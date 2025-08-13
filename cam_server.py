#!/usr/bin/env python3
"""
Servidor Web de Streaming en Vivo con Cámara AI IMX500
Captura video, procesa detecciones de personas y sirve streaming MJPEG con anotaciones
"""

import cv2
import numpy as np
import time
import json
import threading
import queue
import signal
import sys
import os
from flask import Flask, Response, jsonify, render_template_string
from flask_cors import CORS
import psutil
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import subprocess

@dataclass
class Deteccion:
    """Estructura para una detección de persona"""
    timestamp: float
    bbox: List[int]  # [x1, y1, x2, y2]
    confianza: float
    centro: List[int]  # [x, y]
    area: int

@dataclass
class Track:
    """Estructura para tracking de una persona"""
    id: int
    detecciones: deque
    ultima_posicion: List[int]
    ultimo_timestamp: float
    estado: str  # 'entrando', 'saliendo', 'en_habitacion', 'fuera'
    ultimo_evento: Optional[str] = None
    ultimo_evento_timestamp: float = 0.0

class CamaraIMX500:
    """Clase para manejar la cámara IMX500"""
    
    def __init__(self, config):
        self.config = config
        self.proceso_camara = None
        self.frame_actual = None
        self.detecciones_actuales = []
        self.timestamp_ultimo_frame = 0
        self.fps_captura = 0.0
        self.fps_inferencia = 0.0
        
        # Métricas de captura
        self.timestamps_captura = deque(maxlen=30)
        self.timestamps_inferencia = deque(maxlen=30)
        
        # Configurar cámara
        self.configurar_camara()
    
    def configurar_camara(self):
        """Configura la cámara IMX500 para streaming"""
        try:
            print("📷 Configurando cámara IMX500...")
            
            # Verificar si rpicam-vid está disponible
            if not os.path.exists('/usr/bin/rpicam-vid'):
                print("❌ rpicam-vid no disponible")
                return False
            
            # Comando optimizado para streaming
            comando_config = [
                '/usr/bin/rpicam-vid',
                '--width', str(self.config['resolucion'][0]),
                '--height', str(self.config['resolucion'][1]),
                '--framerate', str(self.config['fps_objetivo']),
                '--exposure', 'manual',
                '--shutter', str(self.config['exposure_us']),
                '--gain', str(self.config['gain']),
                '--nopreview',
                '--output', '-',  # Salida a stdout
                '--codec', 'mjpeg',  # Código más rápido
                '--inline',  # Sin buffering
                '--flush',  # Flush inmediato
                '--awb', 'auto',  # Balance de blancos automático
                '--metering', 'centre',  # Medición central
                '--denoise', 'cdn_off'  # Desactivar denoise
            ]
            
            print(f"🔧 Comando cámara: {' '.join(comando_config)}")
            
            # Iniciar proceso de cámara
            self.proceso_camara = subprocess.Popen(
                comando_config,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            print("✅ Cámara configurada e iniciada")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando cámara: {e}")
            return False
    
    def leer_frame(self):
        """Lee un frame de la cámara usando un enfoque más robusto"""
        if not self.proceso_camara:
            return None
        
        try:
            # Verificar si el proceso de cámara sigue activo
            if self.proceso_camara.poll() is not None:
                print("⚠️ Proceso de cámara terminado, reintentando...")
                self.configurar_camara()
                return None
            
            # Leer datos MJPEG con timeout más corto
            buffer_mjpeg = b''
            timeout = time.time() + 0.5  # 500ms timeout
            
            while time.time() < timeout:
                # Leer en chunks más pequeños para mejor control
                chunk = self.proceso_camara.stdout.read(512)
                if not chunk:
                    break
                
                buffer_mjpeg += chunk
                
                # Buscar marcadores de frame MJPEG
                if b'\xff\xd8' in buffer_mjpeg and b'\xff\xd9' in buffer_mjpeg:
                    start = buffer_mjpeg.find(b'\xff\xd8')
                    end = buffer_mjpeg.find(b'\xff\xd9') + 2
                    
                    if start < end:
                        # Extraer frame JPEG
                        frame_jpeg = buffer_mjpeg[start:end]
                        
                        # Verificar tamaño mínimo del frame
                        if len(frame_jpeg) < 1000:  # Frame muy pequeño, probablemente corrupto
                            buffer_mjpeg = buffer_mjpeg[end:]
                            continue
                        
                        # Decodificar a numpy array
                        nparr = np.frombuffer(frame_jpeg, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                            # Actualizar métricas
                            timestamp = time.time()
                            self.timestamps_captura.append(timestamp)
                            
                            # Calcular FPS
                            if len(self.timestamps_captura) >= 2:
                                self.fps_captura = len(self.timestamps_captura) / (self.timestamps_captura[-1] - self.timestamps_captura[0])
                            
                            self.frame_actual = frame
                            self.timestamp_ultimo_frame = timestamp
                            return frame
                        
                        buffer_mjpeg = buffer_mjpeg[end:]
            
            # Si no se pudo leer un frame válido, generar uno de placeholder
            if not self.frame_actual is not None:
                return self.generar_frame_placeholder()
            
            return None
            
        except Exception as e:
            print(f"❌ Error leyendo frame: {e}")
            return self.generar_frame_placeholder()
    
    def generar_frame_placeholder(self):
        """Genera un frame de placeholder cuando no hay video"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Texto de estado
        cv2.putText(frame, "CAMARA NO DISPONIBLE", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Verificando conexion...", (180, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Dibujar icono de cámara
        cv2.circle(frame, (320, 100), 50, (0, 255, 255), 3)
        cv2.circle(frame, (320, 100), 20, (0, 255, 255), -1)
        
        # Mostrar métricas básicas
        cv2.putText(frame, f"FPS: {self.fps_captura:.1f}", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Proceso: {'Activo' if self.proceso_camara else 'Inactivo'}", (50, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def simular_detecciones(self, frame):
        """Simula detecciones de personas (en producción esto vendría de la IMX500)"""
        try:
            # Usar HOG detector como fallback
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detectar personas
            boxes, weights = hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
                hitThreshold=0
            )
            
            detecciones = []
            for (x, y, w, h), weight in zip(boxes, weights):
                # Filtrar por confianza mínima
                if weight > self.config['confianza_minima']:
                    # Filtrar por área mínima
                    area = w * h
                    if area > self.config['area_minima']:
                        # Aplicar ROI de puerta
                        centro_x = x + w // 2
                        centro_y = y + h // 2
                        
                        if self.esta_en_roi_puerta([centro_x, centro_y]):
                            detecciones.append(Deteccion(
                                timestamp=time.time(),
                                bbox=[x, y, x + w, y + h],
                                confianza=float(weight),
                                centro=[centro_x, centro_y],
                                area=area
                            ))
            
            # Actualizar métricas de inferencia
            timestamp = time.time()
            self.timestamps_inferencia.append(timestamp)
            
            if len(self.timestamps_inferencia) >= 2:
                self.fps_inferencia = len(self.timestamps_inferencia) / (self.timestamps_inferencia[-1] - self.timestamps_inferencia[0])
            
            self.detecciones_actuales = detecciones
            return detecciones
            
        except Exception as e:
            print(f"❌ Error simulando detecciones: {e}")
            return []
    
    def esta_en_roi_puerta(self, centro):
        """Verifica si un punto está en el ROI de la puerta"""
        x, y = centro
        x1, y1, x2, y2 = self.config['roi_puerta']
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def cleanup(self):
        """Limpia recursos de la cámara"""
        if self.proceso_camara:
            self.proceso_camara.terminate()
            self.proceso_camara.wait()

class TrackerPersonas:
    """Clase para tracking de personas"""
    
    def __init__(self, config):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        
        # Contadores
        self.contador_entradas = 0
        self.contador_salidas = 0
        self.personas_en_habitacion = 0
    
    def actualizar_tracking(self, detecciones):
        """Actualiza el tracking de personas"""
        # Buscar tracks más cercanos
        personas_actuales = {}
        
        for deteccion in detecciones:
            # Buscar track más cercano
            track_id = None
            distancia_minima = float('inf')
            
            for tid, track in self.tracks.items():
                if track.ultima_posicion:
                    distancia = np.sqrt(
                        (deteccion.centro[0] - track.ultima_posicion[0])**2 +
                        (deteccion.centro[1] - track.ultima_posicion[1])**2
                    )
                    if distancia < distancia_minima and distancia < self.config['distancia_maxima_tracking']:
                        distancia_minima = distancia
                        track_id = tid
            
            # Si no se encontró track, crear uno nuevo
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[track_id] = Track(
                    id=track_id,
                    detecciones=deque(maxlen=self.config['historial_maxlen']),
                    ultima_posicion=deteccion.centro,
                    ultimo_timestamp=deteccion.timestamp,
                    estado='fuera'
                )
            
            # Actualizar track
            track = self.tracks[track_id]
            track.detecciones.append(deteccion)
            track.ultima_posicion = deteccion.centro
            track.ultimo_timestamp = deteccion.timestamp
            
            personas_actuales[track_id] = track
        
        # Limpiar tracks obsoletos
        self.limpiar_tracks_obsoletos()
        
        # Verificar cruce de línea
        self.verificar_cruce_linea()
        
        return personas_actuales
    
    def limpiar_tracks_obsoletos(self):
        """Limpia tracks que no han sido actualizados recientemente"""
        tiempo_actual = time.time()
        tracks_a_eliminar = []
        
        for tid, track in self.tracks.items():
            tiempo_inactivo = (tiempo_actual - track.ultimo_timestamp) * 1000  # ms
            
            if tiempo_inactivo > self.config['track_lost_ms']:
                tracks_a_eliminar.append(tid)
        
        for tid in tracks_a_eliminar:
            del self.tracks[tid]
    
    def verificar_cruce_linea(self):
        """Verifica si una persona cruzó la línea virtual"""
        for tid, track in self.tracks.items():
            # Verificar si hay suficientes detecciones para determinar dirección
            if len(track.detecciones) < 3:
                continue
            
            # Obtener posiciones X de las últimas detecciones
            posiciones_x = [d.centro[0] for d in list(track.detecciones)[-3:]]
            
            # Calcular dirección del movimiento
            x_inicial = posiciones_x[0]
            x_final = posiciones_x[-1]
            diferencia_x = x_final - x_inicial
            
            # Verificar si cruzó la línea
            linea_x = self.config['linea_cruce']
            ancho_banda = self.config['ancho_banda_cruce']
            
            # Verificar si está en la banda de cruce
            en_banda_cruce = abs(track.ultima_posicion[0] - linea_x) <= ancho_banda
            
            if en_banda_cruce:
                # Determinar dirección
                if diferencia_x > self.config['umbral_movimiento']:  # Movimiento hacia la derecha = entrada
                    self.registrar_evento(tid, 'entrada', track.ultimo_timestamp)
                elif diferencia_x < -self.config['umbral_movimiento']:  # Movimiento hacia la izquierda = salida
                    self.registrar_evento(tid, 'salida', track.ultimo_timestamp)
    
    def registrar_evento(self, track_id, tipo_evento, timestamp):
        """Registra un evento de entrada/salida"""
        track = self.tracks[track_id]
        
        # Verificar debounce
        tiempo_desde_ultimo = (timestamp - track.ultimo_evento_timestamp) * 1000  # ms
        
        if tiempo_desde_ultimo < self.config['debounce_ms']:
            return
        
        # Verificar que no sea el mismo evento
        if track.ultimo_evento == tipo_evento:
            return
        
        # Registrar evento
        track.ultimo_evento = tipo_evento
        track.ultimo_evento_timestamp = timestamp
        
        # Actualizar contadores
        if tipo_evento == 'entrada':
            self.contador_entradas += 1
            self.personas_en_habitacion += 1
            track.estado = 'en_habitacion'
            print(f"🚪 PERSONA ENTRÓ - ID: {track_id} - Total: {self.contador_entradas}")
        else:  # salida
            self.contador_salidas += 1
            self.personas_en_habitacion = max(0, self.personas_en_habitacion - 1)
            track.estado = 'fuera'
            print(f"🚪 PERSONA SALIÓ - ID: {track_id} - Total: {self.contador_salidas}")

class ServidorStreaming:
    """Servidor web Flask para streaming en vivo"""
    
    def __init__(self, config):
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Inicializar componentes
        self.camara = CamaraIMX500(config)
        self.tracker = TrackerPersonas(config)
        
        # Estado del sistema
        self.inicio_tiempo = time.time()
        self.frame_count = 0
        
        # Configurar rutas
        self.configurar_rutas()
        
        # Configurar manejo de señales
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Maneja las señales de interrupción"""
        print(f"\n⏹️ Señal recibida ({signum}), deteniendo servidor...")
        self.cleanup()
        sys.exit(0)
    
    def configurar_rutas(self):
        """Configura las rutas del servidor web"""
        
        @self.app.route('/')
        def index():
            """Página principal con información del sistema"""
            return render_template_string(self.get_html_template())
        
        @self.app.route('/stream')
        def stream():
            """Streaming MJPEG en vivo"""
            return Response(
                self.generar_stream(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/metrics')
        def metrics():
            """Métricas del sistema en JSON"""
            return jsonify(self.obtener_metricas())
        
        @self.app.route('/counts')
        def counts():
            """Contadores de entrada/salida"""
            return jsonify({
                'timestamp': time.time(),
                'contador_entradas': self.tracker.contador_entradas,
                'contador_salidas': self.tracker.contador_salidas,
                'personas_en_habitacion': self.tracker.personas_en_habitacion,
                'tracks_activos': len(self.tracker.tracks)
            })
        
        @self.app.route('/health')
        def health():
            """Estado de salud del sistema"""
            return jsonify({
                'status': 'ok',
                'timestamp': time.time(),
                'camara_activa': self.camara.proceso_camara is not None,
                'frame_count': self.frame_count,
                'uptime': time.time() - self.inicio_tiempo
            })
    
    def generar_stream(self):
        """Genera el stream MJPEG en vivo"""
        while True:
            try:
                # Leer frame de la cámara
                frame = self.camara.leer_frame()
                
                if frame is not None:
                    # Procesar detecciones
                    detecciones = self.camara.simular_detecciones(frame)
                    
                    # Actualizar tracking
                    personas_actuales = self.tracker.actualizar_tracking(detecciones)
                    
                    # Dibujar anotaciones en el frame
                    frame_anotado = self.dibujar_anotaciones(frame, detecciones, personas_actuales)
                    
                    # Convertir a JPEG
                    ret, buffer = cv2.imencode('.jpg', frame_anotado, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if ret:
                        # Generar frame MJPEG
                        frame_data = b'--frame\r\n'
                        frame_data += b'Content-Type: image/jpeg\r\n\r\n'
                        frame_data += buffer.tobytes()
                        frame_data += b'\r\n'
                        
                        self.frame_count += 1
                        yield frame_data
                
                # Control de FPS
                time.sleep(1.0 / self.config['fps_objetivo'])
                
            except Exception as e:
                print(f"❌ Error en stream: {e}")
                time.sleep(0.1)
    
    def dibujar_anotaciones(self, frame, detecciones, personas_actuales):
        """Dibuja anotaciones en el frame"""
        frame_anotado = frame.copy()
        
        # Dibujar ROI de la puerta
        x1, y1, x2, y2 = self.config['roi_puerta']
        cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_anotado, 'ROI Puerta', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Dibujar línea de cruce
        linea_x = self.config['linea_cruce']
        cv2.line(frame_anotado, (linea_x, 0), (linea_x, frame.shape[0]), (255, 0, 0), 3)
        cv2.putText(frame_anotado, 'Línea Cruce', (linea_x+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Dibujar detecciones
        for deteccion in detecciones:
            x1, y1, x2, y2 = deteccion.bbox
            confianza = deteccion.confianza
            
            # Color basado en confianza
            if confianza > 0.7:
                color = (0, 255, 0)  # Verde
            elif confianza > 0.5:
                color = (0, 255, 255)  # Amarillo
            else:
                color = (0, 0, 255)  # Rojo
            
            # Bounding box
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta de confianza
            label = f'Persona: {confianza:.2f}'
            cv2.putText(frame_anotado, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Centro de la persona
            cv2.circle(frame_anotado, tuple(deteccion.centro), 3, color, -1)
        
        # Dibujar contadores
        self.dibujar_contadores(frame_anotado)
        
        # Dibujar métricas
        self.dibujar_metricas(frame_anotado)
        
        return frame_anotado
    
    def dibujar_contadores(self, frame):
        """Dibuja los contadores en el frame"""
        # Fondo para contadores
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Contadores
        cv2.putText(frame, f'Entradas: {self.tracker.contador_entradas}', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Salidas: {self.tracker.contador_salidas}', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'En Habitacion: {self.tracker.personas_en_habitacion}', (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Tracks: {len(self.tracker.tracks)}', (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def dibujar_metricas(self, frame):
        """Dibuja las métricas del sistema en el frame"""
        # Fondo para métricas
        cv2.rectangle(frame, (frame.shape[1]-310, 10), (frame.shape[1]-10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (frame.shape[1]-310, 10), (frame.shape[1]-10, 100), (255, 255, 255), 2)
        
        # Métricas
        cv2.putText(frame, f'FPS Captura: {self.camara.fps_captura:.1f}', (frame.shape[1]-300, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'FPS Inferencia: {self.camara.fps_inferencia:.1f}', (frame.shape[1]-300, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Frames: {self.frame_count}', (frame.shape[1]-300, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def obtener_metricas(self):
        """Obtiene las métricas del sistema"""
        # Métricas del sistema
        cpu_percent = psutil.cpu_percent()
        memoria = psutil.virtual_memory()
        
        # Temperatura (Raspberry Pi)
        temperatura = None
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temperatura = float(f.read()) / 1000.0
        except:
            pass
        
        return {
            'timestamp': time.time(),
            'uptime': time.time() - self.inicio_tiempo,
            'frame_count': self.frame_count,
            'fps_captura': self.camara.fps_captura,
            'fps_inferencia': self.camara.fps_inferencia,
            'contador_entradas': self.tracker.contador_entradas,
            'contador_salidas': self.tracker.contador_salidas,
            'personas_en_habitacion': self.tracker.personas_en_habitacion,
            'tracks_activos': len(self.tracker.tracks),
            'cpu': cpu_percent,
            'memoria': memoria.percent,
            'temperatura': temperatura,
            'camara_activa': self.camara.proceso_camara is not None
        }
    
    def get_html_template(self):
        """Retorna la plantilla HTML para la página principal"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cámara AI IMX500 - Streaming en Vivo</title>
            <meta charset="utf-8">
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    text-align: center; 
                    background: #1a1a1a; 
                    color: white; 
                    margin: 0; 
                    padding: 20px; 
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                }
                h1 { 
                    color: #00ff88; 
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5); 
                }
                .stream-container { 
                    border: 3px solid #00ff88; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,255,136,0.3); 
                    margin: 20px 0; 
                    padding: 10px; 
                }
                .stream-image { 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 5px; 
                }
                .metrics { 
                    background: #333; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin: 20px 0; 
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }
                .metric-card { 
                    background: #444; 
                    padding: 15px; 
                    border-radius: 5px; 
                    border-left: 4px solid #00ff88; 
                }
                .metric-value { 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #00ff88; 
                }
                .metric-label { 
                    font-size: 14px; 
                    color: #ccc; 
                    margin-top: 5px; 
                }
                .controls { 
                    margin: 20px 0; 
                }
                button { 
                    background: #00ff88; 
                    color: #000; 
                    border: none; 
                    padding: 10px 20px; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    margin: 5px; 
                    font-weight: bold; 
                }
                button:hover { 
                    background: #00cc6a; 
                }
                .info { 
                    background: #222; 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                    font-size: 14px; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Cámara AI IMX500 - Streaming en Vivo</h1>
                <div class="info">
                    <strong>Sensor:</strong> Sony IMX500 | <strong>Resolución:</strong> 640x480 | <strong>FPS:</strong> 25-30
                </div>
                
                <div class="stream-container">
                    <h3>📹 Video en Vivo con Detecciones</h3>
                    <img class="stream-image" src="/stream" alt="Streaming en Vivo">
                </div>
                
                <div class="controls">
                    <button onclick="location.reload()">🔄 Recargar</button>
                    <button onclick="window.open('/metrics', '_blank')">📊 Ver Métricas JSON</button>
                    <button onclick="window.open('/counts', '_blank')">🔢 Ver Contadores</button>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="entradas">-</div>
                        <div class="metric-label">Entradas</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="salidas">-</div>
                        <div class="metric-label">Salidas</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="en_habitacion">-</div>
                        <div class="metric-label">En Habitación</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="fps">-</div>
                        <div class="metric-label">FPS</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="cpu">-</div>
                        <div class="metric-label">CPU %</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="memoria">-</div>
                        <div class="metric-label">Memoria %</div>
                    </div>
                </div>
            </div>
            
            <script>
                // Actualizar métricas cada 2 segundos
                function actualizarMetricas() {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('entradas').textContent = data.contador_entradas;
                            document.getElementById('salidas').textContent = data.contador_salidas;
                            document.getElementById('en_habitacion').textContent = data.personas_en_habitacion;
                            document.getElementById('fps').textContent = data.fps_captura.toFixed(1);
                            document.getElementById('cpu').textContent = data.cpu.toFixed(1);
                            document.getElementById('memoria').textContent = data.memoria.toFixed(1);
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                // Actualizar cada 2 segundos
                setInterval(actualizarMetricas, 2000);
                
                // Actualizar al cargar la página
                actualizarMetricas();
            </script>
        </body>
        </html>
        """
    
    def iniciar(self, host='0.0.0.0', port=5000, debug=False):
        """Inicia el servidor web"""
        print("🚀 INICIANDO SERVIDOR DE STREAMING EN VIVO")
        print("=" * 60)
        print(f"📱 Cámara AI IMX500 + Raspberry Pi 5")
        print(f"🎯 FPS objetivo: {self.config['fps_objetivo']}")
        print(f"📍 ROI puerta: {self.config['roi_puerta']}")
        print(f"📍 Línea cruce: X={self.config['linea_cruce']}")
        print(f"🌐 Servidor: http://{host}:{port}")
        print(f"📹 Streaming: http://{host}:{port}/stream")
        print(f"📊 Métricas: http://{host}:{port}/metrics")
        print("=" * 60)
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except KeyboardInterrupt:
            print("\n⏹️ Servidor detenido por el usuario")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos del servidor"""
        print("🧹 Limpiando recursos...")
        self.camara.cleanup()
        print("✅ Recursos limpiados")

def main():
    """Función principal"""
    # Configuración por defecto
    config = {
        'resolucion': [640, 480],
        'fps_objetivo': 25,
        'confianza_minima': 0.4,
        'area_minima': 2000,
        'roi_puerta': [80, 80, 560, 420],
        'linea_cruce': 320,
        'ancho_banda_cruce': 3,
        'debounce_ms': 300,
        'track_lost_ms': 700,
        'exposure_us': 4000,
        'gain': 1.0,
        'distancia_maxima_tracking': 100,
        'historial_maxlen': 30,
        'umbral_movimiento': 20
    }
    
    # Cargar configuración si existe
    try:
        if os.path.exists('config_detector.json'):
            with open('config_detector.json', 'r') as f:
                config.update(json.load(f))
            print("📋 Configuración cargada desde config_detector.json")
    except Exception as e:
        print(f"⚠️ Error cargando configuración: {e}")
    
    # Crear y iniciar servidor
    servidor = ServidorStreaming(config)
    servidor.iniciar()

if __name__ == "__main__":
    main() 