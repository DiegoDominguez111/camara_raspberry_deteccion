#!/usr/bin/env python3
"""
Detector de Entrada y Salida de Personas V2 - Cámara AI Raspberry Pi 5
Sistema optimizado para detectar cuando personas entran o salen de una habitación
Usa inferencia de la cámara IMX500 + tracking ligero en Raspberry Pi
"""

import cv2
import numpy as np
import time
import argparse
import subprocess
import os
import json
import threading
import queue
import signal
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import psutil

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

class DetectorEntradaSalidaV2:
    def __init__(self, config=None):
        """
        Inicializa el detector optimizado
        
        Args:
            config: Diccionario de configuración
        """
        # Configuración por defecto
        self.config = {
            'resolucion': (640, 480),
            'fps_objetivo': 30,
            'confianza_minima': 0.4,
            'nms_iou': 0.45,
            'area_minima': 2000,
            'roi_puerta': [80, 80, 560, 420],
            'linea_cruce': 320,  # X central de la puerta
            'ancho_banda_cruce': 3,  # Píxeles de ancho para la línea
            'debounce_ms': 300,
            'track_lost_ms': 700,
            'exposure_us': 4000,  # 1/250s
            'gain': 1.0,
            'ae_lock': True,
            'awb_lock': True,
            'denoise': False,
            'histograma_estable': True
        }
        
        # Actualizar con configuración personalizada si se proporciona
        if config:
            self.config.update(config)
        
        # Estado del sistema
        self.frame_count = 0
        self.inicio_tiempo = time.time()
        self.detenido = False
        self.fps_actual = 0.0
        self.latencia_total = 0.0
        
        # Contadores
        self.contador_entradas = 0
        self.contador_salidas = 0
        self.personas_en_habitacion = 0
        
        # Tracking
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        
        # Colas y buffers
        self.cola_detecciones = queue.Queue(maxsize=100)
        self.cola_eventos = queue.Queue(maxsize=50)
        
        # Métricas de rendimiento
        self.metricas = {
            'fps_captura': 0.0,
            'fps_inferencia': 0.0,
            'fps_post_proceso': 0.0,
            'latencia_captura': 0.0,
            'latencia_inferencia': 0.0,
            'latencia_post_proceso': 0.0,
            'cpu_promedio': 0.0,
            'memoria_promedio': 0.0,
            'temperatura_promedio': 0.0,
            'frames_perdidos': 0,
            'colas_crecientes': 0
        }
        
        # Timestamps para cálculo de FPS
        self.timestamps_captura = deque(maxlen=30)
        self.timestamps_inferencia = deque(maxlen=30)
        self.timestamps_post_proceso = deque(maxlen=30)
        
        # Configurar manejo de señales
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚪 Inicializando detector de entrada/salida V2...")
        print(f"📱 Cámara AI IMX500 + Raspberry Pi 5")
        print(f"🎯 FPS objetivo: {self.config['fps_objetivo']}")
        print(f"📍 ROI puerta: {self.config['roi_puerta']}")
        print(f"📍 Línea cruce: X={self.config['linea_cruce']}")
        
        # Configurar cámara
        self.configurar_camara()
        
        # Iniciar threads
        self.iniciar_threads()
    
    def signal_handler(self, signum, frame):
        """Maneja las señales de interrupción"""
        print(f"\n⏹️ Señal recibida ({signum}), deteniendo detector...")
        self.detenido = True
    
    def configurar_camara(self):
        """Configura la cámara para streaming en tiempo real"""
        try:
            print("📷 Configurando cámara IMX500...")
            
            # Verificar si rpicam-vid está disponible
            if not os.path.exists('/usr/bin/rpicam-vid'):
                print("⚠️ rpicam-vid no disponible, usando modo simulación")
                self.proceso_camara = None
                return
            
            # Comando para configurar cámara con parámetros optimizados
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
                '--flush'  # Flush inmediato
            ]
            
            if self.config['ae_lock']:
                comando_config.extend(['--auto-exposure', 'off'])
            if self.config['awb_lock']:
                comando_config.extend(['--auto-white-balance', 'off'])
            if not self.config['denoise']:
                comando_config.extend(['--denoise', 'off'])
            if self.config['histograma_estable']:
                comando_config.extend(['--histogram', 'on'])
            
            print(f"🔧 Comando cámara: {' '.join(comando_config)}")
            
            # Iniciar proceso de cámara
            self.proceso_camara = subprocess.Popen(
                comando_config,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            print("✅ Cámara configurada e iniciada")
            
        except Exception as e:
            print(f"⚠️ Error configurando cámara: {e}")
            print("📷 Continuando en modo simulación")
            self.proceso_camara = None
    
    def iniciar_threads(self):
        """Inicia los threads de captura y procesamiento"""
        print("🧵 Iniciando threads...")
        
        # Thread de captura de video
        self.thread_captura = threading.Thread(
            target=self.thread_captura_video,
            daemon=True
        )
        self.thread_captura.start()
        
        # Thread de procesamiento de detecciones
        self.thread_procesamiento = threading.Thread(
            target=self.thread_procesamiento_detecciones,
            daemon=True
        )
        self.thread_procesamiento.start()
        
        # Thread de métricas
        self.thread_metricas = threading.Thread(
            target=self.thread_metricas_sistema,
            daemon=True
        )
        self.thread_metricas.start()
        
        print("✅ Threads iniciados")
    
    def thread_captura_video(self):
        """Thread dedicado a capturar video de la cámara"""
        print("📹 Thread de captura iniciado")
        
        # Buffer para MJPEG
        buffer_mjpeg = b''
        frame_count = 0
        
        while not self.detenido:
            try:
                # Leer datos de la cámara
                chunk = self.proceso_camara.stdout.read(1024)
                if not chunk:
                    break
                
                buffer_mjpeg += chunk
                
                # Buscar marcadores de frame MJPEG
                while b'\xff\xd8' in buffer_mjpeg and b'\xff\xd9' in buffer_mjpeg:
                    # Encontrar inicio y fin del frame
                    start = buffer_mjpeg.find(b'\xff\xd8')
                    end = buffer_mjpeg.find(b'\xff\xd9') + 2
                    
                    if start < end:
                        # Extraer frame JPEG
                        frame_jpeg = buffer_mjpeg[start:end]
                        buffer_mjpeg = buffer_mjpeg[end:]
                        
                        # Timestamp de captura
                        timestamp_captura = time.time()
                        self.timestamps_captura.append(timestamp_captura)
                        
                        # Calcular FPS de captura
                        if len(self.timestamps_captura) >= 2:
                            fps_captura = len(self.timestamps_captura) / (self.timestamps_captura[-1] - self.timestamps_captura[0])
                            self.metricas['fps_captura'] = fps_captura
                        
                        # Simular detección de la cámara IMX500 (en producción esto vendría de la cámara)
                        detecciones_simuladas = self.simular_detecciones_imx500(frame_jpeg)
                        
                        # Encolar detecciones con timestamp
                        for deteccion in detecciones_simuladas:
                            deteccion.timestamp = timestamp_captura
                            
                            # Verificar si la cola está llena
                            if self.cola_detecciones.full():
                                # Política "latest-frame": descartar el más viejo
                                try:
                                    self.cola_detecciones.get_nowait()
                                    self.metricas['frames_perdidos'] += 1
                                except queue.Empty:
                                    pass
                            
                            self.cola_detecciones.put(deteccion)
                        
                        frame_count += 1
                        
                        # Log cada 100 frames
                        if frame_count % 100 == 0:
                            print(f"📹 Capturados {frame_count} frames, FPS: {self.metricas['fps_captura']:.1f}")
                
            except Exception as e:
                print(f"❌ Error en thread de captura: {e}")
                time.sleep(0.1)
        
        print("📹 Thread de captura terminado")
    
    def simular_detecciones_imx500(self, frame_jpeg):
        """
        Simula las detecciones que vendrían de la cámara IMX500
        En producción, esto sería reemplazado por la salida real de la cámara
        """
        try:
            # Decodificar JPEG a numpy array
            nparr = np.frombuffer(frame_jpeg, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return []
            
            # Simular detecciones usando OpenCV (en producción esto sería la salida de IMX500)
            # Usar HOG detector como fallback ligero
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
            
            return detecciones
            
        except Exception as e:
            print(f"❌ Error simulando detecciones: {e}")
            return []
    
    def esta_en_roi_puerta(self, centro):
        """Verifica si un punto está en el ROI de la puerta"""
        x, y = centro
        x1, y1, x2, y2 = self.config['roi_puerta']
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def thread_procesamiento_detecciones(self):
        """Thread dedicado a procesar detecciones y tracking"""
        print("🧠 Thread de procesamiento iniciado")
        
        while not self.detenido:
            try:
                # Obtener detección de la cola
                try:
                    deteccion = self.cola_detecciones.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Timestamp de inicio de procesamiento
                timestamp_procesamiento = time.time()
                self.timestamps_post_proceso.append(timestamp_procesamiento)
                
                # Calcular FPS de post-proceso
                if len(self.timestamps_post_proceso) >= 2:
                    fps_post = len(self.timestamps_post_proceso) / (self.timestamps_post_proceso[-1] - self.timestamps_post_proceso[0])
                    self.metricas['fps_post_proceso'] = fps_post
                
                # Calcular latencia
                latencia = (timestamp_procesamiento - deteccion.timestamp) * 1000  # ms
                self.metricas['latencia_post_proceso'] = latencia
                
                # Actualizar tracking
                self.actualizar_tracking(deteccion)
                
                # Verificar cruce de línea
                self.verificar_cruce_linea(deteccion)
                
            except Exception as e:
                print(f"❌ Error en thread de procesamiento: {e}")
                time.sleep(0.01)
        
        print("🧠 Thread de procesamiento terminado")
    
    def actualizar_tracking(self, deteccion):
        """Actualiza el tracking de personas"""
        # Buscar track más cercano
        track_id = None
        distancia_minima = float('inf')
        
        for tid, track in self.tracks.items():
            if track.ultima_posicion:
                distancia = np.sqrt(
                    (deteccion.centro[0] - track.ultima_posicion[0])**2 +
                    (deteccion.centro[1] - track.ultima_posicion[1])**2
                )
                if distancia < distancia_minima and distancia < 100:  # Umbral de 100 píxeles
                    distancia_minima = distancia
                    track_id = tid
        
        # Si no se encontró track, crear uno nuevo
        if track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id] = Track(
                id=track_id,
                detecciones=deque(maxlen=30),
                ultima_posicion=deteccion.centro,
                ultimo_timestamp=deteccion.timestamp,
                estado='fuera'
            )
        
        # Actualizar track
        track = self.tracks[track_id]
        track.detecciones.append(deteccion)
        track.ultima_posicion = deteccion.centro
        track.ultimo_timestamp = deteccion.timestamp
        
        # Limpiar tracks obsoletos
        self.limpiar_tracks_obsoletos()
    
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
    
    def verificar_cruce_linea(self, deteccion):
        """Verifica si una persona cruzó la línea virtual"""
        # Buscar el track correspondiente
        track_id = None
        for tid, track in self.tracks.items():
            if track.ultima_posicion == deteccion.centro:
                track_id = tid
                break
        
        if track_id is None:
            return
        
        track = self.tracks[track_id]
        
        # Verificar si hay suficientes detecciones para determinar dirección
        if len(track.detecciones) < 3:
            return
        
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
        en_banda_cruce = abs(deteccion.centro[0] - linea_x) <= ancho_banda
        
        if en_banda_cruce:
            # Determinar dirección
            if diferencia_x > 20:  # Movimiento hacia la derecha = entrada
                self.registrar_evento(track_id, 'entrada', deteccion.timestamp)
            elif diferencia_x < -20:  # Movimiento hacia la izquierda = salida
                self.registrar_evento(track_id, 'salida', deteccion.timestamp)
    
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
        
        # Encolar evento
        evento = {
            'timestamp': timestamp,
            'track_id': track_id,
            'tipo': tipo_evento,
            'contador_entradas': self.contador_entradas,
            'contador_salidas': self.contador_salidas,
            'personas_en_habitacion': self.personas_en_habitacion
        }
        
        if not self.cola_eventos.full():
            self.cola_eventos.put(evento)
    
    def thread_metricas_sistema(self):
        """Thread dedicado a monitorear métricas del sistema"""
        print("📊 Thread de métricas iniciado")
        
        while not self.detenido:
            try:
                # Métricas del sistema
                cpu_percent = psutil.cpu_percent()
                memoria = psutil.virtual_memory()
                temperatura = self.obtener_temperatura()
                
                # Actualizar métricas promedio
                self.metricas['cpu_promedio'] = cpu_percent
                self.metricas['memoria_promedio'] = memoria.percent
                if temperatura:
                    self.metricas['temperatura_promedio'] = temperatura
                
                # Verificar colas crecientes
                if self.cola_detecciones.qsize() > 50:
                    self.metricas['colas_crecientes'] += 1
                
                # Log de métricas cada 5 segundos
                if self.frame_count % 150 == 0:  # ~5 segundos a 30 FPS
                    self.mostrar_metricas()
                
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error en thread de métricas: {e}")
                time.sleep(1)
        
        print("📊 Thread de métricas terminado")
    
    def mostrar_metricas(self):
        """Muestra las métricas del sistema"""
        print("\n" + "="*60)
        print("📊 MÉTRICAS DEL SISTEMA")
        print("="*60)
        print(f"🎯 FPS Captura: {self.metricas['fps_captura']:.1f}")
        print(f"🧠 FPS Post-proceso: {self.metricas['fps_post_proceso']:.1f}")
        print(f"⏱️ Latencia Post-proceso: {self.metricas['latencia_post_proceso']:.1f} ms")
        print(f"👥 Personas en habitación: {self.personas_en_habitacion}")
        print(f"🚪 Total entradas: {self.contador_entradas}")
        print(f"🚪 Total salidas: {self.contador_salidas}")
        print(f"📊 Tracks activos: {len(self.tracks)}")
        print(f"📈 Cola detecciones: {self.cola_detecciones.qsize()}")
        print(f"📈 Cola eventos: {self.cola_eventos.qsize()}")
        print(f"🖥️ CPU: {self.metricas['cpu_promedio']:.1f}%")
        print(f"💾 Memoria: {self.metricas['memoria_promedio']:.1f}%")
        if self.metricas['temperatura_promedio'] > 0:
            print(f"🌡️ Temperatura: {self.metricas['temperatura_promedio']:.1f}°C")
        print(f"❌ Frames perdidos: {self.metricas['frames_perdidos']}")
        print(f"⚠️ Colas crecientes: {self.metricas['colas_crecientes']}")
    
    def obtener_temperatura(self):
        """Obtiene la temperatura de la Raspberry Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return temp
        except:
            return None
    
    def obtener_metricas_json(self):
        """Retorna las métricas en formato JSON"""
        return {
            'timestamp': time.time(),
            'fps_captura': self.metricas['fps_captura'],
            'fps_post_proceso': self.metricas['fps_post_proceso'],
            'latencia_post_proceso': self.metricas['latencia_post_proceso'],
            'contador_entradas': self.contador_entradas,
            'contador_salidas': self.contador_salidas,
            'personas_en_habitacion': self.personas_en_habitacion,
            'tracks_activos': len(self.tracks),
            'cola_detecciones': self.cola_detecciones.qsize(),
            'cola_eventos': self.cola_eventos.qsize(),
            'cpu': self.metricas['cpu_promedio'],
            'memoria': self.metricas['memoria_promedio'],
            'temperatura': self.metricas['temperatura_promedio'],
            'frames_perdidos': self.metricas['frames_perdidos'],
            'colas_crecientes': self.metricas['colas_crecientes']
        }
    
    def ejecutar_detector(self, duracion=0):
        """Ejecuta el detector optimizado"""
        print("🚪 INICIANDO DETECTOR DE ENTRADA/SALIDA V2")
        print("📱 Cámara AI IMX500 + Raspberry Pi 5")
        print("🎯 Streaming en tiempo real + Tracking optimizado")
        print(f"📍 ROI puerta: {self.config['roi_puerta']}")
        print(f"📍 Línea cruce: X={self.config['linea_cruce']}")
        print(f"⚡ FPS objetivo: {self.config['fps_objetivo']}")
        print("=" * 60)
        
        if duracion > 0:
            print(f"⏰ Duración configurada: {duracion} segundos")
        else:
            print("♾️ Ejecución continua (Ctrl+C para detener)")
        
        tiempo_inicio = time.time()
        
        try:
            while not self.detenido:
                # Verificar duración
                if duracion > 0 and (time.time() - tiempo_inicio) > duracion:
                    print(f"⏰ Tiempo límite alcanzado ({duracion}s)")
                    break
                
                # Sleep principal
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️ Deteniendo detector...")
        
        finally:
            self.cleanup()
            print("\n" + "="*60)
            print("📊 ESTADÍSTICAS FINALES:")
            print(f"🚪 Total entradas: {self.contador_entradas}")
            print(f"🚪 Total salidas: {self.contador_salidas}")
            print(f"👥 Personas en habitación: {self.personas_en_habitacion}")
            print(f"⚡ FPS promedio: {self.metricas['fps_post_proceso']:.1f}")
            print(f"⏱️ Latencia promedio: {self.metricas['latencia_post_proceso']:.1f} ms")
            print("✅ Detector detenido correctamente")
    
    def cleanup(self):
        """Limpia recursos del detector"""
        print("🧹 Limpiando recursos...")
        
        # Detener proceso de cámara
        if hasattr(self, 'proceso_camara'):
            self.proceso_camara.terminate()
            self.proceso_camara.wait()
        
        # Esperar threads
        if hasattr(self, 'thread_captura'):
            self.thread_captura.join(timeout=2)
        if hasattr(self, 'thread_procesamiento'):
            self.thread_procesamiento.join(timeout=2)
        if hasattr(self, 'thread_metricas'):
            self.thread_metricas.join(timeout=2)
        
        print("✅ Recursos limpiados")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Detector de Entrada/Salida V2 - Cámara AI Raspberry Pi 5')
    parser.add_argument('--duracion', type=int, default=0, help='Duración en segundos (0 = continuo)')
    parser.add_argument('--config', type=str, help='Archivo de configuración JSON')
    
    args = parser.parse_args()
    
    # Cargar configuración si se especifica
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"📋 Configuración cargada desde: {args.config}")
        except Exception as e:
            print(f"❌ Error cargando configuración: {e}")
            return
    
    # Crear detector
    detector = DetectorEntradaSalidaV2(config=config)
    
    # Ejecutar detector
    detector.ejecutar_detector(duracion=args.duracion)

if __name__ == "__main__":
    main() 