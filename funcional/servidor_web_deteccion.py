#!/usr/bin/env python3
"""
Servidor Web con Detección en Tiempo Real - Cámara AI Raspberry Pi 5
Muestra la cámara con detecciones de personas, tracking y área de puerta
Optimizado para 30 FPS y conteo preciso de entradas/salidas
"""

import cv2
import numpy as np
import http.server
import socketserver
import subprocess
import time
import os
import signal
import sys
from ultralytics import YOLO
import threading
import json
from collections import deque

class ServidorDeteccion:
    def __init__(self, puerto=8082, modelo='yolov8n.pt', confianza=0.3):
        self.puerto = puerto
        self.modelo = modelo
        self.confianza = confianza
        self.detenido = False
        
        # Configurar zona de la puerta
        self.zona_puerta = [80, 80, 560, 420]  # Más amplia
        
        # Estado de tracking
        self.personas_detectadas = {}
        self.historial_personas = {}
        self.historial_maxlen = 15  # Igual que en detector_entrada_salida
        self.contador_entradas = 0
        self.contador_salidas = 0
        self.personas_en_habitacion = 0
        self.personas_en_zona_puerta = set()
        self.personas_estado = {}  # ID -> {'ultima': None, 'persistencia': 0}
        
        # Configurar señales
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🤖 Inicializando servidor de detección en tiempo real...")
        self.cargar_modelo()
        self.fps_captura = 30
        self.captura = None
        self.captura_thread = threading.Thread(target=self.iniciar_stream_camara, daemon=True)
        self.captura_thread.start()
        self.ultimo_frame_raw = None
        self.ultimo_frame_time = 0
        self.fps_real = 0
        self.ultimo_fps_log = time.time()
        self.ultimo_frame = None  # Para streaming fluido
        self.lock_frame = threading.Lock()
        self.inferencia_thread = threading.Thread(target=self.loop_inferencia, daemon=True)
        self.inferencia_thread.start()
        self.linea_virtual = (self.zona_puerta[0] + self.zona_puerta[2]) // 2  # X central de la puerta
    
    def signal_handler(self, signum, frame):
        """Maneja las señales de interrupción"""
        print(f"\n⏹️ Señal recibida ({signum}), deteniendo servidor...")
        self.detenido = True
        sys.exit(0)
    
    def cargar_modelo(self):
        """Carga el modelo YOLO"""
        try:
            print(f"📦 Cargando modelo: {self.modelo}")
            self.yolo = YOLO(self.modelo)
            print("✅ Modelo cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def iniciar_stream_camara(self):
        # Usa rpicam-vid para emitir MJPEG a un puerto local y OpenCV para leerlo
        # Lanzar rpicam-vid en background
        comando = [
            'rpicam-vid',
            '--width', '640',
            '--height', '480',
            '--framerate', str(self.fps_captura),
            '--codec', 'mjpeg',
            '--listen',
            '--output', 'tcp://127.0.0.1:8888',
            '--timeout', '0'
        ]
        self.proc_camara = subprocess.Popen(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Esperar a que el stream esté listo
        time.sleep(1)
        # Leer el stream MJPEG con OpenCV
        cap = cv2.VideoCapture('tcp://127.0.0.1:8888', cv2.CAP_FFMPEG)
        while True:
            ret, frame = cap.read()
            if ret:
                self.ultimo_frame_raw = frame
                self.ultimo_frame_time = time.time()
            else:
                time.sleep(0.01)
        # NOTA: el proceso se termina al cerrar el servidor

    def capturar_imagen(self):
        # Devuelve el último frame capturado por el hilo de cámara
        if self.ultimo_frame_raw is not None:
            # Calcular FPS real
            ahora = time.time()
            if ahora - self.ultimo_fps_log > 2:
                self.fps_real = 1.0 / max(1e-6, (ahora - self.ultimo_frame_time))
                print(f"[DEBUG] FPS cámara: {self.fps_real:.2f}")
                self.ultimo_fps_log = ahora
            return self.ultimo_frame_raw.copy()
        else:
            return None
    
    def detectar_personas(self, frame):
        """Detecta personas en el frame optimizado"""
        try:
            # Reducir tamaño del frame para detección más rápida
            frame_pequeno = cv2.resize(frame, (320, 240))
            
            resultados = self.yolo(frame_pequeno, conf=self.confianza, verbose=False, classes=[0])
            
            detecciones = []
            for resultado in resultados:
                if resultado.boxes is not None:
                    for box in resultado.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confianza = float(box.conf[0].cpu().numpy())
                        
                        # Escalar coordenadas de vuelta al tamaño original
                        x1, y1, x2, y2 = x1 * 2, y1 * 2, x2 * 2, y2 * 2
                        
                        centro_x = int((x1 + x2) / 2)
                        centro_y = int((y1 + y2) / 2)
                        
                        detecciones.append({
                            'clase': 'person',
                            'confianza': confianza,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'centro': [centro_x, centro_y]
                        })
            
            return detecciones
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []
    
    def loop_inferencia(self):
        while not self.detenido:
            frame = self.capturar_imagen()
            if frame is None:
                continue
            detecciones = self.detectar_personas(frame)
            self.actualizar_tracking(detecciones)
            frame = self.dibujar_detecciones(frame, detecciones)
            with self.lock_frame:
                self.ultimo_frame = frame.copy()

    def actualizar_tracking(self, detecciones):
        """Actualiza el tracking de personas y detecta entradas/salidas con lógica robusta"""
        personas_actuales = {}
        personas_en_zona_actual = set()
        for deteccion in detecciones:
            centro = deteccion['centro']
            persona_id = None
            distancia_minima = float('inf')
            for pid, pos_anterior in self.personas_detectadas.items():
                distancia = np.sqrt((centro[0] - pos_anterior[0])**2 + (centro[1] - pos_anterior[1])**2)
                if distancia < distancia_minima and distancia < 150:
                    distancia_minima = distancia
                    persona_id = pid
            if persona_id is None:
                persona_id = len(self.personas_detectadas) + 1
            personas_actuales[persona_id] = centro
            if persona_id not in self.historial_personas:
                self.historial_personas[persona_id] = deque(maxlen=20)
            self.historial_personas[persona_id].append(centro)
            if persona_id not in self.personas_estado:
                self.personas_estado[persona_id] = {
                    'estado': 'esperando',  # esperando, cruzando, contado
                    'lado_anterior': None,
                    'cooldown': 0
                }
            # Cooldown para evitar rebotes
            if self.personas_estado[persona_id]['cooldown'] > 0:
                self.personas_estado[persona_id]['cooldown'] -= 1
            # Determinar de qué lado de la línea está el centro
            x_centro = centro[0]
            if x_centro < self.linea_virtual:
                lado_actual = 'izquierda'
            else:
                lado_actual = 'derecha'
            # Estado de cruce
            estado = self.personas_estado[persona_id]['estado']
            lado_anterior = self.personas_estado[persona_id]['lado_anterior']
            # Lógica de cruce
            if estado == 'esperando':
                if lado_anterior is not None and lado_anterior != lado_actual:
                    # Cruzó la línea
                    if lado_anterior == 'izquierda' and lado_actual == 'derecha' and self.personas_estado[persona_id]['cooldown'] == 0:
                        self.contador_entradas += 1
                        self.personas_en_habitacion += 1
                        self.personas_estado[persona_id]['estado'] = 'contado'
                        self.personas_estado[persona_id]['cooldown'] = 15
                        print(f"[CRUCE] ID {persona_id} ENTRADA (izq->der) - Total entradas: {self.contador_entradas}")
                    elif lado_anterior == 'derecha' and lado_actual == 'izquierda' and self.personas_estado[persona_id]['cooldown'] == 0:
                        self.contador_salidas += 1
                        self.personas_en_habitacion = max(0, self.personas_en_habitacion - 1)
                        self.personas_estado[persona_id]['estado'] = 'contado'
                        self.personas_estado[persona_id]['cooldown'] = 15
                        print(f"[CRUCE] ID {persona_id} SALIDA (der->izq) - Total salidas: {self.contador_salidas}")
                else:
                    # No ha cruzado, sigue esperando
                    self.personas_estado[persona_id]['estado'] = 'esperando'
            elif estado == 'contado':
                # Esperar a que la persona se aleje de la línea para resetear
                if lado_actual == lado_anterior:
                    self.personas_estado[persona_id]['estado'] = 'esperando'
            # Actualizar lado anterior
            self.personas_estado[persona_id]['lado_anterior'] = lado_actual
        self.personas_detectadas = personas_actuales
        self.personas_en_zona_puerta = personas_en_zona_actual
        inactivos = set(self.historial_personas.keys()) - set(personas_actuales.keys())
        for pid in inactivos:
            del self.historial_personas[pid]
            if pid in self.personas_estado:
                del self.personas_estado[pid]
    
    def persona_ya_contada(self, persona_id, tipo_movimiento):
        """Verifica si una persona ya fue contada para evitar duplicados"""
        if not hasattr(self, 'personas_contadas'):
            self.personas_contadas = {}
        
        if persona_id not in self.personas_contadas:
            self.personas_contadas[persona_id] = {'entrada': False, 'salida': False}
        
        return self.personas_contadas[persona_id][tipo_movimiento]
    
    def marcar_persona_contada(self, persona_id, tipo_movimiento):
        """Marca a una persona como contada para evitar duplicados"""
        if not hasattr(self, 'personas_contadas'):
            self.personas_contadas = {}
        
        if persona_id not in self.personas_contadas:
            self.personas_contadas[persona_id] = {'entrada': False, 'salida': False}
        
        self.personas_contadas[persona_id][tipo_movimiento] = True
    
    def limpiar_historiales_obsoletos(self):
        """Limpia historiales de personas que ya no están siendo detectadas"""
        personas_activas = set(self.personas_detectadas.keys())
        
        # Limpiar historiales de personas inactivas
        for pid in list(self.historial_personas.keys()):
            if pid not in personas_activas:
                del self.historial_personas[pid]
                if hasattr(self, 'personas_contadas') and pid in self.personas_contadas:
                    del self.personas_contadas[pid]
    
    def direccion_simple_debug(self, historial):
        """Determina la dirección simple y retorna debug info"""
        x_ini = historial[0][0]
        x_fin = historial[-1][0]
        diferencia = x_fin - x_ini
        umbral = 25
        if diferencia > umbral:
            return 'entrada', x_ini, x_fin, diferencia
        elif -diferencia > umbral:
            return 'salida', x_ini, x_fin, diferencia
        return None, x_ini, x_fin, diferencia
    
    def dibujar_detecciones(self, frame, detecciones):
        """Dibuja las detecciones en el frame"""
        # Dibujar zona de puerta
        x1, y1, x2, y2 = self.zona_puerta
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "ZONA PUERTA", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Dibujar línea virtual
        x_c = self.linea_virtual
        cv2.line(frame, (x_c, 0), (x_c, frame.shape[0]), (0, 0, 255), 2)
        cv2.putText(frame, "Línea de cruce", (x_c+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # Dibujar detecciones de personas
        for i, deteccion in enumerate(detecciones):
            x1, y1, x2, y2 = deteccion['bbox']
            confianza = deteccion['confianza']
            centro = deteccion['centro']
            
            # Color basado en si está en zona de puerta
            if self.esta_en_zona_puerta(centro):
                color = (0, 255, 0)  # Verde si está en zona de puerta
                estado = "EN PUERTA"
            else:
                color = (255, 0, 0)  # Rojo si no está en zona de puerta
                estado = "FUERA"
            
            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta
            etiqueta = f"Persona {i+1}: {confianza:.2f} - {estado}"
            cv2.putText(frame, etiqueta, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Dibujar centro
            cv2.circle(frame, (centro[0], centro[1]), 5, color, -1)
            
            # Dibujar ID
            cv2.putText(frame, f"ID: {i+1}", (x1, y2+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar información del sistema
        self.dibujar_info_sistema(frame)
        
        return frame
    
    def esta_en_zona_puerta(self, centro_persona):
        """Verifica si una persona está en la zona de puerta"""
        x, y = centro_persona
        x1, y1, x2, y2 = self.zona_puerta
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def dibujar_info_sistema(self, frame):
        """Dibuja información del sistema en el frame"""
        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Información básica
        info_lines = [
            f"Personas detectadas: {len(self.personas_detectadas)}",
            f"Personas en puerta: {len(self.personas_en_zona_puerta)}",
            f"Entradas: {self.contador_entradas}",
            f"Salidas: {self.contador_salidas}",
            f"En habitacion: {self.personas_en_habitacion}",
            f"Historiales activos: {len(self.historial_personas)}"
        ]
        
        for i, linea in enumerate(info_lines):
            y_pos = 30 + (i * 20)
            cv2.putText(frame, linea, (15, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Información de debug del tracking
        if self.personas_detectadas:
            debug_y = 160
            cv2.rectangle(overlay, (10, debug_y), (400, debug_y + 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "DEBUG TRACKING:", (15, debug_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            for i, (pid, pos) in enumerate(list(self.personas_detectadas.items())[:3]):  # Solo mostrar 3
                debug_text = f"ID {pid}: ({pos[0]}, {pos[1]})"
                if pid in self.historial_personas:
                    historial_len = len(self.historial_personas[pid])
                    debug_text += f" - Hist: {historial_len}"
                cv2.putText(frame, debug_text, (15, debug_y + 40 + (i * 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def generar_frame_con_detecciones(self):
        """Genera un frame con detecciones optimizado"""
        # Devuelve el último frame disponible para el streaming
        with self.lock_frame:
            if self.ultimo_frame is not None:
                return self.ultimo_frame.copy()
            else:
                # Frame negro si aún no hay nada
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Esperando cámara...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame

class DeteccionHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        """Maneja HEAD requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Maneja POST requests"""
        if self.path == '/reset':
            # Resetear contadores
            self.server.servidor.contador_entradas = 0
            self.server.servidor.contador_salidas = 0
            self.server.servidor.personas_en_habitacion = 0
            self.server.servidor.personas_contadas = {}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == '/':
            # Página principal
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Detección en Tiempo Real - Cámara AI</title>
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
                        max-width: 1000px; 
                        margin: 0 auto; 
                    }
                    h1 { 
                        color: #00ff88; 
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.5); 
                    }
                    .camera-feed { 
                        border: 3px solid #00ff88; 
                        border-radius: 10px; 
                        box-shadow: 0 0 20px rgba(0,255,136,0.3); 
                        margin: 20px 0; 
                    }
                    .info { 
                        background: #333; 
                        padding: 15px; 
                        border-radius: 8px; 
                        margin: 20px 0; 
                        font-size: 14px; 
                    }
                    .legend { 
                        background: #222; 
                        padding: 15px; 
                        border-radius: 8px; 
                        margin: 20px 0; 
                        text-align: left; 
                    }
                    .legend h3 { 
                        color: #00ff88; 
                        margin-top: 0; 
                    }
                    .legend-item { 
                        margin: 10px 0; 
                        display: flex; 
                        align-items: center; 
                    }
                    .color-box { 
                        width: 20px; 
                        height: 20px; 
                        margin-right: 10px; 
                        border-radius: 3px; 
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Detección en Tiempo Real - Cámara AI</h1>
                    <div class="info">
                        <strong>Sensor:</strong> Sony IMX500 | <strong>Resolución:</strong> 640x480 | <strong>Modelo:</strong> YOLOv8 | <strong>FPS:</strong> 30
                    </div>
                    
                    <div class="camera-feed">
                        <img id="videoFeed" src="/video_feed" alt="Feed de Cámara con Detecciones" style="max-width: 100%; height: auto;">
                    </div>
                    
                    <div class="legend">
                        <h3>📋 Leyenda de Colores:</h3>
                        <div class="legend-item">
                            <div class="color-box" style="background: #00ff88;"></div>
                            <span>Verde: Persona en zona de puerta</span>
                        </div>
                        <div class="legend-item">
                            <div class="color-box" style="background: #ff0000;"></div>
                            <span>Rojo: Persona fuera de zona de puerta</span>
                        </div>
                        <div class="legend-item">
                            <div class="color-box" style="background: #ffff00;"></div>
                            <span>Amarillo: Zona de puerta</span>
                        </div>
                    </div>
                    
                    <div class="info">
                        <strong>Instrucciones:</strong> Muévete por la zona amarilla (puerta) para probar la detección de entrada/salida
                    </div>
                    
                    <div class="controls">
                        <button onclick="resetContadores()" style="background: #ff6b6b; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px;">
                            🔄 Reset Contadores
                        </button>
                    </div>
                </div>
                
                <script>
                    // Recargar imagen si hay error
                    document.getElementById('videoFeed').onerror = function() {
                        console.log('Error cargando video, reintentando...');
                        setTimeout(() => {
                            this.src = '/video_feed?' + new Date().getTime();
                        }, 1000);
                    };
                    
                    // Función para resetear contadores
                    function resetContadores() {
                        if (confirm('¿Estás seguro de que quieres resetear los contadores?')) {
                            fetch('/reset', {method: 'POST'})
                                .then(response => response.json())
                                .then(data => {
                                    if (data.success) {
                                        alert('Contadores reseteados correctamente');
                                        location.reload();
                                    } else {
                                        alert('Error al resetear contadores');
                                    }
                                })
                                .catch(error => {
                                    console.error('Error:', error);
                                    alert('Error al resetear contadores');
                                });
                        }
                    }
                </script>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
            
        elif self.path == '/video_feed':
            # Stream de video con detecciones optimizado para 30 FPS
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            try:
                while not self.server.detenido:
                    # Generar frame con detecciones
                    frame = self.server.servidor.generar_frame_con_detecciones()
                    
                    # Convertir a JPEG con calidad optimizada
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        self.wfile.write(b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Pausa mínima para 30 FPS
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                print(f"❌ Error en streaming: {e}")
                
        else:
            self.send_response(404)
            self.end_headers()

def main():
    """Función principal"""
    puerto = 8082
    
    print("🌐 Iniciando servidor de detección en tiempo real...")
    print(f"📱 Puerto: {puerto}")
    print(f"🔗 URL: http://localhost:{puerto}")
    print("🎯 Optimizado para 30 FPS y conteo preciso de entradas/salidas")
    print("=" * 50)
    
    try:
        # Crear servidor de detección
        servidor = ServidorDeteccion(puerto=puerto)
        
        # Configurar servidor HTTP
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", puerto), DeteccionHandler) as httpd:
            # Asignar servidor de detección al handler
            httpd.servidor = servidor
            httpd.detenido = False
            
            print(f"✅ Servidor iniciado en puerto {puerto}")
            print("🌍 Abre tu navegador en la URL mostrada arriba")
            print("🎯 Verás la cámara con detecciones en tiempo real a 30 FPS")
            print("⏹️ Presiona Ctrl+C para detener el servidor")
            print("=" * 50)
            
            # Iniciar servidor
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo servidor...")
        print("✅ Servidor detenido correctamente")
    except Exception as e:
        print(f"❌ Error en servidor: {e}")

if __name__ == "__main__":
    main() 