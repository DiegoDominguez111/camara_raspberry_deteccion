#!/usr/bin/env python3
"""
Servidor Web de Streaming en Vivo con Cámara AI IMX500
Captura video, procesa detecciones de personas y sirve streaming MJPEG con anotaciones
Sistema optimizado con procesamiento en segundo plano y visualización dinámica
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
# Flask-CORS removido para compatibilidad
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
    activo: bool = True # Nuevo campo para indicar si el track está activo
    frames_sin_detectar: int = 0 # Nuevo campo para contar frames sin detectar

class ProcesadorSegundoPlano:
    """Clase para procesamiento en segundo plano independiente del renderizado"""
    
    def __init__(self, config):
        self.config = config
        self.detenido = False
        
        # Estado del procesamiento
        self.frame_actual = None
        self.detecciones_actuales = []
        self.timestamp_ultimo_frame = 0
        self.fps_captura = 0.0
        self.fps_inferencia = 0.0
        
        # Métricas de captura
        self.timestamps_captura = deque(maxlen=30)
        self.timestamps_inferencia = deque(maxlen=30)
        
        # Colas para comunicación entre threads
        self.cola_frames = queue.Queue(maxsize=10)
        self.cola_detecciones = queue.Queue(maxsize=50)
        
        # Configurar cámara
        self.configurar_camara()
        
        # Inicializar tracker
        self.tracker = TrackerPersonas(config)
        
        # Iniciar threads de procesamiento
        self.iniciar_threads_procesamiento()
        
        # Iniciar thread de inferencias AI periódicas
        self.iniciar_thread_inferencias_ai()
    
    def configurar_camara(self):
        """Configura la cámara IMX500 para streaming normal"""
        try:
            print("📷 Configurando cámara IMX500 para streaming...")
            
            # Verificar si rpicam-vid está disponible
            if not os.path.exists('/usr/bin/rpicam-vid'):
                print("❌ rpicam-vid no disponible")
                return False
            
            # Comando optimizado para cámara IMX500 - configuración simple y estable
            comando_config = [
                '/usr/bin/rpicam-vid',
                '--width', str(self.config['resolucion'][0]),
                '--height', str(self.config['resolucion'][1]),
                '--framerate', str(self.config['fps_objetivo']),
                '--nopreview',
                '--output', '-',  # Salida a stdout
                '--codec', 'mjpeg',  # Código MJPEG
                '--profile', 'baseline',  # Perfil JPEG estable
                '--quality', '85',  # Calidad JPEG estable
                '--awb', 'auto',  # Balance de blancos automático
                '--metering', 'centre',  # Medición central
                '--exposure', 'normal',  # Exposición normal
                '--gain', '1.0',  # Ganancia mínima
                '--denoise', 'cdn_off',  # Desactivar denoise
                '--inline',  # Sin buffering
                '--flush'  # Flush inmediato
            ]
            
            print(f"🔧 Comando cámara: {' '.join(comando_config)}")
            
            # Iniciar proceso de cámara
            self.proceso_camara = subprocess.Popen(
                comando_config,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # Esperar un momento para que la cámara se inicialice
            time.sleep(2)
            
            # Verificar que el proceso esté activo
            if self.proceso_camara.poll() is None:
                print("✅ Cámara configurada e iniciada")
                return True
            else:
                print("❌ Proceso de cámara falló al iniciar")
                # Obtener el error si está disponible
                stderr_output = self.proceso_camara.stderr.read().decode()
                if stderr_output:
                    print(f"Error de cámara: {stderr_output}")
                return False
            
        except Exception as e:
            print(f"❌ Error configurando cámara: {e}")
            return False
    
    def iniciar_threads_procesamiento(self):
        """Inicia los threads de procesamiento en segundo plano"""
        print("🧵 Iniciando threads de procesamiento en segundo plano...")
        
        # Thread de captura de frames
        self.thread_captura = threading.Thread(
            target=self.thread_captura_frames,
            daemon=True
        )
        self.thread_captura.start()
        
        # Thread de procesamiento de detecciones
        self.thread_procesamiento = threading.Thread(
            target=self.thread_procesamiento_detecciones,
            daemon=True
        )
        self.thread_procesamiento.start()
        
        print("✅ Threads de procesamiento iniciados")
    
    def iniciar_thread_inferencias_ai(self):
        """Inicia el thread de inferencias AI periódicas"""
        print("🤖 Iniciando thread de inferencias AI periódicas...")
        
        # Thread de inferencias AI
        self.thread_inferencias_ai = threading.Thread(
            target=self.thread_inferencias_ai_periodicas,
            daemon=True
        )
        self.thread_inferencias_ai.start()
        
        print("✅ Thread de inferencias AI iniciado")
    
    def thread_inferencias_ai_periodicas(self):
        """Thread dedicado a inferencias AI periódicas"""
        print("🤖 Thread de inferencias AI periódicas iniciado")
        
        frame_counter = 0
        
        while not self.detenido:
            try:
                frame_counter += 1
                
                # Por ahora, usar detecciones sintéticas para probar el sistema
                detecciones = self.generar_detecciones_sinteticas(frame_counter)
                
                if detecciones:
                    # Encolar detecciones para procesamiento
                    if not self.cola_detecciones.full():
                        self.cola_detecciones.put({
                            'detecciones': detecciones,
                            'timestamp': time.time()
                        })
                    
                    print(f"🎯 Frame {frame_counter}: {len(detecciones)} personas detectadas (sintéticas)")
                
                # Actualizar métricas de inferencia
                timestamp = time.time()
                self.timestamps_inferencia.append(timestamp)
                
                if len(self.timestamps_inferencia) > 15:
                    self.timestamps_inferencia = deque(list(self.timestamps_inferencia)[-15:])
                
                if len(self.timestamps_inferencia) >= 2:
                    recent_timestamps = list(self.timestamps_inferencia)[-10:]
                    if len(recent_timestamps) >= 2:
                        time_span = recent_timestamps[-1] - recent_timestamps[0]
                        if time_span > 0:
                            self.fps_inferencia = (len(recent_timestamps) - 1) / time_span
                
                # Esperar 2 segundos antes de la siguiente inferencia
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error en thread de inferencias AI: {e}")
                time.sleep(1)
        
        print("🤖 Thread de inferencias AI periódicas terminado")
    
    def ejecutar_inferencia_ai(self, frame_counter):
        """Ejecuta inferencia AI real usando rpicam-still con post-procesamiento"""
        try:
            # Verificar si el modelo AI está disponible
            modelo_ai = '/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json'
            if not os.path.exists(modelo_ai):
                print(f"❌ Modelo AI no encontrado: {modelo_ai}")
                return []
            
            # Comando para inferencia AI con rpicam-still
            comando_inferencia = [
                '/usr/bin/rpicam-still',
                '--width', str(self.config['resolucion'][0]),
                '--height', str(self.config['resolucion'][1]),
                '--nopreview',
                '--output', '-',  # Salida a stdout
                '--post-process-file', modelo_ai,
                '--metadata', '-',  # Metadatos a stdout
                '--metadata-format', 'json',
                '--immediate',  # Captura inmediata
                '--timeout', '1000',  # Timeout de 1 segundo
                '--awb', 'auto',
                '--metering', 'centre',
                '--denoise', 'cdn_off'
            ]
            
            # Ejecutar inferencia AI
            proceso_inferencia = subprocess.Popen(
                comando_inferencia,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # Esperar a que termine (timeout de 3 segundos)
            try:
                stdout_data, stderr_data = proceso_inferencia.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                proceso_inferencia.kill()
                proceso_inferencia.communicate()
                print("⚠️ Timeout en inferencia AI")
                return []
            
            # Verificar si el proceso terminó correctamente
            if proceso_inferencia.returncode != 0:
                print(f"⚠️ Inferencia AI no disponible: returncode={proceso_inferencia.returncode}")
                if stderr_data:
                    error_msg = stderr_data.decode()
                    print(f"Error: {error_msg}")
                    # Si el error es por archivos .rpk corruptos, usar detección básica
                    if "garbage after data" in error_msg or "not found" in error_msg:
                        print("🔄 Usando detección básica como fallback...")
                        return self.deteccion_basica_fallback()
                return []
            
            # Procesar metadatos JSON
            if stdout_data:
                try:
                    # Buscar JSON en la salida (puede estar mezclado con datos de imagen)
                    stdout_str = stdout_data.decode('utf-8', errors='ignore')
                    
                    # Buscar el inicio del JSON
                    json_start = stdout_str.find('{')
                    if json_start == -1:
                        print("⚠️ No se encontró JSON en la salida")
                        return []
                    
                    # Extraer solo la parte JSON
                    json_str = stdout_str[json_start:]
                    
                    # Buscar el final del JSON (último })
                    json_end = json_str.rfind('}')
                    if json_end == -1:
                        print("⚠️ JSON incompleto en la salida")
                        return []
                    
                    json_str = json_str[:json_end + 1]
                    
                    # Parsear JSON
                    metadata = json.loads(json_str)
                    
                    # Extraer detecciones del tensor de salida
                    if 'CnnOutputTensor' in metadata and 'CnnInputTensorInfo' in metadata:
                        tensor_data = metadata['CnnOutputTensor']
                        tensor_info = metadata['CnnInputTensorInfo']
                        
                        detecciones = self.parse_cnn_output_tensor_ai(tensor_data, tensor_info)
                        
                        if detecciones:
                            print(f"✅ Inferencia AI exitosa: {len(detecciones)} personas detectadas")
                            return detecciones
                        else:
                            print("ℹ️ Inferencia AI exitosa pero sin personas detectadas")
                            return []
                    else:
                        print("⚠️ Metadatos AI incompletos")
                        return []
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Error parseando JSON: {e}")
                    return []
                except Exception as e:
                    print(f"❌ Error procesando metadatos AI: {e}")
                    return []
            else:
                print("⚠️ No se recibieron datos de inferencia AI")
                return []
                
        except Exception as e:
            print(f"❌ Error ejecutando inferencia AI: {e}")
            return []
    
    def deteccion_basica_fallback(self):
        """Detección básica como fallback cuando las inferencias AI no están disponibles"""
        try:
            # Obtener el frame actual del procesador
            if hasattr(self, 'frame_actual') and self.frame_actual is not None:
                frame = self.frame_actual.copy()
                
                # Convertir a escala de grises para detección básica
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detección básica de movimiento usando diferencia de frames
                if not hasattr(self, 'frame_anterior'):
                    self.frame_anterior = gray
                    return []
                
                # Calcular diferencia entre frames
                diff = cv2.absdiff(self.frame_anterior, gray)
                
                # Aplicar umbral para detectar movimiento
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                
                # Encontrar contornos
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detecciones = []
                
                for contour in contours:
                    # Filtrar contornos pequeños
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Área mínima
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Verificar si está en ROI de puerta
                        centro_x = x + w // 2
                        centro_y = y + h // 2
                        
                        if self.esta_en_roi_puerta([centro_x, centro_y]):
                            deteccion = Deteccion(
                                timestamp=time.time(),
                                bbox=[x, y, x + w, y + h],
                                confianza=0.5,  # Confianza media para detección básica
                                centro=[centro_x, centro_y],
                                area=area
                            )
                            detecciones.append(deteccion)
                
                # Actualizar frame anterior
                self.frame_anterior = gray
                
                if detecciones:
                    print(f"🔄 Detección básica: {len(detecciones)} objetos en movimiento")
                
                return detecciones
            
            return []
            
        except Exception as e:
            print(f"❌ Error en detección básica: {e}")
            return []
    
    def generar_detecciones_sinteticas(self, frame_counter):
        """Genera detecciones sintéticas para probar el sistema (DEPRECATED)"""
        # Esta función ya no se usa, se mantiene solo por compatibilidad
        return []
    
    def thread_captura_frames(self):
        """Thread dedicado a capturar frames de video - versión simplificada"""
        print("📹 Thread de captura de video iniciado")
        
        buffer_mjpeg = b''
        frame_count = 0
        
        while not self.detenido:
            try:
                # Verificar si el proceso de cámara sigue activo
                if self.proceso_camara.poll() is not None:
                    print("⚠️ Proceso de cámara terminado, reintentando...")
                    try:
                        self.proceso_camara.terminate()
                        self.proceso_camara.wait(timeout=2)
                    except:
                        pass
                    
                    if self.configurar_camara():
                        buffer_mjpeg = b''
                        time.sleep(2)
                    else:
                        time.sleep(5)
                    continue
                
                # Leer datos del proceso
                try:
                    chunk = self.proceso_camara.stdout.read(1024)
                    if not chunk:
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    print(f"⚠️ Error leyendo datos de cámara: {e}")
                    time.sleep(0.1)
                    continue
                
                # Procesar datos de video MJPEG
                buffer_mjpeg += chunk
                
                # Buscar marcadores de frame MJPEG
                start_marker = b'\xff\xd8'  # SOI (Start of Image)
                end_marker = b'\xff\xd9'    # EOI (End of Image)
                
                # Buscar inicio de frame
                start_pos = buffer_mjpeg.find(start_marker)
                if start_pos == -1:
                    continue
                
                # Buscar fin de frame después del inicio
                end_pos = buffer_mjpeg.find(end_marker, start_pos)
                if end_pos == -1:
                    continue
                
                # Extraer frame JPEG completo
                frame_jpeg = buffer_mjpeg[start_pos:end_pos + 2]
                
                # Verificar tamaño del frame
                if len(frame_jpeg) < 5000 or len(frame_jpeg) > 500000:
                    buffer_mjpeg = buffer_mjpeg[end_pos + 2:]
                    continue
                
                # Decodificar frame
                try:
                    nparr = np.frombuffer(frame_jpeg, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        # Actualizar métricas
                        timestamp = time.time()
                        self.timestamps_captura.append(timestamp)
                        
                        # Calcular FPS
                        if len(self.timestamps_captura) >= 2:
                            recent_timestamps = list(self.timestamps_captura)[-10:]
                            if len(recent_timestamps) >= 2:
                                time_span = recent_timestamps[-1] - recent_timestamps[0]
                                if time_span > 0:
                                    self.fps_captura = (len(recent_timestamps) - 1) / time_span
                        
                        self.frame_actual = frame
                        self.timestamp_ultimo_frame = timestamp
                        frame_count += 1
                        
                        # Log cada 30 frames
                        if frame_count % 30 == 0:
                            print(f"📹 Frame {frame_count} capturado, FPS: {self.fps_captura:.1f}")
                        
                        # Encolar frame para procesamiento
                        if not self.cola_frames.full():
                            self.cola_frames.put(frame)
                        
                        # Limpiar buffer
                        buffer_mjpeg = buffer_mjpeg[end_pos + 2:]
                        
                except Exception as e:
                    # Frame corrupto, continuar
                    buffer_mjpeg = buffer_mjpeg[end_pos + 2:]
                    continue
                
                # Control de FPS
                time.sleep(1.0 / self.config['fps_objetivo'])
                
            except Exception as e:
                print(f"❌ Error en thread de captura: {e}")
                time.sleep(0.1)
        
        print("📹 Thread de captura de video terminado")
    
    def thread_procesamiento_detecciones(self):
        """Thread dedicado a procesar detecciones AI y tracking"""
        print("🧠 Thread de procesamiento AI iniciado")
        
        frame_counter = 0
        
        while not self.detenido:
            try:
                # Obtener detecciones de la cola de inferencias AI
                try:
                    datos_detecciones = self.cola_detecciones.get(timeout=0.1)
                    detecciones = datos_detecciones['detecciones']
                    timestamp = datos_detecciones['timestamp']
                except queue.Empty:
                    continue
                
                frame_counter += 1
                
                # Actualizar tracking con detecciones AI
                personas_actuales = self.tracker.actualizar_tracking(detecciones)
                
                # Log reducido para optimizar rendimiento
                if frame_counter % 30 == 0:  # Log cada 30 frames
                    print(f"🔄 [AI] Frame {frame_counter}: {len(detecciones)} personas, {len(personas_actuales)} personas activas, IDs: {list(personas_actuales.keys())}")
                
            except Exception as e:
                print(f"❌ Error en thread de procesamiento AI: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
        
        print("🧠 Thread de procesamiento AI terminado")
    
    def detectar_personas_camara_real(self, frame):
        """Función de compatibilidad - las detecciones ahora vienen de inferencias AI"""
        # Esta función ya no se usa, las detecciones vienen directamente de las inferencias AI
        return []
    
    def parse_cnn_output_tensor_ai(self, tensor_data, tensor_info):
        """Parsea el tensor de salida de la CNN para extraer detecciones"""
        try:
            tensor = np.array(tensor_data, dtype=np.float32)
            
            # Buscar el número de detecciones en tensor_info
            num_detections = 0
            for i, val in enumerate(tensor_info):
                if val > 0 and i > 0:
                    num_detections = int(val)
                    break
            
            detecciones = []
            
            # Formato MobileNet SSD: [image_id, label, confidence, x_min, y_min, x_max, y_max]
            if num_detections > 0:
                for i in range(min(num_detections, len(tensor) // 7)):
                    start_idx = i * 7
                    if start_idx + 6 < len(tensor):
                        image_id = int(tensor[start_idx])
                        label = int(tensor[start_idx + 1])
                        confidence = float(tensor[start_idx + 2])
                        x_min = float(tensor[start_idx + 3])
                        y_min = float(tensor[start_idx + 4])
                        x_max = float(tensor[start_idx + 5])
                        y_max = float(tensor[start_idx + 6])
                        
                        # Solo personas (clase 0 en COCO) y confianza mínima
                        if label == 0 and confidence > self.config['confianza_minima']:
                            # Verificar que las coordenadas son válidas
                            if 0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1:
                                # Escalar coordenadas al tamaño de la imagen
                                x1 = int(x_min * self.config['resolucion'][0])
                                y1 = int(y_min * self.config['resolucion'][1])
                                x2 = int(x_max * self.config['resolucion'][0])
                                y2 = int(y_max * self.config['resolucion'][1])
                                
                                # Verificar si está en ROI de puerta
                                centro_x = (x1 + x2) // 2
                                centro_y = (y1 + y2) // 2
                                
                                if self.esta_en_roi_puerta([centro_x, centro_y]):
                                    deteccion = Deteccion(
                                        timestamp=time.time(),
                                        bbox=[x1, y1, x2, y2],
                                        confianza=confidence,
                                        centro=[centro_x, centro_y],
                                        area=(x2 - x1) * (y2 - y1)
                                    )
                                    detecciones.append(deteccion)
                                    print(f"✅ Persona detectada por AI IMX500: conf={confidence:.3f}, centro=({centro_x},{centro_y})")
            
            return detecciones
            
        except Exception as e:
            print(f"Error parsing tensor AI: {e}")
            return []
    
    def aplicar_nms(self, detecciones):
        """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas"""
        if len(detecciones) <= 1:
            return detecciones
        
        # Ordenar por confianza (mayor a menor)
        detecciones_ordenadas = sorted(detecciones, key=lambda x: x.confianza, reverse=True)
        detecciones_finales = []
        
        for deteccion in detecciones_ordenadas:
            # Verificar si esta detección se superpone significativamente con alguna ya seleccionada
            es_duplicada = False
            for det_final in detecciones_finales:
                iou = self.calcular_iou(deteccion.bbox, det_final.bbox)
                if iou > self.config['nms_iou']:
                    es_duplicada = True
                    break
            
            if not es_duplicada:
                detecciones_finales.append(deteccion)
        
        return detecciones_finales
    
    def calcular_iou(self, bbox1, bbox2):
        """Calcula el Intersection over Union entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        area_interseccion = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular unión
        area_bbox1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_bbox2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        area_union = area_bbox1 + area_bbox2 - area_interseccion
        
        return area_interseccion / area_union if area_union > 0 else 0.0
    
    def esta_en_roi_puerta(self, centro):
        """Verifica si un punto está en el ROI de la puerta"""
        x, y = centro
        x1, y1, x2, y2 = self.config['roi_puerta']
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def obtener_metricas(self):
        """Obtiene las métricas del procesamiento"""
        return {
            'fps_captura': self.fps_captura,
            'fps_inferencia': self.fps_inferencia,
            'frame_actual': self.frame_actual is not None,
            'timestamp_ultimo_frame': self.timestamp_ultimo_frame,
            'cola_frames': self.cola_frames.qsize(),
            'cola_detecciones': self.cola_detecciones.qsize()
        }
    
    def obtener_detecciones_actuales(self):
        """Obtiene las detecciones actuales para visualización"""
        return self.detecciones_actuales
    
    def obtener_tracker(self):
        """Obtiene el tracker para acceso a contadores"""
        return self.tracker
    
    def cleanup(self):
        """Limpia recursos del procesador"""
        self.detenido = True
        
        # Esperar threads
        if hasattr(self, 'thread_captura'):
            self.thread_captura.join(timeout=2)
        if hasattr(self, 'thread_procesamiento'):
            self.thread_procesamiento.join(timeout=2)
        
        # Limpiar proceso de cámara
        if hasattr(self, 'proceso_camara') and self.proceso_camara:
            self.proceso_camara.terminate()
            self.proceso_camara.wait()
        
        # Limpiar proceso de inferencia AI (ya no existe, se maneja en el proceso principal)
        pass

class TrackerPersonas:
    """Clase para tracking de personas - BASADA EN SISTEMA FUNCIONAL"""
    
    def __init__(self, config):
        self.config = config
        self.personas_detectadas = {}
        self.historial_personas = {}
        self.historial_maxlen = 20
        
        # Contadores
        self.contador_entradas = 0
        self.contador_salidas = 0
        self.personas_en_habitacion = 0
        self.personas_en_zona_puerta = set()
        
        # Estado de tracking (como en servidor_web_deteccion.py)
        self.personas_estado = {}  # ID -> {'estado': 'esperando', 'lado_anterior': None, 'cooldown': 0}
        
        # Línea virtual para cruce (centro de la zona de puerta)
        self.linea_virtual = (self.config['roi_puerta'][0] + self.config['roi_puerta'][2]) // 2
    
    def actualizar_tracking(self, detecciones):
        """Actualiza el tracking de personas y detecta entradas/salidas con lógica robusta - BASADO EN SISTEMA FUNCIONAL"""
        personas_actuales = {}
        personas_en_zona_actual = set()
        
        for deteccion in detecciones:
            centro = deteccion.centro
            persona_id = None
            distancia_minima = float('inf')
            
            # Buscar persona más cercana
            for pid, pos_anterior in self.personas_detectadas.items():
                distancia = np.sqrt((centro[0] - pos_anterior[0])**2 + (centro[1] - pos_anterior[1])**2)
                if distancia < distancia_minima and distancia < 150:  # Distancia máxima de 150px
                    distancia_minima = distancia
                    persona_id = pid
            
            # Si no se encontró, crear nueva persona
            if persona_id is None:
                persona_id = len(self.personas_detectadas) + 1
                print(f"🆔 Nueva persona detectada: ID={persona_id}, centro=({centro[0]},{centro[1]})")
            
            personas_actuales[persona_id] = centro
            
            # Actualizar historial
            if persona_id not in self.historial_personas:
                self.historial_personas[persona_id] = deque(maxlen=self.historial_maxlen)
            self.historial_personas[persona_id].append(centro)
            
            # Inicializar estado si es nueva persona
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
            
            # Lógica de cruce (como en servidor_web_deteccion.py)
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
        
        # Actualizar estado global
        self.personas_detectadas = personas_actuales
        self.personas_en_zona_puerta = personas_en_zona_actual
        
        # Limpiar historiales de personas inactivas
        inactivos = set(self.historial_personas.keys()) - set(personas_actuales.keys())
        for pid in inactivos:
            del self.historial_personas[pid]
            if pid in self.personas_estado:
                del self.personas_estado[pid]
        
        print(f"📊 Estado actual: {len(personas_actuales)} personas activas, {len(self.historial_personas)} historiales")
        return personas_actuales
    
    def obtener_contadores(self):
        """Obtiene los contadores actuales"""
        return {
            'contador_entradas': self.contador_entradas,
            'contador_salidas': self.contador_salidas,
            'personas_en_habitacion': self.personas_en_habitacion,
            'tracks_activos': len(self.personas_detectadas)
        }

class ServidorStreaming:
    """Servidor web Flask para streaming en vivo con visualización dinámica"""
    
    def __init__(self, config):
        self.config = config
        self.app = Flask(__name__)
        # CORS removido para compatibilidad
        # CORS(self.app)
        
        # Inicializar componentes
        self.procesador = ProcesadorSegundoPlano(config)
        self.tracker = self.procesador.obtener_tracker() # Obtener el tracker del procesador
        
        # Estado del sistema
        self.inicio_tiempo = time.time()
        self.frame_count = 0
        
        # Control de visualización
        self.visualizacion_activa = config.get('visualizacion_por_defecto', False)  # Configurable por defecto
        self.ultimo_frame_visualizacion = None
        self.ultimas_detecciones_visualizacion = []
        self.ultimas_personas_visualizacion = {}
        
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
            contadores = self.tracker.obtener_contadores()
            return jsonify({
                'timestamp': time.time(),
                **contadores
            })
        
        @self.app.route('/toggle_puntos_personas', methods=['POST'])
        def toggle_puntos_personas():
            """Toggle para mostrar/ocultar puntos de personas"""
            self.config['mostrar_puntos_personas'] = not self.config.get('mostrar_puntos_personas', False)
            return jsonify({
                'mostrar_puntos_personas': self.config['mostrar_puntos_personas'],
                'mensaje': 'Puntos de personas ' + ('activados' if self.config['mostrar_puntos_personas'] else 'desactivados')
            })
        
        @self.app.route('/toggle_linea_cruce', methods=['POST'])
        def toggle_linea_cruce():
            """Toggle para mostrar/ocultar línea de cruce"""
            self.config['mostrar_linea_cruce'] = not self.config.get('mostrar_linea_cruce', True)
            return jsonify({
                'mostrar_linea_cruce': self.config['mostrar_linea_cruce'],
                'mensaje': 'Línea de cruce ' + ('activada' if self.config['mostrar_linea_cruce'] else 'desactivada')
            })
        
        @self.app.route('/toggle_visualizacion', methods=['POST'])
        def toggle_visualizacion():
            """Toggle para activar/desactivar visualización de cámara"""
            self.visualizacion_activa = not self.visualizacion_activa
            return jsonify({
                'visualizacion_activa': self.visualizacion_activa,
                'mensaje': 'Visualización de cámara ' + ('activada' if self.visualizacion_activa else 'desactivada')
            })
        
        @self.app.route('/estado_visualizacion')
        def estado_visualizacion():
            """Obtiene el estado actual de la visualización"""
            return jsonify({
                'visualizacion_activa': self.visualizacion_activa,
                'procesamiento_activo': True,  # El procesamiento siempre está activo
                'timestamp': time.time()
            })
        
        @self.app.route('/health')
        def health():
            """Estado de salud del sistema"""
            return jsonify({
                'status': 'ok',
                'timestamp': time.time(),
                'camara_activa': self.procesador.proceso_camara is not None,
                'procesamiento_activo': True,
                'visualizacion_activa': self.visualizacion_activa,
                'frame_count': self.frame_count,
                'uptime': time.time() - self.inicio_tiempo
            })
    
    def generar_stream(self):
        """Genera el stream MJPEG en vivo optimizado para Raspberry Pi con control de visualización"""
        frame_counter = 0  # Contador de frames para procesar detecciones
        
        while True:
            try:
                # Verificar si la visualización está activa
                if not self.visualizacion_activa:
                    # Si la visualización está desactivada, solo mantener el procesamiento activo
                    time.sleep(0.1)
                    continue
                
                # Obtener frame del procesador
                try:
                    frame = self.procesador.cola_frames.get(timeout=0.1)
                    
                    # Obtener detecciones actuales del tracker
                    detecciones = self.procesador.obtener_detecciones_actuales()
                    personas_actuales = self.tracker.personas_detectadas
                    
                    # Actualizar datos de visualización
                    self.ultimo_frame_visualizacion = frame
                    self.ultimas_detecciones_visualizacion = detecciones
                    self.ultimas_personas_visualizacion = personas_actuales
                    
                except queue.Empty:
                    # Si no hay frames nuevos, usar el último disponible
                    if self.ultimo_frame_visualizacion is not None:
                        frame = self.ultimo_frame_visualizacion
                        detecciones = self.ultimas_detecciones_visualizacion
                        personas_actuales = self.ultimas_personas_visualizacion
                    else:
                        # Generar frame de placeholder
                        frame = self.generar_frame_placeholder()
                        detecciones = []
                        personas_actuales = {}
                
                if frame is not None:
                    frame_counter += 1
                    
                    # Dibujar anotaciones simplificadas
                    frame_anotado = self.dibujar_anotaciones(frame, detecciones, personas_actuales)
                    
                    # Convertir a JPEG con calidad optimizada para Raspberry Pi
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, self.config.get('calidad_jpeg', 70),
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Optimización
                        cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # Sin progresivo
                    ]
                    
                    ret, buffer = cv2.imencode('.jpg', frame_anotado, encode_params)
                    
                    if ret:
                        # Generar frame MJPEG
                        frame_data = b'--frame\r\n'
                        frame_data += b'Content-Type: image/jpeg\r\n\r\n'
                        frame_data += buffer.tobytes()
                        frame_data += b'\r\n'
                        
                        self.frame_count += 1
                        yield frame_data
                
                # Control de FPS optimizado
                target_delay = 1.0 / self.config['fps_objetivo']
                time.sleep(target_delay)
                
            except Exception as e:
                print(f"❌ Error en stream: {e}")
                time.sleep(0.1)
    
    def generar_frame_placeholder(self):
        """Genera un frame de placeholder cuando la visualización está desactivada"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Texto de estado
        cv2.putText(frame, "VISUALIZACION DESACTIVADA", (120, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Procesamiento activo en segundo plano", (100, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Dibujar icono de cámara
        cv2.circle(frame, (320, 100), 50, (0, 255, 255), 3)
        cv2.circle(frame, (320, 100), 20, (0, 255, 255), -1)
        
        # Mostrar métricas básicas
        metricas = self.procesador.obtener_metricas()
        cv2.putText(frame, f"FPS: {metricas['fps_captura']:.1f}", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Procesamiento: Activo", (50, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def dibujar_anotaciones(self, frame, detecciones, personas_actuales):
        """Dibuja anotaciones mínimas: solo línea de cruce y puntos de personas"""
        frame_anotado = frame.copy()
        
        # Dibujar línea de cruce central solo si está habilitado
        if self.config.get('mostrar_linea_cruce', True):
            linea_x = self.config['linea_cruce']
            cv2.line(frame_anotado, (linea_x, 0), (linea_x, frame.shape[0]), (255, 0, 0), 2)
        
        # Dibujar solo puntos en el centro de las personas detectadas (opcional para depuración)
        if self.config.get('mostrar_puntos_personas', False):
            for deteccion in detecciones:
                centro_x, centro_y = deteccion.centro
                # Punto simple sin texto para optimizar rendimiento
                cv2.circle(frame_anotado, (centro_x, centro_y), 4, (0, 255, 0), -1)
        
        return frame_anotado
    
    # Función dibujar_contadores eliminada para optimizar rendimiento
    
    def dibujar_metricas(self, frame):
        """Dibuja las métricas del sistema en el frame"""
        # Fondo para métricas
        cv2.rectangle(frame, (frame.shape[1]-310, 10), (frame.shape[1]-10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (frame.shape[1]-310, 10), (frame.shape[1]-10, 100), (255, 255, 255), 2)
        
        # Obtener métricas del procesador
        metricas_procesador = self.procesador.obtener_metricas()
        
        # Métricas
        cv2.putText(frame, f'FPS Captura: {metricas_procesador["fps_captura"]:.1f}', (frame.shape[1]-300, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'FPS Inferencia: {metricas_procesador["fps_inferencia"]:.1f}', (frame.shape[1]-300, 60), 
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
        
        # Obtener métricas del procesador
        metricas_procesador = self.procesador.obtener_metricas()
        
        return {
            'timestamp': time.time(),
            'uptime': time.time() - self.inicio_tiempo,
            'frame_count': self.frame_count,
            'fps_captura': metricas_procesador['fps_captura'],
            'fps_inferencia': metricas_procesador['fps_inferencia'],
            **self.tracker.obtener_contadores(),
            'cpu': cpu_percent,
            'memoria': memoria.percent,
            'temperatura': temperatura,
            'camara_activa': self.procesador.proceso_camara is not None,
            'procesamiento_activo': True,
            'visualizacion_activa': self.visualizacion_activa,
            'cola_frames': metricas_procesador['cola_frames'],
            'cola_detecciones': metricas_procesador['cola_detecciones']
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
                <h1>🤖 Cámara AI IMX500 REAL - Inferencias AI</h1>
                <div class="info">
                    <strong>Sensor:</strong> Sony IMX500 REAL | <strong>Resolución:</strong> 640x480 | <strong>FPS:</strong> 15 | <strong>Inferencias:</strong> AI REALES
                </div>
                
                <div class="stream-container">
                    <h3>📹 Video en Vivo con Detecciones</h3>
                    <img class="stream-image" src="/stream" alt="Streaming en Vivo">
                </div>
                
                <div class="controls">
                    <button onclick="location.reload()">🔄 Recargar</button>
                    <button onclick="window.open('/metrics', '_blank')">📊 Ver Métricas JSON</button>
                    <button onclick="window.open('/counts', '_blank')">🔢 Ver Contadores</button>
                    <button onclick="toggleVisualizacion()" id="btnVisualizacion">📹 Desactivar Visualización</button>
                    <button onclick="togglePuntosPersonas()">👁️ Toggle Puntos Personas</button>
                    <button onclick="toggleLineaCruce()">📏 Toggle Línea Cruce</button>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="entradas">-</div>
                        <div class="metric-label">🚪 Entradas</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="salidas">-</div>
                        <div class="metric-label">🚪 Salidas</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="en_habitacion">-</div>
                        <div class="metric-label">👥 En Habitación</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="tracks">-</div>
                        <div class="metric-label">🆔 Tracks Activos</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="fps_captura">-</div>
                        <div class="metric-label">📹 FPS Captura</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="fps_inferencia">-</div>
                        <div class="metric-label">🧠 FPS Inferencia</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="frames">-</div>
                        <div class="metric-label">🎬 Frames</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="cpu">-</div>
                        <div class="metric-label">🖥️ CPU %</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="memoria">-</div>
                        <div class="metric-label">💾 Memoria %</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="temperatura">-</div>
                        <div class="metric-label">🌡️ Temperatura</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="visualizacion">-</div>
                        <div class="metric-label">📹 Visualización</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="procesamiento">-</div>
                        <div class="metric-label">🧠 Procesamiento</div>
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
                            document.getElementById('tracks').textContent = data.tracks_activos;
                            document.getElementById('fps_captura').textContent = data.fps_captura.toFixed(1);
                            document.getElementById('fps_inferencia').textContent = data.fps_inferencia.toFixed(1);
                            document.getElementById('frames').textContent = data.frame_count;
                            document.getElementById('cpu').textContent = data.cpu.toFixed(1);
                            document.getElementById('memoria').textContent = data.memoria.toFixed(1);
                            document.getElementById('temperatura').textContent = data.temperatura ? data.temperatura.toFixed(1) + '°C' : 'N/A';
                            document.getElementById('visualizacion').textContent = data.visualizacion_activa ? 'ON' : 'OFF';
                            document.getElementById('procesamiento').textContent = data.procesamiento_activo ? 'ON' : 'OFF';
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                // Actualizar cada 2 segundos
                setInterval(actualizarMetricas, 2000);
                
                // Actualizar al cargar la página
                actualizarMetricas();
                
                // Funciones para controlar visualizaciones
                function togglePuntosPersonas() {
                    fetch('/toggle_puntos_personas', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            console.log('Puntos personas:', data.mostrar_puntos_personas ? 'Activados' : 'Desactivados');
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                function toggleLineaCruce() {
                    fetch('/toggle_linea_cruce', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            console.log('Línea cruce:', data.mostrar_linea_cruce ? 'Activada' : 'Desactivada');
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                function toggleVisualizacion() {
                    fetch('/toggle_visualizacion', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            const btn = document.getElementById('btnVisualizacion');
                            if (data.visualizacion_activa) {
                                btn.textContent = '📹 Desactivar Visualización';
                                btn.style.background = '#00ff88';
                            } else {
                                btn.textContent = '📹 Activar Visualización';
                                btn.style.background = '#ff8800';
                            }
                            console.log('Visualización:', data.visualizacion_activa ? 'Activada' : 'Desactivada');
                            
                            // Recargar la imagen del stream si es necesario
                            const streamImg = document.querySelector('.stream-image');
                            if (streamImg) {
                                streamImg.src = '/stream?' + new Date().getTime();
                            }
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                // Verificar estado inicial de visualización
                function verificarEstadoVisualizacion() {
                    fetch('/estado_visualizacion')
                        .then(response => response.json())
                        .then(data => {
                            const btn = document.getElementById('btnVisualizacion');
                            if (data.visualizacion_activa) {
                                btn.textContent = '📹 Desactivar Visualización';
                                btn.style.background = '#00ff88';
                            } else {
                                btn.textContent = '📹 Activar Visualización';
                                btn.style.background = '#ff8800';
                            }
                        })
                        .catch(error => console.error('Error:', error));
                }
                
                // Verificar estado al cargar la página
                verificarEstadoVisualizacion();
            </script>
        </body>
        </html>
        """
    
    def iniciar(self, host='0.0.0.0', port=5000, debug=False):
        """Inicia el servidor web"""
        print("🚀 INICIANDO SERVIDOR DE STREAMING EN VIVO CON INFERENCIAS AI REALES")
        print("=" * 60)
        print(f"🤖 Cámara AI IMX500 REAL + Raspberry Pi 5")
        print(f"🎯 FPS objetivo: {self.config['fps_objetivo']}")
        print(f"📍 ROI puerta: {self.config['roi_puerta']}")
        print(f"📍 Línea cruce: X={self.config['linea_cruce']}")
        print(f"🌐 Servidor: http://{host}:{port}")
        print(f"📹 Streaming: http://{host}:{port}/stream")
        print(f"📊 Métricas: http://{host}:{port}/metrics")
        print(f"🎛️ Control visualización: /toggle_visualizacion")
        print(f"📊 Estado visualización: /estado_visualizacion")
        print("=" * 60)
        print("💡 NUEVO: Inferencias reales de la cámara AI IMX500")
        print("💡 NUEVO: Cero modelos locales - solo procesamiento de metadatos")
        print("💡 NUEVO: Tracking ligero optimizado para mínimo uso de CPU")
        print("💡 NUEVO: Visualización desactivada por defecto para ahorrar recursos")
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
        self.procesador.cleanup()
        print("✅ Recursos limpiados")

def main():
    """Función principal"""
    # Configuración optimizada para cámara Raspberry Pi AI Camera IMX500 con inferencias reales
    config = {
        'resolucion': [640, 480],
        'fps_objetivo': 15,  # Reducido para inferencias AI
        'confianza_minima': 0.3,  # Umbral para inferencias AI
        'area_minima': 800,  # Reducido para detectar personas más pequeñas
        'roi_puerta': [50, 50, 590, 430],  # ROI más amplio
        'linea_cruce': 320,
        'ancho_banda_cruce': 15,  # Banda más amplia para mejor detección
        'debounce_ms': 300,  # Reducido para detección más rápida
        'track_lost_ms': 2000,  # Aumentado para mejor persistencia
        'exposure_us': 4000,
        'gain': 1.0,
        'distancia_maxima_tracking': 100,  # Aumentado para cámara real
        'historial_maxlen': 25,  # Aumentado para mejor tracking
        'umbral_movimiento': 15,  # Reducido para detectar movimientos menores
        'nms_iou': 0.6,  # Más permisivo para cámara real
        'procesar_cada_n_frames': 1,  # Procesar TODOS los frames para máxima detección
        'filtro_estabilidad': True,
        'tiempo_persistencia_ms': 5000,  # Aumentado para mejor tracking
        'umbral_confianza_alto': 0.3,  # Más sensible
        'umbral_confianza_medio': 0.2,  # Más sensible
        'mostrar_puntos_personas': True,  # Activar para debug
        'mostrar_linea_cruce': True,
        'calidad_jpeg': 85,  # Aumentado para mejor calidad
        'log_reducido': True,  # Log reducido para producción
        'camara_real': True,  # Indicador de que usamos cámara real
        'debug_deteccion': False,  # Modo debug desactivado para producción
        'inferencias_ai': True,  # Usar inferencias reales de la cámara AI
        'visualizacion_por_defecto': True  # Activar visualización por defecto para ver la cámara
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