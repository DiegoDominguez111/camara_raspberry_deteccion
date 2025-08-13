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
            
            # Comando optimizado para streaming con parámetros válidos
            comando_config = [
                '/usr/bin/rpicam-vid',
                '--width', str(self.config['resolucion'][0]),
                '--height', str(self.config['resolucion'][1]),
                '--framerate', str(self.config['fps_objetivo']),
                '--nopreview',
                '--output', '-',  # Salida a stdout
                '--codec', 'mjpeg',  # Código más rápido
                '--inline',  # Sin buffering
                '--flush',  # Flush inmediato
                '--awb', 'auto',  # Balance de blancos automático
                '--metering', 'centre',  # Medición central
                '--denoise', 'cdn_off',  # Desactivar denoise
                '--roi', '0.0,0.0,1.0,1.0',  # ROI completo
                '--brightness', '0.0',  # Brillo neutro
                '--contrast', '1.0',  # Contraste neutro
                '--saturation', '1.0',  # Saturación neutra
                '--sharpness', '0.0'  # Sin sharpening
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
    
    def leer_frame(self):
        """Lee un frame de la cámara usando un enfoque más robusto para MJPEG"""
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
                chunk = self.proceso_camara.stdout.read(1024)
                if not chunk:
                    break
                
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
                
                # Verificar tamaño mínimo del frame
                if len(frame_jpeg) < 5000:  # Frame muy pequeño, probablemente corrupto
                    buffer_mjpeg = buffer_mjpeg[end_pos + 2:]
                    continue
                
                # Decodificar a numpy array
                nparr = np.frombuffer(frame_jpeg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    # Actualizar métricas
                    timestamp = time.time()
                    self.timestamps_captura.append(timestamp)
                    
                    # Calcular FPS de manera más robusta
                    if len(self.timestamps_captura) >= 2:
                        # Usar solo los últimos 10 timestamps para FPS más estable
                        recent_timestamps = list(self.timestamps_captura)[-10:]
                        if len(recent_timestamps) >= 2:
                            time_span = recent_timestamps[-1] - recent_timestamps[0]
                            if time_span > 0:
                                self.fps_captura = (len(recent_timestamps) - 1) / time_span
                    
                    self.frame_actual = frame
                    self.timestamp_ultimo_frame = timestamp
                    
                    # Limpiar buffer para el siguiente frame
                    buffer_mjpeg = buffer_mjpeg[end_pos + 2:]
                    return frame
                
                # Limpiar buffer hasta el fin del frame actual
                buffer_mjpeg = buffer_mjpeg[end_pos + 2:]
            
            # Si no se pudo leer un frame válido, generar uno de placeholder
            if self.frame_actual is None:
                return self.generar_frame_placeholder()
            
            return self.frame_actual
            
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
        """Detecta personas usando HOG detector optimizado para Raspberry Pi con FPS estable"""
        try:
            # Usar HOG detector como detector principal
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Reducir resolución para mejor rendimiento si es necesario
            frame_procesar = frame
            scale_factor = 1.0
            
            # Aplicar redimensionamiento para mejor rendimiento
            if frame.shape[0] > 480 or frame.shape[1] > 640:
                scale_factor = 0.4  # Reducir más para mejor rendimiento
                frame_procesar = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            elif frame.shape[0] > 400 or frame.shape[1] > 500:
                scale_factor = 0.6  # Reducción moderada
                frame_procesar = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            
            # Detectar personas con parámetros optimizados para rendimiento estable
            boxes, weights = hog.detectMultiScale(
                frame_procesar, 
                winStride=(32, 32),    # Aumentado para mejor rendimiento
                padding=(16, 16),      # Padding aumentado
                scale=1.2,             # Escala más eficiente
                hitThreshold=0         # Sin umbral de hit
            )
            
            detecciones = []
            if len(boxes) > 0:
                print(f"🔍 HOG detectó {len(boxes)} candidatos")
            
            for (x, y, w, h), weight in zip(boxes, weights):
                # Filtrar por confianza mínima más estricta
                if weight > self.config['confianza_minima'] * 1.2:  # 20% más estricto
                    # Filtrar por área mínima
                    area = w * h
                    if area > self.config['area_minima']:
                        # Ajustar coordenadas si se redimensionó
                        if frame_procesar is not frame:
                            scale_inv = 1.0 / scale_factor
                            x = int(x * scale_inv)
                            y = int(y * scale_inv)
                            w = int(w * scale_inv)
                            h = int(h * scale_inv)
                        
                        # Aplicar ROI de puerta más estricto
                        centro_x = x + w // 2
                        centro_y = y + h // 2
                        
                        if self.esta_en_roi_puerta([centro_x, centro_y]):
                            deteccion = Deteccion(
                                timestamp=time.time(),
                                bbox=[x, y, x + w, y + h],
                                confianza=float(weight),
                                centro=[centro_x, centro_y],
                                area=area
                            )
                            detecciones.append(deteccion)
                            print(f"✅ Persona detectada: conf={weight:.3f}, centro=({centro_x},{centro_y})")
                        else:
                            print(f"⚠️ Persona fuera del ROI: conf={weight:.3f}, centro=({centro_x},{centro_y})")
                    else:
                        print(f"⚠️ Persona muy pequeña: conf={weight:.3f}, área={area}")
                else:
                    print(f"⚠️ Persona con baja confianza: {weight:.3f}")
            
            # Aplicar NMS para eliminar detecciones duplicadas
            if len(detecciones) > 1:
                detecciones = self.aplicar_nms(detecciones)
                print(f"🎯 Después de NMS: {len(detecciones)} detecciones únicas")
            
            # Actualizar métricas de inferencia de manera más estable
            timestamp = time.time()
            self.timestamps_inferencia.append(timestamp)
            
            # Mantener solo los últimos 15 timestamps para cálculo más estable
            if len(self.timestamps_inferencia) > 15:
                self.timestamps_inferencia = deque(list(self.timestamps_inferencia)[-15:])
            
            if len(self.timestamps_inferencia) >= 2:
                # Calcular FPS de inferencia de manera más robusta
                recent_timestamps = list(self.timestamps_inferencia)[-10:]
                if len(recent_timestamps) >= 2:
                    time_span = recent_timestamps[-1] - recent_timestamps[0]
                    if time_span > 0:
                        self.fps_inferencia = (len(recent_timestamps) - 1) / time_span
            
            self.detecciones_actuales = detecciones
            
            if len(detecciones) > 0:
                print(f"🎯 Total personas detectadas en ROI: {len(detecciones)}")
            else:
                print(f"❌ No se detectaron personas en este frame")
            
            return detecciones
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def filtrar_por_estabilidad(self, detecciones):
        """Filtra detecciones por estabilidad temporal para mejorar persistencia"""
        if not hasattr(self, 'detecciones_historial'):
            self.detecciones_historial = []
        
        # Agregar detecciones actuales al historial
        self.detecciones_historial.append({
            'timestamp': time.time(),
            'detecciones': detecciones
        })
        
        # Mantener solo el historial reciente
        tiempo_limite = time.time() - (self.config.get('tiempo_persistencia_ms', 3000) / 1000.0)
        self.detecciones_historial = [h for h in self.detecciones_historial if h['timestamp'] > tiempo_limite]
        
        if len(self.detecciones_historial) < 2:
            return detecciones
        
        # Filtrar detecciones que aparecen consistentemente
        detecciones_estables = []
        for deteccion in detecciones:
            apariciones = 0
            for historial in self.detecciones_historial:
                for det_hist in historial['detecciones']:
                    # Verificar si es la misma persona (misma área)
                    if self.es_misma_persona(deteccion, det_hist):
                        apariciones += 1
                        break
            
            # Una detección es estable si aparece en al menos 2 frames
            if apariciones >= 2:
                detecciones_estables.append(deteccion)
                print(f"🔒 Detección estable: conf={deteccion.confianza:.3f}, apariciones={apariciones}")
        
        return detecciones_estables
    
    def es_misma_persona(self, det1, det2):
        """Determina si dos detecciones son de la misma persona"""
        # Calcular IoU
        iou = self.calcular_iou(det1.bbox, det2.bbox)
        
        # Calcular distancia entre centros
        distancia = np.sqrt(
            (det1.centro[0] - det2.centro[0])**2 +
            (det1.centro[1] - det2.centro[1])**2
        )
        
        # Es la misma persona si IoU > 0.3 o distancia < 30px
        return iou > 0.3 or distancia < 30
    
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
        
        # Diccionario para almacenar los últimos eventos registrados
        self.eventos_recientes: Dict[int, Dict] = {}
    
    def actualizar_tracking(self, detecciones):
        """Actualiza el tracking de personas de manera ligera para Raspberry Pi"""
        # Buscar tracks más cercanos
        personas_actuales = {}
        
        if len(detecciones) > 0:
            print(f"🔄 Procesando {len(detecciones)} detecciones para tracking...")
        
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
                
                print(f"🆔 Nuevo track creado: ID={track_id}, centro=({deteccion.centro[0]},{deteccion.centro[1]})")
                
                self.tracks[track_id] = Track(
                    id=track_id,
                    detecciones=deque(maxlen=self.config['historial_maxlen']),
                    ultima_posicion=deteccion.centro,
                    ultimo_timestamp=deteccion.timestamp,
                    estado='fuera'
                )
            else:
                print(f"🔄 Track {track_id} actualizado: distancia={distancia_minima:.1f}px")
            
            # Actualizar track
            track = self.tracks[track_id]
            track.detecciones.append(deteccion)
            track.ultima_posicion = deteccion.centro
            track.ultimo_timestamp = deteccion.timestamp
            
            # Marcar track como activo
            track.activo = True
            track.frames_sin_detectar = 0
            
            personas_actuales[track_id] = track
        
        # Actualizar tracks existentes (mantener activos por menos tiempo para ahorrar CPU)
        for tid, track in self.tracks.items():
            if tid not in personas_actuales:
                # Incrementar contador de frames sin detectar
                track.frames_sin_detectar += 1
                
                # Mantener track activo por menos tiempo
                if track.frames_sin_detectar < 3:  # Solo 3 frames
                    track.activo = True
                    personas_actuales[tid] = track
                else:
                    track.activo = False
        
        # Limpiar tracks obsoletos
        tracks_eliminados = self.limpiar_tracks_obsoletos()
        if tracks_eliminados > 0:
            print(f"🧹 {tracks_eliminados} tracks obsoletos eliminados")
        
        # Verificar cruce de línea con tracking ligero
        eventos_detectados = self.verificar_cruce_linea()
        if eventos_detectados > 0:
            print(f"🚪 {eventos_detectados} eventos de cruce detectados")
        
        print(f"📊 Estado actual: {len(personas_actuales)} tracks activos, {len(self.tracks)} total")
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
        return len(tracks_a_eliminar)
    
    def verificar_cruce_linea(self):
        """Verifica cruces de línea con lógica mejorada para conteo preciso"""
        eventos_detectados = 0
        linea_x = self.config['linea_cruce']
        ancho_banda = self.config['ancho_banda_cruce']
        
        for track_id, track in self.tracks.items():
            if not track.activo or len(track.detecciones) < 3:  # Mínimo 3 detecciones para estabilidad
                continue
            
            # Obtener las últimas 3 posiciones para mejor estabilidad
            ultimas_detecciones = list(track.detecciones)[-3:]
            
            # Verificar estabilidad del track
            posiciones_x = [d.centro[0] for d in ultimas_detecciones]
            varianza_x = np.var(posiciones_x)
            
            # Filtrar tracks muy inestables
            if varianza_x > 1000:  # Umbral de estabilidad
                continue
            
            # Verificar si la persona está cruzando la línea
            posicion_actual = ultimas_detecciones[-1].centro
            posicion_anterior = ultimas_detecciones[-2].centro
            posicion_inicial = ultimas_detecciones[0].centro
            
            # Calcular si está dentro de la banda de cruce
            en_banda_actual = abs(posicion_actual[0] - linea_x) <= ancho_banda
            en_banda_anterior = abs(posicion_anterior[0] - linea_x) <= ancho_banda
            
            # Solo procesar si está en la banda de cruce
            if en_banda_actual or en_banda_anterior:
                # Calcular movimiento horizontal total (inicial a final)
                movimiento_total_x = posicion_actual[0] - posicion_inicial[0]
                movimiento_reciente_x = posicion_actual[0] - posicion_anterior[0]
                
                # Verificar si hay movimiento horizontal significativo y consistente
                if (abs(movimiento_total_x) > self.config['umbral_movimiento'] and 
                    abs(movimiento_reciente_x) > self.config['umbral_movimiento'] / 2):
                    
                    # Determinar dirección basada en movimiento total
                    if movimiento_total_x > 0:
                        direccion = 'derecha'  # Entrada
                        evento = 'entrada'
                    else:
                        direccion = 'izquierda'  # Salida
                        evento = 'salida'
                    
                    # Verificar que el track no haya cambiado de dirección recientemente
                    if hasattr(track, 'ultima_direccion') and track.ultima_direccion != direccion:
                        # Cambio de dirección, verificar que sea consistente
                        if abs(movimiento_total_x) < self.config['umbral_movimiento'] * 2:
                            continue  # Movimiento muy corto, ignorar
                    
                    track.ultima_direccion = direccion
                    
                    print(f"🎯 Track {track_id} cruzando línea: {direccion} (mov_total={movimiento_total_x:.1f}, mov_reciente={movimiento_reciente_x:.1f})")
                    
                    # Registrar evento si no se ha registrado recientemente
                    if self.registrar_evento(track_id, evento, posicion_actual):
                        eventos_detectados += 1
                        print(f"✅ Evento {evento} registrado para track {track_id}")
                    else:
                        print(f"⚠️ Evento {evento} ya registrado recientemente para track {track_id}")
        
        return eventos_detectados
    
    def registrar_evento(self, track_id, tipo_evento, posicion):
        """Registra un evento de entrada o salida con anti-rebote mejorado y validación de estado"""
        timestamp_actual = time.time()
        
        # Verificar si este track ya registró un evento recientemente
        if track_id in self.eventos_recientes:
            ultimo_evento = self.eventos_recientes[track_id]
            tiempo_transcurrido = timestamp_actual - ultimo_evento['timestamp']
            
            # Si es el mismo tipo de evento, aplicar debounce estricto
            if ultimo_evento['tipo'] == tipo_evento:
                if tiempo_transcurrido < (self.config['debounce_ms'] / 1000.0):
                    print(f"🔄 Debounce activo para track {track_id}: {tipo_evento} (tiempo: {tiempo_transcurrido:.1f}s)")
                    return False
            else:
                # Si es diferente tipo de evento, verificar coherencia
                if tiempo_transcurrido < (self.config['debounce_ms'] / 1000.0):
                    print(f"🔄 Debounce activo para track {track_id}: cambio de {ultimo_evento['tipo']} a {tipo_evento}")
                    return False
        
        # Verificar coherencia del estado
        track = self.tracks.get(track_id)
        if track:
            if tipo_evento == 'entrada' and track.estado == 'en_habitacion':
                print(f"⚠️ Track {track_id} ya está en habitación, ignorando entrada")
                return False
            elif tipo_evento == 'salida' and track.estado == 'fuera':
                print(f"⚠️ Track {track_id} ya está fuera, ignorando salida")
                return False
        
        # Verificar si hay demasiados eventos del mismo tipo recientemente
        eventos_mismo_tipo = [e for e in self.eventos_recientes.values() 
                             if e['tipo'] == tipo_evento and 
                             timestamp_actual - e['timestamp'] < 3.0]  # Últimos 3 segundos
        
        if len(eventos_mismo_tipo) >= 2:  # Reducido de 3 a 2
            print(f"⚠️ Demasiados eventos {tipo_evento} recientemente ({len(eventos_mismo_tipo)}), aplicando filtro")
            return False
        
        # Registrar el evento
        self.eventos_recientes[track_id] = {
            'tipo': tipo_evento,
            'timestamp': timestamp_actual,
            'posicion': posicion
        }
        
        # Actualizar contadores y estado del track
        if tipo_evento == 'entrada':
            self.contador_entradas += 1
            if track:
                track.estado = 'en_habitacion'
            self.personas_en_habitacion += 1
            print(f"🚪 ENTRADA registrada para track {track_id} - Total: {self.contador_entradas}, En habitación: {self.personas_en_habitacion}")
        elif tipo_evento == 'salida':
            self.contador_salidas += 1
            if track:
                track.estado = 'fuera'
            self.personas_en_habitacion = max(0, self.personas_en_habitacion - 1)
            print(f"🚪 SALIDA registrada para track {track_id} - Total: {self.contador_salidas}, En habitación: {self.personas_en_habitacion}")
        
        # Limpiar eventos antiguos
        self.limpiar_eventos_antiguos()
        
        return True
    
    def limpiar_eventos_antiguos(self):
        """Limpia los eventos registrados que son demasiado antiguos"""
        timestamp_actual = time.time()
        eventos_a_eliminar = []
        for track_id, evento in self.eventos_recientes.items():
            if timestamp_actual - evento['timestamp'] > 10.0: # Mantener eventos por 10 segundos
                eventos_a_eliminar.append(track_id)
        for tid in eventos_a_eliminar:
            del self.eventos_recientes[tid]

class ServidorStreaming:
    """Servidor web Flask para streaming en vivo"""
    
    def __init__(self, config):
        self.config = config
        self.app = Flask(__name__)
        # CORS removido para compatibilidad
        # CORS(self.app)
        
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
        """Genera el stream MJPEG en vivo optimizado para Raspberry Pi"""
        frame_counter = 0  # Contador de frames para procesar detecciones
        detecciones_cache = []  # Cache de detecciones para frames intermedios
        personas_cache = {}  # Cache de tracking para frames intermedios
        
        while True:
            try:
                # Leer frame de la cámara
                frame = self.camara.leer_frame()
                
                if frame is not None:
                    frame_counter += 1
                    
                    # Procesar detecciones cada N frames para optimizar rendimiento y estabilidad
                    procesar_detecciones = frame_counter % self.config.get('procesar_cada_n_frames', 2) == 0
                    
                    # Sistema de cache inteligente: procesar más frames si no hay detecciones
                    if procesar_detecciones:
                        # Procesar detecciones
                        detecciones = self.camara.simular_detecciones(frame)
                        detecciones_cache = detecciones
                        
                        # Actualizar tracking
                        personas_actuales = self.tracker.actualizar_tracking(detecciones)
                        personas_cache = personas_actuales
                        
                        # Log reducido para optimizar rendimiento
                        if frame_counter % 60 == 0:  # Log cada 60 frames para reducir ruido
                            print(f"🔄 Frame {frame_counter}: {len(detecciones)} personas, {len(personas_actuales)} tracks")
                    else:
                        # Usar cache de detecciones y tracking
                        detecciones = detecciones_cache
                        personas_actuales = personas_cache
                        
                        # Si no hay detecciones en cache, procesar ocasionalmente para mantener precisión
                        if len(detecciones_cache) == 0 and frame_counter % 5 == 0:
                            detecciones = self.camara.simular_detecciones(frame)
                            detecciones_cache = detecciones
                            personas_actuales = self.tracker.actualizar_tracking(detecciones)
                            personas_cache = personas_actuales
                    
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
    # Configuración por defecto optimizada para conteo preciso y FPS estable
    config = {
        'resolucion': [640, 480],
        'fps_objetivo': 20,  # Reducido para estabilidad y mejor rendimiento
        'confianza_minima': 0.5,  # Aumentado para mejor precisión
        'area_minima': 2500,  # Aumentado para filtrar ruido
        'roi_puerta': [80, 80, 560, 400],
        'linea_cruce': 320,
        'ancho_banda_cruce': 5,  # Aumentado para mejor detección
        'debounce_ms': 400,  # Aumentado para evitar dobles conteos
        'track_lost_ms': 1000,  # Aumentado para mejor persistencia
        'exposure_us': 4000,
        'gain': 1.0,
        'distancia_maxima_tracking': 60,  # Reducido para mejor precisión
        'historial_maxlen': 15,  # Reducido para mejor rendimiento
        'umbral_movimiento': 20,  # Aumentado para movimientos más claros
        'nms_iou': 0.4,  # Reducido para mejor separación
        'procesar_cada_n_frames': 3,  # Procesar cada 3 frames para mejor rendimiento
        'filtro_estabilidad': True,
        'tiempo_persistencia_ms': 3000,  # Aumentado para mejor tracking
        'umbral_confianza_alto': 0.7,  # Aumentado para mejor precisión
        'umbral_confianza_medio': 0.5,  # Aumentado para mejor precisión
        'mostrar_puntos_personas': False,
        'mostrar_linea_cruce': True,
        'calidad_jpeg': 80,  # Aumentado para mejor calidad
        'log_reducido': True
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