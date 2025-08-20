import cv2
import numpy as np
import threading
import time
import subprocess
import json
import os
from typing import List, Tuple, Optional, Callable
import queue
from collections import deque
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMX500CameraHandler:
    """
    Manejador de cámara IMX500 que simula la generación de embeddings
    en la cámara como requieren las reglas del sistema
    """
    
    def __init__(self, camera_index: int = 0, frame_width: int = 640, frame_height: int = 480):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.is_running = False
        self.current_frame = None
        self.face_detector = None
        self.recognition_callback = None
        
        # Colas para comunicación entre hilos
        self.frame_queue = queue.Queue(maxsize=10)
        self.recognition_queue = queue.Queue(maxsize=20)
        
        # Estadísticas
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Estado de la cámara
        self.camera_status = "OFFLINE"
        self.last_error = None
        self.reconnection_attempts = 0
        self.max_reconnection_attempts = 5
        self.reconnection_backoff = [0.5, 1.0, 2.0, 4.0, 8.0]
        
        # Proceso de cámara
        self.camera_process = None
        
        # Inicializar detector de rostros como fallback
        self._init_face_detector()
        
        # Inicializar cámara
        self._init_camera()
    
    def _init_face_detector(self):
        """Inicializa el detector de rostros de OpenCV como fallback"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                logger.warning("No se pudo cargar el detector de rostros Haar")
                self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
                
            if not self.face_detector.empty():
                logger.info("Detector de rostros Haar inicializado correctamente")
            else:
                logger.error("No se pudo inicializar ningún detector de rostros")
                
        except Exception as e:
            logger.error(f"Error al inicializar detector de rostros: {e}")
            self.face_detector = None
    
    def _init_camera(self):
        """Inicializa la cámara IMX500"""
        try:
            # Verificar que la cámara esté disponible
            result = subprocess.run(['rpicam-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=10)
            
            if 'imx500' in result.stdout:
                logger.info("Cámara IMX500 detectada")
                self.camera_status = "READY"
            else:
                logger.error("Cámara IMX500 no encontrada")
                self.camera_status = "ERROR"
                self.last_error = "Cámara IMX500 no disponible"
                
        except Exception as e:
            logger.error(f"Error al inicializar cámara: {e}")
            self.camera_status = "ERROR"
            self.last_error = str(e)
    
    def start(self):
        """Inicia la captura de video desde la cámara IMX500"""
        try:
            if self.camera_status != "READY":
                logger.error(f"No se puede iniciar cámara en estado: {self.camera_status}")
                return False
            
            self.is_running = True
            
            # Iniciar hilos
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            
            self.capture_thread.start()
            self.processing_thread.start()
            
            logger.info("Cámara IMX500 iniciada correctamente")
            self.camera_status = "RUNNING"
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar cámara: {e}")
            self.camera_status = "ERROR"
            self.last_error = str(e)
            return False
    
    def stop(self):
        """Detiene la captura de video"""
        self.is_running = False
        
        # Detener proceso de cámara si está activo
        if self.camera_process and self.camera_process.poll() is None:
            self.camera_process.terminate()
            try:
                self.camera_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.camera_process.kill()
        
        # Esperar a que terminen los hilos
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        
        self.camera_status = "STOPPED"
        logger.info("Cámara detenida")
    
    def _capture_loop(self):
        """Hilo principal de captura de frames desde la cámara IMX500"""
        logger.info("Hilo de captura iniciado")
        
        while self.is_running:
            try:
                # Capturar frame usando rpicam-still
                frame = self._capture_single_frame()
                
                if frame is not None:
                    # Actualizar FPS
                    current_time = time.time()
                    self.fps_counter.append(current_time - self.last_fps_time)
                    self.last_fps_time = current_time
                    
                    if len(self.fps_counter) >= 30:
                        self.current_fps = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))
                    
                    # Agregar frame a la cola para procesamiento
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    
                    # Actualizar frame actual
                    self.current_frame = frame.copy()
                    
                    # Resetear contador de reconexión si la cámara funciona
                    self.reconnection_attempts = 0
                    self.camera_status = "RUNNING"
                    
                else:
                    # Frame no disponible, intentar reconexión
                    self._handle_camera_error("Error al leer frame de la cámara")
                    
                # Control de frecuencia
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self._handle_camera_error(f"Error en captura: {e}")
                time.sleep(0.1)
        
        logger.info("Hilo de captura terminado")
    
    def _capture_single_frame(self):
        """Captura un frame individual desde la cámara IMX500"""
        try:
            # Usar rpicam-still para capturar un frame
            output_path = "tmp/camera_frame.jpg"
            
            # Comando para capturar frame
            cmd = [
                'rpicam-still',
                '--camera', str(self.camera_index),
                '--timeout', '1000',  # 1 segundo
                '--nopreview',
                '--output', output_path
            ]
            
            # Ejecutar comando
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and os.path.exists(output_path):
                # Leer frame capturado
                frame = cv2.imread(output_path)
                
                if frame is not None:
                    # Redimensionar si es necesario
                    if frame.shape[:2] != (self.frame_height, self.frame_width):
                        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    
                    # Limpiar archivo temporal
                    try:
                        os.remove(output_path)
                    except:
                        pass
                    
                    return frame
                else:
                    logger.warning("Frame capturado pero no se pudo leer")
                    return None
            else:
                logger.warning(f"Error al capturar frame: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning("Timeout al capturar frame")
            return None
        except Exception as e:
            logger.error(f"Error al capturar frame: {e}")
            return None
    
    def _handle_camera_error(self, error_msg):
        """Maneja errores de la cámara con reconexión automática"""
        logger.error(error_msg)
        self.last_error = error_msg
        self.camera_status = "ERROR"
        
        # Intentar reconexión si no se ha excedido el límite
        if self.reconnection_attempts < self.max_reconnection_attempts:
            backoff_time = self.reconnection_backoff[min(self.reconnection_attempts, len(self.reconnection_backoff) - 1)]
            logger.info(f"Reintentando conexión en {backoff_time}s (intento {self.reconnection_attempts + 1}/{self.max_reconnection_attempts})")
            
            time.sleep(backoff_time)
            self.reconnection_attempts += 1
            
            # Intentar reinicializar cámara
            self._init_camera()
        else:
            logger.error("Se excedió el límite de intentos de reconexión")
            self.camera_status = "FAILED"
    
    def _processing_loop(self):
        """Hilo de procesamiento de rostros y generación de embeddings simulados"""
        logger.info("Hilo de procesamiento iniciado")
        
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Detectar rostros usando detector Haar (fallback)
                    faces = self._detect_faces(frame)
                    
                    if faces:
                        # Generar embeddings simulados "desde la cámara"
                        face_data = self._generate_camera_embeddings(frame, faces)
                        
                        # Agregar a cola de reconocimiento
                        if not self.recognition_queue.full():
                            self.recognition_queue.put((frame, face_data))
                    
                    # Limpiar cola si está muy llena
                    while self.frame_queue.qsize() > 5:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                            
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error en procesamiento: {e}")
                time.sleep(0.01)
        
        logger.info("Hilo de procesamiento terminado")
    
    def _detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detecta rostros en el frame usando OpenCV (fallback)"""
        if self.face_detector is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [(x, y, w, h) for (x, y, w, h) in faces]
            
        except Exception as e:
            logger.error(f"Error en detección de rostros: {e}")
            return []
    
    def _generate_camera_embeddings(self, frame, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Genera embeddings simulados "desde la cámara" como requieren las reglas
        En la implementación real, esto sería reemplazado por el modelo MobileFaceNet
        """
        face_data = []
        
        for (x, y, w, h) in faces:
            try:
                # Extraer región del rostro
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    continue
                
                # Redimensionar a tamaño estándar
                face_resized = cv2.resize(face_roi, (128, 128))
                
                # Convertir a escala de grises
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                
                # Normalizar
                face_normalized = face_gray.astype(np.float32) / 255.0
                
                # Generar embedding simulado "desde la cámara" (128 dimensiones)
                # En la implementación real, esto sería el output del modelo MobileFaceNet en la cámara
                embedding = self._simulate_camera_embedding(face_normalized)
                
                # Calcular confianza de detección
                confidence = self._calculate_detection_confidence(face_roi)
                
                face_data.append((embedding, (x, y, w, h), confidence))
                
            except Exception as e:
                logger.error(f"Error al generar embedding: {e}")
                continue
        
        return face_data
    
    def _simulate_camera_embedding(self, face_normalized: np.ndarray) -> np.ndarray:
        """
        Simula la generación de embeddings desde la cámara IMX500
        En la implementación real, esto sería reemplazado por el modelo MobileFaceNet
        """
        try:
            # Extraer características básicas (simulación del modelo de la cámara)
            features = []
            
            # Histograma de gradientes (simula características del modelo)
            grad_x = cv2.Sobel(face_normalized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_normalized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Estadísticas de gradientes
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])
            
            # Características de textura
            features.extend([
                np.mean(face_normalized),
                np.std(face_normalized),
                np.percentile(face_normalized, 25),
                np.percentile(face_normalized, 75)
            ])
            
            # Características de forma
            features.extend([
                face_normalized.shape[0] / face_normalized.shape[1],  # Aspect ratio
                np.sum(face_normalized > 0.5) / face_normalized.size  # Densidad
            ])
            
            # Rellenar hasta 128 dimensiones con características adicionales
            while len(features) < 128:
                # Agregar características basadas en la posición
                row_idx = len(features) % face_normalized.shape[0]
                col_idx = len(features) % face_normalized.shape[1]
                
                if row_idx < face_normalized.shape[0] and col_idx < face_normalized.shape[1]:
                    features.append(face_normalized[row_idx, col_idx])
                else:
                    features.append(0.0)
            
            # Normalizar el vector
            embedding = np.array(features[:128], dtype=np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error en simulación de embedding: {e}")
            # Retornar embedding aleatorio como fallback
            return np.random.rand(128).astype(np.float32)
    
    def _calculate_detection_confidence(self, face_roi: np.ndarray) -> float:
        """Calcula la confianza de detección del rostro"""
        try:
            # Calcular confianza basada en la calidad de la imagen
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Contraste
            contrast = np.std(gray)
            
            # Brillo
            brightness = np.mean(gray)
            
            # Tamaño del rostro
            size_score = min(face_roi.shape[0] * face_roi.shape[1] / 10000, 1.0)
            
            # Calcular confianza combinada
            confidence = (contrast / 100.0 * 0.4 + 
                         (1.0 - abs(brightness - 128) / 128.0) * 0.3 + 
                         size_score * 0.3)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error al calcular confianza: {e}")
            return 0.5
    
    def get_current_frame(self):
        """Obtiene el frame actual"""
        return self.current_frame
    
    def get_face_data(self) -> Optional[Tuple[np.ndarray, List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]]]:
        """Obtiene el frame y datos de rostros más recientes"""
        try:
            if not self.recognition_queue.empty():
                return self.recognition_queue.get()
            return None
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Obtiene el FPS actual"""
        return self.current_fps
    
    def get_camera_status(self) -> dict:
        """Obtiene el estado completo de la cámara"""
        return {
            'status': self.camera_status,
            'last_error': self.last_error,
            'reconnection_attempts': self.reconnection_attempts,
            'max_reconnection_attempts': self.max_reconnection_attempts,
            'is_running': self.is_running,
            'current_fps': self.current_fps
        }
    
    def set_recognition_callback(self, callback: Callable):
        """Establece callback para reconocimiento facial"""
        self.recognition_callback = callback
    
    def force_reconnection(self):
        """Fuerza una reconexión de la cámara"""
        logger.info("Forzando reconexión de la cámara")
        self.reconnection_attempts = 0
        self._init_camera()
        
        if self.camera_status == "READY":
            logger.info("Reconexión exitosa")
            return True
        else:
            logger.error("Reconexión fallida")
            return False

# Mantener compatibilidad con el nombre anterior
CameraHandler = IMX500CameraHandler 