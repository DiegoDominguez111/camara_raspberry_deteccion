import cv2
import numpy as np
import threading
import time
from typing import List, Tuple, Optional, Callable
import queue
from collections import deque
import json

class CameraHandler:
    def __init__(self, camera_index: int = 0, frame_width: int = 640, frame_height: int = 480):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.face_detector = None
        self.face_recognizer = None
        self.recognition_callback = None
        
        # Colas para comunicación entre hilos
        self.frame_queue = queue.Queue(maxsize=10)
        self.recognition_queue = queue.Queue(maxsize=20)
        
        # Estadísticas
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Inicializar detector de rostros
        self._init_face_detector()
        
    def _init_face_detector(self):
        """Inicializa el detector de rostros de OpenCV"""
        try:
            # Usar el detector Haar cascade pre-entrenado
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                print("Error: No se pudo cargar el detector de rostros Haar")
                # Fallback a detector alternativo
                self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
                
        except Exception as e:
            print(f"Error al inicializar detector de rostros: {e}")
            self.face_detector = None
    
    def start(self):
        """Inicia la captura de video"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: No se pudo abrir la cámara {self.camera_index}")
                return False
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Configurar FPS
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            
            # Iniciar hilos
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            
            self.capture_thread.start()
            self.processing_thread.start()
            
            print("Cámara iniciada correctamente")
            return True
            
        except Exception as e:
            print(f"Error al iniciar cámara: {e}")
            return False
    
    def stop(self):
        """Detiene la captura de video"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Esperar a que terminen los hilos
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        
        print("Cámara detenida")
    
    def _capture_loop(self):
        """Hilo principal de captura de frames"""
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
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
                        
                    else:
                        print("Error al leer frame de la cámara")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error en captura: {e}")
                time.sleep(0.1)
    
    def _processing_loop(self):
        """Hilo de procesamiento de rostros y generación de embeddings"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Detectar rostros
                    faces = self._detect_faces(frame)
                    
                    if faces:
                        # Generar embeddings simulados (en la implementación real esto se haría en la cámara)
                        face_data = self._generate_embeddings(frame, faces)
                        
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
                print(f"Error en procesamiento: {e}")
                time.sleep(0.01)
    
    def _detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detecta rostros en el frame usando OpenCV"""
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
            print(f"Error en detección de rostros: {e}")
            return []
    
    def _generate_embeddings(self, frame, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Genera embeddings simulados para los rostros detectados
        En la implementación real, esto se haría en la cámara AI
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
                
                # Generar embedding simulado (128 dimensiones)
                # En la implementación real, esto sería el output del modelo MobileFaceNet
                embedding = self._simulate_embedding(face_normalized)
                
                # Calcular confianza de detección (simulada)
                confidence = self._calculate_detection_confidence(face_roi)
                
                face_data.append((embedding, (x, y, w, h), confidence))
                
            except Exception as e:
                print(f"Error al generar embedding: {e}")
                continue
        
        return face_data
    
    def _simulate_embedding(self, face_normalized: np.ndarray) -> np.ndarray:
        """
        Simula la generación de embeddings usando características básicas de la imagen
        En la implementación real, esto sería reemplazado por el modelo MobileFaceNet
        """
        try:
            # Extraer características básicas (simulación)
            features = []
            
            # Histograma de gradientes
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
            
            # Características de forma (simuladas)
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
            print(f"Error en simulación de embedding: {e}")
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
            print(f"Error al calcular confianza: {e}")
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
    
    def set_recognition_callback(self, callback: Callable):
        """Establece callback para reconocimiento facial"""
        self.recognition_callback = callback
    
    def get_camera_info(self) -> dict:
        """Obtiene información de la cámara"""
        if self.cap and self.cap.isOpened():
            return {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'is_running': self.is_running,
                'current_fps': self.current_fps
            }
        return {'is_running': False} 