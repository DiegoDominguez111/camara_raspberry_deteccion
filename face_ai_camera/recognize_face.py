#!/usr/bin/env python3
"""
Script para reconocimiento facial en tiempo real usando rpicam-vid
Detecta rostros y los compara contra embeddings registrados
"""

import subprocess
import threading
import time
import sys
import os
import cv2
import numpy as np
import face_recognition
from utils import load_all_embeddings, find_best_match, normalize_embedding
import json

class FaceRecognizer:
    def __init__(self, threshold=0.6):
        self.camera_process = None
        self.is_capturing = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Configuración de la cámara
        self.sensor_width = 2028
        self.sensor_height = 1520
        self.display_width = 640
        self.display_height = 480
        
        # Configuración de reconocimiento
        self.threshold = threshold
        self.known_embeddings = {}
        self.known_names = []
        
        # Estado de reconocimiento
        self.current_recognition = None
        self.recognition_stability = 0
        self.stability_threshold = 3
        
        # Cargar embeddings conocidos
        self._load_known_embeddings()
    
    def _load_known_embeddings(self):
        """Carga embeddings conocidos desde el directorio"""
        try:
            self.known_embeddings = load_all_embeddings()
            self.known_names = list(self.known_embeddings.keys())
            print(f"📚 Cargados {len(self.known_names)} rostros conocidos:")
            for name in self.known_names:
                print(f"   - {name}")
        except Exception as e:
            print(f"❌ Error cargando embeddings: {e}")
            self.known_embeddings = {}
            self.known_names = []
    
    def start_camera_stream(self):
        """Inicia el stream de la cámara usando rpicam-vid"""
        try:
            cmd = [
                "rpicam-vid",
                "-n",  # sin preview
                "-t", "0",  # tiempo infinito
                "--codec", "mjpeg",
                "-o", "-",  # salida a stdout
                "--width", str(self.sensor_width),
                "--height", str(self.sensor_height),
            ]
            
            print(f"📹 Iniciando stream: {' '.join(cmd)}")
            self.camera_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Iniciar hilo de procesamiento de frames
            self.is_capturing = True
            frame_thread = threading.Thread(target=self._process_frames, daemon=True)
            frame_thread.start()
            
            # Esperar a que se capture el primer frame
            timeout = 10
            start_time = time.time()
            while self.latest_frame is None and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.latest_frame is not None:
                print("✅ Stream de cámara iniciado correctamente")
                return True
            else:
                print("❌ Timeout esperando primer frame")
                return False
                
        except Exception as e:
            print(f"❌ Error iniciando stream: {e}")
            return False
    
    def _process_frames(self):
        """Procesa frames del stream de rpicam-vid"""
        buffer = b""
        
        while self.is_capturing and self.camera_process:
            try:
                # Leer chunk de datos
                chunk = self.camera_process.stdout.read(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Buscar frames JPEG completos
                while True:
                    # Buscar inicio de frame JPEG (0xFF 0xD8)
                    start_pos = buffer.find(b"\xff\xd8")
                    if start_pos == -1:
                        if buffer and buffer[-1] == 0xFF:
                            buffer = buffer[-1:]
                        else:
                            buffer = b""
                        break
                    
                    # Buscar fin de frame JPEG (0xFF 0xD9)
                    end_pos = buffer.find(b"\xff\xd9", start_pos + 2)
                    if end_pos == -1:
                        buffer = buffer[start_pos:]
                        break
                    
                    # Extraer frame completo
                    frame_data = buffer[start_pos : end_pos + 2]
                    if len(frame_data) > 1000:
                        with self.frame_lock:
                            self.latest_frame = frame_data
                    
                    # Remover frame procesado del buffer
                    buffer = buffer[end_pos + 2 :]
                    
            except Exception as e:
                print(f"❌ Error procesando frames: {e}")
                time.sleep(0.1)
    
    def stop_camera_stream(self):
        """Detiene el stream de la cámara"""
        self.is_capturing = False
        if self.camera_process:
            self.camera_process.terminate()
            self.camera_process.wait()
            self.camera_process = None
        print("🛑 Stream de cámara detenido")
    
    def get_current_frame(self):
        """Obtiene el frame actual como array de numpy"""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            
            try:
                # Convertir bytes JPEG a array numpy
                nparr = np.frombuffer(self.latest_frame, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Redimensionar para display
                    frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
                    return frame_resized
                else:
                    return None
                    
            except Exception as e:
                print(f"❌ Error convirtiendo frame: {e}")
                return None
    
    def detect_faces_opencv(self, frame):
        """Detecta rostros usando OpenCV Haar cascades"""
        try:
            # Cargar clasificador Haar
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            if not os.path.exists(haar_cascade_path):
                print("⚠️  Descargando clasificador Haar...")
                os.makedirs(os.path.dirname(haar_cascade_path), exist_ok=True)
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, haar_cascade_path)
            
            face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            
            if face_cascade.empty():
                print("❌ Error cargando clasificador Haar")
                return []
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces
            
        except Exception as e:
            print(f"❌ Error en detección facial: {e}")
            return []
    
    def generate_embedding(self, frame, face_location):
        """Genera embedding para un rostro detectado"""
        try:
            x, y, w, h = face_location
            
            # Extraer región del rostro
            face_image = frame[y:y+h, x:x+w]
            
            # Generar embedding usando face_recognition
            face_encodings = face_recognition.face_encodings(face_image)
            
            if len(face_encodings) > 0:
                embedding = face_encodings[0]
                # Normalizar embedding
                normalized_embedding = normalize_embedding(embedding)
                return normalized_embedding
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None
    
    def recognize_face(self, embedding):
        """Reconoce un rostro comparando su embedding con los conocidos"""
        if not self.known_embeddings:
            return None, 0.0
        
        try:
            best_match_name, best_match_distance = find_best_match(embedding, self.known_embeddings)
            
            if best_match_distance <= self.threshold:
                return best_match_name, best_match_distance
            else:
                return "Desconocido", best_match_distance
                
        except Exception as e:
            print(f"❌ Error en reconocimiento: {e}")
            return None, 1.0
    
    def stabilize_recognition(self, current_name, current_distance):
        """Estabiliza el reconocimiento para evitar saltos"""
        if current_name == self.current_recognition:
            self.recognition_stability += 1
        else:
            self.recognition_stability = 0
            self.current_recognition = current_name
        
        # Solo retornar reconocimiento estable
        if self.recognition_stability >= self.stability_threshold:
            return self.current_recognition, current_distance
        else:
            return "Analizando...", current_distance
    
    def draw_recognition_info(self, frame, faces, recognitions):
        """Dibuja información de reconocimiento en el frame"""
        for i, (face, recognition) in enumerate(zip(faces, recognitions)):
            x, y, w, h = face
            name, distance = recognition
            
            # Color basado en el reconocimiento
            if name == "Desconocido":
                color = (0, 0, 255)  # Rojo
            elif name == "Analizando...":
                color = (0, 255, 255)  # Amarillo
            else:
                color = (0, 255, 0)  # Verde
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Preparar texto
            if name == "Desconocido":
                text = f"Desconocido ({distance:.2f})"
            elif name == "Analizando...":
                text = "Analizando..."
            else:
                text = f"{name} ({distance:.2f})"
            
            # Calcular posición del texto
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x
            text_y = y - 10 if y - 10 > text_size[1] else y + h + text_size[1]
            
            # Fondo del texto
            cv2.rectangle(frame, 
                         (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0], text_y + 5),
                         color, -1)
            
            # Texto
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_recognition(self):
        """Ejecuta el reconocimiento facial en tiempo real"""
        print("🎯 Iniciando reconocimiento facial en tiempo real")
        print("📹 Presiona 'q' para salir")
        print("📹 Presiona 'r' para recargar embeddings")
        
        if not self.start_camera_stream():
            print("❌ No se pudo iniciar la cámara")
            return False
        
        try:
            while True:
                frame = self.get_current_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Detectar rostros
                faces = self.detect_faces_opencv(frame)
                recognitions = []
                
                # Procesar cada rostro detectado
                for face in faces:
                    # Generar embedding
                    embedding = self.generate_embedding(frame, face)
                    
                    if embedding is not None:
                        # Reconocer rostro
                        name, distance = self.recognize_face(embedding)
                        
                        # Estabilizar reconocimiento
                        stable_name, stable_distance = self.stabilize_recognition(name, distance)
                        recognitions.append((stable_name, stable_distance))
                    else:
                        recognitions.append(("Error", 1.0))
                
                # Dibujar información en el frame
                frame_display = frame.copy()
                self.draw_recognition_info(frame_display, faces, recognitions)
                
                # Mostrar estadísticas
                cv2.putText(frame_display, f"Rostros detectados: {len(faces)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"Conocidos: {len(self.known_names)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, "Presiona 'q' para salir, 'r' para recargar", 
                           (10, frame_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Mostrar frame
                cv2.imshow("Reconocimiento Facial", frame_display)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Reconocimiento detenido por el usuario")
                    break
                elif key == ord('r'):
                    print("🔄 Recargando embeddings...")
                    self._load_known_embeddings()
                
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante reconocimiento: {e}")
            return False
        finally:
            cv2.destroyAllWindows()
            self.stop_camera_stream()


def main():
    """Función principal"""
    print("🚀 SISTEMA DE RECONOCIMIENTO FACIAL")
    print("=" * 40)
    
    # Configurar umbral de reconocimiento
    threshold = 0.6
    try:
        threshold_input = input(f"Umbral de reconocimiento (0.1-1.0, default {threshold}): ").strip()
        if threshold_input:
            threshold = float(threshold_input)
            if threshold < 0.1 or threshold > 1.0:
                print("⚠️  Umbral fuera de rango, usando valor por defecto")
                threshold = 0.6
    except ValueError:
        print("⚠️  Valor inválido, usando umbral por defecto")
        threshold = 0.6
    
    print(f"🎯 Umbral configurado: {threshold}")
    
    recognizer = FaceRecognizer(threshold=threshold)
    
    try:
        success = recognizer.run_recognition()
        if success:
            print("✅ Reconocimiento completado exitosamente")
        else:
            print("❌ Error durante el reconocimiento")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrumpido por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 