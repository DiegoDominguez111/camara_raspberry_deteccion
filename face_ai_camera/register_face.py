#!/usr/bin/env python3
"""
Script para registrar rostros usando rpicam-vid
Captura múltiples imágenes de una persona y genera embeddings
"""

import subprocess
import threading
import time
import sys
import os
import cv2
import numpy as np
import face_recognition
from utils import save_embedding, load_all_embeddings, normalize_embedding
import json

class FaceRegistrar:
    def __init__(self):
        self.camera_process = None
        self.is_capturing = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Configuración de la cámara
        self.sensor_width = 2028
        self.sensor_height = 1520
        self.display_width = 640
        self.display_height = 480
        
        # Cargar embeddings existentes
        self.existing_embeddings = load_all_embeddings()
        print(f"📚 Cargados {len(self.existing_embeddings)} embeddings existentes")
    
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
                print("⚠️  No se pudo generar embedding para el rostro")
                return None
                
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None
    
    def capture_and_register(self, person_name):
        """Captura múltiples imágenes y registra el rostro"""
        print(f"\n👤 Registrando rostro para: {person_name}")
        print("📸 Posiciónate frente a la cámara y presiona 'c' para capturar")
        print("   Presiona 'q' para salir sin guardar")
        
        if not self.start_camera_stream():
            print("❌ No se pudo iniciar la cámara")
            return False
        
        embeddings = []
        captures = 0
        max_captures = 5
        
        try:
            while captures < max_captures:
                frame = self.get_current_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Detectar rostros
                faces = self.detect_faces_opencv(frame)
                
                # Dibujar rectángulos alrededor de los rostros
                frame_display = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Mostrar información
                cv2.putText(frame_display, f"Capturas: {captures}/{max_captures}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_display, "Presiona 'c' para capturar, 'q' para salir", 
                           (10, frame_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Mostrar frame
                cv2.imshow("Registro de Rostro", frame_display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("❌ Registro cancelado por el usuario")
                    break
                elif key == ord('c'):
                    if len(faces) == 0:
                        print("⚠️  No se detectó ningún rostro")
                    elif len(faces) > 1:
                        print("⚠️  Se detectaron múltiples rostros, usa solo uno")
                    else:
                        # Generar embedding
                        embedding = self.generate_embedding(frame, faces[0])
                        if embedding is not None:
                            embeddings.append(embedding)
                            captures += 1
                            print(f"✅ Captura {captures}/{max_captures} completada")
                        else:
                            print("❌ Error generando embedding")
                
                time.sleep(0.1)
            
            cv2.destroyAllWindows()
            
            if captures > 0:
                # Calcular embedding promedio
                if len(embeddings) > 1:
                    avg_embedding = np.mean(embeddings, axis=0)
                else:
                    avg_embedding = embeddings[0]
                
                # Guardar embedding
                success = save_embedding(person_name, avg_embedding)
                if success:
                    print(f"✅ Rostro registrado exitosamente para: {person_name}")
                    print(f"   - Embeddings generados: {len(embeddings)}")
                    return True
                else:
                    print("❌ Error guardando embedding")
                    return False
            else:
                print("❌ No se capturaron embeddings válidos")
                return False
                
        except Exception as e:
            print(f"❌ Error durante captura: {e}")
            return False
        finally:
            self.stop_camera_stream()


def main():
    """Función principal"""
    print("🚀 SISTEMA DE REGISTRO DE ROSTROS")
    print("=" * 40)
    
    registrar = FaceRegistrar()
    
    try:
        while True:
            print("\n📝 Opciones:")
            print("1. Registrar nuevo rostro")
            print("2. Ver rostros registrados")
            print("3. Salir")
            
            choice = input("\nSelecciona una opción (1-3): ").strip()
            
            if choice == "1":
                person_name = input("Ingresa el nombre de la persona: ").strip()
                if person_name:
                    registrar.capture_and_register(person_name)
                else:
                    print("❌ Nombre no válido")
            
            elif choice == "2":
                faces = registrar.existing_embeddings
                if faces:
                    print(f"\n📚 Rostros registrados ({len(faces)}):")
                    for name in faces.keys():
                        print(f"   - {name}")
                else:
                    print("\n📚 No hay rostros registrados")
            
            elif choice == "3":
                print("👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción no válida")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrumpido por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
    finally:
        registrar.stop_camera_stream()


if __name__ == "__main__":
    main() 