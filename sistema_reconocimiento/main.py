#!/usr/bin/env python3
"""
Sistema de Reconocimiento Facial en Tiempo Real
Raspberry Pi 5 + Raspberry Pi AI Camera (Sony IMX500)

Este archivo orquesta todos los componentes del sistema:
- Cámara y procesamiento de video
- Base de datos de rostros
- Reconocimiento facial
- Servidor web
"""

import asyncio
import threading
import time
import signal
import sys
import os
from typing import Optional

# Importar módulos del sistema
from face_db import FaceDatabase
from recognizer import FaceRecognizer
from camera_handler import CameraHandler
from webapp import app
import uvicorn

class FaceRecognitionSystem:
    def __init__(self):
        self.face_db = None
        self.face_recognizer = None
        self.camera_handler = None
        self.web_server = None
        self.is_running = False
        self.recognition_thread = None
        
        # Configuración
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.web_host = "0.0.0.0"
        self.web_port = 8000
        self.recognition_interval = 0.1  # 100ms entre reconocimientos
        
        # Manejo de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Maneja señales de terminación"""
        print(f"\nRecibida señal {signum}, cerrando sistema...")
        self.stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Inicializa todos los componentes del sistema"""
        try:
            print("Inicializando Sistema de Reconocimiento Facial...")
            
            # 1. Inicializar base de datos
            print("  - Inicializando base de datos...")
            self.face_db = FaceDatabase()
            print("    ✓ Base de datos inicializada")
            
            # 2. Inicializar reconocedor facial
            print("  - Inicializando reconocedor facial...")
            self.face_recognizer = FaceRecognizer(self.face_db)
            print("    ✓ Reconocedor facial inicializado")
            
            # 3. Inicializar cámara
            print("  - Inicializando cámara...")
            self.camera_handler = CameraHandler(
                camera_index=self.camera_index,
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
            
            if not self.camera_handler.start():
                print("    ✗ Error al iniciar cámara")
                return False
            
            print("    ✓ Cámara iniciada")
            
            # 4. Configurar callback de reconocimiento
            self.camera_handler.set_recognition_callback(self._on_face_detected)
            
            # 5. Iniciar hilo de reconocimiento
            self.recognition_thread = threading.Thread(
                target=self._recognition_loop,
                daemon=True
            )
            self.recognition_thread.start()
            print("    ✓ Hilo de reconocimiento iniciado")
            
            print("✓ Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"✗ Error en inicialización: {e}")
            return False
    
    def _on_face_detected(self, frame, face_data):
        """Callback cuando se detectan rostros"""
        try:
            if face_data and self.face_recognizer:
                # Preparar datos para reconocimiento
                recognition_input = [(emb, bbox) for emb, bbox, conf in face_data]
                
                # Realizar reconocimiento
                results = self.face_recognizer.batch_recognize(recognition_input)
                
                # Procesar resultados
                for nombre, confianza, es_nuevo, bbox in results:
                    if nombre:
                        print(f"Rostro reconocido: {nombre} (confianza: {confianza:.2f})")
                    else:
                        print(f"Rostro desconocido detectado (confianza: {confianza:.2f})")
                        
        except Exception as e:
            print(f"Error en callback de reconocimiento: {e}")
    
    def _recognition_loop(self):
        """Hilo principal de reconocimiento facial"""
        print("Hilo de reconocimiento iniciado")
        
        while self.is_running:
            try:
                if self.camera_handler and self.face_recognizer:
                    # Obtener datos de rostros de la cámara
                    face_data = self.camera_handler.get_face_data()
                    
                    if face_data:
                        frame, faces = face_data
                        
                        if faces:
                            # Preparar datos para reconocimiento
                            recognition_input = [(emb, bbox) for emb, bbox, conf in faces]
                            
                            # Realizar reconocimiento
                            results = self.face_recognizer.batch_recognize(recognition_input)
                            
                            # Procesar resultados
                            for nombre, confianza, es_nuevo, bbox in results:
                                if nombre and es_nuevo:
                                    print(f"✓ {nombre} reconocido (confianza: {confianza:.2f})")
                                elif not nombre:
                                    print(f"? Rostro desconocido detectado")
                
                # Control de frecuencia
                time.sleep(self.recognition_interval)
                
            except Exception as e:
                print(f"Error en hilo de reconocimiento: {e}")
                time.sleep(0.1)
        
        print("Hilo de reconocimiento terminado")
    
    def start_web_server(self):
        """Inicia el servidor web en un hilo separado"""
        try:
            print(f"Iniciando servidor web en {self.web_host}:{self.web_port}...")
            
            # Configurar uvicorn
            config = uvicorn.Config(
                app=app,
                host=self.web_host,
                port=self.web_port,
                log_level="info",
                access_log=True
            )
            
            # Crear servidor
            self.web_server = uvicorn.Server(config)
            
            # Ejecutar en hilo separado
            web_thread = threading.Thread(
                target=self._run_web_server,
                daemon=True
            )
            web_thread.start()
            
            print("    ✓ Servidor web iniciado")
            return True
            
        except Exception as e:
            print(f"    ✗ Error al iniciar servidor web: {e}")
            return False
    
    def _run_web_server(self):
        """Ejecuta el servidor web"""
        try:
            self.web_server.run()
        except Exception as e:
            print(f"Error en servidor web: {e}")
    
    def start(self) -> bool:
        """Inicia el sistema completo"""
        try:
            if not self.initialize():
                return False
            
            # Iniciar servidor web
            if not self.start_web_server():
                return False
            
            self.is_running = True
            print("\n🎉 Sistema de Reconocimiento Facial iniciado exitosamente!")
            print(f"📱 Interfaz web disponible en: http://{self.web_host}:{self.web_port}")
            print("📷 Cámara funcionando en tiempo real")
            print("🔍 Reconocimiento facial activo")
            print("\nPresiona Ctrl+C para detener el sistema")
            
            return True
            
        except Exception as e:
            print(f"Error al iniciar sistema: {e}")
            return False
    
    def stop(self):
        """Detiene el sistema completo"""
        print("\nDeteniendo sistema...")
        
        self.is_running = False
        
        # Detener cámara
        if self.camera_handler:
            print("  - Deteniendo cámara...")
            self.camera_handler.stop()
            print("    ✓ Cámara detenida")
        
        # Detener servidor web
        if self.web_server:
            print("  - Deteniendo servidor web...")
            try:
                self.web_server.should_exit = True
            except:
                pass
            print("    ✓ Servidor web detenido")
        
        # Esperar hilos
        if self.recognition_thread and self.recognition_thread.is_alive():
            print("  - Esperando hilo de reconocimiento...")
            self.recognition_thread.join(timeout=3)
            print("    ✓ Hilo de reconocimiento terminado")
        
        print("✓ Sistema detenido correctamente")
    
    def get_status(self) -> dict:
        """Obtiene el estado del sistema"""
        status = {
            'is_running': self.is_running,
            'components': {
                'database': self.face_db is not None,
                'recognizer': self.face_recognizer is not None,
                'camera': self.camera_handler is not None and self.camera_handler.is_running,
                'web_server': self.web_server is not None
            }
        }
        
        # Información de la cámara
        if self.camera_handler:
            status['camera_info'] = self.camera_handler.get_camera_info()
        
        # Estadísticas de reconocimiento
        if self.face_recognizer:
            status['recognition_stats'] = self.face_recognizer.get_recognition_stats()
        
        # Estadísticas de la base de datos
        if self.face_db:
            people = self.face_db.list_people()
            recent_logs = self.face_db.get_recent_logs(50)
            status['database_stats'] = {
                'total_people': len(people),
                'total_logs': len(recent_logs),
                'recent_activity': len([log for log in recent_logs if log[0] is not None])
            }
        
        return status
    
    def run_interactive(self):
        """Ejecuta el sistema en modo interactivo"""
        try:
            if not self.start():
                print("Error al iniciar sistema")
                return
            
            # Bucle principal
            while self.is_running:
                try:
                    # Mostrar estado cada 10 segundos
                    time.sleep(10)
                    
                    if self.is_running:
                        status = self.get_status()
                        fps = status.get('camera_info', {}).get('current_fps', 0)
                        people_count = status.get('database_stats', {}).get('total_people', 0)
                        
                        print(f"📊 Estado: FPS={fps:.1f}, Personas={people_count}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error en bucle principal: {e}")
                    time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nInterrupción del usuario")
        except Exception as e:
            print(f"Error en modo interactivo: {e}")
        finally:
            self.stop()

def main():
    """Función principal"""
    print("=" * 60)
    print("SISTEMA DE RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("Raspberry Pi 5 + Raspberry Pi AI Camera")
    print("=" * 60)
    
    # Crear y ejecutar sistema
    system = FaceRecognitionSystem()
    
    try:
        system.run_interactive()
    except Exception as e:
        print(f"Error fatal: {e}")
        system.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 