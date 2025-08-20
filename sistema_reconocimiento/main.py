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
import logging

from face_db import FaceDatabase
from recognizer import FaceRecognizer
from camera_handler import IMX500CameraHandler
from webapp import app
from utils import log_system_event, get_all_metrics
import uvicorn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.recognition_interval = 0.1
        
        # Manejo de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Logging de eventos del sistema
        log_system_event("INFO", "Sistema de reconocimiento facial inicializado")
    
    def _signal_handler(self, signum, frame):
        """Maneja señales de terminación"""
        logger.info(f"Recibida señal {signum}, cerrando sistema...")
        log_system_event("INFO", f"Sistema terminado por señal {signum}")
        self.stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Inicializa todos los componentes del sistema"""
        try:
            logger.info("Inicializando Sistema de Reconocimiento Facial...")
            log_system_event("INFO", "Iniciando inicialización del sistema")
            
            # Inicializar base de datos
            logger.info("  - Inicializando base de datos...")
            self.face_db = FaceDatabase()
            logger.info("    ✓ Base de datos inicializada")
            
            # Inicializar reconocedor facial
            logger.info("  - Inicializando reconocedor facial...")
            self.face_recognizer = FaceRecognizer(self.face_db)
            logger.info("    ✓ Reconocedor facial inicializado")
            
            # Inicializar cámara IMX500
            logger.info("  - Inicializando cámara IMX500...")
            self.camera_handler = IMX500CameraHandler(
                camera_index=self.camera_index,
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
            
            if not self.camera_handler.start():
                logger.error("    ✗ Error al iniciar cámara")
                log_system_event("ERROR", "Error al iniciar cámara IMX500")
                return False
            
            logger.info("    ✓ Cámara IMX500 iniciada")
            
            # Configurar callback de reconocimiento
            self.camera_handler.set_recognition_callback(self._on_face_detected)
            
            # Iniciar hilo de reconocimiento
            self.recognition_thread = threading.Thread(
                target=self._recognition_loop,
                daemon=True
            )
            self.recognition_thread.start()
            logger.info("    ✓ Hilo de reconocimiento iniciado")
            
            logger.info("✓ Sistema inicializado correctamente")
            log_system_event("SUCCESS", "Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error en inicialización: {e}")
            log_system_event("ERROR", f"Error en inicialización: {e}")
            return False
    
    def _on_face_detected(self, frame, face_data):
        """Callback cuando se detectan rostros"""
        try:
            if face_data and self.face_recognizer:
                # Preparar datos para reconocimiento
                recognition_input = [(emb, bbox) for emb, bbox, conf in face_data]
                
                # Realizar reconocimiento
                results = self.face_recognizer.batch_recognize(recognition_input)
                
                for nombre, confianza, es_nuevo, bbox in results:
                    if nombre:
                        logger.info(f"Rostro reconocido: {nombre} (confianza: {confianza:.2f})")
                        log_system_event("INFO", f"Reconocimiento exitoso: {nombre} ({confianza:.2f})")
                    else:
                        logger.info(f"Rostro desconocido detectado (confianza: {confianza:.2f})")
                        log_system_event("INFO", f"Rostro desconocido detectado ({confianza:.2f})")
                        
        except Exception as e:
            logger.error(f"Error en callback de reconocimiento: {e}")
            log_system_event("ERROR", f"Error en callback de reconocimiento: {e}")
    
    def _recognition_loop(self):
        """Hilo principal de reconocimiento facial"""
        logger.info("Hilo de reconocimiento iniciado")
        
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
                                    logger.info(f"✓ {nombre} reconocido (confianza: {confianza:.2f})")
                                elif not nombre:
                                    logger.debug(f"? Rostro desconocido detectado")
                
                # Control de frecuencia
                time.sleep(self.recognition_interval)
                
            except Exception as e:
                logger.error(f"Error en hilo de reconocimiento: {e}")
                log_system_event("ERROR", f"Error en hilo de reconocimiento: {e}")
                time.sleep(0.1)
        
        logger.info("Hilo de reconocimiento terminado")
    
    def start_web_server(self):
        """Inicia el servidor web en un hilo separado"""
        try:
            logger.info(f"Iniciando servidor web en {self.web_host}:{self.web_port}...")
            
            config = uvicorn.Config(
                app=app,
                host=self.web_host,
                port=self.web_port,
                log_level="info",
                access_log=True
            )
            
            self.web_server = uvicorn.Server(config)
            
            web_thread = threading.Thread(
                target=self._run_web_server,
                daemon=True
            )
            web_thread.start()
            
            logger.info("    ✓ Servidor web iniciado")
            log_system_event("SUCCESS", f"Servidor web iniciado en {self.web_host}:{self.web_port}")
            return True
            
        except Exception as e:
            logger.error(f"    ✗ Error al iniciar servidor web: {e}")
            log_system_event("ERROR", f"Error al iniciar servidor web: {e}")
            return False
    
    def _run_web_server(self):
        """Ejecuta el servidor web"""
        try:
            self.web_server.run()
        except Exception as e:
            logger.error(f"Error en servidor web: {e}")
            log_system_event("ERROR", f"Error en servidor web: {e}")
    
    def start(self) -> bool:
        """Inicia el sistema completo"""
        try:
            if not self.initialize():
                return False
            
            if not self.start_web_server():
                return False
            
            self.is_running = True
            
            logger.info("\n🎉 Sistema de Reconocimiento Facial iniciado exitosamente!")
            logger.info(f"📱 Interfaz web disponible en: http://{self.web_host}:{self.web_port}")
            logger.info("📷 Cámara IMX500 funcionando en tiempo real")
            logger.info("🔍 Reconocimiento facial activo")
            logger.info("\nPresiona Ctrl+C para detener el sistema")
            
            log_system_event("SUCCESS", "Sistema iniciado completamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar sistema: {e}")
            log_system_event("ERROR", f"Error al iniciar sistema: {e}")
            return False
    
    def stop(self):
        """Detiene el sistema completo"""
        logger.info("\nDeteniendo sistema...")
        log_system_event("INFO", "Deteniendo sistema")
        
        self.is_running = False
        
        # Detener cámara
        if self.camera_handler:
            logger.info("  - Deteniendo cámara...")
            self.camera_handler.stop()
            logger.info("    ✓ Cámara detenida")
        
        # Detener servidor web
        if self.web_server:
            logger.info("  - Deteniendo servidor web...")
            try:
                self.web_server.should_exit = True
            except:
                pass
            logger.info("    ✓ Servidor web detenido")
        
        # Esperar hilo de reconocimiento
        if self.recognition_thread and self.recognition_thread.is_alive():
            logger.info("  - Esperando hilo de reconocimiento...")
            self.recognition_thread.join(timeout=3)
            logger.info("    ✓ Hilo de reconocimiento terminado")
        
        logger.info("✓ Sistema detenido correctamente")
        log_system_event("INFO", "Sistema detenido correctamente")
    
    def get_status(self) -> dict:
        """Obtiene el estado del sistema"""
        try:
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
                status['camera_info'] = self.camera_handler.get_camera_status()
            
            # Estadísticas de reconocimiento
            if self.face_recognizer:
                status['recognition_stats'] = self.face_recognizer.get_recognition_stats()
            
            # Estadísticas de la base de datos
            if self.face_db:
                db_stats = self.face_db.get_database_stats()
                status['database_stats'] = db_stats
            
            # Métricas del sistema
            try:
                system_metrics = get_all_metrics()
                status['system_metrics'] = system_metrics
            except Exception as e:
                logger.warning(f"No se pudieron obtener métricas del sistema: {e}")
                status['system_metrics'] = {'error': str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Error al obtener estado del sistema: {e}")
            return {
                'is_running': self.is_running,
                'error': str(e)
            }
    
    def run_interactive(self):
        """Ejecuta el sistema en modo interactivo"""
        try:
            if not self.start():
                logger.error("Error al iniciar sistema")
                return
            
            logger.info("Sistema ejecutándose en modo interactivo...")
            
            while self.is_running:
                try:
                    # Esperar 10 segundos
                    time.sleep(10)
                    
                    if self.is_running:
                        # Mostrar estado del sistema
                        status = self.get_status()
                        
                        # FPS de la cámara
                        fps = status.get('camera_info', {}).get('current_fps', 0)
                        
                        # Número de personas
                        people_count = status.get('database_stats', {}).get('total_people', 0)
                        
                        # Estado de la cámara
                        camera_status = status.get('camera_info', {}).get('status', 'UNKNOWN')
                        
                        logger.info(f"📊 Estado: FPS={fps:.1f}, Personas={people_count}, Cámara={camera_status}")
                        
                        # Loggear métricas del sistema
                        if 'system_metrics' in status and 'system' in status['system_metrics']:
                            system_info = status['system_metrics']['system']
                            cpu = system_info.get('cpu_percent', 0)
                            memory = system_info.get('memory_percent', 0)
                            temp = system_info.get('temperature', 'N/A')
                            
                            logger.info(f"💻 Sistema: CPU={cpu:.1f}%, RAM={memory:.1f}%, Temp={temp}°C")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error en bucle principal: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("\nInterrupción del usuario")
        except Exception as e:
            logger.error(f"Error en modo interactivo: {e}")
            log_system_event("ERROR", f"Error en modo interactivo: {e}")
        finally:
            self.stop()

def main():
    """Función principal"""
    print("=" * 60)
    print("SISTEMA DE RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("Raspberry Pi 5 + Raspberry Pi AI Camera (IMX500)")
    print("=" * 60)
    
    # Crear sistema
    system = FaceRecognitionSystem()
    
    try:
        # Ejecutar sistema
        system.run_interactive()
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        log_system_event("ERROR", f"Error fatal del sistema: {e}")
        system.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 