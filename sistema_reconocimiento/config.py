"""
Archivo de configuración centralizada para el Sistema de Reconocimiento Facial
Raspberry Pi 5 + AI Camera
"""

import os
from typing import Dict, Any

class Config:
    """Configuración centralizada del sistema"""
    
    # Configuración de la cámara
    CAMERA = {
        'index': 0,                    # Índice de la cámara (0, 1, etc.)
        'width': 640,                  # Ancho del frame
        'height': 480,                 # Alto del frame
        'fps': 30,                     # FPS objetivo
        'buffer_size': 10,             # Tamaño del buffer de frames
        'recognition_queue_size': 20,  # Tamaño de la cola de reconocimiento
    }
    
    # Configuración del reconocimiento facial
    RECOGNITION = {
        'confidence_threshold': 0.6,   # Umbral de confianza (0.0 - 1.0)
        'min_faces_for_recognition': 3, # Mínimo de frames para confirmar
        'recognition_interval': 0.1,   # Intervalo entre reconocimientos (segundos)
        'duplicate_timeout': 5.0,      # Tiempo para evitar duplicados (segundos)
        'bbox_distance_threshold': 50, # Distancia en píxeles para considerar mismo rostro
    }
    
    # Configuración de la base de datos
    DATABASE = {
        'path': 'face_recognition.db', # Ruta de la base de datos
        'backup_interval': 3600,      # Intervalo de backup automático (segundos)
        'max_logs': 10000,            # Máximo número de logs a mantener
    }
    
    # Configuración del servidor web
    WEB = {
        'host': '0.0.0.0',            # Host del servidor web
        'port': 8000,                 # Puerto del servidor web
        'debug': False,                # Modo debug
        'reload': False,               # Recarga automática
        'workers': 1,                  # Número de workers
        'log_level': 'info',          # Nivel de logging
    }
    
    # Configuración del video
    VIDEO = {
        'jpeg_quality': 80,            # Calidad JPEG (1-100)
        'mjpeg_boundary': 'frame',     # Boundary para MJPEG
        'max_frame_size': 1024 * 1024, # Tamaño máximo de frame (bytes)
    }
    
    # Configuración del sistema
    SYSTEM = {
        'log_level': 'INFO',           # Nivel de logging del sistema
        'max_threads': 4,              # Máximo número de hilos
        'health_check_interval': 10,   # Intervalo de health check (segundos)
        'stats_update_interval': 2,    # Intervalo de actualización de stats (segundos)
        'logs_update_interval': 5,     # Intervalo de actualización de logs (segundos)
    }
    
    # Configuración de rendimiento
    PERFORMANCE = {
        'target_fps': 30,              # FPS objetivo
        'min_fps': 15,                 # FPS mínimo aceptable
        'max_latency': 0.1,            # Latencia máxima (segundos)
        'memory_limit_mb': 512,        # Límite de memoria (MB)
        'cpu_threshold': 80,           # Umbral de CPU (%)
    }
    
    # Configuración de seguridad
    SECURITY = {
        'enable_cors': True,           # Habilitar CORS
        'allowed_origins': ['*'],      # Orígenes permitidos
        'max_file_size': 10 * 1024 * 1024,  # Tamaño máximo de archivo (10MB)
        'rate_limit': 100,             # Límite de requests por minuto
    }
    
    @classmethod
    def get_camera_config(cls) -> Dict[str, Any]:
        """Obtiene configuración de la cámara"""
        return cls.CAMERA.copy()
    
    @classmethod
    def get_recognition_config(cls) -> Dict[str, Any]:
        """Obtiene configuración del reconocimiento"""
        return cls.RECOGNITION.copy()
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Obtiene configuración de la base de datos"""
        return cls.DATABASE.copy()
    
    @classmethod
    def get_web_config(cls) -> Dict[str, Any]:
        """Obtiene configuración del servidor web"""
        return cls.WEB.copy()
    
    @classmethod
    def get_video_config(cls) -> Dict[str, Any]:
        """Obtiene configuración del video"""
        return cls.VIDEO.copy()
    
    @classmethod
    def get_system_config(cls) -> Dict[str, Any]:
        """Obtiene configuración del sistema"""
        return cls.SYSTEM.copy()
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Obtiene configuración de rendimiento"""
        return cls.PERFORMANCE.copy()
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """Obtiene configuración de seguridad"""
        return cls.SECURITY.copy()
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Obtiene toda la configuración"""
        return {
            'camera': cls.get_camera_config(),
            'recognition': cls.get_recognition_config(),
            'database': cls.get_database_config(),
            'web': cls.get_web_config(),
            'video': cls.get_video_config(),
            'system': cls.get_system_config(),
            'performance': cls.get_performance_config(),
            'security': cls.get_security_config(),
        }
    
    @classmethod
    def update_config(cls, section: str, key: str, value: Any) -> bool:
        """Actualiza una configuración específica"""
        try:
            if hasattr(cls, section.upper()) and key in getattr(cls, section.upper()):
                getattr(cls, section.upper())[key] = value
                return True
            return False
        except Exception:
            return False
    
    @classmethod
    def load_from_env(cls):
        """Carga configuración desde variables de entorno"""
        # Cámara
        if os.getenv('CAMERA_INDEX'):
            cls.CAMERA['index'] = int(os.getenv('CAMERA_INDEX'))
        
        if os.getenv('CAMERA_WIDTH'):
            cls.CAMERA['width'] = int(os.getenv('CAMERA_WIDTH'))
        
        if os.getenv('CAMERA_HEIGHT'):
            cls.CAMERA['height'] = int(os.getenv('CAMERA_HEIGHT'))
        
        if os.getenv('CAMERA_FPS'):
            cls.CAMERA['fps'] = int(os.getenv('CAMERA_FPS'))
        
        # Reconocimiento
        if os.getenv('RECOGNITION_THRESHOLD'):
            cls.RECOGNITION['confidence_threshold'] = float(os.getenv('RECOGNITION_THRESHOLD'))
        
        # Web
        if os.getenv('WEB_HOST'):
            cls.WEB['host'] = os.getenv('WEB_HOST')
        
        if os.getenv('WEB_PORT'):
            cls.WEB['port'] = int(os.getenv('WEB_PORT'))
        
        # Base de datos
        if os.getenv('DB_PATH'):
            cls.DATABASE['path'] = os.getenv('DB_PATH')
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Valida la configuración y retorna errores si los hay"""
        errors = []
        
        # Validar cámara
        if cls.CAMERA['width'] < 320 or cls.CAMERA['height'] < 240:
            errors.append("Resolución de cámara muy baja")
        
        if cls.CAMERA['fps'] < 1 or cls.CAMERA['fps'] > 60:
            errors.append("FPS de cámara inválido")
        
        # Validar reconocimiento
        if cls.RECOGNITION['confidence_threshold'] < 0.0 or cls.RECOGNITION['confidence_threshold'] > 1.0:
            errors.append("Umbral de confianza inválido")
        
        # Validar web
        if cls.WEB['port'] < 1024 or cls.WEB['port'] > 65535:
            errors.append("Puerto web inválido")
        
        # Validar rendimiento
        if cls.PERFORMANCE['target_fps'] < cls.PERFORMANCE['min_fps']:
            errors.append("FPS objetivo menor que FPS mínimo")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

# Cargar configuración desde variables de entorno al importar
Config.load_from_env()

# Configuración específica para desarrollo
if os.getenv('ENVIRONMENT') == 'development':
    Config.WEB['debug'] = True
    Config.WEB['reload'] = True
    Config.SYSTEM['log_level'] = 'DEBUG'

# Configuración específica para producción
if os.getenv('ENVIRONMENT') == 'production':
    Config.WEB['debug'] = False
    Config.WEB['reload'] = False
    Config.SYSTEM['log_level'] = 'WARNING'
    Config.SECURITY['enable_cors'] = False
    Config.SECURITY['allowed_origins'] = ['localhost', '127.0.0.1'] 