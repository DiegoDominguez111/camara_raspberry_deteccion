import cv2
import numpy as np
import base64
from typing import Tuple, List, Optional
import json
import time
from datetime import datetime
import psutil
import subprocess
import logging
import os

logger = logging.getLogger(__name__)

def draw_face_boxes(frame: np.ndarray, recognitions: List[Tuple[str, float, bool, Tuple[int, int, int, int]]]) -> np.ndarray:
    """
    Dibuja bounding boxes y nombres en el frame
    
    Args:
        frame: Frame de video
        recognitions: Lista de tuplas (nombre, confianza, es_nuevo, bbox)
    
    Returns:
        Frame con bounding boxes dibujados
    """
    frame_copy = frame.copy()
    
    for nombre, confianza, es_nuevo, (x, y, w, h) in recognitions:
        # Color del bounding box
        if nombre:
            if es_nuevo:
                color = (0, 255, 0)  # Verde para reconocimientos nuevos
            else:
                color = (255, 255, 0)  # Amarillo para reconocimientos existentes
        else:
            color = (0, 0, 255)  # Rojo para desconocidos
        
        # Dibujar bounding box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
        
        # Preparar texto
        if nombre:
            text = f"{nombre} ({confianza:.2f})"
        else:
            text = f"Desconocido ({confianza:.2f})"
        
        # Calcular posición del texto
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 20 else y + h + 20
        
        # Dibujar fondo del texto
        cv2.rectangle(frame_copy, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 10, text_y + 5),
                     color, -1)
        
        # Dibujar texto
        cv2.putText(frame_copy, text, (text_x + 5, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_copy

def frame_to_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    """
    Convierte un frame a formato JPEG
    
    Args:
        frame: Frame de OpenCV (BGR)
        quality: Calidad JPEG (1-100)
    
    Returns:
        Bytes del frame en formato JPEG
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer.tobytes()

def frame_to_base64(frame: np.ndarray, quality: int = 90) -> str:
    """
    Convierte un frame a string base64
    
    Args:
        frame: Frame de OpenCV (BGR)
        quality: Calidad JPEG (1-100)
    
    Returns:
        String base64 del frame
    """
    jpeg_bytes = frame_to_jpeg(frame, quality)
    return base64.b64encode(jpeg_bytes).decode('utf-8')

def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Redimensiona un frame manteniendo la proporción
    
    Args:
        frame: Frame de OpenCV
        target_width: Ancho objetivo
        target_height: Alto objetivo
    
    Returns:
        Frame redimensionado
    """
    h, w = frame.shape[:2]
    
    # Calcular ratio de aspecto
    aspect_ratio = w / h
    target_ratio = target_width / target_height
    
    if aspect_ratio > target_ratio:
        # Frame más ancho que objetivo
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # Frame más alto que objetivo
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    
    # Redimensionar
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Crear frame del tamaño objetivo con padding negro
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Centrar el frame redimensionado
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calcula FPS basado en tiempo y número de frames
    
    Args:
        start_time: Tiempo de inicio
        frame_count: Número de frames procesados
    
    Returns:
        FPS calculado
    """
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0

def format_timestamp(timestamp: str) -> str:
    """
    Formatea timestamp para mostrar en la interfaz
    
    Args:
        timestamp: Timestamp de la base de datos
    
    Returns:
        String formateado
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%d/%m/%Y %H:%M:%S')
    except:
        return timestamp

def validate_embedding(embedding: np.ndarray) -> bool:
    """
    Valida que un embedding sea válido
    
    Args:
        embedding: Vector de características
    
    Returns:
        True si es válido, False en caso contrario
    """
    if embedding is None:
        return False
    
    if not isinstance(embedding, np.ndarray):
        return False
    
    if embedding.size == 0:
        return False
    
    # Verificar que no sea todo ceros
    if np.all(embedding == 0):
        return False
    
    # Verificar que no contenga NaN o infinitos
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        return False
    
    return True

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normaliza un embedding a vector unitario
    
    Args:
        embedding: Vector de características
    
    Returns:
        Vector normalizado
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def get_system_metrics() -> dict:
    """
    Obtiene métricas del sistema: CPU, RAM, temperatura
    
    Returns:
        Diccionario con métricas del sistema
    """
    try:
        metrics = {}
        
        # CPU
        try:
            metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            metrics['cpu_count'] = psutil.cpu_count()
            metrics['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        except Exception as e:
            logger.warning(f"No se pudo obtener métricas de CPU: {e}")
            metrics['cpu_percent'] = 0
            metrics['cpu_count'] = 0
            metrics['cpu_freq'] = None
        
        # Memoria
        try:
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used'] = memory.used
            metrics['memory_total'] = memory.total
            metrics['memory_available'] = memory.available
        except Exception as e:
            logger.warning(f"No se pudo obtener métricas de memoria: {e}")
            metrics['memory_percent'] = 0
            metrics['memory_used'] = 0
            metrics['memory_total'] = 0
            metrics['memory_available'] = 0
        
        # Disco
        try:
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            metrics['disk_used'] = disk.used
            metrics['disk_total'] = disk.total
            metrics['disk_free'] = disk.free
        except Exception as e:
            logger.warning(f"No se pudo obtener métricas de disco: {e}")
            metrics['disk_percent'] = 0
            metrics['disk_used'] = 0
            metrics['disk_total'] = 0
            metrics['disk_free'] = 0
        
        # Temperatura
        try:
            temp = get_raspberry_pi_temperature()
            metrics['temperature'] = temp
        except Exception as e:
            logger.warning(f"No se pudo obtener temperatura: {e}")
            metrics['temperature'] = None
        
        # Uptime del sistema
        try:
            metrics['uptime'] = time.time() - psutil.boot_time()
        except Exception as e:
            logger.warning(f"No se pudo obtener uptime: {e}")
            metrics['uptime'] = 0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error al obtener métricas del sistema: {e}")
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0,
            'temperature': None,
            'uptime': 0
        }

def get_raspberry_pi_temperature() -> Optional[float]:
    """
    Obtiene la temperatura de la Raspberry Pi
    
    Returns:
        Temperatura en grados Celsius o None si no se puede obtener
    """
    try:
        # Intentar obtener temperatura usando vcgencmd
        result = subprocess.run(['vcgencmd', 'measure_temp'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            # El formato es "temp=XX.X'C"
            temp_str = result.stdout.strip()
            temp_value = float(temp_str.split('=')[1].split("'")[0])
            return temp_value
        else:
            logger.warning(f"vcgencmd falló: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.warning("Timeout al obtener temperatura")
        return None
    except Exception as e:
        logger.warning(f"Error al obtener temperatura: {e}")
        return None

def get_camera_metrics() -> dict:
    """
    Intenta obtener métricas de la cámara IMX500
    
    Returns:
        Diccionario con métricas de la cámara o información de error
    """
    try:
        metrics = {}
        
        # Verificar estado de la cámara
        try:
            result = subprocess.run(['rpicam-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                if 'imx500' in result.stdout:
                    metrics['camera_available'] = True
                    metrics['camera_type'] = 'IMX500'
                    
                    # Intentar obtener información de la cámara
                    try:
                        info_result = subprocess.run(['rpicam-hello', '--camera', '0', '--timeout', '1s'], 
                                                   capture_output=True, text=True, timeout=3)
                        
                        if info_result.returncode == 0:
                            # Extraer información del output
                            output_lines = info_result.stdout.split('\n')
                            
                            # Buscar información de FPS
                            for line in output_lines:
                                if 'fps' in line.lower():
                                    metrics['camera_fps'] = line.strip()
                                    break
                            
                            # Buscar información de resolución
                            for line in output_lines:
                                if 'x' in line and ('width' in line.lower() or 'height' in line.lower()):
                                    metrics['camera_resolution'] = line.strip()
                                    break
                        else:
                            metrics['camera_info'] = "No se pudo obtener información detallada"
                            
                    except Exception as e:
                        logger.warning(f"No se pudo obtener información detallada de la cámara: {e}")
                        metrics['camera_info'] = f"Error: {str(e)}"
                else:
                    metrics['camera_available'] = False
                    metrics['camera_type'] = 'None'
            else:
                metrics['camera_available'] = False
                metrics['camera_error'] = result.stderr
                
        except subprocess.TimeoutExpired:
            metrics['camera_available'] = False
            metrics['camera_error'] = 'Timeout al verificar cámara'
        except Exception as e:
            metrics['camera_available'] = False
            metrics['camera_error'] = str(e)
        
        # Verificar modelos disponibles
        try:
            models_dir = '/usr/share/imx500-models'
            if os.path.exists(models_dir):
                models = os.listdir(models_dir)
                metrics['available_models'] = [m for m in models if m.endswith('.rpk')]
                metrics['models_count'] = len(metrics['available_models'])
            else:
                metrics['available_models'] = []
                metrics['models_count'] = 0
        except Exception as e:
            logger.warning(f"No se pudo obtener información de modelos: {e}")
            metrics['available_models'] = []
            metrics['models_count'] = 0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error al obtener métricas de la cámara: {e}")
        return {
            'camera_available': False,
            'camera_error': str(e),
            'available_models': [],
            'models_count': 0
        }

def get_all_metrics() -> dict:
    """
    Obtiene todas las métricas del sistema y la cámara
    
    Returns:
        Diccionario con todas las métricas
    """
    try:
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': get_system_metrics(),
            'camera': get_camera_metrics()
        }
        
        return all_metrics
        
    except Exception as e:
        logger.error(f"Error al obtener todas las métricas: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'system': {},
            'camera': {}
        }

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Crea una imagen de prueba para testing
    
    Args:
        width: Ancho de la imagen
        height: Alto de la imagen
    
    Returns:
        Imagen de prueba
    """
    # Crear imagen con gradiente
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = int(255 * (x + y) / (width + height))
            image[y, x] = [b, g, r]
    
    # Agregar texto
    cv2.putText(image, "Imagen de Prueba", (50, height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def log_system_event(event_type: str, message: str, details: dict = None):
    """
    Registra un evento del sistema
    
    Args:
        event_type: Tipo de evento (ERROR, WARNING, INFO, SUCCESS)
        message: Mensaje del evento
        details: Detalles adicionales del evento
    """
    try:
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'details': details or {}
        }
        
        # Guardar en archivo de log
        log_file = 'tmp/system_events.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # También loggear a consola
        if event_type == 'ERROR':
            logger.error(message)
        elif event_type == 'WARNING':
            logger.warning(message)
        elif event_type == 'SUCCESS':
            logger.info(message)
        else:
            logger.info(message)
            
    except Exception as e:
        logger.error(f"Error al registrar evento del sistema: {e}") 