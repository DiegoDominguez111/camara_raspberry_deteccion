import cv2
import numpy as np
import base64
from typing import Tuple, List
import json
import time
from datetime import datetime

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

def get_system_info() -> dict:
    """
    Obtiene información del sistema
    
    Returns:
        Diccionario con información del sistema
    """
    try:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    except ImportError:
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0
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