import numpy as np
from typing import Tuple, Optional, List
from face_db import FaceDatabase
import cv2
import time
import json
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, db: FaceDatabase, confidence_threshold: float = 0.6):
        self.db = db
        self.confidence_threshold = confidence_threshold
        self.recognition_history = []
        self.min_frames_for_recognition = 3
    
    def recognize_face(self, embedding: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], float, bool]:
        """
        Reconoce un rostro comparando su embedding con la base de datos
        
        Args:
            embedding: Vector de características del rostro
            face_bbox: Bounding box del rostro (x, y, w, h)
        
        Returns:
            Tuple de (nombre, confianza, es_nuevo_reconocimiento)
        """
        try:
            # Buscar coincidencia en la base de datos
            match_result = self.db.find_match(embedding, self.confidence_threshold)
            
            if match_result:
                nombre, confianza = match_result
                
                # Verificar si es un reconocimiento nuevo
                is_new_recognition = self._is_new_recognition(nombre, face_bbox, confianza)
                
                if is_new_recognition:
                    # Guardar log de reconocimiento
                    self._save_recognition_log(nombre, confianza, embedding, face_bbox)
                
                return nombre, confianza, is_new_recognition
            else:
                # Rostro desconocido
                return None, 0.0, False
                
        except Exception as e:
            logger.error(f"Error en reconocimiento facial: {e}")
            return None, 0.0, False
    
    def _is_new_recognition(self, nombre: str, face_bbox: Tuple[int, int, int, int], confianza: float) -> bool:
        """
        Determina si un reconocimiento es nuevo (no repetido recientemente)
        """
        current_time = time.time()
        
        # Calcular centro del bounding box
        bbox_center = (face_bbox[0] + face_bbox[2]//2, face_bbox[1] + face_bbox[3]//2)
        
        # Buscar reconocimientos recientes del mismo nombre
        recent_recognitions = [
            rec for rec in self.recognition_history
            if rec['nombre'] == nombre and 
               current_time - rec['timestamp'] < 5.0  # 5 segundos
        ]
        
        if recent_recognitions:
            # Verificar si el rostro está en la misma posición
            for rec in recent_recognitions:
                rec_center = rec['bbox_center']
                distance = np.sqrt(
                    (bbox_center[0] - rec_center[0])**2 + 
                    (bbox_center[1] - rec_center[1])**2
                )
                
                # Si está muy cerca, considerar el mismo reconocimiento
                if distance < 50:  # 50 píxeles
                    return False
        
        # Agregar a historial
        self.recognition_history.append({
            'nombre': nombre,
            'timestamp': current_time,
            'bbox_center': bbox_center,
            'confianza': confianza
        })
        
        # Limpiar historial antiguo (más de 30 segundos)
        self.recognition_history = [
            rec for rec in self.recognition_history
            if current_time - rec['timestamp'] < 30.0
        ]
        
        return True
    
    def _save_recognition_log(self, nombre: str, confianza: float, embedding: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """Guarda un log de reconocimiento exitoso"""
        try:
            # Buscar ID de la persona
            people = self.db.list_people()
            person_id = None
            
            for person in people:
                if person[1] == nombre:  # person[1] es el nombre
                    person_id = person[0]  # person[0] es el ID
                    break
            
            # Crear payload con información del reconocimiento
            raw_payload = {
                'nombre': nombre,
                'confianza': confianza,
                'bbox': face_bbox,
                'embedding_length': len(embedding),
                'timestamp': time.time(),
                'recognition_type': 'success'
            }
            
            # Guardar log
            if person_id:
                self.db.save_log(person_id, confianza, json.dumps(raw_payload))
                logger.info(f"Log guardado para {nombre} (ID: {person_id})")
            else:
                logger.warning(f"No se encontró ID para persona: {nombre}")
                
        except Exception as e:
            logger.error(f"Error al guardar log de reconocimiento: {e}")
    
    def batch_recognize(self, face_data: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[Optional[str], float, bool, Tuple[int, int, int, int]]]:
        """
        Reconoce múltiples rostros en un frame
        
        Args:
            face_data: Lista de tuplas (embedding, bbox)
        
        Returns:
            Lista de tuplas (nombre, confianza, es_nuevo, bbox)
        """
        results = []
        
        for embedding, bbox in face_data:
            nombre, confianza, es_nuevo = self.recognize_face(embedding, bbox)
            results.append((nombre, confianza, es_nuevo, bbox))
        
        return results
    
    def get_recognition_stats(self) -> dict:
        """Obtiene estadísticas del reconocimiento"""
        current_time = time.time()
        
        # Reconocimientos recientes (últimos 5 minutos)
        recent_recognitions = [
            rec for rec in self.recognition_history
            if current_time - rec['timestamp'] < 300.0
        ]
        
        # Contar por persona
        recognition_counts = {}
        for rec in recent_recognitions:
            nombre = rec['nombre']
            if nombre not in recognition_counts:
                recognition_counts[nombre] = 0
            recognition_counts[nombre] += 1
        
        return {
            'total_recent': len(recent_recognitions),
            'by_person': recognition_counts,
            'confidence_threshold': self.confidence_threshold,
            'history_size': len(self.recognition_history)
        }
    
    def adjust_confidence_threshold(self, new_threshold: float):
        """Ajusta el umbral de confianza para el reconocimiento"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Umbral de confianza ajustado a: {new_threshold}")
        else:
            logger.warning("Umbral debe estar entre 0.0 y 1.0")
    
    def clear_recognition_history(self):
        """Limpia el historial de reconocimientos"""
        self.recognition_history.clear()
        logger.info("Historial de reconocimientos limpiado")
    
    def get_unknown_face_log(self, embedding: np.ndarray, face_bbox: Tuple[int, int, int, int], confianza: float):
        """Guarda log de rostro desconocido"""
        try:
            # Crear payload para rostro desconocido
            raw_payload = {
                'nombre': 'Desconocido',
                'confianza': confianza,
                'bbox': face_bbox,
                'embedding_length': len(embedding),
                'timestamp': time.time(),
                'recognition_type': 'unknown'
            }
            
            # Guardar log sin persona_id (persona_id = NULL)
            self.db.save_log(None, confianza, json.dumps(raw_payload))
            logger.info("Log de rostro desconocido guardado")
            
        except Exception as e:
            logger.error(f"Error al guardar log de rostro desconocido: {e}")
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Valida que un embedding sea válido"""
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
        
        # Verificar longitud esperada (128 para MobileFaceNet)
        if len(embedding) != 128:
            logger.warning(f"Embedding con longitud inesperada: {len(embedding)} (esperado: 128)")
            return False
        
        return True 