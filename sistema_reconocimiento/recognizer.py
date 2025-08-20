import numpy as np
from typing import Tuple, Optional, List
from face_db import FaceDatabase
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import time

class FaceRecognizer:
    def __init__(self, db: FaceDatabase, confidence_threshold: float = 0.6):
        self.db = db
        self.confidence_threshold = confidence_threshold
        self.recognition_history = []
        self.min_frames_for_recognition = 3  # Mínimo de frames para confirmar reconocimiento
        
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
                
                # Verificar si es un reconocimiento nuevo (no repetido recientemente)
                is_new_recognition = self._is_new_recognition(nombre, face_bbox, confianza)
                
                if is_new_recognition:
                    # Guardar log en la base de datos
                    self._save_recognition_log(nombre, confianza)
                
                return nombre, confianza, is_new_recognition
            else:
                return None, 0.0, False
                
        except Exception as e:
            print(f"Error en reconocimiento facial: {e}")
            return None, 0.0, False
    
    def _is_new_recognition(self, nombre: str, face_bbox: Tuple[int, int, int, int], confianza: float) -> bool:
        """
        Determina si un reconocimiento es nuevo (no repetido recientemente)
        """
        current_time = time.time()
        bbox_center = (face_bbox[0] + face_bbox[2]//2, face_bbox[1] + face_bbox[3]//2)
        
        # Filtrar reconocimientos recientes del mismo nombre
        recent_recognitions = [
            rec for rec in self.recognition_history 
            if rec['nombre'] == nombre and 
               current_time - rec['timestamp'] < 5.0  # 5 segundos
        ]
        
        # Si hay reconocimientos recientes, verificar si es la misma ubicación
        if recent_recognitions:
            for rec in recent_recognitions:
                rec_center = rec['bbox_center']
                distance = np.sqrt((bbox_center[0] - rec_center[0])**2 + (bbox_center[1] - rec_center[1])**2)
                
                # Si está muy cerca y es reciente, no es nuevo
                if distance < 50:  # 50 píxeles de tolerancia
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
    
    def _save_recognition_log(self, nombre: str, confianza: float):
        """Guarda un log de reconocimiento exitoso"""
        try:
            # Obtener ID de la persona
            people = self.db.list_people()
            person_id = None
            
            for person in people:
                if person[1] == nombre:
                    person_id = person[0]
                    break
            
            if person_id:
                self.db.save_log(person_id, confianza)
                
        except Exception as e:
            print(f"Error al guardar log de reconocimiento: {e}")
    
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
        
        # Reconocimientos en los últimos 5 minutos
        recent_recognitions = [
            rec for rec in self.recognition_history 
            if current_time - rec['timestamp'] < 300.0
        ]
        
        # Agrupar por nombre
        recognition_counts = {}
        for rec in recent_recognitions:
            nombre = rec['nombre']
            if nombre not in recognition_counts:
                recognition_counts[nombre] = 0
            recognition_counts[nombre] += 1
        
        return {
            'total_recent': len(recent_recognitions),
            'by_person': recognition_counts,
            'confidence_threshold': self.confidence_threshold
        }
    
    def adjust_confidence_threshold(self, new_threshold: float):
        """Ajusta el umbral de confianza para el reconocimiento"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            print(f"Umbral de confianza ajustado a: {new_threshold}")
        else:
            print("Umbral debe estar entre 0.0 y 1.0")
    
    def clear_recognition_history(self):
        """Limpia el historial de reconocimientos"""
        self.recognition_history.clear()
        print("Historial de reconocimientos limpiado") 