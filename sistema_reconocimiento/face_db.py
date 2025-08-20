import sqlite3
import numpy as np
from datetime import datetime
import json
from typing import List, Tuple, Optional
import os

class FaceDatabase:
    def __init__(self, db_path: str = "face_recognition.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos con las tablas necesarias"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de personas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confianza REAL NOT NULL,
                FOREIGN KEY (persona_id) REFERENCES personas (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_person(self, nombre: str, embedding: np.ndarray) -> bool:
        """Registra una nueva persona con su embedding facial"""
        try:
            # Convertir embedding a string JSON
            embedding_str = json.dumps(embedding.tolist())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO personas (nombre, embedding) VALUES (?, ?)",
                (nombre, embedding_str)
            )
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # Nombre ya existe
            return False
        except Exception as e:
            print(f"Error al agregar persona: {e}")
            return False
    
    def find_match(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """Busca una coincidencia en la base de datos usando distancia coseno"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT nombre, embedding FROM personas")
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return None
            
            best_match = None
            best_score = 0
            
            for nombre, embedding_str in results:
                stored_embedding = np.array(json.loads(embedding_str))
                
                # Calcular similitud coseno
                similarity = self._cosine_similarity(embedding, stored_embedding)
                
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    best_match = nombre
            
            if best_match:
                return (best_match, best_score)
            return None
            
        except Exception as e:
            print(f"Error al buscar coincidencia: {e}")
            return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula la similitud coseno entre dos vectores"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def list_people(self) -> List[Tuple[int, str, str]]:
        """Lista todas las personas registradas"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, nombre, fecha_registro FROM personas ORDER BY nombre")
            results = cursor.fetchall()
            conn.close()
            
            return results
        except Exception as e:
            print(f"Error al listar personas: {e}")
            return []
    
    def save_log(self, persona_id: Optional[int], confianza: float) -> bool:
        """Guarda un log de reconocimiento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO logs (persona_id, confianza) VALUES (?, ?)",
                (persona_id, confianza)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error al guardar log: {e}")
            return False
    
    def get_recent_logs(self, limit: int = 50) -> List[Tuple[str, str, float]]:
        """Obtiene los logs más recientes con nombres de personas"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.nombre, l.timestamp, l.confianza 
                FROM logs l 
                LEFT JOIN personas p ON l.persona_id = p.id 
                ORDER BY l.timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
        except Exception as e:
            print(f"Error al obtener logs: {e}")
            return []
    
    def delete_person(self, person_id: int) -> bool:
        """Elimina una persona y sus logs asociados"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Eliminar logs primero
            cursor.execute("DELETE FROM logs WHERE persona_id = ?", (person_id,))
            
            # Eliminar persona
            cursor.execute("DELETE FROM personas WHERE id = ?", (person_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error al eliminar persona: {e}")
            return False
    
    def get_person_by_id(self, person_id: int) -> Optional[Tuple[int, str, np.ndarray]]:
        """Obtiene una persona por su ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, nombre, embedding FROM personas WHERE id = ?", (person_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                embedding = np.array(json.loads(result[2]))
                return (result[0], result[1], embedding)
            return None
            
        except Exception as e:
            print(f"Error al obtener persona: {e}")
            return None 