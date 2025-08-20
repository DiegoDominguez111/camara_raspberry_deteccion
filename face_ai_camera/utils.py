#!/usr/bin/env python3
"""
Funciones auxiliares para el sistema de reconocimiento facial
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional
import pickle
from pathlib import Path


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos embeddings
    
    Args:
        embedding1: Primer embedding
        embedding2: Segundo embedding
        
    Returns:
        Similitud coseno (0-1, donde 1 es idéntico)
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana entre dos embeddings
    
    Args:
        embedding1: Primer embedding
        embedding2: Segundo embedding
        
    Returns:
        Distancia euclidiana (menor = más similar)
    """
    return np.linalg.norm(embedding1 - embedding2)


def find_best_match(query_embedding: np.ndarray, 
                   known_embeddings: List[np.ndarray], 
                   labels: List[str],
                   threshold: float = 0.6,
                   method: str = 'cosine') -> Tuple[Optional[str], float]:
    """
    Encuentra la mejor coincidencia para un embedding dado
    
    Args:
        query_embedding: Embedding a comparar
        known_embeddings: Lista de embeddings conocidos
        labels: Lista de etiquetas correspondientes
        threshold: Umbral de similitud mínima
        method: Método de comparación ('cosine' o 'euclidean')
        
    Returns:
        Tupla (etiqueta, similitud) o (None, 0.0) si no hay coincidencia
    """
    if not known_embeddings or not labels:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for i, known_embedding in enumerate(known_embeddings):
        if method == 'cosine':
            similarity = cosine_similarity(query_embedding, known_embedding)
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = labels[i]
        else:  # euclidean
            distance = euclidean_distance(query_embedding, known_embedding)
            # Convertir distancia a similitud (1 / (1 + distance))
            similarity = 1 / (1 + distance)
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = labels[i]
    
    return best_match, best_score


def save_embedding(embedding: np.ndarray, label: str, encodings_dir: str = "encodings"):
    """
    Guarda un embedding en disco
    
    Args:
        embedding: Embedding a guardar
        label: Etiqueta/nombre de la persona
        encodings_dir: Directorio donde guardar
    """
    os.makedirs(encodings_dir, exist_ok=True)
    
    # Crear nombre de archivo único
    base_filename = f"{label.lower().replace(' ', '_')}"
    counter = 1
    filename = f"{base_filename}_{counter}.pkl"
    
    while os.path.exists(os.path.join(encodings_dir, filename)):
        counter += 1
        filename = f"{base_filename}_{counter}.pkl"
    
    filepath = os.path.join(encodings_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump({
            'embedding': embedding,
            'label': label,
            'timestamp': np.datetime64('now')
        }, f)
    
    print(f"Embedding guardado: {filepath}")
    return filepath


def load_all_embeddings(encodings_dir: str = "encodings") -> Tuple[List[np.ndarray], List[str]]:
    """
    Carga todos los embeddings guardados
    
    Args:
        encodings_dir: Directorio donde buscar embeddings
        
    Returns:
        Tupla (embeddings, labels)
    """
    embeddings = []
    labels = []
    
    if not os.path.exists(encodings_dir):
        return embeddings, labels
    
    for filename in os.listdir(encodings_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(encodings_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    embeddings.append(data['embedding'])
                    labels.append(data['label'])
            except Exception as e:
                print(f"Error cargando {filename}: {e}")
    
    return embeddings, labels


def list_registered_faces(encodings_dir: str = "encodings") -> List[str]:
    """
    Lista todas las personas registradas
    
    Args:
        encodings_dir: Directorio de embeddings
        
    Returns:
        Lista de nombres únicos registrados
    """
    embeddings, labels = load_all_embeddings(encodings_dir)
    return list(set(labels))


def delete_face_encodings(label: str, encodings_dir: str = "encodings"):
    """
    Elimina todos los embeddings de una persona
    
    Args:
        label: Nombre de la persona
        encodings_dir: Directorio de embeddings
    """
    if not os.path.exists(encodings_dir):
        return
    
    deleted_count = 0
    for filename in os.listdir(encodings_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(encodings_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if data['label'].lower() == label.lower():
                        os.remove(filepath)
                        deleted_count += 1
                        print(f"Eliminado: {filename}")
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
    
    print(f"Se eliminaron {deleted_count} embeddings para '{label}'")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normaliza un embedding para mejorar la comparación
    
    Args:
        embedding: Embedding a normalizar
        
    Returns:
        Embedding normalizado
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def validate_embedding(embedding: np.ndarray) -> bool:
    """
    Valida que un embedding tenga el formato correcto
    
    Args:
        embedding: Embedding a validar
        
    Returns:
        True si es válido, False en caso contrario
    """
    if not isinstance(embedding, np.ndarray):
        return False
    
    if embedding.ndim != 1:
        return False
    
    if embedding.size == 0:
        return False
    
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        return False
    
    return True 