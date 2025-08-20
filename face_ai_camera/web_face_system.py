#!/usr/bin/env python3
"""
Sistema de reconocimiento facial web mejorado
- Servidor web con Flask
- Stream de video en tiempo real
- Registro y reconocimiento de rostros simplificado
- Interfaz web moderna y responsive
- Integración con MobileFaceNet
"""

import subprocess
import threading
import time
import sys
import os
import cv2
import numpy as np
import face_recognition
import json
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify, Response, render_template_string, request
import psutil
import pickle
from typing import Dict, List, Tuple, Optional

# ==========================
# Configuración
# ==========================
SENSOR_WIDTH = 2028
SENSOR_HEIGHT = 1520
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# ==========================
# Estado global
# ==========================
camera_process = None
is_capturing = False
latest_frame = None
frame_lock = threading.Lock()
known_embeddings = {}
known_names = []
recognition_threshold = 0.6
current_recognition = None
recognition_stability = 0
stability_threshold = 3
isRecognitionActive = False

# Estadísticas
frames_processed = 0
faces_detected = 0
recognitions = 0
cpu_usage = 0.0
ram_usage = 0.0
cpu_temp = 0.0

# ==========================
# Funciones de embeddings
# ==========================
def save_embedding(embedding: np.ndarray, label: str, encodings_dir: str = "encodings") -> str:
    """Guarda un embedding en disco"""
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
            'timestamp': time.time()
        }, f)
    
    print(f"✅ Embedding guardado: {filepath}")
    return filepath

def load_all_embeddings(encodings_dir: str = "encodings") -> Dict[str, np.ndarray]:
    """Carga todos los embeddings guardados como diccionario"""
    embeddings_dict = {}
    
    if not os.path.exists(encodings_dir):
        return embeddings_dict
    
    for filename in os.listdir(encodings_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(encodings_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    label = data['label']
                    embedding = data['embedding']
                    
                    # Si ya existe el label, usar el embedding más reciente
                    embeddings_dict[label] = embedding
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
    
    return embeddings_dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similitud coseno entre dos vectores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula distancia euclidiana entre dos vectores"""
    return np.linalg.norm(a - b)

def find_best_match(query_embedding: np.ndarray, known_embeddings: Dict[str, np.ndarray], threshold: float = 0.6) -> Tuple[Optional[str], float]:
    """Encuentra la mejor coincidencia para un embedding"""
    if not known_embeddings:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for label, known_embedding in known_embeddings.items():
        # Usar similitud coseno
        similarity = cosine_similarity(query_embedding, known_embedding)
        
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = label
    
    return best_match, best_score

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normaliza un embedding"""
    return embedding / np.linalg.norm(embedding)

# ==========================
# Funciones de cámara
# ==========================
def start_camera_stream():
    """Inicia el stream de la cámara usando rpicam-vid"""
    global camera_process, is_capturing
    
    try:
        cmd = [
            "rpicam-vid",
            "-n",  # sin preview
            "-t", "0",  # tiempo infinito
            "--codec", "mjpeg",
            "-o", "-",  # salida a stdout
            "--width", str(SENSOR_WIDTH),
            "--height", str(SENSOR_HEIGHT),
        ]
        
        print(f"📹 Iniciando stream: {' '.join(cmd)}")
        camera_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Iniciar hilo de procesamiento de frames
        is_capturing = True
        frame_thread = threading.Thread(target=process_frames, daemon=True)
        frame_thread.start()
        
        # Esperar a que se capture el primer frame
        timeout = 10
        start_time = time.time()
        while latest_frame is None and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if latest_frame is not None:
            print("✅ Stream de cámara iniciado correctamente")
            return True
        else:
            print("❌ Timeout esperando primer frame")
            return False
            
    except Exception as e:
        print(f"❌ Error iniciando stream: {e}")
        return False

def process_frames():
    """Procesa frames del stream de rpicam-vid"""
    global latest_frame, frames_processed, faces_detected, recognitions, current_recognition
    buffer = b""
    
    while is_capturing and camera_process:
        try:
            # Leer chunk de datos
            chunk = camera_process.stdout.read(4096)
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
                    with frame_lock:
                        latest_frame = frame_data
                        frames_processed += 1
                    
                    # Procesar reconocimiento si está activo
                    if isRecognitionActive and known_embeddings:
                        process_recognition(frame_data)
                
                # Remover frame procesado del buffer
                buffer = buffer[end_pos + 2 :]
                
        except Exception as e:
            print(f"❌ Error procesando frames: {e}")
            time.sleep(0.1)

def process_recognition(frame_data: bytes):
    """Procesa reconocimiento facial en un frame"""
    global faces_detected, recognitions, current_recognition
    
    try:
        # Convertir bytes a imagen
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Detectar rostros
        face_locations = face_recognition.face_locations(frame)
        faces_detected = len(face_locations)
        
        if face_locations:
            # Generar embeddings
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for face_encoding in face_encodings:
                # Normalizar embedding
                normalized_encoding = normalize_embedding(face_encoding)
                
                # Buscar coincidencia
                best_match, confidence = find_best_match(normalized_encoding, known_embeddings, recognition_threshold)
                
                if best_match:
                    current_recognition = {
                        'name': best_match,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    recognitions += 1
                else:
                    current_recognition = {
                        'name': 'Desconocido',
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    
    except Exception as e:
        print(f"❌ Error en reconocimiento: {e}")

def stop_camera_stream():
    """Detiene el stream de la cámara"""
    global is_capturing, camera_process
    is_capturing = False
    if camera_process:
        camera_process.terminate()
        camera_process.wait()
        camera_process = None
    print("🛑 Stream de cámara detenido")

def get_current_frame():
    """Obtiene el frame actual como array de numpy"""
    with frame_lock:
        if latest_frame is None:
            return None
        
        try:
            # Convertir bytes JPEG a array numpy
            nparr = np.frombuffer(latest_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Redimensionar para display
                frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                return frame_resized
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error convirtiendo frame: {e}")
            return None

# ==========================
# Funciones de reconocimiento
# ==========================
def detect_faces_opencv(frame):
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

def generate_embedding(frame, face_location):
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
            return None
            
    except Exception as e:
        print(f"❌ Error generando embedding: {e}")
        return None

def register_new_face(person_name: str, face_image_base64: str) -> Tuple[bool, str]:
    """Registra un nuevo rostro desde la interfaz web"""
    try:
        # Decodificar imagen base64
        image_data = base64.b64decode(face_image_base64.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return False, "Error decodificando imagen"
        
        # Detectar rostros usando face_recognition
        face_locations = face_recognition.face_locations(frame)
        
        if len(face_locations) == 0:
            return False, "No se detectó ningún rostro"
        elif len(face_locations) > 1:
            return False, "Se detectaron múltiples rostros, usa solo uno"
        
        # Generar embedding
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        if len(face_encodings) == 0:
            return False, "Error generando embedding"
        
        embedding = normalize_embedding(face_encodings[0])
        
        # Guardar embedding
        save_embedding(embedding, person_name)
        
        # Recargar embeddings
        load_known_embeddings()
        
        return True, f"Rostro registrado exitosamente para: {person_name}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def load_known_embeddings():
    """Carga embeddings conocidos desde el directorio"""
    global known_embeddings, known_names
    try:
        known_embeddings = load_all_embeddings()
        known_names = list(known_embeddings.keys())
        print(f"📚 Cargados {len(known_names)} rostros conocidos:")
        for name in known_names:
            print(f"   - {name}")
    except Exception as e:
        print(f"❌ Error cargando embeddings: {e}")
        known_embeddings = {}
        known_names = []

# ==========================
# Funciones de métricas
# ==========================
def update_metrics():
    """Actualiza métricas del sistema"""
    global cpu_usage, ram_usage, cpu_temp
    while True:
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            # Lectura de temperatura en Raspberry Pi
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    cpu_temp = float(f.read().strip()) / 1000.0
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error actualizando métricas: {e}")
            time.sleep(1)

# ==========================
# Servidor web Flask
# ==========================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sistema de Reconocimiento Facial</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .video-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .video-container {
            position: relative;
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
        }
        
        .video-feed {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        .recognition-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .recognition-overlay.known {
            background: rgba(39, 174, 96, 0.9);
        }
        
        .recognition-overlay.unknown {
            background: rgba(231, 76, 60, 0.9);
        }
        
        .controls {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .control-section {
            margin-bottom: 25px;
        }
        
        .control-section h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .btn.danger {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }
        
        .btn.success {
            background: linear-gradient(135deg, #2ecc71, #27ae60);
        }
        
        .btn.warning {
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }
        
        .input-group {
            margin-bottom: 15px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 5px;
            color: #2c3e50;
            font-weight: 600;
        }
        
        .input-group input {
            width: 100%;
            padding: 10px;
            border: 2px solid #ecf0f1;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        .input-group input:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stat-card .number {
            font-size: 1.8em;
            font-weight: bold;
            display: block;
        }
        
        .stat-card .label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .registered-faces {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .face-item {
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .face-name {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .delete-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.8em;
        }
        
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .alert.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .alert.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Sistema de Reconocimiento Facial</h1>
            <p>Controla la AI Camera desde tu navegador</p>
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <h3>📹 Video en Tiempo Real</h3>
                <div class="video-container">
                    <img id="videoFeed" class="video-feed" src="/video_feed" alt="Video Feed" />
                    <div id="recognitionOverlay" class="recognition-overlay" style="display: none;">
                        <span id="recognitionText">Esperando...</span>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <div class="control-section">
                    <h3>📊 Estadísticas</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <span class="number" id="cpu_usage">0%</span>
                            <span class="label">CPU</span>
                        </div>
                        <div class="stat-card">
                            <span class="number" id="ram_usage">0%</span>
                            <span class="label">RAM</span>
                        </div>
                        <div class="stat-card">
                            <span class="number" id="cpu_temp">0°C</span>
                            <span class="label">Temperatura</span>
                        </div>
                        <div class="stat-card">
                            <span class="number" id="faces_detected">0</span>
                            <span class="label">Rostros</span>
                        </div>
                    </div>
                </div>
                
                <div class="control-section">
                    <h3>🎯 Reconocimiento</h3>
                    <button id="startRecognition" class="btn success">▶️ Iniciar Reconocimiento</button>
                    <button id="stopRecognition" class="btn danger">⏹️ Detener Reconocimiento</button>
                    <button id="reloadFaces" class="btn">🔄 Recargar Rostros</button>
                </div>
                
                <div class="control-section">
                    <h3>📝 Registro de Rostros</h3>
                    <div class="input-group">
                        <label for="personName">Nombre de la persona:</label>
                        <input type="text" id="personName" placeholder="Ej: Juan Pérez">
                    </div>
                    <button id="captureAndRegister" class="btn warning">📸 Capturar y Registrar</button>
                    <p style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
                        Posiciónate frente a la cámara y haz clic en el botón
                    </p>
                </div>
                
                <div class="control-section">
                    <h3>👥 Rostros Registrados</h3>
                    <div id="registeredFaces" class="registered-faces">
                        <p>Cargando...</p>
                    </div>
                </div>
                
                <div class="control-section">
                    <h3>⚙️ Configuración</h3>
                    <div class="input-group">
                        <label for="threshold">Umbral de reconocimiento (0.1-1.0):</label>
                        <input type="number" id="threshold" value="0.6" min="0.1" max="1.0" step="0.1">
                    </div>
                    <button id="updateThreshold" class="btn">🔧 Actualizar Umbral</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isRecognitionActive = false;
        
        // Elementos del DOM
        const videoFeed = document.getElementById('videoFeed');
        const recognitionOverlay = document.getElementById('recognitionOverlay');
        const recognitionText = document.getElementById('recognitionText');
        const startRecognitionBtn = document.getElementById('startRecognition');
        const stopRecognitionBtn = document.getElementById('stopRecognition');
        const reloadFacesBtn = document.getElementById('reloadFaces');
        const captureAndRegisterBtn = document.getElementById('captureAndRegister');
        const personNameInput = document.getElementById('personName');
        const thresholdInput = document.getElementById('threshold');
        const updateThresholdBtn = document.getElementById('updateThreshold');
        const registeredFacesDiv = document.getElementById('registeredFaces');
        
        // Funciones de utilidad
        function showAlert(message, type = 'info') {
            const alert = document.createElement('div');
            alert.className = `alert ${type}`;
            alert.textContent = message;
            document.querySelector('.container').insertBefore(alert, document.querySelector('.header').nextSibling);
            setTimeout(() => alert.remove(), 5000);
        }
        
        function updateStats() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cpu_usage').textContent = data.cpu_usage.toFixed(1) + '%';
                    document.getElementById('ram_usage').textContent = data.ram_usage.toFixed(1) + '%';
                    document.getElementById('cpu_temp').textContent = data.cpu_temp.toFixed(1) + '°C';
                    document.getElementById('faces_detected').textContent = data.faces_detected;
                    
                    // Actualizar overlay de reconocimiento
                    if (data.current_recognition) {
                        recognitionOverlay.style.display = 'block';
                        recognitionText.textContent = `${data.current_recognition.name} (${data.current_recognition.confidence.toFixed(2)})`;
                        
                        if (data.current_recognition.name === 'Desconocido') {
                            recognitionOverlay.className = 'recognition-overlay unknown';
                        } else {
                            recognitionOverlay.className = 'recognition-overlay known';
                        }
                    } else {
                        recognitionOverlay.style.display = 'none';
                    }
                })
                .catch(error => console.error('Error actualizando estadísticas:', error));
        }
        
        function loadRegisteredFaces() {
            fetch('/registered_faces')
                .then(response => response.json())
                .then(data => {
                    if (data.faces.length === 0) {
                        registeredFacesDiv.innerHTML = '<p>No hay rostros registrados</p>';
                    } else {
                        registeredFacesDiv.innerHTML = data.faces.map(face => `
                            <div class="face-item">
                                <span class="face-name">${face}</span>
                                <button class="delete-btn" onclick="deleteFace('${face}')">🗑️</button>
                            </div>
                        `).join('');
                    }
                })
                .catch(error => console.error('Error cargando rostros:', error));
        }
        
        // Event listeners
        startRecognitionBtn.addEventListener('click', () => {
            fetch('/start_recognition', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRecognitionActive = true;
                        showAlert('Reconocimiento iniciado', 'success');
                        startRecognitionBtn.disabled = true;
                        stopRecognitionBtn.disabled = false;
                    } else {
                        showAlert('Error iniciando reconocimiento', 'error');
                    }
                });
        });
        
        stopRecognitionBtn.addEventListener('click', () => {
            fetch('/stop_recognition', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRecognitionActive = false;
                        showAlert('Reconocimiento detenido', 'info');
                        startRecognitionBtn.disabled = false;
                        stopRecognitionBtn.disabled = true;
                        recognitionOverlay.style.display = 'none';
                    }
                });
        });
        
        reloadFacesBtn.addEventListener('click', () => {
            loadRegisteredFaces();
            showAlert('Rostros recargados', 'success');
        });
        
        captureAndRegisterBtn.addEventListener('click', () => {
            if (!personNameInput.value.trim()) {
                showAlert('Por favor ingresa un nombre', 'error');
                return;
            }
            
            // Capturar frame actual del video
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = videoFeed.naturalWidth || 640;
            canvas.height = videoFeed.naturalHeight || 480;
            ctx.drawImage(videoFeed, 0, 0);
            
            const capturedImage = canvas.toDataURL('image/jpeg');
            
            // Registrar rostro
            fetch('/register_face', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: personNameInput.value.trim(),
                    image: capturedImage
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert(data.message, 'success');
                    personNameInput.value = '';
                    loadRegisteredFaces();
                } else {
                    showAlert(data.message, 'error');
                }
            });
        });
        
        updateThresholdBtn.addEventListener('click', () => {
            const threshold = parseFloat(thresholdInput.value);
            if (threshold < 0.1 || threshold > 1.0) {
                showAlert('El umbral debe estar entre 0.1 y 1.0', 'error');
                return;
            }
            
            fetch('/update_threshold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ threshold: threshold })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert(`Umbral actualizado a ${threshold}`, 'success');
                } else {
                    showAlert('Error actualizando umbral', 'error');
                }
            });
        });
        
        function deleteFace(name) {
            if (confirm(`¿Estás seguro de que quieres eliminar el rostro de "${name}"?`)) {
                fetch('/delete_face', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name: name })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showAlert(`Rostro de "${name}" eliminado`, 'success');
                        loadRegisteredFaces();
                    } else {
                        showAlert('Error eliminando rostro', 'error');
                    }
                });
            }
        }
        
        // Inicialización
        document.addEventListener('DOMContentLoaded', () => {
            loadRegisteredFaces();
            updateStats();
            
            // Actualizar estadísticas cada segundo
            setInterval(updateStats, 1000);
            
            // Actualizar rostros registrados cada 5 segundos
            setInterval(loadRegisteredFaces, 5000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                
                with frame_lock:
                    frame = latest_frame
                
                if not isinstance(frame, bytes) or len(frame) < 100:
                    time.sleep(0.1)
                    continue
                
                if not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
                    time.sleep(0.1)
                    continue
                
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error en video_feed: {e}")
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global frames_processed, faces_detected, recognitions, current_recognition
    
    return jsonify({
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'cpu_temp': cpu_temp,
        'faces_detected': faces_detected,
        'frames_processed': frames_processed,
        'recognitions': recognitions,
        'current_recognition': current_recognition,
        'recognition_active': isRecognitionActive
    })

@app.route('/registered_faces')
def get_registered_faces():
    return jsonify({
        'faces': known_names
    })

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global isRecognitionActive
    isRecognitionActive = True
    return jsonify({'success': True, 'message': 'Reconocimiento iniciado'})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global isRecognitionActive
    isRecognitionActive = False
    return jsonify({'success': True, 'message': 'Reconocimiento detenido'})

@app.route('/register_face', methods=['POST'])
def register_face_endpoint():
    data = request.get_json()
    name = data.get('name')
    image = data.get('image')
    
    if not name or not image:
        return jsonify({'success': False, 'message': 'Datos incompletos'})
    
    success, message = register_new_face(name, image)
    return jsonify({'success': success, 'message': message})

@app.route('/update_threshold', methods=['POST'])
def update_threshold_endpoint():
    global recognition_threshold
    data = request.get_json()
    threshold = data.get('threshold')
    
    if threshold is None or threshold < 0.1 or threshold > 1.0:
        return jsonify({'success': False, 'message': 'Umbral inválido'})
    
    recognition_threshold = threshold
    return jsonify({'success': True, 'message': f'Umbral actualizado a {threshold}'})

@app.route('/delete_face', methods=['POST'])
def delete_face_endpoint():
    data = request.get_json()
    name = data.get('name')
    
    if not name:
        return jsonify({'success': False, 'message': 'Nombre no proporcionado'})
    
    try:
        # Eliminar archivo de embedding
        embedding_file = os.path.join('encodings', f'{name}.pkl')
        if os.path.exists(embedding_file):
            os.remove(embedding_file)
            load_known_embeddings()  # Recargar
            return jsonify({'success': True, 'message': f'Rostro de {name} eliminado'})
        else:
            return jsonify({'success': False, 'message': 'Rostro no encontrado'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

def start_web_server():
    print("🌐 Iniciando servidor web en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# ==========================
# Main
# ==========================
def main():
    print("🚀 Iniciando sistema de reconocimiento facial web mejorado...")
    
    # Cargar embeddings conocidos
    load_known_embeddings()
    
    # Iniciar cámara
    if not start_camera_stream():
        print("❌ Error iniciando cámara")
        return
    
    # Iniciar hilos
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    
    web_thread.start()
    metrics_thread.start()
    
    print("✅ Sistema iniciado correctamente")
    print("🌐 Accede a http://localhost:5000 en tu navegador")
    print("📱 O desde otro dispositivo: http://[IP_DE_LA_RASPBERRY_PI]:5000")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
    finally:
        stop_camera_stream()
        print("✅ Sistema detenido")

if __name__ == "__main__":
    main() 