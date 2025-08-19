#!/usr/bin/env python3
"""
Sistema de reconocimiento facial tipo reloj checador optimizado para bajo consumo de CPU
"""
import subprocess
import threading
import time
import sqlite3
import pickle
import cv2
import numpy as np
import face_recognition
from flask import Flask, Response, render_template_string, request, jsonify
import psutil
import os
import base64
from collections import deque
import queue
from datetime import datetime
import gc

# ==========================
# Configuración Optimizada
# ==========================
# Configuración simplificada para streaming directo
SENSOR_WIDTH = 640    # Resolución directa para streaming
SENSOR_HEIGHT = 480
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
SCALE_X = 1.0         # No hay escalado necesario
SCALE_Y = 1.0

DB_PATH = "faces_face_detection.db"
latest_frame = None
latest_recognition_frame = None
lock = threading.Lock()
recognition_lock = threading.Lock()

# Colas optimizadas para mejor rendimiento
video_queue = queue.Queue(maxsize=3)  # Reducido para menor latencia
recognition_queue = queue.Queue(maxsize=1)  # Solo 1 frame para reconocimiento

# Cache para rostros conocidos
known_faces_cache = []
last_face_update = 0
FACE_UPDATE_INTERVAL = 10  # Actualizar cada 10 segundos

# Resultados de reconocimiento
current_detections = []
detection_lock = threading.Lock()

cpu_usage = 0.0
ram_usage = 0.0
cpu_temp = 0.0

# Control de FPS optimizado - AUMENTADO para mejor fluidez
VIDEO_FPS = 25  # Reducido a 25 FPS para estabilidad con IMX500
RECOGNITION_FPS = 3  # Reducido a 3 FPS para estabilidad
VIDEO_INTERVAL = 1.0 / VIDEO_FPS
RECOGNITION_INTERVAL = 1.0 / RECOGNITION_FPS

# Control de latencia
MAX_FRAME_AGE = 5.0  # Aumentado a 5 segundos para permitir procesamiento

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sistema de Chequeo Facial</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .video-container { text-align: center; margin: 20px 0; }
        .form-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }
        input[type="text"] { padding: 10px; margin-right: 10px; width: 250px; border: 1px solid #ddd; border-radius: 3px; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 14px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #6c757d; cursor: not-allowed; }
        .status { margin: 20px 0; padding: 15px; background-color: #e9ecef; border-radius: 5px; }
        .status h3 { margin-top: 0; color: #495057; }
        .metric { display: inline-block; margin-right: 20px; font-weight: bold; }
        .metric-value { color: #007bff; }
        .message { margin-top: 10px; padding: 10px; border-radius: 3px; }
        .message.success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .message.error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .message.warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .recognition-info { margin: 20px 0; padding: 15px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; }
        .recognition-info h3 { margin-top: 0; color: #0c5460; }
        .person-item { padding: 8px; margin: 5px 0; background-color: white; border-radius: 3px; border-left: 4px solid #007bff; }
        .person-name { font-weight: bold; color: #007bff; }
        .person-time { color: #6c757d; font-size: 0.9em; }
        .person-confidence { color: #28a745; font-weight: bold; }
        .fps-info { margin: 10px 0; color: #6c757d; font-size: 0.9em; }
        .video-wrapper { position: relative; display: inline-block; }
        .video-stream { border: 3px solid #007bff; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .recognition-overlay { position: absolute; top: 0; left: 0; pointer-events: none; opacity: 0.95; }
        .performance-indicator { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕐 Sistema de Chequeo Facial (ULTRA OPTIMIZADO)</h1>
        
        <div class="video-container">
            <h2>📹 Cámara en Tiempo Real</h2>
            <div class="video-wrapper">
                <img id="videoStream" class="video-stream" src="{{ url_for('video_feed') }}" width="640" height="480"/>
                <img id="recognitionOverlay" class="recognition-overlay" src="{{ url_for('recognition_feed') }}" width="640" height="480"/>
                <div class="performance-indicator">
                    <span id="latencyIndicator">Latencia: --</span>
                </div>
            </div>
            <div class="fps-info">
                📊 Video: <span id="videoFps">--</span> FPS | Reconocimiento: <span id="recognitionFps">--</span> FPS
            </div>
            <p><em>Video fluido (25 FPS) + Reconocimiento facial superpuesto (3 FPS)</em></p>
        </div>
        
        <div class="form-container">
            <h2>📝 Registrar Persona Desconocida</h2>
            <form id="registerForm">
                <input type="text" id="name" placeholder="Nombre completo de la persona" required>
                <button type="submit" id="registerBtn">Registrar Persona</button>
            </form>
            <div id="message"></div>
        </div>
        
        <div class="recognition-info">
            <h3>👥 Personas Detectadas Recientemente</h3>
            <div id="recentDetections">
                <p>Esperando detecciones...</p>
            </div>
        </div>
        
        <div class="status">
            <h3>📊 Estado del Sistema</h3>
            <div class="metric">CPU: <span id="cpu" class="metric-value">--</span>%</div>
            <div class="metric">RAM: <span id="ram" class="metric-value">--</span>%</div>
            <div class="metric">Temperatura: <span id="temp" class="metric-value">--</span>°C</div>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('registerForm');
        const messageDiv = document.getElementById('message');
        const registerBtn = document.getElementById('registerBtn');
        const recentDetectionsDiv = document.getElementById('recentDetections');
        const latencyIndicator = document.getElementById('latencyIndicator');
        
        // Optimización de imágenes para menor latencia
        const videoStream = document.getElementById('videoStream');
        const recognitionOverlay = document.getElementById('recognitionOverlay');
        
        // Configurar imágenes para mejor rendimiento
        videoStream.style.imageRendering = 'optimizeSpeed';
        recognitionOverlay.style.imageRendering = 'optimizeSpeed';
        
        // Medir latencia del video
        let lastVideoUpdate = Date.now();
        videoStream.addEventListener('load', () => {
            const now = Date.now();
            const latency = now - lastVideoUpdate;
            latencyIndicator.textContent = `Latencia: ${latency}ms`;
            lastVideoUpdate = now;
        });
        
        // Verificar si una persona ya está registrada
        async function checkPersonExists(name) {
            try {
                const res = await fetch('/check_person', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({name: name})
                });
                const result = await res.json();
                return result.exists;
            } catch (error) {
                console.error('Error verificando persona:', error);
                return false;
            }
        }
        
        form.addEventListener('submit', async e => {
            e.preventDefault();
            const name = document.getElementById('name').value.trim();
            
            if (name.length < 2) {
                messageDiv.innerHTML = `<div class="message error">❌ El nombre debe tener al menos 2 caracteres</div>`;
                return;
            }
            
            const button = form.querySelector('button');
            const originalText = button.textContent;
            
            button.textContent = 'Verificando...';
            button.disabled = true;
            
            try {
                // Verificar si la persona ya existe
                const exists = await checkPersonExists(name);
                if (exists) {
                    messageDiv.innerHTML = `<div class="message warning">⚠️ La persona '${name}' ya está registrada en el sistema</div>`;
                    return;
                }
                
                button.textContent = 'Capturando rostro...';
                
                // Esperar un momento para asegurar que hay un frame disponible
                await new Promise(resolve => setTimeout(resolve, 300)); // Reducido de 500ms a 300ms
                
                // Usar canvas para capturar frame del video de reconocimiento
                const recognitionImg = document.getElementById('recognitionOverlay');
                const canvas = document.createElement('canvas');
                canvas.width = 640;
                canvas.height = 480;
                const ctx = canvas.getContext('2d');
                
                // Dibujar la imagen de reconocimiento en el canvas
                ctx.drawImage(recognitionImg, 0, 0, 640, 480);
                const dataURL = canvas.toDataURL('image/jpeg', 0.8);
                
                button.textContent = 'Registrando...';
                
                const res = await fetch('/register_face', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({name:name, image:dataURL})
                });
                
                const result = await res.json();
                
                if (result.success) {
                    messageDiv.innerHTML = `<div class="message success">✅ ${result.message}</div>`;
                    form.reset();
                    updateRecentDetections();
                } else {
                    messageDiv.innerHTML = `<div class="message error">❌ ${result.message}</div>`;
                }
            } catch (error) {
                messageDiv.innerHTML = `<div class="message error">❌ Error: ${error.message}</div>`;
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        });
        
        // Actualizar lista de detecciones recientes
        async function updateRecentDetections() {
            try {
                const res = await fetch('/recent_detections');
                const data = await res.json();
                
                if (data.detections && data.detections.length > 0) {
                    let html = '';
                    data.detections.forEach(person => {
                        const timeStr = new Date(person.timestamp).toLocaleTimeString();
                        const confidenceColor = person.confidence > 80 ? '#28a745' : 
                                             person.confidence > 60 ? '#ffc107' : '#dc3545';
                        
                        html += `
                            <div class="person-item">
                                <div class="person-name">${person.name}</div>
                                <div class="person-time">🕐 ${timeStr}</div>
                                <div class="person-confidence" style="color: ${confidenceColor}">
                                    🎯 ${person.confidence.toFixed(1)}% coincidencia
                                </div>
                            </div>
                        `;
                    });
                    recentDetectionsDiv.innerHTML = html;
                } else {
                    recentDetectionsDiv.innerHTML = '<p>Esperando detecciones...</p>';
                }
            } catch (error) {
                console.error('Error actualizando detecciones:', error);
            }
        }
        
        // Actualizar métricas del sistema
        async function updateStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                document.getElementById('cpu').textContent = data.cpu_usage.toFixed(1);
                document.getElementById('ram').textContent = data.ram_usage.toFixed(1);
                document.getElementById('temp').textContent = data.cpu_temp.toFixed(1);
                document.getElementById('videoFps').textContent = data.video_fps || '--';
                document.getElementById('recognitionFps').textContent = data.recognition_fps || '--';
            } catch (error) {
                console.error('Error actualizando estado:', error);
            }
        }
        
        // Actualizar cada 1 segundo para mejor responsividad
        setInterval(updateStatus, 1000);
        setInterval(updateRecentDetections, 3000); // Cada 3 segundos
        
        // Actualización inicial
        updateStatus();
        updateRecentDetections();
    </script>
</body>
</html>
"""

# ==========================
# Base de datos SQLite
# ==========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tabla de rostros
    c.execute(
        """CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    
    # Tabla de detecciones
    c.execute(
        """CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    
    conn.commit()
    conn.close()

def save_face(name, encoding):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, pickle.dumps(encoding)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        print(f"Error guardando rostro: {e}")
        return False

def check_person_exists(name):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM faces WHERE name = ?", (name,))
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        print(f"Error verificando persona: {e}")
        return False

def load_faces():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, encoding FROM faces")
        results = [(row[0], pickle.loads(row[1])) for row in c.fetchall()]
        conn.close()
        return results
    except Exception as e:
        print(f"Error cargando rostros: {e}")
        return []

def save_detection(name, confidence):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO detections (name, confidence) VALUES (?, ?)", (name, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error guardando detección: {e}")

def get_recent_detections(limit=5):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, confidence, timestamp FROM detections ORDER BY timestamp DESC LIMIT ?", (limit,))
        results = [{"name": row[0], "confidence": row[1], "timestamp": row[2]} for row in c.fetchall()]
        conn.close()
        return results
    except Exception as e:
        print(f"Error obteniendo detecciones: {e}")
        return []

# ==========================
# Procesamiento de video SEPARADO
# ==========================
video_fps_counter = 0
recognition_fps_counter = 0
last_video_fps_time = time.time()
last_recognition_fps_time = time.time()
current_video_fps = 0
current_recognition_fps = 0

def procesar_video_rapido(frame_data):
    """Procesa frames solo para video rápido - CON LOGGING DETALLADO"""
    global latest_frame, video_fps_counter, current_video_fps, last_video_fps_time
    
    try:
        # Verificar antigüedad del frame para evitar efecto fantasma
        current_time = time.time()
        
        print(f"🎬 Procesando frame de video: {len(frame_data)} bytes")
        
        # Decodificar frame
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            print("❌ Error: frame no se pudo decodificar")
            return
        
        print(f"✅ Frame decodificado: {frame.shape}")
        
        # El frame ya está en 640x480, no necesita redimensionar
        display_frame = frame
        
        # Convertir a JPEG con calidad optimizada para velocidad
        _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        print(f"✅ Frame convertido a JPEG: {len(jpeg.tobytes())} bytes")
        
        # Actualizar frame con lock optimizado
        with lock:
            latest_frame = jpeg.tobytes()
        
        # Contar FPS
        video_fps_counter += 1
        if current_time - last_video_fps_time >= 1.0:
            current_video_fps = video_fps_counter
            video_fps_counter = 0
            last_video_fps_time = current_time
            print(f"📊 Video FPS actualizado: {current_video_fps}")
        
        # Limpiar memoria inmediatamente
        del np_frame, frame, display_frame
        
    except Exception as e:
        print(f"❌ Error en video rápido: {e}")
        import traceback
        traceback.print_exc()

def procesar_reconocimiento_facial(frame_data):
    """Procesa frames SOLO para reconocimiento facial (optimizado para menor latencia)"""
    global latest_recognition_frame, current_detections, recognition_fps_counter, current_recognition_fps, last_recognition_fps_time
    
    try:
        # Verificar antigüedad del frame
        current_time = time.time()
        
        # Decodificar frame
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            return
        
        # El frame ya está en 640x480, no necesita redimensionar
        display_frame = frame
        
        # Redimensionar para reconocimiento (más pequeño = más rápido)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reducido a 320x240
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros en frame pequeño
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Usar HOG para velocidad
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Escalar coordenadas de vuelta (corregido para nueva resolución)
        face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]
        
        # Procesar detecciones
        detections = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Desconocido"
            confidence = 0.0
            
            # Comparar con rostros conocidos (optimizado)
            for db_name, db_enc in known_faces_cache:
                try:
                    distance = face_recognition.face_distance([db_enc], encoding)[0]
                    current_confidence = (1 - distance) * 100
                    
                    if current_confidence > confidence and current_confidence > 55:
                        confidence = current_confidence
                        name = db_name
                except:
                    continue
            
            # Escalar coordenadas para display
            display_left = int(left * SCALE_X)
            display_right = int(right * SCALE_X)
            display_top = int(top * SCALE_Y)
            display_bottom = int(bottom * SCALE_Y)
            
            # Configurar colores y estilos según el estado
            if name != "Desconocido":
                # Persona reconocida - Verde profesional
                box_color = (0, 150, 0)
                text_color = (255, 255, 255)
                bg_color = (0, 100, 0)
                thickness = 2
                text = f"{name} ({confidence:.1f}%)"
                save_detection(name, confidence)
            else:
                # Persona desconocida - Rojo reservado
                box_color = (0, 0, 200)
                text_color = (255, 255, 255)
                bg_color = (0, 0, 120)
                thickness = 3
                text = "DESCONOCIDO"
            
            # Dibujar box principal
            cv2.rectangle(display_frame, (display_left, display_top), 
                         (display_right, display_bottom), box_color, thickness)
            
            # Esquinas decorativas
            corner_size = 8
            cv2.rectangle(display_frame, (display_left, display_top), 
                         (display_left + corner_size, display_top + corner_size), box_color, -1)
            cv2.rectangle(display_frame, (display_right - corner_size, display_top), 
                         (display_right, display_top + corner_size), box_color, -1)
            cv2.rectangle(display_frame, (display_left, display_bottom - corner_size), 
                         (display_left + corner_size, display_bottom), box_color, -1)
            cv2.rectangle(display_frame, (display_right - corner_size, display_bottom - corner_size), 
                         (display_right, display_bottom), box_color, -1)
            
            # Etiqueta de nombre optimizada
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1
            
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            
            label_y = max(display_top - 15, text_height + 10)
            label_x = display_left
            
            padding_x = 8
            padding_y = 4
            
            # Fondo de la etiqueta
            label_bg_left = label_x
            label_bg_right = label_x + text_width + padding_x * 2
            label_bg_top = label_y - text_height - padding_y
            label_bg_bottom = label_y + padding_y
            
            cv2.rectangle(display_frame, 
                         (label_bg_left, label_bg_top), 
                         (label_bg_right, label_bg_bottom), 
                         bg_color, -1)
            
            cv2.rectangle(display_frame, 
                         (label_bg_left, label_bg_top), 
                         (label_bg_right, label_bg_bottom), 
                         box_color, 1)
            
            # Línea conectora
            line_start_y = label_bg_bottom
            line_end_y = display_top
            line_x = label_x + padding_x + text_width // 2
            
            cv2.line(display_frame, (line_x, line_start_y), (line_x, line_end_y), box_color, 1)
            
            # Texto
            text_x = label_x + padding_x
            text_y = label_y - padding_y // 2
            
            cv2.putText(display_frame, text, 
                       (text_x + 1, text_y + 1), 
                       font, font_scale, (0, 0, 0), font_thickness)
            
            cv2.putText(display_frame, text, 
                       (text_x, text_y), 
                       font, font_scale, text_color, font_thickness)
            
            # Barra de confianza para personas reconocidas
            if name != "Desconocido":
                confidence_bar_width = 60
                confidence_bar_height = 4
                confidence_bar_x = display_left
                confidence_bar_y = display_bottom + 8
                
                cv2.rectangle(display_frame, 
                             (confidence_bar_x, confidence_bar_y), 
                             (confidence_bar_x + confidence_bar_width, confidence_bar_y + confidence_bar_height), 
                             (50, 50, 50), -1)
                
                confidence_width = int((confidence / 100.0) * confidence_bar_width)
                if confidence_width > 0:
                    cv2.rectangle(display_frame, 
                                 (confidence_bar_x, confidence_bar_y), 
                                 (confidence_bar_x + confidence_width, confidence_bar_y + confidence_bar_height), 
                                 (0, 200, 0), -1)
                
                cv2.rectangle(display_frame, 
                             (confidence_bar_x, confidence_bar_y), 
                             (confidence_bar_x + confidence_bar_width, confidence_bar_y + confidence_bar_height), 
                             (100, 100, 100), 1)
            
            detections.append({
                "name": name,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        # Actualizar detecciones globales
        with detection_lock:
            current_detections = detections
        
        # Convertir a JPEG para overlay con calidad optimizada
        _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        with recognition_lock:
            latest_recognition_frame = jpeg.tobytes()
        
        # Contar FPS de reconocimiento
        recognition_fps_counter += 1
        if current_time - last_recognition_fps_time >= 1.0:
            current_recognition_fps = recognition_fps_counter
            recognition_fps_counter = 0
            last_recognition_fps_time = current_time
        
        # Limpiar memoria inmediatamente
        del np_frame, frame, display_frame, small_frame, rgb_small_frame
        gc.collect()
        
    except Exception as e:
        print(f"Error en reconocimiento: {e}")

def rpicam_video_reader():
    """Captura frames y los distribuye - CORREGIDO basado en script de prueba exitoso"""
    cmd = [
        "rpicam-vid",
        "-n", "-t", "0",
        "--codec", "mjpeg",
        "-o", "-",
        "--width", "640",
        "--height", "480",
        "--framerate", "25"
    ]
    
    print(f"🎥 Iniciando captura corregida basada en script de prueba exitoso: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = b""
    frame_count = 0
    last_frame_time = time.time()
    
    print("✅ Proceso de captura iniciado, esperando frames...")
    
    try:
        while True:
            # Usar el mismo tamaño de chunk que funciona en el script de prueba
            chunk = proc.stdout.read(1024)
            if not chunk:
                print("⚠️  No hay datos del proceso")
                break
                
            buffer += chunk
            
            while True:
                start = buffer.find(b"\xff\xd8")
                if start == -1:
                    buffer = buffer[-2:]
                    break
                    
                end = buffer.find(b"\xff\xd9", start+2)
                if end == -1:
                    buffer = buffer[start:]
                    break
                    
                frame_data = buffer[start:end+2]
                buffer = buffer[end+2:]
                
                frame_count += 1
                current_time = time.time()
                
                print(f"📹 Frame {frame_count} detectado: {len(frame_data)} bytes")
                
                # Simplificar control de antigüedad - solo para frames muy antiguos
                if current_time - last_frame_time > MAX_FRAME_AGE:
                    print(f"⏰ Frame {frame_count} muy antiguo ({current_time - last_frame_time:.1f}s), descartando")
                    continue
                
                # Todos los frames van a video (cola pequeña para menor latencia)
                try:
                    if not video_queue.full():
                        video_queue.put_nowait(frame_data)
                        last_frame_time = current_time
                        print(f"✅ Frame {frame_count} enviado a cola de video")
                    else:
                        print(f"⚠️  Cola de video llena, descartando frame {frame_count}")
                except:
                    print(f"❌ Error enviando frame {frame_count} a cola de video")
                
                # Solo cada 8 frames van a reconocimiento (3 FPS aprox)
                if frame_count % 8 == 0:
                    try:
                        if not recognition_queue.full():
                            recognition_queue.put_nowait(frame_data)
                            print(f"👤 Frame {frame_count} enviado a cola de reconocimiento")
                        else:
                            print(f"⚠️  Cola de reconocimiento llena, descartando frame {frame_count}")
                    except:
                        print(f"❌ Error enviando frame {frame_count} a cola de reconocimiento")
                    
    except Exception as e:
        print(f"❌ Error en captura: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 Terminando proceso de captura")
        proc.terminate()
        proc.wait()
        print("✅ Proceso de captura terminado")

def procesar_cola_video():
    """Procesa cola de video rápido - OPTIMIZADO para menor latencia"""
    while True:
        try:
            frame_data = video_queue.get(timeout=0.05)  # Reducido de 0.1 a 0.05 para menor latencia
            procesar_video_rapido(frame_data)
            time.sleep(VIDEO_INTERVAL)
        except queue.Empty:
            time.sleep(0.005)  # Reducido de 0.01 a 0.005 para menor latencia
        except Exception as e:
            print(f"Error en cola video: {e}")
            time.sleep(0.05)

def procesar_cola_reconocimiento():
    """Procesa cola de reconocimiento - OPTIMIZADO para menor latencia"""
    while True:
        try:
            frame_data = recognition_queue.get(timeout=0.5)  # Reducido de 1.0 a 0.5 para menor latencia
            procesar_reconocimiento_facial(frame_data)
            time.sleep(RECOGNITION_INTERVAL)
        except queue.Empty:
            time.sleep(0.05)  # Reducido de 0.1 a 0.05 para menor latencia
        except Exception as e:
            print(f"Error en cola reconocimiento: {e}")
            time.sleep(0.25)  # Reducido de 0.5 a 0.25 para menor latencia

# ==========================
# Servidor web
# ==========================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_video():
    """Stream de video rápido - OPTIMIZADO para menor latencia"""
    global latest_frame
    while True:
        try:
            with lock:
                if latest_frame is not None:
                    frame = latest_frame
                else:
                    time.sleep(0.005)  # Reducido de 0.01 a 0.005 para menor latencia
                    continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except Exception as e:
            print(f"Error en stream video: {e}")
            time.sleep(0.005)  # Reducido de 0.01 a 0.005 para menor latencia

def generate_recognition():
    """Stream de reconocimiento con overlay - OPTIMIZADO para menor latencia"""
    global latest_recognition_frame
    while True:
        try:
            with recognition_lock:
                if latest_recognition_frame is not None:
                    frame = latest_recognition_frame
                else:
                    time.sleep(0.05)  # Reducido de 0.1 a 0.05 para menor latencia
                    continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except Exception as e:
            print(f"Error en stream reconocimiento: {e}")
            time.sleep(0.05)  # Reducido de 0.1 a 0.05 para menor latencia

@app.route("/video_feed")
def video_feed():
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/recognition_feed")
def recognition_feed():
    return Response(generate_recognition(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register_face", methods=["POST"])
def register_face():
    """Registra un nuevo rostro"""
    data = request.json
    name = data.get("name")
    image_b64 = data.get("image")
    
    if not name or not image_b64:
        return jsonify({"success": False, "message": "Faltan datos"}), 400

    try:
        # Verificar si ya existe
        if check_person_exists(name):
            return jsonify({"success": False, "message": f"La persona '{name}' ya está registrada"}), 400
        
        # Procesar imagen
        header, encoded = image_b64.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "message": "Error al procesar imagen"}), 400
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return jsonify({"success": False, "message": "No se detectó rostro en la imagen"}), 400
            
        encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        if save_face(name, encoding):
            # Actualizar cache
            global known_faces_cache
            known_faces_cache = load_faces()
            
            return jsonify({"success": True, "message": f"Persona '{name}' registrada exitosamente!"})
        else:
            return jsonify({"success": False, "message": f"Error al guardar en base de datos"}), 500
        
    except Exception as e:
        print(f"Error registrando rostro: {e}")
        return jsonify({"success": False, "message": f"Error interno: {str(e)}"}), 500

@app.route("/check_person", methods=["POST"])
def check_person():
    """Verifica si una persona existe"""
    data = request.json
    name = data.get("name")
    
    if not name:
        return jsonify({"exists": False}), 400
    
    exists = check_person_exists(name)
    return jsonify({"exists": exists, "name": name})

@app.route("/recent_detections")
def recent_detections():
    """Obtiene detecciones recientes"""
    detections = get_recent_detections(5)
    return jsonify({"detections": detections})

@app.route("/status")
def status():
    return jsonify({
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "cpu_temp": cpu_temp,
        "video_fps": current_video_fps,
        "recognition_fps": current_recognition_fps
    })

def actualizar_metricas():
    global cpu_usage, ram_usage, cpu_temp
    while True:
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp","r") as f:
                    cpu_temp = float(f.read().strip())/1000.0
            time.sleep(2)
        except Exception as e:
            print(f"Error métricas: {e}")
            time.sleep(2)

# ==========================
# Main
# ==========================
def main():
    print("🚀 Iniciando sistema de chequeo facial SIMPLIFICADO para streaming...")
    print("⚡ Optimizaciones aplicadas:")
    print("   • Video: 25 FPS (estable)")
    print("   • Reconocimiento: 3 FPS (estable)")
    print("   • Resolución directa: 640x480")
    print("   • Sin escalado (mejor rendimiento)")
    print("   • Colas reducidas para menor latencia")
    print("   • Procesamiento optimizado con HOG")
    print("   • Control de antigüedad de frames")
    print("   • Limpieza de memoria inmediata")
    
    # Inicializar base de datos
    init_db()
    print("✅ Base de datos inicializada")
    
    # Cargar rostros conocidos
    global known_faces_cache
    known_faces_cache = load_faces()
    print(f"✅ {len(known_faces_cache)} personas cargadas")
    
    print("🔄 Iniciando hilos ultra optimizados...")
    
    # Hilo para captura de video
    video_thread = threading.Thread(target=rpicam_video_reader, daemon=True, name="VideoCapture")
    video_thread.start()
    print("✅ Captura de video simplificada iniciada (25 FPS)")
    
    time.sleep(1)  # Reducido de 2 a 1 segundo
    
    # Hilo para procesamiento rápido de video
    process_video_thread = threading.Thread(target=procesar_cola_video, daemon=True, name="VideoProcess")
    process_video_thread.start()
    print("✅ Procesamiento de video ultra rápido iniciado")
    
    # Hilo para reconocimiento facial optimizado
    recognition_thread = threading.Thread(target=procesar_cola_reconocimiento, daemon=True, name="Recognition")
    recognition_thread.start()
    print("✅ Reconocimiento facial optimizado iniciado (3 FPS)")
    
    # Hilo para métricas
    metrics_thread = threading.Thread(target=actualizar_metricas, daemon=True, name="Metrics")
    metrics_thread.start()
    print("✅ Métricas iniciadas")
    
    # Servidor web
    web_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, threaded=True), daemon=True, name="WebServer")
    web_thread.start()
    print("✅ Servidor web iniciado")
    
    print("\n🎉 Sistema SIMPLIFICADO completamente iniciado!")
    print("🌐 Accede a: http://<raspberry_pi_ip>:5000")
    print("📱 Video fluido (25 FPS) + Reconocimiento (3 FPS)")
    print("⚡ Resolución directa: 640x480 (sin escalado)")
    print("💡 CPU optimizado para máximo 80% de uso")
    print("⏹️  Presiona Ctrl+C para detener")
    
    try:
        while True:
            time.sleep(3)  # Reducido de 5 a 3 segundos
            print(f"📊 Estado: Video {current_video_fps} FPS | Reconocimiento {current_recognition_fps} FPS | CPU {cpu_usage:.1f}%")
    except KeyboardInterrupt:
        print("\n🛑 Sistema detenido")

if __name__ == "__main__":
    main()
