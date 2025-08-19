#!/usr/bin/env python3
"""
Sistema de reconocimiento facial con rpicam-vid y registro desde web
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

# ==========================
# Configuración
# ==========================
SENSOR_WIDTH = 2028
SENSOR_HEIGHT = 1520
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
SCALE_X = CANVAS_WIDTH / SENSOR_WIDTH
SCALE_Y = CANVAS_HEIGHT / SENSOR_HEIGHT

DB_PATH = "faces.db"
latest_frame = None
lock = threading.Lock()

# Colas para separar procesamiento de video y reconocimiento facial
video_queue = queue.Queue(maxsize=30)  # Aumentado para más frames
face_queue = queue.Queue(maxsize=10)   # Cola de frames para reconocimiento facial

cpu_usage = 0.0
ram_usage = 0.0
cpu_temp = 0.0

# Control de FPS optimizado
TARGET_FPS = 25  # Reducido para mejor rendimiento
FRAME_INTERVAL = 1.0 / TARGET_FPS

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Reconocimiento Facial</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Sistema de Reconocimiento Facial</h1>
        
        <div class="video-container">
            <h2>Video en Tiempo Real</h2>
            <img src="{{ url_for('video_feed') }}" width="640" height="480" style="border: 3px solid #007bff; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"/>
        </div>
        
        <div class="form-container">
            <h2>📝 Registrar Rostro Desconocido</h2>
            <form id="registerForm">
                <input type="text" id="name" placeholder="Nombre de la persona" required>
                <button type="submit">Registrar Rostro</button>
            </form>
            <div id="message"></div>
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
        
        form.addEventListener('submit', async e => {
            e.preventDefault();
            const name = document.getElementById('name').value;
            const button = form.querySelector('button');
            const originalText = button.textContent;
            
            button.textContent = 'Procesando...';
            button.disabled = true;
            
            try {
                // Capturar frame actual del video
                const video_img = document.querySelector('img');
                const canvas = document.createElement('canvas');
                canvas.width = video_img.width;
                canvas.height = video_img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video_img, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/jpeg');
                
                const res = await fetch('/register_face', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({name:name, image:dataURL})
                });
                
                const result = await res.json();
                
                if (result.success) {
                    messageDiv.innerHTML = `<div class="message success">✅ ${result.message}</div>`;
                    form.reset();
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
        
        // Actualizar métricas del sistema
        async function updateStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                document.getElementById('cpu').textContent = data.cpu_usage.toFixed(1);
                document.getElementById('ram').textContent = data.ram_usage.toFixed(1);
                document.getElementById('temp').textContent = data.cpu_temp.toFixed(1);
            } catch (error) {
                console.error('Error actualizando estado:', error);
            }
        }
        
        // Actualizar cada 2 segundos
        setInterval(updateStatus, 2000);
        updateStatus(); // Actualización inicial
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
    c.execute(
        """CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding BLOB
        )"""
    )
    conn.commit()
    conn.close()

def save_face(name, encoding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, pickle.dumps(encoding)))
    conn.commit()
    conn.close()

def load_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM faces")
    results = [(row[0], pickle.loads(row[1])) for row in c.fetchall()]
    conn.close()
    return results

# ==========================
# Procesamiento de video optimizado
# ==========================
face_register_queue = []  # almacenar rostros para registrar desde web
known_faces_cache = []    # cache de rostros conocidos
last_face_update = 0      # timestamp de última actualización de rostros
FACE_UPDATE_INTERVAL = 5  # segundos entre actualizaciones de rostros

def procesar_video_rapido(frame_data):
    """Procesa frame JPEG solo para display (sin reconocimiento facial)"""
    global latest_frame
    if not isinstance(frame_data, bytes) or len(frame_data) < 1000:
        return

    try:
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Solo redimensionar para display
        display_frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
        
        # Convertir a JPEG para streaming con calidad optimizada
        _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        with lock:
            latest_frame = jpeg.tobytes()
            
    except Exception as e:
        print(f"Error procesando frame: {e}")

def procesar_reconocimiento_facial():
    """Procesa frames para reconocimiento facial en hilo separado"""
    global known_faces_cache, last_face_update
    
    while True:
        try:
            # Actualizar cache de rostros conocidos periódicamente
            current_time = time.time()
            if current_time - last_face_update > FACE_UPDATE_INTERVAL:
                known_faces_cache = load_faces()
                last_face_update = current_time
            
            # Procesar frame de la cola de reconocimiento facial
            try:
                frame_data = face_queue.get(timeout=2.0)
            except queue.Empty:
                continue
                
            if not isinstance(frame_data, bytes) or len(frame_data) < 1000:
                continue
                
            np_frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Procesar cada rostro detectado
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                name = "Desconocido"
                for db_name, db_enc in known_faces_cache:
                    if face_recognition.compare_faces([db_enc], encoding, tolerance=0.5)[0]:
                        name = db_name
                        break
                
                # Si desconocido, almacenar para registro web
                if name == "Desconocido":
                    with lock:
                        face_register_queue.append((encoding, (top, right, bottom, left)))
                        
        except Exception as e:
            print(f"Error en reconocimiento facial: {e}")
            time.sleep(0.1)

def rpicam_video_reader():
    """Captura frames JPEG desde rpicam-vid y los distribuye a las colas"""
    cmd = [
        "rpicam-vid",
        "-n", "-t", "0",
        "--codec", "mjpeg",
        "-o", "-",
        "--width", str(SENSOR_WIDTH),
        "--height", str(SENSOR_HEIGHT),
        "--framerate", "25"
    ]
    
    print(f"🎥 Iniciando captura con comando: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = b""
    frame_count = 0
    last_log_time = time.time()
    
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                print("❌ No se recibieron datos del proceso rpicam-vid")
                break
                
            buffer += chunk
            
            # Buscar frames JPEG completos
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
                
                # Log cada 30 frames (aproximadamente cada segundo)
                if frame_count % 30 == 0:
                    elapsed = current_time - last_log_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    print(f"📸 Frame {frame_count}: {len(frame_data)} bytes | FPS: {fps:.1f}")
                    last_log_time = current_time
                
                # Cada 4 frames, enviar a reconocimiento facial (para no saturar)
                if frame_count % 4 == 0:
                    try:
                        face_queue.put_nowait(frame_data)
                    except queue.Full:
                        pass  # Ignorar si la cola está llena
                
                # Todos los frames van a procesamiento rápido
                try:
                    video_queue.put_nowait(frame_data)
                except queue.Full:
                    # Si la cola está llena, limpiar frames antiguos
                    try:
                        video_queue.get_nowait()
                        video_queue.put_nowait(frame_data)
                    except:
                        pass
                    
    except Exception as e:
        print(f"Error en captura de video: {e}")
    finally:
        print("🛑 Terminando captura de video...")
        proc.terminate()
        proc.wait()

def procesar_cola_video():
    """Procesa frames de la cola de video para streaming"""
    print("🔄 Iniciando procesamiento de cola de video...")
    frame_count = 0
    last_time = time.time()
    
    while True:
        try:
            frame_data = video_queue.get(timeout=0.1)
            procesar_video_rapido(frame_data)
            frame_count += 1
            
            # Control de FPS
            current_time = time.time()
            if current_time - last_time >= FRAME_INTERVAL:
                last_time = current_time
            else:
                time.sleep(FRAME_INTERVAL - (current_time - last_time))
                
        except queue.Empty:
            time.sleep(0.01)
        except Exception as e:
            print(f"Error procesando cola de video: {e}")
            time.sleep(0.1)

# ==========================
# Servidor web
# ==========================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_video():
    """Genera stream de video MJPEG optimizado"""
    global latest_frame
    frame_count = 0
    last_frame_time = time.time()
    
    while True:
        current_time = time.time()
        
        # Control de FPS para el stream
        if current_time - last_frame_time >= FRAME_INTERVAL:
            with lock:
                if latest_frame is not None:
                    frame = latest_frame
                    frame_count += 1
                    last_frame_time = current_time
                else:
                    time.sleep(0.01)
                    continue
        else:
            time.sleep(0.001)  # Sleep muy corto para mantener responsividad
            continue
            
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register_face", methods=["POST"])
def register_face():
    """Recibe nombre + imagen para registrar rostro"""
    data = request.json
    name = data.get("name")
    image_b64 = data.get("image")
    if not name or not image_b64:
        return jsonify({"success": False, "message": "Faltan datos"}), 400

    try:
        # Convertir base64 a numpy array
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
        save_face(name, encoding)
        
        # Actualizar cache
        global known_faces_cache, last_face_update
        known_faces_cache = load_faces()
        last_face_update = time.time()
        
        return jsonify({"success": True, "message": f"Rostro de '{name}' registrado exitosamente!"})
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Error al registrar rostro: {str(e)}"}), 500

@app.route("/status")
def status():
    return jsonify({
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "cpu_temp": cpu_temp
    })

def actualizar_metricas():
    global cpu_usage, ram_usage, cpu_temp
    while True:
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp","r") as f:
                    cpu_temp = float(f.read().strip())/1000.0
            time.sleep(2)
        except Exception as e:
            print(f"Error actualizando métricas: {e}")
            time.sleep(2)

# ==========================
# Main
# ==========================
def main():
    print("🚀 Iniciando sistema de reconocimiento facial optimizado...")
    
    # Inicializar base de datos
    init_db()
    print("✅ Base de datos inicializada")
    
    # Cargar rostros conocidos inicialmente
    global known_faces_cache, last_face_update
    known_faces_cache = load_faces()
    last_face_update = time.time()
    print(f"✅ {len(known_faces_cache)} rostros cargados de la base de datos")
    
    # Iniciar hilos
    print("🔄 Iniciando hilos de procesamiento...")
    
    # Hilo para captura de video
    video_thread = threading.Thread(target=rpicam_video_reader, daemon=True, name="VideoCapture")
    video_thread.start()
    print("✅ Hilo de captura de video iniciado")
    
    # Esperar un momento para que la cámara se inicialice
    time.sleep(2)
    
    # Hilo para procesamiento rápido de video
    process_thread = threading.Thread(target=procesar_cola_video, daemon=True, name="VideoProcess")
    process_thread.start()
    print("✅ Hilo de procesamiento de video iniciado")
    
    # Hilo para reconocimiento facial
    face_thread = threading.Thread(target=procesar_reconocimiento_facial, daemon=True, name="FaceRecognition")
    face_thread.start()
    print("✅ Hilo de reconocimiento facial iniciado")
    
    # Hilo para métricas del sistema
    metrics_thread = threading.Thread(target=actualizar_metricas, daemon=True, name="Metrics")
    metrics_thread.start()
    print("✅ Hilo de métricas iniciado")
    
    # Hilo para servidor web
    web_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, threaded=True), daemon=True, name="WebServer")
    web_thread.start()
    print("✅ Servidor web iniciado")
    
    print("\n🎉 Sistema completamente iniciado!")
    print("🌐 Accede a: http://<raspberry_pi_ip>:5000")
    print("📱 El video debería mostrarse en tiempo real ahora")
    print("⏹️  Presiona Ctrl+C para detener")
    
    try:
        while True:
            time.sleep(1)
            # Verificar que los hilos principales estén vivos
            if not video_thread.is_alive():
                print("❌ Hilo de captura de video se detuvo")
                break
            if not process_thread.is_alive():
                print("❌ Hilo de procesamiento de video se detuvo")
                break
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.")
        print("🔄 Cerrando sistema...")

if __name__ == "__main__":
    main()
