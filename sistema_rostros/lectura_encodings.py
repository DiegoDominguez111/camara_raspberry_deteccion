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

cpu_usage = 0.0
ram_usage = 0.0
cpu_temp = 0.0

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Reconocimiento Facial</title>
</head>
<body>
    <h1>Video en Tiempo Real</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480"/>
    <h2>Registrar rostro desconocido</h2>
    <form id="registerForm">
        Nombre: <input type="text" id="name" required>
        <button type="submit">Registrar</button>
    </form>
    <script>
        const form = document.getElementById('registerForm');
        form.addEventListener('submit', async e => {
            e.preventDefault();
            const name = document.getElementById('name').value;
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
            alert(result.message);
        });
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
# Procesamiento de video
# ==========================
face_register_queue = []  # almacenar rostros para registrar desde web

def procesar_video(frame_data):
    """Procesa frame JPEG y detecta rostros"""
    global latest_frame
    if not isinstance(frame_data, bytes) or len(frame_data) < 1000:
        return

    np_frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    if frame is None:
        return

    display_frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    known_faces = load_faces()

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Desconocido"
        for db_name, db_enc in known_faces:
            if face_recognition.compare_faces([db_enc], encoding, tolerance=0.5)[0]:
                name = db_name
                break

        # Dibujar box + nombre
        cv2.rectangle(display_frame, (int(left*SCALE_X), int(top*SCALE_Y)),
                      (int(right*SCALE_X), int(bottom*SCALE_Y)), (0,255,0), 2)
        cv2.putText(display_frame, name, (int(left*SCALE_X), int(top*SCALE_Y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Si desconocido, almacenar para registro web
        if name == "Desconocido":
            with lock:
                face_register_queue.append((encoding, (top, right, bottom, left)))

    _, jpeg = cv2.imencode('.jpg', display_frame)
    with lock:
        latest_frame = jpeg.tobytes()

def rpicam_hello_reader():
    """Captura frames JPEG desde rpicam-vid"""
    cmd = [
        "rpicam-vid",
        "-n","-t","0",
        "--codec","mjpeg",
        "-o","-",
        f"--width={SENSOR_WIDTH}",
        f"--height={SENSOR_HEIGHT}"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
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
            procesar_video(frame_data)

# ==========================
# Servidor web
# ==========================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_video():
    global latest_frame
    while True:
        time.sleep(0.03)
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame
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

    # Convertir base64 a numpy array
    header, encoded = image_b64.split(",", 1)
    img_data = base64.b64decode(encoded)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return jsonify({"success": False, "message": "No se detectó rostro"}), 400
    encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    save_face(name, encoding)
    return jsonify({"success": True, "message": f"Rostro de '{name}' registrado!"})

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
            time.sleep(1)
        except:
            time.sleep(1)

# ==========================
# Main
# ==========================
def main():
    init_db()
    threading.Thread(target=rpicam_hello_reader, daemon=True).start()
    threading.Thread(target=actualizar_metricas, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, threaded=True), daemon=True).start()
    print("🚀 Sistema de reconocimiento facial iniciado. Accede a http://<raspberry_pi_ip>:5000")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.")

if __name__ == "__main__":
    main()
