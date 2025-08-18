#!/usr/bin/env python3
"""
Sistema de detección de personas con rpicam-vid
- Procesa frames JPEG desde stdout
- Procesa inferencias desde stderr
- Muestra video en tiempo real en el navegador
"""
import subprocess
import threading
import time
import sys
import re
from flask import Flask, jsonify, Response, render_template_string
from datetime import datetime
import psutil
import os

# ==========================
# Configuración
# ==========================
LINE_X_SENSOR = 2028 // 2
SENSOR_WIDTH = 2028
SENSOR_HEIGHT = 1520
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
SCALE_X = CANVAS_WIDTH / SENSOR_WIDTH
SCALE_Y = CANVAS_HEIGHT / SENSOR_HEIGHT

STALE_TRACK_SEC = 2.0
MAX_DIST_PX = 120

DET_RE = re.compile(
    r"\[(\d+)\]\s*:\s*(\w+)\[\d+\]\s*\(([\d.]+)\)\s*@\s*(\d+),(\d+)\s+(\d+)x(\d+)"
)

# ==========================
# Estado global
# ==========================
tracks = {}
next_id = 0
total_in = 0
total_out = 0
last_update = time.time()
lock = threading.Lock()
latest_frame = None  # Para video feed

cpu_usage = 0.0
ram_usage = 0.0
cpu_temp = 0.0


# ==========================
# Funciones de tracking
# ==========================
def asignar_id(cx, cy):
    global next_id
    best_tid = None
    best_dist = float("inf")
    now = time.time()
    with lock:
        for tid, t in tracks.items():
            dist = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
            if dist < best_dist and dist <= MAX_DIST_PX:
                best_dist = dist
                best_tid = tid
        if best_tid is not None:
            t = tracks[best_tid]
            t["cx"] = cx
            t["cy"] = cy
            t["last_seen"] = now
            return best_tid
        tid = next_id
        next_id += 1
        tracks[tid] = {"cx": cx, "cy": cy, "last_side": "?", "last_seen": now}
        return tid


def limpiar_tracks():
    now = time.time()
    with lock:
        to_del = [
            tid for tid, t in tracks.items() if now - t["last_seen"] > STALE_TRACK_SEC
        ]
        for tid in to_del:
            del tracks[tid]


def procesar_inferencia(line):
    """Procesa una línea de inferencia y actualiza los tracks y conteos"""
    global total_in, total_out, last_update
    m = DET_RE.search(line)
    if not m:
        return

    label = m.group(2).lower()
    conf = float(m.group(3))
    x = int(m.group(4))
    y = int(m.group(5))
    w = int(m.group(6))
    h = int(m.group(7))
    if label != "person" or conf < 0.5:
        return

    cx = x + w // 2
    cy = y + h // 2
    tid = asignar_id(cx, cy)
    side = "L" if cx < LINE_X_SENSOR else "R"

    with lock:
        last_side = tracks[tid]["last_side"]
        if last_side != "?" and last_side != side:
            if last_side == "L" and side == "R":
                total_in += 1
            elif last_side == "R" and side == "L":
                total_out += 1
            last_update = time.time()
        tracks[tid]["last_side"] = side
        tracks[tid]["last_seen"] = time.time()

    limpiar_tracks()


def procesar_video(frame_data):
    """Procesa frame JPEG de video"""
    global latest_frame

    # Verificar que sea un frame JPEG válido
    if (
        isinstance(frame_data, bytes)
        and len(frame_data) > 1000
        and frame_data.startswith(b"\xff\xd8")
        and frame_data.endswith(b"\xff\xd9")
    ):

        latest_frame = frame_data
        print(
            f"📹 Frame JPEG válido procesado: {len(frame_data)} bytes", file=sys.stderr
        )
    else:
        print(
            f"⚠️ Frame inválido ignorado: {type(frame_data)}, tamaño: {len(frame_data) if isinstance(frame_data, bytes) else 'N/A'}",
            file=sys.stderr,
        )


def rpicam_hello_reader():
    global latest_frame

    # Un solo comando que maneja tanto video como inferencias
    cmd = [
        "rpicam-vid",
        "-n",  # sin preview
        "-t",
        "0",  # tiempo infinito
        "--post-process-file",
        "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
        "-v",
        "2",  # verbose para inferencias
        "--codec",
        "mjpeg",
        "-o",
        "-",  # salida a stdout
        f"--width={SENSOR_WIDTH}",
        f"--height={SENSOR_HEIGHT}",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("⏳ Iniciando procesamiento de frames e inferencias...", file=sys.stderr)

    # Procesar frames JPEG desde stdout
    def process_frames():
        buffer = b""
        print("📹 Iniciando captura de frames desde stdout...", file=sys.stderr)

        while True:
            try:
                # Leer chunks de datos del stream
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break

                buffer += chunk

                # Buscar frames JPEG completos en el buffer
                while True:
                    # Buscar inicio de frame JPEG (0xFF 0xD8)
                    start_pos = buffer.find(b"\xff\xd8")
                    if start_pos == -1:
                        # No hay inicio de frame, mantener solo el último byte por si es 0xFF
                        if buffer and buffer[-1] == 0xFF:
                            buffer = buffer[-1:]
                        else:
                            buffer = b""
                        break

                    # Buscar fin de frame JPEG (0xFF 0xD9) después del inicio
                    end_pos = buffer.find(b"\xff\xd9", start_pos + 2)
                    if end_pos == -1:
                        # No hay fin de frame, mantener desde el inicio
                        buffer = buffer[start_pos:]
                        break

                    # Extraer frame completo
                    frame_data = buffer[start_pos : end_pos + 2]
                    if (
                        len(frame_data) > 1000
                    ):  # Verificar que el frame tenga un tamaño mínimo
                        procesar_video(frame_data)

                    # Remover frame procesado del buffer
                    buffer = buffer[end_pos + 2 :]

            except Exception as e:
                print(f"❌ Error en process_frames: {e}", file=sys.stderr)
                time.sleep(0.1)

    # Procesar inferencias desde stderr
    def process_inferences():
        buffer_block = ""
        current_type = None
        print(
            "🔍 Iniciando procesamiento de inferencias desde stderr...", file=sys.stderr
        )

        while True:
            try:
                line = proc.stderr.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()

                # Detectar bloque de video
                if line.startswith("Viewfinder frame"):
                    if buffer_block and current_type:
                        if current_type == "video":
                            procesar_video(buffer_block)
                        elif current_type == "inferencia":
                            procesar_inferencia(buffer_block)
                    buffer_block = line + "\n"
                    current_type = "video"

                # Detectar bloque de inferencia
                elif "Number of objects detected:" in line or line.startswith(
                    "[0] : person"
                ):
                    if buffer_block and current_type:
                        if current_type == "video":
                            procesar_video(buffer_block)
                        elif current_type == "inferencia":
                            procesar_inferencia(buffer_block)
                    buffer_block = line + "\n"
                    current_type = "inferencia"

                # Acumular líneas de contexto
                else:
                    buffer_block += line + "\n"

            except Exception as e:
                print(f"❌ Error en process_inferences: {e}", file=sys.stderr)
                time.sleep(0.1)

    # Ejecutar ambos procesadores en hilos separados
    import threading

    frame_thread = threading.Thread(target=process_frames, daemon=True)
    inference_thread = threading.Thread(target=process_inferences, daemon=True)

    frame_thread.start()
    inference_thread.start()

    # Esperar a que ambos hilos terminen
    frame_thread.join()
    inference_thread.join()


# ==========================
# Servidor web
# ==========================
app = Flask(__name__)

HTML_TEMPLATE="""
<!DOCTYPE html>
<html>
<head>
    <title>Contador de Personas</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #f0f0f0; }
        .container { display: flex; height: 100vh; }

        /* Barra lateral izquierda */
        .sidebar {
            width: 220px;
            background: #fff;
            padding: 20px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .stat-box { margin-bottom: 20px; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-box .number { font-size: 2em; font-weight: bold; display: block; }
        .activos { background: #e3f2fd; color: #1976d2; }
        .entradas { background: #e8f5e8; color: #388e3c; }
        .salidas { background: #fff3e0; color: #f57c00; }
        .cpu { background: #fce4ec; color: #d81b60; }
        .ram { background: #ede7f6; color: #5e35b1; }
        .temp { background: #fff9c4; color: #fbc02d; }

        /* Contenedor derecho */
        .main {
            flex-grow: 1;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        canvas { border: 2px solid #333; background: #000; margin-bottom: 20px; }
        .video-feed { border: 2px solid #333; }
    </style>
</head>
<body>
<div class="container">
    <div class="sidebar">
        <div class="stat-box activos">
            <span class="number" id="activos">0</span>
            <span class="label">Personas Activas</span>
        </div>
        <div class="stat-box entradas">
            <span class="number" id="entradas">0</span>
            <span class="label">Entradas</span>
        </div>
        <div class="stat-box salidas">
            <span class="number" id="salidas">0</span>
            <span class="label">Salidas</span>
        </div>
        <div class="stat-box cpu">
            <span class="number" id="cpu_usage">0%</span>
            <span class="label">CPU Uso</span>
        </div>
        <div class="stat-box ram">
            <span class="number" id="ram_usage">0%</span>
            <span class="label">RAM Uso</span>
        </div>
        <div class="stat-box temp">
            <span class="number" id="cpu_temp">0°C</span>
            <span class="label">CPU Temp</span>
        </div>
    </div>

    <div class="main">
        <canvas id="camCanvas" width="640" height="480"></canvas>
        <img id="videoFeed" class="video-feed" width="640" height="480" src="/video_feed" />
    </div>
</div>

<script>
const videoFeed = document.getElementById('videoFeed');

function drawCanvas(tracksData) {
    const canvas = document.getElementById('camCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Línea central
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(canvas.width/2, 0);
    ctx.lineTo(canvas.width/2, canvas.height);
    ctx.stroke();

    // Dibujar tracks
    ctx.fillStyle = 'lime';
    tracksData.forEach(track => {
        ctx.beginPath();
        ctx.arc(track.cx, track.cy, 10, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// Función que dibuja el canvas sincronizado con cada frame
async function updateCanvasWithFrame() {
    try {
        const response = await fetch('/status');
        const data = await response.json();

        // Actualizar métricas
        document.getElementById('activos').textContent = data.activos;
        document.getElementById('entradas').textContent = data.entradas;
        document.getElementById('salidas').textContent = data.salidas;
        document.getElementById('cpu_usage').textContent = data.cpu_usage.toFixed(1) + '%';
        document.getElementById('ram_usage').textContent = data.ram_usage.toFixed(1) + '%';
        document.getElementById('cpu_temp').textContent = data.cpu_temp.toFixed(1) + '°C';

        // Escalar tracks para el canvas
        const scaledTracks = data.tracks_activos.map(t => {
            return {
                cx: Math.round(t.cx * 640 / 2028),
                cy: Math.round(t.cy * 480 / 1520)
            };
        });

        drawCanvas(scaledTracks);
    } catch (e) {
        console.error('Error fetching status:', e);
    }

    // Solicitar el siguiente frame
    requestAnimationFrame(updateCanvasWithFrame);
}

// Iniciar bucle cuando el video se cargue
videoFeed.onload = () => {
    updateCanvasWithFrame();
};
</script>
</body>
</html>
"""



@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/status")
def status():
    with lock:
        tracks_pos = [{"cx": t["cx"], "cy": t["cy"]} for t in tracks.values()]
        return jsonify(
            {
                "activos": len(tracks),
                "entradas": total_in,
                "salidas": total_out,
                "tracks_activos": tracks_pos,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "cpu_temp": cpu_temp,
                "ultima_actualizacion": datetime.fromtimestamp(last_update).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )


def generate_video():
    global latest_frame
    while True:
        try:
            if latest_frame is None:
                time.sleep(0.01)
                continue

            with lock:
                frame = latest_frame

            # Verificar que frame sea bytes válidos
            if not isinstance(frame, bytes) or len(frame) < 100:
                time.sleep(0.01)
                continue

            # Verificar que sea un JPEG válido
            if not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
                time.sleep(0.01)
                continue

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)  # ~30 FPS

        except Exception as e:
            print(f"❌ Error en generate_video: {e}", file=sys.stderr)
            time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    try:
        return Response(
            generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        print(f"❌ Error en video_feed: {e}", file=sys.stderr)
        return "Error en video feed", 500


def start_web():
    print("🌐 Iniciando servidor web en http://0.0.0.0:5000", file=sys.stderr)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


def actualizar_metricas():
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
            print(f"⚠️ Error actualizando métricas: {e}", file=sys.stderr)
            time.sleep(1)


# ==========================
# Main
# ==========================
def main():
    print("🚀 Iniciando sistema de detección de personas...", file=sys.stderr)

    t_rpi = threading.Thread(target=rpicam_hello_reader, daemon=True)
    t_rpi.start()
    print("📹 Hilo de detección y video iniciado", file=sys.stderr)

    t_web = threading.Thread(target=start_web, daemon=True)
    t_web.start()
    print("🌐 Hilo del servidor web iniciado", file=sys.stderr)

    t_metrics = threading.Thread(target=actualizar_metricas, daemon=True)
    t_metrics.start()
    print("📊 Hilo de métricas iniciado", file=sys.stderr)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.", file=sys.stderr)


if __name__ == "__main__":
    main()
