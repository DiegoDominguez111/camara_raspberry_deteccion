#!/usr/bin/env python3
import subprocess
import threading
import time
import sys
import re
from flask import Flask, jsonify, Response, render_template_string
from datetime import datetime

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


def procesar_video(line):
    """Procesa línea de video; aquí se podrían extraer frames JPEG si se necesita"""
    global latest_frame
    # Por ahora solo placeholder, en el futuro podemos extraer frame
    latest_frame = line  # opcional: guardar referencia al frame o número


def rpicam_hello_reader():
    global latest_frame

    cmd = [
        "rpicam-vid",
        "-n",  # sin preview
        "-t",
        "0",  # tiempo infinito
        "--post-process-file",
        "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
        "-v",
        "2",  # verbose para inferencias
        f"--width={SENSOR_WIDTH}",
        f"--height={SENSOR_HEIGHT}",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    buffer_block = ""
    current_type = None

    print("⏳ Esperando detecciones y video de rpicam-vid...", file=sys.stderr)

    while True:
        line = proc.stderr.readline()
        if not line:
            continue
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
        elif "Number of objects detected:" in line or line.startswith("[0] : person"):
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


# ==========================
# Servidor web
# ==========================
app = Flask(__name__)

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Contador de Personas</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; justify-content: space-around; margin: 30px 0; }}
        .stat-box {{ text-align: center; padding: 20px; border-radius: 8px; }}
        .activos {{ background: #e3f2fd; color: #1976d2; }}
        .entradas {{ background: #e8f5e8; color: #388e3c; }}
        .salidas {{ background: #fff3e0; color: #f57c00; }}
        .number {{ font-size: 2.5em; font-weight: bold; display: block; }}
        .label {{ font-size: 1.2em; margin-top: 5px; }}
        .last-update {{ text-align: center; color: #666; margin-top: 20px; }}
        canvas {{ border: 2px solid #333; background: #000; display: block; margin: 20px auto; }}
        .video-feed {{ border: 2px solid #333; display: block; margin: 20px auto; }}
        .auto-refresh {{ text-align: center; margin-top: 20px; }}
        button {{ padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }}
        .refresh-btn {{ background: #2196f3; color: white; }}
        .auto-btn {{ background: #4caf50; color: white; }}
        .stop-btn {{ background: #f44336; color: white; }}
    </style>
</head>
<body>
<div class="container">
<h1>📊 Contador de Personas en Tiempo Real</h1>

<div class="stats">
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
</div>

<canvas id="camCanvas" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}"></canvas>

<h2>🎥 Video en tiempo real</h2>
<img id="videoFeed" class="video-feed" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" src="/video_feed" />

<div class="last-update">
    <p>Última actualización: <span id="last-update">-</span></p>
</div>

<div class="auto-refresh">
    <button class="refresh-btn" onclick="refreshData()">🔄 Actualizar Manual</button>
    <button class="auto-btn" onclick="startAutoRefresh()">▶️ Auto-refresh ON</button>
    <button class="stop-btn" onclick="stopAutoRefresh()">⏹️ Auto-refresh OFF</button>
</div>
</div>

<script>
let autoRefreshInterval;

function drawCanvas(tracksData) {{
    const canvas = document.getElementById('camCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo({CANVAS_WIDTH//2}, 0);
    ctx.lineTo({CANVAS_WIDTH//2}, canvas.height);
    ctx.stroke();

    ctx.fillStyle = 'lime';
    tracksData.forEach(track => {{
        ctx.beginPath();
        ctx.arc(track.cx, track.cy, 10, 0, 2 * Math.PI);
        ctx.fill();
    }});
}}

function updateData() {{
    fetch('/status')
    .then(response => response.json())
    .then(data => {{
        document.getElementById('activos').textContent = data.activos;
        document.getElementById('entradas').textContent = data.entradas;
        document.getElementById('salidas').textContent = data.salidas;
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

        const scaledTracks = data.tracks_activos.map(t => {{
            return {{
                cx: Math.round(t.cx * {SCALE_X}),
                cy: Math.round(t.cy * {SCALE_Y})
            }};
        }});
        drawCanvas(scaledTracks);
    }})
    .catch(error => console.error('Error:', error));
}}

function refreshData() {{ updateData(); }}
function startAutoRefresh() {{
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    autoRefreshInterval = setInterval(updateData, 1000);
}}
function stopAutoRefresh() {{
    if (autoRefreshInterval) {{ clearInterval(autoRefreshInterval); autoRefreshInterval = null; }}
}}

updateData();
startAutoRefresh();
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
                "ultima_actualizacion": datetime.fromtimestamp(last_update).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )


def generate_video():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with lock:
            frame = latest_frame
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def start_web():
    print("🌐 Iniciando servidor web en http://0.0.0.0:5000", file=sys.stderr)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.", file=sys.stderr)


if __name__ == "__main__":
    main()
