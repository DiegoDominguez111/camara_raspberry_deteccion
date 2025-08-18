#!/usr/bin/env python3
import subprocess
import threading
import time
import sys
import re
from flask import Flask, jsonify, render_template_string
from datetime import datetime

# ==========================
# Configuración
# ==========================
LINE_X = 320  # línea virtual central (ajustable)
STALE_TRACK_SEC = 2.0
MAX_DIST_PX = 120

DET_RE = re.compile(
    r"\[(\d+)\]\s*:\s*(\w+)\[\d+\]\s*\(([\d.]+)\)\s*@\s*(\d+),(\d+)\s+(\d+)x(\d+)"
)

# ==========================
# Estado global
# ==========================
tracks = {}        # track_id -> {"cx", "cy", "last_side", "last_seen"}
next_id = 0
total_in = 0
total_out = 0
last_update = time.time()
lock = threading.Lock()

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

        # crear nuevo ID
        tid = next_id
        next_id += 1
        tracks[tid] = {"cx": cx, "cy": cy, "last_side": "?", "last_seen": now}
        return tid

def limpiar_tracks():
    now = time.time()
    with lock:
        to_del = [tid for tid, t in tracks.items() if now - t["last_seen"] > STALE_TRACK_SEC]
        for tid in to_del:
            del tracks[tid]

# ==========================
# Lectura rpicam-hello
# ==========================
def rpicam_hello_reader():
    global total_in, total_out, last_update
    cmd = [
        "rpicam-hello",
        "-n", "-t", "0", "-v", "2",
        "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
        "--lores-width", "640", "--lores-height", "480"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    print("⏳ Esperando detecciones de rpicam-hello...", file=sys.stderr)
    print("✅ Firmware cargado, comenzando detecciones...", file=sys.stderr)

    while True:
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.01)
            continue
        line = line.strip()
        m = DET_RE.search(line)
        if not m:
            continue
        label = m.group(2).lower()
        conf = float(m.group(3))
        x = int(m.group(4))
        y = int(m.group(5))
        w = int(m.group(6))
        h = int(m.group(7))
        if label != "person" or conf < 0.5:
            continue
        cx = x + w // 2
        cy = y + h // 2
        tid = asignar_id(cx, cy)
        side = "L" if cx < LINE_X else "R"
        
        with lock:
            last_side = tracks[tid]["last_side"]
            if last_side != "?" and last_side != side:
                if last_side == "L" and side == "R":
                    total_in += 1
                    print(f"🚶 ENTRADA detectada! Total entradas: {total_in}")
                elif last_side == "R" and side == "L":
                    total_out += 1
                    print(f"🚶 SALIDA detectada! Total salidas: {total_out}")
                last_update = time.time()
            
            tracks[tid]["last_side"] = side
            tracks[tid]["last_seen"] = time.time()
        
        limpiar_tracks()

        # DEBUG consola
        with lock:
            activos = len(tracks)
            print(f"[DEBUG] Persona ID={tid}, cx={cx}, cy={cy}, lado={side}, conf={conf:.2f}")
            print(f"👥 Activos={activos} | Entradas={total_in} | Salidas={total_out}")

# ==========================
# Servidor web
# ==========================
app = Flask(__name__)

# HTML con canvas para simular cámara y dibujar tracks
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Contador de Personas</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .stats { display: flex; justify-content: space-around; margin: 30px 0; }
        .stat-box { text-align: center; padding: 20px; border-radius: 8px; }
        .activos { background: #e3f2fd; color: #1976d2; }
        .entradas { background: #e8f5e8; color: #388e3c; }
        .salidas { background: #fff3e0; color: #f57c00; }
        .number { font-size: 2.5em; font-weight: bold; display: block; }
        .label { font-size: 1.2em; margin-top: 5px; }
        .last-update { text-align: center; color: #666; margin-top: 20px; }
        canvas { border: 2px solid #333; display: block; margin: 20px auto; }
        .auto-refresh { text-align: center; margin-top: 20px; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .refresh-btn { background: #2196f3; color: white; }
        .auto-btn { background: #4caf50; color: white; }
        .stop-btn { background: #f44336; color: white; }
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

        <canvas id="camCanvas" width="640" height="480"></canvas>

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

        function drawCanvas(tracksData) {
            const canvas = document.getElementById('camCanvas');
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Línea de cruce
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(320, 0); 
            ctx.lineTo(320, canvas.height);
            ctx.stroke();

            // Dibujar tracks activos
            ctx.fillStyle = 'lime';
            tracksData.forEach(track => {
                ctx.beginPath();
                ctx.arc(track.cx, track.cy, 8, 0, 2 * Math.PI);
                ctx.fill();
            });
        }

        function updateData() {
        fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('activos').textContent = data.activos;
            document.getElementById('entradas').textContent = data.entradas;
            document.getElementById('salidas').textContent = data.salidas;
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

            // dibujar directamente usando los datos recibidos
            drawCanvas(data.tracks_activos);
        })
        .catch(error => console.error('Error:', error));
}


        function refreshData() { updateData(); }
        function startAutoRefresh() {
            if (autoRefreshInterval) clearInterval(autoRefreshInterval);
            autoRefreshInterval = setInterval(updateData, 1000);
        }
        function stopAutoRefresh() {
            if (autoRefreshInterval) { clearInterval(autoRefreshInterval); autoRefreshInterval = null; }
        }

        updateData();
        startAutoRefresh();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/counts")
def counts():
    with lock:
        return jsonify({
            "activos": len(tracks),
            "entradas": total_in,
            "salidas": total_out,
            "timestamp": last_update
        })

# endpoint /status
@app.route("/status")
def status():
    with lock:
        tracks_pos = [{"cx": int(t["cx"]), "cy": int(t["cy"])} for t in tracks.values()]
        return jsonify({
            "activos": len(tracks),
            "entradas": total_in,
            "salidas": total_out,
            "tracks_activos": tracks_pos,
            "ultima_actualizacion": datetime.fromtimestamp(last_update).strftime("%Y-%m-%d %H:%M:%S")
        })

def start_web():
    print("🌐 Iniciando servidor web en http://0.0.0.0:5000", file=sys.stderr)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

# ==========================
# Main
# ==========================
def main():
    print("🚀 Iniciando sistema de detección de personas...", file=sys.stderr)
    
    # Hilo del rpicam-hello
    t_rpi = threading.Thread(target=rpicam_hello_reader, daemon=True)
    t_rpi.start()
    print("📹 Hilo de detección iniciado", file=sys.stderr)

    # Hilo servidor web
    t_web = threading.Thread(target=start_web, daemon=True)
    t_web.start()
    print("🌐 Hilo del servidor web iniciado", file=sys.stderr)

    # Mantener script principal vivo
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.", file=sys.stderr)

if __name__ == "__main__":
    main()
