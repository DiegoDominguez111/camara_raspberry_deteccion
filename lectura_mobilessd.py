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
personas_habitacion = 0  # Contador de personas en habitación
last_update = time.time()
lock = threading.Lock()
latest_frame = None  # Para video feed

# Configuración de dirección del flujo
FLOW_DIRECTION_NORMAL = True  # True: L->R = Entrada, R->L = Salida | False: L->R = Salida, R->L = Entrada

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
    global total_in, total_out, personas_habitacion, last_update
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
    # La cámara está invertida horizontalmente, por eso invertimos la lógica
    side = "L" if cx < LINE_X_SENSOR else "R"

    with lock:
        # Obtener el valor actual de la dirección del flujo de manera thread-safe
        current_flow_direction = FLOW_DIRECTION_NORMAL
        
        last_side = tracks[tid]["last_side"]
        if last_side != "?" and last_side != side:
            # Determinar entrada/salida basado en la dirección del flujo
            if current_flow_direction:
                # Dirección normal: L->R = Entrada, R->L = Salida
                if last_side == "L" and side == "R":
                    total_in += 1
                    personas_habitacion += 1  # Incrementar personas en habitación
                    print(f"🟢 ENTRADA detectada (L->R) - Total: {total_in}, En habitación: {personas_habitacion}", file=sys.stderr)
                elif last_side == "R" and side == "L":
                    total_out += 1
                    personas_habitacion = max(0, personas_habitacion - 1)  # Decrementar, mínimo 0
                    print(f"🔴 SALIDA detectada (R->L) - Total: {total_out}, En habitación: {personas_habitacion}", file=sys.stderr)
            else:
                # Dirección invertida: L->R = Salida, R->L = Entrada
                if last_side == "L" and side == "R":
                    total_out += 1
                    personas_habitacion = max(0, personas_habitacion - 1)  # Decrementar, mínimo 0
                    print(f"🔴 SALIDA detectada (L->R) [INVERTIDA] - Total: {total_out}, En habitación: {personas_habitacion}", file=sys.stderr)
                elif last_side == "R" and side == "L":
                    total_in += 1
                    personas_habitacion += 1  # Incrementar personas en habitación
                    print(f"🟢 ENTRADA detectada (R->L) [INVERTIDA] - Total: {total_in}, En habitación: {personas_habitacion}", file=sys.stderr)
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
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container { 
            display: flex; 
            flex-direction: column;
            height: 100vh; 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }
        
        /* Barra superior */
        .topbar {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            color: white;
        }
        .topbar h2 {
            margin: 0;
            color: #ecf0f1;
            font-size: 1.3em;
        }
        .topbar-controls {
            display: flex;
            gap: 15px;
        }
        .topbar-btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .topbar-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            background: linear-gradient(135deg, #2980b9, #1f5f8b);
        }
        .topbar-btn.danger {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }
        .topbar-btn.danger:hover {
            background: linear-gradient(135deg, #c0392b, #a93226);
        }
        
        .main-content {
            display: flex;
            flex: 1;
        }

        /* Barra lateral izquierda */
        .sidebar {
            width: 280px;
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            padding: 25px;
            box-shadow: 4px 0 20px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            color: white;
        }
        .sidebar h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #ecf0f1;
            font-size: 1.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stat-box { 
            margin-bottom: 25px; 
            padding: 20px; 
            border-radius: 15px; 
            text-align: center; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .stat-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .stat-box .number { 
            font-size: 2.2em; 
            font-weight: bold; 
            display: block; 
            margin-bottom: 5px;
        }
        .stat-box .label { 
            font-size: 0.9em; 
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .activos { background: linear-gradient(135deg, #3498db, #2980b9); color: white; }
        .habitacion { background: linear-gradient(135deg, #1abc9c, #16a085); color: white; }
        .entradas { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; }
        .salidas { background: linear-gradient(135deg, #e67e22, #d35400); color: white; }
        .cpu { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
        .ram { background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; }
        .temp { background: linear-gradient(135deg, #f1c40f, #f39c12); color: white; }
        
        /* Controles de flujo */
        .flow-control {
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            text-align: center;
        }
        .flow-info {
            margin-bottom: 15px;
            color: #ecf0f1;
            font-weight: 600;
            font-size: 1.1em;
        }
        .flow-toggle-btn {
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .flow-toggle-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            background: linear-gradient(135deg, #8e44ad, #7d3c98);
        }

        /* Contenedor derecho */
        .main {
            flex-grow: 1;
            padding: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            position: relative;
        }
        .main::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="%23ffffff" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="%23ffffff" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }
        .visualization-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .visualization-title {
            text-align: center;
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        canvas { 
            border: 3px solid #34495e; 
            background: #000; 
            margin-bottom: 20px; 
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .video-feed { 
            border: 3px solid #34495e; 
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
<div class="container">
    <div class="topbar">
        <h2>🎥 Sistema de Monitoreo de Personas</h2>
        <div class="topbar-controls">
            <button id="toggle-flow-btn" class="topbar-btn">
                🔄 Cambiar Dirección
            </button>
            <button id="reset-stats-btn" class="topbar-btn danger">
                🗑️ Reiniciar Estadísticas
            </button>
        </div>
    </div>
    
    <div class="main-content">
        <div class="sidebar">
            <h1>📊 Estadísticas</h1>
            <div class="stat-box activos">
                <span class="number" id="activos">0</span>
                <span class="label">Personas Activas</span>
            </div>
            <div class="stat-box habitacion">
                <span class="number" id="habitacion">0</span>
                <span class="label">En Habitación</span>
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
                <span class="number" id="cpu_temp">0%</span>
                <span class="label">CPU Temp</span>
            </div>
            
            <div class="flow-control">
                <div class="flow-info">
                    <span id="flow-direction-text">Dirección: Normal</span>
                </div>
            </div>
        </div>

    <div class="main">
        <div class="visualization-container">
            <div class="visualization-title">🎯 Mapa de Posiciones</div>
            <canvas id="camCanvas" width="640" height="480"></canvas>
        </div>
        
        <div class="visualization-container">
            <div class="visualization-title">🎥 Video en Tiempo Real</div>
            <img id="videoFeed" class="video-feed" width="640" height="480" src="/video_feed" />
        </div>
    </div>
    </div>
</div>

<script>
const videoFeed = document.getElementById('videoFeed');

function drawCanvas(tracksData) {
    const canvas = document.getElementById('camCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Línea central
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(canvas.width/2, 0);
    ctx.lineTo(canvas.width/2, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Indicadores de dirección del flujo
    const flowDirection = document.getElementById('flow-direction-text').textContent.includes('Normal');
    
    // Lado izquierdo (L)
    ctx.fillStyle = flowDirection ? '#2ecc71' : '#e67e22';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    if (flowDirection) {
        // Dirección normal: L->R = Entrada, R->L = Salida
        ctx.fillText('LADO L', canvas.width * 0.25, 30);
        ctx.fillText('LADO R', canvas.width * 0.75, 30);
        
        // Explicación de la lógica
        ctx.font = '12px Arial';
        ctx.fillText('L→R = ENTRADA', canvas.width * 0.25, 50);
        ctx.fillText('R→L = SALIDA', canvas.width * 0.75, 50);
    } else {
        // Dirección invertida: L->R = Salida, R->L = Entrada
        ctx.fillText('LADO L', canvas.width * 0.25, 30);
        ctx.fillText('LADO R', canvas.width * 0.75, 30);
        
        // Explicación de la lógica
        ctx.font = '12px Arial';
        ctx.fillText('L→R = SALIDA', canvas.width * 0.25, 50);
        ctx.fillText('R→L = ENTRADA', canvas.width * 0.75, 50);
    }
    
    // Flechas de dirección física (siempre muestran L→R y R→L)
    ctx.strokeStyle = '#34495e';
    ctx.lineWidth = 2;
    
    // Flecha L→R (izquierda a derecha)
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.25 + 20, 30);
    ctx.lineTo(canvas.width * 0.25 + 40, 30);
    ctx.lineTo(canvas.width * 0.25 + 35, 25);
    ctx.moveTo(canvas.width * 0.25 + 40, 30);
    ctx.lineTo(canvas.width * 0.25 + 35, 35);
    ctx.stroke();
    
    // Flecha R→L (derecha a izquierda)
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.75 - 20, 30);
    ctx.lineTo(canvas.width * 0.75 - 40, 30);
    ctx.lineTo(canvas.width * 0.75 - 35, 25);
    ctx.moveTo(canvas.width * 0.75 - 40, 30);
    ctx.lineTo(canvas.width * 0.75 - 35, 35);
    ctx.stroke();

    // Dibujar tracks con IDs
    tracksData.forEach((track, index) => {
        // Círculo de la persona
        ctx.fillStyle = '#3498db';
        ctx.strokeStyle = '#2980b9';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(track.cx, track.cy, 12, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // ID de la persona
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 1;
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        const idText = `ID:${index + 1}`;
        const textX = track.cx;
        const textY = track.cy - 25;
        
        // Fondo del texto
        const textMetrics = ctx.measureText(idText);
        const textWidth = textMetrics.width;
        const textHeight = 16;
        
        ctx.fillStyle = 'rgba(44, 62, 80, 0.9)';
        ctx.fillRect(textX - textWidth/2 - 4, textY - textHeight/2 - 2, textWidth + 8, textHeight + 4);
        
        // Texto del ID
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 1;
        ctx.fillText(idText, textX, textY);
        ctx.strokeText(idText, textX, textY);
    });
}

// Función que dibuja el canvas sincronizado con cada frame
async function updateCanvasWithFrame() {
    try {
        const response = await fetch('/status');
        const data = await response.json();

        // Actualizar métricas con animación
        updateMetricWithAnimation('activos', data.activos);
        updateMetricWithAnimation('habitacion', data.habitacion);
        updateMetricWithAnimation('entradas', data.entradas);
        updateMetricWithAnimation('salidas', data.salidas);
        updateMetricWithAnimation('cpu_usage', data.cpu_usage.toFixed(1) + '%');
        updateMetricWithAnimation('ram_usage', data.ram_usage.toFixed(1) + '%');
        updateMetricWithAnimation('cpu_temp', data.cpu_temp.toFixed(1) + '°C');

        // Actualizar indicador de dirección del flujo
        if (data.flow_direction) {
            const directionText = data.flow_direction === 'normal' ? 'Dirección: Normal' : 'Dirección: Invertida';
            document.getElementById('flow-direction-text').textContent = directionText;
        }

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

// Función para actualizar métricas con animación
function updateMetricWithAnimation(elementId, newValue) {
    const element = document.getElementById(elementId);
    if (element && element.textContent !== newValue) {
        element.style.transform = 'scale(1.1)';
        element.style.transition = 'transform 0.2s ease';
        element.textContent = newValue;
        
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 200);
    }
}

// Función para cambiar la dirección del flujo
async function toggleFlowDirection() {
    try {
        const response = await fetch('/toggle_flow_direction');
        const data = await response.json();
        
        if (data.success) {
            // Actualizar el texto del botón temporalmente
            const btn = document.getElementById('toggle-flow-btn');
            const originalText = btn.textContent;
            btn.textContent = '✅ Cambiado!';
            btn.style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
            
            setTimeout(() => {
                btn.textContent = originalText;
                btn.style.background = 'linear-gradient(135deg, #3498db, #2980b9)';
            }, 2000);
            
            console.log('Dirección del flujo cambiada a:', data.direction);
        }
    } catch (e) {
        console.error('Error cambiando dirección del flujo:', e);
    }
}

// Función para reiniciar estadísticas
async function resetStats() {
    if (confirm('¿Estás seguro de que quieres reiniciar todas las estadísticas?')) {
        try {
            const response = await fetch('/reset_stats');
            const data = await response.json();
            
            if (data.success) {
                // Actualizar el texto del botón temporalmente
                const btn = document.getElementById('reset-stats-btn');
                const originalText = btn.textContent;
                btn.textContent = '✅ Reiniciado!';
                btn.style.background = 'linear-gradient(135deg, #2ecc71, #27ae60)';
                
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
                }, 2000);
                
                console.log('Estadísticas reiniciadas:', data.message);
                
                // Actualizar inmediatamente las métricas en pantalla
                document.getElementById('activos').textContent = '0';
                document.getElementById('habitacion').textContent = '0';
                document.getElementById('entradas').textContent = '0';
                document.getElementById('salidas').textContent = '0';
            }
        } catch (e) {
            console.error('Error reiniciando estadísticas:', e);
        }
    }
}

// Agregar eventos a los botones
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('toggle-flow-btn');
    const resetBtn = document.getElementById('reset-stats-btn');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleFlowDirection);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetStats);
    }
});

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
                "habitacion": personas_habitacion,
                "entradas": total_in,
                "salidas": total_out,
                "tracks_activos": tracks_pos,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "cpu_temp": cpu_temp,
                "flow_direction": "normal" if FLOW_DIRECTION_NORMAL else "inverted",
                "ultima_actualizacion": datetime.fromtimestamp(last_update).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )


@app.route("/toggle_flow_direction")
def toggle_flow_direction():
    global FLOW_DIRECTION_NORMAL
    old_direction = "normal" if FLOW_DIRECTION_NORMAL else "invertida"
    FLOW_DIRECTION_NORMAL = not FLOW_DIRECTION_NORMAL
    new_direction = "normal" if FLOW_DIRECTION_NORMAL else "invertida"
    
    print(f"🔄 Dirección del flujo cambiada de '{old_direction}' a '{new_direction}'", file=sys.stderr)
    print(f"🔄 Valor de FLOW_DIRECTION_NORMAL: {FLOW_DIRECTION_NORMAL}", file=sys.stderr)
    
    return jsonify({"success": True, "direction": new_direction})


@app.route("/reset_stats")
def reset_stats():
    global total_in, total_out, personas_habitacion, tracks, next_id
    with lock:
        total_in = 0
        total_out = 0
        personas_habitacion = 0
        tracks = {}
        next_id = 0
        print("🔄 Estadísticas reiniciadas", file=sys.stderr)
    
    return jsonify({"success": True, "message": "Estadísticas reiniciadas"})


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
