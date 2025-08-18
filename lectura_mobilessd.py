#!/usr/bin/env python3
import subprocess
import re
import time
import sys
import math

# Regex para el formato real del log:
# [0] : person[0] (0.73) @ 275,447 1478x1072
LINE_RE = re.compile(
    r"\[(\d+)\]\s*:\s*(\w+)\[\d+\]\s*\(([\d.]+)\)\s*@\s*(\d+),(\d+)\s+(\d+)x(\d+)"
)

# Configuración de la línea de cruce
FRAME_WIDTH = 2028  # ajusta según tu cámara (ver salida "Viewfinder size chosen")
LINE_X = FRAME_WIDTH // 2  # línea vertical en el centro
MAX_DIST = 100  # distancia máxima para considerar que es la misma persona

# Variables globales
tracks = {}            # id -> {cx, cy, last_side, last_seen}
next_id = 0
total_entradas = 0
total_salidas = 0

def launch():
    cmd = [
        "rpicam-hello",
        "-n", "-t", "0", "-v", "2",
        "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
        "--lores-width", "640", "--lores-height", "480"
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

def asignar_id(cx, cy):
    global next_id
    for tid, data in tracks.items():
        dist = math.hypot(cx - data["cx"], cy - data["cy"])
        if dist < MAX_DIST:
            # Actualizamos el track existente
            tracks[tid]["cx"] = cx
            tracks[tid]["cy"] = cy
            tracks[tid]["last_seen"] = time.time()
            return tid
    # Si no coincidió con ninguno, creamos nuevo
    tid = next_id
    next_id += 1
    tracks[tid] = {"cx": cx, "cy": cy, "last_seen": time.time(), "last_side": "?"}
    return tid

def limpiar_tracks():
    # Eliminamos tracks viejos que no se actualizan hace >2s
    ahora = time.time()
    eliminar = [tid for tid, data in tracks.items() if ahora - data["last_seen"] > 2]
    for tid in eliminar:
        del tracks[tid]

def main():
    global total_entradas, total_salidas
    proc = launch()
    print("⏳ Esperando detecciones del IMX500 (MobileNet-SSD)...", file=sys.stderr)
    print("🔄 Cargando firmware de red en el IMX500 (puede tomar varios minutos)...", file=sys.stderr)

    try:
        while True:
            line = proc.stderr.readline()
            if not line:
                line = proc.stdout.readline()
                if not line:
                    continue

            line = line.strip()
            m = LINE_RE.search(line)
            if not m:
                continue

            label = m.group(2).lower()
            conf = float(m.group(3))
            x = int(m.group(4))
            y = int(m.group(5))
            w = int(m.group(6))
            h = int(m.group(7))
            cx, cy = x + w // 2, y + h // 2

            if label == "person" and conf >= 0.5:
                tid = asignar_id(cx, cy)

                # Determinar de qué lado de la línea está
                side = "L" if cx < LINE_X else "R"
                last_side = tracks[tid].get("last_side", "?")

                if last_side != "?" and last_side != side:
                    if last_side == "L" and side == "R":
                        total_entradas += 1
                        print(f"➡️ ENTRADA detectada (ID={tid}) Total entradas={total_entradas}")
                    elif last_side == "R" and side == "L":
                        total_salidas += 1
                        print(f"⬅️ SALIDA detectada (ID={tid}) Total salidas={total_salidas}")

                tracks[tid]["last_side"] = side

                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] ID={tid} PERSONA conf={conf:.2f} cx={cx} side={side}")

            limpiar_tracks()

            # Mostrar resumen cada frame
            activos = len(tracks)
            print(f"👥 Activos={activos} | Entradas={total_entradas} | Salidas={total_salidas}")

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.", file=sys.stderr)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

if __name__ == "__main__":
    main()
