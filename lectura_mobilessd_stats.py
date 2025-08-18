#!/usr/bin/env python3
import subprocess
import re
import time
import sys
from collections import deque
from datetime import datetime

# Regex para el formato real del log:
# [0] : person[0] (0.73) @ 275,447 1478x1072
LINE_RE = re.compile(
    r"\[(\d+)\]\s*:\s*(\w+)\[\d+\]\s*\(([\d.]+)\)\s*@\s*(\d+),(\d+)\s+(\d+)x(\d+)"
)

class DetectionStats:
    def __init__(self, window_size=60):  # 60 segundos de ventana
        self.detections = deque(maxlen=window_size)
        self.start_time = time.time()
        self.total_detections = 0
        self.highest_conf = 0.0
        self.lowest_conf = 1.0
        
    def add_detection(self, confidence):
        now = time.time()
        self.detections.append(now)
        self.total_detections += 1
        self.highest_conf = max(self.highest_conf, confidence)
        self.lowest_conf = min(self.lowest_conf, confidence)
        
    def get_stats(self):
        now = time.time()
        recent_detections = sum(1 for t in self.detections if now - t <= 60)
        
        return {
            'detections_per_minute': recent_detections,
            'total_detections': self.total_detections,
            'highest_conf': self.highest_conf,
            'lowest_conf': self.lowest_conf,
            'uptime': now - self.start_time
        }

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

def print_stats(stats):
    """Imprime estadísticas formateadas"""
    print(f"\n📊 ESTADÍSTICAS DE DETECCIÓN:")
    print(f"   ⏱️  Tiempo activo: {stats['uptime']:.1f}s")
    print(f"   🎯 Detecciones/min: {stats['detections_per_minute']}")
    print(f"   📈 Total detecciones: {stats['total_detections']}")
    print(f"   🔝 Confianza máxima: {stats['highest_conf']:.2f}")
    print(f"   🔻 Confianza mínima: {stats['lowest_conf']:.2f}")
    print("-" * 50)

def main():
    proc = launch()
    stats = DetectionStats()
    last_stats_time = time.time()
    
    print("⏳ Esperando detecciones del IMX500 (MobileNet-SSD)...", file=sys.stderr)
    print("🔄 Cargando firmware de red en el IMX500 (puede tomar varios minutos)...", file=sys.stderr)
    print("📊 Las estadísticas se mostrarán cada 10 segundos", file=sys.stderr)

    try:
        while True:
            # Leemos de stderr primero ya que ahí es donde aparece la información de detección
            line = proc.stderr.readline()
            if not line:
                # Si stderr no tiene datos, intentamos con stdout
                line = proc.stdout.readline()
                if not line:
                    continue
            
            line = line.strip()
            
            m = LINE_RE.search(line)
            if not m:
                continue

            idx = int(m.group(1))
            label = m.group(2).lower()
            conf = float(m.group(3))
            x = int(m.group(4))
            y = int(m.group(5))
            w = int(m.group(6))
            h = int(m.group(7))
            x1, y1 = x + w, y + h

            if label == "person" and conf >= 0.5:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] PERSONA conf={conf:.2f} bbox={x},{y}-{x1},{y1}")
                sys.stdout.flush()
                
                # Actualizar estadísticas
                stats.add_detection(conf)
                
                # Mostrar estadísticas cada 10 segundos
                if time.time() - last_stats_time >= 10:
                    print_stats(stats.get_stats())
                    last_stats_time = time.time()

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.", file=sys.stderr)
        print_stats(stats.get_stats())
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

if __name__ == "__main__":
    main() 