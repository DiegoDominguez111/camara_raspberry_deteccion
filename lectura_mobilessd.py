#!/usr/bin/env python3
import subprocess
import re
import time
import sys

# Regex para el formato real del log:
# [0] : person[0] (0.73) @ 275,447 1478x1072
LINE_RE = re.compile(
    r"\[(\d+)\]\s*:\s*(\w+)\[\d+\]\s*\(([\d.]+)\)\s*@\s*(\d+),(\d+)\s+(\d+)x(\d+)"
)

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

def main():
    proc = launch()
    print("⏳ Esperando detecciones del IMX500 (MobileNet-SSD)...", file=sys.stderr)
    print("🔄 Cargando firmware de red en el IMX500 (puede tomar varios minutos)...", file=sys.stderr)

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

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario.", file=sys.stderr)
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
