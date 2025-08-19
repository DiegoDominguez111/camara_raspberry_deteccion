#!/usr/bin/env python3
"""
Script simple para probar el hilo de captura
"""
import subprocess
import threading
import time
import queue

def simple_capture():
    """Captura simple para probar"""
    print("🎥 Iniciando captura simple...")
    
    cmd = [
        "rpicam-vid",
        "-n", "-t", "0",
        "--codec", "mjpeg",
        "-o", "-",
        "--width", "640",
        "--height", "480",
        "--framerate", "25"
    ]
    
    print(f"📹 Comando: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = b""
    frame_count = 0
    
    try:
        while frame_count < 10:  # Solo 10 frames para prueba
            chunk = proc.stdout.read(1024)
            if not chunk:
                print("⚠️  No hay datos")
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
                print(f"✅ Frame {frame_count} capturado: {len(frame_data)} bytes")
                
                if frame_count >= 10:
                    break
                    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        proc.terminate()
        proc.wait()
        print("🛑 Captura terminada")

def main():
    """Función principal"""
    print("🧪 Probando captura simple...")
    
    # Ejecutar en hilo
    thread = threading.Thread(target=simple_capture, daemon=True)
    thread.start()
    
    # Esperar a que termine
    thread.join(timeout=15)
    
    if thread.is_alive():
        print("⚠️  Hilo aún ejecutándose después de 15 segundos")
    else:
        print("✅ Hilo completado correctamente")

if __name__ == "__main__":
    main() 