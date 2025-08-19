#!/usr/bin/env python3
"""
Script simple para probar la cámara y diagnosticar problemas
"""
import subprocess
import time
import sys

def test_camera_simple():
    """Prueba simple de la cámara"""
    print("🧪 Probando cámara de manera simple...")
    
    # Comando simple para capturar un frame
    cmd = [
        "rpicam-vid",
        "-n", "-t", "2000",  # 2 segundos
        "--codec", "mjpeg",
        "-o", "test_frame.jpg",
        "--width", "640",
        "--height", "480"
    ]
    
    print(f"📷 Comando: {' '.join(cmd)}")
    
    try:
        # Ejecutar comando
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        print(f"✅ Comando ejecutado")
        print(f"   • Código de salida: {result.returncode}")
        print(f"   • STDOUT: {result.stdout[:200]}...")
        print(f"   • STDERR: {result.stderr[:200]}...")
        
        # Verificar si se creó la imagen
        import os
        if os.path.exists("test_frame.jpg"):
            size = os.path.getsize("test_frame.jpg")
            print(f"✅ Imagen creada: test_frame.jpg ({size} bytes)")
        else:
            print("❌ No se creó la imagen")
            
    except subprocess.TimeoutExpired:
        print("❌ Comando expiró por timeout")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_camera_stream():
    """Prueba streaming de la cámara"""
    print("\n🎥 Probando streaming de la cámara...")
    
    cmd = [
        "rpicam-vid",
        "-n", "-t", "3000",  # 3 segundos
        "--codec", "mjpeg",
        "-o", "-",  # Salida a stdout
        "--width", "640",
        "--height", "480"
    ]
    
    print(f"📹 Comando streaming: {' '.join(cmd)}")
    
    try:
        # Ejecutar comando con timeout
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        start_time = time.time()
        data_received = 0
        frames_received = 0
        
        while time.time() - start_time < 3:  # 3 segundos máximo
            try:
                chunk = proc.stdout.read(1024)
                if chunk:
                    data_received += len(chunk)
                    # Buscar frames JPEG
                    if b"\xff\xd8" in chunk:
                        frames_received += 1
                else:
                    break
            except:
                break
        
        proc.terminate()
        proc.wait()
        
        print(f"✅ Streaming completado")
        print(f"   • Datos recibidos: {data_received} bytes")
        print(f"   • Frames detectados: {frames_received}")
        
    except Exception as e:
        print(f"❌ Error en streaming: {e}")

def test_camera_parameters():
    """Prueba diferentes parámetros de la cámara"""
    print("\n🔧 Probando diferentes parámetros...")
    
    # Listar cámaras disponibles
    try:
        result = subprocess.run(["rpicam-vid", "--list-cameras"], 
                              capture_output=True, text=True, timeout=5)
        print("📋 Cámaras disponibles:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error listando cámaras: {e}")
    
    # Probar con parámetros más simples
    print("\n📷 Probando con parámetros mínimos...")
    cmd = [
        "rpicam-vid",
        "-n", "-t", "1000",
        "-o", "test_minimal.jpg"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"✅ Comando mínimo ejecutado")
        print(f"   • Código: {result.returncode}")
        print(f"   • Error: {result.stderr[:200]}...")
        
        import os
        if os.path.exists("test_minimal.jpg"):
            size = os.path.getsize("test_minimal.jpg")
            print(f"✅ Imagen mínima creada: {size} bytes")
        else:
            print("❌ No se creó imagen mínima")
            
    except Exception as e:
        print(f"❌ Error en comando mínimo: {e}")

def main():
    """Función principal"""
    print("🔍 Diagnóstico completo de la cámara")
    print("=" * 50)
    
    test_camera_simple()
    test_camera_stream()
    test_camera_parameters()
    
    print("\n" + "=" * 50)
    print("🎯 Diagnóstico completado")
    print("\n💡 Recomendaciones:")
    print("   1. Verifica que la cámara esté conectada")
    print("   2. Revisa los permisos de /dev/video*")
    print("   3. Prueba con parámetros más simples")
    print("   4. Verifica la versión de libraspberrypi-bin")

if __name__ == "__main__":
    main() 