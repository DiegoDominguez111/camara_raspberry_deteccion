#!/usr/bin/env python3
"""
Script de debug para diagnosticar el streaming de la cámara
"""
import subprocess
import time
import sys

def debug_streaming():
    """Debug del streaming paso a paso"""
    print("🔍 Debug del streaming de la cámara...")
    
    # Comando exacto que usa nuestro sistema
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
    
    try:
        # Ejecutar proceso
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Proceso iniciado")
        print("⏳ Esperando datos...")
        
        # Leer datos por 5 segundos
        start_time = time.time()
        total_bytes = 0
        chunks_received = 0
        frames_detected = 0
        
        while time.time() - start_time < 5:  # 5 segundos
            try:
                # Leer con timeout
                chunk = proc.stdout.read(1024)
                if chunk:
                    total_bytes += len(chunk)
                    chunks_received += 1
                    
                    # Buscar frames JPEG
                    if b"\xff\xd8" in chunk:
                        frames_detected += 1
                        print(f"🎯 Frame {frames_detected} detectado en chunk {chunks_received}")
                    
                    # Mostrar progreso cada segundo
                    elapsed = time.time() - start_time
                    if int(elapsed) > int(elapsed - 0.1):
                        print(f"⏱️  {elapsed:.1f}s: {total_bytes} bytes, {chunks_received} chunks, {frames_detected} frames")
                else:
                    print("⚠️  No hay datos disponibles")
                    break
                    
            except Exception as e:
                print(f"❌ Error leyendo: {e}")
                break
        
        # Terminar proceso
        proc.terminate()
        proc.wait()
        
        print(f"\n📊 Resumen del streaming:")
        print(f"   • Tiempo total: {time.time() - start_time:.1f}s")
        print(f"   • Bytes totales: {total_bytes}")
        print(f"   • Chunks recibidos: {chunks_received}")
        print(f"   • Frames detectados: {frames_detected}")
        
        if frames_detected > 0:
            print("✅ Streaming funcionando correctamente")
        else:
            print("❌ No se detectaron frames")
            
    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")

def debug_with_verbose():
    """Debug con modo verbose"""
    print("\n🔧 Debug con modo verbose...")
    
    cmd = [
        "rpicam-vid",
        "-n", "-t", "3000",  # 3 segundos
        "--codec", "mjpeg",
        "-o", "-",
        "--width", "640",
        "--height", "480",
        "--framerate", "25",
        "-v"  # Modo verbose
    ]
    
    print(f"📹 Comando verbose: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        print(f"✅ Comando verbose completado")
        print(f"   • Código de salida: {result.returncode}")
        print(f"   • STDOUT (primeros 500 chars):")
        print(result.stdout[:500])
        print(f"   • STDERR (primeros 500 chars):")
        print(result.stderr[:500])
        
    except subprocess.TimeoutExpired:
        print("❌ Comando verbose expiró por timeout")
    except Exception as e:
        print(f"❌ Error en comando verbose: {e}")

def main():
    """Función principal"""
    print("🚀 Debug completo del streaming de la cámara")
    print("=" * 60)
    
    debug_streaming()
    debug_with_verbose()
    
    print("\n" + "=" * 60)
    print("🎯 Debug completado")
    print("\n💡 Análisis:")
    print("   • Si se detectan frames: El problema está en el procesamiento")
    print("   • Si no hay frames: El problema está en la captura")
    print("   • Si hay errores: Revisar configuración de la cámara")

if __name__ == "__main__":
    main() 