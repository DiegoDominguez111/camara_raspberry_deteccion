#!/usr/bin/env python3
"""
Script de prueba inicial para la Raspberry Pi AI Camera
Usa rpicam-vid para acceder directamente a la cámara (evita conflictos con PipeWire)
"""

import subprocess
import threading
import time
import sys
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def test_rpicam_vid_basic():
    """
    Prueba básica usando rpicam-vid (método que funciona)
    """
    print("🔍 Iniciando prueba básica con rpicam-vid...")
    
    try:
        # Comando similar al que funciona en entradas_salidas_mobilessd.py
        cmd = [
            "rpicam-vid",
            "-n",  # sin preview
            "-t", "5000",  # 5 segundos
            "--codec", "mjpeg",
            "-o", "test_frame.jpg",  # guardar directamente
            "--width", "1920",
            "--height", "1080",
        ]
        
        print(f"📹 Ejecutando: {' '.join(cmd)}")
        
        # Ejecutar el comando
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ rpicam-vid ejecutado exitosamente")
            
            # Verificar que se creó el archivo
            if os.path.exists("test_frame.jpg"):
                file_size = os.path.getsize("test_frame.jpg")
                print(f"✅ Frame guardado: test_frame.jpg ({file_size} bytes)")
                
                # Cargar y mostrar información de la imagen
                try:
                    img = cv2.imread("test_frame.jpg")
                    if img is not None:
                        print(f"✅ Imagen cargada: {img.shape}")
                        return True
                    else:
                        print("❌ No se pudo cargar la imagen")
                        return False
                except Exception as e:
                    print(f"❌ Error cargando imagen: {e}")
                    return False
            else:
                print("❌ No se creó el archivo test_frame.jpg")
                return False
        else:
            print(f"❌ Error ejecutando rpicam-vid:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout ejecutando rpicam-vid")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def test_rpicam_vid_stream():
    """
    Prueba captura de stream usando rpicam-vid
    """
    print("\n📹 Probando captura de stream con rpicam-vid...")
    
    try:
        # Comando para capturar stream
        cmd = [
            "rpicam-vid",
            "-n",  # sin preview
            "-t", "3000",  # 3 segundos
            "--codec", "mjpeg",
            "-o", "-",  # salida a stdout
            "--width", "640",
            "--height", "480",
        ]
        
        print(f"📹 Ejecutando stream: {' '.join(cmd)}")
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frames_captured = 0
        buffer = b""
        start_time = time.time()
        
        print("📹 Capturando frames...")
        
        while time.time() - start_time < 3.0:  # 3 segundos máximo
            try:
                # Leer chunk de datos
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Buscar frames JPEG completos
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
                    if len(frame_data) > 1000:  # Verificar tamaño mínimo
                        frames_captured += 1
                        print(f"   Frame {frames_captured} capturado: {len(frame_data)} bytes")
                        
                        # Guardar el primer frame como prueba
                        if frames_captured == 1:
                            with open("test_stream_frame.jpg", "wb") as f:
                                f.write(frame_data)
                            print("   💾 Primer frame guardado como test_stream_frame.jpg")
                    
                    # Remover frame procesado del buffer
                    buffer = buffer[end_pos + 2 :]
                    
            except Exception as e:
                print(f"❌ Error en captura: {e}")
                break
        
        # Terminar proceso
        proc.terminate()
        proc.wait()
        
        if frames_captured > 0:
            print(f"✅ Captura exitosa: {frames_captured} frames capturados")
            return True
        else:
            print("❌ No se capturaron frames")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de stream: {e}")
        return False


def test_face_detection_rpicam():
    """
    Prueba detección facial usando frames de rpicam-vid
    """
    print("\n👤 Probando detección facial con frames de rpicam-vid...")
    
    try:
        # Primero capturar un frame
        if not test_rpicam_vid_basic():
            print("❌ No se pudo capturar frame para detección facial")
            return False
        
        # Cargar el clasificador de Haar
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(haar_cascade_path):
            print("⚠️  Clasificador Haar no encontrado, descargando...")
            os.makedirs(os.path.dirname(haar_cascade_path), exist_ok=True)
            
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, haar_cascade_path)
            print("✅ Clasificador Haar descargado")
        
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        
        if face_cascade.empty():
            print("❌ Error: No se pudo cargar el clasificador Haar")
            return False
        
        # Cargar imagen capturada
        frame = cv2.imread("test_frame.jpg")
        if frame is None:
            print("❌ No se pudo cargar la imagen para detección facial")
            return False
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        print(f"✅ Detección facial completada")
        print(f"   - Rostros detectados: {len(faces)}")
        
        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Guardar imagen con detección
        output_path = "test_face_detection_rpicam.jpg"
        cv2.imwrite(output_path, frame)
        print(f"💾 Imagen con detección guardada como: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en detección facial: {e}")
        return False


def test_ai_camera_features():
    """
    Prueba características específicas de la AI Camera
    """
    print("\n🤖 Probando características de la AI Camera...")
    
    try:
        # Verificar si existe el archivo de configuración de la AI Camera
        ai_config_path = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
        
        if os.path.exists(ai_config_path):
            print(f"✅ Configuración AI Camera encontrada: {ai_config_path}")
            
            # Leer información del archivo de configuración
            with open(ai_config_path, 'r') as f:
                config_content = f.read()
                print(f"   - Tamaño del archivo: {len(config_content)} bytes")
                
                if "mobilenet" in config_content.lower():
                    print("   - Modelo: MobileNet SSD detectado")
                if "imx500" in config_content.lower():
                    print("   - Sensor: IMX500 detectado")
        else:
            print(f"⚠️  Configuración AI Camera no encontrada en: {ai_config_path}")
        
        # Probar diferentes resoluciones con rpicam-vid
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        
        for width, height in resolutions:
            try:
                cmd = [
                    "rpicam-vid",
                    "-n",
                    "-t", "1000",  # 1 segundo
                    "--codec", "mjpeg",
                    "-o", f"test_{width}x{height}.jpg",
                    "--width", str(width),
                    "--height", str(height),
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and os.path.exists(f"test_{width}x{height}.jpg"):
                    file_size = os.path.getsize(f"test_{width}x{height}.jpg")
                    print(f"✅ Resolución {width}x{height}: OK ({file_size} bytes)")
                else:
                    print(f"❌ Resolución {width}x{height}: Falló")
                    
            except Exception as e:
                print(f"❌ Resolución {width}x{height}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de características: {e}")
        return False


def main():
    """
    Función principal que ejecuta todas las pruebas
    """
    print("🚀 INICIANDO PRUEBAS DE LA RASPBERRY PI AI CAMERA")
    print("=" * 50)
    print("📹 Usando rpicam-vid para evitar conflictos con PipeWire")
    print("=" * 50)
    
    # Prueba 1: Captura básica con rpicam-vid
    if not test_rpicam_vid_basic():
        print("❌ FALLA EN PRUEBA BÁSICA - Revisar instalación de rpicam-vid")
        return False
    
    # Prueba 2: Captura de stream
    if not test_rpicam_vid_stream():
        print("⚠️  ADVERTENCIA: Captura de stream falló")
    
    # Prueba 3: Características de AI Camera
    if not test_ai_camera_features():
        print("⚠️  ADVERTENCIA: Algunas características de AI Camera no funcionan")
    
    # Prueba 4: Detección facial con frames de rpicam-vid
    if not test_face_detection_rpicam():
        print("⚠️  ADVERTENCIA: Detección facial falló")
    
    print("\n" + "=" * 50)
    print("✅ PRUEBAS COMPLETADAS")
    print("📁 Archivos generados:")
    print("   - test_frame.jpg: Frame de prueba básica")
    print("   - test_stream_frame.jpg: Frame del stream")
    print("   - test_face_detection_rpicam.jpg: Frame con detección facial")
    print("   - test_640x480.jpg, test_1280x720.jpg, test_1920x1080.jpg: Pruebas de resolución")
    print("\n🎯 La cámara está funcionando correctamente con rpicam-vid")
    print("🔄 Continuar con el siguiente paso del proyecto...")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1) 