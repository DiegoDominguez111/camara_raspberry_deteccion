#!/usr/bin/env python3
"""
Prueba de detección facial usando OpenCV
Alternativa a picamera2 para verificar funcionalidad
"""

import cv2
import numpy as np
import time
import sys
import os

def test_camera_access():
    """Prueba acceso básico a la cámara"""
    print("🔍 Probando acceso a cámara...")
    
    # Probar diferentes índices
    for i in range(10):
        print(f"📷 Probando cámara {i}...")
        
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Configurar resolución
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Leer frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                print(f"✅ Cámara {i} funciona: {frame.shape}")
                cap.release()
                return i, cap
            else:
                print(f"   - Error leyendo frame")
                cap.release()
        else:
            print(f"   - No se pudo abrir")
    
    return None, None

def download_haar_cascade():
    """Descarga el clasificador Haar si no existe"""
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(haar_path):
        print("⚠️  Clasificador Haar no encontrado, descargando...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, haar_path)
            print("✅ Clasificador Haar descargado")
        except Exception as e:
            print(f"❌ Error descargando: {e}")
            return False
    
    return True

def test_face_detection(camera_index):
    """Prueba detección facial"""
    print(f"\n👤 Probando detección facial con cámara {camera_index}...")
    
    # Descargar clasificador si es necesario
    if not download_haar_cascade():
        return False
    
    # Cargar clasificador
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ No se pudo cargar clasificador Haar")
        return False
    
    # Abrir cámara
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir cámara")
        return False
    
    print("✅ Cámara abierta, iniciando detección...")
    print("🎮 Controles:")
    print("   - Presiona 'q' para salir")
    print("   - Presiona 's' para guardar frame")
    
    frame_count = 0
    faces_detected = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("⚠️  Error leyendo frame")
                continue
            
            frame_count += 1
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Dibujar rectángulos alrededor de rostros
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                faces_detected += 1
            
            # Mostrar información en frame
            cv2.putText(frame, f"Frames: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Rostros: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Total detectados: {faces_detected}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow("Detección Facial - OpenCV", frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("⏹️  Detección detenida por usuario")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"face_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame guardado: {filename}")
            
            # Mostrar progreso cada 30 frames
            if frame_count % 30 == 0:
                print(f"📊 Procesados {frame_count} frames, {faces_detected} rostros detectados")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n⏹️  Detección interrumpida")
    except Exception as e:
        print(f"❌ Error durante detección: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   - Frames procesados: {frame_count}")
        print(f"   - Rostros detectados: {faces_detected}")
        if frame_count > 0:
            print(f"   - Tasa de detección: {faces_detected/frame_count:.2%}")
    
    return True

def main():
    """Función principal"""
    print("🚀 PRUEBA DE DETECCIÓN FACIAL CON OPENCV")
    print("=" * 45)
    
    # Probar acceso a cámara
    camera_index, cap = test_camera_access()
    
    if camera_index is None:
        print("❌ No se pudo acceder a ninguna cámara")
        print("🔧 Verificar:")
        print("   - Conexión de cámara")
        print("   - Permisos de usuario")
        print("   - Drivers instalados")
        return False
    
    print(f"✅ Cámara {camera_index} disponible")
    
    # Probar detección facial
    if test_face_detection(camera_index):
        print("\n🎯 Detección facial funcionando correctamente")
        print("🔄 Continuar con el proyecto...")
        return True
    else:
        print("\n❌ Error en detección facial")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Prueba interrumpida")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1) 