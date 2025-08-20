#!/usr/bin/env python3
"""
Prueba simple de la cámara para verificar funcionamiento básico
"""

import cv2
import time
import sys

def test_opencv_camera():
    """Prueba la cámara usando OpenCV"""
    print("🔍 Probando cámara con OpenCV...")
    
    # Probar diferentes índices de cámara
    for i in range(5):
        print(f"📷 Probando cámara {i}...")
        
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✅ Cámara {i} abierta exitosamente")
            
            # Leer un frame
            ret, frame = cap.read()
            
            if ret:
                print(f"   - Frame capturado: {frame.shape}")
                print(f"   - Tipo: {frame.dtype}")
                print(f"   - Rango: [{frame.min()}, {frame.max()}]")
                
                # Guardar frame de prueba
                filename = f"test_camera_{i}.jpg"
                cv2.imwrite(filename, frame)
                print(f"   - Frame guardado: {filename}")
                
                cap.release()
                return True
            else:
                print(f"   - Error leyendo frame")
                cap.release()
        else:
            print(f"   - No se pudo abrir cámara {i}")
    
    return False

def test_picamera2():
    """Prueba la cámara usando picamera2"""
    print("\n🔍 Probando cámara con picamera2...")
    
    try:
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        
        # Configuración básica
        config = picam2.create_preview_configuration(
            main={"size": (640, 480)},
            buffer_count=2
        )
        
        picam2.configure(config)
        picam2.start()
        
        print("✅ picamera2 iniciado")
        
        # Esperar estabilización
        time.sleep(2)
        
        # Capturar frame
        frame = picam2.capture_array()
        
        if frame is not None and frame.size > 0:
            print(f"✅ Frame capturado: {frame.shape}")
            print(f"   - Tipo: {frame.dtype}")
            print(f"   - Rango: [{frame.min()}, {frame.max()}]")
            
            # Guardar frame
            filename = "test_picamera2.jpg"
            cv2.imwrite(filename, frame)
            print(f"   - Frame guardado: {filename}")
            
            picam2.stop()
            return True
        else:
            print("❌ No se pudo capturar frame")
            picam2.stop()
            return False
            
    except Exception as e:
        print(f"❌ Error con picamera2: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 PRUEBA SIMPLE DE CÁMARA")
    print("=" * 30)
    
    # Prueba 1: OpenCV
    opencv_success = test_opencv_camera()
    
    # Prueba 2: picamera2
    picamera2_success = test_picamera2()
    
    # Resumen
    print("\n" + "=" * 30)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 30)
    print(f"OpenCV: {'✅' if opencv_success else '❌'}")
    print(f"picamera2: {'✅' if picamera2_success else '❌'}")
    
    if opencv_success or picamera2_success:
        print("\n🎯 Al menos una cámara funciona correctamente")
        print("🔄 Continuar con el proyecto...")
        return True
    else:
        print("\n❌ Ninguna cámara funciona")
        print("🔧 Revisar conexión y permisos")
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