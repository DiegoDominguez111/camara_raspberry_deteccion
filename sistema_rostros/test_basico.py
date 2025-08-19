#!/usr/bin/env python3
"""
Script de prueba básico para validar el sistema optimizado
"""
import time
import sys
import os

def test_dependencias_basicas():
    """Prueba las dependencias básicas disponibles"""
    print("🧪 Probando dependencias básicas...")
    
    try:
        import cv2
        print("✅ OpenCV disponible")
        print(f"   • Versión: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV no disponible: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy disponible")
        print(f"   • Versión: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy no disponible: {e}")
        return False
    
    try:
        import psutil
        print("✅ psutil disponible")
        print(f"   • Versión: {psutil.__version__}")
    except ImportError as e:
        print(f"❌ psutil no disponible: {e}")
        return False
    
    return True

def test_sistema_archivos():
    """Prueba que los archivos del sistema estén disponibles"""
    print("\n📁 Probando archivos del sistema...")
    
    archivos_requeridos = [
        "lectura_encodings.py",
        "test_sistema.py",
        "requirements.txt",
        "install.sh",
        "README.md"
    ]
    
    todos_disponibles = True
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"✅ {archivo} disponible")
        else:
            print(f"❌ {archivo} no encontrado")
            todos_disponibles = False
    
    return todos_disponibles

def test_optimizaciones_implementadas():
    """Verifica que las optimizaciones estén implementadas en el código"""
    print("\n⚡ Verificando optimizaciones implementadas...")
    
    try:
        with open("lectura_encodings.py", "r") as f:
            contenido = f.read()
        
        optimizaciones = {
            "VIDEO_FPS = 30": "FPS de video aumentado a 30",
            "RECOGNITION_FPS = 5": "FPS de reconocimiento aumentado a 5",
            "maxsize=3": "Cola de video reducida a 3 frames",
            "maxsize=1": "Cola de reconocimiento reducida a 1 frame",
            "model=\"hog\"": "Modelo HOG para reconocimiento",
            "fx=0.4, fy=0.4": "Frame de reconocimiento reducido a 40%",
            "MAX_FRAME_AGE = 0.5": "Control de antigüedad de frames",
            "timeout=0.05": "Timeouts reducidos para menor latencia",
            "imageRendering = 'optimizeSpeed'": "Optimización de renderizado frontend"
        }
        
        implementadas = 0
        for codigo, descripcion in optimizaciones.items():
            if codigo in contenido:
                print(f"✅ {descripcion}")
                implementadas += 1
            else:
                print(f"❌ {descripcion} - NO IMPLEMENTADA")
        
        print(f"\n📊 Resumen: {implementadas}/{len(optimizaciones)} optimizaciones implementadas")
        return implementadas == len(optimizaciones)
        
    except Exception as e:
        print(f"❌ Error verificando optimizaciones: {e}")
        return False

def test_rendimiento_sistema():
    """Prueba el rendimiento básico del sistema"""
    print("\n📊 Probando rendimiento básico del sistema...")
    
    try:
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   • CPU: {cpu_percent:.1f}%")
        
        # Memoria
        memoria = psutil.virtual_memory()
        print(f"   • RAM: {memoria.percent:.1f}%")
        print(f"   • RAM Total: {memoria.total / (1024**3):.1f} GB")
        print(f"   • RAM Disponible: {memoria.available / (1024**3):.1f} GB")
        
        # Temperatura (solo en Raspberry Pi)
        try:
            if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read().strip()) / 1000.0
                print(f"   • Temperatura CPU: {temp:.1f}°C")
            else:
                print("   • Temperatura CPU: No disponible (no es Raspberry Pi)")
        except:
            print("   • Temperatura CPU: Error al leer")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando rendimiento: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 Iniciando pruebas básicas del sistema optimizado...")
    print("=" * 60)
    
    # Prueba 1: Dependencias básicas
    if not test_dependencias_basicas():
        print("\n⚠️  Algunas dependencias básicas no están disponibles")
        print("   Ejecuta: pip install -r requirements.txt")
    
    # Prueba 2: Archivos del sistema
    if not test_sistema_archivos():
        print("\n⚠️  Algunos archivos del sistema no están disponibles")
    
    # Prueba 3: Optimizaciones implementadas
    if not test_optimizaciones_implementadas():
        print("\n⚠️  No todas las optimizaciones están implementadas")
    
    # Prueba 4: Rendimiento del sistema
    test_rendimiento_sistema()
    
    print("\n" + "=" * 60)
    print("🎉 Pruebas básicas completadas!")
    
    print("\n📋 Próximos pasos:")
    print("   1. Instalar dependencias faltantes:")
    print("      pip install -r requirements.txt")
    print("   2. Ejecutar el sistema:")
    print("      python3 lectura_encodings.py")
    print("   3. Ejecutar pruebas completas:")
    print("      python3 test_sistema.py")
    
    print("\n💡 Para instalación automática:")
    print("   chmod +x install.sh && ./install.sh")

if __name__ == "__main__":
    main() 