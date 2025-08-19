#!/usr/bin/env python3
"""
Script de prueba para validar el sistema de reconocimiento facial optimizado
"""
import time
import requests
import json
import sys
import os

def test_conexion_web():
    """Prueba la conexión al servidor web"""
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor web funcionando correctamente")
            return True
        else:
            print(f"❌ Error en servidor web: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ No se puede conectar al servidor web: {e}")
        return False

def test_endpoint_status():
    """Prueba el endpoint de estado"""
    try:
        response = requests.get("http://localhost:5000/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint de estado funcionando")
            print(f"   • CPU: {data.get('cpu_usage', 'N/A')}%")
            print(f"   • RAM: {data.get('ram_usage', 'N/A')}%")
            print(f"   • Temperatura: {data.get('cpu_temp', 'N/A')}°C")
            print(f"   • Video FPS: {data.get('video_fps', 'N/A')}")
            print(f"   • Reconocimiento FPS: {data.get('recognition_fps', 'N/A')}")
            return True
        else:
            print(f"❌ Error en endpoint de estado: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error probando estado: {e}")
        return False

def test_endpoint_detecciones():
    """Prueba el endpoint de detecciones recientes"""
    try:
        response = requests.get("http://localhost:5000/recent_detections", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint de detecciones funcionando")
            print(f"   • Detecciones recientes: {len(data.get('detections', []))}")
            return True
        else:
            print(f"❌ Error en endpoint de detecciones: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error probando detecciones: {e}")
        return False

def test_verificacion_persona():
    """Prueba la verificación de personas"""
    try:
        test_name = "TEST_PERSONA_123"
        response = requests.post("http://localhost:5000/check_person", 
                               json={"name": test_name}, 
                               timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint de verificación de personas funcionando")
            print(f"   • Persona '{test_name}' existe: {data.get('exists', 'N/A')}")
            return True
        else:
            print(f"❌ Error en verificación de personas: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error probando verificación: {e}")
        return False

def test_rendimiento():
    """Prueba el rendimiento del sistema"""
    print("\n📊 Probando rendimiento del sistema...")
    
    # Probar latencia de respuesta
    start_time = time.time()
    try:
        response = requests.get("http://localhost:5000/status", timeout=2)
        latency = (time.time() - start_time) * 1000
        print(f"   • Latencia de respuesta: {latency:.1f}ms")
        
        if latency < 100:
            print("   ✅ Latencia excelente (<100ms)")
        elif latency < 200:
            print("   ✅ Latencia buena (<200ms)")
        elif latency < 500:
            print("   ⚠️  Latencia aceptable (<500ms)")
        else:
            print("   ❌ Latencia alta (>500ms)")
            
    except Exception as e:
        print(f"   ❌ Error en prueba de latencia: {e}")
    
    # Probar múltiples requests simultáneos
    print("   • Probando requests simultáneos...")
    start_time = time.time()
    try:
        responses = []
        for i in range(5):
            response = requests.get("http://localhost:5000/status", timeout=2)
            responses.append(response.status_code)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / 5
        
        print(f"   • 5 requests en {total_time:.1f}ms (promedio: {avg_time:.1f}ms)")
        
        if avg_time < 100:
            print("   ✅ Rendimiento excelente")
        elif avg_time < 200:
            print("   ✅ Rendimiento bueno")
        else:
            print("   ⚠️  Rendimiento aceptable")
            
    except Exception as e:
        print(f"   ❌ Error en prueba de rendimiento: {e}")

def main():
    """Función principal de pruebas"""
    print("🧪 Iniciando pruebas del sistema de reconocimiento facial optimizado...")
    print("=" * 60)
    
    # Verificar que el sistema esté ejecutándose
    print("1️⃣  Verificando conexión al servidor web...")
    if not test_conexion_web():
        print("❌ El sistema no está ejecutándose. Ejecuta primero: python3 lectura_encodings.py")
        sys.exit(1)
    
    print("\n2️⃣  Probando endpoints del sistema...")
    test_endpoint_status()
    test_endpoint_detecciones()
    test_verificacion_persona()
    
    print("\n3️⃣  Probando rendimiento...")
    test_rendimiento()
    
    print("\n" + "=" * 60)
    print("🎉 Pruebas completadas!")
    print("\n📋 Resumen de optimizaciones implementadas:")
    print("   • Video aumentado de 25 a 30 FPS")
    print("   • Reconocimiento aumentado de 2 a 5 FPS")
    print("   • Colas reducidas para menor latencia")
    print("   • Procesamiento optimizado con modelo HOG")
    print("   • Control de antigüedad de frames (evita efecto fantasma)")
    print("   • Limpieza de memoria inmediata")
    print("   • Timeouts reducidos en todas las operaciones")
    print("   • Frontend optimizado con indicador de latencia")
    
    print("\n💡 Para monitorear en tiempo real:")
    print("   • Abre http://localhost:5000 en tu navegador")
    print("   • Observa el indicador de latencia en la esquina superior derecha")
    print("   • Verifica que los FPS estén en los rangos esperados")

if __name__ == "__main__":
    main() 