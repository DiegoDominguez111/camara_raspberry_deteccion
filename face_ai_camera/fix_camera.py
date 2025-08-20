#!/usr/bin/env python3
"""
Script para intentar solucionar problemas de cámara
"""

import os
import subprocess
import time
import sys

def run_command(command, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        
        if result.stdout:
            print("✅ Salida:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Errores:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_camera_status():
    """Verifica el estado actual de las cámaras"""
    print("🔍 Verificando estado de cámaras...")
    
    # Verificar dispositivos
    run_command("ls -la /dev/video*", "Dispositivos de video")
    
    # Verificar módulos cargados
    run_command("lsmod | grep -E '(v4l|video|imx500)'", "Módulos de video cargados")
    
    # Verificar procesos usando cámaras
    run_command("lsof /dev/video* 2>/dev/null || echo 'No hay procesos usando cámaras'", "Procesos usando cámaras")

def try_camera_reset():
    """Intenta resetear las cámaras"""
    print("\n🔄 Intentando resetear cámaras...")
    
    # Detener servicios que puedan estar usando cámaras
    services = ['camera-streamer', 'motion', 'zoneminder']
    for service in services:
        run_command(f"sudo systemctl stop {service} 2>/dev/null || echo 'Servicio {service} no encontrado'", 
                   f"Deteniendo {service}")
    
    # Esperar un momento
    time.sleep(2)
    
    # Verificar si hay procesos zombie
    run_command("ps aux | grep -E '(video|v4l|camera)' | grep -v grep", "Procesos relacionados con video")

def try_camera_reload():
    """Intenta recargar módulos de cámara"""
    print("\n🔄 Intentando recargar módulos de cámara...")
    
    # Listar módulos relacionados
    modules = ['imx500', 'rp1_cfe', 'pisp_be']
    
    for module in modules:
        if run_command(f"lsmod | grep {module}", f"Verificando módulo {module}"):
            print(f"📦 Recargando módulo {module}...")
            run_command(f"sudo rmmod {module}", f"Descargando {module}")
            time.sleep(1)
            run_command(f"sudo modprobe {module}", f"Cargando {module}")
            time.sleep(2)

def test_specific_camera(device_index):
    """Prueba una cámara específica con diferentes configuraciones"""
    print(f"\n🧪 Probando cámara {device_index} con diferentes configuraciones...")
    
    # Crear script de prueba
    test_script = f"""
import cv2
import time

print(f"Probando cámara {device_index}...")

# Configuración 1: Básica
print("Configuración 1: Básica")
cap = cv2.VideoCapture({device_index})
if cap.isOpened():
    print("  ✅ Cámara abierta")
    
    # Configurar propiedades
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Intentar leer frame
    ret, frame = cap.read()
    if ret:
        print(f"  ✅ Frame leído: {{frame.shape}}")
        cv2.imwrite(f"test_camera_{device_index}_config1.jpg", frame)
    else:
        print("  ❌ Error leyendo frame")
    
    cap.release()
else:
    print("  ❌ No se pudo abrir cámara")

# Configuración 2: Con buffer
print("Configuración 2: Con buffer")
cap = cv2.VideoCapture({device_index})
if cap.isOpened():
    print("  ✅ Cámara abierta")
    
    # Configurar buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Esperar estabilización
    time.sleep(2)
    
    # Intentar leer frame
    ret, frame = cap.read()
    if ret:
        print(f"  ✅ Frame leído: {{frame.shape}}")
        cv2.imwrite(f"test_camera_{device_index}_config2.jpg", frame)
    else:
        print("  ❌ Error leyendo frame")
    
    cap.release()
else:
    print("  ❌ No se pudo abrir cámara")

# Configuración 3: Con formato específico
print("Configuración 3: Con formato específico")
cap = cv2.VideoCapture({device_index})
if cap.isOpened():
    print("  ✅ Cámara abierta")
    
    # Intentar diferentes formatos
    formats = [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV')),
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'RGB3'))
    ]
    
    for prop, value in formats:
        cap.set(prop, value)
        time.sleep(0.5)
        
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Formato funcionando: {{frame.shape}}")
            cv2.imwrite(f"test_camera_{device_index}_format_{{value}}.jpg", frame)
            break
        else:
            print(f"  ❌ Formato no funcionando")
    
    cap.release()
else:
    print("  ❌ No se pudo abrir cámara")
"""
    
    # Guardar y ejecutar script
    with open(f"temp_camera_test_{device_index}.py", "w") as f:
        f.write(test_script)
    
    run_command(f"python3 temp_camera_test_{device_index}.py", f"Prueba de cámara {device_index}")
    
    # Limpiar
    os.remove(f"temp_camera_test_{device_index}.py")

def check_camera_info(device_index):
    """Obtiene información detallada de una cámara"""
    print(f"\n📊 Información detallada de cámara {device_index}")
    
    # Información V4L2
    run_command(f"v4l2-ctl -d /dev/video{device_index} --all", f"Información V4L2 de video{device_index}")
    
    # Formatos soportados
    run_command(f"v4l2-ctl -d /dev/video{device_index} --list-formats-ext", f"Formatos soportados por video{device_index}")

def main():
    """Función principal"""
    print("🚀 SOLUCIONADOR DE PROBLEMAS DE CÁMARA")
    print("=" * 45)
    
    try:
        # Verificar estado actual
        check_camera_status()
        
        # Intentar resetear
        try_camera_reset()
        
        # Intentar recargar módulos
        try_camera_reload()
        
        # Verificar estado después de reset
        print("\n🔄 Verificando estado después de reset...")
        check_camera_status()
        
        # Probar cámaras específicas
        print("\n🧪 Probando cámaras específicas...")
        
        # Probar las primeras 5 cámaras
        for i in range(5):
            if os.path.exists(f"/dev/video{i}"):
                test_specific_camera(i)
                time.sleep(1)
        
        # Obtener información de la primera cámara
        if os.path.exists("/dev/video0"):
            check_camera_info(0)
        
        print("\n✅ Solución de problemas completada")
        print("📁 Archivos de prueba generados:")
        run_command("ls -la test_camera_*.jpg 2>/dev/null || echo 'No se generaron archivos de prueba'", 
                   "Archivos de prueba")
        
    except KeyboardInterrupt:
        print("\n⏹️  Solución de problemas interrumpida")
    except Exception as e:
        print(f"\n💥 Error durante solución de problemas: {e}")

if __name__ == "__main__":
    main() 