#!/usr/bin/env python3
"""
Script de diagnóstico completo para la cámara
Ayuda a identificar problemas de configuración
"""

import os
import subprocess
import sys
import time

def run_command(command, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n🔍 {description}")
    print(f"Comando: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        
        if result.stdout:
            print("Salida estándar:")
            print(result.stdout)
        
        if result.stderr:
            print("Errores:")
            print(result.stderr)
        
        print(f"Código de salida: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Comando expiró por tiempo")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")
        return False

def check_system_info():
    """Verifica información del sistema"""
    print("🚀 DIAGNÓSTICO DE CÁMARA - RASPBERRY PI")
    print("=" * 50)
    
    # Información del sistema
    run_command("uname -a", "Información del sistema")
    run_command("cat /proc/cpuinfo | grep Model", "Modelo de Raspberry Pi")
    run_command("cat /proc/version", "Versión del kernel")
    run_command("lsb_release -a", "Versión de la distribución")

def check_camera_devices():
    """Verifica dispositivos de cámara"""
    print("\n📷 DISPOSITIVOS DE CÁMARA")
    print("=" * 30)
    
    # Listar dispositivos de video
    run_command("ls -la /dev/video*", "Dispositivos de video disponibles")
    
    # Información de dispositivos
    run_command("v4l2-ctl --list-devices", "Información de dispositivos V4L2")
    
    # Verificar permisos
    run_command("ls -la /dev/video0", "Permisos del primer dispositivo de video")
    
    # Verificar grupo video
    run_command("groups $USER", "Grupos del usuario actual")

def check_camera_modules():
    """Verifica módulos de cámara cargados"""
    print("\n🔧 MÓDULOS DE CÁMARA")
    print("=" * 25)
    
    # Módulos cargados
    run_command("lsmod | grep -i camera", "Módulos de cámara cargados")
    run_command("lsmod | grep -i v4l", "Módulos V4L2 cargados")
    run_command("lsmod | grep -i uvc", "Módulos UVC cargados")
    
    # Configuración de módulos
    run_command("cat /etc/modules | grep -i camera", "Módulos de cámara en /etc/modules")

def check_camera_processes():
    """Verifica procesos relacionados con cámara"""
    print("\n⚙️  PROCESOS DE CÁMARA")
    print("=" * 25)
    
    # Procesos activos
    run_command("ps aux | grep -i camera", "Procesos de cámara activos")
    run_command("ps aux | grep -i v4l", "Procesos V4L2 activos")
    
    # Puertos en uso
    run_command("netstat -tuln | grep -E ':(80|8080|5000)'", "Puertos web comunes")

def check_camera_config():
    """Verifica configuración de cámara"""
    print("\n⚙️  CONFIGURACIÓN DE CÁMARA")
    print("=" * 30)
    
    # Configuración de boot
    run_command("cat /boot/config.txt | grep -i camera", "Configuración de cámara en boot")
    
    # Configuración de interfaces
    run_command("cat /boot/config.txt | grep -i enable", "Interfaces habilitadas")
    
    # Configuración de GPU
    run_command("cat /boot/config.txt | grep -i gpu", "Configuración de GPU")

def check_python_packages():
    """Verifica paquetes Python relacionados con cámara"""
    print("\n🐍 PAQUETES PYTHON")
    print("=" * 20)
    
    # Verificar OpenCV
    run_command("python3 -c 'import cv2; print(f\"OpenCV versión: {cv2.__version__}\")'", "Versión de OpenCV")
    
    # Verificar picamera2
    run_command("python3 -c 'import picamera2; print(\"picamera2 disponible\")'", "picamera2 disponible")
    
    # Verificar numpy
    run_command("python3 -c 'import numpy; print(f\"NumPy versión: {numpy.__version__}\")'", "Versión de NumPy")

def test_camera_access():
    """Prueba acceso básico a la cámara"""
    print("\n🧪 PRUEBA DE ACCESO")
    print("=" * 20)
    
    # Crear script de prueba temporal
    test_script = """
import cv2
import sys

print("Probando acceso a cámara...")
for i in range(5):
    print(f"Probando cámara {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"  ✅ Cámara {i} abierta")
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Frame leído: {frame.shape}")
        else:
            print(f"  ❌ Error leyendo frame")
        cap.release()
    else:
        print(f"  ❌ No se pudo abrir cámara {i}")
"""
    
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    # Ejecutar prueba
    run_command("python3 temp_test.py", "Prueba de acceso a cámara")
    
    # Limpiar
    os.remove("temp_test.py")

def check_network_cameras():
    """Verifica si hay cámaras de red disponibles"""
    print("\n🌐 CÁMARAS DE RED")
    print("=" * 20)
    
    # Verificar interfaces de red
    run_command("ip addr show", "Interfaces de red")
    
    # Verificar conectividad
    run_command("ping -c 3 8.8.8.8", "Prueba de conectividad")

def generate_report():
    """Genera un reporte de diagnóstico"""
    print("\n📋 REPORTE DE DIAGNÓSTICO")
    print("=" * 30)
    
    report_file = f"camera_diagnostic_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"📄 Generando reporte: {report_file}")
    print("\n💡 RECOMENDACIONES:")
    print("1. Verificar conexión física de la cámara")
    print("2. Revisar permisos de usuario")
    print("3. Verificar drivers instalados")
    print("4. Comprobar configuración de boot")
    print("5. Revisar logs del sistema")
    
    if os.path.exists("/var/log/syslog"):
        print("\n📊 Últimos logs del sistema:")
        run_command("tail -20 /var/log/syslog | grep -i camera", "Logs recientes de cámara")

def main():
    """Función principal"""
    try:
        check_system_info()
        check_camera_devices()
        check_camera_modules()
        check_camera_processes()
        check_camera_config()
        check_python_packages()
        test_camera_access()
        check_network_cameras()
        generate_report()
        
        print("\n✅ Diagnóstico completado")
        print("📚 Revisa el reporte generado para más detalles")
        
    except KeyboardInterrupt:
        print("\n⏹️  Diagnóstico interrumpido")
    except Exception as e:
        print(f"\n💥 Error durante diagnóstico: {e}")

if __name__ == "__main__":
    main() 