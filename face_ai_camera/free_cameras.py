#!/usr/bin/env python3
"""
Script para liberar las cámaras de PipeWire y probar acceso
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

def check_pipewire_status():
    """Verifica el estado de PipeWire"""
    print("🔍 Verificando estado de PipeWire...")
    
    run_command("systemctl status pipewire", "Estado del servicio PipeWire")
    run_command("ps aux | grep pipewire | grep -v grep", "Procesos PipeWire activos")
    run_command("lsof /dev/video* 2>/dev/null | head -20", "Dispositivos de video en uso")

def stop_pipewire():
    """Detiene PipeWire temporalmente"""
    print("\n🛑 Deteniendo PipeWire...")
    
    # Detener servicios relacionados
    services = ['pipewire', 'pipewire-pulse', 'wireplumber']
    
    for service in services:
        print(f"Deteniendo {service}...")
        run_command(f"sudo systemctl stop {service}", f"Deteniendo {service}")
        time.sleep(1)
    
    # Verificar que se detuvieron
    run_command("ps aux | grep -E '(pipewire|wireplumber)' | grep -v grep", "Verificando que se detuvieron")
    
    # Esperar un momento para que se liberen los dispositivos
    time.sleep(3)

def start_pipewire():
    """Reinicia PipeWire"""
    print("\n🔄 Reiniciando PipeWire...")
    
    # Iniciar servicios
    services = ['pipewire', 'pipewire-pulse', 'wireplumber']
    
    for service in services:
        print(f"Iniciando {service}...")
        run_command(f"sudo systemctl start {service}", f"Iniciando {service}")
        time.sleep(2)
    
    # Verificar estado
    run_command("systemctl status pipewire", "Estado final de PipeWire")

def test_camera_access():
    """Prueba acceso a la cámara después de liberar"""
    print("\n🧪 Probando acceso a cámara...")
    
    # Crear script de prueba
    test_script = """
import cv2
import time

print("Probando acceso a cámara después de liberar...")

# Probar diferentes índices
for i in range(5):
    print(f"\\nProbando cámara {i}...")
    
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        print(f"  ✅ Cámara {i} abierta")
        
        # Configurar propiedades
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Intentar leer frame
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Frame leído: {frame.shape}")
            print(f"  ✅ Tipo: {frame.dtype}")
            print(f"  ✅ Rango: [{frame.min()}, {frame.max()}]")
            
            # Guardar frame
            filename = f"camera_{i}_working.jpg"
            cv2.imwrite(filename, frame)
            print(f"  ✅ Frame guardado: {filename}")
            
            cap.release()
            print(f"  🎯 ¡Cámara {i} funciona correctamente!")
            break
        else:
            print(f"  ❌ Error leyendo frame")
            cap.release()
    else:
        print(f"  ❌ No se pudo abrir cámara {i}")

print("\\nPrueba completada")
"""
    
    # Guardar y ejecutar script
    with open("temp_camera_test_freed.py", "w") as f:
        f.write(test_script)
    
    run_command("python3 temp_camera_test_freed.py", "Prueba de cámara liberada")
    
    # Limpiar
    os.remove("temp_camera_test_freed.py")

def configure_pipewire_exclusions():
    """Configura exclusiones en PipeWire para las cámaras"""
    print("\n⚙️  Configurando exclusiones en PipeWire...")
    
    # Crear archivo de configuración temporal
    config_content = """
context.modules = [
    {   name = "libpipewire-module-rt"
        args = {
            nice.level = -11
            rt.prio = 88
            rt.time.soft = 200000000
            rt.time.hard = 200000000
        }
        flags = [ ifexists nofail ]
    }
    {   name = "libpipewire-module-protocol-native" }
    {   name = "libpipewire-module-client-node" }
    {   name = "libpipewire-module-adapter" }
    {   name = "libpipewire-module-metadata" }
    
    {   name = "libpipewire-module-rtkit"
        args = {
            nice.level = -11
            rt.prio = 88
            rt.time.soft = 200000000
            rt.time.hard = 200000000
        }
        flags = [ ifexists nofail ]
    }
    
    # Excluir dispositivos de video específicos
    {   name = "libpipewire-module-v4l2"
        args = {
            # Excluir dispositivos de cámara
            exclude.devices = [
                "/dev/video0"
                "/dev/video1" 
                "/dev/video2"
                "/dev/video3"
                "/dev/video4"
            ]
        }
        flags = [ ifexists nofail ]
    }
]
"""
    
    # Guardar configuración temporal
    config_path = "/tmp/pipewire-camera-exclude.conf"
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✅ Configuración temporal guardada en {config_path}")
    print("⚠️  Para aplicar permanentemente, copia este archivo a:")
    print("   /etc/pipewire/pipewire.conf.d/")
    
    return config_path

def main():
    """Función principal"""
    print("🚀 LIBERADOR DE CÁMARAS - PIPEWIRE")
    print("=" * 40)
    
    try:
        # Verificar estado inicial
        check_pipewire_status()
        
        # Preguntar al usuario
        print("\n🤔 ¿Deseas detener PipeWire temporalmente para probar las cámaras?")
        print("⚠️  Esto detendrá el audio y video del sistema")
        response = input("Continuar? (s/n): ").lower()
        
        if response != 's':
            print("❌ Operación cancelada")
            return
        
        # Detener PipeWire
        stop_pipewire()
        
        # Verificar dispositivos liberados
        print("\n🔍 Verificando dispositivos liberados...")
        run_command("lsof /dev/video* 2>/dev/null || echo 'No hay dispositivos en uso'", "Dispositivos de video")
        
        # Probar acceso a cámara
        test_camera_access()
        
        # Configurar exclusiones
        config_path = configure_pipewire_exclusions()
        
        # Preguntar si reiniciar PipeWire
        print("\n🤔 ¿Deseas reiniciar PipeWire ahora?")
        response = input("Reiniciar? (s/n): ").lower()
        
        if response == 's':
            start_pipewire()
        else:
            print("⚠️  PipeWire permanece detenido")
            print("🔧 Para reiniciarlo manualmente:")
            print("   sudo systemctl start pipewire pipewire-pulse wireplumber")
        
        print("\n✅ Operación completada")
        print("📁 Archivos de prueba generados:")
        run_command("ls -la camera_*_working.jpg 2>/dev/null || echo 'No se generaron archivos de prueba'", 
                   "Archivos de prueba")
        
    except KeyboardInterrupt:
        print("\n⏹️  Operación interrumpida")
    except Exception as e:
        print(f"\n💥 Error durante la operación: {e}")

if __name__ == "__main__":
    main() 