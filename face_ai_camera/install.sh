#!/bin/bash

# Script de instalación automática para Face AI Camera
# Raspberry Pi 5 + AI Camera

set -e

echo "🚀 INSTALADOR AUTOMÁTICO - FACE AI CAMERA"
echo "=========================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que estamos en Raspberry Pi
check_raspberry_pi() {
    print_status "Verificando sistema..."
    
    if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
        print_warning "No se detectó Raspberry Pi. Continuando de todas formas..."
    else
        print_success "Raspberry Pi detectado"
    fi
    
    # Verificar versión de Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION encontrado"
    else
        print_error "Python3 no encontrado. Instalando..."
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv
    fi
}

# Actualizar sistema
update_system() {
    print_status "Actualizando sistema..."
    sudo apt update
    sudo apt upgrade -y
    print_success "Sistema actualizado"
}

# Instalar dependencias del sistema
install_system_dependencies() {
    print_status "Instalando dependencias del sistema..."
    
    # Dependencias básicas
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        libatlas-base-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        libatlas-base-dev \
        libjasper-dev \
        libqtcore4 \
        libqtgui4 \
        libqt4-test \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libatlas-base-dev \
        gfortran \
        wget \
        curl \
        git
    
    print_success "Dependencias del sistema instaladas"
}

# Instalar picamera2
install_picamera2() {
    print_status "Instalando picamera2..."
    
    # Verificar si ya está instalado
    if python3 -c "import picamera2" 2>/dev/null; then
        print_success "picamera2 ya está instalado"
        return
    fi
    
    # Instalar picamera2
    sudo apt install -y python3-picamera2
    
    # Verificar instalación
    if python3 -c "import picamera2" 2>/dev/null; then
        print_success "picamera2 instalado correctamente"
    else
        print_error "Error instalando picamera2"
        exit 1
    fi
}

# Crear entorno virtual
create_virtual_environment() {
    print_status "Creando entorno virtual..."
    
    if [ -d "venv" ]; then
        print_warning "Entorno virtual ya existe. Eliminando..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    print_success "Entorno virtual creado"
}

# Activar entorno virtual e instalar dependencias Python
install_python_dependencies() {
    print_status "Instalando dependencias Python..."
    
    source venv/bin/activate
    
    # Actualizar pip
    pip install --upgrade pip
    
    # Instalar dependencias
    pip install -r requirements.txt
    
    print_success "Dependencias Python instaladas"
}

# Verificar cámara
check_camera() {
    print_status "Verificando cámara..."
    
    # Verificar dispositivos de video
    if ls /dev/video* 1> /dev/null 2>&1; then
        print_success "Dispositivos de video detectados:"
        ls /dev/video*
    else
        print_warning "No se detectaron dispositivos de video"
    fi
    
    # Verificar permisos de cámara
    if groups $USER | grep -q "video"; then
        print_success "Usuario en grupo video"
    else
        print_warning "Usuario no está en grupo video. Agregando..."
        sudo usermod -a -G video $USER
        print_warning "Reinicia la sesión para aplicar cambios"
    fi
}

# Crear directorios necesarios
create_directories() {
    print_status "Creando directorios del proyecto..."
    
    mkdir -p encodings
    mkdir -p models
    mkdir -p logs
    
    print_success "Directorios creados"
}

# Configurar permisos
setup_permissions() {
    print_status "Configurando permisos..."
    
    chmod +x *.py
    chmod +x install.sh
    
    print_success "Permisos configurados"
}

# Probar instalación
test_installation() {
    print_status "Probando instalación..."
    
    source venv/bin/activate
    
    # Prueba básica de importaciones
    if python3 -c "
import cv2
import numpy as np
import face_recognition
from picamera2 import Picamera2
import onnxruntime
print('✅ Todas las importaciones exitosas')
"; then
        print_success "Importaciones básicas funcionando"
    else
        print_error "Error en importaciones básicas"
        exit 1
    fi
}

# Mostrar instrucciones post-instalación
show_post_install_instructions() {
    echo ""
    echo "🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE!"
    echo "======================================"
    echo ""
    echo "📋 Próximos pasos:"
    echo "1. Activa el entorno virtual:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Prueba la cámara:"
    echo "   python camera_test.py"
    echo ""
    echo "3. Registra rostros:"
    echo "   python register_face.py --interactive"
    echo ""
    echo "4. Ejecuta reconocimiento:"
    echo "   python recognize_face.py"
    echo ""
    echo "📚 Para más información, consulta README.md"
    echo ""
    echo "⚠️  IMPORTANTE: Si agregaste el usuario al grupo video,"
    echo "   reinicia la sesión SSH o ejecuta: newgrp video"
}

# Función principal
main() {
    echo "Iniciando instalación automática..."
    echo ""
    
    check_raspberry_pi
    update_system
    install_system_dependencies
    install_picamera2
    create_virtual_environment
    install_python_dependencies
    check_camera
    create_directories
    setup_permissions
    test_installation
    show_post_install_instructions
}

# Ejecutar función principal
main "$@" 