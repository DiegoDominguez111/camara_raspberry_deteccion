#!/bin/bash
# Script para configurar Raspberry Pi 5 con reconocimiento facial

set -e

echo "🚀 Actualizando sistema..."
sudo apt update
sudo apt upgrade -y

echo "🔧 Instalando dependencias de compilación..."
sudo apt install -y build-essential cmake
sudo apt install -y libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
sudo apt install -y libboost-python-dev python3-dev
sudo apt install -y libjpeg-dev zlib1g-dev
sudo apt install -y python3-pip

echo "🐍 Actualizando pip..."
python3 -m pip install --upgrade pip

echo "📦 Instalando librerías de Python desde requirements.txt..."
python3 -m pip install -r ../requirements.txt

echo "✅ Configuración completa. El sistema está listo para ejecutar el script de reconocimiento facial."
