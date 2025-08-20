#!/bin/bash

# Script de inicio rápido para el Sistema de Reconocimiento Facial
# Raspberry Pi 5 + AI Camera

echo "🚀 INICIANDO SISTEMA DE RECONOCIMIENTO FACIAL"
echo "=============================================="

# Verificar que estamos en el directorio correcto
if [ ! -f "main.py" ]; then
    echo "❌ Error: Ejecuta este script desde el directorio sistema_reconocimiento/"
    exit 1
fi

# Verificar que el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Error: Entorno virtual no encontrado. Ejecuta primero:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Verificar dependencias
echo "📦 Verificando dependencias..."
python -c "import cv2, numpy, fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Dependencias faltantes. Ejecuta:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Verificar cámara
echo "📷 Verificando cámara..."
if [ -e "/dev/video0" ]; then
    echo "✅ Cámara detectada en /dev/video0"
else
    echo "⚠️  Cámara no detectada. Verifica la conexión y permisos."
    echo "   Ejecuta: sudo usermod -a -G video \$USER"
    echo "   Luego reinicia la sesión."
fi

# Mostrar información del sistema
echo ""
echo "📊 INFORMACIÓN DEL SISTEMA:"
echo "   - Python: $(python --version)"
echo "   - OpenCV: $(python -c "import cv2; print(cv2.__version__)")"
echo "   - FastAPI: $(python -c "import fastapi; print(fastapi.__version__)")"
echo "   - Directorio: $(pwd)"
echo ""

# Preguntar modo de ejecución
echo "🎯 SELECCIONA EL MODO DE EJECUCIÓN:"
echo "   1) Sistema completo (cámara + web + reconocimiento)"
echo "   2) Solo servidor web (sin cámara)"
echo "   3) Solo pruebas del sistema"
echo "   4) Salir"
echo ""

read -p "Selecciona una opción (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 INICIANDO SISTEMA COMPLETO..."
        echo "   - Cámara: Activada"
        echo "   - Web: http://0.0.0.0:8000"
        echo "   - Reconocimiento: Activado"
        echo ""
        echo "Presiona Ctrl+C para detener"
        echo ""
        python main.py
        ;;
    2)
        echo ""
        echo "🌐 INICIANDO SOLO SERVIDOR WEB..."
        echo "   - Web: http://0.0.0.0:8000"
        echo "   - Cámara: Desactivada"
        echo ""
        echo "Presiona Ctrl+C para detener"
        echo ""
        python webapp.py
        ;;
    3)
        echo ""
        echo "🧪 EJECUTANDO PRUEBAS DEL SISTEMA..."
        python test_system.py
        ;;
    4)
        echo "👋 ¡Hasta luego!"
        exit 0
        ;;
    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac 