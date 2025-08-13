#!/bin/bash
"""
Script de Instalación para Sistema de Cámara AI
Raspberry Pi 5 + Cámara IMX500
"""

set -e  # Salir en caso de error

echo "🚀 INSTALANDO SISTEMA DE CÁMARA AI - RASPBERRY PI 5"
echo "=================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Verificar que estamos en Raspberry Pi
check_raspberry_pi() {
    print_header "Verificando sistema..."
    
    if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
        print_warning "No se detectó Raspberry Pi. Continuando de todas formas..."
    else
        print_status "Raspberry Pi detectado"
    fi
    
    # Verificar arquitectura
    ARCH=$(uname -m)
    print_status "Arquitectura: $ARCH"
    
    # Verificar versión de Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_status "Python: $PYTHON_VERSION"
    else
        print_error "Python3 no está instalado"
        exit 1
    fi
    
    # Verificar versión de Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js: $NODE_VERSION"
    else
        print_warning "Node.js no está instalado. Se instalará..."
    fi
}

# Actualizar sistema
update_system() {
    print_header "Actualizando sistema..."
    
    print_status "Actualizando lista de paquetes..."
    sudo apt update
    
    print_status "Actualizando paquetes del sistema..."
    sudo apt upgrade -y
    
    print_status "Instalando dependencias del sistema..."
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-opencv \
        python3-numpy \
        python3-psutil \
        git \
        curl \
        build-essential \
        cmake \
        libatlas-base-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf5-103 \
        libqtgui4 \
        libqtwebkit4 \
        libqt4-test \
        python3-pyqt5 \
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
        unzip \
        vim \
        htop \
        iotop \
        nethogs
}

# Instalar Node.js
install_nodejs() {
    print_header "Instalando Node.js..."
    
    if command -v node &> /dev/null; then
        print_status "Node.js ya está instalado"
        return
    fi
    
    print_status "Descargando Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    
    print_status "Instalando Node.js..."
    sudo apt install -y nodejs
    
    print_status "Verificando instalación..."
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    print_status "Node.js: $NODE_VERSION"
    print_status "npm: $NPM_VERSION"
}

# Instalar dependencias de Python
install_python_deps() {
    print_header "Instalando dependencias de Python..."
    
    print_status "Creando entorno virtual..."
    python3 -m venv venv_camara_ai
    
    print_status "Activando entorno virtual..."
    source venv_camara_ai/bin/activate
    
    print_status "Actualizando pip..."
    pip install --upgrade pip
    
    print_status "Instalando dependencias de Python..."
    pip install \
        opencv-python \
        numpy \
        psutil \
        requests \
        ultralytics \
        torch \
        torchvision \
        pillow \
        matplotlib \
        seaborn \
        pandas \
        scikit-learn \
        imutils \
        dlib \
        face-recognition
    
    print_status "Desactivando entorno virtual..."
    deactivate
}

# Instalar dependencias de Node.js
install_nodejs_deps() {
    print_header "Instalando dependencias de Node.js..."
    
    print_status "Instalando dependencias del proyecto..."
    npm install
    
    print_status "Instalando dependencias globales..."
    sudo npm install -g nodemon pm2
}

# Configurar cámara
configure_camera() {
    print_header "Configurando cámara..."
    
    # Verificar si la cámara está habilitada
    if ! vcgencmd get_camera | grep -q "detected=1"; then
        print_warning "Cámara no detectada. Verifica la conexión física."
    else
        print_status "Cámara detectada"
    fi
    
    # Verificar si la cámara está habilitada en config.txt
    if ! grep -q "camera_auto_detect=1" /boot/config.txt; then
        print_status "Habilitando cámara en config.txt..."
        echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt
        print_warning "Reinicia la Raspberry Pi para aplicar los cambios de cámara"
    fi
    
    # Verificar librerías de cámara
    if ! command -v rpicam-still &> /dev/null; then
        print_status "Instalando librerías de cámara..."
        sudo apt install -y libraspberrypi-dev libraspberrypi-doc
    fi
    
    # Verificar librerías de cámara AI
    if ! command -v imx500-all &> /dev/null; then
        print_warning "Librerías IMX500 no encontradas. Verifica la instalación de imx500-all"
    fi
}

# Crear servicios systemd
create_systemd_services() {
    print_header "Creando servicios systemd..."
    
    # Servicio del detector
    cat > camara-ai-detector.service << EOF
[Unit]
Description=Cámara AI Detector de Personas
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv_camara_ai/bin
ExecStart=$(pwd)/venv_camara_ai/bin/python3 detector_entrada_salida_v2.py --config config_detector.json
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Servicio de Node.js
    cat > camara-ai-node.service << EOF
[Unit]
Description=Cámara AI Servicio Node.js
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/node servicio_camara_node.js
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Instalar servicios
    print_status "Instalando servicios systemd..."
    sudo cp camara-ai-detector.service /etc/systemd/system/
    sudo cp camara-ai-node.service /etc/systemd/system/
    
    # Recargar systemd
    sudo systemctl daemon-reload
    
    # Habilitar servicios
    sudo systemctl enable camara-ai-detector.service
    sudo systemctl enable camara-ai-node.service
    
    print_status "Servicios creados y habilitados"
}

# Configurar firewall
configure_firewall() {
    print_header "Configurando firewall..."
    
    # Verificar si ufw está instalado
    if command -v ufw &> /dev/null; then
        print_status "Configurando reglas de firewall..."
        sudo ufw allow 8080/tcp  # Puerto del servicio Node.js
        sudo ufw allow 8081/tcp  # Puerto del servidor web de cámara
        print_status "Firewall configurado"
    else
        print_warning "ufw no está instalado. Instalando..."
        sudo apt install -y ufw
        sudo ufw --force enable
        sudo ufw allow 8080/tcp
        sudo ufw allow 8081/tcp
        print_status "Firewall instalado y configurado"
    fi
}

# Crear scripts de utilidad
create_utility_scripts() {
    print_header "Creando scripts de utilidad..."
    
    # Script de inicio rápido
    cat > iniciar_sistema.sh << 'EOF'
#!/bin/bash
echo "🚀 Iniciando sistema de cámara AI..."
sudo systemctl start camara-ai-node.service
sleep 5
sudo systemctl start camara-ai-detector.service
echo "✅ Sistema iniciado"
echo "📊 Estado: sudo systemctl status camara-ai-*"
echo "🌐 Servicio: http://localhost:8080"
echo "📷 Cámara: http://localhost:8081"
EOF

    # Script de parada
    cat > parar_sistema.sh << 'EOF'
#!/bin/bash
echo "⏹️ Deteniendo sistema de cámara AI..."
sudo systemctl stop camara-ai-detector.service
sudo systemctl stop camara-ai-node.service
echo "✅ Sistema detenido"
EOF

    # Script de reinicio
    cat > reiniciar_sistema.sh << 'EOF'
#!/bin/bash
echo "🔄 Reiniciando sistema de cámara AI..."
sudo systemctl restart camara-ai-node.service
sleep 5
sudo systemctl restart camara-ai-detector.service
echo "✅ Sistema reiniciado"
EOF

    # Script de estado
    cat > estado_sistema.sh << 'EOF'
#!/bin/bash
echo "📊 Estado del sistema de cámara AI:"
echo "=================================="
sudo systemctl status camara-ai-node.service --no-pager -l
echo ""
sudo systemctl status camara-ai-detector.service --no-pager -l
echo ""
echo "📈 Logs recientes:"
sudo journalctl -u camara-ai-* -n 20 --no-pager
EOF

    # Script de logs
    cat > ver_logs.sh << 'EOF'
#!/bin/bash
echo "📋 Logs del sistema de cámara AI:"
echo "================================="
sudo journalctl -u camara-ai-* -f
EOF

    # Hacer ejecutables
    chmod +x iniciar_sistema.sh
    chmod +x parar_sistema.sh
    chmod +x reiniciar_sistema.sh
    chmod +x estado_sistema.sh
    chmod +x ver_logs.sh
    
    print_status "Scripts de utilidad creados"
}

# Crear archivo de configuración por defecto
create_default_config() {
    print_header "Creando configuración por defecto..."
    
    if [ ! -f "config_detector.json" ]; then
        print_status "Creando archivo de configuración por defecto..."
        cat > config_detector.json << 'EOF'
{
  "resolucion": [640, 480],
  "fps_objetivo": 30,
  "confianza_minima": 0.4,
  "nms_iou": 0.45,
  "area_minima": 2000,
  "roi_puerta": [80, 80, 560, 420],
  "linea_cruce": 320,
  "ancho_banda_cruce": 3,
  "debounce_ms": 300,
  "track_lost_ms": 700,
  "exposure_us": 4000,
  "gain": 1.0,
  "ae_lock": true,
  "awb_lock": true,
  "denoise": false,
  "histograma_estable": true,
  "umbral_movimiento": 20,
  "historial_maxlen": 30,
  "distancia_maxima_tracking": 100,
  "timeout_captura": 5,
  "max_cola_detecciones": 100,
  "max_cola_eventos": 50
}
EOF
        print_status "Archivo de configuración creado"
    else
        print_status "Archivo de configuración ya existe"
    fi
}

# Crear archivo README
create_readme() {
    print_header "Creando documentación..."
    
    cat > README.md << 'EOF'
# Sistema de Cámara AI - Raspberry Pi 5

Sistema de conteo de personas en tiempo real usando la cámara AI IMX500 de la Raspberry Pi 5.

## Características

- 🎯 Detección de personas en tiempo real
- 📊 Conteo de entradas y salidas
- 🚪 Línea virtual configurable
- ⚡ 30 FPS objetivo
- 🧠 Inferencia en la cámara IMX500
- 📱 API REST y WebSocket
- 🔧 Configuración en tiempo real

## Arquitectura

```
Cámara IMX500 → Inferencia AI → Raspberry Pi → Tracking → API/WebSocket
```

## Instalación

1. Ejecutar el script de instalación:
   ```bash
   chmod +x instalar_sistema.sh
   ./instalar_sistema.sh
   ```

2. Reiniciar la Raspberry Pi para aplicar cambios de cámara

3. Iniciar el sistema:
   ```bash
   ./iniciar_sistema.sh
   ```

## Uso

### Scripts de utilidad

- `./iniciar_sistema.sh` - Inicia el sistema
- `./parar_sistema.sh` - Detiene el sistema
- `./reiniciar_sistema.sh` - Reinicia el sistema
- `./estado_sistema.sh` - Muestra el estado
- `./ver_logs.sh` - Muestra logs en tiempo real

### API REST

- `GET /health` - Estado del servicio
- `GET /metrics` - Métricas del sistema
- `GET /counts` - Contadores de entrada/salida
- `GET /config` - Configuración actual
- `POST /config` - Actualizar configuración
- `POST /detector/start` - Iniciar detector
- `POST /detector/stop` - Detener detector
- `GET /detector/status` - Estado del detector

### WebSocket

- Endpoint: `ws://localhost:8080`
- Eventos: métricas en tiempo real, eventos de conteo

### Configuración

Editar `config_detector.json` para ajustar:
- Resolución y FPS
- Zona de la puerta (ROI)
- Línea de cruce
- Parámetros de tracking
- Configuración de cámara

## Pruebas

### Prueba sintética

```bash
python3 prueba_sintetica.py --num-pruebas 20
```

### Prueba en vivo

1. Posicionar cámara en puerta
2. Configurar ROI y línea de cruce
3. Ejecutar pruebas de paso normal (1.0-1.6 m/s)

## Métricas de éxito

- ✅ FPS efectivo: ≥ 25 FPS sostenido
- ✅ Latencia: ≤ 150 ms mediana
- ✅ Precisión: ≥ 95% en pruebas de 20 cruces
- ✅ CPU: ≤ 35% promedio
- ✅ RAM: ≤ 300 MB
- ✅ Estabilidad: 2 horas continuas

## Solución de problemas

### Cámara no detectada

```bash
# Verificar conexión física
vcgencmd get_camera

# Habilitar en config.txt
echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt
sudo reboot
```

### Bajo FPS

1. Verificar configuración de cámara
2. Ajustar resolución
3. Verificar temperatura de la Raspberry Pi

### Errores de detección

1. Ajustar ROI de la puerta
2. Configurar línea de cruce
3. Ajustar umbrales de confianza

## Logs

```bash
# Ver logs del detector
sudo journalctl -u camara-ai-detector.service -f

# Ver logs del servicio Node.js
sudo journalctl -u camara-ai-node.service -f

# Ver logs del sistema
./ver_logs.sh
```

## Desarrollo

### Entorno virtual

```bash
source venv_camara_ai/bin/activate
pip install -r requirements.txt
```

### Estructura del proyecto

```
├── detector_entrada_salida_v2.py    # Detector principal
├── servicio_camara_node.js          # Servicio Node.js
├── config_detector.json             # Configuración
├── prueba_sintetica.py              # Pruebas sintéticas
├── instalar_sistema.sh              # Script de instalación
└── README.md                        # Documentación
```

## Licencia

MIT License
EOF

    print_status "Documentación creada"
}

# Crear archivo requirements.txt
create_requirements() {
    print_header "Creando archivo de dependencias..."
    
    cat > requirements.txt << 'EOF'
opencv-python>=4.8.0
numpy>=1.24.0
psutil>=5.9.0
requests>=2.31.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
scikit-learn>=1.3.0
imutils>=0.5.4
dlib>=19.24.0
face-recognition>=1.3.0
EOF

    print_status "Archivo de dependencias creado"
}

# Función principal
main() {
    print_header "Iniciando instalación del sistema de cámara AI..."
    
    # Verificar permisos de root
    if [ "$EUID" -eq 0 ]; then
        print_error "No ejecutar como root. Usa un usuario normal con sudo."
        exit 1
    fi
    
    # Verificar que estamos en el directorio correcto
    if [ ! -f "detector_entrada_salida_v2.py" ]; then
        print_error "Ejecuta este script desde el directorio del proyecto"
        exit 1
    fi
    
    # Ejecutar pasos de instalación
    check_raspberry_pi
    update_system
    install_nodejs
    install_python_deps
    install_nodejs_deps
    configure_camera
    create_systemd_services
    configure_firewall
    create_utility_scripts
    create_default_config
    create_requirements
    create_readme
    
    print_header "🎉 INSTALACIÓN COMPLETADA"
    echo ""
    echo "📋 Próximos pasos:"
    echo "1. Reinicia la Raspberry Pi: sudo reboot"
    echo "2. Inicia el sistema: ./iniciar_sistema.sh"
    echo "3. Verifica el estado: ./estado_sistema.sh"
    echo "4. Accede al servicio: http://localhost:8080"
    echo "5. Ejecuta pruebas: python3 prueba_sintetica.py"
    echo ""
    echo "📚 Documentación: README.md"
    echo "🔧 Scripts de utilidad creados"
    echo "📊 Servicios systemd configurados"
    echo ""
    echo "⚠️  IMPORTANTE: Reinicia la Raspberry Pi para aplicar cambios de cámara"
}

# Ejecutar función principal
main "$@" 