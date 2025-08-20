# Sistema de Reconocimiento Facial en Tiempo Real

Sistema completo de reconocimiento facial diseñado para Raspberry Pi 5 con cámara AI (Sony IMX500).

## 🚀 Características

- **Procesamiento en tiempo real**: Detección y reconocimiento de rostros a 30 FPS
- **Base de datos SQLite**: Almacenamiento eficiente de embeddings y logs
- **Interfaz web moderna**: Dashboard responsive con Bootstrap 5
- **API REST completa**: Endpoints para gestión y monitoreo
- **Streaming MJPEG**: Video en vivo con bounding boxes y nombres
- **Reconocimiento inteligente**: Evita duplicados y optimiza rendimiento

## 🏗️ Arquitectura

```
sistema_reconocimiento/
│── venv/                    # Entorno virtual Python
│── main.py                  # Orquestador principal del sistema
│── camera_handler.py        # Manejo de cámara + inferencia
│── face_db.py              # Base de datos SQLite
│── recognizer.py           # Comparación de embeddings
│── webapp.py               # Servidor web FastAPI
│── utils.py                # Funciones auxiliares
│── static/                 # Archivos estáticos (CSS, JS)
│── templates/              # Templates HTML (Jinja2)
│── tmp/                    # Archivos temporales
│── requirements.txt        # Dependencias Python
│── README.md              # Este archivo
```

## 📋 Requisitos

### Hardware
- Raspberry Pi 5 (recomendado 4GB+ RAM)
- Raspberry Pi AI Camera (Sony IMX500)
- Tarjeta microSD clase 10+ (32GB+)

### Software
- Raspberry Pi OS (Bullseye o Bookworm)
- Python 3.8+
- OpenCV 4.8+
- FastAPI + Uvicorn

## 🛠️ Instalación

### 1. Clonar/Descargar el proyecto
```bash
cd ~
git clone <url-del-repositorio> sistema_reconocimiento
cd sistema_reconocimiento
```

### 2. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar cámara
```bash
# Verificar que la cámara esté habilitada
sudo raspi-config
# Interface Options > Camera > Enable

# Verificar conexión
ls /dev/video*
```

## 🚀 Uso

### Iniciar el sistema completo
```bash
cd sistema_reconocimiento
source venv/bin/activate
python main.py
```

### Solo servidor web
```bash
cd sistema_reconocimiento
source venv/bin/activate
python webapp.py
```

### Acceder a la interfaz
Abrir navegador en: `http://[IP_RASPBERRY_PI]:8000`

## 📱 Interfaz Web

### Dashboard Principal
- **Stream en vivo**: Video de la cámara con reconocimiento en tiempo real
- **Estadísticas**: FPS, personas registradas, reconocimientos
- **Panel de control**: Botones para gestión del sistema
- **Logs recientes**: Historial de reconocimientos

### Funcionalidades
- ✅ Visualización en tiempo real
- ✅ Registro de nuevas personas
- ✅ Monitoreo de rendimiento
- ✅ Gestión de base de datos
- ✅ Control de cámara

## 🔌 API REST

### Endpoints Principales

#### Personas
- `GET /api/people` - Listar personas registradas
- `POST /api/register` - Registrar nueva persona
- `DELETE /api/people/{id}` - Eliminar persona

#### Logs
- `GET /api/logs?limit=50` - Obtener logs recientes

#### Sistema
- `GET /api/stats` - Estadísticas del sistema
- `GET /health` - Estado de salud
- `POST /api/camera/restart` - Reiniciar cámara

#### Video
- `GET /video_feed` - Stream MJPEG en tiempo real

## 🔧 Configuración

### Ajustar parámetros del sistema
Editar `main.py`:

```python
# Configuración de cámara
self.camera_index = 0          # Índice de cámara
self.frame_width = 640         # Ancho de frame
self.frame_height = 480        # Alto de frame
self.web_port = 8000           # Puerto del servidor web

# Intervalo de reconocimiento
self.recognition_interval = 0.1  # 100ms entre reconocimientos
```

### Umbral de confianza
Editar `recognizer.py`:

```python
class FaceRecognizer:
    def __init__(self, db: FaceDatabase, confidence_threshold: float = 0.6):
        # Ajustar umbral (0.0 - 1.0)
        self.confidence_threshold = confidence_threshold
```

## 📊 Rendimiento

### Métricas Objetivo
- **FPS mínimo**: 15 FPS estables
- **FPS meta**: 30 FPS
- **Latencia**: < 100ms para reconocimiento
- **Precisión**: > 90% con umbral 0.6

### Optimizaciones
- Procesamiento en hilos separados
- Colas de frames optimizadas
- Detección Haar cascade eficiente
- Base de datos SQLite optimizada

## 🐛 Solución de Problemas

### Cámara no funciona
```bash
# Verificar permisos
sudo usermod -a -G video $USER

# Verificar drivers
lsmod | grep bcm2835

# Reiniciar servicios
sudo systemctl restart camera
```

### Bajo rendimiento
```bash
# Reducir resolución
# Editar main.py: frame_width = 320, frame_height = 240

# Ajustar intervalo de reconocimiento
# Editar main.py: recognition_interval = 0.2
```

### Error de memoria
```bash
# Aumentar swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 🔒 Seguridad

### Recomendaciones
- Cambiar puerto por defecto (8000)
- Configurar firewall
- Usar HTTPS en producción
- Limitar acceso por IP

### Firewall básico
```bash
sudo ufw enable
sudo ufw allow 8000
sudo ufw allow ssh
```

## 📈 Monitoreo

### Logs del sistema
```bash
# Ver logs en tiempo real
tail -f /var/log/syslog | grep "face_recognition"

# Ver estadísticas
curl http://localhost:8000/api/stats
```

### Métricas de rendimiento
```bash
# CPU y memoria
htop

# Temperatura
vcgencmd measure_temp

# Uso de disco
df -h
```

## 🔄 Actualizaciones

### Actualizar dependencias
```bash
cd sistema_reconocimiento
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Actualizar código
```bash
git pull origin main
# Reiniciar sistema
```

## 🤝 Contribución

### Reportar bugs
1. Verificar que no sea un problema de configuración
2. Incluir logs de error
3. Especificar versión de Raspberry Pi OS
4. Describir pasos para reproducir

### Sugerencias
- Abrir issue en GitHub
- Describir funcionalidad deseada
- Incluir casos de uso

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

## 🙏 Agradecimientos

- OpenCV por el framework de visión por computadora
- FastAPI por el framework web moderno
- Raspberry Pi Foundation por el hardware
- Comunidad de desarrolladores de Python

## 📞 Soporte

### Canales de ayuda
- Issues de GitHub
- Documentación del proyecto
- Comunidad Raspberry Pi

### Información del sistema
```bash
# Versión del sistema
cat /etc/os-release

# Versión de Python
python3 --version

# Versión de OpenCV
python3 -c "import cv2; print(cv2.__version__)"
```

---

**Nota**: Este sistema está diseñado para uso educativo y de desarrollo. Para uso en producción, considerar aspectos de seguridad adicionales. 