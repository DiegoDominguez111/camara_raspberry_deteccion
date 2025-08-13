# 🚀 Sistema de Conteo de Personas AI - Raspberry Pi 5 + IMX500

## 📋 **DESCRIPCIÓN DEL SISTEMA**

Sistema de conteo de personas en tiempo real optimizado para Raspberry Pi 5 con cámara AI Sony IMX500. El sistema captura video, detecta personas usando YOLO, realiza tracking y cuenta entradas/salidas a través de una línea virtual configurable.

## 🎯 **CARACTERÍSTICAS PRINCIPALES**

- **🎥 Captura en tiempo real** desde cámara IMX500
- **🤖 Detección AI** usando YOLOv8 optimizado
- **📊 Tracking robusto** con IDs persistentes
- **🔢 Conteo automático** de entradas y salidas
- **🌐 Servidor web** con streaming MJPEG y API REST
- **⚙️ Servicios systemd** para operación automática
- **📈 Métricas en tiempo real** (FPS, CPU, memoria, temperatura)

## 🏗️ **ARQUITECTURA DEL SISTEMA**

```
IMX500 Camera → Python Detector → Node.js API → Web Interface
     ↓              ↓              ↓            ↓
  Video Stream → Person Detection → REST API → Browser
     ↓              ↓              ↓            ↓
  MJPEG Feed → Tracking & Count → WebSocket → Real-time Updates
```

## 📁 **ESTRUCTURA DEL PROYECTO**

```
camara_1/
├── detector_entrada_salida_v2.py    # 🎯 Detector principal Python
├── cam_server.py                     # 🌐 Servidor web Flask
├── servicio_camara_node.js           # 🔌 API Node.js
├── config_detector.json              # ⚙️ Configuración del detector
├── camara-ai-detector.service        # 🔧 Servicio systemd detector
├── camara-ai-node.service            # 🔧 Servicio systemd Node.js
├── instalar_sistema.sh               # 🚀 Script de instalación
├── requirements.txt                  # 📦 Dependencias Python
├── package.json                      # 📦 Dependencias Node.js
├── yolov8n.pt                        # 🤖 Modelo YOLO
├── venv_camara_ai/                   # 🐍 Entorno virtual Python
├── node_modules/                     # 📦 Módulos Node.js
├── api/                              # 🔌 API del sistema
└── templates/                        # 🎨 Templates HTML
```

## 🚀 **INSTALACIÓN Y CONFIGURACIÓN**

### 📋 **Requisitos Previos**

- Raspberry Pi 5 con Raspberry Pi OS
- Cámara AI Sony IMX500 conectada
- Python 3.8+
- Node.js 16+
- Acceso root para servicios systemd

### 🔧 **Instalación Automática**

```bash
# Clonar o descargar el proyecto
cd camara_1

# Dar permisos de ejecución
chmod +x instalar_sistema.sh

# Ejecutar instalador automático
sudo ./instalar_sistema.sh
```

### 🔧 **Instalación Manual**

#### 1. **Dependencias Python**
```bash
# Crear entorno virtual
python3 -m venv venv_camara_ai
source venv_camara_ai/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

#### 2. **Dependencias Node.js**
```bash
# Instalar módulos
npm install
```

#### 3. **Configurar Servicios Systemd**
```bash
# Copiar archivos de servicio
sudo cp camara-ai-*.service /etc/systemd/system/

# Recargar systemd
sudo systemctl daemon-reload

# Habilitar servicios
sudo systemctl enable camara-ai-detector
sudo systemctl enable camara-ai-node
```

## 🎮 **USO DEL SISTEMA**

### 🚀 **Inicio Automático (Recomendado)**

```bash
# Iniciar todos los servicios
sudo systemctl start camara-ai-detector
sudo systemctl start camara-ai-node

# Verificar estado
sudo systemctl status camara-ai-*
```

### 🚀 **Inicio Manual (Para desarrollo/pruebas)**

#### **Opción 1: Servidor Web Flask (Streaming + API)**
```bash
# Activar entorno virtual
source venv_camara_ai/bin/activate

# Iniciar servidor web
python3 cam_server.py
```

**Acceso:**
- 🌐 **Interfaz web**: http://localhost:5000
- 📹 **Streaming**: http://localhost:5000/stream
- 📊 **Métricas**: http://localhost:5000/metrics
- 🔢 **Contadores**: http://localhost:5000/counts
- 🏥 **Health**: http://localhost:5000/health

#### **Opción 2: Detector Python Standalone**
```bash
# Activar entorno virtual
source venv_camara_ai/bin/activate

# Ejecutar detector
python3 detector_entrada_salida_v2.py --config config_detector.json
```

#### **Opción 3: API Node.js**
```bash
# Iniciar servidor Node.js
node servicio_camara_node.js
```

## ⚙️ **CONFIGURACIÓN**

### 📝 **Archivo de Configuración Principal**

`config_detector.json` - Configuración del detector Python:

```json
{
  "resolucion": [640, 480],
  "fps_objetivo": 25,
  "confianza_minima": 0.4,
  "area_minima": 2000,
  "roi_puerta": [80, 80, 560, 400],
  "linea_cruce": 240,
  "ancho_banda_cruce": 3,
  "debounce_ms": 300,
  "track_lost_ms": 700,
  "exposure_us": 4000,
  "gain": 1.0
}
```

### 🔧 **Parámetros Importantes**

- **`resolucion`**: Resolución de captura [ancho, alto]
- **`fps_objetivo`**: FPS objetivo de captura
- **`confianza_minima`**: Umbral de confianza para detecciones
- **`roi_puerta`**: Región de interés [x1, y1, x2, y2]
- **`linea_cruce`**: Posición Y de la línea de conteo
- **`debounce_ms`**: Tiempo de anti-rebote en milisegundos

## 🧪 **PRUEBAS Y VALIDACIÓN**

### 🔍 **Pruebas Básicas del Sistema**

#### 1. **Verificar Servicios**
```bash
# Estado de servicios
sudo systemctl status camara-ai-detector
sudo systemctl status camara-ai-node

# Logs de servicios
sudo journalctl -u camara-ai-detector -f
sudo journalctl -u camara-ai-node -f
```

#### 2. **Prueba del Servidor Web**
```bash
# Iniciar servidor
source venv_camara_ai/bin/activate
python3 cam_server.py

# En otra terminal, verificar endpoints
curl http://localhost:5000/health
curl http://localhost:5000/metrics
```

#### 3. **Prueba del Detector**
```bash
# Activar entorno virtual
source venv_camara_ai/bin/activate

# Ejecutar detector
python3 detector_entrada_salida_v2.py --config config_detector.json
```

### 📊 **Métricas de Rendimiento**

#### **Objetivos de Rendimiento**
- **FPS efectivo**: ≥ 20 FPS sostenido
- **Latencia**: < 200ms exposición→evento
- **CPU**: ≤ 35% promedio en 1 núcleo
- **Memoria**: ≤ 300 MB para el proceso de conteo
- **Precisión**: ≥ 95% en conteo a velocidad normal

#### **Monitoreo en Tiempo Real**
```bash
# Métricas del sistema
curl http://localhost:5000/metrics | python3 -m json.tool

# Estado de salud
curl http://localhost:5000/health | python3 -m json.tool

# Contadores
curl http://localhost:5000/counts | python3 -m json.tool
```

### 🎯 **Pruebas de Funcionalidad**

#### **Prueba de Conteo**
1. **Configurar línea de cruce** en `config_detector.json`
2. **Ejecutar sistema** en modo manual
3. **Pasar personas** por la línea de cruce
4. **Verificar contadores** en tiempo real
5. **Validar precisión** del conteo

#### **Prueba de Streaming**
1. **Iniciar servidor web** (`python3 cam_server.py`)
2. **Abrir navegador** en http://localhost:5000
3. **Verificar video** en tiempo real
4. **Comprobar anotaciones** (bounding boxes, línea de cruce)
5. **Validar métricas** en pantalla

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### ❌ **Problemas Comunes**

#### **Cámara no detectada**
```bash
# Verificar conexión física
lsusb | grep -i camera

# Verificar drivers
ls /dev/video*

# Probar comando rpicam
rpicam-hello
```

#### **Servicios no inician**
```bash
# Verificar logs
sudo journalctl -u camara-ai-detector -n 50
sudo journalctl -u camara-ai-node -n 50

# Verificar permisos
ls -la /etc/systemd/system/camara-ai-*.service

# Recargar systemd
sudo systemctl daemon-reload
```

#### **Bajo rendimiento**
```bash
# Verificar temperatura
cat /sys/class/thermal/thermal_zone0/temp

# Verificar uso de CPU
htop

# Verificar uso de memoria
free -h
```

### 🔧 **Comandos de Diagnóstico**

```bash
# Estado del sistema
sudo systemctl status camara-ai-*

# Logs en tiempo real
sudo journalctl -u camara-ai-detector -f
sudo journalctl -u camara-ai-node -f

# Verificar puertos
sudo netstat -tlnp | grep :5000
sudo netstat -tlnp | grep :3000

# Verificar procesos
ps aux | grep python
ps aux | grep node
```

## 📈 **MONITOREO Y MANTENIMIENTO**

### 📊 **Métricas a Monitorear**

- **FPS de captura e inferencia**
- **Uso de CPU y memoria**
- **Temperatura de la Raspberry Pi**
- **Precisión del conteo**
- **Latencia del sistema**

### 🔄 **Mantenimiento Rutinario**

#### **Diario**
- Verificar estado de servicios
- Revisar logs de errores
- Comprobar métricas de rendimiento

#### **Semanal**
- Reiniciar servicios si es necesario
- Verificar espacio en disco
- Actualizar logs del sistema

#### **Mensual**
- Verificar actualizaciones del sistema
- Limpiar logs antiguos
- Validar precisión del conteo

## 🚀 **DESPLIEGUE EN PRODUCCIÓN**

### 📋 **Checklist de Producción**

- [ ] **Hardware verificado** (Raspberry Pi 5 + IMX500)
- [ ] **Sistema operativo actualizado** (Raspberry Pi OS)
- [ ] **Dependencias instaladas** (Python, Node.js)
- [ ] **Servicios systemd configurados** y habilitados
- [ ] **Configuración optimizada** para el entorno
- [ ] **Monitoreo configurado** (logs, métricas)
- [ ] **Backup configurado** del sistema
- [ ] **Documentación actualizada** para el equipo

### 🔒 **Consideraciones de Seguridad**

- **Servicios ejecutándose como usuario no-root** (configurado como root111)
- **Puertos expuestos solo en red local**
- **Logs de acceso y errores habilitados**
- **Monitoreo de recursos del sistema**

### 📦 **Backup y Recuperación**

```bash
# Backup de configuración
sudo cp -r /home/root111/camara_1 /backup/camara_1_$(date +%Y%m%d)

# Backup de servicios systemd
sudo cp /etc/systemd/system/camara-ai-*.service /backup/

# Restaurar desde backup
sudo cp -r /backup/camara_1_YYYYMMDD /home/root111/camara_1
sudo cp /backup/camara-ai-*.service /etc/systemd/system/
sudo systemctl daemon-reload
```

## 📚 **RECURSOS ADICIONALES**

### 🔗 **Enlaces Útiles**

- **Documentación del sistema**: `SISTEMA_FINAL.md`
- **Configuración del detector**: `config_detector.json`
- **Script de instalación**: `instalar_sistema.sh`
- **Servicios systemd**: `camara-ai-*.service`

### 📖 **Referencias Técnicas**

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **OpenCV**: https://opencv.org/
- **Flask**: https://flask.palletsprojects.com/
- **Node.js**: https://nodejs.org/
- **Systemd**: https://systemd.io/

## 🆘 **SOPORTE Y CONTACTO**

### 📧 **Reportar Problemas**

1. **Verificar logs** del servicio correspondiente
2. **Documentar síntomas** y pasos para reproducir
3. **Incluir métricas** del sistema (CPU, memoria, temperatura)
4. **Adjuntar configuración** actual (`config_detector.json`)

### 🔧 **Escalación de Problemas**

1. **Reiniciar servicios** afectados
2. **Verificar recursos** del sistema
3. **Revisar conectividad** de la cámara
4. **Contactar equipo** de desarrollo si persiste

---

## 🎉 **¡SISTEMA LISTO PARA PRODUCCIÓN!**

El sistema de conteo de personas está optimizado, probado y listo para operar en entornos de producción. Sigue las instrucciones de instalación y configuración para un despliegue exitoso.

**🚀 ¡Que tengas éxito con tu implementación!** 