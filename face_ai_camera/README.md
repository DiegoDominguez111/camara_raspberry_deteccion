# 🎯 Sistema de Reconocimiento Facial con Raspberry Pi AI Camera

Sistema completo de reconocimiento facial que aprovecha el procesamiento AI integrado en la Raspberry Pi AI Camera (sensor Sony IMX500).

## 🚀 Características

- **Procesamiento AI en cámara**: Detección facial y generación de embeddings directamente en la AI Camera
- **Reconocimiento en tiempo real**: Comparación de embeddings contra base de datos local
- **Múltiples modelos ONNX**: Soporte para MobileFaceNet, ArcFace, FaceNet
- **Interfaz interactiva**: Scripts para registro y reconocimiento de rostros
- **Optimización Raspberry Pi**: Configurado para máximo rendimiento en Pi 5

## 📋 Requisitos

- Raspberry Pi 5
- Raspberry Pi AI Camera (sensor Sony IMX500)
- Python 3.8+
- 4GB+ RAM recomendado
- Almacenamiento: 2GB+ libre

## 🛠️ Instalación

### 1. Preparar entorno virtual

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Verificar cámara

```bash
# Probar funcionamiento básico
python camera_test.py
```

## 📁 Estructura del Proyecto

```
face_ai_camera/
├── camera_test.py          # Prueba inicial de la cámara
├── register_face.py        # Registro de rostros
├── recognize_face.py       # Reconocimiento en tiempo real
├── test_onnx_models.py     # Prueba de modelos ONNX
├── utils.py               # Funciones auxiliares
├── requirements.txt       # Dependencias
├── README.md             # Este archivo
├── encodings/            # Base de datos de embeddings
└── models/               # Modelos ONNX
```

## 🎮 Uso

### 1. Prueba Inicial

```bash
python camera_test.py
```

Verifica que la AI Camera funciona correctamente y genera imágenes de prueba.

### 2. Registro de Rostros

```bash
# Modo interactivo
python register_face.py --interactive

# Registrar persona específica
python register_face.py --name "Juan Pérez" --samples 5
```

### 3. Reconocimiento en Tiempo Real

```bash
# Reconocimiento básico
python recognize_face.py

# Con parámetros personalizados
python recognize_face.py --threshold 0.7 --method cosine
```

### 4. Prueba de Modelos ONNX

```bash
# Descargar y probar modelo
python test_onnx_models.py --model mobilefacenet

# Comparar todos los modelos
python test_onnx_models.py --compare
```

## ⚙️ Configuración

### Umbral de Reconocimiento

- **0.6 (default)**: Balance entre precisión y sensibilidad
- **0.8+**: Mayor precisión, menos falsos positivos
- **0.4-**: Mayor sensibilidad, más falsos positivos

### Métodos de Comparación

- **cosine**: Similitud coseno (recomendado)
- **euclidean**: Distancia euclidiana

## 🔧 Personalización

### Cambiar Modelo de Embeddings

1. Descargar nuevo modelo:
```bash
python test_onnx_models.py --download arcface
```

2. Modificar `utils.py` para usar el nuevo modelo

### Ajustar Detección Facial

Editar parámetros en `register_face.py` y `recognize_face.py`:

```python
# Sensibilidad de detección
scaleFactor=1.1        # Más bajo = más sensible
minNeighbors=5         # Más alto = menos falsos positivos
minSize=(50, 50)      # Tamaño mínimo de rostro
```

## 📊 Rendimiento

### Métricas Típicas (Raspberry Pi 5)

- **Detección facial**: 15-25 FPS
- **Generación embeddings**: 5-10 FPS
- **Reconocimiento completo**: 3-8 FPS
- **Latencia total**: 100-300ms

### Optimizaciones

1. **Reducir resolución**: Cambiar `size` en configuración de cámara
2. **Ajustar buffer**: Modificar `buffer_count` según memoria disponible
3. **Modelo ligero**: Usar MobileFaceNet en lugar de ArcFace

## 🐛 Solución de Problemas

### Cámara no detectada

```bash
# Verificar conexión
ls /dev/video*

# Reinstalar picamera2
sudo apt update
sudo apt install python3-picamera2
```

### Error de memoria

```bash
# Aumentar swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Cambiar CONF_SWAPSIZE=100 a CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Dependencias faltantes

```bash
# Reinstalar entorno virtual
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🔮 Próximos Pasos

- [ ] Integración con base de datos SQLite
- [ ] API REST para reconocimiento remoto
- [ ] Interfaz web para gestión
- [ ] Soporte para múltiples cámaras
- [ ] Análisis de emociones
- [ ] Detección de mascarillas

## 📚 Referencias

- [Raspberry Pi AI Camera Documentation](https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_face_detection.html)
- [Face Recognition Library](https://github.com/ageitgey/face_recognition)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

**Desarrollado para Raspberry Pi 5 + AI Camera** 🍓📷🤖 