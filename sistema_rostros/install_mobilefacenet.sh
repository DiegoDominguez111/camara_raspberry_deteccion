#!/bin/bash

echo "🚀 Instalando MobileFaceNet para GPU IMX500..."

# Actualizar sistema
echo "📦 Actualizando sistema..."
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev

# Crear entorno virtual
echo "🐍 Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

# Actualizar pip
echo "⬆️ Actualizando pip..."
pip install --upgrade pip setuptools wheel

# Instalar dependencias optimizadas
echo "📚 Instalando dependencias..."
pip install -r requirements_mobilefacenet.txt

# Descargar modelo MobileFaceNet pre-entrenado
echo "🤖 Descargando modelo MobileFaceNet..."
mkdir -p models
cd models

# Descargar modelo ONNX de MobileFaceNet
wget -O mobilefacenet.onnx https://github.com/XiaoyuZuo/MobileFaceNet/raw/master/models/mobilefacenet.onnx

if [ $? -eq 0 ]; then
    echo "✅ Modelo MobileFaceNet descargado exitosamente"
else
    echo "⚠️ No se pudo descargar el modelo, creando uno básico..."
    # Crear un modelo básico de ejemplo
    python3 -c "
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnxruntime.quantization import quantize_dynamic

# Crear un modelo básico de ejemplo
input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 3, 112, 112])
output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 192])

# Crear modelo simple
model = helper.make_model([input_tensor, output_tensor], producer_name='MobileFaceNet-Basic')
onnx.save(model, 'mobilefacenet.onnx')
print('Modelo básico creado')
"
fi

cd ..

echo "🎉 Instalación completada!"
echo "💡 Para activar el entorno virtual: source venv/bin/activate"
echo "🚀 Para ejecutar: python3 lectura_encodings_mobilefacenet.py" 