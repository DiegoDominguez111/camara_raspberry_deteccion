#!/usr/bin/env python3
"""
Script para crear un modelo MobileFaceNet simple para pruebas
"""

import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import os

class SimpleMobileFaceNet(nn.Module):
    """Modelo MobileFaceNet simplificado para pruebas"""
    
    def __init__(self, embedding_size=192):
        super(SimpleMobileFaceNet, self).__init__()
        
        # Capa de entrada
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Bloques móviles
        self.mobile_blocks = nn.Sequential(
            self._make_mobile_block(64, 64, 1),
            self._make_mobile_block(64, 128, 2),
            self._make_mobile_block(128, 128, 1),
            self._make_mobile_block(128, 256, 2),
            self._make_mobile_block(256, 256, 1),
            self._make_mobile_block(256, 512, 2),
            self._make_mobile_block(512, 512, 1),
        )
        
        # Capa de salida
        self.conv_last = nn.Conv2d(512, embedding_size, kernel_size=1)
        self.bn_last = nn.BatchNorm2d(embedding_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def _make_mobile_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.mobile_blocks(x)
        
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), -1)
        return x

def create_mobilefacenet_model():
    """Crea y exporta el modelo MobileFaceNet"""
    
    # Crear modelo
    model = SimpleMobileFaceNet(embedding_size=192)
    model.eval()
    
    # Crear tensor de entrada de ejemplo
    dummy_input = torch.randn(1, 3, 112, 112)
    
    # Exportar a ONNX
    onnx_path = "/usr/share/rpi-camera-assets/mobilefacenet.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Modelo MobileFaceNet creado: {onnx_path}")
    
    # Probar el modelo
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Prueba exitosa - Embedding shape: {output.shape}")
        print(f"   - Tamaño de embedding: {output.shape[1]}")
    
    return onnx_path

if __name__ == "__main__":
    create_mobilefacenet_model() 