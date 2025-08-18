# Detección de Personas con Raspberry Pi AI Camera

Este proyecto contiene scripts para detectar personas en tiempo real usando la Raspberry Pi AI Camera con el modelo MobileNet-SSD en el chip IMX500.

## Requisitos

- Raspberry Pi 5 con AI Camera (IMX500)
- Sistema operativo con soporte para `rpicam-hello`
- Python 3.6+

## Scripts Disponibles

### 1. `lectura_mobilessd.py` - Detección Básica
Script principal que detecta personas y muestra sus coordenadas en tiempo real.

**Uso:**
```bash
python3 lectura_mobilessd.py
```

**Salida:**
```
⏳ Esperando detecciones del IMX500 (MobileNet-SSD)...
🔄 Cargando firmware de red en el IMX500 (puede tomar varios minutos)...
[10:40:16] PERSONA conf=0.73 bbox=563,436-2027,1506
[10:40:16] PERSONA conf=0.78 bbox=602,436-2027,1506
```

### 2. `lectura_mobilessd_stats.py` - Detección con Estadísticas
Versión avanzada que incluye estadísticas en tiempo real de las detecciones.

**Uso:**
```bash
python3 lectura_mobilessd_stats.py
```

**Salida:**
```
⏳ Esperando detecciones del IMX500 (MobileNet-SSD)...
🔄 Cargando firmware de red en el IMX500 (puede tomar varios minutos)...
📊 Las estadísticas se mostrarán cada 10 segundos
[10:40:16] PERSONA conf=0.73 bbox=563,436-2027,1506

📊 ESTADÍSTICAS DE DETECCIÓN:
   ⏱️  Tiempo activo: 15.2s
   🎯 Detecciones/min: 45
   📈 Total detecciones: 67
   🔝 Confianza máxima: 0.88
   🔻 Confianza mínima: 0.62
--------------------------------------------------
```

## Parámetros de Configuración

Los scripts utilizan los siguientes parámetros para `rpicam-hello`:

- `-n`: Sin ventana de preview
- `-t 0`: Sin límite de tiempo
- `-v 2`: Verbosidad nivel 2
- `--post-process-file`: Archivo de configuración del modelo MobileNet-SSD
- `--lores-width 640 --lores-height 480`: Resolución de procesamiento

## Formato de Salida

Cada detección incluye:
- **Timestamp**: Hora de la detección
- **Confianza**: Valor entre 0.0 y 1.0 (solo se muestran detecciones ≥ 0.5)
- **Bounding Box**: Coordenadas (x1,y1)-(x2,y2) de la caja delimitadora

## Interpretación de Coordenadas

Las coordenadas están en el espacio de la imagen:
- **x, y**: Esquina superior izquierda de la caja delimitadora
- **x1, y1**: Esquina inferior derecha de la caja delimitadora
- **Resolución**: Las coordenadas están normalizadas para la resolución de procesamiento

## Solución de Problemas

### El script no detecta nada
1. Verifica que la cámara esté conectada correctamente
2. Asegúrate de que haya personas en el campo de visión
3. El firmware puede tardar varios minutos en cargarse la primera vez

### Error de permisos
```bash
sudo chmod +x lectura_mobilessd.py
```

### El proceso no termina correctamente
Usa Ctrl+C para interrumpir el script. El proceso se cerrará automáticamente.

## Rendimiento

- **Latencia**: ~30ms por detección
- **FPS**: ~30 FPS de detección
- **Precisión**: Confianza típica entre 0.6-0.9 para personas claramente visibles
- **Uso de CPU**: Mínimo (procesamiento en el chip IMX500)

## Personalización

Puedes modificar los parámetros en la función `launch()`:
- Cambiar la resolución de procesamiento
- Ajustar el umbral de confianza (actualmente 0.5)
- Modificar el nivel de verbosidad

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT. 