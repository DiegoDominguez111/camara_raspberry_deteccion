# Ejemplo de Uso del SDK - Múltiples Cámaras

Este ejemplo demuestra cómo usar el SDK genérico para conectar y gestionar múltiples cámaras Raspberry Pi simultáneamente.

## 🎯 Configuración para tu Red

### IP de tu Cámara
La cámara principal está configurada para usar la IP: **192.168.1.74**

### Configuración de Red
- **Puerto**: 8082 (puerto por defecto del servidor Python)
- **Protocolo**: HTTP
- **URL completa**: `http://192.168.1.74:8082`

## 🚀 Instalación y Configuración

### 1. Verificar que el SDK esté disponible
```bash
cd api/node/camera-sdk
ls -la src/index.js
```

### 2. Verificar que Node.js esté instalado
```bash
node --version
# Debe ser 18.0.0 o superior
```

### 3. Configurar la IP de tu cámara
Edita el archivo `multi-camera-example.js` y cambia la IP:

```javascript
const CAMERAS_CONFIG = {
  'mi_camara': {
    baseUrl: 'http://192.168.1.74:8082',  // ← Tu IP aquí
    timeout: 10000,
    retries: 3
  }
  // Puedes agregar más cámaras aquí
};
```

## 🧪 Ejecutar las Pruebas

### Prueba Simple (Recomendada para empezar)
```bash
cd api/node/camera-sdk
node test-simple.js
```

**Esta prueba:**
- Conecta a una sola cámara
- Verifica conectividad básica
- Prueba operaciones individuales
- Es más rápida y fácil de debuggear

### Ejemplo Completo de Múltiples Cámaras
```bash
cd api/node/camera-sdk
node examples/multi-camera-example.js
```

**Este ejemplo:**
- Configura múltiples cámaras (incluyendo la tuya)
- Ejecuta operaciones en lote
- Monitorea continuamente
- Demuestra todas las funcionalidades

## 📋 Configuración Personalizada

### Para una sola cámara (tu caso)
```javascript
import { CameraSDK } from '../src/index.js';

const camera = new CameraSDK({
  baseUrl: 'http://192.168.1.74:8082',
  timeout: 10000,        // 10 segundos
  retries: 3,            // 3 reintentos
  retryDelay: 1000       // 1 segundo entre reintentos
});

// Usar la cámara
const status = await camera.getStatus();
console.log('Estado:', status);
```

### Para múltiples cámaras
```javascript
import { CameraManager } from '../src/index.js';

const manager = new CameraManager();

// Tu cámara principal
manager.addCamera('principal', {
  baseUrl: 'http://192.168.1.74:8082',
  timeout: 10000
});

// Otras cámaras (si las tienes)
manager.addCamera('secundaria', {
  baseUrl: 'http://192.168.1.75:8082',
  timeout: 10000
});

// Operaciones en lote
const { results, errors } = await manager.getAllStatuses();
```

## 🔧 Solución de Problemas

### Error: "Failed to connect to camera"
1. **Verificar IP**: Confirma que 192.168.1.74 es la IP correcta
2. **Verificar puerto**: El servidor Python debe estar en puerto 8082
3. **Verificar red**: Asegúrate de que puedas hacer ping a la IP
4. **Verificar firewall**: El puerto 8082 debe estar abierto

### Error: "ECONNREFUSED"
1. **Servidor no ejecutándose**: El servidor Python debe estar activo
2. **Puerto incorrecto**: Verifica que el servidor use puerto 8082
3. **IP incorrecta**: Confirma la IP de la cámara

### Error: "fetch failed"
1. **Node.js versión**: Necesitas Node.js 18+ para `fetch` nativo
2. **Red**: Problemas de conectividad de red
3. **DNS**: Usa IP directa en lugar de hostname

## 📊 Verificar Funcionamiento

### 1. Verificar que el servidor Python esté ejecutándose
```bash
# En la Raspberry Pi con la cámara
ps aux | grep python3 | grep servidor
```

### 2. Verificar conectividad básica
```bash
# Desde tu máquina
curl http://192.168.1.74:8082/status
```

### 3. Verificar que el SDK funcione
```bash
cd api/node/camera-sdk
node test-simple.js
```

## 🎯 Casos de Uso

### Monitoreo Simple
```javascript
// Obtener estado cada 30 segundos
setInterval(async () => {
  try {
    const status = await camera.getStatus();
    console.log(`[${new Date().toLocaleTimeString()}] Entradas: ${status.entradas}, Salidas: ${status.salidas}`);
  } catch (error) {
    console.error('Error:', error.message);
  }
}, 30000);
```

### Sistema de Alertas
```javascript
// Configurar webhook para recibir eventos
await camera.setWebhook('https://tu-servidor.com/webhook');

// El servidor Python enviará automáticamente eventos cuando detecte entradas/salidas
```

### Control Remoto
```javascript
// Cambiar configuración de la cámara
await camera.setConfig({
  confianza: 0.4,        // Más sensible
  fps_captura: 25         // Menos FPS para ahorrar CPU
});

// Resetear contadores
await camera.resetCounters();
```

## 📱 Acceso Web

Una vez configurado, puedes acceder a:
- **Estado**: `http://192.168.1.74:8082/status`
- **Video**: `http://192.168.1.74:8082/video_feed`
- **Interfaz**: `http://192.168.1.74:8082/`

## 🚨 Notas Importantes

1. **IP fija**: Asegúrate de que tu Raspberry Pi tenga IP fija (192.168.1.74)
2. **Puerto abierto**: El puerto 8082 debe estar accesible desde tu red
3. **Servidor activo**: El servidor Python debe estar ejecutándose
4. **Permisos**: Asegúrate de tener permisos para acceder a la cámara

## 🔄 Próximos Pasos

1. **Configura la IP correcta** en los ejemplos
2. **Ejecuta `test-simple.js`** para verificar conectividad básica
3. **Ejecuta `multi-camera-example.js`** para ver todas las funcionalidades
4. **Personaliza** según tus necesidades específicas
5. **Integra** en tu aplicación principal 