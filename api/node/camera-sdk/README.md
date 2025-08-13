# Camera SDK - Node.js

SDK genérico para Node.js que permite conectarse y gestionar múltiples cámaras Raspberry Pi con detección de personas simultáneamente.

## Características

- **Múltiples Cámaras**: Conecta a múltiples cámaras simultáneamente
- **Configuración Flexible**: Cada cámara puede tener su propia configuración
- **Manejo de Errores Robusto**: Reintentos automáticos, timeouts configurables
- **Gestión Centralizada**: Clase `CameraManager` para operaciones en lote
- **Compatibilidad**: Mantiene compatibilidad con versiones anteriores

## Instalación

```bash
# Copiar el SDK a tu proyecto
cp -r camera-sdk /tu/proyecto/

# Importar en tu código
import { CameraSDK, CameraManager, createSDK, createCameraManager } from './camera-sdk/src/index.js';
```

## Uso Básico - Una Cámara

### Crear una instancia de cámara

```javascript
// Forma simple (compatibilidad hacia atrás)
const camera = new CameraSDK('http://192.168.1.100:8082');

// Forma completa con configuración
const camera = new CameraSDK({
  baseUrl: 'http://192.168.1.100:8082',
  timeout: 15000,        // 15 segundos
  retries: 5,            // 5 reintentos
  retryDelay: 2000       // 2 segundos entre reintentos
});
```

### Operaciones básicas

```javascript
try {
  // Obtener estado
  const status = await camera.getStatus();
  console.log('Estado:', status);
  
  // Resetear contadores
  await camera.resetCounters();
  
  // Configurar webhook
  await camera.setWebhook('https://mi-servidor.com/webhook');
  
  // Obtener configuración
  const config = await camera.getConfig();
  
  // Actualizar configuración
  await camera.setConfig({
    confianza: 0.4,
    fps_captura: 25
  });
  
  // URL del stream de video
  const videoUrl = camera.getVideoFeedUrl();
  
} catch (error) {
  if (error.name === 'CameraConnectionError') {
    console.error('Error de conexión:', error.message);
  } else if (error.name === 'CameraAPIError') {
    console.error('Error de API:', error.message, 'Status:', error.status);
  } else {
    console.error('Error inesperado:', error);
  }
}
```

## Uso Avanzado - Múltiples Cámaras

### Crear un gestor de cámaras

```javascript
const manager = new CameraManager();

// Agregar múltiples cámaras
manager.addCamera('camara_entrada', {
  baseUrl: 'http://192.168.1.100:8082',
  timeout: 10000
});

manager.addCamera('camara_salida', {
  baseUrl: 'http://192.168.1.101:8082',
  timeout: 10000
});

manager.addCamera('camara_estacionamiento', {
  baseUrl: 'http://192.168.1.102:8082',
  timeout: 15000,
  retries: 5
});
```

### Operaciones en lote

```javascript
try {
  // Obtener estado de todas las cámaras
  const { results, errors } = await manager.getAllStatuses();
  
  console.log('Cámaras conectadas:', results);
  if (Object.keys(errors).length > 0) {
    console.log('Errores:', errors);
  }
  
  // Resetear contadores de todas las cámaras
  const resetResults = await manager.resetAllCounters();
  
  // Configurar webhook para todas las cámaras
  await manager.setAllWebhooks('https://mi-servidor.com/webhook');
  
  // Verificar conectividad de todas las cámaras
  const connections = await manager.checkAllConnections();
  
  // Listar todas las cámaras
  const cameraIds = manager.listCameras();
  console.log('Cámaras registradas:', cameraIds);
  
} catch (error) {
  console.error('Error en operación en lote:', error);
}
```

### Gestión individual de cámaras

```javascript
// Obtener una cámara específica
const camaraEntrada = manager.getCamera('camara_entrada');

// Actualizar configuración de conexión
camaraEntrada.updateConnectionConfig({
  timeout: 20000,
  retries: 10
});

// Obtener configuración actual
const config = camaraEntrada.getConnectionConfig();

// Remover una cámara
manager.removeCamera('camara_estacionamiento');
```

## Configuración

### Opciones de conexión

```javascript
const config = {
  baseUrl: 'http://192.168.1.100:8082',  // URL de la cámara (requerido)
  timeout: 10000,                         // Timeout en ms (default: 10000)
  retries: 3,                             // Número de reintentos (default: 3)
  retryDelay: 1000                        // Delay entre reintentos en ms (default: 1000)
};
```

### Configuración de detección

```javascript
const detectionConfig = {
  zona_puerta: [80, 80, 560, 420],       // Zona de detección [x1, y1, x2, y2]
  confianza: 0.3,                         // Umbral de confianza (0.0 - 1.0)
  fps_captura: 30                          // FPS de captura
};
```

## Manejo de Errores

### Tipos de error

- **CameraConnectionError**: Error de conexión (timeout, red, etc.)
- **CameraAPIError**: Error de la API (HTTP status codes)
- **Error**: Errores generales (validación, etc.)

### Ejemplo de manejo robusto

```javascript
async function operacionSegura(camera) {
  try {
    return await camera.getStatus();
  } catch (error) {
    switch (error.name) {
      case 'CameraConnectionError':
        console.error('Error de conexión, reintentando en 5s...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        return await camera.getStatus(); // Reintento manual
        
      case 'CameraAPIError':
        if (error.status === 404) {
          console.error('Endpoint no encontrado');
        } else if (error.status >= 500) {
          console.error('Error del servidor, reintentando...');
          await new Promise(resolve => setTimeout(resolve, 2000));
          return await camera.getStatus();
        }
        break;
        
      default:
        console.error('Error inesperado:', error);
    }
    throw error;
  }
}
```

## Ejemplos de Uso

### Monitoreo de múltiples cámaras

```javascript
import { CameraManager } from './camera-sdk/src/index.js';

const manager = new CameraManager();

// Configurar cámaras
manager.addCamera('entrada_principal', { baseUrl: 'http://192.168.1.100:8082' });
manager.addCamera('entrada_secundaria', { baseUrl: 'http://192.168.1.101:8082' });
manager.addCamera('salida', { baseUrl: 'http://192.168.1.102:8082' });

// Función de monitoreo
async function monitorearCamaras() {
  try {
    const { results, errors } = await manager.getAllStatuses();
    
    for (const [id, status] of Object.entries(results)) {
      console.log(`${id}: ${status.entradas} entradas, ${status.salidas} salidas`);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('Cámaras con errores:', errors);
    }
    
  } catch (error) {
    console.error('Error en monitoreo:', error);
  }
}

// Monitorear cada 30 segundos
setInterval(monitorearCamaras, 30000);
```

### Sistema de alertas con webhooks

```javascript
// Configurar webhook para todas las cámaras
await manager.setAllWebhooks('https://mi-servidor.com/alertas');

// El servidor Python enviará automáticamente eventos cuando se detecten entradas/salidas
```

## Compatibilidad

- **Node.js**: 18.0.0 o superior (requiere `fetch` nativo)
- **ESM**: Solo módulos ES6 (import/export)
- **Navegadores**: No compatible (requiere Node.js)

## Troubleshooting

### Error: "fetch failed"
- Verificar que la URL de la cámara sea correcta
- Usar IP en lugar de hostname si hay problemas de DNS
- Verificar que el puerto esté abierto y accesible

### Error: "ECONNREFUSED"
- La cámara no está ejecutando el servidor Python
- Firewall bloqueando la conexión
- Puerto incorrecto en la configuración

### Error: "CameraAPIError: HTTP 404"
- El endpoint no existe en esa versión del servidor
- Verificar que el servidor Python esté actualizado

### Timeouts frecuentes
- Aumentar el valor de `timeout` en la configuración
- Verificar la latencia de red
- Considerar usar IP local en lugar de hostname 