const DEFAULT_TIMEOUT = 10000; // 10 segundos
const DEFAULT_RETRIES = 3;
const DEFAULT_RETRY_DELAY = 1000; // 1 segundo

class CameraConnectionError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'CameraConnectionError';
    this.cause = cause;
  }
}

class CameraAPIError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'CameraAPIError';
    this.status = status;
    this.response = response;
  }
}

async function http(method, path, body, config) {
  const { baseUrl, timeout = DEFAULT_TIMEOUT, retries = DEFAULT_RETRIES, retryDelay = DEFAULT_RETRY_DELAY } = config;
  
  const url = `${baseUrl}${path}`;
  const opts = { 
    method, 
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(timeout)
  };
  
  if (body) opts.body = JSON.stringify(body);

  let lastError;
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, opts);
      
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new CameraAPIError(
          `HTTP ${res.status}: ${text}`,
          res.status,
          text
        );
      }
      
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        return await res.json();
      }
      return await res.text();
      
    } catch (error) {
      lastError = error;
      
      // Si es un error de API (HTTP), no reintentar
      if (error instanceof CameraAPIError) {
        throw error;
      }
      
      // Si es el último intento, lanzar el error
      if (attempt === retries) {
        throw new CameraConnectionError(
          `Failed to connect to camera at ${baseUrl} after ${retries + 1} attempts`,
          error
        );
      }
      
      // Esperar antes del siguiente intento
      if (retryDelay > 0) {
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }
}

export class CameraSDK {
  constructor(config = {}) {
    if (typeof config === 'string') {
      // Compatibilidad hacia atrás: config puede ser solo la URL
      config = { baseUrl: config };
    }
    
    this.config = {
      baseUrl: config.baseUrl || 'http://127.0.0.1:8082',
      timeout: config.timeout || DEFAULT_TIMEOUT,
      retries: config.retries || DEFAULT_RETRIES,
      retryDelay: config.retryDelay || DEFAULT_RETRY_DELAY,
      ...config
    };
    
    // Validar configuración
    if (!this.config.baseUrl) {
      throw new Error('baseUrl is required');
    }
    
    // Asegurar que la URL termine con /
    if (!this.config.baseUrl.endsWith('/')) {
      this.config.baseUrl = this.config.baseUrl + '/';
    }
  }

  async getStatus() {
    return http('GET', 'status', null, this.config);
  }

  async resetCounters() {
    return http('POST', 'reset', {}, this.config);
  }

  async setWebhook(webhookUrl) {
    return http('POST', 'config_webhook', { webhook_url: webhookUrl }, this.config);
  }

  async getConfig() {
    return http('GET', 'config', null, this.config);
  }

  async setConfig({ zona_puerta, confianza, fps_captura } = {}) {
    const body = {};
    if (zona_puerta) body.zona_puerta = zona_puerta;
    if (typeof confianza === 'number') body.confianza = confianza;
    if (typeof fps_captura === 'number') body.fps_captura = fps_captura;
    return http('POST', 'config', body, this.config);
  }

  getVideoFeedUrl() {
    return `${this.config.baseUrl}video_feed`;
  }

  // Método para obtener la configuración actual de la instancia
  getConnectionConfig() {
    return { ...this.config };
  }

  // Método para actualizar la configuración de conexión
  updateConnectionConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    
    // Asegurar que la URL termine con /
    if (this.config.baseUrl && !this.config.baseUrl.endsWith('/')) {
      this.config.baseUrl = this.config.baseUrl + '/';
    }
  }
}

export class CameraManager {
  constructor() {
    this.cameras = new Map();
  }

  // Agregar una nueva cámara
  addCamera(id, config) {
    if (this.cameras.has(id)) {
      throw new Error(`Camera with ID '${id}' already exists`);
    }
    
    const camera = new CameraSDK(config);
    this.cameras.set(id, camera);
    return camera;
  }

  // Obtener una cámara existente
  getCamera(id) {
    const camera = this.cameras.get(id);
    if (!camera) {
      throw new Error(`Camera with ID '${id}' not found`);
    }
    return camera;
  }

  // Remover una cámara
  removeCamera(id) {
    if (!this.cameras.has(id)) {
      throw new Error(`Camera with ID '${id}' not found`);
    }
    this.cameras.delete(id);
  }

  // Listar todas las cámaras
  listCameras() {
    return Array.from(this.cameras.keys());
  }

  // Obtener el estado de todas las cámaras
  async getAllStatuses() {
    const results = {};
    const errors = {};

    for (const [id, camera] of this.cameras) {
      try {
        results[id] = await camera.getStatus();
      } catch (error) {
        errors[id] = error.message;
      }
    }

    return { results, errors };
  }

  // Resetear contadores de todas las cámaras
  async resetAllCounters() {
    const results = {};
    const errors = {};

    for (const [id, camera] of this.cameras) {
      try {
        results[id] = await camera.resetCounters();
      } catch (error) {
        errors[id] = error.message;
      }
    }

    return { results, errors };
  }

  // Configurar webhook para todas las cámaras
  async setAllWebhooks(webhookUrl) {
    const results = {};
    const errors = {};

    for (const [id, camera] of this.cameras) {
      try {
        results[id] = await camera.setWebhook(webhookUrl);
      } catch (error) {
        errors[id] = error.message;
      }
    }

    return { results, errors };
  }

  // Obtener configuración de todas las cámaras
  async getAllConfigs() {
    const results = {};
    const errors = {};

    for (const [id, camera] of this.cameras) {
      try {
        results[id] = await camera.getConfig();
      } catch (error) {
        errors[id] = error.message;
      }
    }

    return { results, errors };
  }

  // Verificar conectividad de todas las cámaras
  async checkAllConnections() {
    const results = {};
    const errors = {};

    for (const [id, camera] of this.cameras) {
      try {
        await camera.getStatus();
        results[id] = { connected: true, timestamp: Date.now() };
      } catch (error) {
        errors[id] = { connected: false, error: error.message, timestamp: Date.now() };
      }
    }

    return { results, errors };
  }
}

// Función de conveniencia para crear un SDK simple
export default function createSDK(config) {
  return new CameraSDK(config);
}

// Función de conveniencia para crear un manager
export function createCameraManager() {
  return new CameraManager();
} 