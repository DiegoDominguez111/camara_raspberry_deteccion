#!/usr/bin/env node

/**
 * Ejemplo de uso del SDK genérico para múltiples cámaras
 * 
 * Este ejemplo demuestra cómo:
 * - Conectar a múltiples cámaras simultáneamente
 * - Gestionar operaciones en lote
 * - Manejar errores de conexión
 * - Monitorear el estado de todas las cámaras
 */

import { CameraManager, CameraSDK } from '../src/index.js';

// Configuración de ejemplo para múltiples cámaras
const CAMERAS_CONFIG = {
  'entrada_principal': {
    baseUrl: 'http://192.168.1.100:8082',
    timeout: 10000,
    retries: 3
  },
  'entrada_secundaria': {
    baseUrl: 'http://192.168.1.101:8082',
    timeout: 10000,
    retries: 3
  },
  'salida_principal': {
    baseUrl: 'http://192.168.1.102:8082',
    timeout: 15000,
    retries: 5
  },
  'estacionamiento': {
    baseUrl: 'http://192.168.1.103:8082',
    timeout: 20000,
    retries: 3
  }
};

// Crear el gestor de cámaras
const manager = new CameraManager();

async function inicializarCamaras() {
  console.log('🚀 Inicializando sistema de múltiples cámaras...\n');
  
  try {
    // Agregar todas las cámaras
    for (const [id, config] of Object.entries(CAMERAS_CONFIG)) {
      try {
        manager.addCamera(id, config);
        console.log(`✅ Cámara '${id}' agregada: ${config.baseUrl}`);
      } catch (error) {
        console.error(`❌ Error agregando cámara '${id}':`, error.message);
      }
    }
    
    console.log(`\n📊 Total de cámaras registradas: ${manager.listCameras().length}`);
    
  } catch (error) {
    console.error('❌ Error en inicialización:', error.message);
    process.exit(1);
  }
}

async function verificarConectividad() {
  console.log('\n🔍 Verificando conectividad de todas las cámaras...');
  
  try {
    const { results, errors } = await manager.checkAllConnections();
    
    console.log('\n📡 Estado de conectividad:');
    for (const [id, status] of Object.entries(results)) {
      console.log(`  ✅ ${id}: Conectada`);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Cámaras con problemas de conexión:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error.error}`);
      }
    }
    
    return Object.keys(errors).length === 0;
    
  } catch (error) {
    console.error('❌ Error verificando conectividad:', error.message);
    return false;
  }
}

async function obtenerEstados() {
  console.log('\n📊 Obteniendo estado de todas las cámaras...');
  
  try {
    const { results, errors } = await manager.getAllStatuses();
    
    console.log('\n📈 Estado actual:');
    for (const [id, status] of Object.entries(results)) {
      console.log(`  📷 ${id}:`);
      console.log(`    Entradas: ${status.entradas}`);
      console.log(`    Salidas: ${status.salidas}`);
      console.log(`    En habitación: ${status.en_habitacion}`);
      console.log(`    FPS real: ${status.fps_camara_real?.toFixed(1) || 'N/A'}`);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Errores al obtener estado:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error obteniendo estados:', error.message);
  }
}

async function configurarWebhooks() {
  console.log('\n🔗 Configurando webhooks para todas las cámaras...');
  
  const webhookUrl = 'https://mi-servidor-central.com/api/eventos-camara';
  
  try {
    const { results, errors } = await manager.setAllWebhooks(webhookUrl);
    
    console.log(`✅ Webhook configurado: ${webhookUrl}`);
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Errores configurando webhooks:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error configurando webhooks:', error.message);
  }
}

async function resetearContadores() {
  console.log('\n🔄 Reseteando contadores de todas las cámaras...');
  
  try {
    const { results, errors } = await manager.resetAllCounters();
    
    console.log('✅ Contadores reseteados en todas las cámaras');
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Errores reseteando contadores:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error reseteando contadores:', error.message);
  }
}

async function monitoreoContinuo() {
  console.log('\n🔄 Iniciando monitoreo continuo (30 segundos)...');
  
  const intervalId = setInterval(async () => {
    try {
      const { results, errors } = await manager.getAllStatuses();
      
      const timestamp = new Date().toLocaleTimeString();
      console.log(`\n[${timestamp}] 📊 Monitoreo:`);
      
      for (const [id, status] of Object.entries(results)) {
        const total = status.entradas + status.salidas;
        console.log(`  📷 ${id}: ${status.entradas}E/${status.salidas}S (Total: ${total})`);
      }
      
      if (Object.keys(errors).length > 0) {
        console.log(`  ⚠️  ${Object.keys(errors).length} cámara(s) con errores`);
      }
      
    } catch (error) {
      console.error(`[${new Date().toLocaleTimeString()}] ❌ Error en monitoreo:`, error.message);
    }
  }, 30000);
  
  // Detener monitoreo después de 2 minutos
  setTimeout(() => {
    clearInterval(intervalId);
    console.log('\n⏹️  Monitoreo detenido');
  }, 120000);
  
  return intervalId;
}

async function ejemploIndividual() {
  console.log('\n🔧 Ejemplo de operaciones individuales...');
  
  try {
    // Obtener una cámara específica
    const camaraEntrada = manager.getCamera('entrada_principal');
    
    // Obtener configuración actual
    const config = await camaraEntrada.getConfig();
    console.log('📋 Configuración de entrada principal:', config);
    
    // Actualizar configuración
    await camaraEntrada.setConfig({
      confianza: 0.35,
      fps_captura: 25
    });
    console.log('✅ Configuración actualizada');
    
    // Obtener nueva configuración
    const nuevaConfig = await camaraEntrada.getConfig();
    console.log('📋 Nueva configuración:', nuevaConfig);
    
  } catch (error) {
    console.error('❌ Error en operaciones individuales:', error.message);
  }
}

async function main() {
  try {
    // Inicializar sistema
    await inicializarCamaras();
    
    // Verificar conectividad
    const todasConectadas = await verificarConectividad();
    
    if (!todasConectadas) {
      console.log('\n⚠️  Algunas cámaras no están conectadas. Continuando con las disponibles...');
    }
    
    // Obtener estados iniciales
    await obtenerEstados();
    
    // Configurar webhooks
    await configurarWebhooks();
    
    // Ejemplo de operaciones individuales
    await ejemploIndividual();
    
    // Resetear contadores
    await resetearContadores();
    
    // Iniciar monitoreo continuo
    await monitoreoContinuo();
    
    console.log('\n🎉 Sistema de múltiples cámaras iniciado correctamente!');
    console.log('💡 Presiona Ctrl+C para detener el monitoreo');
    
  } catch (error) {
    console.error('\n💥 Error fatal en el sistema:', error.message);
    process.exit(1);
  }
}

// Manejar señales de terminación
process.on('SIGINT', () => {
  console.log('\n\n👋 Cerrando sistema de múltiples cámaras...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n\n👋 Cerrando sistema de múltiples cámaras...');
  process.exit(0);
});

// Ejecutar el programa principal
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('💥 Error no manejado:', error);
    process.exit(1);
  });
} 