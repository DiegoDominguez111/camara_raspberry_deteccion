#!/usr/bin/env node

/**
 * Ejemplo específico para tu cámara en 192.168.1.74
 * 
 * Este ejemplo está configurado para conectarse a tu cámara específica
 * y demuestra todas las funcionalidades del SDK.
 */

import { CameraSDK, CameraManager } from '../src/index.js';

// Configuración específica para tu cámara
const MI_CAMARA_CONFIG = {
  'mi_camara': {
    baseUrl: 'http://192.168.1.74:8082',  // ← Tu IP específica
    timeout: 10000,
    retries: 3,
    retryDelay: 1000
  }
};

// Crear el gestor de cámaras
const manager = new CameraManager();

async function inicializarMiCamara() {
  console.log('🚀 Inicializando tu cámara en 192.168.1.74...\n');
  
  try {
    // Agregar tu cámara
    manager.addCamera('mi_camara', MI_CAMARA_CONFIG.mi_camara);
    console.log('✅ Tu cámara agregada: http://192.168.1.74:8082');
    console.log(`📊 Total de cámaras registradas: ${manager.listCameras().length}`);
    
  } catch (error) {
    console.error('❌ Error agregando tu cámara:', error.message);
    throw error;
  }
}

async function verificarConectividad() {
  console.log('\n🔍 Verificando conectividad de tu cámara...');
  
  try {
    const { results, errors } = await manager.checkAllConnections();
    
    console.log('\n📡 Estado de conectividad:');
    for (const [id, status] of Object.entries(results)) {
      console.log(`  ✅ ${id}: Conectada`);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Problemas de conexión:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error.error}`);
      }
      return false;
    }
    
    return true;
    
  } catch (error) {
    console.error('❌ Error verificando conectividad:', error.message);
    return false;
  }
}

async function obtenerEstado() {
  console.log('\n📊 Obteniendo estado de tu cámara...');
  
  try {
    const { results, errors } = await manager.getAllStatuses();
    
    console.log('\n📈 Estado actual:');
    for (const [id, status] of Object.entries(results)) {
      console.log(`  📷 ${id}:`);
      console.log(`    Entradas: ${status.entradas}`);
      console.log(`    Salidas: ${status.salidas}`);
      console.log(`    En habitación: ${status.en_habitacion}`);
      console.log(`    FPS real: ${status.fps_camara_real?.toFixed(1) || 'N/A'}`);
      console.log(`    Webhook: ${status.webhook_url || 'No configurado'}`);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Errores al obtener estado:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error obteniendo estado:', error.message);
  }
}

async function obtenerConfiguracion() {
  console.log('\n⚙️  Obteniendo configuración de tu cámara...');
  
  try {
    const { results, errors } = await manager.getAllConfigs();
    
    console.log('\n🔧 Configuración actual:');
    for (const [id, config] of Object.entries(results)) {
      console.log(`  📷 ${id}:`);
      console.log(`    Zona puerta: [${config.zona_puerta.join(', ')}]`);
      console.log(`    Confianza: ${config.confianza}`);
      console.log(`    FPS captura: ${config.fps_captura}`);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('\n⚠️  Errores al obtener configuración:');
      for (const [id, error] of Object.entries(errors)) {
        console.log(`  ❌ ${id}: ${error}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error obteniendo configuración:', error.message);
  }
}

async function configurarWebhook() {
  console.log('\n🔗 Configurando webhook para tu cámara...');
  
  // Cambia esta URL por tu servidor real
  const webhookUrl = 'https://tu-servidor.com/api/eventos-camara';
  
  try {
    const { results, errors } = await manager.setAllWebhooks(webhookUrl);
    
    console.log(`✅ Webhook configurado: ${webhookUrl}`);
    console.log('💡 Tu cámara enviará eventos automáticamente cuando detecte entradas/salidas');
    
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

async function cambiarConfiguracion() {
  console.log('\n🔧 Cambiando configuración de tu cámara...');
  
  try {
    const camara = manager.getCamera('mi_camara');
    
    // Cambiar configuración para mejor detección
    await camara.setConfig({
      confianza: 0.35,        // Más sensible
      fps_captura: 25          // Balance entre fluidez y CPU
    });
    
    console.log('✅ Configuración actualizada');
    console.log('  - Confianza: 0.35 (más sensible)');
    console.log('  - FPS: 25 (balanceado)');
    
    // Obtener nueva configuración
    const nuevaConfig = await camara.getConfig();
    console.log('\n📋 Nueva configuración aplicada:', nuevaConfig);
    
  } catch (error) {
    console.error('❌ Error cambiando configuración:', error.message);
  }
}

async function resetearContadores() {
  console.log('\n🔄 Reseteando contadores de tu cámara...');
  
  try {
    const { results, errors } = await manager.resetAllCounters();
    
    console.log('✅ Contadores reseteados');
    
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
  console.log('💡 Presiona Ctrl+C para detener');
  
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
  
  return intervalId;
}

async function mostrarURLs() {
  console.log('\n🌐 URLs de acceso a tu cámara:');
  console.log('  📊 Estado: http://192.168.1.74:8082/status');
  console.log('  🎥 Video: http://192.168.1.74:8082/video_feed');
  console.log('  🖥️  Interfaz: http://192.168.1.74:8082/');
  console.log('  ⚙️  Config: http://192.168.1.74:8082/config');
}

async function main() {
  try {
    console.log('🎯 SDK de Cámara - Ejemplo para tu cámara en 192.168.1.74\n');
    
    // Inicializar tu cámara
    await inicializarMiCamara();
    
    // Verificar conectividad
    const conectada = await verificarConectividad();
    
    if (!conectada) {
      console.log('\n❌ Tu cámara no está conectada. Verifica:');
      console.log('  1. Que la IP 192.168.1.74 sea correcta');
      console.log('  2. Que el servidor Python esté ejecutándose');
      console.log('  3. Que el puerto 8082 esté abierto');
      console.log('  4. Que puedas hacer ping a 192.168.1.74');
      process.exit(1);
    }
    
    // Obtener estado inicial
    await obtenerEstado();
    
    // Obtener configuración actual
    await obtenerConfiguracion();
    
    // Mostrar URLs de acceso
    await mostrarURLs();
    
    // Configurar webhook
    await configurarWebhook();
    
    // Cambiar configuración
    await cambiarConfiguracion();
    
    // Resetear contadores
    await resetearContadores();
    
    // Iniciar monitoreo continuo
    const monitor = await monitoreoContinuo();
    
    console.log('\n🎉 ¡Tu cámara está funcionando correctamente!');
    console.log('📱 Puedes acceder a la interfaz web en: http://192.168.1.74:8082/');
    
    // Detener monitoreo después de 5 minutos
    setTimeout(() => {
      clearInterval(monitor);
      console.log('\n⏹️  Monitoreo detenido automáticamente');
      process.exit(0);
    }, 300000);
    
  } catch (error) {
    console.error('\n💥 Error fatal:', error.message);
    process.exit(1);
  }
}

// Manejar señales de terminación
process.on('SIGINT', () => {
  console.log('\n\n👋 Cerrando ejemplo de tu cámara...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n\n👋 Cerrando ejemplo de tu cámara...');
  process.exit(0);
});

// Ejecutar el programa principal
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('💥 Error no manejado:', error);
    process.exit(1);
  });
} 