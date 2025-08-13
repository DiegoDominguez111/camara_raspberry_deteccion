#!/usr/bin/env node

/**
 * Prueba simple del SDK genérico
 * Conecta a la cámara local y prueba las operaciones básicas
 */

import { CameraSDK, CameraManager } from './src/index.js';

async function testSingleCamera() {
  console.log('🧪 Probando cámara individual...\n');
  
  try {
    // Crear instancia de cámara local
    const camera = new CameraSDK({
      baseUrl: 'http://127.0.0.1:8082',
      timeout: 5000,
      retries: 2
    });
    
    console.log('✅ Cámara creada');
    console.log('📋 Configuración:', camera.getConnectionConfig());
    
    // Probar getStatus
    console.log('\n📊 Obteniendo estado...');
    const status = await camera.getStatus();
    console.log('✅ Estado obtenido:', status);
    
    // Probar getConfig
    console.log('\n⚙️  Obteniendo configuración...');
    const config = await camera.getConfig();
    console.log('✅ Configuración obtenida:', config);
    
    // Probar getVideoFeedUrl
    console.log('\n🎥 URL del stream de video:');
    console.log('✅', camera.getVideoFeedUrl());
    
    return true;
    
  } catch (error) {
    console.error('❌ Error en prueba de cámara individual:', error.message);
    if (error.cause) {
      console.error('  Causa:', error.cause.message);
    }
    return false;
  }
}

async function testCameraManager() {
  console.log('\n\n🧪 Probando gestor de cámaras...\n');
  
  try {
    const manager = new CameraManager();
    
    // Agregar cámara local
    manager.addCamera('local', {
      baseUrl: 'http://127.0.0.1:8082',
      timeout: 5000
    });
    
    console.log('✅ Cámara agregada al gestor');
    console.log('📋 Cámaras registradas:', manager.listCameras());
    
    // Probar operaciones en lote
    console.log('\n📊 Obteniendo estado de todas las cámaras...');
    const { results, errors } = await manager.getAllStatuses();
    
    if (Object.keys(results).length > 0) {
      console.log('✅ Estados obtenidos:', results);
    }
    
    if (Object.keys(errors).length > 0) {
      console.log('⚠️  Errores:', errors);
    }
    
    // Probar verificación de conectividad
    console.log('\n🔍 Verificando conectividad...');
    const connections = await manager.checkAllConnections();
    console.log('✅ Estado de conectividad:', connections);
    
    return true;
    
  } catch (error) {
    console.error('❌ Error en prueba del gestor:', error.message);
    return false;
  }
}

async function main() {
  console.log('🚀 Iniciando pruebas del SDK genérico...\n');
  
  try {
    // Probar cámara individual
    const singleCameraOk = await testSingleCamera();
    
    // Probar gestor de cámaras
    const managerOk = await testCameraManager();
    
    console.log('\n\n📊 Resumen de pruebas:');
    console.log(`  📷 Cámara individual: ${singleCameraOk ? '✅ PASÓ' : '❌ FALLÓ'}`);
    console.log(`  🎛️  Gestor de cámaras: ${managerOk ? '✅ PASÓ' : '❌ FALLÓ'}`);
    
    if (singleCameraOk && managerOk) {
      console.log('\n🎉 ¡Todas las pruebas pasaron! El SDK está funcionando correctamente.');
    } else {
      console.log('\n⚠️  Algunas pruebas fallaron. Revisa los errores arriba.');
      process.exit(1);
    }
    
  } catch (error) {
    console.error('\n💥 Error fatal en las pruebas:', error.message);
    process.exit(1);
  }
}

// Ejecutar pruebas
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('💥 Error no manejado:', error);
    process.exit(1);
  });
} 