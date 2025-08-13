#!/usr/bin/env node
/**
 * Servicio de Configuración y Telemetría para Cámara AI
 * Raspberry Pi 5 + Cámara IMX500
 */

const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const cors = require('cors');

class ServicioCamara {
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.wss = new WebSocket.Server({ server: this.server });
        
        // Configuración
        this.config = this.cargarConfiguracion();
        this.puerto = process.env.PUERTO || 8080;
        
        // Estado del sistema
        this.estado = {
            camara_activa: false,
            detector_activo: false,
            ultima_actualizacion: null,
            errores: [],
            metricas: {}
        };
        
        // Procesos
        this.procesoDetector = null;
        this.procesoCamara = null;
        
        // Métricas en tiempo real
        this.metricasTiempoReal = {
            fps_captura: 0,
            fps_inferencia: 0,
            fps_post_proceso: 0,
            latencia_total: 0,
            cpu: 0,
            memoria: 0,
            temperatura: 0,
            contador_entradas: 0,
            contador_salidas: 0,
            personas_en_habitacion: 0
        };
        
        // Configurar middleware
        this.configurarMiddleware();
        
        // Configurar rutas
        this.configurarRutas();
        
        // Configurar WebSocket
        this.configurarWebSocket();
        
        // Iniciar monitoreo del sistema
        this.iniciarMonitoreoSistema();
        
        // Configurar manejo de señales
        this.configurarManejoSenales();
    }
    
    cargarConfiguracion() {
        try {
            const configPath = path.join(__dirname, 'config_detector.json');
            if (fs.existsSync(configPath)) {
                const configData = fs.readFileSync(configPath, 'utf8');
                return JSON.parse(configData);
            }
        } catch (error) {
            console.error('❌ Error cargando configuración:', error.message);
        }
        
        // Configuración por defecto
        return {
            resolucion: [640, 480],
            fps_objetivo: 30,
            confianza_minima: 0.4,
            nms_iou: 0.45,
            area_minima: 2000,
            roi_puerta: [80, 80, 560, 420],
            linea_cruce: 320,
            ancho_banda_cruce: 3,
            debounce_ms: 300,
            track_lost_ms: 700,
            exposure_us: 4000,
            gain: 1.0,
            ae_lock: true,
            awb_lock: true,
            denoise: false
        };
    }
    
    configurarMiddleware() {
        // CORS para cliente externo
        this.app.use(cors({
            origin: '*',
            methods: ['GET', 'POST', 'PUT', 'DELETE'],
            allowedHeaders: ['Content-Type', 'Authorization']
        }));
        
        // Parsear JSON
        this.app.use(express.json({ limit: '10mb' }));
        
        // Logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
            next();
        });
    }
    
    configurarRutas() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'ok',
                timestamp: new Date().toISOString(),
                camara_activa: this.estado.camara_activa,
                detector_activo: this.estado.detector_activo,
                version: '1.0.0'
            });
        });
        
        // Métricas del sistema
        this.app.get('/metrics', (req, res) => {
            res.json({
                timestamp: new Date().toISOString(),
                ...this.metricasTiempoReal,
                estado: this.estado,
                configuracion: this.config
            });
        });
        
        // Contadores
        this.app.get('/counts', (req, res) => {
            res.json({
                timestamp: new Date().toISOString(),
                contador_entradas: this.metricasTiempoReal.contador_entradas,
                contador_salidas: this.metricasTiempoReal.contador_salidas,
                personas_en_habitacion: this.metricasTiempoReal.personas_en_habitacion,
                ultima_actualizacion: this.estado.ultima_actualizacion
            });
        });
        
        // Configuración actual
        this.app.get('/config', (req, res) => {
            res.json({
                timestamp: new Date().toISOString(),
                configuracion: this.config
            });
        });
        
        // Actualizar configuración
        this.app.post('/config', (req, res) => {
            try {
                const nuevaConfig = req.body;
                
                // Validar configuración
                if (this.validarConfiguracion(nuevaConfig)) {
                    this.config = { ...this.config, ...nuevaConfig };
                    
                    // Guardar en archivo
                    this.guardarConfiguracion();
                    
                    // Reiniciar detector si está activo
                    if (this.estado.detector_activo) {
                        this.reiniciarDetector();
                    }
                    
                    res.json({
                        status: 'ok',
                        message: 'Configuración actualizada',
                        configuracion: this.config
                    });
                } else {
                    res.status(400).json({
                        status: 'error',
                        message: 'Configuración inválida'
                    });
                }
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            }
        });
        
        // Iniciar detector
        this.app.post('/detector/start', (req, res) => {
            try {
                this.iniciarDetector();
                res.json({
                    status: 'ok',
                    message: 'Detector iniciado'
                });
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            }
        });
        
        // Detener detector
        this.app.post('/detector/stop', (req, res) => {
            try {
                this.detenerDetector();
                res.json({
                    status: 'ok',
                    message: 'Detector detenido'
                });
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            }
        });
        
        // Estado del detector
        this.app.get('/detector/status', (req, res) => {
            res.json({
                timestamp: new Date().toISOString(),
                detector_activo: this.estado.detector_activo,
                camara_activa: this.estado.camara_activa,
                proceso_id: this.procesoDetector ? this.procesoDetector.pid : null
            });
        });
        
        // Logs del detector
        this.app.get('/detector/logs', (req, res) => {
            const logs = this.obtenerLogsDetector();
            res.json({
                timestamp: new Date().toISOString(),
                logs: logs
            });
        });
        
        // Reiniciar detector
        this.app.post('/detector/restart', (req, res) => {
            try {
                this.reiniciarDetector();
                res.json({
                    status: 'ok',
                    message: 'Detector reiniciado'
                });
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            }
        });
        
        // Configurar cámara
        this.app.post('/camera/config', (req, res) => {
            try {
                const configCamara = req.body;
                this.configurarCamara(configCamara);
                res.json({
                    status: 'ok',
                    message: 'Cámara configurada'
                });
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            }
        });
        
        // Estado de la cámara
        this.app.get('/camera/status', (req, res) => {
            res.json({
                timestamp: new Date().toISOString(),
                camara_activa: this.estado.camara_activa,
                configuracion: this.config,
                metricas: {
                    fps_captura: this.metricasTiempoReal.fps_captura,
                    latencia_total: this.metricasTiempoReal.latencia_total
                }
            });
        });
        
        // Capturar imagen de prueba
        this.app.get('/camera/test-image', (req, res) => {
            try {
                this.capturarImagenPrueba(res);
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            }
        });
        
        // Ruta raíz con información del sistema
        this.app.get('/', (req, res) => {
            res.json({
                servicio: 'Cámara AI - Raspberry Pi 5',
                version: '1.0.0',
                timestamp: new Date().toISOString(),
                endpoints: {
                    health: '/health',
                    metrics: '/metrics',
                    counts: '/counts',
                    config: '/config',
                    detector: '/detector/*',
                    camera: '/camera/*'
                },
                estado: {
                    camara_activa: this.estado.camara_activa,
                    detector_activo: this.estado.detector_activo
                }
            });
        });
    }
    
    configurarWebSocket() {
        this.wss.on('connection', (ws) => {
            console.log('🔌 Cliente WebSocket conectado');
            
            // Enviar estado inicial
            ws.send(JSON.stringify({
                tipo: 'estado_inicial',
                data: {
                    timestamp: new Date().toISOString(),
                    estado: this.estado,
                    metricas: this.metricasTiempoReal
                }
            }));
            
            ws.on('close', () => {
                console.log('🔌 Cliente WebSocket desconectado');
            });
        });
        
        // Broadcast de métricas cada segundo
        setInterval(() => {
            if (this.wss.clients.size > 0) {
                const mensaje = JSON.stringify({
                    tipo: 'metricas_tiempo_real',
                    data: {
                        timestamp: new Date().toISOString(),
                        ...this.metricasTiempoReal
                    }
                });
                
                this.wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(mensaje);
                    }
                });
            }
        }, 1000);
    }
    
    iniciarMonitoreoSistema() {
        // Monitorear métricas del sistema cada 2 segundos
        setInterval(() => {
            this.actualizarMetricasSistema();
        }, 2000);
        
        // Verificar salud del detector cada 10 segundos
        setInterval(() => {
            this.verificarSaludDetector();
        }, 10000);
    }
    
    actualizarMetricasSistema() {
        try {
            // CPU y memoria
            const cpuUsage = require('os').loadavg()[0] * 100;
            const memoriaTotal = require('os').totalmem();
            const memoriaLibre = require('os').freemem();
            const memoriaUso = ((memoriaTotal - memoriaLibre) / memoriaTotal) * 100;
            
            // Temperatura
            let temperatura = 0;
            try {
                const tempData = fs.readFileSync('/sys/class/thermal/thermal_zone0/temp', 'utf8');
                temperatura = parseFloat(tempData) / 1000;
            } catch (error) {
                // Temperatura no disponible
            }
            
            this.metricasTiempoReal.cpu = Math.round(cpuUsage);
            this.metricasTiempoReal.memoria = Math.round(memoriaUso);
            this.metricasTiempoReal.temperatura = Math.round(temperatura * 10) / 10;
            
        } catch (error) {
            console.error('❌ Error actualizando métricas del sistema:', error.message);
        }
    }
    
    verificarSaludDetector() {
        if (this.estado.detector_activo && this.procesoDetector) {
            // Verificar si el proceso sigue activo
            if (this.procesoDetector.killed) {
                console.log('⚠️ Proceso del detector terminado inesperadamente');
                this.estado.detector_activo = false;
                this.estado.errores.push({
                    timestamp: new Date().toISOString(),
                    error: 'Proceso del detector terminado inesperadamente'
                });
                
                // Intentar reiniciar automáticamente
                setTimeout(() => {
                    this.iniciarDetector();
                }, 5000);
            }
        }
    }
    
    iniciarDetector() {
        if (this.estado.detector_activo) {
            console.log('⚠️ El detector ya está activo');
            return;
        }
        
        try {
            console.log('🚀 Iniciando detector...');
            
            // Iniciar proceso del detector Python
            this.procesoDetector = spawn('python3', [
                'detector_entrada_salida_v2.py',
                '--config', 'config_detector.json'
            ], {
                stdio: ['pipe', 'pipe', 'pipe']
            });
            
            // Configurar manejo de salida
            this.procesoDetector.stdout.on('data', (data) => {
                const output = data.toString();
                console.log('📊 Detector:', output.trim());
                
                // Parsear métricas si es posible
                this.parsearMetricasDetector(output);
            });
            
            this.procesoDetector.stderr.on('data', (data) => {
                console.error('❌ Error detector:', data.toString().trim());
            });
            
            this.procesoDetector.on('close', (code) => {
                console.log(`🔚 Proceso del detector terminado con código: ${code}`);
                this.estado.detector_activo = false;
                this.procesoDetector = null;
            });
            
            this.procesoDetector.on('error', (error) => {
                console.error('❌ Error iniciando detector:', error);
                this.estado.detector_activo = false;
                this.procesoDetector = null;
            });
            
            this.estado.detector_activo = true;
            this.estado.ultima_actualizacion = new Date().toISOString();
            
            console.log('✅ Detector iniciado correctamente');
            
        } catch (error) {
            console.error('❌ Error iniciando detector:', error);
            throw error;
        }
    }
    
    detenerDetector() {
        if (!this.estado.detector_activo || !this.procesoDetector) {
            console.log('⚠️ El detector no está activo');
            return;
        }
        
        try {
            console.log('⏹️ Deteniendo detector...');
            
            this.procesoDetector.kill('SIGTERM');
            
            // Esperar un poco y forzar si es necesario
            setTimeout(() => {
                if (this.procesoDetector && !this.procesoDetector.killed) {
                    this.procesoDetector.kill('SIGKILL');
                }
            }, 5000);
            
            this.estado.detector_activo = false;
            this.procesoDetector = null;
            
            console.log('✅ Detector detenido correctamente');
            
        } catch (error) {
            console.error('❌ Error deteniendo detector:', error);
            throw error;
        }
    }
    
    reiniciarDetector() {
        console.log('🔄 Reiniciando detector...');
        this.detenerDetector();
        
        setTimeout(() => {
            this.iniciarDetector();
        }, 2000);
    }
    
    parsearMetricasDetector(output) {
        try {
            // Buscar patrones en la salida del detector
            const lines = output.split('\n');
            
            for (const line of lines) {
                // FPS
                const fpsMatch = line.match(/FPS.*?(\d+\.?\d*)/);
                if (fpsMatch) {
                    this.metricasTiempoReal.fps_post_proceso = parseFloat(fpsMatch[1]);
                }
                
                // Contadores
                const entradaMatch = line.match(/Total entradas.*?(\d+)/);
                if (entradaMatch) {
                    this.metricasTiempoReal.contador_entradas = parseInt(entradaMatch[1]);
                }
                
                const salidaMatch = line.match(/Total salidas.*?(\d+)/);
                if (salidaMatch) {
                    this.metricasTiempoReal.contador_salidas = parseInt(salidaMatch[1]);
                }
                
                // Personas en habitación
                const personasMatch = line.match(/Personas en habitación.*?(\d+)/);
                if (personasMatch) {
                    this.metricasTiempoReal.personas_en_habitacion = parseInt(personasMatch[1]);
                }
            }
            
            this.estado.ultima_actualizacion = new Date().toISOString();
            
        } catch (error) {
            console.error('❌ Error parseando métricas del detector:', error.message);
        }
    }
    
    configurarCamara(configCamara) {
        try {
            console.log('📷 Configurando cámara...');
            
            // Actualizar configuración
            this.config = { ...this.config, ...configCamara };
            
            // Guardar configuración
            this.guardarConfiguracion();
            
            // Si el detector está activo, reiniciarlo para aplicar cambios
            if (this.estado.detector_activo) {
                this.reiniciarDetector();
            }
            
            console.log('✅ Cámara configurada correctamente');
            
        } catch (error) {
            console.error('❌ Error configurando cámara:', error);
            throw error;
        }
    }
    
    capturarImagenPrueba(res) {
        try {
            console.log('📸 Capturando imagen de prueba...');
            
            const proceso = spawn('rpicam-still', [
                '--width', '640',
                '--height', '480',
                '--output', '-',
                '--nopreview',
                '--timeout', '1000',
                '--immediate'
            ]);
            
            let imagenData = Buffer.alloc(0);
            
            proceso.stdout.on('data', (data) => {
                imagenData = Buffer.concat([imagenData, data]);
            });
            
            proceso.on('close', (code) => {
                if (code === 0 && imagenData.length > 0) {
                    res.set('Content-Type', 'image/jpeg');
                    res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
                    res.send(imagenData);
                } else {
                    res.status(500).json({
                        status: 'error',
                        message: 'Error capturando imagen'
                    });
                }
            });
            
            proceso.on('error', (error) => {
                res.status(500).json({
                    status: 'error',
                    message: error.message
                });
            });
            
        } catch (error) {
            res.status(500).json({
                status: 'error',
                message: error.message
            });
        }
    }
    
    validarConfiguracion(config) {
        // Validaciones básicas
        if (config.fps_objetivo && (config.fps_objetivo < 10 || config.fps_objetivo > 60)) {
            return false;
        }
        
        if (config.confianza_minima && (config.confianza_minima < 0.1 || config.confianza_minima > 1.0)) {
            return false;
        }
        
        if (config.exposure_us && (config.exposure_us < 100 || config.exposure_us > 100000)) {
            return false;
        }
        
        return true;
    }
    
    guardarConfiguracion() {
        try {
            const configPath = path.join(__dirname, 'config_detector.json');
            fs.writeFileSync(configPath, JSON.stringify(this.config, null, 2));
            console.log('💾 Configuración guardada');
        } catch (error) {
            console.error('❌ Error guardando configuración:', error.message);
        }
    }
    
    obtenerLogsDetector() {
        // Simular logs del detector (en producción esto vendría de un archivo)
        return [
            {
                timestamp: new Date().toISOString(),
                nivel: 'info',
                mensaje: 'Detector iniciado correctamente'
            },
            {
                timestamp: new Date().toISOString(),
                nivel: 'info',
                mensaje: `FPS actual: ${this.metricasTiempoReal.fps_post_proceso}`
            }
        ];
    }
    
    configurarManejoSenales() {
        process.on('SIGINT', () => {
            console.log('\n⏹️ Señal SIGINT recibida, cerrando servicio...');
            this.cleanup();
            process.exit(0);
        });
        
        process.on('SIGTERM', () => {
            console.log('⏹️ Señal SIGTERM recibida, cerrando servicio...');
            this.cleanup();
            process.exit(0);
        });
    }
    
    cleanup() {
        console.log('🧹 Limpiando recursos...');
        
        // Detener detector
        if (this.estado.detector_activo) {
            this.detenerDetector();
        }
        
        // Cerrar WebSocket
        this.wss.close();
        
        // Cerrar servidor HTTP
        this.server.close();
        
        console.log('✅ Recursos limpiados');
    }
    
    iniciar() {
        this.server.listen(this.puerto, () => {
            console.log('🚀 Servicio de Cámara AI iniciado');
            console.log(`🌐 Puerto: ${this.puerto}`);
            console.log(`🔗 URL: http://localhost:${this.puerto}`);
            console.log(`📊 Health: http://localhost:${this.puerto}/health`);
            console.log(`📈 Métricas: http://localhost:${this.puerto}/metrics`);
            console.log('=' * 50);
        });
    }
}

// Crear e iniciar servicio
const servicio = new ServicioCamara();
servicio.iniciar(); 