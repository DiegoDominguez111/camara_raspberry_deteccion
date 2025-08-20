from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import json
import base64
import io
from typing import Optional, List
import time
from datetime import datetime
import asyncio
import logging

from face_db import FaceDatabase
from recognizer import FaceRecognizer
from camera_handler import CameraHandler
from utils import draw_face_boxes, frame_to_jpeg, frame_to_base64, format_timestamp, get_all_metrics, log_system_event

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sistema de Reconocimiento Facial", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Variables globales
face_db = None
face_recognizer = None
camera_handler = None
last_recognition_time = 0
recognition_results = []
system_start_time = time.time()

# WebSocket connections
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema al arrancar"""
    global face_db, face_recognizer, camera_handler
    
    try:
        # Inicializar base de datos
        face_db = FaceDatabase()
        logger.info("Base de datos inicializada")
        
        # Inicializar reconocedor facial
        face_recognizer = FaceRecognizer(face_db)
        logger.info("Reconocedor facial inicializado")
        
        # Inicializar cámara
        camera_handler = CameraHandler()
        if camera_handler.start():
            logger.info("Cámara iniciada")
            log_system_event("SUCCESS", "Sistema iniciado correctamente")
        else:
            logger.error("Error al iniciar cámara")
            log_system_event("ERROR", "Error al iniciar cámara")
            
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        log_system_event("ERROR", f"Error en inicialización: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpia recursos al cerrar"""
    global camera_handler
    
    if camera_handler:
        camera_handler.stop()
        logger.info("Cámara detenida")
        log_system_event("INFO", "Sistema detenido")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Página principal del dashboard"""
    try:
        people_count = len(face_db.list_people()) if face_db else 0
        recent_logs = face_db.get_recent_logs(10) if face_db else []
        camera_status = camera_handler.get_camera_status() if camera_handler else {}
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "people_count": people_count,
            "recent_logs": recent_logs,
            "camera_status": camera_status
        })
    except Exception as e:
        logger.error(f"Error en dashboard: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/video_feed")
async def video_feed():
    """Stream de video en tiempo real con reconocimiento facial"""
    async def generate_frames():
        global last_recognition_time, recognition_results
        
        while True:
            try:
                if camera_handler and camera_handler.is_running:
                    frame = camera_handler.get_current_frame()
                    
                    if frame is not None:
                        current_time = time.time()
                        
                        # Procesar reconocimiento cada 100ms
                        if current_time - last_recognition_time > 0.1:
                            face_data = camera_handler.get_face_data()
                            
                            if face_data:
                                frame, faces = face_data
                                if faces:
                                    # Preparar datos para reconocimiento
                                    recognition_input = [(emb, bbox) for emb, bbox, conf in faces]
                                    
                                    # Realizar reconocimiento
                                    recognition_results = face_recognizer.batch_recognize(recognition_input)
                                    
                                    # Dibujar bounding boxes y nombres
                                    frame = draw_face_boxes(frame, recognition_results)
                            
                            last_recognition_time = current_time
                        
                        # Convertir a JPEG
                        jpeg_frame = frame_to_jpeg(frame, quality=80)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                    else:
                        # Frame no disponible
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Esperando cámara...", (200, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        jpeg_frame = frame_to_jpeg(placeholder, quality=80)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                else:
                    # Cámara no disponible
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "CAMARA OFFLINE", (180, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    jpeg_frame = frame_to_jpeg(placeholder, quality=80)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                
                # Control de frecuencia
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error en video feed: {e}")
                await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para actualizaciones en tiempo real"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Enviar métricas cada 2 segundos
            await asyncio.sleep(2)
            
            if websocket in active_connections:
                metrics = get_all_metrics()
                await websocket.send_text(json.dumps(metrics))
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/api/people")
async def get_people():
    """Obtiene lista de personas registradas"""
    try:
        if not face_db:
            raise HTTPException(status_code=500, detail="Base de datos no disponible")
        
        people = face_db.list_people()
        return {"people": people}
    except Exception as e:
        logger.error(f"Error al obtener personas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register-via-camera")
async def register_person_via_camera(name: str = Form(...)):
    """Registra una nueva persona usando la cámara en vivo"""
    try:
        if not face_db or not camera_handler:
            raise HTTPException(status_code=500, detail="Sistema no disponible")
        
        if not camera_handler.is_running:
            raise HTTPException(status_code=500, detail="Cámara no disponible")
        
        # Obtener frame actual de la cámara
        frame = camera_handler.get_current_frame()
        if frame is None:
            raise HTTPException(status_code=500, detail="No se pudo capturar frame de la cámara")
        
        # Detectar rostros en el frame
        faces = camera_handler._detect_faces(frame)
        if not faces:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la cámara")
        
        # Usar el primer rostro detectado
        face_bbox = faces[0]
        x, y, w, h = face_bbox
        
        # Extraer región del rostro
        face_roi = frame[y:y+h, x:x+w]
        
        # Generar embedding "desde la cámara" (simulado)
        face_resized = cv2.resize(face_roi, (128, 128))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = face_gray.astype(np.float32) / 255.0
        
        # Generar embedding usando el método de la cámara
        embedding = camera_handler._simulate_camera_embedding(face_normalized)
        
        # Validar embedding
        if not face_recognizer.validate_embedding(embedding):
            raise HTTPException(status_code=400, detail="Embedding generado no es válido")
        
        # Registrar persona
        success = face_db.add_person(name, embedding)
        if success:
            log_system_event("SUCCESS", f"Persona {name} registrada desde cámara")
            return {"message": f"Persona {name} registrada exitosamente desde la cámara"}
        else:
            raise HTTPException(status_code=400, detail="Nombre ya existe en la base de datos")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al registrar persona desde cámara: {e}")
        log_system_event("ERROR", f"Error al registrar persona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """Obtiene logs de reconocimiento"""
    try:
        if not face_db:
            raise HTTPException(status_code=500, detail="Base de datos no disponible")
        
        logs = face_db.get_recent_logs(limit)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error al obtener logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/people/{person_id}")
async def delete_person(person_id: int):
    """Elimina una persona"""
    try:
        if not face_db:
            raise HTTPException(status_code=500, detail="Base de datos no disponible")
        
        success = face_db.delete_person(person_id)
        if success:
            log_system_event("INFO", f"Persona con ID {person_id} eliminada")
            return {"message": "Persona eliminada exitosamente"}
        else:
            raise HTTPException(status_code=404, detail="Persona no encontrada")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar persona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Obtiene estadísticas del sistema"""
    try:
        stats = {}
        
        if face_recognizer:
            stats['recognition'] = face_recognizer.get_recognition_stats()
        
        if camera_handler:
            stats['camera'] = camera_handler.get_camera_status()
        
        if face_db:
            db_stats = face_db.get_database_stats()
            stats['database'] = db_stats
        
        # Agregar métricas del sistema
        stats['system'] = get_all_metrics()
        
        return stats
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Obtiene métricas detalladas del sistema y la cámara"""
    try:
        metrics = get_all_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error al obtener métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Endpoint de salud del sistema"""
    try:
        # Obtener estado de la cámara
        camera_status = "UNKNOWN"
        last_error = None
        
        if camera_handler:
            camera_info = camera_handler.get_camera_status()
            camera_status = camera_info.get('status', 'UNKNOWN')
            last_error = camera_info.get('last_error')
        
        # Calcular uptime
        uptime = time.time() - system_start_time
        
        # Obtener estadísticas de la base de datos
        db_stats = {}
        if face_db:
            db_stats = face_db.get_database_stats()
        
        status = {
            "status": "healthy" if camera_status in ["RUNNING", "READY"] else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime),
            "camera_status": camera_status,
            "last_camera_error": last_error,
            "database": face_db is not None,
            "recognizer": face_recognizer is not None,
            "database_stats": db_stats
        }
        
        # Loggear estado de salud
        if status["status"] == "healthy":
            log_system_event("INFO", "Health check: Sistema saludable")
        else:
            log_system_event("WARNING", f"Health check: Sistema no saludable - Cámara: {camera_status}")
        
        return status
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        log_system_event("ERROR", f"Error en health check: {e}")
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/camera/restart")
async def restart_camera():
    """Reinicia la cámara"""
    try:
        global camera_handler
        
        if camera_handler:
            # Detener cámara actual
            camera_handler.stop()
            await asyncio.sleep(1)
            
            # Crear nueva instancia
            camera_handler = CameraHandler()
            
            if camera_handler.start():
                log_system_event("SUCCESS", "Cámara reiniciada exitosamente")
                return {"message": "Cámara reiniciada exitosamente"}
            else:
                log_system_event("ERROR", "Error al reiniciar cámara")
                raise HTTPException(status_code=500, detail="Error al reiniciar cámara")
        else:
            raise HTTPException(status_code=500, detail="Cámara no disponible")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al reiniciar cámara: {e}")
        log_system_event("ERROR", f"Error al reiniciar cámara: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/camera/status")
async def get_camera_status():
    """Obtiene el estado de la cámara"""
    try:
        if camera_handler:
            return camera_handler.get_camera_status()
        else:
            return {"status": "NOT_INITIALIZED", "error": "Cámara no inicializada"}
    except Exception as e:
        logger.error(f"Error al obtener estado de cámara: {e}")
        return {"status": "ERROR", "error": str(e)}

@app.post("/api/camera/force-reconnect")
async def force_camera_reconnect():
    """Fuerza una reconexión de la cámara"""
    try:
        if camera_handler:
            success = camera_handler.force_reconnection()
            if success:
                log_system_event("SUCCESS", "Reconexión forzada de cámara exitosa")
                return {"message": "Reconexión forzada exitosa"}
            else:
                log_system_event("ERROR", "Reconexión forzada de cámara fallida")
                return {"message": "Reconexión forzada fallida"}
        else:
            raise HTTPException(status_code=500, detail="Cámara no disponible")
    except Exception as e:
        logger.error(f"Error en reconexión forzada: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 