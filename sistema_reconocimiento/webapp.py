from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File
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

# Importar módulos del sistema
from face_db import FaceDatabase
from recognizer import FaceRecognizer
from camera_handler import CameraHandler
from utils import draw_face_boxes, frame_to_jpeg, frame_to_base64, format_timestamp

app = FastAPI(title="Sistema de Reconocimiento Facial", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Variables globales del sistema
face_db = None
face_recognizer = None
camera_handler = None
current_frame = None
last_recognition_time = 0
recognition_results = []

@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema al arrancar"""
    global face_db, face_recognizer, camera_handler
    
    try:
        # Inicializar base de datos
        face_db = FaceDatabase()
        print("Base de datos inicializada")
        
        # Inicializar reconocedor
        face_recognizer = FaceRecognizer(face_db)
        print("Reconocedor facial inicializado")
        
        # Inicializar cámara
        camera_handler = CameraHandler()
        if camera_handler.start():
            print("Cámara iniciada")
        else:
            print("Error al iniciar cámara")
            
    except Exception as e:
        print(f"Error en inicialización: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpia recursos al cerrar"""
    global camera_handler
    
    if camera_handler:
        camera_handler.stop()
        print("Cámara detenida")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Página principal del dashboard"""
    try:
        # Obtener estadísticas
        people_count = len(face_db.list_people()) if face_db else 0
        recent_logs = face_db.get_recent_logs(10) if face_db else []
        
        # Obtener información de la cámara
        camera_info = camera_handler.get_camera_info() if camera_handler else {}
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "people_count": people_count,
            "recent_logs": recent_logs,
            "camera_info": camera_info
        })
    except Exception as e:
        print(f"Error en dashboard: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/video_feed")
async def video_feed():
    """Stream de video en tiempo real con reconocimiento facial"""
    async def generate_frames():
        global current_frame, last_recognition_time, recognition_results
        
        while True:
            try:
                if camera_handler and camera_handler.is_running:
                    # Obtener frame actual
                    frame = camera_handler.get_current_frame()
                    
                    if frame is not None:
                        # Procesar reconocimiento cada 100ms
                        current_time = time.time()
                        if current_time - last_recognition_time > 0.1:
                            face_data = camera_handler.get_face_data()
                            
                            if face_data:
                                frame, faces = face_data
                                
                                if faces:
                                    # Preparar datos para reconocimiento
                                    recognition_input = [(emb, bbox) for emb, bbox, conf in faces]
                                    
                                    # Realizar reconocimiento
                                    recognition_results = face_recognizer.batch_recognize(recognition_input)
                                    
                                    # Dibujar bounding boxes
                                    frame = draw_face_boxes(frame, recognition_results)
                            
                            last_recognition_time = current_time
                        
                        # Convertir frame a JPEG
                        jpeg_frame = frame_to_jpeg(frame, quality=80)
                        
                        # Generar respuesta MJPEG
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                    else:
                        # Frame vacío, enviar imagen de placeholder
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Esperando cámara...", (200, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        jpeg_frame = frame_to_jpeg(placeholder, quality=80)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                else:
                    # Cámara no disponible
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Cámara no disponible", (180, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    jpeg_frame = frame_to_jpeg(placeholder, quality=80)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                
                # Control de FPS
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Error en video feed: {e}")
                await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/people")
async def get_people():
    """Obtiene lista de personas registradas"""
    try:
        if not face_db:
            raise HTTPException(status_code=500, detail="Base de datos no disponible")
        
        people = face_db.list_people()
        return {"people": people}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
async def register_person(name: str = Form(...), embedding_data: str = Form(...)):
    """Registra una nueva persona"""
    try:
        if not face_db:
            raise HTTPException(status_code=500, detail="Base de datos no disponible")
        
        # Decodificar embedding desde base64
        try:
            embedding_bytes = base64.b64decode(embedding_data)
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Formato de embedding inválido")
        
        # Validar embedding
        if embedding.size != 128:
            raise HTTPException(status_code=400, detail="Embedding debe tener 128 dimensiones")
        
        # Registrar persona
        success = face_db.add_person(name, embedding)
        
        if success:
            return {"message": f"Persona {name} registrada exitosamente"}
        else:
            raise HTTPException(status_code=400, detail="Nombre ya existe en la base de datos")
            
    except HTTPException:
        raise
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/people/{person_id}")
async def delete_person(person_id: int):
    """Elimina una persona"""
    try:
        if not face_db:
            raise HTTPException(status_code=500, detail="Base de datos no disponible")
        
        success = face_db.delete_person(person_id)
        
        if success:
            return {"message": "Persona eliminada exitosamente"}
        else:
            raise HTTPException(status_code=404, detail="Persona no encontrada")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Obtiene estadísticas del sistema"""
    try:
        stats = {}
        
        # Estadísticas de reconocimiento
        if face_recognizer:
            stats['recognition'] = face_recognizer.get_recognition_stats()
        
        # Estadísticas de la cámara
        if camera_handler:
            stats['camera'] = camera_handler.get_camera_info()
        
        # Estadísticas de la base de datos
        if face_db:
            people = face_db.list_people()
            recent_logs = face_db.get_recent_logs(100)
            
            stats['database'] = {
                'total_people': len(people),
                'total_logs': len(recent_logs),
                'recent_activity': len([log for log in recent_logs if log[0] is not None])
            }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/camera/restart")
async def restart_camera():
    """Reinicia la cámara"""
    try:
        global camera_handler
        
        if camera_handler:
            camera_handler.stop()
            time.sleep(1)
            
            camera_handler = CameraHandler()
            if camera_handler.start():
                return {"message": "Cámara reiniciada exitosamente"}
            else:
                raise HTTPException(status_code=500, detail="Error al reiniciar cámara")
        else:
            raise HTTPException(status_code=500, detail="Cámara no disponible")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/camera/status")
async def get_camera_status():
    """Obtiene el estado de la cámara"""
    try:
        if camera_handler:
            return camera_handler.get_camera_info()
        else:
            return {"is_running": False, "error": "Cámara no inicializada"}
            
    except Exception as e:
        return {"is_running": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Endpoint de salud del sistema"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": face_db is not None,
            "recognizer": face_recognizer is not None,
            "camera": camera_handler is not None and camera_handler.is_running
        }
        
        return status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000) 