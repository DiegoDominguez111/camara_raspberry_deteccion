#!/usr/bin/env python3
"""
Script para probar modelos ONNX y preparar integración con AI Camera
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import sys
from picamera2 import Picamera2
import urllib.request
import zipfile
from pathlib import Path
import argparse


class ONNXModelTester:
    """
    Clase para probar modelos ONNX de reconocimiento facial
    """
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.picam2 = None
        self.current_model = None
        self.session = None
        
        # Crear directorio de modelos si no existe
        os.makedirs(models_dir, exist_ok=True)
        
        # URLs de modelos pre-entrenados
        self.model_urls = {
            "mobilefacenet": "https://github.com/XiaoyuGuo/MobileFaceNet_Pytorch/releases/download/v1.0/mobilefacenet.onnx",
            "arcface": "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcface_r100.onnx",
            "facenet": "https://github.com/onnx/models/raw/main/vision/body_analysis/facenet/model/facenet-1.onnx"
        }
        
        # Inicializar cámara
        self._init_camera()
    
    def _init_camera(self):
        """Inicializa la AI Camera"""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)},
                buffer_count=2
            )
            self.picam2.configure(config)
            print("✅ AI Camera inicializada")
        except Exception as e:
            print(f"❌ Error al inicializar la cámara: {e}")
            raise
    
    def download_model(self, model_name):
        """
        Descarga un modelo ONNX
        
        Args:
            model_name: Nombre del modelo a descargar
        """
        if model_name not in self.model_urls:
            print(f"❌ Modelo '{model_name}' no disponible")
            return False
        
        model_path = os.path.join(self.models_dir, f"{model_name}.onnx")
        
        if os.path.exists(model_path):
            print(f"✅ Modelo {model_name} ya existe")
            return True
        
        print(f"📥 Descargando modelo {model_name}...")
        try:
            urllib.request.urlretrieve(
                self.model_urls[model_name],
                model_path
            )
            print(f"✅ Modelo {model_name} descargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            return False
    
    def load_model(self, model_name):
        """
        Carga un modelo ONNX
        
        Args:
            model_name: Nombre del modelo a cargar
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.onnx")
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo {model_name} no encontrado")
            return False
        
        try:
            # Configurar ONNX Runtime para optimización
            providers = ['CPUExecutionProvider']
            
            # Intentar usar GPU si está disponible
            try:
                gpu_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in gpu_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
                    print("🚀 GPU CUDA detectada")
                elif 'OpenVINOExecutionProvider' in gpu_providers:
                    providers.insert(0, 'OpenVINOExecutionProvider')
                    print("🚀 GPU OpenVINO detectada")
            except:
                pass
            
            # Crear sesión ONNX
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            self.current_model = model_name
            print(f"✅ Modelo {model_name} cargado exitosamente")
            print(f"   - Proveedores: {providers}")
            
            # Mostrar información del modelo
            self._show_model_info()
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def _show_model_info(self):
        """Muestra información del modelo cargado"""
        if not self.session:
            return
        
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        print(f"📊 Información del modelo:")
        print(f"   - Entradas:")
        for inp in inputs:
            print(f"     * {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"   - Salidas:")
        for out in outputs:
            print(f"     * {out.name}: {out.shape} ({out.type})")
    
    def preprocess_image(self, image, target_size=(112, 112)):
        """
        Preprocesa una imagen para el modelo ONNX
        
        Args:
            image: Imagen de entrada
            target_size: Tamaño objetivo
            
        Returns:
            Imagen preprocesada
        """
        try:
            # Redimensionar
            resized = cv2.resize(image, target_size)
            
            # Convertir a float32 y normalizar
            normalized = resized.astype(np.float32) / 255.0
            
            # Agregar dimensión de batch si es necesario
            if len(normalized.shape) == 3:
                normalized = np.expand_dims(normalized, axis=0)
            
            return normalized
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento: {e}")
            return None
    
    def extract_embedding(self, face_image):
        """
        Extrae embedding usando el modelo ONNX cargado
        
        Args:
            face_image: Imagen del rostro
            
        Returns:
            Embedding extraído o None
        """
        if not self.session:
            print("❌ No hay modelo cargado")
            return None
        
        try:
            # Preprocesar imagen
            input_data = self.preprocess_image(face_image)
            if input_data is None:
                return None
            
            # Obtener nombres de entrada y salida
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            # Ejecutar inferencia
            start_time = time.time()
            outputs = self.session.run([output_name], {input_name: input_data})
            inference_time = time.time() - start_time
            
            # Extraer embedding
            embedding = outputs[0].flatten()
            
            print(f"⚡ Inferencia completada en {inference_time*1000:.2f}ms")
            print(f"   - Embedding shape: {embedding.shape}")
            print(f"   - Rango: [{embedding.min():.4f}, {embedding.max():.4f}]")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error en inferencia: {e}")
            return None
    
    def test_model_performance(self, num_frames=100):
        """
        Prueba el rendimiento del modelo actual
        
        Args:
            num_frames: Número de frames a procesar
        """
        if not self.session:
            print("❌ No hay modelo cargado")
            return
        
        print(f"\n🚀 PROBANDO RENDIMIENTO DEL MODELO {self.current_model.upper()}")
        print("=" * 50)
        
        # Iniciar cámara
        self.picam2.start()
        time.sleep(2)
        
        inference_times = []
        embeddings_generated = 0
        
        try:
            for i in range(num_frames):
                # Capturar frame
                frame = self.picam2.capture_array()
                if frame is None:
                    continue
                
                # Detectar rostros (usar OpenCV básico)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                
                if len(faces) > 0:
                    # Procesar primer rostro detectado
                    (x, y, w, h) = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Extraer embedding
                    start_time = time.time()
                    embedding = self.extract_embedding(face_roi)
                    end_time = time.time()
                    
                    if embedding is not None:
                        inference_times.append(end_time - start_time)
                        embeddings_generated += 1
                
                # Mostrar progreso
                if (i + 1) % 10 == 0:
                    print(f"📊 Procesados {i + 1}/{num_frames} frames")
                
                time.sleep(0.01)
            
            # Mostrar estadísticas
            if inference_times:
                avg_time = np.mean(inference_times)
                min_time = np.min(inference_times)
                max_time = np.max(inference_times)
                
                print(f"\n📊 ESTADÍSTICAS DE RENDIMIENTO:")
                print(f"   - Frames procesados: {num_frames}")
                print(f"   - Embeddings generados: {embeddings_generated}")
                print(f"   - Tiempo promedio: {avg_time*1000:.2f}ms")
                print(f"   - Tiempo mínimo: {min_time*1000:.2f}ms")
                print(f"   - Tiempo máximo: {max_time*1000:.2f}ms")
                print(f"   - FPS estimado: {1/avg_time:.1f}")
                
            else:
                print("⚠️  No se generaron embeddings durante la prueba")
                
        except Exception as e:
            print(f"❌ Error en prueba de rendimiento: {e}")
        finally:
            self.picam2.stop()
    
    def compare_models(self, model_names=None):
        """
        Compara el rendimiento de múltiples modelos
        
        Args:
            model_names: Lista de nombres de modelos a comparar
        """
        if model_names is None:
            model_names = list(self.model_urls.keys())
        
        print(f"\n🔍 COMPARANDO MODELOS: {', '.join(model_names)}")
        print("=" * 50)
        
        results = {}
        
        for model_name in model_names:
            print(f"\n📊 Probando modelo: {model_name}")
            
            # Descargar si no existe
            if not self.download_model(model_name):
                continue
            
            # Cargar modelo
            if not self.load_model(model_name):
                continue
            
            # Probar rendimiento
            start_time = time.time()
            self.test_model_performance(num_frames=50)
            total_time = time.time() - start_time
            
            results[model_name] = {
                'total_time': total_time,
                'model_loaded': True
            }
            
            print(f"✅ Modelo {model_name} probado en {total_time:.2f}s")
        
        # Mostrar resumen de comparación
        print(f"\n📋 RESUMEN DE COMPARACIÓN:")
        print("=" * 30)
        
        for model_name, result in results.items():
            status = "✅" if result['model_loaded'] else "❌"
            time_str = f"{result['total_time']:.2f}s" if result['model_loaded'] else "Falló"
            print(f"{status} {model_name}: {time_str}")
    
    def cleanup(self):
        """Limpia recursos"""
        if self.picam2:
            self.picam2.stop()
        cv2.destroyAllWindows()


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Probador de modelos ONNX para AI Camera")
    parser.add_argument("--model", "-m", help="Modelo específico a probar")
    parser.add_argument("--download", "-d", help="Descargar modelo específico")
    parser.add_argument("--compare", "-c", action="store_true", help="Comparar todos los modelos")
    parser.add_argument("--performance", "-p", type=int, default=100, help="Frames para prueba de rendimiento")
    parser.add_argument("--models-dir", default="models", help="Directorio de modelos")
    
    args = parser.parse_args()
    
    try:
        tester = ONNXModelTester(args.models_dir)
        
        if args.download:
            # Solo descargar modelo
            if tester.download_model(args.download):
                print(f"✅ Modelo {args.download} descargado exitosamente")
            else:
                print(f"❌ Fallo al descargar modelo {args.download}")
                sys.exit(1)
                
        elif args.model:
            # Probar modelo específico
            if not tester.download_model(args.model):
                print(f"❌ No se pudo descargar modelo {args.model}")
                sys.exit(1)
            
            if not tester.load_model(args.model):
                print(f"❌ No se pudo cargar modelo {args.model}")
                sys.exit(1)
            
            tester.test_model_performance(args.performance)
            
        elif args.compare:
            # Comparar todos los modelos
            tester.compare_models()
            
        else:
            # Modo interactivo
            print("🎯 PROBADOR INTERACTIVO DE MODELOS ONNX")
            print("=" * 40)
            
            while True:
                print("\nOpciones:")
                print("1. Descargar modelo")
                print("2. Cargar modelo")
                print("3. Probar rendimiento")
                print("4. Comparar modelos")
                print("5. Salir")
                
                choice = input("\nSelecciona una opción (1-5): ").strip()
                
                if choice == "1":
                    model_name = input("Nombre del modelo: ").strip()
                    if model_name:
                        tester.download_model(model_name)
                        
                elif choice == "2":
                    model_name = input("Nombre del modelo: ").strip()
                    if model_name:
                        tester.load_model(model_name)
                        
                elif choice == "3":
                    if tester.current_model:
                        frames = input("Número de frames (default 100): ").strip()
                        try:
                            frames = int(frames) if frames else 100
                            tester.test_model_performance(frames)
                        except ValueError:
                            print("❌ Número de frames inválido")
                    else:
                        print("❌ Debes cargar un modelo primero")
                        
                elif choice == "4":
                    tester.compare_models()
                    
                elif choice == "5":
                    print("👋 ¡Hasta luego!")
                    break
                    
                else:
                    print("❌ Opción no válida")
        
    except Exception as e:
        print(f"💥 Error fatal: {e}")
        sys.exit(1)
    finally:
        try:
            tester.cleanup()
        except:
            pass


if __name__ == "__main__":
    main() 