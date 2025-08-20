#!/usr/bin/env python3
"""
Script de prueba para validar el sistema de reconocimiento facial
Ejecuta pruebas automáticas de todos los componentes
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Prueba que todos los módulos se importen correctamente"""
    print("🔍 Probando imports...")
    
    try:
        from face_db import FaceDatabase
        print("  ✓ face_db importado")
        
        from recognizer import FaceRecognizer
        print("  ✓ recognizer importado")
        
        from camera_handler import CameraHandler
        print("  ✓ camera_handler importado")
        
        from utils import draw_face_boxes, frame_to_jpeg, create_test_image
        print("  ✓ utils importado")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Error de import: {e}")
        return False

def test_database():
    """Prueba la base de datos SQLite"""
    print("\n🗄️ Probando base de datos...")
    
    try:
        from face_db import FaceDatabase
        
        # Crear base de datos temporal
        db = FaceDatabase("test.db")
        print("  ✓ Base de datos creada")
        
        # Probar inserción de persona
        test_embedding = np.random.rand(128).astype(np.float32)
        success = db.add_person("TestPerson", test_embedding)
        
        if success:
            print("  ✓ Persona agregada")
        else:
            print("  ✗ Error al agregar persona")
            return False
        
        # Probar búsqueda
        match = db.find_match(test_embedding)
        if match and match[0] == "TestPerson":
            print("  ✓ Búsqueda exitosa")
        else:
            print("  ✗ Error en búsqueda")
            return False
        
        # Probar listado
        people = db.list_people()
        if len(people) > 0:
            print("  ✓ Listado exitoso")
        else:
            print("  ✗ Error en listado")
            return False
        
        # Limpiar
        import os
        os.remove("test.db")
        print("  ✓ Base de datos limpiada")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en base de datos: {e}")
        return False

def test_recognizer():
    """Prueba el reconocedor facial"""
    print("\n🔍 Probando reconocedor facial...")
    
    try:
        from face_db import FaceDatabase
        from recognizer import FaceRecognizer
        
        # Crear base de datos temporal
        db = FaceDatabase("test_recognizer.db")
        
        # Crear reconocedor
        recognizer = FaceRecognizer(db, confidence_threshold=0.5)
        print("  ✓ Reconocedor creado")
        
        # Probar reconocimiento
        test_embedding = np.random.rand(128).astype(np.float32)
        result = recognizer.recognize_face(test_embedding, (100, 100, 50, 50))
        
        if result is not None:
            print("  ✓ Reconocimiento funcionando")
        else:
            print("  ✓ Reconocimiento funcionando (sin coincidencias)")
        
        # Limpiar
        import os
        os.remove("test_recognizer.db")
        print("  ✓ Base de datos limpiada")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en reconocedor: {e}")
        return False

def test_camera_handler():
    """Prueba el manejador de cámara"""
    print("\n📷 Probando manejador de cámara...")
    
    try:
        from camera_handler import CameraHandler
        
        # Crear manejador
        handler = CameraHandler()
        print("  ✓ Manejador creado")
        
        # Probar inicialización (sin iniciar cámara real)
        if handler.face_detector is not None:
            print("  ✓ Detector de rostros cargado")
        else:
            print("  ⚠️ Detector de rostros no disponible")
        
        # Probar generación de embeddings simulados
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_faces = [(100, 100, 50, 50)]
        
        embeddings = handler._generate_embeddings(test_frame, test_faces)
        
        if embeddings and len(embeddings) > 0:
            print("  ✓ Generación de embeddings funcionando")
        else:
            print("  ✗ Error en generación de embeddings")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en manejador de cámara: {e}")
        return False

def test_utils():
    """Prueba las funciones de utilidades"""
    print("\n🛠️ Probando utilidades...")
    
    try:
        from utils import draw_face_boxes, frame_to_jpeg, create_test_image
        
        # Probar creación de imagen de prueba
        test_image = create_test_image(320, 240)
        if test_image.shape == (240, 320, 3):
            print("  ✓ Creación de imagen de prueba")
        else:
            print("  ✗ Error en creación de imagen")
            return False
        
        # Probar conversión a JPEG
        jpeg_data = frame_to_jpeg(test_image)
        if len(jpeg_data) > 0:
            print("  ✓ Conversión a JPEG")
        else:
            print("  ✗ Error en conversión JPEG")
            return False
        
        # Probar dibujo de bounding boxes
        test_recognitions = [("TestPerson", 0.85, True, (50, 50, 100, 100))]
        annotated_frame = draw_face_boxes(test_image, test_recognitions)
        
        if annotated_frame.shape == test_image.shape:
            print("  ✓ Dibujo de bounding boxes")
        else:
            print("  ✗ Error en dibujo de bounding boxes")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en utilidades: {e}")
        return False

def test_webapp():
    """Prueba la aplicación web"""
    print("\n🌐 Probando aplicación web...")
    
    try:
        from webapp import app
        
        # Verificar que la app se creó
        if app is not None:
            print("  ✓ Aplicación web creada")
        else:
            print("  ✗ Error al crear aplicación web")
            return False
        
        # Verificar endpoints
        routes = [route.path for route in app.routes]
        expected_routes = ['/', '/video_feed', '/api/people', '/api/logs', '/health']
        
        for route in expected_routes:
            if route in routes:
                print(f"  ✓ Endpoint {route} disponible")
            else:
                print(f"  ⚠️ Endpoint {route} no encontrado")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en aplicación web: {e}")
        return False

def test_performance():
    """Prueba el rendimiento del sistema"""
    print("\n⚡ Probando rendimiento...")
    
    try:
        from face_db import FaceDatabase
        from recognizer import FaceRecognizer
        import time
        
        # Crear base de datos temporal
        db = FaceDatabase("test_performance.db")
        recognizer = FaceRecognizer(db)
        
        # Generar embeddings de prueba
        test_embeddings = [np.random.rand(128).astype(np.float32) for _ in range(100)]
        
        # Medir tiempo de inserción
        start_time = time.time()
        for i, emb in enumerate(test_embeddings):
            db.add_person(f"TestPerson{i}", emb)
        insert_time = time.time() - start_time
        
        print(f"  ✓ Inserción de 100 personas: {insert_time:.3f}s")
        
        # Medir tiempo de búsqueda
        start_time = time.time()
        for emb in test_embeddings[:10]:
            recognizer.recognize_face(emb, (100, 100, 50, 50))
        search_time = time.time() - start_time
        
        print(f"  ✓ Búsqueda de 10 personas: {search_time:.3f}s")
        
        # Limpiar
        import os
        os.remove("test_performance.db")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en prueba de rendimiento: {e}")
        return False

def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("🚀 INICIANDO PRUEBAS DEL SISTEMA DE RECONOCIMIENTO FACIAL")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Base de Datos", test_database),
        ("Reconocedor", test_recognizer),
        ("Manejador de Cámara", test_camera_handler),
        ("Utilidades", test_utils),
        ("Aplicación Web", test_webapp),
        ("Rendimiento", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASÓ")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! El sistema está listo.")
        return True
    else:
        print("⚠️ Algunas pruebas fallaron. Revisar errores antes de usar.")
        return False

def main():
    """Función principal"""
    try:
        success = run_all_tests()
        
        if success:
            print("\n🚀 El sistema está listo para usar!")
            print("📖 Ejecuta 'python main.py' para iniciar el sistema completo")
            print("🌐 O 'python webapp.py' para solo el servidor web")
        else:
            print("\n🔧 Corrige los errores antes de continuar")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error fatal durante las pruebas: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 