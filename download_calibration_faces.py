import os
import io
import requests
from PIL import Image
import random
import time

# Carpeta de salida
OUT_DIR = "calibration_images"
os.makedirs(OUT_DIR, exist_ok=True)

# URLs de servicios que generan rostros de personas
URLS = [
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/",
    "https://thispersondoesnotexist.com/"
]

# URLs alternativas de rostros reales
ALTERNATIVE_URLS = [
    "https://randomuser.me/api/portraits/men/1.jpg",
    "https://randomuser.me/api/portraits/women/1.jpg",
    "https://randomuser.me/api/portraits/men/2.jpg",
    "https://randomuser.me/api/portraits/women/2.jpg",
    "https://randomuser.me/api/portraits/men/3.jpg",
    "https://randomuser.me/api/portraits/women/3.jpg",
    "https://randomuser.me/api/portraits/men/4.jpg",
    "https://randomuser.me/api/portraits/women/4.jpg",
    "https://randomuser.me/api/portraits/men/5.jpg",
    "https://randomuser.me/api/portraits/women/5.jpg",
    "https://randomuser.me/api/portraits/men/6.jpg",
    "https://randomuser.me/api/portraits/women/6.jpg",
    "https://randomuser.me/api/portraits/men/7.jpg",
    "https://randomuser.me/api/portraits/women/7.jpg",
    "https://randomuser.me/api/portraits/men/8.jpg",
    "https://randomuser.me/api/portraits/women/8.jpg",
    "https://randomuser.me/api/portraits/men/9.jpg",
    "https://randomuser.me/api/portraits/women/9.jpg",
    "https://randomuser.me/api/portraits/men/10.jpg",
    "https://randomuser.me/api/portraits/women/10.jpg"
]

NUM_IMAGES = 20
SIZE = 112

def download_and_resize(url, save_path, size=112):
    try:
        print(f"Descargando desde: {url}")
        
        # Configurar headers para evitar bloqueos
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        }
        
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        
        if len(resp.content) == 0:
            print(f"✗ Error: respuesta vacía desde {url}")
            return False
            
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        
        # Verificar que la imagen tenga un tamaño mínimo
        if img.size[0] < 50 or img.size[1] < 50:
            print(f"✗ Error: imagen demasiado pequeña {img.size}")
            return False
        
        # Redimensionar con letterbox
        w, h = img.size
        scale = min(size / w, size / h)
        new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Crear canvas negro y centrar la imagen
        canvas = Image.new("RGB", (size, size), (0,0,0))
        canvas.paste(img, ((size-new_w)//2, (size-new_h)//2))
        
        canvas.save(save_path, "JPEG", quality=95)
        print(f"✓ Guardada {save_path} ({new_w}x{new_h} -> {size}x{size})")
        return True
        
    except requests.exceptions.Timeout:
        print(f"✗ Timeout descargando {url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Error de red descargando {url}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error procesando {url}: {e}")
        return False

def main():
    print(f"Descargando {NUM_IMAGES} rostros de personas...")
    print(f"Guardando en: {os.path.abspath(OUT_DIR)}")
    
    # Usar URLs alternativas que son más confiables para rostros
    all_urls = ALTERNATIVE_URLS + URLS
    random.shuffle(all_urls)
    
    successful_downloads = 0
    for i in range(NUM_IMAGES):
        url = all_urls[i % len(all_urls)]
        save_path = os.path.join(OUT_DIR, f"face_{i+1:03d}.jpg")
        
        print(f"\n--- Descarga {i+1}/{NUM_IMAGES} ---")
        
        if download_and_resize(url, save_path, SIZE):
            successful_downloads += 1
        
        # Pequeña pausa para no sobrecargar los servidores
        time.sleep(1)
    
    print(f"\nDescarga completada: {successful_downloads}/{NUM_IMAGES} rostros descargados exitosamente")
    
    # Verificar archivos descargados
    if os.path.exists(OUT_DIR):
        files = [f for f in os.listdir(OUT_DIR) if f.endswith('.jpg')]
        print(f"Archivos JPG encontrados en {OUT_DIR}: {len(files)}")
        for file in sorted(files):
            file_path = os.path.join(OUT_DIR, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size} bytes)")

if __name__ == "__main__":
    main()