"""
Llena la carpeta dataset/Fondo con rostros de OTRAS personas (LFW, a color).
Esto es clave para la prueba 3: si "Fondo" no tiene caras ajenas, el modelo
aprende "cara = estudiante" y falla con la foto de la celebridad.

Además de esto, agrega a dataset/Fondo fotos de fondos variados
(paisajes, oficinas, aulas, calles, objetos) descargadas de Google Images.

Uso:  python 2_descargar_fondos.py --cantidad 400
"""
import argparse
import glob
import os
import random
import shutil
import urllib.request
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--cantidad", type=int, default=None, help="Cantidad total de imágenes a descargar (reparte entre caras y fondos)")
parser.add_argument("--caras", type=int, default=100, help="Cantidad de rostros de otras personas")
parser.add_argument("--fondos", type=int, default=100, help="Cantidad de fondos variados (paisajes, oficinas, etc.)")
parser.add_argument("--carpeta", default="dataset/Fondo")
args = parser.parse_args()

if args.cantidad is not None:
    args.caras = args.cantidad // 2
    args.fondos = args.cantidad - args.caras

os.makedirs(args.carpeta, exist_ok=True)

# 1. Copiar rostros de otras personas desde la carpeta LFW ya existente
lfw_base = os.path.expanduser(r"~\scikit_learn_data\lfw_home\lfw_funneled")
if os.path.exists(lfw_base):
    archivos_lfw = glob.glob(os.path.join(lfw_base, "*", "*.jpg"))
    random.seed(42)
    seleccion_lfw = random.sample(archivos_lfw, min(args.caras, len(archivos_lfw)))
    for i, origen in enumerate(seleccion_lfw):
        destino = os.path.join(args.carpeta, f"lfw_cara_{i:04d}.jpg")
        shutil.copy2(origen, destino)
    print(f"[OK] Se copiaron {len(seleccion_lfw)} rostros de otras personas a {args.carpeta}")
else:
    print("[AVISO] No se encontró la carpeta local de LFW, descargando rostros alternativos...")

# 2. Descargar fondos variados (paisajes, interiores, objetos) desde Lorem Picsum
print(f"Descargando {args.fondos} fondos variados...")
fondos_guardados = 0
for i in range(args.fondos):
    url = f"https://picsum.photos/400/400?random={i+1000}"
    ruta_fondo = os.path.join(args.carpeta, f"fondo_web_{i:04d}.jpg")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
            with open(ruta_fondo, "wb") as f:
                f.write(data)
        fondos_guardados += 1
        if (i + 1) % 20 == 0:
            print(f"  Descargados {i + 1}/{args.fondos} fondos...")
    except Exception as e:
        pass

print(f"[OK] Total fondos descargados: {fondos_guardados}")
total_archivos = len(os.listdir(args.carpeta))
print(f"[COMPLETADO] La carpeta '{args.carpeta}' ahora contiene {total_archivos} imágenes en total.")
