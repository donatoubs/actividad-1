"""
Captura fotos con la webcam para la clase del estudiante.
Uso:  python 1_capturar_fotos.py --nombre Donato --cantidad 300
Teclas: ESPACIO = iniciar/pausar captura continua, Q = salir.
Muévete, cambia de ángulo, luz y expresión mientras captura.
"""
import argparse
import os
import time

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--nombre", required=True, help="Nombre del estudiante (será la etiqueta)")
parser.add_argument("--cantidad", type=int, default=300)
parser.add_argument("--carpeta", default="dataset")
parser.add_argument("--intervalo", type=float, default=0.15, help="segundos entre fotos")
args = parser.parse_args()

destino = os.path.join(args.carpeta, args.nombre)
os.makedirs(destino, exist_ok=True)
inicio = len(os.listdir(destino))

cam = cv2.VideoCapture(0)
capturando = False
guardadas = 0
ultimo = 0.0

while guardadas < args.cantidad:
    ok, frame = cam.read()
    if not ok:
        print("No se pudo leer la webcam")
        break

    if capturando and time.time() - ultimo >= args.intervalo:
        ruta = os.path.join(destino, f"{args.nombre}_{inicio + guardadas:04d}.jpg")
        cv2.imwrite(ruta, frame)
        guardadas += 1
        ultimo = time.time()

    vista = frame.copy()
    estado = "CAPTURANDO" if capturando else "PAUSA (ESPACIO para iniciar)"
    cv2.putText(vista, f"{estado}  {guardadas}/{args.cantidad}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Captura", vista)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord(" "):
        capturando = not capturando
    elif tecla == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print(f"Guardadas {guardadas} fotos en {destino}")
