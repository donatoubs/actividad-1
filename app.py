"""
App de Streamlit para presentar las 3 pruebas.
Uso:  streamlit run app.py
"""
import json
import os

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

TAM = 224


@st.cache_resource
def cargar():
    if not os.path.exists("modelo_vgg16.keras") or not os.path.exists("clases.json"):
        return None, None
    modelo = load_model("modelo_vgg16.keras")
    with open("clases.json", encoding="utf-8") as f:
        clases = json.load(f)
    return modelo, clases


def preparar(img_bgr):
    img = cv2.resize(img_bgr, (TAM, TAM))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_input(img.astype("float32"))


def leer(archivo):
    datos = np.frombuffer(archivo.getvalue(), np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


modelo, clases = cargar()

if modelo is None:
    st.title("Clasificador VGG-16: Donato Oña vs Fondo")
    st.error("⚠️ **No se encontró el modelo entrenado (`modelo_vgg16.keras`) ni `clases.json`.**")
    st.info(
        "Para que la aplicación funcione, primero debes entrenar el modelo desde tu terminal:\n\n"
        "1. `python 2_descargar_fondos.py --cantidad 400`\n"
        "2. `python entrenar.py --nombre Donato`"
    )
    st.stop()

st.title("Clasificador VGG-16: Donato Oña vs Fondo")
st.caption(f"Clases: {clases[0]} / {clases[1]}")

pruebas = {
    "Prueba 1 - Foto con webcam (esperado: " + clases[0] + ")": "camara",
    "Prueba 2 - Fondo de Google Images (esperado: Fondo)": "archivo",
    "Prueba 3 - Rostro de celebridad (esperado: Fondo)": "archivo",
}
opcion = st.radio("Prueba", list(pruebas))

if pruebas[opcion] == "camara":
    archivo = st.camera_input("Tómate una foto")
else:
    archivo = st.file_uploader("Sube la imagen", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    img = leer(archivo)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=320)

    pred = modelo.predict(np.expand_dims(preparar(img), 0), verbose=0)[0]
    i = int(np.argmax(pred))

    st.subheader(f"Predicción: {clases[i]}")
    st.write(f"{clases[0]}: {pred[0]:.2%}  |  {clases[1]}: {pred[1]:.2%}")
    st.progress(float(pred[i]))
