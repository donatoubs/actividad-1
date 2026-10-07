"""
Transfer learning con VGG-16: <Nombre del estudiante> vs Fondo.
Misma estructura que el notebook de clase (perros vs gatos):
  cargar imágenes con OpenCV -> 224x224 -> one-hot -> train_test_split
  -> VGG16 congelada + Flatten + Dense(256) + Dropout + Dense(2, softmax)
  -> EarlyStopping -> guardar modelo.

Estructura esperada:
  dataset/
    Donato/   (fotos tuyas)
    Fondo/       (fondos + caras de otras personas)

Uso:  python entrenar.py --nombre Donato
"""
import argparse
import json
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument("--nombre", required=True)
parser.add_argument("--carpeta", default="dataset")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--loss", default="categorical_crossentropy")
args = parser.parse_args()

CLASES = [args.nombre, "Fondo"]  # índice 0 = estudiante, 1 = fondo
TAM = 224


def preparar(img_bgr):
    """Misma preparación en entrenamiento y en la app."""
    img = cv2.resize(img_bgr, (TAM, TAM))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_input(img.astype("float32"))


def load_images_from_folder(folder, etiqueta):
    images, labels = [], []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(preparar(img))
            labels.append(etiqueta)
    return images, labels


# Cargamos las dos clases y las unimos
x_est, y_est = load_images_from_folder(os.path.join(args.carpeta, CLASES[0]), 0)
x_fon, y_fon = load_images_from_folder(os.path.join(args.carpeta, CLASES[1]), 1)
print(f"{CLASES[0]}: {len(x_est)} imágenes | Fondo: {len(x_fon)} imágenes")

X = np.array(x_est + x_fon)
y = to_categorical(np.array(y_est + y_fon), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.argmax(1))

# Modelo base VGG16 pre-entrenado en ImageNet, sin la cabecera de clasificación
baseModel = VGG16(weights="imagenet", include_top=False, input_shape=(TAM, TAM, 3))

model = Sequential()
model.add(baseModel)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))  # estudiante / fondo

# Congelamos las capas del modelo base
for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=args.epochs, batch_size=args.batch_size, callbacks=[early_stopping])

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy en prueba: {acc:.4f}")

model.save("modelo_vgg16.keras")
with open("clases.json", "w", encoding="utf-8") as f:
    json.dump(CLASES, f, ensure_ascii=False)
print("Modelo guardado en modelo_vgg16.keras")
