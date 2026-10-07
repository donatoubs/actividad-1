# Visión por Computador – Actividad calificable: Donato Oña vs Fondo (VGG-16)

Clasificador con transfer learning sobre VGG-16 que distingue mi rostro (`Donato`) de cualquier otra imagen (`Fondo`). Basado en el notebook de clase `UC.03 Transfer Learning & Model Improvement/Image classification/Clasificacion_imagenes_perros_gatos_VGG-16_transfer_learning.ipynb` del repositorio [eugeniomorocho/ComputerVision](https://github.com/eugeniomorocho/ComputerVision).

## Archivos

| Archivo | Uso |
|---|---|
| `1_capturar_fotos.py` | Captura fotos con la webcam en `dataset/Donato` |
| `2_descargar_fondos.py` | Llena `dataset/Fondo` con caras de otras personas (LFW) |
| `Clasificacion_Donato_Fondo_VGG-16.ipynb` | Entrenamiento, evaluación, pruebas e hiperparámetros |
| `entrenar.py` | Lo mismo que el notebook, desde la terminal |
| `app.py` | App de Streamlit para la presentación |

## Cómo correrlo

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python 1_capturar_fotos.py --nombre Donato --cantidad 300
python 2_descargar_fondos.py --cantidad 400
# agregar a dataset/Fondo 100-200 fondos variados de Google Images

# entrenar: correr el notebook completo, o bien
python entrenar.py --nombre Donato

streamlit run app.py
```

## Pruebas

1. Foto tomada con la webcam el día de la presentación → `Donato`
2. Fondo de Google Images (palabra dada por un compañero) → `Fondo`
3. Celebridad parecida a mí (nombre dado por un compañero) → `Fondo`

La clase `Fondo` incluye caras de otras personas para que el modelo no aprenda "cara = Donato" y la prueba 3 salga bien.
