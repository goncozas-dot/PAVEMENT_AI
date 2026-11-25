# PAVEMENT AI

Proyecto individual SENATI para estimar el índice de condición del pavimento rígido mediante regresión lineal.

## Instalación

```powershell
python -m pip install -r requirements.txt
python generate_dataset.py
python -m streamlit run app.py
```

## Contenido

- `app.py`: dashboard, modelo predictivo, registro y exportación.
- `image_classifier.py`: extracción HOG y clasificador local de patologías desde imágenes.
- `pavimento_simulado.csv`: 60 registros simulados identificados como tales.
- `generate_dataset.py`: generación reproducible del dataset.
- `requirements.txt`: dependencias del proyecto.
- `test_app.py`: prueba básica del modelo y métricas.
- `generate_report.py`: genera el informe PDF con métricas y conclusiones.

## Clasificación de imágenes

En la pestaña **Imágenes**, carga un ZIP con fotos JPG/PNG organizadas por clase,
por ejemplo `dataset/grieta/foto.jpg` y `dataset/despostillamiento/foto.jpg`.
Cada clase requiere al menos dos imágenes. La aplicación entrena un clasificador
local con descriptores HOG; luego puedes arrastrar una foto nueva al Dashboard
principal para clasificarla. Para mejores
resultados, utiliza numerosas fotografías variadas y revisa el modelo con datos
de prueba independientes.

El clasificador devuelve una clase visual, no estima el PCI ni determina
severidad técnica. Las fotos y las etiquetas deben proceder de inspecciones
correctamente etiquetadas; las imágenes de demostración no validan el estado
estructural de una vía. La puntuación mostrada no es una precisión medida con
un conjunto de prueba independiente.

## Nota técnica

Los datos son ficticios. Una aplicación real debe separar el entrenamiento y la prueba por tramo o cronológicamente, incorporar una evaluación técnica del PCI y considerar tránsito, clima, espesor y mantenimiento.
