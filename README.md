# ¿Héroe o villano? Clasificación de la alineación de personajes de DC

Proyecto de la asignatura **Aprendizaje de máquina (ACIF104), Universidad Andrés Bello**, Sumativa 1 (Semana 6).

El proyecto predice si un personaje de DC Comics es **bueno o malo** a partir de sus atributos (tipo de identidad, sexo, color de ojos y de pelo, si está vivo, número de apariciones y año de primera aparición). Incluye:

1. Un **notebook reproducible** con el análisis exploratorio, la comparación de 4 modelos de *machine learning* (más una línea base) y 3 arquitecturas de *deep learning*, el análisis de balanceo de clases, la evaluación final y la explicación con SHAP.
2. Una **aplicación web** (back end en Flask + front end HTML/JS) que usa el modelo ganador, explica cada predicción y monitorea su desempeño en uso.

## Estructura

```
acif104-dc-alineacion/
├── data/
│   └── dc-wikia-data.csv          # Dataset (FiveThirtyEight, licencia CC BY 4.0)
├── notebooks/
│   └── acif104_s6_grupo1.ipynb    # Todos los experimentos; genera salidas/ y el modelo de la app
├── app/
│   ├── app.py                     # API REST (Flask): predicción, explicación, retroalimentación y monitoreo
│   ├── static/index.html          # Front end: predictor, panel de monitoreo y ficha del modelo
│   ├── modelo/                    # Aquí va app_modelo.joblib (lo genera el notebook)
│   ├── requirements.txt
│   └── Dockerfile
├── docs/                          # Figuras y tablas usadas en el informe
├── requirements.txt               # Dependencias del notebook
└── README.md
```

## 1. Ejecutar el notebook

**Opción recomendada: Google Colab**

1. Abre `notebooks/acif104_s6_grupo1.ipynb` en [Google Colab](https://colab.research.google.com/) (*Archivo → Subir notebook*).
2. Sube `data/dc-wikia-data.csv` al panel de archivos (si no lo encuentra, el notebook lo descarga del repositorio de FiveThirtyEight).
3. *Entorno de ejecución → Ejecutar todo*. Tarda entre 20 y 30 minutos en CPU.
4. Al terminar se descarga `salidas_acif104.zip` con:
   - `salidas/tablas/*.csv` y `salidas/figuras/*.png`: todas las tablas y figuras del informe.
   - `salidas/resumen.json`: cifras clave, entorno y versiones de las bibliotecas.
   - `salidas/app_modelo.joblib`: el modelo desplegado con su preprocesador.

**Opción local**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
cp data/dc-wikia-data.csv notebooks/
cd notebooks && jupyter notebook acif104_s6_grupo1.ipynb
```

Semilla fija (`42`) para reproducibilidad. La partición es 70 % entrenamiento / 30 % prueba (estratificada); los modelos se seleccionan con validación cruzada estratificada de 5 pliegues dentro del 70 %.

## 2. Ejecutar la aplicación web

1. Copia `salidas/app_modelo.joblib` (generado por el notebook) a `app/modelo/`.
2. Instala y ejecuta:

```bash
cd app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Instala la misma versión de scikit-learn que usó el notebook, por ejemplo:
# pip install scikit-learn==1.6.1
python app.py
```

3. Abre <http://localhost:5000>.

**Con Docker (producción)**

```bash
cd app
docker build -t heroe-villano .
docker run -p 8000:8000 -v $(pwd)/monitoreo:/app/monitoreo heroe-villano
```

### API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/salud` | Estado del servicio, modelo cargado y método de explicación |
| GET | `/api/opciones` | Valores válidos del formulario y métricas del modelo en prueba |
| POST | `/api/predecir` | Recibe los atributos en JSON y devuelve probabilidad, clase y aporte de cada variable |
| POST | `/api/retroalimentacion` | Registra la alineación real de una predicción (`{"id": "...", "real": "Héroe"}`) |
| GET | `/api/monitoreo` | Volumen, latencia, deriva de datos, exactitud con retroalimentación y alertas |

Ejemplo:

```bash
curl -X POST http://localhost:5000/api/predecir -H "Content-Type: application/json" \
  -d '{"ID":"Secret Identity","SEX":"Female Characters","EYE":"Blue Eyes","HAIR":"Black Hair","ALIVE":"Living Characters","APPEARANCES":120,"YEAR":1985}'
```

Todos los campos son opcionales: un campo vacío se trata como dato faltante, igual que en el entrenamiento.

### Explicabilidad y monitoreo

- **Explicación por predicción:** valores SHAP del modelo (si `shap` está instalado) agregados por variable original; si no, explicación por oclusión (cambio de probabilidad al reemplazar cada variable por su valor típico del entrenamiento).
- **Monitoreo:** cada predicción se registra en `app/monitoreo/predicciones.csv`. El panel muestra volumen, latencia media y p95, proporción de héroes predichos, deriva en el porcentaje de faltantes respecto del entrenamiento y la exactitud real cuando los usuarios informan la alineación verdadera. Se generan alertas si la deriva supera 15 puntos o si la exactitud en uso cae más de 10 puntos bajo la de prueba.

## Datos y licencia

Dataset: FiveThirtyEight. (2014). *comic-characters* [Conjunto de datos]. GitHub. https://github.com/fivethirtyeight/data/tree/master/comic-characters — licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Autoría

Grupo 1 · ACIF104 · Universidad Andrés Bello · 2026.
