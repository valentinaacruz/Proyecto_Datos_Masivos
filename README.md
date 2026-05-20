# Clasificador Masivo de Noticias
### PySpark + Redes Neuronales — Datos Masivos I

> Pipeline distribuido de NLP de extremo a extremo: almacenamiento, MapReduce, deduplicación con LSH y clasificación neuronal sobre ~500k artículos de noticias reales.

---

## Equipo: DataWhales

| Nombre |
|--------|
| Cardon Carrillo Pedro Manuel | 
| Cruz Mendoza Valentina Ayelen | 
| Rivera Hernandez Milena Fernanda |
---

## Descripción

En este repositorio encontrarás nuestro proyecto final de la asignatura de Datos Masivos I que construye un pipeline distribuido de clasificación de noticias que cubre las siguientes etapas:

1. **Almacenamiento distribuido** — Corpus (~500k documentos) particionado en formato Parquet, simulando HDFS con PySpark local.
2. **Preprocesamiento con MapReduce** — Tokenización, eliminación de stopwords, stemming y cálculo de vectores TF-IDF dispersos.
3. **Deduplicación con LSH** — Detección y eliminación de noticias casi idénticas usando MinHash + Locality-Sensitive Hashing (similitud Jaccard y coseno).
4. **Clasificación neuronal** — Entrenamiento de un MLP (3 capas, ReLU + Softmax) con PySpark MLlib para categorizar noticias en política, deportes, salud, tecnología, entre otras.
5. **Evaluación y análisis de complejidad** — F1-score por clase y análisis del modelo costo–comunicación de cada etapa del pipeline.

---

## Estructura del repositorio

```
clasificador-noticias/
│
├── data/
│   ├── raw/              # JSONs originales (CC-News, MIND Large)
│   ├── processed/        # Vectores TF-IDF en Parquet
│   └── dedup/            # Corpus limpio post-LSH en Parquet
│
├── notebooks/
│   ├── 01_ingesta.ipynb          # Carga y particionado de datos
│   ├── 02_preprocesamiento.ipynb # MapReduce + TF-IDF
│   ├── 03_lsh_dedup.ipynb        # MinHash + LSH
│   ├── 04_clasificador.ipynb     # Red neuronal MLP
│   └── 05_evaluacion.ipynb       # Métricas y análisis de complejidad
│
├── models/               # Pesos del modelo entrenado
├── requirements.txt
└── README.md
```

---

## Datos

El proyecto utiliza dos fuentes públicas combinadas para alcanzar ~5 GB de noticias reales:

| Dataset | Tamaño | Etiquetas | Acceso |
|---------|--------|-----------|--------|
| [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) | ~3.5 GB | No | `load_dataset("vblagoje/cc_news")` |
| [MIND Large](https://msnews.github.io) | ~1.5 GB |  18 categorías | msnews.github.io |

- **CC-News** se usa como corpus para la fase de deduplicación con LSH.
- **MIND Large** se usa como ground truth para entrenar y evaluar el clasificador.

---

##  Requisitos

- Python 3.10+
- Java 8 o 11 (requerido por PySpark)
- PySpark 3.x
- Google Colab **o** entorno local con al menos 8 GB de RAM

---

## Instalación

**1. Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/clasificador-noticias.git
cd clasificador-noticias
```

**2. Crea un entorno virtual e instala dependencias**
```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. (Opcional) Verifica que PySpark funciona**
```bash
python -c "import pyspark; print(pyspark.__version__)"
```

---

## Uso

Ejecuta los notebooks en orden desde la carpeta `notebooks/`:

```bash
jupyter notebook notebooks/01_ingesta.ipynb
```

O si usas Google Colab, sube cada notebook y monta tu Google Drive como almacenamiento:

```python
from google.colab import drive
drive.mount('/content/drive')
```

> **Nota:** Los datos crudos no están incluidos en el repositorio por su tamaño. Consulta la sección de Datos para descargarlos.

---

## Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| PySpark 3.x | Procesamiento distribuido (RDDs, DataFrames, MLlib) |
| NLTK / SpaCy | Tokenización, stopwords, stemming |
| Keras / MLlib | Red neuronal MLP para clasificación |
| MinHash + LSH | Deduplicación por similitud |
| Parquet | Almacenamiento columnar eficiente |
| Jupyter / Colab | Entorno de desarrollo interactivo |

---

## Temas del curso cubiertos

- Almacenamiento distribuido (HDFS, replicación, tolerancia a fallos)
- Algoritmos MapReduce y modelo costo–comunicación
- TF-IDF e índice invertido
- Medidas de similitud (Jaccard, coseno) y LSH
- Redes neuronales con PySpark MLlib
- Evaluación de modelos (F1, accuracy, precisión, recall)
- Teoría de complejidad algorítmica

---

##  Estado del proyecto

> 🟡 **En desarrollo** — Propuesta aprobada. Implementación en progreso.

| Fase | Estado |
|------|--------|
| Fase 1 — Almacenamiento | 🔲 Pendiente |
| Fase 2 — MapReduce + TF-IDF | 🔲 Pendiente |
| Fase 3 — LSH Deduplicación | 🔲 Pendiente |
| Fase 4 — Red Neuronal | 🔲 Pendiente |
| Fase 5 — Evaluación | 🔲 Pendiente |

---
