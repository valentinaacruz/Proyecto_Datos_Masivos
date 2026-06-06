# Clasificador Masivo de Noticias
### PySpark + Redes Neuronales — Datos Masivos I

> Pipeline distribuido de NLP de extremo a extremo: almacenamiento, MapReduce, deduplicación con LSH y clasificación neuronal sobre ~700k artículos de noticias reales.

---

## Equipo: DataWhales

| Nombre |
|--------|
| Cardon Carrillo Pedro Manuel |
| Cruz Mendoza Valentina Ayelen |
| Rivera Hernandez Milena Fernanda |

---

## Descripción

Proyecto final de la asignatura Datos Masivos I. Construye un pipeline distribuido de clasificación de noticias que cubre las siguientes etapas:

1. **Almacenamiento distribuido** — Corpus (~700k documentos entre CC-News y MIND Large) particionado en formato Parquet, simulando HDFS con PySpark en modo `local[*]`.
2. **Preprocesamiento con MapReduce** — Tokenización y eliminación de stopwords con operadores nativos JVM de MLlib (`RegexTokenizer`, `StopWordsRemover`), stemming con NLTK (`PorterStemmer`) y cálculo de vectores TF-IDF dispersos de 65,536 dimensiones con `HashingTF` + `IDF`.
3. **Deduplicación con LSH** — Detección y eliminación de noticias casi idénticas usando MinHash LSH (similitud Jaccard ≥ 0.85) y Random Projection LSH (similitud coseno ≥ 0.90), con agrupación de duplicados via Union-Find.
4. **Clasificación neuronal** — Entrenamiento de un MLP de 3 capas (ReLU + Softmax) con `MLPClassifier` de scikit-learn sobre vectores TF-IDF reducidos a 4,096 features (ChiSqSelector) para clasificar noticias de MIND Large en 18 categorías.
5. **Evaluación y análisis de complejidad** — Accuracy, F1, precisión y recall por clase, matriz de confusión, y análisis del modelo costo–comunicación de cada etapa del pipeline.

---

## Estructura del repositorio

> **NOTA:** Los datos no se encuentran en el repositorio porque son 5GB. Se pueden descargar directamente de las funtes que vienen más abajo.


```
clasificador-noticias/
│
├── datos/
│   ├── MINDlarge_train/      # TSV original (news.tsv, behaviors.tsv)
│   ├── MINDlarge_dev/        # TSV original (news.tsv, behaviors.tsv)
│   ├── cc_news/              # Cache HuggingFace + Parquet crudo
│   ├── processed/
│   │   ├── cc_news/          # Parquet limpio CC-News (8 particiones)
│   │   ├── mind_large/       # Parquet MIND particionado por categoría
│   │   ├── tokens_cc/        # Tokens CC-News (muestra 100k docs)
│   │   ├── tfidf_mind/       # Vectores TF-IDF MIND (173k docs)
│   │   └── predicciones/     # Salida del clasificador MLP
│   └── dedup/                # Corpus limpio post-LSH en Parquet
│
├── notebooks/
│   ├── 01_ingesta.ipynb          # Carga y particionado de datos
│   ├── 02_preprocesamiento.ipynb # MapReduce + TF-IDF
│   ├── 03_lsh_dedup.ipynb        # MinHash + LSH
│   ├── 04_clasificador.ipynb     # Red neuronal MLP
│   └── 05_evaluacion.ipynb       # Métricas y análisis de complejidad
│
├── models/
│   ├── mlp_sklearn.joblib    # Modelo MLP entrenado
│   └── idf_model_mind/       # Modelo IDF de MLlib
├── resultados/               # Matriz de confusión y reportes
├── requirements.txt
└── README.md
```

---

## Datos

El proyecto combina dos fuentes públicas (~5 GB en total):

| Dataset | Tamaño | Etiquetas | Uso en el proyecto | Acceso |
|---------|--------|-----------|--------------------|--------|
| [CC-News](https://huggingface.co/datasets/vblagoje/cc_news) | ~3.5 GB | No | Deduplicación con LSH (Fase 3) | `load_dataset("vblagoje/cc_news")` |
| [MIND Large](https://msnews.github.io) | ~1.5 GB | 18 categorías | TF-IDF + clasificación (Fases 2 y 4) | msnews.github.io |

- **CC-News** (~703k artículos): corpus sin etiquetas usado para demostrar MinHash + LSH. Se trabaja con una muestra de 100k documentos en local.
- **MIND Large** (~173k artículos): dataset de Microsoft News con categorías etiquetadas, usado como ground truth para entrenar y evaluar el clasificador.

> Los datos crudos no están incluidos en el repositorio por su tamaño. Ver instrucciones de descarga abajo.

---

## Requisitos

- Python 3.10+
- Java 8 u 11 (requerido por PySpark)
- PySpark 3.5.x
- Al menos 10 GB de RAM recomendados para correr todos los notebooks

---

## Instalación

**1. Clona el repositorio**
```bash
git clone https://github.com/valentinaacruz/Proyecto_Datos_Masivos.git
cd Proyecto_Datos_Masivos
```

**2. Crea un entorno virtual e instala dependencias**
```bash
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> El directorio `venv/` está en `.gitignore` — no se sube al repo, cada integrante lo crea localmente.

**3. Descarga los datos**

CC-News se descarga automáticamente al correr `01_ingesta.ipynb` (via HuggingFace `datasets`).

MIND Large debe descargarse manualmente desde [msnews.github.io](https://msnews.github.io) y descomprimirse en:
```
datos/MINDlarge_train/
datos/MINDlarge_dev/
```

**4. Verifica la instalación**
```bash
python3 -c "import pyspark; print('PySpark', pyspark.__version__)"
python3 -c "import nltk; print('NLTK OK')"
python3 -c "import sklearn; print('scikit-learn OK')"
python3 -c "import datasets; print('Datasets OK')"
```

---

## Uso

Ejecuta los notebooks en orden desde la carpeta `notebooks/`:

```bash
jupyter notebook notebooks/01_ingesta.ipynb
```

> **Nota:** El orden importa — cada notebook produce archivos Parquet que el siguiente consume.

---

## Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| PySpark 3.5 | Procesamiento distribuido (RDDs, DataFrames, MLlib) |
| NLTK | Stopwords y stemming (PorterStemmer) |
| scikit-learn | MLP (`MLPClassifier`), métricas, matriz de confusión |
| MinHash + LSH (MLlib) | Deduplicación por similitud Jaccard y coseno |
| Parquet + Snappy | Almacenamiento columnar eficiente |
| matplotlib + seaborn | Visualización de métricas |
| Jupyter Notebook | Entorno de desarrollo interactivo |

---

## Temas del curso cubiertos

- Almacenamiento distribuido (simulación HDFS, particionado, replicación)
- Algoritmos MapReduce y modelo costo–comunicación
- TF-IDF e índice invertido
- Medidas de similitud (Jaccard, coseno) y LSH
- Redes neuronales (MLP, ReLU, Softmax, backpropagation con Adam)
- Evaluación de modelos (F1, accuracy, precisión, recall por clase)
- Complejidad algorítmica y escalabilidad

---
