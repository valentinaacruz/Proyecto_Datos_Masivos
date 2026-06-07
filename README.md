# Clasificador Masivo de Noticias

**Equipo:** DataWhales  
**Curso:** Datos Masivos I — Licenciatura en Ciencia de Datos, UNAM, semestre 2026-2  
**Integrantes:** 

| Nombre |
|--------|
| Cardon Carrillo Pedro Manuel |
| Cruz Mendoza Valentina Ayelen |
| Rivera Hernandez Milena Fernanda |

---

## Descripción general

Pipeline de procesamiento de lenguaje natural a gran escala para clasificar noticias en 18 categorías temáticas. El sistema procesa dos datasets públicos (~5 GB combinados) usando PySpark 3.5 en modo local, lo que permite demostrar el paradigma de procesamiento distribuido con el mismo código que escalaría a un cluster real.

**Nota de implementación:** El proyecto corre íntegramente en modo `local[*]` sobre una laptop (MacBook, 10 GB RAM asignados al driver). No hay cluster real ni HDFS — los datos viven en el filesystem local y el paralelismo proviene de los múltiples cores de la misma máquina. Cuando la documentación menciona "simulación de HDFS" se refiere a que el código y la estructura de archivos Parquet son idénticos a los que se usarían en producción; solo cambia la URL del master de Spark.

---

## Datasets

| Dataset | Tamaño crudo | Documentos ingestados | Etiquetas | Uso en el pipeline |
|---|---|---|---|---|
| **CC-News** (HuggingFace `vblagoje/cc_news`) | ~3.5 GB | 703,488 | ✗ | Fase 3 — deduplicación LSH |
| **MIND Large** (Microsoft News Dataset) | ~144 MB (TSV) | 173,550 | ✓ 18 categorías | Fases 2 y 4 — TF-IDF + clasificador |
| **Total ingestado** | | **877,038** | | |

### Categorías MIND (distribución real)

| Categoría | Docs | Categoría | Docs |
|---|---|---|---|
| sports | 53,599 | weather | 7,046 |
| news | 52,304 | autos | 5,519 |
| finance | 10,247 | health | 5,306 |
| travel | 8,336 | tv | 2,402 |
| lifestyle | 8,015 | music | 2,179 |
| foodanddrink | 7,876 | entertainment | 1,561 |
| video | 7,536 | movies | 1,492 |

> Las categorías `kids` (125), `middleeast` (4), `games` (2) y `northamerica` (1) tienen muy pocos ejemplos, lo que afecta el F1 de esas clases.

---

## Estructura del repositorio

```
Proyecto_Datos_Masivos/
├── notebooks/
│   ├── 01_ingesta.ipynb          # Descarga, limpieza y guardado en Parquet
│   ├── 02_preprocesamiento.ipynb # Tokenización MapReduce + TF-IDF
│   ├── 03_lsh_dedup.ipynb        # Deduplicación con MinHash + LSH
│   ├── 04_clasificador.ipynb     # MLP con sklearn sobre vectores TF-IDF
│   └── 05_evaluacion.ipynb       # Métricas + análisis costo-comunicación
├── datos/
│   ├── MINDlarge_train/          # TSV crudos de MIND (train)
│   ├── MINDlarge_dev/            # TSV crudos de MIND (dev)
│   ├── cc_news/                  # Parquet crudo de CC-News (se genera al correr fase 1)
│   └── processed/
│       ├── cc_news/              # Parquet limpio CC-News (703,488 docs, 18 particiones)
│       ├── mind_large/           # Parquet MIND particionado por categoría
│       ├── tokens_cc/            # Tokens CC-News (muestra 100k para LSH)
│       ├── tfidf_mind/           # Vectores TF-IDF MIND 65,536-dim
│       └── predicciones/         # Salida del clasificador (5,000 docs de test)
├── models/
│   ├── idf_model_mind/           # Modelo IDF serializado (PySpark MLlib)
│   └── mlp_sklearn.joblib        # Clasificador MLP serializado (sklearn)
└── resultados/
    ├── confusion_matrix.png
    └── escalabilidad.png
```

---

## Pipeline — 5 fases

### Fase 1 — Ingesta y almacenamiento (`01_ingesta.ipynb`)

Descarga CC-News desde HuggingFace y carga MIND desde TSV. Ambos datasets se limpian (filtro de nulos y textos < 100 caracteres) y se guardan en Parquet con compresión Snappy.

**Lo que hace Spark aquí:**
- CC-News: 8 particiones Parquet (simulación de bloques distribuidos)
- MIND: particionado por columna `category` (18 carpetas, una por clase)

**Formato de almacenamiento — por qué Parquet:**
- Lectura columnar: solo carga las columnas necesarias en cada fase, reduciendo el I/O
- Compresión Snappy integrada (default de PySpark): rápida de descomprimir, viable para procesamiento en paralelo
- Predicado pushdown: filtra filas antes de deserializar

**Resultado:** 703,488 docs CC-News (1.1 GB Parquet) + 173,550 docs MIND (64 MB Parquet)

---

### Fase 2 — Preprocesamiento MapReduce + TF-IDF (`02_preprocesamiento.ipynb`)

Tokenización y vectorización del corpus usando las APIs nativas de PySpark MLlib (JVM puro, sin UDFs Python, 10–20× más rápido).

**Pipeline de procesamiento:**
1. `RegexTokenizer` — tokenización por regex `[^a-z]+`, minúsculas, tokens ≥ 3 chars (MAP)
2. `StopWordsRemover` — eliminación de 198 stopwords en inglés (COMBINER)
3. `HashingTF` — vectores TF dispersos de 65,536 dimensiones (2^16)
4. `IDF.fit()` — ajuste global del IDF sobre MIND (**shuffle global**: cada partición agrega conteos locales, se shufflean al reducer, que calcula `idf(t) = log(N / df(t))`)

**Qué se vectoriza con TF-IDF:**
- MIND Large completo (173,550 docs) → vectores 65,536-dim → guardado en `processed/tfidf_mind/`
- CC-News: solo se guardan tokens (sin TF-IDF) porque MinHash LSH no los necesita

**Muestra CC-News para LSH:** de los 703,488 docs disponibles, se guardan 100,000 tokens en `processed/tokens_cc/`. El corpus completo tardaría varias horas en local para la fase de auto-join de LSH.

**Resultado:** vectores TF-IDF MIND (4 particiones Parquet) + tokens CC-News (muestra 100k)

---

### Fase 3 — Deduplicación con MinHash + LSH (`03_lsh_dedup.ipynb`)

Detecta y elimina noticias casi idénticas en CC-News usando dos métricas de similitud complementarias y Union-Find para agrupar duplicados transitivos.

**Corpus de entrada:** muestra de 10,000 docs de CC-News (de los 100,000 disponibles). El `approxSimilarityJoin` es un self-join cuadrático en el peor caso; 100k docs tardarían horas en local.

**Dos métricas de similitud:**

| Método | Similitud | Umbral | Implementación |
|---|---|---|---|
| MinHash LSH | Jaccard | ≥ 0.85 | `MinHashLSH`, 3 tablas hash |
| Random Projection LSH | Coseno | ≥ 0.90 | `BucketedRandomProjectionLSH` sobre vectores L2-normalizados |

**Por qué 3 tablas hash:** con `numHashTables=3` y similitud s=0.85, la probabilidad de detectar un par duplicado es P(0.85) = 1-(1-0.85)³ ≈ 99.7%. Con 1 sola tabla sería solo 85%.

**Union-Find en el driver:** los 4,343 pares de duplicados detectados se traen al driver con `collect()` y se procesan con Union-Find en Python (compresión de caminos + union by rank, O(α(n)) amortizado). Esto agrupa duplicados transitivos: si A≈B y B≈C, los tres forman un mismo cluster.

**Resultado real:**
- Documentos procesados: 10,000
- Pares duplicados detectados: 4,343
- Clusters de duplicados: 229
- Documentos eliminados: 536 (5.36%)
- Corpus deduplicado: 9,464 docs

---

### Fase 4 — Clasificador MLP (`04_clasificador.ipynb`)

Clasificador de noticias en 18 categorías entrenado sobre los vectores TF-IDF de MIND Large.

**Decisión de implementación — sklearn vs PySpark MLlib:**  
PySpark MLlib `MultilayerPerceptronClassifier` materializa bloques densos en la JVM. Con vectores de 65,536 dimensiones genera bloques de ~33 MB por partición → `OutOfMemoryError` en local. Se usa `sklearn MLPClassifier` porque acepta matrices `scipy.sparse.csr_matrix` directamente, sin densificar en memoria.

**Flujo híbrido Spark + sklearn:**
1. Spark carga los vectores TF-IDF desde Parquet y codifica etiquetas con `StringIndexer`
2. `ChiSqSelector` reduce 65,536 → **4,096 features** (las de mayor correlación chi-cuadrado con la categoría). Esto corre en Spark distribuido sobre las 172,422 filas
3. Split aleatorio 80/20 con `seed=42`: 137,887 train / 34,535 test
4. `spark_to_scipy()` extrae una muestra de 20,000 train y 5,000 test al driver como matrices sparse
5. sklearn entrena el MLP sobre esa muestra (sin JVM)
6. Las predicciones regresan a Spark y se guardan en Parquet

**Arquitectura MLP entrenada:**
```
Entrada (4,096) → Oculta (128, ReLU) → Salida (18, Softmax)
Optimizador: Adam  |  max_iter: 30  |  Parámetros: 526,738
```

> El modelo no converge en 30 iteraciones (sklearn emite `ConvergenceWarning`). En producción se aumentaría `max_iter` o se usaría early stopping.

**Resultado:**
- Accuracy train: 0.9964 (sobreajuste esperado con 30 iteraciones en muestra pequeña)
- Accuracy test: 0.7560
- F1 ponderado: 0.7543
- Modelo guardado: `models/mlp_sklearn.joblib`

---

### Fase 5 — Evaluación y análisis de complejidad (`05_evaluacion.ipynb`)

Análisis de métricas del clasificador y complejidad algorítmica del pipeline completo.

**Métricas por categoría (test, 5,000 docs):**

| Categoría | F1 | Soporte |
|---|---|---|
| sports | 0.90 | 1,550 |
| news | 0.78 | 1,542 |
| foodanddrink | 0.72 | 223 |
| weather | 0.74 | 224 |
| autos | 0.68 | 165 |
| finance | 0.59 | 279 |
| travel | 0.56 | 237 |
| kids | 0.00 | 1 |

**Análisis de complejidad del pipeline:**

| Fase | Algoritmo clave | Complejidad | Shuffle |
|---|---|---|---|
| Ingesta | Lectura + particionado | O(N) | O(N) escritura |
| TF (Fase 2) | HashingTF por doc | O(N · \|tokens\|) | No |
| IDF (Fase 2) | Conteo global | O(N · \|V\|) | **Sí** — shuffle de conteos |
| ChiSqSelector (Fase 4) | Test chi-cuadrado | O(N · \|V\|) | **Sí** |
| MinHash (Fase 3) | Firma por doc | O(N · K · \|D\|) | O(N · K) |
| LSH join (Fase 3) | approxSimilarityJoin | O(C), C ≪ N² | O(C) |
| Union-Find (Fase 3) | Componentes conexas | O(\|E\| · α(N)) | O(\|E\|) collect |
| MLP fit (Fase 4) | Backprop Adam | O(iter · muestra · W) | 0 — driver solo |

**Ventaja LSH vs fuerza bruta:** para N=500k docs, fuerza bruta requiere ~1.25×10¹¹ comparaciones. LSH lo reduce a O(N·b) en la fase de buckets más O(C) verificaciones de candidatos.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Motor de procesamiento | PySpark 3.5.1, modo `local[*]` |
| Formato de almacenamiento | Apache Parquet + compresión Snappy |
| Tokenización / stopwords | PySpark MLlib `RegexTokenizer`, `StopWordsRemover` |
| Vectorización | PySpark MLlib `HashingTF`, `IDF`, `ChiSqSelector` |
| Deduplicación | PySpark MLlib `MinHashLSH`, `BucketedRandomProjectionLSH` |
| Clasificador | **sklearn `MLPClassifier`** + `scipy.sparse` |
| Serialización de modelos | PySpark MLlib (modelo IDF) + `joblib` (modelo MLP) |
| Lenguaje | Python 3.12.5 |
| Entorno | Jupyter Notebook / VS Code, venv |

---

## Temas del curso cubiertos

- Almacenamiento distribuido
- Paradigma MapReduce: tokenización (MAP), eliminación de stopwords (COMBINER), IDF (REDUCE con shuffle global)
- Modelo costo-comunicación: análisis por fase de costo computacional y de comunicación
- Medidas de similitud: Jaccard (MinHash) y coseno (Random Projection)
- Locality-Sensitive Hashing: MinHash LSH + Bucketed Random Projection LSH
- Estructuras de datos para grafos: Union-Find con compresión de caminos y union by rank

---

## Cómo reproducir

```bash
# 1. Clonar el repositorio
git clone https://github.com/valentinaacruz/Proyecto_Datos_Masivos.git
cd Proyecto_Datos_Masivos

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate
pip install pyspark==3.5.1 nltk scikit-learn scipy joblib datasets matplotlib seaborn

# 3. Descargar MIND Large manualmente
# https://msnews.github.io/ → MINDlarge_train.zip + MINDlarge_dev.zip
# Descomprimir en datos/MINDlarge_train/ y datos/MINDlarge_dev/


# 4. Verifica la instalación

python3 -c "import pyspark; print('PySpark', pyspark.__version__)"
python3 -c "import nltk; print('NLTK OK')"
python3 -c "import sklearn; print('scikit-learn OK')"
python3 -c "import datasets; print('Datasets OK')"



# 5. Correr los notebooks en orden
# CC-News se descarga automáticamente desde HuggingFace en la Fase 1
jupyter notebook notebooks/01_ingesta.ipynb
jupyter notebook notebooks/02_preprocesamiento.ipynb
jupyter notebook notebooks/03_lsh_dedup.ipynb
jupyter notebook notebooks/04_clasificador.ipynb
jupyter notebook notebooks/05_evaluacion.ipynb
```

**Requisitos mínimos de hardware:** 16 GB RAM recomendados (10 GB asignados al driver de Spark), ~8 GB de espacio en disco.
