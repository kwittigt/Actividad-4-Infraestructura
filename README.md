# Actividad 4.4: Optimización de Pipelines de Datos con Python

Este proyecto es una demostración práctica de diversas técnicas de ingeniería de datos para procesar grandes volúmenes de información de manera eficiente utilizando Python y librerías del ecosistema de PyData. A través de una serie de experimentos, se analiza el rendimiento (tiempo y memoria) de diferentes estrategias para la lectura, procesamiento y almacenamiento de un dataset simulado de consumo eléctrico.

## 🎯 Objetivo

El objetivo principal es ilustrar y cuantificar el impacto de decisiones de diseño clave en un pipeline de datos, tales como:
- La especificación de tipos de datos (`dtype`) al leer archivos.
- El procesamiento de datos en fragmentos (`chunks`) vs. monolítico.
- El uso de formatos de almacenamiento columnar como Parquet.
- El particionamiento de datos para optimizar consultas.
- La aplicación de motores de procesamiento paralelo como Dask y Polars.

## 📂 Estructura del Proyecto

Al ejecutar el script, se generará la siguiente estructura de directorios:

```
.
├── mainfinal4.py         # Script principal con los experimentos
├── requirements.txt      # Dependencias del proyecto
├── datos/                # Directorio para el dataset generado
│   └── consumo_electrico.csv
├── datos_parquet/        # Directorio para los archivos Parquet
│   ├── consumo_completo.parquet
│   └── particionado_jerarquico/
│       └── year=.../
│           └── month=.../
│               └── comuna=.../
└── visualizaciones/      # Directorio para las gráficas generadas
    ├── fig1_tiempo_lectura.png
    ├── fig2_memoria_lectura.png
    ├── fig3_chunksize.png
    ├── fig4_comparacion_formatos.png
    └── fig5_consumo_por_comuna.png
```

## 🛠️ Requisitos

Para ejecutar el proyecto, necesitas Python 3.8+ y las siguientes librerías. Se recomienda crear un entorno virtual.

Puedes instalar todas las dependencias con el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

Para ejecutar todos los experimentos y generar los resultados, simplemente corre el script principal desde tu terminal:

```bash
python mainfinal4.py
```

El script se encargará de:
1.  Generar el archivo `consumo_electrico.csv` si no existe.
2.  Ejecutar los experimentos B, C y D.
3.  Imprimir los resultados de rendimiento en la consola.
4.  Guardar las visualizaciones en la carpeta `visualizaciones/`.

## 🔬 Experimentos

El script está dividido en cuatro experimentos principales:

### Experimento A: Generación del Dataset
- **Acción:** Crea un archivo CSV (`consumo_electrico.csv`) con **1,000,000 de registros** simulados de consumo eléctrico.
- **Propósito:** Generar un dataset lo suficientemente grande como para que las optimizaciones de memoria y tiempo sean medibles y significativas.

### Experimento B: Lectura Monolítica vs. por Fragmentos
- **Acción:** Compara el tiempo y el uso de memoria pico al leer el CSV de dos formas:
    1.  **Monolítica:** Cargando el archivo completo en un solo DataFrame de Pandas.
    2.  **Chunked:** Leyendo el archivo en fragmentos de diferentes tamaños (50k, 200k, 500k filas).
- **Propósito:** Demostrar cómo el procesamiento por `chunks` permite manejar archivos más grandes que la RAM disponible, controlando el consumo de memoria.

### Experimento C: Pipeline Incremental y Alternativas
- **Acción:** Implementa un pipeline de procesamiento de datos (filtrado y agregación) sobre los `chunks` del CSV.
- **Demostración:** Compara el rendimiento de este bucle secuencial de Pandas con dos alternativas de alto rendimiento:
    - **Dask:** Una librería de computación paralela que ejecuta el mismo pipeline sobre múltiples núcleos.
    - **Polars:** Un motor de DataFrames escrito en Rust que utiliza paralelismo columnar y de tareas para una ejecución ultra-rápida.
- **Propósito:** Mostrar cómo las librerías modernas pueden acelerar drásticamente los pipelines de ETL en un solo nodo.

### Experimento D: Parquet, Particionamiento y Arquitectura
- **Acción:**
    1.  Convierte el CSV a formato Parquet, un formato de almacenamiento columnar optimizado para analítica.
    2.  Crea dos versiones: un archivo monolítico y un dataset particionado jerárquicamente (`year/month/comuna`).
    3.  Compara el rendimiento de una consulta de agregación sobre el CSV, el Parquet monolítico y el Parquet particionado.
- **Propósito:** Ilustrar las ventajas de Parquet en términos de compresión (menor tamaño en disco) y velocidad de consulta, especialmente cuando los datos están particionados (*predicate pushdown* y *partition pruning*).

## 📊 Resultados y Conclusiones

Al final de la ejecución, el script presenta una tabla con recomendaciones técnicas basadas en los resultados de los experimentos, aplicables a diferentes escenarios de escalabilidad de datos:

- **< 10 GB:** Pandas con tipos de datos optimizados es suficiente.
- **10-500 GB:** Un Data Lake con Parquet particionado es ideal para consultas ad-hoc.
- **Pipelines en un solo nodo:** Dask o Polars ofrecen un rendimiento superior al bucle de `chunks` de Pandas.
- **Análisis en tiempo real:** Se recomiendan arquitecturas de streaming (Kafka, Flink).
- **> 500 GB (escala Petabyte):** Se sugiere una arquitectura Lakehouse (Delta Lake, Iceberg) para combinar flexibilidad y garantías ACID.

Este proyecto sirve como una guía práctica para tomar decisiones informadas al diseñar y construir sistemas de procesamiento de datos en Python.
