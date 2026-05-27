
import os
import platform
import shutil
import sys
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=UserWarning,    module="pyarrow")
warnings.filterwarnings("ignore", category=FutureWarning,  module="pyarrow")
warnings.filterwarnings("ignore", category=UserWarning,    module="dask")
warnings.filterwarnings("ignore", category=FutureWarning,  module="dask")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORES = sns.color_palette("muted", 8)

DIR_BASE    = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS   = os.path.join(DIR_BASE, "datos")
DIR_PARQUET = os.path.join(DIR_BASE, "datos_parquet")
DIR_IMGS    = os.path.join(DIR_BASE, "visualizaciones")

for d in [DIR_DATOS, DIR_PARQUET, DIR_IMGS]:
    os.makedirs(d, exist_ok=True)

ARCHIVO_CSV = os.path.join(DIR_DATOS, "consumo_electrico.csv")

DTYPE_MAP = {
    "comuna"        : "category",
    "edificio"      : "category",
    "estado"        : "category",
    "tipo_cliente"  : "category",
    "consumo_kwh"   : "float32",
    "voltaje_v"     : "float32",
    "temperatura_amb": "float32",
    "tarifa_clp"    : "int64",
}


def leer_csv_optimizado(usecols=None) -> pd.DataFrame:
    """
    ▶ MEJORA 1: Lectura con dtype explícito.
    Equivalente de pd.read_csv pero con control total sobre los tipos.

    Parámetros
    ----------
    usecols : list[str] | None
        Subconjunto de columnas a leer. Si es None, lee todas.
    """
    dtype_activo = (
        {k: v for k, v in DTYPE_MAP.items() if k in usecols}
        if usecols else DTYPE_MAP
    )

    parse_ts = ["timestamp"] if (usecols is None or "timestamp" in usecols) else []

    return pd.read_csv(
        ARCHIVO_CSV,
        dtype       = dtype_activo,
        parse_dates = parse_ts,
        usecols     = usecols,
    )



def medir(func):
    """
    Decorador que mide tiempo de ejecución y memoria pico.

    NOTA METODOLÓGICA:
      tracemalloc solo captura asignaciones gestionadas por el GC de Python.
      No mide buffers de extensiones C (NumPy/Arrow), caché del SO ni copias
      en la frontera Python-C → los valores reportados son una cota inferior
      conservadora (~3-4x menor que el RSS real del proceso).
      Se añade medición de RSS vía psutil como referencia complementaria.
    """
    def wrapper(*args, **kwargs):
        proc = psutil.Process()
        rss_antes = proc.memory_info().rss
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            resultado = func(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            _, pico_gc = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        rss_delta = proc.memory_info().rss - rss_antes
        pico_mb = round(pico_gc / 1_048_576, 2)
        rss_mb  = round(rss_delta / 1_048_576, 2)
        return resultado, round(t1 - t0, 4), pico_mb, rss_mb
    return wrapper


def info_sistema():
    """Imprime información del entorno experimental."""
    ram_total = psutil.virtual_memory().total / 1_073_741_824
    ram_disp  = psutil.virtual_memory().available / 1_073_741_824
    disco_lib = psutil.disk_usage(DIR_BASE).free / 1_073_741_824

    print("=" * 60)
    print("  ENTORNO EXPERIMENTAL")
    print("=" * 60)
    print(f"  Sistema operativo : {platform.system()} {platform.release()}")
    print(f"  Versión Python    : {sys.version.split()[0]}")
    print(f"  RAM total         : {ram_total:.1f} GB")
    print(f"  RAM disponible    : {ram_disp:.1f} GB")
    print(f"  Espacio en disco  : {disco_lib:.1f} GB libres")
    print(f"  Librerías clave   : pandas {pd.__version__}, "
          f"numpy {np.__version__}, pyarrow {pa.__version__}")
    print(f"  Fecha de ejecución: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)


def guardar_figura(nombre):
    ruta = os.path.join(DIR_IMGS, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"  [OK] Figura guardada → {ruta}")
    plt.close()



def experimento_a(n_filas: int = 1_000_000, semilla: int = 42) -> None:
    """
    Genera un CSV con n_filas registros simulados de consumo eléctrico.

    Parámetros
    ----------
    n_filas : int
        Número de filas a generar.
    semilla : int
        Semilla del generador aleatorio. Valor por defecto: 42.
        Documentar explícitamente para reproducibilidad científica:
        cambiar la semilla altera las distribuciones generadas y, por tanto,
        las métricas de los experimentos B, C y D.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO A — Generación del dataset")
    print("=" * 60)

    if os.path.exists(ARCHIVO_CSV):
        muestra = pd.read_csv(ARCHIVO_CSV, usecols=["comuna"], nrows=200)
        if {"Maipú", "Ñuñoa"} & set(muestra["comuna"].unique()):
            print("  CSV antiguo (con tildes) detectado. Regenerando...")
            os.remove(ARCHIVO_CSV)
        else:
            tam = os.path.getsize(ARCHIVO_CSV) / 1_048_576
            print(f"  El archivo ya existe ({tam:.1f} MB). Se omite generación.")
            return

    rng = np.random.default_rng(semilla)

    comunas      = ["Santiago", "Providencia", "Nunoa", "Las Condes",
                    "Maipu", "Pudahuel", "La Florida", "Vitacura"]
    tipos        = ["Residencial", "Comercial", "Industrial", "Gubernamental"]
    estados      = ["Normal", "Alerta", "Falla", "Mantenimiento"]
    probs_estado = [0.88, 0.07, 0.03, 0.02]

    n_medidores = 2_000
    n_edificios = 200

    medidores = [f"MED-{i:05d}" for i in rng.choice(n_medidores, n_filas)]
    edificios = [f"EDIF-{i:03d}" for i in rng.choice(n_edificios, n_filas)]

    timestamps = pd.date_range(
        start="2023-01-01 00:00:00",
        periods=n_filas,
        freq="30s",
        tz="UTC",
    )

    hora = timestamps.hour.values
    factor_hora = 0.5 + 0.5 * np.sin((hora - 6) * np.pi / 12)
    consumo = (rng.exponential(scale=5, size=n_filas) * (1 + factor_hora)).round(3)

    df = pd.DataFrame({
        "timestamp"      : timestamps,
        "medidor_id"     : medidores,
        "edificio"       : edificios,
        "comuna"         : rng.choice(comunas, n_filas),
        "tipo_cliente"   : rng.choice(tipos, n_filas, p=[0.50, 0.30, 0.15, 0.05]),
        "consumo_kwh"    : consumo,
        "voltaje_v"      : (rng.normal(220, 5, n_filas)).round(2),
        "temperatura_amb": (rng.normal(18, 6, n_filas)).round(1),
        "estado"         : rng.choice(estados, n_filas, p=probs_estado),
        "tarifa_clp"     : (consumo * rng.uniform(95, 130, n_filas)).round(0),
    })

    t0 = time.perf_counter()
    df.to_csv(ARCHIVO_CSV, index=False)
    t_gen = time.perf_counter() - t0

    tam_mb = os.path.getsize(ARCHIVO_CSV) / 1_048_576
    print(f"  Filas            : {len(df):,}")
    print(f"  Columnas         : {len(df.columns)}")
    print(f"  Tamaño en disco  : {tam_mb:.1f} MB")
    print(f"  Tiempo generación: {t_gen:.2f} s")
    print(f"  Tipos de datos   :")
    for col, dtype in df.dtypes.items():
        print(f"      {col:<18} {dtype}")



CHUNK_SIZES = [50_000, 200_000, 500_000]

@medir
def leer_completo():
    return leer_csv_optimizado()

@medir
def leer_por_chunks(chunksize):
    dtype_activo = DTYPE_MAP.copy()
    fragmentos = []
    for chunk in pd.read_csv(
        ARCHIVO_CSV,
        chunksize   = chunksize,
        dtype       = dtype_activo,
        parse_dates = ["timestamp"],
    ):
        fragmentos.append(chunk)
    return len(fragmentos)


def experimento_b() -> pd.DataFrame:
    """
    Compara lectura monolítica vs. chunked con 3 tamaños distintos.
    Mide tiempo y memoria pico.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO B — Lectura monolítica vs. por fragmentos")
    print("=" * 60)

    registros = []

    print("  → Lectura monolítica (con dtypes optimizados)...")
    _, t, mem_gc, mem_rss = leer_completo()
    registros.append({
        "estrategia" : "Monolítica",
        "chunksize"  : "Completo",
        "tiempo_s"   : t,
        "memoria_mb" : mem_gc,
        "rss_mb"     : mem_rss,
    })
    print(f"     Tiempo: {t:.3f} s | Memoria GC pico: {mem_gc:.1f} MB | RSS delta: {mem_rss:.1f} MB")

    for cs in CHUNK_SIZES:
        print(f"  → Lectura chunked (chunksize={cs:,}, dtypes optimizados)...")
        _, t, mem_gc, mem_rss = leer_por_chunks(cs)
        registros.append({
            "estrategia" : "Chunked",
            "chunksize"  : f"{cs:,}",
            "tiempo_s"   : t,
            "memoria_mb" : mem_gc,
            "rss_mb"     : mem_rss,
        })
        print(f"     Tiempo: {t:.3f} s | Memoria GC pico: {mem_gc:.1f} MB | RSS delta: {mem_rss:.1f} MB")

    df_res = pd.DataFrame(registros)
    print("\n  Tabla resumen Experimento B:")
    print(df_res.to_string(index=False))
    return df_res



def demo_dask_alternativa():
    """
    DEMO — Reemplazo distribuido del pipeline incremental usando Dask.
    Requiere: pip install dask[dataframe]

    Equivalencia funcional con pipeline_incremental():
      - Filtra estado != 'Falla' y consumo > 0
      - Agrega consumo_total y n_registros por comuna + tipo_cliente

    DIFERENCIA CLAVE vs. chunksize loop:
      ddf.groupby().agg() construye un grafo dirigido acíclico (DAG).
      .compute() ejecuta ese DAG en paralelo sobre todos los núcleos disponibles.
    """
    try:
        import dask.dataframe as dd
    except ImportError:
        print("  [SKIP] Dask no está instalado. Ejecuta: pip install dask[dataframe]")
        return

    print("\n  ── DEMO DASK (alternativa distribuida) ──────────────────")
    t0 = time.perf_counter()

    ddf = dd.read_csv(
        ARCHIVO_CSV,
        dtype       = DTYPE_MAP,
        parse_dates = ["timestamp"],
        blocksize   = "64MB",
    )

    ddf_filtrado = ddf[
        (ddf["estado"] != "Falla") &
        (ddf["consumo_kwh"] > 0)
    ]

    resultado_dask = (
        ddf_filtrado
        .groupby(["comuna", "tipo_cliente"])
        .agg({"consumo_kwh": ["sum", "count"]})
    )

    df_resultado = resultado_dask.compute()

    t1 = time.perf_counter()
    print(f"  Tiempo total Dask (paralelo): {t1 - t0:.3f} s")
    print(f"  Filas en resultado          : {len(df_resultado):,}")
    print(f"  (vs. bucle chunksize secuencial en Experimento C)")


def demo_polars_alternativa():
    """
    DEMO — Reemplazo de alta performance usando Polars.
    Requiere: pip install polars

    Polars ejecuta el pipeline completo en un solo pase multihilo
    sin necesidad de dividir manualmente en chunks.
    Su motor de ejecución usa SIMD y paralelismo columnar en Rust.
    """
    try:
        import polars as pl
    except ImportError:
        print("  [SKIP] Polars no está instalado. Ejecuta: pip install polars")
        return

    print("\n  ── DEMO POLARS (alternativa columnar/paralela) ──────────")
    t0 = time.perf_counter()

    POLARS_SCHEMA = {
        "comuna"         : pl.Categorical,
        "edificio"       : pl.Categorical,
        "estado"         : pl.Categorical,
        "tipo_cliente"   : pl.Categorical,
        "consumo_kwh"    : pl.Float32,
        "voltaje_v"      : pl.Float32,
        "temperatura_amb": pl.Float32,
        "tarifa_clp"     : pl.Float32,
    }

    resultado_polars = (
        pl.scan_csv(
            ARCHIVO_CSV,
            schema_overrides = POLARS_SCHEMA,
            try_parse_dates  = True,
        )
        .filter(
            (pl.col("estado") != "Falla") &
            (pl.col("consumo_kwh") > 0)
        )
        .group_by(["comuna", "tipo_cliente"])
        .agg([
            pl.col("consumo_kwh").sum().alias("consumo_total"),
            pl.col("consumo_kwh").count().alias("n_registros"),
        ])
        .sort("consumo_total", descending=True)
        .collect()
    )

    t1 = time.perf_counter()
    print(f"  Tiempo total Polars (multihilo): {t1 - t0:.3f} s")
    print(f"  Filas en resultado             : {len(resultado_polars):,}")
    print(resultado_polars.head(5))



@medir
def pipeline_incremental(chunksize: int = 200_000):
    """
    Pipeline incremental con chunksize.
    ▶ MEJORA 1 aplicada: dtype explícito en la lectura de cada chunk.
    """
    parciales_consumo = []
    parciales_voltaje = []
    total_filas_proc   = 0
    total_filas_filtro = 0

    cols = ["timestamp", "comuna", "tipo_cliente",
            "consumo_kwh", "voltaje_v", "estado"]

    dtype_cols = {k: v for k, v in DTYPE_MAP.items() if k in cols}

    for chunk in pd.read_csv(
        ARCHIVO_CSV,
        usecols     = cols,
        chunksize   = chunksize,
        dtype       = dtype_cols,
        parse_dates = ["timestamp"],
    ):
        total_filas_proc += len(chunk)

        chunk = chunk[
            (chunk["estado"] != "Falla") &
            (chunk["consumo_kwh"] > 0)
        ].copy()
        total_filas_filtro += len(chunk)


        agg1 = chunk.groupby(["comuna", "tipo_cliente"], observed=True).agg(
            consumo_total=("consumo_kwh", "sum"),
            n_registros  =("consumo_kwh", "count"),
        )
        parciales_consumo.append(agg1)

        agg2 = chunk.groupby(["comuna"], observed=True).agg(
            suma_voltaje=("voltaje_v", "sum"),
            n_volt      =("voltaje_v", "count"),
        )
        parciales_voltaje.append(agg2)

    df_consumo = (
        pd.concat(parciales_consumo)
        .groupby(level=[0, 1], observed=True)
        .sum()
        .reset_index()
    )
    df_consumo["consumo_promedio_kwh"] = (
        df_consumo["consumo_total"] / df_consumo["n_registros"]
    ).round(4)

    df_voltaje = (
        pd.concat(parciales_voltaje)
        .groupby(level=0, observed=True)
        .sum()
    )
    df_voltaje["voltaje_promedio_v"] = (
        df_voltaje["suma_voltaje"] / df_voltaje["n_volt"]
    ).round(3)
    df_voltaje = df_voltaje[["voltaje_promedio_v"]].reset_index()

    return {
        "consumo"            : df_consumo,
        "voltaje"            : df_voltaje,
        "filas_procesadas"   : total_filas_proc,
        "filas_tras_filtrado": total_filas_filtro,
    }


def experimento_c():
    print("\n" + "=" * 60)
    print("  EXPERIMENTO C — Pipeline incremental")
    print("=" * 60)

    resultado, t, mem, _ = pipeline_incremental(200_000)

    print(f"  Tiempo total      : {t:.3f} s")
    print(f"  Memoria pico      : {mem:.1f} MB")
    print(f"  Filas procesadas  : {resultado['filas_procesadas']:,}")
    print(f"  Filas tras filtro : {resultado['filas_tras_filtrado']:,}")

    print("\n  Consumo por comuna y tipo de cliente:")
    print(resultado["consumo"].sort_values("consumo_total", ascending=False)
          .head(12).to_string(index=False))

    print("\n  Voltaje promedio por comuna:")
    print(resultado["voltaje"].to_string(index=False))

    print("\n" + "─" * 60)
    print("  ALTERNATIVAS DISTRIBUIDAS (MEJORA 3)")
    print("─" * 60)
    demo_dask_alternativa()
    demo_polars_alternativa()

    return resultado, t, mem



def _normalizar_columna_string(tabla: pa.Table, col: str) -> pa.Table:
    """
    Normaliza una columna de strings en una Arrow Table:
    - Convierte a minúsculas ASCII (sin tildes) para que los directorios
      Hive-style sean compatibles con Athena/Glue Crawler.
    - 'Ñuñoa' → 'nunoa', 'Las Condes' → 'las_condes'
    Sin esta normalización, pq.write_to_dataset() crea directorios con tildes
    o espacios (comuna=Ñuñoa/) que Athena URL-encodea de forma inconsistente,
    generando particiones duplicadas o fallos silenciosos en el Glue Crawler.
    """
    import unicodedata

    def _limpiar(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.lower().replace(" ", "_")

    col_idx = tabla.schema.get_field_index(col)
    if col_idx == -1:
        return tabla
    nueva_col = pa.array(
        [_limpiar(v.as_py()) if v.as_py() is not None else None
         for v in tabla.column(col)],
        type=pa.string(),
    )
    return tabla.set_column(col_idx, col, nueva_col)


def _escribir_parquet_monolitico(tabla: pa.Table, dir_parquet: str) -> str:
    """Escribe un único archivo Parquet y devuelve su ruta."""
    ruta = os.path.join(dir_parquet, "consumo_completo.parquet")
    pq.write_table(tabla, ruta, compression="snappy")
    return ruta


def _escribir_parquet_particionado(tabla: pa.Table, dir_parquet: str) -> str:
    """
    Escribe el dataset particionado year/month/comuna y devuelve la ruta raíz.

    ADVERTENCIA sobre existing_data_behavior='overwrite_or_ignore':
      En PyArrow ≥ 21.x, el comportamiento exacto depende de la versión y
      puede dejar datos parciales o duplicados si el job se re-ejecuta sobre
      un rango temporal que ya existe. NO es idempotente de forma garantizada.
      En producción con cargas diarias se recomienda Delta Lake o Apache Iceberg,
      que ofrecen garantías ACID y solventan exactamente este problema.
    """
    ruta = os.path.join(dir_parquet, "particionado_jerarquico")
    pq.write_to_dataset(
        tabla,
        root_path      = ruta,
        partition_cols = ["year", "month", "comuna"],
        compression    = "snappy",
        existing_data_behavior = "overwrite_or_ignore",
    )
    return ruta


@medir
def convertir_a_parquet():
    """
    Convierte el CSV a Parquet monolítico y particionado.

    CORRECCIÓN
      Se usa pyarrow.csv.read_csv() directamente en lugar de leer primero con
      Pandas y luego convertir con pa.Table.from_pandas(). La ruta original
      tenía ambas estructuras (DataFrame Pandas ~150 MB + Arrow Table ~150 MB)
      coexistiendo en RAM antes de que el GC liberara la primera → no escala
      a datasets de TBs.

    CORRECCIÓN
      Las comunas se normalizan a ASCII-lowercase antes de escribir para
      garantizar compatibilidad con Athena, Glue Crawler y sistemas POSIX.

    CORRECCIÓN
      Esta función orquesta; la lógica de escritura está en funciones privadas
      (_escribir_parquet_monolitico, _escribir_parquet_particionado).
    """
    import pyarrow.csv as pa_csv

    convert_options = pa_csv.ConvertOptions(
        column_types={
            "comuna"         : pa.dictionary(pa.int32(), pa.string()),
            "edificio"       : pa.dictionary(pa.int32(), pa.string()),
            "estado"         : pa.dictionary(pa.int32(), pa.string()),
            "tipo_cliente"   : pa.dictionary(pa.int32(), pa.string()),
            "consumo_kwh"    : pa.float32(),
            "voltaje_v"      : pa.float32(),
            "temperatura_amb": pa.float32(),
            "tarifa_clp"     : pa.float64(),
        }
    )
    tabla = pa_csv.read_csv(ARCHIVO_CSV, convert_options=convert_options)

    idx_tarifa = tabla.schema.get_field_index("tarifa_clp")
    tabla = tabla.set_column(
        idx_tarifa, "tarifa_clp",
        tabla.column("tarifa_clp").cast(pa.int64()),
    )

    ts = pa.compute.cast(
        pa.compute.strptime(tabla.column("timestamp"), "%Y-%m-%d %H:%M:%S%z",
                            error_is_null=True),
        pa.timestamp("us", tz="UTC"),
    ) if False else pa.array(
        pd.to_datetime(tabla.column("timestamp").to_pylist(), utc=True)
    )
    ts_pd = pd.array(tabla.column("timestamp").to_pylist(), dtype="object")
    ts_pd = pd.to_datetime(ts_pd, utc=True)
    tabla = tabla.append_column("year",  pa.array(ts_pd.year.astype("int32")))
    tabla = tabla.append_column("month", pa.array(ts_pd.month.astype("int32")))

    tabla = _normalizar_columna_string(tabla, "comuna")

    ruta_mono = _escribir_parquet_monolitico(tabla, DIR_PARQUET)
    ruta_part = _escribir_parquet_particionado(tabla, DIR_PARQUET)

    return ruta_mono, ruta_part


@medir
def consulta_csv(comuna_filtro: str):
    df = leer_csv_optimizado(usecols=["comuna", "consumo_kwh", "tarifa_clp"])
    return df[df["comuna"] == comuna_filtro]["consumo_kwh"].sum()


@medir
def consulta_parquet_mono(ruta_mono: str, comuna_filtro: str):
    df = pq.read_table(
        ruta_mono,
        columns=["comuna", "consumo_kwh", "tarifa_clp"],
        filters=[("comuna", "==", comuna_filtro)],
    ).to_pandas()
    return df["consumo_kwh"].sum()


@medir
def consulta_parquet_jerarquico(ruta_part: str, comuna_filtro: str,
                                year_filtro: int = 2023,
                                month_filtro: int = 1):
    """
    ▶ MEJORA 2: Consulta con partition pruning en los 3 niveles.
    PyArrow aplica el filtro ANTES de leer los datos (predicate pushdown):
    solo abre los archivos en year=2023/month=1/comuna=<filtro>/.
    """
    df = pq.read_table(
        ruta_part,
        columns = ["consumo_kwh", "tarifa_clp"],
        filters = [
            ("year",   "==", year_filtro),
            ("month",  "==", month_filtro),
            ("comuna", "==", comuna_filtro),
        ],
    ).to_pandas()
    return df["consumo_kwh"].sum()


def experimento_d():
    print("\n" + "=" * 60)
    print("  EXPERIMENTO D — Parquet, particionamiento y arquitectura")
    print("=" * 60)

    print("  → Convirtiendo CSV a Parquet (partición jerárquica year/month/comuna)...")
    if os.path.exists(DIR_PARQUET):
        shutil.rmtree(DIR_PARQUET)
        os.makedirs(DIR_PARQUET, exist_ok=True)

    (ruta_mono, ruta_part), t_conv, mem_conv, _ = convertir_a_parquet()


    n_archivos_part = sum(
        len(files)
        for _, _, files in os.walk(ruta_part)
        if files
    )

    tam_csv  = os.path.getsize(ARCHIVO_CSV) / 1_048_576
    tam_mono = os.path.getsize(ruta_mono) / 1_048_576
    tam_part = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(ruta_part)
        for f in files
    ) / 1_048_576

    print(f"  Tamaño CSV                : {tam_csv:.1f} MB")
    print(f"  Tamaño Parquet monolítico : {tam_mono:.1f} MB")
    print(f"  Tamaño Parquet jerárquico : {tam_part:.1f} MB")
    print(f"  Archivos en partición     : {n_archivos_part} "
          f"(year × month × comuna)")
    print(f"  Tiempo conversión         : {t_conv:.2f} s")

    COMUNA = "Maipu"
    YEAR   = 2023
    MONTH  = 1
    print(f"\n  → Comparando consultas para "
          f"comuna='{COMUNA}', year={YEAR}, month={MONTH}...")

    res_csv, t_csv, mem_csv, _ = consulta_csv(COMUNA)
    print(f"  CSV completo              → {t_csv:.3f} s | {mem_csv:.1f} MB "
          f"| suma={res_csv:.2f}")

    res_mono, t_mono, mem_mono, _ = consulta_parquet_mono(ruta_mono, COMUNA)
    print(f"  Parquet monolítico        → {t_mono:.3f} s | {mem_mono:.1f} MB "
          f"| suma={res_mono:.2f}")

    res_part, t_part, mem_part, _ = consulta_parquet_jerarquico(
        ruta_part, COMUNA, YEAR, MONTH
    )
    print(f"  Parquet jerárquico        → {t_part:.3f} s | {mem_part:.1f} MB "
          f"| suma={res_part:.2f} (solo {YEAR}/{MONTH:02d})")

    tabla_formato = pd.DataFrame({
        "Formato"     : ["CSV", "Parquet monolítico", "Parquet jerárquico"],
        "Tiempo (s)"  : [t_csv, t_mono, t_part],
        "Memoria (MB)": [mem_csv, mem_mono, mem_part],
        "Tamaño (MB)" : [tam_csv, tam_mono, tam_part],
    })

    return tabla_formato, ruta_mono, ruta_part



def viz_1_tiempos_lectura(df_b: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    colores  = [COLORES[0]] + [COLORES[2]] * len(CHUNK_SIZES)
    etiquetas = df_b["estrategia"] + "\n(" + df_b["chunksize"] + ")"
    bars = ax.bar(etiquetas, df_b["tiempo_s"], color=colores,
                  edgecolor="white", linewidth=0.8, width=0.55)
    for bar, val in zip(bars, df_b["tiempo_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f} s", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title("Figura 1 — Tiempo de lectura: Monolítica vs. Chunked",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Estrategia de lectura")
    ax.set_ylabel("Tiempo (segundos)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    sns.despine()
    plt.tight_layout()
    guardar_figura("fig1_tiempo_lectura.png")


def viz_2_memoria(df_b: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    colores   = [COLORES[3]] + [COLORES[4]] * len(CHUNK_SIZES)
    etiquetas = df_b["estrategia"] + "\n(" + df_b["chunksize"] + ")"
    bars = ax.bar(etiquetas, df_b["memoria_mb"], color=colores,
                  edgecolor="white", linewidth=0.8, width=0.55)
    for bar, val in zip(bars, df_b["memoria_mb"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f} MB", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title("Figura 2 — Uso de memoria pico por estrategia de lectura",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Estrategia de lectura")
    ax.set_ylabel("Memoria pico (MB)")
    sns.despine()
    plt.tight_layout()
    guardar_figura("fig2_memoria_lectura.png")


def viz_3_chunksize(df_b: pd.DataFrame):
    df_chunk = df_b[df_b["estrategia"] == "Chunked"].copy()
    df_chunk["cs_num"] = CHUNK_SIZES
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df_chunk["cs_num"], df_chunk["tiempo_s"],
             marker="o", color=COLORES[0], linewidth=2.5, markersize=9)
    for x, y in zip(df_chunk["cs_num"], df_chunk["tiempo_s"]):
        ax1.annotate(f"{y:.2f} s", (x, y),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10)
    ax1.set_title("Tiempo según chunksize", fontweight="bold")
    ax1.set_xlabel("Chunksize (filas por fragmento)")
    ax1.set_ylabel("Tiempo (s)")
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

    ax2.plot(df_chunk["cs_num"], df_chunk["memoria_mb"],
             marker="s", color=COLORES[1], linewidth=2.5, markersize=9)
    for x, y in zip(df_chunk["cs_num"], df_chunk["memoria_mb"]):
        ax2.annotate(f"{y:.0f} MB", (x, y),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10)
    ax2.set_title("Memoria pico según chunksize", fontweight="bold")
    ax2.set_xlabel("Chunksize (filas por fragmento)")
    ax2.set_ylabel("Memoria pico (MB)")
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

    fig.suptitle("Figura 3 — Efecto del tamaño de chunksize en tiempo y memoria",
                 fontsize=13, fontweight="bold", y=1.01)
    sns.despine()
    plt.tight_layout()
    guardar_figura("fig3_chunksize.png")


def viz_4_formatos(tabla_formato: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metricas = [
        ("Tiempo (s)",    "Tiempo (s)",    COLORES[0]),
        ("Memoria (MB)",  "Memoria (MB)",  COLORES[2]),
        ("Tamaño (MB)",   "Tamaño (MB)",   COLORES[4]),
    ]
    for ax, (col, ylabel, color) in zip(axes, metricas):
        bars = ax.bar(tabla_formato["Formato"], tabla_formato[col],
                      color=color, edgecolor="white", width=0.5)
        for bar, val in zip(bars, tabla_formato[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(tabla_formato[col]) * 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        sns.despine(ax=ax)
    fig.suptitle("Figura 4 — Comparación de formatos: CSV vs. Parquet",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    guardar_figura("fig4_comparacion_formatos.png")


def viz_5_consumo_comuna(resultado_c):
    df = resultado_c["consumo"].copy()
    df_agg = df.groupby("comuna", observed=True)["consumo_total"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_agg.index, df_agg.values / 1e6,
                  color=COLORES[:len(df_agg)], edgecolor="white")
    for bar, val in zip(bars, df_agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val/1e6:.2f} GWh", ha="center", va="bottom", fontsize=9)
    ax.set_title(
        "Figura 5 (bonus) — Consumo total por comuna (pipeline incremental)",
        fontsize=13, fontweight="bold")
    ax.set_xlabel("Comuna")
    ax.set_ylabel("Consumo total (GWh)")
    ax.tick_params(axis="x", rotation=25)
    sns.despine()
    plt.tight_layout()
    guardar_figura("fig5_consumo_por_comuna.png")



def main():
    info_sistema()

    df_b = resultado_c = t_c = mem_c = tabla_formato = ruta_mono = ruta_part = None

    try:
        experimento_a(n_filas=1_000_000, semilla=42)
    except Exception as exc:
        print(f"\n  [ERROR] Experimento A falló: {exc}")
        return

    try:
        df_b = experimento_b()
    except Exception as exc:
        print(f"\n  [ERROR] Experimento B falló: {exc}")

    try:
        resultado_c, t_c, mem_c = experimento_c()
    except Exception as exc:
        print(f"\n  [ERROR] Experimento C falló: {exc}")

    try:
        tabla_formato, ruta_mono, ruta_part = experimento_d()
    except Exception as exc:
        print(f"\n  [ERROR] Experimento D falló: {exc}")

    print("\n" + "=" * 60)
    print("  GENERANDO VISUALIZACIONES")
    print("=" * 60)
    if df_b is not None:
        viz_1_tiempos_lectura(df_b)
        viz_2_memoria(df_b)
        viz_3_chunksize(df_b)
    if tabla_formato is not None:
        viz_4_formatos(tabla_formato)
    if resultado_c is not None:
        viz_5_consumo_comuna(resultado_c)

    if tabla_formato is not None:
        print("\n  Tabla comparativa de formatos (Experimento D):")
        print(tabla_formato.to_string(index=False))
    else:
        print("\n  [OMITIDO] Tabla de formatos no disponible (Experimento D falló).")


    print("\n" + "=" * 60)
    print("  DECISIONES TÉCNICAS POR ESCENARIO")
    print("=" * 60)
    recomendaciones = [
        (
            "< 10 GB, un analista, cálculos diarios",
            "Batch local con pandas + dtype optimizado (MEJORA 1)",
            "RAM suficiente; category dtype reduce footprint hasta 80%; CAPEX mínimo.",
        ),
        (
            "10–500 GB, múltiples analistas, consultas ad-hoc",
            "Data Lake con Parquet jerárquico year/month/comuna (MEJORA 2)",
            "Partition pruning reduce I/O; compatible con Athena/Spark/BigQuery.",
        ),
        (
            "Pipeline analítico en un solo nodo con > 4 cores",
            "Dask o Polars en lugar de bucle chunksize (MEJORA 3)",
            "Paralelismo real sobre todos los núcleos; Polars: x10-20 vs. Pandas.",
        ),
        (
            "Detección de anomalías (estado='Alerta') con latencia < 1 s",
            "Streaming: Kafka + Bytewax o Flink Python API (MEJORA 4)",
            "Latencia estimada ~50-300 ms según benchmarks; escala horizontal; integra con Parquet histórico.",
        ),
        (
            "500 GB–PB, BI + ML compartidos",
            "Lakehouse (Delta Lake / Apache Iceberg)",
            "Combina flexibilidad del lago con garantías ACID del DWH.",
        ),
        (
            "Operación distribuida geográfica",
            "Cloud híbrida (on-premise + cloud burst)",
            "Federación de datos; cumplimiento regulatorio local; escalado bajo demanda.",
        ),
    ]

    for escenario, rec, justif in recomendaciones:
        print(f"\n  Escenario     : {escenario}")
        print(f"  Recomendación : {rec}")
        print(f"  Justificación : {justif}")

    print("\n" + "=" * 60)
    print("  EJECUCIÓN FINALIZADA")
    print("  Directorio de salida:", DIR_BASE)
    print("  Figuras en          :", DIR_IMGS)
    print("=" * 60)


if __name__ == "__main__":
    main()