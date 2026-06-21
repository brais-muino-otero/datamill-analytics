# -*- coding: utf-8 -*-
"""
utils/data_loader.py
====================
Ingesta UNICA del dataset estatico local.

Punto de entrada centralizado para cargar los datos cerrados de la app en un
DataFrame de pandas. Todo el pipeline aguas abajo (limpieza, PCA, clustering,
modelos supervisados, Monte Carlo) consume su salida, de modo que cambiar la
fuente de datos se hace en UN solo sitio.

Soporta CSV/TSV y Parquet (autodeteccion por extension): migrar a .parquet es
tan simple como apuntar config.DATA_FILE al nuevo fichero (Parquet requiere
'pyarrow' o 'fastparquet' instalado).

Sin red, sin claves, sin cuotas: 100% local, estable y reproducible.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_FILE

# Cache de modulo: el fichero estatico no cambia en tiempo de ejecucion, asi que
# lo leemos del disco una sola vez por ruta y reutilizamos el DataFrame.
_CACHE: dict[str, pd.DataFrame] = {}


def load_dataset(path: str | Path | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    """Carga el dataset estatico local en un DataFrame de pandas.

    Parametros
    ----------
    path : ruta al fichero de datos. Por defecto, config.DATA_FILE.
    use_cache : si True (por defecto), reutiliza la lectura previa de esa ruta.

    Devuelve SIEMPRE una copia, para que quien la reciba pueda transformar el
    DataFrame sin contaminar la cache compartida (mismo comportamiento que tenia
    pd.read_csv: cada consumidor recibe su propio DataFrame independiente).
    """
    file_path = Path(path) if path is not None else Path(DATA_FILE)
    key = str(file_path.resolve())

    if use_cache and key in _CACHE:
        return _CACHE[key].copy()

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset estatico: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)             # requiere pyarrow/fastparquet
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(file_path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Formato no soportado: '{suffix}'. Usa .csv, .tsv o .parquet")

    if df.empty:
        raise ValueError(f"El dataset esta vacio: {file_path}")

    _CACHE[key] = df
    return df.copy()


def clear_cache() -> None:
    """Vacia la cache de lectura (util en tests o si se recarga el fichero)."""
    _CACHE.clear()
