"""
utils/unsupervised_engine.py
============================
Motor NO supervisado del dashboard (pagina "Aprendizaje No Supervisado").

Objetivo: descubrir ESTILOS DE JUEGO (perfiles) de los 20 equipos de la Premier
League 2022-23 mediante experimentacion exhaustiva (PCA + K-Means + clustering
jerarquico + SOM), evaluando la calidad con el Silhouette Score.

Pipeline:
- Agregacion por equipo (promedios por partido) y 5 KPIs avanzados.
- Estandarizacion Z-score (StandardScaler) de los 5 KPIs.
- PCA: nº de componentes necesarios para explicar [0.70..0.95] de varianza
  (reporte analitico) + proyeccion 2D (PC1, PC2) para la visualizacion.
- K-Means (k = 2..7), jerarquico (5 metodos via scipy.cluster.hierarchy con
  k = 2..7) y SOM (minisom, 6 cuadriculas simetricas), todos con Silhouette.

Todo se cachea en memoria del proceso: la agregacion/escalado/PCA se calculan una
sola vez y cada agrupamiento (algoritmo+config) se memoiza.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from minisom import MiniSom

from config import DATA_FILE, RANDOM_STATE
from utils.data_loader import load_dataset

# --------------------------------------------------------------------------- #
# Espacios de busqueda EXACTOS del enunciado
# --------------------------------------------------------------------------- #
KPI_KEYS = ["Volume_Attack", "Efficiency_Attack", "Defensive_Solidity",
            "Game_Control", "Aggressiveness"]
VARIANCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
KMEANS_K = [2, 3, 4, 5, 6, 7]
HIER_METHODS = ["ward", "complete", "average", "single", "centroid"]
HIER_K = [2, 3, 4, 5, 6, 7]
SOM_GRIDS = [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (10, 10)]

# Caches de proceso.
_kpi_cache: pd.DataFrame | None = None
_scaled_cache: tuple[list[str], np.ndarray] | None = None
_pca_cache: dict | None = None
_cluster_cache: dict[tuple, dict] = {}


# =========================================================================== #
# Agregacion por equipo + KPIs avanzados
# =========================================================================== #
def get_kpis() -> pd.DataFrame:
    """
    Agrega los 20 equipos (suma de sus 38 partidos como local y visitante) y
    calcula 5 KPIs avanzados (promedios/ratios por partido):

      - Volume_Attack     = tiros a puerta a favor / partido
      - Efficiency_Attack = goles a favor / tiros a puerta a favor
      - Defensive_Solidity= 1 - (goles en contra / tiros a puerta en contra)
      - Game_Control      = corners a favor / corners totales
      - Aggressiveness    = (faltas + tarjetas) / partido
    """
    global _kpi_cache
    if _kpi_cache is not None:
        return _kpi_cache

    df = load_dataset(DATA_FILE)
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    rows = {}
    for tm in teams:
        h = df[df["HomeTeam"] == tm]
        a = df[df["AwayTeam"] == tm]
        games = len(h) + len(a)
        gf = h["FTHG"].sum() + a["FTAG"].sum()        # goles a favor
        ga = h["FTAG"].sum() + a["FTHG"].sum()        # goles en contra
        sot_f = h["HST"].sum() + a["AST"].sum()       # tiros a puerta a favor
        sot_a = h["AST"].sum() + a["HST"].sum()       # tiros a puerta en contra
        cor_f = h["HC"].sum() + a["AC"].sum()         # corners a favor
        cor_a = h["AC"].sum() + a["HC"].sum()         # corners en contra
        fouls = h["HF"].sum() + a["AF"].sum()         # faltas cometidas
        yel = h["HY"].sum() + a["AY"].sum()           # amarillas
        red = h["HR"].sum() + a["AR"].sum()           # rojas
        rows[tm] = {
            "Volume_Attack": sot_f / games,
            "Efficiency_Attack": gf / sot_f,
            "Defensive_Solidity": 1.0 - (ga / sot_a),
            "Game_Control": cor_f / (cor_f + cor_a),
            "Aggressiveness": (fouls + yel + red) / games,
        }
    _kpi_cache = pd.DataFrame(rows).T[KPI_KEYS]
    return _kpi_cache


def scaled_matrix() -> tuple[list[str], np.ndarray]:
    """Devuelve (equipos, matriz Z-score 20x5). StandardScaler sobre los KPIs."""
    global _scaled_cache
    if _scaled_cache is not None:
        return _scaled_cache
    K = get_kpis()
    Xs = StandardScaler().fit_transform(K.values)
    _scaled_cache = (list(K.index), Xs)
    return _scaled_cache


# =========================================================================== #
# PCA: reporte analitico de varianza + proyeccion 2D
# =========================================================================== #
def pca_report() -> dict:
    """
    Reporte de PCA (cacheado):
      - explained / cumulative: ratios de varianza explicada por componente.
      - thresholds: nº de componentes para alcanzar cada umbral [0.70..0.95].
      - coords: proyeccion 2D (PC1, PC2) de los 20 equipos (para el scatter).
      - pc_var: % de varianza de PC1 y PC2 (para los ejes del scatter).
    """
    global _pca_cache
    if _pca_cache is not None:
        return _pca_cache

    _, Xs = scaled_matrix()
    pca = PCA(random_state=RANDOM_STATE).fit(Xs)        # 5 componentes (min(n,p))
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    thresholds = [{"threshold": thr,
                   "n_components": int(np.searchsorted(cumulative, thr) + 1)}
                  for thr in VARIANCE_THRESHOLDS]
    coords = pca.transform(Xs)[:, :2]
    _pca_cache = {
        "explained": explained.tolist(),
        "cumulative": cumulative.tolist(),
        "thresholds": thresholds,
        "coords": coords.tolist(),
        "pc_var": [float(explained[0] * 100), float(explained[1] * 100)],
    }
    return _pca_cache


# =========================================================================== #
# Etiquetas de cada algoritmo
# =========================================================================== #
def _compact(labels) -> np.ndarray:
    """Reindexa etiquetas arbitrarias a 0..n-1 (orden estable)."""
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv


def _kmeans_labels(k: int) -> np.ndarray:
    _, Xs = scaled_matrix()
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)


def _hierarchical_labels(method: str, k: int) -> np.ndarray:
    """Jerarquico con scipy: linkage(metodo) + fcluster(maxclust=k)."""
    _, Xs = scaled_matrix()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # centroid/ward avisan de no-monotonia
        Z = linkage(Xs, method=method)
    return _compact(fcluster(Z, t=k, criterion="maxclust"))


def _som_labels(grid: tuple[int, int]) -> np.ndarray:
    """SOM (minisom): etiqueta = neurona ganadora de cada equipo (reindexada)."""
    _, Xs = scaled_matrix()
    x, y = grid
    som = MiniSom(x, y, Xs.shape[1], sigma=1.0, learning_rate=0.5,
                  random_seed=RANDOM_STATE)
    som.random_weights_init(Xs)
    som.train_random(Xs, 1000)
    winners = [som.winner(v) for v in Xs]
    return _compact([i * y + j for (i, j) in winners])


def cluster_result(algo: str, params) -> dict:
    """
    Agrupa segun (algoritmo, config) y evalua con Silhouette (cacheado).

    params:
      - "kmeans"       -> int k
      - "hierarchical" -> (method, k)
      - "som"          -> (x, y)

    El Silhouette solo esta definido para 2 <= nº clusters <= n-1; si el SOM
    sobre-segmenta (casi un equipo por neurona) se devuelve None.
    """
    key = (algo, params if not isinstance(params, list) else tuple(params))
    if key in _cluster_cache:
        return _cluster_cache[key]

    _, Xs = scaled_matrix()
    if algo == "kmeans":
        labels = _kmeans_labels(int(params))
    elif algo == "hierarchical":
        method, k = params
        labels = _hierarchical_labels(method, int(k))
    elif algo == "som":
        labels = _som_labels(tuple(params))
    else:
        raise ValueError(f"Algoritmo desconocido: {algo}")

    labels = np.asarray(labels)
    n_clusters = int(len(np.unique(labels)))
    if 2 <= n_clusters <= len(Xs) - 1:
        sil = float(silhouette_score(Xs, labels))
    else:
        sil = None

    result = {"algo": algo, "params": params, "labels": labels.tolist(),
              "n_clusters": n_clusters, "silhouette": sil}
    _cluster_cache[key] = result
    return result
