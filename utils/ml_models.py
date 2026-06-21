"""
utils/ml_models.py
==================
Motor de aprendizaje supervisado del dashboard (pagina "Aprendizaje Supervisado").

Replica ESTRICTA de la metodologia de la Practica 2 de AA1 (Wine -> aqui EPL),
trasladada de Julia a scikit-learn. Predice el resultado final del partido
(FTR: "H"/"D"/"A"  =  1/X/2) a partir UNICAMENTE de informacion PRE-PARTIDO.

Pilares metodologicos (identicos al .jl / a la memoria del Wine):
- ANTI-FUGA DE DATOS: FTR queda determinado por FTHG/FTAG, y el resto de columnas
  (HS, HST, HC, HF, HY, HR, ...) se registran DURANTE el partido. Usarlas como
  entrada seria fuga catastrofica. Por eso, para cada partido construimos las
  MEDIAS MOVILES de los `ROLLING_WINDOW` encuentros ANTERIORES de cada equipo
  (forma reciente). Las estadisticas intra-partido solo alimentan ese historial.
- Semilla global UNICA (RANDOM_STATE = 1234) -> reproducibilidad total.
- Validacion cruzada ESTRATIFICADA de 10 particiones, FIJA (mismo split para
  todos los modelos -> comparativa justa y test de Wilcoxon emparejado valido).
- Estandarizacion Z-score calculada SOLO con el train de cada fold y aplicada al
  fold de validacion (Pipeline StandardScaler->modelo). PROHIBIDO normalizar todo
  el dataset antes del CV (evita fuga de informacion).
- Espacios de busqueda CERRADOS (exactamente los de la practica; sin rangos extra).
- Tests estadisticos: Shapiro-Wilk (normalidad del vector de 10 accuracies) y
  Wilcoxon de rangos con signo (comparacion emparejada de dos modelos).

Rendimiento: cada evaluacion (algoritmo+config) se cachea en memoria del proceso,
de modo que reentrenar una configuracion identica o reusarla en una comparacion es
instantaneo. Las features se construyen una sola vez.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import CV_FOLDS, DATA_FILE, MIN_HISTORY, RANDOM_STATE, ROLLING_WINDOW
from utils.data_loader import load_dataset

# Clases del problema en orden 1/X/2 (filas = real, columnas = predicha en la matriz).
LABELS = ["H", "D", "A"]

# --------------------------------------------------------------------------- #
# Espacios de busqueda CERRADOS (identicos a practica_epl.jl / la memoria).
#   Cada algoritmo expone: clave canonica, etiqueta i18n y su rejilla exacta.
# --------------------------------------------------------------------------- #
ANN_TOPOLOGIES = [(5,), (10,), (15,), (20,), (5, 5), (10, 5), (10, 10), (15, 10)]
SVM_CONFIGS = [
    ("linear", 0.1), ("linear", 1), ("linear", 10),
    ("poly", 1), ("poly", 10),
    ("rbf", 0.1), ("rbf", 1), ("rbf", 10),
]
TREE_DEPTHS = [2, 3, 4, 5, 6, 7]
KNN_NEIGHBORS = [1, 3, 5, 7, 9, 11]
DOME_NODES = [2, 3, 4, 5, 6, 7, 8, 9]

# Algoritmos "complejos" vs "simples" (para la lectura de la Navaja de Ockham).
COMPLEX_ALGOS = {"ann", "svm", "dome"}
SIMPLE_ALGOS = {"knn", "tree", "baseline"}

# Caches de proceso.
_data_cache: tuple[pd.DataFrame, pd.Series] | None = None
_eval_cache: dict[tuple, dict] = {}
_grid_cache: dict[str, dict] = {}


# =========================================================================== #
# Construccion de features PRE-PARTIDO (sin fuga de datos)
# =========================================================================== #
def _points(gf: int, ga: int) -> int:
    """Puntos del partido para un equipo: 3 victoria, 1 empate, 0 derrota."""
    return 3 if gf > ga else (1 if gf == ga else 0)


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye (X, y) replicando `construir_features_prepartido` del .jl.

    Para cada partido (en orden cronologico) y si AMBOS equipos acumulan al menos
    MIN_HISTORY partidos previos, se calculan las medias moviles de sus ultimos
    ROLLING_WINDOW encuentros: goles a favor/en contra, tiros, tiros a puerta,
    corners, faltas, amarillas, rojas y puntos (forma), para local (L_) y
    visitante (V_), mas 4 diferenciales clave (local - visitante). El partido se
    anade al historial DESPUES de usarlo (jamas se mira el futuro).
    """
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    df = load_dataset(DATA_FILE)
    # Date dd/mm/YYYY + Time HH:MM -> orden cronologico estricto.
    dt = pd.to_datetime(df["Date"] + " " + df["Time"],
                        format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.assign(_dt=dt).sort_values("_dt").reset_index(drop=True)

    stat_keys = ["gf", "ga", "tir", "tp", "cor", "fal", "amar", "roj", "pts"]
    history: dict[str, list[dict]] = {
        tm: [] for tm in pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    }

    rows, targets = [], []
    for m in df.itertuples(index=False):
        hl, hv = history[m.HomeTeam], history[m.AwayTeam]

        # Solo si ambos equipos tienen historial suficiente (descarta jornadas iniciales).
        if len(hl) >= MIN_HISTORY and len(hv) >= MIN_HISTORY:
            last_l, last_v = hl[-ROLLING_WINDOW:], hv[-ROLLING_WINDOW:]
            ml = {k: float(np.mean([d[k] for d in last_l])) for k in stat_keys}
            mv = {k: float(np.mean([d[k] for d in last_v])) for k in stat_keys}
            rows.append({
                # --- Medias moviles del LOCAL ---
                "L_GF": ml["gf"], "L_GA": ml["ga"], "L_TIR": ml["tir"], "L_TP": ml["tp"],
                "L_COR": ml["cor"], "L_FAL": ml["fal"], "L_AMAR": ml["amar"],
                "L_ROJ": ml["roj"], "L_PTS": ml["pts"],
                # --- Medias moviles del VISITANTE ---
                "V_GF": mv["gf"], "V_GA": mv["ga"], "V_TIR": mv["tir"], "V_TP": mv["tp"],
                "V_COR": mv["cor"], "V_FAL": mv["fal"], "V_AMAR": mv["amar"],
                "V_ROJ": mv["roj"], "V_PTS": mv["pts"],
                # --- Diferenciales clave (local - visitante) ---
                "DIF_GF": ml["gf"] - mv["gf"], "DIF_GA": ml["ga"] - mv["ga"],
                "DIF_TP": ml["tp"] - mv["tp"], "DIF_PTS": ml["pts"] - mv["pts"],
            })
            targets.append(m.FTR)

        # El partido pasa a formar parte del historial de ambos (tras predecirlo).
        history[m.HomeTeam].append(dict(
            gf=m.FTHG, ga=m.FTAG, tir=m.HS, tp=m.HST, cor=m.HC, fal=m.HF,
            amar=m.HY, roj=m.HR, pts=_points(m.FTHG, m.FTAG)))
        history[m.AwayTeam].append(dict(
            gf=m.FTAG, ga=m.FTHG, tir=m.AS, tp=m.AST, cor=m.AC, fal=m.AF,
            amar=m.AY, roj=m.AR, pts=_points(m.FTAG, m.FTHG)))

    _data_cache = (pd.DataFrame(rows), pd.Series(targets, name="FTR"))
    return _data_cache


def majority_baseline() -> float:
    """Tasa de no-informacion: proporcion de la clase mayoritaria (referencia)."""
    _, y = build_features()
    return float(y.value_counts().iloc[0] / len(y))


# =========================================================================== #
# Estimadores (configuraciones EXACTAS de la practica)
# =========================================================================== #
def _make_estimator(algo: str, config):
    """
    Devuelve el estimador de scikit-learn para (algoritmo, configuracion).
    La estandarizacion se anade fuera, en un Pipeline, para hacerla por-fold.
    """
    if algo == "baseline":
        # Baseline: regresion logistica con parametros por defecto del enunciado.
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, random_state=RANDOM_STATE)
    if algo == "ann":
        # MLP. solver='lbfgs' (cuasi-Newton): determinista e idoneo para datasets
        # pequenos; reproduce el regimen "bien entrenado" de la ANN de Flux
        # (que promediaba 50 ejecuciones) sin el coste/varianza de 'adam'.
        return MLPClassifier(hidden_layer_sizes=tuple(config), solver="lbfgs",
                             max_iter=1000, random_state=RANDOM_STATE)
    if algo == "svm":
        kernel, C = config
        # gamma='auto' = 1/n_features  ==  el 1.0/size(inputs,2) del .jl (y de LIBSVM).
        if kernel == "poly":
            return SVC(kernel="poly", degree=3, C=C, gamma="auto", random_state=RANDOM_STATE)
        return SVC(kernel=kernel, C=C, gamma="auto", random_state=RANDOM_STATE)
    if algo == "tree":
        return DecisionTreeClassifier(max_depth=int(config), random_state=RANDOM_STATE)
    if algo == "knn":
        return KNeighborsClassifier(n_neighbors=int(config))
    if algo == "dome":
        # PROXY de DoME: RandomForest acotado por nº de hojas (max_nodes -> max_leaf_nodes).
        # >>> SUSTITUIR AQUI por la llamada a la API de Julia (SymDoME) cuando se integre,
        #     manteniendo la misma firma evaluate(algo, config). <<<
        return RandomForestClassifier(max_leaf_nodes=int(config),
                                      n_estimators=100, random_state=RANDOM_STATE)
    raise ValueError(f"Algoritmo desconocido: {algo}")


def _cv() -> StratifiedKFold:
    """Particionado 10-fold estratificado FIJO (mismo split para todos los modelos)."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# =========================================================================== #
# Evaluacion por validacion cruzada
# =========================================================================== #
def evaluate(algo: str, config=None) -> dict:
    """
    10-fold CV estratificado de (algo, config) con Z-score por-fold (Pipeline).

    Devuelve accuracy/error/F1 medios (+ std), el vector de 10 accuracies, y la
    matriz de confusion GLOBAL (agregando las predicciones de los 10 folds, en
    los que cada patron aparece en test exactamente una vez). Resultado cacheado.
    """
    key = (algo, config if not isinstance(config, list) else tuple(config))
    if key in _eval_cache:
        return _eval_cache[key]

    X, y = build_features()
    accs, f1s = [], []
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in _cv().split(X, y):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", _make_estimator(algo, config))])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], pred))
        f1s.append(f1_score(y.iloc[test_idx], pred, labels=LABELS, average="macro",
                            zero_division=0))
        y_true_all.extend(y.iloc[test_idx]); y_pred_all.extend(pred)

    accs = np.asarray(accs, dtype=float)
    f1s = np.asarray(f1s, dtype=float)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)

    result = {
        "algo": algo,
        "config": config,
        "accuracies": accs.tolist(),         # vector de 10 (uno por fold)
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std()),
        "error_mean": float(1.0 - accs.mean()),
        "error_std": float(accs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "confusion": cm.tolist(),            # 3x3, filas=real, columnas=predicha
        "labels": LABELS,
        "n": int(len(y)),
    }
    _eval_cache[key] = result
    return result


# Espacio de configuraciones por algoritmo (para la rejilla y la UI).
def config_space(algo: str) -> list:
    return {
        "baseline": [None],
        "ann": ANN_TOPOLOGIES,
        "svm": SVM_CONFIGS,
        "tree": TREE_DEPTHS,
        "knn": KNN_NEIGHBORS,
        "dome": DOME_NODES,
    }[algo]


def grid_search(algo: str) -> dict:
    """
    Evalua TODO el espacio cerrado del algoritmo (cacheado) y localiza la mejor
    configuracion por accuracy media. Tambien deja cacheada cada config evaluada.
    """
    if algo in _grid_cache:
        return _grid_cache[algo]

    rows = []
    for cfg in config_space(algo):
        r = evaluate(algo, cfg)
        rows.append({"config": cfg, "acc_mean": r["acc_mean"],
                     "acc_std": r["acc_std"], "accuracies": r["accuracies"]})
    best = max(rows, key=lambda d: d["acc_mean"])
    summary = {"algo": algo, "results": rows,
               "best_config": best["config"], "best_acc": best["acc_mean"],
               "best_accuracies": best["accuracies"]}
    _grid_cache[algo] = summary
    return summary


def best_evaluation(algo: str) -> dict:
    """Evaluacion completa del algoritmo en su MEJOR configuracion (cacheada)."""
    return evaluate(algo, grid_search(algo)["best_config"])


# =========================================================================== #
# Tests estadisticos (scipy.stats)
# =========================================================================== #
def shapiro_test(accuracies) -> dict:
    """Shapiro-Wilk de normalidad sobre el vector de accuracies por fold."""
    acc = np.asarray(accuracies, dtype=float)
    # Si no hay varianza (todos iguales) Shapiro no esta definido: lo tratamos aparte.
    if np.allclose(acc, acc[0]):
        return {"stat": float("nan"), "p_value": 1.0, "normal": True, "degenerate": True}
    stat, p = shapiro(acc)
    return {"stat": float(stat), "p_value": float(p), "normal": bool(p > 0.05),
            "degenerate": False}


def wilcoxon_test(acc_a, acc_b, alpha: float = 0.05) -> dict:
    """
    Wilcoxon de rangos con signo EMPAREJADO entre dos vectores de accuracies
    (mismos folds). Veredicto: 'significativo' si p <= alpha.
    """
    a = np.asarray(acc_a, dtype=float)
    b = np.asarray(acc_b, dtype=float)
    diff = a - b
    # Si no hay diferencias (p.ej. comparar un modelo consigo mismo) el test no
    # aplica: son trivialmente equivalentes.
    if np.allclose(diff, 0.0):
        return {"stat": 0.0, "p_value": 1.0, "significant": False, "degenerate": True}
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        return {"stat": float("nan"), "p_value": 1.0, "significant": False,
                "degenerate": True}
    return {"stat": float(stat), "p_value": float(p),
            "significant": bool(p <= alpha), "degenerate": False}


# =========================================================================== #
# Senal de sobreajuste (para el data storytelling)
# =========================================================================== #
def overfitting_signal(algo: str) -> dict | None:
    """
    Detecta el patron "mas complejidad NO mejora (o empeora)" en algoritmos con
    un eje de complejidad creciente (arboles, DoME, ANN). Devuelve None si no hay
    senal o el algoritmo no aplica.

    Para arboles/DoME el eje es el escalar (profundidad/nodos). Para la ANN se
    ordena por (nº de capas, nº total de neuronas). Hay senal si la configuracion
    mas compleja rinde por debajo de la mejor y la mejor no es la mas compleja.
    """
    if algo not in {"tree", "dome", "ann"}:
        return None
    grid = grid_search(algo)
    rows = grid["results"]

    def complexity(cfg):
        if algo == "ann":
            return (len(cfg), sum(cfg))
        return cfg  # escalar (profundidad o nodos)

    most_complex = max(rows, key=lambda d: complexity(d["config"]))
    best = max(rows, key=lambda d: d["acc_mean"])
    if complexity(best["config"]) >= complexity(most_complex["config"]):
        return None
    if most_complex["acc_mean"] >= best["acc_mean"]:
        return None

    param = {"tree": "max_depth", "dome": "max_nodes", "ann": "topology"}[algo]
    return {
        "algo": algo,
        "param": param,
        "best_config": best["config"],
        "best_acc": best["acc_mean"],
        "complex_config": most_complex["config"],
        "complex_acc": most_complex["acc_mean"],
    }
