"""
tests/test_ml_engine.py
=======================
Tests del MOTOR MATEMATICO (no de la UI). Garantizan que el nucleo de ML/estadistica
es correcto y no se rompe en casos limite. Se ejecutan con:

    pytest -q

Cubren:
  * Estandarizacion Z-score -> media 0 y varianza 1.
  * PCA -> numero EXACTO de componentes esperado (por umbral de varianza y directo).
  * Prevencion de fuga de datos (medias moviles pre-partido) -> sin NaN/inf en limites.
Mas comprobaciones de robustez del motor supervisado y no supervisado.
"""
import os
import sys

import numpy as np
import pytest

# Permite `from utils import ...` sin instalar el paquete (raiz del proyecto al path).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.decomposition import PCA  # noqa: E402

from config import CV_FOLDS  # noqa: E402
from utils import ml_models as ml  # noqa: E402
from utils import unsupervised_engine as ue  # noqa: E402

TOL = 1e-9


# --------------------------------------------------------------------------- #
# 1) Estandarizacion Z-score: media 0, varianza 1
# --------------------------------------------------------------------------- #
def test_zscore_zero_mean_unit_variance():
    """StandardScaler (scaled_matrix) deja cada KPI con media ~0 y varianza ~1."""
    _, Xs = ue.scaled_matrix()
    means = Xs.mean(axis=0)
    variances = Xs.var(axis=0, ddof=0)          # StandardScaler usa ddof=0
    assert np.allclose(means, 0.0, atol=1e-9), f"medias no nulas: {means}"
    assert np.allclose(variances, 1.0, atol=1e-9), f"varianzas != 1: {variances}"


def test_zscore_shape_and_finiteness():
    """La matriz estandarizada es 20x5 y totalmente finita."""
    teams, Xs = ue.scaled_matrix()
    assert Xs.shape == (20, len(ue.KPI_KEYS))
    assert len(teams) == 20
    assert np.isfinite(Xs).all()


# --------------------------------------------------------------------------- #
# 2) PCA: numero EXACTO de componentes
# --------------------------------------------------------------------------- #
def test_pca_direct_n_components_exact():
    """PCA(n_components=k) devuelve EXACTAMENTE k componentes."""
    _, Xs = ue.scaled_matrix()
    for k in (1, 2, 3, 4, 5):
        pca = PCA(n_components=k, random_state=1234).fit(Xs)
        assert pca.n_components_ == k
        assert pca.transform(Xs).shape == (20, k)


def test_pca_report_thresholds_match_cumulative():
    """El nº de componentes por umbral coincide con la varianza acumulada real."""
    rep = ue.pca_report()
    cumulative = np.array(rep["cumulative"])
    # La proyeccion 2D para el scatter debe ser (20, 2).
    assert np.array(rep["coords"]).shape == (20, 2)
    # Con 20 muestras y 5 features hay exactamente 5 componentes.
    assert len(rep["explained"]) == 5
    assert pytest.approx(cumulative[-1], abs=1e-9) == 1.0
    for item in rep["thresholds"]:
        expected = int(np.searchsorted(cumulative, item["threshold"]) + 1)
        assert item["n_components"] == expected, (
            f"umbral {item['threshold']}: {item['n_components']} != {expected}")
        # Sanidad: ese nº de componentes realmente alcanza el umbral.
        assert cumulative[item["n_components"] - 1] >= item["threshold"] - 1e-12


# --------------------------------------------------------------------------- #
# 3) Prevencion de fuga de datos: medias moviles sin NaN en casos limite
# --------------------------------------------------------------------------- #
def test_rolling_features_no_nan_or_inf():
    """build_features (medias moviles pre-partido) no produce NaN ni inf."""
    X, y = ml.build_features()
    assert not X.isnull().values.any(), "hay NaN en las features"
    assert np.isfinite(X.to_numpy()).all(), "hay inf en las features"
    assert len(X) == len(y) > 0


def test_rolling_features_schema_and_min_history_drop():
    """22 features exactas y se descartan las jornadas sin historial (caso limite)."""
    X, y = ml.build_features()
    assert X.shape[1] == 22, f"esperadas 22 features, hay {X.shape[1]}"
    # Caso limite: los primeros partidos (historial < MIN_HISTORY) NO generan fila.
    assert len(X) < 380, "no se descarto ninguna jornada inicial (posible fuga de limite)"
    # El target solo contiene clases validas.
    assert set(np.unique(y)).issubset({"H", "D", "A"})


def test_rolling_features_deterministic():
    """El builder es determinista (mismas filas en dos llamadas)."""
    X1, _ = ml.build_features()
    X2, _ = ml.build_features()
    assert X1.shape == X2.shape
    assert np.allclose(X1.to_numpy(), X2.to_numpy())


# --------------------------------------------------------------------------- #
# 4) Robustez del motor SUPERVISADO
# --------------------------------------------------------------------------- #
def test_evaluate_baseline_structure_and_ranges():
    """evaluate() devuelve 10 accuracies, matriz 3x3 y metricas en rango."""
    r = ml.evaluate("baseline", None)
    assert len(r["accuracies"]) == CV_FOLDS
    assert all(0.0 <= a <= 1.0 for a in r["accuracies"])
    assert 0.0 <= r["acc_mean"] <= 1.0
    assert pytest.approx(r["error_mean"], abs=1e-9) == 1.0 - r["acc_mean"]
    cm = np.array(r["confusion"])
    assert cm.shape == (3, 3)
    # La matriz global agrega todos los patrones (cada uno en test una vez).
    assert cm.sum() == r["n"]


def test_statistical_tests_guard_degenerate_cases():
    """Shapiro/Wilcoxon manejan vectores degenerados sin lanzar excepciones."""
    accs = ml.evaluate("baseline", None)["accuracies"]
    # Wilcoxon de un modelo consigo mismo -> no significativo (degenerado).
    w = ml.wilcoxon_test(accs, accs)
    assert w["degenerate"] is True and w["significant"] is False
    # Shapiro de un vector constante -> degenerado, no peta.
    sh = ml.shapiro_test([0.5] * CV_FOLDS)
    assert sh["degenerate"] is True
    # Shapiro de un vector normal -> p-valor en [0, 1].
    sh2 = ml.shapiro_test(accs)
    assert 0.0 <= sh2["p_value"] <= 1.0


# --------------------------------------------------------------------------- #
# 5) Robustez del motor NO SUPERVISADO
# --------------------------------------------------------------------------- #
def test_kmeans_cluster_result_valid():
    """K-Means: 20 etiquetas, k clusters y Silhouette en [-1, 1]."""
    res = ue.cluster_result("kmeans", 3)
    assert len(res["labels"]) == 20
    assert res["n_clusters"] == 3
    assert res["silhouette"] is not None and -1.0 <= res["silhouette"] <= 1.0


def test_som_oversegmentation_reports_none_silhouette():
    """SOM 10x10 sobre 20 equipos sobre-segmenta -> Silhouette no definido (None)."""
    res = ue.cluster_result("som", (10, 10))
    # Casi un equipo por neurona: el Silhouette deja de estar definido.
    assert res["silhouette"] is None
