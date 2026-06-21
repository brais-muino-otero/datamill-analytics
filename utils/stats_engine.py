"""
utils/stats_engine.py
=====================
Motor estadistico del dashboard de Futbol Analytics.

Aqui vive TODA la logica matematica (y solo ella): carga de datos, calculo de
la clasificacion, matriz de correlaciones, contraste t de Student sobre la
ventaja de campo, ajuste del modelo Dixon-Coles por maxima verosimilitud y
simulacion Monte Carlo de la temporada. La capa de presentacion (pages/) solo
consume estas funciones y dibuja; no hace cuentas.

Decisiones de diseno:
- Los datos y los parametros de Dixon-Coles se calculan UNA vez y se cachean en
  memoria del proceso (load_data / fit_dixon_coles), porque son caros y no
  cambian: el dataset es fijo y el ajuste por MV es determinista.
- Se reproduce la formulacion del analisis original en R para que los resultados
  (ventaja de campo ~1.34, ranking de ataque, probabilidades Monte Carlo)
  coincidan con el informe del proyecto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, ttest_ind, norm

from config import DATA_FILE, TOP_N_TEAMS
from utils.data_loader import load_dataset

# Variables del mapa de calor (claves canonicas; pages/ las traduce a etiquetas).
# El orden define filas/columnas de la matriz de correlaciones.
CORRELATION_VARS = ["goals", "shots", "shots_on_target", "fouls", "cards"]

# Metricas disponibles para el contraste de ventaja de campo:
#   clave -> (columna_local, columna_visitante)
TTEST_METRICS = {
    "goals": ("FTHG", "FTAG"),   # goles
    "shots": ("HST", "AST"),     # tiros a puerta
}

# Caches de proceso (se rellenan en la primera llamada).
_data_cache: pd.DataFrame | None = None
_dc_cache: dict | None = None


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Carga el CSV de la temporada (una sola vez) y anade variables derivadas."""
    global _data_cache
    if _data_cache is None:
        df = load_dataset(DATA_FILE)
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        # Variables derivadas (totales por partido) usadas en el analisis.
        df["TotalGoals"] = df["FTHG"] + df["FTAG"]
        df["HomeCards"] = df["HY"] + df["HR"]
        df["AwayCards"] = df["AY"] + df["AR"]
        _data_cache = df
    return _data_cache


# --------------------------------------------------------------------------- #
# Clasificacion
# --------------------------------------------------------------------------- #
def compute_standings(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clasificacion real: puntos, posicion y goles a favor/en contra por equipo."""
    if df is None:
        df = load_data()
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    table = {tm: {"Points": 0, "GF": 0, "GA": 0} for tm in teams}
    for h, a, hg, ag, res in zip(df["HomeTeam"], df["AwayTeam"],
                                 df["FTHG"], df["FTAG"], df["FTR"]):
        table[h]["GF"] += hg; table[h]["GA"] += ag
        table[a]["GF"] += ag; table[a]["GA"] += hg
        if res == "H":
            table[h]["Points"] += 3
        elif res == "A":
            table[a]["Points"] += 3
        else:
            table[h]["Points"] += 1
            table[a]["Points"] += 1
    standings = pd.DataFrame(
        [{"Team": tm, **v, "GD": v["GF"] - v["GA"]} for tm, v in table.items()]
    )
    # Orden: puntos y, a igualdad, diferencia de goles (criterio de la Premier).
    standings = standings.sort_values(["Points", "GD"], ascending=False).reset_index(drop=True)
    standings.insert(0, "Position", range(1, len(standings) + 1))
    return standings


def get_top_teams(n: int = TOP_N_TEAMS) -> list[str]:
    """Devuelve los n mejores equipos por clasificacion (filtro 'Top 6')."""
    return compute_standings()["Team"].head(n).tolist()


# --------------------------------------------------------------------------- #
# Perfil largo equipo-partido (base de las correlaciones)
# --------------------------------------------------------------------------- #
def _team_match_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por equipo y partido, desde la perspectiva de ese equipo.
    Permite calcular correlaciones y filtrarlas por subconjunto de equipos.
    """
    home = pd.DataFrame({
        "Team": df["HomeTeam"],
        "goals": df["FTHG"], "shots": df["HS"], "shots_on_target": df["HST"],
        "fouls": df["HF"], "cards": df["HY"] + df["HR"],
    })
    away = pd.DataFrame({
        "Team": df["AwayTeam"],
        "goals": df["FTAG"], "shots": df["AS"], "shots_on_target": df["AST"],
        "fouls": df["AF"], "cards": df["AY"] + df["AR"],
    })
    return pd.concat([home, away], ignore_index=True)


def correlation_matrix(scope: str = "all") -> tuple[list[str], np.ndarray]:
    """
    Matriz de correlaciones de Pearson entre CORRELATION_VARS.

    scope='all'  -> todos los equipos.
    scope='top6' -> solo los partidos de los 6 primeros clasificados.
    Devuelve (claves_de_variables, matriz NxN como ndarray).
    """
    long = _team_match_long(load_data())
    if scope == "top6":
        long = long[long["Team"].isin(get_top_teams(TOP_N_TEAMS))]
    corr = long[CORRELATION_VARS].corr(method="pearson")
    return CORRELATION_VARS, corr.to_numpy()


# --------------------------------------------------------------------------- #
# Inferencia: contraste t de Student (ventaja de campo)
# --------------------------------------------------------------------------- #
def home_advantage_ttest(metric: str = "goals", alpha: float = 0.05) -> dict:
    """
    Contraste t de Welch unilateral (local > visitante) sobre la metrica dada.

    Devuelve estadistico, p-valor, medias, errores estandar y las curvas de la
    distribucion muestral de cada media (listas para dibujar sin mas calculo).
    """
    df = load_data()
    home_col, away_col = TTEST_METRICS[metric]
    home = df[home_col].to_numpy(dtype=float)
    away = df[away_col].to_numpy(dtype=float)

    # Welch (no asume varianzas iguales); H1: media local > media visitante.
    t_stat, p_value = ttest_ind(home, away, equal_var=False, alternative="greater")

    mean_h, mean_a = home.mean(), away.mean()
    # Error estandar de la media = s / sqrt(n)  (s muestral, ddof=1).
    se_h = home.std(ddof=1) / np.sqrt(home.size)
    se_a = away.std(ddof=1) / np.sqrt(away.size)

    # Distribucion muestral de cada media (Teorema Central del Limite): N(media, se).
    lo = min(mean_h - 4 * se_h, mean_a - 4 * se_a)
    hi = max(mean_h + 4 * se_h, mean_a + 4 * se_a)
    grid = np.linspace(lo, hi, 240)

    return {
        "metric": metric,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_home": float(mean_h),
        "mean_away": float(mean_a),
        "se_home": float(se_h),
        "se_away": float(se_a),
        "diff": float(mean_h - mean_a),
        "significant": bool(p_value < alpha),
        "curve_x": grid.tolist(),
        "density_home": norm.pdf(grid, mean_h, se_h).tolist(),
        "density_away": norm.pdf(grid, mean_a, se_a).tolist(),
    }


# --------------------------------------------------------------------------- #
# Modelo Dixon-Coles (maxima verosimilitud)
# --------------------------------------------------------------------------- #
def fit_dixon_coles() -> dict:
    """
    Estima ataque y defensa de cada equipo + ventaja de campo por MV.

    Modelo: goles ~ Poisson, con
        lambda_local     = exp(ventaja + ataque_local  - defensa_visitante)
        lambda_visitante = exp(          ataque_visit.  - defensa_local)

    Nota de identificabilidad: el modelo tiene una direccion redundante (sumar
    una constante a todos los ataques y defensas no cambia la verosimilitud),
    por lo que el optimizador puede no "converger" en el sentido estricto del
    gradiente. Las cantidades con sentido -ventaja de campo y las DIFERENCIAS de
    ataque/defensa entre equipos- si son unicas y reproducibles. Se mantiene
    esta formulacion para reproducir el informe original en R.
    """
    global _dc_cache
    if _dc_cache is not None:
        return _dc_cache

    df = load_data()
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    index = {tm: i for i, tm in enumerate(teams)}
    n = len(teams)

    home_idx = df["HomeTeam"].map(index).to_numpy()
    away_idx = df["AwayTeam"].map(index).to_numpy()
    home_goals = df["FTHG"].to_numpy()
    away_goals = df["FTAG"].to_numpy()

    def neg_log_likelihood(params: np.ndarray) -> float:
        attack = params[:n]
        defense = params[n:2 * n]
        home_adv = params[2 * n]
        lambda_home = np.exp(home_adv + attack[home_idx] - defense[away_idx])
        lambda_away = np.exp(attack[away_idx] - defense[home_idx])
        ll = poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(away_goals, lambda_away)
        return -float(ll.sum())

    # Mismos valores iniciales que el informe en R: ataques/defensas a 0, ventaja 0.1.
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1]])
    result = minimize(neg_log_likelihood, x0, method="BFGS", options={"maxiter": 10000})

    attack = result.x[:n]
    defense = result.x[n:2 * n]
    home_adv = float(result.x[2 * n])

    _dc_cache = {
        "teams": teams,
        "index": index,
        "attack": attack,
        "defense": defense,
        "home_adv": home_adv,
        "home_factor": float(np.exp(home_adv)),   # multiplicador sobre goles esperados
        # Medias de Poisson de cada partido real (pre-calculadas para simular).
        "lambda_home": np.exp(home_adv + attack[home_idx] - defense[away_idx]),
        "lambda_away": np.exp(attack[away_idx] - defense[home_idx]),
        "home_idx": home_idx,
        "away_idx": away_idx,
    }
    return _dc_cache


def attack_defense_table() -> pd.DataFrame:
    """Tabla ataque/defensa por equipo (util para depurar o ampliar el dashboard)."""
    dc = fit_dixon_coles()
    return (
        pd.DataFrame({
            "Team": dc["teams"],
            "Attack": np.round(dc["attack"], 4),
            "Defense": np.round(dc["defense"], 4),
        })
        .sort_values("Attack", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Simulacion Monte Carlo de la temporada
# --------------------------------------------------------------------------- #
def simulate_season(n_sims: int = 10000, seed: int | None = None) -> pd.DataFrame:
    """
    Juega la temporada n_sims veces con marcadores Poisson(lambda) y agrega
    probabilidades de titulo, Top 4 y descenso, mas los puntos medios.

    Implementacion vectorizada en NumPy: matrices (n_sims, n_partidos). Devuelve
    la clasificacion predictiva ordenada por puntos medios.
    """
    dc = fit_dixon_coles()
    teams = dc["teams"]
    n = len(teams)
    home_idx, away_idx = dc["home_idx"], dc["away_idx"]
    n_matches = home_idx.size

    rng = np.random.default_rng(seed)
    # Marcadores simulados (int16 sobra: rara vez >10 goles).
    home_goals = rng.poisson(dc["lambda_home"], size=(n_sims, n_matches)).astype(np.int16)
    away_goals = rng.poisson(dc["lambda_away"], size=(n_sims, n_matches)).astype(np.int16)

    # Puntos por partido (3/1/0) para local y visitante.
    home_pts = np.where(home_goals > away_goals, 3,
                        np.where(home_goals == away_goals, 1, 0)).astype(np.int16)
    away_pts = np.where(away_goals > home_goals, 3,
                        np.where(home_goals == away_goals, 1, 0)).astype(np.int16)

    # Acumular puntos por equipo (bucle corto: 20 equipos).
    points = np.zeros((n_sims, n), dtype=np.int32)
    for t in range(n):
        points[:, t] = home_pts[:, home_idx == t].sum(axis=1) + away_pts[:, away_idx == t].sum(axis=1)

    # Ranking por simulacion, rompiendo empates al azar (como rank(ties="random")).
    jitter = points + rng.uniform(0, 0.5, size=points.shape)
    order = np.argsort(-jitter, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)

    champion = (ranks == 1).mean(axis=0) * 100
    top4 = (ranks <= 4).mean(axis=0) * 100
    relegation = (ranks >= 18).mean(axis=0) * 100
    avg_points = points.mean(axis=0)

    standings = pd.DataFrame({
        "Team": teams,
        "AvgPoints": np.round(avg_points, 1),
        "ChampionPct": np.round(champion, 1),
        "Top4Pct": np.round(top4, 1),
        "RelegationPct": np.round(relegation, 1),
    }).sort_values("AvgPoints", ascending=False).reset_index(drop=True)
    standings.insert(0, "Position", range(1, len(standings) + 1))
    return standings
