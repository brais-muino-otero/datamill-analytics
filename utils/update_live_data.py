"""
utils/update_live_data.py
=========================
Live Data Pipeline (MLOps) — script INDEPENDIENTE para ejecutar via cron.

Descarga los resultados mas recientes de la Premier League desde la API de
football-data.org, los transforma al MISMO formato que data/epl_results_2022-23.csv
y sobrescribe ese fichero (de forma atomica) para que la app sirva datos frescos.

Uso:
    export FOOTBALL_API_TOKEN="tu_token"
    python -m utils.update_live_data        # o: python utils/update_live_data.py

Cron (cada dia a las 04:30):
    30 4 * * *  cd /ruta/al/proyecto && /ruta/al/.venv/bin/python -m utils.update_live_data >> /var/log/datamill_update.log 2>&1

Variables de entorno:
    FOOTBALL_API_TOKEN   (obligatoria)  Token de https://www.football-data.org/
    FOOTBALL_SEASON      (opcional)     Año de inicio de temporada, p.ej. "2024".
                                        Si no se indica, la API usa la temporada actual.

Nota de honestidad de datos
---------------------------
El plan gratuito de football-data.org expone marcador y fecha, pero NO las
estadisticas de tiros/corners/faltas/tarjetas ni el arbitro. Para mantener el
ESQUEMA del CSV intacto, esas columnas se rellenan con marcadores (0 / vacio) y se
emite un WARNING. Para estadisticas completas usa football-data.co.uk o un plan
de pago, manteniendo este mismo esquema de salida.

Requiere: requests  (pip install requests)
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
API_BASE = "https://api.football-data.org/v4"
COMPETITION = "PL"                      # Premier League
REQUEST_TIMEOUT = 30                    # segundos
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "epl_results_2022-23.csv"

# Orden EXACTO de columnas del CSV (no cambiar: la app depende de el).
CSV_COLUMNS = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "Referee",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]
# Columnas de estadisticas no disponibles en el plan gratuito (placeholders).
STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

# Nombres de football-data.org -> nombres cortos usados en nuestro CSV.
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update_live_data")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalize_team(name: str) -> str:
    """Mapea el nombre de la API al nombre corto del CSV (con fallback robusto)."""
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # Fallback: limpiar sufijos/prefijos habituales.
    cleaned = name.removesuffix(" FC").removeprefix("AFC ").strip()
    logger.warning("Equipo sin mapear: '%s' -> usando '%s'", name, cleaned)
    return cleaned


def _outcome(home: int, away: int) -> str:
    """Resultado 1X2: 'H' local, 'A' visitante, 'D' empate."""
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def fetch_matches(token: str, season: str | None) -> list[dict]:
    """Descarga los partidos de la competicion desde football-data.org."""
    url = f"{API_BASE}/competitions/{COMPETITION}/matches"
    params = {"season": season} if season else {}
    headers = {"X-Auth-Token": token}

    logger.info("Solicitando datos a %s (season=%s)...", url, season or "actual")
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 403:
        raise PermissionError("Token invalido o sin permisos (HTTP 403).")
    if resp.status_code == 429:
        raise RuntimeError("Limite de peticiones excedido (HTTP 429). Reintenta mas tarde.")
    resp.raise_for_status()

    matches = resp.json().get("matches", [])
    logger.info("Recibidos %d partidos en total.", len(matches))
    return matches


def transform(matches: list[dict]) -> list[dict]:
    """Convierte los partidos FINALIZADOS al esquema de filas del CSV."""
    rows: list[dict] = []
    skipped = 0
    for m in matches:
        if m.get("status") != "FINISHED":
            skipped += 1
            continue
        try:
            score = m["score"]
            ft = score["fullTime"]
            ht = score.get("halfTime", {}) or {}
            fthg, ftag = int(ft["home"]), int(ft["away"])
            # Marcador de descanso: puede venir nulo en algunos partidos.
            hthg = ht.get("home")
            htag = ht.get("away")

            dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(timezone.utc)

            row = {
                "Date": dt.strftime("%d/%m/%Y"),
                "Time": dt.strftime("%H:%M"),
                "HomeTeam": _normalize_team(m["homeTeam"]["name"]),
                "AwayTeam": _normalize_team(m["awayTeam"]["name"]),
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": _outcome(fthg, ftag),
                "HTHG": "" if hthg is None else int(hthg),
                "HTAG": "" if htag is None else int(htag),
                "HTR": "" if hthg is None or htag is None else _outcome(int(hthg), int(htag)),
                "Referee": "",   # No disponible en el plan gratuito.
            }
            # Estadisticas no disponibles -> placeholders (esquema intacto).
            for col in STAT_COLUMNS:
                row[col] = 0
            rows.append((dt, row))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Partido omitido por datos incompletos: %s", exc)

    # Orden cronologico ascendente (como el CSV original).
    rows.sort(key=lambda pair: pair[0])
    logger.info("Transformados %d partidos finalizados (%d no finalizados omitidos).",
                len(rows), skipped)
    return [r for _, r in rows]


def write_csv_atomic(rows: list[dict], path: Path) -> None:
    """Escribe el CSV de forma ATOMICA (temp + replace) para no servir datos a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)       # operacion atomica en el mismo sistema de ficheros
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    logger.info("=== Live Data Pipeline: inicio ===")
    token = os.environ.get("FOOTBALL_API_TOKEN")
    if not token:
        logger.error("Falta la variable de entorno FOOTBALL_API_TOKEN. Abortando.")
        return 1

    season = os.environ.get("FOOTBALL_SEASON")  # opcional
    try:
        matches = fetch_matches(token, season)
        rows = transform(matches)
        if not rows:
            logger.error("No hay partidos finalizados para escribir. Se conserva el CSV actual.")
            return 1

        write_csv_atomic(rows, DATA_FILE)
        logger.warning(
            "football-data.org (plan gratuito) no aporta tiros/corners/faltas/tarjetas/arbitro: "
            "esas columnas quedan a 0/vacio. Usa football-data.co.uk o un plan de pago para stats completas."
        )
        logger.info("CSV actualizado correctamente: %s (%d partidos).", DATA_FILE, len(rows))
        logger.info("=== Live Data Pipeline: fin OK ===")
        return 0

    except requests.exceptions.Timeout:
        logger.exception("Timeout al contactar con la API.")
    except requests.exceptions.RequestException as exc:
        logger.exception("Error de red/HTTP: %s", exc)
    except (PermissionError, RuntimeError) as exc:
        logger.exception("Error de la API: %s", exc)
    except Exception as exc:  # red de seguridad: nada debe tumbar el cron sin loguear
        logger.exception("Error inesperado: %s", exc)
    logger.info("=== Live Data Pipeline: fin con ERRORES ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
