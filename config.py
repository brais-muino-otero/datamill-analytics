"""
config.py
=========
Configuracion global y centralizada de la aplicacion.

Mantener TODAS las constantes "magicas" (rutas, idiomas, paths, nombre de la
app) en un unico modulo evita strings duplicados por el codigo y hace que
cambiar algo (p.ej. anadir un idioma o renombrar una ruta) sea trivial.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Identidad de la aplicacion
# --------------------------------------------------------------------------- #
APP_TITLE = "Premier League Analytics"   # Marca: NO se traduce (nombre propio)

# --------------------------------------------------------------------------- #
# Internacionalizacion (i18n)
# --------------------------------------------------------------------------- #
# Idioma por defecto / de respaldo (fallback) si falta una clave o un idioma.
DEFAULT_LANGUAGE = "es"

# Idiomas soportados: {codigo ISO 639-1: endonimo (nombre en su propia lengua)}.
# Los endonimos NO se traducen: un selector de idioma siempre muestra cada
# lengua escrita en si misma ("Deutsch", no "Aleman"). Por eso son constantes.
LANGUAGES = {
    "es": "Español",
    "en": "English",
    "pt": "Português",
    "it": "Italiano",
    "fr": "Français",
    "de": "Deutsch",
}

# --------------------------------------------------------------------------- #
# Rutas (URL paths) de cada pagina
# --------------------------------------------------------------------------- #
class Routes:
    """Rutas de la SPA. Usar constantes evita typos en hrefs y en el router."""
    FOOTBALL = "/"                          # Home = Futbol Analytics
    SUPERVISED = "/aprendizaje-supervisado"
    UNSUPERVISED = "/aprendizaje-no-supervisado"
    KNOWLEDGE_HUB = "/knowledge-hub"

# --------------------------------------------------------------------------- #
# Paths del sistema de ficheros
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent      # Carpeta raiz del proyecto
LOCALES_DIR = BASE_DIR / "i18n" / "locales"     # Diccionarios JSON de traduccion
DATA_DIR = BASE_DIR / "data"                    # Datasets del proyecto
DATA_FILE = DATA_DIR / "epl_results_2022-23.csv"  # Resultados Premier League 22/23

# --------------------------------------------------------------------------- #
# Parametros de analisis
# --------------------------------------------------------------------------- #
TOP_N_TEAMS = 6   # "Top 6": tamano del subconjunto de elite para los filtros

# --------------------------------------------------------------------------- #
# Aprendizaje supervisado (replica de la metodologia de la Practica 2 de AA1)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 1234   # Semilla global UNICA: reproducibilidad total (igual que el .jl)
CV_FOLDS = 10         # Validacion cruzada estratificada de 10 particiones
ROLLING_WINDOW = 5    # Medias moviles de los 5 partidos previos (features pre-partido)
MIN_HISTORY = 3       # Historial minimo por equipo (descarta jornadas iniciales)
