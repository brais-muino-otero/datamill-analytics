"""
utils/plotly_theme.py
=====================
Tema visual de Plotly: coherente con DARKLY (Bootstrap) y, sobre todo,
SEGURO PARA DALTONICOS (colorblind-safe) por defecto en TODAS las graficas.

Por que esta paleta:
- Categorica: paleta de Okabe & Ito (el estandar "Color Universal Design").
  Disenada para ser distinguible por personas con deuteranopia/protanopia
  (los tipos de daltonismo mas comunes). Sustituimos el negro original por
  blanco porque sobre fondo oscuro el negro seria invisible.
- Secuencial (escalas continuas): Viridis -> perceptualmente uniforme y CVD-safe.
- Divergente: RdBu (azul<->rojo). Evita el eje rojo-verde, problematico para CVD.

Aplicamos el tema de DOS formas para cubrir todo el ecosistema Plotly:
  1. Un go.layout.Template registrado y puesto como default -> afecta a graph_objects.
  2. Defaults de Plotly Express (px.defaults.*)             -> afecta a px.
"""
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# --------------------------------------------------------------------------- #
# Paletas seguras para daltonicos
# --------------------------------------------------------------------------- #
# Okabe-Ito adaptada a fondo oscuro (negro -> blanco al final).
COLORBLIND_SAFE_QUALITATIVE = [
    "#56B4E9",  # azul cielo
    "#E69F00",  # naranja
    "#009E73",  # verde azulado
    "#F0E442",  # amarillo
    "#0072B2",  # azul
    "#D55E00",  # bermellon
    "#CC79A7",  # purpura rosado
    "#FFFFFF",  # blanco (sustituye al negro original, invisible en dark mode)
]
COLORBLIND_SAFE_SEQUENTIAL = "Viridis"   # continua, perceptualmente uniforme
COLORBLIND_SAFE_DIVERGING = "RdBu"       # divergente sin eje rojo-verde

# Colores de UI alineados con el tema DARKLY (#222 de fondo).
_TEXT_COLOR = "#E0E0E0"     # texto claro legible sobre oscuro
_GRID_COLOR = "#444444"     # rejilla sutil
_AXIS_COLOR = "#666666"     # ejes / lineas cero
TEMPLATE_NAME = "pl_darkly_cvd"   # premier league + darkly + colorblind-safe


def _build_template() -> go.layout.Template:
    """Construye el Template de Plotly con estetica dark + colores CVD-safe."""
    template = go.layout.Template()
    template.layout = go.Layout(
        # Color de las trazas categoricas (lineas, barras, etc.)
        colorway=COLORBLIND_SAFE_QUALITATIVE,
        # Fondos transparentes: la grafica hereda el fondo del dbc.Card/pagina
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Tipografia
        font=dict(color=_TEXT_COLOR, family="Helvetica Neue, Helvetica, Arial, sans-serif"),
        title=dict(font=dict(color="#FFFFFF", size=20)),
        # Ejes
        xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_AXIS_COLOR, linecolor=_AXIS_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_AXIS_COLOR, linecolor=_AXIS_COLOR),
        # Escalas de color continuas / divergentes
        colorscale=dict(
            sequential=COLORBLIND_SAFE_SEQUENTIAL,
            sequentialminus=COLORBLIND_SAFE_SEQUENTIAL,
            diverging=COLORBLIND_SAFE_DIVERGING,
        ),
        # Leyenda
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID_COLOR),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return template


def apply_plotly_theme() -> None:
    """
    Registra el template y lo fija como DEFAULT global.
    Llamar UNA vez al arrancar la app (en app.py).

    Componemos 'plotly_dark+pl_darkly_cvd': partimos del tema oscuro oficial de
    Plotly y le superponemos nuestros colores CVD-safe.
    """
    pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = f"plotly_dark+{TEMPLATE_NAME}"

    # Plotly Express no usa pio.templates.default para los colores discretos,
    # asi que fijamos tambien sus defaults explicitamente.
    px.defaults.template = f"plotly_dark+{TEMPLATE_NAME}"
    px.defaults.color_discrete_sequence = COLORBLIND_SAFE_QUALITATIVE
    px.defaults.color_continuous_scale = COLORBLIND_SAFE_SEQUENTIAL
