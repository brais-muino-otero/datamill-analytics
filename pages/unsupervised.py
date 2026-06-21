"""
pages/unsupervised.py
====================
Pagina "Aprendizaje No Supervisado" (ruta /aprendizaje-no-supervisado).

Descubre estilos de juego (perfiles) de los 20 equipos de la Premier League
2022-23. Toda la matematica vive en utils/unsupervised_engine; aqui solo se
construye la UI, se dibujan los graficos (Plotly CVD-safe) y se redacta el
storytelling.

Estructura (segun el enunciado):
- Controles: dropdown maestro (K-Means / Jerarquico / SOM) + controles dinamicos
  que fuerzan las configuraciones requeridas (k 2-7, los 5 metodos jerarquicos
  incluido centroid, las 6 cuadriculas SOM simetricas).
- Espectaculo visual: scatter PCA 2D (equipos en PC1/PC2, color = cluster) +
  radar (un poligono por cluster con la media de los 5 KPIs estandarizados).
- Panel de rigor: componentes PCA para [0.70..0.95] de varianza + Silhouette
  Score de la configuracion elegida.
- Storytelling dinamico (dcc.Markdown): ADN de cada cluster (forma del radar) y
  justificacion matematica via Silhouette.
"""
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, clientside_callback, dcc, html

from config import DEFAULT_LANGUAGE
from i18n.translator import translator
from utils import unsupervised_engine as ue
from utils.plotly_theme import COLORBLIND_SAFE_QUALITATIVE as PALETTE

_P = "pages.unsupervised"
_ALGOS = ["kmeans", "hierarchical", "som"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _params(algo, k, method, grid):
    """Construye el objeto de config para el motor a partir de los controles."""
    if algo == "kmeans":
        return int(k)
    if algo == "hierarchical":
        return (method, int(k))
    return tuple(int(v) for v in grid.split(","))     # som "x,y" -> (x, y)


def _config_human(algo, params, t) -> str:
    if algo == "kmeans":
        return t(f"{_P}.k_option", n=params)
    if algo == "hierarchical":
        method, k = params
        return f"{t(f'{_P}.method.{method}')}, {t(f'{_P}.k_option', n=k)}"
    x, y = params
    return f"{x}\u00d7{y}"


def _kpi_labels(t):
    return [t(f"{_P}.kpi.volume"), t(f"{_P}.kpi.efficiency"),
            t(f"{_P}.kpi.solidity"), t(f"{_P}.kpi.control"),
            t(f"{_P}.kpi.aggressiveness")]


def _sil_quality(sil):
    """(clave_verdicto, clave_frase, color_badge) segun el Silhouette."""
    if sil is None:
        return "na", "na", "secondary"
    if sil >= 0.50:
        return "strong", "strong", "success"
    if sil >= 0.35:
        return "reasonable", "reasonable", "info"
    if sil >= 0.25:
        return "weak", "weak", "warning"
    return "poor", "poor", "danger"


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _placeholder_fig(t) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=t(f"{_P}.run_prompt"), showarrow=False,
                       font={"size": 15, "color": "#888"})
    fig.update_layout(height=420, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def _scatter_figure(labels, t) -> go.Figure:
    """Scatter PCA 2D: 20 equipos en PC1/PC2, texto = nombre, color = cluster."""
    teams, _ = ue.scaled_matrix()
    rep = ue.pca_report()
    coords = np.array(rep["coords"])
    labels = np.asarray(labels)

    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        mask = labels == c
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers+text",
            text=[teams[i] for i in np.where(mask)[0]],
            textposition="top center", textfont={"size": 11},
            marker={"size": 14, "color": _color(int(c)),
                    "line": {"width": 1, "color": "#222"}},
            name=t(f"{_P}.cluster_name", n=int(c) + 1),
            hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title=t(f"{_P}.scatter_title"),
        xaxis_title=t(f"{_P}.pc_axis", n=1, pct=f"{rep['pc_var'][0]:.1f}"),
        yaxis_title=t(f"{_P}.pc_axis", n=2, pct=f"{rep['pc_var'][1]:.1f}"),
        height=460, legend={"orientation": "h", "y": -0.18},
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _radar_figure(labels, t) -> go.Figure:
    """Radar: un poligono por cluster con la media de los 5 KPIs estandarizados."""
    _, Xs = ue.scaled_matrix()
    labels = np.asarray(labels)
    axes = _kpi_labels(t)
    theta = axes + [axes[0]]                          # cerrar el poligono

    means = {int(c): Xs[labels == c].mean(axis=0) for c in np.unique(labels)}
    all_vals = np.concatenate(list(means.values()))
    lo, hi = float(all_vals.min()) - 0.4, float(all_vals.max()) + 0.4

    fig = go.Figure()
    for c in sorted(means):
        r = means[c].tolist()
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]], theta=theta, fill="toself", opacity=0.55,
            name=t(f"{_P}.cluster_name", n=c + 1),
            line={"color": _color(c), "width": 2},
        ))
    fig.update_layout(
        title=t(f"{_P}.radar_title"),
        polar={"radialaxis": {"range": [lo, hi], "showline": False,
                              "tickfont": {"size": 9}}},
        height=460, legend={"orientation": "h", "y": -0.12},
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


# --------------------------------------------------------------------------- #
# Storytelling
# --------------------------------------------------------------------------- #
def _cluster_dna_story(labels, t) -> list:
    """Una linea por cluster describiendo su ADN (KPI dominante / deficitario)."""
    teams, Xs = ue.scaled_matrix()
    labels = np.asarray(labels)
    axes = _kpi_labels(t)
    parts = [t(f"{_P}.story_dna_header")]
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        mean = Xs[idx].mean(axis=0)
        names = [teams[i] for i in idx]
        sample = ", ".join(names[:3]) + ("\u2026" if len(names) > 3 else "")
        hi_i, lo_i = int(mean.argmax()), int(mean.argmin())
        high, low = mean[hi_i], mean[lo_i]
        common = dict(name=t(f"{_P}.cluster_name", n=c + 1), size=len(idx), teams=sample)
        if high > 0.5 and low < -0.5:
            parts.append(t(f"{_P}.story_cluster_both", high=axes[hi_i], low=axes[lo_i], **common))
        elif high > 0.5:
            parts.append(t(f"{_P}.story_cluster_high", high=axes[hi_i], **common))
        elif low < -0.5:
            parts.append(t(f"{_P}.story_cluster_low", low=axes[lo_i], **common))
        else:
            parts.append(t(f"{_P}.story_cluster_balanced", **common))
    return parts


def _story(algo, params, res, t) -> str:
    intro = t(f"{_P}.story_intro", algo=t(f"{_P}.algo.{algo}"),
              config=_config_human(algo, params, t), n=res["n_clusters"])
    sil = res["silhouette"]
    if sil is None:
        quality = t(f"{_P}.story_silhouette_na")
    else:
        _, phrase_key, _ = _sil_quality(sil)
        quality = t(f"{_P}.story_silhouette", sil=f"{sil:.3f}",
                    quality=t(f"{_P}.quality_phrase.{phrase_key}"))
    return "\n\n".join([intro, quality] + _cluster_dna_story(res["labels"], t))


# =========================================================================== #
# Layout
# =========================================================================== #
def _pca_panel(t) -> dbc.Card:
    """Panel estatico: nº de componentes PCA para cada umbral de varianza."""
    rep = ue.pca_report()
    cards = []
    for item in rep["thresholds"]:
        cards.append(dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(t(f"{_P}.pca_threshold", pct=int(item["threshold"] * 100)),
                     className="stat-label"),
            html.Div(str(item["n_components"]), className="stat-value"),
            html.Div(t(f"{_P}.pca_components_unit"), className="stat-sub"),
        ]), className="stat-card"), xs=6, md=2))
    return dbc.Card(dbc.CardBody([
        html.H5(t(f"{_P}.pca_panel_title")),
        dbc.Row(cards, className="g-2"),
    ]), className="mb-3")


def layout(t) -> dbc.Container:
    algo_options = [{"label": t(f"{_P}.algo.{a}"), "value": a} for a in _ALGOS]
    k_options = [{"label": t(f"{_P}.k_option", n=k), "value": str(k)} for k in ue.KMEANS_K]
    method_options = [{"label": t(f"{_P}.method.{m}"), "value": m} for m in ue.HIER_METHODS]
    grid_options = [{"label": f"{x}\u00d7{y}", "value": f"{x},{y}"} for (x, y) in ue.SOM_GRIDS]

    controls = dbc.Card(dbc.CardBody([
        html.Label(t(f"{_P}.algo_label"), htmlFor="uns-algo", className="control-label"),
        dcc.Dropdown(id="uns-algo", options=algo_options, value="kmeans",
                     clearable=False, className="control-dropdown mb-3"),
        # k (K-Means y Jerarquico)
        html.Div([
            html.Label(t(f"{_P}.k_label"), htmlFor="uns-k", className="control-label"),
            dcc.Dropdown(id="uns-k", options=k_options, value="3",
                         clearable=False, className="control-dropdown mb-3"),
        ], id="uns-k-wrap"),
        # metodo (Jerarquico)
        html.Div([
            html.Label(t(f"{_P}.method_label"), htmlFor="uns-method", className="control-label"),
            dcc.Dropdown(id="uns-method", options=method_options, value="ward",
                         clearable=False, className="control-dropdown mb-3"),
        ], id="uns-method-wrap", style={"display": "none"}),
        # cuadricula (SOM)
        html.Div([
            html.Label(t(f"{_P}.grid_label"), htmlFor="uns-grid", className="control-label"),
            dcc.Dropdown(id="uns-grid", options=grid_options, value="5,5",
                         clearable=False, className="control-dropdown mb-3"),
        ], id="uns-grid-wrap", style={"display": "none"}),
        dbc.Button(t(f"{_P}.run_button"), id="uns-run", color="primary",
                   n_clicks=0, className="mc-run-btn w-100"),
    ]), className="controls-card")

    silhouette_card = dbc.Card(dbc.CardBody([
        html.Div(t(f"{_P}.silhouette_label"), className="stat-label"),
        html.Div("\u2014", id="uns-sil-value", className="stat-value"),
        html.Div(dbc.Badge(t(f"{_P}.run_prompt_badge"), color="secondary",
                           className="verdict-badge"), id="uns-sil-verdict"),
    ]), className="stat-card")

    visuals = dcc.Loading(dbc.Row([
        dbc.Col(dcc.Graph(id="uns-scatter", figure=_placeholder_fig(t),
                          config={"displayModeBar": False}), lg=7, md=12),
        dbc.Col(dcc.Graph(id="uns-radar", figure=_placeholder_fig(t),
                          config={"displayModeBar": False}), lg=5, md=12),
    ], className="g-3"), type="default")

    return dbc.Container([
        html.H1(t(f"{_P}.title")),
        html.P(t(f"{_P}.description"), className="lead"),
        html.Div(
            dbc.Button(
                [html.Span("\U0001F4C4", className="me-2"), t(f"{_P}.export_report")],
                id="uns-export-btn", color="light", outline=True, size="sm",
                className="export-btn"),
            className="d-flex justify-content-end mb-2 no-print",
        ),
        html.Hr(),
        dcc.Markdown(t(f"{_P}.theory"), className="theory-block"),
        _pca_panel(t),
        dbc.Row([
            dbc.Col(controls, lg=4, md=12),
            dbc.Col([
                visuals,
                dbc.Row(dbc.Col(silhouette_card, md=6), className="mt-3"),
            ], lg=8, md=12),
        ], className="g-3"),
        dcc.Markdown(id="uns-story", className="story-block",
                     children=t(f"{_P}.run_prompt")),
        # Anclaje invisible para el callback de impresion (exportar a PDF).
        dcc.Store(id="uns-print-anchor"),
    ], fluid=True)


# =========================================================================== #
# Callback 1: el algoritmo elegido muestra/oculta los controles que aplican
# =========================================================================== #
@callback(
    Output("uns-k-wrap", "style"),
    Output("uns-method-wrap", "style"),
    Output("uns-grid-wrap", "style"),
    Input("uns-algo", "value"),
)
def _toggle_controls(algo):
    show, hide = {}, {"display": "none"}
    if algo == "kmeans":
        return show, hide, hide
    if algo == "hierarchical":
        return show, show, hide
    return hide, hide, show          # som


# =========================================================================== #
# Callback 2: "Ejecutar" -> clustering, scatter, radar, Silhouette y storytelling
# =========================================================================== #
@callback(
    Output("uns-scatter", "figure"),
    Output("uns-radar", "figure"),
    Output("uns-sil-value", "children"),
    Output("uns-sil-verdict", "children"),
    Output("uns-story", "children"),
    Input("uns-run", "n_clicks"),
    State("uns-algo", "value"),
    State("uns-k", "value"),
    State("uns-method", "value"),
    State("uns-grid", "value"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def _run(n_clicks, algo, k, method, grid, lang):
    t = translator.scoped(lang or DEFAULT_LANGUAGE)
    algo = algo or "kmeans"
    params = _params(algo, k, method, grid)
    res = ue.cluster_result(algo, params)

    scatter = _scatter_figure(res["labels"], t)
    radar = _radar_figure(res["labels"], t)

    sil = res["silhouette"]
    sil_value = f"{sil:.3f}" if sil is not None else t(f"{_P}.silhouette_na")
    verdict_key, _, color = _sil_quality(sil)
    verdict = dbc.Badge(t(f"{_P}.sil_verdict.{verdict_key}"), color=color,
                        className="verdict-badge")

    story = _story(algo, params, res, t)
    return scatter, radar, sil_value, verdict, story


# =========================================================================== #
# Callback (clientside): "Exportar Informe" -> dialogo de impresion del navegador
# =========================================================================== #
# window.print() nativo (sin weasyprint/pdfkit). El CSS @media print de
# assets/custom.css limpia la salida (oculta sidebar, fondos blancos, 100% ancho).
clientside_callback(
    "function (n) { if (n) { window.print(); } return window.dash_clientside.no_update; }",
    Output("uns-print-anchor", "data"),
    Input("uns-export-btn", "n_clicks"),
    prevent_initial_call=True,
)
