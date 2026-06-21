"""
pages/football_analytics.py
==========================
Pagina "Futbol Analytics" (home, ruta "/").

Dashboard con tres pestanas (dbc.Tabs):
  1. Correlaciones         -> mapa de calor interactivo con filtro de ambito.
  2. Inferencia            -> contraste t de la ventaja de campo + storytelling.
  3. Simulador Monte Carlo -> clasificacion simulada (Dixon-Coles) bajo demanda.

Arquitectura:
- TODA la matematica vive en utils/stats_engine. Aqui solo se construye la UI y
  se dibujan los resultados (separacion de responsabilidades).
- Los callbacks usan el registro global de Dash (@callback) para no depender de
  la instancia `app` (evita import circular: app importa pages, no al reves).
- Cada callback lee el idioma activo desde dcc.Store(language-store) y traduce el
  "data storytelling" en tiempo real con el sistema i18n del proyecto.
"""
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from config import DEFAULT_LANGUAGE
from i18n.translator import translator
from utils import stats_engine
from utils.plotly_theme import COLORBLIND_SAFE_QUALITATIVE as CVD

# Prefijo comun de las claves i18n de esta pagina (para no repetirlo).
_P = "pages.football_analytics"

# Colores fijos (de la paleta CVD-safe del proyecto) para el grafico de medias:
# necesitamos rellenos translucidos, asi que los fijamos en vez de heredarlos.
_C_HOME = CVD[0]   # azul cielo
_C_AWAY = CVD[1]   # naranja


# --------------------------------------------------------------------------- #
# Utilidades de presentacion (formato; nada de matematica real aqui)
# --------------------------------------------------------------------------- #
_SUPERSCRIPT = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _format_p(p: float) -> str:
    """p-valor legible: decimal si es grande, notacion cientifica si es infimo."""
    if p <= 0:
        return "\u2248 0"
    if p >= 0.001:
        return f"{p:.3f}"
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / 10 ** exponent
    return f"{mantissa:.2f} \u00d7 10{str(exponent).translate(_SUPERSCRIPT)}"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte '#RRGGBB' a 'rgba(r,g,b,a)' para rellenos translucidos."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _stat_card(label_key: str, value_id: str, t) -> dbc.Card:
    """Tarjeta KPI compacta: etiqueta fija + valor que rellena un callback."""
    return dbc.Card(
        dbc.CardBody([
            html.Div(t(label_key), className="stat-label"),
            html.Div(id=value_id, className="stat-value"),
        ]),
        className="stat-card",
    )


# =========================================================================== #
# PESTANA 1 - Correlaciones
# =========================================================================== #
def _tab_correlations(t) -> dbc.Container:
    return dbc.Container([
        dcc.Markdown(t(f"{_P}.correlations.theory"), className="theory-block"),
        html.Label(t(f"{_P}.correlations.scope_label"), htmlFor="corr-scope",
                   className="control-label"),
        dbc.RadioItems(
            id="corr-scope",
            options=[
                {"label": t(f"{_P}.correlations.scope_all"), "value": "all"},
                {"label": t(f"{_P}.correlations.scope_top6"), "value": "top6"},
            ],
            value="all",
            inline=True,
            className="mb-3",
        ),
        dcc.Graph(id="corr-heatmap", config={"displayModeBar": False}),
    ], fluid=True, className="tab-body")


@callback(
    Output("corr-heatmap", "figure"),
    Input("corr-scope", "value"),
    State("language-store", "data"),
)
def _update_correlations(scope, lang):
    """Filtro de ambito -> mapa de calor de correlaciones (diverging CVD-safe)."""
    t = translator.scoped(lang or DEFAULT_LANGUAGE)
    scope = scope or "all"
    keys, matrix = stats_engine.correlation_matrix(scope)
    labels = [t(f"{_P}.correlations.vars.{k}") for k in keys]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,   # azul/rojo: evita rojo-verde
        texttemplate="%{z:.2f}", textfont={"size": 13},
        colorbar=dict(title=dict(text="r"), outlinewidth=0),
        hovertemplate="%{y} \u2013 %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    scope_label = t(f"{_P}.correlations.scope_all" if scope == "all"
                    else f"{_P}.correlations.scope_top6")
    fig.update_layout(
        title=f"{t(f'{_P}.correlations.heatmap_title')} \u00b7 {scope_label}",
        yaxis={"autorange": "reversed"},   # diagonal de arriba-izq a abajo-der
        height=480,
        margin=dict(l=120, r=30, t=60, b=90),
    )
    return fig


# =========================================================================== #
# PESTANA 2 - Inferencia estadistica
# =========================================================================== #
def _tab_inference(t) -> dbc.Container:
    return dbc.Container([
        dcc.Markdown(t(f"{_P}.inference.theory"), className="theory-block"),
        html.Label(t(f"{_P}.inference.metric_label"), htmlFor="inf-metric",
                   className="control-label"),
        dcc.Dropdown(
            id="inf-metric",
            options=[
                {"label": t(f"{_P}.inference.metric_goals"), "value": "goals"},
                {"label": t(f"{_P}.inference.metric_shots"), "value": "shots"},
            ],
            value="goals", clearable=False, className="control-dropdown mb-3",
        ),
        dbc.Row([
            dbc.Col(_stat_card(f"{_P}.inference.stat_t", "inf-tstat", t), md=3, xs=6),
            dbc.Col(_stat_card(f"{_P}.inference.stat_p", "inf-pvalue", t), md=3, xs=6),
            dbc.Col(_stat_card(f"{_P}.inference.stat_mean_home", "inf-mean-home", t), md=3, xs=6),
            dbc.Col(_stat_card(f"{_P}.inference.stat_mean_away", "inf-mean-away", t), md=3, xs=6),
        ], className="g-2 mb-3"),
        dcc.Graph(id="inf-plot", config={"displayModeBar": False}),
        dcc.Markdown(id="inf-story", className="story-block"),
    ], fluid=True, className="tab-body")


@callback(
    Output("inf-tstat", "children"),
    Output("inf-pvalue", "children"),
    Output("inf-mean-home", "children"),
    Output("inf-mean-away", "children"),
    Output("inf-plot", "figure"),
    Output("inf-story", "children"),
    Input("inf-metric", "value"),
    State("language-store", "data"),
)
def _update_inference(metric, lang):
    """Metrica elegida -> contraste t, tarjetas, grafico de medias y storytelling."""
    t = translator.scoped(lang or DEFAULT_LANGUAGE)
    metric = metric if metric in stats_engine.TTEST_METRICS else "goals"
    r = stats_engine.home_advantage_ttest(metric)

    metric_label = t(f"{_P}.inference.metric_goals" if metric == "goals"
                     else f"{_P}.inference.metric_shots")
    p_disp = _format_p(r["p_value"])

    # Grafico: distribucion muestral de cada media (curvas normales N(media, se)).
    home_label = t(f"{_P}.inference.plot_home")
    away_label = t(f"{_P}.inference.plot_away")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r["curve_x"], y=r["density_home"], mode="lines", name=home_label,
        line={"color": _C_HOME, "width": 2}, fill="tozeroy",
        fillcolor=_hex_to_rgba(_C_HOME, 0.35),
    ))
    fig.add_trace(go.Scatter(
        x=r["curve_x"], y=r["density_away"], mode="lines", name=away_label,
        line={"color": _C_AWAY, "width": 2}, fill="tozeroy",
        fillcolor=_hex_to_rgba(_C_AWAY, 0.35),
    ))
    fig.add_vline(x=r["mean_home"], line_dash="dash", line_color=_C_HOME, opacity=0.7)
    fig.add_vline(x=r["mean_away"], line_dash="dash", line_color=_C_AWAY, opacity=0.7)
    fig.update_layout(
        title=t(f"{_P}.inference.plot_title"),
        xaxis_title=metric_label,
        yaxis_title=t(f"{_P}.inference.plot_density"),
        height=360,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin=dict(l=60, r=30, t=70, b=50),
    )

    story_key = "story_significant" if r["significant"] else "story_not_significant"
    story = t(
        f"{_P}.inference.{story_key}",
        metric=metric_label.lower(),
        t_stat=f"{r['t_stat']:.3f}",
        p_value=p_disp,
        mean_home=f"{r['mean_home']:.3f}",
        mean_away=f"{r['mean_away']:.3f}",
        diff=f"{r['diff']:.3f}",
    )

    return (
        f"{r['t_stat']:.3f}",
        p_disp,
        f"{r['mean_home']:.2f}",
        f"{r['mean_away']:.2f}",
        fig,
        story,
    )


# =========================================================================== #
# PESTANA 3 - Simulador Monte Carlo
# =========================================================================== #
def _tab_montecarlo(t) -> dbc.Container:
    return dbc.Container([
        dcc.Markdown(t(f"{_P}.montecarlo.theory"), className="theory-block"),
        dbc.Row([
            dbc.Col([
                html.Label(t(f"{_P}.montecarlo.nsims_label"), htmlFor="mc-nsims",
                           className="control-label"),
                dbc.Select(
                    id="mc-nsims",
                    options=[
                        {"label": "1 000", "value": "1000"},
                        {"label": "5 000", "value": "5000"},
                        {"label": "10 000", "value": "10000"},
                    ],
                    value="10000",
                ),
            ], md=4, xs=7),
            dbc.Col(
                dbc.Button(t(f"{_P}.montecarlo.run_button"), id="mc-run",
                           color="primary", n_clicks=0, className="mc-run-btn"),
                width="auto", className="d-flex align-items-end",
            ),
        ], className="g-2 mb-3"),
        dcc.Loading(
            html.Div(
                dcc.Markdown(t(f"{_P}.montecarlo.placeholder"), className="placeholder-block"),
                id="mc-results",
            ),
            type="default",
        ),
        dcc.Markdown(id="mc-story", className="story-block"),
    ], fluid=True, className="tab-body")


def _build_standings_table(standings, t) -> dbc.Table:
    """Construye la tabla de clasificacion predictiva a partir del DataFrame."""
    header = html.Thead(html.Tr([
        html.Th(t(f"{_P}.montecarlo.col_pos")),
        html.Th(t(f"{_P}.montecarlo.col_team")),
        html.Th(t(f"{_P}.montecarlo.col_points"), className="num"),
        html.Th(t(f"{_P}.montecarlo.col_champion"), className="num"),
        html.Th(t(f"{_P}.montecarlo.col_top4"), className="num"),
        html.Th(t(f"{_P}.montecarlo.col_relegation"), className="num"),
    ]))
    body_rows = []
    for _, row in standings.iterrows():
        body_rows.append(html.Tr([
            html.Td(int(row["Position"])),
            html.Td(row["Team"], className="team-cell"),
            html.Td(f"{row['AvgPoints']:.1f}", className="num"),
            html.Td(f"{row['ChampionPct']:.1f}", className="num pct-champion"),
            html.Td(f"{row['Top4Pct']:.1f}", className="num"),
            html.Td(f"{row['RelegationPct']:.1f}", className="num pct-relegation"),
        ]))
    return dbc.Table([header, html.Tbody(body_rows)],
                     bordered=False, hover=True, striped=True,
                     responsive=True, className="mc-table")


@callback(
    Output("mc-results", "children"),
    Output("mc-story", "children"),
    Input("mc-run", "n_clicks"),
    State("mc-nsims", "value"),
    State("language-store", "data"),
    prevent_initial_call=True,   # solo corre al pulsar el boton, no en la carga
)
def _run_montecarlo(n_clicks, nsims, lang):
    """Boton -> simula la temporada, dibuja la tabla y narra el resultado."""
    t = translator.scoped(lang or DEFAULT_LANGUAGE)
    n_sims = int(nsims or 10000)

    standings = stats_engine.simulate_season(n_sims=n_sims)
    dc = stats_engine.fit_dixon_coles()

    champion = standings.iloc[0]
    runner_up = standings.iloc[1]
    relegated = standings.sort_values("RelegationPct", ascending=False).iloc[0]

    story = t(
        f"{_P}.montecarlo.story",
        n_sims=f"{n_sims:,}".replace(",", "\u202f"),   # separador de millares fino
        champion=champion["Team"],
        champion_prob=f"{champion['ChampionPct']:.1f}",
        runner_up=runner_up["Team"],
        relegated=relegated["Team"],
        relegation_prob=f"{relegated['RelegationPct']:.1f}",
    )

    note = dcc.Markdown(
        t(f"{_P}.montecarlo.note", factor=f"{dc['home_factor']:.2f}"),
        className="mc-note",
    )
    return [note, _build_standings_table(standings, t)], story


# =========================================================================== #
# Layout de la pagina
# =========================================================================== #
def layout(t) -> dbc.Container:
    """Layout de la pagina en el idioma activo (`t` = traductor scoped)."""
    return dbc.Container([
        html.H1(t(f"{_P}.title")),
        html.P(t(f"{_P}.description"), className="lead"),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(_tab_correlations(t), tab_id="tab-corr",
                        label=t(f"{_P}.tabs.correlations")),
                dbc.Tab(_tab_inference(t), tab_id="tab-inf",
                        label=t(f"{_P}.tabs.inference")),
                dbc.Tab(_tab_montecarlo(t), tab_id="tab-mc",
                        label=t(f"{_P}.tabs.montecarlo")),
            ],
            id="football-tabs", active_tab="tab-corr", className="mt-2",
        ),
    ], fluid=True)
