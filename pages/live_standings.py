# -*- coding: utf-8 -*-
"""
pages/live_standings.py
=======================
Clasificacion EN VIVO de la Premier League (API-Football) + grafico analitico.

- Usa api_client.get_standings() (peticion barata + cache con TTL).
- Es INDEPENDIENTE del dataset del clustering: no toca el CSV ni los motores.
- layout(t) solo dibuja el cascaron; el callback rellena tabla y grafico, de modo
  que un fallo de la API NUNCA rompe el render (muestra un aviso elegante).
- Debajo de la tabla, un grafico de barras horizontales DIVERGENTES con la
  diferencia de goles (GF - GC) de cada equipo: positivas a la derecha,
  negativas a la izquierda. Cada barra se colorea segun la ZONA del equipo en la
  clasificacion (Champions / Europa / Descenso / zona media), con la misma paleta
  CVD-safe que los acentos de la tabla. Fondo transparente (se funde con DARKLY).
- Totalmente traducible y con la misma firma layout(t) que el resto de paginas.
"""
import datetime as dt

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, ctx, dcc, html

import api_client
from i18n.translator import translator

_P = "pages.live_standings"

# Etiqueta de temporada legible a partir del ano de inicio (2025 -> "2025/26").
_SEASON_LABEL = f"{api_client.SEASON}/{str(api_client.SEASON + 1)[-2:]}"

# --------------------------------------------------------------------------- #
# Zonas de la clasificacion: UMBRALES EN UN SOLO SITIO (tabla y grafico iguales)
# Paleta CVD-safe (Okabe-Ito), identica a los acentos de la tabla.
# --------------------------------------------------------------------------- #
_ZONE_CLASS = {  # clase CSS del borde izquierdo en la tabla
    "ucl": "ls-zone-ucl", "uel": "ls-zone-uel",
    "rel": "ls-zone-rel", "mid": "ls-zone-none",
}
_ZONE_COLOR = {  # color de relleno de la barra en el grafico
    "ucl": "#56B4E9",   # azul    -> Champions League
    "uel": "#009E73",   # verde   -> Europa League
    "rel": "#E69F00",   # naranja -> Descenso
    "mid": "#868e96",   # gris neutro -> zona media
}


def _zone_of(pos: int, total: int) -> str:
    """Devuelve la clave de zona ('ucl' / 'uel' / 'rel' / 'mid') segun la posicion."""
    if pos <= 4:
        return "ucl"            # Champions League
    if pos == 5:
        return "uel"            # Europa League
    if pos >= total - 2:
        return "rel"            # Descenso
    return "mid"                # zona media


# --------------------------------------------------------------------------- #
# Helpers de construccion de la tabla
# --------------------------------------------------------------------------- #
def _header(t) -> html.Thead:
    """Cabecera con codigos cortos + tooltip (atributo title) con el nombre completo."""
    # (clave_corta, clave_completa_para_tooltip, alineacion)
    spec = [
        (f"{_P}.col.pos", f"{_P}.full.pos", "text-center"),
        (f"{_P}.col.team", None, "text-start"),
        (f"{_P}.col.played", f"{_P}.full.played", "text-end"),
        (f"{_P}.col.win", f"{_P}.full.win", "text-end"),
        (f"{_P}.col.draw", f"{_P}.full.draw", "text-end"),
        (f"{_P}.col.lose", f"{_P}.full.lose", "text-end"),
        (f"{_P}.col.gf", f"{_P}.full.gf", "text-end"),
        (f"{_P}.col.ga", f"{_P}.full.ga", "text-end"),
        (f"{_P}.col.points", f"{_P}.full.points", "text-end"),
    ]
    cells = []
    for short_key, full_key, align in spec:
        kwargs = {"className": align}
        if full_key:
            kwargs["title"] = t(full_key)   # tooltip accesible al pasar el raton
        cells.append(html.Th(t(short_key), **kwargs))
    return html.Thead(html.Tr(cells))


def _row(rec: dict, total: int) -> html.Tr:
    """Una fila de la tabla a partir de un registro de la clasificacion."""
    pos = int(rec["Pos"])
    zone_class = _ZONE_CLASS[_zone_of(pos, total)]
    return html.Tr(
        [
            html.Td(pos, className=f"ls-pos text-center {zone_class}"),
            html.Td(rec["Team"], className="ls-team text-start"),
            html.Td(rec["Played"], className="text-end"),
            html.Td(rec["Win"], className="text-end"),
            html.Td(rec["Draw"], className="text-end"),
            html.Td(rec["Lose"], className="text-end"),
            html.Td(rec["GF"], className="text-end"),
            html.Td(rec["GA"], className="text-end"),
            html.Td(rec["Points"], className="ls-pts text-end"),
        ]
    )


def _build_table(df, t) -> dbc.Table:
    """Construye la tabla dbc completa (cabecera + cuerpo) con estilo Darkly."""
    total = len(df)
    body = html.Tbody([_row(rec, total) for rec in df.to_dict("records")])
    return dbc.Table(
        [_header(t), body],
        hover=True,
        responsive=True,
        color="dark",                    # base oscura coherente con DARKLY
        className="live-standings-table align-middle",
    )


def _legend_item(dot_class: str, label: str) -> html.Span:
    return html.Span(
        [html.Span(className=f"ls-dot {dot_class}"), label],
        className="d-inline-flex align-items-center",
    )


# --------------------------------------------------------------------------- #
# Grafico de barras DIVERGENTES: diferencia de goles (GF - GC), color por zona
# --------------------------------------------------------------------------- #
def _gd_figure(df, t) -> go.Figure:
    """Barras horizontales divergentes con la diferencia de goles de cada equipo.

    Direccion: positivas -> derecha, negativas -> izquierda (sigue siendo divergente).
    Color: segun la ZONA del equipo en la clasificacion (misma paleta que la tabla).
    Fondo transparente para fundirse con el tema DARKLY.
    """
    total = len(df)
    # Diferencia de goles = goles a favor - goles en contra.
    data = df.copy()
    data["GD"] = data["GF"] - data["GA"]
    # Orden ascendente: el mejor GD queda ARRIBA (en barras 'h', el ultimo va arriba).
    data = data.sort_values("GD", ascending=True)

    gd = data["GD"].tolist()
    teams = data["Team"].tolist()
    # Color por zona (no por signo): usa la POSICION real de cada equipo.
    colors = [_ZONE_COLOR[_zone_of(int(p), total)] for p in data["Pos"]]
    labels = [f"{v:+d}" for v in gd]   # etiqueta con signo: +15 / -19

    fig = go.Figure(
        go.Bar(
            x=gd,
            y=teams,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=labels,
            textposition="outside",
            textfont={"color": "#E0E0E0", "size": 12},
            cliponaxis=False,                       # que el texto no se recorte en los bordes
            hovertemplate="%{y}: %{x:+d}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",              # fondo del lienzo transparente
        plot_bgcolor="rgba(0,0,0,0)",               # fondo del area de trazado transparente
        showlegend=False,
        height=max(380, 26 * len(df) + 90),         # alto proporcional al nº de equipos
        margin=dict(l=20, r=40, t=10, b=40),
        bargap=0.28,
        font={"color": "#E0E0E0"},
        xaxis=dict(
            title=t(f"{_P}.chart_axis"),
            zeroline=True, zerolinecolor="#5a5f66", zerolinewidth=1.5,
            gridcolor="rgba(255,255,255,0.06)",
            color="#adb5bd",
        ),
        yaxis=dict(color="#E0E0E0", automargin=True),  # automargin: deja sitio a los nombres
    )
    return fig


def _empty_fig() -> go.Figure:
    """Figura vacia y transparente (estado sin datos: no muestra recuadro blanco)."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False}, yaxis={"visible": False},
        height=10, margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# Layout (solo cascaron; el callback rellena tabla y grafico)
# --------------------------------------------------------------------------- #
def layout(t):
    return dbc.Container(
        [
            html.H1(t(f"{_P}.title")),
            html.P(t(f"{_P}.description"), className="lead"),

            # Barra de control: badge "en vivo" + temporada + actualizado + boton.
            html.Div(
                [
                    html.Div(
                        [
                            dbc.Badge(
                                [html.Span(className="ls-live-dot"), t(f"{_P}.live_badge")],
                                className="ls-live-badge",
                            ),
                            html.Span(t(f"{_P}.season", season=_SEASON_LABEL),
                                      className="ls-season"),
                            html.Span(id="live-standings-updated", className="ls-updated"),
                        ],
                        className="d-flex align-items-center gap-3 flex-wrap",
                    ),
                    dbc.Button(
                        [html.Span("\U0001F504 "), t(f"{_P}.refresh")],
                        id="live-standings-refresh",
                        color="primary",
                        size="sm",
                        className="ls-refresh",
                    ),
                ],
                className="ls-controls d-flex justify-content-between "
                          "align-items-center flex-wrap mb-3",
            ),

            # Tabla (se rellena por callback) con spinner de carga.
            dcc.Loading(
                html.Div(id="live-standings-table"),
                type="default",
                color="#56B4E9",
            ),

            # Leyenda de zonas (compartida: explica los colores de tabla y grafico).
            html.Div(
                [
                    _legend_item("ls-zone-ucl-dot", t(f"{_P}.legend.ucl")),
                    _legend_item("ls-zone-uel-dot", t(f"{_P}.legend.uel")),
                    _legend_item("ls-zone-rel-dot", t(f"{_P}.legend.rel")),
                ],
                className="ls-legend d-flex gap-3 flex-wrap mt-3",
            ),

            # ---- Grafico analitico: diferencia de goles divergente (color por zona) ----
            html.H4(t(f"{_P}.chart_title"), className="ls-chart-title"),
            dcc.Loading(
                dcc.Graph(id="live-standings-gd-chart",
                          config={"displayModeBar": False}),
                type="default",
                color="#56B4E9",
            ),
        ],
        fluid=True,
        className="live-standings-wrapper py-2",
    )


# --------------------------------------------------------------------------- #
# Callback: carga/refresca tabla y grafico
# --------------------------------------------------------------------------- #
@callback(
    Output("live-standings-table", "children"),
    Output("live-standings-updated", "children"),
    Output("live-standings-gd-chart", "figure"),
    Input("live-standings-refresh", "n_clicks"),
    Input("language-store", "data"),
)
def _load_standings(n_clicks, language):
    t = translator.scoped(language)
    # force=True solo cuando el disparador es el boton (ignora la cache).
    force = ctx.triggered_id == "live-standings-refresh"

    df, err = api_client.get_standings(force=force)

    if df is None:
        # Fallo controlado: mensaje claro, sin romper la pagina, grafico vacio.
        msg = t(f"{_P}.no_key") if err == "no_key" else t(f"{_P}.error", reason=err or "")
        return dbc.Alert(msg, color="warning", className="ls-alert mb-0"), "", _empty_fig()

    now = dt.datetime.now().strftime("%H:%M:%S")
    return _build_table(df, t), t(f"{_P}.updated", time=now), _gd_figure(df, t)
