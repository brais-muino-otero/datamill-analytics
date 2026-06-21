"""
pages/supervised_learning.py
===========================
Pagina: Aprendizaje Supervisado (ruta "/aprendizaje-supervisado").

Modelos predictivos de resultados de partidos.
"""
import dash_bootstrap_components as dbc
from dash import html


def layout(t):
    """Layout de la pagina en el idioma activo (`t` = traductor scoped)."""
    return dbc.Container(
        [
            html.H1(t("pages.supervised.title")),
            html.P(t("pages.supervised.description"), className="lead"),
            html.Hr(),
            # Placeholder mientras desarrollamos el contenido real.
            dbc.Alert(t("common.under_construction"), color="info"),
            # ------------------------------------------------------------- #
            # TODO (proximas iteraciones):
            #   - Dixon-Coles, Poisson, regresion logistica
            #   - LDA / KNN para clasificacion de resultados
            #   - Metricas de evaluacion y backtesting
            # ------------------------------------------------------------- #
        ],
        fluid=True,
    )
