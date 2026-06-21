"""
pages/unsupervised_learning.py
=============================
Pagina: Aprendizaje No Supervisado (ruta "/aprendizaje-no-supervisado").

Clustering y reduccion de dimensionalidad de equipos y jugadores.
"""
import dash_bootstrap_components as dbc
from dash import html


def layout(t):
    """Layout de la pagina en el idioma activo (`t` = traductor scoped)."""
    return dbc.Container(
        [
            html.H1(t("pages.unsupervised.title")),
            html.P(t("pages.unsupervised.description"), className="lead"),
            html.Hr(),
            # Placeholder mientras desarrollamos el contenido real.
            dbc.Alert(t("common.under_construction"), color="info"),
            # ------------------------------------------------------------- #
            # TODO (proximas iteraciones):
            #   - PCA (reduccion de dimensionalidad)
            #   - K-means / clustering de estilos de juego
            #   - Visualizaciones 2D de los clusters
            # ------------------------------------------------------------- #
        ],
        fluid=True,
    )
