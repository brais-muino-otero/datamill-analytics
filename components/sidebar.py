"""
components/sidebar.py
=====================
Menu lateral (sidebar) fijo: marca, navegacion y selector de idioma.

Separacion de responsabilidades:
- build_sidebar()     -> "cascaron" ESTATICO que se monta una sola vez.
                         Contiene contenedores con id que se rellenan por callback.
- build_nav_links(t)  -> genera los NavLink TRADUCIDOS. Lo llama un callback de
                         app.py cada vez que cambia el idioma.

Por que el selector de idioma es estatico y la navegacion dinamica:
los nombres de idioma (endonimos) no se traducen, pero las etiquetas de
navegacion si. Asi solo re-renderizamos lo que realmente cambia.
"""
import dash_bootstrap_components as dbc
from dash import html

from config import APP_TITLE, DEFAULT_LANGUAGE, LANGUAGES, Routes


def build_nav_links(t) -> list:
    """
    Construye la lista de enlaces de navegacion con textos traducidos.
    `t` es la funcion de traduccion "scoped" (ver Translator.scoped).

    active='exact' hace que dbc resalte automaticamente el enlace cuya `href`
    coincide EXACTAMENTE con la URL actual (via dcc.Location). Cero callbacks
    extra para gestionar el estado activo.
    """
    return [
        dbc.NavLink(t("nav.football_analytics"), href=Routes.FOOTBALL, active="exact"),
        dbc.NavLink(t("nav.supervised"), href=Routes.SUPERVISED, active="exact"),
        dbc.NavLink(t("nav.unsupervised"), href=Routes.UNSUPERVISED, active="exact"),
        dbc.NavLink(t("nav.knowledge_hub"), href=Routes.KNOWLEDGE_HUB, active="exact"),
    ]


def build_sidebar() -> html.Div:
    """
    Cascaron estatico del sidebar. Se monta una sola vez en el layout raiz.
    Los textos traducibles viven en contenedores vacios (id=...) que un callback
    rellena segun el idioma activo.
    """
    # Opciones del selector: {label: endonimo, value: codigo ISO}
    language_options = [{"label": name, "value": code} for code, name in LANGUAGES.items()]

    return html.Div(
        [
            # --- Marca (nombre propio, no se traduce) ---
            html.H2(APP_TITLE, className="sidebar-brand"),
            # Subtitulo traducible -> rellenado por callback
            html.Div(id="sidebar-subtitle", className="sidebar-subtitle"),

            html.Hr(),

            # --- Navegacion (NavLinks traducidos) -> rellenado por callback ---
            dbc.Nav(id="sidebar-nav", vertical=True, pills=True),

            html.Hr(),

            # --- Selector de idioma (estatico) ---
            html.Div(
                [
                    # Etiqueta traducible ("Idioma" / "Language" / ...) -> callback.
                    # htmlFor enlaza label<->select (accesibilidad / lectores de pantalla).
                    html.Label(
                        id="language-label",
                        htmlFor="language-selector",
                        className="language-label",
                    ),
                    dbc.Select(
                        id="language-selector",
                        options=language_options,
                        value=DEFAULT_LANGUAGE,   # idioma inicial
                    ),
                ],
                className="language-selector-wrapper",
            ),
        ],
        className="sidebar",
    )
