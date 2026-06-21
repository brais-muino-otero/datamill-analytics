"""
app.py
======
Punto de entrada de la aplicacion Dash (multi-pagina / SPA).

Arquitectura:
- Routing EXPLICITO con dcc.Location + un callback que mapea pathname -> layout.
  (Elegido sobre Dash Pages a proposito: nos permite inyectar el traductor en
   cada pagina y re-renderizarlas al cambiar de idioma, ver mas abajo.)
- i18n: un dcc.Store guarda el idioma activo; los textos se re-renderizan por callback.
- Tema: DARKLY (Bootstrap) + Plotly con paleta colorblind-safe por defecto.

Flujo de datos (callbacks):
    language-selector.value            --(1)-->  language-store.data
    language-store.data                --(2)-->  sidebar (nav + subtitulo + label)
    url.pathname + language-store.data --(3)-->  page-content
"""
import flask
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from config import APP_TITLE, DEFAULT_LANGUAGE, Routes
from components.sidebar import build_sidebar, build_nav_links
from i18n.translator import translator
from utils.plotly_theme import apply_plotly_theme

# Paginas (cada modulo expone layout(t))
from pages import (
    football_analytics,
    supervised,
    unsupervised,
    knowledge_hub,
)

# --------------------------------------------------------------------------- #
# Inicializacion
# --------------------------------------------------------------------------- #
# Aplica el tema Plotly (colorblind-safe) globalmente, una sola vez.
apply_plotly_theme()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],   # Tema oscuro de Bootstrap
    title=APP_TITLE,                            # Titulo de la pestana del navegador
    suppress_callback_exceptions=True,          # Necesario en apps multi-pagina:
                                               # hay componentes que aun no existen
                                               # en el layout inicial.
)
server = app.server   # Objeto Flask subyacente (necesario para desplegar con gunicorn)

# --------------------------------------------------------------------------- #
# PWA (Progressive Web App): manifest + service worker
# --------------------------------------------------------------------------- #
# Inyecta en el <head> el enlace al manifest, los metadatos de instalacion y el
# registro del service worker, conservando TODOS los placeholders de Dash.
app.index_string = """<!DOCTYPE html>
<html lang="es">
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="manifest" href="/assets/manifest.json">
    <meta name="theme-color" content="#0d6efd">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="DataMill">
    <link rel="apple-touch-icon" href="/assets/icons/icon-192.png">
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
    <script>
      // Registro del Service Worker (scope '/', servido desde la raiz).
      if ("serviceWorker" in navigator) {
        window.addEventListener("load", function () {
          navigator.serviceWorker.register("/sw.js", { scope: "/" })
            .then(function (reg) { console.log("[PWA] SW registrado, scope:", reg.scope); })
            .catch(function (err) { console.error("[PWA] Fallo al registrar SW:", err); });
        });
      }
    </script>
  </body>
</html>"""

# El Service Worker DEBE servirse desde la raiz para controlar todo el sitio
# (scope '/'). Se guarda como sw.js junto a app.py y se expone aqui.
_BASE_DIR = Path(__file__).resolve().parent


@server.route("/sw.js")
def _serve_service_worker():
    response = flask.send_from_directory(_BASE_DIR, "sw.js")
    response.headers["Content-Type"] = "application/javascript"
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response

# --------------------------------------------------------------------------- #
# Layout raiz
# --------------------------------------------------------------------------- #
app.layout = html.Div(
    [
        # Router: lee/escribe la URL del navegador SIN recargar la pagina.
        dcc.Location(id="url"),

        # Estado global del idioma activo. storage_type='memory' = se reinicia al
        # recargar. Para persistir entre recargas: cambiar a 'local' (ver README).
        dcc.Store(id="language-store", data=DEFAULT_LANGUAGE),

        # Menu lateral fijo (cascaron estatico; sus textos los rellena un callback).
        build_sidebar(),

        # Area de contenido: aqui se inyecta la pagina activa.
        html.Div(id="page-content", className="content"),
    ]
)

# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

# (1) El selector escribe el idioma elegido en el Store (unica fuente de verdad).
#     prevent_initial_call=True -> no se dispara en la carga inicial (ya tenemos
#     el valor por defecto en el Store), solo cuando el usuario cambia de idioma.
@app.callback(
    Output("language-store", "data"),
    Input("language-selector", "value"),
    prevent_initial_call=True,
)
def update_language(selected_language: str) -> str:
    return selected_language


# (2) Al cambiar el idioma, re-renderiza los textos traducibles del sidebar.
@app.callback(
    Output("sidebar-nav", "children"),
    Output("sidebar-subtitle", "children"),
    Output("language-label", "children"),
    Input("language-store", "data"),
)
def translate_sidebar(language: str):
    t = translator.scoped(language)            # funcion de traduccion fijada al idioma
    return build_nav_links(t), t("app.subtitle"), t("language.label")


# (3) Router: segun la URL (y el idioma), renderiza la pagina correspondiente.
# Mapa ruta -> funcion layout de cada pagina.
ROUTES_MAP = {
    Routes.FOOTBALL: football_analytics.layout,
    Routes.SUPERVISED: supervised.layout,
    Routes.UNSUPERVISED: unsupervised.layout,
    Routes.KNOWLEDGE_HUB: knowledge_hub.layout,
}


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("language-store", "data"),
)
def render_page_content(pathname: str, language: str):
    t = translator.scoped(language)
    page_layout = ROUTES_MAP.get(pathname)

    # 404: ruta desconocida.
    if page_layout is None:
        return dbc.Container(
            [
                html.H1("404", className="display-1"),
                html.P(t("common.not_found", path=pathname)),
                dbc.Button(t("common.go_home"), href=Routes.FOOTBALL, color="primary"),
            ],
            fluid=True,
            className="py-5",
        )

    return page_layout(t)


# --------------------------------------------------------------------------- #
# Arranque (solo en ejecucion local; en produccion usar gunicorn sobre `server`)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(debug=True)
