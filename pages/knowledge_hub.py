"""
pages/knowledge_hub.py
=====================
Pagina: Knowledge Hub (ruta "/knowledge-hub").

Whitepaper tecnico del motor analitico "DataMill Analytics". Documenta el producto,
las dos arquitecturas de ML (supervisada y no supervisada), el stack tecnologico y
el equipo. Pagina 100% de presentacion: sin callbacks ni computo; todo el texto
proviene de i18n (dcc.Markdown) y el stack se luce con dbc.Badge.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

_P = "pages.knowledge_hub"


# --------------------------------------------------------------------------- #
# Helpers de presentacion
# --------------------------------------------------------------------------- #
def _badge(text: str, color: str) -> dbc.Badge:
    return dbc.Badge(text, color=color, pill=True, className="me-2 mb-2 px-3 py-2")


def _stack_group(label: str, items: list[tuple[str, str]]) -> html.Div:
    """Grupo de badges con su etiqueta de categoria. items = [(texto, color), ...]."""
    return html.Div(
        [
            html.Div(label, className="fw-bold mb-2"),
            html.Div([_badge(text, color) for text, color in items], className="mb-1"),
        ],
        className="mb-3",
    )


def _section(title: str, body_md: str, item_id: str, extra=None) -> dbc.AccordionItem:
    """Item de acordeon: titulo + cuerpo Markdown (+ contenido extra opcional)."""
    children = [dcc.Markdown(body_md, className="kh-markdown", link_target="_blank")]
    if extra is not None:
        children.extend(extra)
    return dbc.AccordionItem(children, title=title, item_id=item_id)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def layout(t) -> dbc.Container:
    # --- Stack tecnologico (nombres propios literales; etiquetas traducibles) ---
    stack = html.Div([
        _stack_group(t(f"{_P}.stack_languages"), [("Python 3.12", "primary")]),
        _stack_group(t(f"{_P}.stack_frontend"), [
            ("Dash", "info"), ("Plotly", "info"), ("dash-bootstrap-components", "info"),
        ]),
        _stack_group(t(f"{_P}.stack_ml"), [
            ("scikit-learn", "success"), ("SciPy", "success"), ("MiniSom", "success"),
            ("NumPy", "success"), ("pandas", "success"),
        ]),
        _stack_group(t(f"{_P}.stack_i18n"), [
            ("i18n custom \u00b7 6 idiomas", "warning"), ("CVD-safe", "warning"),
        ]),
    ])

    # --- Cabecera tipo portada de whitepaper ---
    hero = html.Div([
        html.H1(t(f"{_P}.title"), className="display-5 mb-1"),
        html.P(t(f"{_P}.subtitle"), className="lead text-muted mb-3"),
        html.Div([
            _badge("Premier League 2022-23", "secondary"),
            _badge("Machine Learning", "secondary"),
            _badge("Dash \u00b7 Plotly", "secondary"),
            _badge("v1.0", "secondary"),
        ]),
    ], className="mb-4")

    accordion = dbc.Accordion(
        [
            _section(t(f"{_P}.sec1_title"), t(f"{_P}.sec1_body"), "sec-engine"),
            _section(t(f"{_P}.sec2_title"), t(f"{_P}.sec2_body"), "sec-supervised"),
            _section(t(f"{_P}.sec3_title"), t(f"{_P}.sec3_body"), "sec-unsupervised"),
            _section(t(f"{_P}.sec4_title"), t(f"{_P}.sec4_body"), "sec-stack",
                     extra=[stack]),
            _section(t(f"{_P}.sec5_title"), t(f"{_P}.sec5_body"), "sec-team"),
        ],
        active_item=["sec-engine"],     # primera seccion abierta por defecto
        always_open=True,               # el lector puede expandir varias a la vez
        flush=False,
    )

    return dbc.Container([
        hero,
        dcc.Markdown(t(f"{_P}.lead"), className="kh-markdown mb-4"),
        accordion,
    ], fluid=True)
