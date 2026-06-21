"""
pages/supervised.py
===================
Pagina "Aprendizaje Supervisado" (ruta /aprendizaje-supervisado).

Predice el resultado del partido (1/X/2 = H/D/A) replicando la metodologia de la
Practica 2 de AA1. TODA la matematica/ML vive en utils/ml_models; aqui solo se
construye la UI, se dibujan resultados y se redacta el data storytelling.

Estructura (segun el enunciado):
- Controles RESTRINGIDOS: dropdown maestro de algoritmo + control secundario que
  ofrece EXACTAMENTE el espacio de busqueda cerrado de cada modelo (para SVM, un
  unico selector con las 8 combinaciones literales, evitando cruces invalidos).
- Panel de resultados: tarjetas KPI (accuracy/error/F1) + matriz de confusion
  (heatmap CVD-safe, filas = clase real, columnas = predicha) + storytelling.
- Panel de robustez: Shapiro-Wilk del modelo actual + comparacion por pares
  (Wilcoxon) contra otro algoritmo en su mejor configuracion + storytelling
  (Navaja de Ockham / sobreajuste).

Los callbacks usan el registro global (@callback) y leen el idioma del Store, igual
que el resto de paginas. Las evaluaciones se cachean en utils/ml_models, asi que
reentrenar/compa­rar configuraciones repetidas es instantaneo.
"""
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, clientside_callback, dcc, html

from config import DEFAULT_LANGUAGE
from i18n.translator import translator
from utils import ml_models as ml

_P = "pages.supervised"

# Orden de algoritmos en el selector maestro.
_ALGOS = ["baseline", "ann", "svm", "tree", "knn", "dome"]


# --------------------------------------------------------------------------- #
# Helpers de presentacion de configuraciones (UI <-> objeto de config del motor)
# --------------------------------------------------------------------------- #
def _topo_label(topo) -> str:
    """Topologia de ANN como '[20]' o '[10, 5]'."""
    return "[" + ", ".join(str(n) for n in topo) + "]"


def _svm_label(kernel: str, C, t) -> str:
    """Etiqueta legible de una config SVM: 'RBF, C=1', 'Lineal, C=0.1'..."""
    name = {
        "linear": t(f"{_P}.kernel_linear"),
        "poly": t(f"{_P}.kernel_poly"),
        "rbf": t(f"{_P}.kernel_rbf"),
    }[kernel]
    return f"{name}, C={C}"


def _config_human(algo: str, config, t) -> str:
    """Texto humano de una configuracion concreta (para titulos y storytelling)."""
    if algo == "baseline":
        return t(f"{_P}.algo.baseline")
    if algo == "ann":
        return _topo_label(config)
    if algo == "svm":
        return _svm_label(config[0], config[1], t)
    if algo == "tree":
        return t(f"{_P}.opt_depth", n=config)
    if algo == "knn":
        return t(f"{_P}.opt_k", n=config)
    if algo == "dome":
        return t(f"{_P}.opt_nodes", n=config)
    return str(config)


def _model_label(algo: str, config, t) -> str:
    """Nombre completo 'Algoritmo (config)' usado en tarjetas y narracion."""
    if algo == "baseline":
        return t(f"{_P}.algo.baseline")
    return f"{t(f'{_P}.algo.{algo}')} ({_config_human(algo, config, t)})"


def _config_options(algo: str, t):
    """
    Devuelve (options, value, disabled) del control secundario para `algo`,
    forzando EXACTAMENTE el espacio cerrado del motor. Valores codificados como
    string para dcc.Dropdown; `_parse_config` los reconvierte.
    """
    if algo == "baseline":
        return [], None, True
    if algo == "ann":
        opts = [{"label": _topo_label(tp), "value": ",".join(map(str, tp))}
                for tp in ml.ANN_TOPOLOGIES]
        return opts, "20", False                      # por defecto [20]
    if algo == "svm":
        opts = [{"label": _svm_label(k, C, t), "value": f"{k}|{C}"}
                for (k, C) in ml.SVM_CONFIGS]
        return opts, "rbf|1", False                   # por defecto RBF, C=1
    if algo == "tree":
        opts = [{"label": t(f"{_P}.opt_depth", n=d), "value": str(d)}
                for d in ml.TREE_DEPTHS]
        return opts, "3", False
    if algo == "knn":
        opts = [{"label": t(f"{_P}.opt_k", n=k), "value": str(k)}
                for k in ml.KNN_NEIGHBORS]
        return opts, "9", False
    if algo == "dome":
        opts = [{"label": t(f"{_P}.opt_nodes", n=n), "value": str(n)}
                for n in ml.DOME_NODES]
        return opts, "6", False
    return [], None, True


def _parse_config(algo: str, value):
    """Reconvierte el value del control secundario al objeto de config del motor."""
    if algo == "baseline" or value is None:
        return None
    if algo == "ann":
        return tuple(int(x) for x in value.split(","))
    if algo == "svm":
        kernel, c = value.split("|")
        return (kernel, float(c) if "." in c else int(c))
    return int(value)   # tree / knn / dome


def _fmt_p(p: float) -> str:
    """p-valor legible (decimal normal; cientifico si es muy pequeno)."""
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


# --------------------------------------------------------------------------- #
# Tarjetas
# --------------------------------------------------------------------------- #
def _kpi_card(label_key: str, value_id: str, sub_id: str, t) -> dbc.Card:
    """Tarjeta KPI: etiqueta fija + valor grande + subtexto (±std), via callback."""
    return dbc.Card(
        dbc.CardBody([
            html.Div(t(label_key), className="stat-label"),
            html.Div(id=value_id, className="stat-value"),
            html.Div(id=sub_id, className="stat-sub"),
        ]),
        className="stat-card",
    )


def _test_card(label_id: str, value_id: str, verdict_id: str) -> dbc.Card:
    """Tarjeta de test estadistico: etiqueta + p-valor + badge de veredicto."""
    return dbc.Card(
        dbc.CardBody([
            html.Div(id=label_id, className="stat-label"),
            html.Div(id=value_id, className="stat-value"),
            html.Div(id=verdict_id, className="stat-sub"),
        ]),
        className="stat-card",
    )


# --------------------------------------------------------------------------- #
# Matriz de confusion (heatmap CVD-safe)
# --------------------------------------------------------------------------- #
def _confusion_figure(cm, t) -> go.Figure:
    classes = [t(f"{_P}.classes.home"), t(f"{_P}.classes.draw"), t(f"{_P}.classes.away")]
    z = np.array(cm, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=classes, y=classes,
        # Azul oscuro CONSTANTE en todas las celdas: los numeros en blanco siempre se leen.
        colorscale=[[0.0, "#0d3b66"], [1.0, "#0d3b66"]],
        showscale=False,                        # la barra de color ya no aporta (celdas uniformes)
        xgap=2, ygap=2,                          # separacion fina entre celdas
        texttemplate="%{z:.0f}", textfont={"size": 15, "color": "white"},
        hovertemplate=(t(f"{_P}.axis_real") + ": %{y}<br>" +
                       t(f"{_P}.axis_pred") + ": %{x}<br>n = %{z:.0f}<extra></extra>"),
    ))
    fig.update_layout(
        title=t(f"{_P}.confusion_title"),
        xaxis_title=t(f"{_P}.axis_pred"),
        yaxis_title=t(f"{_P}.axis_real"),
        yaxis={"autorange": "reversed"},        # primera clase (1/Local) arriba
        height=420,
        margin=dict(l=110, r=30, t=60, b=60),
    )
    return fig


# --------------------------------------------------------------------------- #
# Construccion del storytelling
# --------------------------------------------------------------------------- #
def _dominant_error(cm):
    """(real_idx, pred_idx, n) del mayor error fuera de la diagonal."""
    best = (0, 1, -1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > best[2]:
                best = (i, j, cm[i][j])
    return best


def _worst_recall_class(cm):
    """Indice de la clase con peor recall (la mas dificil de acertar)."""
    recalls = []
    for i in range(len(cm)):
        total = sum(cm[i])
        recalls.append(cm[i][i] / total if total else 0.0)
    return int(np.argmin(recalls))


def _results_story(algo, config, r, t) -> str:
    """Narracion del panel de resultados: intro + sobreajuste + lectura de la matriz."""
    classes = [t(f"{_P}.classes.home"), t(f"{_P}.classes.draw"), t(f"{_P}.classes.away")]
    parts = [t(
        f"{_P}.story_intro",
        model=_model_label(algo, config, t),
        accuracy=f"{r['acc_mean'] * 100:.2f}",
        error=f"{r['error_mean'] * 100:.2f}",
        f1=f"{r['f1_mean']:.3f}",
        baseline=f"{ml.majority_baseline() * 100:.1f}",
    )]

    # Senal de sobreajuste (solo arboles/DoME/ANN, eje de complejidad creciente).
    sig = ml.overfitting_signal(algo)
    if sig is not None:
        param_word = {
            "max_depth": t(f"{_P}.param_depth"),
            "max_nodes": t(f"{_P}.param_nodes"),
            "topology": t(f"{_P}.param_topology"),
        }[sig["param"]]
        parts.append(t(
            f"{_P}.story_overfit",
            param=param_word,
            best=_config_human(algo, sig["best_config"], t),
            best_acc=f"{sig['best_acc'] * 100:.2f}",
            complex=_config_human(algo, sig["complex_config"], t),
            complex_acc=f"{sig['complex_acc'] * 100:.2f}",
        ))

    # Lectura de la matriz de confusion.
    ri, pj, n = _dominant_error(r["confusion"])
    worst = _worst_recall_class(r["confusion"])
    parts.append(t(
        f"{_P}.story_confusion",
        n_err=n, real_cls=classes[ri], pred_cls=classes[pj], worst_cls=classes[worst],
    ))
    return "\n\n".join(parts)


# =========================================================================== #
# Layout
# =========================================================================== #
def layout(t) -> dbc.Container:
    algo_options = [{"label": t(f"{_P}.algo.{a}"), "value": a} for a in _ALGOS]

    controls = dbc.Card(dbc.CardBody([
        html.Label(t(f"{_P}.algo_label"), htmlFor="sup-algo", className="control-label"),
        dcc.Dropdown(id="sup-algo", options=algo_options, value="svm",
                     clearable=False, className="control-dropdown mb-3"),
        html.Label(t(f"{_P}.config_label"), htmlFor="sup-config", className="control-label"),
        html.Div(dcc.Dropdown(id="sup-config", clearable=False,
                              className="control-dropdown mb-2"),
                 id="sup-config-container"),
        html.Div(t(f"{_P}.baseline_hint"), id="sup-config-hint",
                 className="config-hint mb-3"),
        dbc.Button(t(f"{_P}.train_button"), id="sup-train", color="primary",
                   n_clicks=0, className="mc-run-btn w-100"),
    ]), className="controls-card")

    results = dcc.Loading(html.Div([
        dbc.Row([
            dbc.Col(_kpi_card(f"{_P}.kpi_accuracy", "sup-acc", "sup-acc-sub", t), md=4, xs=12),
            dbc.Col(_kpi_card(f"{_P}.kpi_error", "sup-err", "sup-err-sub", t), md=4, xs=12),
            dbc.Col(_kpi_card(f"{_P}.kpi_f1", "sup-f1", "sup-f1-sub", t), md=4, xs=12),
        ], className="g-2 mb-3"),
        dcc.Graph(id="sup-confusion", config={"displayModeBar": False}),
    ]), type="default")

    robustness = dbc.Container([
        html.Hr(),
        html.H4(t(f"{_P}.robustness_title"), className="mt-2"),
        dcc.Markdown(t(f"{_P}.robustness_intro"), className="theory-block"),
        dbc.Row([
            dbc.Col(_test_card("sup-shapiro-label", "sup-shapiro-p", "sup-shapiro-verdict"),
                    md=4, xs=12),
            dbc.Col([
                html.Label(t(f"{_P}.compare_label"), htmlFor="sup-compare",
                           className="control-label"),
                dcc.Dropdown(id="sup-compare", clearable=True,
                             placeholder=t(f"{_P}.compare_placeholder"),
                             className="control-dropdown"),
            ], md=4, xs=12, className="d-flex flex-column justify-content-center"),
            dbc.Col(dcc.Loading(
                _test_card("sup-wilcoxon-label", "sup-wilcoxon-p", "sup-wilcoxon-verdict"),
                type="default"), md=4, xs=12),
        ], className="g-2 mb-2"),
    ], fluid=True)

    return dbc.Container([
        html.H1(t(f"{_P}.title")),
        html.P(t(f"{_P}.description"), className="lead"),
        html.Div(
            dbc.Button(
                [html.Span("\U0001F4C4", className="me-2"), t(f"{_P}.export_report")],
                id="sup-export-btn", color="light", outline=True, size="sm",
                className="export-btn"),
            className="d-flex justify-content-end mb-2 no-print",
        ),
        html.Hr(),
        dcc.Markdown(t(f"{_P}.theory"), className="theory-block"),
        dbc.Row([
            dbc.Col(controls, lg=4, md=12),
            dbc.Col(results, lg=8, md=12),
        ], className="g-3"),
        dcc.Markdown(id="sup-story-results", className="story-block"),
        robustness,
        dcc.Markdown(id="sup-story-compare", className="story-block"),
        # Estado del modelo actual (para el test de Wilcoxon emparejado).
        dcc.Store(id="sup-current-store"),
        # Anclaje invisible para el callback de impresion (exportar a PDF).
        dcc.Store(id="sup-print-anchor"),
    ], fluid=True)


# =========================================================================== #
# Callback 1: el algoritmo elegido reconfigura el control secundario y los rivales
# =========================================================================== #
@callback(
    Output("sup-config", "options"),
    Output("sup-config", "value"),
    Output("sup-config", "disabled"),
    Output("sup-config-hint", "style"),
    Output("sup-compare", "options"),
    Output("sup-compare", "value"),
    Input("sup-algo", "value"),
    State("language-store", "data"),
)
def _on_algo_change(algo, lang):
    t = translator.scoped(lang or DEFAULT_LANGUAGE)
    algo = algo or "svm"
    options, value, disabled = _config_options(algo, t)
    # El "hint" de baseline solo se muestra cuando no hay hiperparametros.
    hint_style = {} if algo == "baseline" else {"display": "none"}
    # Rivales del test: la baseline y los demas algoritmos (excluyendo el actual).
    compare_opts = [{"label": t(f"{_P}.algo.{a}"), "value": a}
                    for a in _ALGOS if a != algo]
    return options, value, disabled, hint_style, compare_opts, None


# =========================================================================== #
# Callback 2: "Entrenar Modelo" -> CV, KPIs, matriz, Shapiro y storytelling
# =========================================================================== #
@callback(
    Output("sup-acc", "children"),
    Output("sup-acc-sub", "children"),
    Output("sup-err", "children"),
    Output("sup-err-sub", "children"),
    Output("sup-f1", "children"),
    Output("sup-f1-sub", "children"),
    Output("sup-confusion", "figure"),
    Output("sup-shapiro-label", "children"),
    Output("sup-shapiro-p", "children"),
    Output("sup-shapiro-verdict", "children"),
    Output("sup-story-results", "children"),
    Output("sup-current-store", "data"),
    Output("sup-wilcoxon-label", "children"),
    Output("sup-wilcoxon-p", "children"),
    Output("sup-wilcoxon-verdict", "children"),
    Output("sup-story-compare", "children"),
    Input("sup-train", "n_clicks"),
    State("sup-algo", "value"),
    State("sup-config", "value"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def _train(n_clicks, algo, config_value, lang):
    t = translator.scoped(lang or DEFAULT_LANGUAGE)
    algo = algo or "svm"
    config = _parse_config(algo, config_value)

    r = ml.evaluate(algo, config)

    # KPIs
    acc = f"{r['acc_mean'] * 100:.2f} %"
    acc_sub = f"± {r['acc_std'] * 100:.2f}"
    err = f"{r['error_mean'] * 100:.2f} %"
    err_sub = f"± {r['error_std'] * 100:.2f}"
    f1 = f"{r['f1_mean']:.3f}"
    f1_sub = f"± {r['f1_std']:.3f}"

    fig = _confusion_figure(r["confusion"], t)

    # Shapiro-Wilk del modelo actual
    sh = ml.shapiro_test(r["accuracies"])
    shapiro_label = t(f"{_P}.shapiro_label")
    shapiro_p = f"p = {_fmt_p(sh['p_value'])}"
    verdict_txt = t(f"{_P}.verdict_normal") if sh["normal"] else t(f"{_P}.verdict_not_normal")
    shapiro_verdict = dbc.Badge(verdict_txt, color="info" if sh["normal"] else "warning",
                                className="verdict-badge")

    story = _results_story(algo, config, r, t)

    store = {"algo": algo, "config": config, "accuracies": r["accuracies"],
             "acc_mean": r["acc_mean"], "label": _model_label(algo, config, t)}

    # Al entrenar de nuevo, reseteamos el panel de Wilcoxon (evita resultados obsoletos).
    wil_reset_label = t(f"{_P}.wilcoxon_label_idle")
    wil_reset_p = "—"
    wil_reset_verdict = dbc.Badge(t(f"{_P}.compare_prompt"), color="secondary",
                                  className="verdict-badge")

    return (acc, acc_sub, err, err_sub, f1, f1_sub, fig,
            shapiro_label, shapiro_p, shapiro_verdict, story, store,
            wil_reset_label, wil_reset_p, wil_reset_verdict, "")


# =========================================================================== #
# Callback 3: elegir rival -> Wilcoxon emparejado + storytelling (Ockham/sig.)
# =========================================================================== #
@callback(
    Output("sup-wilcoxon-label", "children", allow_duplicate=True),
    Output("sup-wilcoxon-p", "children", allow_duplicate=True),
    Output("sup-wilcoxon-verdict", "children", allow_duplicate=True),
    Output("sup-story-compare", "children", allow_duplicate=True),
    Input("sup-compare", "value"),
    State("sup-current-store", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def _compare(other_algo, current, lang):
    t = translator.scoped(lang or DEFAULT_LANGUAGE)

    # Sin modelo entrenado o sin rival: pedimos accion al usuario.
    if not current:
        return (t(f"{_P}.wilcoxon_label_idle"), "—",
                dbc.Badge(t(f"{_P}.train_first"), color="secondary",
                          className="verdict-badge"), "")
    if not other_algo:
        return (t(f"{_P}.wilcoxon_label_idle"), "—",
                dbc.Badge(t(f"{_P}.compare_prompt"), color="secondary",
                          className="verdict-badge"), "")

    # Rival en su MEJOR configuracion (grid cerrado, cacheado).
    best = ml.grid_search(other_algo)
    other_eval = ml.best_evaluation(other_algo)
    other_label = _model_label(other_algo, best["best_config"], t)

    w = ml.wilcoxon_test(current["accuracies"], other_eval["accuracies"])
    p_disp = f"p = {_fmt_p(w['p_value'])}"
    wil_label = t(f"{_P}.wilcoxon_label", rival=other_label)

    if w["significant"]:
        verdict = dbc.Badge(t(f"{_P}.verdict_significant"), color="warning",
                            className="verdict-badge")
        better, worse = ((current["label"], other_label)
                         if current["acc_mean"] >= other_eval["acc_mean"]
                         else (other_label, current["label"]))
        story = t(f"{_P}.story_significant", better=better, worse=worse, p=_fmt_p(w["p_value"]))
    else:
        verdict = dbc.Badge(t(f"{_P}.verdict_equivalent"), color="info",
                            className="verdict-badge")
        cur_a, oth_a = current["algo"], other_algo
        mixed = ((cur_a in ml.SIMPLE_ALGOS and oth_a in ml.COMPLEX_ALGOS) or
                 (cur_a in ml.COMPLEX_ALGOS and oth_a in ml.SIMPLE_ALGOS))
        if mixed:
            simple_lbl = current["label"] if cur_a in ml.SIMPLE_ALGOS else other_label
            complex_lbl = current["label"] if cur_a in ml.COMPLEX_ALGOS else other_label
            story = t(f"{_P}.story_ockham", simple=simple_lbl, complex=complex_lbl,
                      p=_fmt_p(w["p_value"]))
        else:
            story = t(f"{_P}.story_equivalent", model_a=current["label"],
                      model_b=other_label, p=_fmt_p(w["p_value"]))

    return wil_label, p_disp, verdict, story


# =========================================================================== #
# Callback (clientside): "Exportar Informe" -> dialogo de impresion del navegador
# =========================================================================== #
# Solucion nativa, sin dependencias pesadas (weasyprint/pdfkit): window.print()
# abre el dialogo del navegador, que permite "Guardar como PDF". El CSS @media
# print de assets/custom.css limpia la salida (oculta sidebar, fondos blancos...).
clientside_callback(
    "function (n) { if (n) { window.print(); } return window.dash_clientside.no_update; }",
    Output("sup-print-anchor", "data"),
    Input("sup-export-btn", "n_clicks"),
    prevent_initial_call=True,
)
