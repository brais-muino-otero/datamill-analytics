# DataMill Analytics

> Premier League 2022-23 analytics & machine learning web app, built with **Dash** and **Plotly**.

A portfolio-grade data science project: a leakage-safe supervised match predictor, an unsupervised team-profiling engine, classical statistical inference, and a Monte Carlo season simulator — all wrapped in a multi-page, 6-language, accessible (CVD-safe) interface.

**Live demo:** [huggingface.co/spaces/Braizito/datamill-analytics](https://huggingface.co/spaces/Braizito/datamill-analytics)

---

## What this project demonstrates

This is a closed, fully reproducible dataset (no live API calls), by design: the goal is to showcase rigorous ML/statistics methodology with results that are identical on every run, not to operate a live data pipeline. (See [`docs/architecture-notes.md`](#architecture-notes) below for the reasoning.)

- **Anti-leakage feature engineering** for time-series sports data.
- **Stratified 10-fold cross-validation** with per-fold Z-score standardization (no leakage across folds).
- **Statistical rigor**: Shapiro-Wilk normality tests, Wilcoxon signed-rank tests for paired model comparison, Ockham's razor applied explicitly when models are statistically equivalent.
- **Unsupervised team profiling**: PCA + three clustering families (K-Means, hierarchical, Self-Organizing Maps), evaluated with the silhouette score — including honest handling of SOM over-segmentation (reported as undefined, never faked).
- **Classical inference & simulation**: Welch's t-test for the home-advantage hypothesis, Dixon-Coles model + Monte Carlo season simulation for title/relegation probabilities.
- **Production concerns**: Dockerized deployment, PWA support (offline shell, installable), a from-scratch i18n system (no external dependency) covering 6 languages, and a CVD-safe (colorblind-accessible) Plotly theme applied app-wide.

## Pages

| Page | What it does |
|---|---|
| **Football Analytics** | Correlation heatmaps (whole league vs. Top 6), Welch's t-test on the home-advantage hypothesis, Dixon-Coles Monte Carlo season simulator (title / top-4 / relegation probabilities). |
| **Supervised Learning** | Predicts each match outcome (Home/Draw/Away) using only pre-match information — 5-match rolling averages per team, never in-match stats. Six model families (logistic regression baseline, ANN, SVM, decision tree, k-NN, a bounded random-forest proxy for DoME), each tuned over a closed, predefined hyperparameter grid. Confusion matrix, KPI cards, and pairwise Wilcoxon model comparison. |
| **Unsupervised Learning** | Summarizes each team's season into 5 advanced KPIs (attack volume, efficiency, defensive solidity, game control, aggressiveness), then runs PCA + K-Means / hierarchical / SOM clustering to discover playing-style profiles. |
| **Knowledge Hub** | Technical whitepaper: methodology, architecture, and engineering decisions, written for a non-trivial reader. |

## Tech stack

- **Python**, **Dash** + **Dash Bootstrap Components** (DARKLY theme), **Plotly** (custom CVD-safe theme: Okabe-Ito palette, transparent backgrounds).
- **scikit-learn**, **SciPy**, **MiniSom** — the ML/stats engine.
- **pandas** / **NumPy** for the data pipeline.
- Custom **i18n** module (dictionary-based, hierarchical keys, cascading fallback) — no external translation library.
- **Docker** (gunicorn, non-root user, healthcheck) for deployment; **PWA** (service worker, offline shell, manifest) for installability.
- **pytest** test suite covering the ML engine's statistical guarantees (Z-score correctness, PCA component count, leakage-free rolling features, clustering validity, SOM over-segmentation handling).

## Project structure

```
premier-league-analytics/
├── app.py                  # Dash app entrypoint, routing, sidebar
├── config.py                # Routes, languages, model/CV constants
├── pages/                    # One module per page (each exposes layout(t))
│   ├── football_analytics.py
│   ├── supervised.py
│   └── unsupervised.py
│   └── knowledge_hub.py
├── utils/
│   ├── data_loader.py        # Single static-dataset ingestion point (CSV/Parquet)
│   ├── ml_models.py           # Supervised engine: feature build, CV, grid search
│   ├── unsupervised_engine.py # KPI build, PCA, clustering
│   ├── stats_engine.py        # Dixon-Coles, Welch's t-test, Monte Carlo
│   └── plotly_theme.py        # CVD-safe Plotly theme, applied once at startup
├── i18n/
│   ├── translator.py          # Dependency-free translation engine
│   └── locales/*.json         # es, en, pt, it, fr, de — 171 parity-checked keys each
├── components/sidebar.py
├── assets/                    # custom.css, manifest.json, icons
├── data/epl_results_2022-23.csv
├── tests/test_ml_engine.py
├── Dockerfile
└── sw.js                       # Service worker (PWA)
```

## Running locally

```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:8050

## Running the test suite

```bash
pip install pytest
pytest tests/ -v
```

## Running with Docker

```bash
docker build -t datamill-analytics .
docker run -p 8050:8050 datamill-analytics
```

## Adding a language

1. Copy `i18n/locales/es.json` to `i18n/locales/<code>.json` and translate every leaf value (keep all `{placeholders}` unchanged).
2. Add `"<code>": "<native name>"` to `LANGUAGES` in `config.py`.

Currently supported: Español, English, Português, Italiano, Français, Deutsch.

## Architecture notes

- **Why a static dataset, not a live API.** An earlier iteration of this project integrated a live football API. It was deliberately removed: the supervised and unsupervised engines depend on rigorous, *reproducible* evaluation (fixed seeds, stratified CV, paired statistical tests) — results need to be identical on every run for the methodology to mean anything. A live feed would also have required a paid API tier to keep per-match statistics (shots, corners, cards) populated, which power the clustering KPIs; a free tier's quota silently degrades that data. The data ingestion layer (`utils/data_loader.py`) is intentionally a single, swappable seam: pointing `config.DATA_FILE` at a Parquet file, or re-introducing a live source for a different use case, requires no changes anywhere else in the pipeline.
- **Explicit `dcc.Location`-based routing** (rather than Dash Pages) so each page layout can be rebuilt on language change, with the translator injected as a parameter.
- **Active language lives in a `dcc.Store`** (memory). To persist across reloads, switch `storage_type` to `"local"`.
- **The Plotly theme is applied once**, at startup (`apply_plotly_theme()` in `app.py`); every chart — `graph_objects` or Plotly Express — inherits the CVD-safe palette automatically.

## Author

**Brais Muiño Otero** — Data Science student (UDC), targeting ML/applied-science roles.

🔗 [github.com/brais-muino-otero](https://github.com/brais-muino-otero)

## License

MIT — see [LICENSE](LICENSE).
