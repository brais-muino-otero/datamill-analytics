---
title: DataMill Analytics
emoji: ⚽
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8050
pinned: false
---

# Premier League Analytics

Web app interactiva (Dash + Plotly) para analisis y prediccion de la Premier
League. Multi-pagina, multi-idioma (6 lenguas) y con paleta de colores segura
para daltonicos por defecto.

## Stack
- Dash + Dash Bootstrap Components (tema DARKLY)
- Plotly (tema oscuro + paleta colorblind-safe por defecto)
- i18n propio basado en diccionarios JSON

## Ejecucion (local)
```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```
Abre http://127.0.0.1:8050

## Anadir un idioma
1. Crea `i18n/locales/<codigo>.json` copiando el esquema de `es.json`.
2. Anade `"<codigo>": "<endonimo>"` a `LANGUAGES` en `config.py`.

## Notas de arquitectura
- Routing explicito con `dcc.Location` (en vez de Dash Pages) para poder
  re-renderizar cada pagina al cambiar de idioma, inyectando el traductor.
- El idioma activo vive en un `dcc.Store` (memory). Para persistirlo entre
  recargas, cambia `storage_type` a `"local"`.
- El tema Plotly se aplica una vez en `app.py` (`apply_plotly_theme()`); todas
  las graficas (graph_objects y plotly express) heredan colores CVD-safe.
