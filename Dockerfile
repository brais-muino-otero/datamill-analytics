# ----------------------------------------------------------------------------- #
# DataMill Analytics - imagen de produccion (ligera)                            #
# Base slim + gunicorn sirviendo el objeto WSGI `server` de app.py.             #
# ----------------------------------------------------------------------------- #
FROM python:3.12-slim

# Buenas practicas de runtime: sin .pyc, logs sin buffer, pip sin cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8050

WORKDIR /app

# Dependencia de sistema imprescindible para scikit-learn/scipy (OpenMP runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1) Dependencias primero (mejor cacheo de capas: solo se reinstala si cambian).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 2) Codigo de la aplicacion.
COPY . .

# 3) Usuario sin privilegios (seguridad).
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8050

# Healthcheck: la home debe responder 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8050/').status==200 else 1)" || exit 1

# Servidor de produccion. `app:server` = objeto Flask (server = app.server en app.py).
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "4", "--timeout", "120", "--preload", "app:server"]
