# syntax=docker/dockerfile:1

# ---------- builder: install deps into an isolated venv ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------- runtime: slim image, no build tooling, non-root ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

# Non-root user (defense in depth; required by k8s "restricted" PodSecurity).
RUN groupadd -r app && useradd -r -g app -u 10001 app

# Copy only the built virtualenv from the builder — no compilers/caches shipped.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . /app
# Ensure the non-root user can write the paths it needs at runtime (sqlite db,
# uploads, predictions, outputs). In production these are typically mounted
# volumes; baking them writable lets the image also run standalone.
RUN chmod +x /app/docker-entrypoint.sh \
 && mkdir -p /app/data/uploads /app/data/processed /app/data/raw/datasets /app/outputs \
 && chown -R app:app /app \
 && chmod -R u+rwX /app/data /app/outputs

USER app

EXPOSE 8000

# Liveness from inside the container (no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "scripts/run_api.py", "--config", "configs/prod.yaml"]
