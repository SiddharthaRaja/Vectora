# Two stages: wheels are built where the compilers are, and the runtime image
# carries only what it needs to run.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY vectora ./vectora
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[ingest,metrics,eval]" \
    # CPU-only torch: the CUDA build is ~2GB and useless without a GPU.
    && /opt/venv/bin/pip install --force-reinstall \
       --index-url https://download.pytorch.org/whl/cpu torch

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/vectora/.cache/huggingface \
    VECTORA_HOST=0.0.0.0 \
    VECTORA_PORT=5000

# libgomp is needed by torch; the rest is for the optional PDF export.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash vectora

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=vectora:vectora vectora ./vectora
COPY --chown=vectora:vectora static ./static
COPY --chown=vectora:vectora eval ./eval
COPY --chown=vectora:vectora pyproject.toml README.md ./

RUN mkdir -p /app/data && chown -R vectora:vectora /app /home/vectora
USER vectora

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "vectora.api.app:app", "--host", "0.0.0.0", "--port", "5000", \
     "--log-config", "/dev/null", "--no-access-log"]
