# syntax=docker/dockerfile:1.7
# Inference image: CPU PyTorch + the `serve` extra only. Build from the repository root:
#   docker build -f docker/Dockerfile -t decider-serve .

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra cpu --extra serve
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra cpu --extra serve

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 decider     && mkdir -p /home/decider/logs /home/decider/data/feedback     && chown -R decider:decider /home/decider/logs /home/decider/data
COPY --from=builder --chown=decider:decider /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    DECIDER_BUNDLE=/models/bundle
USER decider
WORKDIR /home/decider
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')"
ENTRYPOINT ["decider-serve", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]
