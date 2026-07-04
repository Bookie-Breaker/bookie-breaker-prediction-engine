FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra train --no-install-project

COPY src/ src/
COPY scripts/ scripts/

# Bake the synthetic bootstrap model into the image, then prune the venv
# back to runtime-only dependencies (drops sklearn/pandas/nba_api).
RUN uv run --no-sync python scripts/train.py --synthetic --out /app/models --rounds 100 && \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim-bookworm

RUN useradd --uid 10001 --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv .venv
COPY --from=builder --chown=appuser:appuser /app/src src
COPY --from=builder --chown=appuser:appuser /app/models /models

USER appuser

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src MODEL_DIR=/models

EXPOSE 8004

CMD ["uvicorn", "prediction_engine.main:app", "--host", "0.0.0.0", "--port", "8004"]
