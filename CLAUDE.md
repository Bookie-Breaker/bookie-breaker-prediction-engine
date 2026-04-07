# bookie-breaker-prediction-engine

## Service Purpose

Python/FastAPI ML-based probability calibration layer. Takes simulation distributions and contextual features (injuries, rest, weather, line movement) to produce calibrated probabilities with confidence intervals.

## Language & Conventions

- **Language:** Python 3.12
- **Framework:** FastAPI + uvicorn, XGBoost/LightGBM for ML
- **Project layout:** `src/prediction_engine/` package, `main.py` FastAPI entry point
- **Naming:** `snake_case.py` files, `snake_case` functions, `PascalCase` classes
- **Package manager:** uv
- **Testing:** pytest in `tests/`

## Key Files

- `src/prediction_engine/main.py` — FastAPI app entry point
- `src/prediction_engine/api/` — Route handlers
- `src/prediction_engine/core/` — Feature engineering, model inference, calibration
- `src/prediction_engine/models/` — Pydantic models
- `pyproject.toml` — Dependencies and tool config

## Service-Specific Commands

```bash
task dev          # uvicorn with --reload on port 8004
task lint         # ruff check + format
task test         # pytest --cov
task typecheck    # mypy src/
```

## Dependencies

- **statistics-service** (port 8002) — Features: team stats, injury data
- **lines-service** (port 8001) — Features: line movement data
- **PostgreSQL** (predictions schema) — Store predictions, model versions, feature vectors
- **Redis** — Caching

## Environment Variables

See `.env.example`. Key: `DATABASE_URL`, `STATISTICS_SERVICE_URL`, `LINES_SERVICE_URL`, `REDIS_URL`, `MODEL_DIR`, `PORT=8004`.
