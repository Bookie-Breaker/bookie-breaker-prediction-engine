# bookie-breaker-prediction-engine

ML calibration layer (port 8004) that turns simulation distributions plus contextual features
(injuries, rest, weather, line movement) into calibrated probabilities with confidence
intervals, using XGBoost. Serves champion models per league/market with optional ensembles.
Challenger models register from operator-triggered retrains
(`POST /api/v1/predict/models/retrain`), shadow-score alongside the champion, and promote
through criteria-gated experiments endpoints. Model artifacts persist in `MODEL_DIR`; metadata
lives in the `predictions` schema (Alembic migrations).

## Quickstart

### With Docker Compose (recommended)

```bash
task up  # from BookieBreaker/ root
```

### Standalone

```bash
cp .env.example .env  # fill in values
task bootstrap
task dev
```

## API

Interactive docs at `http://localhost:8004/docs` when running. All endpoints live under
`/api/v1/predict`.

Full contract:
[prediction-engine-api.md](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/api-contracts/prediction-engine-api.md)

## Architecture Decisions

- [Hybrid Prediction Approach (ADR-002)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/002-hybrid-prediction-approach.md)
- [Probability Calibration (ADR-014)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/014-probability-calibration.md)
- [Ensemble and Challenger Serving (ADR-032)](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/decisions/032-ensemble-and-challenger-serving.md)

Model lifecycle operations (retrain, shadow, promote):
[Maintenance playbook](https://github.com/Bookie-Breaker/bookie-breaker-docs/blob/main/playbooks/09-maintenance.md)

## Environment Variables

See `.env.example` for all variables with descriptions. Key ones: `DATABASE_URL`,
`STATISTICS_SERVICE_URL`, `LINES_SERVICE_URL`, `REDIS_URL`, `MODEL_DIR`, `PORT=8004`.
