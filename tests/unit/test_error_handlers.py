"""Error envelope handlers (api/errors.py) against the api-contracts code table."""

from typing import Annotated

import pytest
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient

from prediction_engine.api.errors import NotFoundError, register_error_handlers


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/missing")
    async def missing() -> None:
        raise NotFoundError("thing not found", details={"id": "42"})

    @app.get("/typed")
    async def typed(limit: Annotated[int, Query()]) -> dict[str, int]:
        return {"limit": limit}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal state")

    return TestClient(app, raise_server_exceptions=False)


def test_api_errors_render_their_code_and_details(client: TestClient) -> None:
    response = client.get("/missing")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert body["error"]["message"] == "thing not found"
    assert body["error"]["details"] == {"id": "42"}
    assert "timestamp" in body["meta"]


def test_request_validation_failures_are_400_with_field_locations(client: TestClient) -> None:
    response = client.get("/typed", params={"limit": "not-a-number"})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    errors = body["error"]["details"]["errors"]
    assert errors and errors[0]["loc"] == ["query", "limit"]
    assert errors[0]["msg"]


def test_unexpected_exceptions_are_opaque_500s(client: TestClient) -> None:
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    # the raw exception message must not leak into the envelope
    assert "secret" not in body["error"]["message"]
    assert body["error"]["message"] == "An unexpected error occurred"
    assert body["error"]["details"] == {}
