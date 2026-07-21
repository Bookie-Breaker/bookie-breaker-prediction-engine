"""ServiceClient envelope handling and error mapping (clients/base.py)."""

import httpx
import pytest

from prediction_engine.api.errors import DependencyError, DependencyTimeoutError, NotFoundError
from prediction_engine.clients.base import ServiceClient


def make_client(handler) -> ServiceClient:
    transport = httpx.MockTransport(handler)
    return ServiceClient("http://svc.test/", httpx.AsyncClient(transport=transport))


def enveloped_response(data, meta: dict | None = None) -> httpx.Response:
    payload = {"data": data}
    if meta is not None:
        payload["meta"] = meta
    return httpx.Response(200, json=payload)


class TestGetData:
    async def test_returns_the_envelope_data_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/things/42"
            assert request.url.params["limit"] == "5"
            return enveloped_response({"id": "42"})

        data = await make_client(handler).get_data("/api/v1/things/42", "thing", params={"limit": 5})
        assert data == {"id": "42"}

    async def test_timeout_maps_to_dependency_timeout_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow upstream", request=request)

        with pytest.raises(DependencyTimeoutError, match="upstream timed out fetching thing"):
            await make_client(handler).get_data("/api/v1/things", "thing")

    async def test_transport_error_maps_to_dependency_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(DependencyError, match="upstream is unavailable"):
            await make_client(handler).get_data("/api/v1/things", "thing")

    async def test_404_maps_to_not_found_error(self) -> None:
        client = make_client(lambda request: httpx.Response(404, json={"error": {}}))
        with pytest.raises(NotFoundError, match="thing not found in upstream"):
            await client.get_data("/api/v1/things/missing", "thing")

    async def test_5xx_maps_to_dependency_error_with_status(self) -> None:
        client = make_client(lambda request: httpx.Response(503, json={}))
        with pytest.raises(DependencyError, match="upstream returned 503 for thing"):
            await client.get_data("/api/v1/things", "thing")

    async def test_missing_data_key_is_a_malformed_envelope(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json={"items": []}))
        with pytest.raises(DependencyError, match="malformed envelope"):
            await client.get_data("/api/v1/things", "thing")


class TestGetMetaTimestamp:
    async def test_returns_the_meta_timestamp(self) -> None:
        client = make_client(lambda request: enveloped_response([], meta={"timestamp": "2026-07-20T00:00:00Z"}))
        assert await client.get_meta_timestamp("/api/v1/things") == "2026-07-20T00:00:00Z"

    async def test_missing_timestamp_returns_none(self) -> None:
        client = make_client(lambda request: enveloped_response([], meta={}))
        assert await client.get_meta_timestamp("/api/v1/things") is None

    async def test_non_200_returns_none(self) -> None:
        client = make_client(lambda request: httpx.Response(500, json={}))
        assert await client.get_meta_timestamp("/api/v1/things") is None

    async def test_transport_error_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        assert await make_client(handler).get_meta_timestamp("/api/v1/things") is None


class TestIsHealthy:
    async def test_200_is_healthy(self) -> None:
        client = make_client(lambda request: httpx.Response(200, json={"data": {"status": "healthy"}}))
        assert await client.is_healthy("/api/v1/things/health") is True

    async def test_non_200_is_unhealthy(self) -> None:
        client = make_client(lambda request: httpx.Response(503, json={}))
        assert await client.is_healthy("/api/v1/things/health") is False

    async def test_transport_error_is_unhealthy(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        assert await make_client(handler).is_healthy("/api/v1/things/health") is False
