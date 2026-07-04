"""Shared envelope-unwrapping HTTP client base."""

from typing import Any

import httpx

from prediction_engine.api.errors import DependencyError, DependencyTimeoutError, NotFoundError


class ServiceClient:
    service_name = "upstream"

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def get_data(self, path: str, resource: str, params: dict[str, Any] | None = None) -> Any:
        """GET an enveloped endpoint and return its data payload.

        Returns dict or list depending on the endpoint. Raises NotFoundError
        on 404, DependencyError/DependencyTimeoutError on upstream failures.
        """
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise DependencyTimeoutError(f"{self.service_name} timed out fetching {resource}") from exc
        except httpx.HTTPError as exc:
            raise DependencyError(f"{self.service_name} is unavailable: {exc}") from exc
        if response.status_code == 404:
            raise NotFoundError(f"{resource} not found in {self.service_name}")
        if response.status_code >= 500:
            raise DependencyError(f"{self.service_name} returned {response.status_code} for {resource}")
        payload: dict[str, Any] = response.json()
        if "data" not in payload:
            raise DependencyError(f"{self.service_name} returned a malformed envelope for {resource}")
        return payload["data"]

    async def get_meta_timestamp(self, path: str, params: dict[str, Any] | None = None) -> str | None:
        """Fetch just the envelope meta timestamp (used for feature_sources)."""
        try:
            response = await self._client.get(f"{self._base_url}{path}", params=params)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        meta = response.json().get("meta", {})
        timestamp = meta.get("timestamp")
        return str(timestamp) if timestamp else None

    async def is_healthy(self, health_path: str) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}{health_path}", timeout=1.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
