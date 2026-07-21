"""JSON log formatting and OTEL wiring (telemetry.py, ADR-012)."""

import json
import logging

import pytest
from fastapi import FastAPI
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prediction_engine import telemetry
from prediction_engine.config import Settings
from prediction_engine.telemetry import JsonLogFormatter, configure_logging, configure_telemetry


def make_record(message: str, level: int = logging.INFO, exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test.logger", level=level, pathname=__file__, lineno=1, msg=message, args=(), exc_info=exc_info
    )


@pytest.fixture
def restore_root_logging():
    """configure_logging replaces the root handlers; keep the suite's intact."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


class TestJsonLogFormatter:
    def test_emits_single_line_json_with_level_logger_and_message(self) -> None:
        entry = json.loads(JsonLogFormatter().format(make_record("hello", level=logging.WARNING)))
        assert entry["level"] == "warning"
        assert entry["logger"] == "test.logger"
        assert entry["message"] == "hello"
        assert "timestamp" in entry
        # no active span: trace correlation fields are omitted entirely
        assert "trace_id" not in entry and "span_id" not in entry

    def test_includes_the_formatted_exception(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = make_record("failed", level=logging.ERROR, exc_info=sys.exc_info())
        entry = json.loads(JsonLogFormatter().format(record))
        assert "ValueError: boom" in entry["exception"]

    def test_correlates_with_the_active_span(self) -> None:
        # a local (non-global) provider is enough to make the current span context valid
        tracer = TracerProvider().get_tracer("test")
        with tracer.start_as_current_span("fmt") as span:
            entry = json.loads(JsonLogFormatter().format(make_record("in-span")))
            ctx = span.get_span_context()
        assert entry["trace_id"] == format(ctx.trace_id, "032x")
        assert entry["span_id"] == format(ctx.span_id, "016x")


class TestConfigureLogging:
    def test_installs_a_json_handler_at_the_requested_level(self, restore_root_logging) -> None:
        configure_logging("warning")
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonLogFormatter)


class TestConfigureTelemetry:
    def test_without_an_endpoint_only_logging_is_configured(self, restore_root_logging) -> None:
        app = FastAPI()
        configure_telemetry(app, Settings(otel_exporter_otlp_endpoint=None, log_level="debug"))
        assert logging.getLogger().level == logging.DEBUG
        # no instrumentation happened
        assert not getattr(app, "_is_instrumented_by_opentelemetry", False)

    def test_with_an_endpoint_wires_exporters_and_instruments_the_app(
        self, restore_root_logging, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # swap the OTLP exporters for in-memory stand-ins so no gRPC channel
        # or export thread is spawned; capture the constructor kwargs to
        # assert the endpoint actually flows through
        span_exporter_kwargs: dict = {}
        metric_exporter_kwargs: dict = {}

        def fake_span_exporter(**kwargs):
            span_exporter_kwargs.update(kwargs)
            return InMemorySpanExporter()

        def fake_metric_exporter(**kwargs):
            metric_exporter_kwargs.update(kwargs)
            return object()

        monkeypatch.setattr(telemetry, "OTLPSpanExporter", fake_span_exporter)
        monkeypatch.setattr(telemetry, "OTLPMetricExporter", fake_metric_exporter)
        monkeypatch.setattr(telemetry, "BatchSpanProcessor", SimpleSpanProcessor)
        monkeypatch.setattr(telemetry, "PeriodicExportingMetricReader", lambda exporter: InMemoryMetricReader())

        app = FastAPI()
        settings = Settings(otel_exporter_otlp_endpoint="collector:4317", otel_service_name="prediction-engine-test")
        try:
            configure_telemetry(app, settings, engine=None)
        finally:
            # httpx instrumentation is process-global; undo it for the rest of the suite
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().uninstrument()

        assert span_exporter_kwargs == {"endpoint": "collector:4317", "insecure": True}
        assert metric_exporter_kwargs == {"endpoint": "collector:4317", "insecure": True}
        assert getattr(app, "_is_instrumented_by_opentelemetry", False)
