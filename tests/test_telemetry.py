"""Tests for Telemetry (no Langfuse installed — tests no-op path)."""

from model_router.telemetry import Telemetry, _NoOpTrace, _NoOpSpan


def test_telemetry_noop_without_creds():
    telemetry = Telemetry()
    ready = telemetry.setup()
    assert ready is False
    assert telemetry.is_available() is False


def test_telemetry_trace_noop():
    telemetry = Telemetry()
    trace = telemetry.start_trace("test query")
    assert isinstance(trace, _NoOpTrace)

    span = trace.span(name="test")
    assert isinstance(span, _NoOpSpan)
    span.end()
    span.update()

    gen = trace.generation()
    assert isinstance(gen, _NoOpSpan)
    gen.end()

    trace.end()
    trace.update()


def test_telemetry_flush_noop():
    telemetry = Telemetry()
    telemetry.flush()  # should not raise


def test_telemetry_no_langfuse_package():
    """Without langfuse installed, setup returns False."""
    telemetry = Telemetry()
    result = telemetry.setup(
        public_key="pk-test",
        secret_key="sk-test",
    )
    assert result is False
    # langfuse isn't installed in test env
