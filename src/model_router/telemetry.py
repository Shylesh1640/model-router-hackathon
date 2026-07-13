"""Langfuse telemetry — optional observability wrapper for the routing pipeline.

Wraps pipeline calls with Langfuse traces. Each pipeline stage
(intent, decomposition, classification, routing, generation)
is logged as a span with timing, tokens, and model info.

Fully optional — if langfuse isn't installed, tracing is a no-op.

Usage:
    from .telemetry import Telemetry

    tracer = Telemetry()
    tracer.setup()

    # In pipeline:
    trace = tracer.trace_route(query)
    # ... do stages ...
    trace.end(response)
"""

import logging
import os
import time
from typing import Optional, Any

from .models import RouteResponse

logger = logging.getLogger(__name__)


class _NoOpSpan:
    """Stand-in when Langfuse is not available. All methods are no-ops."""

    def update(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass

    def span(self, **kwargs):
        return _NoOpSpan()

    def generation(self, **kwargs):
        return _NoOpSpan()


class _NoOpTrace:
    """Stand-in trace when Langfuse is not available."""

    def __init__(self):
        self._spans = []

    def span(self, **kwargs):
        s = _NoOpSpan()
        self._spans.append(s)
        return s

    def generation(self, **kwargs):
        return _NoOpSpan()

    def update(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass


class Telemetry:
    """Optional Langfuse observability wrapper.

    Lazily initialises Langfuse — no-op if the package is missing
    or no API key is configured.

    Usage:
        telemetry = Telemetry()
        telemetry.setup(public_key=..., secret_key=..., host=...)

        # Inside pipeline.route():
        trace = telemetry.start_trace(query)
        intent_span = trace.span(name="intent")
        # ... detect intent ...
        intent_span.end(output=intent)
        # etc.
        trace.end(output=response)
    """

    def __init__(self):
        self._langfuse = None
        self._available = False

    def setup(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> bool:
        """Initialise Langfuse client. Returns True if ready, False if no-op.

        Reads env vars LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        if keys aren't provided directly.
        """
        if self._langfuse is not None:
            return self._available

        pk = public_key or os.getenv("LANGFUSE_PUBLIC_KEY") or os.getenv("LANGFUSE_PK")
        sk = secret_key or os.getenv("LANGFUSE_SECRET_KEY") or os.getenv("LANGFUSE_SK")
        host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not pk or not sk:
            logger.info("Langfuse: no credentials — telemetry disabled")
            self._available = False
            return False

        try:
            from langfuse import Langfuse  # type: ignore
            self._langfuse = Langfuse(
                public_key=pk,
                secret_key=sk,
                host=host,
            )
            self._available = True
            logger.info("Langfuse: telemetry enabled")
            return True
        except ImportError:
            logger.info("Langfuse: package not installed — telemetry disabled")
            self._available = False
            return False
        except Exception as e:
            logger.warning("Langfuse: init failed (%s) — telemetry disabled", e)
            self._available = False
            return False

    def is_available(self) -> bool:
        return self._available and self._langfuse is not None

    def start_trace(
        self,
        query: str,
        name: str = "model-router",
        metadata: Optional[dict] = None,
    ):
        """Start a new trace for a routing request."""
        if not self.is_available():
            return _NoOpTrace()

        try:
            trace = self._langfuse.trace(
                name=name,
                input={"query": query},
                metadata=metadata or {},
            )
            return _LangfuseTraceAdapter(trace)
        except Exception as e:
            logger.warning("Langfuse: failed to start trace (%s)", e)
            return _NoOpTrace()

    def flush(self):
        """Flush any pending Langfuse events."""
        if self.is_available():
            try:
                self._langfuse.flush()
            except Exception:
                pass


class _LangfuseSpanAdapter:
    """Wraps a Langfuse span in a simple start/end interface."""

    def __init__(self, span):
        self._span = span
        self._start_time = time.perf_counter()

    def update(self, **kwargs):
        try:
            self._span.update(**kwargs)
        except Exception:
            pass

    def end(self, **kwargs):
        latency = round((time.perf_counter() - self._start_time) * 1000, 1)
        try:
            self._span.end(**kwargs)
        except Exception:
            pass

    def span(self, **kwargs):
        try:
            child = self._span.span(**kwargs)
            return _LangfuseSpanAdapter(child)
        except Exception:
            return _NoOpSpan()

    def generation(self, **kwargs):
        try:
            child = self._span.generation(**kwargs)
            return _LangfuseSpanAdapter(child)
        except Exception:
            return _NoOpSpan()


class _LangfuseTraceAdapter:
    """Wraps a Langfuse trace, working like _NoOpTrace but real."""

    def __init__(self, trace):
        self._trace = trace
        self._spans = []

    def span(self, **kwargs):
        try:
            s = self._trace.span(**kwargs)
            adapter = _LangfuseSpanAdapter(s)
            self._spans.append(adapter)
            return adapter
        except Exception:
            return _NoOpSpan()

    def generation(self, **kwargs):
        try:
            g = self._trace.generation(**kwargs)
            return _LangfuseSpanAdapter(g)
        except Exception:
            return _NoOpSpan()

    def update(self, **kwargs):
        try:
            self._trace.update(**kwargs)
        except Exception:
            pass

    def end(self, **kwargs):
        try:
            self._trace.end(**kwargs)
        except Exception:
            pass


# Singleton
_telemetry: Optional[Telemetry] = None


def get_telemetry() -> Telemetry:
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry()
    return _telemetry
