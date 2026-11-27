"""Tracing for every turn. Do not edit.

Every span is always written to traces/<run_id>.jsonl in the shared QanoonArena format,
which is what you submit. If you also want to see traces in Langfuse or Arize Phoenix,
install `opentelemetry-sdk opentelemetry-exporter-otlp` and set the environment variables
in README.md (OTEL_EXPORTER_OTLP_ENDPOINT, and headers for Langfuse). The same spans are
then sent there too.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

REQUIRED_SPANS = ("memory_read", "retrieve", "generate", "cite_check", "memory_write")
REQUIRED_TURN_ATTRS = (
    "run_id", "group_id", "config", "memory", "adapter", "seed", "split", "session_id", "turn",
    "config_hash", "prompt_hash", "adapter_hash", "tokens_in", "tokens_out", "latency_ms", "mem_tokens",
)


class _OTel:
    """Optional OpenTelemetry mirror (Langfuse / Phoenix both accept OTLP)."""

    def __init__(self):
        self.tracer = None
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create({"service.name": "qanoonarena"}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("qanoonarena")
            self._provider = provider
        except Exception as e:  # tracing must never break a run
            print(f"[tracing] OpenTelemetry export disabled: {e}")

    def flush(self):
        if self.tracer is not None:
            try:
                self._provider.force_flush()
            except Exception:
                pass


def _clean(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


class Tracer:
    def __init__(self, run_id: str, traces_dir: str = "traces"):
        self.run_id = run_id
        Path(traces_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(traces_dir) / f"{run_id}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")
        self._stack: list[dict] = []
        self._otel = _OTel()
        self._otel_stack: list = []

    @contextmanager
    def span(self, name: str, **attrs):
        sp = {
            "run_id": self.run_id,
            "trace_id": self._stack[0]["trace_id"] if self._stack else uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "parent_id": self._stack[-1]["span_id"] if self._stack else None,
            "name": name,
            "start_ms": int(time.time() * 1000),
            "attributes": {k: _clean(v) for k, v in attrs.items()},
        }
        self._stack.append(sp)
        otel_cm = None
        if self._otel.tracer is not None:
            otel_cm = self._otel.tracer.start_as_current_span(name)
            self._otel_stack.append(otel_cm.__enter__())
        t0 = time.perf_counter()
        try:
            yield sp["attributes"]
        except Exception as e:
            sp["attributes"]["error"] = repr(e)
            raise
        finally:
            sp["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            sp["end_ms"] = sp["start_ms"] + sp["latency_ms"]
            self._stack.pop()
            if otel_cm is not None:
                o = self._otel_stack.pop()
                for k, v in sp["attributes"].items():
                    if isinstance(v, (str, int, float, bool)):
                        o.set_attribute(f"qa.{k}", v)
                    elif v is not None:
                        o.set_attribute(f"qa.{k}", json.dumps(v)[:4000])
                otel_cm.__exit__(None, None, None)
            self._f.write(json.dumps(sp, ensure_ascii=False) + "\n")
            self._f.flush()

    def close(self):
        self._f.close()
        self._otel.flush()


class NullTracer:
    """Used when you call the agent interactively without tracing."""

    run_id = "interactive"

    @contextmanager
    def span(self, name: str, **attrs):
        yield dict(attrs)

    def close(self):
        pass
