"""
benchmark_adapter — makes FraudPipeline look like a harness detector.

The benchmark evaluate.py expects a detector callable with signature:
    detect(event, history: list[Event]) -> tuple[float, list[str]]

This adapter wraps FraudPipeline so it can be dropped into the
DETECTORS dict in benchmark/evaluate.py without any changes there.

Usage in evaluate.py:

    from fraud.integrations.benchmark_adapter import FraudPipelineDetector

    adapter = FraudPipelineDetector()
    DETECTORS["gordon_fraud_v2"] = adapter.detect

Or use it as the detector callable directly:

    detector = FraudPipelineDetector()
    score, flags = detector(event, history)

Session-level scoring (used by benchmark replay.py):

    context = detector.score_session(session)  # FraudContext
"""
from __future__ import annotations
from typing import Callable
from fraud.pipeline import FraudPipeline
from fraud.context  import FraudContext


class FraudPipelineDetector:
    """
    Adapter: wraps FraudPipeline in the (event, history) → (score, flags) interface.

    Because the benchmark calls detect(event, history) per-event with a flat
    list of prior events, we reconstruct a lightweight session object per call
    and run the full pipeline to derive the context.

    This is less efficient than streaming per-event, but lets the adapter
    drop into the existing benchmark harness with zero changes to evaluate.py.
    """

    def __init__(self, pipeline: FraudPipeline = None):
        self._pipeline = pipeline or FraudPipeline.default()
        self.__name__  = "detect"   # required by replay.py for naming
        self.__module__ = "fraud.integrations.benchmark_adapter"

    def detect(self, event, history: list) -> tuple[float, list[str]]:
        """
        Per-event interface consumed by benchmark/harness/simulate/replay.py.

        Builds a synthetic session from `history + [event]` and runs the
        full pipeline up to and including the current event.
        Skips settlement guards (they need a complete session).
        """
        all_events = list(history) + [event]
        session_id = getattr(event, "session_id", "benchmark")
        agent_id   = getattr(event, "agent_id",   "agent-0")
        persona    = getattr(event, "persona",     None)

        context = FraudContext.for_session(session_id, agent_id, persona)

        for ev in all_events:
            context, _ = self._pipeline.process_event(ev, context)

        return context.risk_score, context.risk_flags

    def __call__(self, event, history: list) -> tuple[float, list[str]]:
        return self.detect(event, history)

    def score_session(self, session) -> FraudContext:
        """Full session scoring including settlement layer."""
        return self._pipeline.process_session(session)


def make_detector(pipeline: FraudPipeline = None) -> Callable:
    """Factory: returns a bare detect() function for use in DETECTORS dicts."""
    adapter = FraudPipelineDetector(pipeline)
    return adapter.detect
