"""
fraud — Agentic Payment Fraud Detection for Gordon AI.

Three-layer guard pipeline:

    Pre-Commerce  →  At-Commerce  →  Settlement
    (agent harness)  (wallet/rail)   (network graph)

Layers communicate through FraudContext — the inter-layer state object.
No guard is blind to what came before.

Quickstart:

    from fraud.pipeline import FraudPipeline
    from fraud.context  import FraudContext, Decision

    pipeline = FraudPipeline.default()
    context  = pipeline.process_session(session)

    if context.decision == Decision.BLOCK:
        raise PaymentFraudError(context.risk_flags)
"""
from fraud.context  import FraudContext, GuardSignal, Phase, Decision
from fraud.pipeline import FraudPipeline

__all__ = ["FraudPipeline", "FraudContext", "GuardSignal", "Phase", "Decision"]
__version__ = "0.1.0"
