"""
Pre-Commerce Guards — fire BEFORE any AUTHORIZE.

These run in the agent harness, on FIND_SERVICE and GET_SERVICE calls.
They have zero cost on the payment rail — they operate entirely on the
agent's stated intent and observed behavior.

Guards (in order of cheapness):
    PayloadScanGuard    — regex + base64 pattern matching (L1)
    IntentGuard         — category / amount vs persona profile (L3)
    ToolTrustGuard      — response latency / schema baseline anomaly (L3b)
    SequenceModelGuard  — persona-conditioned Markov sequence model (novel)
"""
from .payload    import PayloadScanGuard
from .intent     import IntentGuard
from .tool_trust import ToolTrustGuard
from .sequence   import SequenceModelGuard

__all__ = [
    "PayloadScanGuard",
    "IntentGuard",
    "ToolTrustGuard",
    "SequenceModelGuard",
]
