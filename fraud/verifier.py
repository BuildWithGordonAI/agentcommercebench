"""
AgentVerifier — formal policy verification layer.

This is distinct from ML guards: instead of a probability score,
the verifier gives a deterministic PASS/FAIL against a declared policy.

Positioning: "verifier framework" — we check invariants, not patterns.
An agent that passes all ML guards can still FAIL formal verification.

Policy → Verifier → PolicyViolation list → Decision

Usage:
    policy   = AgentPolicy(max_spend_per_tx=500_000, allowed_categories={"search"})
    verifier = AgentVerifier(policy)

    for event in session.events:
        violations = verifier.verify(event, context)
        if any(v.severity == "BLOCK" for v in violations):
            raise PolicyViolationError(violations)

This maps cleanly to the fraud/ pipeline:
    FraudPipeline.default() runs ML guards.
    AgentVerifier runs formal checks.
    Both produce signals that flow into FraudContext.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime, timezone


# ── Policy language ───────────────────────────────────────────────────────────

@dataclass
class AgentPolicy:
    """
    Formal policy for one agent session.
    All limits are in μUSDC (1 USDC = 1_000_000 μUSDC).

    Defaults are permissive — callers tighten per persona/role.
    """
    # Spend limits
    max_spend_per_tx:      int = 10_000_000     # $10 USDC per transaction
    max_spend_per_session: int = 100_000_000    # $100 USDC per session

    # Call rate limits
    max_find_service_calls: int = 30
    max_authorize_calls:    int = 10

    # Category / vendor allow-list (empty = unrestricted)
    allowed_categories:  frozenset[str] = field(default_factory=frozenset)
    blocked_categories:  frozenset[str] = field(default_factory=frozenset)
    require_known_vendor: bool = False
    known_vendors:       frozenset[str] = field(default_factory=frozenset)

    # Temporal constraints (24h clock, UTC)
    active_hours: tuple[int, int] = (0, 24)     # (start_hour, end_hour)

    # Session duration cap (minutes)
    max_session_duration_minutes: int = 240

    @classmethod
    def for_persona(cls, persona: str) -> "AgentPolicy":
        """Opinionated defaults per persona."""
        presets: dict[str, dict] = {
            "procurement": {
                "max_spend_per_tx": 500_000_000,        # $500
                "max_spend_per_session": 2_000_000_000,  # $2000
                "allowed_categories": frozenset({"procurement", "finance"}),
                "max_authorize_calls": 20,
            },
            "travel": {
                "max_spend_per_tx": 300_000_000,         # $300
                "max_spend_per_session": 1_000_000_000,  # $1000
                "allowed_categories": frozenset({"travel", "finance"}),
            },
            "research": {
                "max_spend_per_tx": 100_000_000,         # $100
                "max_spend_per_session": 300_000_000,    # $300
                "allowed_categories": frozenset({"ai", "search", "finance"}),
                "max_find_service_calls": 50,
            },
        }
        kwargs = presets.get(str(persona).lower(), {})
        return cls(**kwargs)


# ── Violation record ──────────────────────────────────────────────────────────

@dataclass
class PolicyViolation:
    rule:     str                           # machine-readable rule ID
    severity: Literal["WARN", "BLOCK"]
    details:  str                           # human-readable explanation
    event_id: str

    def to_dict(self) -> dict:
        return {
            "rule":     self.rule,
            "severity": self.severity,
            "details":  self.details,
            "event_id": self.event_id,
        }


# ── Stateful verifier ─────────────────────────────────────────────────────────

class AgentVerifier:
    """
    Stateful, real-time policy verifier for one agent session.

    Instantiate per-session. Call verify(event, history) for each event.
    Unlike ML guards, violations are deterministic — same input → same result.

    The verifier runs BEFORE ML guards in the pipeline so hard policy
    violations are caught without paying ML inference cost.
    """

    def __init__(self, policy: AgentPolicy):
        self.policy = policy
        self._session_spend     = 0
        self._find_service_calls = 0
        self._authorize_calls   = 0
        self._session_start: Optional[datetime] = None
        self.violations: list[PolicyViolation] = []

    def verify(self, event, history: list = None) -> list[PolicyViolation]:
        """
        Verify one event against the policy.
        Returns list of violations emitted by this event (may be empty).
        Accumulates state (spend, call counts) across calls.
        """
        action  = str(getattr(event, "action_type", "")).lower()
        eid     = getattr(event, "event_id", "?")
        ts      = getattr(event, "timestamp", None)
        amount  = getattr(event, "amount_units", 0) or 0
        cat     = getattr(event, "category",     None)
        vendor  = getattr(event, "vendor",        None)

        if ts and self._session_start is None:
            self._session_start = ts

        new_violations: list[PolicyViolation] = []

        def flag(rule: str, severity: str, details: str):
            v = PolicyViolation(rule=rule, severity=severity, details=details, event_id=eid)
            new_violations.append(v)
            self.violations.append(v)

        # ── Per-transaction spend limit ───────────────────────────────────
        if "authorize" in action and amount > self.policy.max_spend_per_tx:
            flag(
                "spend.per_tx",
                "BLOCK",
                f"Transaction {amount} μUSDC > policy max {self.policy.max_spend_per_tx}",
            )

        # ── Session spend accumulation ────────────────────────────────────
        if "authorize" in action or "settle" in action:
            self._session_spend += amount
            if self._session_spend > self.policy.max_spend_per_session:
                flag(
                    "spend.per_session",
                    "BLOCK",
                    f"Session total {self._session_spend} μUSDC > policy max {self.policy.max_spend_per_session}",
                )

        # ── Call rate limits ──────────────────────────────────────────────
        if "find_service" in action or "get_service" in action:
            self._find_service_calls += 1
            if self._find_service_calls > self.policy.max_find_service_calls:
                flag(
                    "rate.find_service",
                    "WARN",
                    f"find_service call #{self._find_service_calls} > policy max {self.policy.max_find_service_calls}",
                )

        if "authorize" in action:
            self._authorize_calls += 1
            if self._authorize_calls > self.policy.max_authorize_calls:
                flag(
                    "rate.authorize",
                    "BLOCK",
                    f"authorize call #{self._authorize_calls} > policy max {self.policy.max_authorize_calls}",
                )

        # ── Category allow/block list ─────────────────────────────────────
        if cat:
            if self.policy.blocked_categories and cat in self.policy.blocked_categories:
                flag("category.blocked", "BLOCK", f"Category '{cat}' is blocked by policy")
            elif self.policy.allowed_categories and cat not in self.policy.allowed_categories:
                flag("category.not_allowed", "WARN", f"Category '{cat}' not in policy allow-list")

        # ── Vendor allow list ─────────────────────────────────────────────
        if "authorize" in action and self.policy.require_known_vendor:
            if vendor and self.policy.known_vendors and vendor not in self.policy.known_vendors:
                flag("vendor.unknown", "BLOCK", f"Vendor '{vendor[:40]}' not in known_vendors")

        # ── Temporal constraint ───────────────────────────────────────────
        if ts:
            h = ts.hour
            start, end = self.policy.active_hours
            if not (start <= h < end):
                flag(
                    "temporal.off_hours",
                    "WARN",
                    f"Action at hour={h} outside active window [{start},{end})",
                )

        # ── Session duration ──────────────────────────────────────────────
        if ts and self._session_start:
            duration_minutes = (ts - self._session_start).total_seconds() / 60
            if duration_minutes > self.policy.max_session_duration_minutes:
                flag(
                    "duration.exceeded",
                    "WARN",
                    f"Session duration {duration_minutes:.0f}m > policy max {self.policy.max_session_duration_minutes}m",
                )

        return new_violations

    # ── Summary ───────────────────────────────────────────────────────────────

    @property
    def is_blocked(self) -> bool:
        return any(v.severity == "BLOCK" for v in self.violations)

    @property
    def block_reasons(self) -> list[str]:
        return [v.details for v in self.violations if v.severity == "BLOCK"]

    def summary(self) -> dict:
        return {
            "blocked":       self.is_blocked,
            "n_violations":  len(self.violations),
            "n_blocks":      sum(1 for v in self.violations if v.severity == "BLOCK"),
            "n_warnings":    sum(1 for v in self.violations if v.severity == "WARN"),
            "session_spend": self._session_spend,
            "violations":    [v.to_dict() for v in self.violations],
        }
