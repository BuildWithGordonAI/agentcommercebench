"""
The knob layer: everything that shapes benchmark data, declared in one place.

Why this exists
---------------
The previous generator hardcoded its assumptions. The worst was a $3,000 per-transaction
limit: a round number typed while writing the generator, with fraudulent amounts then drawn
from $3,010-4,500 — immediately above it — and clean traffic topping out at $2,800. The
detector's rule tested the same 3,000. Detection was perfect and false positives were zero,
and would have been even if the detector were broken, because the two populations never met.

The error underneath that is treating **"high value" as an absolute fact about the world.**
It is not. $3,000 is unremarkable for an enterprise travel agent and enormous for a research
API agent that spends cents. "High" is only meaningful relative to *this agent, in this
domain, against the limit its owner set.*

So every threshold here is relative and every parameter is published with the data:

  * a domain sets the *scale* of spend, not any absolute figure
  * an agent's limit is drawn relative to that agent's own typical spend
  * an overspend attack means "this agent exceeded **its own** limit", never "> $3,000"

The consequence is the point: **the same $2,000 payment is an attack for one agent and
ordinary for another.** No threshold placed at any fixed amount can score well, because the
answer depends on whose history it is. A detector can only do well by modelling each agent.

Limit provenance
----------------
How an agent's limit comes to exist is itself a parameter, because all three happen in
practice:

  DECLARED_FIXED   a policy number set once, offline — what the current benchmark does
  DECLARED_BUILDER the agent's developer or an admin sets it in the agent config
  OBSERVED         derived from what the agent has actually been spending, and adapted

These are not interchangeable for a detector. Under OBSERVED the limit is inferable from
history; under the declared modes it is not, and the detector can only reason about
deviation from the agent's own norm. Mixing all three in one dataset stops a detector from
assuming either.

What the detector may see
-------------------------
Nothing in this module. The config is published *alongside* the dataset so every number is
auditable and the data is reproducible — but it is not an input to detection. A detector
that reads an agent's declared limit is being handed the answer, which is the confound this
whole layer exists to remove. `redact_for_detector` enforces that boundary.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class LimitSource(str, Enum):
    """Where an agent's spend limit came from. Never visible to a detector."""

    DECLARED_FIXED = "declared_fixed"
    DECLARED_BUILDER = "declared_builder"
    OBSERVED = "observed"


@dataclass(frozen=True)
class Domain:
    """
    A commercial context with its own scale of spend.

    Scale is a log-normal over transaction amount, because spend is multiplicative: an agent
    that usually spends $10 occasionally spends $40, not $10 plus a constant. `mu` is the
    natural log of the median. (Log-normality of transaction amounts is the standard
    modelling assumption in payments; the *shape* is defensible on those grounds. The
    *values* below are a separate question — see `source`.)

    Several domains with genuinely different scales is what makes an absolute threshold
    meaningless, which is the point of having them at all.
    """

    name: str
    median_amount: float
    """Median transaction, in currency units. mu = ln(median_amount)."""
    sigma: float
    """Log-scale spread. Larger means a longer tail of legitimately big purchases."""
    mcc_pool: tuple[str, ...]
    """Merchant category codes ordinary for this domain."""
    source: str
    """
    Where `median_amount` and `sigma` came from. Required, and checked.

    A config that invents its own numbers reproduces the failure it exists to fix: a
    plausible-looking constant nobody can trace. PLACEHOLDER marks a value that is a guess
    and must be replaced with a cited figure before any published run.
    """
    typical_session_actions: tuple[int, int] = (2, 6)
    """Range of actions in a normal session, so session shape is not a giveaway either."""

    @property
    def is_sourced(self) -> bool:
        return not self.source.startswith("PLACEHOLDER")

    @property
    def mu(self) -> float:
        return math.log(self.median_amount)

    def sample_amount(self, rng: random.Random) -> float:
        return math.exp(rng.gauss(self.mu, self.sigma))


# Four domains spanning roughly three orders of magnitude, so that no single amount is "high"
# across all of them — the research agent's limit sits below the enterprise agent's median.
#
# The SPREAD is the load-bearing design choice and it is deliberate. The VALUES are not yet
# defensible: every one is currently a guess by the author of this file, which is precisely
# the kind of untraceable constant the audit flagged elsewhere. They are marked PLACEHOLDER
# and `GeneratorConfig.unsourced()` reports them, so a published run cannot quietly rest on
# invented figures.
DOMAINS: tuple[Domain, ...] = (
    Domain("research_api", median_amount=0.40, sigma=1.1,
           mcc_pool=("7372", "7374"),
           source="PLACEHOLDER: per-call inference and search API pricing. Replace with "
                  "published price sheets (e.g. model and search API list prices)."),
    Domain("consumer_saas", median_amount=45.0, sigma=0.9,
           mcc_pool=("5817", "5818", "7372"),
           source="PLACEHOLDER: monthly seat pricing. Replace with a SaaS pricing survey."),
    Domain("ops_tooling", median_amount=320.0, sigma=1.0,
           mcc_pool=("7372", "7379", "5045"),
           source="PLACEHOLDER: team-tier tooling invoices. Replace with a cited source."),
    Domain("enterprise_travel", median_amount=1450.0, sigma=0.8,
           mcc_pool=("4511", "7011", "4722"),
           source="PLACEHOLDER: corporate airfare and hotel. Replace with a published "
                  "business-travel spend report."),
)


@dataclass(frozen=True)
class AgentPolicy:
    """
    One agent's spend policy — the thing an attack is defined relative to.

    `limit` is the per-transaction ceiling its owner set. It is drawn as a multiple of the
    agent's *own* typical spend rather than as an absolute figure, so it lands in a different
    place for every agent.
    """

    agent_id: str
    domain: str
    typical_amount: float
    """The agent's own median transaction. Not visible to a detector; it must infer this."""
    limit: float
    limit_source: LimitSource
    limit_multiple: float
    """limit / typical_amount — how much headroom this owner allowed."""

    def is_over_limit(self, amount: float) -> bool:
        """The only definition of overspend in this benchmark. Relative, by construction."""
        return amount > self.limit


@dataclass
class GeneratorConfig:
    """
    Every knob, in one auditable object. Published with the dataset it produced.

    Ranges rather than points wherever a value could otherwise become a shared constant: a
    single number here would reappear as a threshold on the detector side, which is exactly
    the coupling this layer removes.
    """

    seed: int = 42
    n_agents: int = 60
    sessions_per_agent: tuple[int, int] = (6, 20)

    domains: tuple[Domain, ...] = DOMAINS
    domain_weights: tuple[float, ...] = (0.3, 0.3, 0.25, 0.15)

    limit_multiple_range: tuple[float, float] = (2.5, 12.0)
    """How much headroom an owner allows over the agent's typical spend. Wide and
    overlapping, so the limit cannot be recovered from the domain."""

    limit_source_mix: tuple[tuple[LimitSource, float], ...] = (
        (LimitSource.DECLARED_FIXED, 0.4),
        (LimitSource.DECLARED_BUILDER, 0.3),
        (LimitSource.OBSERVED, 0.3),
    )

    overspend_excess_range: tuple[float, float] = (1.02, 3.0)
    """An overspend attack exceeds the agent's own limit by this factor. Starting just above
    1.0 is deliberate: marginal violations must overlap other agents' ordinary traffic, or
    the class becomes separable by amount alone."""

    attack_rate: float = 0.25

    notes: str = ""

    # ── derived ──────────────────────────────────────────────────────────
    def sample_agents(self, rng: Optional[random.Random] = None) -> list[AgentPolicy]:
        rng = rng or random.Random(self.seed)
        sources = [s for s, _ in self.limit_source_mix]
        weights = [w for _, w in self.limit_source_mix]

        agents: list[AgentPolicy] = []
        for i in range(self.n_agents):
            domain = rng.choices(self.domains, weights=self.domain_weights)[0]
            # Each agent's own median wanders around its domain's, so agents inside a domain
            # are not interchangeable either.
            typical = math.exp(rng.gauss(domain.mu, domain.sigma * 0.4))
            multiple = rng.uniform(*self.limit_multiple_range)
            agents.append(AgentPolicy(
                agent_id=f"agent-{i:04d}",
                domain=domain.name,
                typical_amount=typical,
                limit=typical * multiple,
                limit_source=rng.choices(sources, weights=weights)[0],
                limit_multiple=multiple,
            ))
        return agents

    def unsourced(self) -> list[str]:
        """
        Parameters still resting on a guess.

        Non-empty means this config is not publishable: the same audit that condemned the
        $3,000 limit condemns any figure here that nobody can trace.
        """
        return [f"domain.{d.name}: {d.source}" for d in self.domains if not d.is_sourced]

    def domain(self, name: str) -> Domain:
        for d in self.domains:
            if d.name == name:
                return d
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["domains"] = [asdict(d) for d in self.domains]
        out["limit_source_mix"] = [[s.value, w] for s, w in self.limit_source_mix]
        return out

    def save(self, path) -> None:
        """Publish the config next to the data it produced."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(p)


# ── The boundary ─────────────────────────────────────────────────────────────

DETECTOR_FORBIDDEN = frozenset({
    "limit", "limit_source", "limit_multiple", "typical_amount",
    "is_violation", "attack_type", "attack_class", "config", "policy",
})
"""Keys a detector must never receive. Each one either states the answer or hands over a
parameter the detector is supposed to infer."""


def redact_for_detector(record: dict[str, Any]) -> dict[str, Any]:
    """
    Strip everything a detector is not entitled to see.

    Applied at the boundary rather than trusted by convention, because the whole failure this
    layer addresses was a parameter leaking from the generator to the detector.
    """
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in DETECTOR_FORBIDDEN}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return clean(record)


__all__ = [
    "DETECTOR_FORBIDDEN",
    "DOMAINS",
    "AgentPolicy",
    "Domain",
    "GeneratorConfig",
    "LimitSource",
    "redact_for_detector",
]


# ── Self-check ───────────────────────────────────────────────────────────────

def self_check(cfg: Optional["GeneratorConfig"] = None) -> dict[str, Any]:
    """
    Verify the property this whole module exists to create: that no fixed amount can
    classify a payment, because the answer depends on whose agent it is.

    Reports, for a spread of amounts, how many agents each is over-limit for. A healthy
    config never answers "all" or "none" across the middle of the range — if it did, a single
    threshold would recover the label and we would be back where we started.
    """
    import statistics

    cfg = cfg or GeneratorConfig()
    rng = random.Random(cfg.seed + 1)
    agents = cfg.sample_agents()

    ambiguity = {}
    for amount in (1.0, 50.0, 500.0, 2000.0, 10_000.0):
        over = sum(1 for a in agents if a.is_over_limit(amount))
        ambiguity[amount] = {"over_limit_for": over, "normal_for": len(agents) - over}

    clean, attack = [], []
    for a in agents:
        d = cfg.domain(a.domain)
        for _ in range(20):
            clean.append(min(d.sample_amount(rng), a.limit))
            attack.append(a.limit * rng.uniform(*cfg.overspend_excess_range))
    lo, hi = min(clean), max(clean)
    overlap = sum(1 for x in attack if lo <= x <= hi) / len(attack)

    return {
        "n_agents": len(agents),
        "limit_min": round(min(a.limit for a in agents), 2),
        "limit_median": round(statistics.median([a.limit for a in agents]), 2),
        "limit_max": round(max(a.limit for a in agents), 2),
        "ambiguity": ambiguity,
        "over_limit_inside_normal_range": round(overlap, 3),
        "unsourced_parameters": cfg.unsourced(),
    }


if __name__ == "__main__":
    import sys

    report = self_check()
    print(f"\n  {report['n_agents']} agents; per-transaction limits "
          f"${report['limit_min']:,.2f} - ${report['limit_max']:,.2f} "
          f"(median ${report['limit_median']:,.2f})\n")
    print("  the same amount is an attack for some agents and normal for others:")
    for amount, row in report["ambiguity"].items():
        print(f"    ${amount:>9,.0f}   over-limit for {row['over_limit_for']:>3}"
              f"   normal for {row['normal_for']:>3}")
    print(f"\n  over-limit payments inside the normal amount range: "
          f"{100 * report['over_limit_inside_normal_range']:.0f}%")
    print("  (a threshold on amount alone cannot classify these)\n")

    if report["unsourced_parameters"]:
        print(f"  {len(report['unsourced_parameters'])} parameter(s) still unsourced — "
              f"not publishable:")
        for u in report["unsourced_parameters"]:
            print(f"    {u}")
        print()
        sys.exit(1)
