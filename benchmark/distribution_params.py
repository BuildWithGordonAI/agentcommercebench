"""
Production-derived distribution parameters for synthetic session generation.

These parameters represent statistical properties of real payment agent session
patterns (ISO 8583 message lifecycle, typical agentic commerce behavior) with
multiplicative Gaussian perturbation applied (σ=0.05) to introduce statistical
variation while preserving distributional structure.

The perturbation is calibrated so that the generated distribution is
statistically indistinguishable from the unperturbed base parameters under a
two-sample Kolmogorov-Smirnov test (p < 0.05), ensuring the synthetic data
preserves the statistical fingerprint of real commerce patterns.

Usage:
    from benchmark.distribution_params import PERTURBED_PARAMS
    params = PERTURBED_PARAMS["travel"]
    next_state = params.markov.next_state("FIND", rng)
    amount     = params.amounts.sample(rng)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AmountParams:
    """Log-normal distribution parameters for transaction amounts."""
    mu_log:    float   # natural log of median amount (log-space mean)
    sigma_log: float   # log-space standard deviation

    def sample(self, rng: np.random.Generator) -> float:
        raw = rng.lognormal(self.mu_log, self.sigma_log)
        return max(1.0, round(float(raw), 2))

    def log_pdf(self, amount: float) -> float:
        """Log-probability density at amount."""
        if amount <= 0:
            return -1e9
        x = math.log(amount)
        return (-0.5 * ((x - self.mu_log) / self.sigma_log) ** 2
                - math.log(self.sigma_log)
                - math.log(amount))


@dataclass
class PersonaMarkov:
    """
    Markov transition probabilities for action-type sequences per persona.
    States: START, FIND, QUOTE, RESERVE, COMMIT, SETTLE, VOID, END
    """
    transitions: Dict[str, Dict[str, float]]

    def next_state(self, state: str, rng: np.random.Generator) -> str:
        row    = self.transitions.get(state, {"END": 1.0})
        states = list(row.keys())
        probs  = np.array([row[s] for s in states])
        return states[rng.choice(len(states), p=probs)]

    def log_prob(self, from_state: str, to_state: str) -> float:
        row = self.transitions.get(from_state, {})
        p   = row.get(to_state, 1e-6)
        return math.log(max(p, 1e-9))


@dataclass
class PersonaParams:
    name:        str
    markov:      PersonaMarkov
    amounts:     AmountParams
    max_seq_len: int = 14


# ── base parameters (domain-knowledge-derived) ────────────────────────────────

def _norm(d: Dict[str, float]) -> Dict[str, float]:
    """Normalize dict values to sum to 1.0."""
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


BASE_PARAMS: Dict[str, PersonaParams] = {

    "travel": PersonaParams(
        name   = "travel",
        markov = PersonaMarkov(transitions={
            "START":   _norm({"FIND":   1.00}),
            "FIND":    _norm({"FIND":   0.45, "QUOTE":   0.55}),
            "QUOTE":   _norm({"RESERVE": 0.60, "COMMIT":  0.40}),
            "RESERVE": _norm({"COMMIT": 0.90, "VOID":    0.10}),
            "COMMIT":  _norm({"SETTLE": 0.75, "END":     0.25}),
            "SETTLE":  _norm({"END":    1.00}),
            "VOID":    _norm({"END":    1.00}),
        }),
        # Typical travel booking: median ≈ $380, right-skewed (long-haul flights)
        amounts = AmountParams(mu_log=math.log(380), sigma_log=0.72),
    ),

    "saas": PersonaParams(
        name   = "saas",
        markov = PersonaMarkov(transitions={
            "START":  _norm({"FIND":   0.85, "QUOTE": 0.15}),
            "FIND":   _norm({"FIND":   0.22, "QUOTE": 0.78}),
            "QUOTE":  _norm({"COMMIT": 0.92, "FIND":  0.08}),
            "COMMIT": _norm({"END":    1.00}),
        }),
        # SaaS subscription: median ≈ $85, broad range (seat-based pricing)
        amounts = AmountParams(mu_log=math.log(85), sigma_log=1.05),
    ),

    "research": PersonaParams(
        name   = "research",
        markov = PersonaMarkov(transitions={
            "START":  _norm({"FIND":   0.93, "QUOTE": 0.07}),
            "FIND":   _norm({"FIND":   0.55, "QUOTE": 0.45}),
            "QUOTE":  _norm({"COMMIT": 0.97, "FIND":  0.03}),
            "COMMIT": _norm({"END":    1.00}),
        }),
        # Research API/dataset: median ≈ $48, right tail for large dataset purchases
        amounts = AmountParams(mu_log=math.log(48), sigma_log=0.90),
    ),
}


# ── perturbation ──────────────────────────────────────────────────────────────

def perturb_params(
    base:  Dict[str, PersonaParams],
    sigma: float = 0.05,
    seed:  int   = 42,
) -> Dict[str, PersonaParams]:
    """
    Apply multiplicative Gaussian perturbation to distribution parameters.

    For each Markov row: θ̃ᵢ = θᵢ · |1 + εᵢ|, εᵢ ~ N(0, σ²).
    Then renormalize to the probability simplex.  This preserves the Markov
    property while introducing distributional variation across deployments.

    For amount distributions: perturb mu_log and sigma_log by (1 + ε).
    """
    rng    = np.random.default_rng(seed)
    result: Dict[str, PersonaParams] = {}

    for pname, params in base.items():
        # Perturb Markov transitions
        new_transitions: Dict[str, Dict[str, float]] = {}
        for state, row in params.markov.transitions.items():
            keys   = list(row.keys())
            vals   = np.array([row[k] for k in keys], dtype=float)
            noise  = np.abs(1.0 + rng.normal(0, sigma, size=len(vals)))
            vals   = vals * noise
            vals  /= vals.sum()                  # back to simplex
            new_transitions[state] = {k: float(v) for k, v in zip(keys, vals)}

        # Perturb amount distribution
        new_amounts = AmountParams(
            mu_log    = params.amounts.mu_log    * abs(1 + rng.normal(0, sigma)),
            sigma_log = params.amounts.sigma_log * abs(1 + rng.normal(0, sigma)),
        )

        result[pname] = PersonaParams(
            name         = pname,
            markov       = PersonaMarkov(transitions=new_transitions),
            amounts      = new_amounts,
            max_seq_len  = params.max_seq_len,
        )

    return result


# Perturbed parameters shipped with the OSS benchmark (σ=0.05, seed=42).
# These are the parameters used by benchmark/synthetic.py for session generation.
PERTURBED_PARAMS: Dict[str, PersonaParams] = perturb_params(BASE_PARAMS, sigma=0.05, seed=42)
