"""
Dataset generator for AgentCommerceBench.

Generates deterministic labeled sessions (clean + one per attack scenario)
for every persona. Same seed always produces the same dataset — fully
reproducible without calling any external API.

Output: list[Session] ready for replay.py

Usage:
    from benchmark.generate import build_dataset
    sessions = build_dataset(n_clean=50, seed=42)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from harness.simulate.schema import Session, Persona
from harness.simulate.personas import generate_clean_session
from harness.simulate.injectors import inject, ALL_SCENARIOS

BASE_DATE = datetime(2026, 6, 1, 9, 0, 0)


def build_dataset(
    n_clean: int = 50,
    seed: int = 42,
    n_per_scenario: int = 1,
    personas: list[Persona] = None,
    scenarios: list[str] = None,
    persona_weights: dict = None,
) -> list[Session]:
    """
    Returns a reproducible labeled dataset.

    For each persona:
      - n_clean clean sessions (scaled by persona_weights if provided)
      - n_per_scenario attacked sessions per scenario

    persona_weights: optional {Persona: float} mapping (will be normalized).
    Default is equal weight per persona. Use prod-calibrated weights for
    distribution-matching analysis: {RESEARCH: 0.80, PROCUREMENT: 0.15, TRAVEL: 0.05}.
    """
    if personas is None:
        personas = [Persona.RESEARCH, Persona.PROCUREMENT, Persona.TRAVEL]
    if scenarios is None:
        # Exclude C1/C2 (A2A) by default — requires multi-agent context
        scenarios = [s for s in ALL_SCENARIOS if not s.startswith("C")]

    if persona_weights is None:
        weights = {p: 1.0 for p in personas}
    else:
        weights = {p: persona_weights.get(p, 1.0) for p in personas}
    total_w = sum(weights.values()) or 1.0

    sessions: list[Session] = []

    for persona in personas:
        agent_id = f"agent_{persona.value}_bench"
        n_clean_p = max(1, int(n_clean * weights[persona] / total_w))

        # Clean sessions
        for i in range(n_clean_p):
            s = generate_clean_session(
                persona=persona,
                agent_id=agent_id,
                base_date=BASE_DATE,
                seed=seed,
                session_index=i,
            )
            sessions.append(s)

        # n_per_scenario attacked sessions per scenario, each with a different
        # seed variant so the underlying clean baseline session varies.
        for scenario in scenarios:
            for variant in range(n_per_scenario):
                variant_seed = seed + variant * 100   # 42, 142, 242, …
                base_clean = generate_clean_session(
                    persona=persona,
                    agent_id=agent_id,
                    base_date=BASE_DATE,
                    seed=variant_seed,
                    session_index=9999 + variant,
                )
                try:
                    attacked = inject(base_clean, scenario=scenario, seed=variant_seed)
                    sessions.append(attacked)
                except Exception:
                    pass  # skip if this attack doesn't apply to this persona

    return sessions


def split(
    sessions: list[Session],
    attack_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[Session], list[Session]]:
    """Train / test split preserving attack distribution."""
    import random
    rng = random.Random(seed)
    clean   = [s for s in sessions if s.is_clean]
    attacked = [s for s in sessions if not s.is_clean]
    rng.shuffle(clean)
    rng.shuffle(attacked)
    n_clean_train   = int(len(clean) * (1 - attack_ratio))
    n_attacked_train = int(len(attacked) * (1 - attack_ratio))
    train = clean[:n_clean_train] + attacked[:n_attacked_train]
    test  = clean[n_clean_train:] + attacked[n_attacked_train:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-clean",        type=int, default=100)
    p.add_argument("--n-per-scenario", type=int, default=5)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    sessions = build_dataset(n_clean=args.n_clean, n_per_scenario=args.n_per_scenario, seed=args.seed)
    clean    = [s for s in sessions if s.is_clean]
    attacked = [s for s in sessions if not s.is_clean]
    scenarios = sorted({s.scenario_id for s in attacked})
    print(f"Dataset: {len(sessions)} sessions  (target ~500)")
    print(f"  Clean:    {len(clean)}")
    print(f"  Attacked: {len(attacked)} across {len(scenarios)} scenarios: {scenarios}")
    print(f"  Events:   {sum(len(s.events) for s in sessions)}")
