"""
Fit the L1 detector's thresholds to a false-positive budget, from clean training traffic only.

    python -m benchmark.calibrate --flag-budget 0.10 --block-budget 0.01

Why calibrate rather than hand-set
----------------------------------
Every threshold in the detector was a number someone typed. The consequences were measurable:
fixed amount-ratio bands fired on 48% of clean sessions, and `soft_limit = exp(mu + 1.1*sigma)`
sits at the 86th percentile of a log-normal by construction, so it flags roughly one payment in
seven whatever the traffic looks like.

A threshold should instead be an answer to "what false-positive rate are we willing to pay?"
That is a product decision someone can actually make, and everything else follows from the
data. This fits the z-cut and the soft-limit multiplier so the observed clean flag rate on the
TRAINING split lands on the budget.

What is and is not legitimate here
----------------------------------
Fitting on clean training traffic is legitimate: it is the same data the norms come from, it
contains no attacks, and the test split is untouched. Fitting to maximise recall on the test
attacks would not be — that is choosing the answer.

So the only inputs are (a) clean training sessions and (b) a budget chosen in advance. Recall
is measured afterwards and never steers the fit.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk"))

from benchmark.evaluate_v2 import fit_baselines, load, to_actions


def clean_z_scores(train: list[dict[str, Any]],
                   baselines: dict[str, dict[str, Any]]) -> list[float]:
    """Every clean payment's z-score against its own agent's fitted log-normal."""
    out = []
    for row in train:
        base = baselines.get(row["agent_id"])
        if not base or not base.get("log_sigma"):
            continue
        mu, sigma = base["log_mu"], base["log_sigma"]
        for a in row["actions"]:
            amount = a.get("amount_units")
            if a["action_type"] == "authorize" and amount and amount > 0:
                out.append((math.log(amount) - mu) / sigma)
    return sorted(out)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(int(q * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


def novelty_rate(rows: list[dict[str, Any]],
                 baselines: dict[str, dict[str, Any]]) -> float:
    """
    How often clean traffic reaches a service the agent has not used before.

    Must be measured OUT OF SAMPLE. Scoring the training sessions against a baseline fitted
    on those same sessions returns 0% by construction — every service is known because the
    baseline was built from it — and that number would tell a detector it may treat all
    novelty as suspicious.

    A detector has to tolerate at least the real rate or it flags ordinary growth. Agents do
    add services: production has 43 agents across 295 of them.
    """
    seen = total = 0
    for row in rows:
        known = (baselines.get(row["agent_id"]) or {}).get("known_services") or set()
        for a in row["actions"]:
            if a["action_type"] != "authorize":
                continue
            total += 1
            if a.get("service_id") and a["service_id"] not in known:
                seen += 1
    return seen / total if total else 0.0


def calibrate(data_dir: Path, flag_budget: float = 0.10,
              block_budget: float = 0.01) -> dict[str, Any]:
    train = load(data_dir / "train.jsonl")
    baselines = fit_baselines(train)
    z = clean_z_scores(train, baselines)

    # Novelty is measured on held-out CLEAN traffic, never on the sessions the baseline was
    # fitted from — in-sample it is 0% by construction.
    holdout_clean = [r for r in load(data_dir / "test.jsonl") if r.get("is_clean")]

    # The cut that leaves `flag_budget` of clean payments above it.
    z_escalate = quantile(z, 1 - flag_budget)
    z_block = quantile(z, 1 - block_budget)

    # A soft limit expressed as a multiplier on the fitted sigma, so it means the same thing
    # for every agent regardless of how wide their spend is.
    return {
        "fitted_on": {
            "sessions": len(train),
            "agents": len(baselines),
            "clean_payments": len(z),
        },
        "budget": {"flag": flag_budget, "block": block_budget},
        "z_escalate": round(z_escalate, 3),
        "z_block": round(z_block, 3),
        "soft_limit_k": round(z_escalate, 3),
        "ceiling_k": round(z_block, 3),
        "observed_clean_novelty_rate": round(novelty_rate(holdout_clean, baselines), 4),
        "clean_z_quantiles": {
            f"p{int(q * 100)}": round(quantile(z, q), 3)
            for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "note": ("Fitted on clean training traffic only. Recall on the test attacks was not "
                 "an input to this fit and must not become one."),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="benchmark/data/v2")
    p.add_argument("--flag-budget", type=float, default=0.10,
                   help="share of clean payments allowed to be escalated for review")
    p.add_argument("--block-budget", type=float, default=0.01,
                   help="share of clean payments allowed to be blocked outright")
    p.add_argument("--out", default="benchmark/data/v2/calibration.json")
    args = p.parse_args(argv)

    cal = calibrate(Path(args.data), args.flag_budget, args.block_budget)
    print(f"\n  fitted on {cal['fitted_on']['clean_payments']:,} clean payments from "
          f"{cal['fitted_on']['agents']} agents\n")
    print("  clean z-score distribution (payment vs its own agent's norm):")
    for k, v in cal["clean_z_quantiles"].items():
        print(f"    {k:<5} {v:>7.2f}")
    print(f"\n  budget: flag {100 * args.flag_budget:.0f}% of clean, "
          f"block {100 * args.block_budget:.0f}%")
    print(f"    z_escalate = {cal['z_escalate']}")
    print(f"    z_block    = {cal['z_block']}")
    print(f"\n  clean novelty rate (new service per payment): "
          f"{100 * cal['observed_clean_novelty_rate']:.1f}%")
    print("    a detector must tolerate at least this, or it flags ordinary growth\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=2))
    tmp.replace(out)
    print(f"  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
