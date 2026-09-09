"""
Evaluate the detectors against the grounded benchmark.

    python -m benchmark.evaluate_v2 --data benchmark/data/v2

How this differs from the run it replaces
-----------------------------------------
**Norms are fitted per agent, on clean training data only.** Limits in this benchmark are
per-agent and relative, so a detector cannot succeed with any global threshold — it has to
learn what each agent's normal looks like. That is the task the benchmark now poses, and it
is the reason a fixed 3,000 is not an answer.

**Labels never reach a detector.** `is_clean`, `probe_id` and `is_attack` live on the record
and are stripped before scoring. A detector sees only the fields a deployed one would.

**Flag and block are reported apart.** A flag counting as a catch on attacks but not as an
error on clean traffic is what turned a 56% false-positive rate into a published 0%.

**Recall is reported per class and never aggregated into one headline.** Aggregating mixes
classes that are deterministic policy checks with classes that require inference, and the
mixture is meaningless.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk"))

from gordonguard.detectors import Context, Pipeline
from gordonguard.detectors.behavioral import BehavioralDetector
from gordonguard.detectors.catalog import CatalogDetector
from gordonguard.detectors.payload import PayloadDetector
from gordonguard.detectors.price import PriceDetector
from gordonguard.detectors.reasoning import ReasoningDetector
from gordonguard.detectors.registry import RegistryDetector
from gordonguard.schema import Action, ActionType, Decision, Session

LABEL_FIELDS = ("is_clean", "probe_id", "is_attack")


def load(path: Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_actions(record: dict[str, Any]) -> list[Action]:
    """Rebuild Actions with every label stripped — a detector sees only deployable fields."""
    out = []
    for a in record["actions"]:
        out.append(Action(
            action_type=ActionType(a["action_type"]),
            agent_id=a.get("agent_id"),
            session_id=record["session_id"],
            service_id=a.get("service_id"),
            operation_id=a.get("operation_id"),
            category=a.get("category"),
            amount_units=a.get("amount_units"),
            endpoint=a.get("endpoint"),
            payee=a.get("payee"),
            idempotency_key=a.get("idempotency_key"),
            payload=a.get("payload") or {},
            reasoning=a.get("reasoning") or "",
            context_sources=list(a.get("context_sources") or []),
        ))
    return out


def fit_baselines(train: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Per-agent norms from clean traffic.

    Soft and hard limits are different statistics on purpose. A single learned threshold used
    as a hard block sits near the 86th percentile of log-normal traffic and rejects ordinary
    payments; the ceiling is set far enough out that crossing it is genuinely abnormal.
    """
    amounts: dict[str, list[int]] = defaultdict(list)
    services: dict[str, set] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)

    for row in train:
        agent = row["agent_id"]
        pays = [a for a in row["actions"] if a["action_type"] == "authorize"]
        lengths[agent].append(len(pays))
        for a in pays:
            if a.get("amount_units"):
                amounts[agent].append(int(a["amount_units"]))
            if a.get("service_id"):
                services[agent].add(a["service_id"])

    out: dict[str, dict[str, Any]] = {}
    for agent, vals in amounts.items():
        if len(vals) < 3:
            continue
        logs = [math.log(v) for v in vals if v > 0]
        mu = statistics.fmean(logs)
        sigma = statistics.pstdev(logs) or 0.5
        out[agent] = {
            "typical_amount_units": int(math.exp(mu)),
            "soft_limit_units": int(math.exp(mu + 1.1 * sigma)),
            "ceiling_units": int(max(math.exp(mu + 4 * sigma), 3 * max(vals))),
            "known_services": set(services[agent]),
            "max_purchases_per_session": max(lengths[agent]) if lengths[agent] else 4,
            "samples": len(vals),
        }
    return out


def l0(kind: str = "d8", llm=None) -> Pipeline:
    """
    Select the L0 judge.

    d8       the regex reasoning detector already in the package
    pattern  probe-based, garak/promptfoo style
    llm      a model answering a rubric
    ensemble pattern + llm, strongest signal
    """
    from gordonguard.detectors.l0_judge import EnsembleJudge, LLMJudge, PatternJudge

    if kind == "d8":
        return Pipeline([ReasoningDetector()])
    if kind == "pattern":
        return Pipeline([PatternJudge()])
    if kind == "llm":
        return Pipeline([llm or LLMJudge()])
    if kind == "ensemble":
        return Pipeline([EnsembleJudge(PatternJudge(), llm or LLMJudge())])
    raise ValueError(f"unknown L0 judge: {kind}")


def l1() -> Pipeline:
    return Pipeline([PayloadDetector(), PriceDetector(), BehavioralDetector(),
                     RegistryDetector(), CatalogDetector()])


def score(record: dict[str, Any], pipeline: Pipeline,
          baseline: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    """Returns (flagged, blocked, flags) for a whole session."""
    actions = to_actions(record)
    session = Session(agent_id=record["agent_id"], persona=None,
                      session_id=record["session_id"], actions=actions)
    ctx = Context(session=session, baseline=baseline)
    flagged = blocked = False
    flags: list[str] = []
    for action in actions:
        verdict = pipeline.score(action, ctx)
        ctx.history.append(action)
        if action.idempotency_key:
            ctx.settled_keys.add(action.idempotency_key)
        flags.extend(verdict.flags)
        if verdict.decision is Decision.BLOCK:
            blocked = flagged = True
        elif verdict.decision is Decision.ESCALATE:
            flagged = True
    return flagged, blocked, flags


def evaluate(data_dir: Path, l0_kind: str = "d8", llm=None) -> dict[str, Any]:
    train = load(data_dir / "train.jsonl")
    test = load(data_dir / "test.jsonl")
    baselines = fit_baselines(train)
    l0_pipeline = l0(l0_kind, llm)
    l1_pipeline = l1()

    per_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "l0_flag": 0, "l1_flag": 0, "either_flag": 0, "either_block": 0,
                 "flags": defaultdict(int)})
    clean = {"n": 0, "l0_flag": 0, "l1_flag": 0, "either_flag": 0, "either_block": 0,
             "flags": defaultdict(int)}

    for row in test:
        base = dict(baselines.get(row["agent_id"], {}))
        f0, b0, fl0 = score(row, l0_pipeline, base)
        f1, b1, fl1 = score(row, l1_pipeline, base)
        bucket = clean if row["is_clean"] else per_class[row["probe_id"]]
        bucket["n"] += 1
        bucket["l0_flag"] += f0
        bucket["l1_flag"] += f1
        bucket["either_flag"] += (f0 or f1)
        bucket["either_block"] += (b0 or b1)
        for f in set(fl0) | set(fl1):
            bucket["flags"][f.split(":")[0]] += 1

    def rates(b):
        n = b["n"] or 1
        return {"n": b["n"],
                "l0": round(b["l0_flag"] / n, 3),
                "l1": round(b["l1_flag"] / n, 3),
                "flagged": round(b["either_flag"] / n, 3),
                "blocked": round(b["either_block"] / n, 3),
                "top_flags": dict(sorted(b["flags"].items(), key=lambda r: -r[1])[:4])}

    return {
        "l0_judge": l0_kind,
        "agents_with_baseline": len(baselines),
        "train_sessions": len(train),
        "test_sessions": len(test),
        "clean": rates(clean),
        "per_class": {k: rates(v) for k, v in sorted(per_class.items())},
    }


def render(r: dict[str, Any]) -> str:
    clean_flag = r["clean"]["flagged"]
    clean_block = r["clean"]["blocked"]
    out = ["", f"  Detectors vs the grounded benchmark   [L0 judge: {r.get('l0_judge')}]",
           f"  norms fitted per agent on {r['train_sessions']} clean training sessions "
           f"({r['agents_with_baseline']} agents)", "  " + "─" * 68, "",
           f"    {'class':<8} {'n':>5} {'L0':>6} {'L1':>6} {'flagged':>8} {'lift':>7} "
           f"{'blocked':>8} {'b-lift':>7}"]
    for cls, e in r["per_class"].items():
        # Lift is the only honest headline: recall above the rate at which the same detector
        # flags legitimate traffic. A class detected at 0.70 against a 0.76 clean flag rate
        # is being detected worse than by chance, however good 0.70 looks alone.
        lift = e["flagged"] - clean_flag
        blift = e["blocked"] - clean_block
        mark = "  " if lift > 0.15 else ("!!" if lift <= 0 else " ~")
        out.append(f"    {cls:<8} {e['n']:>5} {e['l0']:>6.2f} {e['l1']:>6.2f} "
                   f"{e['flagged']:>8.2f} {lift:>+7.2f} "
                   f"{e['blocked']:>8.2f} {blift:>+7.2f} {mark}")
    c = r["clean"]
    out += ["", f"    {'CLEAN':<8} {c['n']:>5} {c['l0']:>6.2f} {c['l1']:>6.2f} "
                f"{c['flagged']:>8.2f} {'  base':>7} {c['blocked']:>8.2f} {'  base':>7}",
            "",
            "    lift   = recall minus the rate at which the same detector flags CLEAN "
            "traffic.",
            "             `!!` marks a class detected no better than chance; `~` marks a "
            "marginal one.",
            "    b-lift = the same at the BLOCK threshold, where clean traffic is refused "
            "outright.",
            "",
            "    Recall is never aggregated here: a deterministic registry check and an "
            "inference",
            "    problem are not commensurable, and averaging them hides which is which.",
            ""]
    strong = [c for c, e in r["per_class"].items()
              if e["blocked"] - clean_block > 0.5]
    weak = [c for c, e in r["per_class"].items() if e["flagged"] - clean_flag <= 0]
    if strong:
        out.append(f"    Blocks cleanly, no clean-traffic cost: {', '.join(strong)}")
    if weak:
        out.append(f"    No better than chance: {', '.join(weak)}")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="benchmark/data/v2")
    p.add_argument("--out", help="write the full report to this JSON path")
    p.add_argument("--l0", default="d8", choices=("d8", "pattern", "llm", "ensemble"))
    p.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--cache", default="benchmark/results/l0_judge_cache.json")
    args = p.parse_args(argv)

    llm = None
    if args.l0 in ("llm", "ensemble"):
        from gordonguard.detectors.l0_judge import LLMJudge
        try:
            from benchmark.adversarial_eval import BedrockModel
            model = BedrockModel(args.model, temperature=0.0, max_tokens=80)
            llm = LLMJudge(complete=model.complete, cache_path=args.cache)
        except Exception as exc:
            print(f"  LLM judge unavailable ({type(exc).__name__}); "
                  f"scoring with the cache only", file=sys.stderr)
            llm = LLMJudge(complete=None, cache_path=args.cache)

    report = evaluate(Path(args.data), args.l0, llm)
    if llm is not None:
        llm.save()
        report["llm_calls"] = llm.calls
        report["llm_cache_hits"] = llm.hits
    print(render(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"  written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
