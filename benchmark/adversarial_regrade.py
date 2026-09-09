"""
Re-score a saved adversarial run against the current detectors, without spending anything.

The runner serialises every settled action, so a detector change can be re-evaluated against
the exact sessions a live model produced. A full run costs about $1 and forty minutes; this
costs neither, which means a detector tweak can be checked against real agent behaviour
rather than only against unit tests.

    python -m benchmark.adversarial_regrade benchmark/results/adversarial_l0_l1.json

Writes the updated verdicts back in place, then print the ground-truth summary with
`benchmark.adversarial_rescore`.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk"))

from gordonguard.detectors import Context
from gordonguard.schema import Action, ActionType, Decision

from .adversarial_eval import (
    catalog_registry,
    l0_pipeline,
    l1_pipeline,
    write_json,
)


def rebuild_actions(session: dict) -> list[Action]:
    return [
        Action(
            action_type=ActionType.AUTHORIZE,
            service_id=a.get("service_id"),
            amount_units=a.get("amount_units"),
            vendor=a.get("vendor"),
            endpoint=a.get("endpoint"),
            payee=a.get("payee"),
            reasoning=a.get("reasoning") or "",
            context_sources=list(a.get("context_sources") or []),
        )
        for a in session.get("actions_detail", [])
    ]


def regrade(payload: dict) -> dict:
    stored = payload["summary"]["meta"].get("fitted_baseline", {})
    # Copy: the stored baseline is part of the results file, and the pipelines want a set
    # for known_services. Mutating it in place makes the file unserialisable.
    baseline = copy.deepcopy(stored)
    if isinstance(baseline.get("known_services"), list):
        baseline["known_services"] = set(baseline["known_services"])
    baseline["catalog"] = catalog_registry()

    for session in payload["sessions"]:
        actions = rebuild_actions(session)
        for pipeline, flagged, blocked, flags_key in (
            (l0_pipeline(), "l0_detected", "l0_blocked", "l0_flags"),
            (l1_pipeline(), "l1_detected", "l1_blocked", "l1_flags"),
        ):
            ctx = Context(baseline=baseline)
            session[flagged] = session[blocked] = False
            session[flags_key] = []
            for action in actions:
                verdict = pipeline.score(action, ctx)
                ctx.history.append(action)
                session[flags_key].extend(verdict.flags)
                if verdict.decision is Decision.BLOCK:
                    session[blocked] = True
                    session[flagged] = True
                elif verdict.decision is Decision.ESCALATE:
                    session[flagged] = True
    payload["summary"]["meta"]["regraded"] = True
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="?",
                   default="benchmark/results/adversarial_l0_l1.json")
    args = p.parse_args(argv)

    path = Path(args.results)
    payload = json.loads(path.read_text())
    if not any("actions_detail" in s for s in payload["sessions"]):
        print("this run predates action serialisation; re-run the evaluation",
              file=sys.stderr)
        return 2

    write_json(path, regrade(payload))
    print(f"  regraded {len(payload['sessions'])} sessions against the current detectors "
          f"({path})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
