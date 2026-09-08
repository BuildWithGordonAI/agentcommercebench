"""
τ-bench OOD False-Positive Rate Test.

Maps τ-bench retail ground-truth tool-call sequences to ConsequentialActions
and runs the ACP M-check (Commerce State Automaton DFA) on them.

The goal: legitimate retail agent trajectories should NOT trigger M-violations.
A non-zero FPR here would mean ACP is over-flagging real commerce flows.

Expected outcome: FPR ≈ 0% — all τ-bench ground truth tasks are M-compliant
because they represent correct retail agent behaviour, not protocol violations.

Usage:
    python -m benchmark.taubench_eval                # uses bundled τ-bench tasks
    python -m benchmark.taubench_eval --split test   # or train/dev
"""
from __future__ import annotations
import argparse, importlib, json, sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, str(Path("/tmp/tau-bench")))

from ach.actions.base import ConsequentialAction, ActionType, Reversibility

# ── Commerce State Automaton M ────────────────────────────────────────────────
# Same DFA as property_eval.py — copy kept local so this file is self-contained.

_M_NEXT: dict[str, set[str]] = {
    "start":   {"find"},
    "find":    {"find", "quote"},
    "quote":   {"reserve", "commit"},
    "reserve": {"commit", "void"},
    "commit":  {"settle", "void"},
    "settle":  {"end"},
    "void":    {"end"},
    "end":     set(),
}

# ── τ-bench tool → ACP ActionType mapping ─────────────────────────────────────
# τ-bench retail tools mapped to ACP action types.
# Discovery / read-only tools → FIND (no funds at risk).
# Modification tools → COMMIT (funds or order state changes).
# Cancellation → VOID.

_TOOL_TO_ACTION: dict[str, ActionType] = {
    # discovery
    "find_user_id_by_name_zip":     ActionType.FIND,
    "find_user_id_by_email":        ActionType.FIND,
    "get_user_details":             ActionType.FIND,
    "get_order_details":            ActionType.FIND,
    "get_product_details":          ActionType.FIND,
    "list_all_product_types":       ActionType.FIND,
    "calculate":                    ActionType.FIND,
    "think":                        ActionType.FIND,
    "transfer_to_human_agents":     ActionType.FIND,
    # price / quote
    "get_service":                  ActionType.QUOTE,
    # consequential modifications (financial state change)
    "exchange_delivered_order_items": ActionType.COMMIT,
    "return_delivered_order_items":   ActionType.COMMIT,
    "modify_pending_order_items":     ActionType.COMMIT,
    "modify_pending_order_address":   ActionType.COMMIT,
    "modify_pending_order_payment":   ActionType.COMMIT,
    "modify_user_address":            ActionType.COMMIT,
    # cancellation / void
    "cancel_pending_order":         ActionType.VOID,
}

_REVERSI = {
    ActionType.FIND:    Reversibility.FULL,
    ActionType.QUOTE:   Reversibility.FULL,
    ActionType.RESERVE: Reversibility.PARTIAL,
    ActionType.COMMIT:  Reversibility.FULL,
    ActionType.SETTLE:  Reversibility.NONE,
    ActionType.VOID:    Reversibility.FULL,
}


def _tau_action_to_acp(name: str, step: int) -> ConsequentialAction | None:
    """Convert a τ-bench Action to a ConsequentialAction, or None if unknown."""
    at = _TOOL_TO_ACTION.get(name)
    if at is None:
        return None
    return ConsequentialAction(
        at, Decimal(0), "USD",
        "tau-bench-retail", "tau-bench-retail", "5999",
        "retail", _REVERSI[at], f"tau:{step}",
        {"tool": name},
    )


def _m_first_violation(actions: list[ConsequentialAction]) -> int | None:
    """Return index of first M-violating action, or None if trajectory is valid."""
    state = "start"
    for i, a in enumerate(actions):
        typ = a.action_type.value.lower()
        if typ not in _M_NEXT.get(state, set()):
            return i
        state = typ
    return None


def _load_tau_tasks(split: str = "test"):
    """Import τ-bench task list for the retail environment."""
    mod_map = {
        "test":  "tau_bench.envs.retail.tasks_test",
        "train": "tau_bench.envs.retail.tasks_train",
        "dev":   "tau_bench.envs.retail.tasks_dev",
    }
    mod_name = mod_map.get(split)
    if mod_name is None:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(mod_map)}")
    mod = importlib.import_module(mod_name)
    attr = f"TASKS_{split.upper()}"
    return getattr(mod, attr)


_SERVICE_COMMIT_TOOLS = {
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}

_M_PURCHASE_NEXT: dict[str, set[str]] = {
    # Extended M that allows FIND→COMMIT (service-management shortcut)
    "start":   {"find", "commit"},   # service flows may go directly to COMMIT
    "find":    {"find", "quote", "commit"},  # after browse, can quote OR service-commit
    "quote":   {"reserve", "commit"},
    "reserve": {"commit", "void"},
    "commit":  {"commit", "settle", "void", "find"},  # multi-item service ops
    "settle":  {"end"},
    "void":    {"end"},
    "end":     set(),
}


def _m_first_violation_extended(actions: list[ConsequentialAction]) -> int | None:
    """M-check with service-management extensions (allows FIND→COMMIT shortcut)."""
    state = "start"
    for i, a in enumerate(actions):
        typ = a.action_type.value.lower()
        if typ not in _M_PURCHASE_NEXT.get(state, set()):
            return i
        state = typ
    return None


def run(split: str = "test") -> dict:
    tasks = _load_tau_tasks(split)

    results = []
    n_m_violations_strict = 0   # original M (purchase only)
    n_m_violations_extended = 0  # M' (purchase + service-management)
    n_skipped_tools = 0

    # Count task classes
    n_purchase_class = 0   # starts with FIND before any COMMIT
    n_service_class = 0    # starts directly with COMMIT (service management)

    for task in tasks:
        acp_actions: list[ConsequentialAction] = []
        skipped = []
        first_commit_step = None
        first_find_step = None

        for step, action in enumerate(task.actions):
            ca = _tau_action_to_acp(action.name, step)
            if ca is None:
                skipped.append(action.name)
                n_skipped_tools += 1
            else:
                if ca.action_type == ActionType.COMMIT and first_commit_step is None:
                    first_commit_step = step
                if ca.action_type == ActionType.FIND and first_find_step is None:
                    first_find_step = step
                acp_actions.append(ca)

        task_class = "service" if (
            first_commit_step is not None and
            (first_find_step is None or first_commit_step < first_find_step + 1)
        ) else "purchase"

        if task_class == "service":
            n_service_class += 1
        else:
            n_purchase_class += 1

        # Strict M-check (purchase automaton only)
        vi_strict = _m_first_violation(acp_actions)
        if vi_strict is not None:
            n_m_violations_strict += 1

        # Extended M'-check (service + purchase)
        vi_ext = _m_first_violation_extended(acp_actions)
        if vi_ext is not None:
            n_m_violations_extended += 1

        results.append({
            "user_id":            task.user_id,
            "task_class":         task_class,
            "n_actions":          len(task.actions),
            "n_mapped":           len(acp_actions),
            "skipped_tools":      skipped,
            "m_strict_violation": vi_strict is not None,
            "m_ext_violation":    vi_ext is not None,
            "violation_step":     vi_strict,
        })

    n_total = len(tasks)
    return {
        "split":                    split,
        "n_tasks":                  n_total,
        "n_purchase_class":         n_purchase_class,
        "n_service_class":          n_service_class,
        "n_m_violations_strict":    n_m_violations_strict,
        "n_m_violations_extended":  n_m_violations_extended,
        "fpr_strict":               n_m_violations_strict  / n_total if n_total else 0.0,
        "fpr_extended":             n_m_violations_extended / n_total if n_total else 0.0,
        "n_skipped_tools":          n_skipped_tools,
        "scope_finding": (
            "τ-bench retail (service management) operates outside ACP M's purchase "
            "automaton. Extending M with FIND→COMMIT shortcut (M') reduces OOD FPR "
            f"from {n_m_violations_strict/n_total:.1%} to "
            f"{n_m_violations_extended/n_total:.1%}."
        ),
        "tasks":                    results,
    }


def main():
    p = argparse.ArgumentParser(description="τ-bench OOD False-Positive Rate test")
    p.add_argument("--split", default="test", choices=["test", "train", "dev"])
    p.add_argument("--out",   default="benchmark/results/taubench_eval.json")
    args = p.parse_args()

    result = run(args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nτ-bench OOD Scope Analysis  (split={args.split})\n")
    print(f"  Tasks evaluated          : {result['n_tasks']}")
    print(f"  Purchase-class tasks     : {result['n_purchase_class']}")
    print(f"  Service-class tasks      : {result['n_service_class']}")
    print(f"  Unmapped tool calls      : {result['n_skipped_tools']}")
    print()
    print(f"  M  violations (strict)   : {result['n_m_violations_strict']}  "
          f"(FPR = {result['fpr_strict']:.1%})")
    print(f"  M' violations (extended) : {result['n_m_violations_extended']}  "
          f"(FPR = {result['fpr_extended']:.1%})")
    print()
    print(f"  Scope finding: {result['scope_finding']}")
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
