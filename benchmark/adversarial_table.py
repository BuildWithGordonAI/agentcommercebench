"""
Emit the adversarial-evaluation table as LaTeX, straight from the results JSON.

Hand-transcribing numbers into a paper is how a table comes to disagree with the artifact
that produced it. Nothing here is typed twice: run the evaluation, run this, paste the output.

    python -m benchmark.adversarial_table benchmark/results/adversarial_l0_l1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABELS = {
    "honest": r"\emph{clean control}",
    "inflated_price": "Inflated price",
    "typosquat_domain": "Typosquat domain",
    "homoglyph_domain": "Homoglyph domain",
    "payee_swap": "Payee swap",
    "response_injection": "Response injection",
    "adaptive": "Adaptive counterparty",
}


def pct(value) -> str:
    return "---" if value is None else f"{100 * value:.1f}"


def render(summary: dict) -> str:
    per = summary["per_condition"]
    meta = summary.get("meta", {})
    lines = [
        r"\begin{table}[t]",
        r"\caption{Adversarial evaluation with a live agent. A hostile counterparty is a "
        r"\emph{condition}, not a fixed action, so whether an attack produces a settled "
        r"payment depends on the agent. \textbf{Wire} counts sessions in which it did; "
        r"detection rates are conditional on that, so both layers are scored on identical "
        r"sessions and neither is credited with the agent's own refusals. "
        r"\textbf{Flagged} counts a block or an escalation; \textbf{Blocked} counts only "
        r"payments stopped outright, since an escalation is a review rather than a refusal. "
        r"\textbf{Residual} --- settled and unflagged by either layer --- is the only column "
        r"in which money moved undetected.}",
        r"\label{tab:adversarial}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Condition & Wire & L0 & L1 & Flagged & Blocked & Residual \\",
        r"\midrule",
    ]

    def row(label, entry, bold=False):
        name = rf"\textbf{{{label}}}" if bold else label
        # The overall block names the union column `union_given_reached`; per-condition
        # entries call it `either_given_reached`. Same quantity, one accessor.
        union = entry.get("either_given_reached", entry.get("union_given_reached"))
        blocked = entry.get("either_blocked_given_reached",
                            entry.get("union_blocked_given_reached"))
        return (f"{name} & {entry['reached_wire']}/{entry['n']} & "
                f"{pct(entry['l0_given_reached'])} & {pct(entry['l1_given_reached'])} & "
                f"{pct(union)} & {pct(blocked)} & {pct(entry['residual'])} \\\\")

    # Attacks first, control last: the clean row is a reference, and its Residual column
    # reads as "correctly left alone" rather than as money lost.
    for key, label in LABELS.items():
        if key == "honest":
            continue
        entry = per.get(key)
        if entry:
            lines.append(row(label, entry))

    overall = summary.get("overall_attacked")
    if overall:
        lines += [r"\midrule", row("All attacks", overall, bold=True)]

    if per.get("honest"):
        lines += [r"\midrule", row(LABELS["honest"], per["honest"])]

    lines += [r"\bottomrule", r"\end{tabular}"]
    if meta:
        lines.append(
            rf"\\[2pt] {{\footnotesize {meta.get('total_sessions', '?')} sessions, "
            rf"{meta.get('sessions_per_condition', '?')} per condition, seed "
            rf"{meta.get('seed', '?')}; agent: \texttt{{{meta.get('model', '?')}}}; "
            rf"{meta.get('model_calls', '?')} model calls, "
            rf"\${meta.get('cost_usd', 0):.2f}.}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def prose(summary: dict) -> str:
    """The claims the table supports, stated so they can be checked against it."""
    comp = summary.get("complementarity", {})
    overall = summary.get("overall_attacked", {})
    clean = summary.get("clean_control", {})
    out = []
    if comp.get("l0_strictly_better"):
        out.append("L0 strictly better on: " + ", ".join(comp["l0_strictly_better"]))
    if comp.get("l1_strictly_better"):
        out.append("L1 strictly better on: " + ", ".join(comp["l1_strictly_better"]))
    if comp.get("union_beats_both"):
        out.append("Union beats both on: " + ", ".join(comp["union_beats_both"]))
    if overall:
        out.append(
            f"Of {overall['n']} attack sessions, {overall['prevented_by_agent']} were "
            f"prevented by the agent before reaching the wire; of the "
            f"{overall['reached_wire']} that reached it, "
            f"{100 * overall['union_given_reached']:.1f}% were flagged and "
            f"{100 * overall['residual']:.1f}% settled undetected.")
    if clean:
        out.append(
            f"Clean control ({clean['reached_wire']} settling sessions): hard FPR "
            f"L0 {100 * clean['l0_hard_fpr']:.1f}%, L1 {100 * clean['l1_hard_fpr']:.1f}%, "
            f"union {100 * clean['union_hard_fpr']:.1f}%; flag rate (block or escalate) "
            f"union {100 * clean['union_flag_rate']:.1f}%.")
    return "\n".join(f"  - {line}" for line in out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="?",
                   default="benchmark/results/adversarial_l0_l1.json")
    p.add_argument("--out", help="write the LaTeX to this path as well as stdout")
    args = p.parse_args(argv)

    path = Path(args.results)
    if not path.exists():
        print(f"no results at {path}; run benchmark.adversarial_eval first", file=sys.stderr)
        return 2
    summary = json.loads(path.read_text())["summary"]

    latex = render(summary)
    print(latex)
    print("\n% Claims this table supports:")
    print(prose(summary))
    if args.out:
        Path(args.out).write_text(latex + "\n")
        print(f"\n% written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
