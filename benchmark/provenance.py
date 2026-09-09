"""
Provenance and confounding checks for every constant that affects a benchmark result.

The rule this enforces
----------------------
A detector is blind. It may only get its parameters from two places:

    LEARNED   fitted from training data the detector is allowed to see
    HUMAN     domain knowledge, a published standard, a policy someone set

Anything else is **confounded**: a value that exists because of how the benchmark was built.
A threshold nobody can trace is a cheat even when it happens to work, and a threshold copied
from the generator is a cheat that cannot fail.

The classic shape, from this repo's own history: the generator drew fraudulent amounts from
`uniform(3010, 4500)` and the rule blocked above `3000`, with clean traffic topping out at
`2800`. Detection was perfect and false positives were zero — not because the detector was
good, but because the two populations never overlapped and the threshold sat in the gap.

Two checks
----------
`check_shared_constants`  mechanical. Any literal appearing in both a generator module and a
detector module is reported. Mechanical checks find copied values; they cannot find a value
that was *chosen* with knowledge of the other side, which is why the ledger exists too.

`check_ledger`  every constant declared in PROVENANCE must have a source, a justification and,
where it interacts with a distribution, evidence that the supports overlap. A constant that is
not in the ledger is itself a finding.

    python -m benchmark.provenance
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent

GENERATOR_FILES = ["benchmark/synthetic.py"]
DETECTOR_FILES = [
    "benchmark/behavioral_ml.py",
    "benchmark/acp_baselines.py",
    "ach/verifiers/local_rules.py",
    "ach/verifiers/uncertainty.py",
]

# Values so common they carry no information about coupling.
BORING = {0, 1, 2, 3, -1, 10, 100, 1000, 0.0, 0.5, 1.0, 2.0, 100.0}


@dataclass
class Constant:
    """One parameter that can move a benchmark number."""

    name: str
    value: Any
    source: str          # LEARNED | HUMAN | CONFOUNDED | UNTRACED
    where: str
    justification: str
    overlap_evidence: Optional[str] = None
    """For a threshold sitting between two distributions: proof their supports overlap. A
    threshold in a gap cannot be wrong, so it measures nothing."""

    @property
    def ok(self) -> bool:
        return self.source in {"LEARNED", "HUMAN"}


# ── The ledger ───────────────────────────────────────────────────────────────
#
# Every entry is a claim that can be checked. Where the claim is that a value is confounded,
# that is a finding, not an excuse — the list below is the current state, not the target.

PROVENANCE: list[Constant] = [
    Constant(
        name="per_transaction_limit",
        value=3000,
        source="CONFOUNDED",
        where="benchmark/synthetic.py:110 (wallet) / ach/verifiers/local_rules.py:59 (rule)",
        justification=(
            "Nothing derives this. It is a round number chosen when the generator was "
            "written, and the B8 attack builder then draws uniform(3010, 4500) — immediately "
            "above it. Clean traffic tops out at 2800 (_AMOUNT_RANGES_HIGHVAL['travel']), so "
            "the threshold sits in an empty gap between the two populations."
        ),
        overlap_evidence=(
            "NONE. clean max 2800 < 3010 min attack. The supports are disjoint, so no clean "
            "session can trigger the rule and no attack session can evade it."
        ),
    ),
    Constant(
        name="velocity_block_threshold",
        value=8,
        source="CONFOUNDED",
        where="ach/verifiers/local_rules.py:22 vs benchmark/synthetic.py:220",
        justification=(
            "The rule blocks at 8 commits; the B2 builder draws randint(9, 15). The threshold "
            "is one below the attack's minimum. No clean session has more than 1 commit."
        ),
        overlap_evidence="NONE. clean commit count is a point mass at 1; attacks are 9-15.",
    ),
    Constant(
        name="injection_keywords",
        value="_INJECTION_KEYWORDS",
        source="CONFOUNDED",
        where="benchmark/behavioral_ml.py:44 vs benchmark/synthetic.py:294",
        justification=(
            "The detector's keyword list is a superset of the substrings of the four strings "
            "the generator injects. 40/40 generated payloads match. The test set was written "
            "against the answer key."
        ),
        overlap_evidence="NONE. No clean session contains any keyword.",
    ),
    Constant(
        name="mcc_allowlist",
        value="_MCC_OK",
        source="CONFOUNDED",
        where="benchmark/synthetic.py:40 (both wallet policy and attack construction)",
        justification=(
            "The generator builds B7 attacks by drawing from _MCC_BAD_OPTIONS, constructed to "
            "be disjoint from _MCC_OK, and the wallet policy is _MCC_OK. Same object on both "
            "sides. As a policy conformance check this is legitimate; as a detection result "
            "it is circular."
        ),
        overlap_evidence="NONE, by construction of _MCC_BAD_OPTIONS.",
    ),
    Constant(
        name="threshold_sigmas",
        value=1.1,
        source="UNTRACED",
        where="benchmark/behavioral_ml.py:125",
        justification=(
            "Fitted as mu + k*sigma on clean training scores, which is the right shape — but k "
            "is 1.1 while the module docstring says 1.5 and predicts a 6-10% false-positive "
            "rate. The measured rate is 56%. Whatever produced 1.1, it is not what is "
            "documented, so it cannot be audited."
        ),
    ),
    Constant(
        name="clean_amount_ranges",
        value="_AMOUNT_RANGES",
        source="HUMAN",
        where="benchmark/synthetic.py:79",
        justification=(
            "Per-persona spend ranges chosen as plausible for the personas. Legitimate as a "
            "modelling choice; the problem is not this constant but that attack builders draw "
            "from different ranges, making the attack detectable by amount alone."
        ),
        overlap_evidence=(
            "PARTIAL. Attacks unrelated to money (B4, B5, B7s) draw from ranges whose means "
            "sit well above the clean per-persona means, so they are separable by amount."
        ),
    ),
    Constant(
        name="training_seed",
        value=7,
        source="HUMAN",
        where="benchmark/behavioral_ml.py:133",
        justification=(
            "Deliberately different from the evaluation seed (42) so training and test draws "
            "are independent. Verified: n_per_attack=0, so norms are fitted on clean data "
            "only. This one is correct."
        ),
    ),
    Constant(
        name="zero_variance_fallback",
        value=1.0,
        source="CONFOUNDED",
        where="benchmark/behavioral_ml.py:168",
        justification=(
            "When a feature has zero variance in training the code substitutes sigma=1.0. "
            "Clean commit count is constant at 1, so this manufactures z=(9-1)/1.0=8 for a "
            "velocity attack. The detector appears to have learned a distribution that does "
            "not exist. A zero-variance feature carries no information and must be dropped."
        ),
    ),
]


# ── Mechanical check ─────────────────────────────────────────────────────────

def literals(path: Path) -> dict[Any, list[int]]:
    """Every numeric and string literal in a module, with the lines it appears on."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return {}
    found: dict[Any, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            value = node.value
            if isinstance(value, str) and (len(value) < 4 or len(value) > 80):
                continue
            if isinstance(value, (int, float)) and value in BORING:
                continue
            found.setdefault(value, []).append(node.lineno)
    return found


def check_shared_constants(root: Path = ROOT) -> list[dict[str, Any]]:
    """Literals appearing on both sides of the generator/detector boundary."""
    gen: dict[Any, list[str]] = {}
    for name in GENERATOR_FILES:
        for value, lines in literals(root / name).items():
            gen.setdefault(value, []).extend(f"{name}:{n}" for n in lines)

    findings = []
    for name in DETECTOR_FILES:
        path = root / name
        if not path.exists():
            continue
        for value, lines in literals(path).items():
            if value in gen:
                findings.append({
                    "value": value,
                    "generator": sorted(set(gen[value]))[:4],
                    "detector": [f"{name}:{n}" for n in lines][:4],
                })
    return sorted(findings, key=lambda f: str(f["value"]))


def check_ledger() -> list[Constant]:
    """Constants whose provenance is not LEARNED or HUMAN."""
    return [c for c in PROVENANCE if not c.ok]


def render() -> str:
    out: list[str] = ["", "  Parameter provenance", "  " + "─" * 70]
    for group in ("CONFOUNDED", "UNTRACED", "HUMAN", "LEARNED"):
        rows = [c for c in PROVENANCE if c.source == group]
        if not rows:
            continue
        out.append(f"\n  {group}  ({len(rows)})")
        for c in rows:
            out.append(f"    {c.name} = {c.value}")
            out.append(f"      {c.where}")
            out.append(f"      {c.justification}")
            if c.overlap_evidence:
                out.append(f"      support overlap: {c.overlap_evidence}")

    shared = check_shared_constants()
    out += ["", "  Literals shared across the generator/detector boundary", "  " + "─" * 70]
    if not shared:
        out.append("    none")
    for f in shared:
        out.append(f"    {f['value']!r}")
        out.append(f"      generator: {', '.join(f['generator'])}")
        out.append(f"      detector:  {', '.join(f['detector'])}")

    bad = check_ledger()
    out += ["", "  " + "─" * 70,
            f"  {len(bad)} of {len(PROVENANCE)} declared constants are not traceable to "
            f"training data or human judgement.",
            f"  {len(shared)} literal(s) appear on both sides of the boundary.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if anything is confounded or untraced (for CI)")
    args = p.parse_args(argv)

    if args.json:
        print(json.dumps({
            "constants": [vars(c) for c in PROVENANCE],
            "shared_literals": check_shared_constants(),
        }, indent=2, default=str))
    else:
        print(render())

    if args.strict and (check_ledger() or check_shared_constants()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
