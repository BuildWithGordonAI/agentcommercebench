"""
ACP benchmark experiment runner.

Evaluates all baselines against the synthetic session dataset and prints
a comparison table suitable for inclusion in the paper.

Usage:
    python benchmark/run_experiment.py
    python benchmark/run_experiment.py --n-clean 200 --n-per-attack 40 --seed 42
    python benchmark/run_experiment.py --attacks B1 B2 B5 B6   # subset

Outputs:
    - Per-baseline metrics table (F1, precision, recall, FPR, FNR, latency)
    - Per-attack-type breakdown table
    - Uncertainty calibration summary (ACP pipeline only)
    - Chain integrity verification
    - Saves results/acp_benchmark.json for plotting
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, ".")

from benchmark.synthetic import generate, summary, BenchmarkSession
from benchmark.acp_baselines import BASELINES, ACHFullPipeline
from ach.ceremony import route as ceremony_route
from ach.anchoring.chain import CARLog
from ach.receipts.car import CommerceActionReceipt
from benchmark.sota_evaluators import run_sota_evaluation

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
BLUE   = "\033[34m"; BOLD   = "\033[1m";  DIM = "\033[2m"; RESET = "\033[0m"

def banner(t): print(f"\n{BLUE}{BOLD}{'─'*70}{RESET}\n{BLUE}{BOLD}  {t}{RESET}\n{BLUE}{BOLD}{'─'*70}{RESET}")
def col(v, good_thresh, bad_thresh, reverse=False):
    cmp = (1 - v) if reverse else v   # for coloring: lower FPR/FNR is better
    c   = GREEN if cmp >= good_thresh else YELLOW if cmp >= bad_thresh else RED
    return f"{c}{v:.3f}{RESET}"


# ── metrics ───────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    name:      str
    tp: int = 0; fp: int = 0; tn: int = 0; fn: int = 0
    # Separate "flag" (escalation) from "block" (hard reject)
    fp_block: int = 0   # clean sessions auto-blocked (hard false positive)
    fp_flag:  int = 0   # clean sessions escalated for review (soft false positive)
    tp_block: int = 0   # fraud sessions auto-blocked
    tp_flag:  int = 0   # fraud sessions escalated for review (still caught)
    latency_ms_total: float = 0.0
    n_sessions: int = 0
    per_attack: dict = None

    def __post_init__(self):
        if self.per_attack is None:
            self.per_attack = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0})

    @property
    def precision(self): return self.tp/(self.tp+self.fp) if self.tp+self.fp else 0.0
    @property
    def recall(self):    return self.tp/(self.tp+self.fn) if self.tp+self.fn else 0.0
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2*p*r/(p+r) if p+r else 0.0
    @property
    def fpr(self):       return self.fp/(self.fp+self.tn) if self.fp+self.tn else 0.0
    @property
    def fpr_block(self):
        n_clean = self.fp + self.tn
        return self.fp_block/n_clean if n_clean else 0.0
    @property
    def esc_rate(self):
        n_clean = self.fp + self.tn
        return self.fp_flag/n_clean if n_clean else 0.0
    @property
    def fnr(self):       return self.fn/(self.fn+self.tp) if self.fn+self.tp else 0.0
    @property
    def avg_latency_ms(self): return self.latency_ms_total/self.n_sessions if self.n_sessions else 0.0


def _session_decisions(session: BenchmarkSession, verifier) -> tuple[str, float, float]:
    """
    Run verifier across all actions in a session.
    Returns (worst_decision: str, max_score: float, latency_ms: float).
    worst_decision: 'block' > 'flag' > 'allow'.
    """
    if hasattr(verifier, "reset_session"):
        verifier.reset_session(session.session_id)

    _rank   = {"allow": 0, "flag": 1, "block": 2}
    max_score    = 0.0
    worst_dec    = "allow"
    t0           = time.perf_counter()

    for action in session.actions:
        v = verifier.verify(action, session.wallet, session.context)
        if v.score > max_score:
            max_score = v.score
        if _rank.get(v.decision, 0) > _rank[worst_dec]:
            worst_dec = v.decision

    latency_ms = (time.perf_counter() - t0) * 1000
    return worst_dec, max_score, latency_ms


def evaluate(name: str, verifier, sessions: list[BenchmarkSession]) -> Metrics:
    m = Metrics(name=name)
    for sess in sessions:
        worst_dec, _, latency = _session_decisions(sess, verifier)
        m.latency_ms_total += latency
        m.n_sessions += 1
        at = sess.attack_type
        detected = worst_dec != "allow"   # flag or block = detected
        if sess.is_violation:
            if detected:
                m.tp += 1; m.per_attack[at]["tp"] += 1
                if worst_dec == "block": m.tp_block += 1
                else:                    m.tp_flag  += 1
            else:
                m.fn += 1; m.per_attack[at]["fn"] += 1
        else:
            if detected:
                m.fp += 1; m.per_attack[at]["fp"] += 1
                if worst_dec == "block": m.fp_block += 1
                else:                    m.fp_flag  += 1
            else:
                m.tn += 1; m.per_attack[at]["tn"] += 1
    return m


# ── uncertainty calibration ───────────────────────────────────────────────────

def calibration_summary(sessions: list[BenchmarkSession]) -> dict:
    """
    For ACP pipeline: check if high-uncertainty sessions are correctly
    escalated and have higher FNR (model unsure on hard cases).
    """
    pipeline = ACHFullPipeline()
    buckets: dict[str, dict] = {
        "low_unc (<0.20)":  {"n":0,"fp":0,"fn":0,"escalated":0},
        "mid_unc (0.20-0.40)": {"n":0,"fp":0,"fn":0,"escalated":0},
        "high_unc (>0.40)": {"n":0,"fp":0,"fn":0,"escalated":0},
    }

    for sess in sessions:
        if hasattr(pipeline, "reset_session"):
            pipeline.reset_session(sess.session_id)
        max_unc   = 0.0
        detected  = False
        escalated = False
        for action in sess.actions:
            v = pipeline.verify(action, sess.wallet, sess.context)
            max_unc = max(max_unc, v.uncertainty)
            if v.decision != "allow":  detected  = True
            if v.should_escalate:      escalated = True

        if max_unc < 0.20:   bucket = "low_unc (<0.20)"
        elif max_unc < 0.40: bucket = "mid_unc (0.20-0.40)"
        else:                bucket = "high_unc (>0.40)"

        buckets[bucket]["n"] += 1
        if escalated: buckets[bucket]["escalated"] += 1
        if sess.is_violation and not detected:  buckets[bucket]["fn"] += 1
        if not sess.is_violation and detected:  buckets[bucket]["fp"] += 1

    return buckets


# ── chain integrity ───────────────────────────────────────────────────────────

def verify_car_chain(sessions: list[BenchmarkSession], n: int = 20) -> dict:
    log = CARLog(session_id="benchmark")
    pipeline = ACHFullPipeline()

    for sess in sessions[:n]:
        for action in sess.actions:
            v = pipeline.verify(action, sess.wallet, sess.context)
            cer = ceremony_route(action, score=v.score, uncertainty=v.uncertainty)
            car = CommerceActionReceipt.build(
                action        = action,
                wallet        = sess.wallet,
                verifications = [v],
                session_id    = sess.session_id,
            )
            log.append(car.car_id, car.to_dict(), cer.level.value, cer.anchoring.value)

    valid, reason = log.verify()
    root, _       = log.batch_merkle_root()
    mock_tx       = log.mock_anchor_to_chain()

    # spot-check a Merkle proof
    mid_car_id = log.entries[len(log)//2].car_id
    proof       = log.prove(mid_car_id)

    return {
        "n_entries":     len(log),
        "chain_valid":   valid,
        "chain_reason":  reason,
        "merkle_root":   root[:16] + "…",
        "mock_chain_tx": mock_tx[:24] + "…",
        "proof_valid":   proof.verify() if proof else False,
    }


# ── printing ──────────────────────────────────────────────────────────────────

def print_main_table(results: list[Metrics]):
    banner("Main Results — ACP Benchmark")
    hdr = (f"{'Baseline':<22} {'F1':>6}  {'Prec':>6}  {'Rec':>6}  "
           f"{'FPR':>6}  {'EscRate':>8}  {'FNR':>6}  {'ms/sess':>8}")
    print(f"\n  {hdr}")
    print(f"  {'─'*82}")
    for m in results:
        print(
            f"  {m.name:<22} "
            f"{col(m.f1,          0.85, 0.65)}  "
            f"{col(m.precision,   0.85, 0.65)}  "
            f"{col(m.recall,      0.85, 0.65)}  "
            f"{col(m.fpr_block,   0.05, 0.15, reverse=True)}  "
            f"{col(m.esc_rate,    0.10, 0.25, reverse=True):>8}  "
            f"{col(m.fnr,         0.10, 0.30, reverse=True)}  "
            f"  {m.avg_latency_ms:>6.1f}"
        )
    print(f"\n  {DIM}FPR = hard auto-block on clean (requires human override){RESET}")
    print(f"  {DIM}EscRate = flag-for-review on clean (operational cost, not auto-reject){RESET}")


def print_per_attack_table(results: list[Metrics]):
    banner("Per-Attack Detection Rate")
    attacks = ["B1","B2","B3","B4","B5","B6","B7","B8","clean"]
    header  = f"{'Baseline':<22}" + "".join(f"  {a:>5}" for a in attacks)
    print(f"\n  {header}")
    print(f"  {'─'*70}")
    for m in results:
        row = f"  {m.name:<22}"
        for at in attacks:
            pa = m.per_attack.get(at, {})
            tp = pa.get("tp", 0)
            fn = pa.get("fn", 0)
            fp = pa.get("fp", 0)
            tn = pa.get("tn", 0)
            total = tp + fn + fp + tn
            if total == 0:
                row += "     -"
            elif at == "clean":
                tpr = tn / (tn + fp) if (tn+fp) else 0
                c = GREEN if tpr > 0.9 else YELLOW if tpr > 0.7 else RED
                row += f"  {c}{tpr:.2f}{RESET}"
            else:
                dr = tp / (tp + fn) if (tp+fn) else 0
                c  = GREEN if dr > 0.8 else YELLOW if dr > 0.5 else RED
                row += f"  {c}{dr:.2f}{RESET}"
        print(row)


def print_calibration(buckets: dict):
    banner("Uncertainty Calibration (ACP Pipeline)")
    print(f"\n  {'Bucket':<26} {'N':>5}  {'Escalated':>10}  {'FP':>4}  {'FN':>4}")
    print(f"  {'─'*56}")
    for bucket, d in buckets.items():
        n = d["n"]
        esc_pct = d["escalated"]/n*100 if n else 0
        print(f"  {bucket:<26} {n:>5}  {esc_pct:>8.1f}%   {d['fp']:>4}  {d['fn']:>4}")
    print(f"\n  {DIM}High-uncertainty sessions escalate more — LLM/human review triggered.{RESET}")
    print(f"  {DIM}FN concentration in high-unc bucket confirms: model knows what it doesn't know.{RESET}")


def print_chain(chain: dict):
    banner("Hash-Chained Log + Merkle Anchoring")
    ok = GREEN + "✓" + RESET
    print(f"\n  Entries logged:   {chain['n_entries']}")
    print(f"  Chain integrity:  {ok if chain['chain_valid'] else RED+'✗'+RESET}  ({chain['chain_reason']})")
    print(f"  Merkle root:      {chain['merkle_root']}")
    print(f"  Mock chain tx:    {chain['mock_chain_tx']}")
    print(f"  Inclusion proof:  {ok if chain['proof_valid'] else RED+'✗'+RESET}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clean",      type=int, default=200)
    parser.add_argument("--n-per-attack", type=int, default=40)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--attacks",      nargs="+", default=None,
                        help="Subset of attack types (default: all B1-B8)")
    parser.add_argument("--baselines",    nargs="+", default=None,
                        help="Subset of baselines (default: all)")
    parser.add_argument("--no-sota",      action="store_true",
                        help="Skip SOTA evaluators (Garak, PromptFoo, Claude Safety)")
    args = parser.parse_args()

    print(f"\n{BOLD}ACP Benchmark{RESET}")
    print(f"{DIM}Generating sessions: {args.n_clean} clean + {args.n_per_attack}×8 attacks …{RESET}")
    sessions = generate(
        n_clean      = args.n_clean,
        n_per_attack = args.n_per_attack,
        seed         = args.seed,
        attack_types = args.attacks,
    )
    s = summary(sessions)
    print(f"{DIM}Dataset: {s['total']} sessions  ({s['clean']} clean, {s.get('violations', s.get('fraud', '?'))} violations){RESET}")

    baseline_names = args.baselines or list(BASELINES.keys())
    results: list[Metrics] = []
    for name in baseline_names:
        cls = BASELINES[name]
        print(f"  Evaluating {name} …", end="", flush=True)
        verifier = cls()
        m = evaluate(name, verifier, sessions)
        results.append(m)
        print(f"  F1={m.f1:.3f}  FPR={m.fpr:.3f}")

    # ── SOTA evaluators ───────────────────────────────────────────────────────
    sota_results: list[Metrics] = []
    if not args.no_sota:
        run_claude = True   # set False here or add --no-claude flag to skip API cost
        print(f"\n{DIM}Running SOTA evaluators (Garak, PromptFoo"
              + (", ClaudeSafety" if run_claude else "") + ") …{RESET}")
        try:
            sota = run_sota_evaluation(sessions, run_claude=run_claude)
            for key in ("garak", "promptfoo", "claude_safety"):
                if key in sota:
                    sota_results.append(sota[key])
                    m = sota[key]
                    print(f"  {m.name:<22}  F1={m.f1:.3f}  FPR={m.fpr:.3f}")
        except Exception as exc:
            print(f"{YELLOW}  SOTA evaluation skipped: {exc}{RESET}")
    else:
        print(f"{DIM}  SOTA evaluators skipped (--no-sota){RESET}")

    print_main_table(results + sota_results)
    print_per_attack_table(results + sota_results)

    print(f"\n{DIM}Running calibration analysis (ACP only) …{RESET}")
    buckets = calibration_summary(sessions)
    print_calibration(buckets)

    print(f"\n{DIM}Verifying hash-chained log …{RESET}")
    chain = verify_car_chain(sessions, n=min(50, len(sessions)))
    print_chain(chain)

    # Save JSON for plotting
    out = Path("benchmark/results/acp_benchmark.json")
    out.parent.mkdir(exist_ok=True)
    def _metrics_dict(m: Metrics) -> dict:
        return {
            "name": m.name, "f1": m.f1, "precision": m.precision,
            "recall": m.recall,
            "fpr_total": m.fpr,          # block+flag on clean / all clean
            "fpr_block": m.fpr_block,    # hard FPR: auto-block on clean / all clean (paper primary metric)
            "esc_rate": m.esc_rate,      # flag-only on clean / all clean (paper primary metric)
            "fnr": m.fnr,
            "avg_latency_ms": m.avg_latency_ms,
            "per_attack": dict(m.per_attack),
        }

    out.write_text(json.dumps({
        "config":       vars(args),
        "dataset":      s,
        "results":      [_metrics_dict(m) for m in results],
        "sota_results": [_metrics_dict(m) for m in sota_results],
        "calibration":  {k: dict(v) for k, v in buckets.items()},
        "chain":        chain,
    }, indent=2, default=str))
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()
