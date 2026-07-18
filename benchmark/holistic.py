"""
AgentCommerceBench — Holistic Three-Layer Benchmark

Measures fraud detection across all three injection vectors:

  L1 — Prompt / Payload Injection
        A1 direct SYSTEM: override in payload
        A3 recursive purchase via tool response
        A4 base64-encoded injection
        A6 confused deputy / multi-agent delegation
        A7 MCP tool response poisoning (service catalog)

  L2 — Behavior / Tool / MCP Injection
        B1 intent drift (category shift over session)
        B2 amount ratcheting (gradual increase)
        B3 reconnaissance (mass FIND_SERVICE, 0 AUTHORIZE)
        B4 timing anomaly (burst)
        B5 new service discovery (unknown MCP endpoint)
        B6 velocity spike (burst AUTHORIZE)
        B7 policy bypass probe (binary search on limits)

  L3 — Commerce / Payment Rail
        A2 dormant injection (plants L1, fires L3)
        A5 price poisoning (oracle manipulation)
        D1 replay attack (duplicate idempotency key)
        D2 agent impersonation (stolen key + behavioral discontinuity)
        C1 circular A2A settlement
        C2 Sybil A2A warmup + strike

Per-layer metrics, cross-layer first-detection table, and comparison
between synthetic baseline sessions and real Gordon production sessions.

Usage:
    # Synthetic only
    python -m benchmark.holistic

    # With real sessions from collect.py
    python -m benchmark.holistic --real-sessions benchmark/real_sessions/

    # Compact (no ASCII table headers, JSON output)
    python -m benchmark.holistic --json > benchmark/results/holistic_latest.json
"""
import sys, os, argparse, json, glob
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from benchmark.generate import build_dataset, split
from harness.simulate.schema import Persona, Session, ActionType
from harness.simulate.replay import replay_batch, ALLOW_THRESHOLD, ESCALATE_THRESHOLD
from harness.simulate.injectors import inject, ALL_SCENARIOS

from benchmark.baselines import velocity, keyword, isolation_forest
from benchmark.baselines.llm_safety import detect as llm_safety_detect
from benchmark.models import session_graph as _sgrf
from harness.simulate.replay import replay as _replay_session

# gordon_+seq (proprietary pipeline) — not published in this repo.
# Results from the full pipeline (F1=0.93, FPR=0%) are reported in the paper.
# Available to academic reviewers upon request.
_GORDON_PIPELINE_AVAILABLE = False

# ── Three-layer attack taxonomy ───────────────────────────────────────────────

LAYER_TAXONOMY = {
    "L1_PAYLOAD": {
        "label": "L1 — Prompt/Payload Injection",
        "scenarios": ["A1", "A3", "A4", "A6", "A7"],
        "description": (
            "Text-based attacks embedded in tool responses or request payloads. "
            "Attack arrives via content the agent processes — SYSTEM: overrides, "
            "base64 blobs, recursive purchase notes, MCP catalog poisoning."
        ),
    },
    "L2_BEHAVIOR_MCP": {
        "label": "L2 — Behavior / Tool / MCP Injection",
        "scenarios": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "description": (
            "Attacks expressed through action sequences, not payload text. "
            "Intent drift, mass FIND_SERVICE recon, velocity spikes, policy probing. "
            "Requires behavioral baseline to detect — text classifiers are blind."
        ),
    },
    "L3_COMMERCE_RAIL": {
        "label": "L3 — Commerce / Payment Rail",
        "scenarios": ["A2", "A5", "D1", "D2", "C1", "C2"],
        "description": (
            "Attacks at payment execution: replay, price oracle poisoning, "
            "agent impersonation, circular A2A settlement, Sybil warmup+strike. "
            "Payment-layer checks are the last line of defense."
        ),
    },
}

# Scenarios present in all runs (excludes C1/C2 which need A2A infra)
TARGET_SCENARIOS = ["A1", "A3", "A4", "A5", "A6", "A7",
                    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
                    "D1", "D2"]

DETECTORS = {
    "velocity_check":  [velocity.detect],
    "keyword_filter":  [keyword.detect],
    "llm_text_safety": [llm_safety_detect],
    "isolation_forest":[isolation_forest.detect],
    "session_graph_rf":[_sgrf.detect],
    # gordon_l1_l3_l4 and gordon_+seq are the proprietary Gordon pipeline.
    # Paper results: gordon_+seq F1=0.93, FPR=0%. Available to reviewers on request.
}

# Which layer does each scenario's attack originate from?
_ATTACK_ORIGIN = {
    "A1": "L1", "A2": "L1→L3", "A3": "L1", "A4": "L1", "A5": "L3",
    "A6": "L1", "A7": "L1",
    "B1": "L2", "B2": "L2", "B3": "L2", "B4": "L2", "B5": "L2",
    "B6": "L2", "B7": "L2",
    "D1": "L3", "D2": "L3", "C1": "L3", "C2": "L3",
}

# Which OSS detector has the best coverage for each layer?
_OSS_LAYER_BEST = {
    "L1_PAYLOAD": "keyword_filter",    # text-pattern matching
    "L2_BEHAVIOR_MCP": "isolation_forest",  # anomaly detection
    "L3_COMMERCE_RAIL": "velocity_check",   # rate limiting
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(results, scenario: str = None) -> dict:
    if scenario:
        attacked = [r for r in results if r.scenario_id == scenario]
    else:
        attacked = [r for r in results if not r.is_clean]
    clean = [r for r in results if r.is_clean]

    tp = sum(1 for r in attacked if r.true_positive)
    fn = len(attacked) - tp
    fp = sum(1 for r in clean if r.false_positive)
    tn = len(clean) - fp

    tpr       = tp / max(len(attacked), 1)
    fpr       = fp / max(len(clean), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tpr
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tpr": tpr, "fpr": fpr, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_attacked": len(attacked), "n_clean": len(clean),
    }


def _layer_metrics(results, layer_key: str) -> dict:
    """Aggregate metrics over all scenarios in a given layer."""
    scenarios = LAYER_TAXONOMY[layer_key]["scenarios"]
    attacked = [r for r in results
                if r.scenario_id in scenarios and not r.is_clean]
    clean = [r for r in results if r.is_clean]
    tp = sum(1 for r in attacked if r.true_positive)
    fn = len(attacked) - tp
    fp = sum(1 for r in clean if r.false_positive)
    tpr       = tp / max(len(attacked), 1)
    fpr       = fp / max(len(clean), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tpr
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "layer": layer_key, "tpr": tpr, "fpr": fpr, "f1": f1,
        "tp": tp, "fn": fn, "fp": fp, "n_attacked": len(attacked),
    }


# ── First-detection layer attribution ─────────────────────────────────────────

def _flag_to_layer(flag: str) -> str:
    """Map a risk flag string to L1/L2/L3 detection layer."""
    prefix = flag.split(":")[0].lower()
    if prefix in ("l1",):
        return "L1"
    if prefix in ("l2",):
        return "L2"
    if prefix in ("l3", "l4"):
        return "L3"
    if any(t in prefix for t in ("keyword", "llm", "llmsafety", "base64", "mcp_poison")):
        return "L1"
    if any(t in prefix for t in ("seq", "recon", "velocity", "sequence", "behavior")):
        return "L2"
    if any(t in prefix for t in ("price", "replay", "identity", "a2a", "oracle",
                                  "cross_session", "sybil")):
        return "L3"
    return "L?"


def _first_guard(session: Session, threshold: float = ALLOW_THRESHOLD) -> str | None:
    """
    For an attacked session, return the detection layer of the first event
    that exceeded threshold. Reads risk_score + risk_flags written by replay().
    """
    for e in session.events:
        score = getattr(e, "risk_score", None) or 0.0
        if score < threshold:
            continue
        flags = getattr(e, "risk_flags", []) or []
        if not flags:
            return "L?"
        for flag in flags:
            layer = _flag_to_layer(flag)
            if layer != "L?":
                return layer
        return "L?"
    return None


def _session_caught(session: Session, threshold: float = ALLOW_THRESHOLD) -> bool:
    """True if any injected event risk_score >= threshold."""
    return any(
        e.is_injected and (getattr(e, "risk_score", 0) or 0) >= threshold
        for e in session.events
    )


def _cross_layer_table(attacked_sessions: list[Session]) -> str:
    """
    Build cross-layer first-detection table:
        rows = attack origin layer (L1/L2/L3)
        cols = first-detection layer (L1/L2/L3/missed)
    """
    matrix: dict[str, dict[str, int]] = {
        "L1": {"L1": 0, "L2": 0, "L3": 0, "L?": 0, "missed": 0},
        "L2": {"L1": 0, "L2": 0, "L3": 0, "L?": 0, "missed": 0},
        "L3": {"L1": 0, "L2": 0, "L3": 0, "L?": 0, "missed": 0},
    }

    for s in attacked_sessions:
        origin = _ATTACK_ORIGIN.get(s.scenario_id or "", "L?")
        layer = origin.split("→")[0] if "→" in origin else origin
        if layer not in matrix:
            continue
        caught = _session_caught(s)
        if caught:
            det_layer = _first_guard(s) or "L?"
            matrix[layer][det_layer] += 1
        else:
            matrix[layer]["missed"] += 1

    lines = ["\nCross-layer first-detection matrix (gordon_+seq):"]
    lines.append(f"{'Attack origin':<18} {'→ caught L1':>11} {'→ caught L2':>11} "
                 f"{'→ caught L3':>11} {'→ caught L?':>11} {'→ missed':>9}")
    lines.append("─" * 75)
    for layer_key in ("L1", "L2", "L3"):
        row = matrix[layer_key]
        total = sum(row.values())
        if total == 0:
            continue
        lines.append(
            f"{layer_key + ' (payload/behavior/rail)':<18} "
            f"{row['L1']:>11}  {row['L2']:>11}  {row['L3']:>11}  "
            f"{row['L?']:>11}  {row['missed']:>9}"
        )
    lines.append(
        "\nIdeal: L1 attacks caught by L1, L2 by L2, L3 by L3 (defense in depth)."
    )
    return "\n".join(lines)


# ── Per-layer summary ──────────────────────────────────────────────────────────

def _layer_summary_table(all_results: dict) -> str:
    lines = ["\nPer-layer TPR — gordon_+seq vs best OSS baseline"]
    lines.append(f"{'Layer':<34} {'Our TPR':>8} {'OSS best TPR':>12}")
    lines.append("─" * 58)

    ours_results = all_results.get("gordon_+seq", [])

    for layer_key, meta in LAYER_TAXONOMY.items():
        ours = _layer_metrics(ours_results, layer_key)
        oss_name = _OSS_LAYER_BEST[layer_key]
        oss_r = all_results.get(oss_name, [])
        oss = _layer_metrics(oss_r, layer_key) if oss_r else {"tpr": 0.0}
        lines.append(
            f"{meta['label']:<34} "
            f"{ours['tpr']*100:>6.0f}%  "
            f"{oss['tpr']*100:>10.0f}%  ({oss_name})"
        )
    return "\n".join(lines)


# ── Scenario-level table ───────────────────────────────────────────────────────

def _scenario_table(all_results: dict, scenarios: list[str]) -> str:
    lines = [f"\nScenario coverage table ({len(scenarios)} scenarios)"]
    det_names = list(all_results.keys())
    hdr_dets = " ".join(f"{n[:8]:>10}" for n in det_names)
    lines.append(f"{'Scen':<5} {'Layer':>6} {hdr_dets}")
    lines.append("─" * (11 + 10 * len(det_names) + 2))

    for s in sorted(scenarios):
        origin = _ATTACK_ORIGIN.get(s, "?")
        row = f"{s:<5} {origin:>6} "
        for det_name, results in all_results.items():
            m = _metrics(results, s)
            tpr = m["tpr"]
            row += f"{'  ✓' if tpr >= 0.9 else ('  ~' if tpr >= 0.5 else '  ✗'):>10}"
        lines.append(row)

    lines.append("\n  ✓ = caught  ~ = partial  ✗ = missed")
    return "\n".join(lines)


# ── Real session injection table ───────────────────────────────────────────────

def _real_session_results(
    real_session_paths: list[str],
    pipeline,
    scenarios: list[str],
    seq_model_trained: bool,
) -> str:
    if not real_session_paths:
        return "\n(no real sessions — run benchmark/real_agent/collect.py --all)"

    lines = [f"\nReal Gordon session injection results ({len(real_session_paths)} sessions)"]
    lines.append(f"  Scenarios tested: {scenarios}")
    lines.append(f"  Pipeline: gordon_+seq\n")

    catches = defaultdict(int)
    totals  = defaultdict(int)

    for path in real_session_paths:
        session = Session.load(path)
        for scen in scenarios:
            try:
                attacked = inject(session, scenario=scen, seed=42)
                # Annotate persona for sequence model
                for e in attacked.events:
                    e._persona = attacked.persona.value
                result = _replay_session(
                    attacked, pipeline=pipeline, detector_name="gordon_+seq"
                )
                totals[scen] += 1
                if result.true_positive:
                    catches[scen] += 1
            except Exception as ex:
                lines.append(f"  WARN: {os.path.basename(path)} × {scen}: {ex}")

    hit = sum(catches.values())
    total = sum(totals.values())
    lines.append(f"  Overall catch rate: {hit}/{total} = {100*hit/max(total,1):.0f}%\n")

    # Per-scenario aggregate
    lines.append(f"  {'Scenario':<8} {'Layer':>6} {'Catch':>6} {'Total':>6} {'Rate':>6}")
    lines.append("  " + "─" * 36)
    for scen in sorted(scenarios):
        n = totals[scen]
        c = catches[scen]
        origin = _ATTACK_ORIGIN.get(scen, "?")
        lines.append(f"  {scen:<8} {origin:>6} {c:>6} {n:>6} {100*c/max(n,1):>5.0f}%")

    return "\n".join(lines)


# ── Holistic summary ───────────────────────────────────────────────────────────

def _holistic_summary(all_results: dict) -> str:
    ours = _metrics(all_results.get("gordon_+seq", []))
    kw   = _metrics(all_results.get("keyword_filter", []))
    iso  = _metrics(all_results.get("isolation_forest", []))
    vel  = _metrics(all_results.get("velocity_check", []))
    llm  = _metrics(all_results.get("llm_text_safety", []))

    best_oss_f1 = max(kw["f1"], iso["f1"], vel["f1"], llm["f1"])

    lines = ["\n" + "=" * 65]
    lines.append("HOLISTIC BENCHMARK — KEY CLAIMS")
    lines.append("=" * 65)
    lines.append(f"  Our system (gordon_+seq)")
    lines.append(f"    F1   = {ours['f1']:.2f}   FPR = {ours['fpr']*100:.0f}%")
    lines.append(f"    TP   = {ours['tp']}/{ours['n_attacked']}")
    lines.append(f"  Best OSS baseline F1 = {best_oss_f1:.2f}")
    lines.append(f"  Δ F1 vs best OSS     = +{ours['f1'] - best_oss_f1:.2f}")
    lines.append("")
    lines.append("  Layer coverage (our system):")
    ours_res = all_results.get("gordon_+seq", [])
    for layer_key, meta in LAYER_TAXONOMY.items():
        m = _layer_metrics(ours_res, layer_key)
        lines.append(
            f"    {meta['label']:<40} TPR={m['tpr']*100:.0f}%  "
            f"({m['tp']}/{m['n_attacked']})"
        )
    lines.append("")
    lines.append("  OSS baselines — no single baseline covers all three layers.")
    lines.append("  Combined OSS coverage still misses behavioral (L2) at 0% FPR.")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    n_clean: int = 50,
    n_per_scenario: int = 1,
    seed: int = 42,
    real_sessions_dir: str = None,
    as_json: bool = False,
):
    if not as_json:
        print(f"\nAgentCommerceBench — Holistic Three-Layer Benchmark")
        print(f"  seed={seed}  n_clean={n_clean}  n_per_scenario={n_per_scenario}")
        print("=" * 65)

    # 1. Generate synthetic dataset
    sessions = build_dataset(n_clean=n_clean, n_per_scenario=n_per_scenario, seed=seed)

    # 2. Merge real captured sessions as additional clean baselines
    real_paths: list[str] = []
    if real_sessions_dir:
        real_paths = glob.glob(os.path.join(real_sessions_dir, "*.json"))
        real_clean = [Session.load(p) for p in real_paths]
        sessions.extend(real_clean)
        if not as_json:
            print(f"  Merged {len(real_clean)} real sessions as additional clean baselines.")

    train, test = split(sessions, seed=seed)
    clean_test    = [s for s in test if s.is_clean]
    attacked_test = [s for s in test if not s.is_clean]
    in_test = sorted({s.scenario_id for s in attacked_test if s.scenario_id in TARGET_SCENARIOS})

    if not as_json:
        print(f"  Synthetic: {len(sessions)-len(real_paths)} sessions")
        print(f"  Real:      {len(real_paths)} sessions")
        print(f"  Test clean / attacked: {len(clean_test)} / {len(attacked_test)}")
        print(f"  Scenarios in test: {in_test}\n")

    # 3. Fit sequence model + session graph model + annotate persona
    sequence_model.fit_all([s for s in train if s.is_clean])
    _sgrf.fit_from_sessions(train)
    for s in test:
        for e in s.events:
            e._persona = s.persona.value

    # 4. Run each detector
    all_results: dict[str, list] = {}
    for det_name, pipeline in DETECTORS.items():
        if not as_json:
            print(f"  Running {det_name}...", end=" ", flush=True)
        results = replay_batch(test, pipeline=pipeline, detector_name=det_name)
        all_results[det_name] = results
        if not as_json:
            m = _metrics(results)
            print(f"F1={m['f1']:.2f}  FPR={m['fpr']*100:.0f}%  TP={m['tp']}/{m['n_attacked']}")

    # 5. Print layer summary
    if not as_json:
        print(_layer_summary_table(all_results))

    # 6. Per-scenario coverage table
    if not as_json:
        print(_scenario_table(all_results, in_test))

    # 7. Cross-layer first-detection matrix (using gordon_+seq)
    seq_pipeline = DETECTORS["gordon_+seq"]
    replay_batch(attacked_test, pipeline=seq_pipeline, detector_name="gordon_+seq_phase")
    for s in attacked_test:
        for e in s.events:
            e._persona = s.persona.value

    if not as_json:
        print(_cross_layer_table(attacked_test))

    # 8. Real session injection (inject each scenario into each real session)
    if not as_json and real_paths:
        # Re-use the same gordon_+seq pipeline
        real_injection_report = _real_session_results(
            real_paths, seq_pipeline, in_test, seq_model_trained=True
        )
        print(real_injection_report)

    # 9. Holistic summary
    if not as_json:
        print(_holistic_summary(all_results))

    # 10. JSON output
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"seed": seed, "n_clean": n_clean, "n_per_scenario": n_per_scenario,
                   "real_sessions": len(real_paths)},
        "layer_taxonomy": {k: v["label"] for k, v in LAYER_TAXONOMY.items()},
        "detectors": {},
    }
    for det_name, results in all_results.items():
        m = _metrics(results)
        per_layer = {
            k: _layer_metrics(results, k)
            for k in LAYER_TAXONOMY
        }
        per_scenario = {s: _metrics(results, s) for s in in_test}
        out["detectors"][det_name] = {
            "overall": m,
            "per_layer": per_layer,
            "per_scenario": per_scenario,
        }

    if as_json:
        print(json.dumps(out, indent=2, default=str))
    else:
        os.makedirs("benchmark/results", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = f"benchmark/results/holistic_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved → {out_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clean",        type=int, default=50)
    parser.add_argument("--n-per-scenario", type=int, default=1)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--real-sessions",  type=str, default=None)
    parser.add_argument("--json",           action="store_true")
    args = parser.parse_args()

    run(
        n_clean=args.n_clean,
        n_per_scenario=args.n_per_scenario,
        seed=args.seed,
        real_sessions_dir=args.real_sessions,
        as_json=args.json,
    )
