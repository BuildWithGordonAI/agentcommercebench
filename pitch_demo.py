"""
Gordon AI — Fraud Detection Demo
Runs as a live console showing the agent payment rail with the fraud detector
intercepting every call in real-time. Clean sessions pass. Fraudulent ones
get caught mid-stream.

Usage:
    python pitch_demo.py              # scenario 1 (A7)
    python pitch_demo.py --all        # all 3 scenarios
    python pitch_demo.py --scenario 2
    python pitch_demo.py --fast       # no animation
"""
import argparse, os, sys, time, random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from harness.simulate.schema import Persona, ActionType
from harness.simulate.personas import generate_clean_session
from harness.simulate.injectors import inject
import benchmark.baselines.keyword  as kw
import benchmark.baselines.velocity as vel
from benchmark.models.session_graph import session_to_features, SessionGraphModel
from benchmark.generate import build_dataset

# ── ANSI ──────────────────────────────────────────────────────────────────────
R    = "\033[0m"
BOLD = "\033[1m"; DIM  = "\033[2m"
RED  = "\033[91m"; GRN  = "\033[92m"; YLW  = "\033[93m"
CYN  = "\033[96m"; MGT  = "\033[95m"; BLU  = "\033[94m"; GRY = "\033[90m"
BG_RED = "\033[41m"; BG_GRN = "\033[42m"; BG_YLW = "\033[43m"

def b(t):   return f"{BOLD}{t}{R}"
def d(t):   return f"{DIM}{GRY}{t}{R}"
def red(t): return f"{RED}{t}{R}"
def grn(t): return f"{GRN}{t}{R}"
def ylw(t): return f"{YLW}{t}{R}"
def cyn(t): return f"{CYN}{t}{R}"

W = 70

def rule(char="─"): print(d(char * W))
def blank():        print()

# ── Simulated vendor catalog (what the agent sees in MCP responses) ───────────

VENDOR_CATALOG = {
    "finance": [
        ("AlphaFinance API",  "$0.007/call", "4.9"),
        ("DataStream Pro",    "$0.009/call", "4.7"),
        ("EconPulse",         "$0.006/call", "4.4"),
    ],
    "search": [
        ("Exa Search",        "$0.008/call", "4.8"),
        ("BraveSearch API",   "$0.007/call", "4.6"),
        ("Tavily Pro",        "$0.009/call", "4.9"),
    ],
    "procurement": [
        ("ProcureAI",         "$0.011/call", "4.7"),
        ("VendorHub",         "$0.013/call", "4.5"),
        ("SourceLink",        "$0.010/call", "4.6"),
    ],
}


def _catalog_lines(category: str, inject_text: str | None = None) -> list[str]:
    vendors = VENDOR_CATALOG.get(category, VENDOR_CATALOG["search"])
    lines = []
    for name, price, rating in vendors:
        lines.append(f"             {name:<22}  {price}  ★{rating}")
    if inject_text:
        lines.append(f"             {red('─' * 52)}")
        lines.append(f"             {red(inject_text)}")
    return lines


# ── SGRF loader ────────────────────────────────────────────────────────────────

def load_sgrf() -> SessionGraphModel:
    import joblib
    from pathlib import Path
    p = Path("benchmark/models/artifacts/session_graph_rf.joblib")
    if p.exists():
        return joblib.load(p)
    print(d("  [Training session graph model — first run only, ~5s...]"))
    m = SessionGraphModel()
    m.fit(build_dataset(n_clean=200, n_per_scenario=15, seed=42))
    return m


# ── Live per-event fraud check ─────────────────────────────────────────────────

def _event_fraud_check(event, session_so_far, sgrf: SessionGraphModel) -> tuple[float, str]:
    """Run all detectors on a partial session ending with this event."""
    # keyword on this event's payload
    kw_max, kw_lbl = 0.0, ""
    hits = kw._scan_text(str(event.original_request or ""))
    for sc, lb in hits:
        if sc > kw_max:
            kw_max, kw_lbl = sc, lb

    # velocity against prior events
    vel_sc, vel_flags = vel.detect(event, session_so_far)
    vel_lbl = vel_flags[0] if vel_flags else ""

    # session graph on the full partial session
    import copy
    from harness.simulate.schema import Session
    partial = copy.copy(event)
    # build a minimal Session for feature extraction
    from dataclasses import replace
    tmp_events = session_so_far + [event]

    class _FakeSession:
        def __init__(self, events): self.events = events
    tmp = _FakeSession(tmp_events)
    feats = session_to_features(tmp)
    proba = sgrf.clf.predict_proba(feats.reshape(1, -1))[0]
    sgrf_sc = float(proba[1])

    combined = max(kw_max, vel_sc, sgrf_sc)
    signals  = []
    if kw_max  >= 0.70: signals.append(f"keyword={kw_max:.2f}({kw_lbl})")
    if vel_sc  >= 0.70: signals.append(f"velocity={vel_sc:.2f}({vel_lbl})")
    if sgrf_sc >= 0.70: signals.append(f"behavior={sgrf_sc:.2f}")
    if not signals and combined >= 0.40:
        signals.append(f"risk={combined:.2f}")

    return combined, "  ".join(signals)


def _check_line(score: float, detail: str) -> str:
    if score >= 0.70:
        return f"           {d('fraud check:')}  {BG_RED}{BOLD} ALERT {R}  {red(detail)}"
    if score >= 0.40:
        return f"           {d('fraud check:')}  {BG_YLW} WARN  {R}  {ylw(detail)}"
    return f"           {d('fraud check:  pass')}"


# ── Session runner — the core of the demo ────────────────────────────────────

def run_live_session(
    session,
    label: str,
    category: str,
    sgrf: SessionGraphModel,
    fast: bool,
    show_catalog: bool = True,
) -> tuple[bool, float]:
    """
    Stream a session event-by-event with live fraud checks.
    Returns (was_blocked, max_score).
    """
    delay    = 0.0 if fast else 0.5
    blocked  = False
    max_sc   = 0.0
    history  = []
    probe_n  = 0

    for event in session.events:
        if blocked:
            break

        a = str(event.action_type).replace("ActionType.", "")
        ts = event.timestamp.strftime("%H:%M:%S")

        blank()

        if a == "FIND_SERVICE":
            probe_n += 1
            req = event.original_request or {}
            q   = req.get("query", category) or category

            if probe_n > 3:
                print(f"  {d(ts)}  {cyn(f'gordon_find_service')}  {d(f'[probe #{probe_n}]  query={str(q)[:30]}')}")
                print(f"           {ylw('→ agent querying additional vendor categories...')}")
            else:
                print(f"  {d(ts)}  {cyn('gordon_find_service')}")
                print(f"           {d(f'query={str(q)[:50]}')}")
                if show_catalog:
                    inj_text = (req.get("_mcp_response_injected") or
                                req.get("_service_description"))
                    print(f"           {d('→ Gordon catalog response:')}")
                    for line in _catalog_lines(category, inj_text if event.is_injected else None):
                        print(line)

        elif a == "AUTHORIZE":
            vendor = (event.vendor or "—")[:36]
            amt    = f"${event.amount_units / 1_000_000:.4f} USDC" if event.amount_units else "?"
            print(f"  {d(ts)}  {cyn('gordon_authorize')}")
            print(f"           {d(f'vendor={vendor}  amount={amt}')}")

        else:
            print(f"  {d(ts)}  {cyn(a.lower())}")

        if delay: time.sleep(delay * 0.6)

        # live fraud check
        score, detail = _event_fraud_check(event, history, sgrf)
        max_sc = max(max_sc, score)

        print(_check_line(score, detail))

        if score >= 0.70 and a == "AUTHORIZE":
            print(f"           {BG_RED}{BOLD} TRANSACTION BLOCKED {R}  fraud score={red(f'{score:.2f}')}")
            blocked = True
        elif score >= 0.70:
            print(f"           {ylw('→ session flagged — monitoring closely')}")
        elif a == "AUTHORIZE" and not blocked:
            txid = f"{random.randint(0x1000, 0xffff):04x}{random.randint(0x1000, 0xffff):04x}"
            print(f"           {grn(f'→ authorized  txid={txid}...')}")

        history.append(event)
        if delay: time.sleep(delay)

    return blocked, max_sc


# ── Scenario definitions ───────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":        "A7",
        "persona":   Persona.RESEARCH,
        "category":  "finance",
        "title":     "Vendor response poisoning",
        "agent_id":  "research_agent_001",
        "task":      "Query financial market data for portfolio rebalancing",
        "clean_note":
            "A clean session. Agent finds a finance vendor, authorizes a micropayment. "
            "Fraud detector runs on every call.",
        "attack_note":
            "Same agent, 4 minutes later. Same task. "
            "This time the vendor catalog response has been tampered.",
    },
    {
        "id":        "B3",
        "persona":   Persona.RESEARCH,
        "category":  "search",
        "title":     "Reconnaissance sweep",
        "agent_id":  "research_agent_002",
        "task":      "Find the best-priced search API for a batch data job",
        "clean_note":
            "Normal session: agent finds one search vendor and pays. Two events.",
        "attack_note":
            "Same agent, different session. "
            "Watch the FIND_SERVICE call count.",
    },
    {
        "id":        "B2",
        "persona":   Persona.PROCUREMENT,
        "category":  "procurement",
        "title":     "Amount ratcheting  [open gap]",
        "agent_id":  "procurement_agent_001",
        "task":      "Source API vendors for quarterly procurement run",
        "clean_note":
            "Clean procurement session. One find, one payment.",
        "attack_note":
            "This session looks identical to the clean one. "
            "The attack escalates amounts across sessions — invisible to single-session detectors. "
            "This is the honest gap.",
    },
]


# ── Scenario orchestrator ──────────────────────────────────────────────────────

def run_scenario(cfg: dict, n: int, total: int, sgrf: SessionGraphModel, fast: bool):
    base_date = datetime(2026, 7, 19, 10, 43, 45)
    clean     = generate_clean_session(
        persona=cfg["persona"],
        agent_id=cfg["agent_id"],
        base_date=base_date,
        seed=42,
    )
    # Attacked session 4 minutes later
    attacked_base  = base_date + timedelta(minutes=4)
    clean_for_atk  = generate_clean_session(
        persona=cfg["persona"],
        agent_id=cfg["agent_id"],
        base_date=attacked_base,
        seed=43,
    )
    attacked = inject(clean_for_atk, scenario=cfg["id"], seed=43)

    pause = 0.0 if fast else 1.0

    # ── Scenario header ───────────────────────────────────────────────────────
    blank()
    print(d("═" * W))
    title_line = f'SCENARIO {n}/{total}  ·  {cfg["title"]}'
    print(f"  {b(title_line)}")
    print(d("═" * W))
    blank()
    print(f"  {d('Agent:  ')}{cfg['agent_id']}")
    print(f"  {d('Task:   ')}{cfg['task']}")

    # ── Clean session ─────────────────────────────────────────────────────────
    blank()
    rule()
    print(f"  {b('Session 1 of 2')}  {d('·  ' + base_date.strftime('%H:%M:%S') + ' UTC')}")
    print(f"  {d(cfg['clean_note'])}")
    rule()

    clean_blocked, clean_sc = run_live_session(
        clean, "clean", cfg["category"], sgrf, fast, show_catalog=True
    )

    blank()
    n_ev  = len(clean.events)
    total_usdc = sum(e.amount_units or 0 for e in clean.events) / 1_000_000
    print(f"  {d('─'*W)}")
    print(f"  Session 1 complete  ·  {n_ev} events  ·  {f'${total_usdc:.4f} USDC authorized'}")
    print(f"  Status: {grn('CLEAN')  if not clean_blocked else red('BLOCKED')}")

    if pause: time.sleep(pause)

    # ── Adversarial session ───────────────────────────────────────────────────
    blank()
    rule()
    print(f"  {b('Session 2 of 2')}  {d('·  ' + attacked_base.strftime('%H:%M:%S') + ' UTC')}")
    print(f"  {d(cfg['attack_note'])}")
    rule()

    atk_blocked, atk_sc = run_live_session(
        attacked, "attacked", cfg["category"], sgrf, fast,
        show_catalog=(cfg["id"] == "A7"),
    )

    blank()
    n_ev_a = len(attacked.events)
    print(f"  {d('─'*W)}")
    print(f"  Session 2 complete  ·  {n_ev_a} events  ·  peak fraud score: {red(f'{atk_sc:.2f}') if atk_sc >= 0.70 else ylw(f'{atk_sc:.2f}') if atk_sc >= 0.40 else d(f'{atk_sc:.2f}')}")
    if atk_blocked:
        print(f"  Status: {red('BLOCKED')}  {d('— payment did not go through')}")
    elif atk_sc >= 0.40:
        print(f"  Status: {ylw('FLAGGED')}  {d('— escalated for review')}")
    else:
        print(f"  Status: {BG_RED} NOT DETECTED {R}  {d('— open gap: cross-session history required')}")

    return {
        "scenario_id": cfg["id"],
        "title":       cfg["title"],
        "clean_score": clean_sc,
        "atk_score":   atk_sc,
        "blocked":     atk_blocked,
    }


# ── Summary ────────────────────────────────────────────────────────────────────

def show_summary(results: list):
    blank()
    print(d("═" * W))
    print(b("  Results"))
    rule()
    print(f"  {d('Scenario'):<38}  {d('clean'):>6}  {d('attack'):>8}  {d('outcome')}")
    rule("·")

    for r in results:
        cs    = f"{r['clean_score']:.2f}"
        asp   = r["atk_score"]
        as_   = red(f"{asp:.2f}") if asp >= 0.70 else (ylw(f"{asp:.2f}") if asp >= 0.40 else d(f"{asp:.2f}"))
        out   = red("BLOCKED")     if r["blocked"] else (
                ylw("FLAGGED")     if asp >= 0.40 else
                f"{BG_RED} MISSED {R}")
        print(f"  {r['title']:<38}  {d(cs):>6}  {as_:>8}  {out}")

    blank()
    print(d("  Gordon fraud detector runs as middleware on every gordon_find_service"))
    print(d("  and gordon_authorize call. No action taken until a signal fires."))
    blank()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3])
    parser.add_argument("--fast",     action="store_true")
    args = parser.parse_args()

    if args.scenario:
        indices = [args.scenario - 1]
    elif args.all:
        indices = [0, 1, 2]
    else:
        indices = [0]

    blank()
    print(b("  Gordon AI  ·  payment fraud detection"))
    print(d("  MCP middleware  ·  every agent call intercepted  ·  real-time scoring"))
    rule()

    sgrf    = load_sgrf()
    results = []

    for i, idx in enumerate(indices):
        r = run_scenario(SCENARIOS[idx], i + 1, len(indices), sgrf, args.fast)
        results.append(r)

        if not args.fast and i != len(indices) - 1:
            time.sleep(1.5)

    if len(results) > 1:
        show_summary(results)


if __name__ == "__main__":
    main()
