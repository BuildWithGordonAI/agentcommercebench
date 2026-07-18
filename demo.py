"""
Gordon Fraud Harness — Interactive CLI Demo

Usage:
    python demo.py                    # interactive menu
    python demo.py --scenario A1      # run one scenario
    python demo.py --persona travel   # change persona (research/procurement/travel)
    python demo.py --all              # run all scenarios, print benchmark table
    python demo.py --compare          # side-by-side: our pipeline vs. OSS baselines
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from harness.simulate.schema import Persona, ActionType
from harness.simulate.personas import generate_clean_session
from harness.simulate.injectors import inject, ALL_SCENARIOS
from fraud import FraudPipeline, FraudContext

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
GREY    = "\033[90m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
BG_RED  = "\033[41m"
BG_GRN  = "\033[42m"

def c(colour, text): return f"{colour}{text}{RESET}"
def bold(text):      return c(BOLD, text)
def dim(text):       return c(DIM + GREY, text)
def ok(text):        return c(GREEN, text)
def warn(text):      return c(YELLOW, text)
def bad(text):       return c(RED, text)
def hi(text):        return c(CYAN, text)
def label(text):     return c(MAGENTA, text)

# ── Scenario catalogue ────────────────────────────────────────────────────────

SCENARIO_META = {
    "A1": ("Direct prompt injection",    "Payload",      "Rogue SYSTEM: directive hidden in AUTHORIZE payload"),
    "A3": ("Recursive purchase loop",    "Payload",      "Agent told to authorize repeatedly until wallet drained"),
    "A4": ("Base64 encoded injection",   "Evasion",      "Prompt injection encoded to bypass text scanners"),
    "A5": ("Price oracle poisoning",     "Price",        "Tool response manipulates displayed price upward"),
    "A6": ("Confused deputy attack",     "Auth",         "Agent tricked into authorizing on behalf of attacker"),
    "B1": ("Intent drift",               "Behavioral",   "Category of purchases silently shifts mid-session"),
    "B2": ("Amount ratcheting",          "Behavioral",   "Payment amounts escalate slowly under the threshold"),
    "B3": ("Reconnaissance sweep",       "Behavioral",   "Excessive FIND_SERVICE before any payment — probing"),
    "B4": ("Timing anomaly",             "Behavioral",   "Payments at 3am outside declared operating hours"),
    "B5": ("New service discovery",      "Behavioral",   "Agent contacts unknown category not in its profile"),
    "B6": ("Velocity spike",             "Rate",         "Sudden burst of authorizations in seconds"),
    "B7": ("Policy bypass probe",        "Policy",       "Calls exceed declared rate limits systematically"),
    "D1": ("Replay attack",              "Infra",        "Idempotency key reused from prior settled transaction"),
    "D2": ("Agent impersonation",        "Infra",        "agent_id spoofed to impersonate a trusted agent"),
}

PERSONAS = {
    "research":    Persona.RESEARCH,
    "procurement": Persona.PROCUREMENT,
    "travel":      Persona.TRAVEL,
}

PERSONA_EMOJI = {
    Persona.RESEARCH:    "🔬",
    Persona.PROCUREMENT: "🛒",
    Persona.TRAVEL:      "✈️ ",
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_amount(units: int) -> str:
    if units is None:
        return "—"
    usdc = units / 1_000_000
    return f"${usdc:.4f}"

def _fmt_action(action_type) -> str:
    a = str(action_type).replace("ActionType.", "")
    colours = {
        "FIND_SERVICE": CYAN,
        "GET_SERVICE":  BLUE,
        "AUTHORIZE":    YELLOW,
        "SETTLE":       GREEN,
        "A2A_TRANSFER": MAGENTA,
    }
    col = colours.get(a, WHITE)
    return f"{col}{a:<16}{RESET}"

def _fmt_decision(score: float, blocked: bool) -> str:
    if blocked:
        return f"{BG_RED}{BOLD} BLOCK {RESET}  [{RED}{score:.2f}{RESET}]"
    if score >= 0.40:
        return f"{YELLOW}ESCALATE{RESET}  [{YELLOW}{score:.2f}{RESET}]"
    return f"{GREEN}ALLOW   {RESET}  [{DIM}{score:.2f}{RESET}]"

def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    empty  = width - filled
    col = RED if score >= 0.70 else (YELLOW if score >= 0.40 else GREEN)
    return f"{col}{'█' * filled}{DIM}{'░' * empty}{RESET}"


# ── Core replay ───────────────────────────────────────────────────────────────

def run_scenario(
    scenario_id: str,
    persona: Persona = Persona.RESEARCH,
    seed: int = 42,
    animate: bool = True,
    show_clean: bool = False,
) -> dict:
    """
    Run one scenario end-to-end, printing a live event timeline.
    Returns summary dict.
    """
    base_date = datetime(2026, 7, 16, 9, 0, 0)

    # 1. Generate clean session
    clean_session = generate_clean_session(
        persona=persona,
        agent_id=f"demo_agent_{persona.value}",
        base_date=base_date,
        seed=seed,
    )

    # 2. Inject attack
    attacked_session = inject(clean_session, scenario=scenario_id, seed=seed)

    meta = SCENARIO_META.get(scenario_id, (scenario_id, "Unknown", ""))
    attack_name, attack_type, attack_desc = meta

    # 3. Header
    emoji = PERSONA_EMOJI.get(persona, "  ")
    print()
    print(c(BOLD + CYAN, f"  Attack scenario: {scenario_id} — {attack_name}"))
    print(f"  {dim('Category:')} {label(attack_type)}   "
          f"{dim('Agent persona:')} {emoji} {hi(persona.value.title())}")
    print(f"  {dim(attack_desc)}")
    print()
    print(c(BOLD, "  Event timeline"))
    print(f"  {dim('─' * 70)}")
    print(f"  {dim('Time       Action           Amount        Vendor              Score')}")
    print(f"  {dim('─' * 70)}")

    # 4. Run pipeline event by event
    pipeline = FraudPipeline.default()
    context  = FraudContext.for_session(
        session_id=attacked_session.session_id,
        agent_id=attacked_session.agent_id,
        persona=attacked_session.persona,
    )

    first_auth_idx = next(
        (i for i, e in enumerate(attacked_session.events)
         if e.action_type == ActionType.AUTHORIZE),
        len(attacked_session.events),
    )

    detection_event  = None
    detection_guard  = None
    detection_phase  = None
    cumulative_spend = 0
    blocked          = False
    attack_shown     = False

    for i, event in enumerate(attacked_session.events):
        context, signals = pipeline.process_event(event, context)

        # Annotate phase
        phase = "pre" if i < first_auth_idx else "at"

        # Show attack marker once
        if event.is_injected and not attack_shown:
            print(f"\n  {RED}{'⚡ ATTACK INJECTED':^70}{RESET}")
            print(f"  {dim('─' * 70)}")
            attack_shown = True

        ts    = event.timestamp.strftime("%H:%M:%S")
        amt   = _fmt_amount(event.amount_units)
        vendor = (event.vendor or "—")[:18]
        score = context.risk_score
        is_blocked = score >= 0.70 and event.is_injected

        action_str = _fmt_action(event.action_type)
        decision   = _fmt_decision(score, is_blocked)

        line = f"  {dim(ts)}  {action_str}  {YELLOW}{amt:<12}{RESET}  {dim(vendor):<26}  {decision}"
        print(line)

        # Show guard signals for this event
        for sig in signals:
            reason = "  ".join(sig.flags)[:55] if sig.flags else ""
            print(f"           {DIM}└─ {RESET}{label(sig.guard):<22} "
                  f"[{RED}{sig.score:.2f}{RESET}]  {dim(reason)}")

        if event.action_type in (ActionType.AUTHORIZE, ActionType.SETTLE):
            cumulative_spend += (event.amount_units or 0)

        if is_blocked and detection_event is None:
            detection_event = i
            if signals:
                detection_guard = signals[0].guard
            elif context.risk_flags:
                # Score came from accumulated signals (e.g., sequence model fired earlier)
                detection_guard = context.risk_flags[0].split(":")[0] if context.risk_flags else "accumulated"
            else:
                detection_guard = "accumulated"
            detection_phase = "pre-commerce" if phase == "pre" else "at-commerce"
            blocked = True

        if animate and i < len(attacked_session.events) - 1:
            time.sleep(0.08)

    # End-of-session settlement
    context, settle_sigs = pipeline.end_session(context)
    if settle_sigs:
        print(f"\n  {dim('── Settlement layer ──────────────────────────────────────────────────')}")
        for sig in settle_sigs:
            reason = "  ".join(sig.flags)[:55] if sig.flags else ""
            print(f"  {label(sig.guard):<26} [{RED}{sig.score:.2f}{RESET}]  {dim(reason)}")

    # 5. Summary
    final_score = context.risk_score
    is_fraud    = final_score >= 0.70
    n_events    = len(attacked_session.events)
    n_attacked  = sum(1 for e in attacked_session.events if e.is_injected)
    saved       = cumulative_spend - (cumulative_spend if blocked else 0)

    print()
    print(f"  {dim('─' * 70)}")
    if is_fraud:
        verdict = f"{BG_RED}{BOLD}  FRAUD DETECTED  {RESET}"
        colour_score = bad
    elif final_score >= 0.40:
        verdict = f"{YELLOW}{BOLD}  ESCALATED  {RESET}"
        colour_score = warn
    else:
        verdict = f"{BG_GRN}{BOLD}  MISSED  {RESET}"
        colour_score = dim

    print(f"\n  {verdict}  risk={colour_score(f'{final_score:.2f}')}")
    print()
    if detection_event is not None:
        print(f"  {dim('Caught at event:')}  {hi(str(detection_event + 1))}/{n_events}")
        print(f"  {dim('First detection:')}  {hi(detection_phase)}")
        print(f"  {dim('Guard:')}            {label(detection_guard)}")
    elif final_score >= 0.40:
        top_flags = context.risk_flags[:2]
        flags_str = "  ".join(f[:40] for f in top_flags)
        print(f"  {warn('Escalated — requires human review.')}")
        print(f"  {dim('Top signals:')}      {warn(flags_str)}")
    else:
        print(f"  {bad('Not detected.')} Score below ESCALATE threshold.")
    print(f"  {dim('Final risk score:')} {_bar(final_score, 30)} {final_score:.3f}")
    print()

    return {
        "scenario":        scenario_id,
        "persona":         persona.value,
        "detected":        is_fraud,
        "phase":           detection_phase,
        "guard":           detection_guard,
        "risk_score":      final_score,
        "n_events":        n_events,
        "detection_event": detection_event,
    }


# ── All-scenarios benchmark table ─────────────────────────────────────────────

def run_all(persona: Persona = Persona.RESEARCH, animate: bool = False):
    """Run all scenarios and print a benchmark summary table."""
    from benchmark.baselines import velocity, keyword, isolation_forest
    from benchmark.baselines.llm_safety import detect as llm_detect
    from benchmark.generate import build_dataset, split
    from benchmark.detectors import sequence_model

    print()
    print(c(BOLD + CYAN, "  AgentCommerceBench — Full Run"))
    print(f"  {dim('Generating dataset (seed=42, n_clean=50, n_per_scenario=1)...')}", end=" ", flush=True)

    sessions = build_dataset(n_clean=50, n_per_scenario=1, seed=42)
    _, test  = split(sessions, seed=42)
    attacked = [s for s in test if not s.is_clean]
    clean    = [s for s in test if s.is_clean]

    # Fit sequence model
    train, _ = split(sessions, seed=42)
    sequence_model.fit_all([s for s in train if s.is_clean])
    for s in test:
        for e in s.events:
            e._persona = s.persona.value

    print(f"done. {len(test)} test sessions ({len(attacked)} attacked, {len(clean)} clean)")

    scenarios = sorted({s.scenario_id for s in attacked})

    # Run detectors
    from harness.simulate.replay import replay_batch, ALLOW_THRESHOLD
    from fraud.integrations.benchmark_adapter import FraudPipelineDetector

    _fp = FraudPipelineDetector()
    detectors = {
        "velocity":    [velocity.detect],
        "keyword":     [keyword.detect],
        "llm_safety":  [llm_detect],
        "iso_forest":  [isolation_forest.detect],
        "gordon_L2+L3": [_fp.detect],
    }

    print()
    results = {}
    for name, pipeline in detectors.items():
        r = replay_batch(test, pipeline=pipeline, detector_name=name)
        results[name] = r
        tp  = sum(1 for x in r if not x.is_clean and x.true_positive)
        fp  = sum(1 for x in r if x.is_clean and x.false_positive)
        n_a = sum(1 for x in r if not x.is_clean)
        n_c = len(clean)
        f1  = tp / max(tp + (n_a - tp + fp) / 2, 1e-9)
        print(f"  {name:<18}  F1={f1:.2f}  TP={tp}/{n_a}  FP={fp}/{n_c}")

    # Print table
    print()
    print(c(BOLD, f"  {'Detector':<18} " + "  ".join(f"{s:>3}" for s in scenarios) + f"   {'FPR':>4}  {'F1':>4}"))
    print(f"  {dim('─' * (18 + 6 * len(scenarios) + 16))}")

    for name, res in results.items():
        row = []
        for sc in scenarios:
            sc_res = [r for r in res if r.scenario_id == sc]
            tp     = sum(1 for r in sc_res if r.true_positive)
            n      = len(sc_res)
            pct    = f"{tp * 100 // max(n,1):3d}%" if n > 0 else "  —"
            col    = GREEN if tp == n and n > 0 else (YELLOW if tp > 0 else RED)
            row.append(f"{col}{pct}{RESET}")

        tp_all  = sum(1 for r in res if not r.is_clean and r.true_positive)
        fp_all  = sum(1 for r in res if r.is_clean and r.false_positive)
        n_a     = sum(1 for r in res if not r.is_clean)
        n_c     = len(clean)
        fpr     = fp_all / max(n_c, 1)
        prec    = tp_all / max(tp_all + fp_all, 1)
        rec     = tp_all / max(n_a, 1)
        f1      = 2 * prec * rec / max(prec + rec, 1e-9)
        col_f1  = GREEN if f1 >= 0.80 else (YELLOW if f1 >= 0.50 else RED)

        print(f"  {name:<18} {'  '.join(row)}   {dim(f'{fpr:.0%}'):>6}  {col_f1}{f1:.2f}{RESET}")

    print()
    our_tp  = sum(1 for r in results["gordon_L2+L3"] if not r.is_clean and r.true_positive)
    oss_best = max(
        sum(1 for r in results[n] if not r.is_clean and r.true_positive)
        for n in ("velocity", "keyword", "llm_safety", "iso_forest")
    )
    n_a = sum(1 for r in results["gordon_L2+L3"] if not r.is_clean)
    print(f"  {bold('Gordon L2+L3:')}  {GREEN}{our_tp}/{n_a} scenarios{RESET}")
    print(f"  {bold('Best OSS:    ')}  {YELLOW}{oss_best}/{n_a} scenarios{RESET}")
    print(f"  {bold('Advantage:   ')}  {GREEN}+{our_tp - oss_best} scenarios{RESET}")
    print()


# ── Interactive menu ──────────────────────────────────────────────────────────

def print_banner():
    banner = r"""
   ____               _                 _   _
  / ___|  ___  _ __ | |_   ___  _ __  | | | | __ _  _ __  _ __    ___  ___  ___
 | |  _  / _ \| '__|| __| / _ \| '_ \ | |_| |/ _` || '__|| '_ \  / _ \/ __|/ __|
 | |_| || (_) | |   | |_ | (_) | | | ||  _  | (_| || |   | | | ||  __/\__ \\__ \
  \____| \___/|_|    \__| \___/|_| |_||_| |_|\__,_||_|   |_| |_| \___||___/|___/
"""
    print(c(CYAN, banner))
    print(c(BOLD, "  Agent Commerce Fraud Detection Harness") + dim("  v0.1.0 · AgentCommerceBench"))
    print()

def print_menu():
    print(c(BOLD, "  Pick a scenario to simulate:"))
    print()
    groups = {
        "A — Payload Attacks":    [(k, v) for k, v in SCENARIO_META.items() if k.startswith("A")],
        "B — Behavioral Drift":   [(k, v) for k, v in SCENARIO_META.items() if k.startswith("B")],
        "D — Infrastructure":     [(k, v) for k, v in SCENARIO_META.items() if k.startswith("D")],
    }
    keys = []
    for group, items in groups.items():
        print(f"  {dim(group)}")
        for kid, (name, typ, _) in items:
            idx = len(keys) + 1
            keys.append(kid)
            print(f"    {dim(f'[{idx}]')} {hi(kid)} {dim('—')} {name:<38} {dim(typ)}")
        print()

    print(f"  {dim('[a]')} {ok('Run all scenarios (benchmark table)')}")
    print(f"  {dim('[q]')} Quit")
    print()
    return keys

def choose_persona() -> Persona:
    print()
    print(c(BOLD, "  Agent persona:"))
    for i, (name, p) in enumerate(PERSONAS.items()):
        print(f"    {dim(f'[{i+1}]')} {PERSONA_EMOJI[p]} {name.title()}")
    print()
    while True:
        raw = input(f"  {dim('Persona [1-3, default=1]:')} ").strip()
        if not raw:
            return Persona.RESEARCH
        plist = list(PERSONAS.values())
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(plist):
                return plist[idx]
        except ValueError:
            if raw.lower() in PERSONAS:
                return PERSONAS[raw.lower()]
        print(f"  {bad('Invalid choice.')}")


def interactive():
    print_banner()

    while True:
        keys = print_menu()
        raw  = input(f"  {dim('Select:')} ").strip().lower()
        if raw == "q":
            print(f"\n  {dim('Goodbye.')}\n")
            break
        if raw == "a":
            run_all()
            continue

        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                sid = keys[idx]
            else:
                print(f"  {bad('Out of range.')}")
                continue
        except ValueError:
            sid = raw.upper()
            if sid not in SCENARIO_META:
                print(f"  {bad(f'Unknown scenario: {sid}')}")
                continue

        persona = choose_persona()
        print()
        run_scenario(sid, persona=persona)
        print()
        input(f"  {dim('Press Enter to continue...')}")
        print("\n" + "═" * 76 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gordon Fraud Harness Demo")
    parser.add_argument("--scenario", "-s",  help="Scenario ID (A1, B3, D1, ...)")
    parser.add_argument("--persona",  "-p",  default="research",
                        choices=list(PERSONAS), help="Agent persona")
    parser.add_argument("--all",      action="store_true", help="Run full benchmark table")
    parser.add_argument("--no-anim",  action="store_true", help="Disable animation delay")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    persona = PERSONAS.get(args.persona, Persona.RESEARCH)
    animate = not args.no_anim

    if args.all:
        print_banner()
        run_all(persona=persona, animate=animate)
        return

    if args.scenario:
        sid = args.scenario.upper()
        if sid not in SCENARIO_META:
            print(f"Unknown scenario '{sid}'. Valid: {', '.join(sorted(SCENARIO_META))}")
            sys.exit(1)
        print_banner()
        run_scenario(sid, persona=persona, seed=args.seed, animate=animate)
        return

    interactive()


if __name__ == "__main__":
    main()
