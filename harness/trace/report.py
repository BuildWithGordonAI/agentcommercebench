"""
Report Renderer — turns a GameTrace into console output and Markdown.

Console output is rich ASCII with color-like indicators.
Markdown is saved to harness/data/traces/{run_id}/report.md.
"""
import os, json
from .recorder import GameTrace, RoundTrace, TraceEvent


# ── ANSI helpers (works in most terminals) ──────────────────────────────

def _bold(s):   return f"\033[1m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _green(s):  return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _cyan(s):   return f"\033[96m{s}\033[0m"
def _dim(s):    return f"\033[2m{s}\033[0m"

DECISION_COLOR = {
    "ALLOWED": _green,
    "ALLOWED\n": _green,
    "BLOCKED": _red,
    "ESCALATED": _yellow,
}


def render_console(game_trace: GameTrace, no_color: bool = False) -> str:
    """Render a full game trace to a rich console string."""
    lines = []

    def h1(s):  lines.append(_bold(f"\n{'═' * 64}\n  {s}\n{'═' * 64}") if not no_color else f"\n{'='*64}\n  {s}\n{'='*64}")
    def h2(s):  lines.append(_cyan(f"\n{'─' * 64}\n  {s}") if not no_color else f"\n{'-'*64}\n  {s}")
    def h3(s):  lines.append((_bold(f"\n  {s}")) if not no_color else f"\n  {s}")
    def p(s):   lines.append(f"  {s}")
    def sub(s): lines.append(_dim(f"    {s}") if not no_color else f"    {s}")

    h1(f"GORDON FRAUD DETECTION GAME TRACE")
    p(f"Run ID:     {game_trace.run_id}")
    p(f"Persona:    {game_trace.persona}")
    p(f"Started:    {game_trace.started_at}")
    p(f"Finished:   {game_trace.finished_at}")

    winner = "ADVERSARY" if game_trace.adversary_total > game_trace.defender_total \
             else "DEFENDER" if game_trace.defender_total > game_trace.adversary_total \
             else "DRAW"
    color = _red if winner == "ADVERSARY" else _green if winner == "DEFENDER" else _yellow
    p("")
    p(_bold(f"FINAL SCORE: ADVERSARY {game_trace.adversary_total} — DEFENDER {game_trace.defender_total}") if not no_color
      else f"FINAL SCORE: ADVERSARY {game_trace.adversary_total} — DEFENDER {game_trace.defender_total}")
    p((color(f"WINNER: {winner}")) if not no_color else f"WINNER: {winner}")
    p(f"USDC allowed through: ${game_trace.total_usdc_allowed:.4f}")
    p(f"USDC blocked:         ${game_trace.total_usdc_blocked:.4f}")

    for round_trace in game_trace.rounds:
        _render_round(round_trace, lines, no_color, h1, h2, h3, p, sub)

    return "\n".join(lines)


def _render_round(rt: RoundTrace, lines, no_color, h1, h2, h3, p, sub):
    def p(s): lines.append(f"  {s}")

    h1(f"ROUND {rt.round_num}  |  Scenario: {rt.scenario}  |  Persona: {rt.persona}")

    # Scenario explanation
    p(f"Attack type:  {rt.scenario}")
    p(f"What's happening:")
    for sentence in rt.scenario_narrative.split(". "):
        if sentence.strip():
            lines.append(f"    → {sentence.strip()}.")

    winner_str = ("🔴 ADVERSARY" if rt.winner == "ADVERSARY"
                  else "🔵 DEFENDER" if rt.winner == "DEFENDER"
                  else "⚪ DRAW")
    p(f"Round winner: {winner_str} | Duration: {rt.duration_s:.1f}s")

    for event in rt.events:
        _render_event(event, lines, no_color, h2, h3)

    lines.append("")
    lines.append(f"  {'─' * 60}")
    pts = f"ADVERSARY {rt.adversary_pts} pts  |  DEFENDER {rt.defender_pts} pts"
    lines.append(f"  {pts}")
    lines.append("")


def _render_event(ev: TraceEvent, lines, no_color, h2, h3):
    def p(s): lines.append(f"    {s}")
    def sub(s): lines.append(f"      {s}")

    dec_color = {"allowed": _green, "blocked": _red, "escalated": _yellow}
    color_fn = dec_color.get(ev.decision.lower(), lambda x: x) if not no_color else lambda x: x

    tool_label = ev.tool.upper()
    decision_label = color_fn(ev.decision.upper()) if not no_color else ev.decision.upper()
    h2(f"Move {ev.turn_num}: [{tool_label}] → {decision_label}")

    p(f"What the adversary tried:")
    p(f"  {ev.what_adversary_tried}")

    if ev.was_intercepted:
        p("")
        if no_color:
            p(f"[INTERCEPTOR FIRED — attack payload injected]")
        else:
            p(f"\033[91m[INTERCEPTOR FIRED — attack payload injected]\033[0m")
        for line in ev.what_changed.split("\n"):
            p(f"  {line}")

    if ev.layer_scores:
        p("")
        p("Detector analysis:")
        for layer, score in ev.layer_scores.items():
            bar = "█" * int(score * 20)
            rest = "░" * (20 - int(score * 20))
            score_str = f"{score:.2f}"
            color_fn2 = (_red if score >= 0.70 else _yellow if score >= 0.30 else _dim) if not no_color else lambda x: x
            p(f"  {layer:<20} {color_fn2(score_str)} [{bar}{rest}]")

        if ev.risk_flags:
            p("  Flags:")
            for flag in ev.risk_flags:
                p(f"    ⚑ {flag}")

    p("")
    if no_color:
        p(f"Outcome: {ev.outcome_explanation}")
    else:
        color_fn3 = _red if ev.decision.lower() == "blocked" else \
                    _yellow if ev.decision.lower() == "escalated" else _green
        p(f"Outcome: {color_fn3(ev.outcome_explanation)}")

    if ev.amount_usdc > 0:
        p(f"Amount:  ${ev.amount_usdc:.4f} USDC  ({ev.amount_units:,} μ)")


def save_report(game_trace: GameTrace, base_dir: str = "harness/data/traces") -> dict[str, str]:
    """
    Save trace as JSON + Markdown report.
    Returns dict of saved file paths.
    """
    run_dir = os.path.join(base_dir, game_trace.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # JSON trace
    json_path = os.path.join(run_dir, "trace.json")
    with open(json_path, "w") as f:
        json.dump(_game_to_dict(game_trace), f, indent=2)

    # Markdown report
    md_path = os.path.join(run_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(_render_markdown(game_trace))

    return {"json": json_path, "markdown": md_path}


def _game_to_dict(game_trace: GameTrace) -> dict:
    import dataclasses
    return dataclasses.asdict(game_trace)


def _render_markdown(game_trace: GameTrace) -> str:
    lines = [
        f"# Gordon Fraud Detection Game — Trace Report",
        f"",
        f"**Run ID:** `{game_trace.run_id}`  ",
        f"**Persona:** {game_trace.persona}  ",
        f"**Started:** {game_trace.started_at}  ",
        f"**Finished:** {game_trace.finished_at}",
        f"",
        f"## Final Score",
        f"",
        f"| Team | Points |",
        f"|------|--------|",
        f"| 🔴 Adversary | {game_trace.adversary_total} |",
        f"| 🔵 Defender  | {game_trace.defender_total} |",
        f"",
        f"**USDC allowed through:** ${game_trace.total_usdc_allowed:.4f}  ",
        f"**USDC blocked:** ${game_trace.total_usdc_blocked:.4f}",
        f"",
    ]

    winner = ("🔴 ADVERSARY" if game_trace.adversary_total > game_trace.defender_total
              else "🔵 DEFENDER" if game_trace.defender_total > game_trace.adversary_total
              else "⚪ DRAW")
    lines += [f"**Winner: {winner}**", ""]

    for rt in game_trace.rounds:
        lines += [
            f"---",
            f"",
            f"## Round {rt.round_num} — Scenario {rt.scenario}",
            f"",
            f"**What's happening:**  ",
            f"{rt.scenario_narrative}",
            f"",
            f"**Round winner:** {'🔴 ADVERSARY' if rt.winner == 'ADVERSARY' else '🔵 DEFENDER' if rt.winner == 'DEFENDER' else '⚪ DRAW'}  ",
            f"**Duration:** {rt.duration_s:.1f}s",
            f"",
        ]

        for ev in rt.events:
            dec_emoji = {"allowed": "✅", "blocked": "🔴", "escalated": "⚠️"}.get(ev.decision.lower(), "?")
            lines += [
                f"### Move {ev.turn_num}: `{ev.tool}` → {dec_emoji} {ev.decision.upper()}",
                f"",
                f"**What the adversary tried:** {ev.what_adversary_tried}",
                f"",
            ]
            if ev.was_intercepted:
                lines += [
                    f"**🔥 Interceptor fired — attack payload injected:**",
                    f"```",
                    ev.what_changed,
                    f"```",
                    f"",
                ]
            if ev.layer_scores:
                lines += [f"**Detector scores:**", f""]
                lines += [f"| Layer | Score | Decision |"]
                lines += [f"|-------|-------|----------|"]
                for layer, score in ev.layer_scores.items():
                    d = "🔴 BLOCK" if score >= 0.70 else "⚠️ ESCALATE" if score >= 0.30 else "✅ PASS"
                    lines.append(f"| {layer} | {score:.3f} | {d} |")
                lines.append("")
                if ev.risk_flags:
                    lines += [f"**Flags fired:**", ""]
                    for flag in ev.risk_flags:
                        lines.append(f"- `{flag}`")
                    lines.append("")
            lines += [
                f"**Outcome:** {ev.outcome_explanation}",
                f"",
            ]
            if ev.amount_usdc > 0:
                lines += [f"**Amount:** ${ev.amount_usdc:.4f} USDC ({ev.amount_units:,} μ)", ""]

        lines += [
            f"**Score:** Adversary {rt.adversary_pts} pts — Defender {rt.defender_pts} pts",
            f"",
        ]

    return "\n".join(lines)
