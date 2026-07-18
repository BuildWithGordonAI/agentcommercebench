"""
Fraud Detection Game — Adversary vs Defender

The game:
  - ADVERSARY: real LLM (Claude Haiku via AWS Bedrock) with malicious system prompt
  - DEFENDER:  Gordon L1+L3+L4 detector pipeline running inline
  - BACKEND:   real Gordon MCP (if GORDON_AGENT_KEY set) or deterministic dry-run

  Test ID: every call tags X-Harness-Test-Id header → Gordon logs it under the
  test run. The adversary LLM never sees this header.

Usage:
    game = FraudDetectionGame(persona="research", rounds=5)
    result = game.play()              # runs, prints trace, saves report
    print(result.trace_path)          # harness/data/traces/{run_id}/

    python game.py --persona research --rounds 5 --scenarios A1 A5 B1
    python game.py --real             # use real Gordon MCP (needs GORDON_AGENT_KEY)
"""
import time, uuid, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import load_env  # noqa: F401 — side-effect: resolves AWS_KEY_SLOT → AWS_ACCESS_KEY_ID/SECRET

from dataclasses import dataclass, field
from typing import Optional

from harness.agent.adversary_agent import AdversaryAgent
from harness.agent.adversarial import SCENARIO_DESCRIPTIONS, AdversarialSimulator
from harness.agent.mcp_client import GordonMCPClient
from harness.agent.wallet import wallet_for
from harness.detect.l1_payload import L1PayloadClassifier
from harness.detect.l4_price import L4PriceOracle
from harness.detect.l3_behavioral import L3BehavioralFingerprint
from harness.detect.pipeline import DetectorPipeline
from harness.trace.recorder import TraceRecorder
from harness.trace.report import render_console, save_report


SCENARIOS_BY_DIFFICULTY = {
    "easy":   ["A1", "A5", "B2"],
    "medium": ["A3", "A4", "B1", "B7", "D1"],
    "hard":   ["A4", "C1", "C2", "D2"],
    "all":    ["A1", "A3", "A4", "A5", "A6",
               "B1", "B2", "B7", "C1", "C2", "D1", "D2"],
}


@dataclass
class RoundResult:
    round_num:       int
    scenario:        str
    persona:         str
    description:     str
    total_moves:     int
    auth_attempts:   int
    allowed:         int
    blocked:         int
    escalated:       int
    adversary_pts:   int
    defender_pts:    int
    total_authorized_usdc: float
    total_blocked_usdc:    float
    move_log:        list[dict] = field(default_factory=list)
    duration_s:      float = 0.0

    def winner(self) -> str:
        if self.adversary_pts > self.defender_pts:  return "ADVERSARY"
        if self.defender_pts > self.adversary_pts:  return "DEFENDER"
        return "DRAW"

    def banner(self) -> str:
        w = self.winner()
        return {"ADVERSARY": "🔴 ADVERSARY", "DEFENDER": "🔵 DEFENDER", "DRAW": "⚪ DRAW"}[w]


@dataclass
class GameResult:
    run_id:          str
    persona:         str
    rounds:          list[RoundResult]
    total_rounds:    int
    adversary_total: int
    defender_total:  int
    total_usdc_allowed: float
    total_usdc_blocked: float
    duration_s:      float
    trace_path:      str = ""
    report_path:     str = ""

    def winner(self) -> str:
        if self.adversary_total > self.defender_total: return "ADVERSARY"
        if self.defender_total > self.adversary_total: return "DEFENDER"
        return "DRAW"

    def summary(self) -> str:
        lines = [
            "",
            "═" * 64,
            f"  GAME OVER — {self.winner()} WINS  |  Run: {self.run_id}",
            "═" * 64,
            f"  Persona:           {self.persona}",
            f"  Rounds:            {self.total_rounds}",
            f"  Duration:          {self.duration_s:.1f}s",
            "",
            f"  ADVERSARY:         {self.adversary_total} pts",
            f"  DEFENDER:          {self.defender_total} pts",
            f"  USDC through:      ${self.total_usdc_allowed:.4f}",
            f"  USDC blocked:      ${self.total_usdc_blocked:.4f}",
            "",
            "  ROUND BREAKDOWN",
            "  " + "─" * 60,
        ]
        for r in self.rounds:
            lines.append(
                f"  Round {r.round_num:>2} | {r.scenario:<3} | "
                f"Auth:{r.allowed}/{r.auth_attempts} Blocked:{r.blocked} | "
                f"{r.banner()}"
            )
        lines += [
            "═" * 64,
            f"  Full trace: {self.trace_path}",
            f"  Report:     {self.report_path}",
        ]
        return "\n".join(lines)


class _TracingDetector:
    """
    Wraps DetectorPipeline and feeds the full DetectorResult to the recorder.
    The recorder needs the per-layer scores, not just (score, flags).
    """
    def __init__(self, pipeline: DetectorPipeline, recorder: TraceRecorder,
                 round_num: int, turn_counter: list):
        self.pipeline     = pipeline
        self.recorder     = recorder
        self.round_num    = round_num
        self.turn_counter = turn_counter  # mutable list so we can increment
        self._last_result = None

    def __call__(self, tool: str, params: dict, response: dict):
        result = self.pipeline.score(tool, params, response)
        self._last_result = result
        return result.risk_score, result.flags


class FraudDetectionGame:
    """
    Main game loop.

    Parameters:
        persona:        "procurement" | "research" | "travel"
        scenarios:      list[str] or difficulty preset string
        rounds:         number of rounds
        gordon_key:     full Gordon agent key (gak_pub...:gak_sec...).
                        If set, makes real MCP calls. Otherwise dry_run.
        use_llm:        enable LLM-based L1 scoring
        verbose:        print console trace
        save_dir:       where to save trace JSON + Markdown report
    """

    def __init__(
        self,
        persona:     str = "research",
        scenarios:   list[str] | str = "medium",
        rounds:      int = 3,
        gordon_key:  str = None,
        use_llm:     bool = False,
        verbose:     bool = True,
        save_dir:    str = "harness/data/traces",
    ):
        self.persona    = persona
        self.rounds_n   = rounds
        self.gordon_key = gordon_key or os.environ.get("GORDON_AGENT_KEY", "")
        self.use_llm    = use_llm
        self.verbose    = verbose
        self.save_dir   = save_dir

        if isinstance(scenarios, str):
            self.scenarios = SCENARIOS_BY_DIFFICULTY.get(scenarios,
                             SCENARIOS_BY_DIFFICULTY["medium"])
        else:
            self.scenarios = list(scenarios)

        if rounds > len(self.scenarios):
            self.scenarios = (self.scenarios * ((rounds // len(self.scenarios)) + 1))[:rounds]
        elif rounds < len(self.scenarios):
            self.scenarios = self.scenarios[:rounds]

    def play(self) -> GameResult:
        run_id   = f"run-{uuid.uuid4().hex[:10]}"
        recorder = TraceRecorder(run_id=run_id, persona=self.persona)
        t_start  = time.perf_counter()
        round_results: list[RoundResult] = []

        for i, scenario in enumerate(self.scenarios, 1):
            result = self._run_round(i, scenario, recorder)
            round_results.append(result)

        total_dur       = time.perf_counter() - t_start
        adversary_total = sum(r.adversary_pts for r in round_results)
        defender_total  = sum(r.defender_pts  for r in round_results)
        usdc_allowed    = sum(r.total_authorized_usdc for r in round_results)
        usdc_blocked    = sum(r.total_blocked_usdc    for r in round_results)

        recorder.finish_game(adversary_total, defender_total, usdc_allowed, usdc_blocked)

        # Print rich console trace
        if self.verbose:
            print(render_console(recorder.game_trace))

        # Save JSON + Markdown
        paths = save_report(recorder.game_trace, self.save_dir)

        return GameResult(
            run_id=run_id,
            persona=self.persona,
            rounds=round_results,
            total_rounds=self.rounds_n,
            adversary_total=adversary_total,
            defender_total=defender_total,
            total_usdc_allowed=usdc_allowed,
            total_usdc_blocked=usdc_blocked,
            duration_s=total_dur,
            trace_path=paths["json"],
            report_path=paths["markdown"],
        )

    def _run_round(self, round_num: int, scenario: str,
                   recorder: TraceRecorder) -> RoundResult:
        t0   = time.perf_counter()
        desc = SCENARIO_DESCRIPTIONS.get(scenario, scenario)
        recorder.start_round(round_num, scenario)

        # Fresh detectors per round (resets idempotency state)
        pipeline = DetectorPipeline([
            L1PayloadClassifier(use_llm=self.use_llm),
            L4PriceOracle(),
            L3BehavioralFingerprint(self.persona),
        ])

        # Adversarial interceptor (injects attack payload mid-flight)
        interceptor = AdversarialSimulator(scenario=scenario, intensity=1.0)

        # Build MCP client — real or dry_run
        if self.gordon_key:
            from harness.agent.gordon_mcp import GordonRealMCPClient
            client = GordonRealMCPClient(
                agent_key=self.gordon_key,
                test_run_id=recorder.run_id,
                interceptor=interceptor.intercept,
                detector=pipeline,
            )
        else:
            agent_id = f"harness-{scenario.lower()}-{uuid.uuid4().hex[:6]}"
            wallet   = wallet_for(agent_id)
            client   = GordonMCPClient(
                agent_id=agent_id,
                api_key="harness-dry-run",
                wallet=wallet,
                interceptor=interceptor.intercept,
                detector=pipeline,
                dry_run=True,
            )

        # Wrap detector so we capture full DetectorResult (with layer_scores)
        _last_pipeline_result: list = [None]
        _orig_detector = pipeline

        def tracing_detector(tool, params, response):
            result = pipeline.score(tool, params, response)
            _last_pipeline_result[0] = result
            return result.risk_score, result.flags

        client.detector = tracing_detector

        # Run the adversary LLM
        adversary = AdversaryAgent(persona=self.persona, scenario=scenario)
        adversary.run(client)

        # Score round + record trace events
        allowed = blocked = escalated = auth_attempts = 0
        auth_total_units = 0
        blocked_total_units = 0
        move_log = []

        # The interceptor fired once; we need to match it back to calls
        inj_events = {e.tool: e for e in interceptor.injections}

        for turn, call in enumerate(client.call_log, 1):
            # Figure out original vs mutated params
            params_mutated   = call.request
            was_intercepted  = call.intercepted

            # Try to reconstruct original params (pre-interceptor)
            if was_intercepted and call.tool in inj_events:
                inj = inj_events[call.tool]
                params_original = inj.original
            else:
                params_original = params_mutated

            # Use the pipeline to re-score for the trace (we want DetectorResult not just floats)
            det_result = pipeline.score(call.tool, params_mutated, call.response or {})

            recorder.record_call(
                tool=call.tool,
                params_original=params_original,
                params_mutated=params_mutated,
                was_intercepted=was_intercepted,
                mcp_result=call,
                detector_result=det_result,
            )

            entry = {
                "turn": turn, "tool": call.tool, "status": call.status,
                "risk_score": call.risk_score, "flags": call.risk_flags or [],
            }
            if call.tool == "authorize":
                auth_attempts += 1
                amount = call.request.get("max_payment_units", 0)
                entry["amount_units"] = amount
                if call.status == "allowed":
                    allowed += 1
                    auth_total_units += amount
                elif call.status == "blocked":
                    blocked += 1
                    blocked_total_units += amount
                else:
                    escalated += 1
                    blocked_total_units += amount
            move_log.append(entry)

        adversary_pts = allowed
        defender_pts  = blocked + escalated
        dur = time.perf_counter() - t0

        recorder.end_round(adversary_pts, defender_pts, dur)

        return RoundResult(
            round_num=round_num,
            scenario=scenario,
            persona=self.persona,
            description=desc,
            total_moves=len(client.call_log),
            auth_attempts=auth_attempts,
            allowed=allowed,
            blocked=blocked,
            escalated=escalated,
            adversary_pts=adversary_pts,
            defender_pts=defender_pts,
            total_authorized_usdc=auth_total_units / 1_000_000,
            total_blocked_usdc=blocked_total_units / 1_000_000,
            move_log=move_log,
            duration_s=dur,
        )
