#!/usr/bin/env python3
"""
Gordon Fraud Detection Game — CLI

Usage:
    python game.py                                  # 3 medium rounds, research persona
    python game.py --persona procurement --rounds 5
    python game.py --scenarios A1 A5 B1 D1
    python game.py --difficulty hard --rounds 4
    python game.py --llm                            # enable LLM scorer (slower)
    python game.py --all                            # run all 12 scenarios

Examples:
    # Quick test
    python game.py --rounds 2 --persona travel --scenarios B1 D1

    # Full benchmark
    python game.py --all --persona research
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.agent.game import FraudDetectionGame, SCENARIOS_BY_DIFFICULTY
from harness.agent.adversary_agent import ADVERSARIAL_OBJECTIVES
from harness.agent.adversarial import SCENARIO_DESCRIPTIONS


def main():
    parser = argparse.ArgumentParser(
        description="Gordon AI Fraud Detection Game — Adversary vs Defender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--persona",    default="research",
                        choices=["research", "procurement", "travel"],
                        help="Agent persona (default: research)")
    parser.add_argument("--rounds",     type=int, default=3,
                        help="Number of rounds (default: 3)")
    parser.add_argument("--scenarios",  nargs="+",
                        help="Specific scenarios to run (e.g. A1 A5 B1)")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard", "all"],
                        help="Difficulty preset (default: medium)")
    parser.add_argument("--llm",        action="store_true",
                        help="Enable LLM-based L1 payload scoring (slower)")
    parser.add_argument("--all",        action="store_true",
                        help="Run all 12 scenarios")
    parser.add_argument("--list",       action="store_true",
                        help="List all available scenarios and exit")
    parser.add_argument("--real",       action="store_true",
                        help="Call real Gordon MCP (needs GORDON_AGENT_KEY env var)")
    parser.add_argument("--key",        default=None,
                        help="Gordon agent key (gak_pub_...:gak_sec_...) if not in env")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for sid, desc in SCENARIO_DESCRIPTIONS.items():
            has_llm_objective = sid in ADVERSARIAL_OBJECTIVES
            llm_marker = " [LLM adversary]" if has_llm_objective else " [scripted]"
            print(f"  {sid:<4}  {desc}{llm_marker}")
        print("\nDifficulty presets:")
        for level, sids in SCENARIOS_BY_DIFFICULTY.items():
            if level != "all":
                print(f"  {level:<8} {' '.join(sids)}")
        return

    # Determine scenario list
    if args.scenarios:
        scenarios = args.scenarios
        rounds    = args.rounds or len(scenarios)
    elif args.all:
        scenarios = "all"
        rounds    = len(SCENARIOS_BY_DIFFICULTY["all"])
    else:
        scenarios = args.difficulty
        rounds    = args.rounds

    gordon_key = args.key or (os.environ.get("GORDON_AGENT_KEY", "") if args.real else "")
    mode = "REAL Gordon MCP" if gordon_key else "dry-run (mock responses)"

    print(f"\n  Gordon AI — Fraud Detection Game")
    print(f"  Persona: {args.persona} | Rounds: {rounds}")
    print(f"  Mode:    {mode}")
    print(f"  LLM scorer: {'ON' if args.llm else 'OFF (rule-based)'}")

    game = FraudDetectionGame(
        persona=args.persona,
        scenarios=scenarios,
        rounds=rounds,
        gordon_key=gordon_key,
        use_llm=args.llm,
        verbose=True,
    )

    result = game.play()
    sys.exit(0 if result.defender_total >= result.adversary_total else 1)


if __name__ == "__main__":
    main()
