"""
Prod Replicator — replays the exact calls real prod agents made,
feeding them back through Gordon with the new fraud detectors active.

This gives you:
  1. Shadow mode: see what the new detectors would have decided on real traffic
  2. Regression: verify new detectors don't false-positive on known-good traffic
  3. Attack replay: inject scenarios into real prod sessions
  4. Baseline: real prod sessions seed L3 behavioral fingerprints

The real prod injection case:
  https://exa.ai/search?cmd=ignore+previous+instructions
  → was ALLOWED by prod Gordon
  → should be BLOCKED by L1 classifier
  → replicator shows the before/after
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from harness.simulate.recorder import record_from_prod
from harness.simulate.injectors import inject as sim_inject
from .mcp_client import GordonMCPClient
from .wallet import wallet_for
from .adversarial import AdversarialSimulator
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReplicationResult:
    agent_id:           str
    session_count:      int
    total_calls:        int
    prod_decisions:     dict          # original decisions from prod
    new_decisions:      dict          # decisions with new detectors
    regressions:        list[dict]    # cases where decision changed
    real_injection_caught: bool       # did we catch the known prod injection?


def replicate_agent(
    agent_id:   str,
    api_key:    str,
    detector:   callable = None,
    dry_run:    bool = True,
) -> ReplicationResult:
    """
    Load all recorded sessions for an agent from prod,
    replay them through Gordon (or dry_run mock) with detectors active.
    Compare new decisions to what prod originally decided.
    """
    sessions = record_from_prod(agent_id=agent_id)
    wallet = wallet_for(agent_id)
    client = GordonMCPClient(
        agent_id=agent_id,
        api_key=api_key,
        wallet=wallet,
        detector=detector,
        dry_run=dry_run,
    )

    prod_decisions = {}
    new_decisions = {}
    regressions = []
    real_injection_caught = False

    for session in sessions:
        for event in session.events:
            call_id = event.event_id

            # Replay the call through the client
            if event.action_type.value == "authorize":
                result = client.authorize(
                    service_id=event.service_id or "",
                    operation_id=event.operation_id or "",
                    max_payment_units=event.amount_units or 0,
                    original_request=event.original_request or {},
                    pay_to=event.vendor,
                )
            elif event.action_type.value == "find_service":
                query = (event.original_request or {}).get("query", "")
                result = client.find_service(query)
            elif event.action_type.value == "a2a_transfer":
                result = client.a2a_transfer(
                    to_agent_id=event.vendor or "",
                    amount_units=event.amount_units or 0,
                )
            else:
                continue

            # Prod originally decided...
            prod_decision = event.decision.value if event.decision else "allow"
            prod_decisions[call_id] = prod_decision
            new_decisions[call_id] = result.status

            # Check for the known real injection
            ep = event.raw_endpoint or ""
            if "ignore+previous+instructions" in ep or "cmd=ignore" in ep:
                real_injection_caught = (result.status in ("blocked", "escalated"))

            # Regression: prod allowed but new system blocks (or vice versa)
            if prod_decision == "allow" and result.status == "blocked":
                regressions.append({
                    "type": "new_block",
                    "call_id": call_id,
                    "endpoint": event.raw_endpoint,
                    "amount": event.amount_units,
                    "risk_score": result.risk_score,
                    "risk_flags": result.risk_flags,
                })
            elif prod_decision == "block" and result.status == "allowed":
                regressions.append({
                    "type": "new_allow",
                    "call_id": call_id,
                    "endpoint": event.raw_endpoint,
                })

    return ReplicationResult(
        agent_id=agent_id,
        session_count=len(sessions),
        total_calls=len(client.call_log),
        prod_decisions=prod_decisions,
        new_decisions=new_decisions,
        regressions=regressions,
        real_injection_caught=real_injection_caught,
    )


def replicate_all_prod_agents(
    api_key:    str,
    detector:   callable = None,
    dry_run:    bool = True,
) -> list[ReplicationResult]:
    """Replicate every agent we have in prod."""
    sessions = record_from_prod()
    agent_ids = list({s.agent_id for s in sessions})
    print(f"Replicating {len(agent_ids)} prod agents...")
    return [
        replicate_agent(aid, api_key, detector, dry_run)
        for aid in agent_ids
    ]
