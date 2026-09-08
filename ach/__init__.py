"""
ACH — Agentic Commerce Harness

Core abstractions for building commerce-safe agents.
Verifier is a pluggable interface; Gordon is the production implementation.
"""
from ach.actions.base   import ConsequentialAction, ActionType, Reversibility
from ach.actions.wallet import AgentWallet, SpendLimits, ConsentLevel
from ach.verifiers.base import Verifier, Verification, SessionContext
from ach.verifiers.local_rules import LocalRulesVerifier
from ach.verifiers.gordon       import GordonVerifier
from ach.receipts.car           import CommerceActionReceipt, ProvenanceStep

__all__ = [
    "ConsequentialAction", "ActionType", "Reversibility",
    "AgentWallet", "SpendLimits", "ConsentLevel",
    "Verifier", "Verification", "SessionContext",
    "LocalRulesVerifier", "GordonVerifier",
    "CommerceActionReceipt", "ProvenanceStep",
]
