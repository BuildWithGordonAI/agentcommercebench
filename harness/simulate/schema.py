"""
Core session schema — every event in the harness is this shape.
Matches Gordon prod schema exactly so sessions can be written to the real DB.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
from enum import Enum
import uuid, json


class ActionType(str, Enum):
    FIND_SERVICE    = "find_service"
    GET_SERVICE     = "get_service"
    AUTHORIZE       = "authorize"
    A2A_TRANSFER    = "a2a_transfer"
    SETTLE          = "settle"


class Decision(str, Enum):
    ALLOW     = "allow"
    BLOCK     = "block"
    ESCALATE  = "escalate"


class Persona(str, Enum):
    PROCUREMENT = "procurement"
    RESEARCH    = "research"
    TRAVEL      = "travel"


@dataclass
class Event:
    event_id:       str
    session_id:     str
    agent_id:       str
    action_type:    ActionType
    timestamp:      datetime
    service_id:     Optional[str]
    operation_id:   Optional[str]
    amount_units:   Optional[int]          # USDC micro-units (1 USDC = 1_000_000)
    vendor:         Optional[str]          # wallet address or agent_id
    category:       Optional[str]
    raw_endpoint:   Optional[str]
    original_request: Optional[dict]       # the payload — L1's primary signal
    network:        str = "eip155:8453"    # Base mainnet

    # Ground truth — set by injectors, null on clean sessions
    is_injected:    bool = False
    attack_scenario: Optional[str] = None  # A1, B3, C1, etc.
    expected_detector: Optional[str] = None  # L1, L2, L3, L4, L5

    # Filled by detectors during replay
    decision:       Optional[Decision] = None
    decision_reason: Optional[str] = None
    risk_score:     Optional[float] = None
    risk_flags:     list[str] = field(default_factory=list)
    detector_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['action_type'] = self.action_type.value
        d['decision'] = self.decision.value if self.decision else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        d = dict(d)
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        d['action_type'] = ActionType(d['action_type'])
        if d.get('decision'):
            d['decision'] = Decision(d['decision'])
        return cls(**d)


@dataclass
class Session:
    session_id:   str
    persona:      Persona
    agent_id:     str
    seed:         int                      # for deterministic replay
    created_at:   datetime
    events:       list[Event] = field(default_factory=list)

    # Attack metadata — null for clean sessions
    scenario_id:      Optional[str] = None   # A1–D2
    injection_point:  Optional[int] = None   # index into events list
    is_clean:         bool = True

    # Populated after replay
    session_risk_score: Optional[float] = None
    true_positive:      Optional[bool] = None   # if attacked: did we catch it?
    false_positive:     Optional[bool] = None   # if clean: did we fire?

    def save(self, path: str):
        d = {
            'session_id': self.session_id,
            'persona': self.persona.value,
            'agent_id': self.agent_id,
            'seed': self.seed,
            'created_at': self.created_at.isoformat(),
            'scenario_id': self.scenario_id,
            'injection_point': self.injection_point,
            'is_clean': self.is_clean,
            'session_risk_score': self.session_risk_score,
            'true_positive': self.true_positive,
            'false_positive': self.false_positive,
            'events': [e.to_dict() for e in self.events],
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path) as f:
            d = json.load(f)
        events = [Event.from_dict(e) for e in d.pop('events')]
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        d['persona'] = Persona(d['persona'])
        s = cls(**d)
        s.events = events
        return s

    def clone(self) -> "Session":
        """Return a deep copy for mutation without affecting the original."""
        import copy
        return copy.deepcopy(self)

    def make_event(self, action_type: ActionType, timestamp: datetime = None, **kwargs) -> Event:
        return Event(
            event_id=str(uuid.uuid4()),
            session_id=self.session_id,
            agent_id=self.agent_id,
            action_type=action_type,
            timestamp=timestamp or datetime.utcnow(),
            **kwargs
        )
