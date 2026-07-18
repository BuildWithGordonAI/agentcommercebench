"""
Clean session generators for each persona.
Distributions are seeded from real prod data (503 transactions, July 2026).
Every session is deterministic given the same seed — fully replayable.
"""
import random, uuid
from datetime import datetime, timedelta
from .schema import Session, Event, ActionType, Persona

# ── Real prod distributions (from gordon prod, July 2026) ──────────────────
# Observed services and their operation_ids from real transactions
PROD_SERVICES = {
    "search": [
        {"service_id": "a9fdc0fb-b8a4-47c9-bfbd-10aa389bfff1",
         "operation_id": "search.web",
         "endpoint": "https://api.exa.ai/search",
         "vendor": "0x6d6E695b09861467c7d462f5AAF31cF3540B9192",
         "amount_range": (7000, 10000)},
        {"service_id": "stableenrich-001",
         "operation_id": "exa.search",
         "endpoint": "https://stableenrich.dev/api/exa/search",
         "vendor": "0xStableEnrich...",
         "amount_range": (5000, 10000)},
        {"service_id": "tavily-001",
         "operation_id": "search",
         "endpoint": "https://x402.tavily.com/search",
         "vendor": "0xTavily...",
         "amount_range": (7000, 15000)},
    ],
    "finance": [
        {"service_id": "untitled-fin-001",
         "operation_id": "macro-stress",
         "endpoint": "https://intelligence.untitledfinancial.com/v1/intelligence/macro-stress",
         "vendor": "0xUntitledFin...",
         "amount_range": (150000, 250000)},
        {"service_id": "untitled-fin-002",
         "operation_id": "commodity",
         "endpoint": "https://intelligence.untitledfinancial.com/v1/intelligence/commodity",
         "vendor": "0xUntitledFin...",
         "amount_range": (200000, 250000)},
        {"service_id": "untitled-fin-003",
         "operation_id": "currency-stress",
         "endpoint": "https://intelligence.untitledfinancial.com/v1/intelligence/currency-stress",
         "vendor": "0xUntitledFin...",
         "amount_range": (150000, 250000)},
    ],
    "ai": [
        {"service_id": "blockrun-001",
         "operation_id": "pm.markets",
         "endpoint": "https://blockrun.ai/api/v1/pm/markets",
         "vendor": "0xBlockRun...",
         "amount_range": (5000, 50000)},
        {"service_id": "nansen-001",
         "operation_id": "token-info",
         "endpoint": "https://api.nansen.ai/api/v1/tgm/token-information",
         "vendor": "0xNansen...",
         "amount_range": (10000, 100000)},
    ],
    "procurement": [
        {"service_id": "procureai-001",
         "operation_id": "vendor.lookup",
         "endpoint": "https://procureai.io/api/v1/vendors",
         "vendor": "0xProcureAI...",
         "amount_range": (5000, 50000)},
        {"service_id": "supplychainai-001",
         "operation_id": "inventory.check",
         "endpoint": "https://supplychainai.io/api/v1/inventory",
         "vendor": "0xSupplyChain...",
         "amount_range": (8000, 100000)},
    ],
    "travel": [
        {"service_id": "travelai-001",
         "operation_id": "flights.search",
         "endpoint": "https://travelai.io/api/v1/flights",
         "vendor": "0xTravelAI...",
         "amount_range": (20000, 200000)},
        {"service_id": "hotelai-001",
         "operation_id": "hotels.search",
         "endpoint": "https://hotelai.io/api/v1/hotels",
         "vendor": "0xHotelAI...",
         "amount_range": (50000, 300000)},
    ],
}

# ── Persona definitions ────────────────────────────────────────────────────
PERSONA_CONFIG = {
    Persona.PROCUREMENT: {
        "category_weights": {"search": 0.10, "finance": 0.08, "ai": 0.02,
                              "procurement": 0.80},
        "amount_range": (5000, 100000),
        "session_length": (2, 5),           # events per session
        "sessions_per_day": (1, 5),
        "active_hours": (9, 17),            # 09:00–17:00
        "vendor_set_size": (3, 8),
        # GET_SERVICE optional — only 6% of prod events are get_service
        "sequence": [ActionType.FIND_SERVICE, ActionType.AUTHORIZE],
    },
    Persona.RESEARCH: {
        "category_weights": {"search": 0.65, "finance": 0.33, "ai": 0.02},
        "amount_range": (5000, 150000),
        "session_length": (1, 4),
        "sessions_per_day": (50, 500),      # high frequency
        "active_hours": (9, 20),            # market hours + after
        "vendor_set_size": (5, 15),
        "sequence": [ActionType.FIND_SERVICE, ActionType.AUTHORIZE],
    },
    Persona.TRAVEL: {
        "category_weights": {"search": 0.10, "ai": 0.02, "travel": 0.88},
        "amount_range": (20000, 200000),
        "session_length": (2, 6),
        "sessions_per_day": (0, 20),        # bursty
        "active_hours": (8, 22),
        "vendor_set_size": (10, 40),
        "sequence": [ActionType.FIND_SERVICE, ActionType.AUTHORIZE],
    },
}


def _pick_service(persona: Persona, rng: random.Random) -> dict:
    config = PERSONA_CONFIG[persona]
    weights = config["category_weights"]
    categories = [c for c in weights if c in PROD_SERVICES]
    if not categories:
        categories = list(PROD_SERVICES.keys())
    category = rng.choices(categories,
                           weights=[weights.get(c, 0.1) for c in categories])[0]
    return rng.choice(PROD_SERVICES[category])


def _active_timestamp(base_date: datetime, config: dict, rng: random.Random) -> datetime:
    h_start, h_end = config["active_hours"]
    hour = rng.randint(h_start, h_end - 1)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=second,
                              microsecond=rng.randint(0, 999999))


def generate_clean_session(
    persona: Persona,
    agent_id: str,
    base_date: datetime,
    seed: int,
    session_index: int = 0,
) -> Session:
    """
    Generate one clean (non-attacked) session for a persona.
    Fully deterministic: same seed + session_index → same session.
    """
    rng = random.Random(seed + session_index * 997)
    config = PERSONA_CONFIG[persona]

    session = Session(
        session_id=str(uuid.UUID(int=rng.getrandbits(128))),
        persona=persona,
        agent_id=agent_id,
        seed=seed,
        created_at=base_date,
        is_clean=True,
    )

    svc = _pick_service(persona, rng)
    seq = config["sequence"]
    n_events = rng.randint(*config["session_length"])
    t = _active_timestamp(base_date, config, rng)

    for i, action in enumerate(seq[:n_events]):
        amount = None
        original_request = None

        if action == ActionType.AUTHORIZE:
            amount = rng.randint(*svc["amount_range"])
            # Normal original_request — no injection artifacts
            original_request = {
                "service": svc["endpoint"],
                "operation": svc["operation_id"],
                "query": _normal_query(persona, svc["operation_id"], rng),
            }

        event = session.make_event(
            action_type=action,
            service_id=svc["service_id"],
            operation_id=svc["operation_id"],
            raw_endpoint=svc["endpoint"] if action == ActionType.AUTHORIZE else None,
            amount_units=amount,
            vendor=svc["vendor"] if action == ActionType.AUTHORIZE else None,
            category=next((c for c, svcs in PROD_SERVICES.items()
                           if svc in svcs), "search"),
            original_request=original_request,
            timestamp=t,
        )
        session.events.append(event)
        t += timedelta(seconds=rng.randint(1, 30))

    return session


def generate_agent_history(
    persona: Persona,
    agent_id: str,
    n_sessions: int,
    start_date: datetime,
    seed: int,
) -> list[Session]:
    """
    Generate a full history of clean sessions for one agent.
    Used to build L3 behavioral baselines before injecting attacks.
    The more sessions, the stronger the baseline.
    Recommended: n_sessions >= 100 for L3 to stabilize.
    """
    sessions = []
    config = PERSONA_CONFIG[persona]
    rng = random.Random(seed)

    current_date = start_date
    for i in range(n_sessions):
        session = generate_clean_session(persona, agent_id, current_date, seed, i)
        sessions.append(session)
        # Advance date by cadence
        sessions_today = rng.randint(*config["sessions_per_day"]) or 1
        hours_between = 24 / sessions_today
        current_date += timedelta(hours=hours_between + rng.uniform(-1, 1))

    return sessions


def _normal_query(persona: Persona, operation_id: str, rng: random.Random) -> str:
    """Realistic query strings that look like normal agent requests."""
    research_queries = [
        "AAPL Q2 2026 earnings sentiment",
        "Fed rate decision June 2026 probability",
        "BTC price action last 24h",
        "ETH/USDC liquidity depth Uniswap",
        "commodity stress macro regime signal",
        "currency stress EUR/USD cascade",
        "Polymarket open interest top markets",
    ]
    procurement_queries = [
        "office supplies vendor approved list",
        "industrial parts SKU-X reorder",
        "SaaS license renewal vendor Y",
        "data storage procurement Q3",
    ]
    travel_queries = [
        "SFO to NRT business class under 8000",
        "hotel NYC Midtown 3 nights",
        "car rental LAX economy",
        "flight JFK to LHR 2 passengers",
    ]
    if persona == Persona.RESEARCH:
        return rng.choice(research_queries)
    elif persona == Persona.PROCUREMENT:
        return rng.choice(procurement_queries)
    else:
        return rng.choice(travel_queries)
