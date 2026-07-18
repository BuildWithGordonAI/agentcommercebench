"""
Recorder — imports real Gordon prod transactions as replayable sessions.

Prod sessions are unlabeled (is_clean=True by default).
Once recorded, any session can be:
  1. Replayed as-is (clean baseline)
  2. Cloned + mutated with inject() to produce a labeled attack session
  3. Used to seed L3 behavioral baselines

Usage:
    sessions = record_from_prod(conn, agent_id="agent_yzmxwq3v48")
    Session.save(sessions[0], "data/sessions/prod_agent_yzmxwq3v48_001.json")

    # Later: replay clean
    result = replay(sessions[0], pipeline=GORDON_PIPELINE)

    # Or mutate and replay
    attacked = inject(sessions[0], scenario="A1")
    result = replay(attacked, pipeline=GORDON_PIPELINE)
"""
import psycopg2, json, uuid
from datetime import datetime
from pathlib import Path
from .schema import Session, Event, ActionType, Persona


DB_CONFIG = dict(
    host="gordon-postgres.cc1muq0ccfo1.us-east-1.rds.amazonaws.com",
    port=5432, database="gordon", user="gordon",
    password="JxukVDwc3pvlwd5T4Rtk1qkohgypzlYx",
    sslmode="require", connect_timeout=10,
)


def _infer_persona(category_counts: dict) -> Persona:
    top = max(category_counts, key=category_counts.get, default="search")
    if top in ("finance", "ai"):
        return Persona.RESEARCH
    if top == "travel":
        return Persona.TRAVEL
    return Persona.PROCUREMENT


def record_from_prod(
    agent_id: str = None,
    session_window_minutes: int = 30,
    conn=None,
) -> list[Session]:
    """
    Pull all transactions from prod and group them into sessions
    by agent_id + time gap (> session_window_minutes = new session).

    Returns list of Session objects, all marked is_clean=True.
    """
    close_conn = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        close_conn = True

    cur = conn.cursor()
    query = """
        SELECT t.id, t.agent_id, t.user_id, t.created_at,
               t.request, t.decision,
               ss.service_id, ss.operation_id, ss.raw_endpoint,
               ss.amount_units, ss.protocol, ss.network,
               ss.receipt_status, ss.risk_score, ss.risk_flags
        FROM transactions t
        LEFT JOIN service_settlements ss ON ss.transaction_id = t.id
        {where}
        ORDER BY t.agent_id, t.created_at
    """
    where = f"WHERE t.agent_id = %s" if agent_id else ""
    params = (agent_id,) if agent_id else ()
    cur.execute(query.format(where=where), params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    if close_conn:
        conn.close()

    # Group by agent_id + session window
    by_agent: dict[str, list] = {}
    for row in rows:
        r = dict(zip(cols, row))
        aid = r['agent_id']
        by_agent.setdefault(aid, []).append(r)

    sessions = []
    for aid, txns in by_agent.items():
        category_counts: dict[str, int] = {}
        current_session_rows = []
        last_ts = None

        def flush_session(rows, aid, category_counts):
            if not rows:
                return None
            persona = _infer_persona(category_counts)
            s = Session(
                session_id=str(uuid.uuid4()),
                persona=persona,
                agent_id=aid,
                seed=hash(aid + str(rows[0]['created_at'])) & 0xFFFFFFFF,
                created_at=rows[0]['created_at'],
                is_clean=True,
            )
            for r in rows:
                req = r.get('request') or {}
                meta = req.get('metadata', {})
                decision_raw = r.get('decision') or {}
                cat = req.get('category', 'search')
                action_type = ActionType.AUTHORIZE  # all prod rows are authorizations

                e = Event(
                    event_id=r['id'],
                    session_id=s.session_id,
                    agent_id=aid,
                    action_type=action_type,
                    timestamp=r['created_at'],
                    service_id=r.get('service_id') or meta.get('service_id'),
                    operation_id=r.get('operation_id') or meta.get('operation_id'),
                    amount_units=r.get('amount_units') or req.get('amount'),
                    vendor=req.get('vendor'),
                    category=cat,
                    raw_endpoint=r.get('raw_endpoint') or meta.get('raw_endpoint'),
                    original_request=req,
                    network=r.get('network', 'eip155:8453'),
                    is_injected=False,
                )
                s.events.append(e)
            return s

        for r in txns:
            ts = r['created_at']
            if (last_ts is not None and
                    (ts - last_ts).total_seconds() > session_window_minutes * 60):
                s = flush_session(current_session_rows, aid, category_counts)
                if s:
                    sessions.append(s)
                current_session_rows = []
                category_counts = {}
            current_session_rows.append(r)
            cat = (r.get('request') or {}).get('category', 'search')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            last_ts = ts

        s = flush_session(current_session_rows, aid, category_counts)
        if s:
            sessions.append(s)

    return sessions


def save_all(sessions: list[Session], output_dir: str = "data/sessions"):
    """Save all sessions to disk for replayability."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in sessions:
        fname = f"{s.agent_id}_{s.session_id[:8]}.json"
        path = str(out / fname)
        s.save(path)
        paths.append(path)
    print(f"Saved {len(paths)} sessions → {output_dir}/")
    return paths


def load_all(session_dir: str = "data/sessions") -> list[Session]:
    """Load all recorded sessions from disk."""
    return [Session.load(str(p)) for p in Path(session_dir).glob("*.json")]
