"""
Load .env and resolve the active AWS key slot into AWS_ACCESS_KEY_ID / SECRET.

Usage:
    import load_env  # noqa — call at top of game.py / any entrypoint
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_root = Path(__file__).parent
load_dotenv(_root / ".env", override=True)

slot = os.getenv("AWS_KEY_SLOT", "1")
key_id = os.getenv(f"AWS_ACCESS_KEY_ID_{slot}")
secret  = os.getenv(f"AWS_SECRET_ACCESS_KEY_{slot}")

if key_id:
    os.environ["AWS_ACCESS_KEY_ID"] = key_id
if secret:
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret

ADVERSARY_MODEL: str = (
    os.getenv("ADVERSARY_MODEL")
    or os.getenv("BEDROCK_MODEL_HAIKU", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
)

GORDON_MCP_URL:   str = os.getenv("GORDON_MCP_URL", "https://api.withgordon.ai/mcp")
GORDON_AGENT_KEY: str = os.getenv("GORDON_AGENT_KEY", "")
