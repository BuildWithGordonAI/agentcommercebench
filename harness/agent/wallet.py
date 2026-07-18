"""
Wallet simulator — signs x402 payment authorizations without spending real USDC.

In prod, Gordon's wallet (0xf9E1b03739b6a4B25Dac70542895515A078CB943) signs
EIP-712 transfer-with-authorization messages on Base mainnet.

This simulator:
  - Generates a fresh deterministic test wallet per agent_id (no real funds)
  - Signs the same EIP-712 message structure Gordon expects
  - Works against Gordon's dev backend (testnet) or a mock
  - For prod replay: can load recorded signatures from session files

The wallet address changes per agent_id so each replicated prod agent
has its own identity without sharing keys.
"""
import os, hashlib, time, secrets
from eth_account import Account
from eth_account.messages import encode_typed_data


# USDC contract on Base mainnet
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
BASE_CHAIN_ID = 8453

# EIP-712 domain + type for USDC transferWithAuthorization
EIP712_DOMAIN = {
    "name": "USD Coin",
    "version": "2",
    "chainId": BASE_CHAIN_ID,
    "verifyingContract": USDC_BASE,
}

EIP712_TYPES = {
    "TransferWithAuthorization": [
        {"name": "from",        "type": "address"},
        {"name": "to",          "type": "address"},
        {"name": "value",       "type": "uint256"},
        {"name": "validAfter",  "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce",       "type": "bytes32"},
    ]
}


class WalletSimulator:
    """
    One wallet per agent. Deterministic from agent_id so replay is consistent.
    """

    def __init__(self, agent_id: str, private_key: str = None):
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            # Derive deterministic key from agent_id (test only — not secure)
            seed = hashlib.sha256(f"gordon-test-wallet:{agent_id}".encode()).digest()
            self.account = Account.from_key(seed)

        self.agent_id = agent_id
        self.address = self.account.address

    def sign_payment(
        self,
        pay_to: str,
        amount_units: int,
        valid_window_seconds: int = 300,
    ) -> dict:
        """
        Sign an x402 payment authorization.
        Returns the full payment_response_header Gordon expects.
        """
        valid_after = 0
        valid_before = int(time.time()) + valid_window_seconds
        nonce = "0x" + secrets.token_hex(32)

        message_data = {
            "from":        self.address,
            "to":          pay_to,
            "value":       amount_units,
            "validAfter":  valid_after,
            "validBefore": valid_before,
            "nonce":       bytes.fromhex(nonce[2:]),
        }

        signed = Account.sign_typed_data(
            self.account.key,
            full_message={
                "domain": EIP712_DOMAIN,
                "types": EIP712_TYPES,
                "primaryType": "TransferWithAuthorization",
                "message": message_data,
            }
        )

        return {
            "scheme":       "exact",
            "network":      "eip155:8453",
            "x402Version":  2,
            "payload": {
                "signature": signed.signature.hex()
                             if isinstance(signed.signature, bytes)
                             else signed.signature,
                "authorization": {
                    "from":        self.address,
                    "to":          pay_to,
                    "value":       str(amount_units),
                    "validAfter":  str(valid_after),
                    "validBefore": str(valid_before),
                    "nonce":       nonce,
                }
            }
        }

    def __repr__(self):
        return f"WalletSimulator(agent={self.agent_id}, addr={self.address})"


def wallet_for(agent_id: str) -> WalletSimulator:
    """Get (or create) a test wallet for a given agent_id."""
    return WalletSimulator(agent_id)


def wallet_from_prod_session(recorded_signature: dict) -> dict:
    """
    For replay mode: return the exact signature from a recorded session
    instead of signing fresh. Used when replaying prod sessions exactly.
    """
    return recorded_signature
