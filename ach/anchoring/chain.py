"""
Hash-chained CAR log — tamper-evident audit trail.

Each entry signs its own content AND the previous entry's hash,
forming a chain where any retroactive modification breaks all subsequent hashes.

Structure:
  entry_0: car_hash | prev_hash=GENESIS | entry_signature
  entry_1: car_hash | prev_hash=entry_0.entry_hash | entry_signature
  ...
  entry_N: car_hash | prev_hash=entry_(N-1).entry_hash | entry_signature

Merkle anchoring:
  Call batch_merkle_root() periodically to get a root hash for on-chain anchoring.
  Call prove(car_id) to get a Merkle inclusion proof for any CAR.

Verification:
  chain.verify() — checks every prev_hash link and every entry signature.
  proof.verify() — checks a Merkle inclusion proof without revealing other CARs.
"""
from __future__ import annotations
import hashlib, hmac, json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ach.anchoring.merkle import MerkleTree, MerkleProof

GENESIS_HASH = "0" * 64   # sentinel for the first entry


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _hmac_sign(payload: dict, key: bytes = b"ach-log-key") -> str:
    body = json.dumps(payload, sort_keys=True)
    return hmac.new(key, body.encode(), hashlib.sha256).hexdigest()


@dataclass
class CARLogEntry:
    seq:              int
    car_id:           str
    ceremony_level:   str
    anchoring_level:  str
    car_hash:         str         # SHA-256 of the CAR payload
    prev_hash:        str         # hash of the previous entry (chain link)
    entry_hash:       str         # SHA-256 of this entry's content
    entry_signature:  str         # HMAC-SHA256 of this entry
    timestamp:        str

    # Populated after Merkle batch anchoring
    merkle_leaf:  Optional[str] = None
    merkle_root:  Optional[str] = None
    chain_tx:     Optional[str] = None   # blockchain tx id when anchored


@dataclass
class CARLog:
    """
    Append-only, hash-chained log of Commerce Action Receipts.

    One log per session (or global — caller's choice).
    """
    session_id: str
    _entries:   list[CARLogEntry] = field(default_factory=list)
    _sign_key:  bytes = field(default=b"ach-log-key")

    @property
    def entries(self) -> list[CARLogEntry]:
        return list(self._entries)

    @property
    def head_hash(self) -> str:
        if not self._entries:
            return GENESIS_HASH
        return self._entries[-1].entry_hash

    def append(
        self,
        car_id:         str,
        car_payload:    dict,
        ceremony_level: str,
        anchoring_level: str,
    ) -> CARLogEntry:
        car_hash  = _sha256(json.dumps(car_payload, sort_keys=True, default=str))
        prev_hash = self.head_hash
        timestamp = datetime.now(timezone.utc).isoformat()
        seq       = len(self._entries)

        entry_content = {
            "seq":             seq,
            "car_id":          car_id,
            "car_hash":        car_hash,
            "prev_hash":       prev_hash,
            "ceremony_level":  ceremony_level,
            "anchoring_level": anchoring_level,
            "timestamp":       timestamp,
        }
        entry_hash = _sha256(json.dumps(entry_content, sort_keys=True))
        signature  = _hmac_sign(entry_content, self._sign_key)

        entry = CARLogEntry(
            seq              = seq,
            car_id           = car_id,
            ceremony_level   = ceremony_level,
            anchoring_level  = anchoring_level,
            car_hash         = car_hash,
            prev_hash        = prev_hash,
            entry_hash       = entry_hash,
            entry_signature  = signature,
            timestamp        = timestamp,
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, str]:
        """
        Verify the entire chain.  Returns (True, "ok") or (False, reason).
        """
        expected_prev = GENESIS_HASH
        for e in self._entries:
            if e.prev_hash != expected_prev:
                return False, f"chain break at seq={e.seq}: expected {expected_prev[:8]}… got {e.prev_hash[:8]}…"
            # re-derive entry_hash
            content = {
                "seq": e.seq, "car_id": e.car_id, "car_hash": e.car_hash,
                "prev_hash": e.prev_hash, "ceremony_level": e.ceremony_level,
                "anchoring_level": e.anchoring_level, "timestamp": e.timestamp,
            }
            expected_hash = _sha256(json.dumps(content, sort_keys=True))
            if e.entry_hash != expected_hash:
                return False, f"entry_hash mismatch at seq={e.seq}"
            expected_sig = _hmac_sign(content, self._sign_key)
            if e.entry_signature != expected_sig:
                return False, f"signature mismatch at seq={e.seq}"
            expected_prev = e.entry_hash
        return True, "ok"

    def batch_merkle_root(self, tag_entries: bool = True) -> tuple[str, MerkleTree]:
        """
        Build a Merkle tree over all entry hashes.
        Optionally writes merkle_leaf and merkle_root back to each entry.
        Returns (root_hash, tree).
        """
        hashes = [e.entry_hash for e in self._entries]
        tree   = MerkleTree(hashes)
        if tag_entries:
            for i, e in enumerate(self._entries):
                e.merkle_leaf = hashes[i]
                e.merkle_root = tree.root
        return tree.root, tree

    def prove(self, car_id: str) -> Optional[MerkleProof]:
        """Return a Merkle inclusion proof for the given car_id."""
        hashes = [e.entry_hash for e in self._entries]
        tree   = MerkleTree(hashes)
        for i, e in enumerate(self._entries):
            if e.car_id == car_id:
                return tree.prove(i)
        return None

    def mock_anchor_to_chain(self, tx_id: str | None = None) -> str:
        """
        Simulate on-chain anchoring.  In production: write merkle_root to
        Ethereum OP_RETURN or Solana tx memo and store the tx_id.
        Returns the mock tx_id.
        """
        root, _ = self.batch_merkle_root(tag_entries=True)
        tx      = tx_id or f"0xmock_{root[:16]}"
        for e in self._entries:
            e.chain_tx = tx
        return tx

    def __len__(self) -> int:
        return len(self._entries)
