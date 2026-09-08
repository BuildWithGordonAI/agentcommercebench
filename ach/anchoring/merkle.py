"""
Minimal Merkle tree for batch CAR anchoring.

Builds a balanced binary tree over a list of leaf hashes.
Returns the root hash and inclusion proofs for any leaf.

Production usage:
  1. Collect N CARs into a batch.
  2. Build MerkleTree(leaf_hashes).
  3. Anchor root = tree.root to Ethereum OP_RETURN / Solana tx memo.
  4. Store tree.prove(i) alongside each CAR for future verification.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass


def _sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _pair_hash(left: str, right: str) -> str:
    return _sha256(left + right)


@dataclass
class MerkleProof:
    leaf_hash: str
    path:      list[tuple[str, str]]   # list of (sibling_hash, "left"|"right")
    root:      str

    def verify(self) -> bool:
        current = self.leaf_hash
        for sibling, side in self.path:
            if side == "right":
                current = _pair_hash(current, sibling)
            else:
                current = _pair_hash(sibling, current)
        return current == self.root


class MerkleTree:
    def __init__(self, leaves: list[str]):
        if not leaves:
            raise ValueError("need at least one leaf")
        self._leaves = list(leaves)
        self._layers = self._build(list(leaves))

    @property
    def root(self) -> str:
        return self._layers[-1][0]

    def prove(self, index: int) -> MerkleProof:
        path: list[tuple[str, str]] = []
        idx = index
        for layer in self._layers[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1
                side = "right"
            else:
                sibling_idx = idx - 1
                side = "left"
            sibling_idx = min(sibling_idx, len(layer) - 1)
            path.append((layer[sibling_idx], side))
            idx //= 2
        return MerkleProof(
            leaf_hash = self._leaves[index],
            path      = path,
            root      = self.root,
        )

    @staticmethod
    def _build(layer: list[str]) -> list[list[str]]:
        layers = [layer]
        while len(layer) > 1:
            next_layer: list[str] = []
            for i in range(0, len(layer), 2):
                left  = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else layer[i]
                next_layer.append(_pair_hash(left, right))
            layers.append(next_layer)
            layer = next_layer
        return layers
