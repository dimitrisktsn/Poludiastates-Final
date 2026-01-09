from __future__ import annotations
from typing import List, Set, Sequence, Dict, Tuple, Iterable
import hashlib
import random

class MinHashLSH:
    
    "Υλοποίηση MinHash + LSH για σύνολα tokens, με determinism και banding buckets."
    def __init__(
        self,
        num_perm: int = 64,
        num_bands: int = 8,
        seed: int = 42,
        fallback_all: bool = False,
    ):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.seed = seed
        self.fallback_all = fallback_all

        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"
        self.rows_per_band = num_perm // num_bands

        random.seed(seed)
        self.permutations = [
            (random.randint(1, 2**31 - 1), random.randint(0, 2**31 - 1))
            for _ in range(num_perm)
        ]

        self.doc_tokens: List[Set[str]] = []
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}

    def _hash_token(self, token: str) -> int:
        return int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)

    def _minhash_signature(self, tokens: Set[str]) -> List[int]:
        if not tokens:
            return [2**63 - 1] * self.num_perm

        hashed_tokens = [self._hash_token(t) for t in tokens]
        signature = []

        for a, b in self.permutations:
            min_val = min((a * x + b) % (2**31 - 1) for x in hashed_tokens)
            signature.append(min_val)

        return signature

    def _band_hashes(self, signature: List[int]) -> List[Tuple[int, Tuple[int, ...]]]:
        band_hashes = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(signature[start:end])
            band_hashes.append((band_idx, band_tuple))
        return band_hashes

    def fit(self, documents: Sequence[Iterable[str]]):
        self.doc_tokens = [set(doc) for doc in documents]
        self.buckets.clear()

        for doc_id, tokens in enumerate(self.doc_tokens):
            sig = self._minhash_signature(tokens)
            for band_key in self._band_hashes(sig):
                self.buckets.setdefault(band_key, []).append(doc_id)

    def query(self, query_tokens: Iterable[str], top_n: int = 10) -> List[int]:
        qset = set(query_tokens)
        qsig = self._minhash_signature(qset)

        candidates: Set[int] = set()
        for band_key in self._band_hashes(qsig):
            for doc_id in self.buckets.get(band_key, []):
                candidates.add(doc_id)

        if self.fallback_all and not candidates:
            candidates = set(range(len(self.doc_tokens)))

        scored = []
        for doc_id in candidates:
            sim = self.jaccard(qset, self.doc_tokens[doc_id])
            scored.append((sim, doc_id))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc_id for sim, doc_id in scored][:top_n]

    @staticmethod
    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union


