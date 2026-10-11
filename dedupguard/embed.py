"""
Voter 3 / retrieval signal - content similarity.

Deliberately *IDF-free* and *stateless*: a question's vector never changes when the
corpus grows, so the index is append-only and there is nothing to re-embed.
Template words ("what is", "define", "explain the meaning of") are stripped first so
two questions are compared on *content*, not on how they are phrased.

Features: content-lemma unigrams + adjacent bigrams + character trigrams of each
lemma (the trigrams give typo / morphology tolerance: 'reflection' ~ 'reflexion').

A neural embedder can be dropped in via `DenseEmbedder` (see bottom of file); it is
used as an *additional* voter, never as a replacement.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from .normalize import content_lemmas, tokens

TEMPLATE_WORDS = frozenset("""define definition meaning understand explain describe state list name mention give write
find calculate compute determine difference differences differentiate distinguish compare term concept
short note brief discuss elaborate tell example""".split())

SparseVec = dict  # feature -> weight (L2 normalised)


def content_text(text: str) -> list[str]:
    return [w for w in content_lemmas(text) if w not in TEMPLATE_WORDS]


def encode(text: str) -> SparseVec:
    ws = content_text(text)
    feats: dict[str, float] = {}
    for w in ws:
        feats["w:" + w] = feats.get("w:" + w, 0.0) + 1.0
        padded = f"#{w}#"
        for i in range(len(padded) - 2):
            k = "c:" + padded[i:i + 3]
            feats[k] = feats.get(k, 0.0) + 0.25
    for a, b in zip(ws, ws[1:]):
        feats[f"b:{a}_{b}"] = feats.get(f"b:{a}_{b}", 0.0) + 0.5
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {k: v / norm for k, v in feats.items()}


def cosine(a: SparseVec, b: SparseVec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())



class DenseEmbedder:
    """Wrap any `encode_fn(list[str]) -> list[list[float]]` (sentence-transformers, OpenAI, Voyage, ...).

    Example:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        dense = DenseEmbedder(lambda xs: m.encode(xs, normalize_embeddings=True))
    """

    def __init__(self, encode_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]]):
        self.encode_fn = encode_fn
        self._cache: dict[str, list[float]] = {}

    def vec(self, text: str) -> list[float]:
        if text not in self._cache:
            v = list(self.encode_fn([text])[0])
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            self._cache[text] = [x / n for x in v]
        return self._cache[text]

    def sim(self, a: str, b: str) -> float:
        va, vb = self.vec(a), self.vec(b)
        return sum(x * y for x, y in zip(va, vb))
