"""
Question store + the three indexes that make candidate retrieval sub-linear:

  exact bucket   sha1(canonical text)        -> ids     O(1)
  head bucket    head-noun lemma             -> ids     O(bucket)
  inverted idx   content feature             -> postings   (classic IR; no full scan)

Tags are *schemaless* ({"subject": "physics", "grade": 10, anything: anything}) and are
stored but NEVER consulted for duplicate decisions - only for optional scoping/UI.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import embed
from .frame import Frame, parse
from .normalize import exact_key


@dataclass
class Question:
    id: str
    text: str
    tags: dict[str, Any] = field(default_factory=dict)
    frame: Frame | None = None
    key: str = ""
    vec: dict = field(default_factory=dict)


class QuestionStore:
    def __init__(self, max_posting_frac: float = 0.25, min_posting_cap: int = 400):
        self.items: dict[str, Question] = {}
        self.by_key: dict[str, list[str]] = {}
        self.by_head: dict[str, list[str]] = {}
        self.postings: dict[str, list[tuple[str, float]]] = {}
        self.max_posting_frac = max_posting_frac
        self.min_posting_cap = min_posting_cap
        self._ids = itertools.count(1)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items.values())

    def add(self, text: str, tags: dict | None = None, id: str | None = None) -> Question:
        qid = id or f"Q{next(self._ids)}"
        while qid in self.items:
            qid = f"Q{next(self._ids)}"
        q = Question(qid, text, dict(tags or {}), parse(text), exact_key(text), embed.encode(text))
        self.items[qid] = q
        self.by_key.setdefault(q.key, []).append(qid)
        head = q.frame.concept.head
        if head:
            self.by_head.setdefault(head, []).append(qid)
        for f, w in q.vec.items():
            self.postings.setdefault(f, []).append((qid, w))
        return q

    def add_many(self, texts: Iterable[str]) -> list[Question]:
        return [self.add(t) for t in texts]

    def neighbors(self, vec: dict, k: int = 25) -> list[tuple[str, float]]:
        """Top-k by content cosine using the inverted index (skips ubiquitous features)."""
        n = len(self.items)
        cap_word = max(self.min_posting_cap, int(self.max_posting_frac * n))      # word / bigram features: selective
        cap_char = 300 + int(0.02 * n)                                             # char-gram features: prune the common ones
        acc: dict[str, float] = {}
        for f, w in vec.items():
            plist = self.postings.get(f)
            if not plist or len(plist) > (cap_char if f.startswith("c:") else cap_word):
                continue
            for qid, qw in plist:
                acc[qid] = acc.get(qid, 0.0) + w * qw
        return sorted(acc.items(), key=lambda kv: -kv[1])[:k]

    def candidates(self, text: str, frame: Frame | None = None, k: int = 25) -> dict[str, float]:
        """Union of exact hits, same-head bucket and content neighbours -> {id: content_sim}."""
        frame = frame or parse(text)
        vec = embed.encode(text)
        ids: dict[str, float] = {}
        for qid in self.by_key.get(exact_key(text), []):
            ids[qid] = 1.0
        for qid in self.by_head.get(frame.concept.head, []) if frame.concept.head else []:
            ids.setdefault(qid, embed.cosine(vec, self.items[qid].vec))
        for qid, s in self.neighbors(vec, k):
            ids.setdefault(qid, s)
        return ids
