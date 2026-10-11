"""
Generate -> verify -> feedback loop.

    +------------------+     prompt (bounded)     +-----+
    | CoverageMap      | -----------------------> | LLM |
    | (what's covered, |                          +-----+
    |  what's open)    |                             | candidates
    +------------------+                             v
            ^                         +--------------------------+
            | feedback ("X repeats Y")|  DuplicateDetector       |  <- independent safety net
            +-------------------------|  (never sees the prompt) |
                                      +--------------------------+
                                         accept | hold(review) | reject

Prevention shapes WHAT the LLM is asked; the detector decides what is ALLOWED IN.  Either can
fail without the other noticing - that is the point.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from .coverage import DEFAULT_INTENTS, CoverageMap, PromptStats, build_prompt
from .detector import DuplicateDetector, Label, Verdict
from .frame import Intent
from .llm import LLM


@dataclass
class Rejection:
    text: str
    verdict: Verdict
    round: int


@dataclass
class RoundStats:
    round: int
    prompt: PromptStats
    candidates: int
    accepted: int
    held: int
    rejected: int


@dataclass
class GenerationReport:
    accepted: list = field(default_factory=list)
    held_for_review: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    rounds: list = field(default_factory=list)
    llm_calls: int = 0

    @property
    def first_pass_rate(self) -> float:
        r = self.rounds[0] if self.rounds else None
        return r.accepted / r.candidates if r and r.candidates else 0.0

    def summary(self) -> str:
        total = sum(r.candidates for r in self.rounds)
        return (f"{len(self.accepted)} accepted, {len(self.held_for_review)} held for review, {len(self.rejected)} rejected "
                f"of {total} candidates over {len(self.rounds)} round(s) / {self.llm_calls} LLM call(s); "
                f"first-pass acceptance {self.first_pass_rate:.0%}; "
                f"prompt tokens per round: {[r.prompt.tokens for r in self.rounds]}")


def parse_candidates(raw: str) -> list[str]:
    raw = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in data:
        q = item.get("question") if isinstance(item, dict) else item
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out


class QuestionGenerator:
    def __init__(self, detector: DuplicateDetector, llm: LLM, *, budget_tokens: int = 900,
                 review_policy: str = "hold", intents: Sequence[Intent] = tuple(DEFAULT_INTENTS)):
        assert review_policy in ("hold", "accept", "reject")
        self.detector, self.store, self.llm = detector, detector.store, llm
        self.budget, self.policy, self.intents = budget_tokens, review_policy, intents
        self.coverage = CoverageMap(self.store)

    def generate(self, subject: str, n: int, facets: Sequence = (), tags: dict | None = None,
                 max_rounds: int = 4) -> GenerationReport:
        rep = GenerationReport()
        feedback: list[str] = []
        for rnd in range(1, max_rounds + 1):
            need = n - len(rep.accepted)
            if need <= 0:
                break
            self.coverage.sync()
            prompt, pstats, targets = build_prompt(subject, need, self.coverage, facets, self.intents,
                                                   self.budget, feedback=feedback)
            raw = self.llm.complete(prompt)
            rep.llm_calls += 1
            acc = held = rej = 0
            cands = parse_candidates(raw)
            for text in cands:
                if len(rep.accepted) >= n:
                    break
                v = self.detector.check(text)       # also sees everything accepted earlier in this run
                if v.label is Label.DISTINCT or (v.label is Label.REVIEW and self.policy == "accept"):
                    t = dict(tags or {}, source="llm", subject=subject)
                    if v.label is Label.REVIEW:
                        t["needs_review"] = True
                    q = self.store.add(text, t)
                    rep.accepted.append(q)
                    acc += 1
                elif v.label is Label.REVIEW and self.policy == "hold":
                    rep.held_for_review.append(v)
                    held += 1
                else:
                    rep.rejected.append(Rejection(text, v, rnd))
                    rej += 1
                    b = v.best
                    feedback.append(f"\"{text}\" repeats {b.question.id} ({v.frame.signature().split('|')[0]} | "
                                    f"{v.frame.concept.surface or v.frame.concept.label()}) - pick a different cell")
            rep.rounds.append(RoundStats(rnd, pstats, len(cands), acc, held, rej))
        return rep
