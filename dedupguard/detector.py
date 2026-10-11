"""
Independent duplicate detector - a cascade of *decorrelated voters*.

    V1  exact      canonical-text hash                 catches formatting / punctuation / case / numbering noise
    V2  frame      intent x concept x numbers          catches paraphrase; VETOES false friends ("laws of X" vs "what is X")
    V3  lexical    content-lemma + char-gram cosine    catches what the parser cannot parse; guards against parser blind spots
    V4  dense      optional neural embedding           extra paraphrase recall when a model is available
    V5  judge      optional LLM, gray-zone only        arbitrates REVIEW cases (cost-bounded: only called when needed)

This module never reads the generator's prompt, coverage map, or any generation-time
state.  It only needs a QuestionStore (the raw questions), so it still protects you when
the prevention layer fails, is bypassed, or when questions arrive from another source
(imports, human authors, a different LLM).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Protocol

from . import embed
from .frame import Frame, Rel, compare, extract_numbers, parse
from .index import Question, QuestionStore


class Label(IntEnum):
    DISTINCT = 0
    REVIEW = 1
    SEMANTIC_DUPLICATE = 2
    EXACT_DUPLICATE = 3

    @property
    def is_duplicate(self) -> bool:
        return self >= Label.SEMANTIC_DUPLICATE


class Judge(Protocol):
    def judge(self, a: str, b: str) -> tuple[str, str]:  # ("duplicate"|"different"|"unsure", rationale)
        ...


@dataclass
class Evidence:
    voter: str
    vote: str
    detail: str


@dataclass
class Match:
    question: Question
    label: Label
    confidence: float
    content_sim: float
    evidence: list[Evidence] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    text: str
    label: Label
    confidence: float
    matches: list[Match]
    frame: Frame
    candidates_examined: int = 0

    @property
    def best(self) -> Optional[Match]:
        return self.matches[0] if self.matches else None

    def explain(self, limit: int = 3) -> str:
        lines = [f"{self.label.name}  (confidence {self.confidence:.2f})   frame={self.frame.signature()}"]
        for m in self.matches[:limit]:
            lines.append(f"  vs {m.question.id}: {m.question.text!r} -> {m.label.name} ({m.confidence:.2f}) sim={m.content_sim:.2f}")
            for e in m.evidence:
                lines.append(f"       [{e.voter:<7}] {e.vote:<9} {e.detail}")
        return "\n".join(lines)


@dataclass
class Thresholds:
    lex_dup: float = 0.90          # lexical-only duplicate (used only when the frame parser is unsure)
    lex_review: float = 0.65
    safety_net: float = 0.95       # near-identical content words but frame said DISTINCT -> escalate to REVIEW
    dissent_floor: float = 0.20    # frame says DUP but content similarity this low -> downgrade to REVIEW
    dense_dup: float = 0.90
    dense_review: float = 0.78


class DuplicateDetector:
    def __init__(self, store: QuestionStore | None = None, *, dense: "embed.DenseEmbedder | None" = None,
                 judge: Judge | None = None, thresholds: Thresholds | None = None, top_k: int = 25):
        self.store = store if store is not None else QuestionStore()
        self.dense, self.judge, self.th, self.top_k = dense, judge, thresholds or Thresholds(), top_k
        self.judge_calls = 0

    # ---- corpus management -------------------------------------------------
    def add(self, text: str, tags: dict | None = None, id: str | None = None) -> Question:
        return self.store.add(text, tags, id)

    # ---- single pairwise decision -------------------------------------------
    def compare_pair(self, cand_text: str, cand_frame: Frame, other: Question, sim: float) -> Match:
        ev: list[Evidence] = []
        flags: list[str] = []
        # V1 ---------------------------------------------------------------
        from .normalize import exact_key
        if exact_key(cand_text) == other.key:
            ev.append(Evidence("exact", "DUP", "identical after canonicalisation (case/punctuation/numbering/format)"))
            return Match(other, Label.EXACT_DUPLICATE, 1.0, sim, ev)
        ev.append(Evidence("exact", "no", "texts differ after canonicalisation"))
        # V2 ---------------------------------------------------------------
        fr = compare(cand_frame, other.frame)
        flags += fr.flags
        ev.append(Evidence("frame", fr.rel.value, "; ".join(fr.reasons)))
        # V3 ---------------------------------------------------------------
        lex_vote = "DUP" if sim >= self.th.lex_dup else "REVIEW" if sim >= self.th.lex_review else "no"
        ev.append(Evidence("lexical", lex_vote, f"content cosine {sim:.2f}"))
        # V4 ---------------------------------------------------------------
        dsim = None
        if self.dense is not None:
            dsim = self.dense.sim(cand_text, other.text)
            dv = "DUP" if dsim >= self.th.dense_dup else "REVIEW" if dsim >= self.th.dense_review else "no"
            ev.append(Evidence("dense", dv, f"embedding cosine {dsim:.2f}"))

        # ---- combine ------------------------------------------------------
        if fr.rel is Rel.DUP:
            label, conf = Label.SEMANTIC_DUPLICATE, fr.score
            dense_dissent = dsim is not None and dsim < 0.4
            if sim < self.th.dissent_floor or dense_dissent:
                label, conf = Label.REVIEW, 0.5
                ev.append(Evidence("cascade", "downgrade", "frame says duplicate but content voters strongly dissent"))
        elif fr.rel is Rel.REVIEW:
            label, conf = Label.REVIEW, fr.score
            # dense model can promote an ambiguous case only if it is *very* confident
            if dsim is not None and dsim >= 0.95 and sim >= self.th.lex_dup:
                label, conf = Label.SEMANTIC_DUPLICATE, 0.7
                ev.append(Evidence("cascade", "upgrade", "frame ambiguous; lexical and dense voters both near-identical"))
        elif fr.rel is Rel.DISTINCT:
            label, conf = Label.DISTINCT, 1.0 - fr.score
            if sim >= self.th.safety_net and extract_numbers(cand_text) == extract_numbers(other.text) and not flags:
                label, conf = Label.REVIEW, 0.5
                ev.append(Evidence("cascade", "escalate", "frame says distinct but content words are near-identical - parser blind spot?"))
        else:  # UNKNOWN: parser abstained, lexical (+dense) decide
            if sim >= self.th.lex_dup or (dsim is not None and dsim >= self.th.dense_dup and sim >= self.th.lex_review):
                label, conf = Label.SEMANTIC_DUPLICATE, max(sim, dsim or 0.0)
            elif sim >= self.th.lex_review or (dsim is not None and dsim >= self.th.dense_review):
                label, conf = Label.REVIEW, max(sim, dsim or 0.0)
            else:
                label, conf = Label.DISTINCT, 1.0 - sim
            ev.append(Evidence("cascade", "lexical", "parser abstained; decided by content voters"))

        # V5: bounded LLM arbitration for gray-zone only ---------------------------------
        if label is Label.REVIEW and self.judge is not None:
            self.judge_calls += 1
            verdict, why = self.judge.judge(cand_text, other.text)
            ev.append(Evidence("judge", verdict, why))
            if verdict == "duplicate":
                label, conf = Label.SEMANTIC_DUPLICATE, 0.8
            elif verdict == "different":
                label, conf = Label.DISTINCT, 0.8
        return Match(other, label, conf, sim, ev, flags)

    # ---- public API ------------------------------------------------------------
    def check(self, text: str) -> Verdict:
        frame = parse(text)
        cands = self.store.candidates(text, frame, self.top_k)
        matches = [self.compare_pair(text, frame, self.store.items[qid], sim) for qid, sim in cands.items()]
        matches.sort(key=lambda m: (-m.label, -m.confidence, -m.content_sim))
        if not matches:
            return Verdict(text, Label.DISTINCT, 1.0, [], frame, 0)
        top = matches[0]
        return Verdict(text, top.label, top.confidence, matches, frame, len(matches))

    def check_and_add(self, text: str, tags: dict | None = None, id: str | None = None):
        v = self.check(text)
        q = self.add(text, tags, id)
        return v, q

    def audit(self) -> list[list[str]]:
        """Scan an *existing* corpus for duplicate clusters (union-find over pairwise duplicate verdicts)."""
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for q in self.store:
            frame = q.frame
            for qid, sim in self.store.candidates(q.text, frame, self.top_k).items():
                if qid == q.id:
                    continue
                m = self.compare_pair(q.text, frame, self.store.items[qid], sim)
                if m.label.is_duplicate:
                    parent[find(q.id)] = find(qid)
        groups: dict[str, list[str]] = {}
        for x in list(parent):
            groups.setdefault(find(x), []).append(x)
        return [sorted(g, key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0)) for g in groups.values() if len(g) > 1]
