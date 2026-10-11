"""
Generation-time prevention: the COVERAGE MAP.

The naive approach - paste every existing question into the prompt - fails three ways:
it overflows the context window, it makes the LLM *anchor* on the wording of what it
sees, and it still can't say what is *missing*.

Instead the corpus is folded into a grid of CELLS:

        cell = (kind of ask)  x  (concept)          e.g.  (DEFINE, light reflection)

All paraphrases of "define reflection of light" land in ONE cell no matter how they are
worded, so a corpus of 10,000 questions is described by (number of distinct ideas), not
(number of questions).  The prompt then says two things the LLM can act on:

    TARGET CELLS    open slots that nobody has asked yet  -> generate *into* these
    COVERED CELLS   already taken (with a count and one example) -> stay out of these

i.e. prevention by *construction* (slot-filling) instead of by *blacklist*.  Prompt size
is bounded by a token budget regardless of corpus size; relevance to the requested
subject is found with the inverted index (top-k neighbours), never a full scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import embed
from .frame import Concept, Frame, Intent, parse, parse_np
from .index import QuestionStore

INTENT_HINT = {
    Intent.DEFINE: "asks for a definition / meaning",
    Intent.ENUMERATE: "asks to list or state several items (laws, types, properties...)",
    Intent.WHY: "asks for a reason or cause",
    Intent.HOW: "asks how a process or mechanism works",
    Intent.EXAMPLE: "asks for real-life examples",
    Intent.COMPUTE: "numerical problem with given values (use NEW numbers)",
    Intent.DIFFERENTIATE: "asks to contrast two ideas",
}
DEFAULT_INTENTS = [Intent.DEFINE, Intent.WHY, Intent.HOW, Intent.EXAMPLE, Intent.ENUMERATE]
QUANTITY_HEADS = {"angle", "speed", "velocity", "force", "distance", "area", "volume", "mass", "energy", "power",
                  "pressure", "temperature", "time", "frequency", "wavelength", "current", "voltage", "density",
                  "acceleration", "height", "length", "focal", "magnification"}


def est_tokens(text: str) -> int:
    """~4 chars/token (rounded up). Swap for a real tokenizer in production; the budget logic is unchanged."""
    return max(1, -(-len(text) // 4))


@dataclass
class Cell:
    intent: Intent
    key: str                       # canonical concept label
    phrase: str                    # readable concept phrase
    ids: list = field(default_factory=list)
    numbers: set = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.ids)


@dataclass
class OpenCell:
    intent: Intent
    facet: str
    quantitative: bool = False

    def line(self) -> str:
        return f"[{self.intent.value}] {self.facet} - {INTENT_HINT.get(self.intent, '')}"


class CoverageMap:
    def __init__(self, store: QuestionStore):
        self.store = store
        self.cells: dict[tuple[Intent, str], Cell] = {}
        self._seen: set[str] = set()
        self.sync()

    # ---- maintenance (incremental: O(new questions)) --------------------------------
    def sync(self) -> None:
        for q in self.store:
            if q.id not in self._seen:
                self._seen.add(q.id)
                self._add(q.id, q.frame)

    def _add(self, qid: str, frame: Frame) -> None:
        for f in (frame.parts or (frame,)):
            if f.confidence < 0.5 or not f.concept.tokens:
                continue
            key = f.concept.label()
            cell = self.cells.setdefault((f.intent, key), Cell(f.intent, key, f.concept.surface or key))
            cell.ids.append(qid)
            cell.numbers |= set(f.numbers)

    # ---- queries ---------------------------------------------------------------------
    @staticmethod
    def facet_key(facet: str) -> str:
        c, _ = parse_np(facet)
        return c.label()

    def is_covered(self, intent: Intent, facet: str) -> bool:
        return (intent, self.facet_key(facet)) in self.cells

    def open_cells(self, facets: Sequence, intents: Sequence[Intent] = tuple(DEFAULT_INTENTS)) -> list[OpenCell]:
        """Open (intent x facet) slots, interleaved so early slots vary in BOTH intent and facet."""
        out: list[OpenCell] = []
        flat = [f for f in facets if isinstance(f, str)]
        pairs = [f for f in facets if isinstance(f, tuple)]
        m = max(len(intents), 1)
        for r in range(m):
            for i, f in enumerate(flat):
                it = intents[(i + r) % m]
                if not self.is_covered(it, f):
                    out.append(OpenCell(it, f))
        for f in flat:                                           # numerical problems for quantity-like facets
            c, _ = parse_np(f)
            if c.head in QUANTITY_HEADS:
                cell = self.cells.get((Intent.COMPUTE, c.label()))
                if cell is None or cell.count < 2:
                    out.append(OpenCell(Intent.COMPUTE, f, True))
        for a, b in pairs:                                       # contrasts
            key = self.facet_key(f"{a} {b}")
            if (Intent.DIFFERENTIATE, key) not in self.cells:
                out.append(OpenCell(Intent.DIFFERENTIATE, f"{a} vs {b}"))
        seen, uniq = set(), []
        for oc in out:
            k = (oc.intent, oc.facet)
            if k not in seen:
                seen.add(k)
                uniq.append(oc)
        return uniq

    def relevant_cells(self, subject: str, facets: Sequence = (), k: int = 200) -> list[Cell]:
        """Cells touched by the k content-nearest questions to the subject (+ facet-matching cells)."""
        qids = [qid for qid, _ in self.store.neighbors(embed.encode(subject + " " + " ".join(f for f in facets if isinstance(f, str))), k)]
        keys, seen = [], set()
        for qid in qids:
            fr = self.store.items[qid].frame
            for f in (fr.parts or (fr,)):
                ck = (f.intent, f.concept.label())
                if ck in self.cells and ck not in seen:
                    seen.add(ck)
                    keys.append(ck)
        cells = [self.cells[k_] for k_ in keys]
        cells.sort(key=lambda c: -c.count)
        return cells

    # ---- exemplar selection: Maximal Marginal Relevance -------------------------------------------
    def exemplars(self, subject: str, k: int = 3, pool: int = 60, lam: float = 0.6) -> list[str]:
        qv = embed.encode(subject)
        cand = self.store.neighbors(qv, pool)
        chosen: list[tuple[str, dict]] = []
        while cand and len(chosen) < k:
            best, best_s = None, -9.0
            for qid, rel in cand:
                v = self.store.items[qid].vec
                red = max((embed.cosine(v, cv) for _, cv in chosen), default=0.0)
                s = lam * rel - (1 - lam) * red
                if s > best_s:
                    best, best_s = qid, s
            chosen.append((self.store.items[best].text, self.store.items[best].vec))
            cand = [(i, r) for i, r in cand if i != best]
        return [t for t, _ in chosen]


@dataclass
class PromptStats:
    tokens: int
    open_listed: int
    covered_listed: int
    covered_omitted: int
    exemplars: int
    feedback: int


def build_prompt(subject: str, n: int, cov: CoverageMap, facets: Sequence, intents: Sequence[Intent],
                 budget_tokens: int = 900, n_exemplars: int = 3, feedback: Iterable[str] = (),
                 targets: int | None = None) -> tuple[str, PromptStats, list[OpenCell]]:
    """Priority order when the budget is tight: header > TARGET cells > feedback > COVERED cells > style examples."""
    header = (f"You are writing NEW exam questions for the subject: {subject}.\n"
              f"Write exactly {n} questions. Each must fill ONE of the TARGET CELLS below.\n"
              "A cell = (kind of ask) x (concept). Two questions in the same cell are duplicates even when worded "
              "differently, so NEVER write into a COVERED cell, and never re-phrase an existing question.\n")
    footer = ('\nReturn ONLY a JSON array, no prose: [{"cell": <number>, "question": "..."}]\n')
    used = est_tokens(header) + est_tokens(footer)

    open_all = cov.open_cells(facets, intents)
    want = targets if targets is not None else max(n + 3, int(n * 1.5))
    open_lines, open_used = [], []
    for oc in open_all[:want]:
        line = f" {len(open_lines) + 1}. {oc.line()}"
        if open_lines and used + est_tokens(line) > budget_tokens * 0.5:   # always keep >=1 target
            break
        open_lines.append(line)
        open_used.append(oc)
        used += est_tokens(line)

    fb_lines = []
    for f in list(feedback)[-6:]:
        line = f" - {f}"
        if used + est_tokens(line) > budget_tokens * 0.8:
            break
        fb_lines.append(line)
        used += est_tokens(line)

    cells = cov.relevant_cells(subject, facets)
    cov_lines, omitted = [], 0
    for c in cells:
        ex = cov.store.items[c.ids[0]].text
        line = f" - [{c.intent.value}] {c.phrase} (x{c.count}) e.g. \"{ex}\""
        if used + est_tokens(line) > budget_tokens * 0.92:
            omitted += 1
            continue
        cov_lines.append(line)
        used += est_tokens(line)

    ex_lines = []
    for t in cov.exemplars(subject, n_exemplars):
        line = f" - {t}"
        if used + est_tokens(line) > budget_tokens:
            break
        ex_lines.append(line)
        used += est_tokens(line)

    def assemble() -> str:
        so = "\nTARGET CELLS (open - nothing exists yet):\n" + "\n".join(open_lines) + "\n"
        sf = ("\nREJECTED LAST ROUND (an independent duplicate detector caught these):\n" + "\n".join(fb_lines) + "\n") if fb_lines else ""
        sc = "\nCOVERED CELLS (do NOT write here):\n" + "\n".join(cov_lines)
        if omitted:
            sc += f"\n (+{omitted} more covered cells omitted for space; a detector will reject repeats)"
        sc += "\n"
        se = ("\nSTYLE EXAMPLES (tone/format only - do not reuse their topic or wording):\n" + "\n".join(ex_lines) + "\n") if ex_lines else ""
        return header + so + sf + sc + se + footer

    prompt = assemble()
    while est_tokens(prompt) > budget_tokens:
        if ex_lines:
            ex_lines.pop()
        elif cov_lines:
            cov_lines.pop(); omitted += 1
        elif fb_lines:
            fb_lines.pop(0)
        elif len(open_lines) > 1:
            open_lines.pop(); open_used.pop()
        else:
            break
        prompt = assemble()
    return prompt, PromptStats(est_tokens(prompt), len(open_lines), len(cov_lines), omitted, len(ex_lines), len(fb_lines)), open_used
