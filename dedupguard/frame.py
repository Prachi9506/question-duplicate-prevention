"""
Layer 2 - the *semantic frame*.

Embedding / n-gram similarity answers "do these two strings share words?".
Duplicate detection needs a different question: "do these two questions *ask for
the same thing about the same thing*?"  "What is reflection of light?" and
"State the laws of reflection of light." share 4/5 words but ask for different
things; "Explain the meaning of light reflection." shares only 2 yet asks for the
same thing.

So every question is parsed into a small structured frame:

    intent   WHAT is being asked   (DEFINE / ENUMERATE / WHY / HOW / COMPUTE / ...)
    concept  ABOUT what            (head noun + modifier set, order-insensitive)
    numbers  GIVEN data            (value, unit) pairs - for numerical problems

and two frames are compared structurally.  The parser is rule-based (deterministic,
auditable, zero latency, no API cost) and reports its own confidence so the
cascade can fall back to other voters when it is unsure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .normalize import STOP, canonical, lemma, strip_presentation, words_to_numbers

# --------------------------------------------------------------------------- #
# vocab
# --------------------------------------------------------------------------- #
DETERMINERS = {"the", "a", "an", "any", "some", "few", "all", "each", "every", "various", "different",
               "following", "given", "short", "brief", "detailed", "two", "three", "four", "five", "one"}
PREPOSITIONS = {"of", "in", "on", "for", "by", "with", "from", "to", "at", "during", "through", "using",
                "inside", "under", "over", "across", "about", "regarding"}
WRAPPERS = {"concept", "phenomenon", "process", "idea", "notion", "term", "word", "topic", "principle_of",
            "meaning", "definition"}
# nouns that make "what ARE the <X> of Y?" an enumeration question
ENUM_HEADS = {"law", "type", "kind", "property", "characteristic", "feature", "application", "use",
              "advantage", "disadvantage", "step", "stage", "component", "part", "factor", "cause",
              "effect", "principle", "rule", "postulate", "assumption", "limitation", "category",
              "source", "element", "function", "benefit", "drawback", "example", "reason"}
ENUM_VERBS = {"list", "enumerate", "name", "mention"}
UNIT_WORDS = {"degrees", "m", "cm", "mm", "km", "kg", "g", "mg", "s", "ms", "hz", "n", "j", "w", "v", "a",
              "k", "c", "mol", "l", "ml", "pa", "ohm", "rad", "percent", "units", "kmph", "mph"}


class Intent(str, Enum):
    DEFINE = "DEFINE"
    ENUMERATE = "ENUMERATE"
    EXPLAIN = "EXPLAIN"
    WHY = "WHY"
    HOW = "HOW"
    COMPUTE = "COMPUTE"
    DIFFERENTIATE = "DIFFERENTIATE"
    EXAMPLE = "EXAMPLE"
    DRAW = "DRAW"
    VERIFY = "VERIFY"
    FACT = "FACT"
    OTHER = "OTHER"


# intents whose concept is a free clause rather than a noun phrase -> compared as a bag of lemmas
BAG_INTENTS = {Intent.WHY, Intent.HOW, Intent.DIFFERENTIATE, Intent.EXPLAIN, Intent.OTHER, Intent.VERIFY, Intent.FACT}
LIGHT_VERBS = {"appear", "seem", "get", "become", "happen", "occur", "take", "make", "have", "look", "exist", "work_"}
NEG_PREFIX = {"non", "anti"}
IMPORTANCE = {"important", "importance", "matter", "significance", "significant", "necessary", "essential", "needed", "need", "vital", "crucial"}
_CONTEXT_FILLER = re.compile(r"\b(?:in |from |of |to )?(?:our |the )?(?:everyday|daily|real|day to day|practical) (?:life|lives|world)\b")
ADJACENT = {frozenset({Intent.DEFINE, Intent.EXPLAIN}), frozenset({Intent.EXPLAIN, Intent.HOW}),
            frozenset({Intent.EXPLAIN, Intent.WHY})}


@dataclass(frozen=True)
class Concept:
    head: str                      # lemma of head noun; "" for bag concepts
    mods: frozenset                # modifier lemmas (or the whole bag when head == "")
    surface: str = field(default="", compare=False)   # human-readable phrase, for prompts only

    @property
    def tokens(self) -> frozenset:
        return self.mods | ({self.head} if self.head else frozenset())

    def label(self) -> str:
        if self.head:
            return " ".join(sorted(self.mods)) + (" " if self.mods else "") + self.head
        return " ".join(sorted(self.mods))


@dataclass(frozen=True)
class Frame:
    intent: Intent
    concept: Concept
    numbers: frozenset = frozenset()          # {(value, unit)}
    givens: frozenset = frozenset()           # lemmas from the "given" part of a numerical problem
    parts: tuple = ()                         # sub-frames for compound questions ("what is X and why ...")
    confidence: float = 0.0
    rule: str = ""
    plural_head: bool = False

    def signature(self) -> str:
        c = self.concept.label()
        nums = ",".join(f"{v:g}{u}" for v, u in sorted(self.numbers))
        return f"{self.intent.value}|{c}" + (f"|{nums}" if nums else "")


# --------------------------------------------------------------------------- #
# intent rules: (compiled regex on canonical sentence, intent, rule-name, confidence)
# the LAST capture group is always the noun phrase / clause that follows the cue.
# --------------------------------------------------------------------------- #
_R = lambda p: re.compile(p)
RULES: list[tuple[re.Pattern, Intent, str, float]] = [
    (_R(r"^(?:what )?(?:is |are )?(?:the )?(?:differences?|distinctions?)\s+(?:between|among)\s+(.+)$"), Intent.DIFFERENTIATE, "difference-between", 0.95),
    (_R(r"^(?:state|write|give|explain|list|mention)\s+(?:the )?(?:differences?|distinctions?)\s+(?:between|among)\s+(.+)$"), Intent.DIFFERENTIATE, "state-difference", 0.95),
    (_R(r"^(?:differentiate|distinguish|compare|contrast)\s+(?:between |among )?(.+)$"), Intent.DIFFERENTIATE, "differentiate-verb", 0.95),
    (_R(r"^how (?:is|are|does|do) (.+?) differ(?:ent)? (?:from|than) (.+)$"), Intent.DIFFERENTIATE, "how-differ", 0.9),
    # definitional cues -----------------------------------------------------
    (_R(r"^what does (.+?) mean$"), Intent.DEFINE, "what-does-mean", 0.95),
    (_R(r"^(?:what do you (?:understand|mean) by|what is meant by|what do you understand from|what is understood by)\s+(.+)$"), Intent.DEFINE, "understand-by", 0.95),
    (_R(r"^(?:what is|what are|what was) (?:the )?(?:meaning|definition) of\s+(.+)$"), Intent.DEFINE, "meaning-of", 0.95),
    (_R(r"^(?:explain|describe|state|give|write)(?: me)? (?:the )?(?:meaning|definition|term|concept of the term)(?: of)?\s+(.+)$"), Intent.DEFINE, "explain-meaning", 0.95),
    (_R(r"^(?:explain|describe) (?:what is meant by|what you understand by|what is understood by)\s+(.+)$"), Intent.DEFINE, "explain-what-meant", 0.95),
    (_R(r"^(?:define|definition of|meaning of|give the meaning of)\s+(?:the term )?(.+)$"), Intent.DEFINE, "define", 0.97),
    # compute ---------------------------------------------------------------
    (_R(r"^(?:find|calculate|compute|determine|evaluate|obtain|work out|solve for|solve)\s+(?:the )?(?:value of )?(.+)$"), Intent.COMPUTE, "compute-verb", 0.9),
    (_R(r"^how (?:much|many|far|long|fast|high|tall|wide)\s+(.+)$"), Intent.COMPUTE, "how-much", 0.8),
    # why / how --------------------------------------------------------------
    (_R(r"^what (?:causes|caused|makes|made|leads to|is the reason for|is the cause of|are the reasons for)\s+(.+)$"), Intent.WHY, "what-causes", 0.85),
    (_R(r"^(?:who|whom|when|where|which)\s+(.+)$"), Intent.FACT, "wh-fact", 0.8),
    (_R(r"^why\s+(?:do |does |is |are |did |can |should |would )?(.+)$"), Intent.WHY, "why", 0.9),
    (_R(r"^how (?:do |does |is |are |can |to |would |will |did |should )\s*(.+)$"), Intent.HOW, "how", 0.9),
    # misc asks ---------------------------------------------------------------
    (_R(r"^(?:give|state|write|mention|cite|provide)\s+(?:an? |two |three |some |any )?(?:real life |daily life |everyday )?examples?\s+(?:of |for )?(.+)$"), Intent.EXAMPLE, "example", 0.9),
    (_R(r"^(?:draw|sketch|label)\s+(?:a |an |the )?(?:neat |labelled |ray )?(?:diagram|figure|sketch)?\s*(?:of |to show |showing )?(.+)$"), Intent.DRAW, "draw", 0.9),
    (_R(r"^(?:true or false|is it true that|state whether)\s*(.+)$"), Intent.VERIFY, "verify", 0.8),
    # list-type verbs ----------------------------------------------------------
    (_R(r"^(list|enumerate)\s+(?:down )?(.+)$"), Intent.ENUMERATE, "list-verb", 0.9),
    (_R(r"^(?:state|write|give|write down|name|mention)\s+(.+)$"), Intent.DEFINE, "state-recall", 0.85),   # may become ENUMERATE
    (_R(r"^(?:explain|describe|discuss|elaborate on|illustrate|elaborate|tell me about|write (?:a )?(?:short )?(?:note|essay|paragraph) on)\s+(.+)$"), Intent.EXPLAIN, "explain", 0.75),
    # generic wh-is (must be last of the definitional family) ----------------
    (_R(r"^what (?:is|are|was|were)\s+(.+)$"), Intent.DEFINE, "what-is", 0.85),
]
_SPLIT_IF = re.compile(r"\s+(?:if|when|given that|given|where|provided that|provided|for)\s+(?=.*\d)")
_COMPOUND = re.compile(r"\s+(?:and|also|then)\s+(?=(?:why|how|what|state|explain|give|list|write|define|describe|derive|draw)\b)")
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])(?:\s+(" + "|".join(sorted(UNIT_WORDS, key=len, reverse=True)) + r")\b)?")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sentences(text: str) -> list[str]:
    t = words_to_numbers(strip_presentation(text))
    parts = re.split(r"(?<=[\.\?\!;])\s+(?=\S)", t)
    return [canonical(p) for p in parts if canonical(p)]


def extract_numbers(s: str) -> frozenset:
    out = set()
    for m in _NUM.finditer(s):
        out.add((float(m.group(1)), m.group(2) or ""))
    return frozenset(out)


def _clean(toks: list[str]) -> list[str]:
    """drop possessive 's, numbers and their units; glue 'non'+'renewable' -> 'nonrenewable'."""
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "s":
            i += 1
        elif t.replace(".", "").isdigit():
            i += 2 if i + 1 < len(toks) and toks[i + 1] in UNIT_WORDS else 1
        elif t in NEG_PREFIX and i + 1 < len(toks):
            out.append(t + toks[i + 1]); i += 2
        else:
            out.append(t); i += 1
    return out


def parse_np(np: str) -> tuple[Concept, bool]:
    """Noun phrase -> (head, modifiers).  'X of Y' -> head X; compound 'Y X' -> head X."""
    toks = _clean(np.split())
    while toks and toks[0] in DETERMINERS:
        toks.pop(0)
    # drop classifier nouns: "the concept of X", "the term X"
    while len(toks) > 1 and toks[0] in WRAPPERS and (toks[1] in PREPOSITIONS or toks[1] not in STOP):
        toks.pop(0)
        if toks and toks[0] in PREPOSITIONS:
            toks.pop(0)
        while toks and toks[0] in DETERMINERS:
            toks.pop(0)
    cut = next((i for i, t in enumerate(toks) if t in PREPOSITIONS), None)
    head_part, rest = (toks, []) if cut is None else (toks[:cut], toks[cut + 1:])
    head_part = [t for t in head_part if t not in STOP and t not in WRAPPERS or (t in WRAPPERS and len(head_part) == 1)]
    if not head_part:                       # e.g. "of light" - fall back to the remainder
        head_part, rest = [t for t in rest if t not in STOP], []
    if not head_part:
        return Concept("", frozenset()), False
    surface_head = head_part[-1]
    head = lemma(surface_head)
    plural = surface_head != head and surface_head.endswith("s")
    mods = {lemma(t) for t in head_part[:-1] + rest if t not in STOP and t not in DETERMINERS and t not in WRAPPERS}
    return Concept(head, frozenset(mods), " ".join(toks)), plural


def _bag(np: str) -> Concept:
    clean_np = _CONTEXT_FILLER.sub(" ", np)
    toks = [lemma(t) for t in _clean(clean_np.split()) if t not in STOP and t not in DETERMINERS]
    toks = ["importance" if t in IMPORTANCE or t == "importanc" else t for t in toks]
    toks = [t for t in toks if t not in LIGHT_VERBS]
    return Concept("", frozenset(toks), np)


def tok_eq(a: str, b: str) -> bool:
    """Fuzzy token equivalence inside free-clause bags: reflect ~ reflected ~ reflection."""
    if a == b:
        return True
    lo, hi = sorted((a, b), key=len)
    return len(lo) >= 5 and hi.startswith(lo) and len(hi) - len(lo) <= 4


def bag_relation(A: frozenset, B: frozenset) -> tuple[str, float]:
    """SAME / BROADER (A broader than B) / NARROWER / OVERLAP / DIFFERENT, plus jaccard."""
    if not A or not B:
        return "DIFFERENT", 0.0
    a_in_b = all(any(tok_eq(x, y) for y in B) for x in A)
    b_in_a = all(any(tok_eq(y, x) for x in A) for y in B)
    inter = sum(1 for x in A if any(tok_eq(x, y) for y in B))
    jac = inter / (len(A) + len(B) - inter)
    if a_in_b and b_in_a:
        return "SAME", jac
    if a_in_b:
        return "BROADER", jac
    if b_in_a:
        return "NARROWER", jac
    return ("OVERLAP" if inter else "DIFFERENT"), jac


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #
def _parse_sentence(s: str) -> Frame | None:
    for rx, intent, name, conf in RULES:
        m = rx.match(s)
        if not m:
            continue
        verb = m.group(1) if rx.groups > 1 else ""
        np = m.groups()[-1]
        if intent is Intent.DIFFERENTIATE:
            np = re.sub(r"\s+(?:and|vs|versus|from)\s+", " ", np)
            return Frame(intent, _bag(np), confidence=conf, rule=name)
        if intent in (Intent.WHY, Intent.HOW):
            return Frame(intent, _bag(np), confidence=conf, rule=name)
        if intent is Intent.EXPLAIN:
            c = _bag(np)
            return Frame(intent, c, confidence=conf, rule=name)
        if intent in (Intent.VERIFY, Intent.DRAW):
            return Frame(intent, _bag(np), confidence=conf, rule=name)
        # noun-phrase intents ---------------------------------------------------
        if intent is Intent.COMPUTE:
            target, *tail = _SPLIT_IF.split(np, maxsplit=1)
            c, pl = parse_np(target)
            givens = frozenset(lemma(t) for t in " ".join(tail).split() if t not in STOP and not t.replace(".", "").isdigit())
            return Frame(intent, c, extract_numbers(np), givens, confidence=conf, rule=name)
        c, plural = parse_np(np)
        if intent is Intent.EXAMPLE:
            return Frame(intent, c, confidence=conf, rule=name, plural_head=plural)
        # "meaning"/"definition" smuggled into the NP ("what is the meaning of ...") are handled by rules
        if intent is Intent.ENUMERATE or (c.head in ENUM_HEADS and plural and name in ("what-is", "state-recall")):
            if c.head == "example":
                return Frame(Intent.EXAMPLE, c, confidence=conf, rule=name, plural_head=plural)
            return Frame(Intent.ENUMERATE, c, confidence=conf, rule=name + "+enum", plural_head=plural)
        return Frame(intent, c, confidence=conf if c.head else 0.3, rule=name, plural_head=plural)
    return None


def parse(text: str) -> Frame:
    """Question text -> Frame.  Never raises; unknown shapes get a low-confidence OTHER frame."""
    sents = _sentences(text)
    if not sents:
        return Frame(Intent.OTHER, Concept("", frozenset()), confidence=0.0, rule="empty")
    # compound questions: "what is X and why does it happen"
    if len(sents) == 1:
        pieces = _COMPOUND.split(sents[0])
        if len(pieces) > 1:
            subs = tuple(f for f in (parse(p) for p in pieces) if f.confidence > 0)
            if len(subs) > 1:
                return Frame(Intent.OTHER, _bag(sents[0]), parts=subs, confidence=min(f.confidence for f in subs), rule="compound")
    ask, ask_idx = None, None
    for i in range(len(sents) - 1, -1, -1):
        f = _parse_sentence(sents[i])
        if f is not None:
            ask, ask_idx = f, i
            break
    allnums = extract_numbers(" ".join(sents))
    if ask is None:
        return Frame(Intent.OTHER, _bag(" ".join(sents)), allnums, confidence=0.3, rule="fallback")
    # numerical problem?  numbers anywhere + a "what is / find" style ask  ->  COMPUTE
    if allnums and ask.intent in (Intent.DEFINE, Intent.COMPUTE, Intent.ENUMERATE) and ask.rule in ("what-is", "compute-verb", "how-much"):
        givens_txt = " ".join(s for j, s in enumerate(sents) if j != ask_idx)
        givens = set(ask.givens) | {lemma(t) for t in givens_txt.split() if t not in STOP and not t.replace(".", "").isdigit()}
        return Frame(Intent.COMPUTE, ask.concept, allnums, frozenset(givens), confidence=0.9, rule=ask.rule + "+numbers")
    return ask


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
class Rel(str, Enum):
    DUP = "DUP"
    REVIEW = "REVIEW"
    DISTINCT = "DISTINCT"
    UNKNOWN = "UNKNOWN"


@dataclass
class FrameRelation:
    rel: Rel
    score: float
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)


def _concept_relation(a: Frame, b: Frame) -> tuple[str, float]:
    ca, cb = a.concept, b.concept
    if a.intent in BAG_INTENTS or not ca.head or not cb.head:
        return bag_relation(ca.tokens, cb.tokens)
    if ca.head != cb.head:
        return "DIFFERENT", 0.0
    rel, jac = bag_relation(ca.mods | {"_"}, cb.mods | {"_"})   # '_' keeps empty modifier sets comparable
    return rel, jac


def compare(a: Frame, b: Frame) -> FrameRelation:
    # compound questions: compare constituent parts
    if a.parts or b.parts:
        pa, pb = a.parts or (a,), b.parts or (b,)
        rels = [compare(x, y) for x in pa for y in pb]
        dups = [r for r in rels if r.rel is Rel.DUP]
        if dups and len(pa) == len(pb) and len(dups) >= len(pa):
            return FrameRelation(Rel.DUP, 0.9, ["all sub-questions of the compound question match"])
        if dups or any(r.rel is Rel.REVIEW for r in rels):
            return FrameRelation(Rel.REVIEW, 0.6, ["compound question overlaps with an existing question"])
        return FrameRelation(Rel.DISTINCT, 0.1, ["no sub-question overlaps"])
    if min(a.confidence, b.confidence) < 0.5:
        return FrameRelation(Rel.UNKNOWN, 0.0, ["frame parser not confident; deferring to other voters"])

    ask = f"{a.intent.value} vs {b.intent.value}"
    if a.intent is not b.intent:
        if frozenset({a.intent, b.intent}) in ADJACENT:
            crel, _ = _concept_relation(a, b)
            if crel == "SAME":
                return FrameRelation(Rel.REVIEW, 0.65, [f"adjacent asks ({ask}) about the same concept"])
        return FrameRelation(Rel.DISTINCT, 0.05, [f"different ask: {ask}"])

    crel, jac = _concept_relation(a, b)
    ca, cb = a.concept.label() or "-", b.concept.label() or "-"
    if crel == "SAME":
        if a.intent is Intent.COMPUTE:
            if a.numbers == b.numbers:
                return FrameRelation(Rel.DUP, 0.95, [f"same target '{ca}' with identical given values"])
            return FrameRelation(Rel.DISTINCT, 0.2, [f"same target '{ca}' but different given values"], ["numeric_variant"])
        return FrameRelation(Rel.DUP, 0.95, [f"same ask ({a.intent.value}) about the same concept '{ca}'"])
    if crel in ("BROADER", "NARROWER"):
        extra = (a.concept.tokens ^ b.concept.tokens) - {"_"}
        side = "broader" if crel == "BROADER" else "narrower"
        return FrameRelation(Rel.REVIEW, 0.6, [f"same ask, but candidate is {side}: concept differs only by {sorted(extra)}"])
    if a.intent in BAG_INTENTS and jac >= 0.6:
        return FrameRelation(Rel.REVIEW, 0.5, [f"free-form clauses overlap strongly (jaccard {jac:.2f})"])
    return FrameRelation(Rel.DISTINCT, 0.1, [f"same ask ({a.intent.value}) but different concept: '{ca}' vs '{cb}'"])
