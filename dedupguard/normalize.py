"""
Layer 0 - canonicalisation.

Everything downstream (hashing, frame parsing, similarity) works on text that
has been scrubbed of *presentation* noise so that only *content* is compared:
casing, punctuation, unicode look-alikes, numbering ("Q3."), mark annotations
("[2 marks]"), trailing MCQ options and spelled-out numbers ("thirty" -> 30).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# --------------------------------------------------------------------------- #
# number words
# --------------------------------------------------------------------------- #
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}


def words_to_numbers(text: str) -> str:
    """'thirty five degrees' -> '35 degrees'; 'twenty-five' -> '25'; 'two hundred' -> '200'."""
    flat = re.sub(r"(?<=\w)-(?=\w)", " ", text).split()
    out: list[str] = []
    i = 0
    while i < len(flat):
        w = flat[i].lower().strip(".,;:!?")
        if w in _UNITS or w in _TENS:
            val = 0
            if w in _TENS:
                val = _TENS[w]
                i += 1
                nxt = flat[i].lower().strip(".,;:!?") if i < len(flat) else ""
                if nxt in _UNITS and 0 < _UNITS[nxt] < 10:
                    val += _UNITS[nxt]
                    i += 1
            else:
                val = _UNITS[w]
                i += 1
            nxt = flat[i].lower().strip(".,;:!?") if i < len(flat) else ""
            if nxt == "hundred" and 0 < val < 10:
                val *= 100
                i += 1
                n2 = flat[i].lower().strip(".,;:!?") if i < len(flat) else ""
                if n2 in _TENS:
                    val += _TENS[n2]; i += 1
                    n3 = flat[i].lower().strip(".,;:!?") if i < len(flat) else ""
                    if n3 in _UNITS and 0 < _UNITS[n3] < 10:
                        val += _UNITS[n3]; i += 1
            out.append(str(val))
        else:
            out.append(flat[i])
            i += 1
    return " ".join(out)


# --------------------------------------------------------------------------- #
# presentation noise
# --------------------------------------------------------------------------- #
_NUMBERING = re.compile(r"^\s*(?:q(?:uestion)?\s*\.?\s*)?\d{1,3}\s*[\).:\-]\s+|^\s*q(?:uestion)?\s*[:.]\s*", re.I)
_MARKS = re.compile(r"[\[\(]\s*\d+\s*(?:marks?|pts?|points?)\s*[\]\)]", re.I)
_MCQ_START = re.compile(r"(?:^|\s)\(?\b[aA][\)\.]\s+\S", re.S)
_MCQ_B = re.compile(r"\(?\b[bB][\)\.]\s+\S")
_PREFIX_NOISE = re.compile(r"^\s*(?:fill in the blanks?|choose the correct (?:option|answer)|answer the following|true or false)\s*[:\-]\s*", re.I)

_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}


def strip_presentation(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    for k, v in _QUOTES.items():
        t = t.replace(k, v)
    t = _MARKS.sub(" ", t)
    t = _NUMBERING.sub("", t)
    t = _PREFIX_NOISE.sub("", t)
    m = _MCQ_START.search(t)
    if m and _MCQ_B.search(t, m.end()):          # looks like "(a) ... (b) ..." options
        t = t[: m.start()]
    return t.strip()


_UNIT_ALIASES = [
    (r"centi?met(?:er|re)s?", "cm"), (r"milli?met(?:er|re)s?", "mm"), (r"kilo?met(?:er|re)s?", "km"),
    (r"met(?:er|re)s?", "m"), (r"kilograms?|kilos?", "kg"), (r"grams?", "g"), (r"seconds?|secs?", "s"),
    (r"hertz", "hz"), (r"newtons?", "n"), (r"joules?", "j"), (r"watts?", "w"), (r"volts?", "v"),
    (r"%|percent|per cent", "percent"),
]


def canonical(text: str) -> str:
    """Aggressive canonical form used for the exact-match hash."""
    t = strip_presentation(text).lower()
    t = t.replace("\u00b0", " degrees ")
    t = words_to_numbers(t)
    t = re.sub(r"[^\w\s\.]", " ", t)               # keep '.' only between digits (decimals)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"\b(degrees?|deg)\b", "degrees", t)
    t = re.sub(r"(?<![a-z\d])(\d+(?:\.\d+)?)(?=[a-z])", r"\1 ", t)          # "7cm" -> "7 cm"
    for pat, unit in _UNIT_ALIASES:                                           # only after a digit ("second law" is safe)
        t = re.sub(r"(?<=\d )" + pat + r"\b", unit, t)
    return re.sub(r"\s+", " ", t).strip()


def exact_key(text: str) -> str:
    return hashlib.sha1(canonical(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# light-weight lemmatiser (no NLTK needed)
# --------------------------------------------------------------------------- #
_IRREGULAR = {"incidence": "incident", "reflecting": "reflect", "reflected": "reflect", "indices": "index",
              "analyses": "analysis", "bases": "base", "children": "child", "lenses": "lens", "lens": "lens", "mirrors": "mirror"}


def lemma(tok: str) -> str:
    t = tok.lower()
    if t in _IRREGULAR:
        return _IRREGULAR[t]
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("sses", "ches", "shes", "xes")):
        return t[:-2]
    if t.endswith("s") and len(t) > 3 and not t.endswith(("ss", "us", "is", "ics")):
        t = t[:-1]
    if t.endswith("ence") and len(t) > 6:      # incidence/difference -> incident/different
        t = t[:-4] + "ent"
    if t.endswith("ance") and len(t) > 6:
        t = t[:-4] + "ant"
    return t


STOP = frozenset("""a an the of in on at for to from by with and or is are was were be been being do does did
this that these those it its as into onto about between among what which who whom whose how why when where
you your we our i me my can could should would will shall may might please also""".split())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", canonical(text))


def content_lemmas(text: str) -> list[str]:
    return [lemma(t) for t in tokens(text) if t not in STOP]
