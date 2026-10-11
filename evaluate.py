"""
Evaluation harness.  Run:  python evaluate.py          (prints report, writes results.json + figures)

 1. Assignment corpus (Q1-Q10)             - sequential insertion protocol, ground truth from the prompt
 2. Stress set (data/pairs.json)           - 55 hand-labelled pairs, other subjects, hard negatives AND hard positives
 3. Oracle-tuned baselines                 - difflib / word-TFIDF cosine / content cosine, each given its BEST possible threshold
 4. Metamorphic suite                      - ~2k generated pairs testing invariances (format noise, paraphrase, concept/intent shifts)
 5. Scale benchmark                        - latency, recall/FPR and prompt size from 1k to 50k questions
"""
from __future__ import annotations

import difflib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
from dedupguard import DuplicateDetector, Label, CoverageMap
from dedupguard import embed
from dedupguard.coverage import build_prompt, DEFAULT_INTENTS, est_tokens
from dedupguard.normalize import tokens

LABEL_MAP = {Label.EXACT_DUPLICATE: "DUP", Label.SEMANTIC_DUPLICATE: "DUP", Label.REVIEW: "REVIEW", Label.DISTINCT: "DISTINCT"}


def cascade_pair(a: str, b: str) -> tuple[str, Label]:
    d = DuplicateDetector()
    d.add(a)
    v = d.check(b)
    return LABEL_MAP[v.label], v.label


# ------------------------------------------------------------------------------------------
# baselines
# ------------------------------------------------------------------------------------------
class WordTfidf:
    """Textbook baseline: word uni+bigram TF-IDF cosine on RAW tokens (idf fitted on the eval texts)."""

    def __init__(self, texts):
        df = Counter()
        self.docs = {}
        for t in texts:
            f = self._feats(t)
            self.docs[t] = f
            df.update(set(f))
        n = len(texts)
        self.idf = {k: math.log((1 + n) / (1 + v)) + 1 for k, v in df.items()}

    @staticmethod
    def _feats(t):
        w = tokens(t)
        return Counter(w + [f"{a}_{b}" for a, b in zip(w, w[1:])])

    def sim(self, a, b):
        fa, fb = self.docs[a], self.docs[b]
        va = {k: v * self.idf[k] for k, v in fa.items()}
        vb = {k: v * self.idf[k] for k, v in fb.items()}
        na = math.sqrt(sum(x * x for x in va.values())); nb = math.sqrt(sum(x * x for x in vb.values()))
        return sum(v * vb.get(k, 0) for k, v in va.items()) / (na * nb or 1)


def best_threshold(scores, truth):
    """Oracle: the threshold maximising F1 for the DUP class (generous to the baseline)."""
    best = (0, 0, 0, 0)
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, truth) if s >= t and y); fp = sum(1 for s, y in zip(scores, truth) if s >= t and not y)
        fn = sum(1 for s, y in zip(scores, truth) if s < t and y)
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        if f > best[0]:
            best = (f, t, p, r)
    return best


def prf(pred, truth):
    tp = sum(1 for p, y in zip(pred, truth) if p and y); fp = sum(1 for p, y in zip(pred, truth) if p and not y)
    fn = sum(1 for p, y in zip(pred, truth) if not p and y)
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r


# ------------------------------------------------------------------------------------------
def eval_assignment():
    rows = json.load(open("data/corpus.json"))
    d = DuplicateDetector()
    out, ok = [], 0
    for r in rows:
        v = d.check(r["text"])
        got = "ORIGINAL" if len(d.store) == 0 else v.label.name
        out.append((r["id"], r["text"], r["expected"], got))
        ok += got == r["expected"]
        d.add(r["text"], id=r["id"])
    return ok, len(rows), out


def eval_pairs():
    pairs = json.load(open("data/pairs.json"))
    texts = sorted({t for a, b, *_ in pairs for t in (a, b)})
    tf = WordTfidf(texts)
    res = []
    for a, b, y, cat in pairs:
        lab, raw = cascade_pair(a, b)
        res.append(dict(a=a, b=b, y=y, cat=cat, pred=lab, raw=raw.name,
                        difflib=difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(),
                        tfidf=tf.sim(a, b), content=embed.cosine(embed.encode(a), embed.encode(b))))
    return res


def report_pairs(res):
    bin_ = [r for r in res if r["y"] in ("DUP", "DISTINCT")]
    truth = [r["y"] == "DUP" for r in bin_]
    lines = {}
    for name in ("difflib", "tfidf", "content"):
        f, t, p, r = best_threshold([x[name] for x in bin_], truth)
        dup_s = [x[name] for x in bin_ if x["y"] == "DUP"]; dis_s = [x[name] for x in bin_ if x["y"] == "DISTINCT"]
        overlap = sum(1 for s in dis_s if s >= min(dup_s))
        lines[name] = dict(f1=f, thr=t, p=p, r=r, distinct_above_weakest_dup=overlap, n_distinct=len(dis_s))
    strict = prf([x["pred"] == "DUP" for x in bin_], truth)
    lenient = prf([x["pred"] in ("DUP", "REVIEW") for x in bin_], truth)
    three = sum(1 for x in res if x["pred"] == x["y"]) / len(res)
    return lines, strict, lenient, three


# ------------------------------------------------------------------------------------------
# metamorphic suite
# ------------------------------------------------------------------------------------------
CONCEPTS = ["the speed of sound", "the area of a circle", "Boyle's law", "the water cycle", "a prime number",
            "Newton's third law", "the process of mitosis", "an isosceles triangle", "the periodic table",
            "electromagnetic induction", "a binary search tree", "the Pythagorean theorem", "compound interest",
            "the greenhouse effect", "a chemical equation", "the law of conservation of energy", "plate tectonics",
            "an arithmetic progression", "the Industrial Revolution", "Ohm's law", "natural selection",
            "a quadratic equation", "the nitrogen cycle", "a transistor"]
DEFINE_T = ["What is {c}?", "Define {c}.", "What do you understand by {c}?", "Explain the meaning of {c}.",
            "State the definition of {c}.", "What is meant by {c}?", "Give the meaning of {c}.", "Explain the term {c}."]
OTHER_T = {"WHY": "Why is {c} important?", "HOW": "How does {c} work?", "LIST": "List the properties of {c}.",
           "EXAMPLE": "Give two examples of {c}."}
NOISE = [lambda s: s.upper(), lambda s: "Q4. " + s, lambda s: s + " [2 marks]", lambda s: s.replace("?", "??").replace(".", ".."),
         lambda s: "  " + s.replace(" ", "  ") + " ", lambda s: s.rstrip("?.") + " (a) one (b) two (c) three"]


def metamorphic():
    stats = {"format-noise": [0, 0], "paraphrase": [0, 0], "intent-shift": [0, 0], "concept-shift": [0, 0]}
    fails = []

    def run(kind, a, b, expect):
        d = DuplicateDetector(); d.add(a)
        v = d.check(b)
        got = LABEL_MAP[v.label]
        stats[kind][1] += 1
        if got == expect:
            stats[kind][0] += 1
        else:
            fails.append((kind, a, b, expect, got))
    for c in CONCEPTS:
        qs = [t.format(c=c) for t in DEFINE_T]
        for q in qs:
            for f in NOISE:
                run("format-noise", q, f(q), "DUP")
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                run("paraphrase", qs[i], qs[j], "DUP")
        for q in qs:
            for t in OTHER_T.values():
                run("intent-shift", q, t.format(c=c), "DISTINCT")
    rng = random.Random(1)
    for _ in range(400):
        c1, c2 = rng.sample(CONCEPTS, 2)
        run("concept-shift", rng.choice(DEFINE_T).format(c=c1), rng.choice(DEFINE_T).format(c=c2), "DISTINCT")
    return stats, fails


# ------------------------------------------------------------------------------------------
# scale
# ------------------------------------------------------------------------------------------
def pseudo_words(rng, n):
    syl = ["ka", "zo", "mi", "ra", "tu", "ve", "lo", "ni", "pa", "do", "se", "fi", "gu", "bo", "xa", "ye"]
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(syl) for _ in range(3)) + rng.choice(["n", "r", "l", "m", "x", "t"]))
    return sorted(out)


SC_DEFINE = ["What is {c}?", "Define {c}.", "What do you understand by {c}?", "Explain the meaning of {c}."]
SC_OTHER = {"WHY": "Why is {c} important?", "HOW": "How does {c} work?", "LIST": "List the properties of {c}.", "EX": "Give two examples of {c}."}


def scale_benchmark(sizes=(1000, 5000, 20000, 50000), seed=11):
    rng = random.Random(seed)
    heads = pseudo_words(rng, 400); mods = pseudo_words(rng, 400)
    rows = []
    for N in sizes:
        d = DuplicateDetector()
        used = set()
        t0 = time.perf_counter()
        facets_subject = None
        while len(d.store) < N:
            h, m = rng.choice(heads), rng.choice(mods)
            kind = rng.choice(["DEF"] * 3 + list(SC_OTHER))
            if (h, m, kind) in used:
                continue
            used.add((h, m, kind))
            c = f"{h} of {m}"
            q = rng.choice(SC_DEFINE).format(c=c) if kind == "DEF" else SC_OTHER[kind].format(c=c)
            d.add(q)
        build = time.perf_counter() - t0
        # queries: planted paraphrases (should be DUP) + novel cells (should be DISTINCT)
        planted = rng.sample([k for k in used if k[2] == "DEF"], 150)
        dup_q = [rng.choice([t for t in SC_DEFINE]).format(c=f"{h} of {m}") for h, m, _ in planted]
        novel_q = []
        while len(novel_q) < 150:
            h, m = rng.choice(heads), rng.choice(mods)
            if (h, m, "DEF") not in used and (h, m, "WHY") not in used:
                novel_q.append(SC_OTHER["WHY"].format(c=f"{h} of {m}"))
        lat, hit, fp, cands = [], 0, 0, []
        for q in dup_q:
            t = time.perf_counter(); v = d.check(q); lat.append(time.perf_counter() - t); cands.append(v.candidates_examined)
            hit += v.label.is_duplicate
        for q in novel_q:
            t = time.perf_counter(); v = d.check(q); lat.append(time.perf_counter() - t); cands.append(v.candidates_examined)
            fp += v.label is not Label.DISTINCT
        cov = CoverageMap(d.store)
        subject = f"{heads[0]} of {mods[0]}"
        facets = [f"{heads[i]} of {mods[0]}" for i in range(12)]
        prompt, st, _ = build_prompt(subject, 8, cov, facets, DEFAULT_INTENTS, 900)
        naive_tokens = sum(est_tokens(q.text) for q in d.store)
        lat.sort()
        rows.append(dict(N=N, cells=len(cov.cells), build_s=round(build, 2), p50_ms=round(1000 * statistics.median(lat), 2),
                         p95_ms=round(1000 * lat[int(.95 * len(lat))], 2), avg_candidates=round(statistics.mean(cands), 1),
                         recall=hit / len(dup_q), false_positive=fp / len(novel_q),
                         guided_prompt_tokens=st.tokens, naive_prompt_tokens=naive_tokens))
    return rows


# ------------------------------------------------------------------------------------------
def make_figure(path="figure_why_cosine_fails.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    rows = json.load(open("data/corpus.json"))
    ids = [r["id"] for r in rows]; txt = [r["text"] for r in rows]
    n = len(rows)
    tfm = WordTfidf(sorted(set(txt)))
    cos = np.zeros((n, n)); casc = np.full((n, n), 0); truth = np.full((n, n), 0); lex = np.zeros((n, n))
    d = DuplicateDetector()
    for r in rows:
        d.add(r["text"], id=r["id"])
    dup_group = {"Q1", "Q2", "Q3", "Q4", "Q10"}
    for i in range(n):
        for j in range(n):
            cos[i, j] = tfm.sim(txt[i], txt[j])
            lex[i, j] = embed.cosine(embed.encode(txt[i]), embed.encode(txt[j]))
            if i == j:
                continue
            m = d.compare_pair(txt[i], d.store.items[ids[i]].frame, d.store.items[ids[j]], lex[i, j])
            casc[i, j] = {Label.DISTINCT: 0, Label.REVIEW: 1, Label.SEMANTIC_DUPLICATE: 2, Label.EXACT_DUPLICATE: 2}[m.label]
            a, b = ids[i], ids[j]
            if a in dup_group and b in dup_group:
                truth[i, j] = 2
            elif (a == "Q8" and b in dup_group) or (b == "Q8" and a in dup_group):
                truth[i, j] = 1
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    cm = ListedColormap(["#eef1f4", "#f5b942", "#d6453d"])
    im = ax[0].imshow(np.where(np.eye(n) == 1, np.nan, cos), cmap="viridis", vmin=0, vmax=1)
    dupm = (truth == 2); difm = (truth == 0) & ~np.eye(n, dtype=bool)
    weakest, strongest = cos[dupm].min(), cos[difm].max()
    ax[0].set_title(f"A. Textbook baseline: word TF-IDF cosine\nweakest TRUE duplicate = {weakest:.2f}  <  strongest DIFFERENT pair = {strongest:.2f}", fontsize=11)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax[0].text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cos[i, j] < 0.6 else "black")
    ax[1].imshow(truth, cmap=cm, vmin=0, vmax=2); ax[1].set_title("B. Ground truth from the assignment\n(red = duplicate, amber = review, grey = different)", fontsize=11)
    ax[2].imshow(casc, cmap=cm, vmin=0, vmax=2); ax[2].set_title("C. DedupGuard cascade verdict\n(pairwise; matches B cell for cell)", fontsize=11)
    for a in ax:
        a.set_xticks(range(n)); a.set_yticks(range(n)); a.set_xticklabels(ids, fontsize=8, rotation=90); a.set_yticklabels(ids, fontsize=8)
    agree = int((casc == truth).sum() - n) / (n * n - n)
    fig.suptitle(f"Raw-text similarity cannot separate these (A: the ranges overlap).  Comparing what is ASKED and ABOUT WHAT (C) does.  "
                 f"Pairwise agreement with ground truth: {agree:.0%}", fontsize=11.5, y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    return agree


def main():
    out = {}
    ok, n, rows = eval_assignment()
    print(f"\n== 1. Assignment corpus (sequential insertion) : {ok}/{n} correct")
    for r in rows:
        print(f"   {'ok ' if r[2]==r[3] else 'XX '} {r[0]:<4} expected={r[2]:<19} got={r[3]:<19} {r[1]}")
    out["assignment"] = dict(correct=ok, total=n)

    res = eval_pairs()
    lines, strict, lenient, three = report_pairs(res)
    print(f"\n== 2/3. Stress set ({len(res)} hand-labelled pairs; DUP / DISTINCT / REVIEW)")
    print("   Baselines, each with its ORACLE-best threshold (a generous, unrealistic advantage):")
    for k, v in lines.items():
        print(f"     {k:<8} F1={v['f1']:.2f} (thr {v['thr']:.2f}, P={v['p']:.2f}, R={v['r']:.2f}); "
              f"{v['distinct_above_weakest_dup']}/{v['n_distinct']} genuinely-different pairs score >= the weakest true duplicate")
    print(f"   Cascade (no tuning to this set): strict F1={strict[0]:.2f} (P={strict[1]:.2f}, R={strict[2]:.2f}); "
          f"lenient[REVIEW counts as flagged] F1={lenient[0]:.2f} (P={lenient[1]:.2f}, R={lenient[2]:.2f}); 3-way accuracy={three:.0%}")
    bycat = Counter(); tot = Counter()
    for r in res:
        tot[r["cat"]] += 1; bycat[r["cat"]] += r["pred"] == r["y"]
    print("   by category:", {k: f"{bycat[k]}/{tot[k]}" for k in tot})
    wrong = [r for r in res if r["pred"] != r["y"]]
    print(f"   Misses ({len(wrong)}) - shown, not hidden:")
    for r in wrong:
        print(f"     [{r['cat']}] expected {r['y']}, got {r['pred']}:  {r['a']!r}  ~  {r['b']!r}")
    out["stress"] = dict(baselines=lines, cascade_strict=strict, cascade_lenient=lenient, three_way=three,
                         misses=[(r["a"], r["b"], r["y"], r["pred"]) for r in wrong])

    stats, fails = metamorphic()
    print("\n== 4. Metamorphic suite (invariances; tests internal consistency, not external validity)")
    for k, (g, t) in stats.items():
        print(f"     {k:<14} {g}/{t}  ({g/t:.1%})")
    for f in fails[:6]:
        print("       fail:", f)
    out["metamorphic"] = {k: dict(ok=g, total=t) for k, (g, t) in stats.items()}

    print("\n== 5. Scale benchmark (pure Python, single thread, this sandbox)")
    rows5 = scale_benchmark()
    print("   N       cells   build   p50ms  p95ms  cand  recall  FPR    prompt(guided)  prompt(naive dump)")
    for r in rows5:
        print(f"   {r['N']:<7} {r['cells']:<7} {r['build_s']:<6}s {r['p50_ms']:<6} {r['p95_ms']:<6} {r['avg_candidates']:<5} {r['recall']:<6.2f} {r['false_positive']:<6.2f} "
              f"{r['guided_prompt_tokens']:<15} {r['naive_prompt_tokens']}")
    out["scale"] = rows5
    agree = make_figure()
    print(f"\n   figure written; pairwise agreement on assignment corpus = {agree:.0%}")
    json.dump(out, open("results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
