"""
python demo.py   -  a guided tour of DedupGuard on the assignment's reflection-of-light corpus.
Everything runs offline; the LLM is a deterministic simulation (see llm.MockLLM) so the plumbing is visible.
"""
import json
import sys
sys.path.insert(0, ".")
from dedupguard import CoverageMap, DuplicateDetector, MockLLM, QuestionGenerator
from dedupguard.coverage import DEFAULT_INTENTS, build_prompt

BAR = "=" * 94
corpus = json.load(open("data/corpus.json"))
FACETS = ["reflection of light", "laws of reflection of light", "regular reflection", "diffuse reflection",
          "plane mirror", "angle of incidence", "periscope", "lateral inversion", "angle of reflection"]


def h(t):
    print(f"\n{BAR}\n{t}\n{BAR}")


# ------------------------------------------------------------------------------------------------
h("PART 1 - INDEPENDENT DETECTION: each question vs. everything before it (assignment protocol)")
det = DuplicateDetector()
for r in corpus:
    v = det.check(r["text"])
    got = "ORIGINAL" if len(det.store) == 0 else v.label.name
    print(f"{r['id']:<4}{r['text']:<66} -> {got:<18} {'[OK]' if got == r['expected'] else '[MISMATCH expected ' + r['expected'] + ']'}")
    det.add(r["text"], id=r["id"], tags={"subject": "physics"})

h("PART 1b - WHY it decides that way (evidence trail; each probed against only the questions that precede it)")
for qid, upto, note in [("Q4", 3, "paraphrase in different template wording -> caught"),
                        ("Q5", 4, "shares 4 of 5 content words with Q1 -> correctly NOT a duplicate"),
                        ("Q8", 7, "subsumption -> escalated to a human, with the exact reason")]:
    probe = DuplicateDetector()
    for r in corpus[:upto]:
        probe.add(r["text"], id=r["id"])
    text = corpus[int(qid[1:]) - 1]["text"]
    print(f"{qid}: {text!r}   <- {note}")
    print("   " + probe.check(text).explain(limit=2).replace("\n", "\n   ") + "\n")

h("PART 1c - robustness: formats, spelled-out numbers, compound questions")
for q in ["  Q7)  DEFINE   the Reflection of Light??  [2 marks]",
          "Find the angle of reflection if the angle of incidence is thirty degrees.",
          "The angle of incidence is 45\u00b0. What is the angle of reflection?",
          "What is reflection of light and why does it occur?"]:
    v = det.check(q)
    b = v.best
    print(f"{q!r}\n    -> {v.label.name} (conf {v.confidence:.2f})  closest: {b.question.id if b else '-'}  "
          f"{'| ' + '; '.join(b.flags) if b and b.flags else ''}")

h("PART 2 - GENERATION-TIME PREVENTION: the coverage map replaces 'paste the whole corpus'")
cov = CoverageMap(det.store)
print(f"{len(det.store)} questions collapse into {len(cov.cells)} cells (kind-of-ask x concept):")
for (it, key), c in cov.cells.items():
    print(f"   [{it.value:<13}] {c.phrase:<28} x{c.count}   {c.ids}")
prompt, st, _ = build_prompt("reflection of light", 8, cov, FACETS, DEFAULT_INTENTS, 900)
print(f"\nThe prompt the LLM actually receives ({st.tokens} est. tokens, budget 900):\n" + "-" * 94)
print(prompt)

h("PART 3 - THE FULL LOOP (simulated LLM): prevention + independent safety net + feedback")
for label, llm in [("PLAIN prompt  (no coverage map; LLM gravitates to obvious questions)", MockLLM(naive=True, seed=3)),
                   ("GUIDED prompt (leaky: still ignores the avoid-list 30% / copies examples 15%)", MockLLM(leak_rate=0.30, copy_rate=0.15, seed=3))]:
    d = DuplicateDetector()
    for r in corpus:
        d.add(r["text"], id=r["id"])
    rep = QuestionGenerator(d, llm).generate("reflection of light", 8, FACETS, max_rounds=4)
    print(f"\n>> {label}\n   {rep.summary()}")
    for q in rep.accepted:
        print(f"     + {q.id:<4}{q.text}")
    for r in rep.rejected[:4]:
        b = r.verdict.best
        print(f"     x {r.text!r:<60} {r.verdict.label.name} of {b.question.id}")
    if len(rep.rejected) > 4:
        print(f"     x ... and {len(rep.rejected) - 4} more rejected")
    if label.startswith("GUIDED"):
        print(f"\n   Independent audit of the final {len(d.store)}-question store -> duplicate clusters: {d.audit()}")
        print("   (only the original Q1/Q2/Q3/Q4/Q10 cluster from the assignment data remains; nothing generated got in)")
print("\nNOTE: MockLLM is a behavioural simulation. It demonstrates the pipeline and the safety net; it is NOT evidence of how\n"
      "often a real LLM repeats itself. See README 'How I would evaluate this for real'.")
