import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dedupguard import (CoverageMap, DuplicateDetector, Label, MockLLM, QuestionGenerator, QuestionStore, parse)
from dedupguard.coverage import DEFAULT_INTENTS, build_prompt, est_tokens
from dedupguard.detector import Judge
from dedupguard.embed import DenseEmbedder
from dedupguard.frame import Intent
from dedupguard.normalize import canonical, exact_key, words_to_numbers

with open(os.path.join(os.path.dirname(__file__), "..", "data", "corpus.json")) as _fh:
    CORPUS = json.load(_fh)


def loaded(tags=None, upto=None):
    d = DuplicateDetector()
    for r in CORPUS[:upto]:
        d.add(r["text"], id=r["id"], tags=tags)
    return d


class TestNormalize(unittest.TestCase):
    def test_exact_key_ignores_presentation(self):
        base = exact_key("Define the reflection of light.")
        for v in ["DEFINE THE REFLECTION OF LIGHT??", "Q3.  define the reflection of light [2 marks]",
                  "  Define   the reflection   of light ", "Define the reflection of light. (a) x (b) y"]:
            self.assertEqual(exact_key(v), base, v)

    def test_number_words(self):
        self.assertEqual(words_to_numbers("thirty five degrees"), "35 degrees")
        self.assertEqual(words_to_numbers("twenty-five"), "25")
        self.assertIn("30 degrees", canonical("thirty degrees"))
        self.assertEqual(canonical("30\u00b0"), canonical("30 degrees"))

    def test_units_only_alias_after_digit(self):
        self.assertIn("7 cm", canonical("radius 7 centimetres"))
        self.assertIn("second law", canonical("Newton's second law"))        # 'second' must survive


class TestFrame(unittest.TestCase):
    def test_assignment_intents(self):
        want = ["DEFINE", "DEFINE", "DEFINE", "DEFINE", "ENUMERATE", "DEFINE", "COMPUTE", "DEFINE", "DIFFERENTIATE", "DEFINE"]
        self.assertEqual([parse(r["text"]).intent.value for r in CORPUS], want)

    def test_light_reflection_equals_reflection_of_light(self):
        self.assertEqual(parse("What is reflection of light?").concept, parse("Explain the meaning of light reflection.").concept)

    def test_head_differs_for_angle_of_reflection(self):
        self.assertNotEqual(parse("What is reflection?").concept.head, parse("What is the angle of reflection?").concept.head)

    def test_unknown_shapes_abstain(self):
        self.assertLess(parse("Reflection. Discuss quickly with reference to nothing").confidence, 0.9)


class TestDetectorAssignment(unittest.TestCase):
    def test_sequential_protocol_matches_ground_truth(self):
        d = DuplicateDetector()
        for r in CORPUS:
            v = d.check(r["text"])
            got = "ORIGINAL" if len(d.store) == 0 else v.label.name
            self.assertEqual(got, r["expected"], f'{r["id"]}: {v.explain()}')
            d.add(r["text"], id=r["id"])

    def test_q8_is_review_and_explains_why(self):
        v = loaded(upto=7).check("What is reflection?")        # Q8 arriving after Q1-Q7
        self.assertEqual(v.label, Label.REVIEW)
        self.assertTrue(any("light" in e.detail for e in v.best.evidence if e.voter == "frame"))

    def test_numeric_variant_is_distinct_and_flagged(self):
        d = loaded()
        v = d.check("The angle of incidence is 45\u00b0. What is the angle of reflection?")
        self.assertEqual(v.label, Label.DISTINCT)
        self.assertIn("numeric_variant", [f for m in v.matches for f in m.flags])
        v2 = d.check("Find the angle of reflection if the angle of incidence is thirty degrees.")
        self.assertTrue(v2.label.is_duplicate)

    def test_audit_finds_the_known_cluster(self):
        self.assertEqual(loaded().audit(), [["Q1", "Q2", "Q3", "Q4", "Q10"]])


class TestTagsDoNotDecide(unittest.TestCase):
    def test_duplicate_found_across_different_and_unknown_tag_schemas(self):
        d = DuplicateDetector()
        d.add("What is photosynthesis?", tags={"grade": 7, "board": "CBSE"})
        d.add("What is osmosis?", tags={"brand_new_key": ["any", "shape"]})
        v = d.check("Define photosynthesis.")        # candidate carries no tags at all
        self.assertTrue(v.label.is_duplicate)
        self.assertEqual(v.best.question.tags["board"], "CBSE")

    def test_same_tags_do_not_create_false_duplicates(self):
        d = DuplicateDetector()
        d.add("What is photosynthesis?", tags={"topic": "bio"})
        self.assertEqual(d.check("Why is photosynthesis important?").label, Label.DISTINCT)


class TestPluggableVoters(unittest.TestCase):
    def test_dense_voter_runs_and_cannot_override_confident_frame(self):
        import hashlib

        def fake(xs):
            out = []
            for x in xs:
                v = [0.0] * 32
                for w in x.lower().split():
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1
                out.append(v)
            return out
        d = DuplicateDetector(dense=DenseEmbedder(fake))
        d.add("What is reflection of light?")
        v = d.check("State the laws of reflection of light.")
        self.assertEqual(v.label, Label.DISTINCT)
        self.assertTrue(any(e.voter == "dense" for e in v.best.evidence))

    def test_judge_called_only_for_gray_zone(self):
        class J:
            n = 0

            def judge(self, a, b):
                J.n += 1
                return "different", "stub"
        d = DuplicateDetector(judge=J())
        for r in CORPUS[:7]:
            d.add(r["text"])
        d.check("Define the reflection of light.")        # exact -> no judge
        d.check("What is the difference between mass and weight?")       # clearly different -> no judge
        self.assertEqual(J.n, 0)
        v = d.check("What is reflection?")                  # REVIEW -> judge arbitrates
        self.assertGreater(J.n, 0)
        self.assertEqual(v.label, Label.DISTINCT)


class TestPromptBudget(unittest.TestCase):
    def test_budget_is_a_hard_bound_and_keeps_a_target(self):
        cov = CoverageMap(loaded().store)
        facets = ["plane mirror", "angle of incidence", "periscope", "lateral inversion", "regular reflection"]
        for b in (350, 500, 900, 2000):
            p, st, targets = build_prompt("reflection of light", 8, cov, facets, DEFAULT_INTENTS, b)
            self.assertLessEqual(est_tokens(p), b, b)
            self.assertGreaterEqual(len(targets), 1)

    def test_prompt_size_independent_of_corpus_size(self):
        small = loaded()
        big = loaded()
        for i in range(3000):
            big.add(f"What is zorbit{i} of kalm{i % 40}?")
        facets = ["plane mirror", "periscope"]
        a = build_prompt("reflection of light", 8, CoverageMap(small.store), facets, DEFAULT_INTENTS, 900)[1].tokens
        b = build_prompt("reflection of light", 8, CoverageMap(big.store), facets, DEFAULT_INTENTS, 900)[1].tokens
        self.assertLessEqual(b, 900)
        self.assertLess(abs(a - b), 400)


FACETS = ["reflection of light", "laws of reflection of light", "regular reflection", "diffuse reflection", "plane mirror",
          "angle of incidence", "periscope", "lateral inversion", "angle of reflection"]


class TestGeneration(unittest.TestCase):
    def test_generated_set_has_no_duplicates_even_with_a_leaky_llm(self):
        d = loaded()
        rep = QuestionGenerator(d, MockLLM(leak_rate=0.4, copy_rate=0.2, seed=5)).generate("reflection of light", 8, FACETS, max_rounds=8)
        self.assertGreaterEqual(len(rep.accepted), 6)
        self.assertGreater(len(rep.rejected), 0)                                  # the LLM really did leak...
        self.assertEqual(d.audit(), [["Q1", "Q2", "Q3", "Q4", "Q10"]])         # ...and nothing new clustered anywhere

    def test_hostile_llm_cannot_get_a_duplicate_in(self):
        class Hostile:
            def complete(self, prompt):
                return json.dumps([{"question": q} for q in ["Define the reflection of light.", "What is reflection of light?",
                                                              "Explain the meaning of light reflection.", "WHAT IS REFLECTION OF LIGHT??"]])
        d = loaded()
        before = len(d.store)
        rep = QuestionGenerator(d, Hostile()).generate("reflection of light", 4, FACETS, max_rounds=3)
        self.assertEqual(len(d.store), before)
        self.assertEqual(len(rep.accepted), 0)
        self.assertGreaterEqual(len(rep.rejected), 4)

    def test_feedback_reaches_the_next_prompt(self):
        prompts = []

        class Spy:
            def __init__(self):
                self.inner = MockLLM(leak_rate=1.0, copy_rate=0.0, seed=1)

            def complete(self, p):
                prompts.append(p)
                return self.inner.complete(p)
        QuestionGenerator(loaded(), Spy()).generate("reflection of light", 3, FACETS, max_rounds=2)
        self.assertNotIn("REJECTED LAST ROUND", prompts[0])
        self.assertIn("REJECTED LAST ROUND", prompts[1])


class TestIndependence(unittest.TestCase):
    def test_detector_has_no_dependency_on_generation_layer(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "dedupguard", "detector.py")) as fh:
            src = fh.read()
        self.assertIsNone(re.search(r"^\s*(from|import)\s+\.?(coverage|generate|llm)\b", src, re.M))

    def test_detector_works_with_prevention_entirely_absent(self):
        d = DuplicateDetector()
        d.add("What is osmosis?")
        self.assertTrue(d.check("Define osmosis.").label.is_duplicate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
