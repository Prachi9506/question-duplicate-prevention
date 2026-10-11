"""LLM adapters.  The pipeline only needs `complete(prompt) -> str`."""
from __future__ import annotations

import json
import os
import random
import re
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class AnthropicLLM:
    """Real backend (not exercised in the offline sandbox this was developed in)."""

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 1500, api_key: str | None = None):
        import anthropic  # lazy import
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model, self.max_tokens = model, max_tokens

    def complete(self, prompt: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=0.9,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class LLMJudge:
    """Optional gray-zone arbiter for the detector (only invoked on REVIEW cases)."""

    PROMPT = ("Two exam questions are DUPLICATES if a student answering one has necessarily answered the other "
              "(same ask, same concept, same given data) even if worded differently. They are DIFFERENT if the ask, "
              "the concept or the numbers differ.\nA: {a}\nB: {b}\nAnswer with one word - duplicate, different or "
              "unsure - then a short reason.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def judge(self, a: str, b: str) -> tuple[str, str]:
        out = self.llm.complete(self.PROMPT.format(a=a, b=b)).strip()
        first = out.split()[0].lower().strip(".,:") if out else "unsure"
        return (first if first in ("duplicate", "different", "unsure") else "unsure"), out[:200]


# --------------------------------------------------------------------------------------------
# MockLLM - a *behavioural simulation* so the whole loop runs offline and deterministically.
#
# It reads the same prompt a real model would.  Three knobs model real failure modes:
#   leak_rate    probability it ignores the COVERED list and re-asks a covered idea in new words
#   copy_rate    probability it copies a STYLE EXAMPLE verbatim (a classic LLM failure)
#   naive        ignores TARGET/COVERED entirely (what you get with a plain "generate 8 questions" prompt)
# It proves the *plumbing and safety net*, NOT real-world LLM duplicate rates.
# --------------------------------------------------------------------------------------------
_REAL = {
    "DEFINE": ["What is {f}?", "Define {f}.", "What do you understand by {f}?", "Explain the meaning of {f}."],
    "WHY": ["Why is {f} important?", "Why does {f} matter in everyday life?"],
    "HOW": ["How does {f} work?", "How is {f} observed in daily life?"],
    "EXAMPLE": ["Give two examples of {f} from daily life.", "Give an example of {f}."],
    "ENUMERATE": ["List the main properties of {f}.", "List the characteristics of {f}."],
    "COMPUTE": ["Given the value {n} units, calculate the {f}.", "Find the {f} if the given quantity is {n} units."],
    "DIFFERENTIATE": ["Differentiate between {f}.", "What is the difference between {f}?"],
}


class MockLLM:
    def __init__(self, leak_rate: float = 0.25, copy_rate: float = 0.1, naive: bool = False, seed: int = 7):
        self.leak_rate, self.copy_rate, self.naive = leak_rate, copy_rate, naive
        self.rng = random.Random(seed)
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        n = int(re.search(r"Write exactly (\d+) questions", prompt).group(1))
        targets = re.findall(r"^\s*(\d+)\. \[(\w+)\] (.+?) - ", prompt, re.M)
        covered = re.findall(r'^\s*- \[(\w+)\] (.+?) \(x\d+\) e\.g\. "(.+?)"$', prompt, re.M)
        examples = re.findall(r"STYLE EXAMPLES.*?:\n((?: - .+\n?)+)", prompt)
        ex_lines = [l[3:].strip() for l in examples[0].splitlines()] if examples else []
        out = []
        pool = targets[:]
        for i in range(n):
            r = self.rng.random()
            if self.naive or not pool:
                # plain prompt: LLM gravitates to the most "obvious" questions about the subject
                subj = re.search(r"subject: (.+?)\.", prompt).group(1)
                intent = self.rng.choice(["DEFINE", "DEFINE", "DEFINE", "WHY", "ENUMERATE"])
                out.append({"cell": 0, "question": self.rng.choice(_REAL[intent]).format(f=subj, n=30)})
            elif r < self.copy_rate and ex_lines:
                out.append({"cell": 0, "question": self.rng.choice(ex_lines)})
            elif r < self.copy_rate + self.leak_rate and covered:
                it, phrase, _ = self.rng.choice(covered)
                out.append({"cell": 0, "question": self.rng.choice(_REAL.get(it, _REAL["DEFINE"])).format(f=phrase, n=30)})
            else:
                num, it, facet = pool.pop(0)
                facet = facet.strip()
                out.append({"cell": int(num), "question": self.rng.choice(_REAL[it]).format(f=facet, n=self.rng.choice([20, 35, 50, 65]))})
        return "```json\n" + json.dumps(out, indent=1) + "\n```"
