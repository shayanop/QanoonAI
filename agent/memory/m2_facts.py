"""M2: Structured fact memory. Build this only if your group drew M2.

Idea: extract typed case facts from each user message and keep them in a small table,
for example:

    {"type": "offence", "value": "theft of a mobile phone", "turn": 1, "score": 0.82}
    {"type": "date",    "value": "2025-03-03",              "turn": 1, "score": 0.91}
    {"type": "party",   "value": "neighbour",               "turn": 2, "score": 0.64}

What you must decide and document in MEMORY.md:
  - the fact schema (which types you keep);
  - how candidates are scored before they are kept (use the Lecture 10 memory-scoring
    formula: relevance, recency and importance) and what is never kept;
  - how a correction ("sorry, it was 5 March, not 3 March") replaces the old fact;
  - which facts are read back when the budget is tight.

Tip: extraction can use rules, the backbone model (`self.backend.generate`), or both.
"""
from __future__ import annotations

from .base import Memory, MemoryRead


class FactMemory(Memory):
    code = "M2"
    backend = None
    seed = 13

    def __init__(self):
        self.facts: dict[str, list[dict]] = {}

    def reset(self, session_id):
        self.facts[session_id] = []

    def read(self, session_id, query, budget_tokens, count_tokens) -> MemoryRead:
        # TODO (Step 2): score stored facts for this query and return the best ones as text.
        raise NotImplementedError("M2 FactMemory.read is not implemented yet")

    def write(self, session_id, turn, user_msg, answer, state):
        # TODO (Step 2): extract, score and store facts; handle corrections.
        raise NotImplementedError("M2 FactMemory.write is not implemented yet")
