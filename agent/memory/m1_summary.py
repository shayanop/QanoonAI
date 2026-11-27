"""M1: Summary memory. Build this only if your group drew M1.

Idea: after every turn, rewrite a short summary of the conversation so far, and give
the bot that summary instead of raw turns.

What you must decide and document in MEMORY.md:
  - how the summary is produced (the same backbone model with a summary prompt is fine);
  - how you stop it from losing legally important facts (dates, sections, parties);
  - how you stop it from drifting or inventing facts over many turns;
  - what happens when the summary does not fit memory_budget_tokens.

Tip: you can reach the model through `self.backend.generate(messages, seed)`; the
agent sets `self.backend` and `self.seed` for you before the first call.
"""
from __future__ import annotations

from .base import Memory, MemoryRead


class SummaryMemory(Memory):
    code = "M1"
    backend = None
    seed = 13

    def __init__(self):
        self.summary: dict[str, str] = {}

    def reset(self, session_id):
        self.summary[session_id] = ""

    def read(self, session_id, query, budget_tokens, count_tokens) -> MemoryRead:
        # TODO (Step 2): return the current summary, trimmed to budget_tokens.
        raise NotImplementedError("M1 SummaryMemory.read is not implemented yet")

    def write(self, session_id, turn, user_msg, answer, state):
        # TODO (Step 2): rewrite self.summary[session_id] using the new turn.
        raise NotImplementedError("M1 SummaryMemory.write is not implemented yet")
