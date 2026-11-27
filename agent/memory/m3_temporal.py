"""M3: Temporal graph memory. Build this only if your group drew M3.

Idea: store facts and the laws they concern as a small graph whose edges carry dates:

    (matter)  --occurred_on--> (2025-03-03)
    (matter)  --concerns-----> (PPC-379)
    (PPC-489F) --amended_by--> (Ord. 85 of 2002)   year=2002

Amended sections in the corpus end with their official amendment notes, e.g.
"[Amendment notes: Ins. by Ord. 85 of 2002, s.2.]", and carry them as a list in
`Corpus().get("PPC-489F").meta["amended_by"]`. When the user's matter has a date, memory
should let the bot reason about which amendments came before or after that date.

What you must decide and document in MEMORY.md:
  - node and edge types, and where the graph lives (NetworkX in memory is enough;
    Neo4j is allowed);
  - how dates in user messages are found and normalised;
  - how a correction to a date or fact updates the graph;
  - what is read back when the budget is tight.

Tip: return useful words in MemoryRead.retrieval_hint (for example the Act, the section
number and the date) so the fixed retriever finds the right section.
"""
from __future__ import annotations

from .base import Memory, MemoryRead


class TemporalGraphMemory(Memory):
    code = "M3"
    backend = None
    seed = 13

    def __init__(self):
        self.graphs: dict[str, object] = {}

    def reset(self, session_id):
        self.graphs[session_id] = None

    def read(self, session_id, query, budget_tokens, count_tokens) -> MemoryRead:
        # TODO (Step 2): find facts and law versions relevant to this query and date.
        raise NotImplementedError("M3 TemporalGraphMemory.read is not implemented yet")

    def write(self, session_id, turn, user_msg, answer, state):
        # TODO (Step 2): add or update dated facts in the graph.
        raise NotImplementedError("M3 TemporalGraphMemory.write is not implemented yet")
