"""The memory interface every memory type implements.

The agent calls read() in the memory_read node BEFORE answering, and write() in the
memory_write node AFTER answering. Keep tool outputs and retrieved documents in graph
state and traces; write to memory only information you deliberately choose to keep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MemoryRead:
    text: str = ""                                   # what is placed in the prompt under MEMORY
    doc_ids: list[str] = field(default_factory=list)  # document ids mentioned in that text (count as "seen")
    items: list[dict] = field(default_factory=list)   # structured view of what was read, for traces
    retrieval_hint: str = ""                          # extra words added to the retrieval query (may be empty)


class Memory:
    code = "M?"

    def reset(self, session_id: str) -> None:
        """Forget everything about one conversation (called before each new session)."""
        raise NotImplementedError

    def read(self, session_id: str, query: str, budget_tokens: int,
             count_tokens: Callable[[str], int]) -> MemoryRead:
        """Return what the bot should remember for this message, within budget_tokens."""
        raise NotImplementedError

    def write(self, session_id: str, turn: int, user_msg: str, answer, state: dict) -> list[dict]:
        """Update memory after the bot answered. Return the items written (for traces)."""
        raise NotImplementedError


def fit_to_budget(lines: list[str], budget: int, count_tokens: Callable[[str], int]) -> str:
    """Keep the LAST lines that fit in the token budget (newest information wins)."""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        t = count_tokens(line) + 1
        if used + t > budget:
            break
        kept.append(line)
        used += t
    return "\n".join(reversed(kept))
