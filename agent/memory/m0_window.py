"""M0: the stateless baseline. The bot sees only the last k raw turns (sliding window).

This is Version 1 for every group. Do not change it; build your own type in its file.
"""
from __future__ import annotations

from harness.common import load_config

from .base import Memory, MemoryRead, fit_to_budget


class WindowMemory(Memory):
    code = "M0"

    def __init__(self, window_turns: int | None = None):
        self.k = window_turns if window_turns is not None else load_config()["context"]["window_turns"]
        self.turns: dict[str, list[tuple[str, str, list[str]]]] = {}

    def reset(self, session_id):
        self.turns[session_id] = []

    def read(self, session_id, query, budget_tokens, count_tokens):
        recent = self.turns.get(session_id, [])[-self.k:] if self.k > 0 else []
        lines, ids = [], []
        for user, bot, cites in recent:
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {bot}")
            ids.extend(cites)
        text = fit_to_budget(lines, budget_tokens, count_tokens)
        hint = " ".join(u for u, _, _ in recent)
        return MemoryRead(text=text, doc_ids=ids, items=[{"user": u} for u, _, _ in recent], retrieval_hint=hint)

    def write(self, session_id, turn, user_msg, answer, state):
        self.turns.setdefault(session_id, []).append((user_msg, answer.answer, list(answer.citations)))
        return [{"turn": turn, "stored": "raw turn"}]
