"""Runtime citation monitor (Step 4, item 2).

The agent calls check() in the cite_check node, after generation and before memory_write.
`allowed_ids` holds every document id the bot saw this turn (retrieved + memory).

Baseline behaviour: pass the answer through unchanged and only report flags.
Your task: decide what to do with ungrounded citations (flag only, drop them, or block
the answer and abstain), implement it here, and state your choice in your report.
The flags you return are written to the cite_check span of every trace.
"""
from __future__ import annotations


def check(answer, allowed_ids: set[str], corpus) -> tuple[object, dict]:
    allowed_base = {a.split("@")[0] for a in allowed_ids}
    ungrounded = [
        c for c in answer.citations
        if c not in allowed_ids and c.split("@")[0] not in allowed_base
    ]
    flags = {
        "ungrounded": ungrounded,
        "n_ungrounded": len(ungrounded),
        "action": "none",  # TODO (Step 4): change to "flagged", "dropped" or "blocked" when you implement a policy
    }
    return answer, flags
