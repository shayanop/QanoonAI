"""The answer format every chatbot must return, and a tolerant parser. Do not edit."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

DISCLAIMER = (
    "This is an AI-4016 student research prototype, not legal advice. "
    "Consult a qualified lawyer before acting on any legal matter."
)

# The fixed system prompt for A0 (no training). Groups may ADD memory text to the
# prompt through their memory_read node, but must not change this text.
SYSTEM_PROMPT = """You are a legal research assistant for Pakistani law.
Answer ONLY from the DOCUMENTS and MEMORY given to you. Do not use outside knowledge.
Cite every document you rely on by its id in square brackets as it appears in DOCUMENTS, for example [PPC-378].
If the question asks about the law on a specific date, use only the version in force on that date.
If the documents do not contain the answer, or two provisions conflict, set "abstain" to true and say you are not sure.

Reply with ONE JSON object and nothing else:
{"answer": "<your answer in plain English>", "citations": ["<doc id>", ...], "abstain": <true or false>}"""


@dataclass
class Answer:
    answer: str
    citations: list[str] = field(default_factory=list)
    abstain: bool = False
    parse_ok: bool = True        # False when the model output was not valid JSON (error code E8)

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> str:
        cites = ", ".join(self.citations) if self.citations else "none"
        return f"{self.answer}\n\nSources: {cites}\n\n{DISCLAIMER}"


_ID = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9\-_.@#/]*)\]")


def parse_answer(raw: str) -> Answer:
    """Parse model output. Accepts JSON (possibly inside ``` fences or with text around it).
    Falls back to the raw text with [ID] citations extracted, marked parse_ok=False."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "answer" in obj:
            cites = obj.get("citations") or []
            if isinstance(cites, str):
                cites = [cites]
            cites = [str(x).strip().strip("[]") for x in cites if str(x).strip()]
            ab = obj.get("abstain", False)
            ab = ab if isinstance(ab, bool) else str(ab).lower() in ("true", "yes", "1")
            return Answer(str(obj["answer"]), _dedupe(cites), ab, True)
    cites = [c.split("#")[0] for c in _ID.findall(text)]
    unsure = bool(re.search(r"not sure|cannot answer|can't answer|do not contain|insufficient", text, re.I))
    return Answer(text, _dedupe(cites), unsure, False)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = x.split("#")[0]
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out
