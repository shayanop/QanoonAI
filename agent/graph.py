"""The QanoonArena legal agent as a LangGraph graph.

    memory_read -> retrieve -> generate -> cite_check -> memory_write

Only the memory (agent/memory/) and the citation monitor (agent/cite_check.py) are
yours to change. The prompt, retriever and backend are fixed by the harness so that
all groups are compared fairly.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from harness.answer import SYSTEM_PROMPT, Answer, parse_answer
from harness.backends import Backend
from harness.common import Corpus, load_config, sha256_text
from harness.retriever import BM25Retriever
from harness.tracing import NullTracer

from . import cite_check as cite_check_module
from .memory import Memory

PROMPT_HASH = sha256_text(SYSTEM_PROMPT)[:12]

try:  # LangGraph is the required framework; the fallback only keeps CI smoke tests alive.
    from langgraph.graph import END, START, StateGraph

    HAVE_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAVE_LANGGRAPH = False


class TurnState(TypedDict, total=False):
    session_id: str
    turn: int
    user_msg: str
    memory_text: str
    memory_ids: list
    mem_tokens: int
    retrieval_hint: str
    hits: list
    messages: list
    raw: str
    answer: Any
    tokens_in: int
    tokens_out: int
    cite_flags: dict
    written: list


class LegalAgent:
    def __init__(self, memory: Memory, backend: Backend, cfg: dict | None = None,
                 corpus: Corpus | None = None, retriever: BM25Retriever | None = None,
                 tracer=None, seed: int = 13):
        self.cfg = cfg or load_config()
        self.memory, self.backend, self.seed = memory, backend, seed
        self.corpus = corpus or Corpus()
        self.retriever = retriever or BM25Retriever(self.corpus, self.cfg)
        self.tracer = tracer or NullTracer()
        # give model-based memories access to the same backbone and seed
        if hasattr(memory, "backend"):
            memory.backend = backend
        if hasattr(memory, "seed"):
            memory.seed = seed
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def memory_read(self, s: TurnState) -> TurnState:
        with self.tracer.span("memory_read", memory=self.memory.code) as a:
            mr = self.memory.read(s["session_id"], s["user_msg"],
                                  self.cfg["context"]["memory_budget_tokens"], self.backend.count_tokens)
            mt = self.backend.count_tokens(mr.text) if mr.text else 0
            a.update(mem_tokens=mt, items_read=len(mr.items), items=mr.items[:20], text=mr.text[:2000])
        return {"memory_text": mr.text, "memory_ids": list(mr.doc_ids), "mem_tokens": mt,
                "retrieval_hint": mr.retrieval_hint}

    def retrieve(self, s: TurnState) -> TurnState:
        query = (s["user_msg"] + " " + s.get("retrieval_hint", "")).strip()
        with self.tracer.span("retrieve", query=query[:1000]) as a:
            hits = self.retriever.search(query)
            a.update(doc_ids=[h.doc_id for h in hits], scores=[h.score for h in hits])
        return {"hits": hits}

    def _build_messages(self, s: TurnState) -> list[dict]:
        cap = self.cfg["context"]["max_prompt_tokens"]
        ct = self.backend.count_tokens
        memory = s.get("memory_text") or "(empty)"
        head = SYSTEM_PROMPT
        tail = f"\n\nMEMORY:\n{memory}\n\nQUESTION:\n{s['user_msg']}"
        used = ct(head) + ct(tail) + ct("DOCUMENTS:\n") + 16
        docs = []
        for h in s["hits"]:
            block = h.text
            t = ct(block)
            if used + t > cap:
                remaining = cap - used
                if remaining > 64:  # keep a trimmed piece of the next document
                    block = block[: remaining * 4]
                    docs.append(block)
                break
            docs.append(block)
            used += t
        user = "DOCUMENTS:\n" + ("\n\n".join(docs) if docs else "(none found)") + tail
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def generate(self, s: TurnState) -> TurnState:
        messages = self._build_messages(s)
        with self.tracer.span("generate", backend=self.backend.name, prompt_hash=PROMPT_HASH) as a:
            raw, tin, tout = self.backend.generate(messages, self.seed)
            ans = parse_answer(raw)
            a.update(tokens_in=tin, tokens_out=tout, raw_output=raw[:4000], parse_ok=ans.parse_ok,
                     citations=ans.citations, abstain=ans.abstain)
        return {"messages": messages, "raw": raw, "answer": ans, "tokens_in": tin, "tokens_out": tout}

    def cite_check(self, s: TurnState) -> TurnState:
        allowed = {h.doc_id for h in s["hits"]} | set(s.get("memory_ids") or [])
        with self.tracer.span("cite_check") as a:
            ans, flags = cite_check_module.check(s["answer"], allowed, self.corpus)
            a.update(**flags)
        return {"answer": ans, "cite_flags": flags}

    def memory_write(self, s: TurnState) -> TurnState:
        with self.tracer.span("memory_write", memory=self.memory.code) as a:
            written = self.memory.write(s["session_id"], s["turn"], s["user_msg"], s["answer"], dict(s)) or []
            a.update(items_written=len(written), items=written[:20])
        return {"written": written}

    # ------------------------------------------------------------------ graph
    def _build(self):
        order = ["memory_read", "retrieve", "generate", "cite_check", "memory_write"]
        if HAVE_LANGGRAPH:
            g = StateGraph(TurnState)
            for name in order:
                g.add_node(name, getattr(self, name))
            g.add_edge(START, order[0])
            for a, b in zip(order, order[1:]):
                g.add_edge(a, b)
            g.add_edge(order[-1], END)
            return g.compile()
        print("[agent] langgraph is not installed; running nodes in order (install requirements.txt).")
        return _Sequential([getattr(self, n) for n in order])

    # ------------------------------------------------------------------ public
    def reset(self, session_id: str) -> None:
        self.memory.reset(session_id)

    def respond(self, session_id: str, turn: int, user_msg: str, **trace_attrs) -> tuple[Answer, dict]:
        """Answer one user message. Returns the Answer and turn info for scoring."""
        t0 = time.perf_counter()
        with self.tracer.span("turn", session_id=session_id, turn=turn, prompt_hash=PROMPT_HASH,
                              user_msg=user_msg, **trace_attrs) as a:
            out = self.graph.invoke({"session_id": session_id, "turn": turn, "user_msg": user_msg})
            latency = int((time.perf_counter() - t0) * 1000)
            ans: Answer = out["answer"]
            info = {
                "tokens_in": out.get("tokens_in", 0), "tokens_out": out.get("tokens_out", 0),
                "latency_ms": latency, "mem_tokens": out.get("mem_tokens", 0),
                "allowed_ids": {h.doc_id for h in out.get("hits", [])} | set(out.get("memory_ids") or []),
                "retrieved": [h.doc_id for h in out.get("hits", [])],
                "cite_flags": out.get("cite_flags", {}),
            }
            a.update(tokens_in=info["tokens_in"], tokens_out=info["tokens_out"], latency_ms=latency,
                     mem_tokens=info["mem_tokens"], answer=ans.to_dict())
        return ans, info


class _Sequential:
    def __init__(self, fns):
        self.fns = fns

    def invoke(self, state):
        state = dict(state)
        for fn in self.fns:
            state.update(fn(state) or {})
        return state
