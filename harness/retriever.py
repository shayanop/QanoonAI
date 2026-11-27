"""Fixed BM25 retriever over the QanoonArena corpus. Do not edit.

Every group uses this retriever with the settings in config.yaml, so that retrieval
is the same for everyone and only memory and training differ.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .common import Corpus, load_config

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with "
    "what who whom whose shall any such may under section sections act code".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    score: float
    title: str
    text: str


class BM25Retriever:
    def __init__(self, corpus: Corpus | None = None, cfg: dict | None = None):
        cfg = cfg or load_config()
        r = cfg["retrieval"]
        self.top_k, self.k1, self.b = r["top_k"], r["bm25_k1"], r["bm25_b"]
        self.chunk_chars = r["chunk_chars"]
        self.corpus = corpus or Corpus()
        self.chunks: list[tuple[str, str, str, str]] = []  # (doc_id, chunk_id, title, text)
        for d in self.corpus.docs.values():
            header = self._header(d)
            pieces = self._split(d.text)
            for i, piece in enumerate(pieces):
                cid = d.doc_id if len(pieces) == 1 else f"{d.doc_id}#{i}"
                self.chunks.append((d.doc_id, cid, d.title, f"{header}\n{piece}"))
        self.tf = [Counter(tokenize(c[3])) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    @staticmethod
    def _header(d) -> str:
        m = d.meta
        if d.kind == "statute":
            ver = ""
            if m.get("valid_from") or m.get("valid_to"):
                ver = f" [in force {m.get('valid_from') or 'start'} to {m.get('valid_to') or 'present'}]"
            return f"[{d.doc_id}] {m.get('act', '')}, section {m.get('section', '')}: {d.title}{ver}"
        return f"[{d.doc_id}] {d.title} ({m.get('court', '')}, {m.get('date', '')})"

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_chars:
            return [text]
        out, start = [], 0
        while start < len(text):
            end = min(len(text), start + self.chunk_chars)
            cut = text.rfind(" ", start, end) if end < len(text) else end
            cut = cut if cut > start + self.chunk_chars // 2 else end
            out.append(text[start:cut].strip())
            start = cut
        return [p for p in out if p]

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        q = tokenize(query)
        # exact section mentions ("section 378", "s. 378") get a strong boost
        sec_nums = set(re.findall(r"(?:section|sec\.?|s\.)\s*(\d+[a-z]*)", query.lower()))
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if sec_nums:
                doc = self.corpus.get(self.chunks[i][0])
                if doc is not None and str(doc.meta.get("section", "")).lower() in sec_nums:
                    s += 5.0
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        hits = []
        for s, i in scores[: top_k or self.top_k]:
            doc_id, cid, title, text = self.chunks[i]
            hits.append(Hit(doc_id, cid, round(s, 4), title, text))
        return hits
