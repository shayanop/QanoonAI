"""Check a question file before you submit it (Step 1). Do not edit.

    python -m harness.validate_sessions contrib/G07.jsonl --contrib

Checks: the format, that every cited document exists, that every evidence quote really
appears in the cited document, that memory questions point to earlier turns, and that
T5 questions expect "I am not sure". With --contrib it also checks the Step 1 rule:
10 sessions, 2 of each type T1 to T5.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from .common import TYPES, Corpus, norm_text, read_jsonl

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
GOLD_KEYS = {"answer", "answer_keywords", "citations", "exact_version", "abstain",
             "memory_dependent", "depends_on_turns", "as_of", "evidence"}


def validate(sessions: list[dict], corpus: Corpus, contrib: bool = False) -> list[str]:
    errs: list[str] = []
    ids = Counter(s.get("session_id") for s in sessions)
    for sid, n in ids.items():
        if n > 1:
            errs.append(f"{sid}: session_id used {n} times")

    for s in sessions:
        sid = s.get("session_id", "<missing id>")
        if not s.get("session_id"):
            errs.append("a session has no session_id")
        typ = s.get("type")
        if typ not in TYPES:
            errs.append(f"{sid}: type must be one of {TYPES}, got {typ!r}")
        turns = s.get("turns")
        if not isinstance(turns, list) or not turns:
            errs.append(f"{sid}: turns must be a non-empty list")
            continue
        if not 1 <= len(turns) <= 6:
            errs.append(f"{sid}: a session has 1 to 6 turns (has {len(turns)})")
        if typ in ("T3", "T4") and len(turns) < 2:
            errs.append(f"{sid}: {typ} sessions need at least 2 turns")
        n_gold = 0
        for i, t in enumerate(turns, 1):
            where = f"{sid} turn {i}"
            if not isinstance(t.get("user"), str) or not t["user"].strip():
                errs.append(f"{where}: 'user' text is missing")
            g = t.get("gold")
            if g is None:
                continue
            n_gold += 1
            unknown = set(g) - GOLD_KEYS
            if unknown:
                errs.append(f"{where}: unknown gold fields {sorted(unknown)}")
            if not isinstance(g.get("answer"), str) or not g["answer"].strip():
                errs.append(f"{where}: gold.answer (reference answer) is missing")
            if not isinstance(g.get("abstain"), bool):
                errs.append(f"{where}: gold.abstain must be true or false")
            cites = g.get("citations", [])
            if not isinstance(cites, list):
                errs.append(f"{where}: gold.citations must be a list")
                cites = []
            if not g.get("abstain") and not cites:
                errs.append(f"{where}: an answerable turn needs at least one gold citation")
            for c in cites:
                if not corpus.exists(c):
                    errs.append(f"{where}: cited document {c!r} is not in the corpus")
            kws = g.get("answer_keywords", [])
            if not isinstance(kws, list) or not all(isinstance(k, str) and k.strip() for k in kws):
                errs.append(f"{where}: gold.answer_keywords must be a list of non-empty strings")
            elif not g.get("abstain") and not kws and not g.get("exact_version"):
                errs.append(f"{where}: give 1-4 answer_keywords so the answer can be marked")
            if g.get("as_of") is not None and not DATE.match(str(g["as_of"])):
                errs.append(f"{where}: gold.as_of must be YYYY-MM-DD or null")
            if typ == "T4" and not g.get("abstain") and not any(re.search(r"\b(1[89]|20)\d\d\b", k) for k in kws or []):
                errs.append(f"{where}: T4 gold turns need the amendment year among answer_keywords")
            if g.get("memory_dependent"):
                dep = g.get("depends_on_turns") or []
                if not dep or not all(isinstance(d, int) and 1 <= d < i for d in dep):
                    errs.append(f"{where}: memory_dependent turns must list earlier turn numbers in depends_on_turns")
            if typ == "T5" and not g.get("abstain"):
                errs.append(f"{where}: T5 gold turns must expect abstain true")
            ev = g.get("evidence", [])
            if not g.get("abstain") and not ev:
                errs.append(f"{where}: give evidence: the exact supporting text from the corpus")
            for e in ev or []:
                doc = corpus.get(e.get("doc_id", ""))
                if doc is None:
                    errs.append(f"{where}: evidence doc_id {e.get('doc_id')!r} not found (use the exact id, with @version if any)")
                elif norm_text(e.get("quote", "")) not in norm_text(doc.text) or len(e.get("quote", "")) < 20:
                    errs.append(f"{where}: evidence quote is not found word-for-word in {doc.doc_id} (min 20 characters)")
        if n_gold == 0:
            errs.append(f"{sid}: at least one turn needs a gold answer")
        if typ in ("T3", "T4") and not any((t.get("gold") or {}).get("memory_dependent") for t in turns):
            errs.append(f"{sid}: {typ} sessions need a gold turn with memory_dependent true")

    if contrib:
        c = Counter(s.get("type") for s in sessions)
        if len(sessions) != 10:
            errs.append(f"Step 1 needs exactly 10 sessions (found {len(sessions)})")
        for t in TYPES:
            if c.get(t, 0) != 2:
                errs.append(f"Step 1 needs 2 sessions of type {t} (found {c.get(t, 0)})")
    return errs


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate a QanoonArena question file.")
    p.add_argument("file")
    p.add_argument("--contrib", action="store_true", help="also check the Step 1 rule (10 sessions, 2 per type)")
    a = p.parse_args(argv)
    sessions = read_jsonl(a.file)
    errs = validate(sessions, Corpus(), a.contrib)
    if errs:
        print(f"{len(errs)} problem(s) in {a.file}:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {len(sessions)} sessions in {a.file} are valid.")


if __name__ == "__main__":
    main()
