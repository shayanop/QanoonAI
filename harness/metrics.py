"""Per-turn scoring and the QanoonArena composite score. Do not edit.

The same code scores the practice set on your machine and the secret test at the end.
For the secret test the instructor also runs a fixed LLM judge for "correct"; on the
practice set "correct" uses the keyword rule below.
"""
from __future__ import annotations

import math
from statistics import mean

from .answer import Answer
from .common import Corpus, norm_text, same_citation


def _nan() -> float:
    return float("nan")


def keywords_hit(answer_text: str, keywords: list[str]) -> bool:
    """Every keyword must appear. A keyword may list alternatives separated by '|'."""
    text = norm_text(answer_text)
    for kw in keywords:
        alts = [norm_text(a) for a in kw.split("|") if a.strip()]
        if not any(a in text for a in alts):
            return False
    return True


def score_turn(pred: Answer, gold: dict, corpus: Corpus, allowed_ids: set[str]) -> dict:
    """Score one answered turn against its gold record.

    allowed_ids: doc ids the bot actually saw this turn (retrieved + memory). Citing
    anything else counts as ungrounded.
    """
    exact = bool(gold.get("exact_version"))
    gold_cites = gold.get("citations") or []
    expect_abstain = bool(gold.get("abstain"))

    # correctness
    if expect_abstain:
        correct = int(pred.abstain)
    elif pred.abstain:
        correct = 0
    elif gold.get("answer_keywords"):
        correct = int(keywords_hit(pred.answer, gold["answer_keywords"]))
    else:
        correct = int(bool(gold_cites) and all(any(same_citation(p, g, exact) for p in pred.citations) for g in gold_cites))

    # citations (not scored on turns where the bot should abstain with no gold citations)
    if expect_abstain and not gold_cites:
        cp = cr = _nan()
    else:
        matched_pred = sum(1 for p in pred.citations if any(same_citation(p, g, exact) for g in gold_cites))
        matched_gold = sum(1 for g in gold_cites if any(same_citation(p, g, exact) for p in pred.citations))
        cp = matched_pred / len(pred.citations) if pred.citations else 0.0
        cr = matched_gold / len(gold_cites) if gold_cites else (1.0 if not pred.citations else 0.0)

    allowed_base = {a.split("@")[0] for a in allowed_ids}
    ungrounded = sum(
        1 for p in pred.citations
        if not corpus.exists(p) or (p not in allowed_ids and p.split("@")[0] not in allowed_base)
    )

    return {
        "correct": correct,
        "cite_precision": round(cp, 4) if not math.isnan(cp) else "",
        "cite_recall": round(cr, 4) if not math.isnan(cr) else "",
        "abstain_expected": int(expect_abstain),
        "abstained": int(pred.abstain),
        "memory_recall": correct if gold.get("memory_dependent") else "",
        "ungrounded_cites": ungrounded,
        "parse_ok": int(pred.parse_ok),
    }


def _num(v):
    if v == "" or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def aggregate(rows: list[dict], weights: dict, baseline_tokens_per_turn: float | None = None) -> dict:
    """Composite score (0-100) for one run, from its results rows."""
    if not rows:
        return {k: 0.0 for k in ("score", "accuracy", "citation_f1", "abstain_f1", "memory_recall", "efficiency",
                                 "tokens_per_turn", "latency_p50_ms", "latency_p95_ms", "ungrounded_rate")} | {"n_turns": 0}
    acc = mean(int(r["correct"]) for r in rows)

    cite_f1s = []
    for r in rows:
        p, rc = _num(r["cite_precision"]), _num(r["cite_recall"])
        if p is not None and rc is not None:
            cite_f1s.append(f1(p, rc))
    cite_f1 = mean(cite_f1s) if cite_f1s else 0.0

    tp = sum(1 for r in rows if int(r["abstain_expected"]) and int(r["abstained"]))
    fp = sum(1 for r in rows if not int(r["abstain_expected"]) and int(r["abstained"]))
    fn = sum(1 for r in rows if int(r["abstain_expected"]) and not int(r["abstained"]))
    abst_f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

    mem = [_num(r["memory_recall"]) for r in rows if _num(r["memory_recall"]) is not None]
    mem_recall = mean(mem) if mem else 0.0

    tokens = mean(int(r["tokens_in"]) + int(r["tokens_out"]) for r in rows)
    eff = 1.0 if not baseline_tokens_per_turn else min(1.0, baseline_tokens_per_turn / max(tokens, 1))

    w = weights
    score = 100 * (
        w["accuracy"] * acc + w["citation_f1"] * cite_f1 + w["abstain_f1"] * abst_f1
        + w["memory_recall"] * mem_recall + w["efficiency"] * eff
    )
    lat = sorted(int(r["latency_ms"]) for r in rows)
    return {
        "score": round(score, 2),
        "accuracy": round(acc, 4),
        "citation_f1": round(cite_f1, 4),
        "abstain_f1": round(abst_f1, 4),
        "memory_recall": round(mem_recall, 4),
        "efficiency": round(eff, 4),
        "tokens_per_turn": round(tokens, 1),
        "latency_p50_ms": lat[len(lat) // 2],
        "latency_p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "ungrounded_rate": round(mean(1 if int(r["ungrounded_cites"]) else 0 for r in rows), 4),
        "n_turns": len(rows),
    }
