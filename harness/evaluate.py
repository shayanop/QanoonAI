"""Run one version of your chatbot on a question set and record every turn. Do not edit.

Examples (from the repository root):
    python -m harness.evaluate --group G07 --memory M0 --adapter A0 --split dev --seed 13
    python -m harness.evaluate --group G07 --memory M2 --adapter A1 --adapter-path ./adapters/a1 --split dev --all-seeds
    python -m harness.evaluate --group G07 --memory M0 --adapter A0 --split smoke --backend mock

Outputs (never edit them by hand):
    results/<group>_<config>_<split>_s<seed>.csv          one row per scored turn
    results/<group>_<config>_<split>_s<seed>.pred.jsonl   every answer, for re-runs and the Arena
    results/<group>_<config>_<split>_s<seed>.json         run manifest and summary scores
    traces/<run_id>.jsonl                                 full trace of every turn
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

from .answer import SYSTEM_PROMPT
from .common import ROOT, Corpus, config_hash, file_hash, load_config, read_jsonl, sha256_text
from .metrics import aggregate, score_turn
from .retriever import BM25Retriever
from .tracing import Tracer

RESULT_COLUMNS = [
    "group_id", "config", "memory", "adapter", "seed", "split", "session_id", "turn", "family",
    "correct", "cite_precision", "cite_recall", "abstain_expected", "abstained", "memory_recall",
    "ungrounded_cites", "parse_ok", "tokens_in", "tokens_out", "latency_ms", "mem_tokens", "failure_code",
]

SPLIT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl", "smoke": "smoke.jsonl", "example": "example_sessions.jsonl"}


def load_sessions(split: str, questions: str | None, cfg: dict) -> list[dict]:
    if questions:
        return read_jsonl(questions)
    if split not in SPLIT_FILES:
        raise SystemExit(f"For split '{split}' pass the question file with --questions.")
    return read_jsonl(ROOT / cfg["paths"]["data_dir"] / SPLIT_FILES[split])


def baseline_tokens(results_dir: Path, group: str, split: str, seed: int) -> float | None:
    fp = results_dir / f"{group}_M0+A0_{split}_s{seed}.json"
    if fp.exists():
        return json.loads(fp.read_text())["summary"].get("tokens_per_turn")
    return None


def run(args) -> dict:
    cfg = load_config()
    from agent import build_agent  # imported here so harness tools work without the agent installed

    config_id = f"{args.memory.upper()}+{args.adapter.upper()}"
    if args.adapter.upper() != "A0" and not args.adapter_path and args.backend != "mock":
        raise SystemExit("A1/A2 runs need --adapter-path (your trained adapter folder or Hugging Face id).")
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.group}_{config_id}_{args.split}_s{args.seed}"
    run_id = f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"

    sessions = load_sessions(args.split, args.questions, cfg)
    if args.limit:
        sessions = sessions[: args.limit]
    corpus = Corpus()
    retriever = BM25Retriever(corpus, cfg)
    tracer = Tracer(run_id, str(ROOT / cfg["paths"]["traces_dir"]))
    agent = build_agent(args.memory, args.backend, args.adapter_path, args.seed, tracer, cfg)
    agent.retriever = retriever

    adapter_hash = file_hash(args.adapter_path) if args.adapter_path and Path(args.adapter_path).exists() else (args.adapter_path or "none")
    trace_attrs = dict(run_id=run_id, group_id=args.group, config=config_id, memory=args.memory.upper(),
                       adapter=args.adapter.upper(), seed=args.seed, split=args.split,
                       config_hash=config_hash(cfg), adapter_hash=adapter_hash)

    rows, preds = [], []
    t_start = time.time()
    for si, sess in enumerate(sessions, 1):
        sid = sess["session_id"]
        agent.reset(sid)
        for ti, turn in enumerate(sess["turns"], 1):
            ans, info = agent.respond(sid, ti, turn["user"], **trace_attrs)
            preds.append({"session_id": sid, "turn": ti, "answer": ans.to_dict(),
                          "retrieved": info["retrieved"], "allowed_ids": sorted(info["allowed_ids"]),
                          "tokens_in": info["tokens_in"], "tokens_out": info["tokens_out"],
                          "latency_ms": info["latency_ms"], "mem_tokens": info["mem_tokens"]})
            gold = turn.get("gold")
            if not gold:
                continue
            sc = score_turn(ans, gold, corpus, info["allowed_ids"])
            rows.append({
                "group_id": args.group, "config": config_id, "memory": args.memory.upper(),
                "adapter": args.adapter.upper(), "seed": args.seed, "split": args.split,
                "session_id": sid, "turn": ti, "family": sess.get("type", ""), **sc,
                "tokens_in": info["tokens_in"], "tokens_out": info["tokens_out"],
                "latency_ms": info["latency_ms"], "mem_tokens": info["mem_tokens"], "failure_code": "",
            })
        if not args.quiet:
            print(f"\r[{config_id} seed {args.seed}] session {si}/{len(sessions)}", end="", file=sys.stderr)
    if not args.quiet:
        print(file=sys.stderr)
    tracer.close()

    with open(results_dir / f"{stem}.pred.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary = {}
    if rows:
        with open(results_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in RESULT_COLUMNS} for r in rows)
        base = None if config_id == "M0+A0" else baseline_tokens(results_dir, args.group, args.split, args.seed)
        summary = aggregate(rows, cfg["scoring"]["weights"], base)
    manifest = {
        "run_id": run_id, **{k: v for k, v in trace_attrs.items() if k != "run_id"},
        "backend": agent.backend.name, "prompt_hash": sha256_text(SYSTEM_PROMPT)[:12],
        "corpus_hash": file_hash(ROOT / cfg["paths"]["corpus_dir"]),
        "n_sessions": len(sessions), "wall_clock_s": round(time.time() - t_start, 1),
        "python": platform.python_version(), "summary": summary,
        "warning": "MOCK BACKEND - not valid for submission" if agent.backend.name == "mock" else "",
    }
    (results_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2))
    if summary and not args.quiet:
        print(json.dumps(summary, indent=2))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True, help="your group id, e.g. G07")
    p.add_argument("--memory", required=True, choices=["M0", "M1", "M2", "M3", "m0", "m1", "m2", "m3"])
    p.add_argument("--adapter", required=True, choices=["A0", "A1", "A2", "a0", "a1", "a2"])
    p.add_argument("--adapter-path", default=None, help="LoRA adapter folder or Hugging Face id (A1/A2)")
    p.add_argument("--split", default="dev", help="dev, smoke, example, or test (with --questions)")
    p.add_argument("--questions", default=None, help="a JSONL question file (used for the secret test)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true", help="run the 3 fixed seeds one after another")
    p.add_argument("--backend", default=None, help="hf (default), openai, or mock")
    p.add_argument("--limit", type=int, default=0, help="only the first N sessions (quick checks)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    seeds = load_config()["seeds"]
    if args.all_seeds:
        todo = seeds
    else:
        todo = [args.seed if args.seed is not None else seeds[0]]
    bad = [s for s in todo if s not in seeds]
    if bad:
        print(f"WARNING: seed {bad} is not one of the fixed seeds {seeds}; log it in DEVIATIONS.md.", file=sys.stderr)
    for s in todo:
        args.seed = s
        run(args)


if __name__ == "__main__":
    main()
