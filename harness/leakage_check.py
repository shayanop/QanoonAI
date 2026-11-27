"""Check that your training data contains no evaluation questions (Step 3). Do not edit.

    python -m harness.leakage_check my_sft_data.jsonl [my_dpo_pairs.jsonl ...]

Accepted training files are JSONL with any of these text fields per line:
"messages" (chat list), "prompt", "question", "user", or sessions with "turns".
Every user message is compared (after normalising case and spaces) with:
  - the practice (dev) and smoke questions,
  - the protected hashes of the secret test and of all contributed questions
    (harness/data/protected_hashes.txt; the questions themselves are not shipped).
Any overlap is a leak and must be removed before training.
"""
from __future__ import annotations

import argparse
import sys

from .common import ROOT, load_config, question_hash, read_jsonl


def user_texts(row: dict) -> list[str]:
    out = []
    if isinstance(row.get("messages"), list):
        out += [m.get("content", "") for m in row["messages"] if m.get("role") == "user"]
    for k in ("prompt", "question", "user"):
        v = row.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):  # prompt in chat format (DPO)
            out += [m.get("content", "") for m in v if isinstance(m, dict) and m.get("role") == "user"]
    for t in row.get("turns", []) or []:
        out.append(t.get("user", ""))
    return [t for t in out if t and t.strip()]


def protected_hashes() -> dict[str, str]:
    cfg = load_config()
    data = ROOT / cfg["paths"]["data_dir"]
    # Lines that also occur in the official training split are shared templates, not leaks.
    allowed = set()
    if (data / "train.jsonl").exists():
        allowed = {question_hash(t["user"]) for s in read_jsonl(data / "train.jsonl") for t in s["turns"]}
    hashes: dict[str, str] = {}
    for name in ("dev.jsonl", "smoke.jsonl"):
        fp = data / name
        if fp.exists():
            for s in read_jsonl(fp):
                for t in s["turns"]:
                    hashes[question_hash(t["user"])] = name
    fp = data / "protected_hashes.txt"
    if fp.exists():
        for line in fp.read_text().split():
            hashes[line.strip()] = "secret test / contributed questions"
    return {h: src for h, src in hashes.items() if h not in allowed}


def candidates(text: str) -> set[str]:
    """The whole message, its QUESTION part, and earlier user turns carried in MEMORY."""
    out = {text, text.split("QUESTION:")[-1].strip()}
    for line in text.splitlines():
        if line.startswith("User: "):
            out.add(line[6:].strip())
    return out


def check(files: list[str]) -> list[str]:
    prot = protected_hashes()
    leaks = []
    for f in files:
        for n, row in enumerate(read_jsonl(f), 1):
            for text in user_texts(row):
                for cand in candidates(text):
                    src = prot.get(question_hash(cand))
                    if src:
                        leaks.append(f"{f} line {n}: matches a question in {src}: {cand[:80]!r}")
                        break
    return leaks


def main(argv=None):
    p = argparse.ArgumentParser(description="Check training files for evaluation leakage.")
    p.add_argument("files", nargs="+")
    a = p.parse_args(argv)
    leaks = check(a.files)
    if leaks:
        print(f"LEAKAGE FOUND ({len(leaks)}):")
        for l in leaks[:50]:
            print("  -", l)
        sys.exit(1)
    print("OK: no overlap with evaluation questions.")


if __name__ == "__main__":
    main()
