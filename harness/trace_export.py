"""Check and bundle your traces for submission (Step 4, item 1). Do not edit.

    python -m harness.trace_export --group G07              # all runs of your group
    python -m harness.trace_export --group G07 --split test # only the Arena runs

Checks that every turn has the five required spans (memory_read, retrieve, generate,
cite_check, memory_write) and the required attributes, then writes
submission/<group>_traces.jsonl.gz in the shared QanoonArena trace format.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from .common import ROOT, load_config
from .tracing import REQUIRED_SPANS, REQUIRED_TURN_ATTRS


def check_file(path: Path) -> tuple[list[dict], list[str]]:
    spans = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_trace = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    errs = []
    for tid, group in by_trace.items():
        names = {s["name"] for s in group}
        root = [s for s in group if s["name"] == "turn"]
        if not root:
            errs.append(f"{path.name}: trace {tid[:8]} has no 'turn' span")
            continue
        missing = [n for n in REQUIRED_SPANS if n not in names]
        if missing:
            errs.append(f"{path.name}: trace {tid[:8]} is missing spans {missing}")
        attrs = root[0]["attributes"]
        miss_a = [a for a in REQUIRED_TURN_ATTRS if a not in attrs]
        if miss_a:
            errs.append(f"{path.name}: trace {tid[:8]} turn span is missing {miss_a}")
    return spans, errs


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate and bundle traces")
    p.add_argument("--group", required=True)
    p.add_argument("--split", default=None)
    a = p.parse_args(argv)
    cfg = load_config()
    tdir = ROOT / cfg["paths"]["traces_dir"]
    files = sorted(tdir.glob(f"{a.group}_*.jsonl"))
    if a.split:
        files = [f for f in files if f"_{a.split}_" in f.name]
    if not files:
        sys.exit(f"No trace files for group {a.group} in {tdir}")
    out_dir = ROOT / "submission"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{a.group}_traces{'_' + a.split if a.split else ''}.jsonl.gz"
    all_errs, n_spans, n_turns = [], 0, 0
    with gzip.open(out, "wt", encoding="utf-8") as g:
        for f in files:
            spans, errs = check_file(f)
            all_errs += errs
            for s in spans:
                n_spans += 1
                n_turns += s["name"] == "turn"
                g.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{len(files)} runs, {n_turns} turns, {n_spans} spans -> {out}")
    if all_errs:
        print(f"{len(all_errs)} problem(s):")
        for e in all_errs[:30]:
            print("  -", e)
        sys.exit(1)
    print("OK: traces are complete.")


if __name__ == "__main__":
    main()
