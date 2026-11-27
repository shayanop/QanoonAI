"""Regression gate for the automatic test (Step 4, item 3). Do not edit.

Runs one version on the 30 smoke conversations and compares the composite score with
the last accepted score in results/accepted_smoke.json. Exits with code 1 (fails CI)
when the score drops by more than regression_tolerance points.

    python -m harness.smoke --group G07 --memory M2 --adapter A0            # check
    python -m harness.smoke --group G07 --memory M2 --adapter A0 --accept   # accept new score
"""
from __future__ import annotations

import argparse
import json
import sys

from .common import ROOT, load_config
from .evaluate import main as evaluate_main


def main(argv=None):
    p = argparse.ArgumentParser(description="Smoke-set regression gate")
    p.add_argument("--group", required=True)
    p.add_argument("--memory", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--backend", default=None)
    p.add_argument("--accept", action="store_true", help="store this score as the new accepted score")
    a = p.parse_args(argv)
    cfg = load_config()
    seed = cfg["seeds"][0]
    cli = ["--group", a.group, "--memory", a.memory, "--adapter", a.adapter, "--split", "smoke",
           "--seed", str(seed), "--quiet"]
    if a.adapter_path:
        cli += ["--adapter-path", a.adapter_path]
    if a.backend:
        cli += ["--backend", a.backend]
    evaluate_main(cli)
    config_id = f"{a.memory.upper()}+{a.adapter.upper()}"
    import os
    backend = (a.backend or os.environ.get("QA_BACKEND") or "hf").lower()
    key = f"{config_id}|{backend}"   # real-model and mock scores are kept apart
    res_dir = ROOT / cfg["paths"]["results_dir"]
    summary = json.loads((res_dir / f"{a.group}_{config_id}_smoke_s{seed}.json").read_text())["summary"]
    score = summary["score"]
    acc_file = res_dir / "accepted_smoke.json"
    accepted = json.loads(acc_file.read_text()) if acc_file.exists() else {}
    prev = accepted.get(key)
    tol = cfg["scoring"]["regression_tolerance"]
    print(f"smoke score for {config_id} ({backend}): {score:.2f}" + (f" (accepted {prev:.2f})" if prev is not None else " (no accepted score yet)"))
    if a.accept or prev is None:
        accepted[key] = score
        acc_file.write_text(json.dumps(accepted, indent=2))
        print("accepted as the new reference score.")
        return
    if score < prev - tol:
        print(f"FAIL: score dropped by {prev - score:.2f} points (allowed {tol}).")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
