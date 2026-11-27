#!/usr/bin/env bash
# Run your four versions with the three fixed seeds on the practice (dev) set.
#   bash scripts/run_all.sh <GROUP> <MEMORY> <ADAPTER> <ADAPTER_PATH> [split] [backend]
#   bash scripts/run_all.sh G07 M2 A1 ./adapters/a1
set -euo pipefail
GROUP=${1:?group id, e.g. G07}
MEM=${2:?your memory type: M1, M2 or M3}
AD=${3:?your training type: A1 or A2}
AD_PATH=${4:?path or Hugging Face id of your adapter}
SPLIT=${5:-dev}
BACKEND=${6:-hf}

run () { python -m harness.evaluate --group "$GROUP" --memory "$1" --adapter "$2" --split "$SPLIT" --all-seeds --backend "$BACKEND" "${@:3}"; }

run M0   A0                                  # Version 1: baseline
run "$MEM" A0                                # Version 2: memory only
run M0   "$AD" --adapter-path "$AD_PATH"     # Version 3: training only
run "$MEM" "$AD" --adapter-path "$AD_PATH"   # Version 4: memory and training

python -m harness.summarize --group "$GROUP" --split "$SPLIT"
