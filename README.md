# QanoonArena starter kit — AI-4016 Assignment 3

This repository is your starting point for Assignment 3. It contains a working legal
chatbot (the **baseline**, Version 1), the law documents, the practice questions, and
every tool you need to run, test, trace, train and report. Read the assignment first;
this file tells you how to use the code.

> **Not legal advice.** This chatbot is a student research prototype.

---

## 1. What is in the box

```
qanoonarena-starter/
├── agent/                    <- YOUR WORK GOES HERE
│   ├── graph.py              the LangGraph agent: memory_read -> retrieve -> generate -> cite_check -> memory_write
│   ├── cite_check.py         Step 4: your runtime citation checker
│   └── memory/
│       ├── m0_window.py      M0 baseline (do not change)
│       ├── m1_summary.py     Step 2: build this if you drew M1
│       ├── m2_facts.py       Step 2: build this if you drew M2
│       └── m3_temporal.py    Step 2: build this if you drew M3
├── harness/                  FIXED: do not change anything here
│   ├── config.yaml           the fixed rules (model, retrieval, budgets, seeds)
│   ├── FORMAT.md             every file format, with examples
│   ├── corpus/               the law documents (statutes and judgments)
│   ├── data/                 train, dev (practice), smoke (30 conversations), examples
│   ├── evaluate.py           run a version and record results
│   ├── validate_sessions.py  check your Step 1 questions
│   ├── build_sft_data.py     make fine-tuning data in the bot's exact prompt format
│   ├── leakage_check.py      check training data for test questions
│   ├── smoke.py              regression gate for the automatic test
│   ├── summarize.py          the 2 x 2 table with confidence intervals
│   ├── trace_export.py       check and bundle traces for submission
│   └── paper/section_template.tex   Step 6 report template
├── notebooks/finetune_qlora.ipynb   Step 3: QLoRA SFT and DPO on Colab
├── templates/                MEMORY.md, MODEL_CARD.md, DEVIATIONS.md, ERRORS.csv, review_template.csv
├── contrib/  review/         Step 1 outputs
├── scripts/                  run_all.sh, pre-push hook, chat.py
├── results/                  written by the evaluator; submit as is
└── .github/workflows/regression.yml   the automatic test on every push
```

---

## 2. Setup (15 minutes)

You need Python 3.10 or newer.

```bash
git clone <your group's copy of this repository>
cd qanoonarena-starter
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check that everything works, with no model and no GPU:

```bash
python -m harness.evaluate --group G00 --memory M0 --adapter A0 --split smoke --backend mock
```

You should see a score table. The mock backend only proves the pipeline runs; never
submit mock results.

### Running the real model

All submitted results must use the **hf** backend (the default), which loads the pinned
model `Qwen/Qwen3-4B-Instruct-2507` from `harness/config.yaml`.

| Where | How |
| --- | --- |
| **Google Colab (free T4 GPU)** | Open `notebooks/finetune_qlora.ipynb`; its first cells clone your repo and install everything. You can run evaluations from the same notebook. |
| Your own NVIDIA GPU (8 GB+) | `pip install -r requirements.txt` already includes everything. The model loads in 4-bit automatically. |
| CPU only | Works but is very slow (minutes per answer). Use it only for a few turns with `--limit 2`. |

For fast development you may serve the **same** model with Ollama, vLLM or LM Studio and
use `--backend openai` (set `QA_OPENAI_BASE_URL` and `QA_OPENAI_MODEL`). Final results
must come from `--backend hf`.

### Talk to the bot

```bash
python scripts/chat.py --memory M0              # type your messages; 'reset' starts a new conversation
```

---

## 3. Commands for each step

### Step 1: write test questions

1. Read `harness/FORMAT.md` and look at `harness/data/example_sessions.jsonl`.
2. Write `contrib/<group>.jsonl` (10 sessions, 2 per type).
3. Check it: `python -m harness.validate_sessions contrib/G07.jsonl --contrib`
4. Review the file the instructor assigns you, using `templates/review_template.csv`, and save it as `review/G07.csv`.

### Step 2: build your memory

Implement the file for your memory type in `agent/memory/`. Do not change the method
signatures. Try it:

```bash
python -m harness.evaluate --group G07 --memory M2 --adapter A0 --split dev --limit 10
```

Fill in `templates/MEMORY.md` and save it as `MEMORY.md` at the repository root.

### Step 3: train your model

1. Build training data in the bot's exact prompt format:
   `python -m harness.build_sft_data --out data_sft.jsonl`
2. Check for leakage: `python -m harness.leakage_check data_sft.jsonl`
3. Train with `notebooks/finetune_qlora.ipynb` (SFT for A1; SFT then DPO for A2).
4. Upload the adapter to Hugging Face or Google Drive and fill in `MODEL_CARD.md`.
5. Evaluate it: `python -m harness.evaluate --group G07 --memory M0 --adapter A1 --adapter-path ./adapters/a1 --split dev`

### Step 4: monitor your chatbot

- **Traces** are recorded automatically for every run in `traces/`. To also see them in
  a dashboard, run Langfuse or Arize Phoenix and point OpenTelemetry at it:

  ```bash
  pip install opentelemetry-sdk opentelemetry-exporter-otlp
  # Arize Phoenix (local):  pip install arize-phoenix && phoenix serve
  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006
  # Langfuse (self-hosted or cloud): see Langfuse's OpenTelemetry docs for the endpoint and auth header
  export OTEL_EXPORTER_OTLP_ENDPOINT=https://<your-langfuse-host>/api/public/otel
  export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64 of public_key:secret_key>"
  ```

- **Citation checker:** implement your policy in `agent/cite_check.py`.
- **Automatic test:** the GitHub workflow in `.github/workflows/regression.yml` runs the
  smoke set on every push (pipeline check with the mock backend). Before pushing, run the
  real check locally with the hook: `cp scripts/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push`.
  When a change is good, accept its score and commit `results/accepted_smoke.json`:
  `python -m harness.smoke --group G07 --memory M2 --adapter A0 --accept` (real model) and
  `python -m harness.smoke --group G07 --memory M2 --adapter A0 --backend mock --accept` (the score GitHub checks).
  In your repository settings set the Actions variables `QA_GROUP` and `QA_MEMORY`.
- **Dashboard:** compare your four versions on score, tokens, latency and ungrounded
  citations (the `.json` files in `results/` and `python -m harness.summarize` give every number).
- **Mistakes:** label 60 failed turns in `ERRORS.csv` (template in `templates/`).

### Step 5: run your experiments

```bash
bash scripts/run_all.sh G07 M2 A1 ./adapters/a1      # 4 versions x 3 seeds on dev
python -m harness.summarize --group G07 --split dev   # table, effects and confidence intervals
python -m harness.summarize --group G07 --split dev --latex > report/table_2x2.tex
```

### Competition window (secret test)

When the instructor releases `secret_test_questions.jsonl`, run your versions on it:

```bash
python -m harness.evaluate --group G07 --memory M2 --adapter A1 --adapter-path ./adapters/a1 \
       --split test --questions secret_test_questions.jsonl --all-seeds
python -m harness.trace_export --group G07 --split test
```

Submit the `results/*_test_*` files and `submission/G07_traces_test.jsonl.gz`.

### Step 6: write your paper section

Copy `harness/paper/section_template.tex` to `report/`, fill it in, and compile with
`pdflatex`. Put the numbers from `harness.summarize --latex` into its tables.

---

## 4. Rules the code checks

- `harness/` must be unchanged. The run manifest records the config, prompt and corpus
  hashes, and the instructor re-runs 10% of your answers.
- Only the three fixed seeds in `config.yaml` count.
- Any unavoidable change goes in `DEVIATIONS.md` (template in `templates/`).

## 5. Final submission checklist

- [ ] `contrib/<group>.jsonl` passes the checker, and `review/<group>.csv` is complete
- [ ] `agent/` contains your memory and citation checker
- [ ] `MEMORY.md`, `MODEL_CARD.md`, `ERRORS.csv`, and `DEVIATIONS.md` (if needed)
- [ ] adapter link in `MODEL_CARD.md`
- [ ] `results/`: 4 versions x 3 seeds on dev and on the secret test, unchanged
- [ ] `submission/<group>_traces*.jsonl.gz` passes `harness.trace_export`
- [ ] dashboard screenshot and report PDF with its LaTeX source in `report/`
