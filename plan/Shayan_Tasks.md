# Member A: Memory (M1 rolling summary)

Name: Shayan · Branch: `a-memory` · Marks you mainly carry: Step 2 (13), the Step 5 memory experiment (6), plus your share of Steps 4 and 6

This file is all you need. You never wait for B or C: none of your runs needs the trained adapter.

---

## Rules (same for all three of us)

1. **Only edit and commit the files listed under "Your files".** That's what lets the three branches merge with zero conflicts.
2. **Never edit `harness/`.** CI fails if you do.
3. **`agent/` is frozen.** Your only allowed change is the ablation knob in task A1, which is off by default. The instructor re-runs 10% of our answers from our code, so the code must not change after the real runs start.
4. **Final runs use the real model** (`--backend hf`, which is the default). For quick checks with `--backend mock`, always use `--group G10MOCK` so you don't overwrite real result files, and never commit those.
5. **Commit only your own result patterns:** `git add "results/G10_M1+A0_*" "results/G10ABL*"`. Never `git add results/` or `git add .`.
6. **Colab deletes everything when the runtime dies.** After every run, copy `results/` and `traces/` to Drive `G10/A/`. C needs your traces on the final day.
7. **Nice to have:** before running on a split, copy C's `results/G10_M0+A0_<split>_s*.json` from Drive `G10/C/` into your local `results/`. If they aren't there yet, **run anyway** (summarize fixes the efficiency number later).

## Your files

`agent/memory/m1_summary.py` (knob only) · `tests/test_m1.py` · `MEMORY.md` · `results/G10_M1+A0_*` · `results/G10ABL*` · `errors/ERRORS_A.csv` · `report/parts/memory.tex` · `report/parts/ablation.tex` · `report/parts/trace_failure.tex` · `report/parts/contrib_A.tex` · `notebooks/run_A.ipynb` · (optional) branch `open-arena`

## Colab / Kaggle setup (first cell of `notebooks/run_A.ipynb`)

```python
!git clone -b a-memory https://github.com/<owner>/<repo>.git && %cd <repo>
!pip install -q -r requirements.txt opentelemetry-sdk opentelemetry-exporter-otlp
import os, base64
from google.colab import userdata, drive          # Langfuse keys go in Colab "Secrets", never in the notebook
drive.mount("/content/drive")
pk, sk = userdata.get("LANGFUSE_PK"), userdata.get("LANGFUSE_SK")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"   # US: us.cloud.langfuse.com
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Basic%20" + base64.b64encode(f"{pk}:{sk}".encode()).decode()
```
After every run: `!mkdir -p /content/drive/MyDrive/G10/A && cp -r results traces /content/drive/MyDrive/G10/A/`
If you run out of Colab GPU, use Kaggle (free T4, 30 h/week) with the same commands.

---

## Tasks

### A1. Ablation knob (first hour, then never touch `agent/` again)
At the top of `SummaryMemory.read()` in `agent/memory/m1_summary.py` (and add `import os` at the top of the file):
```python
cap = int(os.environ.get("QA_M1_BUDGET", "0"))   # ablation only; 0 = use the config budget (default)
if cap:
    budget_tokens = min(budget_tokens, cap)
```
Run `python tests/test_m1.py`, then commit. Behaviour without the variable is identical.

### A2. Real M1+A0 runs on the practice set (3 seeds)
```bash
python -m harness.evaluate --group G10 --memory M1 --adapter A0 --split dev --all-seeds
```
This takes about 1.5× as long as a baseline run because of the extra summary call (roughly 45–75 min per seed on a T4). Copy to Drive, then commit `results/G10_M1+A0_dev_*`.

### A3. Memory experiment (Step 5, question 2: 6 marks)
Same version, smaller memory budgets. The config budget is 768 and we can't go above it without a deviation.
```bash
QA_M1_BUDGET=256 python -m harness.evaluate --group G10ABL256 --memory M1 --adapter A0 --split dev --all-seeds
QA_M1_BUDGET=128 python -m harness.evaluate --group G10ABL128 --memory M1 --adapter A0 --split dev --all-seeds
python -m harness.summarize --group G10ABL256 --split dev
python -m harness.summarize --group G10ABL128 --split dev
```
- PowerShell: `$env:QA_M1_BUDGET=256` before the command. Colab: `os.environ["QA_M1_BUDGET"]="256"`.
- Our memory is usually only 60–120 tokens, so if 256 shows no change, 128 (or 64) is the interesting point.
- Compare score, memory recall and tokens per turn against the default 768 (your A2 runs).
- If GPU time is short, use seed 13 only (`--seed 13`) and say so in the report.
- Tell C that you used `QA_M1_BUDGET` 256 and 128 under separate group ids, so C can log it in `DEVIATIONS.md`.
- Optional, if the adapter is ready in time: repeat the best setting with `--adapter A2 --adapter-path <B's link>`.

### A4. `MEMORY.md` (Step 2: 3 marks for the rules, 4 for corrections)
- Replace examples 1 and 3 (currently from mock runs) with **real-model** transcripts from A2. Read the `memory_read` and `memory_write` spans in `traces/G10_M1+A0_dev_s13_*.jsonl` or in Langfuse.
- Keep example 2 (the correction). If you can, also show a real correction: run `python scripts/chat.py --memory M1`, type "My phone was stolen on 3 March 2025", then "Correction: it was 5 May 2024", and paste the trace.
- Remove the "must be replaced with hf traces" warning. Keep it to one page.

### A5. Secret test (released Sat 28 Nov, open 48 h)
```bash
python -m harness.evaluate --group G10 --memory M1 --adapter A0 --split test --questions secret_test_questions.jsonl --all-seeds
```
Copy to Drive, then commit `results/G10_M1+A0_test_*`. Never commit `secret_test_questions.jsonl`.

### A6. Label 20 mistakes → `errors/ERRORS_A.csv`
Header: `config,seed,session_id,turn,code,secondary_code,note`. Take them from your M1+A0 practice runs. Focus on **E3** (forgot an earlier fact) and **E4** (used a wrong fact from memory), but use whatever code fits. List the failed turns with:
```bash
python -c "import csv,sys; [print(r['config'],r['seed'],r['session_id'],r['turn'],r['family'],'ungrounded='+r['ungrounded_cites'],'abst_exp='+r['abstain_expected'],'abst='+r['abstained'],'parse='+r['parse_ok']) for r in csv.DictReader(open(sys.argv[1])) if r['correct']=='0' or r['ungrounded_cites'] not in ('','0')]" results/G10_M1+A0_dev_s13.csv
```
Open each one in Langfuse (or `traces/`) to see what memory held, what was retrieved and what the model said. Give one code, plus an optional second code and a one-line note.

| Code | Mistake |
| --- | --- |
| E1 | Did not find the right document |
| E2 | Wrong amendment, or an old version of the law |
| E3 | Forgot a fact from an earlier message |
| E4 | Used a wrong or outdated fact from memory |
| E5 | Cited a section or case that doesn't support the answer, or doesn't exist |
| E6 | Answered when it should have said "I am not sure" |
| E7 | Said "I am not sure" when it could have answered |
| E8 | Answer in the wrong format |

### A7. Report parts (LaTeX; C's `report/section.tex` pulls them in)
- `report/parts/memory.tex`: Method → Memory. What is stored, the write rules, the read rules and how corrections work, precise enough for another group to rebuild it (condense `MEMORY.md`).
- `report/parts/ablation.tex`: the ablation table (budget 128 / 256 / 768: score, memory recall, tokens per turn) plus 2–3 sentences on what happened.
- `report/parts/trace_failure.tex`: one annotated **failure** trace, turn by turn, quoting the memory read/write and the citations from the trace.
- `report/parts/contrib_A.tex`: one line, `\item <Name>: ...`, saying what you built, ran and wrote.

### A8. Optional bonus: open leaderboard (+5/+3/+2)
Only once everything above is done. On branch `open-arena`, which is **never merged into main**, try any trick with the same model and documents, and run it as `--group G10OPEN`. Check the instructor's instructions for how to submit it.

---

## Your timeline

| When | What |
| --- | --- |
| Thu 26 Nov, evening | Day-0 call · A1 knob · start A2 |
| Fri 27 Nov | finish A2 · A3 ablation · A4 MEMORY.md |
| Sat 28 Nov | A5 secret test (start as soon as it's released) |
| Sun 29 Nov | A6 errors · A7 report parts · **open a PR from `a-memory` into `main` by 22:00** |
| Mon 30 Nov | proof-read the whole report · viva walkthrough in the evening |

## Your checklist
**Status 26 Nov (local prep in this repo, not committed yet):** knob is in `m1_summary.py` and `tests/test_m1.py` passes (10/10, incl. `test_ablation_budget_knob`); a mock run with `QA_M1_BUDGET` works through the harness. Ready: `notebooks/run_A.ipynb` (A2/A3/A5/A6 cells), `report/parts/memory.tex` and `contrib_A.tex` (final), `ablation.tex` and `trace_failure.tex` (layout with `\todo{}` for the real numbers), `errors/ERRORS_A.csv` (header). Left: everything that needs the T4 or the secret test.

- [ ] Knob committed, `tests/test_m1.py` passes (passes; commit after C's first commit)
- [ ] `results/G10_M1+A0_dev_s{13,42,2026}.*` and `results/G10_M1+A0_test_s{13,42,2026}.*` committed (real model, no `MOCK BACKEND` warning in the `.json`)
- [ ] `results/G10ABL256_*` and `results/G10ABL128_*` committed
- [ ] All your traces copied to Drive `G10/A/`
- [ ] `MEMORY.md`: 3 real examples, one of them a correction, one page
- [ ] `errors/ERRORS_A.csv`: 20 rows
- [ ] 4 report part files (memory + contrib done; ablation + trace_failure need hf numbers)
- [ ] Told C what to put in `DEVIATIONS.md`

## Viva prep (you're asked about everything, not just memory)
Be ready to explain every line of `m1_summary.py`: when a message is stored word for word (anchors), the drift guard on rewrites, how corrections replace old values, why the retrieval hint helps BM25, what's dropped when over budget, and what the ablation showed. Also read `TASKS_B_training.md` and `TASKS_C_monitoring.md` so you can answer basic questions on DPO, the cite checker and the score formula.
