# G10 Team Plan: QanoonArena (M1 summary memory + A2 SFT→DPO)

Fill in names: **A (Memory)** = ______ **B (Training)** = ______ **C (Monitoring + report)** = ______

**Each member's own task list, which stands on its own:** [A: Memory](plan/TASKS_A_memory.md) · [B: Training](plan/TASKS_B_training.md) · [C: Monitoring + report](plan/TASKS_C_monitoring.md). This file is the overview for the whole team.

Deadlines: secret test released **Sat 28 Nov** (open 48 h) · final submission **Mon 30 Nov 23:59** · viva after that (each of us is questioned alone, on *all* of it).

---

## 0. Where we stand (checked 26 Nov)

| Step | Marks | Status |
| --- | --- | --- |
| 1 Test questions | 10 | `contrib/G10.jsonl` passes the checker. Review of G04 is written but saved in the wrong place (`G10_review.csv` at the root, should be `review/G10.csv`) → moved on Day 0 |
| 2 Memory (M1) | 13 | `agent/memory/m1_summary.py` is built, and `tests/test_m1.py` passes. `MEMORY.md` is drafted, but examples 1 and 3 come from **mock** runs and must be replaced with **hf** runs |
| 3 Training (A2) | 12 | Notebook is ready, and `MODEL_CARD.md` is drafted with DPO pair stats. **Training has not run.** No adapter yet |
| 4 Monitoring | 20 | Citation checker is built (`agent/cite_check.py`, drop policy). Not done: Langfuse, CI vars, accepted smoke score, dashboard, ERRORS.csv |
| 5 Experiments | 20 | Nothing has run on the real model. `results/` is empty |
| Competition | 10 (+5) | Waits for the secret test on 28 Nov |
| 6 Paper | 15 | Not started |

Not a git repo yet. No GitHub. No Langfuse.

---

## 1. Golden rules (read these before touching anything)

1. **Each file has exactly one owner** (table in §3). You only edit and commit files you own. That way the three branches merge with **zero conflicts**, in any order.
2. **Never edit `harness/`.** CI fails if it differs from the first commit.
3. **`agent/` is frozen as of the Day-0 commit.** The instructor re-runs 10% of our answers from our code, so the code on `main` must be the code that produced the results. The only change allowed is A's ablation knob (§4), and it is **off by default**, so behaviour doesn't change. If someone finds a real bug in `agent/`, stop and decide together, because every affected run has to be redone.
4. **Result files never collide** because each config has its own file name (`results/G10_<config>_<split>_s<seed>.*`), and each config has exactly one owner:
   - C: `M0+A0` · A: `M1+A0` and `G10ABL*` (ablation) · B: `M0+A2` and `M1+A2`
5. **Mock or test runs use `--group G10MOCK`**, never `G10`. Otherwise they overwrite real files. Never commit them.
6. **Commit only your own patterns**, for example `git add "results/G10_M1+A0_*"`. Never `git add results/` or `git add .`.
7. **Colab wipes everything when the runtime dies.** After *every* run, copy `results/` and `traces/` to the shared Drive folder `G10/<A|B|C>/`. `traces/` is gitignored, so Drive is the only place the traces live until C bundles them.
8. **Final results use `--backend hf` only.** `mock` is just a pipeline check. `openai` (Ollama/vLLM) is OK for quick experiments only.
9. **Repo stays PRIVATE.** The law documents are course-only, and we must not share answers.
10. **Baseline first (nice to have, not blocking):** before running a non-baseline config on a split, try to drop C's `results/G10_M0+A0_<split>_s*.json` into your local `results/` (from Drive). The evaluator uses it for the *efficiency* number in the run manifest. If it isn't there yet, **run anyway**: `harness.summarize` recomputes efficiency from the M0+A0 files for the paper table, and the instructor scores test answers separately.

---

## 2. Day 0: setup together on one call (≈30 min, tonight 26 Nov)

C drives. A and B watch, so all three know the setup for the viva.

1. ✅ *Done 26 Nov:* review files moved to `review/G10.csv` and `review/to_review_G04.jsonl`; `git init -b main` run; `git status` lists 61 files (no `.venv/`, no `__pycache__`, largest is the 2.4 MB judgments file). **Still to do (C):** make the **first commit** with everything as it is (harness untouched):
   ```bash
   git add -A
   git commit -m "G10 starting point: starter kit, M1 memory, cite check, Step 1 files, plan"
   ```
2. Create a **private** GitHub repo, push `main`, add A and B as collaborators (and the instructor, if they want repo access).
3. GitHub → Settings → Secrets and variables → Actions → **Variables**: `QA_GROUP=G10`, `QA_MEMORY=M1`.
4. Create a **Langfuse Cloud** project (free) and invite A and B. Share the public and secret keys **privately** (WhatsApp/DM, never in the repo). Everyone uses the same project, so all traces from all 3 machines land in one dashboard.
5. Create the shared Google Drive folder `G10/` with subfolders `A/`, `B/`, `C/`, `traces/`, `adapter/`.
6. Each member: `git checkout -b a-memory` / `b-training` / `c-monitoring` and push it.
7. Each member: fill in the **consent form** on Google Classroom (optional, doesn't affect marks, but do it).
8. **B starts training right after the call** (it's the critical path, see §5).

### Running on Colab or Kaggle with Langfuse (everyone)

```python
# first cell of your own run notebook (A: notebooks/run_A.ipynb, C: notebooks/run_C.ipynb, B: the finetune notebook)
!git clone -b <your-branch> https://github.com/<owner>/<repo>.git && %cd <repo>
!pip install -q -r requirements.txt opentelemetry-sdk opentelemetry-exporter-otlp
import os, base64
from google.colab import userdata, drive          # keep keys in Colab "Secrets", not in the notebook
drive.mount("/content/drive")
pk, sk = userdata.get("LANGFUSE_PK"), userdata.get("LANGFUSE_SK")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"   # US: us.cloud.langfuse.com
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Basic%20" + base64.b64encode(f"{pk}:{sk}".encode()).decode()
```

After each run: `!mkdir -p /content/drive/MyDrive/G10/<X> && cp -r results traces /content/drive/MyDrive/G10/<X>/`. Then on your laptop, copy *your* result files into your branch and commit them.
Backup GPU if Colab quota runs out: **Kaggle** (free T4, 30 h/week, 12 h sessions). Use the same commands, and write the GPU you used in the model card and report.

---

## 3. Who owns what (no file is shared)

| Member | Owns these files (only they edit them) |
| --- | --- |
| **A, Memory** | `agent/memory/m1_summary.py` (ablation knob only) · `tests/test_m1.py` · `MEMORY.md` · `results/G10_M1+A0_*` · `results/G10ABL*` · `errors/ERRORS_A.csv` · `report/parts/memory.tex` · `report/parts/ablation.tex` · `report/parts/trace_failure.tex` · `report/parts/contrib_A.tex` · `notebooks/run_A.ipynb` · (optional) branch `open-arena` |
| **B, Training** | `notebooks/finetune_qlora.ipynb` · `MODEL_CARD.md` · `assets/sft_loss.png` · `assets/dpo_margin.png` · `results/G10_M0+A2_*` · `results/G10_M1+A2_*` · `errors/ERRORS_B.csv` · `report/parts/training.tex` · `report/parts/trace_success.tex` · `report/parts/contrib_B.tex` · adapter on HF/Drive |
| **C, Monitoring + report** | `.gitignore` · `.github/workflows/regression.yml` · `scripts/pre-push` · `tests/test_cite_check.py` · `agent/cite_check.py` (frozen, but C answers for it in the viva) · `results/accepted_smoke.json` · `results/G10_M0+A0_*` · `dashboard/*` · `errors/ERRORS_C.csv` · `ERRORS.csv` (merged on the final day) · `DEVIATIONS.md` · `report/section.tex` · `report/tables/*` · `report/parts/monitor.tex` · `report/parts/results.tex` · `report/parts/analysis.tex` · `report/parts/limitations.tex` · `report/parts/contrib_C.tex` · `notebooks/run_C.ipynb` · `submission/*` (final only) |

The report is split into `report/parts/*.tex` files that C's `report/section.tex` pulls in with `\input{parts/...}`. Everyone writes their own part at any time. If a part doesn't exist yet, C uses a placeholder until merge.

---

## 4. Member A: Memory (Step 2, Step 5 memory experiment)

**A1. Ablation knob** (first hour, then never touch `agent/` again). In `SummaryMemory.read()`, first lines:
```python
cap = int(os.environ.get("QA_M1_BUDGET", "0"))   # ablation only; 0 = use config budget (default)
if cap:
    budget_tokens = min(budget_tokens, cap)
```
(`import os` at the top.) Run `python tests/test_m1.py`, then commit. The default behaviour is identical, so B and C can run M1 before this is merged.

**A2. Real M1+A0 runs on dev**, all 3 seeds (≈1.5× the time of an M0 run, because of the extra summary call):
```bash
python -m harness.evaluate --group G10 --memory M1 --adapter A0 --split dev --all-seeds
```
Copy to Drive. Commit `results/G10_M1+A0_dev_*`.

**A3. Memory experiment (Step 5, Q2, 6 marks).** Same config, two smaller budgets (the config budget is 768, and we can't go above it without a deviation):
```bash
QA_M1_BUDGET=256 python -m harness.evaluate --group G10ABL256 --memory M1 --adapter A0 --split dev --all-seeds
QA_M1_BUDGET=128 python -m harness.evaluate --group G10ABL128 --memory M1 --adapter A0 --split dev --all-seeds
python -m harness.summarize --group G10ABL256 --split dev ; python -m harness.summarize --group G10ABL128 --split dev
```
(PowerShell: `$env:QA_M1_BUDGET=256` first. On Colab: `os.environ[...]`.) If GPU time is short, use seed 13 only and say so in the report. Our typical memory size is only 60–120 tokens, so if 256 shows no change, 128 or 64 is the interesting point. Report score, memory recall and tokens per turn against the default 768 (the M1+A0 runs above). Optional, if the adapter is ready in time: repeat the best setting with `--adapter A2`.

**A4. `MEMORY.md`.** Replace examples 1 and 3 with **hf** transcripts from A2 (read the `memory_read` and `memory_write` spans in `traces/G10_M1+A0_dev_s13_*.jsonl` or in Langfuse). Keep example 2 (the correction), but also show one real correction if possible: run `python scripts/chat.py --memory M1` with "my phone was stolen on 3 March 2025" and then "correction: it was 5 May 2024", and paste the trace. Remove the "must be replaced" warning. One page.

**A5. Secret test (28 Nov):** `python -m harness.evaluate --group G10 --memory M1 --adapter A0 --split test --questions secret_test_questions.jsonl --all-seeds`, then copy to Drive and commit `results/G10_M1+A0_test_*`.

**A6. Label 20 mistakes** from your M1+A0 dev runs into `errors/ERRORS_A.csv` (same header as `templates/ERRORS.csv`). Focus on memory errors **E3/E4**, but use whatever code fits. See §7 for how to find them.

**A7. Report parts:** `parts/memory.tex` (Method → Memory: store, write, read and correction rules, precise enough to rebuild; condense `MEMORY.md`) · `parts/ablation.tex` (ablation table plus 2–3 sentences) · `parts/trace_failure.tex` (one annotated **failure** trace, turn by turn, quoting memory read/write and citations) · `parts/contrib_A.tex` (one `\item` line saying what you did).

**A8 (optional, bonus +5/+3/+2): Open leaderboard.** On branch `open-arena`, **never merged into main**, try any trick with the same model and documents (prompt tweaks are allowed here) and run it as `--group G10OPEN`. Only once everything else is done. Check the instructor's instructions for how to submit it.

---

## 5. Member B: Training (Step 3), and all adapter runs

**B1. Train tonight** (critical path: everything with A2 waits for it). In `notebooks/finetune_qlora.ipynb`, set `REPO_URL` (your branch), `GROUP="G10"`, `TRAIN_TYPE="A2"`. Run sections 1–6 on a T4. Training settings come from `config.yaml`, so **do not change them** (any change goes in `DEVIATIONS.md` via C). Leave `USE_SFT_NEGATIVES=False` unless you have spare GPU time, and if you switch it on, say so in the model card.
- Save the leakage-check output lines for SFT and DPO (they must say `OK`).
- Save `sft_loss.png` and the DPO reward-margin plot (`dpo_margin.png`) into `assets/`.
- Write down the GPU type and the wall-clock minutes for SFT and DPO.
- Upload `adapters/a2` to **Hugging Face (public)** or to Drive with "anyone with the link" access, because the instructor must be able to load it for re-runs. Copy it to `G10/adapter/` too.
- **Post the adapter link in the group chat** the moment it exists (C uses it for the fallback in §9).

**B2. Adapter runs on dev**, all 3 seeds:
```bash
python -m harness.evaluate --group G10 --memory M0 --adapter A2 --adapter-path adapters/a2 --split dev --all-seeds
python -m harness.evaluate --group G10 --memory M1 --adapter A2 --adapter-path adapters/a2 --split dev --all-seeds
```
(`--adapter-path` can also be the HF id.) Copy to Drive and commit `results/G10_M0+A2_dev_*` and `results/G10_M1+A2_dev_*`.

**B3. Secret test (28 Nov):** the same two configs with `--split test --questions secret_test_questions.jsonl --all-seeds`, then commit `results/G10_M0+A2_test_*` and `results/G10_M1+A2_test_*`.

**B4. `MODEL_CARD.md`.** Fill in every **TODO (Colab)**: adapter link, `adapter_hash` (from any A2 `results/*.json`), the leakage line, GPU, SFT + DPO minutes, GPU-hours, the two plots, at least 3 failure cases with E-codes (from your runs), and the practice results table (M0+A0 vs M0+A2: run `python -m harness.summarize --group G10 --split dev` after merging `main` on the final day).

**B5. Label 20 mistakes** (10 from M0+A2 and 10 from M1+A2) into `errors/ERRORS_B.csv`. Watch especially for **E5/E6/E7** (citation and abstain behaviour is what DPO targets).

**B6. Report parts:** `parts/training.tex` (data, 1851 examples, leakage check, hyperparameters, how rejected answers were built, adapter link + hash, GPU-hours) · `parts/trace_success.tex` (one annotated **success** trace, ideally M1+A2) · `parts/contrib_B.tex`.

---

## 6. Member C: Monitoring (Step 4), baseline runs, report and submission

**C1. Baseline runs on dev** (start on Day 0, right after the call):
```bash
python -m harness.evaluate --group G10 --memory M0 --adapter A0 --split dev --all-seeds
```
Put the 3 `results/G10_M0+A0_dev_s*.json` files in Drive `G10/C/` straight away (rule 10), then commit `results/G10_M0+A0_dev_*`.

**C2. Automatic test (4 marks):**
- `.gitignore`: add `results/G10MOCK_*`, `results/*_smoke_s*`, `secret_test_questions.jsonl`.
- Mock reference score (what CI checks): `python -m harness.smoke --group G10 --memory M1 --adapter A0 --backend mock --accept`
- Real reference score (on Colab, hf): `python -m harness.smoke --group G10 --memory M1 --adapter A0 --accept`. Copy the updated `results/accepted_smoke.json` back and commit it (it holds both the `|mock` and `|hf` keys).
- Push, and check that the GitHub Action **regression** is green. Screenshot it for the report.
- Pre-push hook (real model before push): `cp scripts/pre-push .git/hooks/pre-push`, with `QA_GROUP=G10 QA_MEMORY=M1`. On a laptop without a GPU, set `QA_BACKEND=openai` against a local Ollama/LM Studio serving the same Qwen3-4B-Instruct-2507, or rely on the Colab hf smoke run plus CI. Explain the choice in `parts/monitor.tex`.
- Fix `tests/test_cite_check.py` so it runs directly (add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` like `scripts/chat.py` does).

**C3. Traces (6 marks):** every member's evaluations already write `traces/` and (with the env vars from §2) mirror to Langfuse. Check in Langfuse that runs from A and B appear and that every turn has all 5 spans. On the final day, gather everyone's `traces/G10_*.jsonl` from Drive into `traces/`, then:
```bash
python -m harness.trace_export --group G10              # all runs
python -m harness.trace_export --group G10 --split test # Arena bundle
```
Both must print `OK: traces are complete.`

**C4. Dashboard (3 marks):** `dashboard/make_dashboard.py` reads `results/G10_*_<split>_s*.json` and draws one PNG per split: 4 bars (one per version) × 4 panels, **score, tokens/turn, latency p50, ungrounded-citation rate**, with mean ± error bar across the 3 seeds. Build and test it now on your M0+A0 files plus G10MOCK files for the other configs. On the final day it just re-runs on the full `results/`. Save `dashboard/dashboard_dev.png` and `dashboard/dashboard_test.png`, and also take a **Langfuse dashboard screenshot** (`dashboard/langfuse.png`). Both go in the submission. The same figure gives the "comparison with error bars" for Step 5.

**C5. Secret test (28 Nov), first thing:** `python -m harness.evaluate --group G10 --memory M0 --adapter A0 --split test --questions secret_test_questions.jsonl --all-seeds`, then put the 3 `.json` files in Drive for A and B, and commit `results/G10_M0+A0_test_*`.

**C6. Label 20 mistakes** from M0+A0 into `errors/ERRORS_C.csv` (expect lots of **E3**, since the baseline forgets anything older than 2 turns, plus E1 and E8).

**C7. `DEVIATIONS.md`** from `templates/`. Expected content: "None" for fixed controls, plus one row saying that the memory ablation used `QA_M1_BUDGET` 256/128 (below the config's 768) under separate group ids, and that the main results use the default. Add anything A or B tell you about (GPU type isn't a deviation).

**C8. Report skeleton now:** copy `harness/paper/section_template.tex` to `report/section.tex`, fill in `\groupid{G10}`, `\memtype{M1 (rolling summary memory)}`, `\traintype{A2 (QLoRA SFT + DPO)}` and the names, and replace each section body with `\input{parts/<file>}`. Write `parts/monitor.tex` (citation monitor: drop policy and why, CI gate, pre-push, Langfuse), `parts/results.tex` (2×2 table, effects, cost table: latency p50/p95 from summarize, GPU-hours from MODEL_CARD), `parts/analysis.tex` (error table per version from ERRORS.csv, interaction discussion; it `\input`s A's and B's trace parts) and `parts/limitations.tex` (not legal advice, wrong-Act citations a lay user can't spot, known failures listed in MEMORY.md and MODEL_CARD.md, deviations).

---

## 7. How anyone finds mistakes to label (for A6, B5, C6)

```bash
python -c "import csv,sys; [print(r['config'],r['seed'],r['session_id'],r['turn'],r['family'],'ungrounded='+r['ungrounded_cites'],'abst_exp='+r['abstain_expected'],'abst='+r['abstained'],'parse='+r['parse_ok']) for r in csv.DictReader(open(sys.argv[1])) if r['correct']=='0' or r['ungrounded_cites'] not in ('','0')]" results/G10_M1+A0_dev_s13.csv
```
For each row, open that session's turn in Langfuse (or `traces/`) to see what memory held, what was retrieved and what the model said. Then give **one** code (E1–E8, see FORMAT.md §7) plus an optional secondary code, and a one-line note. CSV header: `config,seed,session_id,turn,code,secondary_code,note`. Don't hand-edit any `results/` file.

---

## 8. Timeline

| When | A (Memory) | B (Training) | C (Monitoring) |
| --- | --- | --- | --- |
| **Thu 26 Nov, evening** | Day-0 call · ablation knob · start M1+A0 dev | Day-0 call · **start SFT+DPO training** | Day-0 call (drives) · start M0+A0 dev |
| **Fri 27 Nov** | finish M1+A0 dev · ablation runs · MEMORY.md hf examples | finish training · upload adapter · post link · start M0+A2 dev | CI + accepted smoke · Langfuse check · dashboard script · report skeleton · DEVIATIONS |
| **Sat 28 Nov** (test released) | M1+A0 test | M1+A2 dev · M0+A2 test | **M0+A0 test first** → baseline json to Drive · monitor.tex |
| **Sun 29 Nov** | 20 errors · report parts | M1+A2 test · 20 errors · MODEL_CARD · report parts | 20 errors · results/analysis/limitations drafts |
| **Sun 29 Nov, 22:00** | **All three branches merged into `main`** (see §10). All test runs finished. | | |
| **Mon 30 Nov** | proof-read whole report | proof-read whole report | final tables, dashboard, trace export, compile, **submit by 18:00** (6 h buffer) |

The test window closes about 48 h after release, so all 12 test runs must be finished and copied to Drive before then.
Rough GPU needs per dev run (241 turns, T4): M0 ≈ 30–50 min per seed, M1 ≈ 1.5× that. B carries the most GPU work (training + 12 runs). C has the least, which is why C is B's fallback in §9.

---

## 9. If something goes wrong

| Problem | What to do (no one has to wait) |
| --- | --- |
| Training is late | A and C keep going (their configs don't need the adapter). Once the link is posted, **C runs `M1+A2` test** with it, as C's GPU load is lightest. File names don't collide, so the only change is who commits `results/G10_M1+A2_test_*`. Agree this in chat first so it isn't run twice. |
| Colab quota is gone | Switch to Kaggle. Or run fewer seeds on the ablation only (never on the main 4×3). |
| A run crashed halfway | Re-run that seed only: `--seed 42` instead of `--all-seeds`. Partial files get overwritten. |
| Someone lost traces | Re-run that config and seed (same seed gives the same answers). The trace export needs every turn. |
| A2 scores *below* baseline | That's still a valid result for the paper and costs no marks outside the competition. Report it honestly. |
| Merge conflict | Someone edited a file they don't own. The owner's version wins. |

---

## 10. Final integration (Sun 29 Nov 22:00 → Mon 30 Nov), C drives with A and B on a call

1. Each member opens a PR from their branch into `main`. Merge them in any order (no shared files). CI must stay green.
2. Everyone's traces from Drive → `traces/`. Then:
```bash
(head -1 errors/ERRORS_A.csv; tail -q -n +2 errors/ERRORS_A.csv errors/ERRORS_B.csv errors/ERRORS_C.csv) > ERRORS.csv   # must be 60 rows
python -m harness.summarize --group G10 --split dev
python -m harness.summarize --group G10 --split dev  --latex > report/tables/table_2x2_dev.tex
python -m harness.summarize --group G10 --split test --latex > report/tables/table_2x2_test.tex
python dashboard/make_dashboard.py
python -m harness.trace_export --group G10
python -m harness.trace_export --group G10 --split test
```
3. B fills the practice table in `MODEL_CARD.md`. C pastes the numbers into `parts/results.tex`. **Every number in the report must match `results/`** (4 marks depend on it).
4. Compile `report/section.tex` with `pdflatex` (or Overleaf) → `report/section.pdf`, at most 3 pages plus appendix. All three proof-read it.
5. Check that `results/` has exactly the 4 configs × 3 seeds for dev **and** for test, plus the ablation `G10ABL*` files, and that none of them say `MOCK BACKEND` in their `.json`.
6. Commit, push, tag `submission`. **One member (C)** submits on Google Classroom: repo link, `results/` folder, `submission/G10_traces*.jsonl.gz`, report PDF (+ dashboard screenshot).

---

## 11. Submission checklist (mapped to marks)

- [ ] **S1 (10):** `contrib/G10.jsonl` valid · `review/G10.csv` with 10 rows, every row has a reason
- [ ] **S2 (13):** M1 works and switches off with `--memory M0` · `MEMORY.md` (rules + 3 hf examples, one with a correction) · tests pass
- [ ] **S3 (12):** adapter link works for outsiders · notebook re-runs end to end · `MODEL_CARD.md` has no TODOs left (settings, loss plot, GPU, time, failure cases)
- [ ] **S4 (20):** traces for every run of all 4 versions (`trace_export` OK) · citation checker · CI green + accepted smoke score + pre-push · dashboard PNG + Langfuse screenshot · `ERRORS.csv` with 60 rows
- [ ] **S5 (20):** 4 versions × 3 seeds on dev and test, unchanged · error bars (CIs from summarize, dashboard) · memory ablation · cost (tokens/turn, latency, GPU-hours)
- [ ] **S6 (15):** PDF + LaTeX source · method rebuildable · tables match results · 2 annotated traces (success: B, failure: A) · honest limitations · one contribution line per member
- [ ] `DEVIATIONS.md` present (even if it says "None")
- [ ] Every answer shows the disclaimer (built in through `Answer.display()`, so don't remove it)
- [ ] Consent forms filled in (each member)

---

## 12. Viva prep (final marks × viva score, each of us alone)

We each get asked about **everything**, not just our own part. On **Mon 30 Nov evening**, each member gives a 15-minute walkthrough of their part to the other two:
- **A:** how `m1_summary.py` writes and reads, the loss and drift guards, how corrections work, why the retrieval hint helps, what the ablation showed
- **B:** QLoRA (4-bit NF4 + LoRA r16/α32), SFT prompt-completion loss, DPO (β = 0.1, chosen/rejected, how negatives are built), why there's no leakage, cost
- **C:** the graph's 5 nodes, the cite-check drop policy and why it can only help precision, the composite score weights, the regression gate (tolerance 2), how bootstrap CIs and the 2×2 effects are computed, the trace format

You must be able to explain every line you committed, including anything an AI assistant wrote.
