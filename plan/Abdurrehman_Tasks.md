# Member C: Monitoring, baseline, report and submission

Name: ______ · Branch: `c-monitoring` · Marks you mainly carry: Step 4 (20), the baseline and comparison for Step 5, assembling Step 6 (15), and the submission

This file is all you need. You don't wait for A or B: everything you build (CI, dashboard, report skeleton) can be tested on your own baseline runs plus mock runs, and simply re-run on the full results at the end. You also run the Day-0 call, the final merge and the submission.

---

## Rules (same for all three of us)

1. **Only edit and commit the files listed under "Your files".** That's what lets the three branches merge with zero conflicts.
2. **Never edit `harness/`.** CI fails if you do.
3. **`agent/` is frozen** (that includes `agent/cite_check.py`, which is already built). You answer for it in the viva, but don't change it.
4. **Final runs use the real model** (`--backend hf`, the default). For quick checks with `--backend mock`, always use `--group G10MOCK` and never commit those.
5. **Commit only your own result patterns:** `git add "results/G10_M0+A0_*" results/accepted_smoke.json`. Never `git add results/` or `git add .`.
6. **Colab deletes everything when the runtime dies.** After every run, copy `results/` and `traces/` to Drive `G10/C/`.
7. **Put your baseline `.json` files in Drive `G10/C/` as soon as each baseline run finishes.** A and B use them to get the right efficiency number in their run summaries. It doesn't block them, but do it quickly.

## Your files

`.gitignore` · `.github/workflows/regression.yml` · `scripts/pre-push` · `tests/test_cite_check.py` · `results/accepted_smoke.json` · `results/G10_M0+A0_*` · `dashboard/*` · `errors/ERRORS_C.csv` · `ERRORS.csv` (final day) · `DEVIATIONS.md` · `report/section.tex` · `report/tables/*` · `report/parts/monitor.tex` · `report/parts/results.tex` · `report/parts/analysis.tex` · `report/parts/limitations.tex` · `report/parts/contrib_C.tex` · `notebooks/run_C.ipynb` · `submission/*` (final day)

---

## Day 0 (you drive, Thu 26 Nov evening, ≈30 min with A and B on a call)

Already done on the laptop: the review files were moved into `review/`, and `git init -b main` was run (no commit yet).
1. First commit, with `harness/` untouched (CI compares against this commit):
   ```bash
   git add -A
   git commit -m "G10 starting point: starter kit, M1 memory, cite check, Step 1 files, plan"
   ```
2. Create a **private** GitHub repo, push `main`, and add A and B (plus the instructor, if they want repo access).
3. GitHub → Settings → Secrets and variables → Actions → **Variables**: `QA_GROUP=G10` and `QA_MEMORY=M1`.
4. Create a **Langfuse Cloud** project (free) and invite A and B. Share the public and secret keys privately, never in the repo.
5. Create the shared Drive folder `G10/` with subfolders `A/`, `B/`, `C/`, `traces/` and `adapter/`, and share it with A and B.
6. Everyone creates and pushes their branch: `a-memory`, `b-training`, `c-monitoring`.
7. Remind everyone to fill in the consent form on Google Classroom.

## Colab / Kaggle setup (first cell of `notebooks/run_C.ipynb`)

```python
!git clone -b c-monitoring https://github.com/<owner>/<repo>.git && %cd <repo>
!pip install -q -r requirements.txt opentelemetry-sdk opentelemetry-exporter-otlp
import os, base64
from google.colab import userdata, drive          # Langfuse keys go in Colab "Secrets", never in the notebook
drive.mount("/content/drive")
pk, sk = userdata.get("LANGFUSE_PK"), userdata.get("LANGFUSE_SK")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"   # US: us.cloud.langfuse.com
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Basic%20" + base64.b64encode(f"{pk}:{sk}".encode()).decode()
```
After every run: `!mkdir -p /content/drive/MyDrive/G10/C && cp -r results traces /content/drive/MyDrive/G10/C/`

---

## Tasks

### C1. Baseline on the practice set (start right after the Day-0 call)
```bash
python -m harness.evaluate --group G10 --memory M0 --adapter A0 --split dev --all-seeds
```
Roughly 30–50 min per seed on a T4. Put the 3 `results/G10_M0+A0_dev_s*.json` files in Drive `G10/C/` straight away, then commit `results/G10_M0+A0_dev_*`.

### C2. Automatic test (Step 4: 4 marks)
- Add these lines to `.gitignore`:
  ```
  results/G10MOCK_*
  results/*_smoke_s*
  secret_test_questions.jsonl
  ```
- Reference score for CI (mock): `python -m harness.smoke --group G10 --memory M1 --adapter A0 --backend mock --accept`
- Reference score with the real model (on Colab): `python -m harness.smoke --group G10 --memory M1 --adapter A0 --accept`. Copy the updated `results/accepted_smoke.json` back and commit it. It must hold both the `M1+A0|mock` and `M1+A0|hf` keys.
- Push, check that the GitHub Action **regression** goes green, and screenshot it for the report.
- Pre-push hook: `cp scripts/pre-push .git/hooks/pre-push`, with `QA_GROUP=G10` and `QA_MEMORY=M1`. On a laptop without a GPU, either set `QA_BACKEND=openai` against a local Ollama/LM Studio serving the same Qwen3-4B-Instruct-2507, or rely on CI plus the Colab real-model smoke run. Explain the choice in `monitor.tex`.
- Make `tests/test_cite_check.py` runnable directly by adding the same path line `scripts/chat.py` uses: `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`.

### C3. Traces (Step 4: 6 marks)
- Every evaluation writes `traces/` automatically, and also sends to Langfuse when the env vars are set. Check in Langfuse that runs from A and B show up, and that every turn has all 5 spans: `memory_read`, `retrieve`, `generate`, `cite_check`, `memory_write`.
- On the final day, collect everyone's `traces/G10_*.jsonl` from Drive (`G10/A`, `G10/B`, `G10/C`) into your local `traces/`, then:
  ```bash
  python -m harness.trace_export --group G10
  python -m harness.trace_export --group G10 --split test
  ```
  Both must print `OK: traces are complete.` If any trace is missing, the owner re-runs that version and seed.

### C4. Dashboard (Step 4: 3 marks; it also gives the Step 5 error bars)
Write `dashboard/make_dashboard.py`. It reads `results/G10_*_<split>_s*.json` (the `summary` block) and saves one PNG per split, `dashboard/dashboard_dev.png` and `dashboard/dashboard_test.png`. Each PNG has 4 panels: **score, tokens per turn, latency p50 and ungrounded-citation rate**. Each panel has 4 bars (M0+A0, M1+A0, M0+A2, M1+A2), showing the mean with an error bar across the 3 seeds. Use matplotlib only.
- Test it now with your M0+A0 files plus mock runs for the other versions:
  ```bash
  python -m harness.evaluate --group G10MOCK --memory M1 --adapter A0 --split dev --backend mock --all-seeds
  ```
  Point the script at a `--group` argument so it works for `G10MOCK` and `G10`.
- On the final day, run it on the real `G10` results. Also take a **Langfuse dashboard screenshot** and save it as `dashboard/langfuse.png`.

### C5. Secret test (Sat 28 Nov, open 48 h): run the baseline FIRST
```bash
python -m harness.evaluate --group G10 --memory M0 --adapter A0 --split test --questions secret_test_questions.jsonl --all-seeds
```
Put the 3 `.json` files in Drive `G10/C/` for A and B, then commit `results/G10_M0+A0_test_*`.
**Backup for B:** if B is short on GPU time, run `M1+A2` on the test set with B's adapter link. Agree on it in the chat first.

### C6. Label 20 mistakes → `errors/ERRORS_C.csv`
Take them from M0+A0. Header: `config,seed,session_id,turn,code,secondary_code,note`. Expect lots of **E3** (the baseline only sees the last 2 turns), plus E1 and E8. List the failed turns with:
```bash
python -c "import csv,sys; [print(r['config'],r['seed'],r['session_id'],r['turn'],r['family'],'ungrounded='+r['ungrounded_cites'],'abst_exp='+r['abstain_expected'],'abst='+r['abstained'],'parse='+r['parse_ok']) for r in csv.DictReader(open(sys.argv[1])) if r['correct']=='0' or r['ungrounded_cites'] not in ('','0')]" results/G10_M0+A0_dev_s13.csv
```

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

### C7. `DEVIATIONS.md`
Copy it from `templates/DEVIATIONS.md`. Expected content:
- "None" for the fixed controls
- one row saying the memory ablation used `QA_M1_BUDGET` 256 and 128 (below the config's 768) under the separate group ids `G10ABL256` and `G10ABL128`, and that the main results use the default
- B's `ABSTAIN_FRAC` choice, if the group kept 0.2 (it isn't a config.yaml value, but it's worth stating)
- anything else A or B tell you about (the GPU type is not a deviation)

### C8. Report skeleton and your parts (start Friday)
- Copy `harness/paper/section_template.tex` to `report/section.tex`. Set `\groupid{G10}`, `\memtype{M1 (rolling summary memory)}`, `\traintype{A2 (QLoRA SFT + DPO)}` and the three names. Keep the template's headings, tables and macros, and replace each section body with `\input{parts/...}`:
  - Method → `memory` (A), `training` (B), `monitor` (you)
  - Results → `results` (you; it inputs A's `ablation`)
  - Analysis → `analysis` (you; it inputs A's `trace_failure` and B's `trace_success`)
  - Limitations → `limitations` (you)
  - Contributions → `contrib_A`, `contrib_B`, `contrib_C`
  Until A's and B's files arrive, create **empty placeholder files** for their parts *in a scratch folder only*, not committed, so the merge doesn't conflict.
- `report/parts/monitor.tex`: the citation monitor (it **drops** citations that don't exist or that the bot never saw this turn, and why that can only improve precision), the CI gate (tolerance 2 points), the pre-push hook, Langfuse.
- `report/parts/results.tex`: the 2×2 table with confidence intervals (`\input{tables/table_2x2_dev.tex}` and the test version), effects of memory, training and the interaction, and the cost table (latency p50/p95 from summarize, tokens per turn, GPU-hours from `MODEL_CARD.md`).
- `report/parts/analysis.tex`: error counts per version from `ERRORS.csv`, what the interaction and failure counts tell us, then the two traces.
- `report/parts/limitations.tex`: not legal advice; wrong-Act citations a lay user can't detect; known failures from `MEMORY.md` and `MODEL_CARD.md`; the deviations.
- `report/parts/contrib_C.tex`: one line, `\item <Name>: ...`.

---

## Final integration (Sun 29 Nov 22:00 → Mon 30 Nov), you drive with A and B on a call

1. A, B and you each open a PR into `main`. Merge them in any order (no shared files). CI must stay green.
2. Copy everyone's traces from Drive into `traces/`, then:
   ```bash
   (head -1 errors/ERRORS_A.csv; tail -q -n +2 errors/ERRORS_A.csv errors/ERRORS_B.csv errors/ERRORS_C.csv) > ERRORS.csv   # must be 60 rows
   python -m harness.summarize --group G10 --split dev
   python -m harness.summarize --group G10 --split dev  --latex > report/tables/table_2x2_dev.tex
   python -m harness.summarize --group G10 --split test --latex > report/tables/table_2x2_test.tex
   python dashboard/make_dashboard.py --group G10
   python -m harness.trace_export --group G10
   python -m harness.trace_export --group G10 --split test
   ```
3. B fills in the practice table in `MODEL_CARD.md`. You put the numbers into `results.tex`. **Every number in the report must match `results/`** (4 marks depend on it).
4. Compile `report/section.tex` (`pdflatex` or Overleaf) into `report/section.pdf`: at most 3 pages plus appendix. All three proof-read it.
5. Check `results/`:
   - exactly the 4 versions × 3 seeds for both dev and test
   - the `G10ABL*` ablation files
   - no `MOCK BACKEND` warning in any `G10_*.json`
   - nothing edited by hand
6. Commit, push and tag `submission`. **You submit on Google Classroom, by Mon 30 Nov 18:00** (6 hours of buffer before 23:59): the repo link, the `results/` folder, `submission/G10_traces*.jsonl.gz`, the report PDF and the dashboard screenshot.

---

## Your timeline

| When | What |
| --- | --- |
| Thu 26 Nov, evening | **Day-0 call (you drive)** · start C1 baseline |
| Fri 27 Nov | C2 CI + smoke · C3 Langfuse check · C4 dashboard script · C7 deviations · C8 skeleton |
| Sat 28 Nov | **C5 baseline secret test first**, then `.json` to Drive · `monitor.tex` |
| Sun 29 Nov | C6 errors · results/analysis/limitations drafts · **open your PR by 22:00** |
| Mon 30 Nov | final integration · submit by 18:00 · viva walkthrough in the evening |

## Your checklist
- [ ] Day 0 done: first commit, private repo, Actions variables, Langfuse, Drive, branches
- [ ] `results/G10_M0+A0_{dev,test}_s{13,42,2026}.*` committed (real model)
- [ ] `results/accepted_smoke.json` with both `|mock` and `|hf` keys · CI green · screenshot
- [ ] Langfuse shows traces from all three members · `dashboard/langfuse.png`
- [ ] `dashboard/dashboard_dev.png` and `dashboard/dashboard_test.png`
- [ ] `ERRORS.csv` has 60 rows · `DEVIATIONS.md`
- [ ] Both `trace_export` runs print OK
- [ ] Report PDF + LaTeX source in `report/`, tables match `results/`
- [ ] Submitted on Google Classroom

## Viva prep (you're asked about everything, not just monitoring)
- **The graph:** the 5 nodes in `agent/graph.py` and what flows between them.
- **The cite checker:** the drop policy in `agent/cite_check.py`, and why dropping beats flagging or blocking.
- **The score:** the composite score weights (35/20/15/15/15) and how efficiency compares against the baseline's tokens.
- **The regression gate:** how the smoke test works and its 2-point tolerance.
- **Confidence intervals:** how `harness.summarize` gets them (bootstrap over conversations) and the memory, training and interaction effects.
- **The trace format.**

Also read `TASKS_A_memory.md` and `TASKS_B_training.md` so you can answer basic questions on the summary memory and DPO.
