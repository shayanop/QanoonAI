# Member B: Training (A2: QLoRA SFT, then DPO)

Name: ______ · Branch: `b-training` · Marks you mainly carry: Step 3 (12), the two trained versions for Step 5, plus your share of Steps 4 and 6

This file is all you need. You don't wait for A or C. **You are the critical path**: both trained versions need your adapter, so start training on Thursday night.

---

## Rules (same for all three of us)

1. **Only edit and commit the files listed under "Your files".** That's what lets the three branches merge with zero conflicts.
2. **Never edit `harness/`, and never change the training settings** (epochs, LoRA size, learning rates, DPO beta, number of examples, seeds). They come from `harness/config.yaml`. If you're forced to change something, tell C so it goes in `DEVIATIONS.md`.
3. **`agent/` is frozen.** Don't change memory or cite-check code.
4. **Final runs use the real model** (`--backend hf`, the default). For quick checks with `--backend mock`, always use `--group G10MOCK` and never commit those.
5. **Commit only your own result patterns:** `git add "results/G10_M0+A2_*" "results/G10_M1+A2_*"`. Never `git add results/` or `git add .`.
6. **Colab deletes everything when the runtime dies.** After training, copy `adapters/a2` to Drive `G10/adapter/`. After every run, copy `results/` and `traces/` to Drive `G10/B/`.
7. **Never train on test questions or other groups' questions.** The notebook's leakage check must say `OK`.
8. **Nice to have:** before running on a split, copy C's `results/G10_M0+A0_<split>_s*.json` from Drive `G10/C/` into your local `results/`. If they aren't there yet, **run anyway**.

## Your files

`notebooks/finetune_qlora.ipynb` · `MODEL_CARD.md` · `assets/sft_loss.png` · `assets/dpo_margin.png` · `results/G10_M0+A2_*` · `results/G10_M1+A2_*` · `errors/ERRORS_B.csv` · `report/parts/training.tex` · `report/parts/trace_success.tex` · `report/parts/contrib_B.tex` · the adapter on Hugging Face/Drive

## Langfuse tracing on Colab (add this before the evaluation cells)

```python
!pip install -q opentelemetry-sdk opentelemetry-exporter-otlp
import os, base64
from google.colab import userdata                 # Langfuse keys go in Colab "Secrets", never in the notebook
pk, sk = userdata.get("LANGFUSE_PK"), userdata.get("LANGFUSE_SK")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"   # US: us.cloud.langfuse.com
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Basic%20" + base64.b64encode(f"{pk}:{sk}".encode()).decode()
```
If you run out of Colab GPU, use Kaggle (free T4, 30 h/week, 12 h sessions). Write down which GPU you used.

---

## Tasks

### B0. Decide on `ABSTAIN_FRAC` before you train (one line in section 5 of the notebook)
The assignment says DPO pairs are **good = correct citation, bad = wrong citation**. Right now `ABSTAIN_FRAC = 0.2`, so about 19% of the pairs are "correct answer vs a wrong 'I am not sure'". The bad answer there has *no* citation rather than a wrong one.
- **Recommended:** set `ABSTAIN_FRAC = 0.0`. Every answerable pair then becomes correct vs wrong citation, matching the assignment and other groups' A2. The pairs on "I am not sure" turns (confident wrong citation as the bad answer) still teach the model when to refuse.
- If the group keeps 0.2, justify it in `MODEL_CARD.md` and in `training.tex`.
- Either way, update the pair-count table in `MODEL_CARD.md` with the counts the notebook prints.

### B1. Train (start Thu 26 Nov, night)
In `notebooks/finetune_qlora.ipynb`, set `REPO_URL` (your branch), `GROUP = "G10"` and `TRAIN_TYPE = "A2"`, then run sections 1 to 6 on a T4. Leave `USE_SFT_NEGATIVES = False` unless you have spare GPU time; if you switch it on, say so in the model card and paste the counts. Save as you go:
- both leakage-check output lines (SFT and DPO), which must say `OK: no overlap with evaluation questions.`
- `sft_loss.png` and the DPO reward-margin plot as `dpo_margin.png`, both into `assets/`
- the GPU type, plus wall-clock minutes for SFT and for DPO
- `adapters/a2`: upload it to **Hugging Face (public)** or to Drive with "anyone with the link" access, because the instructor must be able to load it for the re-run check. Also copy it to Drive `G10/adapter/`.

**Post the adapter link in the group chat the moment it exists.** C uses it if you need help with runs, and A may use it for an optional extra ablation.

### B2. Trained versions on the practice set (3 seeds each)
```bash
python -m harness.evaluate --group G10 --memory M0 --adapter A2 --adapter-path adapters/a2 --split dev --all-seeds
python -m harness.evaluate --group G10 --memory M1 --adapter A2 --adapter-path adapters/a2 --split dev --all-seeds
```
`--adapter-path` can also be the Hugging Face id. Roughly 30–50 min per seed for M0 and 1.5× that for M1 on a T4. Copy to Drive, then commit `results/G10_M0+A2_dev_*` and `results/G10_M1+A2_dev_*`.

### B3. Secret test (released Sat 28 Nov, open 48 h)
```bash
python -m harness.evaluate --group G10 --memory M0 --adapter A2 --adapter-path adapters/a2 --split test --questions secret_test_questions.jsonl --all-seeds
python -m harness.evaluate --group G10 --memory M1 --adapter A2 --adapter-path adapters/a2 --split test --questions secret_test_questions.jsonl --all-seeds
```
Copy to Drive, then commit `results/G10_M0+A2_test_*` and `results/G10_M1+A2_test_*`. Never commit `secret_test_questions.jsonl`.
**If you're running out of GPU time,** say so in the chat early. C can run `M1+A2` on the test set with your adapter link. Agree on it first so it isn't run twice.

### B4. `MODEL_CARD.md` (Step 3: 3 marks)
Fill in every **TODO (Colab)**:
- adapter link and `adapter_hash` (copy it from any `results/G10_M0+A2_*.json`)
- the leakage-check lines
- GPU, SFT + DPO minutes and total GPU-hours (C needs the GPU-hours for the cost table)
- the two plots from `assets/`
- at least 3 examples where the trained model is wrong, each with its error code
- the practice-set table comparing M0+A0 with M0+A2: on the final day, after merging `main` into your branch, run `python -m harness.summarize --group G10 --split dev` and copy the numbers

### B5. Label 20 mistakes → `errors/ERRORS_B.csv`
10 from M0+A2 and 10 from M1+A2. Header: `config,seed,session_id,turn,code,secondary_code,note`. Watch especially for **E5/E6/E7**, since citation and refusal behaviour is what DPO targets. List the failed turns with:
```bash
python -c "import csv,sys; [print(r['config'],r['seed'],r['session_id'],r['turn'],r['family'],'ungrounded='+r['ungrounded_cites'],'abst_exp='+r['abstain_expected'],'abst='+r['abstained'],'parse='+r['parse_ok']) for r in csv.DictReader(open(sys.argv[1])) if r['correct']=='0' or r['ungrounded_cites'] not in ('','0')]" results/G10_M0+A2_dev_s13.csv
```
Open each one in Langfuse (or `traces/`), then give one code, plus an optional second code and a one-line note.

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

### B6. Report parts (LaTeX; C's `report/section.tex` pulls them in)
- `report/parts/training.tex`: Method → Training. Data (1851 examples from `harness.build_sft_data`, broken down by type), the leakage check, hyperparameters (they must match `config.yaml`), how the rejected answers were built and the final `ABSTAIN_FRAC`, adapter link and hash, GPU-hours.
- `report/parts/trace_success.tex`: one annotated **success** trace, turn by turn, ideally from M1+A2, quoting the memory read, the retrieved documents and the citations.
- `report/parts/contrib_B.tex`: one line, `\item <Name>: ...`.

---

## Your timeline

| When | What |
| --- | --- |
| Thu 26 Nov, evening | Day-0 call · B0 decision · **start B1 training** |
| Fri 27 Nov | finish training · upload the adapter · post the link · start B2 (M0+A2) |
| Sat 28 Nov | B2 (M1+A2) · B3 secret test, M0+A2 |
| Sun 29 Nov | B3 secret test, M1+A2 · B4 model card · B5 errors · B6 report parts · **open a PR from `b-training` into `main` by 22:00** |
| Mon 30 Nov | practice table in the model card · proof-read the whole report · viva walkthrough in the evening |

## Your checklist
- [ ] `ABSTAIN_FRAC` decided and documented
- [ ] Adapter on Hugging Face or Drive, and the link works for someone logged out
- [ ] Both leakage checks `OK`
- [ ] `results/G10_M0+A2_{dev,test}_s{13,42,2026}.*` and `results/G10_M1+A2_{dev,test}_s{13,42,2026}.*` committed (real model)
- [ ] All your traces copied to Drive `G10/B/`
- [ ] `MODEL_CARD.md` with no TODOs left, plots in `assets/`
- [ ] `errors/ERRORS_B.csv`: 20 rows
- [ ] 3 report part files

## Viva prep (you're asked about everything, not just training)
- **QLoRA:** the 4-bit NF4 base model with LoRA r = 16, α = 32, dropout 0.05, and why it fits on a T4.
- **SFT loss:** the prompt/completion format, where loss is computed only on the answer.
- **DPO:** β = 0.1, what chosen and rejected mean, how the hard negatives are built, and what the reward margin plot shows.
- Why there's no leakage, and what the training cost.

Also read `TASKS_A_memory.md` and `TASKS_C_monitoring.md` so you can answer basic questions on the summary memory, the cite checker and the score formula.
