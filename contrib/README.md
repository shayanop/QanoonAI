# Step 1: your test questions

Save your 10 sessions here as `<group>.jsonl` (for example `G07.jsonl`): 2 sessions of
each type T1 to T5, written in the format in `harness/FORMAT.md`. Start from the real
examples in `harness/data/example_sessions.jsonl`.

Check the file before you submit:

```bash
python -m harness.validate_sessions contrib/G07.jsonl --contrib
```

Rules: no copying from the practice set, and every gold answer and quote must be checked
by a group member against the corpus (an LLM may help you draft, but not decide).
