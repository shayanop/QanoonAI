# Step 1: your peer review

The instructor assigns you another group's question file. Run the checker on it first:

```bash
python -m harness.validate_sessions <their file>.jsonl --contrib
```

Then read every session and record one row per session in `review/<your group>.csv`,
using `templates/review_template.csv`:

- `accept`: correct as written;
- `fix`: right idea but something is wrong (say exactly what);
- `reject`: wrong answer, wrong source, or the question does not test what its type says.

Check in particular: does the quote really support the answer, does a T3/T4 question
truly need the earlier turn, and is a T5 question truly unanswerable from the corpus?
