# Deviations from Fixed Controls (`harness/config.yaml`)

- **Fixed controls (`harness/config.yaml`):** None. Model, corpus, retriever, prompt length, training caps, and random seeds (`13, 42, 2026`) match the starter config.
- **Memory budget ablation (`Step 5`):** Evaluated `QA_M1_BUDGET=256` and `QA_M1_BUDGET=128` under separate group IDs `G10ABL256` and `G10ABL128` (below the default `768`). All main `G10` evaluations use the default budget `768`.
- **DPO abstention fraction (`Step 3`):** Set `ABSTAIN_FRAC=0.2` when constructing DPO preference pairs (external training parameter, not part of `config.yaml`).
