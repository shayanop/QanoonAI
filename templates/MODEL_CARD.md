# MODEL_CARD.md — Group <G..>, training type <A1 | A2>

## Adapter
| Item | Value |
| --- | --- |
| Download link (Hugging Face or Google Drive) | |
| Base model | Qwen/Qwen3-4B-Instruct-2507 (from harness/config.yaml) |
| Method | QLoRA SFT  /  QLoRA SFT then DPO |
| Adapter hash (from the run manifest `adapter_hash`) | |

## Training data
| Item | Value |
| --- | --- |
| Source | `python -m harness.build_sft_data` on the train split |
| Number of SFT examples | |
| Leakage check passed (paste the output line) | |
| A2 only: number of preference pairs | |
| A2 only: how the rejected answers were made | |

## Settings (must match harness/config.yaml)
| Setting | Value |
| --- | --- |
| Epochs | |
| LoRA r / alpha / dropout | |
| Learning rate (SFT) | |
| DPO beta and learning rate (A2) | |
| Max sequence length | |
| Seed | |

## Cost
| Item | Value |
| --- | --- |
| GPU (e.g. Colab T4 16 GB) | |
| Training time (wall clock) | |
| GPU-hours | |

## Training curves
<!-- Insert the training loss plot (and the DPO reward-margin plot for A2). -->

## Results on the practice set (mean of 3 seeds)
| Version | Score | Accuracy | Citation F1 |
| --- | --- | --- | --- |
| M0+A0 | | | |
| M0+<your A> | | | |

## Known failure cases
<!-- At least 3 examples where the trained model is wrong, with the error code (E1 to E8). -->
