# Final model decision

Written 2026-11-25, before the test set was used. Every number in this file is a validation number. Source: `experiments/results.csv` (Section 3 and Section 5 rows), `python -m src.selection --section 3 --section 5`.

## Rule

Fixed before Section 5 ran (`EXPERIMENT_LOG.md`, Section 5 plan, Selection rule):

1. Candidates: configurations with 5 seeds under the full protocol (max 150 epochs, early stopping on validation loss with best weights restored, ReduceLROnPlateau).
2. Eligible only if the worst seed has validation macro F1 0.85 or more.
3. Winner: highest mean validation macro F1. Another eligible configuration whose mean is within one std of the top mean (the larger of the two stds) is a tie, and the lower std wins the tie.

## Evidence (validation, 144 images, 5 seeds each)

| configuration | macro F1 mean +- std | worst seed | rainy F1 mean +- std (min) | best epoch (mean) | epochs run (mean) | min per run | parameters |
|---|---|---|---|---|---|---|---|
| r0_bn_momentum_0.9 | 0.9403 +- 0.0046 | 0.9340 | 0.974 +- 0.019 (0.947) | 33.8 | 48.8 | 1.38 | 305,380 |
| **r0_patience30_min20** | **0.9368 +- 0.0041** | **0.9342** | 0.965 +- 0.014 (0.950) | 43.4 | 73.4 | 1.91 | 305,380 |
| r0_bn_momentum_0.8 | 0.9364 +- 0.0181 | 0.9099 | 0.971 +- 0.040 (0.927) | 39.6 | 54.6 | 1.50 | 305,380 |
| baseline (Section 3) | 0.9080 +- 0.0271 | 0.8716 | 0.963 +- 0.046 (0.889) | 7.6 | 22.6 | 0.29 | 8,408,644 |

Not eligible (worst seed below 0.85): R0 (0.429), r0_lr3e-4 (0.125), r0_warmup5 (0.125), r0_augment_light (0.233). Not candidates (3 seeds): the six ablation arms of Section 5 group C.

## Applying the rule

- Top mean: r0_bn_momentum_0.9, 0.9403 +- 0.0046.
- r0_patience30_min20 is 0.0035 below it, less than 0.0046, so it is a tie. r0_bn_momentum_0.8 is 0.0039 below, less than 0.0181, also a tie. The baseline is 0.0323 below, more than 0.0271, so it is not a tie.
- Lowest std among the tied configurations: r0_patience30_min20 (0.0041, against 0.0046 and 0.0181).
- **Selected: r0_patience30_min20.** It is R0 (batch norm momentum 0.99) with early stopping patience 30, and neither early stopping nor the best-weight restore may act on epochs 1 to 20.

The three R0 arms are equivalent on validation (all differences are below one std, rule 2 of the experiment rules), and the tie-break is decided by 0.0004 of std. The rule was fixed before the results were seen, so it stands. The playbook tie-break (lower std of rainy F1) gives the same answer: 0.014 against 0.019 and 0.040. Parameters and latency do not separate them (same network).

## Rejected options

| option | reason |
|---|---|
| r0_bn_momentum_0.9 | Tied on mean, lost the tie-break by 0.0004 std. It is about 30% cheaper to train (49 against 73 epochs) and removes the batch norm lag instead of waiting it out. It is the fallback if the final training of the selected config fails. |
| r0_bn_momentum_0.8 | Tied on mean, std 4 times larger, worst seed 0.910. |
| baseline | 0.032 below the top mean (more than one std), std 6 times larger, 27 times more parameters. |
| R0, r0_lr3e-4, r0_warmup5, r0_augment_light | Seeds collapse under batch norm lag (worst seed 0.125 to 0.429). |
| Ablation arms (group C) | 3 seeds, not candidates. None beats the baseline by more than one std. |

## Differences from the playbook Section 7 rule

- The playbook ranks by the mean of a stratified K-fold over train plus valid. K-fold was not run, so the decision uses 5 seeds on the fixed validation split. With 144 validation images (20 rainy), 0.004 differences are far below what this split can resolve.
- Still open for Section 7: the latency benchmark, and running the ONNX export and parity inside the Docker trainer image.

## Final training and test protocol (Section 7, Kaggle group `final`)

- `grids.final_config()` is the selected configuration (a test checks that its hash equals the Section 5 row and that `src.selection` still picks it).
- Train it with seeds 11, 22, 33, 44, 55 under the same protocol. Pick the seed by validation macro F1, then lower validation loss. Only that model is exported.
- Export `weather_cnn.h5` and `weather_cnn.onnx` (`src/final_model.py`). ONNX parity on all 144 validation images, Keras on CPU against ONNX Runtime on CPU: max absolute probability difference below 1e-4 and the same predicted class for every image. If parity fails, the run stops before the test set is loaded.
- Then the test set (134 images) is loaded once. `test_touch.json` is written as soon as it is loaded, with the SHA-256 of the ONNX file. A second evaluation is refused, and the local install refuses an ONNX file the test set did not see.
- Reported: accuracy, loss, macro precision, recall and F1, per-class metrics, ROC AUC, confusion matrix, bootstrap 95% confidence intervals (2000 draws), parity on the test images, and the gap to validation.
