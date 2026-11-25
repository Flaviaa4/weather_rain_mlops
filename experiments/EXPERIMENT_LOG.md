# Experiment log

Format per entry: id, date, section, what changes, hypothesis (direction, size, why), then result, right or wrong, conclusion.

## S1-E1: Is dHash Hamming 4 or less enough to find duplicates? (2026-11-22)

- Change: none, this is a data check.
- Hypothesis: dHash at Hamming 4 or less finds near-duplicates with few false matches, because Roboflow exports often contain re-encoded copies of the same source photo. Expected a few dozen pairs, mostly same label.
- Result: 619 pairs, 151 with different labels. Visual review: every cross-label pair was a false match (dark scene with a bright horizontal band, for example rain splashes vs sunsets). Color thumbnail distance separates them: true duplicates below 0.07, false matches above 0.12, with nothing between 0.02 and 0.03.
- Hypothesis: wrong on false matches. Simple sky and horizon structures collide in a 64-bit structural hash.
- Conclusion: confirm hash candidates with pixel checks (color thumbnail MAE and grayscale correlation) and record visually rejected pairs in `experiments/audit/reviewed_not_duplicates.json`.

## S1-E2: Does the hash miss duplicates? (2026-11-22)

- Change: add a hash-independent scan (grayscale thumbnail correlation of 0.93 or more over all pairs).
- Hypothesis: a few crops or tone-edited copies have a Hamming distance above 4 and are missed. Expected fewer than 10.
- Result: 6 more true duplicates (Hamming 5 to 15), 3 of them across splits (shine94 train = shine16 test, shine229 train = shine136 valid, shine190 train = shine213 valid). 1 false match passed the numbers (sunrise92 vs cloudy263) and was rejected after review.
- Hypothesis: right.
- Conclusion: the audit uses both stages. Final result: 114 confirmed pairs in 90 groups, 102 files quarantined, 0 confirmed duplicates left.

## S1-E3: Colab environment check (2026-11-24)

- Change: none, this is an environment check (`colab/run_section.ipynb`, SECTION = 1).
- Hypothesis: Colab's preinstalled TensorFlow and Keras export to ONNX with `tf2onnx.convert.from_keras`, and a T4 GPU is visible.
- Result: Python 3.13.18, TensorFlow 2.21.0, Keras 3.13.2, tf2onnx 1.17.0, onnx 1.23.2, onnxruntime 1.32.0. `.h5` save passed. `from_keras` round trip passed (max abs difference 4.5e-08). No GPU was visible: the preinstalled torch builds are `+cpu`, so the session was a CPU runtime.
- Hypothesis: right on export, wrong on GPU (runtime setting, not a software problem).
- Conclusion: pin requirements to these versions. Confirm the GPU before the first training run in Section 3. `check_env` now warns when no GPU is visible.

## S2-E1: How much information survives at 96, 128 and 160 pixels? (2026-11-24)

- Change: input size 96, 128, 160 against a 224 reference. No model, only image features.
- Hypothesis: color statistics (brightness, saturation, blue-to-red ratio) are almost size independent (correlation with 224 above 0.99 at every size). Fine detail is not: reconstruction error grows clearly at 96, and rainy loses the most because rain streaks and splashes are thin. The sharpness gap between rainy and the rest (a possible source-quality shortcut) shrinks at smaller sizes, because downscaling erases the extra detail of the large rainy photos.
- Result (863 train and valid images, `experiments/eda/size_*.csv`): color features correlate 0.998 to 1.000 with the 224 reference at every size. Sharpness correlates 0.83 (96), 0.90 (128), 0.96 (160). Median reconstruction PSNR at 96 / 128 / 160: rainy 28.7 / 30.8 / 33.1 dB, shiny 31.9 / 34.6 / 37.3, sunrise 34.2 / 37.0 / 39.9, cloudy 36.3 / 39.3 / 42.3. Rainy-vs-rest sharpness AUC: 0.79 (96), 0.83 (128), 0.85 (160), 0.88 (224).
- Surprise 1: within images of 0.06 MP or less, rainy-vs-cloudy sharpness AUC is still 0.93. The texture gap is not caused by the larger rainy source photos. It is in the content (rain streaks, splashes, busy street scenes).
- Surprise 2: the share of images whose height is below the target size (so they get upscaled) is 0 to 1.4% at 128 for every class, 12 to 16% for cloudy, shiny and sunrise but 0% for rainy at 160, and 72% for cloudy vs 35% for rainy at 224. Above 128, "upscaled and blurry" starts to mean "not rainy", which is a resolution shortcut the model could learn.
- Hypothesis: right on color, right on rainy losing the most detail, right on the gap shrinking at smaller sizes. Wrong on the cause: resolution is not what makes rainy sharper.
- Conclusion: keep 128 px for R0. It is the largest size where almost nothing is upscaled, and it keeps 90% of the sharpness ranking. 160 keeps more rain detail but adds a resolution difference between classes, so if 160 enters the Section 5 grid it must be judged together with the Section 6 downscale-then-upscale test.

## S2-E2: Does any augmentation strength destroy the label? (2026-11-24)

- Change: augmentation strength light, medium, strong, 8 samples of one image per class.
- Hypothesis: light and medium keep every label. Strong brightness and hue changes make some sunrise images look like cloudy or shiny (a darkened or color-shifted sunset), because sunrise is defined by warm color at the horizon.
- Run: Colab CPU runtime, 2026-11-24, `experiments/colab_section2.log`. Previews `experiments/figures/augmentation_preview_{light,medium,strong}.png`, 6 rows (one random train image per class, the least saturated sunrise, the brightest cloudy), 8 augmented copies each, seed 11. Parameters at preview time: light hue 0, medium hue 0.02 and saturation 0.2, strong hue 0.05 and saturation 0.35.
- Result (reviewed by the user and by Claude Code): light and medium keep every label. Strong does not: sunrise copies turn washed-out pink-gray (saturation 0.35) or nearly black (brightness 0.2 with contrast 0.35), the brightest cloudy image goes almost white and loses cloud detail, and the shiny sea gets a teal cast from the hue shift.
- Hypothesis: right that light and medium are safe and that strong breaks sunrise. Partly wrong on the cause: saturation and brightness do more damage than hue, and hue also hurts shiny.
- Conclusion (user decision): medium is the R0 default. Hue is 0 at every level. Strong stays only as an experiment arm, with saturation capped at 0.2. These changes only make augmentation milder, so the medium review still holds. The new strong setting has not been previewed: rerun the previews before strong is used in a grid.

## Section 3 reference numbers (2026-11-24, local, labels only)

Validation set, `experiments/sanity/reference_baselines.json`. Majority class (always sunrise): accuracy 0.299, macro F1 0.115, rainy F1 0. Random guess with the train class prior (1000 draws): accuracy 0.260 +- 0.036, macro F1 0.243 +- 0.035, rainy F1 0.160 +- 0.068. Uniform random: accuracy 0.248, macro F1 0.240 +- 0.035.

## Section 3 compute platform change (2026-11-24)

Colab gave no GPU. On the Colab CPU the timing step measured R0 at 124 s per epoch, which puts group C alone at about 10 hours, so CPU training is not workable. The Section 3 grid moves to Kaggle (GPU T4, one device via CUDA_VISIBLE_DEVICES=0) with `kaggle/run_section.ipynb`. The launcher enforces the TensorFlow 2.21.0 and Keras 3.13.2 pins and records the Kaggle Python version, which may differ from 3.13. The rule 0B.4 budget is applied to Kaggle GPU hours: COLAB_UNITS_PER_HOUR=1 and COLAB_UNITS_REMAINING set to the weekly GPU hours left. No result was produced on the CPU, so nothing is discarded.

## S3-E1: Can the pipeline overfit one batch? (2026-11-24)

- Change: R0 architecture without augmentation, dropout, L2 and class weights, trained for 300 steps on one stratified batch of 32 train images (8 per class, seed 11).
- Hypothesis: train accuracy reaches 100% within about 50 steps, and accuracy in inference mode (batch norm moving statistics) on the same batch is at least 97% after 300 steps. If not, the labels, preprocessing or batch norm handling has a bug.
- Result (Kaggle T4, 2026-11-24, first attempt): failed. After 300 steps the training-mode accuracy was 1.0 (loss 1.2e-05), but inference-mode accuracy on the same 32 images was 0.406 (loss 5.86). In smoke (20 steps) it was 0.25. Also R0 smoke validation macro F1 was 0.100 (rainy 0.000) after 2 epochs, against 0.281 for the baseline. Setup, 61 tests and smoke passed. Timing: R0 1.33 s per epoch, baseline 0.55 s per epoch.
- Hypothesis right or wrong: wrong as stated. The check was not measuring what it should. See S3-E1b. The check was not weakened or skipped.

## S3-E1b: Why does inference mode disagree with training mode on the overfit batch? (2026-11-24)

- Change: diagnostic group `diag` (`src/diagnostics.py`), same 32 images (8 per class, seed 11), 300 steps each, for six variants: baseline, R0, R0 without batch norm, R0 with dropout 0 and no augmentation, R0 with batch norm momentum 0.9, and the exact check config. For each: accuracy and loss in training mode and in inference mode, inference through `model.evaluate`, `model.predict` and a manual `model(x, training=False)`, then inference again after refreshing the batch norm moving statistics on the batch (one forward pass in training mode with momentum 0, no weight update). Also an input check on the same files: the eval dataset equals the inference path exactly, the training dataset (augmentation off) has the same tensors in another order, float32, 128x128x3, range [0, 1], RGB channel order checked against an independent TensorFlow JPEG decode.
- Hypothesis: cause (a), batch norm moving statistics. The check config has no augmentation or dropout and `fit` and `evaluate` get the same arrays, so only batch norm differs between modes. Keras updates moving statistics with momentum 0.99: after 20 steps 0.99^20 = 82% of each moving statistic is still its initial value (mean 0, variance 1), which matches the chance-level 0.25 in smoke. After 300 steps the initial share is 4.9%, but the weights keep changing while the loss is pushed to 1e-5, so the moving statistics lag behind the current batch statistics, and saturated logits amplify the small shift (0.406). The R0 smoke F1 of 0.100 fits the same story: 2 epochs of one step each are 2 batch norm updates. Expected: the check config and R0 have a large gap that disappears after the refresh, the no-batch-norm variant and the baseline have no gap, momentum 0.9 shrinks the gap, the three inference paths agree, and the input check passes. If the gap survives the refresh or the paths disagree, it is (c), a real eval-path bug.
- Consequence for the check: refresh the batch norm statistics on the batch before the inference-mode test, keep the stale-statistics accuracy in the JSON for the record, and pass only if training accuracy and refreshed inference accuracy are both at least 0.97. The refreshed test still fails on any real eval-path or label bug, because then the two modes compute different functions. Full training runs are not affected in the same way: 23 steps per epoch over dozens of epochs give hundreds of batch norm updates, while the weights change slowly near the best epoch.
- Result (Kaggle T4, 2026-11-24, `experiments/sanity/diag_overfit_one_batch.json`): cause (a). Input check passed: float32, 128x128x3, range [0, 1], eval dataset equal to the inference path, training dataset equal up to order, RGB channel-mean gap 0.008 against 0.657 if the channels were swapped. The three inference paths agree in every variant. After 300 steps every variant has training-mode accuracy 1.0. Inference-mode accuracy: baseline 1.0, R0 0.375, R0 without batch norm 1.0, R0 with dropout 0 and no augmentation 0.375, R0 with momentum 0.9 1.0, check config 0.406. After the batch norm refresh every variant is at 1.0. The largest moving-mean gap before the refresh was 0.48 std (R0) against 0.09 std at momentum 0.9.
- Hypothesis: right. Only batch norm differs between modes, and momentum 0.9 removes the gap on its own.
- Side note: the smoke diag (3 steps) printed verdict (c), because 3 steps cannot fit 32 images in training mode. That is expected in smoke. The full diag decides.
- Conclusion: the overfit check with refreshed statistics is valid (passed, inference accuracy 1.0, stale accuracy 0.406 kept in the JSON). Batch norm lag is real for this model and is the lead suspect for the R0 stall in S3-E4.

## S3-E2: Does a shuffled-label run stay at chance? (2026-11-24)

- Change: R0 with train labels randomly permuted (seed 11). Validation labels stay true. Full protocol (max 150 epochs, early stopping patience 15).
- Hypothesis: validation macro F1 stays inside the random-guess range (0.24 +- 0.07) and accuracy at or below about 0.33. Validation loss never improves much, so early stopping fires around epoch 16 to 25. A clearly higher score would mean leakage between train labels and validation.
- Result (Kaggle T4): validation macro F1 0.381, accuracy 0.479, rainy F1 0.000, 44 epochs, best epoch 29. Train accuracy stayed at 0.24 to 0.34 the whole run (train loss 1.40 to 1.61, near log 4 = 1.39), so the model did not memorize the noise labels. Validation accuracy climbed steadily from 0.10 (epoch 7) to 0.51 (epoch 33). Confusion: cloudy mostly predicted cloudy (37 of 48), shiny mostly predicted cloudy (17 of 33), rainy split between shiny and cloudy, sunrise predicted sunrise with precision 0.93.
- Chance band (`python -m src.chance_band`, `experiments/sanity/chance_band.json`, 20000 draws): random predictions with the train class prior on the 144 validation images give macro F1 0.245 +- 0.036, 95% band 0.177 to 0.318, 99.9% quantile 0.362. Best of 44 draws (to mimic picking the best epoch; an upper bound, because consecutive epochs are correlated and early stopping picks by loss, not F1): 0.327 +- 0.018, 95% band 0.296 to 0.369, 99.9% quantile 0.400. Share of simulations at or above 0.381: 0.04% (single draw), 0.94% (best of 44). So 0.381 is outside both random-prediction bands.
- Why, without leakage: the permutation is one fixed draw, and with 719 images it keeps some chance correlation with the true classes. Class-weighted, the share of each shuffled label inside each true class is 0.21 to 0.31 instead of 0.25. True sunrise images carry the shuffled label sunrise most often (0.285). Cloudy and shiny images taken together carry cloudy most often. Rainy images carry shiny most often (0.277). A regularized model that cannot memorize learns only this generalizable part: visual cluster to the majority shuffled label. The rule "cloudy and shiny to cloudy, rainy to shiny, sunrise to sunrise" predicts macro F1 0.436 on validation, and the observed confusion follows that rule. The cloudy and shiny merge was chosen after seeing the confusion, so it is an explanation, not a test. Across 2000 fresh permutations, a model that separates the four classes perfectly and outputs each cluster's weighted-majority shuffled label scores 0.22 +- 0.21 (95% band 0 to 0.69), and 24% of permutations give 0.381 or more. Other leakage checks are clean: duplicates were removed in Section 1, the shuffle touches train labels only, and validation labels are never shuffled.
- Hypothesis: wrong. A shuffled-label control with one permutation and a strongly regularized model is not a chance-level control. Its expected score depends on the permutation and ranges from 0 to about 0.7.
- Conclusion: no evidence of leakage. The useful signal from this control is that train accuracy stays at the prior level. If the control is repeated, it needs several permutations and must be judged against the permutation null, not the random-guess band.

## S3-E3: Vanilla CNN baseline, 5 seeds (2026-11-24)

- Change: reference point, no change. Two conv-pool layers (32, 64 filters), flatten, dense 128, no batch norm, dropout, L2, augmentation or class weights. Same training protocol as R0 (Adam 1e-3, early stopping, ReduceLROnPlateau). About 8.4M parameters, almost all in the dense layer after flatten.
- Hypothesis: it overfits fast. Clean train accuracy close to 100%, best epoch early (about 5 to 15), validation macro F1 about 0.80 +- 0.03. Rainy has the lowest F1 (about 0.70) because rainy vs cloudy is the hardest pair (Section 2: only sharpness separates them) and rainy has only 20 validation images.
- Result (Kaggle T4, 5 seeds): macro F1 0.908 +- 0.027 (min 0.872, max 0.943), rainy F1 0.963 +- 0.046 (min 0.889), accuracy 0.904. Best epoch 3 to 11, runs of 18 to 26 epochs, 0.29 min per run. Clean train accuracy at the best epoch 0.953. Per-class F1: cloudy 0.880, rainy 0.963, shiny 0.822, sunrise 0.967. Main confusion over all seeds: cloudy and shiny (27 cloudy called shiny, 21 shiny called cloudy, out of 720 predictions). Only 6 rainy images were called cloudy.
- Hypothesis: right on fast overfitting and the early best epoch. Wrong on the level (0.908, not 0.80) and on rainy: rainy is the second best class. The hard pair for this model is cloudy vs shiny (fair-weather clouds on blue sky, see the Section 1 label review), not rainy vs cloudy. Rainy F1 on 20 images still moves by up to 0.11 between seeds.
- Conclusion: the vanilla CNN is a strong reference. Any optimized model has to beat 0.908 +- 0.027 by more than one std to count as better.

## S3-E4: R0 against the vanilla CNN, 5 seeds (2026-11-24)

- Change: R0 vs S3-E3, all optimizations together: 3 blocks of two convs (32, 64, 128 filters), batch norm, dropout 0.3, L2 1e-4, global average pooling, class weights, medium augmentation. About 0.3M parameters.
- Hypothesis: R0 is better by 0.05 to 0.10 macro F1 (about 0.87 to 0.90), with a lower std across seeds, a later best epoch (about 40 to 80) and a smaller gap between clean train and validation accuracy. Rainy F1 gains the most, from class weights and augmentation. This is a bundle, so it says how much the optimizations buy together, not which one matters. Section 5 ablations separate them.
- Result (Kaggle T4, 5 seeds): macro F1 0.659 +- 0.260 (min 0.429, max 0.941), rainy F1 0.390 +- 0.534. Bimodal. Seeds 33 and 55 converged (0.941 both, rainy F1 0.976 and 0.974, best epochs 49 and 62 of 64 and 77). Seeds 11, 22 and 44 stopped after 16 epochs with best epoch 1 (0.451, 0.536, 0.429, rainy F1 0.000; 55 of 60 rainy predictions were cloudy). 1.29 s per epoch, 1.1 min per run.
- Hypothesis: wrong. R0 is not better on average, and its std is ten times larger. The two converged seeds are 0.033 above the baseline mean, slightly more than one baseline std, with fewer cloudy and shiny confusions.
- Why it stalls (from `history.csv` and `curves.png`, no new run): training is not stuck. In every seed, training-mode accuracy climbs smoothly from 0.66 to 0.70 at epoch 1 to 0.91 to 0.93 at epoch 16, with no plateau or divergence, so learning rate, class weights and augmentation are not the problem. What breaks is the validation measurement in inference mode. Validation loss is 1.16 to 1.32 at epoch 1 (near-uniform outputs: after one epoch of 23 steps, 0.99^23 = 79% of each moving statistic is still its initial value), then jumps to 2.3 to 4.5 between epochs 6 and 10, then falls again. The initial share is 31% after 5 epochs, 10% after 10, 3% after 15 and 1% after 20, which matches when validation loss recovers. With the restored epoch-1 weights, the failed seeds score 0.49 to 0.59 in inference mode on their own training images, against 0.66 to 0.70 training-mode accuracy at that epoch. The outcome is a race against early stopping. Seeds 33 and 55 happened to set their best validation loss at epoch 2 (1.299 and 0.987), which moved the stop deadline to epoch 17, and validation loss beat that value at epochs 16 and 14. Seeds 11, 22 and 44 had their best at epoch 1 and were still at 1.34 to 1.59 at epoch 16, against 1.16 to 1.25, while their validation accuracy was rising (0.53 to 0.67). ReduceLROnPlateau also reacts to the inflated validation loss and halves the rate at epochs 7 and 12 in every seed.
- Conclusion: the R0 failure is a measurement and stopping problem caused by batch norm moving statistics lagging over only 23 steps per epoch, not a training problem. Candidate fixes for Section 5: a lower momentum (the moving statistics then follow the weights within about one epoch) or a stopping rule that cannot fire during the lag. Learning rate, warmup and augmentation are not expected to fix it.

## Section 3 status (2026-11-25)

Section 3 ran on Kaggle in 19 minutes (one T4): setup and tests, smoke, timing, diag, groups A, B and C. 11 runs in `results.csv`. GPU runs are not bit-for-bit reproducible even with fixed seeds, so every comparison uses 5-seed means. The test set was not used.

## Section 5 plan: fix the R0 stall, then measure each optimization (hypotheses written 2026-11-25, before any run)

This is the first Section 5 Kaggle version. The playbook grid (sensitivity, imbalance, interactions, K-fold, 10 seeds) needs a working R0 first, so it comes after this version.

### Selection rule (fixed before running)

- Candidates: configurations with 5 seeds under the full protocol: the Section 3 baseline and R0, and the six Section 5 R0 arms. The ablation arms have 3 seeds and are not candidates. An ablation arm that looks best gets 2 more seeds before it can be chosen.
- Eligible only if the worst seed has validation macro F1 0.85 or more.
- Winner: highest mean validation macro F1. If another eligible configuration is within one std of the top mean (the larger of the two stds), that is a tie (rule 2), and the lower std wins.
- Implemented in `src/selection.py` (`python -m src.selection --section 3 --section 5`). The winner becomes the reference R1 for the rest of Section 5. Nothing here uses the test set.
- Today the rule picks the baseline (0.908 +- 0.027, worst 0.872). R0 is not eligible (worst seed 0.429).

### Budget and order

- One Kaggle version, GPU time at most 60 minutes. Measured speeds: R0 family 1.29 s per epoch, baseline family 0.51 s per epoch, plus about 17 s (R0) and 8 s (baseline) per run for building, the first epoch and saving.
- Order by value: A (momentum), B (stopping rule), C (ablation), D (learning rate, warmup, light augmentation). Within a group, all seeds of one configuration run before the next configuration.
- Expected minutes: A 15, B 11, C 9, D 17 to 27 (depends on how many seeds stall), smoke and timing 2, setup and tests about 5. Total about 59 to 69.
- Guard: `SECTION_BUDGET_MINUTES[5] = 50` in `src/grids.py`. No new run starts after 50 minutes of the experiment process. A run that already started finishes (worst case about 3.5 more minutes), so the version ends near 59 minutes at most. Runs that did not start are listed in the log and run in a second version with this version's output attached. No epochs are cut.
- Dropped to fit: the ablation uses 3 seeds (11, 22, 33) instead of 5. It is there to show what each technique does in the notebook, not to pick a model. The Section 3 baseline and R0 runs are reused as references, not rerun.

### S5-A1: R0 with batch norm momentum 0.9 (5 seeds)

- Change: batch norm momentum 0.99 to 0.9, nothing else.
- Hypothesis: the moving statistics keep 0.9^23 = 9% of their old value after one epoch instead of 79%, so validation loss tracks training from epoch 2 or 3, the early false minimum disappears, and all 5 seeds converge. Expected macro F1 0.93 to 0.95, std 0.02 or less, worst seed above 0.90, rainy F1 above 0.90 in every seed, best epoch about 25 to 50, about 1.5 min per run. Diag support: momentum 0.9 closed the gap on the 32-image batch.
- Risk: the moving statistics average only about 10 batches, so validation loss gets a little noisier.

### S5-A2: R0 with batch norm momentum 0.8 (5 seeds)

- Change: batch norm momentum 0.99 to 0.8.
- Hypothesis: same outcome as A1. The difference from A1 is smaller than one std (no difference). If anything, slightly noisier validation loss and a slightly earlier best epoch.

### S5-B1: R0 with patience 30 and at least 20 epochs (5 seeds)

- Change: the stopping rule only. Early stopping patience 15 to 30, and both early stopping and best-weight restore ignore epochs 1 to 20. This is one factor, "when may the run stop", with two settings.
- Hypothesis: every seed survives the lag phase (the failed seeds were still improving at epoch 16), so all 5 converge to about 0.93 to 0.94 with the worst seed above 0.90. Best epoch later than A1 (about 45 to 65) and runs about twice as long (85 epochs on average), because the learning rate is still halved during the lag phase. Expected to tie with A1 on mean and lose on cost.

### S5-C1 to C6: ablation, add one R0 technique to the baseline (3 seeds each)

Reference: the Section 3 baseline (0.908 +- 0.027). The baseline is now built by the same `build_cnn` function as R0 (2 blocks of one conv, flatten, dense 128). It has the same layers and the same 8,408,644 parameters as in Section 3, and a test checks the count. Each arm takes the R0 value of one setting.

- C1 + batch norm (momentum 0.99): the same lag appears, but with only 2 batch norm layers and a fast-overfitting flatten head it is milder. Expected lower mean and higher std than the baseline, with some seeds stopping early.
- C2 + dropout 0.3: no difference or a small gain (within one std), best epoch a few epochs later.
- C3 + L2 1e-4: no difference (within one std). 8.4M flatten weights are hard to regularize with a small L2.
- C4 + class weights: no difference. Rainy is already 0.96 F1 in the baseline, so weights have little to fix.
- C5 + medium augmentation: the largest gain of the six, +0.01 to +0.03, with best epochs 2 to 4 times later. The baseline overfits by epoch 3 to 11 and augmentation slows that down.
- C6 + global average pooling: a clear drop (0.70 to 0.85). Pooling 64 channels after only two convs removes most spatial detail and leaves about 28k parameters, so the model underfits and trains much longer.

### S5-D1: R0 with learning rate 3e-4 (5 seeds)

- Hypothesis: does not fix the stall. Batch norm lag depends on the number of steps, not the learning rate, and training was never unstable. Some seeds still stop at epoch 16. Mean within one std of R0. Smaller steps make the weights drift more slowly, so a few more seeds may survive, but the std stays above 0.1.

### S5-D2: R0 with a 5-epoch linear warmup (5 seeds)

- Change: the learning rate rises linearly from 2e-4 to 1e-3 over epochs 1 to 5 (`warmup_epochs = 5`). During those epochs the warmup overrides ReduceLROnPlateau.
- Hypothesis: no fix. Training accuracy rose smoothly from epoch 1 in every seed, so there is no early instability for warmup to remove. Mean within one std of R0, still bimodal.

### S5-D3: R0 with light augmentation (5 seeds)

- Hypothesis: no fix for the stall, which is not caused by augmentation. Seeds that converge reach the same F1 as converged R0 within one std, maybe slightly lower on cloudy vs shiny because medium augmentation covers more lighting variation.

### Run files

From Section 5 on (user decision, 2026-11-25), `model.keras` is kept only for the best seed of each configuration (highest validation macro F1, then lower validation loss) and is saved without optimizer state. The Section 3 zip was 556 MB because each baseline-family model with Adam state is 97 MB. Every run still keeps config, history, curves, metrics, confusion matrix and validation predictions. This changes rule 6, which kept weights for every run.

## Test set touches

| date | section | what | by |
|---|---|---|---|
| 2026-11-22 | 1 | Data audit only: file properties, duplicate detection, quarantine of 31 duplicate test copies. No model was evaluated. | Claude Code |
| 2026-11-24 | 2 | File counts per class only, for `class_distribution.png`. No image features, no model. | Claude Code |
