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

From Section 5 on (user decision, 2026-11-25), `model.keras` is kept only for the best seed (highest validation macro F1, then lower validation loss) of the top 3 configurations of the section by mean validation macro F1 (ties by lower std), and is saved without optimizer state. After each run only the best seed of that configuration keeps its model. At the end of each group and of the version, everything outside the top 3 is deleted, and the log prints the folder size before the notebook zips it. The Section 3 zip was 556 MB because each baseline-family model with Adam state is 97 MB. Every run still keeps config, history, curves, metrics, confusion matrix and validation predictions, so any run can be retrained exactly from its config. This changes rule 6, which kept weights for every run.

Incident (2026-11-25, first Section 5 version): the version stopped in the test step before any training. 139 passed, 2 skipped, 1 failed: `test_save_without_optimizer_is_smaller_and_predicts_the_same`. Predictions matched, but the file without optimizer was 3,772,746 bytes against 3,773,671 with it. Cause, read in the Keras 3.13.2 source and reproduced locally on Keras 3.13.2 with the JAX backend (3,772,625 against 3,773,500 bytes): `keras.models.clone_model` compiles the clone again from the original's compile config (`models/cloning.py`). `compile_from_config` then builds a fresh Adam whose zero-filled slots are saved, so the clone file is just as large. `include_optimizer` is ignored for `.keras` files. Fix: save an uncompiled copy made with `keras.Model.from_config` plus `set_weights`. On the same check it gives 1,317,328 bytes (35%), loads uncompiled and predicts the same. The test now asserts equal predictions and only the weak invariant (size without optimizer at most the size with it), and prints both sizes. Worst-case zip: 3 baseline-family models of about 34 MB plus about 15 MB of other files, about 120 MB. It stays near 320 MB even if the optimizer state came back. The notebook does not set `USE_BACKUP`, so no checkpoint is left behind.

## Section 5 results (Kaggle T4, 2026-11-25, commit 32d0c7b)

One version: 144 tests passed, smoke, timing, groups A to D, 48 runs, 49 minutes of training, about 56 minutes in total. No run was cut by the budget guard. The zip held 13.1 MB with 3 kept models of 1.3 MB. Merged into `results.csv` (11 Section 3 rows unchanged, 48 Section 5 rows added). Validation only.

| config | seeds | macro F1 mean +- std | min | rainy F1 mean (min) | best epoch mean | collapsed seeds |
|---|---|---|---|---|---|---|
| r0_bn_momentum_0.9 | 5 | 0.940 +- 0.005 | 0.934 | 0.974 (0.947) | 33.8 | 0 |
| r0_patience30_min20 | 5 | 0.937 +- 0.004 | 0.934 | 0.965 (0.950) | 43.4 | 0 |
| r0_bn_momentum_0.8 | 5 | 0.936 +- 0.018 | 0.910 | 0.971 (0.927) | 39.6 | 0 |
| r0_warmup5 | 5 | 0.777 +- 0.365 | 0.125 | 0.771 (0.000) | 32.4 | 1 (seed 22) |
| r0_lr3e-4 | 5 | 0.767 +- 0.359 | 0.125 | 0.760 (0.000) | 30.8 | 1 (seed 22) |
| r0_augment_light | 5 | 0.481 +- 0.278 | 0.233 | 0.200 (0.000) | 15.6 | 4 (seeds 11, 22, 33, 44) |
| baseline_plus_augment | 3 | 0.905 +- 0.010 | 0.894 | 0.946 (0.889) | 13.7 | 0 |
| baseline_plus_class_weights | 3 | 0.900 +- 0.015 | 0.886 | 0.944 (0.927) | 6.3 | 0 |
| baseline_plus_l2 | 3 | 0.894 +- 0.011 | 0.882 | 0.940 (0.919) | 6.0 | 0 |
| baseline_plus_dropout | 3 | 0.891 +- 0.004 | 0.886 | 0.958 (0.947) | 5.0 | 0 |
| baseline_plus_gap | 3 | 0.860 +- 0.050 | 0.803 | 0.846 (0.667) | 66.0 | 0 |
| baseline_plus_batch_norm | 3 | 0.547 +- 0.090 | 0.443 | 0.359 (0.095) | 1.3 | 3 |

A collapsed seed is a run whose best epoch is 1 or 2 and that stopped at epoch 16 or 17.

- S5-A1 (momentum 0.9): right. All 5 seeds converge, 0.940 +- 0.005, worst 0.934, rainy F1 at least 0.947, best epochs 28 to 42, 1.4 min per run. Validation loss already follows training at epoch 2 (seed 22: 0.75 at momentum 0.9 against 1.40 at 0.99).
- S5-A2 (momentum 0.8): right. Same mean as A1 (difference 0.004, below one std), noisier (std 0.018, worst seed 0.910).
- S5-B1 (patience 30, no stop before epoch 21): right on convergence (5 of 5, 0.937 +- 0.004) and on cost (73 epochs, 1.9 min per run, against 49 and 1.4 for A1). Wrong on timing: best epochs 33 to 51, earlier than the 45 to 65 expected.
- S5-C1 (+ batch norm): right on direction, wrong on size. All 3 seeds collapse (best epoch 1 or 2), 0.547 +- 0.090. The lag is not milder with 2 batch norm layers.
- S5-C2 (+ dropout 0.3): no difference (0.891, 0.017 below the baseline, within its std 0.027). Wrong on later best epochs (5, against 7.6).
- S5-C3 (+ L2): right, no difference (0.894).
- S5-C4 (+ class weights): right, no difference (0.900).
- S5-C5 (+ medium augmentation): wrong. No gain (0.905, 0.003 below the baseline). Best epochs are later (13.7 against 7.6, about 1.8 times), less than the 2 to 4 times expected.
- S5-C6 (+ global average pooling): right on direction and on much longer training (best epochs 42 to 80), wrong on size. 0.860 +- 0.050 is 0.048 below the baseline, just inside one std (0.050), so by rule 2 not a clear drop. With 28k parameters it still reaches 0.89 on 2 of 3 seeds.
- S5-D1 (learning rate 3e-4): right, no fix. 1 of 5 seeds collapses (seed 22, macro F1 0.125: every validation image called cloudy). The 4 converged seeds give 0.928 +- 0.009.
- S5-D2 (5-epoch warmup): right, no fix. Seed 22 collapses the same way. The 4 converged seeds give 0.940 +- 0.013.
- S5-D3 (light augmentation): right, no fix. 4 of 5 seeds collapse. The converged seed (55) gives 0.956, the best single run of the section.
- Collapse mechanism, checked in `history.csv` for all 12 collapsed runs of Sections 3 and 5: training accuracy rises smoothly to 0.91 to 1.00 by the stop. Validation loss starts at 0.96 to 1.63, peaks at 2.6 to 6.0 between epochs 5 and 11, and is still above its epoch 1 or 2 value at epoch 16 or 17. So early stopping fires and restores the epoch 1 or 2 weights. Those weights score 0.26 to 0.66 in inference mode on their own training images. At the stop, validation accuracy was 0.53 to 0.72, at or above its epoch 1 to 2 value in 11 of 12 runs, so the runs were recovering when they were stopped. It is the S3-E4 race again.
- Every collapse in Sections 3 and 5 is at batch norm momentum 0.99 with patience 15 and no minimum epoch count: 12 of 23 such runs. At momentum 0.9 or 0.8: 0 of 10. At momentum 0.99 with patience 30 and no stop before epoch 21: 0 of 5. Seed 22 collapsed in all four R0-type arms at momentum 0.99 (R0, D1, D2, D3), and seed 55 converged in all four. Which seed collapses depends on the seed, not only on the arm.
- Converged seeds land at the same level whatever the arm (0.93 to 0.96). Learning rate, warmup and augmentation strength do not change the converged level by more than one std.
- Conclusion: the batch norm moving statistics lag (momentum 0.99, 23 steps per epoch) is the cause of every R0 failure. Two independent fixes work: a lower momentum, or a stopping rule that cannot fire during the lag. Converged R0 (0.937 to 0.940) beats the baseline (0.908) by more than one baseline std, with 27 times fewer parameters. No single R0 technique added to the baseline improves it by more than one std. Batch norm alone breaks it at momentum 0.99.
- Selection (rule fixed before running): `r0_patience30_min20`. It ties with momentum 0.9 and 0.8 on mean and has the lowest std. Details and rejected options: `DECISION.md`.

## S7-E1: Final model of r0_patience30_min20 on the locked test set (hypothesis written 2026-11-25, before the test set is used)

- Plan: Section 7 group `final` on Kaggle. Train the selected configuration with 5 seeds under the same protocol, pick the seed by validation, export H5 and ONNX, check ONNX parity on all 144 validation images before the test set is loaded, then evaluate the 134 test images once (`DECISION.md`).
- Hypothesis: validation macro F1 of the chosen seed about 0.94 (the Section 5 seeds gave 0.934 to 0.944). Test macro F1 between 0.90 and 0.95, a little below validation, because the seed is picked on validation (an optimistic pick of about 0.005) and the test set has a different class mix (40 shiny images; shiny is the weakest class at validation F1 0.887). The bootstrap 95% interval about +-0.04 wide. Rainy test F1 0.90 or more, with an interval about +-0.07 wide (29 images). A test score more than 0.05 below validation would point to over-tuning on validation or to a shift between the splits, and would be investigated before export.
- ONNX parity: maximum probability difference near 1e-6 (Keras on CPU against ONNX Runtime on CPU). Section 5 showed that GPU against CPU alone moves the probabilities by 1.5e-4, so parity is measured on CPU only.
- Incident (2026-11-25, first Section 7 version): the version stopped in the test step before any training, so the locked test set was not loaded (the failing test uses a synthetic dataset). `test_final_export_writes_h5_and_onnx_with_parity` failed in `keras_probabilities` with "Trying to access resource conv2d_88/kernel located in device GPU:0 from device CPU:0". Cause: the run model was loaded on the default device (GPU:0), then `predict` ran under a CPU scope, so the CPU kernels had to read variables that live on the GPU. Fix: every Keras parity check (validation, test, and after install when TensorFlow is present) reloads the exported `weather_cnn.h5` inside `tf.device("/CPU:0")`, so its variables are created on the CPU, and predicts inside the same scope. Parity is now also measured on the shipped H5 file rather than the in-memory model. A mocked test checks the order (enter CPU scope, load H5, predict, leave) for every check, and a TensorFlow test builds a model under `/GPU:0` and runs the CPU parity on it. The real reload path gives a difference of 0.0 against the original model on Keras 3.13.2 with JAX locally; the GPU case itself can only be checked on Kaggle.

### S7-E1 results (Kaggle T4, 2026-11-25, commit 2b96356, installed locally 2026-11-25)

- Runs: 5 seeds of `r0_patience30_min20` (config hash d2450b51, the same as Section 5), 71 to 91 epochs, best epochs 41 to 61, about 2 minutes each. Validation macro F1 by seed: 11: 0.9526, 22: 0.9438, 33: 0.9406, 44: 0.9467, 55: 0.9619. Mean 0.9491 +- 0.0084, worst 0.9406.
- Selected model: seed 55 (`s7_r0_patience30_min20_seed55`), the best validation macro F1 of the 5. Exported with `tf2onnx.convert.from_keras`.
- Validation (144 images, seed 55): macro F1 0.962, accuracy 0.965, macro precision 0.966, macro recall 0.959, log loss 0.144, ROC AUC 0.998. Per-class F1: cloudy 0.980, rainy 0.950, shiny 0.952, sunrise 0.966.
- Test (134 images, evaluated once, bootstrap 95% CI from 2000 draws): macro F1 0.900 (0.845 to 0.947), accuracy 0.903 (0.851 to 0.948), macro precision 0.900, macro recall 0.907, log loss 0.303, ROC AUC 0.982. Per-class F1: cloudy 0.820 (0.696 to 0.912), rainy 0.918 (0.833 to 0.983), shiny 0.877 (0.790 to 0.950), sunrise 0.986 (0.953 to 1.000). Rainy recall 0.966 (28 of 29), rainy precision 0.875.
- Test confusion (rows true, columns predicted, order cloudy, rainy, shiny, sunrise): cloudy 25, 2, 1, 1; rainy 1, 28, 0, 0; shiny 6, 2, 32, 0; sunrise 0, 0, 0, 36. 13 errors, 8 of them shiny images (6 called cloudy, 2 called rainy).
- ONNX parity (Keras on CPU from the reloaded H5 against ONNX Runtime on CPU): validation 144 images, max difference 5.1e-7; test 134 images, max difference 1.4e-6; same predicted class for every image. Both passed (tolerance 1e-4).
- Install check (local, 2026-11-25): the ONNX file matches the sha256 recorded when the test set was evaluated, the record has validation and test keys and passed parity, and ONNX Runtime on the local `data/valid` reproduces the recorded validation macro F1 (0.96187, difference below 1e-4). The Keras parity after install was skipped because TensorFlow is not installed locally.
- Validation is optimistic: seed 55 was picked on validation as the best of 5 seeds, so its 0.962 is the top of 5 draws. The pick alone adds about 0.013 over the 5-seed mean (0.949). The test number is the estimate to report.

Hypothesis check:
- Validation macro F1 of the chosen seed about 0.94: wrong, it was 0.962. All 5 Section 7 seeds scored above their Section 5 runs (mean +0.012, see the surprise below).
- Test macro F1 between 0.90 and 0.95: right, but only just (0.9002).
- Bootstrap interval about +-0.04: a little wider, -0.055 to +0.047.
- Rainy test F1 0.90 or more with an interval about +-0.07: right (0.918, 0.833 to 0.983).
- Shiny weakest: partly right. Shiny has the lowest test recall (0.80), and its errors make cloudy the class with the lowest F1 (precision 0.78).
- ONNX parity near 1e-6: right.
- "A test score more than 0.05 below validation would be investigated": test is 0.062 below the chosen seed, so it was investigated, from the saved test predictions only (nothing was evaluated again):
  - Against the 5-seed validation mean the gap is 0.049, and against the 10-run mean over Sections 5 and 7 (0.943 +- 0.009) it is 0.043. The test interval (0.845 to 0.947) contains 0.943.
  - The errors repeat the validation pattern: shiny predicted cloudy is the largest error on both splits (2 of 33 on validation, 6 of 40 on test), the ambiguity of white clouds on blue sky seen in the Section 5 gallery and the Section 1 label review. The test split has more shiny images than validation (40 against 33).
  - Wrong test predictions are much less confident than right ones (median 0.76 against 0.996), and 5 of the 13 are below 0.6, which the API flags as uncertain. Two are confident errors: `cloudy75` called sunrise at 0.999 and `rain72` called cloudy at 0.997. Their images were not opened.
  - Leakage would push test up, not down, and cross-split duplicates were removed in Section 1.
- Conclusion: the gap is explained by the optimistic seed pick, the small test set (an interval about 0.10 wide) and the cloudy and shiny ambiguity. No decision changes, and the test set is not used again.

Surprise: Section 7 ran the same configuration hash, seeds, TensorFlow 2.21.0 and T4 GPU as Section 5, but every seed scored higher (11: +0.016, 22: +0.010, 33: +0.007, 44: +0.012, 55: +0.018). The epoch-1 validation loss agrees to about 3 decimals (seed 55: 1.1643 against 1.1645), then the runs drift apart. GPU kernels are not deterministic, so each run is a new draw, but all 5 going the same way happens 1 time in 32 by chance. Not explained; logged as an open question in FINDINGS. It does not affect the selection, which was fixed before Section 7.

## Test set touches

| date | section | what | by |
|---|---|---|---|
| 2026-11-22 | 1 | Data audit only: file properties, duplicate detection, quarantine of 31 duplicate test copies. No model was evaluated. | Claude Code |
| 2026-11-24 | 2 | File counts per class only, for `class_distribution.png`. No image features, no model. | Claude Code |
| 2026-11-25 | 7 | Final model `s7_r0_patience30_min20_seed55` evaluated once on all 134 test images inside `src/final_model.py` on Kaggle (12:19:10 UTC, guard file `experiments/final_model/test_touch.json`, ONNX sha256 31892c5c...). Metrics, bootstrap CI and ONNX parity written to `metrics.json`. The smoke run before it used the valid split as its stand-in test split, not the test set. | Kaggle job |
| 2026-11-25 | 7 | Local install: read the saved test record and predictions to write up S7-E1. No model was run on test images and no image was opened. | Claude Code |
