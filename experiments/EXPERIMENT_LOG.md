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

## S3-E1: Can the pipeline overfit one batch? (2026-11-24)

- Change: R0 architecture without augmentation, dropout, L2 and class weights, trained for 300 steps on one stratified batch of 32 train images (8 per class, seed 11).
- Hypothesis: train accuracy reaches 100% within about 50 steps, and accuracy in inference mode (batch norm moving statistics) on the same batch is at least 97% after 300 steps. If not, the labels, preprocessing or batch norm handling has a bug.
- Result: pending (Colab, group A).

## S3-E2: Does a shuffled-label run stay at chance? (2026-11-24)

- Change: R0 with train labels randomly permuted (seed 11). Validation labels stay true. Full protocol (max 150 epochs, early stopping patience 15).
- Hypothesis: validation macro F1 stays inside the random-guess range (0.24 +- 0.07) and accuracy at or below about 0.33. Validation loss never improves much, so early stopping fires around epoch 16 to 25. A clearly higher score would mean leakage between train labels and validation.
- Result: pending (Colab, group A).

## S3-E3: Vanilla CNN baseline, 5 seeds (2026-11-24)

- Change: reference point, no change. Two conv-pool layers (32, 64 filters), flatten, dense 128, no batch norm, dropout, L2, augmentation or class weights. Same training protocol as R0 (Adam 1e-3, early stopping, ReduceLROnPlateau). About 8.4M parameters, almost all in the dense layer after flatten.
- Hypothesis: it overfits fast. Clean train accuracy close to 100%, best epoch early (about 5 to 15), validation macro F1 about 0.80 +- 0.03. Rainy has the lowest F1 (about 0.70) because rainy vs cloudy is the hardest pair (Section 2: only sharpness separates them) and rainy has only 20 validation images.
- Result: pending (Colab, group B).

## S3-E4: R0 against the vanilla CNN, 5 seeds (2026-11-24)

- Change: R0 vs S3-E3, all optimizations together: 3 blocks of two convs (32, 64, 128 filters), batch norm, dropout 0.3, L2 1e-4, global average pooling, class weights, medium augmentation. About 0.3M parameters.
- Hypothesis: R0 is better by 0.05 to 0.10 macro F1 (about 0.87 to 0.90), with a lower std across seeds, a later best epoch (about 40 to 80) and a smaller gap between clean train and validation accuracy. Rainy F1 gains the most, from class weights and augmentation. This is a bundle, so it says how much the optimizations buy together, not which one matters. Section 5 ablations separate them.
- Result: pending (Colab, group C).

## Test set touches

| date | section | what | by |
|---|---|---|---|
| 2026-11-22 | 1 | Data audit only: file properties, duplicate detection, quarantine of 31 duplicate test copies. No model was evaluated. | Claude Code |
| 2026-11-24 | 2 | File counts per class only, for `class_distribution.png`. No image features, no model. | Claude Code |
