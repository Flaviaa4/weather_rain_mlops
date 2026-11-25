# Findings

## Section 1: Environment, skeleton, data audit

### Environment

- Local: conda env at `.venv/` (Python 3.12.16). Installed and import-checked: numpy, pandas, Pillow, matplotlib, pytest, onnxruntime, FastAPI, uvicorn, Locust. See `experiments/audit/check_env_local.json`.
- Decision (user, 2026-11-23): TensorFlow is not installed locally. All TensorFlow work (training, .h5 save, ONNX export) runs in Colab. Local disk on C: has about 1.2 GB free.
- Colab check: `colab/run_section.ipynb` with SECTION = 1 records Colab's preinstalled versions, installs `requirements-trainer.txt` constrained to Colab's own TensorFlow, Keras, protobuf and numpy, then runs `python -m src.check_env --profile trainer` (GPU, .h5 save, Keras to ONNX round trip). `requirements.txt` gets pinned from that report.
- Disk space is also a blocker for Section 12: Docker Desktop stores images on C:, and the trainer image alone (TensorFlow) is about 2 to 3 GB.

### Colab check and pinned versions (2026-11-24)

Report: `experiments/audit/check_env_colab.json`, `pip_list_preinstalled.txt`, `python_version.txt`.

| Package | Colab | Pinned | Source of pin |
|---|---|---|---|
| Python | 3.13.18 | 3.13 (Docker base image) | Colab |
| tensorflow | 2.21.0 | 2.21.0 | Colab preinstalled |
| keras | 3.13.2 | 3.13.2 | Colab preinstalled |
| tf2onnx | 1.17.0 | 1.17.0 | installed by the launcher, checked |
| onnx | 1.23.2 | 1.23.2 | installed by the launcher, checked |
| onnxruntime | 1.32.0 | 1.32.0 | installed by the launcher, checked |
| numpy, pandas, pillow, matplotlib | 2.4.6, 2.2.3, 11.3.0, 3.10.0 | same | Colab preinstalled |
| protobuf, h5py, ml_dtypes, scikit-learn | 6.33.6, 3.14.0, 0.6.0, 1.6.1 | same | Colab preinstalled |
| fastapi, uvicorn, python-multipart | 0.141.1, 0.54.0, 0.0.33 | same | Colab preinstalled |
| requests | 2.32.4 | 2.34.2 | locust 2.47.0 needs 2.33.1 or newer; not used in training |
| streamlit | not installed | 1.66.1 | PyPI, resolver-checked |
| locust, pytest | not installed | 2.47.0, 9.2.0 | local install, import-checked |

- Checks passed: every import, `.h5` save, and the Keras to ONNX round trip with `tf2onnx.convert.from_keras` (max abs difference 4.5e-08). The SavedModel fallback was not needed.
- Surprise: `"gpus": []`. The preinstalled torch packages are `+cpu` builds, so the check ran on a CPU runtime, not a T4. Versions, `.h5` save and ONNX export do not depend on the GPU, so the pins stand. GPU training itself is not verified yet. `check_env` now prints a warning when no GPU is visible.
- `requirements.txt` (all packages) and `requirements-trainer.txt`, `requirements-api.txt`, `requirements-ui.txt`, `requirements-dev.txt` (subsets) all resolve for Python 3.13 on manylinux_2_28 x86_64 (`uv pip compile`). The trainer set resolves to exactly the Colab versions.
- Docker images: Python 3.13 on a Debian-based slim image. The TensorFlow wheel needs glibc 2.27 or newer and onnxruntime needs 2.28 or newer, so Alpine is not an option.
- Local drift: the local `.venv` is Python 3.12 with newer numpy (2.5.5), pandas (3.1.1), pillow (13.0.0), matplotlib (3.11.2), protobuf (7.36.2) and fastapi (0.165.0). Local tests pass on these, but local results are not proof for the pinned versions. The Docker images and Colab are the reference.

### Data integrity

- 1099 JPG files, counts match the dataset card exactly. 0 corrupt files, 0 unexpected folders.
- All files are JPEG, all are mode RGB, and none has an EXIF rotation tag (orientation 1 everywhere). So no CMYK or grayscale-mode conversion is needed, but `to_rgb` will still handle them for uploads.

### Duplicates and leakage (main finding)

- 114 confirmed duplicate pairs in 90 groups, 55 of them across splits. 31 test images and 21 valid images were copies of another image, mostly of a train image. Without cleaning, about 19% of the test set (31 of 165) would have been near-copies of training or validation images, which inflates test scores.
- No duplicate pair has conflicting labels.
- Action: 102 extra copies were moved to `data/quarantine/` (kept for traceability, excluded from all splits). Kept copy per group: train first, then valid, then test.
- Clean dataset: 997 images. Train 719 (182 cloudy, 142 rainy, 145 shiny, 250 sunrise), valid 144 (48, 20, 33, 43), test 134 (29, 29, 40, 36).
- Method lesson: dHash with Hamming 4 or less alone is not reliable here. It produced 619 pairs, including 151 cross-label false matches (rain splashes vs sunsets). Confirmation with a color thumbnail distance and grayscale correlation was needed, plus visual review. Details in `experiments/audit/DUPLICATE_REPORT.md`.
- Residual risk: several different shots of the same place (for example a mountain sunset series) remain and some cross splits. These are not duplicates but are close relatives.

### Shortcut risks

| Property | Evidence | Risk |
|---|---|---|
| Resolution | Median size: rainy 580x360 (0.196 MP), others about 275x180 (0.050 MP). 115 of 191 rainy images are above 0.1 MP vs 15 to 31% for the other classes. Same pattern in train, valid and test. | High for rainy. Megapixels alone separates rainy from the rest with AUC 0.73. |
| File size | Median 32 KB rainy vs 6 to 8 KB others. File size alone gives rainy-vs-rest AUC 0.82. Bytes per pixel: rainy 0.188 vs 0.115 to 0.147 (AUC 0.80). | High. After resizing to 128 px, the model never sees resolution directly, but a downscaled large photo is sharper with more fine texture than an upscaled small one. Part of this is real (rain streaks are fine texture), part may be a source-quality shortcut. |
| Aspect ratio | Medians 1.50 to 1.51 in every class, AUC 0.46 to 0.57. | Low. |
| Color mode / format / EXIF | All RGB, all JPEG, all orientation 1. | None in the dataset, but real uploads will differ. |
| Low-color images | 95 near-grayscale images (mean chroma below 0.03): 50 cloudy, 45 rainy, 0 shiny, 0 sunrise. | Medium. The model can learn "no color means cloudy or rainy". Partly real (overcast skies are gray), but a grayscale photo of a sunny scene will likely be misread. |
| Scene content | Many rainy images are street scenes with people and umbrellas, or water splashes. The other classes are mostly sky and landscape. | Medium for the Rwanda use case: a sky photo during rain without people or umbrellas may not look like the training rainy class. |
| Watermarks | Stock-photo watermarks seen in shiny and sunrise (fotolia, dreamstime). | Low to medium, to check in the Section 4 misclassified gallery. |

Because these patterns are the same in all three splits, the test set cannot reveal them. Tests planned: Section 4 performance by original resolution and file-size bins, Section 6 downscale-then-upscale and grayscale perturbations, and the field test photos.

### Label noise (visual review)

Sheets: `experiments/audit/sheets/label_review_*.jpg` (20 random per class, seed 11, plus 20 boundary cases per ambiguous pair, picked by brightness and warmth).

- Random samples (80 images): about 1 clear error (rain64 looks like a snow scene) and about 5 ambiguous images (cloudy7 and cloudy47 have the sun or a sunset glow, cloudy109/113/130 are blue sky with fair-weather clouds that could pass as shiny, rain11 shows no visible rain at this size).
- Cloudy vs rainy boundary (20): darkest cloudy images include storms with lightning and rain shafts (cloudy62, cloudy137, cloudy294), which are arguably rainy. Brightest rainy images include water splashes and waves (rain115, rain124, rain125) and a snow scene (rain65), which are arguably not rain.
- Shiny vs sunrise boundary (20): labels hold up. Blue-toned sunrises still show the sun at the horizon. One shiny image is a flower close-up with no sky (shine252), and shine236 is a dark cloud with the sun behind it.
- Estimate: about 2 to 3% clear label errors and 5 to 10% ambiguous images, concentrated on the cloudy vs rainy boundary. This boundary is the one that matters most for rain detection. No labels were changed: the evidence goes into Section 4 and the misclassified gallery will be checked against it.

### Stratification

| split | cloudy | rainy | shiny | sunrise |
|---|---|---|---|---|
| train | 25.3% | 19.7% | 20.2% | 34.8% |
| valid | 33.3% | 13.9% | 22.9% | 29.9% |
| test | 21.6% | 21.6% | 29.9% | 26.9% |

The splits are not stratified (chi-square 21.7 before cleaning, p about 0.001; 14.3 after, df 6, p below 0.05). Validation has the fewest rainy images (20) and test has 29. Consequences: accuracy is not comparable between valid and test, so macro F1 is the main metric. Single-run rainy F1 on 20 images is very noisy (one image is 5 points of recall). Decisions should rely on multi-seed means and stratified K-fold over train plus valid.

### Open questions

1. Closed 2026-11-24: Colab runs Python 3.13.18, TensorFlow 2.21.0, Keras 3.13.2. Requirements are pinned. Still open: confirm a T4 is visible before the first Section 3 run, because the Section 1 check ran on a CPU runtime.
2. Can disk space be freed on C:? Needed: about 2 GB for local TensorFlow now, and about 8 to 10 GB more for Docker images in Section 12.
3. Is the size and sharpness difference for rainy a shortcut or real signal? To test in Sections 4 and 6.
4. Should the ambiguous storm-cloud images stay labeled cloudy? Not changed for now. The Rwanda use case cares most about this boundary.

## Section 2: Preprocessing and exploratory analysis

### Preprocessing

- `src/preprocessing.py`: one PIL path for training and inference. `preprocess_image` = EXIF transpose, transparent pixels flattened on white, convert to RGB, bilinear resize to a 128x128 square (aspect ratios are 1.50 to 1.51 in every class, so stretching adds no class signal), uint8 array, divide by 255. The API will call `preprocess_image_bytes`.
- Training loads each split into memory with that same function (`load_split`). `build_dataset` only adds shuffling, augmentation and batching with `tf.data`. So training and inference preprocessing are identical by construction, and two tests prove it: `test_training_loader_equals_inference_path` (local) and `test_eval_dataset_equals_inference_path` (TensorFlow, Colab).
- The module does not import TensorFlow (a test checks this in a fresh interpreter). TensorFlow is imported inside the augmentation and dataset functions only.
- Augmentation lives in `tf.data` (stateless random ops, seeds from `Dataset.random(seed, rerandomize_each_iteration=True)`), so the exported model stays a plain inference graph and each epoch gets new augmentations. Strengths in `config.AUGMENT_STRENGTHS`: flip, zoom crop, brightness, contrast, saturation, hue. Light has no hue shift.
- Tests: 24 local tests pass. In Colab (TensorFlow 2.21.0, CPU runtime) all 22 preprocessing tests pass, including the 6 TensorFlow tests: the eval dataset equals the inference path exactly, training without augmentation only reorders, augmentation is seeded and stays in [0, 1], and each epoch gets new augmentations (`experiments/colab_section2.log`).

### Figures (`experiments/figures/`)

`class_distribution.png`, `sample_grid.png`, `image_size_distribution.png`, `feature_boxplots.png`, `image_size_comparison.png`. Feature statistics use train and valid only (863 images). The test set was used for file counts only. Tables and generated interpretations are in `experiments/eda/`.

### What three features say

Medians at 128 px. AUC is one-vs-rest or pairwise (0.5 = no signal).

1. **Mean saturation: color separates the sunny classes from the gray ones.** Sunrise 0.58, shiny 0.45, cloudy 0.18, rainy 0.11. 47% of rainy and 24% of cloudy images are nearly colorless (saturation below 0.1), and no shiny or sunrise image is. Story: overcast and rainy skies are gray. Shortcut risk: medium. It is real weather signal, but the model can learn "gray means cloudy or rainy", so a gray-toned or black-and-white sunny photo will likely be misread, and saturation barely separates rainy from cloudy (AUC 0.36). To test with the Section 6 grayscale perturbation.
2. **Blue-to-red ratio: the shiny vs sunrise boundary is almost pure color.** Shiny 1.52, cloudy 1.09, rainy 0.99, sunrise 0.48. Sunrise vs shiny AUC 0.04 (mean blue alone: 0.01). Story: shiny is a blue sky, sunrise is a warm horizon. Shortcut risk: low for the label (it is what the label means), but it makes hue augmentation risky for these two classes, and a blue-toned sunrise (several exist, see Section 1) is a likely error. Rainy vs cloudy AUC 0.28: rainy is slightly warmer, probably street scenes rather than sky.
3. **Sharpness (Laplacian variance): the only feature that separates rainy from cloudy.** Rainy 0.0118, shiny 0.0068, sunrise 0.0037, cloudy 0.0027. Rainy vs cloudy AUC 0.90, rainy vs rest 0.83. Brightness gives no help on that pair (AUC 0.55). Story: rain streaks, splashes, wet streets and crowds are fine, high-contrast texture, while cloud fields are smooth. The Section 1 worry was that this is the large-file shortcut. S2-E1 says mostly not: among small images (0.06 MP or less) the AUC is still 0.93. The remaining risk is scene content: a busy street on a dry day may be called rainy, and a sky photo during light rain (the Rwanda use case) may be called cloudy. To check in the Section 4 misclassified gallery and with field photos.

Brightness is also informative (shiny brightest at 0.56, sunrise darkest at 0.38, shiny vs rest AUC 0.81) but adds nothing for rainy vs cloudy. Contrast separates nothing (all AUC 0.39 to 0.60).

### Image size (S2-E1)

- Color statistics do not change with size (correlation 0.998 or more with the 224 reference at 96, 128 and 160). Fine detail does: rainy loses the most (median PSNR vs 224: 28.7 dB at 96, 30.8 at 128, 33.1 at 160, against 36.3 / 39.3 / 42.3 for cloudy). In `image_size_comparison.png` rain streaks are faint at 96 and clear at 160 (bottom row: the largest rainy photo).
- 128 is the largest size where almost no image is upscaled (0 to 1.4% per class). At 160, 12 to 16% of cloudy, shiny and sunrise images are upscaled and no rainy image is. At 224 it is 72% of cloudy vs 35% of rainy. Larger sizes add a "blurry means not rainy" resolution shortcut.
- Decision for R0: 128 px. 160 is a reasonable Section 5 grid candidate for rain detail, but only together with the downscale-then-upscale robustness test.

### Augmentation preview (S2-E2)

- Previews: `experiments/figures/augmentation_preview_{light,medium,strong}.png`. Rows: one random train image per class, plus the least saturated sunrise and the brightest cloudy (the images most likely to change label).
- Light and medium keep every label. Strong (saturation 0.35, hue 0.05, brightness 0.2) does not: sunrise turns washed-out pink-gray or nearly black, the brightest cloudy goes almost white, shiny gets a teal cast.
- Decision (user): medium is the R0 default. Hue is 0 at every level. Strong is only an experiment arm, with saturation capped at 0.2. Final settings in `config.AUGMENT_STRENGTHS`: medium = flip, zoom 0.1, brightness 0.1, contrast 0.2, saturation 0.2. The capped strong setting has not been previewed yet.

### Open questions

5. Closed: medium is safe for every class, see S2-E2. Still open: preview the capped strong setting before it is used in a grid.
6. Rainy images are partly street scenes: how much of the rainy class is "people with umbrellas" rather than weather? Section 4 gallery.

### Section 2 gate

Closed 2026-11-24. Preprocessing tests pass locally and in Colab, including the training-vs-inference equality tests. Plots are in `experiments/figures/`. Three feature interpretations are above. R0 inputs are fixed: 128 px, medium augmentation without hue.

## Section 3: Experiment harness, sanity checks, baselines

Kaggle T4 (one device), TensorFlow 2.21.0, 2026-11-24, 19 minutes for the whole version (setup, 118 tests, smoke, timing, diag, 11 runs). Validation set only. The test set was not touched. Details and hypotheses: S3-E1 to S3-E4 in `EXPERIMENT_LOG.md`.

### Baseline vs R0 (5 seeds each, validation)

| | baseline (vanilla CNN) | R0 (all optimizations) |
|---|---|---|
| macro F1 mean +- std | **0.908 +- 0.027** | 0.659 +- 0.260 |
| macro F1 min / max | 0.872 / 0.943 | 0.429 / 0.941 |
| macro F1 per seed (11, 22, 33, 44, 55) | 0.872, 0.894, 0.909, 0.943, 0.923 | 0.451, 0.536, 0.941, 0.429, 0.941 |
| rainy F1 mean +- std (min) | 0.963 +- 0.046 (0.889) | 0.390 +- 0.534 (0.000) |
| epochs run / best epoch | 18 to 26 / 3 to 11 | 16, 16, 64, 16, 77 / 1, 1, 49, 1, 62 |
| parameters | 8,408,644 | 305,380 |
| minutes per run | 0.29 | 1.10 |

Std is the sample std over seeds. `python -m src.selection --section 3` prints this table.

- The baseline is strong and stable. Its weak pair is cloudy vs shiny (48 of 720 predictions), not rainy vs cloudy (6). Rainy F1 is 0.96 but moves by up to 0.11 between seeds on 20 images.
- R0 is bimodal. Converged seeds (33, 55) reach 0.941 with rainy F1 0.97, which is 0.033 above the baseline mean (slightly more than one baseline std). Stalled seeds (11, 22, 44) stop at epoch 16 with best epoch 1 and rainy F1 0.000, with almost every rainy image called cloudy.
- Selection rule today: the baseline wins, R0 is not eligible (worst seed below 0.85).

### Sanity checks

- **Overfit one batch:** passed. Training accuracy 1.0 at step 7. Inference accuracy 1.0 after refreshing the batch norm statistics on the batch, against 0.406 with stale statistics.
- **Diag verdict (a), confirmed:** batch norm moving statistics lag in inference mode. The input pipeline is identical for training, evaluation and inference (exact equality, RGB order checked against an independent decode), and the three inference paths agree. Only batch norm variants show the gap, a refresh closes it in every variant, and momentum 0.9 closes it without a refresh.
- **Shuffled labels:** validation macro F1 0.381, outside the random-prediction chance band. With the train class prior on 144 images, single draws give 0.245 +- 0.036 (95% band 0.18 to 0.32). The best of 44 draws gives 0.327 +- 0.018 (95% band 0.30 to 0.37), and 0.9% of those reach 0.381. It is not leakage. The single fixed permutation keeps a chance correlation with the true classes (class-weighted shares 0.21 to 0.31 instead of 0.25). A model that cannot memorize learns exactly that part, and the observed confusion follows it. Across 2000 permutations this mechanism gives 0.22 +- 0.21, and 24% of permutations reach 0.381. Train accuracy stayed at 0.24 to 0.34, so the noise labels were not memorized. Simulation: `python -m src.chance_band`, output `experiments/sanity/chance_band.json`.
- **Reference levels:** majority class 0.115 macro F1, random with train prior 0.243 +- 0.035.

### Why R0 stalls (from history.csv, no new run)

Training itself is not stuck. Validation in inference mode is.

| seed | train acc epoch 1 / 16 | val loss epoch 1 | peak val loss (epoch) | best val loss of epochs 1 and 2 | first epoch that beats it | outcome |
|---|---|---|---|---|---|---|
| 11 | 0.69 / 0.91 | 1.205 | 3.17 (9) | 1.205 (epoch 1) | none by 16 | stopped at 16 |
| 22 | 0.66 / 0.93 | 1.157 | 2.60 (8) | 1.157 (epoch 1) | none by 16 | stopped at 16 |
| 33 | 0.68 / 0.93 | 1.321 | 4.48 (6) | 1.299 (epoch 2) | 16 | converged, 0.941 |
| 44 | 0.70 / 0.93 | 1.253 | 3.48 (10) | 1.253 (epoch 1) | none by 16 | stopped at 16 |
| 55 | 0.68 / 0.93 | 1.165 | 2.27 (10) | 0.987 (epoch 2) | 14 | converged, 0.941 |

- Training-mode accuracy rises smoothly in every seed, with no plateau or divergence. Learning rate, class weights and augmentation do not block training.
- Validation loss follows the batch norm moving statistics, not the weights. With 23 steps per epoch and momentum 0.99, the initial values (mean 0, variance 1) still make up 79% of each moving statistic after epoch 1, 31% after 5, 10% after 10 and 3% after 15. Validation loss is near-uniform at epoch 1, explodes at epochs 6 to 10 while the statistics are a stale mix, and recovers around epochs 14 to 20.
- Early stopping (patience 15) fires during that lag unless the best loss happened to land on epoch 2. That is what separated the two converged seeds from the three stalled ones. ReduceLROnPlateau is fooled the same way and halves the rate at epochs 7 and 12.
- Direct evidence: the stalled seeds' restored weights score 0.49 to 0.59 in inference mode on their own training images.

### Other notes

- GPU runs are not bit-for-bit reproducible with fixed seeds. Decisions use 5-seed means.
- From Section 5 on, `model.keras` is kept only for the best seed of the top 3 configurations by mean validation macro F1, and saved without optimizer state (user decision; Section 3 zip was 556 MB, 97 MB per baseline-family model). This changes rule 6 (`EXPERIMENT_LOG.md`, Run files).
- In Keras 3.13, `clone_model` compiles the clone again and saves a fresh optimizer of the same size, so a clone does not drop optimizer state. An uncompiled copy from `Model.from_config` does (35% of the size). This stopped the first Section 5 version in the test step (`EXPERIMENT_LOG.md`, Run files, incident).
- The baseline is now built by the same `build_cnn` function as R0. It has the same layers and parameter count as in Section 3, so the Section 3 baseline runs stay valid as the reference.

### Open questions

7. Does a lower batch norm momentum or a later stopping rule make R0 converge on every seed, and does converged R0 really beat the baseline by more than one std? Section 5 groups A and B.
8. Which single optimization explains the gap between the baseline and converged R0, and does adding batch norm alone hurt the baseline? Section 5 group C (ablation).
9. The baseline's weak pair is cloudy vs shiny. Is that label ambiguity (fair-weather clouds on blue sky, Section 1 review) or a model limit? Section 4 misclassified gallery.
10. The shuffled-label control is weak with one permutation. If it is rerun, use 3 or more permutations and the permutation null.
11. Alternative fix not planned yet: evaluate validation with refreshed batch norm statistics each epoch. It keeps momentum 0.99, but the exported model would then need refreshed statistics too.

### Section 5 plan (first Kaggle version)

Written in `EXPERIMENT_LOG.md` before any run, with the selection rule: highest mean validation macro F1 over 5 seeds, ties within one std go to the lower std, and the worst seed must be 0.85 or more. Groups: A batch norm momentum 0.9 and 0.8, B patience 30 with at least 20 epochs, C ablation from the baseline (3 seeds), D learning rate 3e-4, 5-epoch warmup, light augmentation. Budget guard: no new run after 50 minutes of the experiment process.
