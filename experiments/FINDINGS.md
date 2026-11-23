# Findings

## Section 1: Environment, skeleton, data audit

### Environment

- Local: conda env at `.venv/` (Python 3.12.16). Installed and import-checked: numpy, pandas, Pillow, matplotlib, pytest, onnxruntime, FastAPI, uvicorn, Locust. See `experiments/audit/check_env_local.json`.
- Decision (user, 2026-11-23): TensorFlow is not installed locally. All TensorFlow work (training, .h5 save, ONNX export) runs in Colab. Local disk on C: has about 1.2 GB free.
- Colab check: `colab/run_section.ipynb` with SECTION = 1 records Colab's preinstalled versions, installs `requirements-trainer.txt` constrained to Colab's own TensorFlow, Keras, protobuf and numpy, then runs `python -m src.check_env --profile trainer` (GPU, .h5 save, Keras to ONNX round trip). `requirements.txt` gets pinned from that report.
- Disk space is also a blocker for Section 12: Docker Desktop stores images on C:, and the trainer image alone (TensorFlow) is about 2 to 3 GB.

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

1. Colab Python, TensorFlow and Keras versions (needed to pin requirements and the Docker trainer image).
2. Can disk space be freed on C:? Needed: about 2 GB for local TensorFlow now, and about 8 to 10 GB more for Docker images in Section 12.
3. Is the size and sharpness difference for rainy a shortcut or real signal? To test in Sections 4 and 6.
4. Should the ambiguous storm-cloud images stay labeled cloudy? Not changed for now. The Rwanda use case cares most about this boundary.
