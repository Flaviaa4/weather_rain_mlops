# Weather Rain MLOps: Claude Code Playbook

## How to use this file

1. Copy Section 0 into a file named `CLAUDE.md` at the repo root. Claude Code reads it automatically in every session.
2. Paste ONE section at a time using this wrapper:
   `Read CLAUDE.md. Do Section N of the playbook only. Stop at the "Done when" gate, update experiments/FINDINGS.md, and give me a short summary.`
3. Read `experiments/EXPERIMENT_LOG.md` and `experiments/FINDINGS.md` before pasting the next section. Do not move on until the gate is true.

Think of the model as a recipe. We do not rewrite the recipe on a hunch. We change one ingredient at a time, taste the result, and write down what happened and why. That habit is what tells us where the model fails and what a change will do.

---

## SECTION 0: Ground rules (goes into CLAUDE.md)

### Project

Weather image classifier (cloudy, rainy, shiny, sunrise) used to support rain detection for a Rwanda climate use case. It is a full MLOps pipeline: data acquisition, preprocessing, model creation, testing, retraining with a trigger, a Python API, a UI, cloud deployment, and a load test with Locust. The data is images, not tabular.

### Code rules

- No comments, no docstrings, no commented-out code. The only exception is a one-line comment explaining a non-obvious reason.
- No em-dashes or en-dashes anywhere: code, logs, README, notebook text. Use commas, colons or hyphens.
- Many small single-purpose functions (about 20 lines or fewer). Names describe the action. No function may do the whole pipeline. Orchestrator functions only call other functions in order.
- Stack: TensorFlow/Keras, tf2onnx, onnxruntime, FastAPI, Streamlit, SQLite, Docker, nginx. Experiments train on Google Colab (GPU). Deployment runs in Docker on a cloud VM (CPU). The TensorFlow version must be the same in Colab and in the Docker trainer image, so first run `python --version` and `pip list` in Colab, then pin the Docker images and requirements to versions that work in both. Verify imports before anything else. If `tf2onnx.convert.from_keras` fails, export a SavedModel and use `python -m tf2onnx.convert --saved-model`.
- Requirements: the repo root has a `requirements.txt` listing every package with pinned versions (the facilitator expects this file). The per-service files (`requirements-api.txt`, `requirements-trainer.txt`, `requirements-ui.txt`) are subsets of it.
- No transfer learning and no pretrained weights. The model is a custom CNN trained from scratch. Retraining starts from our own saved model.
- Saved artifacts: `models/weather_cnn.h5`, `models/weather_cnn.onnx`, `models/class_names.json`, `models/metrics.json`.

### Experiment rules (the most important part)

1. Every reported result comes from a run trained to convergence: max 150 epochs, EarlyStopping on validation loss with patience 15 and best weights restored, plus ReduceLROnPlateau. Never judge a configuration from a few epochs. Log the epochs actually run and the best epoch.
2. Seeds: 11, 22, 33, 44, 55. Any configuration that matters runs with at least 3 seeds, finalists with 5. Report mean and std. A difference smaller than one std is "no difference".
3. Change one thing at a time against a named reference configuration.
4. Before running, write a hypothesis in `experiments/EXPERIMENT_LOG.md`: what changes, expected direction and size, and why. After running: observed result, hypothesis right or wrong, conclusion.
5. The test set is locked. No decision uses it. Tune on the validation set and on stratified K-fold over train plus valid. The test set is evaluated only in Section 7, once per finalist at most. Log every touch.
6. Every run saves to `experiments/runs/<run_id>/`: `config.json`, `history.csv`, `curves.png`, `val_metrics.json`, `confusion_matrix.png`, `val_predictions.csv` (path, true, predicted, confidence). Best weights are saved for every run (the model is small), so any analysis can be repeated later without retraining.
7. `experiments/results.csv` has one row per run: run_id, section, config fields, seed, epochs_run, best_epoch, train and val loss and accuracy, macro precision, recall and F1, per-class F1, minutes.
8. Long jobs must be resumable (skip run_ids already in `results.csv`), log to a file and print an ETA. First time one epoch on the Colab GPU and estimate the total runtime and compute units for the planned grid, report it, then run everything. If the grid is too long, drop grid points and document which. Never shorten epochs to fit.
9. When a run fails or surprises you, investigate and log it. Do not silently patch around it.
10. At the end of every section, update `experiments/FINDINGS.md` with short conclusions and open questions, then stop.

### Dataset facts

Source: Roboflow "weather classification" (folder-structure export), 1099 JPG images, classes `cloudy`, `rainy`, `shiny`, `sunrise`. Layout: `data/train`, `data/valid`, `data/test`, each with one folder per class. Uploaded retraining data goes in `data/uploads/<class>/`.

| Split | cloudy | rainy | shiny | sunrise | total |
|---|---|---|---|---|---|
| train | 200 | 149 | 157 | 263 | 769 |
| valid | 57 | 23 | 39 | 46 | 165 |
| test | 34 | 34 | 54 | 43 | 165 |

Most images are small (about 300x168), a few are 1920x1080. Rainy has only 23 validation and 34 test images, so single-run rainy metrics are noisy. That is why we use multiple seeds, K-fold and bootstrap confidence intervals.

### Facilitator requirements (must all be true at the end)

- Deployment instructions written in the README, step by step, including ONNX Runtime.
- Many small functions, not one function or one pipeline that does everything.
- The app keeps running until grading is finished.
- Everything the facilitator needs is visible in the repo: Dockerfiles, compose file, nginx config, `requirements.txt`, locustfile, model files, sample data.
- The server has spare disk space so uploaded files can be saved and retraining can run.
- The model is new (trained from scratch), not transfer learning.

### Repo structure

```
weather_rain_mlops/
  README.md
  CLAUDE.md
  notebook/weather_rain_classification.ipynb
  src/ config.py preprocessing.py model.py prediction.py retrain.py database.py insights.py experiments.py
  app/ api.py trainer_api.py ui.py
  data/ train/ valid/ test/ uploads/
  models/ weather_cnn.h5 weather_cnn.onnx class_names.json metrics.json archive/
  experiments/ EXPERIMENT_LOG.md FINDINGS.md DECISION.md results.csv runs/
  docker/ Dockerfile.api Dockerfile.trainer Dockerfile.ui nginx.conf
  docker-compose.yml
  locust/ locustfile.py run_flood_test.sh summarize.py results/
  tests/
  colab/ run_section.ipynb
  requirements.txt requirements-api.txt requirements-trainer.txt requirements-ui.txt requirements-dev.txt
```

### Failure triage cheat sheet

| Symptom | Likely cause | First experiment |
|---|---|---|
| Train accuracy high, val low | Overfitting | More augmentation, dropout, L2, fewer filters |
| Train and val both low | Underfitting | More depth or filters, higher LR, longer patience |
| Val loss noisy, jumps | Tiny val set or LR too high | Lower LR, K-fold, more seeds |
| One class has low recall | Imbalance or ambiguous labels | Class weights, oversampling, inspect images |
| Confident wrong predictions | Label noise or shortcut features | Misclassified gallery, duplicate check |
| Test far below val | Over-tuning or leakage | Duplicate check, K-fold, bootstrap CI |
| Great accuracy, bad on real photos | Domain shift | Field test photos, robustness tests |

---

## SECTION 0B: Colab workflow (also goes into CLAUDE.md)

Claude Code cannot control Colab, so the work is split in two:

- **Claude Code (local):** writes all code, smoke-tests it on a tiny subset on CPU (a handful of images, 2 epochs, only to prove it runs), writes the Colab launcher, and runs every analysis step that needs only saved files.
- **Colab (GPU, run by the user):** runs the full-convergence experiment grids.

Rules:

1. Create `colab/run_section.ipynb`, a launcher that takes a section number and a chunk name. It mounts Google Drive, clones the GitHub repo (or unzips the repo from Drive), installs `requirements.txt`, copies the dataset from Drive to local `/content/data` (reading from Drive during training is slow), sets `EXPERIMENTS_DIR` to a folder on Drive so results survive disconnects, runs `python -m src.experiments --section N --group G`, and prints the ETA.
2. Everything a run writes goes to Drive: `results.csv` rows, run folders, logs. A disconnect may lose at most the current run. Long runs also use the Keras `BackupAndRestore` callback pointed at Drive, so an interrupted run resumes from the last epoch.
3. Colab sessions can disconnect when idle or after hours of use, and GPU access can be limited. So split big grids into chunks (`--group A`, `--group B1`, `--group B2` and so on), each finishing in well under 3 hours. After every chunk, print which chunk to run next.
4. Before starting a section, report the estimated GPU hours and compute units. The user has about 74 compute units, visible in Colab's Resources panel. Prioritize groups by value (A, then B, then C and the rest), and stop to ask the user before running a plan that would use more than 60% of the remaining units.
5. GPU runs are not bit-for-bit reproducible even with fixed seeds. Note this in FINDINGS.md and rely on multi-seed means.
6. Hand-off loop: every Colab-bound section ends with exact instructions: what to open in Colab, which chunk to run, and which Drive folder to copy back into the local repo's `experiments/` folder. After the user copies results back, Claude Code runs the analysis and writes FINDINGS.md.
7. Never put passwords or GitHub tokens in the launcher notebook. Use a public clone or a Drive upload.
8. Where things run: training grids (Sections 3, 5, 6) and final training (Section 7) run in Colab. Analysis of saved predictions, histories and weights (Sections 1, 2, 4 and the analysis parts of 5 to 7) runs locally. Sections 9 to 14 run locally and on the VM, with no GPU.

---

## SECTION 1: Environment, skeleton, data audit

**Goal:** a clean working base, and proof that the data is trustworthy before any model is trained.

**Tasks**
- Create the repo skeleton and a local virtual environment. Check the Python and TensorFlow versions in Colab and match them. Pin requirements in `requirements.txt`. Verify that TensorFlow, tf2onnx, onnxruntime, FastAPI and Locust import, locally and in Colab.
- Write `config.py` (paths, class names, image size, seeds, thresholds, all overridable by environment variables).
- Write the data audit functions: list files per split and class, find corrupt files, record original size, mode (RGB, grayscale, CMYK), file size, aspect ratio.

**Questions to answer (write the answers in FINDINGS.md)**
- Are there duplicates or near-duplicates across train, valid and test? Use a perceptual hash (dHash) with Hamming distance of 4 or less. Duplicates across splits inflate scores. Report the count and remove them from the evaluation splits, or move them so no leakage remains.
- Does image resolution, aspect ratio or file size differ by class? If rainy images are mostly one resolution, the model can learn that shortcut instead of the weather.
- Open 20 random images per class and 20 from the ambiguous pairs (shiny vs sunrise, cloudy vs rainy). Estimate label noise.
- Are the splits stratified? Note the small rainy valid and test counts.

**Done when:** `import` checks pass, an audit table and duplicate report exist, and FINDINGS.md lists any leakage, shortcut or label-noise risks with evidence.

---

## SECTION 2: Preprocessing and exploratory analysis

**Goal:** the preprocessing module and the visualizations that tell the data's story.

**Functions (examples, one job each):** `is_image_file`, `is_valid_image_bytes`, `to_rgb` (with EXIF orientation fix), `resize_image`, `to_array`, `normalize_array`, `preprocess_image`, `preprocess_image_bytes`, `add_batch_dimension`, `list_labeled_files`, `load_split`, `augment_image`, `build_dataset`. The inference path (everything except TensorFlow dataset code) must not import TensorFlow, because the API image uses only ONNX Runtime. Import TensorFlow lazily inside the dataset and augmentation functions.

**Rules:** augmentation lives in the `tf.data` pipeline, not as Keras layers, so the exported ONNX model is a plain inference graph. Training and inference preprocessing must be identical, and a test must prove it.

**Insights module (`insights.py`):** per-image features: brightness, contrast, mean saturation, mean R, G, B, blue-to-red ratio, original width, height and aspect ratio. Functions to summarize by class and to write plain-language interpretations from the numbers.

**Visualizations:** class distribution per split, sample grid per class, image size distribution, and boxplots by class for at least 3 features. For each of the 3 features write what story it tells (for example, whether rainy skies are darker) and whether the model could be using it as a shortcut.

**Experiments**
- Image size effect on information: compare 96, 128 and 160 pixels visually and with a quick feature-preservation check.
- Augmentation preview: show 8 augmented versions of the same image at each strength level (light, medium, strong). Check that none destroys the label (for example, brightness so strong that a sunrise looks like cloudy).

**Done when:** preprocessing tests pass, including the training-vs-inference equality test, plots are saved in `experiments/figures/`, and 3 feature interpretations are in FINDINGS.md.

---

## SECTION 3: Experiment harness, sanity checks, baselines

**Goal:** a trustworthy way to run and compare experiments.

**Harness functions (`experiments.py`):** `make_run_id`, `load_config`, `set_seeds`, `run_one`, `log_result_row`, `already_done`, `save_run_artifacts`, `plot_curves`, `collect_predictions`, `estimate_total_time`, `run_grid`. `run_grid` takes a list of configs and seeds, skips finished runs and reports an ETA.

**Models (`model.py`):** `build_baseline_cnn` (plain conv-pool x2, flatten, dense, no regularization, no batch norm), `build_cnn` (configurable blocks, filters, depth, batch norm, dropout, L2, global average pooling, dense units), `compile_model`, `get_callbacks`, `compute_class_weights`, `train_model`, `evaluate_model` (accuracy, loss, macro precision, recall, F1, per-class report, confusion matrix, ROC AUC one-vs-rest).

**Sanity checks (must pass before any comparison)**
- Overfit one batch of 32 images to near 100% train accuracy. If it cannot, the pipeline has a bug.
- Shuffled-label run: accuracy should be near chance (25%). If it is much higher, something leaks.
- Majority-class baseline and random baseline numbers recorded for reference.

**Runs**
- Baseline vanilla CNN: 5 seeds, full convergence.
- Reference config R0 (optimized CNN: 3 blocks of 32, 64, 128 filters, batch norm, dropout 0.3, L2 1e-4, Adam 1e-3, class weights, medium augmentation, batch 32, 128 px): 5 seeds, full convergence.

**Done when:** sanity checks pass, `results.csv` has both baselines with mean and std, and FINDINGS.md says how much the optimizations buy over the vanilla model.

---

## SECTION 4: Where does it fail?

**Goal:** a ranked list of failure modes, with evidence, before changing anything.

Use the reference models from Section 3 on the validation set (not the test set).

- Per-class precision, recall, F1 and the confusion matrix. Name the top 3 confused pairs.
- Misclassified gallery sorted by confidence. Confident mistakes are the most informative. After viewing, tag each with a likely cause: label noise, ambiguous scene, low light, tiny image, watermark or text, unusual composition.
- Performance by image property: bins of brightness, contrast, saturation, original resolution and aspect ratio. Table and plot.
- Calibration: reliability diagram, expected calibration error, mean confidence when right vs wrong, and a coverage-vs-accuracy curve for confidence thresholds (this will drive an "uncertain" response in the UI).
- Learning-curve diagnosis: train vs val loss gap, where the best epoch falls, whether the model is underfitting or overfitting. Use the triage cheat sheet.
- Bootstrap 95% confidence intervals (1000 resamples) on per-class F1, to show how noisy the small classes are.
- Seed spread: how much do per-class results vary across the 5 seeds?

**Output:** a table with columns: failure mode, evidence, affected class, hypothesis for a fix, experiment that would test it.

**Done when:** the failure table is in FINDINGS.md, the gallery and plots are saved, and every proposed fix has a written hypothesis in EXPERIMENT_LOG.md.

---

## SECTION 5: The full experiment grid

**Goal:** know which knobs matter, how much, and what each change does to the failure modes.

Run everything with the experiment rules from Section 0 (full convergence, multiple seeds, hypothesis first) and the chunked Colab workflow from Section 0B. Reference is R0.

**A. Ablation, remove one ingredient at a time:** augmentation off, batch norm off, dropout 0, L2 0, class weights off, LR scheduler off, global average pooling replaced by flatten.

**B. Sensitivity, change one knob at a time:**
- Learning rate: 3e-4, 1e-3, 3e-3
- Batch size: 16, 32, 64
- Dropout: 0.2, 0.3, 0.5
- L2: 0, 1e-5, 1e-4, 1e-3
- Base filters: 16, 32, 64
- Depth: 2, 3, 4, 5 blocks
- Dense units: 64, 128, 256
- Image size: 96, 128, 160
- Optimizer: Adam, AdamW, SGD with Nesterov momentum, RMSprop
- Label smoothing: 0, 0.1
- Augmentation: none, light, medium, strong, plus each augmentation type alone

**C. Imbalance handling:** none, class weights, oversampling the smallest classes.

**D. Interactions:** combine the best settings from B in a small grid (top 2 or 3 values per knob for the 3 most sensitive knobs).

**E. Stability:** stratified 5-fold cross-validation over train plus valid for the top 5 configurations. The test set stays untouched.

**F. Seed variance:** the best configuration with 10 seeds.

After each group, re-run the Section 4 diagnostics on the best and worst run of that group, and write which classes and confused pairs moved. Update the hypothesis table with right or wrong.

**Output:** a sensitivity ranking (which knob moves macro F1 most, with effect size and std), a plot per group, and the top 3 candidate configurations.

**Done when:** all groups are in `results.csv`, the log has a hypothesis and conclusion for every group, and FINDINGS.md has the ranking and candidates.

---

## SECTION 6: Robustness and data-size studies

**Goal:** know how the model behaves outside the clean validation set, and whether more data would help (this decides how valuable retraining is).

- **Data-size curve:** train on 25%, 50%, 75% and 100% of the training data (stratified, 3 seeds each) for the top candidate. Is the curve still rising? If yes, uploaded data should help.
- **Robustness:** apply perturbations to validation images at several severities: brightness up and down, contrast, Gaussian blur, Gaussian noise, JPEG quality 90, 50 and 20, rotation, random crop, grayscale, downscale then upscale. Plot accuracy vs severity per class. Compare the vanilla baseline, R0 and the candidates: does augmentation buy robustness?
- **Out-of-distribution inputs:** feed synthetic images (noise, solid colors, gradients, checkerboards) and any non-weather photos placed in `data/ood/`. Record confidence. Decide a confidence threshold below which the API answers "uncertain".
- **Field test (optional but valuable):** if the user places 10 to 20 phone photos of local skies per class in `data/field_test/`, evaluate on them and report the domain-shift gap.

**Done when:** plots and tables are saved, FINDINGS.md states the data-size trend, the most damaging perturbations, and the chosen uncertainty threshold with evidence.

---

## SECTION 7: Decision, final training, test evaluation, export

**Goal:** choose the final model by a written rule, evaluate once on the test set, export.

**Decision rule (write it in `DECISION.md` before looking at test):**
1. Highest mean macro F1 in cross-validation across the candidates.
2. Tie-break by lower std of rainy F1.
3. Then fewer parameters, then lower latency.
Include the evidence tables and a list of rejected options with reasons.

**Final training:** the chosen config, 5 seeds, early stopping on validation. Pick the seed by validation performance only.

**Test evaluation (once):** accuracy, loss, macro precision, recall, F1, per-class metrics, ROC AUC, confusion matrix, bootstrap 95% confidence intervals. Compare against validation. A big gap means over-tuning or leakage, so investigate.

**Export and checks**
- Save `weather_cnn.h5`, then export `weather_cnn.onnx`. Run the ONNX export and the parity test inside the Docker trainer image (the same TensorFlow version as production), not only in Colab.
- ONNX parity on every test image: max absolute probability difference below 1e-4 and 100% identical predicted class.
- Latency benchmark: Keras vs ONNX Runtime, single image and batch, with 1 and 2 threads.
- Save `class_names.json` and `metrics.json` (version, config, seed, dataset hash, metrics, date).

**Done when:** DECISION.md exists, the test results are in FINDINGS.md next to validation, parity passes, and all four model files are in `models/`.

---

## SECTION 8: The notebook

**Goal:** `notebook/weather_rain_classification.ipynb` that runs top to bottom with outputs saved.

**Required sections:** data acquisition, detailed preprocessing steps (each step shown with shapes and value ranges on a real image), visualizations with interpretations, baseline model, optimized model with the optimization techniques named (regularization, optimizer, early stopping, LR schedule, hyperparameter tuning, augmentation), evaluation with at least 4 metrics plus confusion matrix, test and prediction functions with a demo on test images (right and wrong), ONNX export and parity.

**Rules:** import small functions from `src/` and show their output, no copy-pasted logic. The notebook loads `results.csv` to show the experiment summary (ablation, sensitivity and robustness plots) and retrains only the final configuration, so it finishes in a reasonable time. Write the retraining explanation as markdown that points to `src/retrain.py`. No unnecessary comments in code cells.

**Done when:** it runs clean from a fresh kernel, outputs are saved, and the metrics match `metrics.json`.

---

## SECTION 9: Prediction module and prediction API

**Functions (`prediction.py`, ONNX Runtime only, no TensorFlow):** `load_class_names`, `load_onnx_session` (configurable thread count), `get_session` (reload when the model file changes), `run_inference`, `ensure_probabilities`, `format_prediction` (label, confidence, all probabilities, uncertain flag from the Section 6 threshold), `predict_image_bytes`, `model_version`.

**API (`app/api.py`):** `GET /health`, `GET /uptime`, `GET /model-info`, `POST /predict` (validates type and size, returns label, confidence, probabilities, latency in ms, and the container hostname so load balancing is visible). Sync endpoints so inference runs in the thread pool.

**Experiments and tests**
- Unit tests for validation errors (non-image, oversized, empty, corrupt).
- Pick 5 test images per class that are predicted correctly with high confidence and save the list for the video demo. Include one honest "uncertain" example.
- Hot reload test: replace the ONNX file while requests are in flight. No request may fail.

**Done when:** tests pass, and a curl command and Swagger both return a correct prediction.

---

## SECTION 10: Retraining pipeline and retraining experiments

**Goal:** upload, save to a database, preprocess, retrain from our own saved model, promote safely, with a trigger.

**Functions (`retrain.py`, `database.py`):** `validate_label`, `save_upload`, `save_uploads`, `extract_zip_uploads`, `add_upload` (SQLite), `count_unused_uploads`, `should_retrain`, `load_upload_arrays`, `prepare_retraining_data`, `fine_tune_model` (loads our `.h5`, lower learning rate), `should_promote`, `archive_current_model`, `promote_model` (writes `.h5`, `.onnx`, `metrics.json`), `start_retraining` (background thread with a lock, one job at a time), `get_status`. Database tables: uploads and training runs.

**Trigger:** manual button, plus an automatic rule when unused uploads reach a threshold (default 40, configurable). Retraining uses the base training data plus all uploads, with the uploads marked as used afterwards.

**Promotion rule:** the new model replaces the current one only if its validation macro F1 is at least the current model's on the same validation set. Archive the old model first.

**Retraining experiments (full convergence, multiple seeds)**
- Simulate uploads: hold out 30% of train as "uploads" in batches of 20, 40 and 80.
- Fine-tune learning rate: 1e-4, 3e-4, 1e-3. Compare against training from scratch on all data.
- Forgetting check: per-class accuracy on the original validation set before and after fine-tuning.
- Bad uploads: 10% and 30% wrong labels. Does the promotion rule block the damage?
- Skewed upload: only rainy images. What happens to the other classes?
- Threshold study: 20, 40, 80 images, retrain cost vs gain across 10 simulated batches. Count wrongly promoted and wrongly rejected models.
- Retrain duration on the target server size, and peak memory.

**Done when:** an upload then retrain then promote cycle works end to end from a script, the experiment results and the recommended threshold are in FINDINGS.md, and the unit tests cover the rejection, trigger and promotion logic.

---

## SECTION 11: Trainer API and UI

**Trainer API (`app/trainer_api.py`):** `GET /health`, `GET /uptime`, `GET /system` (disk free and used, upload count, model file sizes), `POST /upload` (multiple files plus a label), `POST /upload_zip` (folders named by class), `GET /uploads/summary`, `POST /retrain` (409 if already running), `GET /retrain/status`, `GET /metrics`, `GET /insights`, `GET /runs`.

**UI (`app/ui.py`, Streamlit), four pages**
1. Predict: upload one image, show the image, the predicted class, confidence, probability chart, uncertain warning, container name and latency.
2. Insights: the visualizations from Section 2 with the interpretations.
3. Upload and Retrain: bulk upload, upload summary, a retrain button with live status, run history, old vs new metrics.
4. Monitoring: model uptime, health of both services, disk space, model version, metrics and confusion matrix.

**Experiments**
- Upload 200 images at once and a 50 MB zip. Does it hold? Check the nginx body size limit.
- Click retrain twice quickly, upload during training, kill the trainer mid-run and restart. Record behavior and fix what breaks.

**Done when:** every page works against the running services, and the edge cases above are documented with their fixes.

---

## SECTION 12: Docker, nginx, deployment, and the README steps

**Platform decision:** Docker on a cloud VM. AWS Lambda is not used.

**Design:** `api` image has ONNX Runtime only and is scalable. `trainer` image has TensorFlow and runs a single instance. `ui` image runs Streamlit. nginx routes `/predict`, `/health`, `/model-info`, `/uptime`, `/docs` to the `api` service and `/trainer/` to the trainer. The `data/` and `models/` folders are bind mounts so the facilitator can see everything and so uploads survive restarts. nginx resolves the service name at request time so `--scale api=N` works.

**Files:** three Dockerfiles, a root `requirements.txt` plus per-service requirements files, `docker-compose.yml` (restart always, healthchecks, CPU limit per api container so scaling is visible on a small server), `nginx.conf` (large body size, long read timeout for the trainer), `.dockerignore`, `.gitignore`.

**Deployment on a VM (document every step in the README):** create the VM with at least 4 GB RAM and 30 GB disk, open ports, install Docker, clone the repo, `docker compose up -d --build`, verify `/health`, set Docker to start on boot, assign a fixed IP or domain. Explain where ONNX Runtime is used and why (small image, fast inference), and how `.h5` is converted to `.onnx` after each retrain.

**Keep it running until graded:** a checklist covering billing alerts, no stopping or terminating the instance, restart policy, disk usage alert, a daily health check, and a backup copy of `models/`.

**Experiments**
- Fresh clone on a clean machine: do the README steps work exactly as written?
- Reboot the VM: do all containers come back?
- Fill the uploads folder to a set size: does retraining still run, and what is the failure when the disk is nearly full? Add a guard that refuses uploads below a free-space threshold with a clear message.
- Image sizes and cold start times for each container.

**Done when:** the public URL works, the fresh-clone and reboot tests pass, and the README deployment section is complete.

---

## SECTION 13: Flood simulation with Locust

**Goal:** show how latency and response time change with the number of containers.

**Files:** `locust/locustfile.py` (mostly `/predict` with sample test images, a small share of `/health`), `locust/run_flood_test.sh` (loops over replica counts and user counts), `locust/summarize.py` (turns the Locust CSV files into a markdown table).

**Protocol (document it in the README)**
- Replicas: 1, 2, 3, 4 (`docker compose up -d --scale api=N`), with the same CPU limit per container.
- Users: 10, 50, 100, 200, with a fixed spawn rate. Each run lasts at least 3 minutes. Discard the first 30 seconds as warm-up. Repeat each combination twice.
- Record per run: requests, failures and failure rate, requests per second, median, p95 and p99 latency, plus CPU and memory from `docker stats`.
- Prove load balancing: count how many responses came from each container hostname.
- Find the breaking point: the user count where failures start or p95 exceeds 2 seconds, for each replica count.
- Mixed scenario: run the load while a retraining job is active. Does prediction latency suffer?
- Upload scenario: a smaller flood against `/upload` to see the disk and CPU effect.

**Output:** a results table and two charts (latency vs replicas, throughput vs replicas) saved in `locust/results/` and pasted into the README, with a short interpretation of what scaling bought and where it stopped helping.

**Done when:** all runs are recorded, the table and charts are in the README, and the breaking point per replica count is stated.

---

## SECTION 14: README, rubric checklist, video script

**README must contain:** project description, the YouTube video link (placeholder), the live URL, setup steps (local and Docker), the deployment steps from Section 12, the repo structure, the dataset source and license, the model summary and test results, the experiment summary (ablation and sensitivity highlights, failure modes, robustness), the flood simulation results, and how retraining and the trigger work.

**Rubric self-check (write a table in the README or a separate file):**
- Video demo: prediction and retraining, camera on, clear sound.
- Retraining: upload, save to database, preprocessing, retrain from our custom model. Script and model file present.
- Prediction: insert a data point and show the correct class. Script and model file present.
- Evaluation: notebook with preprocessing, optimization techniques and at least 4 metrics.
- Deployment: UI on a public URL or Docker, with data insights visible.

**Video script (5 to 7 minutes, camera on):** intro, the data story from the insights page, a correct prediction using a pre-verified image, upload a batch, preprocessing and database record, retrain, the new metrics and promotion result, the Locust results, the monitoring page. Rehearse once and record in a single take.

**Final checks:** run the whole thing from a fresh clone, grep the repo for em-dashes and en-dashes, grep for stray comments, confirm both submission formats are ready (zip of the repository for the first attempt, GitHub URL for the second).

**Done when:** a stranger can follow the README to a working system, and every rubric line has a pointer to proof.
