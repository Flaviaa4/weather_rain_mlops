# CLAUDE.md

Project instructions for Claude Code. Sections 0 and 0B of CLAUDE_CODE_PLAYBOOK.md, followed for the whole project. The full playbook is in CLAUDE_CODE_PLAYBOOK.md; work one section at a time and wait for the user to say "next".

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

## Local setup notes (project state, not playbook text)

- Python env: `.venv/` (conda, Python 3.12). Run tools as `./.venv/python.exe -m ...` from the repo root.
- TensorFlow is not installed locally (user decision). TensorFlow code is smoke-tested in Colab, not on this machine. Local code that runs without TensorFlow (audit, preprocessing without tf.data, ONNX inference, API, UI) is tested locally.
- Disk on C: is nearly full (about 1.2 GB free on 2026-11-22). Install with `pip install --no-cache-dir`. Check `df -h /c` before any large install or Docker build.
- Data was cleaned in Section 1: 102 duplicate copies moved to `data/quarantine/` (see `experiments/audit/DUPLICATE_REPORT.md`). The original export stays untouched in `~/Documents/SummativeDataset`. Clean counts: train 719, valid 144, test 134.
