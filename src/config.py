import os
from pathlib import Path


def env_path(name, default):
    return Path(os.environ.get(name, default)).resolve()


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_list(name, default):
    raw = os.environ.get(name)
    return [item.strip() for item in raw.split(",")] if raw else list(default)


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = env_path("DATA_DIR", ROOT_DIR / "data")
MODELS_DIR = env_path("MODELS_DIR", ROOT_DIR / "models")
EXPERIMENTS_DIR = env_path("EXPERIMENTS_DIR", ROOT_DIR / "experiments")
UPLOADS_DIR = env_path("UPLOADS_DIR", DATA_DIR / "uploads")
QUARANTINE_DIR = env_path("QUARANTINE_DIR", DATA_DIR / "quarantine")
AUDIT_DIR = EXPERIMENTS_DIR / "audit"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
DB_PATH = env_path("DB_PATH", DATA_DIR / "weather.db")

SPLITS = ["train", "valid", "test"]
CLASS_NAMES = env_list("CLASS_NAMES", ["cloudy", "rainy", "shiny", "sunrise"])
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_SIZE = env_int("IMAGE_SIZE", 128)
CHANNELS = 3
BATCH_SIZE = env_int("BATCH_SIZE", 32)
SEEDS = [int(s) for s in env_list("SEEDS", ["11", "22", "33", "44", "55"])]

MAX_EPOCHS = env_int("MAX_EPOCHS", 150)
EARLY_STOP_PATIENCE = env_int("EARLY_STOP_PATIENCE", 15)
LR_PATIENCE = env_int("LR_PATIENCE", 5)
LR_FACTOR = env_float("LR_FACTOR", 0.5)
MIN_LR = env_float("MIN_LR", 1e-6)
# optional keys keep the Section 3 config hashes unchanged
OPTIONAL_CONFIG_DEFAULTS = {"bn_momentum": 0.99, "warmup_epochs": 0, "min_epochs": 0}


def config_setting(cfg, key):
    return cfg.get(key, OPTIONAL_CONFIG_DEFAULTS[key])

DHASH_SIZE = env_int("DHASH_SIZE", 8)
DUPLICATE_MAX_DISTANCE = env_int("DUPLICATE_MAX_DISTANCE", 4)
THUMB_SIZE = env_int("THUMB_SIZE", 32)
DUPLICATE_MAX_MAE = env_float("DUPLICATE_MAX_MAE", 0.07)
DUPLICATE_MIN_CORR = env_float("DUPLICATE_MIN_CORR", 0.98)
CORR_STAGE_MIN_CORR = env_float("CORR_STAGE_MIN_CORR", 0.94)
CORR_STAGE_MAX_MAE = env_float("CORR_STAGE_MAX_MAE", 0.10)

AUGMENT_STRENGTH = os.environ.get("AUGMENT_STRENGTH", "medium")
AUGMENT_STRENGTHS = {
    "none": None,
    "light": {"flip": True, "zoom": 0.05, "brightness": 0.05, "contrast": 0.10, "saturation": 0.10, "hue": 0.0},
    "medium": {"flip": True, "zoom": 0.10, "brightness": 0.10, "contrast": 0.20, "saturation": 0.20, "hue": 0.0},
    "strong": {"flip": True, "zoom": 0.20, "brightness": 0.20, "contrast": 0.35, "saturation": 0.20, "hue": 0.0},
}
EDA_SIZES = [96, 128, 160]
EDA_REFERENCE_SIZE = env_int("EDA_REFERENCE_SIZE", 224)
SAMPLE_SEED = env_int("SAMPLE_SEED", 11)

UNCERTAIN_THRESHOLD = env_float("UNCERTAIN_THRESHOLD", 0.6)
RETRAIN_UPLOAD_THRESHOLD = env_int("RETRAIN_UPLOAD_THRESHOLD", 40)
MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 10)
MIN_FREE_DISK_MB = env_int("MIN_FREE_DISK_MB", 500)
ONNX_THREADS = env_int("ONNX_THREADS", 1)

MODEL_NAME = os.environ.get("MODEL_NAME", "placeholder_model")
IS_PLACEHOLDER = MODEL_NAME.startswith("placeholder")
H5_PATH = MODELS_DIR / f"{MODEL_NAME}.h5"
ONNX_PATH = MODELS_DIR / f"{MODEL_NAME}.onnx"
METRICS_PATH = MODELS_DIR / (f"{MODEL_NAME}_metrics.json" if IS_PLACEHOLDER else "metrics.json")
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
ARCHIVE_DIR = MODELS_DIR / "archive"
CANDIDATE_DIR = MODELS_DIR / "candidate"

MAX_ZIP_MB = env_int("MAX_ZIP_MB", 200)
MAX_FILES_PER_UPLOAD = env_int("MAX_FILES_PER_UPLOAD", 500)
RETRAIN_AUTO = os.environ.get("RETRAIN_AUTO", "1") == "1"
RETRAIN_LEARNING_RATE = env_float("RETRAIN_LEARNING_RATE", 1e-4)
RETRAIN_MAX_EPOCHS = env_int("RETRAIN_MAX_EPOCHS", 10)
RETRAIN_PATIENCE = env_int("RETRAIN_PATIENCE", 3)
RETRAIN_SEED = env_int("RETRAIN_SEED", 11)
PROMOTION_TOLERANCE = env_float("PROMOTION_TOLERANCE", 0.0)
EDA_FEATURES_CSV = EXPERIMENTS_DIR / "eda" / "features_train_valid.csv"
INSIGHTS_CACHE = env_path("INSIGHTS_CACHE", DATA_DIR / "cache" / "features_train_valid.csv")
