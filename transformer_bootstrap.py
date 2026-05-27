from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "news_transformer"
MODEL_PATH = MODEL_DIR / "model.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CONFIG_PATH = MODEL_DIR / "config.json"
BASE_TRAIN_PATH = BASE_DIR / "data" / "news_train.jsonl"
GENERATED_TRAIN_PATH = BASE_DIR / "data" / "news_train_generated.jsonl"


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(primary: str, fallback: str, default: str) -> str:
    value = os.getenv(primary) or os.getenv(fallback) or default
    return value.strip() or default


def transformer_artifacts_ready() -> bool:
    return MODEL_PATH.exists() and VOCAB_PATH.exists() and CONFIG_PATH.exists()


def _training_data_paths() -> list[Path]:
    return [path for path in [BASE_TRAIN_PATH, GENERATED_TRAIN_PATH] if path.exists()]


def ensure_news_transformer_model() -> bool:
    if not env_flag("NEWS_BOOTSTRAP_TRAIN_ENABLED", True):
        print("Transformer bootstrap training is disabled.")
        return transformer_artifacts_ready()

    force_retrain = env_flag("NEWS_BOOTSTRAP_FORCE_RETRAIN", False)
    if transformer_artifacts_ready() and not force_retrain:
        print("Transformer model artifacts already exist.")
        return True

    data_paths = _training_data_paths()
    if not data_paths:
        print("No transformer training data found; keyword classifier will be used.")
        return False

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    epochs = _env_value("NEWS_BOOTSTRAP_TRAIN_EPOCHS", "NEWS_TRAIN_EPOCHS", "4")
    batch_size = _env_value("NEWS_BOOTSTRAP_TRAIN_BATCH_SIZE", "NEWS_TRAIN_BATCH_SIZE", "8")

    command = [
        sys.executable,
        "train_transformer_news.py",
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
    ]
    for path in data_paths:
        command.extend(["--data-path", str(path)])

    print("Transformer model artifacts are missing; starting bootstrap training.")
    completed = subprocess.run(command, cwd=BASE_DIR, check=False)
    if completed.returncode != 0:
        print(f"Transformer bootstrap training failed with exit code {completed.returncode}.")
        return False

    if transformer_artifacts_ready():
        print("Transformer bootstrap training completed.")
        return True

    print("Transformer bootstrap training finished without complete artifacts.")
    return False
