from __future__ import annotations

import argparse
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.preprocess import preprocess
from src.evaluate import evaluate, evaluate_lstm
from src.train_lstm import train_lstm
from src.train_optuna import train_optuna

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Logging-конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.config.dictConfig(cfg)


def resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return str(p)


def resolve_tracking_uri(uri: str) -> str:
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db_path = uri.replace("sqlite:///", "", 1)
        abs_path = (BASE_DIR / db_path).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{abs_path}"
    return uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PdM MLflow pipeline")
    parser.add_argument("--config", default=os.environ.get("PIPELINE_CONFIG", str(BASE_DIR / "configs" / "config.yaml")))
    parser.add_argument("--logging", default=os.environ.get("PIPELINE_LOGGING", str(BASE_DIR / "configs" / "logging.yaml")))
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    return parser.parse_args()


def run_pipeline(
    config_path: Path,
    experiment_override: Optional[str] = None,
    tracking_uri_override: Optional[str] = None,
) -> None:
    cfg = load_config(config_path)

    if experiment_override:
        cfg["experiment_name"] = experiment_override
    if tracking_uri_override:
        cfg["tracking_uri"] = tracking_uri_override

    cfg["tracking_uri"] = resolve_tracking_uri(cfg["tracking_uri"])
    cfg["preprocess"]["telemetry_path"] = resolve_path(cfg["preprocess"]["telemetry_path"])
    cfg["preprocess"]["failures_path"] = resolve_path(cfg["preprocess"]["failures_path"])

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("Experiment: %s", cfg["experiment_name"])
    logger.info("CWD: %s | BASE_DIR: %s", os.getcwd(), BASE_DIR)

    seed = cfg.get("train", {}).get("random_state", 42)
    np.random.seed(seed)

    with mlflow.start_run(run_name="pdm_run") as run:
        logger.info("MLflow run_id=%s", run.info.run_id)

        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("config_file", str(config_path))

        df_features = preprocess(config=cfg)
        mlflow.log_metric("total_rows", len(df_features))
        mlflow.log_metric("total_features", df_features.shape[1])

        test_size = cfg.get("pipeline", {}).get("test_size", 0.2)
        split_seed = cfg.get("pipeline", {}).get("split_seed", seed)

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
        train_idx, test_idx = next(gss.split(df_features, groups=df_features["machineID"]))

        df_train = df_features.iloc[train_idx].copy()
        df_test = df_features.iloc[test_idx].copy()

        train_machines = df_train["machineID"].nunique()
        test_machines = df_test["machineID"].nunique()
        train_pos_rate = float(df_train["target"].mean())
        test_pos_rate = float(df_test["target"].mean())

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("split_seed", split_seed)
        mlflow.log_metric("train_rows", len(df_train))
        mlflow.log_metric("test_rows", len(df_test))
        mlflow.log_metric("train_machines", train_machines)
        mlflow.log_metric("test_machines", test_machines)
        mlflow.log_metric("train_positive_rate", train_pos_rate)
        mlflow.log_metric("test_positive_rate", test_pos_rate)

        logger.info(
            "Split | Train: %d rows (%d machines, pos=%.4f) | Test: %d rows (%d machines, pos=%.4f)",
            len(df_train), train_machines, train_pos_rate,
            len(df_test), test_machines, test_pos_rate,
        )

        model_types = cfg["train"].get("model_types", ["catboost"])
        use_lstm = "lstm" in model_types
        
        if use_lstm:
            logger.info("Выбрана модель LSTM (BiGRU + Focal Loss).")
            # Теперь train_lstm возвращает 4 элемента
            model, best_threshold, scaler, train_meta = train_lstm(df_train, config=cfg)
        else:
            logger.info("Выбрана модель бустинга.")
            model, best_threshold, train_meta = train_optuna(df_train, config=cfg)

        if use_lstm:
            seq_length = cfg["train"].get("lstm", {}).get("seq_length", 24)
            evaluate_lstm(
                model=model,
                df_test=df_test,
                config=cfg,
                best_threshold=best_threshold,
                seq_length=seq_length,
                scaler=scaler,  # <--- Передаем scaler
            )
        else:
            drop_cols = [c for c in ("target", "machineID", "datetime") if c in df_test.columns]
            X_test = df_test.drop(columns=drop_cols)
            y_test = df_test["target"]
            evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
                config=cfg,
                best_threshold=best_threshold,
            )

        mlflow.set_tag("pipeline_status", "completed")

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    args = parse_args()
    setup_logging(Path(args.logging))
    try:
        run_pipeline(
            Path(args.config),
            experiment_override=args.experiment,
            tracking_uri_override=args.tracking_uri,
        )
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)