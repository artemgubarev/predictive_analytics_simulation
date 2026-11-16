from __future__ import annotations

import logging
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Пытаемся импортировать TabPFN
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

logger = logging.getLogger(__name__)

# Отключаем логирование каждого шага Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    """Подбирает порог, максимизирующий F1."""
    best_threshold, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
    return best_threshold, best_f1


def _suggest_params(trial: optuna.Trial, model_type: str, cfg: dict) -> dict:
    """Предлагает гиперпараметры для конкретной модели через Optuna."""
    if model_type == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "loss_function": "Logloss",
            "random_seed": cfg["random_state"],
            "thread_count": -1,
            "allow_writing_files": False,
            "early_stopping_rounds": cfg.get("early_stopping_rounds", 100),
            "task_type": "GPU",  
            "devices": "1", 
        }

    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "binary",
            "metric": "average_precision",
            "random_state": cfg["random_state"],
            "n_jobs": 8,
            "verbose": -1,
        }

    if model_type == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "max_iter": 2000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": cfg["random_state"],
        }

    if model_type == "tabpfn":
        return {}

    raise ValueError(f"Unknown model_type: {model_type}")


def _build_model(model_type: str, params: dict, scale_pos_weight: float = 1.0) -> Any:
    """Создаёт модель с заданными параметрами."""
    if model_type == "catboost":
        return CatBoostClassifier(**params, scale_pos_weight=scale_pos_weight)
    
    if model_type == "lightgbm":
        return LGBMClassifier(**params, scale_pos_weight=scale_pos_weight)
    
    if model_type == "logreg":
        clean_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**clean_params)),
        ])
        
    if model_type == "tabpfn":
        if not HAS_TABPFN:
            raise ImportError("Установи tabpfn: pip install tabpfn")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNClassifier(device=device, **params)
        
        # Определяем устройство
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # TabPFN не принимает scale_pos_weight. N_estimators можно оставить по умолчанию.
        return TabPFNClassifier(device=device)

    raise ValueError(f"Unknown model_type: {model_type}")


def _fit_model(model: Any, model_type: str, X_train, y_train, X_val, y_val):
    """Универсальный вызов fit для разных библиотек."""
    if model_type == "catboost":
        model.fit(
            X_train, y_train, 
            eval_set=(X_val, y_val), 
            use_best_model=True,
            logging_level="Silent"
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_X=X_val, eval_y=y_val,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
    elif model_type == "tabpfn":
        # TabPFN просто обучается на train
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def _objective(
    trial: optuna.Trial,
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cfg: dict,
) -> float:
    """Целевая функция Optuna: возвращает OOF PR-AUC."""
    params = _suggest_params(trial, model_type, cfg)

    # TabPFN и LogReg не поддерживают scale_pos_weight
    if model_type not in ["logreg", "tabpfn"]:
        spw = trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True)
    else:
        spw = 1.0

    gkf = GroupKFold(n_splits=cfg["n_splits"])
    oof_preds = np.zeros(len(X))
    pr_auc_scores = []

    with mlflow.start_run(nested=True, run_name=f"{model_type}_trial_{trial.number}"):
        mlflow.log_params({f"param_{k}": str(v) for k, v in params.items()})
        mlflow.log_param("scale_pos_weight", spw)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("trial_number", trial.number)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Ограничиваем выборку для TabPFN, чтобы не было OOM
            if model_type == "tabpfn":
                max_samples = cfg.get("tabpfn_max_samples", 15000)
                if len(X_train) > max_samples:
                    # Берем случайную подвыборку
                    idx = np.random.RandomState(cfg["random_state"]).choice(
                        len(X_train), max_samples, replace=False
                    )
                    X_train = X_train.iloc[idx]
                    y_train = y_train.iloc[idx]

            # Применяем SMOTE только для Train (кроме logreg и tabpfn)
            if model_type not in ["logreg", "tabpfn"]:
                try:
                    smote = SMOTE(random_state=cfg["random_state"])
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                except ValueError:
                    X_train_res, y_train_res = X_train, y_train
            else:
                X_train_res, y_train_res = X_train, y_train

            model = _build_model(model_type, params, scale_pos_weight=spw)
            model = _fit_model(model, model_type, X_train_res, y_train_res, X_val, y_val)

            preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = preds
            pr_auc_scores.append(average_precision_score(y_val, preds))

        mean_pr_auc = float(np.mean(pr_auc_scores))
        oof_pr_auc = float(average_precision_score(y, oof_preds))
        
        mlflow.log_metric("cv_pr_auc_mean", mean_pr_auc)
        mlflow.log_metric("cv_pr_auc_std", float(np.std(pr_auc_scores)))
        mlflow.log_metric("oof_pr_auc", oof_pr_auc)

    return oof_pr_auc


def train_optuna(
    df_features: pd.DataFrame,
    config: dict,
    model_types: list[str] | None = None,
    n_trials: int | None = None, 
) -> tuple[Any, float, dict[str, Any]]:
    """Основная функция поиска и обучения."""
    logger.info("=== Train (Optuna) started ===")
    cfg = config["train"]

    drop_cols = ["target", "machineID", "datetime"]
    feature_cols = [c for c in df_features.columns if c not in drop_cols]

    X = df_features[feature_cols]
    y = df_features["target"].astype(int)
    groups = df_features["machineID"]

    logger.info("Признаков: %d | Строк: %d | Positive rate: %.4f",
                len(feature_cols), len(X), y.mean())

    if model_types is None:
        model_types = cfg.get("model_types", [cfg.get("model_type", "catboost")])
    if isinstance(model_types, str):
        model_types = [model_types]
        
    if n_trials is None:
        n_trials = int(cfg.get("n_trials", 20))

    logger.info("Поиск моделей: %s | n_trials на модель: %d", model_types, n_trials)

    best_results = {}

    # 3. Перебираем модели
    for mt in model_types:
        logger.info("--- Поиск гиперпараметров для %s ---", mt)
        
        # Для TabPFN не нужен большой n_trials (он zero-shot), ставим 1
        current_n_trials = 1 if mt == "tabpfn" else n_trials
        
        with mlflow.start_run(run_name=f"{mt}_study", nested=True):
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=cfg["random_state"]),
            )
            
            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = Path(tmpdir) / f"{mt}_train_log.txt"
                
                with open(log_path, "w", encoding="utf-8") as f:
                    with redirect_stdout(f):
                        study.optimize(
                            lambda trial: _objective(trial, mt, X, y, groups, cfg),
                            n_trials=current_n_trials,
                            show_progress_bar=False,
                        )
                
                mlflow.log_artifact(str(log_path), artifact_path="logs")

            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_oof_pr_auc", best_score)
            mlflow.set_tag("best_trial_run_id", best_trial.user_attrs.get("mlflow_run_id", ""))

            logger.info("Лучший OOF PR-AUC для %s: %.4f", mt, best_score)
            logger.info("Лучшие параметры: %s", best_params)

            best_results[mt] = {
                "params": best_params,
                "score": best_score,
                "trial": best_trial,
            }

    # 4. Выбираем лучшую модель
    best_model_type = max(best_results, key=lambda k: best_results[k]["score"])
    best_params = best_results[best_model_type]["params"]
    logger.info("Победитель: %s (OOF PR-AUC=%.4f)", best_model_type, best_results[best_model_type]["score"])

    # 5. Подготовка финальной модели
    final_params = {
        k: v for k, v in best_params.items() 
        if k not in ["early_stopping_rounds", "scale_pos_weight"]
    }
    
    final_spw = float(best_params.get("scale_pos_weight", 1.0))

    # Для финальной модели TabPFN тоже ограничиваем данные
    if best_model_type == "tabpfn":
        max_samples = cfg.get("tabpfn_max_samples", 15000)
        if len(X) > max_samples:
            idx = np.random.RandomState(cfg["random_state"]).choice(len(X), max_samples, replace=False)
            X_final = X.iloc[idx]
            y_final = y.iloc[idx]
        else:
            X_final = X
            y_final = y
    else:
        X_final = X
        y_final = y

    final_model = _build_model(best_model_type, final_params, scale_pos_weight=final_spw)
    logger.info("Обучение финальной модели %s на всех данных...", best_model_type)
    
    if best_model_type == "catboost":
        final_model.fit(X_final, y_final, verbose=False)
    elif best_model_type == "lightgbm":
        final_model.fit(X_final, y_final)
    else:
        final_model.fit(X_final, y_final)

    # 6. Подбираем порог на OOF предсказаниях
    gkf = GroupKFold(n_splits=cfg["n_splits"])
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Для TabPFN и здесь ограничиваем train
        if best_model_type == "tabpfn":
            max_samples = cfg.get("tabpfn_max_samples", 15000)
            if len(X_train) > max_samples:
                idx = np.random.RandomState(cfg["random_state"]).choice(len(X_train), max_samples, replace=False)
                X_train = X_train.iloc[idx]
                y_train = y_train.iloc[idx]

        model = _build_model(best_model_type, final_params, scale_pos_weight=final_spw)
        model = _fit_model(model, best_model_type, X_train, y_train, X_val, y_val)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    final_threshold, final_f1 = _find_best_threshold(y, oof_preds)
    logger.info("Финальный порог: %.3f (F1 на OOF=%.4f)", final_threshold, final_f1)

    # 7. Логируем финальную модель в MLflow
    with mlflow.start_run(run_name=f"final_{best_model_type}", nested=True):
        mlflow.log_params({f"final_{k}": v for k, v in final_params.items()})
        mlflow.log_param("final_model_type", best_model_type)
        mlflow.log_metric("final_threshold", final_threshold)
        mlflow.log_metric("final_oof_f1", final_f1)

        input_example = X.head(5)
        
        if best_model_type == "catboost":
            mlflow.catboost.log_model(final_model, artifact_path="model", input_example=input_example)
        elif best_model_type == "lightgbm":
            mlflow.lightgbm.log_model(final_model, artifact_path="model", input_example=input_example)
        else:
            mlflow.sklearn.log_model(final_model, artifact_path="model", input_example=input_example)

    logger.info("=== Train (Optuna) finished ===")
    return final_model, final_threshold, {
        "model_type": best_model_type,
        "params": best_params,
        "oof_pr_auc": best_results[best_model_type]["score"],
        "threshold": final_threshold,
    }