from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
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
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Отключаем логирование каждого шага Optuna, чтобы не засорять консоль
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
            #"eval_metric": "PRAUC",
            "eval_metric": "Logloss",
            "random_seed": cfg["random_state"],
            "thread_count": -1,
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": cfg.get("early_stopping_rounds", 100),
            
            "task_type": "GPU",  
            "devices": "1",
        }

    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": cfg["random_state"],
            "n_jobs": -1,
            "early_stopping_rounds": cfg.get("early_stopping_rounds", 100),
            "device": "cuda:0"
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
            "n_jobs": -1,
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

    raise ValueError(f"Unknown model_type: {model_type}")


def _build_model(model_type: str, params: dict, scale_pos_weight: float = 1.0) -> Any:
    """Создаёт модель с заданными параметрами."""
    if model_type == "catboost":
        return CatBoostClassifier(**params, scale_pos_weight=scale_pos_weight)

    if model_type == "xgboost":
        return XGBClassifier(**params, scale_pos_weight=scale_pos_weight)

    if model_type == "lightgbm":
        return LGBMClassifier(**params, scale_pos_weight=scale_pos_weight)

    if model_type == "logreg":
        # Логрег чувствителен к масштабу → pipeline + imputer
        clean_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**clean_params)),
        ])

    raise ValueError(f"Unknown model_type: {model_type}")


def _fit_model(model: Any, model_type: str, X_train, y_train, X_val, y_val):
    """Универсальный вызов fit для разных библиотек."""
    if model_type == "catboost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    elif model_type == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
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

    # Балансировка классов
    pos_count = max(int((y == 1).sum()), 1)
    neg_count = int((y == 0).sum())
    spw = min(neg_count / pos_count, 100.0) if model_type != "logreg" else 1.0

    gkf = GroupKFold(n_splits=cfg["n_splits"])
    oof_preds = np.zeros(len(X))
    pr_auc_scores = []

    # Вложенный запуск MLflow для этого trial
    with mlflow.start_run(nested=True, run_name=f"{model_type}_trial_{trial.number}"):
        mlflow.log_params({f"param_{k}": str(v) for k, v in params.items()})
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("trial_number", trial.number)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = _build_model(model_type, params, scale_pos_weight=spw)
            model = _fit_model(model, model_type, X_train, y_train, X_val, y_val)

            preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = preds
            pr_auc_scores.append(average_precision_score(y_val, preds))

        mean_pr_auc = float(np.mean(pr_auc_scores))
        oof_pr_auc = float(average_precision_score(y, oof_preds))
        
        # Логируем метрики в дочерний run
        mlflow.log_metric("cv_pr_auc_mean", mean_pr_auc)
        mlflow.log_metric("cv_pr_auc_std", float(np.std(pr_auc_scores)))
        mlflow.log_metric("oof_pr_auc", oof_pr_auc)

    return oof_pr_auc


def train_optuna(
    df_features: pd.DataFrame,
    config: dict,
    model_types: list[str] | None = None,
    n_trials: int = 20,
) -> tuple[Any, float, dict[str, Any]]:
    """
    Основная функция: ищет лучшие гиперпараметры для каждой модели,
    выбирает лучшую по OOF PR-AUC и обучает финальную модель.
    """
    logger.info("=== Train (Optuna) started ===")
    cfg = config["train"]

    # 1. Подготовка данных
    drop_cols = ["target", "machineID", "datetime"]
    feature_cols = [c for c in df_features.columns if c not in drop_cols]

    X = df_features[feature_cols]
    y = df_features["target"].astype(int)
    groups = df_features["machineID"]

    logger.info("Признаков: %d | Строк: %d | Positive rate: %.4f",
                len(feature_cols), len(X), y.mean())

    # 2. Определяем, какие модели искать
    if model_types is None:
        model_types = cfg.get("model_types", [cfg.get("model_type", "catboost")])
    if isinstance(model_types, str):
        model_types = [model_types]

    logger.info("Поиск моделей: %s | n_trials на модель: %d", model_types, n_trials)

    best_results = {}

    # 3. Перебираем модели
    for mt in model_types:
        logger.info("--- Поиск гиперпараметров для %s ---", mt)
        
        # Родительский запуск для исследования (study)
        with mlflow.start_run(run_name=f"{mt}_study", nested=True):
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=cfg["random_state"]),
            )
            study.optimize(
                lambda trial: _objective(trial, mt, X, y, groups, cfg),
                n_trials=n_trials,
                show_progress_bar=True,
            )

            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value

            # Логируем лучшие параметры в родительский run
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_oof_pr_auc", best_score)
            
            # Сохраняем ID лучшего trial, чтобы можно было найти его артефакты
            mlflow.set_tag("best_trial_run_id", best_trial.user_attrs.get("mlflow_run_id", ""))

            logger.info("Лучший OOF PR-AUC для %s: %.4f", mt, best_score)
            logger.info("Лучшие параметры: %s", best_params)

            best_results[mt] = {
                "params": best_params,
                "score": best_score,
                "trial": best_trial,
            }

    # 4. Выбираем лучшую модель по OOF PR-AUC
    best_model_type = max(best_results, key=lambda k: best_results[k]["score"])
    best_params = best_results[best_model_type]["params"]
    logger.info("Победитель: %s (OOF PR-AUC=%.4f)", best_model_type, best_results[best_model_type]["score"])

    # 5. Подготовка финальной модели
    # Для финального обучения убираем early_stopping_rounds
    final_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
    
    # Усредняем best_iteration по фолдам, если это возможно
    # (оптимальнее было бы брать из лучшего trial, но для простоты оставим как есть)
    total_pos = max(int((y == 1).sum()), 1)
    total_neg = int((y == 0).sum())
    total_spw = min(total_neg / total_pos, 100.0) if best_model_type != "logreg" else 1.0

    final_model = _build_model(best_model_type, final_params, scale_pos_weight=total_spw)
    logger.info("Обучение финальной модели %s на всех данных...", best_model_type)
    
    # Обучаем на всех данных
    if best_model_type == "catboost":
        final_model.fit(X, y)
    elif best_model_type == "xgboost":
        final_model.fit(X, y, verbose=False)
    elif best_model_type == "lightgbm":
        final_model.fit(X, y)
    else:
        final_model.fit(X, y)

    # 6. Подбираем порог на OOF предсказаниях лучшей модели
    # (нужно пересчитать OOF для лучшей модели с лучшими параметрами)
    gkf = GroupKFold(n_splits=cfg["n_splits"])
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = _build_model(best_model_type, final_params, scale_pos_weight=total_spw)
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

        if best_model_type == "catboost":
            mlflow.catboost.log_model(final_model, artifact_path="model")
        elif best_model_type == "xgboost":
            mlflow.xgboost.log_model(final_model, artifact_path="model")
        elif best_model_type == "lightgbm":
            mlflow.lightgbm.log_model(final_model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(final_model, artifact_path="model")

    logger.info("=== Train (Optuna) finished ===")
    return final_model, final_threshold, {
        "model_type": best_model_type,
        "params": best_params,
        "oof_pr_auc": best_results[best_model_type]["score"],
        "threshold": final_threshold,
    }