import logging
from typing import Any
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

def resolve_params(model_type: str, cfg: dict) -> dict:
    """Собирает итоговые параметры: common + <model_type> из конфига + служебные."""
    mt = model_type.lower()
    common = cfg.get("common", {})
    specific = cfg.get(mt, {})

    # базовые «служебные» параметры, которые есть у всех
    base = {
        "random_state": cfg["random_state"],
        "learning_rate": common.get("learning_rate", 0.05),
        "n_estimators": common.get("n_estimators", 1000),
    }

    # маппинг общих имён в имена конкретной библиотеки
    if mt == "catboost":
        base.update({
            "iterations": base.pop("n_estimators"),
            "depth": common.get("depth", 6),
            "l2_leaf_reg": common.get("l2_leaf_reg", 3),
            "loss_function": "Logloss",
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "early_stopping_rounds": cfg["early_stopping_rounds"],
        })
    elif mt == "xgboost":
        base.update({
            "max_depth": common.get("depth", 6),
            "reg_lambda": common.get("l2_leaf_reg", 1.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "early_stopping_rounds": cfg["early_stopping_rounds"],
        })
    elif mt == "lightgbm":
        base.update({
            "max_depth": common.get("depth", -1),
            "reg_lambda": common.get("l2_leaf_reg", 0.0),
            "objective": "binary",
            "metric": "average_precision",
            "n_jobs": -1,
            "verbose": -1,
        })
    elif mt == "logreg":
        base.update({
            "max_iter": 2000,
            "solver": "lbfgs",
            "class_weight": "balanced",
        })
    else:
        raise ValueError(f"Unknown model_type: {mt}")

    # специфичные параметры из конфига ПЕРЕОПРЕДЕЛЯЮТ всё
    base.update(specific)
    return base

def build_model(model_type: str, params: dict, scale_pos_weight: float = 1.0) -> Any:
    """Создаёт модель по имени. Все модели поддерживают predict_proba."""
    model_type = model_type.lower()

    if model_type == "catboost":
        return CatBoostClassifier(**params, scale_pos_weight=scale_pos_weight)

    if model_type == "xgboost":
        # XGBoost: scale_pos_weight — родной параметр
        return XGBClassifier(**params, scale_pos_weight=scale_pos_weight)

    if model_type == "lightgbm":
        return LGBMClassifier(**params, scale_pos_weight=scale_pos_weight)

    if model_type == "logreg":
        # Логрег чувствителен к масштабу → pipeline + imputer
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])

    raise ValueError(f"Unknown model_type: {model_type}")