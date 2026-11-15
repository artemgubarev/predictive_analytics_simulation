import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from src.models_fabric import build_model, resolve_params

logger = logging.getLogger(__name__)

def _find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
    return best_threshold, best_f1

def _fit_fold(model, model_type: str, X_train, y_train, X_val, y_val):
    """Универсальный fit для разных библиотек. Возвращает best_iteration или None."""
    mt = model_type.lower()

    if mt == "catboost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        return model.best_iteration_

    if mt == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # у XGBClassifier early_stopping задаётся в конструкторе
        return getattr(model, "best_iteration", None)

    if mt == "lightgbm":
        # LightGBM ждёт callbacks для early stopping в новых версиях
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return model.best_iteration_

    # logreg — без early stopping
    model.fit(X_train, y_train)
    return None

def _final_fit(model, model_type: str, X, y):
    mt = model_type.lower()
    if mt in ("catboost", "xgboost", "lightgbm"):
        model.fit(X, y)
    else:
        model.fit(X, y)
    return model

def train(df_features: pd.DataFrame, config: dict):
    logger.info("=== Train started ===")

    cfg = config["train"]
    model_type = cfg.get("model_type", "catboost").lower()
    logger.info("Model type: %s", model_type)

    # --- 1. Данные ---
    drop_cols = ["target", "machineID", "datetime"]
    feature_cols = [c for c in df_features.columns if c not in drop_cols]

    X = df_features[feature_cols]
    y = df_features["target"].astype(int)
    groups = df_features["machineID"]

    logger.info("Признаков: %d | Строк: %d | Positive rate: %.4f",
                len(feature_cols), len(X), y.mean())

    # --- 2. Параметры ---
    params = resolve_params(model_type, cfg)
    
    # mlflow.log_params({f"{model_type}_{k}": str(v) for k, v in params.items()})
    # mlflow.log_param("n_splits", cfg["n_splits"])
    # mlflow.log_param("n_features", len(feature_cols))
    # mlflow.log_param("model_type", model_type)

    # --- 3. CV ---
    gkf = GroupKFold(n_splits=cfg["n_splits"])
    oof_preds = pd.Series(0.0, index=X.index)

    pr_auc_scores, f1_scores, thresholds, best_iterations = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pos_count = max(int((y_train == 1).sum()), 1)
        neg_count = int((y_train == 0).sum())
        spw = min(neg_count / pos_count, 100.0) if model_type != "logreg" else 1.0

        model = build_model(model_type, params, scale_pos_weight=spw)
        best_iter = _fit_fold(model, model_type, X_train, y_train, X_val, y_val)

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds.iloc[val_idx] = preds

        pr_auc = average_precision_score(y_val, preds)
        best_thr, best_f1 = _find_best_threshold(y_val, preds)
        y_pred_bin = (preds > best_thr).astype(int)
        rec = recall_score(y_val, y_pred_bin, zero_division=0)
        prec = precision_score(y_val, y_pred_bin, zero_division=0)

        logger.info(
            "Fold %d | best_iter=%s | PR-AUC=%.4f | F1=%.4f | Rec=%.4f | Prec=%.4f | thr=%.3f",
            fold, best_iter, pr_auc, best_f1, rec, prec, best_thr,
        )

        pr_auc_scores.append(pr_auc)
        f1_scores.append(best_f1)
        thresholds.append(best_thr)
        if best_iter is not None:
            best_iterations.append(best_iter)

    # --- 4. Метрики ---
    mean_pr_auc = float(np.mean(pr_auc_scores))
    mean_f1 = float(np.mean(f1_scores))
    oof_pr_auc = float(average_precision_score(y, oof_preds))
    final_threshold = float(np.mean(thresholds))

    # mlflow.log_metric("cv_pr_auc_mean", mean_pr_auc)
    # mlflow.log_metric("cv_pr_auc_std", float(np.std(pr_auc_scores)))
    # mlflow.log_metric("cv_f1_mean", mean_f1)
    # mlflow.log_metric("oof_pr_auc", oof_pr_auc)

    logger.info("CV PR-AUC=%.4f (±%.4f) | CV F1=%.4f | OOF PR-AUC=%.4f | thr=%.3f",
                mean_pr_auc, float(np.std(pr_auc_scores)), mean_f1,
                oof_pr_auc, final_threshold)

    # --- 5. Финальная модель ---
    final_params = dict(params)
    # для бустингов подставляем усреднённый best_iter
    if best_iterations:
        avg_best_iter = int(np.mean(best_iterations))
        if model_type in ("catboost",):
            final_params["iterations"] = avg_best_iter
        elif model_type in ("xgboost", "lightgbm"):
            final_params["n_estimators"] = avg_best_iter
        final_params.pop("early_stopping_rounds", None)
    else:
        final_params.pop("early_stopping_rounds", None)

    total_pos = max(int((y == 1).sum()), 1)
    total_neg = int((y == 0).sum())
    total_spw = min(total_neg / total_pos, 100.0) if model_type != "logreg" else 1.0

    logger.info("Обучение финальной модели на всех данных...")
    final_model = build_model(model_type, final_params, scale_pos_weight=total_spw)
    final_model = _final_fit(final_model, model_type, X, y)

    # --- 6. Логирование в MLflow ---
    if model_type == "catboost":
        mlflow.catboost.log_model(final_model, artifact_path="model")
    elif model_type == "xgboost":
        mlflow.xgboost.log_model(final_model, artifact_path="model")
    elif model_type == "lightgbm":
        mlflow.lightgbm.log_model(final_model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(final_model, artifact_path="model")

    logger.info("=== Train finished ===")
    return final_model, final_threshold