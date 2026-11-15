import logging
import tempfile
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _plot_pr_curve(y_true, y_prob, pr_auc, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", linewidth=2)
    ax.axhline(baseline, linestyle="--", color="grey",
               label=f"Baseline = {baseline:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _plot_confusion_matrix(y_true, y_pred, threshold, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Health", "Failure"])
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _plot_score_distribution(y_true, y_prob, threshold, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.5,
            label=f"Health (0), n={int((y_true == 0).sum())}", color="steelblue")
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6,
            label=f"Failure (1), n={int((y_true == 1).sum())}", color="crimson")
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _plot_threshold_sweep(y_true, y_prob, best_threshold, out_path):
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s = [], [], []
    for thr in thresholds:
        y_pred = (y_prob > thr).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    ax.axvline(best_threshold, color="black", linestyle="--",
               label=f"Current = {best_threshold:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs. Threshold")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _log_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    Строит classification_report в двух форматах:
      - текстовый (красиво для логов и артефакта .txt)
      - dict (для логирования per-class метрик в MLflow)
    Возвращает dict с метриками.
    """
    target_names = ["Health (0)", "Failure (1)"]

    # Текстовая версия — для артефакта и логов
    report_text = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report_text)

    # Dict версия — для MLflow
    report_dict = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Сохраняем .txt
    txt_path = out_dir / "classification_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")

    # Сохраняем .csv (DataFrame-формат — удобно открывать в Excel)
    report_df = pd.DataFrame(report_dict).T
    csv_path = out_dir / "classification_report.csv"
    report_df.to_csv(csv_path)

    # Per-class метрики в MLflow
    for class_name in target_names:
        class_key = class_name.split()[0].lower()  # "Health (0)" -> "health"
        metrics = report_dict[class_name]
        mlflow.log_metric(f"test_{class_key}_precision", metrics["precision"])
        mlflow.log_metric(f"test_{class_key}_recall", metrics["recall"])
        mlflow.log_metric(f"test_{class_key}_f1", metrics["f1-score"])
        mlflow.log_metric(f"test_{class_key}_support", metrics["support"])

    # Макро/взвешенные средние
    mlflow.log_metric("test_macro_avg_f1", report_dict["macro avg"]["f1-score"])
    mlflow.log_metric("test_weighted_avg_f1", report_dict["weighted avg"]["f1-score"])
    mlflow.log_metric("test_accuracy", report_dict["accuracy"])

    return report_dict


def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
    best_threshold: float = 0.5,
) -> dict:
    """
    Полная оценка модели: метрики + графики + classification_report.
    """
    logger.info("=== Evaluate started ===")

    # --- Валидация ---
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Длины X_test ({len(X_test)}) и y_test ({len(y_test)}) не совпадают."
        )

    y_test = y_test.astype(int).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "В тесте только один класс (pos=%d, neg=%d).", n_pos, n_neg,
        )

    logger.info(
        "Test size: %d | positives: %d (%.4f) | negatives: %d",
        len(y_test), n_pos, n_pos / len(y_test), n_neg,
    )

    # --- Предсказания ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > best_threshold).astype(int)

    # --- Основные метрики ---
    metrics = {
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "threshold": float(best_threshold),
    }
    if n_pos > 0 and n_neg > 0:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    for name, value in metrics.items():
        mlflow.log_metric(f"test_{name}", value)

    logger.info(
        "Test | PR-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f | thr=%.3f",
        metrics["pr_auc"], metrics["f1"],
        metrics["recall"], metrics["precision"], metrics["threshold"],
    )

    # --- Артефакты ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        logger.info("Сборка артефактов в: %s", tmp)

        # Графики
        _plot_pr_curve(y_test, y_prob, metrics["pr_auc"], tmp / "pr_curve.png")
        _plot_confusion_matrix(y_test, y_pred, best_threshold, tmp / "confusion_matrix.png")
        _plot_score_distribution(y_test, y_prob, best_threshold, tmp / "score_distribution.png")
        _plot_threshold_sweep(y_test, y_prob, best_threshold, tmp / "threshold_sweep.png")

        # Classification report (txt + csv + MLflow per-class metrics)
        report_dict = _log_classification_report(y_test, y_pred, tmp)
        metrics["classification_report"] = report_dict

        # Конфиг
        config_out = {
            **config,
            "evaluation": {
                "best_threshold": float(best_threshold),
                "test_metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
                "test_size": len(y_test),
                "test_positives": n_pos,
                "test_negatives": n_neg,
            },
        }
        with open(tmp / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_out, f, sort_keys=False, allow_unicode=True)

        mlflow.log_artifacts(str(tmp), artifact_path="evaluation_artifacts")
        logger.info("Артефакты сохранены")

    logger.info("=== Evaluate finished ===")
    return metrics
