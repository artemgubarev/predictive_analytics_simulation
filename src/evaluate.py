import tempfile
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.logger import get_logger
from src.metrics import compute_regression_metrics
from src.mlflow_utils import log_json
from src.plots import (
    plot_error_distribution,
    plot_feature_importance,
    plot_forecast_vs_actual,
    plot_residuals)
from src.preprocessors.base import BasePreprocessor

logger = get_logger('evaluate')


def evaluate_model(
    model,
    preprocessor: BasePreprocessor,
    df_test: pd.DataFrame,
    tag: str = 'test',
) -> dict:
    """Evaluates on already-loaded data. Logs to the ACTIVE MLflow run."""
    X_test, y_test = preprocessor.transform(df_test)
    preds = model.predict(X_test)

    result_df = preprocessor.clean_and_sort(df_test)
    result_df['Predicted'] = preds

    timestamps = (
        result_df[preprocessor.time_col]
        if preprocessor.time_col in result_df.columns
        else None
    )

    metrics = {}
    plots = {}

    if y_test is not None:
        result_df['Residual'] = y_test - preds
        metrics = compute_regression_metrics(y_test, preds)
        metrics = {f'{tag}_{k}': v for k, v in metrics.items()}
        logger.info(f'{tag} Metrics: {metrics}')
        mlflow.log_metrics(metrics)

        plots[f'{tag}_forecast_vs_actual.png'] = plot_forecast_vs_actual(
            y_test, preds, timestamps, tag=tag
        )
        plots[f'{tag}_residuals.png'] = plot_residuals(y_test, preds, tag=tag)
        plots[f'{tag}_error_distribution.png'] = plot_error_distribution(
            y_test, preds, tag=tag
        )

    plots[f'{tag}_feature_importance.png'] = plot_feature_importance(
        model, X_test.columns.tolist(), tag=tag
    )

    for filename, fig in plots.items():
        if fig is not None:
            mlflow.log_figure(fig, f'plots/{filename}')
            plt.close(fig)

    with tempfile.TemporaryDirectory() as tmp:
        preds_file = Path(tmp) / f'predictions_{tag}.csv'
        result_df.to_csv(preds_file, index=False)
        mlflow.log_artifact(str(preds_file), artifact_path='predictions')

    log_json(metrics, f'{tag}_metrics.json', artifact_path='reports')
    return metrics