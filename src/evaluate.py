import mlflow
import pandas as pd

from src.logger import get_logger
from src.metrics import compute_regression_metrics
from src.mlflow_utils import log_json
from src.preprocessors.base import BasePreprocessor

logger = get_logger('evaluate')

def evaluate_model(
    model,
    preprocessor: BasePreprocessor,
    df_test: pd.DataFrame,
    config: dict,
    tag: str = 'test',
) -> dict:
    """Evaluates on already-loaded data. Logs to the ACTIVE MLflow run."""
    X_test, y_test = preprocessor.transform(df_test)
    preds = model.predict(X_test)

    result_df = preprocessor.clean_and_sort(df_test)
    result_df['Predicted'] = preds

    metrics = {}
    if y_test is not None:
        result_df['Residual'] = y_test - preds
        metrics = compute_regression_metrics(y_test, preds)
        metrics = {f'{tag}_{k}': v for k, v in metrics.items()}
        logger.info(f'{tag} Metrics: {metrics}')
        mlflow.log_metrics(metrics)

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        preds_file = Path(tmp) / f'predictions_{tag}.csv'
        result_df.to_csv(preds_file, index=False)
        mlflow.log_artifact(str(preds_file), artifact_path='predictions')

    log_json(metrics, f'{tag}_metrics.json', artifact_path='reports')
    return metrics