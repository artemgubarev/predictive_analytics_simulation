import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor

from src.logger import get_logger
from src.metrics import compute_regression_metrics
from src.mlflow_utils import log_files_to_mlflow, log_json
from src.preprocessors.base import BasePreprocessor

logger = get_logger('train')


def train_model(
    df_train,
    df_val,
    preprocessor: BasePreprocessor,
    config: dict,
):
    """Fits the model on already-split data. Logs to the ACTIVE MLflow run."""
    model_cfg = config['model']
    model_name = model_cfg['name'].lower()
    model_params = model_cfg['hyperparameters'].copy()
    early_stopping = model_params.pop('early_stopping_rounds', 50)
    seed = config['experiment']['random_seed']

    X_train, y_train = preprocessor.fit_transform(df_train)
    X_val, y_val = preprocessor.transform(df_val)
    logger.info(f'Features ready | train={X_train.shape} | val={X_val.shape}')

    logger.info(f'Training {model_name}...')
    if model_name == 'catboost':
        model = CatBoostRegressor(
            **model_params,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(
            X_train, y_train,
            cat_features=preprocessor.cat_cols,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping,
        )
    else:
        raise ValueError(f"Unsupported model name: '{model_name}'")

    metrics = compute_regression_metrics(y_val, model.predict(X_val))
    logger.info(f'Validation Metrics: {metrics}')

    # log to the active run — pipeline owns run lifecycle
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('feature_version', config['features']['version'])
    mlflow.log_param('train_rows', len(df_train))
    mlflow.log_param('val_rows', len(df_val))
    mlflow.log_params(model_params)
    mlflow.log_metrics(metrics)

    mlflow.catboost.log_model(model, artifact_path='model')
    log_json(metrics, 'metrics.json', artifact_path='reports')

    import tempfile, joblib
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        prep_file = Path(tmp) / 'preprocessor.joblib'
        joblib.dump(preprocessor, prep_file)
        log_files_to_mlflow({prep_file: 'preprocessor'})

    return model, metrics