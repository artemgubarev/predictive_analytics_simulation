import argparse
import mlflow
import pandas as pd
import yaml

from src.logger import get_logger
from src.preprocessors import get_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import chronological_split_3way, seed_everything

logger = get_logger('pipeline')


def run_experiment(config_path: str = '../configs/config.yaml') -> str:
    # 1. Читаем конфиг
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp_cfg = config['experiment']
    data_cfg = config['data']
    feat_version = config['features']['version']
    model_name = config['model']['name']

    # 2. Фиксируем сиды
    seed_everything(exp_cfg['random_seed'])

    # 3. Настраиваем MLflow
    mlflow.set_tracking_uri(exp_cfg['tracking_uri'])
    mlflow.set_experiment(exp_cfg['name'])

    run_name = f"{model_name}_{feat_version}"
    
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f'Started MLflow Run ID: {run.info.run_id} ({run_name})')
        
        # Логируем сам конфиг
        mlflow.log_artifact(config_path, artifact_path='config')

        # 4. Загружаем данные
        logger.info(f"Loading dataset from {data_cfg['raw_path']}...")
        df = pd.read_csv(data_cfg['raw_path'])

        # 5. Делим данные хронологически на 3 части (Train / Val / Test)
        train_ratio = data_cfg.get('train_ratio', 0.6)
        val_ratio = data_cfg.get('val_ratio', 0.2)
        
        df_train, df_val, df_test = chronological_split_3way(
            df, 
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            time_col=data_cfg['time_col']
        )
        logger.info(f"Split complete | Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        # 6. Инициализируем препроцессор (фабрика сама подтянет нужные фичи)
        preprocessor = get_preprocessor(
            model_family=model_name,  # 'catboost'
            feature_version=feat_version,  # 'v2'
            target_col=data_cfg['target_col'],
            time_col=data_cfg['time_col']
        )

        # 7. Обучаем модель (train_model сам сделает fit_transform для train и transform для val)
        model, _ = train_model(df_train, df_val, preprocessor, config)
        
        # 8. Оцениваем на тесте
        evaluate_model(model, preprocessor, df_test, config, tag='test')

        logger.info(f'EXPERIMENT COMPLETE | Run ID: {run.info.run_id}')
        return run.info.run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run end-to-end MLflow simulation pipeline')
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    args = parser.parse_args()
    
    run_experiment(config_path=args.config)