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
    
    # read config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # config arguments
    exp_cfg = config['experiment']
    data_cfg = config['data']
    feat_version = config['features']['version']
    model_name = config['model']['name']

    # seed
    seed_everything(exp_cfg['random_seed'])

    # connect to mlflow
    mlflow.set_tracking_uri(exp_cfg['tracking_uri'])
    mlflow.set_experiment(exp_cfg['name'])

    run_name = f"{model_name}_{feat_version}"
    
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f'Started MLflow Run ID: {run.info.run_id} ({run_name})')
        
        # log artifact
        mlflow.log_artifact(config_path, artifact_path='config')

        # load dataset
        logger.info(f"Loading dataset from {data_cfg['raw_path']}...")
        df = pd.read_csv(data_cfg['raw_path'])

        # (Train / Val / Test)
        train_ratio = data_cfg.get('train_ratio', 0.6)
        val_ratio = data_cfg.get('val_ratio', 0.2)
        
        df_train, df_val, df_test = chronological_split_3way(
            df, 
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            time_col=data_cfg['time_col']
        )
        logger.info(f"Split complete | Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        # init preprocessor
        preprocessor = get_preprocessor(
            model_family=model_name,  # 'catboost'
            feature_version=feat_version,  # 'v2'
            target_col=data_cfg['target_col'],
            time_col=data_cfg['time_col'])

        # train model
        model, _ = train_model(df_train, df_val, preprocessor, config)
        
        # evaluate model
        evaluate_model(model, preprocessor, df_test, tag='test')

        logger.info(f'EXPERIMENT COMPLETE | Run ID: {run.info.run_id}')
        return run.info.run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run end-to-end MLflow simulation pipeline')
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    args = parser.parse_args()
    
    run_experiment(config_path=args.config)