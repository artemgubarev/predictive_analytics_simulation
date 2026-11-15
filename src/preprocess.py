import logging
import pandas as pd
from pathlib import Path
from src.features.v1 import EquipmentFeatureBuilderV1

logger = logging.getLogger(__name__)

def load_data(telemetry_path: str, failures_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает телеметрию и отказы из файлов (CSV или Parquet)."""
    
    logger.info("Loading telemetry: %s", telemetry_path)
    logger.info("Loading failures: %s", failures_path)
    
    def _read(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {path}")

    df_telemetry = _read(telemetry_path)
    df_failures = _read(failures_path)
    
    logger.info("Телеметрия: %d строк, Отказы: %d строк", len(df_telemetry), len(df_failures))
    
    return df_telemetry, df_failures


def preprocess(config: dict) -> pd.DataFrame:
    """Data preprocessing"""
    
    logger.info("=== Preprocess started ===")
    
    cfg_data = config["preprocess"]
    cfg_features = config["features"]
    
    df_telemetry, df_failures = load_data(
        telemetry_path=cfg_data["telemetry_path"],
        failures_path=cfg_data["failures_path"],
    )
    
    builder = EquipmentFeatureBuilderV1(
        window_short=cfg_features["window_short"],
        window_long=cfg_features["window_long"],
        lead_time_min=cfg_features["lead_time_min"],
        lead_time_max=cfg_features["lead_time_max"],
        nan_drop_threshold=cfg_features["nan_drop_threshold"],
    )
    
    logger.info("Генерация фичей...")
    df_features = builder.prepare_dataset(df_telemetry, df_failures)
    logger.info("Фичи готовы. Итоговая форма: %s", df_features.shape)
    
    return df_features