from __future__ import annotations
import logging
import pandas as pd

from src.features.v1 import EquipmentFeatureBuilderV1
from src.features.v2 import EquipmentFeatureBuilderV2
from src.features.v3 import EquipmentFeatureBuilderV3
from src.features.v4 import EquipmentFeatureBuilderV4
from src.features.raw_builder import EquipmentFeatureBuilderRaw

logger = logging.getLogger(__name__)

def _load_data(telemetry_path: str, failures_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Вспомогательная функция для загрузки данных (пример)."""
    logger.info("Загрузка телеметрии: %s", telemetry_path)
    df_tel = pd.read_csv(telemetry_path)  # или pd.read_parquet, если у тебя parquet
    
    logger.info("Загрузка отказов: %s", failures_path)
    df_fail = pd.read_csv(failures_path)
    
    return df_tel, df_fail


def preprocess(config: dict) -> pd.DataFrame:
    """
    Главная функция препроцессинга.
    Загружает данные, выбирает версию фичей из конфига и генерирует датасет.
    """
    logger.info("=== Preprocess started ===")
    
    # 1. Достаем пути и параметры из конфига
    cfg = config["preprocess"]
    tel_path = cfg["telemetry_path"]
    fail_path = cfg["failures_path"]
    
    # 2. Загружаем сырые данные
    df_tel, df_fail = _load_data(tel_path, fail_path)
    
    # 3. Выбираем версию фичей (по умолчанию v1)
    feature_version = cfg.get("feature_version", "v1").lower()
    logger.info("Используемая версия фичей: %s", feature_version)
    
    # 4. Собираем параметры для билдера (окна и т.д.)
    # Если в конфиге есть секция builder_params, берем оттуда, иначе дефолты
    builder_kwargs = cfg.get("builder_params", {})
    
    # 5. Создаем нужный билдер
    if feature_version == "v1":
        builder = EquipmentFeatureBuilderV1(**builder_kwargs)
    elif feature_version == "v2":
        builder = EquipmentFeatureBuilderV2(**builder_kwargs)
    elif feature_version == "v3":
        valid_keys = {"lead_time_min", "lead_time_max", "scaler_type"}
        clean_kwargs = {k: v for k, v in builder_kwargs.items() if k in valid_keys}
        builder = EquipmentFeatureBuilderV3(**clean_kwargs)
    elif feature_version == "v4":
        valid_keys = {"lead_time_min", "lead_time_max", "scaler_type"}
        clean_kwargs = {k: v for k, v in builder_kwargs.items() if k in valid_keys}
        builder = EquipmentFeatureBuilderV4(**clean_kwargs)
    elif feature_version == "raw":
        builder = EquipmentFeatureBuilderRaw(
            lead_time_min=builder_kwargs.get("lead_time_min", 1),
            lead_time_max=builder_kwargs.get("lead_time_max", 24),
            scaler_type=builder_kwargs.get("scaler_type", "standard"))
    else:
        raise ValueError(f"Неизвестная версия фичей: {feature_version}. Ожидается 'v1' или 'v2'.")
        
    # 6. Генерируем датасет
    logger.info("Генерация фичей...")
    df_features = builder.prepare_dataset(df_tel, df_fail)
    
    logger.info("=== Preprocess finished ===")
    return df_features