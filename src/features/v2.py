from __future__ import annotations
import logging
import pandas as pd
from src.features.v1 import EquipmentFeatureBuilderV1

logger = logging.getLogger(__name__)

__all__ = ["EquipmentFeatureBuilderV2"]


class EquipmentFeatureBuilderV2(EquipmentFeatureBuilderV1):
    """
    V2: Наследует всю логику V1 (таргет, очистка, mean/std/max),
    но добавляет:
    - Разницу (diff) между текущим значением и скользящим средним
    - Отношение (ratio) текущего значения к скользящему среднему
    """

    def _build_rolling_features(
        self,
        df_tel: pd.DataFrame,
        sensor_cols: list[str],
    ) -> pd.DataFrame:
        logger.info("Генерация фичей V2 (с diff и ratio)...")

        result = super()._build_rolling_features(df_tel, sensor_cols)

        for w in [self.window_short, self.window_long]:
            for sensor in sensor_cols:
                mean_col = f"{sensor}_mean_{w}h"
                
                if mean_col in result.columns:
                    current_vals = df_tel[sensor].values
                    mean_vals = result[mean_col].values
                    
                    result[f"{sensor}_diff_mean_{w}h"] = current_vals - mean_vals
                    result[f"{sensor}_ratio_mean_{w}h"] = current_vals / (mean_vals + 1e-8)

        logger.debug("Итоговое количество фичей: %d", result.shape[1] - 2) 
        return result