from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from src.features.raw_builder import EquipmentFeatureBuilderRaw

logger = logging.getLogger(__name__)

__all__ = ["EquipmentFeatureBuilderV3"]


class EquipmentFeatureBuilderV3(EquipmentFeatureBuilderRaw):
    """
    V3: Продвинутый feature engineering для BiGRU + Attention.
    Добавляет:
    - Скользящую волатильность (std) и наклоны трендов.
    - Взаимодействия между датчиками (разности).
    - Время до отказа (если есть).
    """

    def _add_lstm_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        logger.info("Добавление LSTM-фичей V3 (тренды, волатильность, взаимодействия)...")

        # 1. Разности и отклонения (как в V2)
        for col in sensor_cols:
            df[f"{col}_diff_1h"] = df.groupby("machineID")[col].diff(1).fillna(0)
            df[f"{col}_diff_3h"] = df.groupby("machineID")[col].diff(3).fillna(0)
            rolling_mean = (
                df.groupby("machineID")[col]
                .transform(lambda x: x.rolling(24, min_periods=1).mean())
            )
            df[f"{col}_deviation_24h"] = df[col] - rolling_mean

        # 2. Волатильность (Скользящее стандартное отклонение)
        # Показывает, насколько "шумно" ведет себя датчик (часто предвещает отказ)
        for col in sensor_cols:
            df[f"{col}_std_6h"] = (
                df.groupby("machineID")[col]
                .transform(lambda x: x.rolling(6, min_periods=1).std())
                .fillna(0)
            )
            df[f"{col}_std_24h"] = (
                df.groupby("machineID")[col]
                .transform(lambda x: x.rolling(24, min_periods=1).std())
                .fillna(0)
            )

        # 3. Наклон тренда (Slope) — самая мощная фича для временных рядов
        # Считаем разницу между средним за последние 3 часа и за последние 12 часов
        # Это показывает, ускоряется ли изменение датчика
        for col in sensor_cols:
            mean_3h = df.groupby("machineID")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
            mean_12h = df.groupby("machineID")[col].transform(lambda x: x.rolling(12, min_periods=1).mean())
            df[f"{col}_trend_slope"] = mean_3h - mean_12h

        # 4. Взаимодействия между датчиками (для пар из топ-5 важных)
        # Если у тебя много датчиков, это может создать слишком много колонок.
        # Мы берем 3 самых важных (например, по важности из CatBoost) и считаем их разности.
        # ВАЖНО: замени эти имена на реальные имена твоих датчиков!
        # Пример: sensor_cols[0], sensor_cols[1], sensor_cols[2]
        if len(sensor_cols) >= 3:
            s1, s2, s3 = sensor_cols[0], sensor_cols[1], sensor_cols[2]
            
            # Разность между датчиками
            df[f"{s1}_minus_{s2}"] = df[s1] - df[s2]
            df[f"{s2}_minus_{s3}"] = df[s2] - df[s3]
            
            # Отношение (ratio) — полезно, если датчики разной природы
            df[f"{s1}_div_{s2}"] = df[s1] / (df[s2] + 1e-8)

        # 5. Время (циклические признаки)
        if "datetime" in df.columns:
            hours = df["datetime"].dt.hour
            df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
            
            dayofweek = df["datetime"].dt.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)

        logger.info("Добавлены фичи V3: std_6h/24h, trend_slope, взаимодействия датчиков, время")
        return df