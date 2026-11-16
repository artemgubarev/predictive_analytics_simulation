from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from src.features.raw_builder import EquipmentFeatureBuilderRaw

logger = logging.getLogger(__name__)

__all__ = ["EquipmentFeatureBuilderV4"]


class EquipmentFeatureBuilderV4(EquipmentFeatureBuilderRaw):
    """
    V4: Heavy Duty Feature Engineering для BiGRU + Attention.
    Добавляет физику сигнала: Kurtosis, Skewness, RMS, Acceleration, Correlations.
    """

    def _add_lstm_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        logger.info("Добавление LSTM-фичей V4 (Heavy Duty: Kurtosis, RMS, Accel, Corr)...")

        # Выбираем топ-3 датчика для сложных взаимодействий (чтобы не раздувать фичи)
        # ВАЖНО: в идеале тут должны быть имена самых важных датчиков (например, vibration)
        top_sensors = sensor_cols[:3] 

        for col in sensor_cols:
            grouped = df.groupby("machineID")[col]
            
            # 1. Скорость изменения (diff) и Ускорение (вторая производная)
            df[f"{col}_diff_1h"] = grouped.diff(1).fillna(0)
            df[f"{col}_acceleration"] = grouped.diff(1).diff(1).fillna(0)
            
            # 2. Остаток от EWM (Exponential Weighted Mean)
            # Показывает, насколько текущее значение выбивается из тренда
            ewm_24h = grouped.transform(lambda x: x.ewm(span=24, adjust=False).mean())
            df[f"{col}_ewm_residual"] = df[col] - ewm_24h

            # 3. RMS (Root Mean Square) - энергия сигнала за 12 часов
            # ФИКС: Используем быструю формулу (sqrt(mean(x^2))) вместо медленного rolling.apply
            rolling_sq_mean = grouped.transform(lambda x: (x**2).rolling(12, min_periods=1).mean())
            df[f"{col}_rms_12h"] = np.sqrt(rolling_sq_mean)

            # 4. Kurtosis (Эксцесс) и Skewness (Асимметрия) за 24 часа
            # Показывают, насколько сигнал "сломался" (тяжелые хвосты в вибрации)
            df[f"{col}_kurt_24h"] = grouped.transform(lambda x: x.rolling(24, min_periods=5).kurt()).fillna(0)
            df[f"{col}_skew_24h"] = grouped.transform(lambda x: x.rolling(24, min_periods=5).skew()).fillna(0)

        # 5. Скользящая корреляция между топ-3 датчиками
        if len(top_sensors) >= 3:
            s1, s2, s3 = top_sensors[0], top_sensors[1], top_sensors[2]
            
            for pair in [(s1, s2), (s2, s3), (s1, s3)]:
                col1, col2 = pair
                corr_name = f"corr_{col1}_{col2}_24h"
                
                # ФИКС: Более быстрый и безопасный способ через transform
                def calc_rolling_corr(group):
                    return group[col1].rolling(24, min_periods=5).corr(group[col2])
                
                corr_series = df.groupby("machineID").apply(calc_rolling_corr)
                
                # Убираем MultiIndex и выравниваем по исходному индексу
                if isinstance(corr_series.index, pd.MultiIndex):
                    corr_series = corr_series.droplevel(0)
                df[corr_name] = corr_series.fillna(0)

        # 6. Время (циклические признаки)
        if "datetime" in df.columns:
            hours = df["datetime"].dt.hour
            df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
            
            dayofweek = df["datetime"].dt.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)

        logger.info("Добавлены V4 фичи: diff, acceleration, ewm_residual, rms, kurtosis, skewness, rolling_corr")
        return df