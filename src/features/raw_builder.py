from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.features.base import BaseFeatureBuilder

logger = logging.getLogger(__name__)

__all__ = ["EquipmentFeatureBuilderRaw"]


class EquipmentFeatureBuilderRaw(BaseFeatureBuilder):
    """
    Билдер для LSTM. Не делает окон и агрегатов.
    Только разметка таргета, нормализация, LSTM-фичи и базовая очистка.
    """

    def __init__(
        self,
        lead_time_min: int = 1,
        lead_time_max: int = 24,
        scaler_type: str = "standard",  # "standard", "minmax" или "robust"
    ) -> None:
        self.lead_time_min = lead_time_min
        self.lead_time_max = lead_time_max
        self.scaler_type = scaler_type

    def prepare_dataset(
        self,
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("prepare_dataset (RAW): telemetry=%d rows, failures=%d rows",
            len(df_telemetry), len(df_failures))

        # 1. Нормализация времени и сортировка
        df_tel = df_telemetry.copy()
        df_fail = df_failures.copy()
        df_tel["datetime"] = pd.to_datetime(df_tel["datetime"]).dt.floor("h")
        df_fail["datetime"] = pd.to_datetime(df_fail["datetime"]).dt.floor("h")
        
        df_tel = df_tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        df_fail = df_fail.sort_values(["machineID", "datetime"]).reset_index(drop=True)

        # 2. Разметка таргета
        df_tel = self._label_targets(df_tel, df_fail)

        # 3. Определяем колонки датчиков
        sensor_cols = [c for c in df_tel.columns if c not in ["datetime", "machineID", "target"]]

        # 4. СГЛАЖИВАНИЕ (важно: ДО расчета diff и deviation)
        df_tel = self._smooth_telemetry(df_tel, sensor_cols)

        # 5. Добавляем LSTM-специфичные фичи (разности, отклонения, время)
        # Важно: мы считаем diff от сглаженных данных, это нормально.
        df_tel = self._add_lstm_features(df_tel, sensor_cols)

        # 6. Нормализация всех числовых признаков (включая новые)
        all_feature_cols = [c for c in df_tel.columns if c not in ["datetime", "machineID", "target"]]
        df_tel = self._normalize_sensors(df_tel, all_feature_cols)

        # 7. Очистка от NaN
        initial = len(df_tel)
        df_tel = df_tel.dropna().reset_index(drop=True)
        dropped = initial - len(df_tel)
        logger.info("Dropna: %d строк (%.1f%%)", dropped, 100 * dropped / max(initial, 1))

        logger.info(
            "Итог (RAW + LSTM features + Smooth): %d строк | positive class: %d (%.2f%%)",
            len(df_tel),
            int(df_tel["target"].sum()),
            100 * df_tel["target"].mean() if len(df_tel) else 0.0)
        
        return df_tel

    def _smooth_telemetry(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Сглаживание датчиков через EWM (экспоненциальное)."""
        df = df.copy()
        logger.info("Сглаживание телеметрии (EWM, span=3)...")
        
        # Группируем по машине, чтобы не смешивать ряды
        grouped = df.groupby("machineID")[sensor_cols]
        
        # Применяем EWM. span=3 означает, что вес затухает за ~3 часа.
        # Можно поставить 6 или 12, если нужно более сильное сглаживание.
        smoothed = grouped.transform(lambda x: x.ewm(span=3, adjust=False).mean())
        
        df[sensor_cols] = smoothed
        return df

    def get_feature_names(self) -> list[str]:
        return []  # Динамически после prepare_dataset

    def _add_lstm_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Добавляет разности, отклонения и циклические признаки времени."""
        df = df.copy()
        
        logger.info("Добавление LSTM-фичей (diff, deviation, time)...")
        
        # 1. Разности (скорость изменения)
        for col in sensor_cols:
            df[f"{col}_diff_1h"] = df.groupby("machineID")[col].diff(1).fillna(0)
            df[f"{col}_diff_3h"] = df.groupby("machineID")[col].diff(3).fillna(0)
        
        # 2. Отклонение от скользящего среднего (за 24 часа)
        for col in sensor_cols:
            rolling_mean = (
                df.groupby("machineID")[col]
                .transform(lambda x: x.rolling(24, min_periods=1).mean())
            )
            df[f"{col}_deviation_24h"] = df[col] - rolling_mean
        
        # 3. Время (циклические признаки)
        if "datetime" in df.columns:
            hours = df["datetime"].dt.hour
            df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
            
            dayofweek = df["datetime"].dt.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
            
        logger.info("Добавлены LSTM-фичи: diff (1h, 3h), deviation_24h, hour_sin/cos, dow_sin/cos")
        return df

    def _label_targets(self, df: pd.DataFrame, df_fail: pd.DataFrame) -> pd.DataFrame:
        """Разметка таргета (идентична V1)."""
        df = df.copy()
        df["target"] = 0

        if df_fail.empty:
            return df

        df_fail_windows = df_fail[["machineID", "datetime"]].copy()
        df_fail_windows.rename(columns={"datetime": "fail_time"}, inplace=True)
        df_fail_windows["window_start"] = df_fail_windows["fail_time"] - pd.Timedelta(hours=self.lead_time_max)
        df_fail_windows["window_end"] = df_fail_windows["fail_time"] - pd.Timedelta(hours=self.lead_time_min)

        df = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
        df_sorted = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)
        df_fail_sorted = df_fail_windows.sort_values("fail_time", kind="mergesort").reset_index(drop=True)

        merged = pd.merge_asof(
            df_sorted, df_fail_sorted,
            left_on="datetime", right_on="fail_time",
            by="machineID", direction="forward",
            tolerance=pd.Timedelta(hours=self.lead_time_max))

        in_warning_zone = (
            merged["fail_time"].notna()
            & (merged["datetime"] >= merged["window_start"])
            & (merged["datetime"] <= merged["window_end"]))
        merged.loc[in_warning_zone, "target"] = 1

        too_close = (
            merged["fail_time"].notna()
            & (merged["datetime"] > merged["window_end"])
            & (merged["datetime"] <= merged["fail_time"]))
        
        merged = merged.loc[~too_close]
        merged = merged.sort_values("_orig_idx").reset_index(drop=True)
        merged = merged.drop(columns=["_orig_idx", "fail_time", "window_start", "window_end"])
        return merged

    def _normalize_sensors(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Нормализация (Standard, MinMax или Robust)."""
        df = df.copy()
        
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {self.scaler_type}")

        df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
        
        logger.info("Признаки нормализованы (%s).", self.scaler_type)
        return df