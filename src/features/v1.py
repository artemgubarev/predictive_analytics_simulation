from __future__ import annotations
import logging
import pandas as pd
from src.features.base import BaseFeatureBuilder

logger = logging.getLogger(__name__)

__all__ = ["EquipmentFeatureBuilderV1"]

# Обязательные колонки во входных данных
_REQUIRED_TELEMETRY_COLS = {"datetime", "machineID"}
_REQUIRED_FAILURE_COLS = {"datetime", "machineID"}


class EquipmentFeatureBuilderV1(BaseFeatureBuilder):
    """
    Feature generation based on time series (v1)
    Feature:
        - {sensor}_mean_{w}h  — скользящее среднее
        - {sensor}_std_{w}h   — скользящее стд. отклонение
        - {sensor}_max_{w}h   — скользящий максимум
    """

    def __init__(
        self,
        window_short: int = 3,
        window_long: int = 24,
        lead_time_min: int = 1,
        lead_time_max: int = 24,
        nan_drop_threshold: float = 0.3,
    ) -> None:
        self.window_short = window_short
        self.window_long = window_long
        self.lead_time_min = lead_time_min
        self.lead_time_max = lead_time_max
        self.nan_drop_threshold = nan_drop_threshold

    def prepare_dataset(
        self,
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> pd.DataFrame:
        """Полный пайплайн: фичи → таргет → очистка."""
        logger.info("prepare_dataset: telemetry=%d rows, failures=%d rows",
            len(df_telemetry), len(df_failures))

        self._validate_inputs(df_telemetry, df_failures)
        df_tel, df_fail = self._normalize_inputs(df_telemetry, df_failures)
        sensor_cols = self._get_sensor_cols(df_tel)
        
        df_features = self._build_rolling_features(df_tel, sensor_cols)
        logger.info("Rolling features: %d rows × %d cols", len(df_features), df_features.shape[1])

        df_features = self._label_targets(df_features, df_fail)
        df_features = self._clean(df_features)

        logger.info(
            "Итог: %d строк | positive class: %d (%.2f%%)",
            len(df_features),
            int(df_features["target"].sum()),
            100 * df_features["target"].mean() if len(df_features) else 0.0)
        return df_features

    def get_feature_names(self) -> list[str]:
        return []  # Будет реализовано в V2 или позже

    @staticmethod
    def _validate_inputs(df_telemetry: pd.DataFrame, df_failures: pd.DataFrame) -> None:
        for name, df, required in [
            ("df_telemetry", df_telemetry, _REQUIRED_TELEMETRY_COLS),
            ("df_failures", df_failures, _REQUIRED_FAILURE_COLS)]:
            if df.empty:
                raise ValueError(f"{name} is empty")
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")

    @staticmethod
    def _normalize_inputs(
        df_telemetry: pd.DataFrame, df_failures: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_tel = df_telemetry.copy()
        df_fail = df_failures.copy()

        df_tel["datetime"] = pd.to_datetime(df_tel["datetime"]).dt.floor("h")
        df_fail["datetime"] = pd.to_datetime(df_fail["datetime"]).dt.floor("h")

        df_tel = df_tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        df_fail = df_fail.sort_values(["machineID", "datetime"]).reset_index(drop=True)

        return df_tel, df_fail

    @staticmethod
    def _get_sensor_cols(df_tel: pd.DataFrame) -> list[str]:
        exclude = {"datetime", "machineID"}
        cols = [c for c in df_tel.columns if c not in exclude]
        if not cols:
            raise ValueError("No sensor columns found in telemetry")
        return cols

    def _build_rolling_features(
        self, df_tel: pd.DataFrame, sensor_cols: list[str],
    ) -> pd.DataFrame:
        """Генерация rolling-статистик и лагов."""
        result = df_tel[["machineID", "datetime"]].copy()

        # 1. Один раз создаем группировку (чтобы не делать это в каждом цикле)
        grouped = df_tel.groupby("machineID")[sensor_cols]

        # 2. Лаги (используем ту же группировку)
        for lag in [1, 3]:
            lag_df = grouped.shift(lag)
            lag_df.columns = [f"{c}_lag_{lag}h" for c in sensor_cols]
            result = pd.concat([result, lag_df], axis=1)

        # 3. Скользящие окна
        for w in [self.window_short, self.window_long]:
            # ВАЖНО: rolling() внутри groupby возвращает MultiIndex
            rolling = grouped.rolling(window=w, min_periods=w)

            for stat_name, stat_fn in [
                ("mean", rolling.mean),
                ("std", rolling.std),
                ("max", rolling.max)]:
                
                stat_df = stat_fn()
                # Сбрасываем MultiIndex, чтобы индексы совпали с result
                if isinstance(stat_df.index, pd.MultiIndex):
                    stat_df = stat_df.droplevel(0)
                
                stat_df.columns = [f"{c}_{stat_name}_{w}h" for c in sensor_cols]
                result = pd.concat([result, stat_df], axis=1)

        return result

    def _label_targets(self, df: pd.DataFrame, df_fail: pd.DataFrame) -> pd.DataFrame:
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

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = initial - len(df)
        pct = dropped / max(initial, 1)
        log_fn = logger.warning if pct > self.nan_drop_threshold else logger.info
        log_fn("Dropna: %d строк (%.1f%%)", dropped, 100 * pct)
        return df