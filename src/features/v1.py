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

    Target:
        target = 1, если до отказа осталось от lead_time_min до lead_time_max часов.
        Строки ближе lead_time_min к отказу — удаляются (слишком поздно).
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

        logger.debug(
            "EquipmentFeatureBuilderV1: windows=(%d, %d)h, lead=(%d, %d)h, nan_threshold=%.0f%%",
            window_short, window_long, lead_time_min, lead_time_max,
            nan_drop_threshold * 100)

    def prepare_dataset(
        self,
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> pd.DataFrame:
        """Полный пайплайн: фичи → таргет → очистка."""

        logger.info("prepare_dataset: telemetry=%d rows, failures=%d rows",
            len(df_telemetry), len(df_failures))

        # Валидация входных данных
        self._validate_inputs(df_telemetry, df_failures)
        # Копирование и нормализация времени
        df_tel, df_fail = self._normalize_inputs(df_telemetry, df_failures)
        # Имена колонок датчиков
        sensor_cols = self._get_sensor_cols(df_tel)
        # скользящие окна
        df_features = self._build_rolling_features(df_tel, sensor_cols)
        
        logger.info("Rolling features: %d rows × %d cols",
            len(df_features), df_features.shape[1])

        # 3. Разметка таргета (векторизованная)
        df_features = self._label_targets(df_features, df_fail)

        # 4. Очистка
        df_features = self._clean(df_features)

        logger.info(
            "Итог: %d строк | positive class: %d (%.2f%%)",
            len(df_features),
            int(df_features["target"].sum()),
            100 * df_features["target"].mean() if len(df_features) else 0.0)
        return df_features

    def get_feature_names(self) -> list[str]:
        """Имена признаков для текущей конфигурации (без target/ID/time)."""
        # Возвращаем шаблон — реальные имена зависят от датчиков
        # В продакшене лучше кэшировать после первого prepare_dataset
        return []  # будет заполнено после _build_rolling_features

    @staticmethod
    def _validate_inputs(
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> None:
        """Проверка наличия обязательных колонок и непустоты."""
        
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
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Копирование, приведение datetime, сортировка."""
        df_tel = df_telemetry.copy()
        df_fail = df_failures.copy()

        df_tel["datetime"] = pd.to_datetime(df_tel["datetime"]).dt.floor("h")
        df_fail["datetime"] = pd.to_datetime(df_fail["datetime"]).dt.floor("h")

        df_tel = df_tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        df_fail = df_fail.sort_values(["machineID", "datetime"]).reset_index(drop=True)

        logger.debug("Unique machines: %d", df_tel["machineID"].nunique())
        return df_tel, df_fail

    @staticmethod
    def _get_sensor_cols(df_tel: pd.DataFrame) -> list[str]:
        """Извлекает имена колонок-датчиков."""
        exclude = {"datetime", "machineID"}
        cols = [c for c in df_tel.columns if c not in exclude]
        if not cols:
            raise ValueError("No sensor columns found in telemetry")
        logger.debug("Sensors (%d): %s", len(cols), cols)
        return cols

    def _build_rolling_features(
        self,
        df_tel: pd.DataFrame,
        sensor_cols: list[str],
    ) -> pd.DataFrame:
        """
        Генерация rolling-статистик.
        Использует groupby + rolling с ручной очисткой MultiIndex.
        """
        result = df_tel[["machineID", "datetime"]].copy()

        for w in [self.window_short, self.window_long]:
            logger.debug("Rolling window %dh...", w)
            
            # группировка по id машины
            grouped = df_tel.groupby("machineID")[sensor_cols]
            # извлечение скользящих окон
            rolling = grouped.rolling(window=w, min_periods=w)

            for stat_name, stat_fn in [
                ("mean", rolling.mean),
                ("std", rolling.std),
                ("max", rolling.max)]:
                
                stat_df = stat_fn()
                # Убираем MultiIndex от groupby, если появился
                if isinstance(stat_df.index, pd.MultiIndex):
                    stat_df = stat_df.droplevel(0)
                # переименовыние колонок: sensor_stat_windowh (например, temp_mean_3h)
                stat_df.columns = [f"{c}_{stat_name}_{w}h" for c in sensor_cols]
                # конкатенация новых колонок со статистиками к основной таблице
                result = pd.concat([result, stat_df], axis=1)

        return result
    
    def _label_targets(
        self,
        df: pd.DataFrame,
        df_fail: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Векторизованная разметка таргета через merge_asof.
        Заменяет O(N×M) цикл по iterrows на O(N log N).
        """
        df = df.copy()
        df["target"] = 0

        if df_fail.empty:
            logger.warning("df_failures пуст — все target=0")
            return df

        # Для каждого отказа создаём «зону предупреждения»
        # и «зону удаления» (слишком близко к поломке)
        df_fail_windows = df_fail[["machineID", "datetime"]].copy()
        df_fail_windows.rename(columns={"datetime": "fail_time"}, inplace=True)

        # начало зоны предупреждения
        df_fail_windows["window_start"] = (
            df_fail_windows["fail_time"]
            - pd.Timedelta(hours=self.lead_time_max))
    
        # конец зоны предупреждения
        df_fail_windows["window_end"] = (
            df_fail_windows["fail_time"]
            - pd.Timedelta(hours=self.lead_time_min))

        # Merge: для каждой строки телеметрии находим ближайший будущий отказ
        # Сохраняем исходный индекс в колонку, чтобы потом вернуть порядок
        df = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    
        df_sorted = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)
        df_fail_sorted = df_fail_windows.sort_values("fail_time", kind="mergesort").reset_index(drop=True)
    
        merged = pd.merge_asof(
            df_sorted,
            df_fail_sorted,
            left_on="datetime",
            right_on="fail_time",
            by="machineID",
            direction="forward",
            tolerance=pd.Timedelta(hours=self.lead_time_max))

        # target = 1: отказ впереди, и мы в окне [window_start, window_end]
        in_warning_zone = (
            merged["fail_time"].notna()
            & (merged["datetime"] >= merged["window_start"])
            & (merged["datetime"] <= merged["window_end"]))
        merged.loc[in_warning_zone, "target"] = 1

        # Удаляем строки слишком близко к поломке (между window_end и fail_time)
        too_close = (
            merged["fail_time"].notna()
            & (merged["datetime"] > merged["window_end"])
            & (merged["datetime"] <= merged["fail_time"])
        )
        n_drop = int(too_close.sum())
        if n_drop > 0:
            logger.info("Удалено строк слишком близко к отказу: %d", n_drop)
        
        merged = merged.loc[~too_close]
        merged = merged.sort_values("_orig_idx").reset_index(drop=True)
        merged = merged.drop(columns=["_orig_idx", "fail_time", "window_start", "window_end"])

        n_pos = int(merged["target"].sum())
        logger.debug("Target=1: %d строк", n_pos)

        return merged

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление NaN с контролем порога."""
        initial = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = initial - len(df)
        pct = dropped / max(initial, 1)

        log_fn = logger.warning if pct > self.nan_drop_threshold else logger.info
        log_fn("Dropna: %d строк (%.1f%%)", dropped, 100 * pct)

        return df