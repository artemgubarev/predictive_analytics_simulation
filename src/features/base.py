from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod

class BaseFeatureBuilder(ABC):
    """Абстрактный базовый класс для всех генераторов признаков."""

    @abstractmethod
    def prepare_dataset(
        self,
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> pd.DataFrame:
        """Преобразует сырые телеметрию и отказы в датасет с фичами и таргетом."""
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Возвращает список имен сгенерированных признаков."""
        ...