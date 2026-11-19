from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureBuilder(ABC):
    """Интерфейс для генераторов фичей."""
    def __init__(self, time_col: str = 'Timestamp'):
        self.time_col = time_col

    @abstractmethod
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет фичи к датафрейму."""
        pass