from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureBuilder(ABC):
    
    @abstractmethod
    def prepare_dataset(
        self,
        df_telemetry: pd.DataFrame,
        df_failures: pd.DataFrame,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        ...