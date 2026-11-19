import pandas as pd
from .base import BaseFeatureBuilder

class FeatureBuilderV1(BaseFeatureBuilder):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if self.time_col in data.columns:
            ts = pd.to_datetime(data[self.time_col])
            data['Hour'] = ts.dt.hour
            data['Month'] = ts.dt.month
        return data