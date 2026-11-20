import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.base import BaseFeatureBuilder

class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        feature_builder: BaseFeatureBuilder, 
        target_col: str = 'EnergyConsumption', 
        time_col: str = 'Timestamp'
    ):
        self.feature_builder = feature_builder
        self.target_col = target_col
        self.time_col = time_col
        
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self._num_medians: dict[str, float] = {}

    def clean_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names, deduplicate by timestamp, and sort chronologically."""
        
        data = df.copy()
        data.columns = data.columns.str.strip()
        if self.time_col in data.columns:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
            data = data.dropna(subset=[self.time_col])
            data = data.drop_duplicates(subset=[self.time_col], keep='last')
            data = data.sort_values(self.time_col).reset_index(drop=True)
        return data

    def _impute_and_cast(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Impute missing values and cast feature types without data leakage."""
        
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num_cols = X.select_dtypes(include=['number']).columns.tolist()

        for col in cat_cols:
            X[col] = X[col].astype(str).fillna('Missing')

        if fit:
            self._num_medians = {c: float(X[c].median()) for c in num_cols}
        for col in num_cols:
            X[col] = X[col].fillna(self._num_medians.get(col, 0.0))
            
        return X

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray | None]:
        data = self.feature_builder.add_features(self.clean_and_sort(df))
        # delete time and target
        cols_to_drop = [c for c in [self.time_col, self.target_col] if c in data.columns]
        X = data.drop(columns=cols_to_drop)
        y = data[self.target_col].values if self.target_col in data.columns else None
        return X, y