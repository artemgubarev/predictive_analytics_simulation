import pandas as pd
import numpy as np
from typing import Tuple
from .base import BasePreprocessor

class CatBoostPreprocessor(BasePreprocessor):
    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray | None]:
        X, y = super()._prepare_xy(df)
        # Для деревьев удаляем синусы/косинусы, оставляем сырой Hour
        cols_to_drop = [c for c in ['Hour_sin', 'Hour_cos'] if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        return X, y

    def fit_transform(self, df_train: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        X, y = self._prepare_xy(df_train)
        X = self._impute_and_cast(X, fit=True)
        self.cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()
        return X, y

    def transform(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray | None]:
        if not self.cat_cols:
            raise RuntimeError('Preprocessor is not fitted!')
        X, y = self._prepare_xy(df_test)
        X = self._impute_and_cast(X, fit=False)
        return X, y