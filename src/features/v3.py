import pandas as pd
import numpy as np
from typing import List, Optional
from .v2 import FeatureBuilderV2

class FeatureBuilderV3(FeatureBuilderV2):
    def __init__(
        self,
        time_col: str = 'Timestamp',
        smooth_cols: Optional[List[str]] = None,
        ewm_spans: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        super().__init__(time_col=time_col)
        # Список признаков датчиков по умолчанию, которые обычно шумят
        self.smooth_cols = smooth_cols if smooth_cols is not None else [
            'Temperature', 
            'Humidity', 
            'Occupancy', 
            'RenewableEnergy'
        ]
        self.ewm_spans = ewm_spans if ewm_spans is not None else [3, 6]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [6, 12]

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Сначала получаем все признаки из V1 и V2 (включая очищенную температуру, часы и т.д.)
        data = super().add_features(df)

        # 2. Добавляем сглаженные признаки для выбранных колонок
        for col in self.smooth_cols:
            if col not in data.columns:
                continue

            # Экспоненциальное сглаживание (быстрая реакция на тренд, меньше запаздывания)
            for span in self.ewm_spans:
                data[f'{col}_ewm_{span}'] = (
                    data[col].ewm(span=span, adjust=False).mean())

            # Скользящее среднее (хорошо убирает случайные выбросы за длинный период)
            for win in self.rolling_windows:
                data[f'{col}_roll_mean_{win}'] = (
                    data[col].rolling(window=win, min_periods=1).mean())

        # 3. Улучшаем физическое взаимодействие из V2 на основе сглаженной температуры
        # (защищает от резких скачков датчика температуры)
        first_span = self.ewm_spans[0] if self.ewm_spans else None
        smooth_temp_col = f'Temperature_ewm_{first_span}'
        
        if smooth_temp_col in data.columns:
            data['Temp_dist_22_smooth'] = np.abs(data[smooth_temp_col] - 22.0)

        return data
