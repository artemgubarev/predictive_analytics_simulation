import pandas as pd
import numpy as np
from .v1 import FeatureBuilderV1

class FeatureBuilderV2(FeatureBuilderV1):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().add_features(df) # Сначала берем базовые фичи из v1
        
        if self.time_col in data.columns:
            ts = pd.to_datetime(data[self.time_col])
            data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
            data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)

        # Физические взаимодействия
        if 'Temperature' in data.columns:
            data['Temp_dist_22'] = np.abs(data['Temperature'] - 22.0)
        if 'Occupancy' in data.columns and 'SquareFootage' in data.columns:
            data['Occupancy_density'] = data['Occupancy'] / (data['SquareFootage'] + 1.0)
        if 'RenewableEnergy' in data.columns and 'SquareFootage' in data.columns:
            data['Renewable_per_sqft'] = data['RenewableEnergy'] / (data['SquareFootage'] + 1.0)
            
        return data