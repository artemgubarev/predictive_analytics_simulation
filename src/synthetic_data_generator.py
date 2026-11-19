import argparse
from pathlib import Path
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer


class EnergySyntheticGenerator:
    """Hybrid synthetic data generator for energy consumption (GaussianCopula + CatBoost)."""

    def __init__(self, noise_std: float = 5.0, random_seed: int = 42):
        self.noise_std = noise_std
        self.random_seed = random_seed
        self.copula = None
        self.cb_model = None
        self.last_timestamp = None
        self.original_columns = None

    @staticmethod
    def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """Creates temporal and domain-specific features for the CatBoost model."""
        data = df.copy()
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['Hour'] = data['Timestamp'].dt.hour
        data['Month'] = data['Timestamp'].dt.month
        data['Temp_dist_22'] = np.abs(data['Temperature'] - 22)
        data['Occupancy_density'] = data['Occupancy'] / (
            data['SquareFootage'] + 1
        )
        data['Renewable_per_sqft'] = data['RenewableEnergy'] / (
            data['SquareFootage'] + 1
        )
        return data

    def fit(self, df: pd.DataFrame):
        """Fits the feature generator (Copula) and the process physics model (CatBoost)."""
        data = df.copy()
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data = data.sort_values('Timestamp').reset_index(drop=True)

        self.last_timestamp = data['Timestamp'].max()
        self.original_columns = data.columns.tolist()

        # 1. Fit GaussianCopula on input features (excluding Timestamp)
        features_for_copula = data.drop(columns=['Timestamp'])
        metadata = Metadata.detect_from_dataframes(
            data={'energy_table': features_for_copula}
        )
        self.copula = GaussianCopulaSynthesizer(metadata)
        self.copula.fit({'energy_table': features_for_copula})

        # 2. Prepare features and train CatBoost to predict EnergyConsumption
        enriched = self._add_engineered_features(data)
        X = enriched.drop(columns=['Timestamp', 'EnergyConsumption'])
        y = enriched['EnergyConsumption']

        cat_cols = X.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        for col in cat_cols:
            X[col] = X[col].astype(str).fillna('Missing')

        self.cb_model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5,
            loss_function='RMSE',
            random_seed=self.random_seed,
            verbose=0,
            allow_writing_files=False
        )
        self.cb_model.fit(X, y, cat_features=cat_cols)
        return self

    def generate(
        self,
        n_rows: int = 1000,
        freq: str = '1h',
        temp_shift: float = 0.0,
        occupancy_mult: float = 1.0,
        seed: int = None,
    ) -> pd.DataFrame:
        """Generates n_rows of synthetic time-series records.

        Supports what-if scenario parameters:
        - temp_shift: Temperature offset in degrees (e.g., +5.0 for heatwave simulation)
        - occupancy_mult: Multiplier for occupancy count (e.g., 1.5 for peak load)
        """
        if self.copula is None or self.cb_model is None:
            raise RuntimeError(
                'Generator is not fitted! Call fit() or load() first.'
            )

        rng = np.random.default_rng(
            seed if seed is not None else self.random_seed
        )

        # 1. Sample base features using Copula
        sampled = self.copula.sample(num_rows=n_rows)
        synth = (
            sampled['energy_table'] if isinstance(sampled, dict) else sampled
        )

        # 2. Apply what-if scenario modifications
        if temp_shift != 0.0 and 'Temperature' in synth.columns:
            synth['Temperature'] = synth['Temperature'] + temp_shift

        if occupancy_mult != 1.0 and 'Occupancy' in synth.columns:
            synth['Occupancy'] = np.round(
                synth['Occupancy'] * occupancy_mult
            ).astype(int)

        # 3. Generate continuous timeline and consistent day-of-week labels
        start_date = self.last_timestamp + pd.DateOffset(hours=1)
        synth['Timestamp'] = pd.date_range(
            start=start_date, periods=n_rows, freq=freq
        )
        if 'DayOfWeek' in synth.columns:
            synth['DayOfWeek'] = synth['Timestamp'].dt.day_name()

        # 4. Predict energy consumption with CatBoost and add natural noise
        enriched = self._add_engineered_features(synth)
        X_cb = enriched[self.cb_model.feature_names_].copy()

        cat_cols = X_cb.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        for col in cat_cols:
            X_cb[col] = X_cb[col].astype(str).fillna('Missing')

        noise = rng.normal(0, self.noise_std, size=n_rows)
        synth['EnergyConsumption'] = self.cb_model.predict(X_cb) + noise

        return synth[self.original_columns]

    def save(self, folder_path: str = 'saved_generator'):
        """Saves the fitted generator artifacts to disk."""
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        self.copula.save(str(path / 'copula.pkl'))
        self.cb_model.save_model(str(path / 'catboost.cbm'))
        meta = pd.Series({
            'last_timestamp': str(self.last_timestamp),
            'original_columns': ','.join(self.original_columns),
            'noise_std': self.noise_std,
            'random_seed': self.random_seed,
        })
        meta.to_json(path / 'meta.json')

    @classmethod
    def load(cls, folder_path: str = 'saved_generator'):
        """Loads a pre-trained generator from disk."""
        path = Path(folder_path)
        meta = pd.read_json(path / 'meta.json', typ='series')

        instance = cls(
            noise_std=float(meta['noise_std']),
            random_seed=int(meta['random_seed']),
        )
        instance.last_timestamp = pd.to_datetime(meta['last_timestamp'])
        instance.original_columns = meta['original_columns'].split(',')
        instance.copula = GaussianCopulaSynthesizer.load(
            str(path / 'copula.pkl')
        )
        instance.cb_model = CatBoostRegressor()
        instance.cb_model.load_model(str(path / 'catboost.cbm'))
        return instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Synthetic energy consumption data generator'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../data/energy_consumption.csv',
        help='Path to the input CSV dataset',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='synthetic_energy.csv',
        help='Path to save the generated CSV',
    )
    parser.add_argument(
        '--rows', type=int, default=5000, help='Number of synthetic rows to generate'
    )
    parser.add_argument(
        '--temp-shift',
        type=float,
        default=0.0,
        help='Temperature shift (°C) for stress-testing',
    )
    parser.add_argument(
        '--occ-mult',
        type=float,
        default=1.0,
        help='Occupancy multiplier for peak load simulation',
    )
    args = parser.parse_args()

    print(f'Loading source data from {args.input}...')
    raw_df = pd.read_csv(args.input)

    generator = EnergySyntheticGenerator()
    print('Fitting generator...')
    generator.fit(raw_df)
    generator.save('saved_generator')

    print(f'Generating {args.rows} synthetic rows...')
    sim_df = generator.generate(
        n_rows=args.rows,
        temp_shift=args.temp_shift,
        occupancy_mult=args.occ_mult,
    )
    sim_df.to_csv(args.output, index=False)
    print(
        f'Done! Output saved to {args.output}, model artifacts saved to saved_generator/'
    )