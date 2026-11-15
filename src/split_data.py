import pandas as pd
import numpy as np
from pathlib import Path

telemetry_path = '../data/raw/PdM_telemetry.csv'
failures_path = '../data/raw/PdM_failures.csv'
errors_path = '../data/raw/PdM_errors.csv'

output_path = '../data/splited/main_data.csv'
prod_path = '../data/splited/prod_data.csv'

telemetry_df = pd.read_csv(telemetry_path)
failures_df = pd.read_csv(failures_path)
errors_df = pd.read_csv(errors_path)

all_ids = (
    set(telemetry_df["machineID"])
    | set(errors_df["machineID"])
    | set(failures_df["machineID"])
)

ids = np.array(sorted(all_ids))

# деление данных в пропорции
n = len(ids)
n_prod  = int(round(n * 0.40))
n_main = n - n_prod

main_ids = ids[:n_main]
prod_ids = ids[n_main:]

print(main_ids)

print(f"main: {len(main_ids)} machines")
print(f"prod:  {len(prod_ids)} machines")

assert set(main_ids).isdisjoint(prod_ids)

def split_by_ids(df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    return df[df["machineID"].isin(ids)].copy()

# telemetry
telemetry_main = split_by_ids(telemetry_df, main_ids)
telemetry_prod = split_by_ids(telemetry_df, prod_ids)

# errors
errors_main = split_by_ids(errors_df, main_ids)
errors_prod = split_by_ids(errors_df, prod_ids)

# failures
failures_main = split_by_ids(failures_df, main_ids)
failures_prod = split_by_ids(failures_df, prod_ids)

base_dir = Path("../data/splited")

splits = {
    "main": {
        "telemetry": telemetry_main,
        "errors":    errors_main,
        "failures":  failures_main,
    },
    "prod": {
        "telemetry": telemetry_prod,
        "errors":    errors_prod,
        "failures":  failures_prod,
    },
}

for split_name, datasets in splits.items():
    split_dir = base_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, df in datasets.items():
        out_path = split_dir / f"{ds_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"{out_path}: {len(df)} строк, "
              f"{df['machineID'].nunique()} машин")