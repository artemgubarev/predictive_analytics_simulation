import pandas as pd
import numpy as np

def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8, time_col: str = 'Timestamp'):
    """Делит датафрейм на две части (train/test или fit/val)."""
    data = df.copy()
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(time_col).reset_index(drop=True)

    split_idx = max(1, int(len(data) * train_ratio))
    return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()

def chronological_split_3way(
    df: pd.DataFrame, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.2, 
    time_col: str = 'Timestamp'
):
    """Делит датафрейм на Train / Val / Test хронологически."""
    # Сначала отделяем тест (то, что останется справа)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
        
    df_trainval, df_test = chronological_split(df, train_ratio=(train_ratio + val_ratio), time_col=time_col)
    
    # Теперь делим оставшуюся левую часть на train и val
    # Внутри df_trainval пропорция val должна быть пересчитана:
    # (val_ratio) / (train_ratio + val_ratio)
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = chronological_split(df_trainval, train_ratio=(1.0 - relative_val_ratio), time_col=time_col)
    
    return df_train, df_val, df_test

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)