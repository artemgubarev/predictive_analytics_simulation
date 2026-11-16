import numpy as np
import pandas as pd

def create_sequences(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str, 
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, groups_list = [], [], []
    
    for machine_id, group in df.groupby("machineID"):
        group = group.sort_values("datetime")
        features = group[feature_cols].values
        targets = group[target_col].values
        
        for i in range(len(group) - seq_length):
            X_list.append(features[i : i + seq_length])
            y_list.append(targets[i + seq_length])
            groups_list.append(machine_id)  # <--- ВАЖНО: сохраняем группы
            
    return np.array(X_list), np.array(y_list), np.array(groups_list)