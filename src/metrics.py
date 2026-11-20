import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true, y_pred) -> dict:
    """Single source of truth for regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    return {
        'MAE': round(float(mean_absolute_error(y_true, y_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'R2': round(float(r2_score(y_true, y_pred)), 4),
        'MAPE': round(mape, 2),
    }