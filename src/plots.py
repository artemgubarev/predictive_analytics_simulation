import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series | None = None,
    tag: str = 'test',
    max_points: int = 500,
) -> plt.Figure:
    """Plots predicted values over actual values across time or sample index."""
    fig, ax = plt.subplots(figsize=(14, 4))
    y_t_slice = y_true[-max_points:]
    y_p_slice = y_pred[-max_points:]
    x_axis = (
        timestamps.iloc[-max_points:]
        if timestamps is not None
        else np.arange(len(y_t_slice))
    )

    ax.plot(x_axis, y_t_slice, label='Actual', linewidth=1.3)
    ax.plot(x_axis, y_p_slice, label='Predicted', linewidth=1.3, alpha=0.85)
    ax.set_title(f'[{tag.upper()}] Forecast vs Actual (last {len(y_t_slice)} points)')
    ax.set_xlabel('Time' if timestamps is not None else 'Sample Index')
    ax.set_ylabel('Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag: str = 'test',
) -> plt.Figure:
    """Plots residuals (Actual - Predicted) against predicted values."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
    ax.set_title(f'[{tag.upper()}] Residuals vs Predicted')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag: str = 'test',
    bins: int = 40,
) -> plt.Figure:
    """Plots a histogram of prediction errors with zero and mean reference lines."""
    residuals = y_true - y_pred
    mean_err = float(np.mean(residuals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=bins, edgecolor='black', alpha=0.75)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
    ax.axvline(mean_err, color='orange', linestyle=':', linewidth=1.5, label=f'Mean: {mean_err:.2f}')
    ax.set_title(f'[{tag.upper()}] Error Distribution')
    ax.set_xlabel('Prediction Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    tag: str = 'test',
    top_n: int = 15,
) -> plt.Figure | None:
    """Plots top-N feature importances for CatBoost or scikit-learn models."""
    importances = None
    if hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

    if importances is None or len(feature_names) == 0:
        return None

    k = min(top_n, len(feature_names))
    indices = np.argsort(importances)[-k:]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.array(feature_names)[indices], np.array(importances)[indices])
    ax.set_title(f'[{tag.upper()}] Top-{k} Feature Importance')
    ax.set_xlabel('Importance')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def create_regression_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series | None = None,
    model=None,
    feature_names: list[str] | None = None,
    tag: str = 'test',
) -> dict[str, plt.Figure]:
    """Builds all diagnostic regression plots and returns a {filename: Figure} mapping."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    figures: dict[str, plt.Figure] = {
        f'{tag}_forecast_vs_actual.png': plot_forecast_vs_actual(y_true, y_pred, timestamps, tag),
        f'{tag}_residuals.png': plot_residuals(y_true, y_pred, tag),
        f'{tag}_error_distribution.png': plot_error_distribution(y_true, y_pred, tag),
    }

    if model is not None and feature_names is not None:
        fi_fig = plot_feature_importance(model, feature_names, tag)
        if fi_fig is not None:
            figures[f'{tag}_feature_importance.png'] = fi_fig

    return figures