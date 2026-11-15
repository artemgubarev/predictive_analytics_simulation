from pathlib import Path
from typing import Union

# Определяем корень проекта один раз для всего модуля
# __file__ = /raid/agubarev/predictive_analytics_simulation/src/path_utils.py
# .parent = /raid/agubarev/predictive_analytics_simulation/src/
# .parent = /raid/agubarev/predictive_analytics_simulation/ (корень проекта)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Преобразует относительный путь в абсолютный относительно PROJECT_ROOT.
    
    Примеры:
        resolve_path("data/splited/main/telemetry.csv")
        -> /raid/agubarev/predictive_analytics_simulation/data/splited/main/telemetry.csv
        
        resolve_path("logs/preprocess.log")
        -> /raid/agubarev/predictive_analytics_simulation/logs/preprocess.log
    """
    path = Path(relative_path)
    
    # Если путь уже абсолютный, возвращаем как есть
    if path.is_absolute():
        return path
    
    # Иначе резолвим относительно PROJECT_ROOT
    return PROJECT_ROOT / path


def ensure_dir(path: Union[str, Path]) -> Path:
    """Создаёт директорию, если её нет, и возвращает Path объект."""
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path