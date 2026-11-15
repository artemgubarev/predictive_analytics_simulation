import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str | Path,
    level: int = logging.INFO,
    mode: str = "w",   # ← "w" = truncate (новый лог каждый раз), "a" = append
) -> None:
    """
    Настраивает root-логгер: stdout + файл.
    
    По умолчанию файл перезаписывается при каждом запуске (mode="w"),
    чтобы не захламлять логи прошлыми экспериментами.
    Исторические логи и так остаются в MLflow artifacts.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Удаляем старые хендлеры (критично при повторных вызовах в тестах/ноутбуках)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File (mode="w" → перезапись)
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # в файл пишем всё, включая DEBUG
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)