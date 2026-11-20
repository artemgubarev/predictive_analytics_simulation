import json
import mlflow
import tempfile
from pathlib import Path
from typing import Callable, Dict
from src.logger import get_logger

logger = get_logger('mlflow_utils')


# def run_or_reuse(config: dict, active_run: bool, run_name: str, fn: Callable[[], None]) -> None:
#     """Runs fn() inside a new MLflow run, or reuses the active one."""
#     if active_run:
#         fn()
#         return

#     mlflow.set_tracking_uri(config['experiment']['tracking_uri'])
#     mlflow.set_experiment(config['experiment']['name'])
#     with mlflow.start_run(run_name=run_name):
#         fn()


def log_files_to_mlflow(files: Dict[Path, str]) -> None:
    """Writes {local_path: artifact_path} pairs to the active MLflow run via a temp dir.

    Keys are target filenames inside the temp dir, values are artifact subpaths.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for local_path, artifact_path in files.items():
            local_path = Path(local_path)
            if not local_path.exists():
                raise FileNotFoundError(f'Artifact to log does not exist: {local_path}')
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


def log_json(data: dict, filename: str, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)