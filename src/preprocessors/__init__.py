from src.features import get_feature_extractor
from .catboost import CatBoostPreprocessor

MODEL_PREPROCESSORS = {
    'catboost': CatBoostPreprocessor,
}

def get_preprocessor(
    model_family: str, 
    feature_version: str, 
    target_col: str = 'EnergyConsumption', 
    time_col: str = 'Timestamp'
):
    if model_family not in MODEL_PREPROCESSORS:
        raise ValueError(f"Unknown model family: {model_family}")

    # Делегируем создание и проверку версии фичей модулю features
    builder = get_feature_extractor(version=feature_version, time_col=time_col)
    
    preprocessor_cls = MODEL_PREPROCESSORS[model_family]
    return preprocessor_cls(feature_builder=builder, target_col=target_col, time_col=time_col)