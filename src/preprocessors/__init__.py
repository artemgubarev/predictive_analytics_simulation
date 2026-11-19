from src.features.v1 import FeatureBuilderV1
from src.features.v2 import FeatureBuilderV2
from .catboost import CatBoostPreprocessor

FEATURE_VERSIONS = {
    'v1': FeatureBuilderV1,
    'v2': FeatureBuilderV2,
}

def get_preprocessor(
    model_family: str, 
    feature_version: str, 
    target_col: str = 'EnergyConsumption', 
    time_col: str = 'Timestamp'
):
    if feature_version not in FEATURE_VERSIONS:
        raise ValueError(f"Unknown feature version: {feature_version}")
        
    builder = FEATURE_VERSIONS[feature_version](time_col=time_col)
    
    if model_family == 'catboost':
        return CatBoostPreprocessor(feature_builder=builder, target_col=target_col, time_col=time_col)
        
    raise ValueError(f"Unknown model family: {model_family}")