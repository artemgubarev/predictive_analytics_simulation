from typing import Dict, Type
from .base import BaseFeatureBuilder
from .v1 import FeatureBuilderV1
from .v2 import FeatureBuilderV2

FEATURE_REGISTRY: Dict[str, Type[BaseFeatureBuilder]] = {
    'v1': FeatureBuilderV1,
    'v2': FeatureBuilderV2,
}

def get_feature_extractor(version: str = 'v2', **kwargs) -> BaseFeatureBuilder:
    """Factory function to instantiate a feature extractor by version tag."""
    if version not in FEATURE_REGISTRY:
        available = ', '.join(FEATURE_REGISTRY.keys())
        raise ValueError(f"Unknown feature version '{version}'. Available: {available}")
    return FEATURE_REGISTRY[version](**kwargs)