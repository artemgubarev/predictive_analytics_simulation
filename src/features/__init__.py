from src.features.base import BaseFeatureBuilder
from src.features.v1 import EquipmentFeatureBuilderV1

FEATURE_BUILDERS: dict[str, type[BaseFeatureBuilder]] = {
    "v1": EquipmentFeatureBuilderV1,
    # "v2": PdMFeatureBuilderV2,
}

__all__ = ["BaseFeatureBuilder", "EquipmentFeatureBuilderV1", "FEATURE_BUILDERS"]