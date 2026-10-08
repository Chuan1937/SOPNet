from typing import Any, Dict, Optional

from sopnet.models.classifier import SOPNetCls
from sopnet.models.sopnet import SOPNet, SOPNetConfig, SOPNetMulti

__all__ = ["SOPNet", "SOPNetConfig", "SOPNetCls", "SOPNetMulti", "count_parameters", "build_model"]


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(config: Optional[Dict[str, Any]] = None):
    """Instantiate a model from a resolved run config (``config['model']``)."""
    config = config or {}
    model_config = config.get("model", config)
    name = str(model_config.get("name", "sopnet"))
    options = {
        "stem_channels": int(model_config.get("stem_channels", 48)),
        "encoder_channels": tuple(model_config.get("encoder_channels", (64, 96, 128))),
        "bottleneck_channels": int(model_config.get("bottleneck_channels", 192)),
        "dilations": tuple(model_config.get("dilations", (1, 2, 4, 8))),
        "window_length": int(model_config.get("window_length", 400)),
        "dropout": float(model_config.get("dropout", 0.0)),
    }
    sop_config = SOPNetConfig(**options)
    if name in ("sopnet_cls", "classifier", "cls"):
        return SOPNetCls(sop_config)
    if name in ("sopnet_multi", "multitask", "multi"):
        return SOPNetMulti(sop_config)
    if name == "sopnet":
        return SOPNet(sop_config)
    raise ValueError(f"Unknown model name '{name}'")
