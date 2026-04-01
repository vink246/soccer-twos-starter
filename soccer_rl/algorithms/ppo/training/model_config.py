from copy import deepcopy
from typing import Any, Dict


MODEL_PRESETS = {
    "small": {
        "vf_share_layers": True,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "baseline": {
        "vf_share_layers": True,
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "relu",
    },
    "large": {
        "vf_share_layers": True,
        "fcnet_hiddens": [1024, 512, 256],
        "fcnet_activation": "relu",
    },
    "residual_mlp": {
        "vf_share_layers": False,
        "fcnet_hiddens": [512, 512, 512],
        "fcnet_activation": "relu",
    },
}


def build_model_config(rllib_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build model config from preset + explicit overrides.
    Explicit keys in rllib.model override preset values.
    """
    model_cfg = deepcopy(rllib_cfg.get("model") or {})
    preset_name = model_cfg.pop("preset", None)
    if preset_name and preset_name in MODEL_PRESETS:
        merged = deepcopy(MODEL_PRESETS[preset_name])
        merged.update(model_cfg)
        return merged
    if not model_cfg:
        return deepcopy(MODEL_PRESETS["baseline"])
    return model_cfg
