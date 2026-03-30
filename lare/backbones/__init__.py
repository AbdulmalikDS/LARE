from .base import BaseEncoder, EncoderConfig, EncoderOutput
from .clip import CLIPEncoder
from .siglip import SigLIPEncoder

_REGISTRY = {
    "clip":          ("clip",         CLIPEncoder),
    "clip-large":    ("clip-large",   CLIPEncoder),
    "clip-336":      ("clip-336",     CLIPEncoder),
    "siglip":        ("siglip",       SigLIPEncoder),
    "siglip-so400m": ("siglip-so400m", SigLIPEncoder),
    "siglip2":       ("siglip2",      SigLIPEncoder),
}


def create_encoder(name: str, device: str = None) -> BaseEncoder:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(_REGISTRY)}")
    model_id, cls = _REGISTRY[key]
    return cls(model_id, device=device)


__all__ = [
    "BaseEncoder", "EncoderConfig", "EncoderOutput",
    "CLIPEncoder", "SigLIPEncoder",
    "create_encoder",
]
