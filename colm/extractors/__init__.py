"""
Model extractors for prosody CKA: load model + return extract_fn(audio_paths) -> list of (T, D) arrays.

Use in one notebook: for each model_id, get loader + extractor, run CKA on both manifests, save.
"""
from .registry import register, get_loader, get_extract_fn_factory, available_models, run_all_models

__all__ = ["register", "get_loader", "get_extract_fn_factory", "available_models", "run_all_models"]

# Register built-in extractors (optional deps)
try:
    from . import qwen2  # noqa: F401
except ImportError:
    pass
