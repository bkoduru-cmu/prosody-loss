"""
Qwen2-Audio-7B: encoder -> projector -> LLM. Extract projector output (or encoder last_hidden_state).
"""
from __future__ import annotations

import numpy as np

from .registry import register


def load_qwen2(device: str = "cuda", cache_dir: str | None = None, **kwargs):
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else device,
        cache_dir=cache_dir,
        **kwargs,
    )
    model.eval()
    return model, processor


def make_qwen2_extractor(model_and_processor, device: str = "cuda"):
    """model_and_processor = (model, processor). Returns extract_fn(audio_paths) -> list of (T, D) np.ndarray."""
    import torch
    import librosa

    model, processor = model_and_processor

    def load_audio(path):
        audio, _ = librosa.load(path, sr=16000)
        return audio

    def extract(audio_paths):
        reprs = []
        for path in audio_paths:
            audio = load_audio(path)
            inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                enc = model.audio_tower(**inputs)
                proj = model.multi_modal_projector(enc.last_hidden_state)
            x = proj[0].cpu().float().numpy()
            reprs.append(x)
        return reprs

    return extract


def _loader(device: str = "cuda", cache_dir: str | None = None, **kwargs):
    """Returns (model, processor) for Qwen2-Audio."""
    return load_qwen2(device=device, cache_dir=cache_dir, **kwargs)


def _make_extract(model_or_tuple, device: str = "cuda"):
    return make_qwen2_extractor(model_or_tuple, device=device)


register("Qwen2-Audio-7B", _loader, _make_extract)
