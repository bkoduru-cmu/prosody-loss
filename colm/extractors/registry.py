"""
Registry: model_id -> (load_fn, make_extract_fn).
load_fn(device, cache_dir, ...) -> model
make_extract_fn(model) -> extract_fn(audio_paths: list[str]) -> list[np.ndarray] of shape (T, D)
"""
from __future__ import annotations

from typing import Callable

REGISTRY: dict[str, tuple[Callable, Callable]] = {}


def register(model_id: str, load_fn: Callable, make_extract_fn: Callable) -> None:
    REGISTRY[model_id] = (load_fn, make_extract_fn)


def get_loader(model_id: str) -> Callable | None:
    entry = REGISTRY.get(model_id)
    return entry[0] if entry else None


def get_extract_fn_factory(model_id: str) -> Callable | None:
    entry = REGISTRY.get(model_id)
    return entry[1] if entry else None


def available_models() -> list[str]:
    return list(REGISTRY.keys())


def run_all_models(
    model_ids: list[str],
    run_cka_fn: Callable,
    esd_manifest: str,
    expresso_manifest: str,
    max_pairs_esd: int | None = 200,
    max_pairs_expresso: int | None = 200,
    device: str = "cuda",
    cache_dir: str | None = None,
    save_dir: str = ".",
) -> dict:
    """
    For each model_id: load model, get extractor, run CKA on ESD and Expresso, save results.
    run_cka_fn(manifest_path, extract_fn, max_pairs, pair_key) -> dict of style_pair -> list of CKA values.
    Returns {model_id: {'esd': results_esd, 'expresso': results_expresso}} or skip on error.
    """
    import json
    import os

    outcomes = {}
    for model_id in model_ids:
        loader = get_loader(model_id)
        make_extract = get_extract_fn_factory(model_id)
        if loader is None or make_extract is None:
            print(f"  Skip {model_id}: no extractor registered.")
            continue
        try:
            model_or_tuple = loader(device=device, cache_dir=cache_dir)
            extract_fn = make_extract(model_or_tuple, device=device)
            pair_key_esd = "pair"
            pair_key_ex = "style_pair"
            res_esd = run_cka_fn(esd_manifest, extract_fn, max_pairs=max_pairs_esd, pair_key=pair_key_esd)
            res_expresso = run_cka_fn(expresso_manifest, extract_fn, max_pairs=max_pairs_expresso, pair_key=pair_key_ex)
            summary_esd = {k: {"mean": float(__import__("numpy").mean(v)), "std": float(__import__("numpy").std(v)), "n": len(v)} for k, v in res_esd.items()}
            summary_ex = {k: {"mean": float(__import__("numpy").mean(v)), "std": float(__import__("numpy").std(v)), "n": len(v)} for k, v in res_expresso.items()}
            esd_path = os.path.join(save_dir, f"results_{model_id.replace('-', '_')}_esd.json")
            ex_path = os.path.join(save_dir, f"results_{model_id.replace('-', '_')}_expresso.json")
            with open(esd_path, "w") as f:
                json.dump(summary_esd, f, indent=2)
            with open(ex_path, "w") as f:
                json.dump(summary_ex, f, indent=2)
            outcomes[model_id] = {"esd": res_esd, "expresso": res_expresso}
            print(f"  OK {model_id}: ESD -> {esd_path}, Expresso -> {ex_path}")
        except Exception as e:
            print(f"  Fail {model_id}: {e}")
            continue
    return outcomes
