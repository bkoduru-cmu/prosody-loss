"""
Run CKA on prosody-contrastive pairs.

Step 1: Load dataset (Expresso or ESD) and build parallel groups.
Step 2: Load audio and extract representations (placeholder: random vectors; plug in your model).
Step 3: Compute CKA between same-text different-style/emotion representations.

Usage:
  python -m colm.scripts.run_cka --dataset expresso --max-groups 20 --output cka_expresso.npz
  python -m colm.scripts.run_cka --dataset esd --max-groups 20 --output cka_esd.npz
  # Or use pre-built pair manifests (e.g. from interspeech):
  python -m colm.scripts.run_cka --manifest /path/to/expresso_local_2.json --max-pairs 100 --output cka.npz
  python -m colm.scripts.run_cka --manifest /path/to/esp_local.json --max-pairs 100 --output cka_esd.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from colm.data.expresso import build_expresso_parallel_groups
from colm.data.esd import build_esd_parallel_groups
from colm.data.pair_manifests import iter_pairs_for_cka
from colm.cka import linear_cka


def dummy_extract(audio_paths: list[str], feat_dim: int = 64) -> list[np.ndarray]:
    """
    Placeholder: return random features per path. Replace with real model forward pass
    (e.g. encoder or projector hidden states) that returns (T, D) per utterance.
    """
    return [np.random.randn(50, feat_dim).astype(np.float32) for _ in audio_paths]


def run_cka_expresso(
    root: str,
    max_groups: int,
    min_styles: int,
    feat_dim: int,
    extract_fn,
) -> dict:
    parallel, emphasis = build_expresso_parallel_groups(
        root=root,
        min_styles_per_group=min_styles,
    )
    groups = parallel[:max_groups]
    if not groups:
        return {"cka_mean": np.nan, "cka_per_group": [], "n_groups": 0}

    ckas_per_group = []
    for g in groups:
        paths = [u["path"] for u in g["utterances"]]
        reprs = extract_fn(paths, feat_dim)
        # Pairwise CKA within group (same text, different style)
        n = len(reprs)
        for i in range(n):
            for j in range(i + 1, n):
                c = linear_cka(reprs[i], reprs[j])
                ckas_per_group.append(c)
    return {
        "cka_mean": float(np.mean(ckas_per_group)) if ckas_per_group else np.nan,
        "cka_std": float(np.std(ckas_per_group)) if ckas_per_group else np.nan,
        "cka_per_group": ckas_per_group,
        "n_groups": len(groups),
        "n_pairs": len(ckas_per_group),
    }


def run_cka_esd(
    root: str,
    max_groups: int,
    min_emotions: int,
    feat_dim: int,
    extract_fn,
) -> dict:
    groups = build_esd_parallel_groups(
        root=root,
        min_emotions_per_group=min_emotions,
    )
    groups = groups[:max_groups]
    if not groups:
        return {"cka_mean": np.nan, "cka_per_group": [], "n_groups": 0}

    ckas_per_group = []
    for g in groups:
        paths = [u["path"] for u in g["utterances"]]
        reprs = extract_fn(paths, feat_dim)
        n = len(reprs)
        for i in range(n):
            for j in range(i + 1, n):
                c = linear_cka(reprs[i], reprs[j])
                ckas_per_group.append(c)
    return {
        "cka_mean": float(np.mean(ckas_per_group)) if ckas_per_group else np.nan,
        "cka_std": float(np.std(ckas_per_group)) if ckas_per_group else np.nan,
        "cka_per_group": ckas_per_group,
        "n_groups": len(groups),
        "n_pairs": len(ckas_per_group),
    }


def run_cka_from_manifest(
    manifest_path: str,
    max_pairs: int | None,
    feat_dim: int,
    extract_fn,
) -> dict:
    """
    Run CKA on pre-built pairs from a JSON manifest (e.g. expresso_local_2.json or esp_local.json).
    Each pair is (audio1_path, audio2_path); we extract features and compute one CKA per pair.
    """
    pairs = list(iter_pairs_for_cka(manifest_path, dataset="auto"))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        return {"cka_mean": np.nan, "cka_per_group": [], "n_groups": 0, "n_pairs": 0}

    ckas = []
    for a1, a2, _ in pairs:
        reprs = extract_fn([a1, a2], feat_dim)
        if len(reprs) == 2:
            c = linear_cka(reprs[0], reprs[1])
            ckas.append(c)
    return {
        "cka_mean": float(np.mean(ckas)) if ckas else np.nan,
        "cka_std": float(np.std(ckas)) if ckas else np.nan,
        "cka_per_group": ckas,
        "n_groups": 0,
        "n_pairs": len(ckas),
    }


def main():
    from colm.config import EXPRESSO_ROOT, ESD_ROOT

    p = argparse.ArgumentParser(description="Run CKA on prosody-contrastive pairs")
    p.add_argument("--dataset", choices=("expresso", "esd"), default="expresso", help="Used when not using --manifest")
    p.add_argument("--manifest", default=None, help="JSON manifest of pairs (audio1_path, audio2_path). Overrides --dataset/--root.")
    p.add_argument("--root", default=None, help="Dataset root (default: EXPRESSO_ROOT or ESD_ROOT)")
    p.add_argument("--max-groups", type=int, default=50, help="Max groups when building from dataset (ignored if --manifest)")
    p.add_argument("--max-pairs", type=int, default=None, help="Max pairs when using --manifest (default: all)")
    p.add_argument("--min-styles", type=int, default=2, help="Expresso: min styles per group")
    p.add_argument("--min-emotions", type=int, default=2, help="ESD: min emotions per group")
    p.add_argument("--feat-dim", type=int, default=64, help="Feature dim for placeholder extractor")
    p.add_argument("--output", default="cka_results.npz", help="Output .npz file")
    args = p.parse_args()

    extract_fn = dummy_extract  # Replace with your model hook

    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"Manifest not found: {args.manifest}")
            sys.exit(1)
        out = run_cka_from_manifest(args.manifest, args.max_pairs, args.feat_dim, extract_fn)
    else:
        root = args.root or (EXPRESSO_ROOT if args.dataset == "expresso" else ESD_ROOT)
        if not os.path.isdir(root):
            print(f"Dataset root not found: {root}. Set EXPRESSO_ROOT or ESD_ROOT or --root.")
            sys.exit(1)
        if args.dataset == "expresso":
            out = run_cka_expresso(root, args.max_groups, args.min_styles, args.feat_dim, extract_fn)
        else:
            out = run_cka_esd(root, args.max_groups, args.min_emotions, args.feat_dim, extract_fn)

    np.savez(
        args.output,
        cka_mean=out["cka_mean"],
        cka_std=out.get("cka_std", np.nan),
        cka_per_group=out.get("cka_per_group", []),
        n_groups=out["n_groups"],
        n_pairs=out.get("n_pairs", 0),
        dataset=args.dataset,
    )
    print(f"Groups: {out['n_groups']}, Pairs: {out.get('n_pairs', 0)}")
    print(f"CKA mean: {out['cka_mean']:.4f} (std: {out.get('cka_std', np.nan):.4f})")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
