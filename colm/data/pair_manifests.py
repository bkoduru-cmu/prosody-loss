"""
Load pre-built same-text pair manifests (JSON) for Expresso and ESD.

Use these when you already have pair lists like:
  - expresso_local_2.json: style1, style2, audio1_path, audio2_path
  - esp_local.json (ESD): style1, style2, audio1_path, audio2_path
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_expresso_pairs_manifest(path: str | Path) -> list[dict]:
    """
    Load Expresso pair manifest (e.g. expresso_local_2.json).
    Returns list of dicts with: audio1_path, audio2_path, text, style1, style2, style_pair, ...
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs


def load_esd_pairs_manifest(path: str | Path) -> list[dict]:
    """
    Load ESD pair manifest (e.g. esp_local.json).
    Returns list of dicts with: audio1_path, audio2_path, style1, style2, pair, ...
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return pairs


def iter_pairs_for_cka(manifest_path: str | Path, dataset: str = "auto") -> Iterator[tuple[str, str, dict]]:
    """
    Yield (audio1_path, audio2_path, meta) for each pair in the manifest.
    dataset: "expresso", "esd", or "auto" (guess from keys: "style_pair" -> expresso, "pair" -> esd).
    """
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if dataset == "auto":
        sample = pairs[0] if pairs else {}
        dataset = "expresso" if "style_pair" in sample else "esd"

    for p in pairs:
        a1 = p.get("audio1_path") or p.get("audio_path_1")
        a2 = p.get("audio2_path") or p.get("audio_path_2")
        if a1 and a2:
            yield (a1, a2, p)
