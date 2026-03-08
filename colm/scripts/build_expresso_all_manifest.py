"""
Build a manifest of ALL Expresso read-speech files (path, style, speaker).
Output: expresso_all.json — list of {"path", "style", "speaker"} for every wav under
  expresso/audio_48khz/read/{speaker}/{style}/base/*.wav (and longform if present).

Usage:
  python -m colm.scripts.build_expresso_all_manifest
  python -m colm.scripts.build_expresso_all_manifest --root /path/to/expresso --out interspeech/expresso_all.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import EXPRESSO_ROOT, EXPRESSO_SPEAKERS


def collect_all_read_paths(root: str | Path) -> list[dict]:
    root = Path(root)
    read_dir = root / "audio_48khz" / "read"
    if not read_dir.exists():
        return []

    out = []
    for speaker in EXPRESSO_SPEAKERS:
        spk_dir = read_dir / speaker
        if not spk_dir.is_dir():
            continue
        for style_dir in spk_dir.iterdir():
            if not style_dir.is_dir():
                continue
            style = style_dir.name
            for corpus in ("base", "longform"):
                corp_dir = style_dir / corpus
                if not corp_dir.is_dir():
                    continue
                for wav in corp_dir.glob("*.wav"):
                    out.append({
                        "path": str(wav.resolve()),
                        "style": style,
                        "speaker": speaker,
                    })
    return out


def main():
    p = argparse.ArgumentParser(description="Build manifest of all Expresso read-speech paths")
    p.add_argument("--root", default=EXPRESSO_ROOT, help="Expresso dataset root")
    p.add_argument("--out", default=None, help="Output JSON path (default: expresso_all.json in cwd)")
    args = p.parse_args()

    entries = collect_all_read_paths(args.root)
    out_path = Path(args.out or "expresso_all.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=0)

    print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
