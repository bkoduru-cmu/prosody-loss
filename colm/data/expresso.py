"""
Expresso read-speech loader for prosody-contrastive CKA.

Same-text parallel utterances across 7 read styles (default, confused, enunciated,
happy, laughing, sad, whisper) and 4 speakers. Supports contrastive emphasis
subset (sentences with * in transcript).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..config import EXPRESSO_ROOT, EXPRESSO_READ_STYLES, EXPRESSO_SPEAKERS


def _normalize_text_for_grouping(text: str, strip_emphasis: bool = True) -> str:
    """Normalize text for grouping same-sentence across styles. Optionally strip *emphasis*."""
    t = text.strip()
    if strip_emphasis:
        t = re.sub(r"\*[^*]*\*", "", t).strip()
        t = re.sub(r"\s+", " ", t)
    return t


def load_expresso_transcriptions(
    root: str | Path | None = None,
    path: str | Path | None = None,
) -> dict[str, str]:
    """
    Load read_transcriptions.txt: utterance_id -> text.
    utterance_id format: ex01_confused_00001 (speaker_style_id).
    """
    root = Path(root or EXPRESSO_ROOT)
    fp = path or root / "read_transcriptions.txt"
    out = {}
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            uid, text = line.split("\t", 1)
            out[uid] = text
    return out


def get_expresso_audio_path(
    root: str | Path,
    speaker: str,
    style: str,
    utterance_id: str,
    corpus: str = "base",
) -> Path:
    """
    Return path to wav for read utterance.
    utterance_id is the numeric part, e.g. 00001, or full ex01_confused_00001 (id part used).
    """
    root = Path(root)
    if "_" in str(utterance_id):
        # ex01_confused_00001 -> id 00001
        utterance_id = str(utterance_id).split("_")[-1]
    fname = f"{speaker}_{style}_{utterance_id}.wav"
    return root / "audio_48khz" / "read" / speaker / style / corpus / fname


def build_expresso_parallel_groups(
    root: str | Path | None = None,
    styles: Iterable[str] | None = None,
    speakers: Iterable[str] | None = None,
    strip_emphasis_for_grouping: bool = True,
    min_styles_per_group: int = 2,
    include_emphasis_subset: bool = True,
    group_by_sentence_id: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Build parallel same-text groups for CKA: same sentence in different styles.

    If group_by_sentence_id is True (default), groups by (speaker, numeric_id) so that
    ex01_default_00001, ex01_confused_00001, ... are one group (same script sentence).
    Otherwise groups by normalized text.

    Returns:
        parallel_groups: list of {
            "text_norm": normalized text,
            "text_raw": one raw text (first seen),
            "utterances": [{"speaker", "style", "uid", "path", "has_emphasis"}],
        }
        emphasis_groups: same structure for contrastive emphasis subset (text contains *).
    """
    root = Path(root or EXPRESSO_ROOT)
    styles = list(styles or EXPRESSO_READ_STYLES)
    speakers = list(speakers or EXPRESSO_SPEAKERS)

    transcriptions = load_expresso_transcriptions(root)

    if group_by_sentence_id:
        # key: (speaker, num_id) -> list of (style, uid, text)
        by_key: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for uid, text in transcriptions.items():
            parts = uid.split("_")
            if len(parts) != 3:
                continue
            speaker, style, num = parts
            if speaker not in speakers or style not in styles:
                continue
            path = get_expresso_audio_path(root, speaker, style, num)
            if not path.exists():
                continue
            key = (speaker, num)
            by_key[key].append((style, uid, text))
    else:
        by_key = defaultdict(list)
        for uid, text in transcriptions.items():
            parts = uid.split("_")
            if len(parts) != 3:
                continue
            speaker, style, num = parts
            if speaker not in speakers or style not in styles:
                continue
            path = get_expresso_audio_path(root, speaker, style, num)
            if not path.exists():
                continue
            text_norm = _normalize_text_for_grouping(text, strip_emphasis=strip_emphasis_for_grouping)
            if not text_norm:
                continue
            key = (speaker, text_norm)
            by_key[key].append((style, uid, text))

    parallel_groups = []
    emphasis_groups = []

    for key, style_list in by_key.items():
        if len(style_list) < min_styles_per_group:
            continue
        by_style = {}
        for style, uid, raw in style_list:
            if style not in by_style:
                by_style[style] = (uid, raw)
        if len(by_style) < min_styles_per_group:
            continue
        speaker = key[0]
        text_norm = key[1] if not group_by_sentence_id else _normalize_text_for_grouping(
            style_list[0][2], strip_emphasis=strip_emphasis_for_grouping
        )
        utterances = []
        has_emphasis = False
        for style, (uid, raw) in by_style.items():
            path = get_expresso_audio_path(root, speaker, style, uid)
            utterances.append({
                "speaker": speaker,
                "style": style,
                "uid": uid,
                "path": str(path),
                "has_emphasis": "*" in raw,
            })
            if "*" in raw:
                has_emphasis = True
        raw_text = style_list[0][2]
        group = {
            "text_norm": text_norm,
            "text_raw": raw_text,
            "utterances": utterances,
        }
        if has_emphasis and include_emphasis_subset:
            emphasis_groups.append(group)
        parallel_groups.append(group)

    return parallel_groups, emphasis_groups


def load_expresso_splits(root: str | Path | None = None) -> dict[str, list[str]]:
    """Load train/dev/test split file names (conversational ids). Not used for read-speech CKA."""
    root = Path(root or EXPRESSO_ROOT)
    out = {}
    for name in ("train", "dev", "test"):
        p = root / "splits" / f"{name}.txt"
        if not p.exists():
            out[name] = []
            continue
        with open(p, "r", encoding="utf-8") as f:
            out[name] = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return out
