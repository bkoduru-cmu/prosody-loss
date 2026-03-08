"""
ESD (Emotional Speech Dataset) loader for emotion-contrastive CKA.

Parallel structure: 350 sentences × 5 emotions × 10 speakers (English + Mandarin).
Same sentence id maps to different emotion files via offset: Neutral 1-350, Angry 351-700,
Happy 701-1050, Sad 1051-1400, Surprise 1401-1750 per speaker.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ..config import ESD_ROOT, ESD_EMOTIONS

# ESD uses 350 parallel utterances per emotion; file ids are offset by 350 per emotion
ESD_UTTERANCES_PER_EMOTION = 350


def _emotion_to_offset(emotion: str) -> int:
    order = {e: i for i, e in enumerate(ESD_EMOTIONS)}
    idx = order.get(emotion, 0)
    return idx * ESD_UTTERANCES_PER_EMOTION


def load_esd_metadata(
    root: str | Path | None = None,
    speaker: str | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Load speaker's .txt: utterance_id -> (text, emotion).
    utterance_id format: 0001_000001. Returns all utterances with text and emotion label.
    """
    root = Path(root or ESD_ROOT)
    out = {}
    speakers = [speaker] if speaker else [d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    for spk in speakers:
        txt_path = root / spk / f"{spk}.txt"
        if not txt_path.exists():
            continue
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t")
                uid = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                emotion_cn = parts[2] if len(parts) > 2 else ""
                # Map Chinese labels to folder names
                emotion = _cn_emotion_to_en(emotion_cn)
                out[uid] = (text, emotion)
    return out


def _cn_emotion_to_en(label: str) -> str:
    m = {
        "中立": "Neutral",
        "生气": "Angry",
        "开心": "Happy",
        "悲伤": "Sad",
        "惊讶": "Surprise",
    }
    return m.get(label.strip(), label)


def get_esd_audio_path(
    root: str | Path,
    speaker: str,
    emotion: str,
    utterance_id: str,
) -> Path:
    """Return path to wav. utterance_id is full e.g. 0001_000351."""
    root = Path(root)
    if "_" in utterance_id:
        uid = utterance_id
    else:
        uid = f"{speaker}_{utterance_id}"
    return root / speaker / emotion / f"{uid}.wav"


def _sentence_index_from_uid(uid: str) -> int:
    """Global utterance number (1-based) -> sentence index (0..349)."""
    parts = uid.split("_")
    if len(parts) != 2:
        return -1
    num = int(parts[1])
    return (num - 1) % ESD_UTTERANCES_PER_EMOTION


def build_esd_parallel_groups(
    root: str | Path | None = None,
    speakers: Iterable[str] | None = None,
    emotions: Iterable[str] | None = None,
    min_emotions_per_group: int = 2,
) -> list[dict]:
    """
    Build parallel same-sentence groups across emotions for CKA.
    Each group has one sentence index, one speaker, and one wav per available emotion.

    Returns:
        list of {
            "speaker": str,
            "sentence_index": int,
            "text": str,
            "utterances": [{"emotion", "uid", "path"}],
        }
    """
    root = Path(root or ESD_ROOT)
    emotions = list(emotions or ESD_EMOTIONS)
    speakers = list(speakers) if speakers else [d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()]

    metadata = load_esd_metadata(root)
    # (speaker, sentence_index) -> list of (emotion, uid, text)
    by_speaker_sent: dict[tuple[str, int], list[tuple[str, str, str]]] = {}
    for uid, (text, emotion) in metadata.items():
        parts = uid.split("_")
        if len(parts) != 2 or emotion not in emotions:
            continue
        spk = parts[0]
        if spk not in speakers:
            continue
        sent_idx = _sentence_index_from_uid(uid)
        if sent_idx < 0:
            continue
        path = get_esd_audio_path(root, spk, emotion, uid)
        if not path.exists():
            continue
        key = (spk, sent_idx)
        if key not in by_speaker_sent:
            by_speaker_sent[key] = []
        by_speaker_sent[key].append((emotion, uid, text))

    groups = []
    for (speaker, sent_idx), emotion_list in by_speaker_sent.items():
        if len(emotion_list) < min_emotions_per_group:
            continue
        by_emotion = {e: (uid, text) for e, uid, text in emotion_list}
        utterances = []
        text = emotion_list[0][2]
        for emotion, (uid, _) in by_emotion.items():
            path = get_esd_audio_path(root, speaker, emotion, uid)
            utterances.append({"emotion": emotion, "uid": uid, "path": str(path)})
        groups.append({
            "speaker": speaker,
            "sentence_index": sent_idx,
            "text": text,
            "utterances": utterances,
        })
    return groups
