"""Paths and defaults for Expresso and ESD."""
import os

# Project root: colm/colm/config.py -> parent of colm package = project root
_COLM_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_COLM_PKG)

# Expresso: read speech 48kHz, 7 styles, 4 speakers
EXPRESSO_ROOT = os.environ.get(
    "EXPRESSO_ROOT",
    os.path.join(_PROJECT_ROOT, "expresso"),
)
EXPRESSO_READ_STYLES = (
    "default",
    "confused",
    "enunciated",
    "happy",
    "laughing",
    "sad",
    "whisper",
)
EXPRESSO_SPEAKERS = ("ex01", "ex02", "ex03", "ex04")

# ESD: 5 emotions × 10 speakers
ESD_ROOT = os.environ.get(
    "ESD_ROOT",
    os.path.join(_PROJECT_ROOT, "Emotion Speech Dataset"),
)
ESD_EMOTIONS = ("Angry", "Happy", "Neutral", "Sad", "Surprise")
