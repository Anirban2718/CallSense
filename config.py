"""
config.py — Central configuration for the Customer Satisfaction AI system.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

for d in [UPLOAD_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Whisper ────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")   # tiny|base|small|medium|large
WHISPER_LANGUAGE  = "en"                                   # set None for auto-detect
WHISPER_DEVICE    = "cpu"                                  # "cuda" if GPU available

# ── HuggingFace NLP ────────────────────────────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# ── Audio Feature Extraction ───────────────────────────────────────────────
SAMPLE_RATE     = 16_000   # Hz — Whisper native
FRAME_LENGTH    = 2048     # librosa STFT frame
HOP_LENGTH      = 512
N_MFCC          = 13       # MFCC coefficients
SILENCE_THRESH  = -40      # dBFS

# ── Scoring Weights ────────────────────────────────────────────────────────
# Must sum to 1.0
WEIGHTS = {
    "text_sentiment": 0.35,
    "text_emotion":   0.25,
    "audio_emotion":  0.20,
    "audio_prosody":  0.20,
}

# ── Alert Thresholds ───────────────────────────────────────────────────────
ALERT_SCORE_THRESHOLD   = 35    # CSAT score below this → red alert
WARNING_SCORE_THRESHOLD = 55    # CSAT score below this → warning

# ── Flask ──────────────────────────────────────────────────────────────────
FLASK_HOST    = "0.0.0.0"
FLASK_PORT    = 5000
FLASK_DEBUG   = os.getenv("FLASK_DEBUG", "false").lower() == "true"
MAX_FILE_MB   = 100
ALLOWED_EXTS  = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}

# ── Emotion label mapping → polarity score ─────────────────────────────────
EMOTION_POLARITY = {
    # Text emotion model labels
    "joy":       1.0,
    "surprise":  0.3,
    "neutral":   0.0,
    "sadness":  -0.5,
    "fear":     -0.6,
    "disgust":  -0.8,
    "anger":    -1.0,
    # Audio emotion labels (same mapping used for voice model)
    "happy":    1.0,
    "calm":     0.4,
    "excited":  0.5,
    "sad":     -0.5,
    "fearful": -0.6,
    "frustrated": -0.7,
    "angry":   -1.0,
}
