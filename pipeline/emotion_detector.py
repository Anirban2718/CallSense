"""
pipeline/emotion_detector.py
──────────────────────────────
Voice-based emotion detection from acoustic features.

Two modes
---------
1. **Rule-based** (default, no extra model download)
   Uses MFCC + pitch + energy statistics to classify into 7 emotion categories
   via a hand-crafted feature heuristic — fast and interpretable.

2. **ML-based** (set use_ml=True)
   Loads a pre-trained SER (Speech Emotion Recognition) model from HuggingFace:
   "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
   Requires ~1.2 GB download on first run.

Emotions: angry | frustrated | sad | neutral | calm | happy | excited
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .audio_processor import AudioFeatures

logger = logging.getLogger(__name__)

# Emotion label → satisfaction polarity
EMOTION_POLARITY: Dict[str, float] = {
    "angry":      -1.0,
    "frustrated": -0.75,
    "sad":        -0.50,
    "neutral":     0.0,
    "calm":        0.35,
    "happy":       0.85,
    "excited":     0.65,
}

ALL_EMOTIONS = list(EMOTION_POLARITY.keys())


@dataclass
class EmotionResult:
    dominant_emotion: str
    emotion_scores:   Dict[str, float]   # each 0–1, sums to 1
    polarity:         float              # –1 … +1
    confidence:       float              # probability of dominant class

    def to_dict(self) -> dict:
        return {
            "dominant_emotion": self.dominant_emotion,
            "emotion_scores":   {k: round(v, 3) for k, v in self.emotion_scores.items()},
            "polarity":         round(self.polarity, 3),
            "confidence":       round(self.confidence, 3),
        }


class EmotionDetector:
    """
    Detect vocal emotions from audio.

    Parameters
    ----------
    use_ml    : bool   Use wav2vec2 SER model (True) or rule-based (False)
    device    : str    "cpu" or "cuda"
    """

    def __init__(self, use_ml: bool = False, device: str = "cpu"):
        self.use_ml = use_ml
        self.device = device
        self._model = None
        self._feature_extractor = None

    # ── Lazy load ML model ─────────────────────────────────────────────────

    def _load_ml(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            import torch

            model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            logger.info("Loading SER model: %s", model_id)
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(model_id)
            self._model.eval()
            logger.info("SER model ready.")

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        audio_features: AudioFeatures,
        audio_path: Optional[str] = None,
    ) -> EmotionResult:
        """
        Detect emotion from audio.

        Parameters
        ----------
        audio_features : Precomputed AudioFeatures object
        audio_path     : Path to audio file (required if use_ml=True)
        """
        if self.use_ml and audio_path:
            return self._detect_ml(audio_path)
        return self._detect_rule_based(audio_features)

    # ── Rule-based detection ───────────────────────────────────────────────

    def _detect_rule_based(self, f: AudioFeatures) -> EmotionResult:
        """
        Heuristic emotion classification using acoustic features.

        Feature dimensions used
        -----------------------
        • pitch_mean        — speaker's average pitch
        • pitch_variability — emotional expressiveness
        • energy_mean       — loudness / arousal
        • energy_variability
        • speaking_rate_wpm — pacing
        • silence_ratio     — pausing / disengagement
        • spectral_centroid — brightness / harshness
        """

        # Normalise energy to rough dB proxy (0–1 after log)
        energy_norm = min(1.0, max(0.0, (np.log1p(f.energy_mean * 1000) / 8)))

        # Arousal: combination of energy, pitch variability, speaking rate
        arousal = (
            0.40 * energy_norm +
            0.35 * min(f.pitch_variability, 1.0) +
            0.25 * min(f.speaking_rate_wpm / 200, 1.0)
        )

        # Valence proxy: high pitch + low ZCR → positive; high ZCR + high energy → negative
        zcr_norm    = min(1.0, float(f.zcr_mean) * 100)
        spectral_norm = min(1.0, f.spectral_centroid_mean / 4000)
        valence = (
            0.40 * (1.0 - zcr_norm) +         # low ZCR → positive
            0.30 * (1.0 - spectral_norm) +     # low spectral centroid → warmer
            0.30 * (1.0 - min(f.silence_ratio, 1.0))
        )

        # Tension: energy variability + high spectral → frustration/anger
        tension = (
            0.50 * min(f.energy_variability, 1.0) +
            0.50 * spectral_norm
        )

        # ── Map (arousal, valence, tension) → emotion scores ──────────────

        scores: Dict[str, float] = {
            # High arousal + low valence + high tension → angry
            "angry":      arousal * (1 - valence) * tension,
            # Moderate arousal + low valence → frustrated
            "frustrated": (arousal * 0.7) * (1 - valence) * (1 - tension * 0.5),
            # Low arousal + low valence → sad
            "sad":        (1 - arousal) * (1 - valence),
            # Low arousal, moderate valence → neutral
            "neutral":    (1 - arousal) * valence * 0.8,
            # Moderate arousal + moderate valence → calm
            "calm":       (1 - min(arousal, 0.7)) * valence,
            # High arousal + high valence → happy
            "happy":      arousal * valence * (1 - tension),
            # Very high arousal + high valence → excited
            "excited":    min(arousal * 1.3, 1.0) * valence * (1 - tension * 0.3),
        }

        # Normalise to probability distribution
        total = sum(scores.values()) + 1e-9
        scores = {k: v / total for k, v in scores.items()}

        dominant = max(scores, key=scores.get)
        polarity = sum(EMOTION_POLARITY[e] * p for e, p in scores.items())

        return EmotionResult(
            dominant_emotion = dominant,
            emotion_scores   = scores,
            polarity         = np.clip(polarity, -1.0, 1.0),
            confidence       = scores[dominant],
        )

    # ── ML-based detection ─────────────────────────────────────────────────

    def _detect_ml(self, audio_path: str) -> EmotionResult:
        """Use wav2vec2 fine-tuned for speech emotion recognition."""
        import torch
        import librosa

        self._load_ml()

        y, sr = librosa.load(audio_path, sr=16_000, mono=True)

        inputs = self._feature_extractor(
            y, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[0].numpy()

        labels = self._model.config.id2label
        raw_scores = {labels[i].lower(): float(p) for i, p in enumerate(probs)}

        # Map model labels → our 7-class scheme
        scores = _map_to_standard_emotions(raw_scores)
        total  = sum(scores.values()) + 1e-9
        scores = {k: v / total for k, v in scores.items()}

        dominant = max(scores, key=scores.get)
        polarity = sum(EMOTION_POLARITY[e] * p for e, p in scores.items())

        return EmotionResult(
            dominant_emotion = dominant,
            emotion_scores   = scores,
            polarity         = float(np.clip(polarity, -1.0, 1.0)),
            confidence       = scores[dominant],
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _map_to_standard_emotions(raw: Dict[str, float]) -> Dict[str, float]:
    """Map arbitrary SER model labels to our 7-class scheme."""
    mapping = {
        "angry":     "angry",
        "anger":     "angry",
        "sad":       "sad",
        "sadness":   "sad",
        "happy":     "happy",
        "happiness": "happy",
        "joy":       "happy",
        "neutral":   "neutral",
        "calm":      "calm",
        "disgusted": "frustrated",
        "disgust":   "frustrated",
        "fear":      "frustrated",
        "fearful":   "frustrated",
        "excited":   "excited",
        "surprise":  "excited",
        "surprised": "excited",
    }
    out = {e: 0.0 for e in ALL_EMOTIONS}
    for raw_label, score in raw.items():
        canonical = mapping.get(raw_label, "neutral")
        out[canonical] += score
    return out
