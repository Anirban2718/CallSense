"""
pipeline/sentiment_analyzer.py
────────────────────────────────
Text-based sentiment + emotion analysis using HuggingFace transformers.

Models used (downloaded on first run):
  • Sentiment : cardiffnlp/twitter-roberta-base-sentiment-latest
                → labels: Negative / Neutral / Positive
  • Emotion   : j-hartmann/emotion-english-distilroberta-base
                → labels: anger | disgust | fear | joy | neutral | sadness | surprise
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    # Overall
    sentiment_label: str          # "positive" | "neutral" | "negative"
    sentiment_score: float        # 0–1 (probability of predicted label)
    sentiment_polarity: float     # –1 … +1 continuous

    # Emotion
    dominant_emotion: str
    emotion_scores: Dict[str, float]

    # Per-segment (optional)
    segment_sentiments: List[dict] = field(default_factory=list)

    # Derived
    negativity_indicators: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentiment_label":    self.sentiment_label,
            "sentiment_score":    round(self.sentiment_score, 3),
            "sentiment_polarity": round(self.sentiment_polarity, 3),
            "dominant_emotion":   self.dominant_emotion,
            "emotion_scores":     {k: round(v, 3) for k, v in self.emotion_scores.items()},
            "segment_sentiments": self.segment_sentiments,
            "negativity_indicators": self.negativity_indicators,
            "key_phrases":        self.key_phrases,
        }


# Polarity map for RoBERTa-sentiment labels
_SENTIMENT_POLARITY = {
    "positive": 1.0,
    "neutral":  0.0,
    "negative": -1.0,
}

# Emotion → polarity contribution
_EMOTION_POLARITY = {
    "joy":      1.0,
    "surprise": 0.2,
    "neutral":  0.0,
    "sadness": -0.5,
    "fear":    -0.6,
    "disgust": -0.8,
    "anger":   -1.0,
}

# Phrases that signal customer frustration / dissatisfaction
_NEGATIVE_KEYWORDS = [
    "unacceptable", "terrible", "horrible", "worst", "never again",
    "waste of time", "useless", "incompetent", "ridiculous", "outrageous",
    "still broken", "not fixed", "no solution", "transfer me again",
    "been waiting", "hours on hold", "not helpful", "angry", "furious",
    "cancel", "refund", "complaint", "disgusted", "disappointed",
]

# Phrases that signal satisfaction
_POSITIVE_KEYWORDS = [
    "thank you so much", "great help", "very helpful", "resolved",
    "perfect", "excellent", "appreciate", "satisfied", "amazing",
    "you're the best", "quick", "efficient", "will recommend",
]


class SentimentAnalyzer:
    """
    Dual-model NLP pipeline for sentiment + emotion analysis.

    Parameters
    ----------
    sentiment_model : str   HuggingFace model ID for sentiment classification
    emotion_model   : str   HuggingFace model ID for emotion classification
    device          : int   -1 = CPU, 0+ = GPU index
    """

    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        emotion_model: str   = "j-hartmann/emotion-english-distilroberta-base",
        device: int = -1,
    ):
        self.sentiment_model_name = sentiment_model
        self.emotion_model_name   = emotion_model
        self.device = device
        self._sentiment_pipe = None
        self._emotion_pipe   = None

    # ── Lazy loading ───────────────────────────────────────────────────────

    def _load(self):
        if self._sentiment_pipe is None:
            from transformers import pipeline
            logger.info("Loading sentiment model: %s", self.sentiment_model_name)
            self._sentiment_pipe = pipeline(
                "text-classification",
                model=self.sentiment_model_name,
                device=self.device,
                top_k=None,          # return all scores
                truncation=True,
                max_length=512,
            )
        if self._emotion_pipe is None:
            from transformers import pipeline
            logger.info("Loading emotion model: %s", self.emotion_model_name)
            self._emotion_pipe = pipeline(
                "text-classification",
                model=self.emotion_model_name,
                device=self.device,
                top_k=None,
                truncation=True,
                max_length=512,
            )

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(
        self,
        text: str,
        segments: Optional[List[dict]] = None,
        customer_only: bool = True,
    ) -> SentimentResult:
        """
        Run sentiment + emotion analysis on transcript text.

        Parameters
        ----------
        text         : Full transcript string
        segments     : Whisper segments with 'speaker' and 'text' fields
        customer_only: If True and segments available, analyse only customer turns
        """
        self._load()

        # Filter to customer text if segments provided
        analysis_text = text
        if segments and customer_only:
            customer_text = " ".join(
                s["text"] for s in segments if s.get("speaker") == "Customer"
            ).strip()
            if customer_text:
                analysis_text = customer_text

        # --- sentiment --------------------------------------------------
        sent_out = self._sentiment_pipe(analysis_text)
        sent_map = {item["label"].lower(): item["score"] for item in sent_out[0]}
        # Normalise possible label variations
        sent_map = _normalise_sentiment_labels(sent_map)

        top_sent  = max(sent_map, key=sent_map.get)
        sent_score = sent_map[top_sent]

        # Weighted polarity: –1 … +1
        polarity = sum(
            _SENTIMENT_POLARITY.get(lbl, 0) * score
            for lbl, score in sent_map.items()
        )

        # --- emotion ----------------------------------------------------
        emo_out  = self._emotion_pipe(analysis_text)
        emo_map  = {item["label"].lower(): item["score"] for item in emo_out[0]}
        top_emo  = max(emo_map, key=emo_map.get)

        # --- per-segment sentiment (lightweight) -------------------------
        seg_results = []
        if segments:
            for seg in segments[:20]:   # cap for performance
                chunk = seg.get("text", "").strip()
                if not chunk:
                    continue
                try:
                    s_out = self._sentiment_pipe(chunk)
                    s_map = _normalise_sentiment_labels(
                        {x["label"].lower(): x["score"] for x in s_out[0]}
                    )
                    seg_results.append({
                        "start":   seg.get("start", 0),
                        "end":     seg.get("end", 0),
                        "speaker": seg.get("speaker", "?"),
                        "text":    chunk[:120],
                        "sentiment": max(s_map, key=s_map.get),
                        "score":   round(max(s_map.values()), 3),
                    })
                except Exception:
                    pass

        # --- keyword extraction -----------------------------------------
        neg_found = _find_keywords(analysis_text, _NEGATIVE_KEYWORDS)
        pos_found = _find_keywords(analysis_text, _POSITIVE_KEYWORDS)
        key_phrases = neg_found[:5] + pos_found[:5]

        return SentimentResult(
            sentiment_label   = top_sent,
            sentiment_score   = sent_score,
            sentiment_polarity = polarity,
            dominant_emotion  = top_emo,
            emotion_scores    = emo_map,
            segment_sentiments = seg_results,
            negativity_indicators = neg_found,
            key_phrases       = key_phrases,
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalise_sentiment_labels(raw: Dict[str, float]) -> Dict[str, float]:
    """Map various model label styles to positive/neutral/negative."""
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
        "positive": "positive",
        "negative": "negative",
        "neutral":  "neutral",
    }
    out: Dict[str, float] = {}
    for k, v in raw.items():
        canonical = mapping.get(k, k)
        out[canonical] = out.get(canonical, 0.0) + v
    # Ensure all three keys exist
    for lbl in ("positive", "neutral", "negative"):
        out.setdefault(lbl, 0.0)
    return out


def _find_keywords(text: str, keywords: List[str]) -> List[str]:
    lower = text.lower()
    return [kw for kw in keywords if re.search(re.escape(kw), lower)]
