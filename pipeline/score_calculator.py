"""
pipeline/score_calculator.py
──────────────────────────────
Fuses all pipeline signals into a final Customer Satisfaction Score (CSAT).

Score range: 0 – 100
  0–34  : Critical   (red alert)
  35–54 : Poor       (warning)
  55–74 : Acceptable (yellow)
  75–89 : Good       (green)
  90–100: Excellent  (star)

Architecture
------------
Each sub-score is independently normalised to [0, 1] then combined with
configurable weights. Confidence modulation reduces the influence of
uncertain signals.

Sub-scores
----------
1. text_sentiment  — from SentimentAnalyzer (polarity + keyword boost)
2. text_emotion    — from SentimentAnalyzer (emotion polarity)
3. audio_emotion   — from EmotionDetector   (voice emotion polarity)
4. audio_prosody   — from AudioProcessor    (prosody_score)

Alert logic
-----------
• CSAT < 35 → critical alert
• CSAT < 55 → warning alert
• Anger / frustration detected in audio → flag regardless of score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CSATResult:
    csat_score:      float   # 0 – 100
    grade:           str     # A / B / C / D / F
    tier:            str     # excellent / good / acceptable / poor / critical
    alert_level:     str     # "none" | "warning" | "critical"
    alert_reasons:   List[str]

    # Sub-scores (each 0–100)
    sub_scores: Dict[str, float]
    sub_weights: Dict[str, float]

    # Component summaries
    summary: str
    recommendations: List[str]

    # Agent performance indicators
    agent_score:     Optional[float] = None   # 0–100
    agent_flags:     List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "csat_score":    round(self.csat_score, 1),
            "grade":         self.grade,
            "tier":          self.tier,
            "alert_level":   self.alert_level,
            "alert_reasons": self.alert_reasons,
            "sub_scores":    {k: round(v, 1) for k, v in self.sub_scores.items()},
            "sub_weights":   self.sub_weights,
            "summary":       self.summary,
            "recommendations": self.recommendations,
            "agent_score":   round(self.agent_score, 1) if self.agent_score is not None else None,
            "agent_flags":   self.agent_flags,
        }


# ── Tier / grade helpers ───────────────────────────────────────────────────

def _tier(score: float) -> str:
    if score >= 90: return "excellent"
    if score >= 75: return "good"
    if score >= 55: return "acceptable"
    if score >= 35: return "poor"
    return "critical"

def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 65: return "C"
    if score >= 50: return "D"
    return "F"


# ── Recommendations database ───────────────────────────────────────────────

_RECOMMENDATIONS: Dict[str, List[str]] = {
    "negative_sentiment": [
        "Review this call for escalation opportunities.",
        "Customer expressed explicit dissatisfaction — consider a follow-up call.",
    ],
    "angry_audio": [
        "Customer showed vocal anger — immediate escalation may be required.",
        "Agent should be debriefed on de-escalation technique.",
    ],
    "frustrated_audio": [
        "Voice analysis detected frustration; review resolution time.",
    ],
    "low_prosody": [
        "Agent's speech energy was low — coaching on vocal engagement recommended.",
        "Low speaking energy may signal low confidence or disengagement.",
    ],
    "high_silence": [
        "Excessive silence detected; review for hold time or dead air issues.",
    ],
    "negative_keywords": [
        "Transcript contains high-risk keywords (cancel, refund, complaint).",
    ],
    "positive": [
        "Interaction shows strong satisfaction indicators — share as best practice.",
    ],
}


class ScoreCalculator:
    """
    Compute the final CSAT score from all pipeline outputs.

    Parameters
    ----------
    weights : dict   Custom weights. Keys: text_sentiment, text_emotion,
                     audio_emotion, audio_prosody. Must sum to ~1.0.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "text_sentiment": 0.35,
            "text_emotion":   0.25,
            "audio_emotion":  0.20,
            "audio_prosody":  0.20,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        sentiment_result,    # SentimentResult
        audio_features,      # AudioFeatures
        emotion_result,      # EmotionResult
        transcript_result,   # TranscriptionResult
    ) -> CSATResult:
        """
        Fuse all results into a CSAT score.
        """

        # ── Sub-score 1: Text Sentiment (0–100) ───────────────────────────
        # polarity: –1 … +1 → map to 0–100
        sent_polarity = sentiment_result.sentiment_polarity
        ts_score = _polarity_to_score(sent_polarity)

        # Keyword boost / penalty
        neg_count = len(sentiment_result.negativity_indicators)
        pos_count = sum(
            1 for kp in sentiment_result.key_phrases
            if kp in sentiment_result.key_phrases and kp not in sentiment_result.negativity_indicators
        )
        keyword_delta = (-5 * neg_count) + (3 * pos_count)
        ts_score = float(max(0, min(100, ts_score + keyword_delta)))

        # ── Sub-score 2: Text Emotion (0–100) ─────────────────────────────
        te_polarity = _EMOTION_POLARITY_MAP.get(
            sentiment_result.dominant_emotion.lower(), 0.0
        )
        # Weighted by emotion scores
        te_weighted = sum(
            _EMOTION_POLARITY_MAP.get(e, 0.0) * s
            for e, s in sentiment_result.emotion_scores.items()
        )
        te_score = _polarity_to_score(te_weighted)

        # ── Sub-score 3: Audio Emotion (0–100) ────────────────────────────
        ae_score = _polarity_to_score(emotion_result.polarity)

        # Confidence modulation: scale toward 50 if confidence is low
        ae_confidence = emotion_result.confidence
        ae_score = 50 + (ae_score - 50) * min(ae_confidence * 2, 1.0)

        # ── Sub-score 4: Audio Prosody (0–100) ────────────────────────────
        ap_score = audio_features.prosody_score() * 100   # already 0–1 → 0–100

        # ── Weighted CSAT ─────────────────────────────────────────────────
        sub_scores = {
            "text_sentiment": ts_score,
            "text_emotion":   te_score,
            "audio_emotion":  ae_score,
            "audio_prosody":  ap_score,
        }

        csat = sum(
            sub_scores[k] * self.weights[k]
            for k in self.weights
            if k in sub_scores
        )
        csat = float(max(0.0, min(100.0, csat)))

        # ── Alert logic ───────────────────────────────────────────────────
        alert_level   = "none"
        alert_reasons = []

        dom_emo = emotion_result.dominant_emotion.lower()

        if csat < 35:
            alert_level = "critical"
            alert_reasons.append(f"CSAT score critically low: {csat:.0f}/100")
        elif csat < 55:
            alert_level = "warning"
            alert_reasons.append(f"CSAT score below threshold: {csat:.0f}/100")

        if dom_emo == "angry":
            if alert_level != "critical":
                alert_level = "critical"
            alert_reasons.append("Vocal anger detected in audio")
        elif dom_emo == "frustrated":
            if alert_level == "none":
                alert_level = "warning"
            alert_reasons.append("Frustration detected in voice analysis")

        if sentiment_result.sentiment_label == "negative" and neg_count >= 2:
            alert_reasons.append(f"{neg_count} negative keywords detected in transcript")
            if alert_level == "none":
                alert_level = "warning"

        # ── Recommendations ───────────────────────────────────────────────
        recs: List[str] = []
        if sentiment_result.sentiment_label == "negative":
            recs += _RECOMMENDATIONS["negative_sentiment"]
        if dom_emo == "angry":
            recs += _RECOMMENDATIONS["angry_audio"]
        elif dom_emo == "frustrated":
            recs += _RECOMMENDATIONS["frustrated_audio"]
        if ap_score < 40:
            recs += _RECOMMENDATIONS["low_prosody"]
        if audio_features.silence_ratio > 0.4:
            recs += _RECOMMENDATIONS["high_silence"]
        if neg_count >= 2:
            recs += _RECOMMENDATIONS["negative_keywords"]
        if csat >= 75:
            recs += _RECOMMENDATIONS["positive"]

        # De-duplicate
        recs = list(dict.fromkeys(recs))[:6]

        # ── Summary ───────────────────────────────────────────────────────
        summary = _build_summary(
            csat, sentiment_result, emotion_result, audio_features,
            transcript_result
        )

        # ── Agent score (experimental) ────────────────────────────────────
        agent_score, agent_flags = _estimate_agent_score(
            audio_features, transcript_result, sentiment_result
        )

        return CSATResult(
            csat_score      = csat,
            grade           = _grade(csat),
            tier            = _tier(csat),
            alert_level     = alert_level,
            alert_reasons   = alert_reasons,
            sub_scores      = sub_scores,
            sub_weights     = self.weights,
            summary         = summary,
            recommendations = recs,
            agent_score     = agent_score,
            agent_flags     = agent_flags,
        )


# ── Internal helpers ───────────────────────────────────────────────────────

_EMOTION_POLARITY_MAP: Dict[str, float] = {
    "joy":      1.0, "happy":    0.9, "excited":  0.6,
    "surprise": 0.2, "calm":     0.4,
    "neutral":  0.0,
    "sadness": -0.5, "sad":     -0.5,
    "fear":    -0.6, "fearful": -0.6, "frustrated": -0.7,
    "disgust": -0.8, "anger":   -1.0, "angry":    -1.0,
}


def _polarity_to_score(polarity: float) -> float:
    """Map polarity –1 … +1 linearly to 0 … 100."""
    return (polarity + 1.0) / 2.0 * 100.0


def _build_summary(csat, sent, emo, audio, transcript) -> str:
    tier = _tier(csat)
    duration = transcript.duration_seconds
    mins = int(duration // 60)
    secs = int(duration % 60)
    dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

    lines = [
        f"Call duration: {dur_str} | Words: {transcript.word_count} | Language: {transcript.language.upper()}.",
        f"Overall CSAT: {csat:.0f}/100 ({tier.capitalize()}) — Grade {_grade(csat)}.",
        f"Text sentiment: {sent.sentiment_label.capitalize()} "
        f"(polarity {sent.sentiment_polarity:+.2f}). "
        f"Dominant text emotion: {sent.dominant_emotion.capitalize()}.",
        f"Voice emotion: {emo.dominant_emotion.capitalize()} "
        f"(confidence {emo.confidence:.0%}). "
        f"Prosody score: {audio.prosody_score():.0%}.",
    ]
    if sent.negativity_indicators:
        lines.append(
            "Negative indicators found: " + ", ".join(f'"{k}"' for k in sent.negativity_indicators[:4]) + "."
        )
    return " ".join(lines)


def _estimate_agent_score(audio, transcript, sentiment) -> tuple:
    """
    Rough agent quality score based on:
    • Agent speaking rate (not too fast/slow)
    • Agent silence ratio (dead air)
    • Segment count (engagement)
    """
    flags: List[str] = []

    rate = audio.speaking_rate_wpm
    if rate < 80:
        flags.append("Agent speaking rate very low")
    elif rate > 200:
        flags.append("Agent speaking rate very high (may be rushed)")

    if audio.silence_ratio > 0.5:
        flags.append("High silence ratio — review for hold time")

    agent_segs = [s for s in transcript.segments if s.speaker == "Agent"]
    agent_word_count = sum(len(s.text.split()) for s in agent_segs)

    # Simple heuristic: good agent talks roughly 40-60% of call
    total_words = max(transcript.word_count, 1)
    agent_talk_ratio = agent_word_count / total_words

    if agent_talk_ratio < 0.25:
        flags.append("Agent contributed fewer words than expected — may be passive")
    elif agent_talk_ratio > 0.75:
        flags.append("Agent dominated conversation — customer had limited voice time")

    # Base score
    score = 75.0
    score += (0.50 - abs(agent_talk_ratio - 0.50)) * 30   # max +15 when ~50%
    score -= len(flags) * 5
    score = max(0.0, min(100.0, score))

    return score, flags
