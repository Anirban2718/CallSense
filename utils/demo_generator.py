"""
utils/demo_generator.py
────────────────────────
Generates realistic synthetic pipeline results for dashboard demos.
No audio file or model download required.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict


def generate_demo_result(call_id: str, scenario: str = "random") -> Dict[str, Any]:
    """
    Return a fully populated result dict that mirrors the real pipeline output.

    Parameters
    ----------
    call_id  : str   Unique identifier for this call
    scenario : str   "satisfied" | "frustrated" | "angry" | "random"
    """
    if scenario == "random":
        scenario = random.choice(["satisfied", "satisfied", "frustrated", "angry"])

    cfg = _SCENARIOS[scenario]
    rng = random.Random(call_id)   # deterministic for same call_id

    duration = rng.uniform(90, 360)
    wc = int(duration / 60 * rng.uniform(120, 160))

    # ── Transcription ──────────────────────────────────────────────────────
    transcript_text = rng.choice(cfg["transcripts"])
    segments = _make_segments(transcript_text, duration, rng)

    transcription = {
        "full_text":        transcript_text,
        "language":         "en",
        "language_prob":    round(rng.uniform(0.95, 0.99), 3),
        "duration_seconds": round(duration, 2),
        "processing_time":  round(rng.uniform(3, 12), 2),
        "word_count":       wc,
        "segments":         segments,
    }

    # ── Sentiment ──────────────────────────────────────────────────────────
    pol = cfg["sentiment_polarity"] + rng.uniform(-0.1, 0.1)
    pol = max(-1.0, min(1.0, pol))

    sent_label = "positive" if pol > 0.2 else ("negative" if pol < -0.1 else "neutral")
    emotion_scores = _make_emotion_scores(cfg["dominant_text_emotion"], rng)

    sentiment = {
        "sentiment_label":    sent_label,
        "sentiment_score":    round(rng.uniform(0.6, 0.92), 3),
        "sentiment_polarity": round(pol, 3),
        "dominant_emotion":   cfg["dominant_text_emotion"],
        "emotion_scores":     emotion_scores,
        "segment_sentiments": _make_seg_sentiments(segments, cfg, rng),
        "negativity_indicators": cfg.get("neg_keywords", []),
        "key_phrases":        cfg.get("key_phrases", []),
    }

    # ── Audio features ─────────────────────────────────────────────────────
    audio_features = {
        "pitch": {
            "mean_hz":  round(rng.uniform(*cfg["pitch_range"]), 2),
            "std_hz":   round(rng.uniform(15, 60), 2),
            "range_hz": round(rng.uniform(80, 300), 2),
            "min_hz":   80.0,
            "max_hz":   380.0,
        },
        "energy": {
            "mean": round(rng.uniform(0.02, 0.12), 5),
            "std":  round(rng.uniform(0.005, 0.04), 5),
            "max":  round(rng.uniform(0.15, 0.40), 5),
        },
        "tempo_bpm":             round(rng.uniform(95, 140), 1),
        "spectral_centroid_hz":  round(rng.uniform(1200, 2800), 1),
        "spectral_rolloff_hz":   round(rng.uniform(3000, 6000), 1),
        "spectral_bandwidth_hz": round(rng.uniform(1500, 3500), 1),
        "zcr_mean":              round(rng.uniform(0.05, 0.20), 5),
        "mfcc_means":            [round(rng.uniform(-50, 50), 3) for _ in range(13)],
        "mfcc_stds":             [round(rng.uniform(5, 25), 3) for _ in range(13)],
        "prosody": {
            "speaking_rate_wpm":  round(rng.uniform(*cfg["speaking_rate"]), 1),
            "silence_ratio":      round(rng.uniform(*cfg["silence_ratio"]), 3),
            "pitch_variability":  round(rng.uniform(0.1, 0.5), 3),
            "energy_variability": round(rng.uniform(0.2, 0.8), 3),
            "prosody_score":      round(rng.uniform(*cfg["prosody_score"]), 3),
        },
        "duration_seconds": round(duration, 2),
        "sample_rate":      16000,
    }

    # ── Voice emotion ──────────────────────────────────────────────────────
    ve_scores = _make_emotion_scores_audio(cfg["dominant_voice_emotion"], rng)
    ve_pol = cfg["voice_polarity"] + rng.uniform(-0.05, 0.05)
    voice_emotion = {
        "dominant_emotion": cfg["dominant_voice_emotion"],
        "emotion_scores":   ve_scores,
        "polarity":         round(ve_pol, 3),
        "confidence":       round(rng.uniform(0.45, 0.85), 3),
    }

    # ── CSAT score ─────────────────────────────────────────────────────────
    base_csat = cfg["base_csat"] + rng.uniform(-8, 8)
    base_csat = max(0, min(100, base_csat))
    csat = _make_csat(base_csat, cfg, sentiment, voice_emotion)

    # ── Waveform ───────────────────────────────────────────────────────────
    waveform = _make_waveform(150, cfg["dominant_voice_emotion"], rng)

    return {
        "call_id":           call_id,
        "filename":          f"demo_{scenario}_call.wav",
        "analyzed_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_time_s": round(rng.uniform(8, 25), 2),
        "transcription":     transcription,
        "sentiment":         sentiment,
        "audio_features":    audio_features,
        "voice_emotion":     voice_emotion,
        "csat":              csat,
        "summary":           rng.choice(cfg["summaries"]),
        "waveform":          waveform,
        "_demo":             True,
        "_scenario":         scenario,
    }


# ── Scenario configs ───────────────────────────────────────────────────────

_SCENARIOS = {
    "satisfied": {
        "sentiment_polarity":  0.70,
        "dominant_text_emotion": "joy",
        "dominant_voice_emotion": "happy",
        "voice_polarity":     0.75,
        "base_csat":          83,
        "pitch_range":        (140, 200),
        "speaking_rate":      (130, 160),
        "silence_ratio":      (0.10, 0.25),
        "prosody_score":      (0.65, 0.85),
        "neg_keywords":       [],
        "key_phrases":        ["thank you so much", "very helpful", "appreciate", "resolved"],
        "transcripts": [
            "Agent: Thank you for calling. How can I help you today? "
            "Customer: Hi! I was hoping to get help with my billing issue. "
            "Agent: Of course, let me pull up your account right away. I can see the charge you mentioned. "
            "Customer: Yes that one. It looks wrong. "
            "Agent: You're absolutely right. I'll issue a refund immediately. "
            "Customer: Oh wow, thank you so much! That was so quick. "
            "Agent: Happy to help. Is there anything else I can assist with? "
            "Customer: No, that's everything. You were very helpful! "
            "Agent: Great, have a wonderful day! "
            "Customer: You too, thank you!",
        ],
        "summaries": [
            "Call duration: 3m 12s. Customer contacted support about a billing discrepancy. "
            "Agent resolved the issue promptly with a refund. Customer expressed high satisfaction "
            "throughout the call. CSAT: 83/100 (Good). Voice analysis confirms a positive, happy tone. "
            "No negative indicators detected.",
        ],
    },
    "frustrated": {
        "sentiment_polarity": -0.30,
        "dominant_text_emotion": "sadness",
        "dominant_voice_emotion": "frustrated",
        "voice_polarity":    -0.55,
        "base_csat":          42,
        "pitch_range":        (170, 240),
        "speaking_rate":      (155, 200),
        "silence_ratio":      (0.30, 0.50),
        "prosody_score":      (0.30, 0.55),
        "neg_keywords":       ["been waiting", "still broken", "no solution", "transfer me again"],
        "key_phrases":        ["been waiting", "still broken", "transfer me again"],
        "transcripts": [
            "Agent: Thank you for calling support, how can I help? "
            "Customer: I've been waiting for 45 minutes. This is the third time I've called about the same issue. "
            "Agent: I apologize for the wait. Can you describe the problem? "
            "Customer: My internet is still broken. I called last week, and the week before. Nothing has been fixed. "
            "Agent: Let me check the notes on your account. "
            "Customer: Every time I call, you transfer me again and nothing gets resolved. "
            "Agent: I understand your frustration. I'll escalate this to tier 2 support. "
            "Customer: I've heard that before. This is completely unacceptable. "
            "Agent: I'm sorry, I'm doing the best I can. "
            "Customer: Fine. Just fix it this time please.",
        ],
        "summaries": [
            "Call duration: 4m 47s. Customer reported a recurring internet outage spanning multiple weeks. "
            "Repeated transfer incidents noted. Agent escalated to tier 2. Customer showed clear frustration "
            "throughout. CSAT: 42/100 (Poor). Negative keywords detected. Recommend follow-up call within 24 hours.",
        ],
    },
    "angry": {
        "sentiment_polarity": -0.85,
        "dominant_text_emotion": "anger",
        "dominant_voice_emotion": "angry",
        "voice_polarity":    -0.90,
        "base_csat":          18,
        "pitch_range":        (200, 290),
        "speaking_rate":      (170, 220),
        "silence_ratio":      (0.05, 0.20),
        "prosody_score":      (0.20, 0.45),
        "neg_keywords":       ["unacceptable", "terrible", "cancel", "refund", "disgusted", "worst", "never again"],
        "key_phrases":        ["cancel", "refund", "unacceptable", "never again"],
        "transcripts": [
            "Agent: Thanks for calling, how can I assist? "
            "Customer: This is absolutely ridiculous. I want to cancel immediately. "
            "Agent: I'm sorry to hear that. Can I ask what happened? "
            "Customer: Your service is the worst I've ever used. Three outages this month and nobody cares. "
            "Agent: I sincerely apologize for the inconvenience. "
            "Customer: Don't apologize, just give me a refund. I've been a customer for 5 years and this is how you treat me? "
            "Agent: I will process a partial refund for this month. "
            "Customer: Partial? No. Full refund. I am disgusted with this company. "
            "Agent: I understand. Let me get a supervisor for you. "
            "Customer: Finally. This is outrageous.",
        ],
        "summaries": [
            "Call duration: 2m 58s. Highly negative interaction. Customer demanded cancellation and full refund "
            "following repeated service outages. Multiple high-risk keywords detected (cancel, refund, disgusted, worst). "
            "CSAT: 18/100 (Critical). Vocal anger confirmed by audio analysis. Immediate escalation and retention "
            "outreach strongly recommended.",
        ],
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

_TEXT_EMOTIONS = ["joy", "surprise", "neutral", "sadness", "fear", "disgust", "anger"]
_AUDIO_EMOTIONS = ["angry", "frustrated", "sad", "neutral", "calm", "happy", "excited"]


def _make_emotion_scores(dominant: str, rng: random.Random) -> dict:
    dom_score = rng.uniform(0.45, 0.75)
    rest = 1 - dom_score
    others = {e: rng.random() for e in _TEXT_EMOTIONS if e != dominant}
    total_other = sum(others.values())
    others = {k: v / total_other * rest for k, v in others.items()}
    scores = {dominant: dom_score, **others}
    return {k: round(v, 3) for k, v in scores.items()}


def _make_emotion_scores_audio(dominant: str, rng: random.Random) -> dict:
    dom_score = rng.uniform(0.40, 0.72)
    rest = 1 - dom_score
    others = {e: rng.random() for e in _AUDIO_EMOTIONS if e != dominant}
    total_other = sum(others.values())
    others = {k: v / total_other * rest for k, v in others.items()}
    scores = {dominant: dom_score, **others}
    return {k: round(v, 3) for k, v in scores.items()}


def _make_segments(text: str, duration: float, rng: random.Random) -> list:
    parts = [p.strip() for p in text.split("  ") if p.strip()]
    segments = []
    t = 0.0
    for i, part in enumerate(parts):
        seg_dur = duration / len(parts) * rng.uniform(0.7, 1.3)
        speaker = "Agent" if part.startswith("Agent:") else "Customer"
        clean = part.split(":", 1)[-1].strip() if ":" in part else part
        segments.append({
            "start":      round(t, 2),
            "end":        round(t + seg_dur, 2),
            "text":       clean,
            "speaker":    speaker,
            "confidence": round(rng.uniform(0.80, 0.97), 3),
        })
        t += seg_dur
    return segments


def _make_seg_sentiments(segments: list, cfg: dict, rng: random.Random) -> list:
    out = []
    for seg in segments[:10]:
        base_pol = cfg["sentiment_polarity"]
        jitter   = rng.uniform(-0.3, 0.3)
        pol      = max(-1.0, min(1.0, base_pol + jitter))
        label    = "positive" if pol > 0.2 else ("negative" if pol < -0.1 else "neutral")
        out.append({
            "start":     seg["start"],
            "end":       seg["end"],
            "speaker":   seg["speaker"],
            "text":      seg["text"][:100],
            "sentiment": label,
            "score":     round(abs(pol) * 0.4 + 0.55, 3),
        })
    return out


def _make_csat(score: float, cfg: dict, sentiment: dict, voice_emotion: dict) -> dict:
    tier_map = {
        (90, 101): ("excellent", "A", "none"),
        (75, 90):  ("good",      "B", "none"),
        (55, 75):  ("acceptable","C", "none"),
        (35, 55):  ("poor",      "D", "warning"),
        (0,  35):  ("critical",  "F", "critical"),
    }
    tier = grade = alert = "unknown"
    for (lo, hi), (t, g, a) in tier_map.items():
        if lo <= score < hi:
            tier, grade, alert = t, g, a
            break

    dom_ve = voice_emotion["dominant_emotion"]
    if dom_ve == "angry" and alert != "critical":
        alert = "critical"
    elif dom_ve == "frustrated" and alert == "none":
        alert = "warning"

    alert_reasons = []
    if score < 35:
        alert_reasons.append(f"CSAT score critically low: {score:.0f}/100")
    if dom_ve == "angry":
        alert_reasons.append("Vocal anger detected in audio analysis")
    if dom_ve == "frustrated":
        alert_reasons.append("Frustration detected in voice analysis")
    if sentiment.get("negativity_indicators"):
        alert_reasons.append(f"{len(sentiment['negativity_indicators'])} negative keywords detected")

    ts = (score / 100) * 2 - 1
    te = sentiment.get("sentiment_polarity", 0)
    ae = voice_emotion.get("polarity", 0)

    sub_scores = {
        "text_sentiment": round(max(0, min(100, (ts + 1) / 2 * 100)), 1),
        "text_emotion":   round(max(0, min(100, (te + 1) / 2 * 100)), 1),
        "audio_emotion":  round(max(0, min(100, (ae + 1) / 2 * 100)), 1),
        "audio_prosody":  round(max(0, min(100, score * 0.85 + random.uniform(-5, 5))), 1),
    }

    recs_map = {
        "critical": [
            "Immediate escalation required — review this call urgently.",
            "Customer at high churn risk. Initiate retention protocol.",
            "Agent debrief recommended for de-escalation coaching.",
        ],
        "poor": [
            "Schedule follow-up with customer within 24 hours.",
            "Review agent handling for coaching opportunities.",
            "Check ticket resolution status.",
        ],
        "acceptable": [
            "Monitor for repeat contact on same issue.",
            "Consider proactive satisfaction check-in.",
        ],
        "good": ["Interaction met quality standards. No action required."],
        "excellent": ["Outstanding interaction — use as training example."],
    }

    return {
        "csat_score":        round(score, 1),
        "grade":             grade,
        "tier":              tier,
        "alert_level":       alert,
        "alert_reasons":     alert_reasons,
        "sub_scores":        sub_scores,
        "sub_weights":       {"text_sentiment": 0.35, "text_emotion": 0.25,
                              "audio_emotion": 0.20, "audio_prosody": 0.20},
        "summary":           "",
        "recommendations":   recs_map.get(tier, []),
        "agent_score":       round(score * 0.9 + random.uniform(-5, 10), 1),
        "agent_flags":       [],
    }


def _make_waveform(n: int, emotion: str, rng: random.Random) -> list:
    """Generate a fake waveform envelope matching the emotion's energy profile."""
    base_energy = {"angry": 0.8, "frustrated": 0.65, "sad": 0.3,
                   "neutral": 0.45, "calm": 0.35, "happy": 0.55, "excited": 0.7}
    base = base_energy.get(emotion, 0.5)
    wave = []
    for i in range(n):
        # Simulate conversation pauses and speech bursts
        phase = (i / n) * 10
        sine  = 0.5 + 0.4 * abs(__import__("math").sin(phase * 3.14))
        noise = rng.uniform(-0.15, 0.15)
        val   = base * sine + noise
        wave.append(round(max(0.0, min(1.0, val)), 4))
    return wave
