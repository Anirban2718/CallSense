"""
pipeline/transcriber.py
────────────────────────
Speech-to-text using OpenAI Whisper.

Produces:
  - Full transcript text
  - Per-segment timestamps (for timeline view)
  - Detected language + confidence
  - Speaker-level segments (via simple energy-based diarisation heuristic)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """One spoken segment with timing metadata."""
    start: float           # seconds
    end:   float           # seconds
    text:  str
    avg_logprob: float = 0.0   # Whisper confidence proxy
    speaker: str = "unknown"   # filled by diarisation


@dataclass
class TranscriptionResult:
    full_text:        str
    segments:         List[TranscriptSegment]
    language:         str
    language_prob:    float
    duration_seconds: float
    processing_time:  float
    word_count:       int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.full_text.split())

    def to_dict(self) -> dict:
        return {
            "full_text":        self.full_text,
            "language":         self.language,
            "language_prob":    round(self.language_prob, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "processing_time":  round(self.processing_time, 2),
            "word_count":       self.word_count,
            "segments": [
                {
                    "start":   round(s.start, 2),
                    "end":     round(s.end, 2),
                    "text":    s.text.strip(),
                    "speaker": s.speaker,
                    "confidence": round(np.exp(s.avg_logprob), 3)
                        if s.avg_logprob < 0 else 0.9,
                }
                for s in self.segments
            ],
        }


class Transcriber:
    """
    Wraps OpenAI Whisper for audio transcription.

    Parameters
    ----------
    model_size : str
        Whisper model variant: tiny | base | small | medium | large
    device : str
        "cpu" or "cuda"
    language : str | None
        Force a language (e.g. "en") or None for auto-detect.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: Optional[str] = "en",
    ):
        self.model_size = model_size
        self.device     = device
        self.language   = language
        self._model     = None   # lazy-loaded

    # ── Lazy model loading ─────────────────────────────────────────────────

    def _load(self):
        if self._model is None:
            logger.info("Loading Whisper model '%s' on %s …", self.model_size, self.device)
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper ready.")

    # ── Public API ─────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """
        Transcribe an audio file and return structured results.

        Returns
        -------
        TranscriptionResult
        """
        self._load()
        import whisper

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t0 = time.perf_counter()

        # --- load & pad/trim to fit Whisper ---
        audio = whisper.load_audio(str(audio_path))
        duration = len(audio) / 16_000   # Whisper always resamples to 16kHz

        options = dict(
            language=self.language,
            task="transcribe",
            fp16=False,         # fp16 often unsupported on CPU
            verbose=False,
            word_timestamps=False,
        )

        result = self._model.transcribe(str(audio_path), **options)

        t1 = time.perf_counter()

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                avg_logprob=seg.get("avg_logprob", -0.5),
            )
            for seg in result["segments"]
        ]

        # --- simple heuristic diarisation (alternating every ~30s) ---------
        segments = self._heuristic_diarize(segments)

        lang      = result.get("language", self.language or "en")
        lang_prob = result.get("language_probability", 1.0) or 1.0

        return TranscriptionResult(
            full_text=result["text"].strip(),
            segments=segments,
            language=lang,
            language_prob=lang_prob,
            duration_seconds=duration,
            processing_time=t1 - t0,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_diarize(
        segments: List[TranscriptSegment],
        switch_interval: float = 25.0,
    ) -> List[TranscriptSegment]:
        """
        Assign 'Agent' / 'Customer' labels based on time-window alternation.
        Real diarisation requires pyannote.audio or speaker embeddings.
        """
        speakers = ["Agent", "Customer"]
        current  = 0
        window_start = 0.0

        for seg in segments:
            if seg.start - window_start >= switch_interval:
                current = 1 - current
                window_start = seg.start
            seg.speaker = speakers[current]

        return segments

    # ── Convenience ────────────────────────────────────────────────────────

    def summarize(self, transcript: str, max_sentences: int = 5) -> str:
        """
        Lightweight extractive summary: pick sentences with highest
        term-frequency weight (no model needed).
        """
        import re
        sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
        if len(sentences) <= max_sentences:
            return transcript.strip()

        # TF-based scoring
        words = transcript.lower().split()
        freq: dict[str, int] = {}
        for w in words:
            w = re.sub(r"[^a-z]", "", w)
            if len(w) > 3:
                freq[w] = freq.get(w, 0) + 1

        def score(sent: str) -> float:
            ws = [re.sub(r"[^a-z]", "", w.lower()) for w in sent.split()]
            return sum(freq.get(w, 0) for w in ws if len(w) > 3) / max(len(ws), 1)

        ranked = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
        top    = sorted(ranked[:max_sentences], key=lambda x: x[0])
        return " ".join(s for _, s in top)
