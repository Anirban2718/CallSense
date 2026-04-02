"""
pipeline/audio_processor.py
────────────────────────────
Low-level acoustic feature extraction with librosa.

Extracts:
  • Pitch (F0)         — fundamental frequency via YIN algorithm
  • Energy / Loudness  — RMS energy per frame
  • Tempo              — beats-per-minute via beat tracking
  • MFCCs              — 13 Mel-Frequency Cepstral Coefficients
  • Spectral features  — centroid, rolloff, bandwidth, ZCR
  • Speaking rate      — estimated from energy envelope
  • Prosody profile    — pitch variability, energy variability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    # Pitch
    pitch_mean:   float   # Hz
    pitch_std:    float   # Hz — higher = more expressive
    pitch_min:    float
    pitch_max:    float
    pitch_range:  float   # max - min

    # Energy
    energy_mean:  float   # RMS
    energy_std:   float
    energy_max:   float

    # Tempo
    tempo_bpm:    float

    # Spectral
    spectral_centroid_mean:  float   # Hz — brightness indicator
    spectral_rolloff_mean:   float
    spectral_bandwidth_mean: float
    zcr_mean:                float   # zero-crossing rate

    # MFCCs (13 coefficients — means & stds)
    mfcc_means: List[float] = field(default_factory=list)
    mfcc_stds:  List[float] = field(default_factory=list)

    # Derived prosody
    speaking_rate_wpm:   float = 0.0  # words per minute (rough)
    silence_ratio:       float = 0.0  # fraction of silent frames
    pitch_variability:   float = 0.0  # normalised pitch std
    energy_variability:  float = 0.0  # normalised energy std

    duration_seconds: float = 0.0
    sample_rate:      int   = 16_000

    def prosody_score(self) -> float:
        """
        Composite prosody health score 0–1.
        High pitch variability + moderate energy + low silence → healthy conversation.
        """
        # Normalise components
        pv  = min(self.pitch_variability,  1.0)          # 0=monotone, 1=very expressive
        ev  = min(self.energy_variability, 1.0)
        sr  = min(self.speaking_rate_wpm / 160, 1.0)     # ~150 wpm is normal
        sil = 1.0 - min(self.silence_ratio, 1.0)         # less silence is better

        return 0.30 * pv + 0.25 * ev + 0.25 * sr + 0.20 * sil

    def to_dict(self) -> dict:
        return {
            "pitch": {
                "mean_hz":   round(self.pitch_mean, 2),
                "std_hz":    round(self.pitch_std, 2),
                "range_hz":  round(self.pitch_range, 2),
                "min_hz":    round(self.pitch_min, 2),
                "max_hz":    round(self.pitch_max, 2),
            },
            "energy": {
                "mean":      round(float(self.energy_mean), 5),
                "std":       round(float(self.energy_std), 5),
                "max":       round(float(self.energy_max), 5),
            },
            "tempo_bpm":              round(self.tempo_bpm, 1),
            "spectral_centroid_hz":   round(self.spectral_centroid_mean, 1),
            "spectral_rolloff_hz":    round(self.spectral_rolloff_mean, 1),
            "spectral_bandwidth_hz":  round(self.spectral_bandwidth_mean, 1),
            "zcr_mean":               round(float(self.zcr_mean), 5),
            "mfcc_means":             [round(v, 3) for v in self.mfcc_means],
            "mfcc_stds":              [round(v, 3) for v in self.mfcc_stds],
            "prosody": {
                "speaking_rate_wpm":  round(self.speaking_rate_wpm, 1),
                "silence_ratio":      round(self.silence_ratio, 3),
                "pitch_variability":  round(self.pitch_variability, 3),
                "energy_variability": round(self.energy_variability, 3),
                "prosody_score":      round(self.prosody_score(), 3),
            },
            "duration_seconds": round(self.duration_seconds, 2),
            "sample_rate":      self.sample_rate,
        }


class AudioProcessor:
    """
    Extracts acoustic features from a WAV/MP3/etc. file using librosa.

    Parameters
    ----------
    sample_rate  : int   Target sample rate (Whisper uses 16 kHz)
    frame_length : int   STFT frame length in samples
    hop_length   : int   STFT hop length in samples
    n_mfcc       : int   Number of MFCC coefficients
    """

    def __init__(
        self,
        sample_rate:  int = 16_000,
        frame_length: int = 2048,
        hop_length:   int = 512,
        n_mfcc:       int = 13,
    ):
        self.sr           = sample_rate
        self.frame_length = frame_length
        self.hop_length   = hop_length
        self.n_mfcc       = n_mfcc

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, audio_path: str | Path) -> AudioFeatures:
        """
        Load audio and compute all acoustic features.

        Parameters
        ----------
        audio_path : path to audio file (WAV / MP3 / FLAC / M4A / etc.)

        Returns
        -------
        AudioFeatures dataclass
        """
        import librosa

        audio_path = Path(audio_path)
        logger.info("Extracting audio features from %s", audio_path.name)

        # Load & resample
        y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # ── Pitch (YIN / pyin) ───────────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            frame_length=self.frame_length,
        )
        f0_voiced = f0[voiced_flag & np.isfinite(f0) & (f0 > 0)]
        if len(f0_voiced) == 0:
            f0_voiced = np.array([150.0])   # fallback

        pitch_mean  = float(np.mean(f0_voiced))
        pitch_std   = float(np.std(f0_voiced))
        pitch_min   = float(np.min(f0_voiced))
        pitch_max   = float(np.max(f0_voiced))

        # ── Energy (RMS) ─────────────────────────────────────────────────
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length,
                                  hop_length=self.hop_length)[0]
        energy_mean = float(np.mean(rms))
        energy_std  = float(np.std(rms))
        energy_max  = float(np.max(rms))

        # Silence ratio — frames below 5 % of max RMS
        silence_mask  = rms < (0.05 * energy_max + 1e-9)
        silence_ratio = float(np.mean(silence_mask))

        # ── Tempo ────────────────────────────────────────────────────────
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # ── Spectral features ────────────────────────────────────────────
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
        spec_rolloff  = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
        spec_bw       = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
        zcr           = librosa.feature.zero_crossing_rate(
            y, frame_length=self.frame_length, hop_length=self.hop_length)[0]

        # ── MFCCs ────────────────────────────────────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                                     n_fft=self.frame_length,
                                     hop_length=self.hop_length)
        mfcc_means = mfccs.mean(axis=1).tolist()
        mfcc_stds  = mfccs.std(axis=1).tolist()

        # ── Speaking rate estimate ───────────────────────────────────────
        # Count energy-envelope peaks as a syllable proxy
        from scipy.signal import find_peaks
        env = np.convolve(rms, np.ones(5) / 5, mode="same")  # smooth
        peaks, _ = find_peaks(env, height=energy_mean * 0.5, distance=int(sr / self.hop_length * 0.15))
        syllables_per_sec = len(peaks) / max(duration, 1)
        speaking_rate_wpm = syllables_per_sec * 60 / 1.5   # ~1.5 syllables/word

        # ── Prosody variability ──────────────────────────────────────────
        pitch_variability  = pitch_std  / max(pitch_mean, 1)
        energy_variability = energy_std / max(energy_mean, 1e-9)

        return AudioFeatures(
            pitch_mean   = pitch_mean,
            pitch_std    = pitch_std,
            pitch_min    = pitch_min,
            pitch_max    = pitch_max,
            pitch_range  = pitch_max - pitch_min,
            energy_mean  = energy_mean,
            energy_std   = energy_std,
            energy_max   = energy_max,
            tempo_bpm    = tempo_bpm,
            spectral_centroid_mean  = float(np.mean(spec_centroid)),
            spectral_rolloff_mean   = float(np.mean(spec_rolloff)),
            spectral_bandwidth_mean = float(np.mean(spec_bw)),
            zcr_mean    = float(np.mean(zcr)),
            mfcc_means  = mfcc_means,
            mfcc_stds   = mfcc_stds,
            speaking_rate_wpm  = speaking_rate_wpm,
            silence_ratio      = silence_ratio,
            pitch_variability  = pitch_variability,
            energy_variability = energy_variability,
            duration_seconds   = duration,
            sample_rate        = sr,
        )

    # ── Timeline (for dashboard waveform) ─────────────────────────────────

    def waveform_envelope(
        self,
        audio_path: str | Path,
        n_points: int = 200,
    ) -> List[float]:
        """Return a downsampled RMS envelope for waveform visualisation."""
        import librosa

        y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length,
                                  hop_length=self.hop_length)[0]
        # Downsample to n_points
        indices = np.linspace(0, len(rms) - 1, n_points).astype(int)
        envelope = rms[indices]
        # Normalise 0-1
        mx = envelope.max() + 1e-9
        return [round(float(v / mx), 4) for v in envelope]
