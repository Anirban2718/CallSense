# CallSense AI

Customer satisfaction detection from call audio, with a Flask dashboard and REST API. The system analyzes uploaded recordings and produces a **Customer Satisfaction Score (CSAT)** with text sentiment, voice emotion, and acoustic prosody breakdowns.

## Status

This repository is intended for local development and demo use.

- First run downloads Whisper and Hugging Face model weights.
- `ffmpeg` is required to decode MP3, M4A, and other compressed audio formats.
- Runtime data is stored locally in `uploads/`, `results/`, and `models/` and is ignored by Git.

---

## Architecture

```
Audio File (.wav / .mp3 / .m4a)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline                                 │
│                                                                 │
│  1. Transcriber        (OpenAI Whisper)                         │
│     └─ Speech-to-text, segment timestamps, speaker diarisation  │
│                                                                 │
│  2. SentimentAnalyzer  (HuggingFace Transformers)               │
│     ├─ Sentiment: cardiffnlp/twitter-roberta-base-sentiment      │
│     └─ Emotion:   j-hartmann/emotion-english-distilroberta-base │
│                                                                 │
│  3. AudioProcessor     (librosa)                                │
│     └─ Pitch, energy, MFCC, tempo, spectral features, prosody  │
│                                                                 │
│  4. EmotionDetector    (Rule-based / wav2vec2 SER)              │
│     └─ Vocal emotion from acoustic features                     │
│                                                                 │
│  5. ScoreCalculator    (Custom fusion logic)                    │
│     └─ Weighted CSAT 0–100 + alerts + recommendations           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
 Flask REST API  →  Interactive Dashboard (Chart.js)
```

---

## CSAT Score Breakdown

| Component         | Weight | Source                        |
|-------------------|--------|-------------------------------|
| Text Sentiment    | 35%    | RoBERTa sentiment model       |
| Text Emotion      | 25%    | DistilRoBERTa emotion model   |
| Voice Emotion     | 20%    | Acoustic feature heuristics   |
| Audio Prosody     | 20%    | librosa feature extraction    |

**Score tiers:**
- `90–100` → Excellent (Grade A)
- `75–89`  → Good      (Grade B)
- `55–74`  → Acceptable (Grade C)
- `35–54`  → Poor       (Grade D) — Warning alert
- `0–34`   → Critical   (Grade F) — Critical alert + auto-escalation

---

## Quick Start

### 1. Clone and set up a virtual environment

```bash
git clone <repo>
cd customer_satisfaction_ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

FFmpeg is required for audio decoding:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add it to PATH
```

### 4. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 5. Try the demo (no audio needed)

Click **⚡ Run Demo** in the top bar — generates a synthetic call result instantly.

### 6. Run the smoke test

```bash
python -m unittest discover -s tests
```

---

## Configuration

Edit `config.py` to customize:

```python
WHISPER_MODEL_SIZE = "base"    # tiny | base | small | medium | large
WHISPER_DEVICE     = "cpu"     # "cuda" for GPU

WEIGHTS = {
    "text_sentiment": 0.35,
    "text_emotion":   0.25,
    "audio_emotion":  0.20,
    "audio_prosody":  0.20,
}

ALERT_SCORE_THRESHOLD   = 35   # CSAT below → red alert
WARNING_SCORE_THRESHOLD = 55   # CSAT below → warning
```

The following directories are created automatically on startup:

- `uploads/` for temporary uploaded audio files
- `results/` for generated JSON analysis output
- `models/` for downloaded model assets

---

## API Endpoints

| Method | Endpoint           | Description                         |
|--------|--------------------|-------------------------------------|
| GET    | `/`                | Dashboard UI                        |
| POST   | `/analyze`         | Upload audio file for analysis      |
| GET    | `/results/<id>`    | Retrieve a stored analysis result   |
| GET    | `/calls`           | List all analyzed calls             |
| GET    | `/demo`            | Run demo with synthetic data        |
| GET    | `/health`          | Health check                        |

### POST `/analyze` — Request

```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@customer_call.wav"
```

### Response structure

```json
{
  "call_id": "a3f7c1b2",
  "filename": "customer_call.wav",
  "analyzed_at": "2025-03-15T14:22:00Z",
  "processing_time_s": 18.4,
  "transcription": {
    "full_text": "Agent: Thank you for calling...",
    "language": "en",
    "word_count": 312,
    "duration_seconds": 187.3,
    "segments": [ ... ]
  },
  "sentiment": {
    "sentiment_label": "negative",
    "sentiment_polarity": -0.62,
    "dominant_emotion": "anger",
    "emotion_scores": { "anger": 0.71, "joy": 0.04, ... },
    "negativity_indicators": ["unacceptable", "cancel", "refund"]
  },
  "audio_features": {
    "pitch": { "mean_hz": 218.4, "std_hz": 42.1 },
    "tempo_bpm": 128.5,
    "prosody": { "speaking_rate_wpm": 162.3, "prosody_score": 0.44 }
  },
  "voice_emotion": {
    "dominant_emotion": "angry",
    "emotion_scores": { "angry": 0.58, "frustrated": 0.21, ... },
    "polarity": -0.83,
    "confidence": 0.58
  },
  "csat": {
    "csat_score": 22.4,
    "grade": "F",
    "tier": "critical",
    "alert_level": "critical",
    "alert_reasons": ["CSAT score critically low: 22/100", "Vocal anger detected"],
    "sub_scores": {
      "text_sentiment": 19.0,
      "text_emotion": 14.5,
      "audio_emotion": 8.5,
      "audio_prosody": 44.0
    },
    "recommendations": [
      "Immediate escalation required — review this call urgently.",
      "Customer at high churn risk. Initiate retention protocol."
    ],
    "agent_score": 71.2
  },
  "summary": "3m 7s call. Customer demanded cancellation after repeated outages..."
}
```

---

## Optional: GPU acceleration

For large call volumes, enable GPU inference:

```python
# config.py
WHISPER_DEVICE = "cuda"     # Whisper on GPU

# pipeline/sentiment_analyzer.py — SentimentAnalyzer(device=0)
# pipeline/emotion_detector.py   — EmotionDetector(use_ml=True)
```

## Optional: ML-based voice emotion (wav2vec2)

The default uses rule-based audio emotion detection. To use the neural SER model:

```python
# In app.py, change:
_pipeline["emotion"] = EmotionDetector(use_ml=True, device="cpu")
```

This downloads `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (~1.2 GB) on first run.

---

## Project Structure

```
customer_satisfaction_ai/
├── app.py                        # Flask app + API routes
├── config.py                     # All configuration constants
├── requirements.txt
├── README.md
├── pipeline/
│   ├── __init__.py
│   ├── transcriber.py            # Whisper speech-to-text
│   ├── sentiment_analyzer.py     # HuggingFace NLP models
│   ├── audio_processor.py        # librosa acoustic features
│   ├── emotion_detector.py       # Voice emotion detection
│   └── score_calculator.py       # CSAT fusion & alerts
├── utils/
│   └── demo_generator.py         # Synthetic demo data
├── templates/
│   └── dashboard.html            # Full dashboard UI
├── uploads/                      # Temp audio uploads (gitignored)
├── results/                      # JSON result storage (gitignored)
├── models/                       # Downloaded model assets (gitignored)
└── tests/
    └── test_smoke.py             # Basic Flask/runtime smoke checks
```

---

## Extending the System

### Add real speaker diarisation
Replace the heuristic in `transcriber.py` with [pyannote.audio](https://github.com/pyannote/pyannote-audio):
```python
from pyannote.audio import Pipeline
diarize = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
```

### Add call summarisation with LLM
In `app.py`, post-process the transcript with any OpenAI-compatible API:
```python
import openai
summary = openai.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role":"user","content": f"Summarize this call:\n{transcript}"}])
```

### Add batch processing
```python
# batch_analyze.py
from pathlib import Path
for audio in Path("calls/").glob("*.wav"):
    result = analyze_audio(str(audio), audio.name, str(uuid.uuid4())[:8])
    save_result(result["call_id"], result)
```

---

## Tech Stack

| Layer           | Technology                              |
|-----------------|----------------------------------------|
| Speech-to-text  | OpenAI Whisper (base)                  |
| Sentiment NLP   | cardiffnlp/twitter-roberta-base        |
| Emotion NLP     | j-hartmann/emotion-english-distilroberta |
| Audio features  | librosa 0.10+                          |
| Voice emotion   | Rule-based + optional wav2vec2 SER     |
| Backend API     | Flask 3.0                              |
| Dashboard       | Chart.js 4.4 + vanilla JS              |
| Storage         | File-based JSON (swap for PostgreSQL)  |
