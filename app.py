"""
app.py — Flask backend for the Customer Satisfaction AI dashboard.

Endpoints
---------
GET  /                   → Dashboard HTML
POST /analyze            → Upload & analyze audio file
GET  /results/<call_id>  → Retrieve stored result
GET  /calls              → List all analyzed calls
GET  /demo               → Run on a synthetic demo call
GET  /health             → Health check
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_MB * 1024 * 1024

# ── Lazy pipeline (loaded once on first use) ───────────────────────────────
_pipeline: dict = {}


def get_pipeline():
    if not _pipeline:
        logger.info("Initialising AI pipeline …")
        from pipeline import (
            AudioProcessor, EmotionDetector,
            ScoreCalculator, SentimentAnalyzer, Transcriber,
        )
        _pipeline["transcriber"]  = Transcriber(
            model_size=config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            language=config.WHISPER_LANGUAGE,
        )
        _pipeline["sentiment"]    = SentimentAnalyzer(device=-1)
        _pipeline["audio"]        = AudioProcessor(
            sample_rate=config.SAMPLE_RATE,
            frame_length=config.FRAME_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mfcc=config.N_MFCC,
        )
        _pipeline["emotion"]      = EmotionDetector(use_ml=False)
        _pipeline["scorer"]       = ScoreCalculator(weights=config.WEIGHTS)
        logger.info("Pipeline ready.")
    return _pipeline


# ── Result storage (file-based for simplicity) ────────────────────────────

def save_result(call_id: str, result: dict):
    path = config.RESULTS_DIR / f"{call_id}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(call_id: str) -> dict | None:
    path = config.RESULTS_DIR / f"{call_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_results() -> list[dict]:
    results = []
    for p in sorted(config.RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                r = json.load(f)
            results.append({
                "call_id":      r.get("call_id"),
                "filename":     r.get("filename", "unknown"),
                "csat_score":   r.get("csat", {}).get("csat_score"),
                "tier":         r.get("csat", {}).get("tier"),
                "alert_level":  r.get("csat", {}).get("alert_level"),
                "duration":     r.get("transcription", {}).get("duration_seconds"),
                "analyzed_at":  r.get("analyzed_at"),
            })
        except Exception:
            pass
    return results


# ── Core analysis function ─────────────────────────────────────────────────

def analyze_audio(audio_path: str, filename: str, call_id: str) -> dict:
    """Run the full pipeline on an audio file and return structured results."""
    pipe = get_pipeline()
    t_start = time.time()

    logger.info("[%s] Starting analysis of '%s'", call_id, filename)

    # 1. Transcription
    logger.info("[%s] Step 1/4: Transcribing …", call_id)
    transcript = pipe["transcriber"].transcribe(audio_path)
    segments_dict = transcript.to_dict()["segments"]

    # 2. Sentiment + Emotion (text)
    logger.info("[%s] Step 2/4: Analysing sentiment …", call_id)
    sentiment = pipe["sentiment"].analyze(
        text=transcript.full_text,
        segments=segments_dict,
        customer_only=True,
    )

    # 3. Audio features
    logger.info("[%s] Step 3/4: Extracting audio features …", call_id)
    audio_feats = pipe["audio"].extract(audio_path)
    waveform    = pipe["audio"].waveform_envelope(audio_path, n_points=150)

    # 4. Voice emotion
    logger.info("[%s] Step 4/4: Detecting voice emotion …", call_id)
    voice_emotion = pipe["emotion"].detect(audio_feats, audio_path)

    # 5. Score
    csat = pipe["scorer"].compute(sentiment, audio_feats, voice_emotion, transcript)

    # 6. Summarise
    summary_text = pipe["transcriber"].summarize(transcript.full_text, max_sentences=4)

    total_time = time.time() - t_start
    logger.info("[%s] Analysis complete in %.1fs  CSAT=%.1f", call_id, total_time, csat.csat_score)

    return {
        "call_id":      call_id,
        "filename":     filename,
        "analyzed_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_time_s": round(total_time, 2),
        "transcription":   transcript.to_dict(),
        "sentiment":       sentiment.to_dict(),
        "audio_features":  audio_feats.to_dict(),
        "voice_emotion":   voice_emotion.to_dict(),
        "csat":            csat.to_dict(),
        "summary":         summary_text,
        "waveform":        waveform,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Upload an audio file and run the full pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in config.ALLOWED_EXTS:
        return jsonify({"error": f"File type .{ext} not supported"}), 415

    call_id  = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    save_path = config.UPLOAD_DIR / f"{call_id}_{filename}"
    file.save(str(save_path))

    try:
        result = analyze_audio(str(save_path), filename, call_id)
        save_result(call_id, result)
        return jsonify(result)
    except Exception as e:
        logger.exception("Analysis failed for %s", call_id)
        return jsonify({"error": str(e)}), 500
    finally:
        # Optionally remove upload after processing
        if save_path.exists():
            os.remove(save_path)


@app.route("/results/<call_id>")
def get_result(call_id: str):
    result = load_result(call_id)
    if result is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(result)


@app.route("/results/<call_id>", methods=["DELETE"])
def delete_result(call_id: str):
    path = config.RESULTS_DIR / f"{call_id}.json"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    path.unlink()
    logger.info("Deleted result %s", call_id)
    return jsonify({"deleted": call_id})


@app.route("/calls")
def list_calls():
    return jsonify(list_results())


@app.route("/demo")
def demo():
    """Generate and analyze a synthetic demo call (no audio file needed)."""
    from utils.demo_generator import generate_demo_result
    call_id = "demo-" + str(uuid.uuid4())[:6]
    result = generate_demo_result(call_id)
    save_result(call_id, result)
    return jsonify(result)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Customer Satisfaction AI on http://%s:%d", config.FLASK_HOST, config.FLASK_PORT)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
