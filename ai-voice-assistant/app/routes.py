"""
API Routes for the AI Voice Assistant.

Provides RESTful endpoints for:
- Audio transcription (single and streaming)
- Voice command processing
- Task automation
- Third-party integration
- Health checks and metrics
"""

import io
import logging
import os
import time
import uuid
from datetime import datetime

from flask import Blueprint, current_app, jsonify, render_template, request

from app.utils.validators import (
    validate_audio_file,
    validate_command_request,
    validate_integration_request,
    validate_task_request,
    validate_webhook_request,
)
from app.utils.metrics import RequestMetrics

logger = logging.getLogger(__name__)

# API Blueprint
api_bp = Blueprint("api", __name__)

# Web Blueprint (dashboard)
web_bp = Blueprint("web", __name__)


# ──────────────────────────────────────────────
#  Web Dashboard
# ──────────────────────────────────────────────

@web_bp.route("/")
def dashboard():
    """Serve the web dashboard."""
    return render_template("dashboard.html")


# ──────────────────────────────────────────────
#  Health Check
# ──────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "preprocessor": "ready",
            "recognizer": "ready" if current_app.recognizer.is_loaded else "loading",
            "command_processor": "ready",
            "task_manager": "ready",
        },
        "version": "1.0.0",
    })


# ──────────────────────────────────────────────
#  Transcription Endpoints
# ──────────────────────────────────────────────

@api_bp.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe an audio file to text.

    Accepts: multipart/form-data with 'audio' file
    Optional params: language (str), use_chunking (bool)

    Returns: JSON with transcription, segments, and metadata.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    # Validate file upload
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided", "field": "audio"}), 400

    audio_file = request.files["audio"]
    is_valid, error_msg = validate_audio_file(
        audio_file.filename,
        audio_file.content_length or 0,
    )

    # Read audio data
    audio_data = audio_file.read()
    if not audio_data:
        return jsonify({"error": "Empty audio file"}), 400

    if not is_valid and error_msg and "empty" not in error_msg.lower():
        # Allow unknown content-length but still validate
        is_valid, error_msg = validate_audio_file(
            audio_file.filename, len(audio_data)
        )
        if not is_valid:
            return jsonify({"error": error_msg}), 400

    language = request.form.get("language")
    use_chunking = request.form.get("use_chunking", "true").lower() == "true"

    try:
        # Stage 1: Preprocessing
        ext = audio_file.filename.rsplit(".", 1)[-1].lower() if audio_file.filename else "wav"
        preprocess_result = current_app.preprocessor.process(audio_data, ext)

        # Stage 2: Transcription
        if use_chunking:
            transcription = current_app.recognizer.transcribe_chunked(
                preprocess_result.audio,
                preprocess_result.sample_rate,
                language=language,
            )
        else:
            transcription = current_app.recognizer.transcribe(
                preprocess_result.audio,
                preprocess_result.sample_rate,
                language=language,
            )

        total_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        current_app.metrics.record(RequestMetrics(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            total_latency_ms=total_ms,
            preprocessing_ms=preprocess_result.preprocessing_time_ms,
            transcription_ms=transcription.transcription_time_ms,
            audio_duration_s=preprocess_result.duration_s,
            transcript_length=len(transcription.text),
            was_successful=True,
        ))

        return jsonify({
            "request_id": request_id,
            "text": transcription.text,
            "language": transcription.language,
            "language_confidence": transcription.language_confidence,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start_time,
                    "end": s.end_time,
                    "confidence": s.confidence,
                }
                for s in transcription.segments
            ],
            "preprocessing": {
                "duration_s": preprocess_result.duration_s,
                "original_duration_s": preprocess_result.original_duration_s,
                "noise_level_db": preprocess_result.noise_level_db,
                "speech_ratio": preprocess_result.speech_ratio,
                "was_denoised": preprocess_result.was_denoised,
                "was_dereverberated": preprocess_result.was_dereverberated,
                "was_tempo_adjusted": preprocess_result.was_tempo_adjusted,
                "preprocessing_time_ms": preprocess_result.preprocessing_time_ms,
            },
            "performance": {
                "total_latency_ms": round(total_ms, 1),
                "preprocessing_ms": round(preprocess_result.preprocessing_time_ms, 1),
                "transcription_ms": round(transcription.transcription_time_ms, 1),
                "model": transcription.model_size,
            },
        })

    except Exception as e:
        total_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Transcription failed [%s]: %s", request_id, e, exc_info=True)

        current_app.metrics.record(RequestMetrics(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            total_latency_ms=total_ms,
            was_successful=False,
            error=str(e),
        ))

        return jsonify({
            "error": "Transcription failed",
            "detail": str(e),
            "request_id": request_id,
        }), 500


# ──────────────────────────────────────────────
#  Voice Command Processing
# ──────────────────────────────────────────────

@api_bp.route("/command", methods=["POST"])
def process_command():
    """
    Process a voice command.

    Accepts either:
    - multipart/form-data with 'audio' file (full pipeline)
    - JSON with 'text' field (text-only processing)

    Returns: JSON with intent, entities, response, and confidence.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    preprocess_ms = 0.0
    transcription_ms = 0.0
    audio_duration = 0.0

    try:
        # Determine input type
        if "audio" in request.files:
            # Full pipeline: audio → preprocess → transcribe → command
            audio_file = request.files["audio"]
            audio_data = audio_file.read()
            if not audio_data:
                return jsonify({"error": "Empty audio file"}), 400

            ext = audio_file.filename.rsplit(".", 1)[-1].lower() if audio_file.filename else "wav"

            # Preprocess
            preprocess_result = current_app.preprocessor.process(audio_data, ext)
            preprocess_ms = preprocess_result.preprocessing_time_ms
            audio_duration = preprocess_result.duration_s

            # Transcribe
            transcription = current_app.recognizer.transcribe_chunked(
                preprocess_result.audio, preprocess_result.sample_rate
            )
            transcription_ms = transcription.transcription_time_ms
            text = transcription.text

        elif request.is_json:
            data = request.get_json()
            is_valid, error_msg = validate_command_request(data)
            if not is_valid:
                return jsonify({"error": error_msg}), 400
            text = data["text"]

        else:
            return jsonify({
                "error": "Provide either an audio file or JSON with 'text' field"
            }), 400

        # Process command
        use_context = request.args.get("context", "false").lower() == "true"
        command = current_app.command_processor.process(text, use_context=use_context)

        total_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        current_app.metrics.record(RequestMetrics(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            total_latency_ms=total_ms,
            preprocessing_ms=preprocess_ms,
            transcription_ms=transcription_ms,
            command_processing_ms=command.processing_time_ms,
            audio_duration_s=audio_duration,
            transcript_length=len(text),
            intent=command.intent,
            confidence=command.confidence,
            was_successful=command.error is None,
        ))

        response_data = command.to_dict()
        response_data["request_id"] = request_id
        response_data["performance"] = {
            "total_latency_ms": round(total_ms, 1),
            "preprocessing_ms": round(preprocess_ms, 1),
            "transcription_ms": round(transcription_ms, 1),
            "command_processing_ms": round(command.processing_time_ms, 1),
        }

        return jsonify(response_data)

    except Exception as e:
        total_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Command processing failed [%s]: %s", request_id, e, exc_info=True)

        current_app.metrics.record(RequestMetrics(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            total_latency_ms=total_ms,
            was_successful=False,
            error=str(e),
        ))

        return jsonify({
            "error": "Command processing failed",
            "detail": str(e),
            "request_id": request_id,
        }), 500


# ──────────────────────────────────────────────
#  Streaming Endpoint
# ──────────────────────────────────────────────

@api_bp.route("/stream", methods=["POST"])
def stream_audio():
    """
    Process streaming audio in real-time chunks.

    Accepts: multipart/form-data with 'audio' file
    Uses optimised audio chunking for reduced latency.

    Returns: JSON with per-chunk results and aggregate transcription.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_data = audio_file.read()
    if not audio_data:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        ext = audio_file.filename.rsplit(".", 1)[-1].lower() if audio_file.filename else "wav"

        # Preprocess
        preprocess_result = current_app.preprocessor.process(audio_data, ext)

        # Chunk audio
        chunks = current_app.audio_chunker.chunk_audio(
            preprocess_result.audio, use_smart_boundaries=True
        )

        # Process each chunk
        chunk_results = []
        full_text_parts = []

        for chunk in chunks:
            if not chunk.has_speech:
                continue

            chunk_start = time.perf_counter()
            result = current_app.recognizer.transcribe(
                chunk.data, preprocess_result.sample_rate
            )
            chunk_ms = (time.perf_counter() - chunk_start) * 1000

            chunk_results.append({
                "chunk_index": chunk.index,
                "text": result.text,
                "duration_s": chunk.duration_s,
                "latency_ms": round(chunk_ms, 1),
                "confidence": result.language_confidence,
            })
            if result.text:
                full_text_parts.append(result.text)

        # Latency analysis
        latency_stats = current_app.audio_chunker.estimate_latency_reduction(
            preprocess_result.audio
        )

        total_ms = (time.perf_counter() - start_time) * 1000

        return jsonify({
            "request_id": request_id,
            "text": " ".join(full_text_parts),
            "chunks": chunk_results,
            "chunking_stats": latency_stats,
            "performance": {
                "total_latency_ms": round(total_ms, 1),
                "preprocessing_ms": round(preprocess_result.preprocessing_time_ms, 1),
                "chunks_processed": len(chunk_results),
                "total_chunks": len(chunks),
            },
        })

    except Exception as e:
        logger.error("Stream processing failed [%s]: %s", request_id, e, exc_info=True)
        return jsonify({
            "error": "Stream processing failed",
            "detail": str(e),
            "request_id": request_id,
        }), 500


# ──────────────────────────────────────────────
#  Task Automation
# ──────────────────────────────────────────────

@api_bp.route("/tasks", methods=["POST"])
def create_task():
    """Create a new automation task."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()
    is_valid, error_msg = validate_task_request(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    task = current_app.task_manager.create_task(
        task_type=data["type"],
        description=data["description"],
        scheduled_at=data.get("scheduled_at"),
        metadata=data.get("metadata", {}),
        webhook_url=data.get("webhook_url"),
    )

    # Auto-execute if requested
    if data.get("execute", False):
        task = current_app.task_manager.execute_task(task.id)

    return jsonify(task.to_dict()), 201


@api_bp.route("/tasks", methods=["GET"])
def list_tasks():
    """List tasks with optional filtering."""
    status = request.args.get("status")
    task_type = request.args.get("type")
    limit = min(int(request.args.get("limit", 50)), 100)

    tasks = current_app.task_manager.list_tasks(status, task_type, limit)
    return jsonify({
        "tasks": [t.to_dict() for t in tasks],
        "count": len(tasks),
    })


@api_bp.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    """Get a specific task by ID."""
    task = current_app.task_manager.get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task.to_dict())


@api_bp.route("/tasks/<task_id>/execute", methods=["POST"])
def execute_task(task_id):
    """Execute a pending task."""
    task = current_app.task_manager.execute_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task.to_dict())


@api_bp.route("/tasks/<task_id>/cancel", methods=["POST"])
def cancel_task(task_id):
    """Cancel a pending task."""
    task = current_app.task_manager.cancel_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task.to_dict())


@api_bp.route("/tasks/stats", methods=["GET"])
def task_stats():
    """Get task statistics."""
    return jsonify(current_app.task_manager.get_stats())


# ──────────────────────────────────────────────
#  Third-Party Integration
# ──────────────────────────────────────────────

@api_bp.route("/integrate", methods=["POST"])
def register_integration():
    """Register a new third-party integration."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    # Check if using a template
    template = data.get("template")
    if template:
        integration = current_app.integration_service.register_from_template(
            template, auth_token=data.get("auth_token", "")
        )
        if not integration:
            return jsonify({"error": f"Unknown template: {template}"}), 400
        return jsonify(integration.to_dict()), 201

    is_valid, error_msg = validate_integration_request(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    integration = current_app.integration_service.register_integration(
        name=data["name"],
        base_url=data["base_url"],
        auth_type=data.get("auth_type", "bearer"),
        auth_token=data.get("auth_token", ""),
        headers=data.get("headers"),
    )

    return jsonify(integration.to_dict()), 201


@api_bp.route("/integrate", methods=["GET"])
def list_integrations():
    """List all registered integrations."""
    integrations = current_app.integration_service.list_integrations()
    return jsonify({
        "integrations": [i.to_dict() for i in integrations],
        "count": len(integrations),
    })


@api_bp.route("/integrate/<integration_id>/call", methods=["POST"])
def call_integration(integration_id):
    """Make an API call through a registered integration."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()
    result = current_app.integration_service.call_integration(
        integration_id=integration_id,
        endpoint=data.get("endpoint", ""),
        method=data.get("method", "POST"),
        data=data.get("data"),
        params=data.get("params"),
    )

    status = 200 if result.success else result.status_code
    return jsonify(result.to_dict()), status


@api_bp.route("/integrate/templates", methods=["GET"])
def list_integration_templates():
    """List available integration templates."""
    return jsonify({
        "templates": current_app.integration_service.get_available_templates()
    })


# ──────────────────────────────────────────────
#  Webhooks
# ──────────────────────────────────────────────

@api_bp.route("/webhooks", methods=["POST"])
def register_webhook():
    """Register a webhook for event notifications."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()
    is_valid, error_msg = validate_webhook_request(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    webhook = current_app.integration_service.register_webhook(
        url=data["url"],
        events=data["events"],
    )

    return jsonify(webhook.to_dict()), 201


@api_bp.route("/webhooks", methods=["GET"])
def list_webhooks():
    """List registered webhooks."""
    webhooks = current_app.integration_service.list_webhooks()
    return jsonify({
        "webhooks": [w.to_dict() for w in webhooks],
        "count": len(webhooks),
    })


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Get performance metrics."""
    return jsonify(current_app.metrics.get_summary())


@api_bp.route("/metrics/recent", methods=["GET"])
def get_recent_metrics():
    """Get recent request metrics."""
    n = min(int(request.args.get("n", 10)), 50)
    return jsonify({"recent": current_app.metrics.get_recent(n)})


# ──────────────────────────────────────────────
#  Command Context Management
# ──────────────────────────────────────────────

@api_bp.route("/context/clear", methods=["POST"])
def clear_context():
    """Clear conversation context."""
    current_app.command_processor.clear_context()
    return jsonify({"status": "context_cleared"})


@api_bp.route("/intents", methods=["GET"])
def list_intents():
    """List supported command intents."""
    return jsonify({
        "intents": current_app.command_processor.get_supported_intents()
    })
