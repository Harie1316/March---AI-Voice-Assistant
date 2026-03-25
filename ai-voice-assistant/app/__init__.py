"""
Flask Application Factory.

Initialises all services and registers API routes.
Configures CORS, logging, and error handlers.
"""

import logging
import os
import sys

from flask import Flask
from flask_cors import CORS

from config.settings import settings


def create_app(config_override=None) -> Flask:
    """Create and configure the Flask application."""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.metrics.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Validate configuration
    warnings = settings.validate()
    for warning in warnings:
        logger.warning("Config: %s", warning)

    # Create Flask app
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.config["MAX_CONTENT_LENGTH"] = settings.flask.max_content_length
    app.config["UPLOAD_FOLDER"] = settings.flask.upload_folder

    if config_override:
        app.config.update(config_override)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": settings.flask.cors_origins}})

    # Ensure upload directory exists
    os.makedirs(settings.flask.upload_folder, exist_ok=True)

    # Initialise services
    from app.services.audio_preprocessor import AudioPreprocessor
    from app.services.speech_recognizer import SpeechRecognizer
    from app.services.command_processor import CommandProcessor
    from app.services.task_manager import TaskManager
    from app.services.integration_service import IntegrationService
    from app.utils.metrics import MetricsTracker
    from app.utils.audio_chunker import AudioChunker

    app.preprocessor = AudioPreprocessor(settings.audio)
    app.recognizer = SpeechRecognizer(settings.whisper)
    app.command_processor = CommandProcessor(settings.openai)
    app.task_manager = TaskManager()
    app.integration_service = IntegrationService()
    app.metrics = MetricsTracker(window_size=settings.metrics.rolling_window_size)
    app.audio_chunker = AudioChunker(
        sample_rate=settings.audio.sample_rate,
        chunk_duration_ms=settings.audio.chunk_duration_ms,
        overlap_ms=settings.performance.audio_chunk_overlap_ms,
    )

    # Load Whisper model
    try:
        app.recognizer.load_model()
    except Exception as e:
        logger.warning("Whisper model load deferred: %s", e)

    # Register routes
    from app.routes import api_bp, web_bp
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    app.register_blueprint(web_bp)

    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        return {
            "error": "File too large",
            "max_size_mb": settings.flask.max_content_length / 1024 / 1024,
        }, 413

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Endpoint not found"}, 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error("Internal server error: %s", e)
        return {"error": "Internal server error"}, 500

    logger.info(
        "Voice Assistant app initialised — port=%d, debug=%s",
        settings.flask.port, settings.flask.debug,
    )

    return app
