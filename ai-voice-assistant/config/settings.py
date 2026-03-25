"""
Configuration management for the AI Voice Assistant.
Supports environment-based configuration for development, testing, and production.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    chunk_duration_ms: int = 500
    max_audio_length_s: int = 300
    supported_formats: tuple = ("wav", "mp3", "flac", "ogg", "m4a", "webm")
    noise_reduction_strength: float = 0.75
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    normalize_db: float = -20.0
    pre_emphasis_coeff: float = 0.97
    min_speech_duration_ms: int = 250
    silence_threshold_db: float = -40.0


@dataclass
class WhisperConfig:
    """Whisper speech recognition configuration."""
    model_size: str = "base"
    language: Optional[str] = None  # None = auto-detect
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    fp16: bool = True
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 1024
    temperature: float = 0.3
    system_prompt: str = (
        "You are a highly capable voice assistant. Parse the user's spoken command "
        "and respond with a structured JSON containing: "
        "'intent' (the action category), "
        "'entities' (key parameters extracted), "
        "'response' (natural language response to speak back), "
        "and 'confidence' (0.0-1.0). "
        "Supported intents: search, reminder, timer, weather, email, calendar, "
        "smart_home, music, navigation, general_query, task_create, unknown."
    )
    timeout: int = 30


@dataclass
class FlaskConfig:
    """Flask server configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    max_content_length: int = 50 * 1024 * 1024  # 50MB
    upload_folder: str = "/tmp/voice_assistant_uploads"
    cors_origins: str = "*"


@dataclass
class PerformanceConfig:
    """Performance and optimisation settings."""
    max_concurrent_requests: int = 50
    request_timeout_s: int = 60
    audio_chunk_overlap_ms: int = 50
    enable_streaming: bool = True
    cache_ttl_s: int = 300
    worker_threads: int = 4
    batch_size: int = 8


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    track_latency: bool = True
    track_accuracy: bool = True
    rolling_window_size: int = 100


class Settings:
    """
    Central settings manager.
    Loads configuration from environment variables with sensible defaults.
    """

    def __init__(self):
        self.audio = AudioConfig()
        self.whisper = WhisperConfig()
        self.openai = OpenAIConfig()
        self.flask = FlaskConfig()
        self.performance = PerformanceConfig()
        self.metrics = MetricsConfig()
        self._load_from_env()

    def _load_from_env(self):
        """Override defaults with environment variables."""
        # OpenAI
        self.openai.api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai.model = os.getenv("OPENAI_MODEL", self.openai.model)

        # Whisper
        self.whisper.model_size = os.getenv("WHISPER_MODEL", self.whisper.model_size)
        lang = os.getenv("WHISPER_LANGUAGE")
        if lang:
            self.whisper.language = lang

        # Flask
        self.flask.debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        self.flask.port = int(os.getenv("PORT", str(self.flask.port)))

        # Audio
        self.audio.sample_rate = int(
            os.getenv("AUDIO_SAMPLE_RATE", str(self.audio.sample_rate))
        )
        self.audio.noise_reduction_strength = float(
            os.getenv("NOISE_REDUCTION_STRENGTH", str(self.audio.noise_reduction_strength))
        )

        # Performance
        self.performance.max_concurrent_requests = int(
            os.getenv("MAX_CONCURRENT", str(self.performance.max_concurrent_requests))
        )
        self.performance.worker_threads = int(
            os.getenv("WORKER_THREADS", str(self.performance.worker_threads))
        )

        # Metrics
        self.metrics.log_level = os.getenv("LOG_LEVEL", self.metrics.log_level)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        if not self.openai.api_key:
            warnings.append("OPENAI_API_KEY not set — API features will be unavailable")
        if self.whisper.model_size not in ("tiny", "base", "small", "medium", "large"):
            warnings.append(f"Unknown Whisper model size: {self.whisper.model_size}")
        return warnings


# Singleton instance
settings = Settings()
