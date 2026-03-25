"""
Input Validation Utilities.

Validates audio files, API request payloads, and configuration parameters.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "flac", "ogg", "m4a", "webm", "mp4"}
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
MAX_AUDIO_DURATION_S = 300  # 5 minutes


def validate_audio_file(
    filename: Optional[str],
    file_size: int,
    content_type: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate an uploaded audio file.

    Returns:
        (is_valid, error_message)
    """
    if not filename:
        return False, "No filename provided"

    # Check extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_AUDIO_FORMATS:
        return False, (
            f"Unsupported format: .{ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )

    # Check size
    if file_size > MAX_AUDIO_SIZE_BYTES:
        return False, (
            f"File too large: {file_size / 1024 / 1024:.1f}MB. "
            f"Maximum: {MAX_AUDIO_SIZE_BYTES / 1024 / 1024:.0f}MB"
        )

    if file_size == 0:
        return False, "File is empty"

    return True, ""


def validate_transcribe_request(data: dict) -> tuple[bool, str]:
    """Validate a transcription API request."""
    language = data.get("language")
    if language and len(language) > 10:
        return False, "Invalid language code"
    return True, ""


def validate_command_request(data: dict) -> tuple[bool, str]:
    """Validate a command processing request."""
    text = data.get("text", "")
    if not text or not text.strip():
        return False, "Text field is required and cannot be empty"
    if len(text) > 5000:
        return False, "Text too long (max 5000 characters)"
    return True, ""


def validate_task_request(data: dict) -> tuple[bool, str]:
    """Validate a task creation request."""
    if not data.get("type"):
        return False, "Task type is required"
    if not data.get("description"):
        return False, "Task description is required"
    if len(data.get("description", "")) > 1000:
        return False, "Description too long (max 1000 characters)"
    return True, ""


def validate_integration_request(data: dict) -> tuple[bool, str]:
    """Validate an integration registration request."""
    if not data.get("name"):
        return False, "Integration name is required"
    if not data.get("base_url"):
        return False, "Base URL is required"
    base_url = data["base_url"]
    if not base_url.startswith(("http://", "https://")):
        return False, "Base URL must start with http:// or https://"
    return True, ""


def validate_webhook_request(data: dict) -> tuple[bool, str]:
    """Validate a webhook registration request."""
    if not data.get("url"):
        return False, "Webhook URL is required"
    url = data["url"]
    if not url.startswith(("http://", "https://")):
        return False, "Webhook URL must start with http:// or https://"
    events = data.get("events", [])
    if not events:
        return False, "At least one event type is required"
    return True, ""
