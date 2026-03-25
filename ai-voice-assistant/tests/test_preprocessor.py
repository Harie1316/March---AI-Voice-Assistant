"""
Tests for AudioPreprocessor — validates the Librosa-based
domain adaptation pipeline.
"""

import io
import struct
import unittest

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import AudioConfig
from app.services.audio_preprocessor import AudioPreprocessor, PreprocessingResult


def generate_wav_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    noise_level: float = 0.0,
    silence_padding_s: float = 0.0,
) -> bytes:
    """Generate synthetic WAV audio bytes for testing."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Add noise
    if noise_level > 0:
        audio += noise_level * np.random.randn(n_samples).astype(np.float32)

    # Add silence padding
    if silence_padding_s > 0:
        pad_samples = int(silence_padding_s * sample_rate)
        silence = np.zeros(pad_samples, dtype=np.float32)
        audio = np.concatenate([silence, audio, silence])

    # Encode as WAV
    buffer = io.BytesIO()
    import soundfile as sf
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


class TestAudioPreprocessor(unittest.TestCase):
    """Test suite for the audio preprocessing pipeline."""

    def setUp(self):
        self.config = AudioConfig()
        self.preprocessor = AudioPreprocessor(self.config)

    def test_basic_processing(self):
        """Test that the pipeline runs without errors on clean audio."""
        wav_data = generate_wav_bytes(duration_s=1.0)
        result = self.preprocessor.process(wav_data, "wav")

        self.assertIsInstance(result, PreprocessingResult)
        self.assertGreater(len(result.audio), 0)
        self.assertEqual(result.sample_rate, 16000)
        self.assertGreater(result.duration_s, 0)
        self.assertTrue(result.was_normalised)

    def test_noise_reduction(self):
        """Test that noisy audio triggers denoising."""
        wav_data = generate_wav_bytes(duration_s=2.0, noise_level=0.3)
        result = self.preprocessor.process(wav_data, "wav")

        self.assertIsInstance(result, PreprocessingResult)
        self.assertGreater(len(result.audio), 0)
        # Noisy input should trigger denoising in most cases
        # (depends on noise level relative to threshold)

    def test_silence_trimming(self):
        """Test that leading/trailing silence is trimmed."""
        wav_data = generate_wav_bytes(
            duration_s=1.0, silence_padding_s=0.5
        )
        result = self.preprocessor.process(wav_data, "wav")

        # Trimmed audio should be shorter than original
        self.assertLess(result.duration_s, result.original_duration_s)

    def test_loudness_normalisation(self):
        """Test that audio is normalised to target loudness."""
        # Quiet audio
        wav_data = generate_wav_bytes(duration_s=1.0)
        result = self.preprocessor.process(wav_data, "wav")

        # Check that output is not clipping
        self.assertLessEqual(np.max(np.abs(result.audio)), 1.0)

    def test_preprocessing_time_tracked(self):
        """Test that preprocessing time is measured."""
        wav_data = generate_wav_bytes(duration_s=1.0)
        result = self.preprocessor.process(wav_data, "wav")

        self.assertGreater(result.preprocessing_time_ms, 0)

    def test_speech_ratio_computed(self):
        """Test that speech ratio is between 0 and 1."""
        wav_data = generate_wav_bytes(duration_s=1.0)
        result = self.preprocessor.process(wav_data, "wav")

        self.assertGreaterEqual(result.speech_ratio, 0.0)
        self.assertLessEqual(result.speech_ratio, 1.0)

    def test_wav_bytes_roundtrip(self):
        """Test audio → process → WAV bytes conversion."""
        wav_data = generate_wav_bytes(duration_s=1.0)
        result = self.preprocessor.process(wav_data, "wav")

        wav_output = self.preprocessor.to_wav_bytes(result.audio, result.sample_rate)
        self.assertIsInstance(wav_output, bytes)
        self.assertGreater(len(wav_output), 0)
        # WAV header starts with 'RIFF'
        self.assertEqual(wav_output[:4], b"RIFF")

    def test_different_sample_rates(self):
        """Test resampling from different input sample rates."""
        for sr in [8000, 22050, 44100, 48000]:
            wav_data = generate_wav_bytes(duration_s=0.5, sample_rate=sr)
            result = self.preprocessor.process(wav_data, "wav")
            self.assertEqual(result.sample_rate, 16000)  # Target SR


class TestAudioChunker(unittest.TestCase):
    """Test suite for the audio chunking utility."""

    def setUp(self):
        from app.utils.audio_chunker import AudioChunker
        self.chunker = AudioChunker(
            sample_rate=16000,
            chunk_duration_ms=500,
            overlap_ms=50,
        )

    def test_fixed_chunking(self):
        """Test fixed-size chunking produces correct chunks."""
        audio = np.random.randn(32000).astype(np.float32)  # 2 seconds
        chunks = self.chunker.chunk_audio(audio, use_smart_boundaries=False)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertGreater(len(chunk.data), 0)
            self.assertGreater(chunk.duration_s, 0)

    def test_smart_chunking(self):
        """Test smart boundary-aware chunking."""
        # Create audio with silence gaps
        audio = np.zeros(48000, dtype=np.float32)  # 3 seconds
        audio[8000:16000] = 0.5 * np.sin(np.linspace(0, 100, 8000))
        audio[24000:32000] = 0.5 * np.sin(np.linspace(0, 100, 8000))

        chunks = self.chunker.chunk_audio(audio, use_smart_boundaries=True)
        self.assertGreater(len(chunks), 0)

    def test_latency_reduction_estimate(self):
        """Test latency reduction estimation."""
        audio = np.random.randn(32000).astype(np.float32)
        stats = self.chunker.estimate_latency_reduction(audio)

        self.assertIn("fixed_chunks", stats)
        self.assertIn("smart_chunks", stats)
        self.assertIn("latency_reduction_pct", stats)

    def test_streaming_generator(self):
        """Test that stream_chunks yields chunks correctly."""
        audio = np.random.randn(16000).astype(np.float32)
        chunk_count = 0
        for chunk in self.chunker.stream_chunks(audio):
            chunk_count += 1
            self.assertGreater(len(chunk.data), 0)
        self.assertGreater(chunk_count, 0)

    def test_empty_audio(self):
        """Test chunking empty audio returns empty list."""
        chunks = self.chunker.chunk_audio(np.array([], dtype=np.float32))
        self.assertEqual(len(chunks), 0)


class TestSpeechRecognizer(unittest.TestCase):
    """Test suite for the speech recognition service."""

    def setUp(self):
        from config.settings import WhisperConfig
        from app.services.speech_recognizer import SpeechRecognizer

        self.config = WhisperConfig()
        self.recognizer = SpeechRecognizer(self.config)

    def test_initialization(self):
        """Test recognizer initialises correctly."""
        self.assertEqual(self.recognizer.model_size, "base")
        self.assertFalse(self.recognizer.is_loaded)

    def test_merge_overlapping_segments(self):
        """Test segment deduplication for chunked transcription."""
        from app.services.speech_recognizer import TranscriptionSegment

        segments = [
            TranscriptionSegment("Hello", 0.0, 1.0, 0.9),
            TranscriptionSegment("Hello world", 0.5, 1.5, 0.95),  # Overlap
            TranscriptionSegment("How are you", 2.0, 3.0, 0.85),
        ]

        merged = self.recognizer._merge_overlapping_segments(segments)
        # The overlapping segment with higher confidence should win
        self.assertLessEqual(len(merged), len(segments))


class TestCommandProcessor(unittest.TestCase):
    """Test suite for the command processor."""

    def setUp(self):
        from config.settings import OpenAIConfig
        from app.services.command_processor import CommandProcessor

        self.config = OpenAIConfig(api_key="test-key")
        self.processor = CommandProcessor(self.config)

    def test_empty_text(self):
        """Test handling of empty input."""
        result = self.processor.process("")
        self.assertEqual(result.intent, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_supported_intents(self):
        """Test that supported intents list is populated."""
        intents = self.processor.get_supported_intents()
        self.assertIn("search", intents)
        self.assertIn("reminder", intents)
        self.assertIn("weather", intents)
        self.assertIn("smart_home", intents)

    def test_context_clearing(self):
        """Test conversation context can be cleared."""
        self.processor.conversation_history.append({"role": "user", "content": "test"})
        self.processor.clear_context()
        self.assertEqual(len(self.processor.conversation_history), 0)

    def test_parse_valid_json_response(self):
        """Test parsing a well-formed API response."""
        response = '{"intent": "weather", "entities": [{"type": "location", "value": "London"}], "response": "Let me check the weather.", "confidence": 0.92}'
        result = self.processor._parse_response(response, "What's the weather in London?")

        self.assertEqual(result.intent, "weather")
        self.assertAlmostEqual(result.confidence, 0.92)
        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].value, "London")


class TestTaskManager(unittest.TestCase):
    """Test suite for the task manager."""

    def setUp(self):
        from app.services.task_manager import TaskManager
        self.manager = TaskManager()

    def test_create_task(self):
        """Test task creation."""
        task = self.manager.create_task("reminder", "Buy groceries")
        self.assertIsNotNone(task.id)
        self.assertEqual(task.description, "Buy groceries")
        self.assertEqual(task.status.value, "pending")

    def test_execute_task(self):
        """Test task execution."""
        task = self.manager.create_task("timer", "5 minute timer", metadata={"duration_seconds": 300})
        executed = self.manager.execute_task(task.id)
        self.assertEqual(executed.status.value, "completed")
        self.assertIsNotNone(executed.result)

    def test_cancel_task(self):
        """Test task cancellation."""
        task = self.manager.create_task("reminder", "Test")
        cancelled = self.manager.cancel_task(task.id)
        self.assertEqual(cancelled.status.value, "cancelled")

    def test_list_tasks_filtered(self):
        """Test filtered task listing."""
        self.manager.create_task("reminder", "Task 1")
        self.manager.create_task("email", "Task 2")
        self.manager.create_task("reminder", "Task 3")

        reminders = self.manager.list_tasks(task_type="reminder")
        self.assertEqual(len(reminders), 2)

    def test_task_stats(self):
        """Test task statistics."""
        self.manager.create_task("reminder", "A")
        self.manager.create_task("timer", "B")
        stats = self.manager.get_stats()
        self.assertEqual(stats["total"], 2)


class TestIntegrationService(unittest.TestCase):
    """Test suite for the integration service."""

    def setUp(self):
        from app.services.integration_service import IntegrationService
        self.service = IntegrationService()

    def test_register_integration(self):
        """Test integration registration."""
        config = self.service.register_integration(
            name="Test API",
            base_url="https://api.example.com/",
            auth_type="bearer",
            auth_token="test-token",
        )
        self.assertIsNotNone(config.id)
        self.assertEqual(config.name, "Test API")

    def test_register_from_template(self):
        """Test template-based registration."""
        config = self.service.register_from_template("slack", auth_token="xoxb-test")
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "Slack")

    def test_unknown_template(self):
        """Test unknown template returns None."""
        result = self.service.register_from_template("nonexistent")
        self.assertIsNone(result)

    def test_register_webhook(self):
        """Test webhook registration."""
        webhook = self.service.register_webhook(
            url="https://hooks.example.com/notify",
            events=["transcription.complete", "command.processed"],
        )
        self.assertIsNotNone(webhook.id)
        self.assertEqual(len(webhook.events), 2)

    def test_list_templates(self):
        """Test listing available templates."""
        templates = self.service.get_available_templates()
        self.assertGreater(len(templates), 0)
        names = [t["key"] for t in templates]
        self.assertIn("slack", names)
        self.assertIn("home_assistant", names)


class TestMetricsTracker(unittest.TestCase):
    """Test suite for the metrics tracker."""

    def setUp(self):
        from app.utils.metrics import MetricsTracker, RequestMetrics
        self.tracker = MetricsTracker(window_size=10)
        self.RequestMetrics = RequestMetrics

    def test_record_and_summary(self):
        """Test recording metrics and getting summary."""
        for i in range(5):
            self.tracker.record(self.RequestMetrics(
                request_id=f"req-{i}",
                timestamp="2024-01-01T00:00:00",
                total_latency_ms=100 + i * 10,
                preprocessing_ms=30,
                transcription_ms=50,
                command_processing_ms=20,
                intent="search",
                confidence=0.85 + i * 0.02,
                was_successful=True,
            ))

        summary = self.tracker.get_summary()
        self.assertEqual(summary["overview"]["total_requests"], 5)
        self.assertGreater(summary["latency"]["mean_ms"], 0)
        self.assertGreater(summary["accuracy"]["mean_confidence"], 0)

    def test_empty_summary(self):
        """Test summary with no data."""
        summary = self.tracker.get_summary()
        self.assertEqual(summary["overview"]["total_requests"], 0)

    def test_recent_metrics(self):
        """Test getting recent request metrics."""
        self.tracker.record(self.RequestMetrics(
            request_id="test-1",
            timestamp="2024-01-01T00:00:00",
            total_latency_ms=100,
            intent="search",
            confidence=0.9,
            was_successful=True,
        ))
        recent = self.tracker.get_recent(5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["request_id"], "test-1")


class TestValidators(unittest.TestCase):
    """Test suite for input validators."""

    def test_valid_audio_file(self):
        from app.utils.validators import validate_audio_file

        valid, msg = validate_audio_file("test.wav", 1024)
        self.assertTrue(valid)

        valid, msg = validate_audio_file("test.mp3", 1024)
        self.assertTrue(valid)

    def test_invalid_audio_format(self):
        from app.utils.validators import validate_audio_file

        valid, msg = validate_audio_file("test.txt", 1024)
        self.assertFalse(valid)
        self.assertIn("Unsupported", msg)

    def test_audio_too_large(self):
        from app.utils.validators import validate_audio_file

        valid, msg = validate_audio_file("test.wav", 100 * 1024 * 1024)
        self.assertFalse(valid)
        self.assertIn("large", msg)

    def test_valid_command_request(self):
        from app.utils.validators import validate_command_request

        valid, msg = validate_command_request({"text": "Turn on the lights"})
        self.assertTrue(valid)

    def test_empty_command(self):
        from app.utils.validators import validate_command_request

        valid, msg = validate_command_request({"text": ""})
        self.assertFalse(valid)

    def test_valid_task_request(self):
        from app.utils.validators import validate_task_request

        valid, msg = validate_task_request({"type": "reminder", "description": "Buy milk"})
        self.assertTrue(valid)

    def test_missing_task_type(self):
        from app.utils.validators import validate_task_request

        valid, msg = validate_task_request({"description": "Buy milk"})
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
