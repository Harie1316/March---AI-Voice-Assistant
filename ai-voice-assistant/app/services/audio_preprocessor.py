"""
Audio Preprocessing Pipeline for Speech Domain Adaptation.

Handles domain variability by normalising for:
- Background noise (spectral gating, adaptive noise reduction)
- Reverberation (dereverberation via spectral processing)
- Speaking rate differences (tempo normalisation)
- Volume normalisation (peak and RMS-based)
- Pre-emphasis filtering for speech clarity

Uses Librosa for all DSP operations, achieving robust performance
across varied acoustic conditions including noisy and accented speech.
"""

import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result container for the preprocessing pipeline."""
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    preprocessing_time_ms: float
    noise_level_db: float
    speech_ratio: float
    was_normalised: bool
    was_denoised: bool
    was_dereverberated: bool
    was_tempo_adjusted: bool
    original_duration_s: float


class AudioPreprocessor:
    """
    Production-grade audio preprocessing pipeline for speech domain adaptation.
    
    Implements a multi-stage pipeline:
    1. Format conversion & resampling
    2. Pre-emphasis filtering
    3. Voice Activity Detection (VAD)
    4. Noise estimation & spectral gating
    5. Dereverberation
    6. Speaking rate normalisation
    7. Loudness normalisation
    """

    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.noise_reduction_strength = config.noise_reduction_strength
        self.normalize_db = config.normalize_db
        self.pre_emphasis_coeff = config.pre_emphasis_coeff
        self.silence_threshold_db = config.silence_threshold_db
        self.min_speech_duration_ms = config.min_speech_duration_ms

        logger.info(
            "AudioPreprocessor initialised — SR=%d, noise_reduction=%.2f",
            self.sample_rate, self.noise_reduction_strength,
        )

    def process(self, audio_data: bytes, source_format: str = "wav") -> PreprocessingResult:
        """
        Run the full preprocessing pipeline on raw audio bytes.

        Args:
            audio_data: Raw audio file bytes.
            source_format: Audio format hint (wav, mp3, flac, etc.).

        Returns:
            PreprocessingResult with cleaned, normalised audio.
        """
        start_time = time.perf_counter()

        # Stage 1: Load & resample
        audio, sr = self._load_audio(audio_data, source_format)
        original_duration = len(audio) / sr
        logger.info("Loaded audio: %.2fs at %d Hz", original_duration, sr)

        # Stage 2: Pre-emphasis filter
        audio = self._apply_pre_emphasis(audio)

        # Stage 3: Voice Activity Detection
        speech_ratio, vad_mask = self._detect_speech(audio, sr)
        logger.info("Speech ratio: %.2f%%", speech_ratio * 100)

        # Stage 4: Noise estimation & reduction
        noise_level_db = self._estimate_noise_level(audio, vad_mask, sr)
        was_denoised = False
        if noise_level_db > self.silence_threshold_db:
            audio = self._reduce_noise(audio, sr, vad_mask)
            was_denoised = True
            logger.info("Noise reduced — estimated noise: %.1f dB", noise_level_db)

        # Stage 5: Dereverberation
        was_dereverberated = False
        reverb_level = self._estimate_reverberation(audio, sr)
        if reverb_level > 0.3:
            audio = self._dereverberate(audio, sr)
            was_dereverberated = True
            logger.info("Dereverberated — reverb level: %.2f", reverb_level)

        # Stage 6: Speaking rate normalisation
        was_tempo_adjusted = False
        tempo = self._estimate_speaking_rate(audio, sr)
        if tempo < 0.8 or tempo > 1.3:
            audio = self._normalise_tempo(audio, sr, tempo)
            was_tempo_adjusted = True
            logger.info("Tempo adjusted from %.2fx to 1.0x", tempo)

        # Stage 7: Loudness normalisation
        audio = self._normalise_loudness(audio, sr)

        # Trim silence
        audio = self._trim_silence(audio, sr)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return PreprocessingResult(
            audio=audio,
            sample_rate=sr,
            duration_s=len(audio) / sr,
            preprocessing_time_ms=elapsed_ms,
            noise_level_db=noise_level_db,
            speech_ratio=speech_ratio,
            was_normalised=True,
            was_denoised=was_denoised,
            was_dereverberated=was_dereverberated,
            was_tempo_adjusted=was_tempo_adjusted,
            original_duration_s=original_duration,
        )

    def _load_audio(self, audio_data: bytes, source_format: str) -> tuple[np.ndarray, int]:
        """Load audio bytes and resample to target sample rate."""
        try:
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = librosa.load(
                audio_buffer,
                sr=self.sample_rate,
                mono=True,
                dtype=np.float32,
            )
        except Exception as e:
            logger.error("Failed to load audio with librosa: %s", e)
            # Fallback: try soundfile
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = sf.read(audio_buffer, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate

        # Ensure float32 in [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        return audio, sr

    def _apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies for speech clarity."""
        return np.append(audio[0], audio[1:] - self.pre_emphasis_coeff * audio[:-1])

    def _detect_speech(self, audio: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
        """
        Energy-based Voice Activity Detection.
        Returns speech ratio and frame-level binary mask.
        """
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        # Compute short-time energy
        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Convert to dB
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)

        # Adaptive threshold: use the noise floor + margin
        sorted_energy = np.sort(energy_db)
        noise_floor = np.mean(sorted_energy[:max(1, len(sorted_energy) // 10)])
        threshold = noise_floor + 12  # 12 dB above noise floor

        # Create speech mask
        vad_mask = energy_db > threshold

        # Smooth the mask with median filter to remove spurious detections
        min_frames = int(self.min_speech_duration_ms / 10)
        if min_frames > 1:
            vad_mask = median_filter(vad_mask.astype(float), size=min_frames) > 0.5

        speech_ratio = np.mean(vad_mask)
        return float(speech_ratio), vad_mask

    def _estimate_noise_level(
        self, audio: np.ndarray, vad_mask: np.ndarray, sr: int
    ) -> float:
        """Estimate background noise level in dB from non-speech segments."""
        hop_length = int(0.010 * sr)
        frame_length = int(0.025 * sr)

        # Extract non-speech frames
        non_speech_mask = ~vad_mask
        if not np.any(non_speech_mask):
            return float(self.silence_threshold_db - 10)  # Very clean signal

        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Only look at non-speech energy
        min_len = min(len(energy), len(non_speech_mask))
        noise_energy = energy[:min_len][non_speech_mask[:min_len]]

        if len(noise_energy) == 0:
            return float(self.silence_threshold_db - 10)

        noise_rms = np.mean(noise_energy)
        noise_db = 20 * np.log10(max(noise_rms, 1e-10))
        return float(noise_db)

    def _reduce_noise(
        self, audio: np.ndarray, sr: int, vad_mask: np.ndarray
    ) -> np.ndarray:
        """
        Spectral gating noise reduction.
        Estimates noise profile from non-speech segments and applies
        frequency-domain gating to suppress noise.
        """
        n_fft = 2048
        hop_length = int(0.010 * sr)

        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise spectrum from non-speech frames
        n_frames = magnitude.shape[1]
        mask_len = min(len(vad_mask), n_frames)
        non_speech_mask = ~vad_mask[:mask_len]

        if np.any(non_speech_mask):
            noise_spectrum = np.mean(magnitude[:, :mask_len][:, non_speech_mask], axis=1)
        else:
            noise_spectrum = np.percentile(magnitude, 10, axis=1)

        # Apply spectral gating
        noise_gate = noise_spectrum[:, np.newaxis] * (1 + self.noise_reduction_strength)
        gain = np.maximum(magnitude - noise_gate, 0) / (magnitude + 1e-10)

        # Smooth gain to avoid musical noise
        gain = median_filter(gain, size=(1, 5))

        # Soft knee: blend between original and gated
        alpha = self.noise_reduction_strength
        clean_magnitude = magnitude * (alpha * gain + (1 - alpha))

        # Reconstruct
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length, length=len(audio))

        return clean_audio.astype(np.float32)

    def _estimate_reverberation(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate reverberation level using energy decay analysis.
        Returns a value between 0 (dry) and 1 (highly reverberant).
        """
        # Compute autocorrelation
        frame_length = int(0.1 * sr)  # 100ms analysis window
        n_frames = min(5, len(audio) // frame_length)

        if n_frames < 1:
            return 0.0

        reverb_scores = []
        for i in range(n_frames):
            start = i * frame_length
            end = start + frame_length
            frame = audio[start:end]

            # Normalised autocorrelation
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]

            # Measure decay — high reverb = slow decay
            decay_point = int(0.05 * sr)  # Check at 50ms
            if decay_point < len(autocorr):
                reverb_scores.append(float(autocorr[decay_point]))

        return float(np.mean(reverb_scores)) if reverb_scores else 0.0

    def _dereverberate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply dereverberation using spectral subtraction of late reflections.
        """
        n_fft = 2048
        hop_length = 512

        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate late reverberation energy (delayed spectral average)
        delay_frames = max(1, int(0.05 * sr / hop_length))  # 50ms delay
        reverb_estimate = np.zeros_like(magnitude)
        if delay_frames < magnitude.shape[1]:
            reverb_estimate[:, delay_frames:] = magnitude[:, :-delay_frames] * 0.5

        # Subtract reverb estimate with spectral floor
        clean_magnitude = np.maximum(
            magnitude - reverb_estimate * self.noise_reduction_strength,
            magnitude * 0.1  # Spectral floor
        )

        # Reconstruct
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length, length=len(audio))

        return clean_audio.astype(np.float32)

    def _estimate_speaking_rate(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate relative speaking rate using onset detection.
        Returns rate relative to normal (1.0 = normal, >1 = fast, <1 = slow).
        """
        # Detect onsets (syllable-level)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units="time"
        )

        if len(onsets) < 2:
            return 1.0

        # Calculate syllable rate
        duration = len(audio) / sr
        syllable_rate = len(onsets) / duration

        # Normal English speaking rate: ~4-5 syllables/second
        normal_rate = 4.5
        relative_rate = syllable_rate / normal_rate

        return float(np.clip(relative_rate, 0.5, 2.0))

    def _normalise_tempo(
        self, audio: np.ndarray, sr: int, current_rate: float
    ) -> np.ndarray:
        """
        Normalise speaking rate using time-stretching (preserves pitch).
        """
        # Stretch factor: if speaking too fast (rate>1), slow down (stretch>1)
        target_rate = 1.0
        stretch_factor = current_rate / target_rate

        # Only adjust if significantly different from normal
        stretch_factor = np.clip(stretch_factor, 0.75, 1.5)

        stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
        return stretched.astype(np.float32)

    def _normalise_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        RMS-based loudness normalisation to target dB level.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio

        current_db = 20 * np.log10(rms)
        gain_db = self.normalize_db - current_db
        gain_linear = 10 ** (gain_db / 20)

        normalised = audio * gain_linear

        # Prevent clipping with soft limiter
        peak = np.max(np.abs(normalised))
        if peak > 0.95:
            normalised = normalised * (0.95 / peak)

        return normalised.astype(np.float32)

    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=abs(self.silence_threshold_db),
            frame_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),
        )
        return trimmed

    def to_wav_bytes(self, audio: np.ndarray, sr: int) -> bytes:
        """Convert processed audio array back to WAV bytes."""
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()
