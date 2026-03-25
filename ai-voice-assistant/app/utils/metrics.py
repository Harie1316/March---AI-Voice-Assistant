"""
Performance Metrics Tracking.

Tracks latency, accuracy, throughput, and error rates
for the voice assistant pipeline.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    timestamp: str
    total_latency_ms: float
    preprocessing_ms: float = 0.0
    transcription_ms: float = 0.0
    command_processing_ms: float = 0.0
    audio_duration_s: float = 0.0
    transcript_length: int = 0
    intent: str = ""
    confidence: float = 0.0
    was_successful: bool = True
    error: Optional[str] = None


class MetricsTracker:
    """
    Thread-safe metrics tracker with rolling window statistics.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: deque[RequestMetrics] = deque(maxlen=window_size)
        self._lock = Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._start_time = time.time()

    def record(self, metrics: RequestMetrics):
        """Record metrics for a single request."""
        with self._lock:
            self._metrics.append(metrics)
            self._total_requests += 1
            if not metrics.was_successful:
                self._total_errors += 1

    def get_summary(self) -> dict:
        """Get summary statistics over the rolling window."""
        with self._lock:
            if not self._metrics:
                return self._empty_summary()

            metrics_list = list(self._metrics)

        latencies = [m.total_latency_ms for m in metrics_list]
        preproc = [m.preprocessing_ms for m in metrics_list]
        transcription = [m.transcription_ms for m in metrics_list]
        command = [m.command_processing_ms for m in metrics_list]
        confidences = [m.confidence for m in metrics_list if m.confidence > 0]
        successful = [m for m in metrics_list if m.was_successful]

        uptime = time.time() - self._start_time

        return {
            "overview": {
                "total_requests": self._total_requests,
                "window_size": len(metrics_list),
                "success_rate": len(successful) / max(len(metrics_list), 1) * 100,
                "error_rate": self._total_errors / max(self._total_requests, 1) * 100,
                "uptime_s": round(uptime, 1),
                "requests_per_minute": round(
                    self._total_requests / max(uptime / 60, 1), 2
                ),
            },
            "latency": {
                "mean_ms": round(np.mean(latencies), 1),
                "median_ms": round(np.median(latencies), 1),
                "p95_ms": round(np.percentile(latencies, 95), 1),
                "p99_ms": round(np.percentile(latencies, 99), 1),
                "min_ms": round(min(latencies), 1),
                "max_ms": round(max(latencies), 1),
            },
            "pipeline_breakdown": {
                "preprocessing_mean_ms": round(np.mean(preproc), 1),
                "transcription_mean_ms": round(np.mean(transcription), 1),
                "command_processing_mean_ms": round(np.mean(command), 1),
            },
            "accuracy": {
                "mean_confidence": round(np.mean(confidences), 3) if confidences else 0,
                "median_confidence": round(np.median(confidences), 3) if confidences else 0,
                "high_confidence_rate": round(
                    len([c for c in confidences if c > 0.8]) / max(len(confidences), 1) * 100, 1
                ),
            },
            "intents": self._intent_distribution(metrics_list),
        }

    def _intent_distribution(self, metrics_list: list[RequestMetrics]) -> dict:
        """Calculate intent distribution."""
        intents = {}
        for m in metrics_list:
            if m.intent:
                intents[m.intent] = intents.get(m.intent, 0) + 1
        return intents

    def _empty_summary(self) -> dict:
        return {
            "overview": {
                "total_requests": 0,
                "window_size": 0,
                "success_rate": 0,
                "error_rate": 0,
                "uptime_s": round(time.time() - self._start_time, 1),
                "requests_per_minute": 0,
            },
            "latency": {
                "mean_ms": 0, "median_ms": 0, "p95_ms": 0,
                "p99_ms": 0, "min_ms": 0, "max_ms": 0,
            },
            "pipeline_breakdown": {
                "preprocessing_mean_ms": 0,
                "transcription_mean_ms": 0,
                "command_processing_mean_ms": 0,
            },
            "accuracy": {
                "mean_confidence": 0, "median_confidence": 0,
                "high_confidence_rate": 0,
            },
            "intents": {},
        }

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the N most recent request metrics."""
        with self._lock:
            recent = list(self._metrics)[-n:]
        return [
            {
                "request_id": m.request_id,
                "timestamp": m.timestamp,
                "total_latency_ms": m.total_latency_ms,
                "intent": m.intent,
                "confidence": m.confidence,
                "was_successful": m.was_successful,
            }
            for m in recent
        ]
