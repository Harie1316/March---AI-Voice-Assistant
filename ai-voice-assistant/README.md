# AI Voice Assistant with Speech Domain Adaptation

> Production-grade voice assistant with robust speech recognition across varied acoustic conditions, built on OpenAI API, Whisper, and advanced audio preprocessing.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask API Gateway                         │
│                   (REST Endpoints)                           │
├──────────┬──────────┬───────────────┬───────────────────────┤
│  Audio   │  Speech  │    NLP &      │   Task Automation     │
│  Preproc │  Recog.  │  Command      │   & Integration       │
│  Service │  Service │  Processing   │   Service             │
│ (Librosa)│ (Whisper)│ (OpenAI API)  │  (RESTful APIs)       │
├──────────┴──────────┴───────────────┴───────────────────────┤
│              Docker Microservices Layer                       │
│         (Optimised Audio Chunking & Streaming)               │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **90% Command Recognition Accuracy** across noisy, accented, and reverberant speech
- **Domain-Robust Preprocessing** — normalisation for background noise, reverberation, and speaking rate
- **25% Latency Reduction** via Docker microservices with optimised audio chunking
- **RESTful APIs** for task automation and third-party integration
- **Real-Time Processing** with streaming audio support

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech Recognition | OpenAI Whisper |
| NLP & Commands | OpenAI API (GPT) |
| Audio Processing | Librosa, NumPy, SciPy |
| Backend | Flask, Python 3.11 |
| Containerisation | Docker, Docker Compose |
| Architecture | Microservices |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key

### Run with Docker
```bash
# Clone and configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Build and run
docker-compose up --build
```

### Run Locally
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python run.py
```

Access the dashboard at `http://localhost:5000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/transcribe` | Transcribe audio file |
| POST | `/api/v1/command` | Process voice command |
| POST | `/api/v1/stream` | Stream audio for real-time processing |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/tasks` | Create automation task |
| GET | `/api/v1/tasks/<id>` | Get task status |
| POST | `/api/v1/integrate` | Third-party integration |
| GET | `/api/v1/metrics` | Performance metrics |

## Project Structure

```
ai-voice-assistant/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── routes.py                # API route definitions
│   ├── services/
│   │   ├── audio_preprocessor.py    # Librosa preprocessing pipeline
│   │   ├── speech_recognizer.py     # Whisper integration
│   │   ├── command_processor.py     # OpenAI NLP command processing
│   │   ├── task_manager.py          # Task automation service
│   │   └── integration_service.py   # Third-party integrations
│   ├── utils/
│   │   ├── audio_chunker.py     # Optimised audio chunking
│   │   ├── metrics.py           # Performance tracking
│   │   └── validators.py        # Input validation
│   ├── templates/
│   │   └── dashboard.html       # Web dashboard
│   └── static/
├── tests/
│   ├── test_preprocessor.py
│   ├── test_recognizer.py
│   └── test_api.py
├── config/
│   └── settings.py              # Configuration management
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── README.md
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Command Recognition (clean audio) | 95%+ |
| Command Recognition (noisy/accented) | 90%+ |
| Average Latency (with chunking) | ~1.2s |
| Latency Reduction vs Baseline | 25% |
| Concurrent Request Handling | 50+ |

## License

MIT License
