![Chatterbox Badge](docs/assets/chatterbox-openai-badge.svg)

# Chatterbox OpenAI-Compatible TTS

Chatterbox OpenAI-Compatible TTS exposes a FastAPI service that mirrors the OpenAI `/v1/audio/speech` contract while running the open-source [chatterbox-tts](https://pypi.org/project/chatterbox-tts/) models locally. It is designed for teams that need a drop-in, GPU-accelerated speech endpoint they can self-host, customize with cloned voices, and stream over Server-Sent Events (SSE).

The repository bundles a lightweight inference engine (`app/engine.py`), an OpenAI-style API surface (`app/server.py`), and an NVIDIA-ready Docker image for production deployment.

## Highlights

- **OpenAI-compatible endpoints** – `/v1/audio/speech`, `/v1/responses`, `/v1/realtime`, `/v1/models`, and `/healthz`.
- **Streaming or buffered output** – SSE with PCM16 plus chunked MP3/Opus/AAC/FLAC streams or JSON/base64 payloads for full responses.
- **Voice cloning** – Serve registered prompts from disk or accept per-request `voice_url` / `voice_b64`.
- **Model routing** – Map OpenAI model IDs (e.g., `gpt-4o-mini-tts`) to English or multilingual checkpoints via `MODEL_ID_MAP`.
- **GPU first** – Ships with an NVIDIA NGC PyTorch base image; CPU execution is also possible for experimentation.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/engine.py` | Wraps chatterbox TTS models, handles resampling, seeding, and streaming segmentation. |
| `app/server.py` | FastAPI application exposing OpenAI-compatible endpoints and request validation. |
| `Dockerfile` | Production-ready CUDA image with preinstalled dependencies and health checks. |
| `requirements.txt` | Minimal Python dependencies for local development or CPU inference. |
| `docs/assets/` | SVG wordmark, badge, and icon used across the documentation set. |

## Requirements

- Python 3.10+ with CUDA-capable PyTorch (when running outside Docker) **or** Docker with an NVIDIA runtime.
- NVIDIA GPU with at least 8 GB VRAM for real-time inference (multilingual models benefit from 16 GB+). The provided Docker image maintains the NGC Torch 2.9 stack with SM_120 (Blackwell) support; avoid reinstalling torch/torchaudio unless you specifically target older GPUs.
- `libsndfile` (already installed in the Docker image; install via your package manager for local runs).
- `ffmpeg` 5.x+ (included in the Docker image) when serving MP3/AAC/Opus/FLAC outputs; the server degrades to WAV/PCM if the binary is missing.
- Outbound HTTPS access if you plan to fetch voice prompts via `voice_url`.

## Quickstart

### 1. Local Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEVICE=cuda           # or "cpu" for debugging
python -m uvicorn app.server:app --host 0.0.0.0 --port 4123 --reload
```

This uses the PyPI build of `chatterbox-tts` (CUDA wheels). If you already have a tailored PyTorch stack, install it **before** `pip install -r requirements.txt` and ensure `torch`/`torchaudio` are importable.
Set `MODEL_ID_MAP` when you need to rename or remove model IDs (for example, `export MODEL_ID_MAP='{"gpt-4o-mini-tts":"multilingual"}'` to force all traffic to the multilingual checkpoint).

### 2. Docker (recommended for production)

```bash
docker build -t chatterbox-openai .
docker run \
  --gpus all \
  -p 4123:4123 \
  -e DEVICE=cuda \
  -v $PWD/voices:/app/voices \
  chatterbox-openai
```

The image is based on `nvcr.io/nvidia/pytorch:25.09-py3` and includes FFmpeg, libsndfile, health checks, and guarded CUDA memory settings. Mount a persistent volume (e.g., `/models/hf`) if you want to retain downloaded model weights between container restarts.

## Configuration

Key environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `SERVICE_PORT` | `4123` | Port served by FastAPI/Uvicorn. |
| `DEVICE` | `cuda` | Torch device string (e.g., `cuda`, `cuda:1`, `cpu`). |
| `SAMPLE_RATE` | `24000` | Default output sample rate (Hz) for non-streaming requests. |
| `VOICE_LIBRARY_PATH` | `/app/voices` | Root directory scanned for `.wav`/`.flac` prompt files used by `voice`. |
| `VOICE_CACHE_PATH` | `/models/hf/voice-cache` | Location of the `manifest.json` voice warmup cache. |
| `DEFAULT_VOICE_NAME` | _unset_ | Optional voice ID to use when the client omits `voice`. |
| `MAX_TEXT_LENGTH` | `1500` | Maximum characters accepted by `/v1/audio/speech`. |
| `MAX_PROMPT_BYTES` | `2097152` | Safety limit for uploaded voice prompts (bytes). |
| `MODEL_ID_MAP` | see code | JSON string mapping OpenAI model IDs to `english` or `multilingual`. |
| `ENABLE_CORS` | `1` | Set to `0` to disable the permissive CORS middleware entirely. |
| `CORS_ORIGINS` | `*` | Comma-separated origins when `ENABLE_CORS=1`. |
| `LOG_LEVEL` | `INFO` | Uvicorn/logger level (e.g., `DEBUG`). |

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for extended explanations, voice library guidance, and tips for multi-GPU deployments.

## API Usage

The service mirrors OpenAI’s schema for `/v1/audio/speech` and supports the following `response_format` values:

- `wav` – buffered response, returned as `audio/wav`.
- `pcm` – SSE stream of base64 PCM16 chunks (`Accept: text/event-stream`) or chunked binary PCM16 when `stream: true`.
- `mp3`, `opus`, `aac`, `flac` – buffered responses or chunked streaming bodies (no SSE) when `stream: true`. Requires FFmpeg.

Use the `stream` request flag to opt into streaming without relying on `Accept: text/event-stream`. PCM streaming respects either trigger; compressed formats always use chunked HTTP bodies.

### Example: buffered WAV response

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o hello.wav \
  -d '{
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "response_format": "wav",
        "input": "Hello from a self-hosted Chatterbox deployment!"
      }'
```

### Example: streaming PCM via SSE

```bash
curl -N http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini-tts",
        "voice": "studio-demo",
        "response_format": "pcm",
        "stream": true,
        "sample_rate": 16000,
        "voice_url": "https://example.com/prompts/demo.wav",
        "input": "Streaming responses are emitted as base64 PCM16 chunks."
      }'
```

Each SSE event carries:

```json
{
  "type": "response.output_audio.delta",
  "index": 0,
  "audio": "<base64 pcm16>",
  "format": "pcm16",
  "sample_rate": 16000
}
```

The stream concludes with a `response.completed` event followed by `data: [DONE]`.

### Example: chunked MP3 stream

```bash
curl --no-buffer http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini-tts",
        "voice": "studio-demo",
        "response_format": "mp3",
        "stream": true,
        "input": "Chunked MP3 streaming enables progressive playback without SSE."
      }' \
  -o stream.mp3
```

The response is delivered as a chunked `audio/mpeg` body; most HTTP clients (including `curl`) can write it directly to disk while data is still arriving.

### OpenAI Responses API compatibility

`POST /v1/responses` accepts a subset of the official Responses schema (text-only inputs with audio outputs). It emits SSE events when `Accept: text/event-stream` is present and otherwise returns a JSON envelope containing a single base64 audio payload. See [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md#post-v1responses) for the exact schema.

### Realtime WebSocket

`GET /v1/realtime` exposes a minimal PCM16 WebSocket. Send one JSON message describing the model, text, and optional audio config; the server streams binary frames containing length-prefixed PCM16 buffers followed by a JSON completion marker. This endpoint is intended for low-latency microphone integrations or preview players.

Refer to [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) for the complete schema, validation rules, and example responses for every endpoint.

## Voice Library & Prompt Uploads

- Drop `.wav` or `.flac` prompt files under `VOICE_LIBRARY_PATH`. Filenames (minus extension) become case-insensitive `voice` identifiers (e.g., `voices/alloy.wav` ⇒ `voice: alloy`).
- At startup the server prewarms each registered voice and saves metadata into `VOICE_CACHE_PATH/manifest.json`. This avoids regenerating speaker embeddings unless the source file changes.
- Clients can override registered voices on a per-request basis via:
  - `voice_url` – HTTPS URL fetched server-side (size-checked).
  - `voice_b64` – Base64-encoded binary payload sent inline.
- Temporary prompt files are removed after synthesis completes.

## Development Tips

- Enable structured logs by setting `UVICORN_LOG_LEVEL=debug` and `LOG_LEVEL=DEBUG`.
- Use `SEED=<int>` when you need deterministic generations for regression tests.
- Keep text inputs below `MAX_TEXT_LENGTH` (default 1500 characters) to avoid HTTP 400 errors.
- The sentence-streaming helper uses 120 ms segments; adjust `STREAM_SEGMENT_SECONDS` in `app/engine.py` if you need different latency/throughput trade-offs.

## Troubleshooting

- **Model download stalls** – ensure `HF_HOME` points to writable storage (default `/models/hf` in Docker). Mount that path as a persistent volume to prevent redownloads.
- **CUDA OOM** – lower `MAX_PROMPT_BYTES`, reduce `cfg_weight`/`temperature`, or map the requested model IDs to the lighter English backend via `MODEL_ID_MAP`.
- **Voice not found** – confirm the voice filename matches the requested `voice` (case-insensitive) and that the container can see the mounted library.

## Additional Documentation

- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) – Environment variables, voice library management, and hardware notes.
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) – Endpoint-by-endpoint contract and example payloads.

Contributions are welcome via pull requests or issues. Please include reproduction details and logs when reporting synthesis failures.
