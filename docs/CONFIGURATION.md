# Configuration Guide

This document explains every configurable knob exposed by the Chatterbox OpenAI-compatible service, along with deployment tips for GPU, voice libraries, and storage.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVICE_PORT` | `4123` | TCP port bound by Uvicorn inside the container or process. |
| `DEVICE` | `cuda` | Torch device string (e.g., `cuda`, `cuda:1`, `cpu`). If you target a single GPU, prefer `DEVICE=cuda:0` and keep `CUDA_VISIBLE_DEVICES` aligned. The default Docker image preserves the Torch 2.9 stack that supports SM_120/Blackwell GPUs. |
| `SAMPLE_RATE` | `24000` | Default sample rate for non-streaming responses. SSE streaming always honors the per-request sample rate. |
| `VOICE_LIBRARY_PATH` | `/app/voices` | Directory recursively scanned for `.wav` or `.flac` prompt files used by the `voice` field. |
| `VOICE_CACHE_PATH` | `/models/hf/voice-cache` | Folder that stores `manifest.json`, tracking file size and mtime of warmed voices. |
| `DEFAULT_VOICE_NAME` | _unset_ | Optional fallback voice identifier when clients omit the `voice` field. |
| `MAX_TEXT_LENGTH` | `1500` | Maximum number of characters accepted in `input`. Requests exceeding this length return HTTP 400. |
| `MAX_PROMPT_BYTES` | `2 * 1024 * 1024` | Upper bound (bytes) for prompt uploads via `voice_url` or `voice_b64`. Oversized payloads are rejected. |
| `HF_HOME` | `/models/hf` (Docker) | Hugging Face cache location used by `chatterbox-tts`. Mount this path to persist weights. |
| `MODEL_ID_MAP` | JSON string | Maps OpenAI model IDs (e.g., `gpt-4o-mini-tts`) to either `english` or `multilingual`. |
| `ENABLE_CORS` | `1` | Disable (`0`) to remove the permissive CORS middleware entirely. |
| `CORS_ORIGINS` | `*` | Comma-separated allowlist consumed by the CORS middleware. |
| `LOG_LEVEL` | `INFO` | Logging level passed to `logging.basicConfig`. |

All variables can be provided through `docker run -e`, a `.env` file, or any process manager that sets environment variables before launching Uvicorn.

## Voice Library Workflow

1. Populate `VOICE_LIBRARY_PATH` with prompt recordings. File requirements:
   - Format: `.wav` or `.flac`.
   - Sampling rate: The server resamples automatically, but providing 16 kHz mono clips keeps preprocessing costs low.
   - Naming: `studio-demo.wav` becomes the case-insensitive ID `studio-demo`.
2. Restart the service (or trigger a reload). On startup the server:
   - Scans the library.
   - Generates a short warmup request per voice (`text="."`) to prime embeddings.
   - Records file metadata to `VOICE_CACHE_PATH/manifest.json`.
3. If you later update a voice file, the warmup logic detects the new `size`/`mtime` and regenerates the cache entry automatically.

When `VOICE_LIBRARY_PATH` is empty or missing, voice cloning is effectively disabled; `/v1/models` still lists the model IDs from `MODEL_ID_MAP`, but clients must rely on zero-shot prompts via `voice_url` or `voice_b64`.

## Prompt Uploads per Request

Clients that cannot rely on a pre-registered voice may pass:

- `voice_url`: HTTPS URL downloaded server-side. The response must fit under `MAX_PROMPT_BYTES` once fully buffered.
- `voice_b64`: Base64-encoded audio payload embedded directly in the JSON request body.

Uploaded prompts are converted to 16 kHz WAV files and stored under the OS temp directory with `0600` permissions. Files are deleted as soon as synthesis finishes (success or failure).

## Model Routing and Multilingual Behavior

- `MODEL_ID_MAP` lets the service expose multiple logical OpenAI model IDs while still routing to a limited set of Chatterbox checkpoints. Provide a JSON string such as `{"gpt-4o-mini-tts":"english", "gpt-4o-mini-tts-multi":"multilingual"}`. Unknown IDs produce HTTP 400 responses.
- Each backend entry currently resolves to one of the shipped checkpoints: `"english"` or `"multilingual"`. Text-normalization and `language_id` hints are only applied when the backend is set to `"multilingual"`.
- To force every request to a single backend, map all published model IDs to either `"english"` or `"multilingual"` in `MODEL_ID_MAP`. Restart (or redeploy) after changing the map so the environment variable is re-read.

## GPU and Multi-GPU Deployments

- Pin to a device using `DEVICE=cuda:0`, `cuda:1`, etc. Helper `_resolve_cuda_index` extracts the index for RNG seeding.
- Combine with `CUDA_VISIBLE_DEVICES` to restrict which GPUs are exposed to the container or process.
- For deterministic debugging, set the `seed` field in the request body. The engine uses `torch.random.fork_rng` with the resolved CUDA device for reproducibility.
- If you need CPU fallbacks, set `DEVICE=cpu`. Expect significantly slower inference.

## Storage and Cache Locations

- **Model weights**: Controlled by `HF_HOME`. In Docker, mount a host path (e.g., `-v $PWD/hf-cache:/models/hf`) to avoid re-downloading weights.
- **Voice cache manifest**: `VOICE_CACHE_PATH/manifest.json`. Safe to delete; it will be re-created on the next start.
- **Temporary prompts**: System temp directory (via `tempfile.mkstemp`). Automatically removed unless the process is terminated abruptly.

## Logging

FastAPI and the module-level loggers submit to the root logging configuration. Set `LOG_LEVEL=DEBUG` (Uvicorn) or configure `logging.basicConfig` before importing `app.server` if you need verbose diagnostics about downloads, warmups, or prompt parsing.

## FFmpeg availability

MP3/AAC/Opus/FLAC encoding paths require an `ffmpeg` binary in `PATH`. On startup the server probes `ffmpeg -version` and sets a module-level guard. If the binary is missing the service logs a warning and any request for those formats returns HTTP 501. The Docker image already contains FFmpeg; install it locally via your package manager when running outside the container.

## CORS

When `ENABLE_CORS=1` the app installs a permissive CORS middleware that reads `CORS_ORIGINS` (default `*`). Disable it when fronting the service with a gateway that already enforces origin policies.
