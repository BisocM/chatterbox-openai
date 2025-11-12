import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Iterator, Literal, Optional, Dict, List, Tuple

import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from app.engine import (
        TtsEngine,
        TtsRequest,
        pcm16_bytes,
        write_prompt_to_wav,
    )
except ImportError:
    from engine import (
        TtsEngine,
        TtsRequest,
        pcm16_bytes,
        write_prompt_to_wav,
    )

logger = logging.getLogger("chatterbox.openai_compat.server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# === Configuration ===
APP_PORT = int(os.environ.get("SERVICE_PORT", "4123"))
DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_SR = int(os.environ.get("SAMPLE_RATE", "24000"))
VOICE_LIBRARY_PATH = Path(os.environ.get("VOICE_LIBRARY_PATH", "/app/voices"))
VOICE_CACHE_PATH = Path(os.environ.get("VOICE_CACHE_PATH", "/models/hf/voice-cache"))
DEFAULT_VOICE_NAME = os.environ.get("DEFAULT_VOICE_NAME")
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "1500"))
MAX_PROMPT_BYTES = int(os.environ.get("MAX_PROMPT_BYTES", str(2 * 1024 * 1024)))

# Model routing: OpenAI model IDs -> { "english" | "multilingual" }
# Can be overridden via env var MODEL_ID_MAP='{"chatterbox-english-tts":"english"}'
DEFAULT_MODEL_ID_MAP = {
    "chatterbox-english-tts": "english",
    "chatterbox-multilingual-tts": "multilingual",
    # Convenience aliases to interop with some SDK defaults:
    "gpt-4o-mini-tts": "english",
}
MODEL_ID_MAP: Dict[str, str] = DEFAULT_MODEL_ID_MAP.copy()
if os.environ.get("MODEL_ID_MAP"):
    try:
        MODEL_ID_MAP.update(json.loads(os.environ["MODEL_ID_MAP"]))
    except Exception as exc:
        logger.warning("Failed to parse MODEL_ID_MAP: %s", exc)

# Engines are heavy; keep a per-backend pool
_ENGINE_POOL: Dict[str, TtsEngine] = {}

AllowedFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]  # Supported response formats.


def get_engine(backend_key: str) -> TtsEngine:
    eng = _ENGINE_POOL.get(backend_key)
    if eng is None:
        logger.info("Loading TTS backend '%s' on device '%s'...", backend_key, DEVICE)
        eng = TtsEngine(model_key=backend_key, device=DEVICE)
        _ENGINE_POOL[backend_key] = eng
    return eng


def resolve_backend(model_id: str) -> str:
    backend = MODEL_ID_MAP.get(model_id)
    if not backend:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'.")
    return backend


# === Voice library & prewarm ===
def _load_voice_library(base_path: Path) -> dict[str, str]:
    """Return a case-insensitive map of voice identifiers to prompt file paths."""
    voices: dict[str, str] = {}
    if not base_path.is_dir():
        logger.warning("Voice library path %s is not a directory; voice cloning disabled.", base_path)
        return voices

    for prompt_file in base_path.rglob("*"):
        if not prompt_file.is_file():
            continue
        if prompt_file.suffix.lower() not in {".wav", ".flac"}:
            continue
        voices[prompt_file.stem.lower()] = str(prompt_file.resolve())

    logger.info("Loaded %d registered voices from %s", len(voices), base_path)
    return voices


VOICE_LIBRARY = _load_voice_library(VOICE_LIBRARY_PATH)
VOICE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
VOICE_CACHE_FILE = VOICE_CACHE_PATH / "manifest.json"


def _load_voice_cache_manifest() -> dict:
    if not VOICE_CACHE_FILE.is_file():
        return {}
    try:
        return json.loads(VOICE_CACHE_FILE.read_text())
    except Exception as exc:
        logger.warning("Failed to read voice cache manifest: %s", exc)
        return {}


def _save_voice_cache_manifest(manifest: dict) -> None:
    try:
        VOICE_CACHE_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    except Exception as exc:
        logger.warning("Failed to write voice cache manifest: %s", exc)


VOICE_CACHE_MANIFEST = _load_voice_cache_manifest()


def _voice_file_metadata(path: str) -> dict:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _prewarm_registered_voices():
    if not VOICE_LIBRARY:
        return
    logger.info("Prewarming %d registered voices...", len(VOICE_LIBRARY))
    manifest_dirty = False
    # Use the default English backend by convention, but respect the first map entry if only multilingual exists.
    default_backend = next(iter(set(MODEL_ID_MAP.values())), "english")
    engine = get_engine(default_backend)

    for voice_name, prompt_path in VOICE_LIBRARY.items():
        metadata = _voice_file_metadata(prompt_path)
        cache_entry = VOICE_CACHE_MANIFEST.get(voice_name)
        if cache_entry and cache_entry.get("size") == metadata["size"] and cache_entry.get("mtime") == metadata[
            "mtime"]:
            continue
        try:
            req = TtsRequest(
                text=".",
                model_key=default_backend,
                voice_path=prompt_path,
                temperature=0.35,
                cfg_weight=0.5,
                exaggeration=0.5,
                speed=1.0,
                sample_rate=DEFAULT_SR,
                language_id=None,
                seed=None,
                normalize_text=True,
            )
            engine.generate_full(req)
            VOICE_CACHE_MANIFEST[voice_name] = metadata
            manifest_dirty = True
        except Exception as exc:
            logger.warning("Failed to prewarm voice '%s': %s", voice_name, exc)
    logger.info("Voice prewarm complete.")
    if manifest_dirty:
        _save_voice_cache_manifest(VOICE_CACHE_MANIFEST)


# === FastAPI app ===
app = FastAPI(title="Chatterbox-OpenAI", version="2.0.0")

# (Optional) CORS allow-all can be narrowed via env if needed
if os.environ.get("ENABLE_CORS", "1") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def _startup():
    # Non-blocking voice prewarm
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _prewarm_registered_voices)


# === Error handling (OpenAI envelope) ===
def _error_envelope(message: str, type_: str = "server_error", code: Optional[str] = None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return JSONResponse(status_code=exc.status_code,
                        content=_error_envelope(detail, "request_error" if exc.status_code < 500 else "server_error"))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content=_error_envelope(str(exc), "validation_error", code="invalid_request"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content=_error_envelope("Internal error.", "server_error"))


# === Content types & format helpers ===
def content_type_for(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",  # ogg-opus stream
        "aac": "audio/aac",  # ADTS stream
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }[fmt]


def can_resample(fmt: str) -> bool:
    """Return True when the format honors arbitrary caller-provided sample rates."""
    return fmt in ("wav", "pcm")


def choose_encoder_sr(fmt: str, requested: Optional[int], fallback: int) -> int:
    """Snap requested sample rates to encoder-friendly values per format."""
    if fmt in ("wav", "pcm"):
        return int(requested or fallback)
    # Snap to sane encoder rates
    if fmt == "opus":
        return 48000  # Opus native
    if fmt in ("mp3", "aac", "flac"):
        # Prefer 44100/48000; keep 24000 if model default and allowed
        if requested in (44100, 48000, 32000, 24000, 22050, 16000):
            return int(requested)
        return 44100
    return fallback


# === Request Models ===
class OpenAISpeechIn(BaseModel):
    """Pydantic model for the OpenAI-compatible /v1/audio/speech payload."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., min_length=1)
    input: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: Optional[str] = Field(default=None)  # allow default voice selection
    response_format: AllowedFormat = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=3.0)
    stream: bool = Field(default=False, description="If true, stream audio. PCM via SSE; compressed via chunked body.")

    sample_rate: Optional[int] = Field(default=None, ge=8000, le=192000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    cfg_weight: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    exaggeration: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    language_id: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0)
    voice_url: Optional[str] = None
    voice_b64: Optional[str] = None
    normalize_text: Optional[bool] = True

    @property
    def format(self) -> AllowedFormat:
        return self.response_format

    @model_validator(mode="after")
    def _validate(self):
        if self.voice_url and self.voice_b64:
            raise ValueError("Provide voice prompt as URL or base64, not both.")
        if self.voice_url and not self.voice_url.lower().startswith("https://"):
            raise ValueError("voice_url must use https.")
        if self.sample_rate is not None and not can_resample(self.response_format):
            # non-PCM/WAV will be quantized internally; we still allow sample_rate but we snap it.
            pass
        return self


# Minimal Responses API subset for audio generation
class ResponsesAudioConfig(BaseModel):
    voice: Optional[str] = None
    format: AllowedFormat = "pcm"
    sample_rate: Optional[int] = None


class ResponsesInput(BaseModel):
    # minimal subset: plain text input
    type: Literal["input_text"] = "input_text"
    text: str


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: List[ResponsesInput]
    audio: Optional[ResponsesAudioConfig] = None

    # Optional tuning knobs
    temperature: Optional[float] = None
    cfg_weight: Optional[float] = None
    exaggeration: Optional[float] = None
    speed: Optional[float] = 1.0
    language_id: Optional[str] = None
    seed: Optional[int] = None
    normalize_text: Optional[bool] = True


# === Utility: voices ===
def _resolve_registered_voice(name: Optional[str]) -> Optional[str]:
    candidate = name or DEFAULT_VOICE_NAME
    if not candidate:
        return None
    if not VOICE_LIBRARY:
        logger.warning("No registered voices available; ignoring requested voice '%s'.", candidate)
        return None
    path = VOICE_LIBRARY.get(candidate.lower())
    if path is None:
        raise HTTPException(404, f"Voice '{candidate}' is not present in the voice library.")
    return path


# === Encoding helpers ===

def _check_ffmpeg_available() -> bool:
    """Return True when an ffmpeg binary is discoverable on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False


FFMPEG_AVAILABLE = _check_ffmpeg_available()


def _encode_full_wav(audio: torch.Tensor, sr: int) -> bytes:
    """Encode a waveform as a WAV file in-memory."""
    import soundfile as sf  # local import to keep import-time light
    buf = io.BytesIO()
    sf.write(buf, audio.detach().cpu().numpy(), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _encode_full_flac(audio: torch.Tensor, sr: int) -> bytes:
    """Encode a waveform as a FLAC file in-memory."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio.detach().cpu().numpy(), sr, format="FLAC")
    return buf.getvalue()


def _encode_full_via_ffmpeg(audio: torch.Tensor, sr: int, fmt: str, bitrate: Optional[str] = None) -> bytes:
    """Encode MP3/AAC/Opus outputs by piping PCM through ffmpeg."""
    if not FFMPEG_AVAILABLE:
        raise HTTPException(501, f"FFmpeg is not available; cannot encode '{fmt}'.")
    pcm = pcm16_bytes(audio, sr)
    args: List[str]
    if fmt == "mp3":
        args = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                "-f", "mp3"]
        if bitrate:
            args += ["-b:a", bitrate]
        args += ["-ar", str(sr), "pipe:1"]
    elif fmt == "opus":
        # encode to Ogg/Opus
        out_sr = 48000
        args = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                "-c:a", "libopus", "-b:a", bitrate or "48k", "-application", "audio",
                "-ar", str(out_sr), "-f", "ogg", "pipe:1"]
    elif fmt == "aac":
        # ADTS stream
        args = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                "-c:a", "aac", "-b:a", bitrate or "128k", "-f", "adts", "pipe:1"]
    else:
        raise HTTPException(400, f"Unsupported encode format '{fmt}'.")

    try:
        proc = subprocess.run(args, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return proc.stdout
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg encode failed: %s", e.stderr.decode("utf-8", errors="ignore"))
        raise HTTPException(500, "Audio encoding failed.") from e


def _encode_stream_via_ffmpeg(fmt: str, in_sr: int, chunk_iter: Iterator[bytes]) -> Iterator[bytes]:
    """Pipe PCM16 chunks into ffmpeg and yield encoded bytes as they are produced."""
    if not FFMPEG_AVAILABLE:
        raise HTTPException(501, f"FFmpeg is not available; cannot stream-encode '{fmt}'.")

    if fmt == "mp3":
        cmd = ["ffmpeg", "-f", "s16le", "-ar", str(in_sr), "-ac", "1", "-i", "pipe:0",
               "-f", "mp3", "-b:a", "192k", "-ar", str(in_sr), "pipe:1"]
    elif fmt == "opus":
        cmd = ["ffmpeg", "-f", "s16le", "-ar", str(in_sr), "-ac", "1", "-i", "pipe:0",
               "-c:a", "libopus", "-b:a", "48k", "-application", "audio", "-ar", "48000",
               "-f", "ogg", "pipe:1"]
    elif fmt == "aac":
        cmd = ["ffmpeg", "-f", "s16le", "-ar", str(in_sr), "-ac", "1", "-i", "pipe:0",
               "-c:a", "aac", "-b:a", "128k", "-f", "adts", "pipe:1"]
    elif fmt == "flac":
        cmd = ["ffmpeg", "-f", "s16le", "-ar", str(in_sr), "-ac", "1", "-i", "pipe:0",
               "-c:a", "flac", "-f", "flac", "pipe:1"]
    else:
        raise HTTPException(400, f"Unsupported streaming format '{fmt}'.")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Writer thread to feed PCM into ffmpeg stdin
    def _writer():
        try:
            for chunk in chunk_iter:
                if not chunk:
                    continue
                try:
                    proc.stdin.write(chunk)  # type: ignore[union-attr]
                except BrokenPipeError:
                    break
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass

    t = threading.Thread(target=_writer, daemon=True)
    t.start()

    try:
        while True:
            data = proc.stdout.read(4096)  # type: ignore[union-attr]
            if not data:
                break
            yield data
    finally:
        t.join(timeout=2.0)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


# === Health & models ===
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    """Return the available TTS model IDs (not voices)."""
    data = []
    for model_id in sorted(MODEL_ID_MAP.keys()):
        data.append(
            {
                "id": model_id,
                "object": "model",
                "owned_by": "chatterbox-openai",
            }
        )
    return {"object": "list", "data": data}


# === Internals shared by endpoints ===
def _make_tts_request(
        backend_key: str,
        text: str,
        sr: int,
        voice_path: Optional[str],
        temperature: Optional[float],
        cfg_weight: Optional[float],
        exaggeration: Optional[float],
        speed: Optional[float],
        language_id: Optional[str],
        seed: Optional[int],
        normalize_text: Optional[bool],
) -> TtsRequest:
    """Build a normalized TtsRequest shared across endpoints."""
    return TtsRequest(
        text=text,
        model_key=backend_key,
        voice_path=voice_path,
        temperature=temperature if temperature is not None else 0.35,
        cfg_weight=cfg_weight if cfg_weight is not None else 0.5,
        exaggeration=exaggeration if exaggeration is not None else 0.5,
        speed=speed if speed is not None else 1.0,
        sample_rate=sr,
        language_id=language_id,
        seed=seed,
        normalize_text=bool(normalize_text),
    )


def _iter_pcm_segments(engine: TtsEngine, req: TtsRequest) -> Iterator[bytes]:
    """Yield PCM16-encoded chunks for either SSE or ffmpeg pipe."""
    for seg in engine.generate_stream_sentences(req):
        yield pcm16_bytes(seg, req.sample_rate)


def _encode_audio_full(engine: TtsEngine, req: TtsRequest, fmt: AllowedFormat) -> Tuple[bytes, int]:
    """Return a fully encoded audio payload plus the realized sample rate."""
    audio = engine.generate_full(req)
    sr = req.sample_rate
    if fmt == "wav":
        return _encode_full_wav(audio, sr), sr
    if fmt == "pcm":
        return pcm16_bytes(audio, sr), sr
    if fmt == "flac":
        return _encode_full_flac(audio, sr), sr
    if fmt in ("mp3", "aac", "opus"):
        return _encode_full_via_ffmpeg(audio, sr, fmt), sr
    raise HTTPException(400, f"response_format '{fmt}' not supported.")


def _encode_stream_body(fmt: AllowedFormat, sr: int, chunk_iter: Iterator[bytes]) -> Iterator[bytes]:
    """Yield encoded bytes for streaming responses, delegating to ffmpeg when necessary."""
    if fmt == "pcm":
        for ch in chunk_iter:
            yield ch
        return
    # stream-encode compressed via ffmpeg
    for data in _encode_stream_via_ffmpeg(fmt, sr, chunk_iter):
        yield data


# === /v1/audio/speech ===
@app.post("/v1/audio/speech")
def audio_speech(body: OpenAISpeechIn, request: Request):
    """Synthesize speech according to the OpenAI-compatible contract.

    - Non-streaming: returns full audio payload (all formats supported).
    - Streaming: if response_format == "pcm" => SSE frames with base64 PCM16;
                 else chunked HTTP body of encoded audio (mp3/opus/aac/flac).
    """
    text = body.input.strip()
    if not text:
        raise HTTPException(400, "input must contain text.")

    backend_key = resolve_backend(body.model)
    engine = get_engine(backend_key)

    # Resolve voice prompt (registered or uploaded)
    voice_prompt_path: Optional[str] = None
    temp_prompt_path: Optional[str] = None
    if body.voice_url or body.voice_b64:
        try:
            voice_prompt_path = write_prompt_to_wav(body.voice_url, body.voice_b64, max_bytes=MAX_PROMPT_BYTES)
            temp_prompt_path = voice_prompt_path
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        voice_prompt_path = _resolve_registered_voice(body.voice)

    out_sr = choose_encoder_sr(body.response_format, body.sample_rate, DEFAULT_SR)

    req = _make_tts_request(
        backend_key=backend_key,
        text=text,
        sr=out_sr,
        voice_path=voice_prompt_path,
        temperature=body.temperature,
        cfg_weight=body.cfg_weight,
        exaggeration=body.exaggeration,
        speed=body.speed,
        language_id=body.language_id,
        seed=body.seed,
        normalize_text=body.normalize_text,
    )

    # Decide streaming mode
    wants_stream = body.stream or request.headers.get("accept", "").lower() == "text/event-stream"

    if wants_stream and body.response_format == "pcm":
        async def sse_wrapper():
            chunk_index = 0
            try:
                for chunk in _iter_pcm_segments(engine, req):
                    payload = {
                        "type": "response.output_audio.delta",
                        "index": chunk_index,
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "format": "pcm16",
                        "sample_rate": out_sr,
                    }
                    chunk_index += 1
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    await asyncio.sleep(0)
                final_payload = {"type": "response.completed", "chunks": chunk_index, "sample_rate": out_sr}
                yield f"data: {json.dumps(final_payload, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if temp_prompt_path:
                    try:
                        os.unlink(temp_prompt_path)
                    except OSError:
                        pass

        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

    if wants_stream and body.response_format != "wav":
        # Chunked compressed body stream (mp3/opus/aac/flac)
        def stream_body():
            try:
                for data in _encode_stream_body(body.response_format, out_sr, _iter_pcm_segments(engine, req)):
                    yield data
            finally:
                if temp_prompt_path:
                    try:
                        os.unlink(temp_prompt_path)
                    except OSError:
                        pass

        return StreamingResponse(stream_body(), media_type=content_type_for(body.response_format),
                                 headers={"x-sample-rate": str(out_sr)})

    # Synchronous (full) path
    try:
        payload, sr = _encode_audio_full(engine, req, body.response_format)
        return Response(payload, media_type=content_type_for(body.response_format), headers={"x-sample-rate": str(sr)})
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(500, "Synthesis failed.") from exc
    finally:
        if temp_prompt_path:
            try:
                os.unlink(temp_prompt_path)
            except OSError:
                pass


# === /v1/responses (minimal audio subset) ===
@app.post("/v1/responses")
async def responses(request: Request):
    """
    Minimal Responses API compatibility for audio-only generation.

    - If Accept: text/event-stream => emits SSE frames:
        * response.output_audio.delta { audio: base64(PCM16), sample_rate }
        * response.completed
    - Otherwise returns JSON with a single base64-encoded audio blob.
    """
    try:
        body = ResponsesRequest.model_validate_json(await request.body())
    except Exception as exc:
        raise HTTPException(400, f"Invalid request body: {exc}")

    text = " ".join([it.text for it in body.input if it.type == "input_text"]).strip()
    if not text:
        raise HTTPException(400, "input is empty.")

    backend_key = resolve_backend(body.model)
    engine = get_engine(backend_key)

    audio_cfg = body.audio or ResponsesAudioConfig()
    fmt: AllowedFormat = audio_cfg.format  # type: ignore[assignment]
    sr = choose_encoder_sr(fmt, audio_cfg.sample_rate, DEFAULT_SR)

    voice_path = _resolve_registered_voice(audio_cfg.voice)

    req = _make_tts_request(
        backend_key=backend_key,
        text=text,
        sr=sr,
        voice_path=voice_path,
        temperature=body.temperature,
        cfg_weight=body.cfg_weight,
        exaggeration=body.exaggeration,
        speed=body.speed,
        language_id=body.language_id,
        seed=body.seed,
        normalize_text=body.normalize_text,
    )

    accepts_sse = request.headers.get("accept", "").lower() == "text/event-stream"
    if accepts_sse:
        async def sse_wrapper():
            chunk_index = 0
            for chunk in _iter_pcm_segments(engine, req):
                payload = {
                    "type": "response.output_audio.delta",
                    "index": chunk_index,
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "format": "pcm16",
                    "sample_rate": sr,
                }
                chunk_index += 1
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                await asyncio.sleep(0)
            final_payload = {"type": "response.completed", "chunks": chunk_index, "sample_rate": sr}
            yield f"data: {json.dumps(final_payload, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

    # Non-stream: return a compact response with a single audio blob
    try:
        audio, out_sr = _encode_audio_full(engine, req, fmt)
        resp = {
            "id": "resp_" + os.urandom(6).hex(),
            "object": "response",
            "type": "response",
            "output": [
                {
                    "type": "output_audio",
                    "audio": {
                        "format": fmt if fmt != "pcm" else "pcm16",
                        "sample_rate": out_sr,
                        "data": base64.b64encode(audio).decode("ascii"),
                    },
                }
            ],
            "model": body.model,
        }
        return JSONResponse(resp)
    except Exception as exc:
        logger.exception("Responses synthesis failed")
        raise HTTPException(500, "Synthesis failed.") from exc


# === Optional: Realtime WebSocket (PCM16 only, minimal) ===
@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket):
    """
    Minimal realtime WS:
    1) Client sends a single JSON config message:
       {
         "model": "<model_id>", "text": "...",
         "audio": {"voice": "alice", "sample_rate": 24000}
       }
    2) Server streams binary PCM16 frames (each frame prefixed with 4-byte length, big-endian)
       and finally sends a JSON {"type":"response.completed"} text frame.
    """
    await ws.accept()
    try:
        # First message must be text JSON with config
        init_msg = await ws.receive_text()
        try:
            payload = json.loads(init_msg)
        except Exception:
            await ws.close(code=1003)
            return

        model_id = payload.get("model")
        if not model_id:
            await ws.send_text(json.dumps(_error_envelope("Missing 'model'", "validation_error")))
            await ws.close(code=1003)
            return

        text = payload.get("text") or ""
        if not text.strip():
            await ws.send_text(json.dumps(_error_envelope("Missing 'text'", "validation_error")))
            await ws.close(code=1003)
            return

        audio_cfg = payload.get("audio") or {}
        voice = audio_cfg.get("voice")
        sr = int(audio_cfg.get("sample_rate") or DEFAULT_SR)

        backend_key = resolve_backend(model_id)
        engine = get_engine(backend_key)

        req = _make_tts_request(
            backend_key=backend_key,
            text=text.strip(),
            sr=sr,
            voice_path=_resolve_registered_voice(voice),
            temperature=payload.get("temperature"),
            cfg_weight=payload.get("cfg_weight"),
            exaggeration=payload.get("exaggeration"),
            speed=payload.get("speed", 1.0),
            language_id=payload.get("language_id"),
            seed=payload.get("seed"),
            normalize_text=payload.get("normalize_text", True),
        )

        # Stream PCM16 frames as binary messages with a length prefix
        for chunk in _iter_pcm_segments(engine, req):
            # 4-byte length prefix (big endian) + data
            length_prefix = len(chunk).to_bytes(4, "big")
            await ws.send_bytes(length_prefix + chunk)
            await asyncio.sleep(0)

        await ws.send_text(json.dumps({"type": "response.completed", "sample_rate": sr}))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("Realtime error")
        try:
            await ws.send_text(json.dumps(_error_envelope("Internal error in realtime.", "server_error")))
        finally:
            await ws.close(code=1011)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=APP_PORT, log_level="info", reload=False)
