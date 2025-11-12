import base64
import binascii
import io
import inspect
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union, List
from urllib.parse import urlparse
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS

logger = logging.getLogger(__name__)

# Stream data in 120 ms increments (used by SSE / WS producers).
STREAM_SEGMENT_SECONDS = 0.12

# Small fade time at sentence boundaries to avoid clicks (~5 ms at 24 kHz).
BOUNDARY_FADE_MS = 5.0


@dataclass
class TtsRequest:
    """Structured request parameters consumed by the TtsEngine."""

    text: str
    model_key: str
    voice_path: Optional[str]
    temperature: float
    cfg_weight: float
    exaggeration: float
    speed: float
    sample_rate: int
    language_id: Optional[str]
    seed: Optional[int]
    normalize_text: bool


class TtsEngine:
    """Wrap Chatterbox models and expose the streaming helpers used by the server."""

    def __init__(self, model_key: str = "english", device: str = "cuda"):
        self.model_key = model_key
        self.device = device
        if model_key.lower() == "multilingual":
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        else:
            self.model = ChatterboxTTS.from_pretrained(device=device)

        self.model_sr = int(getattr(self.model, "sr", 24000))

    def has_streaming(self) -> bool:
        """Return True when the selected backend exposes native streaming."""
        return False

    def _generate_once(self, req: TtsRequest, text: str) -> torch.Tensor:
        """Run a single inference pass and return a float tensor at the request sample rate."""
        kwargs = dict(
            text=text,
            temperature=req.temperature,
            cfg_weight=req.cfg_weight,
            exaggeration=req.exaggeration,
            audio_prompt_path=req.voice_path,
        )
        # Feature discovery against model.generate signature
        try:
            generate_params = inspect.signature(self.model.generate).parameters  # type: ignore[attr-defined]
        except (ValueError, AttributeError):
            generate_params = {}

        if "speed" in generate_params:
            kwargs["speed"] = req.speed
        if "normalize_text" in generate_params:
            kwargs["normalize_text"] = req.normalize_text

        if req.language_id and self.model_key == "multilingual":
            kwargs["language_id"] = req.language_id

        with torch.inference_mode():
            with _seed_guard(req.seed, self.device):
                wav = self.model.generate(**kwargs)  # type: ignore[attr-defined]

        audio = _to_float_mono(wav)
        audio = _resample_if_needed(audio, self.model_sr, req.sample_rate)
        audio = _peak_normalize(audio, target_dbfs=-1.0)
        return audio

    def generate_stream_sentences(self, req: TtsRequest) -> Iterator[torch.Tensor]:
        """Yield sentence-sized audio fragments chunked into STREAM_SEGMENT_SECONDS windows."""
        sentences = _split_into_sentences(req.text)
        for idx, sent in enumerate(sentences):
            audio = self._generate_once(req, sent)
            # Apply small fade at the start/end of each sentence to avoid boundary clicks.
            audio = _apply_fade_io(audio, req.sample_rate, ms=BOUNDARY_FADE_MS)

            hop = max(1, int(req.sample_rate * STREAM_SEGMENT_SECONDS))
            # Stream sentence audio in fixed hops.
            for i in range(0, int(audio.numel()), hop):
                yield audio[i: i + hop]

    def generate_full(self, req: TtsRequest) -> torch.Tensor:
        """Synthesize the entire prompt as a single tensor."""
        return self._generate_once(req, req.text)


# --------- Audio helpers --------- #

def _split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter:
    - Keeps ellipses "..." intact
    - Avoids splitting on common abbreviations
    - Preserves terminal punctuation with sentence
    """
    if not text:
        return []

    # Protect ellipses first
    text = text.replace("...", "…")

    # Temporarily mask common abbreviations so the dot isn't treated as sentence end.
    ABBREV_PATTERN = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K|St)\.")
    ABBR_PLACEHOLDER = "__ABB__"

    def _abbr_sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}{ABBR_PLACEHOLDER}"

    masked = ABBREV_PATTERN.sub(_abbr_sub, text.strip())

    # Split on punctuation followed by whitespace.
    parts = re.split(r"(?<=[\.\?\!])\s+", masked)
    cleaned: List[str] = []
    for part in parts:
        restored = part.replace(ABBR_PLACEHOLDER, ".").replace("…", "...").strip()
        if restored:
            cleaned.append(restored)

    parts = cleaned
    return parts


def _apply_fade_io(x: torch.Tensor, sr: int, ms: float = 5.0) -> torch.Tensor:
    """Apply a short fade-in/out to avoid clicks at boundaries."""
    if x.numel() == 0:
        return x
    n = int(max(1, round(sr * (ms / 1000.0))))
    n = min(n, int(x.numel() // 2))
    if n <= 1:
        return x
    y = x.clone()
    ramp = torch.linspace(0.0, 1.0, n, dtype=y.dtype, device=y.device)
    y[:n] = y[:n] * ramp
    y[-n:] = y[-n:] * torch.flip(ramp, dims=[0])
    return y


def _to_float_mono(wav: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert the input waveform to a contiguous float tensor in mono."""
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    if wav.ndim == 2:
        # (channels, samples) or (samples, channels) – standardize then mean
        if wav.shape[0] < wav.shape[1]:
            # assume (channels, samples)
            wav = wav.mean(dim=0)
        else:
            # assume (samples, channels)
            wav = wav.mean(dim=1)
    return wav.float().contiguous()


def _resample_if_needed(x: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Resample the waveform only when the target sample rate differs."""
    if src_sr == dst_sr:
        return x
    x = x.unsqueeze(0)
    y = torchaudio.functional.resample(x, src_sr, dst_sr)
    return y.squeeze(0)


def _peak_normalize(x: torch.Tensor, target_dbfs: float = -1.0) -> torch.Tensor:
    """Normalize peak to target dBFS (default -1 dBFS)."""
    peak = float(torch.max(torch.abs(x)).cpu())
    if peak <= 1e-9:
        return x
    target_lin = 10.0 ** (target_dbfs / 20.0)
    gain = min(1.0, target_lin / peak)
    return (x * gain).contiguous()


def pcm16_bytes(x: Union[torch.Tensor, np.ndarray], sr: int) -> bytes:
    """Encode the waveform as PCM16 bytes."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.clip(x, -1.0, 1.0)
    return (x * (2 ** 15 - 1)).astype(np.int16).tobytes()


def _resolve_cuda_index(device: str) -> Optional[int]:
    """Extract the CUDA device index from a torch-style device string."""
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        _, _, suffix = device.partition(":")
        if suffix.isdigit():
            return int(suffix)
        return None
    return 0


@contextmanager
def _seed_guard(seed: Optional[int], device: str) -> Iterator[None]:
    """Scope RNG state so optional seeding stays isolated from the global context."""
    if seed is None:
        yield
        return
    cuda_index = _resolve_cuda_index(device)
    devices = [cuda_index] if cuda_index is not None and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def write_prompt_to_wav(voice_url: Optional[str], voice_b64: Optional[str], max_bytes: int) -> Optional[str]:
    """Store an uploaded or downloaded prompt as a temporary 16 kHz WAV file."""
    if not voice_url and not voice_b64:
        return None

    payload = _load_prompt_payload(voice_url, voice_b64, max_bytes)
    try:
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    except RuntimeError as exc:
        raise ValueError("Unsupported voice prompt audio format.") from exc

    waveform = _to_float_mono(audio)
    tensor = waveform.unsqueeze(0)
    if sample_rate != 16000:
        tensor = torchaudio.functional.resample(tensor, sample_rate, 16000)

    fd, tmp_path = tempfile.mkstemp(prefix="prompt_", suffix=".wav")
    os.close(fd)
    path = Path(tmp_path)
    try:
        sf.write(path, tensor.squeeze(0).cpu().numpy(), 16000, subtype="PCM_16")
        try:
            path.chmod(0o600)
        except OSError:
            logger.debug("Voice prompt chmod skipped for %s", path, exc_info=True)
        return str(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _load_prompt_payload(voice_url: Optional[str], voice_b64: Optional[str], max_bytes: int) -> bytes:
    """Fetch or decode the prompt payload while enforcing size limits."""
    if voice_url:
        parsed = urlparse(voice_url)
        if parsed.scheme != "https":
            raise ValueError("voice_url must use https.")
        try:
            import requests
            from requests import RequestException

            with requests.get(voice_url, timeout=10, stream=True) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError("Voice prompt exceeds size allowance.")

                data = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError("Voice prompt exceeds size allowance.")
        except RequestException as exc:
            logger.warning("Voice prompt download failed: %s", exc)
            raise ValueError("Failed to download voice prompt.") from exc
        return bytes(data)

    if voice_b64 is None:
        raise ValueError("Voice prompt payload missing.")

    try:
        decoded = base64.b64decode(voice_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("voice_b64 is not valid base64 content.") from exc

    if len(decoded) > max_bytes:
        raise ValueError("Voice prompt exceeds size allowance.")
    return decoded