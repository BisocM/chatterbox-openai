# API Reference

The service mirrors OpenAI’s audio endpoints while running Chatterbox models locally. All endpoints are served over HTTP/WebSocket and do not require authentication unless you place the service behind your own gateway.

- **Base URL:** `http://{host}:{SERVICE_PORT}`
- **Content types:** JSON for requests, `audio/*` for buffered responses, `text/event-stream` for SSE, and binary frames for realtime WebSockets.

## `GET /healthz`

Lightweight readiness probe.

- **Response 200**

```json
{ "status": "ok" }
```

Use this endpoint for container health checks or load balancer probes.

## `GET /v1/models`

Lists the OpenAI-style model IDs that the server is willing to serve. Each ID maps to either the English or multilingual Chatterbox checkpoint via `MODEL_ID_MAP`.

### Response Example

```json
{
  "object": "list",
  "data": [
    { "id": "chatterbox-english-tts", "object": "model", "owned_by": "chatterbox-openai" },
    { "id": "chatterbox-multilingual-tts", "object": "model", "owned_by": "chatterbox-openai" },
    { "id": "gpt-4o-mini-tts", "object": "model", "owned_by": "chatterbox-openai" }
  ]
}
```

## `POST /v1/audio/speech`

Synthesizes speech from text. Supports buffered or streaming delivery and the response formats `wav`, `pcm`, `mp3`, `opus`, `aac`, and `flac`.

### Request Body

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `model` | `string` | ✅ | — | Must exist in `MODEL_ID_MAP`; determines whether the English or multilingual backend is used. |
| `input` | `string` | ✅ | — | Text to speak (trimmed). Limited to `MAX_TEXT_LENGTH` characters. |
| `voice` | `string` | ❌ | `DEFAULT_VOICE_NAME` | Case-insensitive ID from `VOICE_LIBRARY_PATH`. Leave unset for default or zero-shot prompts. |
| `response_format` | enum | ❌ | `"wav"` | One of `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac`. Non-WAV/PCM formats require an FFmpeg binary on PATH. |
| `speed` | `float` | ❌ | `1.0` | Range 0.25–3.0 when the model exposes the knob. |
| `stream` | `bool` | ❌ | `false` | Forces streaming even if `Accept` is not `text/event-stream`. PCM uses SSE; other formats stream as chunked binary bodies. |
| `sample_rate` | `int` | ❌ | `SAMPLE_RATE` | 8 kHz–192 kHz. WAV/PCM honor the value; compressed formats snap to encoder-friendly rates (e.g., 48 kHz for Opus). |
| `temperature` | `float` | ❌ | `0.35` | 0.0–2.0. |
| `cfg_weight` | `float` | ❌ | `0.5` | 0.0–4.0. |
| `exaggeration` | `float` | ❌ | `0.5` | 0.0–4.0. |
| `language_id` | `string` | ❌ | `null` | Only effective when the backend is `multilingual`. |
| `seed` | `int` | ❌ | `null` | Enables deterministic generations. |
| `voice_url` | `string (https)` | ❌ | `null` | Downloaded prompt audio (mutually exclusive with `voice_b64`). Max size `MAX_PROMPT_BYTES`. |
| `voice_b64` | `string` | ❌ | `null` | Base64-encoded prompt audio (mutually exclusive with `voice_url`). |
| `normalize_text` | `bool` | ❌ | `true` | Forwarded to the model when available. |

Validation rules:

- `voice_url` and `voice_b64` cannot both be set.
- `voice_url` must use HTTPS.
- Empty or whitespace-only `input` yields HTTP 400.

### Streaming Modes

- **PCM streaming** – set `response_format: "pcm"` and either `stream: true` or `Accept: text/event-stream`. The server emits SSE frames with base64 PCM16 payloads plus a terminal `[DONE]` marker.
- **Compressed streaming** – set `stream: true` with `response_format` in `{mp3, opus, aac, flac}`. Responses are chunked binary bodies encoded by FFmpeg (`Content-Type` reflects the format). SSE is not used in this mode.
- **Buffered responses** – omit `stream` (or set `false`). The response body is the full audio file encoded in the requested format with an `x-sample-rate` header.

### SSE Event Payload

```json
{
  "type": "response.output_audio.delta",
  "index": 0,
  "audio": "<base64 pcm16>",
  "format": "pcm16",
  "sample_rate": 16000
}
```

The stream concludes with:

```json
{
  "type": "response.completed",
  "chunks": <int>,
  "sample_rate": <int>
}
```

followed by `data: [DONE]`.

### Chunked MP3 Response Example

```bash
curl --no-buffer http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini-tts",
        "voice": "amber",
        "response_format": "mp3",
        "stream": true,
        "input": "Chunked MP3 lets clients begin playback immediately."
      }' \
  -o stream.mp3
```

### Error Codes

| Status | When it occurs |
| --- | --- |
| `400 Bad Request` | Validation errors (empty text, unknown model ID, invalid `response_format`, oversized prompt, etc.). |
| `404 Not Found` | Requested `voice` does not exist in the voice library. |
| `500 Internal Server Error` | Model inference or encoding failed unexpectedly. |
| `501 Not Implemented` | Requested compressed format but FFmpeg was not detected on the host. |

## `POST /v1/responses`

Minimal compatibility shim for the OpenAI Responses API. Only audio output is implemented; the request body accepts one or more `input_text` entries which are concatenated in order.

### Request Body

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | `string` | ✅ | Must exist in `MODEL_ID_MAP`. |
| `input` | array | ✅ | Each element must be `{ "type": "input_text", "text": "..." }`. Text values are concatenated with spaces. |
| `audio.voice` | `string` | ❌ | Registered voice (same behavior as `/v1/audio/speech`). |
| `audio.format` | enum | ❌ | Defaults to `"pcm"`. Same response formats as `/v1/audio/speech`. |
| `audio.sample_rate` | `int` | ❌ | Passed through `choose_encoder_sr`. |
| `temperature`, `cfg_weight`, `exaggeration`, `speed`, `language_id`, `seed`, `normalize_text` | optional | Forwarded to the underlying model exactly like `/v1/audio/speech`. |

### Responses

- **SSE** – if the request sets `Accept: text/event-stream`, the server emits the same PCM16 SSE events described earlier, regardless of the requested `audio.format`.
- **JSON** – otherwise, the response is an OpenAI-style envelope:

```json
{
  "id": "resp_abcd1234",
  "object": "response",
  "output": [
    {
      "type": "output_audio",
      "audio": {
        "format": "mp3",
        "sample_rate": 44100,
        "data": "<base64 audio>"
      }
    }
  ],
  "model": "gpt-4o-mini-tts"
}
```

Errors mirror `/v1/audio/speech`.

## WebSocket `/v1/realtime`

A minimal realtime channel that streams PCM16 for low-latency playback. The protocol intentionally mirrors the OpenAI Realtime beta in a simplified form.

1. **Client connects** and immediately sends a JSON text frame:

```json
{
  "model": "gpt-4o-mini-tts",
  "text": "Realtime playback test.",
  "audio": { "voice": "studio-demo", "sample_rate": 24000 }
}
```

2. **Server responds** with a sequence of binary frames. Each frame is `4-byte length prefix (big endian)` + `pcm16 payload`.
3. After streaming completes, the server sends a final JSON text frame: `{"type":"response.completed","sample_rate":24000}` and closes the socket.

Validation failures or unexpected exceptions are surfaced as JSON error frames before the socket closes with code `1011`.

## Encoding Notes

- WAV/FLAC encoding uses `soundfile` directly.
- MP3/AAC/Opus streaming relies on FFmpeg. The server performs a startup probe and returns HTTP 501 if the binary is missing. Inspect server logs (`ffmpeg encode failed: ...`) when troubleshooting codec issues.
