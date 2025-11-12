FROM nvcr.io/nvidia/pytorch:25.09-py3

# NOTE: The 25.09 NGC base ships Torch 2.9 with SM_120 support, which is
# required for NVIDIA Blackwell GPUs. Avoid replacing torch/torchaudio to keep
# the Blackwell-compatible stack intact.

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    CUDA_MODULE_LOADING=LAZY \
    MALLOC_ARENA_MAX=2 \
    PYTORCH_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_TORCHVISION=1

# Install runtime dependencies required for audio IO and diagnostics.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg ca-certificates git curl \
 && rm -rf /var/lib/apt/lists/*

# Remove CUDA extensions not required for Chatterbox TTS to keep ABI exposure small.
RUN pip uninstall -y flash-attn flash_attn xformers torchvision || true

# Preserve the NGC Torch 2.9 stack and avoid reinstalling torch or torchaudio.
RUN pip install --upgrade pip wheel setuptools scikit-build-core pybind11 \
 && pip install --no-deps chatterbox-tts==0.1.4 \
 && pip install \
      fastapi==0.115.2 \
      "uvicorn[standard]==0.30.6" \
      librosa==0.11.0 \
      transformers==4.46.3 \
      "huggingface-hub<1.0,>=0.23.2" \
      diffusers==0.29.0 \
      safetensors==0.5.3 \
      s3tokenizer==0.2.0 \
      resemble_perth==1.0.0 \
      conformer==0.3.2 \
      tokenizers==0.20.3 \
      einops==0.8.1 \
      pyyaml==6.0.3 \
      requests==2.32.3 \
      soundfile==0.12.1 \
      pillow==10.4.0

WORKDIR /opt/app
COPY app /opt/app/app

# Default runtime parameters; override with environment variables as needed.
ENV SERVICE_PORT=4123 \
    DEVICE=cuda \
    SAMPLE_RATE=24000

EXPOSE 4123
HEALTHCHECK --interval=30s --timeout=10s --retries=20 \
  CMD curl -fsS http://127.0.0.1:4123/healthz >/dev/null || exit 1

CMD ["python","-m","uvicorn","app.server:app","--host","0.0.0.0","--port","4123","--log-level","info"]
