# Fish Speech TTS RunPod Serverless Handler
# Based on Fish Speech OpenAudio S1-mini model

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies AND Python packages in single layer
# This ensures git is available when pip runs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Copy requirements first
COPY requirements.txt .

# Install git, then pip install in same RUN to ensure git is available
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
        build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Download Fish Speech model weights
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='/app/checkpoints/openaudio-s1-mini')"

# Copy handler
COPY handler.py .

# Set environment variables
ENV CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT=/app/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG=modded_dac_vq
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "-u", "handler.py"]
