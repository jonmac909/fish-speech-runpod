# Fish Speech TTS RunPod Serverless Handler
# Based on Fish Speech OpenAudio S1-mini model

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Fish Speech model weights
RUN pip install huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='/app/checkpoints/openaudio-s1-mini')"

# Copy handler
COPY handler.py .

# Set environment variables
ENV CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT=/app/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG=modded_dac_vq
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "-u", "handler.py"]
