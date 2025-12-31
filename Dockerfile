# Fish Speech TTS RunPod Serverless Handler
# Based on official Fish Speech Docker image

FROM fishaudio/fish-speech:latest

WORKDIR /app

# Install runpod (try pip first, fall back to uv)
RUN pip install --no-cache-dir runpod>=1.6.0 || uv pip install --system runpod>=1.6.0

# Copy handler
COPY handler.py .

# Environment variables
ENV CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT=/app/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG=modded_dac_vq
ENV PYTHONUNBUFFERED=1

# Download model weights (if not already in base image)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='/app/checkpoints/openaudio-s1-mini')" || echo "Model download skipped (may already exist)"

# Run handler
CMD ["python", "-u", "handler.py"]
