# Fish Speech TTS RunPod Serverless Handler
# Based on official Fish Speech Docker image

FROM fishaudio/fish-speech:latest

# Switch to root to install packages
USER root

WORKDIR /app

# Install runpod and huggingface_hub into the venv using uv
RUN uv pip install --python /app/.venv/bin/python runpod>=1.6.0 huggingface_hub

# Copy handler
COPY handler.py .

# Environment variables
ENV CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT=/app/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG=modded_dac_vq
ENV PYTHONUNBUFFERED=1

# Note: Model weights should be mounted at runtime via RunPod volume
# The fishaudio/fish-speech image expects checkpoints at /app/checkpoints

# Override the base image's ENTRYPOINT (which runs run_webui.py)
ENTRYPOINT []

# Run handler using the venv Python (where torch is installed)
CMD ["/app/.venv/bin/python", "-u", "handler.py"]
