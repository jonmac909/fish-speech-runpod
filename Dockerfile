# Fish Speech TTS RunPod Serverless Handler
# Based on official Fish Speech Docker image

FROM fishaudio/fish-speech:latest

# Switch to root to install packages
USER root

WORKDIR /app

# Install runpod and huggingface_hub into the venv using uv
RUN uv pip install --python /app/.venv/bin/python runpod>=1.6.0 huggingface_hub

# Download model weights at build time (not runtime)
# This bakes the ~2GB model into the Docker image
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='/app/checkpoints/openaudio-s1-mini', token='${HF_TOKEN}')"

# Copy handler
COPY handler.py .

# Environment variables
ENV CHECKPOINT_PATH=/app/checkpoints/openaudio-s1-mini
ENV DECODER_CHECKPOINT=/app/checkpoints/openaudio-s1-mini/codec.pth
ENV DECODER_CONFIG=modded_dac_vq
ENV PYTHONUNBUFFERED=1

# Override the base image's ENTRYPOINT (which runs run_webui.py)
ENTRYPOINT []

# Run handler using the venv Python (where torch is installed)
CMD ["/app/.venv/bin/python", "-u", "handler.py"]
