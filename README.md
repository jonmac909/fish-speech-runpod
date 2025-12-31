# Fish Speech TTS - RunPod Serverless Handler

RunPod serverless endpoint for Fish Speech OpenAudio S1-mini text-to-speech with voice cloning.

## Features

- **Voice Cloning**: 10-30 second audio samples for high-quality voice cloning
- **Fast Inference**: Model compilation for ~150 tokens/second on RTX 4090
- **Compatible API**: Same input/output format as ChatterboxTTS for drop-in replacement

## API Format

### Input

```json
{
  "text": "The text to synthesize",
  "reference_audio_base64": "base64 encoded audio for voice cloning (optional)"
}
```

### Output

```json
{
  "audio_base64": "base64 encoded WAV audio",
  "sample_rate": 24000
}
```

## RunPod Deployment

1. Create a new Serverless Endpoint on RunPod
2. Connect to this GitHub repository
3. Configure:
   - **GPU**: RTX 4090 or similar (12GB+ VRAM)
   - **Max Workers**: 5
   - **Idle Timeout**: 5 seconds
   - **Container Disk**: 20GB (for model weights)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_PATH` | `/app/checkpoints/openaudio-s1-mini` | Path to model weights |
| `DECODER_CHECKPOINT` | `{CHECKPOINT_PATH}/codec.pth` | Path to decoder weights |
| `DECODER_CONFIG` | `modded_dac_vq` | Decoder configuration name |

## Model Info

- **Model**: Fish Speech OpenAudio S1-mini (0.5B parameters)
- **Quality**: 0.008 WER, 0.004 CER on English
- **Output**: 24000Hz mono WAV
- **License**: Apache 2.0 (code), CC-BY-NC-SA-4.0 (weights)

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')"

# Test handler
python handler.py
```

## Credits

- [Fish Speech](https://github.com/fishaudio/fish-speech) by Fish Audio
- Built for [HistoryGen AI](https://historygenai.netlify.app)
