"""
Fish Speech TTS RunPod Serverless Handler

Provides voice cloning TTS using Fish Speech OpenAudio S1-mini model.
Compatible with the existing HistoryGen AI audio generation pipeline.

Input format (same as ChatterboxTTS):
{
    "text": "The text to synthesize",
    "reference_audio_base64": "base64 encoded audio for voice cloning"
}

Output format:
{
    "audio_base64": "base64 encoded WAV audio",
    "sample_rate": 24000
}
"""

import runpod
import base64
import io
import os
import tempfile
import traceback

import torch
import torchaudio

# Fish Speech imports
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.modded_dac import ModdedDACModel
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# Configuration
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/openaudio-s1-mini")
DECODER_CHECKPOINT = os.environ.get("DECODER_CHECKPOINT", f"{CHECKPOINT_PATH}/codec.pth")
DECODER_CONFIG = os.environ.get("DECODER_CONFIG", "modded_dac_vq")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()
COMPILE_MODEL = os.environ.get("COMPILE_MODEL", "true").lower() == "true"

# Constants
MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 2000
MIN_REFERENCE_DURATION = 3.0
OUTPUT_SAMPLE_RATE = 24000
AMPLITUDE = 32768  # Scale for 16-bit PCM

# Global engine instance
tts_engine = None


def load_models():
    """Load Fish Speech models on cold start."""
    global tts_engine

    print(f"Loading Fish Speech models from {CHECKPOINT_PATH}...")
    print(f"Device: {DEVICE}, Half precision: {HALF_PRECISION}, Compile: {COMPILE_MODEL}")

    # Determine precision
    precision = torch.float16 if HALF_PRECISION else torch.bfloat16

    # Load LLAMA (text-to-semantic) model queue
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        precision=precision,
        compile_model=COMPILE_MODEL,
    )
    print("LLAMA model loaded")

    # Load decoder (semantic-to-audio) model
    decoder_model = ModdedDACModel.load(
        config_name=DECODER_CONFIG,
        checkpoint_path=DECODER_CHECKPOINT,
        device=DEVICE,
    )
    decoder_model.eval()
    print(f"Decoder model loaded, sample rate: {decoder_model.sample_rate}")

    # Create TTS inference engine
    tts_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=COMPILE_MODEL,
    )
    print("TTS inference engine created")

    # Warm up the models
    print("Warming up models...")
    try:
        warmup_request = ServeTTSRequest(
            text="Hello, this is a warmup test.",
            references=[],
            temperature=0.7,
            repetition_penalty=1.2,
            format="wav",
        )
        # Run warmup inference
        audio_chunks = []
        for result in tts_engine.inference(warmup_request):
            if hasattr(result, 'audio') and result.audio is not None:
                audio_chunks.append(result.audio)
        print("Models warmed up successfully")
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    print("Fish Speech models ready!")


def decode_reference_audio(audio_base64: str) -> tuple:
    """Decode base64 audio and get duration info."""
    audio_bytes = base64.b64decode(audio_base64)

    # Write to temp file for torchaudio to read
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        waveform, sample_rate = torchaudio.load(temp_path)
        duration = waveform.shape[1] / sample_rate
        return audio_bytes, duration, sample_rate
    finally:
        os.unlink(temp_path)


def handler(job):
    """
    RunPod serverless handler for Fish Speech TTS.

    Input:
        text (str): Text to synthesize
        reference_audio_base64 (str, optional): Base64 encoded audio for voice cloning

    Output:
        audio_base64 (str): Base64 encoded WAV audio
        sample_rate (int): Audio sample rate (24000)
    """
    global tts_engine

    try:
        job_input = job.get("input", {})

        # Get text (support both "text" and "prompt" keys for compatibility)
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing required parameter: text"}

        # Validate text length
        text = text.strip()
        if len(text) < MIN_TEXT_LENGTH:
            return {"error": f"Text too short. Minimum length: {MIN_TEXT_LENGTH}"}
        if len(text) > MAX_TEXT_LENGTH:
            return {"error": f"Text too long. Maximum length: {MAX_TEXT_LENGTH}"}

        print(f"Generating TTS for {len(text)} characters...")

        # Process reference audio for voice cloning
        references = []
        reference_audio_base64 = job_input.get("reference_audio_base64")

        if reference_audio_base64:
            print("Processing reference audio for voice cloning...")
            try:
                audio_bytes, duration, sample_rate = decode_reference_audio(reference_audio_base64)
                print(f"Reference audio: {duration:.1f}s, {sample_rate}Hz")

                if duration < MIN_REFERENCE_DURATION:
                    print(f"Warning: Reference audio is short ({duration:.1f}s). Recommend 10-30s.")

                # Create reference audio object
                references.append(
                    ServeReferenceAudio(
                        audio=audio_bytes,
                        text="",  # Empty text for zero-shot cloning
                    )
                )
            except Exception as e:
                print(f"Error processing reference audio: {e}")
                traceback.print_exc()
                # Continue without voice cloning
                references = []

        # Build TTS request
        request = ServeTTSRequest(
            text=text,
            references=references,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2,
            max_new_tokens=1024,
            normalize=True,
            format="wav",
            chunk_length=200,
        )

        # Generate audio using inference engine
        audio_segments = []
        for result in tts_engine.inference(request):
            if hasattr(result, 'audio') and result.audio is not None:
                # Convert float audio to int16
                audio_data = (result.audio * AMPLITUDE).to(torch.int16)
                audio_segments.append(audio_data)

        if not audio_segments:
            return {"error": "No audio generated. Please check the input text."}

        # Concatenate all segments
        audio = torch.cat(audio_segments, dim=-1)

        # Ensure correct shape for torchaudio (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Get sample rate from decoder model
        model_sample_rate = tts_engine.decoder_model.sample_rate

        # Resample to output sample rate if needed
        if model_sample_rate != OUTPUT_SAMPLE_RATE:
            audio = audio.float()  # Resample needs float
            audio = torchaudio.functional.resample(audio, model_sample_rate, OUTPUT_SAMPLE_RATE)
            audio = audio.to(torch.int16)

        # Save to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), OUTPUT_SAMPLE_RATE, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Encode to base64
        audio_base64_out = base64.b64encode(audio_bytes).decode("utf-8")

        print(f"Generated {len(audio_bytes)} bytes of audio ({len(audio_base64_out)} base64 chars)")

        return {
            "audio_base64": audio_base64_out,
            "sample_rate": OUTPUT_SAMPLE_RATE,
        }

    except Exception as e:
        error_msg = f"TTS generation failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}


# Load models on cold start
print("Fish Speech TTS Handler starting...")
load_models()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
