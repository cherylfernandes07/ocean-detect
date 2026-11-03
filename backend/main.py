"""
main.py — FastAPI backend for OceanDetect

This is the entry point for the API server. It:
  1. Loads a trained OrcaCNN model at startup (once, shared across all requests)
  2. Exposes three self-contained endpoints: /health, /debug, /detect
  3. Mounts two sub-routers: /whale-voice (LLM character voice) and /workflow (full chain)

Spectrogram pipeline note:
  The audio_to_specgram() function here is intentionally identical to the one in
  train.py. If you change one, change both — any mismatch means the model sees
  different-looking inputs at inference vs. training, which destroys accuracy.

Run with:
  MODEL_PATH=model_best_f1.pt ANTHROPIC_API_KEY=sk-... \
    python -m uvicorn backend.main:app --reload
"""
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File
from backend.whale_voice import router as whale_voice_router
from backend.workflow import router as workflow_router
from fastapi.responses import JSONResponse
import shutil, os, tempfile
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.fftpack import fft
from scipy.signal.windows import hann

# OrcaCNN lives in backend/model.py; import by package path when running from project root
from backend.model import OrcaCNN

from fastapi.middleware.cors import CORSMiddleware


# ─── Spectrogram parameters — MUST match train.py exactly ─────────────────────
#
# These constants describe how we slice audio into a 2D image (spectrogram).
# Think of it like turning a song into sheet music: instead of notes, we get
# a grid where each row is a tiny slice of time and each column is a frequency.
#
# Orcasound data is recorded at 20 kHz (20,000 samples per second), which is
# half of the typical 44.1 kHz consumer audio — plenty for orca calls, which
# peak around 10–12 kHz.
#
# WINDOW_DURATION = 2.45s → the CNN sees exactly 2.45 seconds per inference pass
# FFT_SIZE = 2048          → each FFT window covers 2048 samples (~0.1s at 20kHz)
# OVERLAP_PCT = 75%        → consecutive windows overlap 75%, so JUMP = 512 samples
# FREQ_BINS = 1020         → we skip the lowest 4 FFT bins (DC + low noise) and
#                            keep bins 4 through 1023 (the top half is mirrored)
# TIME_STEPS = ~71         → how many FFT windows fit in one 2.45s clip
#
SAMPLE_RATE = 20000
WINDOW_DURATION = 2.45
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # 49,000 samples per clip
FFT_SIZE = 2048
OVERLAP_PCT = 75
JUMP = int(FFT_SIZE * (100 - OVERLAP_PCT) / OVERLAP_PCT)  # 683 samples between windows
FREQ_BINS = FFT_SIZE // 2 - 4   # 1020 frequency columns (skip DC and lowest 3 bins)
TIME_STEPS = WINDOW_SAMPLES // JUMP  # ~71 time rows


# ─── FastAPI app + routers ─────────────────────────────────────────────────────
#
# We split the API into three files to keep concerns separate:
#   main.py       — core detection logic, shared model
#   whale_voice.py — LLM character endpoint (/whale-voice)
#   workflow.py    — full LangChain chain (/workflow)
#
app = FastAPI(
    title="OceanDetect API",
    description="Orca call detection + LLM social media workflow",
    version="0.1.0",
)
app.include_router(whale_voice_router)   # POST /whale-voice
app.include_router(workflow_router)      # POST /workflow

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Device selection ─────────────────────────────────────────────────────────
#
# PyTorch can run on different hardware. We prefer GPU (faster) over CPU (slower).
# On an M1/M2 Mac, "mps" is the Apple Silicon GPU. On a Linux server, "cuda".
# In this cloud container or a basic laptop, it falls back to "cpu" — still works,
# just a bit slower for inference (model is small so it's fine).
#
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ─── Load model once at startup ────────────────────────────────────────────────
#
# We load the model ONCE when the server starts, not on every request. This is
# important — loading a model takes ~0.5s and holds it in memory. By doing it
# once we keep requests fast (a few ms each).
#
# MODEL_PATH env var lets you swap between:
#   model.pt          — best validation accuracy (default)
#   model_best_f1.pt  — best F1 score (recommended for detection tasks;
#                        better balance of catching real calls vs. false alarms)
#
# The checkpoint dict saved by train.py contains:
#   "model_state_dict" — the actual weights
#   "val_acc"          — validation accuracy at the time of saving
#   "epoch"            — which training epoch this came from
#
model = OrcaCNN(time_steps=TIME_STEPS, freq_bins=FREQ_BINS).to(device)
model_path = os.environ.get("MODEL_PATH", "model.pt")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {model_path} (val_acc={checkpoint.get('val_acc', '?'):.3f})")
else:
    print(f"WARNING: no model found at {model_path} — using random weights")

# Switch to eval mode: disables Dropout (which randomly zeroes neurons during
# training to prevent memorization). At inference time we want all neurons active.
model.eval()


# ─── Spectrogram helper ────────────────────────────────────────────────────────

def audio_to_specgram(audio: np.ndarray) -> np.ndarray:
    """
    Convert a 1D float32 audio array into a 2D spectrogram.

    This is the Val Veirs / Orcasound FFT pipeline — the same algorithm used in
    train.py. Must stay in sync with training or the model will see unfamiliar input.

    Algorithm:
      - Slide a 2048-sample Hanning window across the audio with 75% overlap
      - Apply FFT to each window to get the frequency content at that moment
      - Normalize each window's power to [0, 1] (so quiet moments aren't drowned out)
      - Stack the windows as rows → output shape is (TIME_STEPS, FREQ_BINS)

    The Hanning window (h) is a bell-shaped weight applied before the FFT. Without it,
    sharp edges at the clip boundaries cause "spectral leakage" — fake frequency spikes.
    It's the same reason you might fade audio in/out instead of cutting abruptly.

    Returns: (TIME_STEPS, FREQ_BINS) float32 array — the "sheet music" the CNN reads
    """
    N = len(audio)
    h = hann(FFT_SIZE)                                          # bell-shaped window weights
    specGram = np.zeros([TIME_STEPS, FREQ_BINS], dtype=np.float32)
    wav_ptr, row = 0, 0
    while wav_ptr + FFT_SIZE <= N and row < TIME_STEPS:
        chunk = audio[wav_ptr: wav_ptr + FFT_SIZE]              # grab one window of audio
        yhf = fft(chunk * h)                                    # apply FFT (with Hanning)
        pwr = np.abs(yhf)                                       # magnitude spectrum
        pwr[0] = 0                                              # zero out DC component (not useful)
        pwr_max = pwr.max()
        if pwr_max > 0:
            pwr /= pwr_max                                      # normalize this window to [0, 1]
        specGram[row, :] = pwr[4: FFT_SIZE // 2]               # keep bins 4–1023 (skip low noise)
        wav_ptr += JUMP                                         # advance by JUMP samples (25% of FFT_SIZE)
        row += 1
    return specGram


def load_audio_window(path: str) -> np.ndarray:
    """
    Load an audio file and prepare it for the spectrogram pipeline.

    Steps:
      1. Read with soundfile — MUST use dtype='float32' so amplitudes are in [-1, 1].
         If we used the default int16, values would be in [-32768, 32767] — a completely
         different scale from the float32 training data, making every inference wrong.
      2. Convert stereo to mono by taking the first channel (orcas are mono anyway)
      3. Resample to 20 kHz if needed (nearest-neighbor; accurate enough for inference)
      4. Normalize amplitude to roughly [-0.5, 0.5]
      5. Pad with silence or trim to exactly WINDOW_SAMPLES (49,000 samples / 2.45s)

    Returns: 1D float32 array of length WINDOW_SAMPLES, ready for audio_to_specgram()
    """
    # dtype='float32' is critical — avoids int16 scale mismatch with training data
    audio, sr = sf.read(path, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # take left channel if stereo

    # Resample to 20 kHz if the file was recorded at a different rate.
    # Uses simple nearest-neighbor interpolation: for each target sample,
    # we find the closest source sample index. Fast but not HiFi — good enough here.
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(audio) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        indices = np.clip(indices, 0, len(audio) - 1)
        audio = audio[indices].astype(np.float32)

    # Normalize: divide by twice the peak amplitude → values in roughly [-0.5, 0.5]
    # (matches the normalization used in training)
    wav_max = np.abs(audio).max()
    if wav_max > 0:
        audio = audio / (wav_max * 2.0)

    # The CNN expects exactly WINDOW_SAMPLES. Pad short clips with silence;
    # trim longer clips to the first 2.45 seconds.
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    else:
        audio = audio[:WINDOW_SAMPLES]

    return audio.astype(np.float32)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Simple liveness check. Returns the device being used and the model path.
    Useful for confirming the server is up and the right model was loaded.
    """
    return {"status": "ok", "device": str(device), "model": model_path}


@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    """
    Diagnostic endpoint — returns spectrogram stats without running the model.

    Use this to verify the audio pipeline is working correctly before trusting
    /detect results. Key things to check in the response:
      - specgram_shape should be [71, 1020] (TIME_STEPS × FREQ_BINS)
      - specgram_max should be 1.0 (normalization working)
      - nonzero_rows should be close to 71 (real audio, not silence)

    If specgram_shape is wrong → sample rate or constants are mismatched
    If specgram_max is tiny → dtype='float32' not being used (int16 scale issue)
    If nonzero_rows is 0   → file is empty or unreadable
    """
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        audio = load_audio_window(tmp_path)
        specgram = audio_to_specgram(audio)
    finally:
        os.remove(tmp_path)
    return JSONResponse(content={
        "audio_len": len(audio),
        "audio_min": float(audio.min()),
        "audio_max": float(audio.max()),
        "specgram_shape": list(specgram.shape),
        "specgram_min": float(specgram.min()),
        "specgram_max": float(specgram.max()),
        "specgram_mean": float(specgram.mean()),
        "nonzero_rows": int((specgram.max(axis=1) > 0).sum()),
    })


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Core detection endpoint — runs the OrcaCNN and returns a yes/no + confidence.

    Pipeline:
      upload → temp file → load_audio_window() → audio_to_specgram()
      → OrcaCNN forward pass → softmax → threshold → response

    The model outputs logits for two classes: [not_orca, orca].
    Softmax converts logits to probabilities (they sum to 1.0).
    We read probs[1] as the "orca confidence" and compare to our threshold.

    Threshold is 0.3 (not the default 0.5) because:
      - Missing a real orca call is worse than a false alarm
      - Lower threshold = higher recall = catch more real calls
      - False positives can be filtered by the LLM moderation step in /workflow
    """
    # Save upload to a temp file so soundfile can read it from disk
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        audio = load_audio_window(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse(status_code=400, content={"error": f"Could not read audio: {e}"})
    finally:
        # Always clean up temp file even if load_audio_window raised
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Build spectrogram tensor: add batch dim and channel dim → shape [1, 1, 71, 1020]
    # CNN expects (batch_size, channels, height, width), like an image with 1 color channel
    specgram = audio_to_specgram(audio)
    x = torch.tensor(specgram).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,71,1020]

    # torch.no_grad() tells PyTorch not to build a computation graph for backprop.
    # We're only doing inference (no training), so this saves memory and speeds things up.
    with torch.no_grad():
        logits = model(x)                         # raw scores for [not_orca, orca]
        probs = F.softmax(logits, dim=1)          # convert to probabilities (sum to 1)
        confidence = probs[0][1].item()           # orca probability for this clip
        detected = confidence > 0.3               # threshold: catch more, miss less

    return JSONResponse(content={
        "detected": detected,
        "confidence": round(confidence, 4),
        "label": "orca" if detected else "not_orca",
    })