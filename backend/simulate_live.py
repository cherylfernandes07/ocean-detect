"""
simulate_live.py — Simulates a live hydrophone feed for testing the OceanDetect pipeline.

WHY THIS EXISTS:
  Real hydrophones stream audio 24/7. We don't have a live feed during development,
  so this script pretends to be one by replaying a pre-recorded WAV file one second
  at a time — the same pace a real hydrophone would produce audio.

HOW IT WORKS (the assembly line analogy):
  Think of this like a conveyor belt in a fish cannery:
    - The WAV file is the ocean (endless audio)
    - 1-second chunks are fish coming down the belt
    - The CNN inspector checks each fish (chunk) — orca or not?
    - The vote counter watches the last 5 fish — if 3+ are orca, sound the alarm
    - The alarm triggers the full workflow (whale voice + post + Facebook)
    - After the alarm, the counter resets and waits 60s before alarming again

SLIDING WINDOW MAJORITY VOTE:
  Instead of firing the workflow on every single CNN detection, we watch a window
  of the last N chunks and require a majority to be positive. This is the same idea
  as asking 5 doctors instead of 1 — a single false positive doesn't ruin the result.

  Window size = 5 chunks (5 seconds of audio)
  Threshold   = 3 out of 5 must be detected
  This means: even if 2 chunks are noisy background, we still fire correctly.

COOLDOWN:
  After the workflow fires, we ignore detections for 60 seconds. Without this,
  a 10-second orca call would trigger 10 workflow runs and 10 Facebook posts.
  The cooldown ensures one call = one post.

USAGE:
  python simulate_live.py --wav /path/to/recording.wav
  python simulate_live.py --wav /path/to/recording.wav --loop          # loop forever
  python simulate_live.py --wav /path/to/recording.wav --orca Tahlequah --call S_19
  python simulate_live.py --wav /path/to/recording.wav --force         # bypass CNN (for testing)

REQUIREMENTS:
  - Backend must be running: uvicorn backend.main:app --reload
  - pip install soundfile numpy requests
"""

import argparse
import io
import time
import numpy as np
import soundfile as sf
import requests
from collections import deque
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────────

API_BASE        = "http://localhost:8000"   # where uvicorn is running
CHUNK_SECONDS   = 1                         # send 1 second of audio at a time
SAMPLE_RATE     = 20000                     # Orcasound hydrophones record at 20kHz
CHUNK_SAMPLES   = SAMPLE_RATE * CHUNK_SECONDS  # 20,000 samples per chunk

# Sliding window vote settings
WINDOW_SIZE     = 5    # look at the last 5 chunks (5 seconds)
VOTE_THRESHOLD  = 3    # need at least 3 out of 5 to be "orca" before firing workflow

# After a workflow fires, don't fire again for this many seconds.
# Prevents the same call from generating multiple posts.
COOLDOWN_SECONDS = 60


# ─── Helpers ──────────────────────────────────────────────────────────────────

def now() -> str:
    """Human-readable timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def chunk_to_wav_bytes(chunk: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a numpy audio array into WAV file bytes in memory.

    WHY: The /detect endpoint expects a file upload (multipart form), not raw bytes.
    We use an in-memory buffer (io.BytesIO) instead of writing to disk — faster
    and cleaner. soundfile can write to a buffer the same way it writes to a file.

    The buffer is like a fake file that lives in RAM. We write WAV bytes into it,
    seek back to the start, then hand it to requests as if it were a file on disk.
    """
    buf = io.BytesIO()
    sf.write(buf, chunk, sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)  # rewind to the beginning so requests reads from the start
    return buf


def send_chunk_to_detect(chunk: np.ndarray, force: bool = False) -> dict:
    """
    POST a 1-second audio chunk to /detect and return the JSON response.

    Returns a dict with at minimum:
      {"detected": bool, "confidence": float, "label": str}
    Returns {"detected": False, "confidence": 0.0} on any error so the
    vote window degrades gracefully if the server is briefly unavailable.
    """
    buf = chunk_to_wav_bytes(chunk, SAMPLE_RATE)
    try:
        resp = requests.post(
            f"{API_BASE}/detect",
            files={"file": ("chunk.wav", buf, "audio/wav")},
            timeout=10,  # don't hang forever if server is slow
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [{now()}] ⚠️  /detect error: {e}")
        return {"detected": False, "confidence": 0.0}


def send_window_to_workflow(
    window_chunks: list,
    orca_name: str,
    call_type: str,
    location: str,
    force: bool = False,
) -> dict:
    """
    Concatenate the last WINDOW_SIZE chunks into one clip and POST to /workflow.

    WHY CONCATENATE: The CNN in /workflow expects 2.45 seconds of audio, not 1 second.
    By joining the last 5 chunks (5 seconds), we give the workflow a richer clip
    that's more likely to contain the full call — not just a fragment.

    The /workflow endpoint handles trimming to 2.45s internally (in load_audio_window).

    ?force=true is passed when --force flag is set, bypassing CNN detection in the
    workflow endpoint itself (useful when the model isn't trained well yet).
    """
    # Stack all chunks into one continuous audio array
    combined = np.concatenate(window_chunks)
    buf = chunk_to_wav_bytes(combined, SAMPLE_RATE)

    params = {
        "orca_name": orca_name,
        "call_type": call_type,
        "location": location,
    }
    if force:
        params["force"] = "true"

    try:
        resp = requests.post(
            f"{API_BASE}/workflow",
            files={"file": ("window.wav", buf, "audio/wav")},
            params=params,
            timeout=60,  # LLM calls can take up to 30s — give it room
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [{now()}] ⚠️  /workflow error: {e}")
        return {"error": str(e)}


def print_workflow_result(result: dict):
    """Pretty-print the workflow result so it's easy to read in the terminal."""
    print()
    print("  ┌─────────────────────────────────────────────────────")
    print(f"  │ 🐋 ORCA DETECTED — confidence {result.get('confidence', '?'):.0%}")
    print("  │")

    voice = result.get("whale_voice", "")
    if voice:
        # Wrap long lines at 60 chars for readability
        print("  │ 🎵 Whale voice:")
        for line in voice.split("\n"):
            print(f"  │   {line}")
    print("  │")

    approved = result.get("approved")
    reason   = result.get("moderation_reason", "")
    if approved:
        print("  │ ✅ Moderation: APPROVED")
        print(f"  │   {reason}")
        post_id = result.get("facebook_post_id")
        if post_id:
            print(f"  │ 📘 Posted to Facebook: {post_id}")
        else:
            fb_err = result.get("facebook_error", "")
            print(f"  │ 📘 Facebook: {fb_err}")
    else:
        print("  │ ❌ Moderation: REJECTED")
        print(f"  │   {reason}")

    print("  └─────────────────────────────────────────────────────")
    print()


# ─── Main loop ────────────────────────────────────────────────────────────────

def run(args):
    """
    Main simulation loop.

    State we track:
      vote_window   — a deque of the last WINDOW_SIZE detection results (True/False)
                      deque is like a list that automatically drops the oldest item
                      when it gets too long — perfect for a sliding window
      cooldown_until — timestamp when the cooldown expires; we ignore detections until then
      chunk_index   — which 1-second chunk we're on (for progress display)
    """
    print(f"\n🌊 OceanDetect Live Simulator")
    print(f"   WAV file : {args.wav}")
    print(f"   Orca     : {args.orca} ({args.call})")
    print(f"   Window   : {WINDOW_SIZE}s, threshold {VOTE_THRESHOLD}/{WINDOW_SIZE}")
    print(f"   Cooldown : {COOLDOWN_SECONDS}s after each detection")
    print(f"   Force    : {'yes (bypassing CNN)' if args.force else 'no'}")
    print()

    # Load the full WAV file into memory once
    print(f"  Loading {args.wav}...")
    audio, sr = sf.read(args.wav, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # mono

    # Resample to 20kHz if needed (same logic as load_audio_window in main.py)
    if sr != SAMPLE_RATE:
        print(f"  Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
        ratio = SAMPLE_RATE / sr
        new_len = int(len(audio) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        indices = np.clip(indices, 0, len(audio) - 1)
        audio = audio[indices].astype(np.float32)

    total_chunks = len(audio) // CHUNK_SAMPLES
    print(f"  {len(audio)} samples → {total_chunks} chunks ({total_chunks}s of audio)\n")

    # Sliding window: stores True/False for the last WINDOW_SIZE chunks.
    # maxlen=WINDOW_SIZE means old entries fall off the left automatically.
    vote_window: deque = deque(maxlen=WINDOW_SIZE)

    # Also keep the raw audio chunks in a parallel deque so we can concatenate
    # them and send to /workflow when a detection fires.
    audio_window: deque = deque(maxlen=WINDOW_SIZE)

    cooldown_until = 0.0   # Unix timestamp; 0 means no cooldown active
    chunk_index    = 0

    while True:
        # ── Get next chunk ─────────────────────────────────────────────────────
        start_sample = chunk_index * CHUNK_SAMPLES
        end_sample   = start_sample + CHUNK_SAMPLES

        if end_sample > len(audio):
            # We've reached the end of the file
            if args.loop:
                print(f"\n  [{now()}] 🔁 End of file — looping back to start\n")
                chunk_index = 0
                continue
            else:
                print(f"\n  [{now()}] ✅ End of file. Exiting.")
                break

        chunk = audio[start_sample:end_sample]
        chunk_index += 1

        # ── Send to /detect ────────────────────────────────────────────────────
        # If --force is set, skip the CNN entirely and treat every chunk as detected.
        # This is purely for testing the voting + workflow logic when the model is weak.
        if args.force:
            detected   = True
            confidence = 1.0
        else:
            result     = send_chunk_to_detect(chunk)
            detected   = result.get("detected", False)
            confidence = result.get("confidence", 0.0)

        # Add this chunk's result to the sliding window
        vote_window.append(detected)
        audio_window.append(chunk)

        # Count how many of the last WINDOW_SIZE chunks were positive
        votes = sum(vote_window)

        # Progress line — shows the vote tally and a mini bar of the window
        window_bar = "".join("🟢" if v else "⚫" for v in vote_window)
        cooldown_remaining = max(0, cooldown_until - time.time())
        cooldown_str = f" | cooldown {cooldown_remaining:.0f}s" if cooldown_remaining > 0 else ""
        print(
            f"  [{now()}] chunk {chunk_index:>4} | conf={confidence:.2f} "
            f"| votes={votes}/{len(vote_window)} {window_bar}{cooldown_str}"
        )

        # ── Check vote threshold ───────────────────────────────────────────────
        # Fire the workflow if:
        #   1. Enough chunks in the window are positive (majority vote)
        #   2. We're not in a cooldown period from a recent detection
        #   3. The vote window is full (we need WINDOW_SIZE chunks before deciding)
        in_cooldown      = time.time() < cooldown_until
        window_is_full   = len(vote_window) == WINDOW_SIZE
        threshold_met    = votes >= VOTE_THRESHOLD

        if threshold_met and window_is_full and not in_cooldown:
            print(f"\n  [{now()}] 🚨 Threshold met ({votes}/{WINDOW_SIZE}) — triggering workflow!")

            # Run the full LangChain chain with the buffered audio window
            workflow_result = send_window_to_workflow(
                list(audio_window),
                orca_name=args.orca,
                call_type=args.call,
                location=args.location,
                force=args.force,
            )
            print_workflow_result(workflow_result)

            # Start cooldown — don't fire again for COOLDOWN_SECONDS
            cooldown_until = time.time() + COOLDOWN_SECONDS
            print(f"  [{now()}] ⏳ Cooldown started — next detection in {COOLDOWN_SECONDS}s\n")

            # Clear the vote window so we start fresh after the cooldown.
            # Without this, the same 5 votes would immediately re-trigger.
            vote_window.clear()
            audio_window.clear()

        # ── Wait 1 second before next chunk ───────────────────────────────────
        # This is what makes it feel "live" — we pace ourselves to real time.
        # In production you'd read from an actual audio stream instead of sleeping.
        time.sleep(CHUNK_SECONDS)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a live hydrophone feed by replaying a WAV file."
    )
    parser.add_argument(
        "--wav", required=True,
        help="Path to the WAV file to replay (e.g. /tmp/orca_recording.wav)"
    )
    parser.add_argument(
        "--orca", default="Tahlequah",
        help="Name of the orca to use for whale voice (default: Tahlequah)"
    )
    parser.add_argument(
        "--call", default="S_19",
        help="Ford-Osborne call type label (default: S_19)"
    )
    parser.add_argument(
        "--location", default="Orcasound Lab hydrophone, Puget Sound",
        help="Hydrophone location for the Facebook post"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop the WAV file continuously instead of stopping at the end"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass CNN detection — always treat chunks as detected (for testing)"
    )

    args = parser.parse_args()
    run(args)