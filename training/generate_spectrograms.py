#!/usr/bin/env python3
"""
generate_spectrograms.py

Converts all labeled orca call WAV files (S_1.wav, S_2.wav, etc.)
into spectrogram numpy arrays (.npy files) ready for CNN training.

A spectrogram is a 2D grid where:
  - Each ROW is a moment in time
  - Each COLUMN is a frequency
  - Each VALUE is how loud that frequency was at that moment (0.0 to 1.0)

Think of it like sheet music: time goes left to right, pitch goes bottom to top,
and brightness shows how loud each note is.

This uses the same algorithm as WAV_preprocessing_val.ipynb (Val Veirs, 2018).
"""

import numpy as np
import soundfile as sf       # reads WAV/FLAC audio files into numpy arrays
from scipy.fftpack import fft  # Fast Fourier Transform — converts time signal to frequencies
from scipy.signal.windows import hann  # Hanning window — smooths edges of each audio chunk before FFT
import os
import glob                  # finds files matching a pattern (like *.wav)


# ─── Configuration ────────────────────────────────────────────────────────────
# These values MUST match the notebook or the spectrogram shape will be different

FFT_SIZE = 2048   # How many audio samples to analyze at once (one "window")
                  # 2048 samples at 44100 Hz = ~46ms per window
                  # Larger = better frequency resolution, worse time resolution

OVERLAP = 75      # How much each window overlaps the previous one (percent)
                  # 75% overlap means we slide forward only 25% each step
                  # More overlap = smoother spectrogram, more rows

INPUT_DIR = "WAV_FLACprocessor_Val/FordOsborneLabeled"   # where the S_*.wav files live
OUTPUT_DIR = "WAV_FLACprocessor_Val/processedFiles"      # where we save the .npy files

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Main Function ─────────────────────────────────────────────────────────────

def wav_to_specgram(wav_path):
    """
    Reads one WAV file and returns its spectrogram as a 2D numpy array.
    Shape will be (num_time_steps, num_frequencies) — e.g. (46, 1020)
    Values are normalized to 0.0–1.0 (0 = silent, 1 = loudest moment)
    """

    # Step 1: Load the audio file
    # audioData = array of amplitude samples (the waveform)
    # sample_rate = how many samples per second (usually 44100 Hz for audio)
    audioData, sample_rate = sf.read(wav_path)

    # Step 2: Handle stereo audio — we only need one channel (left/mono)
    # Shape check: if audioData has 2 dimensions, it's stereo [samples, channels]
    if len(audioData.shape) > 1:
        audioData = audioData[:, 0]  # take channel 0 (left channel)

    # Step 3: Normalize amplitude to -3dB
    # Divide by (max * 2) so the loudest peak becomes 0.5
    # This puts all files on the same loudness scale regardless of recording level
    wav_max = np.amax(np.abs(audioData))
    audioData = audioData / (wav_max * 2.0)

    # Step 4: Set up the FFT sliding window
    N = len(audioData)  # total number of audio samples in this file

    # Hanning window: a bell-shaped curve applied to each chunk before FFT
    # Without it, the sharp start/end of each chunk creates fake frequencies ("spectral leakage")
    h = hann(FFT_SIZE)

    # How far to slide forward each step
    # At 75% overlap: jump = 2048 * (100-75)/75 = 683 samples forward each step
    jump = int(FFT_SIZE * (100 - OVERLAP) / OVERLAP)

    # How many time steps will fit in this file
    # e.g. a file with 32000 samples / 683 jump = 46 rows
    jump_max = int(N // jump)

    # Preallocate the 2D spectrogram array with zeros
    # Rows = time steps, Columns = frequency bins (FFT_SIZE//2 - 4 = 1020)
    # We skip the first 4 bins because they're very low frequency (near DC / below 170 Hz)
    specGram = np.zeros([jump_max, FFT_SIZE // 2 - 4])

    # Step 5: Slide the window through the audio and compute FFT at each step
    wav_ptr = 0   # current position in the audio array
    jump_ptr = 0  # current row index in specGram

    while wav_ptr < N - FFT_SIZE:
        # Grab one chunk of audio (FFT_SIZE samples)
        x = audioData[wav_ptr:wav_ptr + FFT_SIZE]

        # Apply Hanning window and run FFT
        # FFT output: complex numbers where |value| = amplitude at each frequency
        yhf = fft(x * h)

        # Take absolute value to get real amplitude (magnitude) at each frequency
        pwr = np.absolute(yhf)

        # Zero out the DC component (bin 0) — it's just the average signal level, not useful
        pwr[0] = 0

        # Normalize this window: divide by its max so loudest frequency = 1.0
        # Each time window gets its own normalization — this highlights the call pattern
        # regardless of whether this part of the recording is quiet or loud
        pwr_max = np.amax(pwr)
        if pwr_max > 0:  # avoid divide-by-zero on silent chunks
            pwr /= pwr_max

        # Copy the useful frequency range into specGram
        # We skip bins 0-3 (below ~170 Hz) and only keep up to FFT_SIZE//4
        # (above ~10 kHz is mostly noise for orca calls which are 1-10 kHz)
        for i in range(FFT_SIZE // 4 - 4):
            specGram[jump_ptr, i] = pwr[i + 4]

        # Advance the window
        wav_ptr += jump
        jump_ptr += 1

        # Safety check: stop if we've filled all rows
        if jump_ptr >= jump_max:
            break

    # Return the completed spectrogram
    # Shape: (jump_max, FFT_SIZE//2 - 4) — e.g. (46, 1020)
    # Values: 0.0 to 1.0
    return specGram


# ─── Batch Process All S_ Files ───────────────────────────────────────────────

# Find every file matching S_*.wav (S_1.wav, S_2.wav, ... S_46.wav etc.)
wav_files = sorted(glob.glob(os.path.join(INPUT_DIR, "S_*.wav")))
print(f"Found {len(wav_files)} S_ wav files in {INPUT_DIR}/")
print()

for wav_path in wav_files:
    # Extract just the filename without extension: "S_19.wav" -> "S_19"
    name = os.path.splitext(os.path.basename(wav_path))[0]

    # Output path: processedFiles/specGram_S_19.npy
    out_path = os.path.join(OUTPUT_DIR, f"specGram_{name}.npy")

    # Generate the spectrogram
    specGram = wav_to_specgram(wav_path)

    # Save as .npy (numpy binary format — fast to load, preserves float64 precision)
    np.save(out_path, specGram)

    print(f"  {name}: shape={specGram.shape}  min={specGram.min():.2f}  max={specGram.max():.2f}  -> saved")

print(f"\nDone! {len(wav_files)} spectrograms saved to {OUTPUT_DIR}/")