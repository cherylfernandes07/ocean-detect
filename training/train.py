#!/usr/bin/env python3
"""
train.py — OrcaCNN Binary Classifier Training

Trains a CNN to detect Southern Resident Killer Whale (SRKW) calls.
Binary classification: orca call (1) vs background ocean noise (0).

Dataset format (Orcasound acoustic-sandbox):
  - TSV file: columns (filename, start, duration_s, ..., label)
    * Each row = one labeled orca segment (positive example)
    * Gaps between labeled segments = negative examples (auto-extracted)
  - WAV directory: audio files at 20kHz sample rate

Usage:
  python train.py --wav-dir ./wavs --tsv podcast2.tsv --epochs 20 --out model.pt
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from scipy.fftpack import fft
from scipy.signal.windows import hann

# ─── Spectrogram Parameters ────────────────────────────────────────────────────
# Tuned for 20kHz audio (Orcasound hydrophone recordings, downsampled)
# Original Val Veirs notebook used 44.1kHz; we scale accordingly.

SAMPLE_RATE = 20000      # Hz — Orcasound training data is downsampled to 20kHz
WINDOW_DURATION = 2.45   # seconds per training window (matches Pod.Cast baseline)
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # 49,000 samples

FFT_SIZE = 2048          # samples per FFT analysis window (~102ms at 20kHz)
OVERLAP_PCT = 75         # percent overlap between FFT windows
JUMP = int(FFT_SIZE * (100 - OVERLAP_PCT) / OVERLAP_PCT)  # = 683 samples forward

# Frequency bins: skip first 4 (DC + sub-~40Hz noise), keep up to FFT_SIZE//4
# At 20kHz: FFT_SIZE//4 = 512 bins covers up to 5kHz (enough for SRKW calls 1-4kHz)
FREQ_BINS = FFT_SIZE // 2 - 4   # = 1020 bins

# Time steps per window (rows in spectrogram)
TIME_STEPS = WINDOW_SAMPLES // JUMP  # ~71 rows for 2.45s at 20kHz

# Final spectrogram shape per sample: (1, TIME_STEPS, FREQ_BINS)
# The "1" is the channel dimension expected by Conv2d

MIN_SEGMENT_DURATION = 0.5  # seconds — skip segments shorter than this
NEG_SEGMENT_DURATION = 2.45  # seconds — chop negatives into this size


# ─── Spectrogram Generator ─────────────────────────────────────────────────────

def audio_to_specgram(audio: np.ndarray, target_rows: int = TIME_STEPS) -> np.ndarray:
    """
    Convert a 1D audio array to a 2D spectrogram.

    Args:
        audio: 1D float array of audio samples (already at SAMPLE_RATE)
        target_rows: number of time rows to output (pad/trim as needed)

    Returns:
        specGram: shape (target_rows, FREQ_BINS), values 0.0–1.0
    """
    N = len(audio)
    h = hann(FFT_SIZE)
    specGram = np.zeros([target_rows, FREQ_BINS], dtype=np.float32)

    wav_ptr = 0
    row = 0

    while wav_ptr + FFT_SIZE <= N and row < target_rows:
        chunk = audio[wav_ptr: wav_ptr + FFT_SIZE]

        # Apply Hanning window and FFT
        yhf = fft(chunk * h)
        pwr = np.abs(yhf)
        pwr[0] = 0  # zero DC term

        # Per-window normalization: loudest frequency = 1.0
        pwr_max = pwr.max()
        if pwr_max > 0:
            pwr /= pwr_max

        # Store bins 4..FFT_SIZE//2 (skip sub-~40Hz, keep up to ~5kHz)
        specGram[row, :] = pwr[4: FFT_SIZE // 2]

        wav_ptr += JUMP
        row += 1

    return specGram


# ─── Dataset Loader ─────────────────────────────────────────────────────────────

class OrcaDataset(Dataset):
    """
    Loads positive (orca) and negative (background) audio segments.

    Positive examples: segments from TSV label file
    Negative examples: gaps between labeled segments in the same WAV files,
                       plus any WAV files not appearing in the TSV

    TSV format expected: tab-separated with header row
      filename  start  duration_s  location  date  data_source  data_source_id  label
    """

    def __init__(self, wav_dir: str, tsv_path: str, augment: bool = False):
        self.wav_dir = wav_dir
        self.augment = augment
        self.samples = []          # list of (specgram, label)
        self.index = []            # list of (specgram, label, fname) for file-level splitting
        self.samples_by_file = {}  # fname -> list of indices into self.samples

        # Parse TSV: group labeled segments by filename
        labels_by_file = {}  # filename -> list of (start_s, duration_s)
        with open(tsv_path) as f:
            header = f.readline().strip().split('\t')
            # Find column indices (handle different TSV layouts)
            try:
                fn_col  = header.index('filename')
                st_col  = header.index('start')
                dur_col = header.index('duration_s')
            except ValueError:
                # Fall back to positional columns 0,1,2
                fn_col, st_col, dur_col = 0, 1, 2

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= max(fn_col, st_col, dur_col):
                    continue
                fname   = parts[fn_col]
                start   = float(parts[st_col])
                dur     = float(parts[dur_col])
                labels_by_file.setdefault(fname, []).append((start, dur))

        # Process each labeled WAV file
        print(f"Found {len(labels_by_file)} labeled WAV files in TSV")
        total_pos = 0
        total_neg = 0

        for fname, segs in labels_by_file.items():
            wav_path = os.path.join(wav_dir, fname)
            if not os.path.exists(wav_path):
                print(f"  WARNING: {fname} not found in {wav_dir}, skipping")
                continue

            audio, sr = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # mono

            # Normalize amplitude
            wav_max = np.abs(audio).max()
            if wav_max > 0:
                audio = audio / (wav_max * 2.0)

            total_samples = len(audio)
            total_duration = total_samples / sr

            # ── Positive examples: labeled orca segments ──
            segs_sorted = sorted(segs, key=lambda x: x[0])
            for start_s, dur_s in segs_sorted:
                if dur_s < MIN_SEGMENT_DURATION:
                    continue  # skip tiny labels (sometimes used as "no calls" markers)
                start_i = int(start_s * sr)
                end_i   = int((start_s + dur_s) * sr)
                end_i   = min(end_i, total_samples)
                clip = audio[start_i:end_i]
                if len(clip) < WINDOW_SAMPLES:
                    # Pad short clips with zeros
                    clip = np.pad(clip, (0, WINDOW_SAMPLES - len(clip)))
                else:
                    clip = clip[:WINDOW_SAMPLES]
                specgram = audio_to_specgram(clip)
                idx = len(self.samples)
                self.samples.append((specgram, 1))
                self.index.append((specgram, 1, fname))
                self.samples_by_file.setdefault(fname, []).append(idx)
                total_pos += 1

            # ── Negative examples: gaps between labeled segments ──
            # Build a timeline of labeled intervals, then extract the gaps
            gap_regions = []
            prev_end = 0.0
            for start_s, dur_s in segs_sorted:
                if start_s > prev_end + MIN_SEGMENT_DURATION:
                    gap_regions.append((prev_end, start_s))
                prev_end = max(prev_end, start_s + dur_s)
            # Also add the tail after the last call
            if prev_end + MIN_SEGMENT_DURATION < total_duration:
                gap_regions.append((prev_end, total_duration))

            # Chop each gap into NEG_SEGMENT_DURATION windows
            for gap_start, gap_end in gap_regions:
                pos = gap_start
                while pos + NEG_SEGMENT_DURATION <= gap_end:
                    start_i = int(pos * sr)
                    end_i   = int((pos + NEG_SEGMENT_DURATION) * sr)
                    clip = audio[start_i:end_i]
                    if len(clip) < WINDOW_SAMPLES:
                        clip = np.pad(clip, (0, WINDOW_SAMPLES - len(clip)))
                    else:
                        clip = clip[:WINDOW_SAMPLES]
                    specgram = audio_to_specgram(clip)
                    idx = len(self.samples)
                    self.samples.append((specgram, 0))
                    self.index.append((specgram, 0, fname))
                    self.samples_by_file.setdefault(fname, []).append(idx)
                    total_neg += 1
                    pos += NEG_SEGMENT_DURATION

        print(f"Dataset: {total_pos} positive (orca), {total_neg} negative (background)")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        specgram, label = self.samples[idx]
        # Add channel dimension: (TIME_STEPS, FREQ_BINS) -> (1, TIME_STEPS, FREQ_BINS)
        x = torch.tensor(specgram, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        if self.augment and label == 1:
            # Simple data augmentation for positive class:
            # Add small amount of Gaussian noise
            x = x + torch.randn_like(x) * 0.02
            x = x.clamp(0, 1)

        return x, y


# ─── Model Definition ──────────────────────────────────────────────────────────

class OrcaCNN(nn.Module):
    """
    Convolutional neural network for binary orca call detection.

    Input: (batch, 1, TIME_STEPS, FREQ_BINS) — spectrogram with channel dim
    Output: (batch, 2) — logits for [not-orca, orca]

    Architecture: two conv+pool blocks -> flatten -> two FC layers
    """

    def __init__(self, time_steps: int = TIME_STEPS, freq_bins: int = FREQ_BINS):
        super().__init__()

        # Block 1: 1->16 channels, halves spatial dims
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)

        # Block 2: 16->32 channels, halves again
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # Block 3: 32->64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Compute flattened size after 3 pool operations
        t = time_steps // 8   # 3 halving ops
        f = freq_bins  // 8
        flat_size = 64 * t * f

        self.fc1     = nn.Linear(flat_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(wav_dir, tsv_path, epochs=20, batch_size=16, lr=1e-3, out_path="model.pt",
          val_split=0.15):
    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = OrcaDataset(wav_dir, tsv_path, augment=True)

    if len(full_dataset) == 0:
        print("ERROR: No samples loaded. Check wav_dir and tsv_path.")
        sys.exit(1)

    # Train/val split — MUST be by WAV file, not by sample.
    # If we split randomly, samples from the same WAV end up in both train and val,
    # and the model memorizes the recording instead of learning call patterns.
    all_files = list(full_dataset.samples_by_file.keys())
    random.shuffle(all_files)
    n_val_files = max(1, int(len(all_files) * val_split))
    val_files  = set(all_files[:n_val_files])
    train_files = set(all_files[n_val_files:])

    train_indices = [i for i, (_, _, fname) in enumerate(full_dataset.index) if fname in train_files]
    val_indices   = [i for i, (_, _, fname) in enumerate(full_dataset.index) if fname in val_files]

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_indices)} samples  |  Val: {len(val_indices)} samples")

    # Model, loss, optimizer
    model = OrcaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {epochs} epochs...\n")

    best_val_acc = 0.0
    best_f1 = 0.0
    best_f1_path = out_path.replace(".pt", "_best_f1.pt")

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)
                # Track TP/TN/FP/FN for precision/recall
                for pred, true in zip(preds.cpu(), y.cpu()):
                    if true == 1 and pred == 1: tp += 1
                    elif true == 0 and pred == 0: tn += 1
                    elif true == 0 and pred == 1: fp += 1
                    else: fn += 1

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        avg_train_loss = train_loss / train_total
        avg_val_loss   = val_loss   / val_total

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss={avg_train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={avg_val_loss:.4f} acc={val_acc:.3f} | "
              f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'f1': f1,
            'time_steps': TIME_STEPS,
            'freq_bins': FREQ_BINS,
        }

        # Save best accuracy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, out_path)
            print(f"  ✓ Saved best-acc model (val_acc={val_acc:.3f})")

        # Save best F1 model separately — F1 balances precision and recall,
        # so this is often the more useful model for detection tasks
        if f1 > best_f1:
            best_f1 = f1
            torch.save(checkpoint, best_f1_path)
            print(f"  ✓ Saved best-f1  model (F1={f1:.3f}, P={precision:.3f}, R={recall:.3f})")

    print(f"\nTraining complete.")
    print(f"  Best val_acc={best_val_acc:.3f} → {out_path}")
    print(f"  Best F1={best_f1:.3f}      → {best_f1_path}")
    print(f"\nTip: use the best-f1 model for detection (balances recall and precision)")


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OrcaCNN binary classifier")
    parser.add_argument("--wav-dir",    required=True,         help="Directory of WAV files")
    parser.add_argument("--tsv",        required=True,         help="TSV label file path")
    parser.add_argument("--epochs",     type=int, default=20,  help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,  help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out",        default="model.pt",    help="Output model path")
    args = parser.parse_args()

    train(
        wav_dir    = args.wav_dir,
        tsv_path   = args.tsv,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        out_path   = args.out,
    )