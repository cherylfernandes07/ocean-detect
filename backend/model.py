import torch
import torch.nn as nn

# Must match train.py
TIME_STEPS = 71    # rows in spectrogram (2.45s at 20kHz with 75% overlap)
FREQ_BINS  = 1020  # columns (FFT bins 4..FFT_SIZE//2)


class OrcaCNN(nn.Module):
    """
    Binary classifier: orca call (1) vs background ocean noise (0).

    Input:  (batch, 1, TIME_STEPS, FREQ_BINS) — spectrogram, one channel
    Output: (batch, 2) — raw logits for [not_orca, orca]

    Three conv+pool blocks shrink the spatial dims by 8x total,
    then two FC layers make the final decision.

    A CNN slides a small window across the 2D spectrogram (like scanning
    sheet music measure by measure) and learns to recognize the shape of
    an orca call wherever it appears — the same way you'd spot a face in
    a photo without needing to see the whole thing at once.
    """

    def __init__(self, time_steps: int = TIME_STEPS, freq_bins: int = FREQ_BINS):
        super().__init__()

        # Block 1: 1 channel → 16 feature maps, learn low-level patterns (edges, blobs)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)   # stabilizes training, acts like per-layer normalization

        # Block 2: 16 → 32, learn mid-level shapes (the arc of a whale call)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # Block 3: 32 → 64, learn high-level call signatures
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Each MaxPool2d(2,2) halves both dimensions → 3 pools = divide by 8
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flattened size after 3 halvings
        t = time_steps // 8   # ~8 rows
        f = freq_bins  // 8   # ~127 cols
        flat_size = 64 * t * f

        self.fc1     = nn.Linear(flat_size, 256)
        self.dropout = nn.Dropout(0.5)   # randomly zeros half the neurons → prevents memorization
        self.fc2     = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # → (batch, 16, T/2, F/2)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # → (batch, 32, T/4, F/4)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # → (batch, 64, T/8, F/8)
        x = x.view(x.size(0), -1)                            # flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)                                    # raw logits [not_orca, orca]


# A CNN (Convolutional Neural Network) is a type of neural network designed to find patterns in grid-like data — originally images, but spectrograms work the same way since they’re just 2D grids of values.

# The key idea is the convolutional layer — instead of looking at the whole image at once, it slides a small window (3x3 in our case) across the image and asks “is there a pattern here?” It does this for every position and builds up a map of where patterns were found.

# Think of it like looking for a whale call shape in the spectrogram the same way you’d look for a face in a photo — you don’t need to see the whole picture at once, you just need to recognize the local shape wherever it appears.

# The layers build on each other:

# Conv1 finds simple patterns (edges, blobs of energy)
# Conv2 combines those into more complex shapes (the curve of a whale call)
# FC layers take all that and make the final yes/no decision