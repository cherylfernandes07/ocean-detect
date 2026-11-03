# normalize, denoise, extract MFCCs + mel spectrogram
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

# This take a filepath and then
# returns a dict with 
# mfccs - the MFCC array
# spectogram - the mel spectrogram (as a base64 image or numpy array)
# sample_rate - so the caller knows what its working with
def preprocess(audio_path: str) -> dict:
# Load the audio with librosia.load
    y, sr = librosa.load(audio_path, sr=None)
# Normalize the audio (peak normalization)
# y — the audio as a numpy array of samples
# sr — the sample rate (using sr=None keeps the original, no resampling)
    y = y / np.max(np.abs(y))
# Deniose with noisereduce. Spectral gating
    y = nr.reduce_noise(y=y, sr=sr)
# Extract MFCCs with librosa.feature.mfcc()
# Gives you 13 coefficients, which is the standard for audio classification
# The result is a 2D array: 13 rows X time frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# Extract mel spectrogram with librosa.feature.melspectrogram()
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return {
        "sample_rate": sr,
        "mfccs": mfccs.tolist(),
        "mel_spectrogram": mel_spec_db.tolist()
    }    

