"""HiFi-GAN-compatible log-mel, and WORLD pitch on the same frame timeline."""
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from .common import SR, HOP, NFFT, NMELS, check


def load_audio(path):
    y, sr = sf.read(path, dtype='float32')
    check(sr == SR and y.ndim == 1 and np.isfinite(y).all(), f'잘못된 WAV: {path}')
    check(len(y) >= NFFT, '음성이 너무 짧습니다.')
    return y


def mel_spectrogram(y):
    # No per-utterance peak normalization: preserve emotion-related energy differences.
    y = torch.as_tensor(y, dtype=torch.float32).reshape(1, -1)
    y = F.pad(y.unsqueeze(1), ((NFFT-HOP)//2, (NFFT-HOP)//2), mode='reflect').squeeze(1)
    spec = torch.stft(y, NFFT, hop_length=HOP, win_length=NFFT,
                      window=torch.hann_window(NFFT), center=False, return_complex=True)
    magnitude = (spec.abs().square() + 1e-9).sqrt()
    basis = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=NMELS, fmin=0, fmax=8000))
    return torch.log(torch.clamp(basis @ magnitude, min=1e-5)).squeeze(0).T.numpy()


def frame_prosody(y, frames):
    import pyworld
    signal = np.asarray(y, dtype=np.float64)
    f0, t = pyworld.dio(signal, SR, f0_floor=50, f0_ceil=600, frame_period=HOP/SR*1000)
    f0 = pyworld.stonemask(signal, f0, t, SR)
    # HiFi-GAN reflect padding puts each STFT frame center at i*hop + hop/2.
    centers = (np.arange(frames)*HOP + HOP/2)/SR
    index = np.clip(np.rint(centers/(HOP/SR)).astype(int), 0, len(f0)-1)
    pitch = f0[index]
    padded = np.pad(y, ((NFFT-HOP)//2, (NFFT-HOP)//2), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, NFFT)[::HOP][:frames]
    energy = np.sqrt(np.mean(windows.astype(np.float64)**2, axis=1) + 1e-12)
    check(len(energy) == frames, 'energy 프레임 불일치')
    return pitch, energy
