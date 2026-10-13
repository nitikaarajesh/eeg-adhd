"""Band-power features: one row of numbers per child."""
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from src.channels import CHANNELS, SFREQ

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

def band_features(x):
    """x: channels x samples in µV -> dict of features."""
    x = mne.filter.filter_data(x, SFREQ, 1.0, 40.0, verbose=False)   # same clean-up as before
    psd, freqs = psd_array_welch(x, SFREQ, fmin=1, fmax=40,
                                 n_fft=512, verbose=False)            # 4 s Welch segments
    total = psd.sum(axis=1)                                           # all power 1-40 Hz, per channel
    power = {b: psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
             for b, (lo, hi) in BANDS.items()}
    feats = {}
    for i, ch in enumerate(CHANNELS):
        for b in BANDS:
            feats[f"{b}_{ch}"] = power[b][i] / total[i]               # relative power (0-1)
        feats[f"tbr_{ch}"] = power["theta"][i] / power["beta"][i]     # theta/beta ratio
    return feats

def window_features(x, win_s=4):
    """Cut one child's recording into win_s-second pieces -> one feature row per piece."""
    x = mne.filter.filter_data(x, SFREQ, 1.0, 40.0, verbose=False)   # clean the whole recording once
    n = SFREQ * win_s                                                 # samples per piece (512)
    n_win = x.shape[1] // n                                           # how many full pieces fit
    w = x[:, :n_win * n].reshape(x.shape[0], n_win, n).transpose(1, 0, 2)   # (pieces, channels, samples)
    psd, freqs = psd_array_welch(w, SFREQ, fmin=1, fmax=40, n_fft=n, verbose=False)
    total = psd.sum(-1)
    power = {b: psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) for b, (lo, hi) in BANDS.items()}
    cols = {}
    for i, ch in enumerate(CHANNELS):
        for b in BANDS:
            cols[f"{b}_{ch}"] = power[b][:, i] / total[:, i]
        cols[f"tbr_{ch}"] = power["theta"][:, i] / power["beta"][:, i]
    return pd.DataFrame(cols)