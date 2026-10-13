"""Band-power features: one row of numbers per subject."""
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

def window_features(x, win_s=4, max_seconds=None):
    """Cut one subject's recording into win_s-second segments -> one feature row per segment.

    max_seconds: if given, keep only the first max_seconds of the recording before
    windowing, so every subject yields the same number of windows (e.g. 60 s -> 15 x 4 s).
    """
    if max_seconds is not None:
        x = x[:, :int(max_seconds * SFREQ)]                          # crop to the first max_seconds
    x = mne.filter.filter_data(x, SFREQ, 1.0, 40.0, verbose=False)   # clean the whole recording once
    n = SFREQ * win_s                                                 # samples per segment (512)
    n_win = x.shape[1] // n                                           # how many full segments fit
    w = x[:, :n_win * n].reshape(x.shape[0], n_win, n).transpose(1, 0, 2)   # (segments, channels, samples)
    psd, freqs = psd_array_welch(w, SFREQ, fmin=1, fmax=40, n_fft=n, verbose=False)
    total = psd.sum(-1)
    power = {b: psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) for b, (lo, hi) in BANDS.items()}
    cols = {}
    for i, ch in enumerate(CHANNELS):
        for b in BANDS:
            cols[f"{b}_{ch}"] = power[b][:, i] / total[:, i]
        cols[f"tbr_{ch}"] = power["theta"][:, i] / power["beta"][:, i]
    return pd.DataFrame(cols)