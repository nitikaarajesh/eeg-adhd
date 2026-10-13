"""EEGNet (Lawhern et al., 2018): a compact CNN for EEG."""
import torch
from torch import nn

class EEGNet(nn.Module):
    def __init__(self, n_ch=19, n_samples=512, F1=8, D=2, F2=16, k=64, drop=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # 1) temporal filters: learn frequency patterns (like our bands, but learned)
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(F1),
            # 2) spatial filters: learn which electrodes to combine
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            # 3) summarise patterns over time
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_out = self.features(torch.zeros(1, 1, n_ch, n_samples)).shape[1]
        self.classify = nn.Linear(n_out, 2)

    def forward(self, x):                 # x: (batch, 1, channels, samples)
        return self.classify(self.features(x))