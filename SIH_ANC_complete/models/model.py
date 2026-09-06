"""
model.py
Top-level speech-enhancement model: noisy STFT in, enhanced STFT out.

    X (B, F, T) complex  --stack real/imag as channels-->  (B, 2, F, T)
        --> ComplexCRN --> complex ratio mask (B, 2, F, T)
        --> apply_complex_mask(X, mask) --> Y (B, F, T) complex

Same constructor/forward signature as the original stub
(`SpeechEnhancer(hidden)`, `forward(X) -> (Y, mask)`), so training/train.py,
training/validation.py, inference/offline_test.py and
deployment/export_onnx.py all work unchanged.
"""

import torch
from torch import nn
from .crn import ComplexCRN
from .complex_mask import apply_complex_mask


class SpeechEnhancer(nn.Module):
    def __init__(self, hidden=64, freq_bins=257):
        super().__init__()
        self.net = ComplexCRN(hidden=hidden, freq_bins=freq_bins)

    def forward(self, X):
        # X: (B, F, T) complex STFT of the noisy signal
        inp = torch.stack([X.real, X.imag], dim=1)  # (B, 2, F, T)
        mask = self.net(inp)
        Y = apply_complex_mask(X, mask)
        return Y, mask
