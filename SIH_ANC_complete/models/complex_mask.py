"""
complex_mask.py
Applies a predicted complex ratio mask (CRM) to the noisy STFT.

The mask is complex-valued (mask = M_r + jM_i) and is applied by ordinary
complex multiplication to the noisy spectrogram X:

    Y = X * M = (X_r*M_r - X_i*M_i) + j(X_r*M_i + X_i*M_r)

Because the CRN's last layer bounds M_r and M_i independently to [-1, 1]
via tanh, |M| can reach up to sqrt(2) -- letting the network slightly boost
energy where useful (e.g. reconstructing a partially masked harmonic),
while still keeping the mask numerically well-behaved.
"""

import torch


def apply_complex_mask(X, mask):
    """
    X:    (B, F, T) complex tensor -- the noisy STFT.
    mask: (B, 2, F, T) real tensor -- channel 0 = M_r, channel 1 = M_i.
    Returns: (B, F, T) complex tensor -- the masked (enhanced) STFT.
    """
    mr, mi = mask[:, 0], mask[:, 1]
    M = torch.complex(mr, mi)
    return X * M
