"""
crn.py
DCCRN-style complex convolutional-recurrent network for speech-mask estimation.

Interface (unchanged from the original stub, so nothing downstream breaks):
    ComplexCRN(hidden=64)
    forward(x): x is (B, 2, F, T) real-valued  [ch0 = real STFT, ch1 = imag STFT]
                returns a (B, 2, F, T) complex ratio mask in the same layout.

Architecture
------------
Encoder: 4 complex-conv blocks. Each halves the frequency axis only
(stride=(2,1)); the time axis is left untouched so the network can, in
principle, run frame-by-frame on a live stream without buffering more than
one extra frame of time-context per layer.

Bottleneck: a complex LSTM over time, applied to the flattened
(channel x frequency) feature vector of the deepest encoder stage. This is
what gives the network temporal context beyond a single frame.

Decoder: 4 complex transposed-conv blocks that exactly mirror the encoder
(the frequency-axis padding/stride combination is a bijection for any odd
starting frequency size, so encoder and decoder shapes always match with no
cropping needed), with U-Net-style skip connections from the matching
encoder stage. The final block predicts a bounded complex ratio mask (tanh
on both real and imaginary parts).

For the default config (n_fft=512 -> F=257 frequency bins), the encoder
produces frequency sizes 257 -> 129 -> 65 -> 33 -> 17.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Complex building blocks.
#
# A "complex" layer is implemented the standard way (Deep Complex Networks /
# DCCRN): two real-valued layers W_r, W_i stand in for the real and
# imaginary parts of a complex weight W = W_r + jW_i. For complex input
# x = x_r + jx_i:
#     y = W * x = (W_r x_r - W_i x_i) + j(W_r x_i + W_i x_r)
# computed with four real passes through the two real layers.
# ---------------------------------------------------------------------------

class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1, 1), padding=(0, 0)):
        super().__init__()
        self.conv_r = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.conv_r.weight)
        nn.init.xavier_uniform_(self.conv_i.weight)
        nn.init.zeros_(self.conv_r.bias)
        nn.init.zeros_(self.conv_i.bias)

    def forward(self, xr, xi):
        rr, ii = self.conv_r(xr), self.conv_i(xi)
        ri, ir = self.conv_r(xi), self.conv_i(xr)
        return rr - ii, ri + ir


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=(1, 1), padding=(0, 0), output_padding=(0, 0)):
        super().__init__()
        self.conv_r = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        self.conv_i = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        nn.init.xavier_uniform_(self.conv_r.weight)
        nn.init.xavier_uniform_(self.conv_i.weight)
        nn.init.zeros_(self.conv_r.bias)
        nn.init.zeros_(self.conv_i.bias)

    def forward(self, xr, xi):
        rr, ii = self.conv_r(xr), self.conv_i(xi)
        ri, ir = self.conv_r(xi), self.conv_i(xr)
        return rr - ii, ri + ir


class ComplexBatchNorm2d(nn.Module):
    """Naive complex batch-norm: independent real BatchNorm2d on each part.
    (Full covariance-whitening complex BN gives a small extra gain but adds
    real complexity; this simplified version is what most lightweight
    DCCRN reimplementations use, and is fine for an edge-deployed model.)"""
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, xr, xi):
        return self.bn_r(xr), self.bn_i(xi)


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.act_r = nn.PReLU(num_parameters)
        self.act_i = nn.PReLU(num_parameters)

    def forward(self, xr, xi):
        return self.act_r(xr), self.act_i(xi)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_r = nn.Linear(in_features, out_features)
        self.lin_i = nn.Linear(in_features, out_features)

    def forward(self, xr, xi):
        rr, ii = self.lin_r(xr), self.lin_i(xi)
        ri, ir = self.lin_r(xi), self.lin_i(xr)
        return rr - ii, ri + ir


class ComplexLSTM(nn.Module):
    """Complex LSTM built from two real LSTMs, combined with the same
    complex-multiplication rule used above. (An LSTM isn't strictly linear,
    so this is an approximation -- but it's the standard trick used in
    open-source DCCRN implementations and works well in practice.)"""
    def __init__(self, input_size, hidden_size, num_layers=2, batch_first=True):
        super().__init__()
        self.lstm_r = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.lstm_i = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, xr, xi):
        rr, _ = self.lstm_r(xr)
        ii, _ = self.lstm_i(xi)
        ri, _ = self.lstm_r(xi)
        ir, _ = self.lstm_i(xr)
        return rr - ii, ri + ir


# ---------------------------------------------------------------------------
# Encoder / decoder blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Halves the frequency axis (stride=(2,1)); time-axis length unchanged."""
    def __init__(self, in_ch, out_ch, kernel=(5, 3)):
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.conv = ComplexConv2d(in_ch, out_ch, kernel, stride=(2, 1), padding=pad)
        self.bn = ComplexBatchNorm2d(out_ch)
        self.act = ComplexPReLU(out_ch)

    def forward(self, xr, xi):
        xr, xi = self.conv(xr, xi)
        xr, xi = self.bn(xr, xi)
        xr, xi = self.act(xr, xi)
        return xr, xi


class DecoderBlock(nn.Module):
    """Doubles the frequency axis back up (exact inverse of EncoderBlock)."""
    def __init__(self, in_ch, out_ch, kernel=(5, 3), final=False):
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.conv = ComplexConvTranspose2d(in_ch, out_ch, kernel, stride=(2, 1), padding=pad)
        self.final = final
        if not final:
            self.bn = ComplexBatchNorm2d(out_ch)
            self.act = ComplexPReLU(out_ch)

    def forward(self, xr, xi):
        xr, xi = self.conv(xr, xi)
        if self.final:
            xr, xi = torch.tanh(xr), torch.tanh(xi)
        else:
            xr, xi = self.bn(xr, xi)
            xr, xi = self.act(xr, xi)
        return xr, xi


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class ComplexCRN(nn.Module):
    """
    Complex Convolutional-Recurrent Network for time-frequency mask
    estimation. Drop-in replacement for the original stub: same constructor
    signature, same (B, 2, F, T) -> (B, 2, F, T) interface, so it plugs
    straight into models/model.py, training/train.py and
    deployment/export_onnx.py.

    freq_bins must match the STFT config (N_FFT // 2 + 1). Default 257
    matches this project's N_FFT=512.
    """
    def __init__(self, hidden=64, num_encoder_layers=4, rnn_layers=2,
                 rnn_hidden=128, freq_bins=257):
        super().__init__()
        base = max(hidden // 4, 8)
        # e.g. hidden=64 -> channels = [1, 16, 32, 64, 128]
        channels = [1] + [base * (2 ** i) for i in range(num_encoder_layers)]
        self.channels = channels

        self.encoders = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1])
            for i in range(num_encoder_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderBlock(channels[i + 1] * 2, channels[i], final=(i == 0))
            for i in reversed(range(num_encoder_layers))
        ])

        # Work out the bottleneck frequency size analytically so the LSTM
        # (and its input/output projections) can be built eagerly here --
        # not lazily on first forward -- which matters because train.py
        # constructs the optimizer over model.parameters() right after
        # instantiating the model, before any forward pass has happened.
        f = freq_bins
        for _ in range(num_encoder_layers):
            f = (f - 1) // 2 + 1
        self.bottleneck_freq = f
        self.freq_bins = freq_bins

        bottleneck_ch = channels[-1]
        flat = bottleneck_ch * f
        self.bottleneck_ch = bottleneck_ch
        self.proj_in = ComplexLinear(flat, rnn_hidden)
        self.rnn = ComplexLSTM(rnn_hidden, rnn_hidden, rnn_layers)
        self.proj_out = ComplexLinear(rnn_hidden, flat)

    def forward(self, x):
        # x: (B, 2, F, T) real tensor; channel 0 = real part, channel 1 = imag part
        assert x.shape[1] == 2, "ComplexCRN expects input shape (B, 2, F, T)"
        xr, xi = x[:, 0:1], x[:, 1:2]

        skips = []
        for enc in self.encoders:
            xr, xi = enc(xr, xi)
            skips.append((xr, xi))

        b, c, f, t = xr.shape
        if f != self.bottleneck_freq:
            raise ValueError(
                f"Input frequency size {self.freq_bins} does not match the "
                f"size ComplexCRN was built for (got bottleneck f={f}, "
                f"expected {self.bottleneck_freq}). Re-create ComplexCRN "
                f"with the matching freq_bins."
            )

        # (B, C, F, T) -> (B, T, C*F) for the recurrent bottleneck
        zr = xr.permute(0, 3, 1, 2).reshape(b, t, c * f)
        zi = xi.permute(0, 3, 1, 2).reshape(b, t, c * f)
        zr, zi = self.proj_in(zr, zi)
        zr, zi = self.rnn(zr, zi)
        zr, zi = self.proj_out(zr, zi)
        xr = zr.reshape(b, t, c, f).permute(0, 2, 3, 1).contiguous()
        xi = zi.reshape(b, t, c, f).permute(0, 2, 3, 1).contiguous()

        for i, dec in enumerate(self.decoders):
            sr, si = skips[-(i + 1)]
            xr = torch.cat([xr, sr], dim=1)
            xi = torch.cat([xi, si], dim=1)
            xr, xi = dec(xr, xi)

        mask = torch.cat([xr, xi], dim=1)  # (B, 2, F, T)
        return mask
