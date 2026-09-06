import torch

def stft(x, n_fft=512, hop_length=128, win_length=512):
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=torch.hann_window(win_length, device=x.device),
        return_complex=True, center=True
    )

def istft(X, length=None, n_fft=512, hop_length=128, win_length=512):
    return torch.istft(
        X, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=torch.hann_window(win_length, device=X.device),
        length=length, center=True
    )
