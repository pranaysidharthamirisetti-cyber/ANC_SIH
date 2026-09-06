from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from preprocessing.audio import load_audio
from preprocessing.stft import stft
from config import *

class PairedAudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy = sorted(Path(noisy_dir).glob("*.wav"))
        self.clean_dir = Path(clean_dir)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        npth = self.noisy[idx]
        cpth = self.clean_dir / npth.name
        n = load_audio(npth, SAMPLE_RATE)
        c = load_audio(cpth, SAMPLE_RATE)
        L = min(len(n), len(c))
        X = stft(torch.from_numpy(n[:L]), N_FFT, HOP_LENGTH, WIN_LENGTH)
        S = stft(torch.from_numpy(c[:L]), N_FFT, HOP_LENGTH, WIN_LENGTH)
        return X, S


def collate_fn(batch):
    """
    Clips in generated_dataset/ can end up with slightly different lengths
    (generate_dataset.py allows anything from 1s up to 4s), which produces
    STFT tensors with different numbers of time frames. The default
    DataLoader collate can't stack tensors of different shapes, so this
    pads every clip in the batch to the batch's longest time length with
    zeros (silence) before stacking into (B, F, T) complex tensors.
    """
    Xs, Ss = zip(*batch)
    t_max = max(x.shape[-1] for x in Xs)
    Xp = [F.pad(x, (0, t_max - x.shape[-1])) for x in Xs]
    Sp = [F.pad(s, (0, t_max - s.shape[-1])) for s in Ss]
    return torch.stack(Xp), torch.stack(Sp)
