import argparse
from pathlib import Path
import torch
from config import *
from preprocessing.audio import load_audio, save_audio
from preprocessing.stft import stft, istft
from models.model import SpeechEnhancer
from adaptive_filter.nlms import NLMSFilter

def baseline(X, strength=0.6):
    mag=torch.abs(X)
    phase=torch.angle(X)
    noise=torch.quantile(mag,0.2,dim=-1,keepdim=True)
    gain=torch.clamp((mag-strength*noise)/(mag+1e-8),0.05,1.0)
    return gain*mag*torch.exp(1j*phase)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--checkpoint",default=str(CHECKPOINT_PATH))
    p.add_argument("--baseline",action="store_true")
    p.add_argument("--reference",default="")
    a=p.parse_args()

    x=load_audio(a.input,SAMPLE_RATE)
    X=stft(torch.from_numpy(x),N_FFT,HOP_LENGTH,WIN_LENGTH)
    n_fft, hop_length, win_length = N_FFT, HOP_LENGTH, WIN_LENGTH

    if a.baseline or not Path(a.checkpoint).exists():
        Y=baseline(X)
        print("Using classical spectral baseline.")
    else:
        ckpt=torch.load(a.checkpoint,map_location="cpu")
        n_fft = ckpt.get("n_fft", N_FFT)
        hop_length = ckpt.get("hop_length", HOP_LENGTH)
        win_length = ckpt.get("win_length", WIN_LENGTH)
        if (n_fft, hop_length, win_length) != (N_FFT, HOP_LENGTH, WIN_LENGTH):
            X=stft(torch.from_numpy(x),n_fft,hop_length,win_length)
        model=SpeechEnhancer(
            ckpt.get("hidden", HIDDEN),
            freq_bins=ckpt.get("freq_bins", n_fft // 2 + 1),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            Y,_=model(X.unsqueeze(0))
        Y=Y.squeeze(0)

    y=istft(Y,len(x),n_fft,hop_length,win_length).numpy()

    if a.reference:
        ref=load_audio(a.reference,SAMPLE_RATE)
        y=NLMSFilter(NLMS_TAPS,NLMS_STEP,NLMS_EPS).process(y,ref)

    save_audio(Path(a.output),y,SAMPLE_RATE)

if __name__=="__main__":
    main()
