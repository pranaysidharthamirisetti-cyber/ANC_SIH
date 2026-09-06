import argparse
import torch
from pathlib import Path
from config import *
from preprocessing.audio import load_audio,save_audio
from preprocessing.stft import stft,istft
from inference.offline_test import baseline

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    x=load_audio(a.input,SAMPLE_RATE)
    X=stft(torch.from_numpy(x),N_FFT,HOP_LENGTH,WIN_LENGTH)
    y=istft(baseline(X),len(x),N_FFT,HOP_LENGTH,WIN_LENGTH).numpy()
    save_audio(Path(a.output),y,SAMPLE_RATE)

if __name__=="__main__":
    main()
