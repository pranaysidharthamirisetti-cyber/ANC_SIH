from pathlib import Path
import numpy as np
from config import *
from preprocessing.audio import load_audio
from evaluation.metrics import snr_db,si_snr_db

def main():
    files=sorted(TEST_NOISY.glob("*.wav"))
    if not files:
        raise SystemExit("No test data. Generate it first.")
    vals=[snr_db(load_audio(TEST_CLEAN/f.name,SAMPLE_RATE),
                 load_audio(f,SAMPLE_RATE)) for f in files]
    sis=[si_snr_db(load_audio(TEST_CLEAN/f.name,SAMPLE_RATE),
                   load_audio(f,SAMPLE_RATE)) for f in files]
    print("Mean noisy SNR:",float(np.mean(vals)))
    print("Mean noisy SI-SNR:",float(np.mean(sis)))
    print("Run inference on test data and compare enhanced output to clean reference for final metrics.")

if __name__=="__main__":
    main()
