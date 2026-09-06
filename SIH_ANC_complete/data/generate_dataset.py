from pathlib import Path
import random
import numpy as np
from preprocessing.audio import load_audio, save_audio
from config import *

random.seed(42)
np.random.seed(42)

SNR_DB = [-5, 0, 5, 10, 15, 20]
FILES_PER_SPLIT = {"train": 200, "val": 40, "test": 40}

def rms(x):
    return float(np.sqrt(np.mean(x*x) + 1e-12))

def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean)/len(noise))))
    start = random.randint(0, max(0, len(noise)-len(clean)))
    noise = noise[start:start+len(clean)]
    desired = rms(clean) / (10 ** (snr_db / 20))
    noise = noise * (desired / max(rms(noise), 1e-12))
    mixed = clean + noise
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed *= 0.99 / peak
    return mixed.astype(np.float32)

def main():
    clean_files = list(CLEAN_DIR.glob("*.wav"))
    noise_files = list(NOISE_DIR.glob("*.wav"))
    if not clean_files or not noise_files:
        raise SystemExit("Put WAV files into data/clean_speech and data/defence_noise first.")

    for split, count in FILES_PER_SPLIT.items():
        noisy_dir = DATA / "generated_dataset" / split / "noisy"
        clean_dir = DATA / "generated_dataset" / split / "clean"
        noisy_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            clean = load_audio(random.choice(clean_files), SAMPLE_RATE)
            noise = load_audio(random.choice(noise_files), SAMPLE_RATE)

            max_len = min(len(clean), SAMPLE_RATE * 4)
            if len(clean) > max_len:
                s = random.randint(0, len(clean)-max_len)
                clean = clean[s:s+max_len]
            else:
                clean = clean[:max_len]

            if len(clean) < SAMPLE_RATE:
                continue

            noisy = mix_at_snr(clean, noise, random.choice(SNR_DB))
            name = f"{split}_{i:05d}.wav"
            save_audio(noisy_dir / name, noisy, SAMPLE_RATE)
            save_audio(clean_dir / name, clean, SAMPLE_RATE)

        print(f"{split}: generated {count} examples")

if __name__ == "__main__":
    main()
