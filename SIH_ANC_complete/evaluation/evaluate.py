from pathlib import Path
import numpy as np
import torch
from config import *
from preprocessing.audio import load_audio, save_audio
from evaluation.metrics import snr_db,si_snr_db
from inference.offline_test import baseline
from models.model import SpeechEnhancer
from preprocessing.stft import istft, stft

def report(label, clean, estimate):
    return {
        "label": label,
        "snr": float(snr_db(clean, estimate)),
        "si_snr": float(si_snr_db(clean, estimate)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate noisy, baseline, and AI enhancement metrics.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    files=sorted(TEST_NOISY.glob("*.wav"))
    if not files:
        raise SystemExit("No test data. Generate it first.")

    model = None
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = SpeechEnhancer(
            checkpoint.get("hidden", HIDDEN),
            freq_bins=checkpoint.get("freq_bins", N_FFT // 2 + 1),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

    results = {"noisy": [], "baseline": [], "ai": []}
    output_dir = Path(args.output_dir) if args.output_dir else None
    for noisy_path in files:
        clean = load_audio(TEST_CLEAN / noisy_path.name, SAMPLE_RATE)
        noisy = load_audio(noisy_path, SAMPLE_RATE)
        results["noisy"].append(report("noisy", clean, noisy))
        base = baseline(stft(torch.from_numpy(noisy), N_FFT, HOP_LENGTH, WIN_LENGTH))
        enhanced_baseline = istft(
            base, len(noisy), N_FFT, HOP_LENGTH, WIN_LENGTH
        ).numpy()
        results["baseline"].append(report("baseline", clean, enhanced_baseline))

        if model is not None:
            with torch.no_grad():
                enhanced_stft, _ = model(
                    stft(torch.from_numpy(noisy), N_FFT, HOP_LENGTH, WIN_LENGTH).unsqueeze(0)
                )
            enhanced = istft(
                enhanced_stft.squeeze(0), len(noisy), N_FFT, HOP_LENGTH, WIN_LENGTH
            ).numpy()
            results["ai"].append(report("ai", clean, enhanced))
            if output_dir:
                save_audio(output_dir / noisy_path.name, enhanced, SAMPLE_RATE)

    for label in ("noisy", "baseline", "ai"):
        if not results[label]:
            continue
        print(
            f"{label}: "
            f"SNR={np.mean([r['snr'] for r in results[label]]):.2f} dB, "
            f"SI-SNR={np.mean([r['si_snr'] for r in results[label]]):.2f} dB"
        )

if __name__=="__main__":
    main()
