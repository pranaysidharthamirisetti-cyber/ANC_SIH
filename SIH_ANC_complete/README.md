# SIH ANC AI Noise Cancellation

Pipeline:
Primary mic -> STFT -> lightweight complex-domain CRN -> iSTFT
Reference mic -> LMS/NLMS/RLS residual cancellation -> headset

The CRN is the trainable AI component. Its weights are untrained until you
run the training workflow on clean speech and defence/environment noise.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Put clean WAV speech into `data/clean_speech/` and defence/environment noise
WAV files into `data/defence_noise/`.

```powershell
python -m data.generate_dataset
```

## Train

```powershell
python training/train.py
```

This creates `checkpoints/best.pt`.

## Offline inference

Use the trained model:

```powershell
python inference/offline_test.py --input input/noisy.wav --output output/enhanced.wav --checkpoint checkpoints/best.pt
```

Run the classical spectral baseline without a trained checkpoint:

```powershell
python inference/offline_test.py --input input/noisy.wav --output output/enhanced.wav --baseline
```

Evaluate noisy, classical spectral, and trained-model output:

```powershell
python evaluation/evaluate.py --checkpoint checkpoints/best.pt --output-dir output/test
```

The optional reference microphone can be passed to `offline_test.py` with
`--reference`. The default NLMS canceller can be replaced by the LMS or RLS
implementations in `adaptive_filter/`.

## Deployment

Export the neural mask network after training:

```powershell
python deployment/export_onnx.py
```

The real-time module is a hardware integration skeleton. Measure end-to-end
latency on the target audio interface before claiming real-time performance.

Default audio configuration: 16 kHz sample rate, 512-point FFT, 512-sample
window, and 128-sample hop.
