# SIH ANC Complete Starter Codebase

Pipeline:
Primary mic -> STFT -> lightweight complex-domain CRN -> iSTFT
Reference mic -> LMS/NLMS residual cancellation -> headset

The neural model architecture is included, but its weights are untrained.
Train it on your clean-speech + defence-noise dataset to create
checkpoints/best.pt.

## Setup
python -m venv .venv
Windows: .venv\\Scripts\\activate
Linux/Raspberry Pi: source .venv/bin/activate
pip install -r requirements.txt

## Dataset
Put clean WAV speech into data/clean_speech/
Put defence/environment noise WAV files into data/defence_noise/

Then:
python -m data.generate_dataset

Train:
python training/train.py

Offline inference:
python inference/offline_test.py --input input/noisy.wav --output output/enhanced.wav --checkpoint checkpoints/best.pt

Baseline without trained AI:
python inference/offline_test.py --input input/noisy.wav --output output/enhanced.wav --baseline

Export the neural mask network:
python deployment/export_onnx.py

Default audio:
16 kHz, 512-point FFT, 512 window, 128 hop.

Real-time audio is hardware/interface dependent. The included realtime.py is a
safe integration skeleton; do not claim real-time performance until actual
audio latency is measured on the target hardware.
