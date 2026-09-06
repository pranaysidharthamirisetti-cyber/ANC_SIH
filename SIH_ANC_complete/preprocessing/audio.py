import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

def load_audio(path, target_sr=16000):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak
    return x

def save_audio(path, x, sr=16000):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)
