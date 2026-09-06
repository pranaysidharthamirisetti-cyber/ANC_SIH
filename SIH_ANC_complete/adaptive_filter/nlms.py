import numpy as np


class NLMSFilter:
    def __init__(self, taps=128, step=0.3, eps=1e-8):
        if taps < 1:
            raise ValueError("taps must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.taps = taps
        self.step = step
        self.eps = eps
        self.w = np.zeros(taps, dtype=np.float32)
        self.x = np.zeros(taps, dtype=np.float32)

    def process(self, primary, reference):
        primary = np.asarray(primary, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if primary.ndim != 1 or reference.ndim != 1:
            raise ValueError("primary and reference must be one-dimensional")
        out = np.zeros(len(primary), dtype=np.float32)
        for n in range(len(primary)):
            self.x[1:] = self.x[:-1]
            self.x[0] = reference[n] if n < len(reference) else 0.0
            y = np.dot(self.w, self.x)
            e = primary[n] - y
            out[n] = e
            power = np.dot(self.x, self.x) + self.eps
            self.w += (self.step / power) * e * self.x
        return out
