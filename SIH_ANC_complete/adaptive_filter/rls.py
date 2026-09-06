import numpy as np


class RLSFilter:
    """Recursive least-squares adaptive noise canceller."""

    def __init__(self, taps=128, forgetting_factor=0.999, delta=0.01):
        if taps < 1:
            raise ValueError("taps must be positive")
        if not 0 < forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.taps = taps
        self.forgetting_factor = forgetting_factor
        self.w = np.zeros(taps, dtype=np.float32)
        self.x = np.zeros(taps, dtype=np.float32)
        self.inverse_correlation = np.eye(taps, dtype=np.float32) / delta

    def process(self, primary, reference):
        """Return the primary signal after cancelling correlated reference noise."""
        L = min(len(primary), len(reference))
        out = np.zeros(L, dtype=np.float32)
        lam = self.forgetting_factor

        for n in range(L):
            self.x[1:] = self.x[:-1]
            self.x[0] = reference[n]
            px = self.inverse_correlation @ self.x
            gain = px / (lam + np.dot(self.x, px))
            estimate = np.dot(self.w, self.x)
            error = primary[n] - estimate
            out[n] = error
            self.w += gain * error
            self.inverse_correlation = (
                self.inverse_correlation - np.outer(gain, px)
            ) / lam

        return out
