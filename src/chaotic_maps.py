"""Keyed multi-lane chaotic byte streams.

Role of chaos here = key-expansion / mask generation ONLY (class A component, not a novelty claim).
Two structurally different maps give the X and Y operators independent randomness sources:
  X-stream : 2D Logistic-Adjusted-Sine style map (2D-LASM form, Hua et al. 2018)
  Y-stream : Chebyshev-sine coupled map (our own composition; NO Lyapunov / hyperchaos claim)
512 parallel lanes, each seeded from SHA-512 output, are iterated in numpy (vectorised).
"""
import numpy as np
from .key_derivation import seed_uniform

LANES = 512
TRANSIENT = 64


def _q(v):
    return (np.floor(np.abs(v) * 4294967296.0).astype(np.uint64) & np.uint64(255)).astype(np.uint8)


CHUNK = 128


def _healthy(T):
    """T: (CHUNK, LANES) float trajectory. A lane-chunk is kept only if it neither collapsed to a fixed point
    nor fell into a short cycle (all samples distinct) and is not squeezed into a tiny interval."""
    s = np.sort(T, axis=0)
    return np.all(np.diff(s, axis=0) > 0, axis=0) & (T.std(axis=0) > 0.05)


def _run(step, x, y, n):
    """Iterate all lanes; emit bytes only from healthy lane-chunks (deterministic, identical at the decoder)."""
    for _ in range(TRANSIENT):
        x, y = step(x, y)
    out, have = [], 0
    while have < n:
        X = np.empty((CHUNK, LANES)); Y = np.empty((CHUNK, LANES))
        for t in range(CHUNK):
            x, y = step(x, y); X[t], Y[t] = x, y
        ok = _healthy(X) & _healthy(Y)
        if ok.any():
            b = np.stack([_q(X[:, ok]), _q(Y[:, ok])], axis=-1).reshape(-1)
            out.append(b); have += b.size
    return np.concatenate(out)[:n]


class ChaosEngine:
    def __init__(self, master):
        self.m = master

    def x_bytes(self, label, n):
        s = seed_uniform(self.m, "X:" + label, 3 * LANES)
        x, y, mu = s[:LANES], s[LANES:2 * LANES], 0.55 + 0.45 * s[2 * LANES:]
        pi = np.pi

        def step(x, y):
            x2 = np.sin(pi * (4 * mu * x * (1 - x) + (1 - mu) * np.sin(pi * y)))
            y2 = np.sin(pi * (4 * mu * y * (1 - y) + (1 - mu) * np.sin(pi * x2)))
            return x2, y2
        return _run(step, x, y, n)

    def y_bytes(self, label, n):
        s = seed_uniform(self.m, "Y:" + label, 4 * LANES)
        u, v = 2 * s[:LANES] - 1, 2 * s[LANES:2 * LANES] - 1
        beta = 0.5 + 0.5 * s[2 * LANES:3 * LANES]
        k = 3 + np.floor(s[3 * LANES:] * 5)
        pi = np.pi

        def step(u, v):
            Tu = np.cos(k * np.arccos(np.clip(u, -1, 1)))
            u2 = np.sin(pi * (beta * Tu + (1 - beta) * np.sin(pi * v)))
            Tv = np.cos(k * np.arccos(np.clip(v, -1, 1)))
            v2 = np.sin(pi * (beta * Tv + (1 - beta) * np.sin(pi * u2)))
            return u2, v2
        return _run(step, u, v, n)

    def sbox(self, label):
        """Bijective 8-bit S-box: argsort of a keyed chaotic vector (Fisher-Yates-equivalent)."""
        vals = self.x_bytes("sbox:" + label, 1024).view("<u4")
        S = np.argsort(vals, kind="stable").astype(np.uint8)
        return S, np.argsort(S, kind="stable").astype(np.uint8)

    def perm_rows(self, label, nb):   # kept for diagnostics only (permutation operator was removed from the bit set)
        """nb independent keyed permutations of 0..255 (block-local pixel permutation)."""
        v = self.y_bytes("perm:" + label, nb * 512).view("<u2").reshape(nb, 256)
        return np.argsort(v, axis=1, kind="stable")

    def random_modes(self, label, nb, allowed):
        b = self.x_bytes("field:" + label, nb)
        return np.asarray(allowed, dtype=np.uint8)[b.astype(np.int64) % len(allowed)]
