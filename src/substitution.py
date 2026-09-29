"""Nonlinear substitution: bijective keyed 8-bit S-box (identity for the ablation).

Needed because every other stage is affine over Z_256 (or index-free): without a nonlinear layer the whole cipher
is affine and falls to a chosen-plaintext linear reconstruction (see cryptanalysis.affine_reconstruction_attack).
It is treated as ONE primitive; no strength claim is made from these numbers alone.
"""
import numpy as np


def make_sbox(engine, label, kind="keyed"):
    if kind == "identity":
        i = np.arange(256, dtype=np.uint8); return i, i.copy()
    return engine.sbox(label)


def verify_sbox(S, Sinv):
    """Automatic checks: 256 unique outputs (bijective), exact inverse both ways."""
    S = np.asarray(S).astype(int); Sinv = np.asarray(Sinv).astype(int)
    return dict(bijective=len(set(S.tolist())) == 256 and S.min() == 0 and S.max() == 255,
                inverse_ok=bool((Sinv[S] == np.arange(256)).all() and (S[Sinv] == np.arange(256)).all()))


def _fwht(a):
    a = a.astype(np.int64).copy(); h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h); a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], 1).reshape(-1); h *= 2
    return a


def sbox_properties(S):
    """Differential uniformity (DDT), nonlinearity + max linear bias (Walsh / LAT), SAC matrix deviation."""
    S = np.asarray(S).astype(np.int64); x = np.arange(256)
    du = max(int(np.bincount(S ^ S[x ^ dx], minlength=256).max()) for dx in range(1, 256))
    par = np.array([bin(v).count("1") & 1 for v in range(256)])
    maxw = 0
    for b in range(1, 256):
        f = 1 - 2 * par[S & b]; maxw = max(maxw, int(np.abs(_fwht(f)).max()))
    sac = np.array([[(((S ^ S[x ^ (1 << i)]) >> j) & 1).mean() for j in range(8)] for i in range(8)])
    return dict(differential_uniformity=du, nonlinearity=128 - maxw // 2, max_linear_bias=maxw / 512.0,
                sac_mean=float(sac.mean()), sac_max_dev=float(np.abs(sac - 0.5).max()))
