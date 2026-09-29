"""Reversible operator library with a 4-bit per-block configuration  z = (zX, zY, zP, zR).

  zX : X-direction operator  0 = additive chain  c_t = p_t + k_t + c_{t-1}                (linear over Z_256)
                             1 = S-box feedback  c_t = S[p_t + k_t + c_{t-1}]             (nonlinear, full avalanche)
  zY : Y-direction operator, same two choices along the vertical axis
       (opts bx/by = 0 removes the X / Y operator entirely -> used by the X-only / Y-only ablations)
  zB : bidirectional propagation: after each forward chain, a second chain in the REVERSE direction (full row/column coverage)
  zR : triangular RGB coupling  R'=R+S[G+k1], G'=G+S[B+k2], B'=B+S[R'+k3]   (after the chains)
Config index c = zX + 2 zY + 4 zP + 8 zR  (16 configurations / block).

Every stage is c = p + mask(key) (+ known neighbour) mod 256, or a bijection => exact inverse.
Global (field-independent) stages: keyed S-box, then forward+backward chain over the whole image.
"""
import numpy as np

N_BITS, N_CFG = 4, 16
BIT_NAMES = ["zX", "zY", "zB", "zR"]
DEFAULT_OPTS = dict(bx=1, by=1)


def bits_of(c):
    c = np.asarray(c)
    return np.stack([(c >> k) & 1 for k in range(N_BITS)], -1)


def compute_cost(c, opts=DEFAULT_OPTS):
    """Operation count (passes over the block): additive chain = 1, S-box-feedback chain = 2, each reverse pass = 1,
    RGB coupling = 1. Deliberately LINEAR in the bits (a modelling simplification) so that the compute-budget
    constraint is an exact quadratic and the whole objective stays a QUBO."""
    b = bits_of(c)
    return opts["bx"] * (1 + b[..., 0]) + opts["by"] * (1 + b[..., 1]) + b[..., 2] * (opts["bx"] + opts["by"]) + b[..., 3]


def cfg_name(c):
    b = bits_of(c)
    return ("Xn" if b[0] else "Xa") + ("Yn" if b[1] else "Ya") + ("B" if b[2] else "") + ("R" if b[3] else "")


# ------------------------------------------------------------ block reshaping
def to_blocks(a, B):
    H, W = a.shape[:2]
    gh, gw = H // B, W // B
    return a.reshape(gh, B, gw, B, *a.shape[2:]).swapaxes(1, 2).reshape(gh * gw, B, B, *a.shape[2:])


def from_blocks(blk, B, gh, gw):
    return blk.reshape(gh, gw, B, B, *blk.shape[3:]).swapaxes(1, 2).reshape(gh * B, gw * B, *blk.shape[3:])


# ------------------------------------------------------------ chained additive diffusion
def chain_enc(a, K, axis):
    return (np.cumsum(a.astype(np.int64) + K, axis=axis) & 255).astype(np.uint8)


def chain_dec(c, K, axis):
    d = np.diff(c.astype(np.int64), axis=axis, prepend=0)
    return ((d - K) & 255).astype(np.uint8)


def xchain_enc(a, K): return chain_enc(a, K, 2)
def xchain_dec(c, K): return chain_dec(c, K, 2)
def ychain_enc(a, K): return chain_enc(a, K, 1)
def ychain_dec(c, K): return chain_dec(c, K, 1)


def perm_enc(blk, perm):
    n, B = blk.shape[0], blk.shape[1]
    flat = blk.reshape(n, B * B, 3)
    return np.take_along_axis(flat, perm[:, :, None], axis=1).reshape(blk.shape)


def perm_dec(blk, perm):
    return perm_enc(blk, np.argsort(perm, axis=1))


# ------------------------------------------------------------ RGB triangular coupling (works on any (...,3) array)
def rgb_couple(X, S, k1, k2, k3):
    R, G, B = (X[..., c].astype(np.int64) for c in range(3))
    R2 = (R + S[(G + k1) & 255]) & 255
    G2 = (G + S[(B + k2) & 255]) & 255
    B2 = (B + S[(R2 + k3) & 255]) & 255
    return np.stack([R2, G2, B2], -1).astype(np.uint8)


def rgb_uncouple(X, S, k1, k2, k3):
    R2, G2, B2 = (X[..., c].astype(np.int64) for c in range(3))
    B = (B2 - S[(R2 + k3) & 255]) & 255
    G = (G2 - S[(B + k2) & 255]) & 255
    R = (R2 - S[(G + k1) & 255]) & 255
    return np.stack([R, G, B], -1).astype(np.uint8)


# ------------------------------------------------------------ block-local stage, driven by the operator field
def nlchain_enc(a, K, S, axis):
    """S-box feedback chain along `axis` (1=y, 2=x):  c_t = S[(p_t + k_t + c_{t-1}) mod 256]  -- bijective given c_{t-1}."""
    a = np.moveaxis(a.astype(np.int64), axis, 0); K = np.moveaxis(K, axis, 0); out = np.empty_like(a)
    prev = np.zeros_like(a[0])
    for t in range(a.shape[0]):
        prev = S[(a[t] + K[t] + prev) & 255]; out[t] = prev
    return np.moveaxis(out, 0, axis).astype(np.uint8)


def nlchain_dec(c, K, Sinv, axis):
    c = np.moveaxis(c.astype(np.int64), axis, 0); K = np.moveaxis(K, axis, 0); out = np.empty_like(c)
    for t in range(c.shape[0]):
        prevc = c[t - 1] if t else 0
        out[t] = (Sinv[c[t]].astype(np.int64) - K[t] - prevc) & 255
    return np.moveaxis(out, 0, axis).astype(np.uint8)


def _bits(c):
    b = bits_of(int(c)); return bool(b[0]), bool(b[1]), bool(b[2]), bool(b[3])


def _fx(blk, K, S64, axis, nl, rev):
    if rev: blk, K = np.flip(blk, axis), np.flip(K, axis)
    o = nlchain_enc(blk, K, S64, axis) if nl else chain_enc(blk, K, axis)
    return np.flip(o, axis) if rev else o


def _fxd(blk, K, Sinv, axis, nl, rev):
    if rev: blk, K = np.flip(blk, axis), np.flip(K, axis)
    o = nlchain_dec(blk, K, Sinv, axis) if nl else chain_dec(blk, K, axis)
    return np.flip(o, axis) if rev else o


def local_encrypt(Xb, cfgs, R, opts=DEFAULT_OPTS):
    out = Xb.copy(); K = R["K"]; S64 = R["S"].astype(np.int64)
    for c in np.unique(cfgs):
        idx = np.flatnonzero(cfgs == c); zx, zy, zb, rgb = _bits(c); blk = Xb[idx]
        if opts["bx"]:
            blk = _fx(blk, K["kx"][idx], S64, 2, zx, False)
            if zb: blk = _fx(blk, K["kx2"][idx], S64, 2, zx, True)
        if opts["by"]:
            blk = _fx(blk, K["ky"][idx], S64, 1, zy, False)
            if zb: blk = _fx(blk, K["ky2"][idx], S64, 1, zy, True)
        if rgb: blk = rgb_couple(blk, S64, K["c1"][idx], K["c2"][idx], K["c3"][idx])
        out[idx] = blk
    return out


def local_decrypt(Cb, cfgs, R, opts=DEFAULT_OPTS):
    out = Cb.copy(); K = R["K"]; Sinv = R["Sinv"].astype(np.int64); S64 = R["S"].astype(np.int64)
    for c in np.unique(cfgs):
        idx = np.flatnonzero(cfgs == c); zx, zy, zb, rgb = _bits(c); blk = Cb[idx]
        if rgb: blk = rgb_uncouple(blk, S64, K["c1"][idx], K["c2"][idx], K["c3"][idx])
        if opts["by"]:
            if zb: blk = _fxd(blk, K["ky2"][idx], Sinv, 1, zy, True)
            blk = _fxd(blk, K["ky"][idx], Sinv, 1, zy, False)
        if opts["bx"]:
            if zb: blk = _fxd(blk, K["kx2"][idx], Sinv, 2, zx, True)
            blk = _fxd(blk, K["kx"][idx], Sinv, 2, zx, False)
        out[idx] = blk
    return out


# ------------------------------------------------------------ nonlinear substitution (global)
def sbox_enc(X, S, ks):
    return S[(X.astype(np.int64) + ks) & 255]


def sbox_dec(X, Sinv, ks):
    return ((Sinv[X].astype(np.int64) - ks) & 255).astype(np.uint8)


# ------------------------------------------------------------ global bidirectional chain
def global_enc(X, gf, gb):
    v = X.reshape(-1).astype(np.int64)
    c = np.cumsum(v + gf) & 255
    d = np.cumsum((c + gb)[::-1])[::-1] & 255
    return d.astype(np.uint8).reshape(X.shape)


def global_dec(D, gf, gb):
    d = D.reshape(-1).astype(np.int64)
    t = np.diff(d[::-1], prepend=0)[::-1]
    c = (t - gb) & 255
    v = (np.diff(c, prepend=0) - gf) & 255
    return v.astype(np.uint8).reshape(D.shape)


# ------------------------------------------------------------ measured differential-unpredictability table
_CAL = {}


def calibrate(B=16, locs=8, trials=192, seed=12345, opts=DEFAULT_OPTS):
    """Monte-Carlo measurement (image-independent, deterministic, seeded) of how UNPREDICTABLE the block-local
    operator makes the effect of a single-sample change.  For each config c: `locs` random flip locations
    (row y0, col x0, channel ch, non-zero xor-delta); for each, `trials` random (plaintext block, keys, S-box).
    At each output sample the most frequent difference value  max_d P(D_out = d)  is estimated; unpredictability = 1 - mean.
      Dx : samples in the ORIGINAL ROW of the flipped pixel, flipped channel        (X-direction)
      Dy : samples in the ORIGINAL COLUMN                                            (Y-direction)
      Dc : the two OTHER channels, all block positions                               (RGB coupling)
    (an additive chain propagates a CONSTANT difference => 0 unpredictability; S-box feedback => ~uniform difference)
    Also returns quadratic multilinear fits of the supplies in the 4 configuration bits (needed for a QUBO)."""
    key = (B, locs, trials, seed, tuple(sorted(opts.items())))
    if key in _CAL: return _CAL[key]
    import os, pickle
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "calibration_cache_v2_%s.pkl" % "_".join(map(str, key[:4] + tuple(v for _, v in key[4]))))
    if os.path.exists(cache):                                     # deterministic (seeded) => safe to cache on disk
        _CAL[key] = pickle.load(open(cache, "rb")); return _CAL[key]
    rng = np.random.default_rng(seed); n = trials
    acc = {k: np.zeros(N_CFG) for k in ("Dx", "Dy", "Dc")}
    for _ in range(locs):
        S_box = np.stack([rng.permutation(256) for _ in range(n)])                           # per-trial S-box
        K = dict(kx=rng.integers(0, 256, (n, B, B, 3)), ky=rng.integers(0, 256, (n, B, B, 3)),
                 kx2=rng.integers(0, 256, (n, B, B, 3)), ky2=rng.integers(0, 256, (n, B, B, 3)),
                 perm=np.argsort(rng.random((n, B * B)), axis=1), c1=rng.integers(0, 256, (n, B, B)),
                 c2=rng.integers(0, 256, (n, B, B)), c3=rng.integers(0, 256, (n, B, B)))
        P = rng.integers(0, 256, (n, B, B, 3), dtype=np.uint8)
        y0, x0, ch = int(rng.integers(B)), int(rng.integers(B)), int(rng.integers(3)); dl = int(rng.integers(1, 256))
        P2 = P.copy(); P2[:, y0, x0, ch] = (P[:, y0, x0, ch].astype(np.int64) + dl) & 255   # ADDITIVE delta (matches Z_256 chains)
        for c in range(N_CFG):
            cf = np.full(n, c, dtype=np.uint8)
            E1 = np.empty((n, B, B, 3), np.uint8); E2 = np.empty_like(E1)
            for t in range(n):                                                                # per-trial S-box -> loop trials
                Rt = dict(K={k: v[t:t + 1] for k, v in K.items()}, S=S_box[t].astype(np.uint8),
                          Sinv=np.argsort(S_box[t]).astype(np.uint8))
                E1[t] = local_encrypt(P[t:t + 1], cf[:1], Rt, opts)[0]; E2[t] = local_encrypt(P2[t:t + 1], cf[:1], Rt, opts)[0]
            D = (E2.astype(np.int64) - E1) & 255                                              # (n,B,B,3)

            def unpred(sel):                                                                  # sel: (n,m) differences at m samples
                m = sel.shape[1]; cnt = np.zeros((m, 256))
                for v in range(256): cnt[:, v] = (sel == v).sum(0)
                return 1 - float(cnt.max(1).mean() / n)
            acc["Dx"][c] += unpred(D[:, y0, :, ch]); acc["Dy"][c] += unpred(D[:, :, x0, ch])
            oth = [k for k in range(3) if k != ch]
            acc["Dc"][c] += unpred(D[:, :, :, oth].reshape(n, -1))
    out = {k: v / locs for k, v in acc.items()}; out["cost"] = np.array([compute_cost(c, opts) for c in range(N_CFG)], float)
    Z = bits_of(np.arange(N_CFG)).astype(float)
    cols = [np.ones(N_CFG)] + [Z[:, i] for i in range(4)] + [Z[:, i] * Z[:, j] for i in range(4) for j in range(i + 1, 4)]
    A = np.stack(cols, 1); fit = {}
    for k in ("Dx", "Dy", "Dc"):
        y = 1 - out[k]; coef, *_ = np.linalg.lstsq(A, y, rcond=None); res = y - A @ coef
        fit[k] = dict(coef=coef, max_abs_err=float(np.abs(res).max()),
                      r2=float(1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)))
    out["fit"] = fit; _CAL[key] = out
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(out, open(cache, "wb"))
    return out
