"""Security / statistical metrics for RGB ciphertexts."""
import time, tracemalloc
import numpy as np
from .feature_extraction import gray, _pearson
from .operators import to_blocks


def entropy(img):
    """Shannon entropy per channel (bits) -> array(3)."""
    out = []
    for c in range(3):
        p = np.bincount(img[..., c].ravel(), minlength=256) / img[..., c].size
        p = p[p > 0]; out.append(float(-(p * np.log2(p)).sum()))
    return np.array(out)


def chi_square(img):
    """chi^2 vs uniform per channel; df=255, critical value 293.25 (alpha=0.05)."""
    out = []
    for c in range(3):
        cnt = np.bincount(img[..., c].ravel(), minlength=256); e = img[..., c].size / 256
        out.append(float(((cnt - e) ** 2 / e).sum()))
    return np.array(out)


def adjacent_corr(img, direction, n=3000, seed=0):
    """Pearson correlation of n random adjacent pixel pairs, per channel."""
    rng = np.random.default_rng(seed); H, W, _ = img.shape
    dy, dx = {"h": (0, 1), "v": (1, 0), "d": (1, 1)}[direction]
    ys = rng.integers(0, H - dy, n); xs = rng.integers(0, W - dx, n)
    out = []
    for c in range(3):
        a = img[ys, xs, c].astype(np.float64); b = img[ys + dy, xs + dx, c].astype(np.float64)
        out.append(float(_pearson(a[None], b[None])[0]) if a.std() > 0 and b.std() > 0 else 1.0)
    return np.array(out)


def cross_channel_corr(img):
    f = [img[..., c].reshape(1, -1).astype(np.float64) for c in range(3)]
    return dict(RG=float(_pearson(f[0], f[1])[0]), GB=float(_pearson(f[1], f[2])[0]),
                BR=float(_pearson(f[2], f[0])[0]))


def npcr_uaci(a, b):
    d = a != b
    npcr = 100.0 * d.mean()
    uaci = 100.0 * (np.abs(a.astype(np.float64) - b.astype(np.float64)) / 255.0).mean()
    return float(npcr), float(uaci)


def local_residual(img, B=16):
    """Block-level residual |corr| in the CIPHERTEXT (h and v): mean and worst block."""
    L = to_blocks(np.repeat(gray(img)[..., None], 3, -1), B)[..., 0]; nb = L.shape[0]
    ch = _pearson(L[:, :, :-1].reshape(nb, -1), L[:, :, 1:].reshape(nb, -1))
    cv = _pearson(L[:, :-1, :].reshape(nb, -1), L[:, 1:, :].reshape(nb, -1))
    a = np.abs(np.concatenate([ch, cv]))
    return dict(mean=float(a.mean()), max=float(a.max()))


def key_space_bits(key):
    """Honest key space: derivation is SHA-512 (512-bit master), but the *user key* string bounds
    the search space. Reported as min(8*len(key) upper bound on entropy, 512)."""
    kb = key.encode() if isinstance(key, str) else bytes(key)
    return dict(nominal_master_bits=512, user_key_max_entropy_bits=min(8 * len(kb), 512))


def measure_memory(fn, *a, **k):
    tracemalloc.start(); t0 = time.perf_counter()
    r = fn(*a, **k); dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return r, dt, peak / 1e6


def summarize(img_c, img_p=None, B=16):
    """Statistical metrics for one ciphertext (mean over R,G,B)."""
    H = entropy(img_c); X = chi_square(img_c)
    h, v, d = (adjacent_corr(img_c, k) for k in "hvd")
    cc = cross_channel_corr(img_c); lr = local_residual(img_c, B)
    out = dict(entropy_R=H[0], entropy_G=H[1], entropy_B=H[2], entropy_mean=float(H.mean()),
               chi2_mean=float(X.mean()), chi2_pass=bool((X < 293.25).all()),
               corr_h=float(h.mean()), corr_v=float(v.mean()), corr_d=float(d.mean()),
               corr_RG=cc["RG"], corr_GB=cc["GB"], corr_BR=cc["BR"],
               local_resid_mean=lr["mean"], local_resid_max=lr["max"])
    if img_p is not None:
        out.update(plain_corr_h=float(adjacent_corr(img_p, "h").mean()),
                   plain_corr_v=float(adjacent_corr(img_p, "v").mean()),
                   plain_entropy=float(entropy(img_p).mean()))
    return out


def mutual_information(a, b, bins=16, bias_correct=True):
    """Mutual information (bits) between two uint8 vectors on a bins x bins joint histogram (top log2(bins) bits of each value).
    256x256 bins are unusable for a 128x128 image: the plug-in bias (K-1)^2/(2N ln2) is ~2.9 bits and the correction over-shoots
    (negative values were observed in the first full run). 16x16 bins: bias ~0.01 bit at N=16384."""
    sh = 8 - int(np.log2(bins)); n = a.size
    j = np.bincount((a >> sh).astype(np.int64) * bins + (b >> sh), minlength=bins * bins).reshape(bins, bins) / n
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True); m = j > 0
    mi = float((j[m] * np.log2(j[m] / (pa @ pb)[m])).sum())
    if bias_correct: mi -= ((pa > 0).sum() - 1) * ((pb > 0).sum() - 1) / (2 * n * np.log(2))
    return mi


def channel_mi(img):
    f = [img[..., c].ravel() for c in range(3)]
    return dict(MI_RG=mutual_information(f[0], f[1]), MI_GB=mutual_information(f[1], f[2]), MI_BR=mutual_information(f[2], f[0]))
