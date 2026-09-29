"""Per-block features F_ij = [entropy, corr_x, corr_y, corr_diag, edge, std, corr_RG, corr_GB, corr_BR]
and boundary correlations between neighbouring blocks (used for the pairwise QUBO term)."""
import numpy as np
from .operators import to_blocks

NAMES = ["entropy", "corr_x", "corr_y", "corr_diag", "edge", "std",
         "corr_RG", "corr_GB", "corr_BR"]


def _pearson(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    num = (a * b).sum(-1)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1.0)   # flat -> fully redundant


def gray(img):
    f = img.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def block_features(img, B):
    H, W, _ = img.shape
    gh, gw = H // B, W // B
    blk = to_blocks(img, B)                                   # (nb,B,B,3)
    nb = blk.shape[0]
    L = to_blocks(np.repeat(gray(img)[..., None], 3, -1), B)[..., 0]   # (nb,B,B)
    # entropy of all 3*B*B samples of the block, normalised by 8 bits
    vals = blk.reshape(nb, -1).astype(np.int64)
    cnt = np.bincount((np.arange(nb)[:, None] * 256 + vals).ravel(), minlength=nb * 256).reshape(nb, 256)
    p = cnt / vals.shape[1]
    ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1) / 8.0
    cx = _pearson(L[:, :, :-1].reshape(nb, -1), L[:, :, 1:].reshape(nb, -1))
    cy = _pearson(L[:, :-1, :].reshape(nb, -1), L[:, 1:, :].reshape(nb, -1))
    cd = _pearson(L[:, :-1, :-1].reshape(nb, -1), L[:, 1:, 1:].reshape(nb, -1))
    gx = np.abs(np.diff(L, axis=2))[:, :-1, :]; gy = np.abs(np.diff(L, axis=1))[:, :, :-1]
    edge = ((gx + gy) > 32).reshape(nb, -1).mean(1)
    std = np.clip(L.reshape(nb, -1).std(1) / 64.0, 0, 1)
    ch = [blk[..., c].reshape(nb, -1).astype(np.float64) for c in range(3)]
    crg, cgb, cbr = _pearson(ch[0], ch[1]), _pearson(ch[1], ch[2]), _pearson(ch[2], ch[0])
    F = np.stack([ent, cx, cy, cd, edge, std, crg, cgb, cbr], -1)
    return F.reshape(gh, gw, 9)


def boundary_corr(img, B):
    """|Pearson| across the shared boundary of adjacent blocks: rho_x (gh,gw-1), rho_y (gh-1,gw)."""
    L = gray(img); H, W = L.shape; gh, gw = H // B, W // B
    rho_x = np.ones((gh, max(gw - 1, 0))); rho_y = np.ones((max(gh - 1, 0), gw))
    if gw > 1:
        cols = np.arange(1, gw) * B
        a = np.concatenate([L[:, cols - 1], L[:, cols - 2]], 0) if B > 1 else L[:, cols - 1]
        b = np.concatenate([L[:, cols], L[:, np.minimum(cols + 1, W - 1)]], 0)
        # reshape to (2*gh, B, gw-1) -> per boundary samples over 2 pairs x B rows
        a = a.reshape(2, gh, B, gw - 1).transpose(1, 3, 0, 2).reshape(gh, gw - 1, -1)
        b = b.reshape(2, gh, B, gw - 1).transpose(1, 3, 0, 2).reshape(gh, gw - 1, -1)
        rho_x = np.abs(_pearson(a, b))
    if gh > 1:
        rows = np.arange(1, gh) * B
        a = np.concatenate([L[rows - 1, :], L[rows - 2, :]], 0)
        b = np.concatenate([L[rows, :], L[np.minimum(rows + 1, H - 1), :]], 0)
        a = a.reshape(2, gh - 1, gw, B).transpose(1, 2, 0, 3).reshape(gh - 1, gw, -1)
        b = b.reshape(2, gh - 1, gw, B).transpose(1, 2, 0, 3).reshape(gh - 1, gw, -1)
        rho_y = np.abs(_pearson(a, b))
    return rho_x, rho_y


def risk(F):
    """Parameter-free block vulnerability r_b in [0,1]: 1 - normalised local entropy (low entropy = predictable)."""
    return np.clip(1.0 - F[..., 0], 0.0, 1.0)


def demands(F):
    """Per-block demand for X-, Y-diffusion unpredictability and RGB decorrelation, each in [0,1]:
    d_x = r*|corr_x|, d_y = r*|corr_y|, d_c = r*mean(|corr_RG|,|corr_GB|,|corr_BR|)."""
    r = risk(F); crgb = np.mean(np.abs(F[..., 6:9]), -1)
    return dict(dx=r * np.abs(F[..., 1]), dy=r * np.abs(F[..., 2]), dc=r * crgb, r=r)
