"""Cryptanalytic probes. Each test states what it measures and what it does NOT prove.

All probes run on the cipher CORE with a FIXED operator field and FIXED nonce/key (so the map is a deterministic
function E(x) that we can query as a chosen-plaintext oracle). They probe structure, they are not security proofs.

 1 affine_probes           additivity over Z_256 and XOR:  E(a)+E(b)-E(0)-E(a+b) == 0 ?   (fraction of samples violating)
 2 affine_reconstruction   chosen-plaintext attack: query E(0), E(unit_i) for every sample i, predict E(x) = E(0)+sum x_i (E(u_i)-E(0))
                           for fresh random x; success = fraction of exactly predicted samples. 1.0 => cipher is affine => broken.
 3 differential_probe      fixed additive input difference d on one sample, N random plaintexts: for each output sample the largest
                           frequency of any difference value; ~1/256 (+ sampling floor) is ideal, 1.0 means a deterministic differential.
 4 bit_avalanche           flip one plaintext bit (fixed key/nonce/field): fraction of ciphertext BITS flipped, ideal 0.5.
 5 sbox_report             bijectivity, DDT (differential uniformity), Walsh (nonlinearity, max linear bias), SAC.
 6 channel_differential    flip one R sample: fraction of G/B ciphertext samples that change (isolates the RGB coupling).
 7 field_leakage           does ciphertext block statistics depend on the (secret) operator field bits?  Mann-Whitney per bit.
"""
import numpy as np
from scipy import stats
from .encryption import RoundKeys, cipher_core, merged_cfg
from .key_derivation import master_key
from .chaotic_maps import ChaosEngine
from .substitution import sbox_properties, verify_sbox
from . import metrics as M


def _oracle(cfg, H, W, field_value=None, seed=0, key="probe"):
    cfg = merged_cfg(cfg); B = cfg["block_size"]; rk = RoundKeys(ChaosEngine(master_key(key, b"\0" * 16)), H, W, cfg)
    rng = np.random.default_rng(seed); gh, gw = H // B, W // B
    field = rng.integers(0, 16, (gh, gw)).astype(np.uint8) if field_value is None else np.full((gh, gw), field_value, np.uint8)
    return (lambda x: cipher_core(x, field, rk, cfg).astype(np.int64)), field


def affine_probes(cfg, H=64, W=64, trials=3, seed=0):
    E, _ = _oracle(cfg, H, W, seed=seed); rng = np.random.default_rng(seed); zero = np.zeros((H, W, 3), np.uint8)
    add, xor = [], []
    for _ in range(trials):
        a = rng.integers(0, 256, (H, W, 3), dtype=np.uint8); b = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        add.append((((E(a) + E(b) - E(zero) - E((a.astype(np.int64) + b) & 255).astype(np.int64)) & 255) != 0).mean() if False else
                   ((((E(a) + E(b) - E(zero) - E(((a.astype(np.int64) + b) & 255).astype(np.uint8))) & 255) != 0).mean()))
        xor.append(((E(a) ^ E(b) ^ E(zero) ^ E(a ^ b)) != 0).mean())
    return dict(additive_violation=float(np.mean(add)), xor_violation=float(np.mean(xor)))


def affine_reconstruction_attack(cfg, H=16, W=16, seed=0, n_test=8):
    E, _ = _oracle(cfg, H, W, seed=seed); rng = np.random.default_rng(seed); zero = np.zeros((H, W, 3), np.uint8)
    e0 = E(zero); N = H * W * 3; basis = np.empty((N,) + e0.shape, np.int64)
    for i in range(N):
        u = zero.copy().reshape(-1); u[i] = 1; basis[i] = (E(u.reshape(H, W, 3)) - e0) & 255
    ok = []
    for _ in range(n_test):
        x = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        pred = (e0 + np.tensordot(x.reshape(-1).astype(np.int64), basis, axes=(0, 0))) & 255
        ok.append((pred == E(x)).mean())
    return dict(queries=N + 1, exact_prediction_rate=float(np.mean(ok)))


def differential_probe(cfg, H=32, W=32, N=400, seed=0, n_pos=64):
    E, _ = _oracle(cfg, H, W, seed=seed); rng = np.random.default_rng(seed); d = int(rng.integers(1, 256))
    y, x, c = int(rng.integers(H // 2)), int(rng.integers(W // 2)), int(rng.integers(3))
    D = np.empty((N, H, W, 3), np.int64)
    for t in range(N):
        p = rng.integers(0, 256, (H, W, 3), dtype=np.uint8); q = p.copy(); q[y, x, c] = (int(p[y, x, c]) + d) & 255
        D[t] = (E(q) - E(p)) & 255
    pos = rng.choice(H * W * 3, n_pos, replace=False); Dp = D.reshape(N, -1)[:, pos]
    mx = np.array([np.bincount(Dp[:, j], minlength=256).max() / N for j in range(n_pos)])
    return dict(mean_max_freq=float(mx.mean()), worst_max_freq=float(mx.max()), ideal_floor=float(1 / 256), samples=N)


def bit_avalanche(cfg, H=32, W=32, trials=48, seed=0):
    E, _ = _oracle(cfg, H, W, seed=seed); rng = np.random.default_rng(seed); r = []
    for _ in range(trials):
        p = rng.integers(0, 256, (H, W, 3), dtype=np.uint8); q = p.copy(); i = tuple(int(rng.integers(m)) for m in (H, W, 3))
        q[i] ^= np.uint8(1 << int(rng.integers(8)))
        r.append(np.unpackbits((E(p) ^ E(q)).astype(np.uint8)).mean())
    return dict(mean=float(np.mean(r)), std=float(np.std(r)), ideal=0.5)


def sbox_report(cfg, label="r0"):
    from .substitution import make_sbox
    eng = ChaosEngine(master_key("probe", b"\0" * 16)); S, Si = make_sbox(eng, label, cfg.get("sbox", "keyed"))
    return dict(**verify_sbox(S, Si), **sbox_properties(S))


def channel_differential(cfg, field_value, H=32, W=32, trials=32, seed=0):
    """Flip one R sample; report fraction of G and B ciphertext samples that change (field fixed to one config)."""
    E, _ = _oracle(cfg, H, W, field_value=field_value, seed=seed); rng = np.random.default_rng(seed); g, b, r_ = [], [], []
    for _ in range(trials):
        p = rng.integers(0, 256, (H, W, 3), dtype=np.uint8); q = p.copy(); q[int(rng.integers(H)), int(rng.integers(W)), 0] ^= np.uint8(rng.integers(1, 256))
        d = E(p) != E(q); r_.append(d[..., 0].mean()); g.append(d[..., 1].mean()); b.append(d[..., 2].mean())
    return dict(R=float(np.mean(r_)), G=float(np.mean(g)), B=float(np.mean(b)))


def field_leakage(C, cfgs, B=16):
    """Mann-Whitney U (two-sided) between blocks whose field bit k is 0 vs 1, on per-block ciphertext statistics."""
    from .operators import to_blocks
    blk = to_blocks(C, B); nb = blk.shape[0]; flat = blk.reshape(nb, -1).astype(np.int64)
    ent = np.array([-(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(v, minlength=256) / v.size) for v in flat])
    mean = flat.mean(1); chi = np.array([((np.bincount(v, minlength=256) - v.size / 256) ** 2 / (v.size / 256)).sum() for v in flat])
    bits = (np.asarray(cfgs).reshape(-1)[:, None] >> np.arange(4)) & 1; out = {}
    for k, name in enumerate(["zX", "zY", "zB", "zR"]):
        a, b = bits[:, k] == 0, bits[:, k] == 1
        if a.sum() < 5 or b.sum() < 5: out[name] = None; continue
        out[name] = min(stats.mannwhitneyu(s[a], s[b]).pvalue for s in (ent, mean, chi))
    ps = [v for v in out.values() if v is not None]
    return dict(min_p_per_bit=out, bonferroni_threshold=0.05 / max(3 * len(ps), 1), leak_detected=bool(ps and min(ps) < 0.05 / (3 * len(ps))))
