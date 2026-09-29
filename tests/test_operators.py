import numpy as np
from src import operators as ops
from src.chaotic_maps import ChaosEngine
from src.key_derivation import master_key
from src.substitution import make_sbox, verify_sbox, sbox_properties

rng = np.random.default_rng(0); eng = ChaosEngine(master_key("k", b"n" * 16))


def _R(n, B=16):
    S = rng.permutation(256).astype(np.uint8); shp = (n, B, B, 3)
    K = dict(kx=rng.integers(0, 256, shp), ky=rng.integers(0, 256, shp), kx2=rng.integers(0, 256, shp), ky2=rng.integers(0, 256, shp),
             c1=rng.integers(0, 256, (n, B, B)), c2=rng.integers(0, 256, (n, B, B)), c3=rng.integers(0, 256, (n, B, B)))
    return dict(K=K, S=S, Sinv=np.argsort(S).astype(np.uint8))


def test_sbox_bijective_inverse_deterministic_keyed():
    S, Si = make_sbox(eng, "t"); v = verify_sbox(S, Si); assert v["bijective"] and v["inverse_ok"]
    S2, _ = make_sbox(ChaosEngine(master_key("k", b"n" * 16)), "t"); assert (S == S2).all()                 # reproducible
    S3, _ = make_sbox(ChaosEngine(master_key("k2", b"n" * 16)), "t"); assert (S != S3).any()                # key dependent


def test_every_config_and_option_is_reversible():
    Xb = rng.integers(0, 256, (32, 16, 16, 3), dtype=np.uint8); R = _R(32); cf = (np.arange(32) % 16).astype(np.uint8)
    for o in (ops.DEFAULT_OPTS, dict(bx=1, by=0), dict(bx=0, by=1)):
        assert (ops.local_decrypt(ops.local_encrypt(Xb, cf, R, o), cf, R, o) == Xb).all()


def test_nonlinear_chain_inverse_and_avalanche():
    a = rng.integers(0, 256, (4, 16, 16, 3), dtype=np.uint8); K = rng.integers(0, 256, a.shape); S = rng.permutation(256); Si = np.argsort(S)
    for ax in (1, 2): assert (ops.nlchain_dec(ops.nlchain_enc(a, K, S, ax), K, Si, ax) == a).all()
    b = a.copy(); b[:, 0, 0, 0] ^= 1
    d = ops.nlchain_enc(a, K, S, 2) != ops.nlchain_enc(b, K, S, 2); assert d[:, 0, 1:, 0].mean() > 0.95      # change reaches the whole row


def test_rgb_coupling_invert_and_couples_channels():
    X = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8); S = rng.permutation(256).astype(np.int64); k = [rng.integers(0, 256, (16, 16)) for _ in range(3)]
    Y = ops.rgb_couple(X, S, *k); assert (ops.rgb_uncouple(Y, S, *k) == X).all()
    X2 = X.copy(); X2[..., 1] = (X2[..., 1] + 1) & 255; assert (ops.rgb_couple(X2, S, *k)[..., 0] != Y[..., 0]).any()


def test_global_chain_and_sbox_layer_invert():
    X = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8); gf, gb = rng.integers(0, 256, X.size), rng.integers(0, 256, X.size)
    assert (ops.global_dec(ops.global_enc(X, gf, gb), gf, gb) == X).all()
    S, Si = make_sbox(eng, "t"); ks = rng.integers(0, 256, X.shape); assert (ops.sbox_dec(ops.sbox_enc(X, S, ks), Si, ks) == X).all()


def test_block_reshape_roundtrip():
    a = rng.integers(0, 256, (48, 32, 3), dtype=np.uint8); assert (ops.from_blocks(ops.to_blocks(a, 16), 16, 3, 2) == a).all()


def test_compute_cost_is_linear_in_bits():
    c = np.arange(16); b = ops.bits_of(c); w = np.array([1, 1, 2, 1]); assert (ops.compute_cost(c) == 2 + b @ w).all()


def test_chaos_streams_deterministic_and_uniform():
    assert (eng.x_bytes("u", 100000) == eng.x_bytes("u", 100000)).all()
    for s in (eng.x_bytes("u", 200000), eng.y_bytes("u", 200000)):
        cnt = np.bincount(s, minlength=256); e = s.size / 256; assert ((cnt - e) ** 2 / e).sum() < 330


def test_sbox_property_code_against_aes():
    def gmul(a, b):
        p = 0
        for _ in range(8):
            if b & 1: p ^= a
            hi = a & 0x80; a = (a << 1) & 255
            if hi: a ^= 0x1B
            b >>= 1
        return p
    inv = [0] + [next(y for y in range(1, 256) if gmul(x, y) == 1) for x in range(1, 256)]
    aes = [ (lambda v: v ^ ((v << 1 | v >> 7) & 255) ^ ((v << 2 | v >> 6) & 255) ^ ((v << 3 | v >> 5) & 255) ^ ((v << 4 | v >> 4) & 255) ^ 0x63)(inv[x]) for x in range(256)]
    p = sbox_properties(np.array(aes)); assert p["differential_uniformity"] == 4 and p["nonlinearity"] == 112
