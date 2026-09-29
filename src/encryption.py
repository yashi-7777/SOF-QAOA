"""Encrypt: RGB image + key -> (cipher image, header, info).

  HMAC-SHA512(key, image) -> nonce -> SHA-512 master -> keyed chaotic streams / S-box
  features -> constrained QUBO -> [QAOA window decomposition] -> operator field Omega (4 bits / block)
  R rounds of { block-local operators Omega_ij  ->  global keyed S-box  ->  global forward+backward chain }

Header (decoder side information), byte layout:
  ver(1) | H(2) | W(2) | field_src(1) | nonce(16) | field bytes (0 or ceil(nb/2)) | HMAC tag(8)
  * nonce   : HMAC-SHA512(key, plaintext)[:16]  (SIV style: deterministic, key-dependent, reveals nothing without the key)
  * field   : 4 bits/block, XOR-masked with a keystream derived from (key, nonce)  -> only present for adaptive selectors
  * tag     : HMAC over the header (authenticates nonce/field, NOT the ciphertext)
"""
import copy, struct, hashlib, json
import numpy as np
from . import operators as ops
from .key_derivation import derive_nonce, master_key, expand, header_tag
from .chaotic_maps import ChaosEngine
from .preprocessing import validate, pad_to_block
from .spatial_operator_field import (build_problem, select_field, pack_field, unpack_field, op_opts)
from .substitution import make_sbox
from .utils import Timer

VER = 2
FIELD_TRANSMITTED, FIELD_NONE_FIXED, FIELD_NONE_KEYED = 0, 1, 2
_FIELD_CACHE = {}


def merged_cfg(cfg, ablate=None):
    c = copy.deepcopy(cfg)
    for k, v in (ablate or {}).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict): c[k].update(v)
        else: c[k] = v
    return c


class RoundKeys:
    def __init__(self, engine, H, W, cfg):
        B = cfg["block_size"]; n3 = H * W * 3
        blk3 = lambda a: ops.to_blocks(a.reshape(H, W, 3), B).astype(np.int64)
        blk1 = lambda a: ops.to_blocks(a.reshape(H, W), B).astype(np.int64)
        self.rounds = []
        for r in range(cfg["rounds"]):
            lab = f"r{r}"; S, Sinv = make_sbox(engine, lab, cfg["sbox"])
            K = dict(kx=blk3(engine.x_bytes(f"{lab}-kx", n3)), ky=blk3(engine.y_bytes(f"{lab}-ky", n3)),
                     kx2=blk3(engine.y_bytes(f"{lab}-kx2", n3)), ky2=blk3(engine.x_bytes(f"{lab}-ky2", n3)),
                     c1=blk1(engine.x_bytes(f"{lab}-c1", H * W)), c2=blk1(engine.y_bytes(f"{lab}-c2", H * W)),
                     c3=blk1(engine.x_bytes(f"{lab}-c3", H * W)))
            self.rounds.append(dict(K=K, S=S, Sinv=Sinv, ks=engine.y_bytes(f"{lab}-ks", n3).reshape(H, W, 3).astype(np.int64),
                                    gf=engine.x_bytes(f"{lab}-gf", n3).astype(np.int64), gb=engine.y_bytes(f"{lab}-gb", n3).astype(np.int64)))


def cipher_core(P, cfgs, rk, cfg):
    """Exactly invertible core; `cfgs` = operator field (gh,gw) uint8."""
    B = cfg["block_size"]; H, W, _ = P.shape; gh, gw = H // B, W // B; opts = op_opts(cfg)
    flat = np.asarray(cfgs).reshape(-1); X = P
    for R in rk.rounds:
        X = ops.from_blocks(ops.local_encrypt(ops.to_blocks(X, B), flat, R, opts), B, gh, gw)
        X = ops.sbox_enc(X, R["S"], R["ks"]).astype(np.uint8)
        if cfg.get("global_chain", True): X = ops.global_enc(X, R["gf"], R["gb"])
    return X


def make_header(key, hw, nonce, src, field_bytes):
    core = struct.pack(">BHHB", VER, hw[0], hw[1], src) + nonce + field_bytes
    return core + header_tag(key, core)


def encrypt_image(img, key, cfg, method=None, ablate=None, field=None, use_cache=True):
    """`field` (gh,gw) uint8 : reuse a previously derived operator field (skips selection)."""
    cfg = merged_cfg(cfg, ablate); method = method or cfg["selector"]
    img = validate(img); B = cfg["block_size"]
    with Timer() as t_total:
        P, hw = pad_to_block(img, B); H, W, _ = P.shape
        nonce = derive_nonce(key, P, cfg["nonce_mode"]); master = master_key(key, nonce)
        engine = ChaosEngine(master)
        with Timer() as t_keys: rk = RoundKeys(engine, H, W, cfg)
        seed = int.from_bytes(master[:4], "big"); gh, gw = H // B, W // B
        if field is not None:
            cfgs, sel = np.asarray(field, np.uint8), dict(method="given", runtime=0.0, circuit_evals=0, classical_evals=0,
                                                          qaoa_runtime=0.0, classical_runtime=0.0, J=float("nan"), cfg_hist=[], max_qubits=0, windows=0)
            t_feat = 0.0
        else:
            ck = (hashlib.sha256(P.tobytes()).hexdigest(), str(key), method, json.dumps(cfg, sort_keys=True, default=str), seed)
            if use_cache and ck in _FIELD_CACHE:
                cfgs, sel = _FIELD_CACHE[ck]; t_feat = 0.0
            else:
                with Timer() as tf: prob = build_problem(P, cfg)
                t_feat = tf.dt; cfgs, sel = select_field(prob, engine, cfg, method, seed)
                if use_cache: _FIELD_CACHE[ck] = (cfgs, sel)
        if method == "fixed" or method == "fixed_max":
            # uniform fields still depend on the image (chosen under J) -> transmit (cheap, 4 bits/block)
            pass
        if method == "random":
            src, fb = FIELD_NONE_KEYED, b""
        else:
            src = FIELD_TRANSMITTED; pk = pack_field(cfgs); mask = expand(master, "fieldmask", len(pk))
            fb = bytes(a ^ b for a, b in zip(pk, mask))
        with Timer() as t_cipher: C = cipher_core(P, cfgs, rk, cfg)
        header = make_header(key, hw, nonce, src, fb)
    info = dict(sel=sel, cfgs=cfgs, method=method, header_bytes=len(header), field_bytes=len(fb), t_total=t_total.dt,
                t_keys=t_keys.dt, t_features=t_feat, t_cipher=t_cipher.dt, t_select=sel["runtime"], nonce=nonce)
    return C, header, info
