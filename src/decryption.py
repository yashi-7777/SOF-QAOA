"""Decrypt: (cipher image, header, key) -> exact plaintext. Inverse of encryption.encrypt_image."""
import struct
import numpy as np
from . import operators as ops
from .key_derivation import master_key, expand, header_tag
from .chaotic_maps import ChaosEngine
from .encryption import RoundKeys, merged_cfg, VER, FIELD_NONE_KEYED, FIELD_NONE_FIXED
from .preprocessing import crop
from .spatial_operator_field import unpack_field, op_opts, build_problem
from .utils import Timer


def parse_header(header):
    ver, H, W, src = struct.unpack(">BHHB", header[:6])
    if ver != VER: raise ValueError("bad header version")
    return H, W, src, header[6:22], header[22:-8], header[-8:], header[:-8]


def cipher_core_inv(C, cfgs, rk, cfg):
    B = cfg["block_size"]; H, W, _ = C.shape; gh, gw = H // B, W // B; opts = op_opts(cfg); flat = np.asarray(cfgs).reshape(-1); X = C
    for R in reversed(rk.rounds):
        if cfg.get("global_chain", True): X = ops.global_dec(X, R["gf"], R["gb"])
        X = ops.sbox_dec(X, R["Sinv"], R["ks"])
        X = ops.from_blocks(ops.local_decrypt(ops.to_blocks(X, B), flat, R, opts), B, gh, gw)
    return X


def decrypt_image(C, header, key, cfg, ablate=None, verify=True):
    cfg = merged_cfg(cfg, ablate); B = cfg["block_size"]
    hh, ww, src, nonce, fb, tag, core = parse_header(header)
    if verify and tag != header_tag(key, core): raise ValueError("header authentication failed (wrong key or corrupted header)")
    H, W, _ = C.shape; gh, gw = H // B, W // B
    master = master_key(key, nonce); engine = ChaosEngine(master)
    if src == FIELD_NONE_KEYED:
        # keyed pseudo-random field: recomputed from the key alone (needs the allowed-config list of the encoder's problem)
        from .spatial_operator_field import frozen_bits
        fz = frozen_bits(cfg); al = np.array([c for c in range(16) if all(((c >> k) & 1) == v for k, v in fz.items())])
        cfgs = al[engine.x_bytes("field:random", gh * gw).astype(np.int64) % len(al)].astype(np.uint8).reshape(gh, gw)
    else:
        mask = expand(master, "fieldmask", len(fb)); cfgs = unpack_field(bytes(a ^ b for a, b in zip(fb, mask)), gh, gw)
    rk = RoundKeys(engine, H, W, cfg)
    return crop(cipher_core_inv(C, cfgs, rk, cfg), (hh, ww))
