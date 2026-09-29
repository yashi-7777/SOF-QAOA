"""SHA-512 / HMAC-SHA-512 key schedule.

The plaintext enters ONLY through an HMAC keyed with the user key (SIV-style nonce):
the receiver gets the 16-byte nonce from the header, and without the key the nonce
reveals nothing about the plaintext (unlike an unkeyed image hash).
"""
import hashlib, hmac
import numpy as np


def _b(key):
    return key.encode() if isinstance(key, str) else bytes(key)


def derive_nonce(key, img, mode="siv"):
    k = _b(key)
    if mode == "siv":
        msg = b"SOF-NONCE" + np.array(img.shape, dtype=np.uint32).tobytes() + \
              np.ascontiguousarray(img).tobytes()
    else:
        msg = b"SOF-NONCE-FIXED"
    return hmac.new(k, msg, hashlib.sha512).digest()[:16]


def master_key(key, nonce):
    return hashlib.sha512(b"SOF-MASTER" + _b(key) + nonce).digest()


def expand(master, label, n):
    out, c = bytearray(), 0
    while len(out) < n:
        out += hashlib.sha512(master + label.encode() + c.to_bytes(4, "big")).digest()
        c += 1
    return bytes(out[:n])


def seed_uniform(master, label, n):
    """n floats in (0,1), deterministic from (master, label)."""
    u = np.frombuffer(expand(master, label, 8 * n), dtype="<u8")
    return np.clip((u >> np.uint64(11)).astype(np.float64) / 2.0 ** 53, 1e-6, 1 - 1e-6)


def header_tag(key, data):
    return hmac.new(_b(key), b"SOF-TAG" + data, hashlib.sha512).digest()[:8]
