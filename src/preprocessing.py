"""Image validation and block padding (reversible: original size goes in the header)."""
import numpy as np


def validate(img):
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("expected uint8 RGB array of shape (H, W, 3)")
    return img


def pad_to_block(img, B):
    H, W, _ = img.shape
    ph, pw = (-H) % B, (-W) % B
    if ph == 0 and pw == 0:
        return img, (H, W)
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge"), (H, W)


def crop(img, hw):
    return img[: hw[0], : hw[1]]
