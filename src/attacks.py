"""Channel/noise attacks on the CIPHERTEXT, plus a linearity probe of the cipher core."""
import io
import numpy as np
from PIL import Image


def salt_pepper(img, p, seed=0):
    rng = np.random.default_rng(seed); out = img.copy(); m = rng.random(img.shape[:2])
    out[m < p / 2] = 0; out[(m >= p / 2) & (m < p)] = 255
    return out


def gaussian(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, img.shape), 0, 255).round().astype(np.uint8)


def crop_loss(img, frac):
    """Data loss: bottom `frac` of the rows are zeroed."""
    out = img.copy(); r = int(round(img.shape[0] * frac)); 
    if r: out[-r:] = 0
    return out


def occlusion(img, frac):
    """Central square covering `frac` of the area is zeroed."""
    out = img.copy(); H, W, _ = img.shape; s = int(round(np.sqrt(frac) * min(H, W)))
    y0, x0 = (H - s) // 2, (W - s) // 2; out[y0:y0 + s, x0:x0 + s] = 0
    return out


def jpeg(img, quality=90):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)


ATTACKS = {
    "salt_pepper_1%": lambda c: salt_pepper(c, 0.01), "salt_pepper_5%": lambda c: salt_pepper(c, 0.05),
    "gaussian_s5": lambda c: gaussian(c, 5), "gaussian_s20": lambda c: gaussian(c, 20),
    "crop_6%": lambda c: crop_loss(c, 0.0625), "crop_25%": lambda c: crop_loss(c, 0.25),
    "occlusion_6%": lambda c: occlusion(c, 0.0625), "occlusion_25%": lambda c: occlusion(c, 0.25),
    "jpeg_q90": lambda c: jpeg(c, 90),
}
