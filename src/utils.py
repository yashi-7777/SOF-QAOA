"""I/O, timing, plotting and sample-image helpers."""
import os, time
import numpy as np
import yaml
from PIL import Image


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_image(arr, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).save(path)


def ensure_dirs(base="outputs"):
    for s in ["encrypted", "decrypted", "metrics", "plots", "logs"]:
        os.makedirs(os.path.join(base, s), exist_ok=True)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); self.dt = 0.0; return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, bytes): return o.hex()
    return o


def synthetic_image(size=128, seed=0):
    """Gradients + shapes + fine texture + flat region (needs no external data)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    img = np.stack([127 + 100 * np.sin(2 * np.pi * (x + 0.3 * y)),
                    127 + 100 * np.cos(2 * np.pi * (y - 0.2 * x)),
                    60 + 150 * x * y], -1)
    m = (x - 0.35) ** 2 + (y - 0.6) ** 2 < 0.18 ** 2
    img[m] = [230, 60, 40]
    img[: size // 4, size // 2:] = 200                       # flat region
    h = size - size // 2
    img[size // 2:, size // 2:] += rng.normal(0, 25, (h, h, 3))   # textured patch
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def sample_images(size=128):
    """name -> uint8 RGB. Uses scikit-image bundled photos when available."""
    out = {"synthetic": synthetic_image(size, 1)}
    try:
        from skimage import data, transform
        for name in ["astronaut", "chelsea", "coffee"]:
            a = transform.resize(getattr(data, name)(), (size, size),
                                 anti_aliasing=True, preserve_range=True)
            out[name] = np.clip(a, 0, 255).astype(np.uint8)
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_histogram(img, path, title):
    plt = _plt(); fig, ax = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
    for c, (n, col) in enumerate(zip("RGB", "rgb")):
        ax[c].hist(img[..., c].ravel(), bins=256, range=(0, 255), color=col)
        ax[c].set_title(f"{title} - {n}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_correlation(img, path, title, n=3000, seed=0):
    plt = _plt(); rng = np.random.default_rng(seed)
    H, W, _ = img.shape
    ys = rng.integers(0, H, n); xs = rng.integers(0, W - 1, n)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for c, n_ in enumerate("RGB"):
        ax[c].scatter(img[ys, xs, c], img[ys, xs + 1, c], s=2, alpha=.4)
        ax[c].set_title(f"{title}: horizontal pairs - {n_}", fontsize=9)
        ax[c].set_xlim(0, 255); ax[c].set_ylim(0, 255)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_field(cfgs, path, names=None):
    """Operator field: colour = configuration index (0..15), text = configuration name."""
    plt = _plt(); from .operators import cfg_name
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * cfgs.shape[1], 0.8 + 0.6 * cfgs.shape[0]))
    ax.imshow(cfgs, cmap="tab20", vmin=0, vmax=19)
    if cfgs.shape[0] * cfgs.shape[1] <= 100:
        for (i, j), v in np.ndenumerate(cfgs): ax.text(j, i, cfg_name(int(v)), ha="center", va="center", fontsize=6)
    ax.set_title("Spatial operator field  (Xa/Xn: additive/nonlinear X chain, Ya/Yn: Y, B: bidirectional, R: RGB coupling)", fontsize=6)
    ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
