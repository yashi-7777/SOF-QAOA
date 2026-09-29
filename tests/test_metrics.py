import numpy as np
from src import metrics as M
from src.attacks import salt_pepper, crop_loss, occlusion
from src.utils import load_config
from src import cryptanalysis as CA

rng = np.random.default_rng(0); U = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8); CFG = load_config("config.yaml")


def test_entropy_chi_square_npcr_uaci_correlation():
    assert M.entropy(U).min() > 7.97 and M.entropy(np.zeros((8, 8, 3), np.uint8)).max() == 0 and M.chi_square(U).mean() < 293.25
    assert M.npcr_uaci(U, U) == (0.0, 0.0)
    n, u = M.npcr_uaci(U, rng.integers(0, 256, U.shape, dtype=np.uint8)); assert 99 < n <= 100 and 32.5 < u < 34.5
    smooth = np.tile(np.linspace(0, 255, 128).astype(np.uint8)[None, :, None], (128, 1, 3))
    assert M.adjacent_corr(smooth, "h").min() > 0.99 and abs(M.adjacent_corr(U, "h")).max() < 0.06


def test_mutual_information():
    a = rng.integers(0, 256, 200000); assert M.mutual_information(a, a) > 3.9 and abs(M.mutual_information(a, rng.integers(0, 256, 200000))) < 0.01
    small = rng.integers(0, 256, 16384); assert abs(M.mutual_information(small, rng.integers(0, 256, 16384))) < 0.03     # 128x128-sized sample


def test_attacks_change_expected_fraction():
    assert 0.03 < (salt_pepper(U, 0.05) != U).any(-1).mean() < 0.06
    assert (crop_loss(U, 0.25) == 0)[-32:].all() and (occlusion(U, 0.25) == 0)[48:80, 48:80].all()


def test_cryptanalysis_flags_missing_sbox_and_passes_keyed_sbox():
    ident = dict(CFG, sbox="identity")
    assert CA.affine_probes(ident)["additive_violation"] == 0.0 and CA.affine_probes(CFG)["additive_violation"] > 0.9
    assert CA.affine_reconstruction_attack(ident)["exact_prediction_rate"] == 1.0 and CA.affine_reconstruction_attack(CFG)["exact_prediction_rate"] < 0.05
    assert CA.differential_probe(ident, N=100)["mean_max_freq"] == 1.0 and CA.differential_probe(CFG, N=100)["mean_max_freq"] < 0.1


def test_xor_probe_alone_would_miss_the_additive_weakness():
    assert CA.affine_probes(dict(CFG, sbox="identity"))["xor_violation"] > 0.9       # algebra must match the cipher's group law


def test_bit_avalanche_close_to_half():
    assert abs(CA.bit_avalanche(CFG, trials=16)["mean"] - 0.5) < 0.02


def test_sbox_report_fields():
    r = CA.sbox_report(CFG); assert r["bijective"] and r["inverse_ok"] and r["nonlinearity"] > 80
