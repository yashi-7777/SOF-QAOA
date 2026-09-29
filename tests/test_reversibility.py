import numpy as np, pytest
from src.utils import load_config, sample_images
from src.encryption import encrypt_image, merged_cfg
from src.decryption import decrypt_image, parse_header
from src.key_derivation import master_key, expand
from src.metrics import npcr_uaci
from src.spatial_operator_field import unpack_field

CFG = load_config("config.yaml"); IMG = sample_images(64)["synthetic"]


@pytest.mark.parametrize("method", ["fixed", "fixed_max", "random", "greedy", "random_window", "ls_window", "sa_window", "exact_window", "sa_full", "ls_full", "qaoa"])
def test_exact_reconstruction_every_selector(method):
    C, h, _ = encrypt_image(IMG, "research-key", CFG, method); assert np.array_equal(decrypt_image(C, h, "research-key", CFG), IMG)


@pytest.mark.parametrize("ab", [dict(rgb_coupling=True), dict(sbox="identity"), dict(tie_xy=True), dict(xy_mode="x_only"), dict(xy_mode="y_only"),
                                dict(rounds=1), dict(rounds=3), dict(global_chain=False)])
def test_exact_reconstruction_ablations(ab):
    C, h, _ = encrypt_image(IMG, "k", CFG, "qaoa", ab); assert np.array_equal(decrypt_image(C, h, "k", CFG, ab), IMG)


@pytest.mark.parametrize("shape", [(50, 70), (17, 33), (16, 16), (48, 80)])
def test_non_multiple_of_block_sizes(shape):
    im = np.random.default_rng(1).integers(0, 256, (*shape, 3), dtype=np.uint8); C, h, _ = encrypt_image(im, "k", CFG, "qaoa")
    assert np.array_equal(decrypt_image(C, h, "k", CFG), im)


def test_extreme_images():
    for im in (np.zeros((32, 32, 3), np.uint8), np.full((32, 32, 3), 255, np.uint8)):
        C, h, _ = encrypt_image(im, "k", CFG, "qaoa"); assert np.array_equal(decrypt_image(C, h, "k", CFG), im)


def test_same_input_same_ciphertext_and_header():
    a, ha, _ = encrypt_image(IMG, "k", CFG, "qaoa", use_cache=False); b, hb, _ = encrypt_image(IMG, "k", CFG, "qaoa", use_cache=False)
    assert np.array_equal(a, b) and ha == hb


def test_different_key_different_ciphertext():
    a, _, _ = encrypt_image(IMG, "key-A", CFG); b, _, _ = encrypt_image(IMG, "key-B", CFG); n, u = npcr_uaci(a, b); assert n > 99 and 32 < u < 35


def test_wrong_key_rejected_and_not_recovered():
    C, h, _ = encrypt_image(IMG, "k", CFG)
    with pytest.raises(ValueError): decrypt_image(C, h, "k2", CFG)
    assert (decrypt_image(C, h, "k2", CFG, verify=False) == IMG).mean() < 0.02


def test_one_pixel_change_plaintext_sensitivity():
    I2 = IMG.copy(); I2[10, 10, 0] ^= 1
    a, _, _ = encrypt_image(IMG, "k", CFG); b, _, _ = encrypt_image(I2, "k", CFG); assert npcr_uaci(a, b)[0] > 99                       # full system (SIV nonce)
    cf = dict(CFG, nonce_mode="fixed"); a, _, ia = encrypt_image(IMG, "k", cf, "qaoa"); b, _, _ = encrypt_image(I2, "k", cf, "qaoa", field=ia["cfgs"])
    n, u = npcr_uaci(a, b); assert n > 99 and 32 < u < 35                                                                             # pure cipher, frozen field


def test_header_tamper_detected():
    C, h, _ = encrypt_image(IMG, "k", CFG); bad = bytearray(h); bad[10] ^= 1
    with pytest.raises(ValueError): decrypt_image(C, bytes(bad), "k", CFG)


def test_receiver_reconstructs_the_exact_operator_field_from_header_and_key():
    C, h, info = encrypt_image(IMG, "k", CFG, "qaoa"); H, W, src, nonce, fb, tag, core = parse_header(h)
    master = master_key("k", nonce); mask = expand(master, "fieldmask", len(fb))
    field = unpack_field(bytes(a ^ b for a, b in zip(fb, mask)), IMG.shape[0] // 16, IMG.shape[1] // 16)
    assert np.array_equal(field, info["cfgs"]) and len(h) == 6 + 16 + len(fb) + 8


def test_field_bytes_are_masked_not_plain():
    C, h, info = encrypt_image(IMG, "k", CFG, "qaoa"); _, _, _, _, fb, _, _ = parse_header(h)
    from src.spatial_operator_field import pack_field
    assert fb != pack_field(info["cfgs"])                                     # XOR-masked with a keyed stream


def test_keyed_random_field_needs_no_field_bytes():
    C, h, info = encrypt_image(IMG, "k", CFG, "random"); assert info["field_bytes"] == 0 and len(h) == 30
