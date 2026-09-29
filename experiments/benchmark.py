"""Security suite, computational complexity, RGB-coupling investigation, weight sensitivity, robustness, cryptanalysis, leakage."""
import time, numpy as np, pandas as pd
from src.utils import sample_images, psnr
from src.encryption import encrypt_image, merged_cfg
from src.decryption import decrypt_image
from src.attacks import ATTACKS
from src import metrics as M, cryptanalysis as CA
from src.spatial_operator_field import build_problem, combine, select_field, TERMS

KEY = "research-key"


def security_suite(cfg, imgs, out):
    rows = []
    for name, im in imgs.items():
        C, h, info = encrypt_image(im, KEY, cfg); t0 = time.perf_counter(); D = decrypt_image(C, h, KEY, cfg); td = time.perf_counter() - t0
        assert np.array_equal(D, im)
        S = M.summarize(C, im, cfg["block_size"]); S.update(M.channel_mi(C))
        k2 = KEY[:-1] + chr(ord(KEY[-1]) ^ 1); Ck, _, _ = encrypt_image(im, k2, cfg); ks = M.npcr_uaci(C, Ck)
        Dw = decrypt_image(C, h, k2, cfg, verify=False)
        i2 = im.copy(); i2[im.shape[0] // 2, im.shape[1] // 2, 1] ^= 1; C2, _, _ = encrypt_image(i2, KEY, cfg)       # full system, field re-derived
        ps = M.npcr_uaci(C, C2)
        cf = dict(cfg, nonce_mode="fixed"); f = info["cfgs"]; A = encrypt_image(im, KEY, cf, field=f)[0]; B = encrypt_image(i2, KEY, cf, field=f)[0]
        pf = M.npcr_uaci(A, B)
        rows.append(dict(image=name, entropy=S["entropy_mean"], chi2=S["chi2_mean"], chi2_pass=S["chi2_pass"], corr_h=S["corr_h"], corr_v=S["corr_v"],
                         corr_d=S["corr_d"], corr_RG=S["corr_RG"], corr_GB=S["corr_GB"], corr_BR=S["corr_BR"], NPCR_plaintext_full=ps[0], UACI_plaintext_full=ps[1],
                         NPCR_plaintext_frozen_field=pf[0], UACI_plaintext_frozen_field=pf[1], NPCR_key=ks[0], UACI_key=ks[1],
                         wrong_key_exact_pct=100 * float((Dw == im).mean()), enc_s=info["t_total"], dec_s=td,
                         throughput_MBps=im.nbytes / 1e6 / info["t_total"], header_B=info["header_bytes"],
                         key_space_bits_master=512, user_key_entropy_bound_bits=min(8 * len(KEY), 512)))
        print(f"[security] {name}: H={S['entropy_mean']:.4f} NPCR(full)={ps[0]:.3f} NPCR(frozen)={pf[0]:.3f} key NPCR={ks[0]:.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/security_suite.csv", index=False); return df


def complexity(cfg, out, sizes=(128, 256)):
    rows = []
    for size in sizes:
        im = sample_images(size)["synthetic"]
        for m in ["fixed", "sa_window", "qaoa"]:
            C, h, info = encrypt_image(im, KEY, cfg, m, use_cache=False)
            _, _, mem = M.measure_memory(lambda: encrypt_image(im, KEY, cfg, m, use_cache=False))
            t0 = time.perf_counter(); decrypt_image(C, h, KEY, cfg); td = time.perf_counter() - t0
            s = info["sel"]
            rows.append(dict(size=size, blocks=(size // cfg["block_size"]) ** 2, method=m, enc_s=info["t_total"], keys_s=info["t_keys"],
                             qubo_build_s=info["t_features"], select_s=info["t_select"], cipher_s=info["t_cipher"], dec_s=td, peak_MB=mem,
                             circuit_evals=s["circuit_evals"], classical_evals=s["classical_evals"], windows=s["windows"], qubits_per_window=s["max_qubits"],
                             throughput_MBps=im.nbytes / 1e6 / info["t_total"], header_B=info["header_bytes"]))
            print(f"[complexity] {size} {m:9s} enc={info['t_total']:.2f}s select={info['t_select']:.2f}s circ={s['circuit_evals']} peak={mem:.0f}MB", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/complexity.csv", index=False); return df


def rgb_investigation(cfg, out):
    """2x2 design: RGB coupling (field bit zR) x global chain.  Fraction of G/B ciphertext samples changed when ONE R sample flips."""
    rows = []
    for gc in (True, False):
        for zr, fv in ((False, 7), (True, 15)):
            c = dict(cfg, global_chain=gc, rounds=(cfg["rounds"] if gc else 1)); d = CA.channel_differential(c, fv, trials=32)
            rows.append(dict(global_chain=gc, rgb_coupling=zr, changed_R=d["R"], changed_G=d["G"], changed_B=d["B"]))
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/rgb_coupling_investigation.csv", index=False); print(df.round(4).to_string()); return df


def weight_sensitivity(cfg, out, n_samples=24, seed=0):
    """Dirichlet(1,1,1,1) weights on the 4 range-normalised terms; exact-window solver (classical, cheap) on the 128x128 astronaut."""
    im = sample_images(128).get("astronaut", sample_images(128)["synthetic"]); prob = build_problem(im, cfg)
    eng = type("E", (), {"x_bytes": lambda s, l, n: np.zeros(n, np.uint8)})()
    base, binfo = select_field(prob, eng, cfg, "exact_window", 0); Q0 = combine(prob, cfg); rng = np.random.default_rng(seed); rows = []
    for k in range(n_samples):
        w = rng.dirichlet(np.ones(4)); f, info = select_field(prob, eng, cfg, "exact_window", 0, weights=w)
        from src.spatial_operator_field import cfg_to_x
        rows.append(dict(sample=k, **{f"w_{t}": w[i] for i, t in enumerate(TERMS)}, blocks_changed_frac=float((f != base).mean()),
                         J_default_of_solution=float(Q0.energy(cfg_to_x(f))[0]), regret_vs_default=float(Q0.energy(cfg_to_x(f))[0] - binfo["J"]),
                         ops=info["ops_total"], cfgs_used=int(sum(v > 0 for v in info["cfg_hist"]))))
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/weight_sensitivity.csv", index=False)
    print("[sensitivity] mean blocks changed %.3f  mean regret %.4f (default J=%.4f)" % (df.blocks_changed_frac.mean(), df.regret_vs_default.mean(), binfo["J"])); return df


def robustness(cfg, imgs, out):
    rob = []
    for name, im in list(imgs.items())[:2]:
        C, h, _ = encrypt_image(im, KEY, cfg)
        for an, fn in ATTACKS.items():
            D = decrypt_image(fn(C), h, KEY, cfg); rob.append(dict(image=name, attack=an, PSNR_dB=psnr(im, D), exact_pixels_pct=100 * float((D == im).mean())))
    df = pd.DataFrame(rob); df.to_csv(f"{out}/metrics/robustness.csv", index=False); return df


def cryptanalysis_table(cfg, out):
    rows = []
    for nm, ab in [("proposed (keyed S-box)", {}), ("identity S-box (E9)", dict(sbox="identity"))]:
        c = dict(cfg, **ab); ap = CA.affine_probes(c); ar = CA.affine_reconstruction_attack(c); dp = CA.differential_probe(c, N=300)
        av = CA.bit_avalanche(c, trials=48); sb = CA.sbox_report(c)
        rows.append(dict(variant=nm, additive_violation=ap["additive_violation"], xor_violation=ap["xor_violation"], affine_recon_rate=ar["exact_prediction_rate"],
                         recon_queries=ar["queries"], diff_mean_max_freq=dp["mean_max_freq"], diff_worst_max_freq=dp["worst_max_freq"], diff_ideal=dp["ideal_floor"],
                         bit_avalanche=av["mean"], bit_avalanche_std=av["std"], sbox_bijective=bool(sb["bijective"]), sbox_inverse_ok=bool(sb["inverse_ok"]),
                         sbox_DU=sb["differential_uniformity"], sbox_NL=sb["nonlinearity"], sbox_max_lin_bias=sb["max_linear_bias"], sbox_SAC_dev=sb["sac_max_dev"]))
        print("[crypt]", rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/cryptanalysis.csv", index=False); return df


def leakage(cfg, out):
    im = sample_images(256).get("astronaut", sample_images(256)["synthetic"]); C, h, info = encrypt_image(im, KEY, cfg)
    r = CA.field_leakage(C, info["cfgs"], cfg["block_size"]); pd.DataFrame([dict(blocks=info["cfgs"].size, **{f"p_{k}": v for k, v in r["min_p_per_bit"].items()},
                                                                                  bonferroni_threshold=r["bonferroni_threshold"], leak_detected=r["leak_detected"])]).to_csv(f"{out}/metrics/field_leakage.csv", index=False)
    print("[leakage]", r); return r


def run(cfg, imgs, out, quick=False):
    security_suite(cfg, imgs, out); rgb_investigation(cfg, out); cryptanalysis_table(cfg, out); weight_sensitivity(cfg, out, 8 if quick else 24)
    complexity(cfg, out, (128,) if quick else (128, 256)); robustness(cfg, imgs, out)
    if not quick: leakage(cfg, out)
