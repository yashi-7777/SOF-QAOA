"""python main.py --image sample.png --key "research-key" [--selector qaoa] [--config config.yaml]"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from src.utils import (load_config, load_image, save_image, ensure_dirs, jsonable, psnr, plot_histogram, plot_correlation,
                       plot_field, synthetic_image)
from src.encryption import encrypt_image, merged_cfg
from src.decryption import decrypt_image
from src import metrics as M
from src.spatial_operator_field import build_problem, select_field, TERMS
from src.operators import cfg_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--selector", default=None)
    ap.add_argument("--out", default="outputs"); ap.add_argument("--no-reference", action="store_true", help="skip exact/SA reference solves")
    a = ap.parse_args(); cfg = load_config(a.config); ensure_dirs(a.out)
    if not os.path.exists(a.image):
        os.makedirs(os.path.dirname(a.image) or ".", exist_ok=True); save_image(synthetic_image(128, 1), a.image)
        print(f"[info] {a.image} not found -> wrote a synthetic sample")
    img = load_image(a.image); stem = os.path.splitext(os.path.basename(a.image))[0]; method = a.selector or cfg["selector"]
    q = cfg["qaoa"]

    C, header, info = encrypt_image(img, a.key, cfg, method, use_cache=False)
    _, _, mem = M.measure_memory(lambda: encrypt_image(img, a.key, cfg, method, use_cache=False))
    t0 = time.perf_counter(); D = decrypt_image(C, header, a.key, cfg); t_dec = time.perf_counter() - t0
    assert D.shape == img.shape and np.array_equal(D, img), "REVERSIBILITY FAILED"          # hard requirement, before any metric
    C2, h2, _ = encrypt_image(img, a.key, cfg, method, use_cache=False); deterministic = bool(np.array_equal(C, C2) and header == h2)
    save_image(C, f"{a.out}/encrypted/{stem}_encrypted.png"); save_image(D, f"{a.out}/decrypted/{stem}_decrypted.png")
    open(f"{a.out}/encrypted/{stem}_header.bin", "wb").write(header)

    S = M.summarize(C, img, cfg["block_size"]); S.update(M.channel_mi(C))
    # plaintext sensitivity: (i) full system (field re-derived, SIV nonce), (ii) pure cipher (frozen field, key-only nonce), several trials
    rng = np.random.default_rng(1); i2 = img.copy(); i2[img.shape[0] // 2, img.shape[1] // 2, 1] ^= 1
    ps_full = M.npcr_uaci(C, encrypt_image(i2, a.key, cfg, method)[0])
    cf = dict(cfg, nonce_mode="fixed"); f = info["cfgs"]; C0 = encrypt_image(img, a.key, cf, method, field=f)[0]; vals = []
    for _ in range(4):
        y, x, c = rng.integers(img.shape[0]), rng.integers(img.shape[1]), rng.integers(3); j = img.copy(); j[y, x, c] ^= 1
        vals.append(M.npcr_uaci(C0, encrypt_image(j, a.key, cf, method, field=f)[0]))
    npcr, uaci = np.mean(vals, 0)
    k2 = a.key[:-1] + chr(ord(a.key[-1]) ^ 1); Ck, _, _ = encrypt_image(img, k2, cfg, method); ks = M.npcr_uaci(C, Ck)
    Dw = decrypt_image(C, header, k2, cfg, verify=False)

    sel = info["sel"]; qa = dict(method=method, n_qubits_per_window=sel["max_qubits"], windows=sel["windows"], sweeps=cfg["sweeps"], p=q["p"],
                                 shots=q["shots"], maxiter=q["maxiter"], circuit_evals=sel["circuit_evals"], classical_bitstring_evals=sel["classical_evals"],
                                 qaoa_runtime_s=sel["qaoa_runtime"], gammas_per_window=sel["gammas"], betas_per_window=sel["betas"],
                                 p_opt_per_window=sel["p_opt"], expected_energy_per_window=sel["exp_energy"], objective_J=sel["J"], terms=sel["terms"],
                                 ops_total=sel.get("ops_total"), ops_target=sel.get("ops_target"), cfg_histogram=sel["cfg_hist"])
    ref = {}
    if not a.no_reference and method == "qaoa":
        P, _ = __import__("src.preprocessing", fromlist=["x"]).pad_to_block(img, cfg["block_size"]); prob = build_problem(P, cfg)
        for m in ("exact_window", "sa_window", "fixed", "random"):
            eng = type("E", (), {"x_bytes": lambda s, l, n: np.random.default_rng(1).integers(0, 256, n).astype(np.uint8)})()
            _, r = select_field(prob, eng, cfg, m, 0); ref[m] = dict(J=r["J"], seconds=r["runtime"], classical_evals=r["classical_evals"])
        Jo = ref["exact_window"]["J"]; qa["J_exact_window_decomposition"] = Jo
        qa["objective_gap_vs_exact_window"] = float((sel["J"] - Jo) / abs(Jo)); qa["reference"] = ref

    res = dict(image=a.image, size=list(img.shape), selector=method, reversible=True, deterministic=deterministic,
               enc_time_s=info["t_total"], dec_time_s=t_dec, throughput_MBps=img.nbytes / 1e6 / info["t_total"], enc_peak_mem_MB=mem,
               header_bytes=info["header_bytes"], field_bytes=info["field_bytes"], NPCR_pure_cipher=float(npcr), UACI_pure_cipher=float(uaci),
               NPCR_plaintext_full_system=ps_full[0], UACI_plaintext_full_system=ps_full[1], key_sens_NPCR=ks[0], key_sens_UACI=ks[1],
               wrong_key_exact_pixels_pct=float(100 * (Dw == img).mean()), wrong_key_PSNR=psnr(img, Dw),
               key_space=M.key_space_bits(a.key), qaoa=qa, **S)
    res = jsonable(res)
    json.dump(res, open(f"{a.out}/metrics/metrics.json", "w"), indent=2)
    json.dump(jsonable(qa), open(f"{a.out}/metrics/qaoa_stats.json", "w"), indent=2)
    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in res.items()}
    pd.DataFrame([flat]).to_csv(f"{a.out}/metrics/metrics.csv", index=False)
    pd.DataFrame(info["cfgs"]).to_csv(f"{a.out}/metrics/operator_field.csv", index=False, header=False)
    plot_histogram(img, f"{a.out}/plots/original_histogram.png", "original"); plot_histogram(C, f"{a.out}/plots/encrypted_histogram.png", "encrypted")
    plot_correlation(img, f"{a.out}/plots/correlation_original.png", "original"); plot_correlation(C, f"{a.out}/plots/correlation_encrypted.png", "encrypted")
    plot_field(info["cfgs"], f"{a.out}/plots/spatial_operator_field.png")

    print(f"\n=== SOF-QAOA | {a.image} {img.shape} | selector={method} ===")
    print(f"reversibility (pixel-exact)   : PASS    deterministic: {deterministic}")
    print(f"encryption / decryption time  : {info['t_total']:.3f} s / {t_dec:.3f} s  ({res['throughput_MBps']:.3f} MB/s, peak {mem:.1f} MB)")
    print(f"entropy / chi2 (pass<293.25)  : {S['entropy_mean']:.4f} / {S['chi2_mean']:.1f} ({S['chi2_pass']})")
    print(f"correlation h / v / d         : {S['corr_h']:+.4f} / {S['corr_v']:+.4f} / {S['corr_d']:+.4f}")
    print(f"RG / GB / BR  (MI_RG bits)    : {S['corr_RG']:+.4f} / {S['corr_GB']:+.4f} / {S['corr_BR']:+.4f}   ({S['MI_RG']:.4f})")
    print(f"NPCR/UACI pure cipher         : {npcr:.3f}% / {uaci:.3f}%   full system (field re-derived): {ps_full[0]:.3f}% / {ps_full[1]:.3f}%")
    print(f"key sensitivity NPCR / UACI   : {ks[0]:.3f}% / {ks[1]:.3f}%   wrong-key exact pixels: {res['wrong_key_exact_pixels_pct']:.2f}%")
    print(f"operator field                : {info['cfgs'].size} blocks, {sum(v > 0 for v in sel['cfg_hist'])} distinct configs, ops {sel.get('ops_total')}/{sel.get('ops_target'):.0f} (target)")
    if method == "qaoa":
        print(f"QAOA                          : {sel['max_qubits']} qubits/window, {sel['windows']} windows ({cfg['sweeps']} sweeps), p={q['p']}, shots={q['shots']}, "
              f"{sel['circuit_evals']} circuit evals, {sel['qaoa_runtime']:.2f} s")
        print(f"  first-window gamma/beta     : {np.round(sel['gammas'][0], 3).tolist()} / {np.round(sel['betas'][0], 3).tolist()}")
        print(f"  objective J (QAOA)          : {sel['J']:.4f}  terms { {k[:2]: round(v, 3) for k, v in sel['terms'].items()} }")
        for m, r in ref.items(): print(f"  reference {m:13s}      : J={r['J']:.4f} ({r['seconds']:.2f}s, {r['classical_evals']} evals)")
        if ref: print(f"  gap QAOA vs exact-window    : {qa['objective_gap_vs_exact_window']:+.4f} (exact_window = best per-window given neighbours; NOT a global optimum)")
    print(f"header side-info              : {info['header_bytes']} B = ver1+dims4+src1+nonce16+field{info['field_bytes']}+tag8  -> {a.out}/encrypted/{stem}_header.bin")
    print(f"key space                     : 512-bit master; user-key entropy bound {res['key_space']['user_key_max_entropy_bits']} bits")


if __name__ == "__main__":
    main()
