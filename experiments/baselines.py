"""Selector comparison (Table 2b/1) and the shared evaluate() used by the ablation suite."""
import numpy as np, pandas as pd
from src.encryption import encrypt_image, merged_cfg
from src.decryption import decrypt_image
from src import metrics as M
from src.spatial_operator_field import TERMS

KEY = "research-key"
SELECTORS = ["fixed", "random", "greedy", "random_window", "ls_window", "sa_window", "exact_window", "sa_full", "ls_full", "qaoa"]


def evaluate(img, cfg, method, ablate=None, trials=None, key=KEY):
    """One metrics row. NPCR/UACI: one-sample plaintext change, key-only nonce, operator field FROZEN to the original image's
    field so that the diffusion of the cipher itself is measured (re-deriving the field would re-run the optimiser per trial)."""
    trials = trials or cfg["experiments"]["npcr_trials"]; rng = np.random.default_rng(cfg["experiments"]["seed"])
    cfg_m = merged_cfg(cfg, ablate); C, hdr, info = encrypt_image(img, key, cfg, method, ablate)
    assert np.array_equal(decrypt_image(C, hdr, key, cfg, ablate), img), "REVERSIBILITY FAILED"      # before any metric
    S = M.summarize(C, img, cfg_m["block_size"]); S.update(M.channel_mi(C))
    cf = dict(cfg_m, nonce_mode="fixed"); f = info["cfgs"]; C0 = encrypt_image(img, key, cf, method, field=f)[0]; vals = []
    for _ in range(trials):
        y, x, c = rng.integers(img.shape[0]), rng.integers(img.shape[1]), rng.integers(3)
        i2 = img.copy(); i2[y, x, c] ^= 1; vals.append(M.npcr_uaci(C0, encrypt_image(i2, key, cf, method, field=f)[0]))
    npcr, uaci = np.mean(vals, 0); sel = info["sel"]
    row = dict(method=method, J=sel["J"], **{t: sel["terms"].get(t, np.nan) for t in TERMS}, ops_total=sel.get("ops_total", np.nan),
               ops_target=sel.get("ops_target", np.nan), circuit_evals=sel["circuit_evals"], classical_evals=sel["classical_evals"],
               select_s=info["t_select"], qaoa_s=sel["qaoa_runtime"], enc_s=info["t_total"], header_B=info["header_bytes"],
               cfgs_used=int(sum(v > 0 for v in sel.get("cfg_hist", []))), NPCR=npcr, UACI=uaci, **S)
    return row, info


def run(cfg, imgs, out):
    rows = []
    for name, im in imgs.items():
        for m in SELECTORS:
            r, _ = evaluate(im, cfg, m); r["image"] = name; rows.append(r)
            print(f"[baselines] {name:10s} {m:14s} J={r['J']:.4f} ops={r['ops_total']:.0f}/{r['ops_target']:.0f} NPCR={r['NPCR']:.3f} H={r['entropy_mean']:.4f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/baselines.csv", index=False)
    cols = ["J", *TERMS, "ops_total", "circuit_evals", "classical_evals", "select_s", "entropy_mean", "chi2_mean", "corr_h", "corr_v", "corr_d",
            "NPCR", "UACI", "local_resid_max"]
    g = df.groupby("method")[cols].mean().reindex(SELECTORS); g.to_csv(f"{out}/metrics/baselines_summary.csv")
    print("\n" + g.round(4).to_string()); return df
