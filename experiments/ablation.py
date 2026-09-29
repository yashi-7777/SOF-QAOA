"""Ablation E1..E11 (E4 = proposed system; E4/E8 duplicate of v1 removed - every experiment below differs from E4 in exactly one thing).
NOTE: J is comparable only among experiments that share the same variable set (E7/E8 change the operator library)."""
import pandas as pd
from experiments.baselines import evaluate
from src import cryptanalysis as CA

EXPS = [
    ("E1_fixed_best_uniform",   "fixed",         None),
    ("E2_random_keyed_field",   "random",        None),
    ("E3_classical_SA_window",  "sa_window",     None),
    ("E4_QAOA_proposed",        "qaoa",          None),
    ("E5_plus_RGB_coupling",    "qaoa",          dict(rgb_coupling=True)),
    ("E6_tied_X_eq_Y",          "qaoa",          dict(tie_xy=True)),
    ("E7_X_operator_only",      "qaoa",          dict(xy_mode="x_only")),
    ("E8_Y_operator_only",      "qaoa",          dict(xy_mode="y_only")),
    ("E9_no_nonlinear_sbox",    "qaoa",          dict(sbox="identity")),
    ("E10_separable_objective", "greedy",        None),
    ("E11_no_global_chain(diag)", "qaoa",        dict(global_chain=False)),
]


def run(cfg, imgs, out):
    rows = []; algebra = {}
    for eid, m, ab in EXPS:
        c = dict(cfg, **(ab or {})); algebra[eid] = CA.affine_reconstruction_attack(c)["exact_prediction_rate"]
    for name, im in imgs.items():
        for eid, m, ab in EXPS:
            r, _ = evaluate(im, cfg, m, ab); r.update(image=name, experiment=eid, affine_reconstruction=algebra[eid]); rows.append(r)
            print(f"[ablation] {name:10s} {eid:27s} J={r['J']:.4f} NPCR={r['NPCR']:.3f} H={r['entropy_mean']:.4f} affine_recon={r['affine_reconstruction']:.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/ablation.csv", index=False)
    cols = ["J", "ops_total", "cfgs_used", "entropy_mean", "chi2_mean", "corr_h", "corr_v", "corr_d", "corr_RG", "MI_RG", "NPCR", "UACI", "local_resid_max", "affine_reconstruction"]
    g = df.groupby("experiment", sort=False)[cols].mean(); g.to_csv(f"{out}/metrics/ablation_summary.csv"); print("\n" + g.round(4).to_string()); return df
