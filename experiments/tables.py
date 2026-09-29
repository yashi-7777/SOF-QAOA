"""Paper tables from the CSVs written by the experiments (nothing is typed by hand)."""
import os, pandas as pd, numpy as np


def md(df, floatfmt="{:.4g}"):
    df = df.copy(); cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build(out):
    m = f"{out}/metrics"; T = {}; rd = lambda n: pd.read_csv(f"{m}/{n}.csv") if os.path.exists(f"{m}/{n}.csv") else None
    s = rd("security_suite")
    if s is not None: T["Table1_security_metrics"] = s.drop(columns=["chi2_pass"]).round(4)
    q = rd("solver_quality")
    if q is not None:
        g = q.groupby(["n_qubits", "algo"], sort=False).agg(gap=("gap", "mean"), hit_rate=("hit", "mean"), approx_ratio=("AR", "mean"),
                                                        evals=("evals", "mean"), time_s=("t", "mean"), instances=("gap", "size")).reset_index()
        T["Table2_QAOA_vs_SA_vs_Random_vs_Exact"] = g
    f = rd("full_image_problem")
    if f is not None: T["Table2b_full_image_problem"] = f
    sh = rd("shot_study")
    if sh is not None:
        p = sh[sh.shots > 0].groupby(["n_qubits", "algo", "shots"])[["hit", "AR", "gap"]].mean().reset_index()
        T["Table3_shot_study"] = p
    if q is not None:
        p = q[q.algo.isin(["Exact", "QAOA", "SA@equal", "LS@equal", "random@equal"])].pivot_table(index="n_qubits", columns="algo", values="t", aggfunc="mean").reset_index()
        pr = rd("qaoa_21q_probe")
        T["Table4_scaling_runtime_seconds"] = p
        if pr is not None: T["Table4b_21_qubit_probe"] = pr
        if f is not None: T["Table4c_full_image_scaling"] = f[f.method.isin(["sa_full", "ls_full", "exact_window", "sa_window", "qaoa"])][
            ["size", "n_vars", "method", "J", "gap_to_best_known", "circuit_evals", "classical_evals", "qaoa_s", "total_s", "windows", "qubits_per_window"]]
    a = rd("ablation_summary")
    if a is not None: T["Table5_ablation"] = a.round(4)
    c = rd("complexity")
    if c is not None: T["Table6_computational_complexity"] = c.round(4)
    x = rd("cryptanalysis")
    if x is not None: T["Table7_cryptanalysis"] = x.round(4)
    for n, t in (("rgb_coupling_investigation", "Table8_RGB_coupling"), ("weight_sensitivity", "Table9_weight_sensitivity"), ("robustness", "Table10_robustness"), ("field_leakage", "Table11_field_leakage"), ("baselines_summary", "Table12_selector_comparison")):
        d = rd(n)
        if d is not None:
            if n == "robustness": d = d.groupby("attack")[["PSNR_dB", "exact_pixels_pct"]].mean().reset_index()
            if n == "weight_sensitivity": d = d.describe().T.reset_index().rename(columns={"index": "quantity"})
            T[t] = d.round(4)
    os.makedirs(f"{out}/tables", exist_ok=True); doc = []
    for k, d in T.items():
        d.to_csv(f"{out}/tables/{k}.csv", index=False); doc += [f"### {k}\n", md(d), "\n"]
    open(f"{out}/tables/ALL_TABLES.md", "w").write("\n".join(doc)); return T
