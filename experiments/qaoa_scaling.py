"""QAOA vs exact / SA / local-search / random on REAL image QUBOs of growing size, plus whole-image (exact-infeasible) problems."""
import time, numpy as np, pandas as pd
from src.utils import sample_images
from src.spatial_operator_field import build_problem, combine, solve_full, windows, select_field
from src.quantum_optimizer import (qaoa_solve, exact_solve, sa_solve, random_solve, local_search_solve, gap, MAX_SIM, pauli_terms)

SHAPES = {6: (1, 2), 9: (1, 3), 12: (2, 2), 15: (1, 5), 18: (2, 3)}        # blocks per window x 3 free bits/block


def instances(cfg, n_per_size, sizes):
    """Real window sub-QUBOs: neighbours fixed to a whole-image SA solution (context), windows spread over the image."""
    out = {n: [] for n in sizes}
    for name, im in sample_images(cfg["experiments"]["image_size"]).items():
        prob = build_problem(im, cfg); Q = combine(prob, cfg); x, _ = solve_full(prob, Q, cfg, "sa", 0)
        for n in sizes:
            ws = windows(prob["gh"], prob["gw"], SHAPES[n]); rng = np.random.default_rng(n)
            for wi in rng.choice(len(ws), min(n_per_size, len(ws)), replace=False):
                if len(ws[wi]) * 3 != n: continue
                free = np.array([4 * b + k for b in ws[wi] for k in range(3)]); out[n].append((name, int(wi), Q.restrict(free, x)))
    return out


def solver_table(cfg, out, quick=False):
    q = cfg["qaoa"]; sizes = [6, 9, 12] if quick else [6, 9, 12, 15, 18]; rows = []
    per = {6: 12, 9: 12, 12: 12, 15: 6, 18: 3}
    inst = instances(cfg, 3, sizes)
    for n in sizes:
        for (name, wi, sub) in inst[n][: (4 if quick else per[n])]:
            ex = exact_solve(sub); Eo, Emax, Em = ex.energy, ex.extra["E_max"], ex.extra["E_mean"]
            r = qaoa_solve(sub, q["p"], q["maxiter"], q["shots"], wi, q.get("restarts", 1), keep_probs=True)
            budget = r.evals + q["shots"]                              # bitstring evaluations QAOA needs (circuits + readout shots)
            base = {"random@equal": random_solve(sub, budget, wi), "SA@equal": sa_solve(sub, budget, wi),
                    "LS@equal": local_search_solve(sub, budget, wi), "SA@circuit-evals": sa_solve(sub, r.evals, wi),
                    "random@shots": random_solve(sub, q["shots"], wi)}
            AR = lambda E: (Emax - E) / max(Emax - Eo, 1e-12)
            rows.append(dict(n_qubits=n, image=name, win=wi, algo="QAOA", E=r.energy, gap=gap(r.energy, Eo), hit=abs(r.energy - Eo) < 1e-9,
                             AR=AR(r.energy), evals=r.evals, extra_evals=q["shots"], t=r.runtime, p_opt=r.extra["p_opt"],
                             AR_expect=r.extra["approx_ratio_expect"], gap_argmax=gap(r.extra["argmax_energy"], Eo),
                             gammas=str(np.round(r.extra["gammas"], 3).tolist()), betas=str(np.round(r.extra["betas"], 3).tolist()),
                             pauli=str(pauli_terms(sub))))
            rows.append(dict(n_qubits=n, image=name, win=wi, algo="Exact", E=Eo, gap=0.0, hit=True, AR=1.0, evals=ex.evals, t=ex.runtime))
            rows.append(dict(n_qubits=n, image=name, win=wi, algo="uniform-mean", E=Em, gap=gap(Em, Eo), hit=False, AR=AR(Em), evals=0, t=0.0))
            for k, b in base.items():
                rows.append(dict(n_qubits=n, image=name, win=wi, algo=k, E=b.energy, gap=gap(b.energy, Eo), hit=abs(b.energy - Eo) < 1e-9,
                                 AR=AR(b.energy), evals=b.evals, t=b.runtime))
        print(f"[scaling] n={n} done ({sum(1 for r in rows if r['n_qubits']==n and r['algo']=='QAOA')} instances)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/solver_quality.csv", index=False)
    g = df.groupby(["n_qubits", "algo"], sort=False)[["gap", "hit", "AR", "evals", "t"]].mean(); g.to_csv(f"{out}/metrics/solver_quality_summary.csv")
    print(g.round(4).to_string()); return df


def big_qaoa_probe(cfg, out, n_q=21):
    """One instance beyond the routine range to document simulator cost; exact still feasible by enumeration."""
    from src.quantum_optimizer import Qubo
    im = sample_images(128)["astronaut"]; prob = build_problem(im, cfg); Q = combine(prob, cfg); x, _ = solve_full(prob, Q, cfg, "sa", 0)
    ws = windows(prob["gh"], prob["gw"], (1, 7)); w = [b for b in ws if len(b) == 7][0]
    free = np.array([4 * b + k for b in w for k in range(3)]); sub = Q.restrict(free, x); q = cfg["qaoa"]
    ex = exact_solve(sub); r = qaoa_solve(sub, q["p"], q["maxiter"], q["shots"], 0)
    row = dict(n_qubits=len(free), exact_s=ex.runtime, qaoa_s=r.runtime, circuit_evals=r.evals, qaoa_gap=gap(r.energy, ex.energy),
               qaoa_hit=bool(abs(r.energy - ex.energy) < 1e-9), sa_gap=gap(sa_solve(sub, r.evals + q["shots"], 0).energy, ex.energy))
    pd.DataFrame([row]).to_csv(f"{out}/metrics/qaoa_21q_probe.csv", index=False); print("[scaling] 21-qubit probe:", row); return row


def full_image_table(cfg, out, sizes=(128, 256), quick=False):
    """Whole-image QUBOs (192 / 768 free variables): exact enumeration is INFEASIBLE (2^192, 2^768). Best-known = min over all solvers."""
    rows = []
    for size in ([128] if quick else sizes):
        im = sample_images(size)["astronaut"] if "astronaut" in sample_images(size) else sample_images(size)["synthetic"]
        prob = build_problem(im, cfg); eng = type("E", (), {"x_bytes": lambda s, l, n: np.random.default_rng(1).integers(0, 256, n).astype(np.uint8)})()
        res = {}
        for m in ["fixed", "random", "greedy", "random_window", "ls_window", "sa_window", "exact_window", "sa_full", "ls_full", "qaoa"]:
            f, info = select_field(prob, eng, cfg, m); res[m] = info
        best = min(v["J"] for v in res.values())
        for m, info in res.items():
            rows.append(dict(size=size, n_vars=info["n_vars"], method=m, J=info["J"], gap_to_best_known=(info["J"] - best) / abs(best),
                             circuit_evals=info["circuit_evals"], classical_evals=info["classical_evals"], qaoa_s=info["qaoa_runtime"],
                             total_s=info["runtime"], windows=info["windows"], qubits_per_window=info["max_qubits"], ops=info["ops_total"], ops_target=info["ops_target"]))
        print(f"[full-image] {size}x{size}: best-known J={best:.4f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/full_image_problem.csv", index=False)
    print(df.round(4).to_string()); return df


def run(cfg, out, quick=False):
    solver_table(cfg, out, quick)
    if not quick: big_qaoa_probe(cfg, out)
    full_image_table(cfg, out, quick=quick)
