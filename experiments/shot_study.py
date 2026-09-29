"""Shots vs solution quality. QAOA parameters are trained ONCE per instance (p=2, 40 circuit evals); only the read-out varies.
Read-out with k shots = sample k bitstrings from |psi|^2, keep the lowest classical energy. Compared with k uniform-random samples
and k-evaluation simulated annealing / local search (same number of classically evaluated bitstrings)."""
import numpy as np, pandas as pd
from src.utils import sample_images
from src.quantum_optimizer import qaoa_solve, exact_solve, sa_solve, random_solve, local_search_solve, gap
from experiments.qaoa_scaling import instances

SHOTS = [16, 32, 64, 128, 256, 512, 1024]


def run(cfg, out, quick=False, reps=20):
    q = cfg["qaoa"]; rows = []
    for n in ([12] if quick else [12, 15]):
        inst = instances(cfg, 3, [n])[n][: (3 if quick else 8)]
        for (name, wi, sub) in inst:
            ex = exact_solve(sub); Eo, Emax = ex.energy, ex.extra["E_max"]
            r = qaoa_solve(sub, q["p"], q["maxiter"], 16, wi, 1, keep_probs=True); pr, E = r.extra["probs"], r.extra["E"]
            rng = np.random.default_rng(wi)
            for k in SHOTS:
                for rep in range(reps):
                    s = rng.choice(len(pr), size=k, p=pr); e_q = E[s].min()
                    e_r = random_solve(sub, k, rep).energy; e_s = sa_solve(sub, k, rep).energy; e_l = local_search_solve(sub, k, rep).energy
                    for algo, e in (("QAOA", e_q), ("Random", e_r), ("SA", e_s), ("LocalSearch", e_l)):
                        rows.append(dict(n_qubits=n, win=wi, image=name, shots=k, algo=algo, gap=gap(e, Eo), hit=abs(e - Eo) < 1e-9,
                                         AR=(Emax - e) / max(Emax - Eo, 1e-12)))
            rows.append(dict(n_qubits=n, win=wi, image=name, shots=0, algo="QAOA-argmax", gap=gap(r.extra["argmax_energy"], Eo),
                             hit=abs(r.extra["argmax_energy"] - Eo) < 1e-9, AR=(Emax - r.extra["argmax_energy"]) / max(Emax - Eo, 1e-12)))
        print(f"[shots] n={n} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{out}/metrics/shot_study.csv", index=False)
    g = df.groupby(["n_qubits", "algo", "shots"])[["gap", "hit", "AR"]].mean(); g.to_csv(f"{out}/metrics/shot_study_summary.csv")
    plot(df, out); return df


def plot(df, out):
    from src.utils import _plt
    plt = _plt()
    for n in sorted(df.n_qubits.unique()):
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); d = df[(df.n_qubits == n) & (df.shots > 0)]
        for a, (col, lab) in zip(ax, [("hit", "optimum hit rate"), ("AR", "approximation ratio"), ("gap", "objective gap (J-J*)/|J*|")]):
            for algo in ["QAOA", "Random", "SA", "LocalSearch"]:
                g = d[d.algo == algo].groupby("shots")[col].mean(); a.plot(g.index, g.values, marker="o", label=algo)
            a.set_xscale("log", base=2); a.set_xlabel("shots (= classically evaluated bitstrings)"); a.set_ylabel(lab)
            if col == "gap": a.set_yscale("log")
        ax[0].legend(fontsize=8); fig.suptitle(f"Shot study, {n}-qubit window QUBOs (real image instances)", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{out}/plots/shot_study_{n}q.png", dpi=130); plt.close(fig)
