"""Spatial Operator Field  Omega = [c_ij]  (c_ij in 0..15, 4 bits per block) chosen by a constrained QUBO.

Variables  x_{b,k} in {0,1}, b = block index, k in {zX,zY,zB,zR}                       (n = 4 * #blocks)

J(x) = sum_k w_k (T_k - min T_k)/(max T_k - min T_k),   w on the simplex (default 1/4 each = equal influence by construction)
 T1  demand-supply diffusion deficit:  sum_b [ dx_b (1-Dx(z_b)) + dy_b (1-Dy(z_b)) + dc_b (1-Dc(z_b)) ] / sum_b (dx+dy+dc)
       d* = r_b * |feature| from the block's own features; D* = MEASURED differential-unpredictability of each
       configuration (operators.calibrate) fitted by an exact quadratic in the 4 bits.
 T2  boundary continuity:  sum_{adjacent a,b} rho(a,b) * gX(z_a) * gX(z_b)  (horizontal),  gY (vertical)   [bilinear]
       rho = |corr| across the shared boundary, g = least-squares AFFINE fit of the measured deficit (1-D): the seam stays
       predictable only if BOTH sides are weak
 T3  spatial diversity:  fraction of equal configuration bits between adjacent blocks  (anti-ferromagnetic, design assumption)
 T4  computational budget:  ((sum_b ops(z_b) - B)/norm)^2,  B = phi * max-ops   -> dense all-to-all quadratic coupling
All terms are at most quadratic in x  =>  exact QUBO.  Large problems are solved by window (Gauss-Seidel) decomposition:
each window's sub-QUBO is exact given the current values of all other variables and is what QAOA receives.
NPCR/UACI are never used as objectives (they exist only after encryption).
"""
import time
import numpy as np
from . import operators as ops
from .feature_extraction import block_features, boundary_corr, demands
from .quantum_optimizer import (Qubo, qaoa_solve, exact_solve, random_solve, sa_solve, local_search_solve,
                                MAX_SIM, bits_to_int, int_to_bits)

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
TERMS = ["T1_deficit", "T2_boundary", "T3_diversity", "T4_budget"]


def _poly_parts(coef):
    """coef layout [1, z0..z3, z0z1,z0z2,z0z3,z1z2,z1z3,z2z3] -> (const, lin(4), quad(4,4 symmetric))."""
    q = np.zeros((4, 4))
    for m, (i, j) in enumerate(PAIRS): q[i, j] = q[j, i] = coef[5 + m]
    return coef[0], coef[1:5].copy(), q


def frozen_bits(cfg):
    """Bits fixed by ablations (ops.calibrate/optimizer never see them as free variables)."""
    fz = {}
    if not cfg.get("rgb_coupling", True): fz[3] = 0
    m = cfg.get("xy_mode", "full")
    if m == "x_only": fz[1] = 0
    if m == "y_only": fz[0] = 0
    return fz


def op_opts(cfg):
    m = cfg.get("xy_mode", "full")
    return dict(bx=0 if m == "y_only" else 1, by=0 if m == "x_only" else 1)


def build_problem(img_padded, cfg):
    B = cfg["block_size"]; opts = op_opts(cfg)
    F = block_features(img_padded, B); rx, ry = boundary_corr(img_padded, B)
    gh, gw = F.shape[:2]; nb = gh * gw; n = 4 * nb
    cal = ops.calibrate(B, opts=opts); dem = demands(F)
    fx, fy, fc = (_poly_parts(cal["fit"][k]["coef"]) for k in ("Dx", "Dy", "Dc"))   # polys of (1 - D_k)
    Zc = ops.bits_of(np.arange(16)).astype(float); A1 = np.hstack([np.ones((16, 1)), Zc])
    glin = {d: np.linalg.lstsq(A1, 1 - cal[k], rcond=None)[0] for d, k in (("x", "Dx"), ("y", "Dy"))}   # best affine fit of deficit
    glin = {d: (v[0], v[1:]) for d, v in glin.items()}
    idx = lambda b, k: 4 * b + k
    Qs = {t: [np.zeros(n), np.zeros((n, n)), 0.0] for t in TERMS}                        # h, Q, const per term

    # ---- T1
    N1 = float((dem["dx"] + dem["dy"] + dem["dc"]).sum()) or 1.0
    h, Q, c0 = Qs["T1_deficit"]
    for b in range(nb):
        i, j = divmod(b, gw)
        for (const, lin, quad), d in ((fx, dem["dx"][i, j]), (fy, dem["dy"][i, j]), (fc, dem["dc"][i, j])):
            c0 += d * const / N1
            for k in range(4): h[idx(b, k)] += d * lin[k] / N1
            for k in range(4):
                for l in range(k + 1, 4):
                    Q[idx(b, k), idx(b, l)] += d * quad[k, l] / N1; Q[idx(b, l), idx(b, k)] += d * quad[k, l] / N1
    Qs["T1_deficit"][2] = c0
    # ---- T2, T3 over adjacent block pairs
    pairs = []
    for i in range(gh):
        for j in range(gw):
            if j + 1 < gw: pairs.append((i * gw + j, i * gw + j + 1, "x", rx[i, j]))
            if i + 1 < gh: pairs.append((i * gw + j, (i + 1) * gw + j, "y", ry[i, j]))
    N2 = float(sum(p[3] for p in pairs)) or 1.0; npairs = max(len(pairs), 1)
    h2, Q2, c2 = Qs["T2_boundary"]; h3, Q3, c3 = Qs["T3_diversity"]
    for a, b, dr, rho in pairs:
        f0, fl = glin[dr]; w = rho / N2
        c2 += w * f0 * f0
        for k in range(4):
            h2[idx(a, k)] += w * f0 * fl[k]; h2[idx(b, k)] += w * f0 * fl[k]
            for l in range(4):
                Q2[idx(a, k), idx(b, l)] += w * fl[k] * fl[l]; Q2[idx(b, l), idx(a, k)] += w * fl[k] * fl[l]
        for k in range(4):                                                  # eq_k = 1 - za - zb + 2 za zb
            wd = 1.0 / (4 * npairs); c3 += wd
            h3[idx(a, k)] -= wd; h3[idx(b, k)] -= wd
            Q3[idx(a, k), idx(b, k)] += 2 * wd; Q3[idx(b, k), idx(a, k)] += 2 * wd
    Qs["T2_boundary"][2] = c2; Qs["T3_diversity"][2] = c3
    # ---- T4 budget
    wv = np.zeros(n); S0 = 0.0
    for b in range(nb):
        S0 += opts["bx"] + opts["by"]
        wv[idx(b, 0)] = opts["bx"]; wv[idx(b, 1)] = opts["by"]; wv[idx(b, 2)] = opts["bx"] + opts["by"]; wv[idx(b, 3)] = 1
    fz = frozen_bits(cfg)
    allowed = [c for c in range(16) if all(((c >> k) & 1) == v for k, v in fz.items())]
    smax = float(ops.compute_cost(np.array(allowed), opts).max()) * nb
    smin = float(ops.compute_cost(np.array(allowed), opts).min()) * nb
    Bt = cfg["weights"]["budget_frac"] * (smax - smin) + smin                 # target between min and max ops
    Dn = max(Bt - smin, smax - Bt) ** 2 or 1.0
    h4, Q4, c4 = Qs["T4_budget"]
    c4 = (S0 - Bt) ** 2 / Dn; h4 += (2 * (S0 - Bt) * wv + wv ** 2) / Dn
    Q4 += 2 * np.outer(wv, wv) / Dn; np.fill_diagonal(Q4, 0.0); Qs["T4_budget"][2] = c4
    terms = {t: Qubo(*Qs[t][:2], Qs[t][2]) for t in TERMS}
    # ---- min-max normalisation of every term over its achievable range (parameter-free balancing)
    def cfg_x(b, c):
        z = np.zeros(n); z[4 * b:4 * b + 4] = [(c >> k) & 1 for k in range(4)]; return z
    T1 = terms["T1_deficit"]; e0 = T1.energy(np.zeros(n))[0]
    per = np.array([[T1.energy(cfg_x(b, c))[0] - e0 for c in allowed] for b in range(nb)])
    rng_ = {"T1_deficit": (e0 + per.min(1).sum(), e0 + per.max(1).sum())}
    zc = lambda c: np.array([(c >> k) & 1 for k in range(4)])
    g = {d: np.array([f[0] + f[1] @ zc(c) for c in allowed]) for d, f in glin.items()}
    prod = {d: (min(g[d].min() * g[d].max(), g[d].min() ** 2, g[d].max() ** 2), max(g[d].min() * g[d].max(), g[d].min() ** 2, g[d].max() ** 2)) for d in g}
    rng_["T2_boundary"] = (sum(p[3] / N2 * prod[p[2]][0] for p in pairs), sum(p[3] / N2 * prod[p[2]][1] for p in pairs))
    rng_["T3_diversity"] = (len(fz) / 4.0, 1.0)
    rng_["T4_budget"] = (0.0, 1.0)
    for t in TERMS:
        lo, hi = rng_[t]; r = (hi - lo) if hi - lo > 1e-12 else 1.0
        terms[t] = Qubo(terms[t].h / r, terms[t].Q / r, (terms[t].c - lo) / r)
    return dict(F=F, dem=dem, rho_x=rx, rho_y=ry, gh=gh, gw=gw, nb=nb, n=n, terms=terms, cal=cal, opts=opts,
                frozen=fz, pairs=pairs, budget_target=Bt, ops_range=(smin, smax), term_ranges=rng_, allowed=allowed)


def combine(prob, cfg, weights=None):
    w = weights if weights is not None else cfg["weights"]["term_weights"]
    w = np.asarray([w[t] for t in TERMS] if isinstance(w, dict) else w, float); w = w / w.sum()
    Q = Qubo(sum(wi * prob["terms"][t].h for wi, t in zip(w, TERMS)), sum(wi * prob["terms"][t].Q for wi, t in zip(w, TERMS)),
             sum(wi * prob["terms"][t].c for wi, t in zip(w, TERMS)))
    m = cfg.get("xy_mode", "full")
    if cfg.get("tie_xy", False):                                             # constraint zX == zY  (penalty, M >> J range)
        M = 5.0; h = Q.h.copy(); QQ = Q.Q.copy()
        for b in range(prob["nb"]):
            a, c_ = 4 * b, 4 * b + 1; h[a] += M; h[c_] += M; QQ[a, c_] -= 2 * M; QQ[c_, a] -= 2 * M
        Q = Qubo(h, QQ, Q.c)
    return Q


def x_to_cfg(x, nb):
    x = np.asarray(x).reshape(nb, 4).astype(int); return (x[:, 0] + 2 * x[:, 1] + 4 * x[:, 2] + 8 * x[:, 3]).astype(np.uint8)


def cfg_to_x(c):
    return np.concatenate([[(int(v) >> k) & 1 for k in range(4)] for v in np.ravel(c)]).astype(np.uint8)


def term_values(prob, x):
    return {t: float(prob["terms"][t].energy(x)[0]) for t in TERMS}


def windows(gh, gw, shape):
    a, b = shape; out = []
    for i0 in range(0, gh, a):
        for j0 in range(0, gw, b):
            out.append([i * gw + j for i in range(i0, min(i0 + a, gh)) for j in range(j0, min(j0 + b, gw))])
    return out


def _free_vars(blocks, frozen):
    return np.array([4 * b + k for b in blocks for k in range(4) if k not in frozen], dtype=int)


def solve_windows(prob, Q, cfg, method, seed=0, sweeps=None):
    """Gauss-Seidel decomposition. method in qaoa | exact | sa | ls | random  (identical windows/sweeps/objective)."""
    q = cfg["qaoa"]; shape = tuple(cfg["window_shape"]); sweeps = sweeps or cfg["sweeps"]
    x = np.zeros(prob["n"], dtype=np.uint8)
    for k, v in prob["frozen"].items(): x[k::4] = v
    stats = dict(circuit_evals=0, classical_evals=0, qaoa_runtime=0.0, classical_runtime=0.0, windows=0, max_qubits=0,
                 p_opt=[], approx=[], gammas=[], betas=[], exp_energy=[])
    budget = q["maxiter"] + 1 + q["shots"]                                   # bitstring evaluations QAOA needs per window
    for sw in range(sweeps):
        for wi, blocks in enumerate(windows(prob["gh"], prob["gw"], shape)):
            free = _free_vars(blocks, prob["frozen"])
            if len(free) == 0: continue
            sub = Q.restrict(free, x); s = seed + 1000 * sw + wi; stats["windows"] += 1; stats["max_qubits"] = max(stats["max_qubits"], len(free))
            if method == "qaoa":
                r = qaoa_solve(sub, q["p"], q["maxiter"], q["shots"], s, q.get("restarts", 1), q.get("backend", "numpy"))
                stats["circuit_evals"] += r.evals; stats["qaoa_runtime"] += r.runtime; stats["classical_evals"] += q["shots"]
                for k_, key in (("p_opt", "p_opt"), ("exp_energy", "exp_energy")): stats[k_].append(r.extra[key])
                stats["gammas"].append(r.extra["gammas"]); stats["betas"].append(r.extra["betas"])
                if len(free) <= MAX_SIM and len(free) <= 20:                 # approximation ratio needs the spectrum
                    stats["approx"].append(r.extra["approx_ratio_expect"])
            else:
                r = {"exact": lambda: exact_solve(sub), "sa": lambda: sa_solve(sub, budget, s),
                     "ls": lambda: local_search_solve(sub, budget, s), "random": lambda: random_solve(sub, budget, s)}[method]()
                stats["classical_evals"] += r.evals; stats["classical_runtime"] += r.runtime
            # never accept a worse assignment than the incumbent (monotone sweeps; cost = 1 classical evaluation)
            cur = sub.energy(x[free])[0]; stats["classical_evals"] += 1
            if r.energy <= cur + 1e-12: x[free] = r.x
    return x, stats


def solve_full(prob, Q, cfg, method, seed=0, evals=None):
    evals = evals or cfg["full_budget"]
    free = _free_vars(range(prob["nb"]), prob["frozen"]); x = np.zeros(prob["n"], dtype=np.uint8)
    for k, v in prob["frozen"].items(): x[k::4] = v
    sub = Q.restrict(free, x)
    r = {"sa": lambda: sa_solve(sub, evals, seed), "ls": lambda: local_search_solve(sub, evals, seed),
         "random": lambda: random_solve(sub, evals, seed)}[method]()
    x[free] = r.x
    return x, dict(circuit_evals=0, classical_evals=r.evals, qaoa_runtime=0.0, classical_runtime=r.runtime, windows=1,
                   max_qubits=0, p_opt=[], approx=[], gammas=[], betas=[], exp_energy=[])


METHODS = ["fixed", "fixed_max", "random", "greedy", "sa_full", "ls_full", "random_window", "sa_window", "ls_window",
           "exact_window", "qaoa"]


def select_field(prob, engine, cfg, method, seed=0, weights=None):
    """-> (cfgs (gh,gw) uint8, info). `engine` supplies the keyed random field for method 'random'."""
    t0 = time.perf_counter(); nb = prob["nb"]; Q = combine(prob, cfg, weights); fz = prob["frozen"]
    stats = dict(circuit_evals=0, classical_evals=0, qaoa_runtime=0.0, classical_runtime=0.0, windows=0, max_qubits=0,
                 p_opt=[], approx=[], gammas=[], betas=[], exp_energy=[])
    if method == "fixed":                                                    # best UNIFORM configuration under the same J
        best = None
        for c in prob["allowed"]:
            x = cfg_to_x(np.full(nb, c)); e = Q.energy(x)[0]
            if best is None or e < best[0]: best = (e, x)
        x = best[1]; stats["classical_evals"] = 16
    elif method == "fixed_max":
        c = max(prob['allowed'], key=lambda c_: float(ops.compute_cost(np.array([c_]), prob['opts'])[0])); x = cfg_to_x(np.full(nb, c))
    elif method == "random":                                                 # keyed pseudo-random field (needs no header bytes)
        al = np.array(prob["allowed"]); b = al[engine.x_bytes("field:random", nb).astype(np.int64) % len(al)]
        x = cfg_to_x(b.astype(np.uint8))
    elif method == "greedy":                                                 # separable classical rule: block-wise argmin of T1 only
        x = np.zeros(prob["n"], np.uint8); T1 = prob["terms"]["T1_deficit"]
        for b in range(nb):
            best = None
            for c in prob["allowed"]:
                xb = np.array([(c >> k) & 1 for k in range(4)])
                z = np.zeros(prob["n"]); z[4 * b:4 * b + 4] = xb
                e = T1.energy(z)[0]
                if best is None or e < best[0] - 1e-12: best = (e, xb)
            x[4 * b:4 * b + 4] = best[1]
        stats["classical_evals"] = len(prob["allowed"]) * nb
    elif method in ("sa_full", "ls_full", "random_window"):
        if method == "random_window": x, stats = solve_windows(prob, Q, cfg, "random", seed)
        else: x, stats = solve_full(prob, Q, cfg, method.split("_")[0], seed)
    elif method in ("sa_window", "ls_window", "exact_window", "qaoa"):
        x, stats = solve_windows(prob, Q, cfg, {"sa_window": "sa", "ls_window": "ls", "exact_window": "exact", "qaoa": "qaoa"}[method], seed)
    else:
        raise ValueError(method)
    cfgs = x_to_cfg(x, nb).reshape(prob["gh"], prob["gw"])
    info = dict(stats, method=method, J=float(Q.energy(x)[0]), terms=term_values(prob, x), runtime=time.perf_counter() - t0,
                cfg_hist=np.bincount(cfgs.ravel(), minlength=16).tolist(), n_vars=prob["n"] - len(prob["frozen"]) * prob["nb"],
                ops_total=float(ops.compute_cost(cfgs, prob["opts"]).sum()), ops_target=prob["budget_target"])
    return cfgs, info


# ------------------------------------------------------------------ compact transmitted representation (4 bits / block)
def pack_field(c):
    v = np.asarray(c, np.uint8).ravel(); v = np.concatenate([v, np.zeros(len(v) % 2, np.uint8)]).reshape(-1, 2)
    return (v[:, 0] | (v[:, 1] << 4)).astype(np.uint8).tobytes()


def unpack_field(b, gh, gw):
    a = np.frombuffer(b, dtype=np.uint8); v = np.stack([a & 15, a >> 4], 1).reshape(-1)
    return v[: gh * gw].reshape(gh, gw).astype(np.uint8)
