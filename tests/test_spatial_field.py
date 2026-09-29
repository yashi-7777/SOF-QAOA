import numpy as np, pytest
from src.utils import load_config, sample_images
from src import operators as ops
from src.spatial_operator_field import (build_problem, combine, TERMS, cfg_to_x, x_to_cfg, pack_field, unpack_field, select_field, windows,
                                        frozen_bits, solve_windows)
from src.chaotic_maps import ChaosEngine
from src.key_derivation import master_key

CFG = load_config("config.yaml"); IM = sample_images(64)["astronaut"] if "astronaut" in sample_images(64) else sample_images(64)["synthetic"]
CFG4 = dict(CFG, rgb_coupling=True); PROB = build_problem(IM, CFG4)


def direct_terms(prob, cfgs):
    """Independent evaluation of the four (un-normalised-then-range-normalised) terms from their definitions."""
    cal = prob["cal"]; gh, gw = prob["gh"], prob["gw"]; dem = prob["dem"]; Z = ops.bits_of(np.arange(16)).astype(float)
    cols = [np.ones(16)] + [Z[:, i] for i in range(4)] + [Z[:, i] * Z[:, j] for i in range(4) for j in range(i + 1, 4)]; A = np.stack(cols, 1)
    fit = {k: A @ cal["fit"][k]["coef"] for k in ("Dx", "Dy", "Dc")}                               # fitted DEFICITS (1-D), per config
    lin = {k: np.linalg.lstsq(np.hstack([np.ones((16, 1)), Z]), 1 - cal[k], rcond=None)[0] for k in ("Dx", "Dy")}
    g = {k: np.hstack([np.ones((16, 1)), Z]) @ lin[k] for k in lin}
    N1 = (dem["dx"] + dem["dy"] + dem["dc"]).sum(); T1 = sum(dem["dx"][i, j] * fit["Dx"][c] + dem["dy"][i, j] * fit["Dy"][c] + dem["dc"][i, j] * fit["Dc"][c]
                                                                for (i, j), c in np.ndenumerate(cfgs)) / N1
    N2 = sum(p[3] for p in prob["pairs"]); f = cfgs.reshape(-1)
    T2 = sum(p[3] / N2 * (g["Dx"] if p[2] == "x" else g["Dy"])[f[p[0]]] * (g["Dx"] if p[2] == "x" else g["Dy"])[f[p[1]]] for p in prob["pairs"])
    bits = ops.bits_of(f); T3 = sum((bits[p[0]] == bits[p[1]]).sum() for p in prob["pairs"]) / (4 * len(prob["pairs"]))
    smin, smax = prob["ops_range"]; Bt = prob["budget_target"]; Dn = max(Bt - smin, smax - Bt) ** 2
    T4 = (ops.compute_cost(f, prob["opts"]).sum() - Bt) ** 2 / Dn
    return dict(zip(TERMS, [T1, T2, T3, T4]))


def test_qubo_matches_direct_definition_of_every_term():
    rng = np.random.default_rng(0)
    for _ in range(6):
        cfgs = rng.integers(0, 16, (PROB["gh"], PROB["gw"])).astype(np.uint8); d = direct_terms(PROB, cfgs); x = cfg_to_x(cfgs)
        for t in TERMS:
            lo, hi = PROB["term_ranges"][t]; r = (hi - lo) if hi - lo > 1e-12 else 1.0
            assert np.isclose(PROB["terms"][t].energy(x)[0], (d[t] - lo) / r, atol=1e-9), t


def test_qubo_is_at_most_quadratic_and_symmetric():
    for t in TERMS:
        q = PROB["terms"][t]; assert np.allclose(q.Q, q.Q.T) and np.allclose(np.diag(q.Q), 0)


def test_term_weights_change_objective_and_are_normalised():
    Q1 = combine(PROB, CFG4); Q2 = combine(PROB, CFG4, weights=[1, 0, 0, 0]); x = cfg_to_x(np.zeros((PROB["gh"], PROB["gw"]), np.uint8))
    assert np.isclose(Q2.energy(x)[0], PROB["terms"]["T1_deficit"].energy(x)[0]) and Q1.energy(x)[0] != Q2.energy(x)[0]


def test_one_config_per_block_by_construction_and_bit_mapping():
    c = np.arange(16, dtype=np.uint8).reshape(4, 4); assert (x_to_cfg(cfg_to_x(c), 16).reshape(4, 4) == c).all()
    assert (unpack_field(pack_field(c), 4, 4) == c).all()


def test_frozen_bits_respected_by_every_selector():
    prob = build_problem(IM, CFG); assert prob["frozen"] == {3: 0}; eng = ChaosEngine(master_key("k", b"n" * 16))
    for m in ["fixed", "fixed_max", "random", "greedy", "sa_full", "ls_full", "random_window", "sa_window", "exact_window", "qaoa"]:
        f, _ = select_field(prob, eng, CFG, m); assert ((f >> 3) & 1).sum() == 0, m
    for ab, bit in ((dict(xy_mode="x_only"), 1), (dict(xy_mode="y_only"), 0)):
        c = dict(CFG, **ab); p = build_problem(IM, c); f, _ = select_field(p, eng, c, "qaoa"); assert ((f >> bit) & 1).sum() == 0


def test_tied_xy_constraint_enforced():
    c = dict(CFG, tie_xy=True); p = build_problem(IM, c); eng = ChaosEngine(master_key("k", b"n" * 16))
    f, _ = select_field(p, eng, c, "exact_window"); assert ((f & 1) == ((f >> 1) & 1)).all()


def test_window_decomposition_qubit_count_and_monotone_objective():
    prob = build_problem(IM, CFG); Q = combine(prob, CFG); x, st = solve_windows(prob, Q, CFG, "qaoa", 0)
    assert st["max_qubits"] == 12 and st["windows"] == 2 * len(windows(prob["gh"], prob["gw"], CFG["window_shape"]))
    x1, _ = solve_windows(prob, Q, CFG, "qaoa", 0, sweeps=1); assert Q.energy(x)[0] <= Q.energy(x1)[0] + 1e-9        # 2nd sweep never worse


def test_qaoa_beats_random_and_fixed_on_objective():
    prob = build_problem(IM, CFG); eng = ChaosEngine(master_key("k", b"n" * 16)); J = {m: select_field(prob, eng, CFG, m)[1]["J"] for m in ["fixed", "random", "qaoa"]}
    assert J["qaoa"] < J["random"] and J["qaoa"] < J["fixed"]


def test_calibration_table_structure():
    cal = PROB["cal"]; assert cal["Dx"][0] == 0 and cal["Dx"][1] > 0.3 and cal["Dc"][8] > 0.1 and cal["Dc"][0] == 0     # additive chain fully predictable; nonlinear not
