import numpy as np, pytest
from src.quantum_optimizer import (Qubo, qaoa_probs_numpy, qaoa_probs_gates_pennylane, qaoa_solve, sa_solve, exact_solve, random_solve,
                                   local_search_solve, energy_scale, pauli_terms, gap, MAX_SIM)
rng = np.random.default_rng(0)


def rand_qubo(n, seed=0):
    r = np.random.default_rng(seed); Q = r.normal(size=(n, n)); Q = (Q + Q.T) / 2; np.fill_diagonal(Q, 0); return Qubo(r.normal(size=n), Q, 0.3)


def test_qubo_ising_equivalence():
    q = rand_qubo(8); c, hz, J = q.to_ising(); X = rng.integers(0, 2, (60, 8)); S = 1 - 2 * X
    assert np.allclose(c + S @ hz + np.einsum("ij,jk,ik->i", S, J, S), q.energy(X))


def test_restrict_is_exact():
    q = rand_qubo(10); x = rng.integers(0, 2, 10).astype(np.uint8); free = np.array([1, 4, 5, 8]); sub = q.restrict(free, x)
    for _ in range(20):
        y = x.copy(); y[free] = rng.integers(0, 2, 4); assert np.isclose(sub.energy(y[free])[0], q.energy(y)[0])


def test_state_normalised_and_diag_matches_energy():
    q = rand_qubo(6); d = q.diagonal(); assert d.shape == (64,) and np.isclose(q.energy((np.arange(6) * 0 + 1))[0], d[-1])
    assert abs(qaoa_probs_numpy((d - q.c) * energy_scale(q), [.7, 1.1], [.3, .5], 6).sum() - 1) < 1e-12


def test_gate_level_circuit_equals_diagonal_simulation():
    pytest.importorskip("pennylane"); q = rand_qubo(6); sc = energy_scale(q); g, b = [0.7, 1.1], [0.3, 0.5]
    d = (q.diagonal() - q.c) * sc
    # PennyLane wire 0 is the MOST significant bit, our bit i is the i-th least significant -> reverse the index bits
    idx = np.arange(64); rev = np.array([int(format(i, "06b")[::-1], 2) for i in idx])
    assert np.allclose(qaoa_probs_numpy(d, g, b, 6)[rev], qaoa_probs_gates_pennylane(q, sc, g, b), atol=1e-9)


def test_qaoa_output_valid_and_exposes_circuit_details():
    q = rand_qubo(8); r = qaoa_solve(q, p=2, maxiter=30, shots=64, seed=1)
    e = r.extra; assert r.x.shape == (8,) and set(np.unique(r.x)) <= {0, 1} and np.isclose(q.energy(r.x)[0], r.energy)
    assert e["n_qubits"] == 8 and e["p"] == 2 and len(e["gammas"]) == 2 and len(e["betas"]) == 2 and e["shots"] == 64 and r.evals >= 30
    assert r.energy >= exact_solve(q).energy - 1e-12


def test_qaoa_expectation_beats_uniform_state_and_random_guess():
    q = rand_qubo(10, 3); r = qaoa_solve(q, p=2, maxiter=50, shots=128, seed=0)
    assert r.extra["exp_energy"] < r.extra["uniform_energy"] and r.extra["approx_ratio_expect"] > 0.5


def test_classical_solvers_valid_and_bounded_by_exact():
    q = rand_qubo(10, 5); ex = exact_solve(q).energy
    for f in (lambda: sa_solve(q, 300, 0), lambda: local_search_solve(q, 300, 0), lambda: random_solve(q, 300, 0)):
        r = f(); assert np.isclose(q.energy(r.x)[0], r.energy) and r.energy >= ex - 1e-12


def test_sim_cap_and_pauli_terms():
    with pytest.raises(AssertionError): qaoa_solve(rand_qubo(MAX_SIM + 1), 1, 2, 1)
    t = pauli_terms(rand_qubo(8)); assert t["max_locality"] == 2 and t["two_body"] == 28


def test_gap_definition():
    assert np.isclose(gap(1.1, 1.0), 0.1) and gap(1.0, 1.0) == 0.0
