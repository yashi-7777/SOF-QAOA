"""QUBO container + QAOA (statevector simulator) + matched classical solvers.

QUBO   E(x) = c + h.x + sum_{i<j} Q_ij x_i x_j,   x in {0,1}^n           (Q stored symmetric, zero diagonal)
Ising  x_i = (1 - s_i)/2  ->  H_C = c' + sum_i h'_i Z_i + sum_{i<j} J_ij Z_i Z_j    (`to_ising`)
QAOA   |psi(g,b)> = prod_{l=1..p} e^{-i b_l H_M} e^{-i g_l H_C} |+>^n ,  H_M = sum_q X_q
       cost layer  = diagonal phases  exp(-i g s (E(x) - c))  (== RZ/RZZ gates of `to_ising`, verified against PennyLane)
       s = 1/sqrt(sum h'^2 + sum J^2) : analytic energy scale (no enumeration of the spectrum is used).
Read-out: `shots` bitstrings sampled from |psi|^2 (simulated measurement); the best sampled bitstring, by its
classically evaluated QUBO energy, is returned (the same read-out rule a hardware run would use).
"""
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

MAX_ENUM = 26          # enumeration / statevector cap (2^26 doubles = 512 MB is the hard limit we allow)
MAX_SIM = 22           # QAOA statevector cap on this machine


class Qubo:

    def __init__(self, h, Q, c=0.0):
        self.h = np.asarray(h, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.c = float(c)
        self.n = len(self.h)

        if self.Q.shape != (self.n, self.n):
            raise ValueError(
                f"Q must have shape {(self.n, self.n)}, got {self.Q.shape}"
            )

        if not np.all(np.isfinite(self.h)):
            raise ValueError("QUBO linear coefficients contain NaN or Inf.")

        if not np.all(np.isfinite(self.Q)):
            raise ValueError("QUBO quadratic coefficients contain NaN or Inf.")

        if not np.isfinite(self.c):
            raise ValueError("QUBO constant contains NaN or Inf.")

        if not np.allclose(self.Q, self.Q.T, atol=1e-12):
            raise ValueError("QUBO matrix Q must be symmetric.")

        if not np.allclose(np.diag(self.Q), 0.0, atol=1e-12):
            raise ValueError("QUBO matrix diagonal must be zero.")

    def energy(self, X):
        X = np.atleast_2d(X).astype(np.float64, copy=False)

        if X.shape[1] != self.n:
            raise ValueError(
                f"Expected {self.n} binary variables, got {X.shape[1]}."
            )

        result = (
            self.c
            + X @ self.h
            + 0.5 * np.einsum(
                "ij,ij->i",
                X @ self.Q,
                X,
                optimize=True
            )
        )

        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "Non-finite QUBO energy detected."
            )

        return result

    def restrict(self, free, x_full):
        """Exact sub-QUBO over variables `free` with all other variables fixed to x_full (Gauss-Seidel step)."""
        free = np.asarray(free); mask = np.ones(self.n, bool); mask[free] = False; out = np.flatnonzero(mask)
        xo = x_full[out].astype(float)
        h = self.h[free] + self.Q[np.ix_(free, out)] @ xo
        c = self.c + self.h[out] @ xo + 0.5 * xo @ self.Q[np.ix_(out, out)] @ xo
        return Qubo(h, self.Q[np.ix_(free, free)], c)

    def to_ising(self):
        """(c', h', J) with H_C = c' + sum h'_i Z_i + sum_{i<j} J_ij Z_i Z_j ."""
        Qu = np.triu(self.Q, 1)
        hz = -0.5 * self.h - 0.25 * (self.Q.sum(1))
        J = 0.25 * Qu
        c = self.c + 0.5 * self.h.sum() + 0.125 * self.Q.sum()
        return c, hz, J

    def diagonal(self, chunk=1 << 16):
        assert self.n <= MAX_ENUM, "enumeration infeasible"
        N = 1 << self.n; out = np.empty(N); sh = np.arange(self.n)
        for a in range(0, N, chunk):
            idx = np.arange(a, min(a + chunk, N)); X = ((idx[:, None] >> sh) & 1)
            out[a:a + len(idx)] = self.energy(X)
        return out

    def n_terms(self):
        return dict(linear=int((np.abs(self.h) > 1e-12).sum()), quadratic=int((np.abs(np.triu(self.Q, 1)) > 1e-12).sum()))


@dataclass
class Solution:
    x: np.ndarray; energy: float; evals: int; runtime: float; extra: dict = field(default_factory=dict)


def bits_to_int(x): return int(sum(int(b) << i for i, b in enumerate(x)))
def int_to_bits(v, n): return np.array([(v >> i) & 1 for i in range(n)], dtype=np.uint8)


# ------------------------------------------------------------------ QAOA
def _mixer(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape((2,) * n)
    for q in range(n):
        a, b = np.take(psi, 0, axis=q), np.take(psi, 1, axis=q)
        psi = np.stack((c * a + s * b, s * a + c * b), axis=q)
    return psi.reshape(-1)


def qaoa_state_numpy(dscaled, gammas, betas, n):
    psi = np.full(1 << n, 2.0 ** (-n / 2), dtype=np.complex128)
    for g, b in zip(gammas, betas):
        psi = _mixer(psi * np.exp(-1j * g * dscaled), b, n)
    return psi


def qaoa_probs_numpy(dscaled, gammas, betas, n):
    return np.abs(qaoa_state_numpy(dscaled, gammas, betas, n)) ** 2


def qaoa_probs_gates_pennylane(qubo, scale, gammas, betas):
    """Same circuit written with elementary gates: H, RZ(2 g s h'_i), IsingZZ(2 g s J_ij), RX(2 b)."""
    import pennylane as qml
    n = qubo.n; _, hz, J = qubo.to_ising(); dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ():
        for w in range(n): qml.Hadamard(wires=w)
        for g, b in zip(gammas, betas):
            for i in range(n):
                if abs(hz[i]) > 1e-15: qml.RZ(2 * g * scale * hz[i], wires=i)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(J[i, j]) > 1e-15: qml.IsingZZ(2 * g * scale * J[i, j], wires=[i, j])
            for w in range(n): qml.RX(2 * b, wires=w)
        return qml.probs(wires=list(range(n)))
    return np.asarray(circ())


def energy_scale(qubo):
    _, hz, J = qubo.to_ising()
    return 1.0 / max(np.sqrt((hz ** 2).sum() + (J ** 2).sum()), 1e-12)


def qaoa_solve(qubo, p=2, maxiter=60, shots=256, seed=0, restarts=1, backend="numpy", init="ramp", keep_probs=False):
    """Optimise (gamma, beta) with COBYLA on the exact expectation <H_C>, then sample `shots` bitstrings."""
    n = qubo.n; assert n <= MAX_SIM, f"statevector simulation of {n} qubits exceeds cap {MAX_SIM}"
    t0 = time.perf_counter(); E = qubo.diagonal(); sc = energy_scale(qubo); d = (E - qubo.c) * sc
    evals = [0]; rng = np.random.default_rng(seed)

    def probs(g, b):
        evals[0] += 1
        return qaoa_probs_numpy(d, g, b, n) if backend == "numpy" else _pl_probs(d, g, b, n)

    def f(x): return float(probs(x[:p], x[p:]) @ d)

    best = None
    for r in range(restarts):
        base = (np.arange(p) + .5) / p
        x0 = np.concatenate([0.7 * base, 0.7 * (1 - base)]) if (init == "ramp" and r == 0) else rng.uniform(0, 1, 2 * p)
        res = minimize(f, x0, method="COBYLA", options=dict(maxiter=max(maxiter // restarts, 4), rhobeg=0.3))
        if best is None or res.fun < best.fun: best = res
    pr = probs(best.x[:p], best.x[p:]); pr = pr / pr.sum()                     # final read-out circuit
    samp = rng.choice(1 << n, size=shots, p=pr); i_best = int(samp[np.argmin(E[samp])])
    Emin, Emean = E.min(), E.mean()
    ex = dict(gammas=best.x[:p].tolist(), betas=best.x[p:].tolist(), p=p, n_qubits=n, shots=shots, scale=sc,
              exp_energy=float(pr @ E), p_opt=float(pr[np.isclose(E, Emin)].sum()), argmax_energy=float(E[int(np.argmax(pr))]),
              argmax_x=int_to_bits(int(np.argmax(pr)), n), circuit_evals=evals[0], restarts=restarts,
              approx_ratio_expect=float((E.max() - float(pr @ E)) / max(E.max() - Emin, 1e-12)), uniform_energy=float(Emean))
    if keep_probs: ex.update(probs=pr, E=E)
    return Solution(int_to_bits(i_best, n), float(E[i_best]), evals[0], time.perf_counter() - t0, ex)


def _pl_probs(d, g, b, n):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ():
        for w in range(n): qml.Hadamard(wires=w)
        for gg, bb in zip(g, b):
            qml.DiagonalQubitUnitary(np.exp(-1j * gg * d), wires=list(range(n)))
            for w in range(n): qml.RX(2 * bb, wires=w)
        return qml.probs(wires=list(range(n)))
    return np.asarray(circ())


# ------------------------------------------------------------------ classical solvers on the SAME QUBO
def exact_solve(qubo):
    t0 = time.perf_counter(); E = qubo.diagonal(); i = int(np.argmin(E))
    return Solution(int_to_bits(i, qubo.n), float(E[i]), len(E), time.perf_counter() - t0,
                    dict(E_max=float(E.max()), E_mean=float(E.mean())))


def random_solve(qubo, evals, seed=0):
    t0 = time.perf_counter(); rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (evals, qubo.n)); E = qubo.energy(X); i = int(np.argmin(E))
    return Solution(X[i].astype(np.uint8), float(E[i]), evals, time.perf_counter() - t0)


def sa_solve(qubo, evals, seed=0, T0=None, T1=None, x0=None):
    """Single-bit-flip simulated annealing; every proposed move = 1 QUBO-energy evaluation (O(n) incremental)."""
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); n = qubo.n
    scale = np.sqrt((qubo.h ** 2).sum() + (np.triu(qubo.Q, 1) ** 2).sum()) / np.sqrt(n)
    T0 = T0 or 0.5 * scale; T1 = T1 or 0.005 * scale
    x = rng.integers(0, 2, n).astype(float) if x0 is None else x0.astype(float).copy()
    field_ = qubo.h + qubo.Q @ x; E = qubo.energy(x)[0]; best, Eb = x.copy(), E
    for t in range(evals):
        T = T0 * (T1 / T0) ** (t / max(evals - 1, 1)); i = int(rng.integers(n))
        dE = (1 - 2 * x[i]) * field_[i]
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            field_ += (1 - 2 * x[i]) * qubo.Q[:, i]; x[i] = 1 - x[i]; E += dE
            if E < Eb: best, Eb = x.copy(), E
    return Solution(best.astype(np.uint8), float(Eb), evals, time.perf_counter() - t0)


def local_search_solve(qubo, evals, seed=0):
    """Steepest 1-flip descent from random restarts (classical optimisation baseline); counts single-flip evaluations."""
    t0 = time.perf_counter(); rng = np.random.default_rng(seed); n = qubo.n; used = 0; best, Eb = None, np.inf
    while used < evals:
        x = rng.integers(0, 2, n).astype(float); fld = qubo.h + qubo.Q @ x; E = qubo.energy(x)[0]; used += 1
        while used < evals:
            dE = (1 - 2 * x) * fld; used += n; i = int(np.argmin(dE))
            if dE[i] >= -1e-12: break
            fld += (1 - 2 * x[i]) * qubo.Q[:, i]; x[i] = 1 - x[i]; E += dE[i]
        if E < Eb: best, Eb = x.copy(), E
    return Solution(best.astype(np.uint8), float(Eb), used, time.perf_counter() - t0)


def pauli_terms(qubo):
    c, hz, J = qubo.to_ising()
    return dict(one_body=int((np.abs(hz) > 1e-12).sum()), two_body=int((np.abs(J) > 1e-12).sum()), max_locality=2)


def gap(E_sol, E_opt):
    return float((E_sol - E_opt) / abs(E_opt)) if abs(E_opt) > 1e-12 else float(E_sol - E_opt)
