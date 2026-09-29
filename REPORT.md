# SOF-QAOA v2 — Report (all numbers from actual runs; CSVs in `outputs/metrics`, paper tables in `outputs/tables/ALL_TABLES.md`)

**Claim.** A quantum-classical framework in which QAOA solves window-decomposed sub-problems of a constrained QUBO for operator-field
assignment in RGB image encryption. **No quantum speedup or advantage is claimed or observed.** Everything is a classical statevector
simulation; QAOA is slower than exact enumeration at every size tested and no better than simulated annealing at equal evaluations.

## 1. What was wrong in v1
1. The optimisation had 4 modes × 4 blocks = 256 states (8 qubits); exact enumeration ≈ 1 ms.
2. The "residual correlation" surrogate could not separate modes: after one keyed additive pass the block x-correlation is already 0.002 (plain: 1.0).
3. Risk weights (0.35/0.25/0.2/0.2) were hand-set. 4. E4 and E8 were the same configuration by construction. 5. E5 (RGB coupling) changed nothing, unexamined.
6. E7 (identity S-box) looked fine on entropy/NPCR while the cipher was affine; the check existed only as an ad-hoc probe.

## 2. Formulation
Per block z=(zX,zY,zB,zR): zX/zY select additive vs S-box-feedback (nonlinear) chain along x/y; zB adds reverse-direction chains; zR = triangular RGB coupling
(frozen to 0 by default, §5). Cost = operation count (linear in bits). `J = Σ w_k (T_k−min)/(max−min)`, w = ¼:
T1 demand–supply deficit (demand = block risk × |correlation|; supply = **measured** differential unpredictability of each configuration, quadratic fit),
T2 boundary continuity (bilinear), T3 spatial diversity (anti-ferromagnetic), T4 compute budget (dense quadratic, target = min + 0.6(max−min)).
All terms ≤ quadratic ⇒ exact QUBO ⇒ Ising (`H_C = c + Σ h_i Z_i + Σ J_ij Z_i Z_j`); verified against a direct term-by-term evaluator (`test_spatial_field.py`).
NPCR/UACI are not objectives. 128×128 image: 64 blocks, 192 free variables. Large problems: Gauss-Seidel windows of 2×2 blocks = **12 qubits**, 2 sweeps.
Calibration table (measured, `operators.calibrate`): additive chains give unpredictability 0 (fully predictable differences); one nonlinear chain (X **or** Y) gives Dx≈0.51, Dy≈0.72;
with bidirectional ≈0.96–0.98; coupling adds Dc 0.31 (0.73 with bidirectional). Quadratic-fit R²: Dx 0.978, Dy 0.994, Dc 1.0 (max error ≈0.056 — a surrogate approximation).

## 3. QAOA
`|ψ(γ,β)⟩ = Π_l e^{−iβ_l ΣX} e^{−iγ_l H_C}|+⟩^n`, p=2, COBYLA on the exact expectation (40 iterations = 41 circuit evaluations per window), analytic energy scale,
then 128 shots read out; the best sampled bitstring by classical QUBO energy is kept. Gate-level (H, RZ, IsingZZ, RX) PennyLane circuit equals the diagonal simulation (test).
128² image: 12 qubits/window, 32 windows, 1312 circuit evals, 2.8 s. γ, β, p_opt, shots, J per window are saved in `outputs/metrics/qaoa_stats.json`.

## 4. Solver benchmark (real image window QUBOs; equal budget = circuit evals + shots = 169 bitstring evaluations)
| qubits | inst. | QAOA gap / hit / AR | SA gap / hit | LS gap / hit | Random gap / hit | QAOA time | Exact time |
|---|---|---|---|---|---|---|---|
| 6 | 12 | 0 / 100% / 1.0 | 0 / 100% | 0 / 100% | 0.0011 / 92% | 0.048 s | 6e-5 s |
| 9 | 4 | 0 / 100% / 1.0 | 0 / 100% | 0 / 100% | 0.0077 / 0% | 0.060 s | 1e-4 s |
| 12 | 12 | 0.0012 / 67% / 0.996 | 0.0004 / 75% | 0.0011 / 67% | 0.0107 / 17% | 0.084 s | 4e-4 s |
| 15 | 4 | 0.0040 / 50% / 0.989 | 3e-5 / 75% | 0.0025 / 75% | 0.049 / 0% | 0.335 s | 4e-3 s |
| 18 | 3 | 0.0030 / 0% / 0.994 | 0.0003 / 33% | 6e-5 / 67% | 0.071 / 0% | 4.0 s | 0.095 s |
| 21 | 1 | 0.0013 / miss | 0.0013 | – | – | 64.9 s | 0.36 s |
Small instance counts at 15/18/21 qubits: treat as indicative. QAOA ≫ random, ≈ or worse than SA/LS, and 10²–10³× slower than exact enumeration (which is still feasible up to 21 qubits).
**Whole-image problems** (exact impossible: 2^192, 2^768): best-known = lowest J found by any solver (not a proven optimum).
128²: sa_window 0.1117 (best), sa_full 0.1179, exact_window 0.1203, ls_window 0.122, **QAOA 0.1251**, random_window 0.153, random 0.316, fixed 0.334, greedy 0.396.
256²: sa_window 0.1175 (best), **QAOA 0.1199** (gap 2.1 %, second best), exact_window 0.1217, sa_full 0.1234, ls_window 0.1248, ls_full 0.275, random 0.288, fixed 0.334. QAOA: 5248 circuit evals, 11.4 s.
Window decomposition itself costs quality (whole-image SA beat exact_window on 128²).

## 5. Shot study (12-qubit instances, 8 windows × 20 reps; QAOA parameters trained once, 41 evals *not* counted in shots)
| shots | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| QAOA gap / hit | .0070/23% | .0037/34% | .0021/50% | .0012/60% | .0006/77% | .0004/83% | .0001/93% |
| SA gap / hit | .0330/4% | .0144/23% | .0041/52% | .0012/61% | .0008/74% | .0002/87% | .0001/93% |
| LS gap / hit | .0445/3% | .0287/8% | .0075/41% | .0014/71% | .0003/91% | 0/99% | 0/100% |
| Random gap / hit | .0414/1% | .0336/1% | .0258/1% | .0194/3% | .0144/5% | .0103/9% | .0058/22% |
QAOA read-out is better than SA/LS only at very few shots (≤64) and ties at 128; local search wins from 256 shots. Plot: `outputs/plots/shot_study_*q.png`.

## 6. Operator design findings (negative results are results)
* **Permutation removed.** Measured with the same calibration code before removal: it lowered row/column unpredictability (Dx 0.52→0.37) with no cross-channel or other benefit.
* **X/Y asymmetry is redundant here.** XnYa ≈ XaYn (Dx 0.512/0.512, Dy 0.72/0.72); making both nonlinear adds nothing. Ablation: E6 (X=Y tied) raises J 0.126→0.160 in the objective only; E7/E8 (one direction only) halve compute (ops 140 vs 277) with NPCR/UACI/entropy unchanged. So X/Y asymmetry is not supported by security metrics; its only measurable effect is on the surrogate objective.
* **RGB coupling redundant in the full system → removed from the default.** One R sample flipped changes 99.66/99.59/99.62 % of R/G/B ciphertext samples without coupling and 99.58/99.57/99.58 % with it (global interleaved chain already mixes channels). Coupling matters only without the global chain (B: 0 → 0.249). E5 (+coupling) raises ops 277→317 with no metric gain. Hypothesis, untested: coupling would become necessary if the global chain were channel-planar.
* **Field choice barely affects final ciphertext metrics.** Entropy, chi², correlations, NPCR, UACI are indistinguishable across E1–E8 (e.g. NPCR 99.60–99.63 %); the global S-box+chain dominates. The field improves the *surrogate* J (fixed 0.334, random 0.280, QAOA 0.126), not measured final security.
* **Weight sensitivity is high.** 24 Dirichlet weight draws: on average 62 % of blocks change field (min 37 %, max 92 %); regret under the default J: mean 0.030 (max 0.11). The field should be presented as one solution of a trade-off, not as a unique optimum.

## 7. Security evaluation (4 images 128², QAOA selector, key "research-key")
Entropy 7.9881–7.9892; chi² 245–270 (pass < 293.25); correlations |h,v,d| ≤ 0.028; NPCR full system 99.59–99.61 %, UACI 33.27–33.42 %; key-change NPCR 99.59–99.64 %, UACI 33.28–33.67 %;
wrong-key exact pixels 0.36–0.43 %; pure cipher (field frozen) NPCR 99.60–99.68 %. Ciphertext channel MI (16-bin, bias-corrected) ≈ 0.000–0.001 bit vs 1.4–2.0 bit in the plaintext.
(An earlier 256-bin MI estimator was invalid at this image size — it produced values of +2.2 and −0.65 bit — and was replaced; MI columns in `ablation.csv`/`ALL_TABLES.md` produced before the fix should be ignored.)
Key space: 512-bit master; user-key entropy bound 8·len(key) bits (96 for the demo key). Throughput 0.14 MB/s (fixed) — 0.015 MB/s (QAOA selector, dominated by simulation).
Robustness: the cipher is a diffusion cipher, so it is fragile by design: JPEG q90/Gaussian noise → ≈0.4 % exact pixels; 25 % crop → 75 % exact; salt&pepper 1 % → 82.6 % exact.

## 8. Cryptanalysis (fixed field/nonce oracle; identity S-box = E9)
| test | proposed | identity S-box |
|---|---|---|
| additive (Z256) violation | 0.9956 | **0.0** |
| XOR-linearity violation | 0.9958 | 0.9926 (**misses the weakness**: algebra must match the cipher's group law) |
| chosen-plaintext affine reconstruction (769 queries) | 0.46 % exact | **100 %** |
| differential: mean max-frequency (floor ≈ 1/256; 300 samples) | 0.017 | **1.0** |
| bit avalanche (ideal 0.5) | 0.5004 ± 0.003 | 0.203 ± 0.146 |
| S-box DU / nonlinearity / max linear bias / SAC dev | 10 / 92 / 0.141 / 0.156 | 256 / 0 / 0.5 / 0.5 |
E9 in the ablation: entropy 7.988, chi² 266, NPCR 98.44 %, UACI 33.43 % — a superficially acceptable profile for a cipher that is completely affine.
So high entropy + high NPCR/UACI + low correlation do **not** imply cryptographic nonlinearity. The S-box is bijective/invertible/keyed/reproducible (tests) but DU 10, NL 92 is random-like, not AES-grade (DU 4, NL 112, verified by the same code in a test). No strength claim is made for it.
Probes measure structure; they are not security proofs. Chaotic-map components were not analysed.

## 9. Header, field reconstruction and leakage
Header = version(1) | H,W(4) | field-source(1) | nonce(16) | field(⌈blocks/2⌉) | HMAC tag(8): **62 B at 128²** (0.126 % of the image), 158 B at 256² (measured); 30 B for the keyed-random selector.
Nonce = HMAC-SHA512(key, plaintext)[:16] (deterministic; equal plaintexts ⇒ equal ciphertexts, which is inherent to the required determinism). The field is 4 bits/block, XOR-masked by a keystream from (key, nonce); the tag authenticates the header only.
**A zero-side-information design is impossible with plaintext-derived fields**: decoding needs Ω, Ω is a function of the plaintext. The field is therefore transmitted masked (or replaced by a keyed pseudo-random field: `random` selector, 0 field bytes, no adaptivity).
Plaintext features themselves are never transmitted. Leakage test (256² image, 256 blocks, Mann–Whitney on per-block ciphertext entropy/mean/chi² vs field bits): min p = 0.17 (zB), 0.46 (zX), 0.59 (zY) vs Bonferroni threshold 0.0056 → no leakage detected (absence of evidence, not a proof).
Decoder reconstruction is tested to reproduce the encoder field exactly.

## 10. Complexity (1 CPU)
128²: fixed 0.37 s, SA-window 0.41 s, QAOA 3.2 s (select 2.8 s, 1312 circuits); decryption 0.34–0.36 s; peak 11.7 MB.
256²: 1.8 / 2.2 / 13.0 s; QAOA 5248 circuits; QUBO build ≈1 s; decryption 0.54–0.61 s; peak 110 MB. QAOA state simulation costs ≈ 0.02 s (16q), 0.12 s (18q), 0.77 s (20q), 5.8 s (22q) per p=2 circuit.

## 11. Limitations and novelty
Simulator only; surrogate quality (quadratic fit, hand-set linear cost model, diversity as design assumption); field weakly linked to final metrics; window decomposition ≠ global optimum; weight sensitivity high;
few instances at ≥15 qubits; float determinism only relevant to the encoder (decoder reads the transmitted field); S-box DU 10; robustness to noise poor by design.
Novelty (web-only search in the previous session, not repeated; Scopus/WoS/IEEE exact-string search still required): closest work is Sultanow et al. (arXiv 2506.01536, 2-qubit VQC/RL primitive selection, grayscale, no QAOA) and AESeg (ETASR 2026, classical entropy-driven regional strength allocation). No exact prior art found, but "first" must not be claimed.
