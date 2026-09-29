# SOF-QAOA v2 — Constrained-QUBO operator-field optimisation (QAOA) for spatially adaptive RGB image encryption

Research prototype. **No quantum speedup or quantum advantage is claimed** (QAOA is run on a classical statevector
simulator and is slower than exact enumeration at every size tested; see `REPORT.md`).

```
RGB image ──► block features ──► constrained QUBO  J(x)  (3 free bits / block, 4 range-normalised terms)
          ──► QAOA on 12-qubit window sub-QUBOs (Gauss-Seidel decomposition, 2 sweeps)  ──► operator field Ω (4 bits / block)
          ──► R rounds of { block-local operators Ω_ij ─► keyed S-box ─► global fwd+bwd chain } ──► cipher image
```

## Run
```bash
pip install -r requirements.txt
python main.py --image data/sample.png --key "research-key"          # encrypt+decrypt+metrics+plots (writes outputs/)
python main.py --image data/sample.png --key "research-key" --selector sa_window     # any selector
python run_experiment.py                      # everything (≈30–40 min on 1 CPU) -> outputs/tables/ALL_TABLES.md
python run_experiment.py --suite ablation     # baselines | ablation | scaling | shots | benchmark | tables
python run_experiment.py --quick              # smoke run
python -m pytest tests -q                     # 66 tests
```
`main.py` writes `outputs/encrypted|decrypted/*.png`, `outputs/metrics/metrics.json|csv`, `qaoa_stats.json` (qubits, p, γ, β per window,
shots, circuit evaluations, J, exact/SA reference), `operator_field.csv`, and plots (`spatial_operator_field.png`, histograms, correlations).
All parameters are in `config.yaml` — no source editing needed.

## Configuration vector per block  z = (zX, zY, zB, zR)
| bit | 0 | 1 |
|---|---|---|
| zX | additive chain along x (`c=p+k+c_prev`, affine over Z256) | S-box-feedback chain (`c=S[p+k+c_prev]`, nonlinear) |
| zY | same, along y | same |
| zB | forward chains only | + reverse-direction chain (full row/column coverage) |
| zR | – | triangular RGB coupling (**frozen to 0 by default**: measured redundant, `rgb_coupling: false`) |

Operation cost is linear in the bits (`operators.compute_cost`), so the budget constraint is an exact quadratic. The permutation operator was tested and **removed**
(it lowered the measured differential unpredictability of the row/column and helped nothing else; see REPORT §5).

## Objective (all terms ≤ quadratic ⇒ exact QUBO ⇒ Ising ⇒ QAOA)
`J = Σ_k w_k·(T_k−min T_k)/(max T_k−min T_k)`, `w = ¼` each, on the simplex; sensitivity analysis over Dirichlet weights in `experiments/benchmark.py`.
T1 demand–supply deficit (measured differential unpredictability of each configuration vs. block correlation/risk demand) · T2 boundary continuity ·
T3 spatial diversity (anti-ferromagnetic; a design assumption, not a proven security property) · T4 global compute budget (dense all-to-all quadratic).
**NPCR/UACI are never objectives.**

## Header (decoder side information) — exactly what is stored
```
byte 0      version (=2)
bytes 1-4   H, W (public image dimensions, big-endian uint16 each)
byte 5      field source  (0 = transmitted field, 2 = keyed pseudo-random field, no field bytes)
bytes 6-21  nonce = HMAC-SHA512(key, plaintext)[:16]        (SIV-style, deterministic)
next ceil(blocks/2) bytes   operator field, 4 bits/block, XOR-masked with a keystream from (key, nonce)
last 8 bytes  HMAC tag over the header (authenticates nonce + field; not the ciphertext)
```
128×128 image, 16×16 blocks: 6+16+32+8 = **62 bytes** (0.126 % of the 49,152-byte image); keyed-random selector: 30 bytes.
The field cannot be derived from key + dimensions alone because it depends on the plaintext (circular: decoding needs Ω, Ω needs the plaintext).
It is therefore transmitted, masked. See REPORT §6 for the leakage analysis.

## Layout
`src/` operators · feature_extraction · spatial_operator_field (QUBO, decomposition, selectors) · quantum_optimizer (Qubo, QAOA, SA/LS/random/exact) ·
substitution (S-box + LAT/DDT/SAC) · encryption · decryption · metrics · cryptanalysis · attacks · chaotic_maps · key_derivation · preprocessing · utils  
`experiments/` baselines · qaoa_scaling · shot_study · ablation · benchmark · tables  
`tests/` operators, reversibility (all selectors/ablations), QUBO vs. direct definition, QAOA, gate-level circuit equivalence, metrics, cryptanalysis
