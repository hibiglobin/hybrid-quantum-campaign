# Consolidated technical report — FINAL verified artifact summary

**Source of truth:** canonical root `stage1_result.csv`, `stage2_result.json`, `stage3_result.json`, and ideal Qiskit Aer evidence files. Supersedes historical NumPy-only reports (renamed `LEGACY_*.md`).

## Stage 1: N17 + Friend Hybrid

The package **does contain all 100 test customer predictions**, sorted ranks 1–100 and exactly 10 Top10 flags. The 300 labeled training inputs and original teammate model implementations are bundled. See `README.md` for the exact routed architecture and training-only history adjustments, plus `reports/STAGE1_BASELINES.md` for CV. This assembly checks output consistency, **not** an independently verified bit-identical refit.

## Stage 2: selection with cardinality conservation

Ten top-ranked candidates; select exactly six to maximize net value `Σ_i(p_i D_i − c_i)x_i`, budget <=25, capacity <=22. All 210 6-of-10 choices happen to satisfy budget/capacity. Exact optimum `0101001111` (q9…q0), IDs C0081, C0079, C0031, C0327, C0374, C0222, expected net value **43.183204088500**, cost **14/25**, capacity **11/22**.

Dicke+XY p=2, 8,192 ideal Aer shots: optimum observed **236** times, **8192** feasible shots, best-sampled gap **0**; ideal expectation **33.720061823** (below exact optimum). The opaque Dicke initialization is not included in a hardware-native depth calculation. Supplementary p=3–8 comparison: 18 ideal Aer experiments; best recorded expectation Dicke+XY p=8, 37.994954; most optimum hits p=6, 1923/8192. Not a demonstrated quantum advantage.

## Stage 3: independent Bernoulli outcome probability

Same six probabilities from Stage 1 and Stage 2: exact Poisson-binomial **0.677951471871**.

| Method | Shots | Successful outcomes | Estimate | Absolute error |
|---|---:|---:|---:|---:|
| Ideal Qiskit Aer | 8192 | 5515 | 0.673217773 | 0.004733698 |
| Classical NumPy MC (PCG64, seed 42) | 8192 | 5553 | 0.677856445 | 0.000095027 |
| Synthetic-noise Qiskit Aer | 8192 | 5392 | 0.658203125 | 0.019748347 |

The reference MC is reproducible using `python code/stage3_monte_carlo_reference.py`; original simulator evidence retained in `evidence/qiskit_aer/`. Nonzero Monte Carlo/Aer shot error is statistically expected, not a mismatch in the mathematical probability model.

## Methodological disclaimers

No hardware run, no quantum-speedup proof. Stage 1 CV is exploratory due to repeated model selection; no unseen test labels. Approximate shot intervals account for sampling only, not Stage 1 model uncertainty or correlated customers. Original collaborator outputs in their own folder are distinct from official root files.
