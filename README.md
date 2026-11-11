# ATOMIQ Hackathon 2026 — Self-contained hybrid quantum–classical submission

**Canonical pipeline:** Stage 1 (100 subscription-probability predictions) → Stage 2 (choose 6 of the ranked top 10) → Stage 3 (estimate probability of at least 3 subscriptions).  
**Backend evidence:** Qiskit AerSimulator, **not physical hardware**. **Random seed:** 42 for recorded runs.  
**Start:** `python run_all.py verify` (Python standard library only). See `README_START_HERE.md` for setup.

## 1. Motivation and design philosophy

The task is not three unrelated quantum demonstrations. It is a sequential banking-campaign decision problem: use a small labeled data set to estimate which customers will subscribe, allocate a constrained outreach budget to maximize expected net value, then quantify how likely the chosen campaign is to meet a minimum subscription target. **Probabilities from Stage 1 are carried without re-fitting or silently changing them into Stages 2 and 3.**

The team developed a *routed hybrid model* instead of relying on a single quantum classifier. With only 300 labeled cases, three challenges matter: (i) variable types and nonlinear interactions; (ii) the `poutcome = unknown` subpopulation has less explicit prior-campaign evidence; (iii) top-10 ranking and probability calibration are not the same objective. We therefore combined different predictive feature/model families and used a modest history correction only where prior-campaign outcomes were observed. This is an empirical design choice, not a proven superiority or quantum-speedup result.

### Stage 1 — model architecture and rationale

**Input/output.** Fit on `data/Stage1_Train.tsv` (300 labeled customers); predict `data/Stage1_Test.tsv` (100 unlabeled). The authoritative output is `stage1_result.csv`, sorted by probability descending, ties broken by customer ID ascending, ranks 1–100, exactly ten `top10_flag=1`.

The named submitted predictor is **N17 + Friend Hybrid**, with fixed history coefficient **γ = 0.2**. The *implementation* is in `code/atomiq_N17_same_nested_three_models.py` (`fit_all`, `predict`), invoked by `code/atomiq_N17_final_test_top10.py`.

Four probability-producing ingredients:

1. **F2**, a quantum-feature-fusion learner built from angle-encoded quantum correlations and conventional supervised fitting. The code computes quantum-derived features and estimates probabilities from training-only labels. See `code/atomiq_unknown_14q_N7_N9.py` and dependencies.
2. **N8 Ring40**, a ring-type quantum feature specialist for customers with **unknown previous campaign outcome**. The construction uses angle-encoded features and selects 40 correlation terms using **training rows only**; see `code/atomiq_N15_strict_nested_history.py` (`fit_sources`).
3. **Friend LR Compact**, a regularized logistic-regression model with engineered numerical/categorical features, scaling and selected features. It provides a relatively stable classical anchor in small-data settings; original code at `code/friend_atomiq/src/qcampaign/`.
4. **Friend Hybrid**, the teammate's **arithmetic mean** of three independently fitted models: LR Compact, a quantum-kernel logistic regressor, and a variational quantum classifier (VQC). These are genuine separately trained model components, not a renamed single quantum model. See `code/atomiq_N17_same_nested_three_models.py` (`fit_all`).

The routing and weights are **explicit**, allowing auditing rather than a black-box claim:

- `poutcome = unknown`: let \(p_{N16} = 0.5p_{F2}+0.5p_{Ring40}\). Then \(p_{N17} = 0.85p_{N16}+0.15p_{FriendHybrid}\).
- Other history strata: \(p_{base}=0.7p_{F2}+0.3p_{FriendLR}\); then \(p_{final}=\sigma(\operatorname{logit}(p_{base})+0.2\Delta_{history})\). \(\Delta_{history}\) is a shrunk log-odds adjustment based **only on training labels**, with a pseudo-count of 40 toward the global conversion rate; see `history_offsets` in `code/atomiq_N14_history_shared_F2_friend.py`.

**Why these choices?** Routing gives the limited-history group access to ring-based quantum features, while reserving an interpretable, shrinkage-regularized history offset for groups with observed prior outcomes. A 15% hybrid contribution diversifies the unknown-group probabilities without replacing the F2/ring backbone. The exact mixture constants and γ resulted from iterative *exploratory model development*, not an independently validated global optimum.

**Evaluation safeguards and limits.** Model assessment includes stratified CV and a nested-CV implementation that refits sources **within each split** to avoid validation-fold feature/label leakage. Metrics include ROC-AUC, PR-AUC, Brier score, log loss and precision@10. Relevant files: `reports/STAGE1_BASELINES.md`, `results/stage1_cv/`, `stage1_reverification/`. The 300 labeled examples were used repeatedly during development, so selected CV scores remain exploratory and may be optimistic. **The 100 test labels were neither present nor used.** Exact bit-for-bit Stage 1 re-training remains unverified in the assembly environment; `python run_all.py stage1-refit` attempts it separately and does **not** overwrite canonical predictions.

### Stage 2 — economic objective and why Dicke + XY

Let \(x_i\in\{0,1\}\) indicate a contact, \(p_i\) be Stage 1 probability, \(D_i\) the expected deposit contribution on subscription, \(c_i\) the outreach cost and \(h_i\) capacity units. We maximize

\[
Z(x)=\sum_{i=0}^9(p_iD_i-c_i)x_i
\]
subject to \(\sum_i x_i=6\), \(\sum_i c_ix_i\le25\), \(\sum_i h_ix_i\le22\).

**Why the Dicke state?** Preparing \(|D_{10}^6\rangle\), the equal-amplitude superposition of the \(\binom{10}{6}=210\) six-person selections, puts *all* initial weight on exactly-six choices. **Why the XY mixer?** Each nearest-neighbor \(XX+YY\) term conserves total excitation/Hamming weight; consequently every generated bitstring still selects exactly six people. It avoids a large penalty for cardinality, unlike a generic X mixer. The phase separator uses \(w_i=p_iD_i-c_i\) and \(Z\) rotations. Implementation details and p=2 angles are in `stage2_result.json`, `code/run_aer_evidence.py`, and `reports/AER_VERIFIED_TECHNICAL_REPORT.md`.

**Important instance-specific simplification:** all **210** six-person choices also satisfy both budget and capacity bounds (not true in general). Thus the budget/capacity constraints are non-binding **in this data set**; exact optimization can also be done by sorting net values, and there is **no claimed quantum computational advantage**.

Canonical **p=2 QAOA**: 10 qubits, 8,192 ideal Aer samples; optimum bitstring (printed q9…q0) **`0101001111`**, observed **236 times**; **8,192 feasible shots**. The best sampled objective equals the exact classical optimum **43.183204088500**, with cost **14/25** and capacity **11/22**. **Gap zero means best sampled solution only**, not mean performance or guaranteed convergence. Statevector mean objective is **33.720061823**. The most frequent bitstring is different from the optimum.

**Added controlled-depth study, p=3–8:** 3 circuit families × 6 depths = 18 Qiskit Aer cases, each 8,192 shots, with saved parameters, counts and independent NumPy/statevector parity checks (`experiments/p3_p8/results/`). Compared: **Dicke + XY**, **Dicke + X**, **Plus + X**. Mixing type and initial state both vary between certain families, so attribution requires care. For Dicke + XY, highest **expected raw value** in the recorded sweep occurs at p=8 (**37.994954**), whereas most **optimal-bitstring Aer hits** occur at p=6 (**1923/8192**). The study demonstrates an engineering tradeoff, not monotonic improvement in success probability or a quantum advantage. Other families' parameters are those found by the recorded finite classical search, not proven global optima.

**Circuit-cost disclosure:** p=2 has 20 RZ and 18 RXX + 18 RYY logical rotations (36 two-qubit rotations) plus one **opaque Dicke `initialize`**. The reported depth 12 **does not include native-gate decomposition** of the initial state. A separate generic-basis Dicke decomposition generated depth 4,846 and 1,013 CX gates; device topology/noise were not used. These numbers are not physically optimized resource estimates.

### Stage 3 — Ry probability encoding, Monte Carlo and MLAE

The *same six IDs*, in q0…q5 order, are **C0081, C0079, C0031, C0327, C0374, C0222**. For customer i, `Ry(2*asin(sqrt(p_i)))` prepares \(\sqrt{1-p_i}|0
angle+\sqrt{p_i}|1
angle\). Measuring six independent qubits and counting outcomes with at least three 1s estimates \(P(K\ge3)\), where \(K=\sum_iX_i\) and \(X_i\sim\mathrm{Bernoulli}(p_i)\).

The exact **Poisson–binomial** reference is **0.677951471871**. The recorded **ideal Qiskit Aer** sample is **5515/8192 = 0.673217773**, absolute error **0.004733698**. A separately reproducible **classical Monte Carlo** calculation is **5553/8192 = 0.677856445**, absolute error **0.000095027**. Both methods simulate the *same independent-Bernoulli model* but use different random-number generators and should not have identical shot counts. Source and script: `evidence/stage3_monte_carlo.json` and `code/stage3_monte_carlo_reference.py`. The illustrative noisy Aer run gives **5392/8192 = 0.658203125**, using a **synthetic, not device-calibrated noise model**.

The *supplementary* Maximum-Likelihood Amplitude Estimation (MLAE) experiment uses Grover powers k=0,1,2,4,8 with a likelihood fit (`experiments/` evidence if present). **Number of shots is not an equal quantum-cost comparison** when Grover depth differs, and the high-level `diagonal` operation must be decomposed before discussing physical gates. With a simple independent-Bernoulli model and known p_i, the exact classical dynamic-programming reference is trivially available; no quantum speedup is established.

### Scientific assumptions

The six events are modeled as independent **conditional on supplied probabilities**. Neither the 95% shot intervals nor the Aer noisy example quantify uncertainty of learned Stage 1 probabilities or correlations between customer outcomes. No actual quantum processor or physical circuit calibration was used. Classical enumeration (210 cases) and dynamic programming (six Bernoulli variables) are very efficient in this instance.

## 2. Results and source-of-truth hierarchy

| Artifact | Role |
|---|---|
| `stage1_result.csv` | **Authoritative** 100 probabilities/ranks/Top10 flags |
| `stage2_candidates.csv`, `stage2_result.csv/json` | **Authoritative** six-person decision, expected economics, p=2 Aer result |
| `stage3_result.csv/json` | **Authoritative** at-least-three event and Aer result |
| `evidence/qiskit_aer/` | Actual **Aer simulator** count files and recorded environment |
| `evidence/stage3_monte_carlo.json` | Reproducible independent classical Monte Carlo |
| `experiments/p3_p8/results/` | Supplemental 18-condition model/mixer/depth comparison |
| `presentation/` | Editable English slides using team-supplied template |
| `code/friend_atomiq/` | Original teammate project snapshot; its own outputs differ from the canonical submission |
| `supplementary/numpy_baseline/`, `reports/LEGACY_*.md` | Historical experimental artifacts only, **not canonical outputs** |

**Rules:** never substitute a teammate project's independent output for the official root files; never substitute old NumPy sampling counts for recorded Qiskit Aer counts; do not round Stage 1 probabilities when constructing Stage 2 expected net values.

## 3. Reproduce from a clean extraction

Supported environment is **Python 3.11+ with dependencies installed**, not a frozen Python executable. Qiskit/Aer recorded versions: **2.6.2/0.18.2**. Recommended macOS workflow:

```bash
conda activate qiskit-hackathon
python -m pip install -r requirements.txt
python run_all.py verify
python run_all.py stage1-baseline
python run_all.py aer-stages23
python run_all.py aer-p3-p8-replay
```

The optional heavy Stage 1 refit and nested-CV commands are `python run_all.py stage1-refit` and `python run_all.py stage1-nested`; they can be CPU-intensive and have **not** been independently confirmed to reproduce the original bit-for-bit predictions. The `run_all.py verify` command validates the **saved outputs**, not the Stage 1 fitting process. Fresh Aer sampling may produce different counts despite identical state probabilities.

Other instructions: `README_START_HERE.md`; dependencies `requirements.txt`, `code/requirements_aer.txt`, `code/requirements_friend_original.txt` (original teammate's requirements may not all be jointly pin-compatible). Exact source snapshot/provenance at `code/friend_SOURCE_PROVENANCE.md`.

## 4. Submission and limitations

- The archive is self-contained in **project/code/data/evidence** (no extra GitHub clone or companion ZIP), but requires standard Python dependency installation.
- SHA-256 per-file manifest `MANIFEST_SHA256.txt` was rebuilt after final changes. Verify with `python verify_manifest.py`.
- All top-level Stage 1/2/3 artifacts are internally cross-validated. The original author's snapshot does **not** imply their independent final choices are ours.
- **No claim of physical quantum hardware execution, quantum speedup, globally optimal QAOA parameters, or hidden-test predictive accuracy.**
