# ATOMIQ Hackathon 2026 — Reproducible Hybrid Quantum–Classical Campaign Pipeline

**Submission type:** three-stage predictive modelling, constrained selection, and quantum-circuit campaign simulation  
**Canonical workflow:** Stage 1 predicts 100 subscription probabilities → Stage 2 chooses exactly 6 of the top 10 → Stage 3 estimates the probability of at least 3 subscriptions.  
**Quantum execution:** Qiskit **AerSimulator** only; **no physical quantum hardware** was used.  
**Recorded random seed:** 42 where stated in the run records.  
**Quick integrity check:** `python run_all.py verify`  
**Getting started:** [`README_START_HERE.md`](README_START_HERE.md)

> **Scope of the evidence.** This report describes the submitted outputs and the recorded simulator experiments. It distinguishes exact classical reference calculations, ideal quantum-circuit sampling, synthetic-noise simulation, and unverified reproducibility claims. In particular, no quantum speedup, hardware validation, independent hidden-test accuracy, or bit-for-bit reconstruction of the original Stage 1 fit is claimed.

## Executive summary

The business problem is to select a small set of bank customers for a term-deposit outreach campaign. The three stages use one consistent set of model-derived probabilities rather than constructing disconnected demonstrations:

1. **Predict:** fit a routed hybrid classifier to 300 labelled training records and assign subscription probabilities to 100 unlabelled customers.
2. **Decide:** select 6 customers from the 10 highest-ranked candidates to maximise predicted net value, subject to contact-count, budget and capacity limits.
3. **Evaluate:** prepare a 6-qubit circuit with the corresponding individual probabilities and estimate the chance that at least 3 customers subscribe.

For the recorded instance, the best sampled Stage 2 decision matches the exact classical optimum (**43.183204088500** model-defined value units). Its budget usage is **14/25**, and its capacity usage is **11/22**. The exact conditional probability of at least three subscriptions for those six candidates is **0.677951471871** (approximately **67.80%**); ideal Aer sampling yielded **0.673217773**. These are **model-based estimates**, not observed campaign returns.

## 1. End-to-end methodology and data governance

| Stage | Input | Method | Canonical output | Validation |
|---|---|---|---|---|
| 1: prediction | `data/Stage1_Train.tsv` (300 labelled rows), `data/Stage1_Test.tsv` (100 unlabelled rows) | Routed quantum-feature/classical hybrid predictor | `stage1_result.csv` | Exploratory stratified and nested CV; rank/ID checks |
| 2: selection | Stage 1 top 10, economic coefficients, budget and capacity | Dicke-state QAOA with excitation-preserving XY mixer; exhaustive classical reference | `stage2_candidates.csv`, `stage2_result.csv`, `stage2_result.json` | Feasibility checks and exact comparison of 210 six-person choices |
| 3: outcomes | The same six selected IDs and their **unrounded** Stage 1 probabilities | Independent `Ry` state preparation, measurement and event counting | `stage3_result.csv`, `stage3_result.json` | Exact Poisson–binomial evaluation; independent Monte Carlo |

**Data-leakage policy.** Preprocessing and parameter fitting must use training data only. Unlabelled Stage 1 test rows are used for prediction, not for feature selection, threshold tuning, calibration or hyperparameter selection. The 100 test labels were not available to the team. The Stage 1 probabilities and the deterministic top-10 ranking are carried unchanged into Stages 2 and 3; no retrospective substitution of another model's output is permitted.

**Decision provenance.** The root-level CSV and JSON files are the authoritative submission. Historical experiments and teammate snapshots are retained for transparency but are not interchangeable with the final outputs; see [Section 8](#8-artifacts-and-source-of-truth).

## 2. Stage 1 — Subscription probability modelling

### 2.1 Problem, model and motivation

Stage 1 assigns a probability \(p_i=P(Y_i=1\mid\mathrm{features}_i)\) to each of 100 customers. The submitted predictor is **N17 + Friend Hybrid**, with history-adjustment coefficient **\(\gamma=0.2\)**. Its implementation is `code/atomiq_N17_same_nested_three_models.py` (`fit_all`, `predict`), invoked by `code/atomiq_N17_final_test_top10.py`.

Only 300 labelled cases are provided. The architecture addresses mixed feature types, potentially nonlinear responses and a subpopulation with `poutcome = unknown` (limited previous-campaign evidence). It mixes four probability-producing ingredients:

- **F2 quantum-feature fusion:** angle-encoded quantum correlation features followed by supervised probability fitting (`code/atomiq_unknown_14q_N7_N9.py` and dependencies).
- **N8 Ring40:** a ring-based quantum feature representation for customers with unknown historical outcome. Forty correlation terms are selected using training rows only (`code/atomiq_N15_strict_nested_history.py`, `fit_sources`).
- **Friend LR Compact:** a regularised logistic-regression baseline with numerical and categorical feature engineering, scaling and feature selection (`code/friend_atomiq/src/qcampaign/`).
- **Friend Hybrid:** an arithmetic mean of three separately fitted models: LR Compact, quantum-kernel logistic regression and a variational quantum classifier (VQC). This is not a single model relabelled as three components (`code/atomiq_N17_same_nested_three_models.py`).

The routed combination is auditable:

**For `poutcome = unknown`:**

\[
p_{N16}=0.5p_{F2}+0.5p_{Ring40},\qquad
p_{N17}=0.85p_{N16}+0.15p_{FriendHybrid}.
\]

**For other historical-outcome groups:**

\[
p_{base}=0.7p_{F2}+0.3p_{FriendLR},\qquad
p_{final}=\sigma\!\left(\operatorname{logit}(p_{base})+0.2\Delta_{history}\right).
\]

Here \(\sigma(z)=1/(1+e^{-z})\). The history offset \(\Delta_{history}\) uses training labels only and is shrunk toward the global conversion rate with a pseudo-count of 40; its implementation is in `code/atomiq_N14_history_shared_F2_friend.py` (`history_offsets`). The routing and weights were chosen during exploratory development; they are **not** established global optima.

### 2.2 Evaluation and limitations

Model checks include stratified cross-validation and a nested-CV implementation which refits components inside each split to reduce validation-fold leakage. Available metrics include ROC-AUC, PR-AUC, Brier score, log loss and precision@10; records are under `reports/STAGE1_BASELINES.md`, `results/stage1_cv/` and `stage1_reverification/`.

Because the same 300 labelled cases were used repeatedly during model exploration, selected validation results may be optimistic. No independent hidden-test performance estimate is available. The root `stage1_result.csv` is sorted by decreasing probability (customer ID ascending as tie-breaker), contains exactly 100 ranked customers and marks exactly 10 as `top10_flag=1`.

**Reproducibility distinction:** `python run_all.py verify` validates the saved outputs; it does not refit N17. An optional Stage 1 refit path exists, but bit-for-bit reproduction of the original N17 predictions was **not independently verified** in the packaging environment.

## 3. Stage 2 — Constrained optimisation

### 3.1 Binary mathematical formulation

Index the ten highest-ranked Stage 1 candidates by \(i=0,\ldots,9\). Define:

- \(x_i\in\{0,1\}\): whether candidate \(i\) is selected;
- \(p_i\): unchanged Stage 1 predicted subscription probability;
- \(D_i\): contribution if that customer subscribes, in the supplied objective units;
- \(c_i\): contact cost;
- \(h_i\): capacity requirement;
- \(w_i=p_iD_i-c_i\): **predicted expected net value** of contacting that customer.

The binary optimisation problem is

\[
\begin{aligned}
\text{maximise}\quad & Z(\mathbf x)=\sum_{i=0}^{9}w_ix_i,\\
\text{subject to}\quad &\sum_{i=0}^{9}x_i=6,\\
&\sum_{i=0}^{9}c_ix_i\le 25,\\
&\sum_{i=0}^{9}h_ix_i\le22,\\
&x_i\in\{0,1\}.
\end{aligned}
\]

The units of \(D_i\), \(c_i\), and \(Z\) are determined by the supplied problem data. We do not assume the reported objective is a monetary amount in pounds or dollars.

### 3.2 Constraint handling: Dicke initial state and XY mixer

The cardinality requirement is encoded **structurally**, not by a large penalty coefficient. The initial state is the 10-qubit, six-excitation Dicke state:

\[
|D_{10}^{6}\rangle=\frac{1}{\sqrt{\binom{10}{6}}}
\sum_{\mathbf x:\,\sum_i x_i=6}|\mathbf x\rangle,
\qquad \binom{10}{6}=210.
\]

The QAOA cost layer applies single-qubit \(Z\) rotations based on the weights \(w_i\), equivalent to implementing a diagonal objective Hamiltonian up to an additive constant and a conventional sign/scaling choice. The nearest-neighbour XY mixer is built from \(XX+YY\) couplings. Because XY exchange preserves the number of \(|1\rangle\) excitations, the optimisation remains in the six-selected subspace for every QAOA layer.

**Budget and capacity:** in this *particular submitted instance*, an exhaustive enumeration confirms that **all 210** six-person combinations satisfy both inequality constraints. Consequently, budget and capacity are **non-binding here**; they do not need extra penalty terms or slack/ancilla registers for this recorded run. This is an important data-specific property, **not** a general solution to arbitrary inequality-constrained QAOA. Were any cardinality-valid bitstrings to violate those limits, the present mixer alone would not enforce the inequalities; a different feasible-subspace construction, penalty/slack encoding, or rigorous feasible-solution filtering would be required.

### 3.3 Canonical p=2 results and bitstring interpretation

| Quantity | Recorded result |
|---|---:|
| Number of candidate/decision qubits | 10 |
| QAOA depth parameter | \(p=2\) |
| Ideal Aer measurement shots | 8,192 |
| Reported optimal bitstring, displayed **q9…q0** | `0101001111` |
| Occurrences of that bitstring | 236 / 8,192 |
| Feasible measured shots | 8,192 / 8,192 |
| Best sampled objective | **43.183204088500** |
| Exact classical optimum | **43.183204088500** |
| Best-sampled-solution optimality gap | **0** |
| Cost usage | **14 / 25** |
| Capacity usage | **11 / 22** |
| Statevector expected objective | **33.720061823** |

For bitstring interpretation, **the reported string is in q9-to-q0 display order**, whereas the candidate-to-qubit mapping must be read in the implementation/result files. Never interpret the leftmost printed bit as `q0`. The best objective was reached by **at least one sampled feasible string**; a zero *best-sample* optimality gap does not imply that the QAOA distribution is concentrated on the optimum, that the modal string is optimal, or that the variational angles are globally optimal. In fact, the most frequent string differs from the optimum.

The classical benchmark enumerates all \(\binom{10}{6}=210\) selections and checks constraints and net value. Because all choices are feasible, sorting the ten weights also provides an exact solution in this instance. Thus the tiny problem is already easy classically: **no computational quantum advantage is claimed**.

### 3.4 Depth and ansatz comparison (supplementary)

The recorded study covers **18 Aer cases**: three circuit families (**Dicke + XY**, **Dicke + X**, **Plus + X**) at depths \(p=3,4,5,6,7,8\), each with **8,192 shots**, saved optimiser parameters and measurement counts, plus independent NumPy/statevector parity checks in `experiments/p3_p8/results/`.

Within the **Dicke + XY** family:

- The highest recorded **expected raw value** is **37.994954 at p=8**.
- The highest recorded number of **optimal-string hits** occurs at **p=6: 1,923 / 8,192**.

Optimisation quality depends on the metric; higher depth does not necessarily produce monotonic gains in sampled optimal-string probability. Some comparisons change **both** the initial state and mixer, so performance differences cannot be uniquely attributed to either factor. The stored finite parameter search is not a proof of global optimisation.

### 3.5 Circuit resources and transpilation caveat

The canonical p=2 logical construction has **20 RZ rotations**, **18 RXX** and **18 RYY rotations** (36 logical two-qubit rotations), plus one `initialize` of the Dicke state. Its reported **depth 12 excludes the native-gate decomposition of that opaque initialisation**. A separately generated **generic-basis Dicke decomposition** produced **depth 4,846 and 1,013 CX gates**. These figures refer to different circuit descriptions and must not be presented as a single hardware-ready transpiled circuit. No physical coupling map, device-calibrated basis/noise model or hardware execution was used.

**Resource-reporting boundary:** the provided README evidence does not establish the full device-basis physical-qubit count, final routed circuit depth, final SWAP count, or Stage 3 transpilation statistics. These should be read directly from a reproducible transpiler output before being quoted; they are **not inferred** from the abstract gate lists here. Further technical records are referenced in `reports/AER_VERIFIED_TECHNICAL_REPORT.md` and `code/run_aer_evidence.py`.

## 4. Stage 3 — Quantum-circuit campaign simulation

### 4.1 State preparation and measurement event

The six Stage 2 customers are, in **q0 → q5** order:

`C0081`, `C0079`, `C0031`, `C0327`, `C0374`, `C0222`.

For customer \(i\), prepare one qubit by

\[
R_y(\theta_i)|0\rangle
=\sqrt{1-p_i}|0\rangle+\sqrt{p_i}|1\rangle,
\qquad \theta_i=2\arcsin\sqrt{p_i}.
\]

The independent-customer assumption gives the product state

\[
|\Psi\rangle=\bigotimes_{i=0}^{5}
\left(\sqrt{1-p_i}|0\rangle_i+\sqrt{p_i}|1\rangle_i\right).
\]

On each shot, six measured bits represent non-subscription (`0`) or subscription (`1`). A shot is a success if its **Hamming weight is at least three**:

\[
K=\sum_{i=0}^{5}X_i,\quad X_i\sim \operatorname{Bernoulli}(p_i),\qquad
P_{\mathrm{success}}=P(K\ge3).
\]

The stored customer order is q0-to-q5; Qiskit's usual displayed count strings place the highest-index classical bit on the left. The success test is invariant under bit reversal, but mapping individual outcomes back to customers is not. The Stage 3 circuit is a **six-qubit `Ry`-encoded Bernoulli sampler**, not by itself evidence of a quantum computational advantage.

### 4.2 Exact benchmark and recorded sampling results

The exact **Poisson–binomial** result is obtained by multiplying the six probability-generating factors

\[
G(z)=\prod_{i=0}^{5}\bigl[(1-p_i)+p_i z\bigr],
\qquad P(K\ge3)=\sum_{k=3}^{6}[z^k]G(z),
\]

or equivalently by a short classical dynamic programme.

| Method | Successful outcomes | Estimated \(P(K\ge3)\) | Absolute error vs exact |
|---|---:|---:|---:|
| **Exact classical Poisson–binomial** | N/A | **0.677951471871** | 0 |
| **Ideal Qiskit Aer** | 5,515 / 8,192 | **0.673217773** | **0.004733698** |
| **Independent classical Monte Carlo** | 5,553 / 8,192 | **0.677856445** | **0.000095027** |
| **Synthetic-noise Aer illustration** | 5,392 / 8,192 | **0.658203125** | **0.019748347** |

The classical Monte Carlo comparator can be reproduced via

```bash
python code/stage3_monte_carlo_reference.py --seed 42 --shots 8192
```

It samples the **same Bernoulli assumptions** with a separate pseudorandom-number generator; the ideal Aer and classical Monte Carlo counts should **not** be expected to match exactly. Their individual deviations from the analytic reference are subject to finite-shot randomness.

### 4.3 Noise, sampling uncertainty and what the experiment proves

For \(N\) independent shots of a success event with probability \(P\), the standard error of a sample fraction is approximately

\[
\operatorname{SE}(\hat P)\approx\sqrt{\frac{P(1-P)}{N}}.
\]

At \(P=0.677951471871\) and \(N=8192\), this is approximately **0.00516** (0.52 percentage points); a normal-approximation **95% sampling interval half-width** is approximately **0.0101** (1.01 percentage points). The observed ideal-Aer difference of **0.00473** is smaller than one such standard error; the recorded quantum sampler is therefore consistent with finite-shot sampling of the stated model. This calculation characterises **shot noise conditional on the supplied probabilities**, not uncertainty in those learned probabilities.

Relative to the ideal Aer sample, the synthetic-noise run's reported success fraction is lower by **0.015014648** (about **1.50 percentage points**). Relative to the exact reference, it is lower by approximately **1.97 percentage points**. These are **descriptive differences between recorded samples**, not an isolated estimate of physical noise bias: different shot fluctuations can contribute, and the noise model is **synthetic, not calibrated to an actual quantum device**. A defensible hardware noise claim would require a specified channel model, controlled repeats and appropriate transpiled-circuit/depth information.

### 4.4 Supplementary maximum-likelihood amplitude estimation

A separate exploratory **MLAE** experiment uses Grover powers \(k=0,1,2,4,8\) and a likelihood fit, with experiment evidence referenced under `experiments/` where present. Counting only measured shots is **not** a fair quantum-resource comparison with the direct sampler because deeper Grover powers use additional oracle/state-preparation operations. High-level `diagonal` instructions must also be decomposed before physical gate or noise claims. Since the six Bernoulli probabilities are supplied numerically, the exact classical answer is readily accessible; **quantum speedup is not established**.

### 4.5 Error propagation from Stage 1 probabilities to Stage 3 outcomes

The Stage 3 probability is a deterministic function of the six **predicted** probabilities,

\[
F(p_0,\ldots,p_5)=P(K\ge3).
\]

For small prediction perturbations \(\delta p_i\), with **the selected six customers held fixed**, first-order sensitivity is

\[
\delta F\approx\sum_{i=0}^{5}\frac{\partial F}{\partial p_i}\,\delta p_i,
\qquad
\boxed{\frac{\partial F}{\partial p_i}
=P\!\left(\sum_{j\ne i}X_j=2\right)}.
\]

This identity follows by conditioning on customer \(i\): changing that customer's success probability affects the threshold event precisely when **exactly two of the other five** customers subscribe. Each derivative lies in \([0,1]\), so a conservative fixed-selection bound is

\[
|\delta F|\ \le\ \sum_{i=0}^{5}|\delta p_i|
\]

for probability vectors connected by an in-range coordinate path. If uncertainty estimates and their dependence structure are available, one can also propagate a covariance matrix \(\Sigma_p\) through \(\nabla F^\mathsf T\Sigma_p\nabla F\) as a local approximation.

**Important limitation:** the source README did **not** provide per-customer probability uncertainty, a calibrated covariance matrix, or a recorded numerical six-customer sensitivity study. Therefore this section supplies a **mathematically justified sensitivity framework, not a measured prediction-error interval**. If Stage 1 uncertainty changes the *top-10 set or the chosen six*, a fixed-selection derivative is insufficient: the entire ranking–optimisation–outcome pipeline should be rerun under perturbed or bootstrap-fitted probabilities.

### 4.6 Stage 1 error propagation into Stage 2 decisions

For a **fixed contact set** \(S\), predicted Stage 2 value is \(Z_S=\sum_{i\in S}(p_iD_i-c_i)\). Small changes in probabilities change the reported value by

\[
\delta Z_S=\sum_{i\in S}D_i\,\delta p_i.
\]

This relationship is exact for a fixed selected set and fixed \(D_i,c_i\). The identity of the optimal contact set, however, can change discontinuously if ranking or value ordering changes. A practical follow-up analysis would bootstrap Stage 1 fits, recompute the **top 10**, resolve all 210 candidate selections for each replicate, and then recompute Stage 3's success probability. **That end-to-end uncertainty analysis is not claimed as a completed recorded result.**

## 5. Business interpretation and recommendations

**Recommended contact list:** `C0081`, `C0079`, `C0031`, `C0327`, `C0374`, `C0222` (the canonical six selected candidates, in Stage 3 qubit order).

**Decision summary:** Within the supplied model and economic assumptions, this set achieves the exact benchmark value **43.183204088500**, uses **14 of 25** budget units and **11 of 22** capacity units, and has an **estimated 67.80%** chance of reaching at least three subscriptions under the conditional-independence assumption. The success chance is **not** the probability that the campaign will achieve any particular profit level.

**Operational recommendation:** If the bank accepts the underlying cost/deposit definitions and model assumptions, these six customers are the **model-preferred test cohort**. Before production rollout, evaluate probability calibration and ranking stability on fresh labelled data, examine outcome correlations and fairness/operational constraints as appropriate, and validate returns through a pilot or controlled experiment. As all 210 feasible selections meet the reported budget and capacity bounds, this instance provides no evidence that quantum optimisation is operationally necessary: the exact classical procedure is faster and simpler for this scale.

**Do not over-interpret:** the objective values reflect predicted, model-defined economics; the outcome circuit assumes independent customer responses; and the submission contains no observed post-campaign conversions. The small labelled sample and exploratory Stage 1 model selection may impair generalisation.

## 6. Reproduction and environment

The archive is intended to be self-contained at the **project/code/data/evidence** level: an external GitHub checkout or second ZIP is not required, but Python dependencies still need installing. The documented supported environment is **Python 3.11+** with recorded Qiskit **2.6.2** and Qiskit Aer **0.18.2**. A working Conda environment named `qiskit-hackathon` was used in the suggested macOS commands, but users may create an equivalent environment.

From the **repository root**:

```bash
conda activate qiskit-hackathon
python -m pip install -r requirements.txt

# Check canonical files and recorded simulator evidence (no Stage 1 refit):
python run_all.py verify

# Check the per-file hash manifest:
python verify_manifest.py

# Reproduce the independent classical Stage 3 baseline:
python code/stage3_monte_carlo_reference.py --seed 42 --shots 8192
```

Optional, potentially slower and environment-dependent runs:

```bash
python run_all.py stage1-baseline
python run_all.py stage1-refit
python run_all.py stage1-nested
python run_all.py aer-stages23
python run_all.py aer-p3-p8-replay
```

**Verification hierarchy:** `verify` checks the **stored canonical outputs and recorded Aer evidence** (including the 18 supplementary cases), not a bit-for-bit reconstruction of N17 training. `stage1-refit` is available but exact reproduction of original N17 probabilities was **not independently confirmed** in the packaging environment; it should not silently overwrite `stage1_result.csv`. A fresh finite-shot simulator run may produce different measured counts even when the underlying probability distribution agrees.

Dependency and provenance references:

- `requirements.txt`: primary project dependencies.
- `code/requirements_aer.txt`: Aer-related requirements.
- `code/requirements_friend_original.txt`: teammate's original environment requirements; may not be fully jointly pin-compatible.
- `code/friend_SOURCE_PROVENANCE.md`: original teammate source snapshot and provenance.
- `README_START_HERE.md`: short startup and verification instructions.
- `MANIFEST_SHA256.txt`: stored file checksums, to be checked with `python verify_manifest.py`.

### Reproduction checklist for judges

- [ ] Run the root-level verification; confirm all canonical Stage 1–3 files pass their internal consistency tests.
- [ ] Confirm 100 distinct Stage 1 customer IDs, descending probabilities, deterministic tie-breaking and exactly 10 `top10_flag` entries.
- [ ] Check that Stage 2 selects exactly 6 IDs **from those top 10**, stays within both limits and matches the stated candidate-to-bit ordering.
- [ ] Independently enumerate the 210 six-person sets to verify the objective and stated non-binding inequalities.
- [ ] Verify the Stage 3 six IDs and probabilities match the Stage 2/Stage 1 canonical files, without rounding or substitutions.
- [ ] Compare the six-qubit ideal Aer sample and independent Monte Carlo estimate with the exact Poisson–binomial calculation.
- [ ] Distinguish saved results from optional refits/replays, and ideal simulator counts from synthetic-noise runs.
- [ ] Inspect the actual transpilation reports before treating any logical-depth or opaque-state-initialisation number as a device-level hardware estimate.

## 7. Reproducibility and scientific limitations

| Topic | What is supported | What is not established |
|---|---|---|
| Stage 1 | Documented predictor, saved 100 probabilities, exploratory CV and a nested-CV implementation | Independent hidden-test performance or bitwise-identical N17 refit |
| Stage 2 | Explicit binary objective, feasible-subspace design, ideal Aer QAOA samples, exact 210-combination check | General inequality enforcement by the XY mixer; global angle optimisation; quantum speedup |
| Circuit resources | Logical rotations and a separate generic-basis Dicke decomposition | Full device-routed circuit statistics, physical-hardware execution or calibrated noise estimates |
| Stage 3 | Six-qubit probability encoding, exact Poisson–binomial reference, ideal/noisy Aer samples, classical Monte Carlo | Real-world conversion outcomes; advantage over direct classical sampling/dynamic programming |
| Uncertainty | Finite-shot sampling uncertainty and analytic first-order sensitivity formulas | A numerical end-to-end confidence interval for Stage 1 → Stage 2 → Stage 3 |
| Business result | Model-optimal six-person list and model-based success probability | Verified bank revenue, causal uplift or deployment readiness |

No benchmark in this submission demonstrates computational quantum advantage. The project is a reproducible hybrid-method **demonstration and comparison**, with clear classical baselines and explicit caveats about synthetic noise and probabilistic predictions.

## 8. Artifacts and source of truth

| Path | Purpose |
|---|---|
| `stage1_result.csv` | **Canonical** 100 probability predictions, ranks and top-10 flags |
| `stage2_candidates.csv` | **Canonical** candidate details for the optimisation |
| `stage2_result.csv`, `stage2_result.json` | **Canonical** selection, QAOA sample evidence and objective-related values |
| `stage3_result.csv`, `stage3_result.json` | **Canonical** outcome simulation and exact-comparison values |
| `evidence/qiskit_aer/` | Recorded simulator outputs and run-environment evidence |
| `evidence/stage3_monte_carlo.json` | Independent classical baseline evidence |
| `experiments/p3_p8/results/` | Eighteen supplementary depth/ansatz comparisons |
| `reports/AER_VERIFIED_TECHNICAL_REPORT.md` | Referenced additional implementation and circuit-analysis details |
| `reports/STAGE1_BASELINES.md`, `results/stage1_cv/`, `stage1_reverification/` | Referenced Stage 1 model and evaluation records |
| `presentation/` | Editable presentation material |
| `code/friend_atomiq/` | Teammate's original project snapshot; **not** an alternative official submission |
| `supplementary/numpy_baseline/`, `reports/LEGACY_*.md` | Historical or exploratory material; **not** canonical results |

**Submission integrity:** do not replace canonical root outputs with a teammate's alternative set, do not confuse old NumPy samples with recorded Aer counts, and do not round Stage 1 probabilities before computing Stage 2 weights. The file manifest and independent verifier are supplied to assist inspection of the packaged archive.

---

**Final claim:** The submitted pipeline joins predictive modelling, a feasibility-aware quantum-optimisation circuit and probability-encoded outcome simulation into one auditable workflow. Its strongest evidence is **correctness against transparent classical references and coherent use of the same Stage 1 probabilities**, not demonstrated practical quantum advantage.
