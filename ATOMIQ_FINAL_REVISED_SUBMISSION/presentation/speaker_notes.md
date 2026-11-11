# Atomiq Hackathon — Presenter notes (English)

**Suggested duration:** ~7–8 minutes for 13 slides, excluding questions. Slide 9 is exploratory and should not be represented as completed Aer measurements.

## Slide 1 — Introduction (20 sec)
Our project is an end-to-end hybrid quantum-classical pipeline for a bank's term-deposit campaign. It connects customer probability prediction, constrained outreach selection and conversion-risk estimation. We focus on transparent comparisons against classical reference methods.

## Slide 2 — Three linked tasks (30 sec)
We train on 300 labelled customers and predict probabilities for 100 test customers. The ten highest predictions become the only candidates in Stage 2. We select exactly six under budget and capacity constraints, then estimate the probability that at least three subscribe. The probability errors from Stage 1 propagate downstream.

## Slide 3 — Stage 1 model (35 sec)
We combine classical features with our teammate's compact logistic regression, quantum kernel and variational quantum classifier implementations. Our final N17 hybrid includes a history-aware correction with a fixed coefficient of 0.2. Importantly, we preserve the exact same predicted probabilities for all downstream calculations.

## Slide 4 — Benchmarks (40 sec)
We evaluated several classical baselines and the N16/N17 model family. Within the aligned, exploratory three-fold N16/N17 comparison, the N17 Friend Hybrid reaches a PR-AUC of approximately 0.357 and ROC-AUC of 0.734. These improvements are small and should not be interpreted as statistically significant. The independently rerun classical baselines use a slightly different cross-validation procedure.

## Slide 5 — Selected candidates (25 sec)
Here are our ten highest-probability customers. The six shaded rows are the final selection. Note that selection does not simply mirror predicted probability: we also consider contact costs and potential deposit value.

## Slide 6 — Business optimisation (40 sec)
We maximise the sum of expected net contributions, probability times deposit value minus outreach cost. We must select exactly six, stay within a budget of 25 and a capacity limit of 22. All 210 six-person combinations are feasible in this particular dataset, so we can enumerate the exact classical optimum: 43.183204, at cost 14 and capacity 11.

## Slide 7 — Quantum design (40 sec)
We start with a six-excitation Dicke state. The cost phase encodes the customers' net values, and the XY mixer preserves the number of selected customers. That keeps every ideal sample within the six-of-ten subspace. Since all those combinations also satisfy the two inequalities, penalties are unnecessary for this specific instance. This is an adaptation of known techniques, not a new QAOA algorithm.

## Slide 8 — Verified Aer (45 sec)
We ran Qiskit Aer with 8,192 shots per circuit at QAOA depths one to three. For Dicke plus XY at depth three, we observed the exact optimal bitstring 578 times, and all samples were feasible. The best observed solution matches the classical optimum. However, the optimal bitstring is not the most common one in every experiment, and a zero best-sample gap does not mean zero expected-value gap.

## Slide 9 — Deeper, exploratory search (30 sec)
We also optimised depths three through eight using ideal NumPy/SciPy simulation. Expected return generally improves, but the optimum's sampling probability does not increase monotonically. These are local optima, and the new depths require additional Aer verification. Deeper is not automatically more cost-effective on quantum hardware.

## Slide 10 — Stage 3 probability (35 sec)
For each of the six selected customers, an Ry rotation encodes the binary response probability. We count outcomes with at least three ones. Classical exact enumeration gives 67.7951 percent. Ideal Aer yields 67.3218 percent from 5,515 successful shots out of 8,192. With an illustrative, non-calibrated noise model, the estimate is 65.8203 percent.

## Slide 11 — Amplitude estimation (40 sec)
As an optional extension, we implemented maximum-likelihood amplitude estimation with Grover powers zero, one, two, four and eight. The estimated probabilities approach the exact reference well in ideal Aer. However, each amplified shot uses more circuit operations, so these shot counts must not be used as proof of a quantum speed-up.

## Slide 12 — Limitations (30 sec)
This small instance is easy classically, and our generic Dicke state preparation becomes extremely deep when decomposed into basis gates. Aer is a simulator, not real quantum hardware. Stage 3 conditions on fixed Stage 1 probabilities and therefore does not represent model uncertainty. We report these limitations explicitly.

## Slide 13 — Conclusion (20 sec)
Our contribution is a connected and reproducible hybrid workflow, with an exact classical reference at each stage. The main lesson is that structure-aware quantum circuit design and honest benchmarking are more meaningful here than unsupported claims of quantum advantage.

## Anticipated judge questions

**Q: What is actually quantum?** The teammate's quantum kernel and VQC are used in the Stage 1 hybrid; Stage 2 uses QAOA circuit execution; Stage 3 uses amplitude-encoded measurement and an MLAE extension. All currently reported quantum executions are Qiskit Aer simulations.

**Q: Why is the Stage 2 optimum not a proof of quantum advantage?** The best sample is compared with exhaustive enumeration of only 210 combinations. The classical reference is trivial for this instance.

**Q: Why can you avoid a penalty term?** XY dynamics preserve six selected customers, while every six-subset in this particular candidate pool meets both resource limits. This shortcut would fail for a more restrictive instance.

**Q: Why is the transpiled Dicke circuit so deep?** Qiskit's general `initialize` is not a hardware-efficient Dicke preparation strategy; generic decomposition incurs many entangling gates. The reported depth is not an optimised native-hardware resource estimate.

**Q: Can the Stage 1 CV results establish a winning model?** No. The exploratory comparison has limited seeds, different baseline protocols, and no significance testing. The submitted test labels are hidden.
