# Stage 3 — Final Aer / Monte Carlo / exact comparison

## Target and circuit

Selected campaign candidates **C0081, C0079, C0031, C0327, C0374, C0222**, ordered q0…q5, have Stage 1 probabilities `[0.8493520678, 0.7295664424, 0.7017450767, 0.2641117305, 0.2268685727, 0.1827347972]`. A six-qubit circuit applies \(R_y(2\arcsin\sqrt{p_i})\) to each qubit and measures all six. Bitstrings are printed q5…q0 and an outcome succeeds if its Hamming weight is at least 3. No entanglement gates are required under the conditional independence model.

## Exact reference

The Poisson-binomial recurrence \(P_{j+1}(k)=(1-p_j)P_j(k)+p_jP_j(k-1)\) gives **P(K>=3) = 0.677951471871**.

## Three separately identified estimates

| Method | Seed | Shots | K>=3 successes | Estimate | Absolute error | Status |
|---|---:|---:|---:|---:|---:|---|
| Ideal Qiskit AerSimulator | 42 | 8192 | 5515 | 0.673217773 | 0.004733698 | Recorded Aer simulator counts |
| Classical Monte Carlo, NumPy PCG64 | 42 | 8192 | 5553 | 0.677856445 | 0.000095027 | Independently regenerated for this package |
| Qiskit Aer with illustrative synthetic noise | 42 | 8192 | 5392 | 0.658203125 | 0.019748347 | Not device-calibrated |

The NumPy MC has approximate standard error **0.005162958** and Wilson 95% interval **[0.667655710, 0.687890449]**. The independent seed-42 samples need not coincide with Qiskit Aer's seed-42 samples. Ideal and synthetic-noise Aer counts are verified from their respective evidence files.

## Device/complexity interpretation

The ideal Stage 3 circuit is six Ry gates and six measurements, depth 2 and **no two-qubit operations** as counted in saved Qiskit evidence. Shot noise does not incorporate Stage 1 predictive-model uncertainty or between-customer dependence. Exact dynamic programming is cheap for six Bernoulli trials; no quantum advantage is claimed. Supplementary MLAE runs, if used, must count Grover repetitions and decomposed multi-qubit gates rather than shots alone.

## Reproduction

`python code/stage3_monte_carlo_reference.py --shots 8192 --seed 42` is self-contained inside the installed environment (NumPy required). `python run_all.py aer-stages23` reruns the Qiskit Aer circuits into a distinct `reproduced/` folder.
