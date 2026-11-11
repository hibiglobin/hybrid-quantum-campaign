# Getting started — one-folder official submission

This directory contains canonical `stage1_result.csv`, `stage2_candidates.csv`, `stage2_result.csv`, `stage3_result.csv`, complete source/data, and recorded Qiskit Aer simulator evidence. **No external GitHub checkout or additional ZIP is required**; Python dependencies still must be installed.

```bash
conda activate qiskit-hackathon
python run_all.py verify
python verify_manifest.py
```

After Python package installation if needed (`python -m pip install -r requirements.txt`), reproduce the **classical Monte Carlo** result:

```bash
python code/stage3_monte_carlo_reference.py --seed 42 --shots 8192
```

Optional heavier / environment-dependent runs (saved official files never overwritten):

```bash
python run_all.py stage1-baseline
python run_all.py stage1-refit
python run_all.py aer-stages23
python run_all.py aer-p3-p8-replay
```

**What is and is not proved:** `verify` checks saved outputs and all 18 recorded Aer evidence files, not bitwise refit of Stage 1. `stage1-refit` is provided, but bit-for-bit reproducing original N17 predictions is not independently verified in this packaging environment. Aer is a simulator, not real quantum hardware; no quantum speedup is claimed. Read `README.md` for design reasoning, model equations, numerical results, CV caveats, and report hierarchy.
