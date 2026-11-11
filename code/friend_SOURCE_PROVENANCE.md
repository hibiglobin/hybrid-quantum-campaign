# Teammate original source — provenance and integration

- Supplied archive: `atomiq-qiskit-fall-fest-main.zip` (uploaded by the user)
- SHA-256 (archive bytes): `17922b80a60c0d6f83938ce061a45869ccbe9f8d04a57ac73c61fa90bd141bdc`
- Copied into: `code/friend_atomiq/` without editing the original teammate files.
- Source snapshot includes `src/qcampaign/`, `tests/`, `notebooks/`, `requirements.txt`, `results/` and the original README.
- Original Git repository is indicated by prior conversation as `https://github.com/whatdislol/atomiq-qiskit-fall-fest`. This ZIP has no `.git` history; **commit ID is unverified**.
- License: no LICENSE file was present in the received ZIP. Confirm with the teammate that redistribution to contest judges is approved.

## Canonical results and namespace distinction

- Canonical Stage 1–3 final predictions, selected customers and Aer evidence are at the **outer submission folder root**.
- The teammate's **separate** stage1/stage2/stage3 outputs reside only at `code/friend_atomiq/` and are included for source provenance and baseline study; **do not submit those root output CSVs as a substitute for N17**.
- N17 imports `LRCompact` from `qcampaign.models.classical`, and `QuantumKernelKLR`, `QuantumVQC`, `HybridEnsemble` from `qcampaign.models.quantum`, and `FEATURES` from `qcampaign.data`, all present in the received archive.
- Re-run from this folder root: `python code/atomiq_N17_final_test_top10.py --train data/Stage1_Train.tsv --test data/Stage1_Test.tsv --friend-repo code/friend_atomiq --gamma 0.2`.
- Dependency warning: teammate pinned Python 3.13 Linux libraries while the original Stage 2/3 Aer evidence was captured on Python 3.11/macOS. No new model run is claimed here; use tested compatible environments for each.
