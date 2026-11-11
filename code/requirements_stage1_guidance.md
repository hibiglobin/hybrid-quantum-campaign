# Stage 1 installation guidance

The original run used Python 3.11.18, Qiskit 2.6.2 and Qiskit Aer 0.18.2. The teammate repository pins NumPy 2.5.3 (requires Python 3.12+), which may conflict with that environment. **Do not blindly run its full requirements.txt on Python 3.11.**

Prefer the preserved working `qiskit-hackathon` conda environment from the author. Add only missing imports and record `pip freeze` to fix the actual run. If the teammate package cannot be imported, create a separate Python >=3.12 environment and verify compatible Qiskit packages.

The teammate's source is needed at `friend_atomiq/src/qcampaign`; `python code/atomiq_N17_final_test_top10.py --train data/Stage1_Train.tsv --test data/Stage1_Test.tsv --friend-repo friend_atomiq --gamma 0.2`
