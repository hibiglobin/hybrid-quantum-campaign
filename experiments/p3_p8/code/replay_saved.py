#!/usr/bin/env python3
"""Replay the exact *saved* p=3..8 circuit parameters in Qiskit Aer.
This deliberately does not re-optimize, and does not claim bit-for-bit shot equality.
"""
import argparse,json,csv,sys
from pathlib import Path
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import compare_p3_p8 as b

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--shots',type=int,default=8192);a.add_argument('--seed',type=int,default=42);v=a.parse_args()
 r=v.root.resolve();source=r/'experiments/p3_p8/results';out=r/'reproduced/p3_p8_replay';out.mkdir(parents=True,exist_ok=True)
 ids,p,cost,capacity,profit=b.read_data(r);data=b.prepare(cost,capacity,profit)
 backend=AerSimulator(method='statevector',seed_simulator=v.seed)
 output=[]
 for model in b.MODELS:
  for depth in range(3,9):
   saved=json.loads((source/f'{model}_p{depth}_parameters.json').read_text())
   params=np.asarray(saved['parameters'],dtype=float)
   qc=b.make_circuit(model,params,profit,data)
   _,prob=b.evaluate(model,params,data)
   exact=np.abs(np.asarray(Statevector.from_instruction(qc).data))**2
   error=float(np.max(np.abs(prob-exact)))
   if error>1e-8:raise RuntimeError(f'Circuit mismatch {model} p{depth}: {error}')
   qc.measure_all()
   tq=transpile(qc,backend,seed_transpiler=v.seed,optimization_level=1)
   counts={str(k).replace(' ',''):int(x) for k,x in backend.run(tq,shots=v.shots,seed_simulator=v.seed).result().get_counts().items()}
   assert sum(counts.values())==v.shots
   b.write_json(out/f'{model}_p{depth}_aer_counts.json',counts)
   hits=counts.get(format(data['optimum'],'010b'),0)
   feasible=sum(n for bits,n in counts.items() if data['valid'][int(bits,2)])
   item=dict(model=model,p=depth,shots=v.shots,best_hits=hits,feasible_shots=feasible,
      statevector_probability_discrepancy=error,depth_initialize_atomic=tq.depth(),
      original_source='saved optimized parameter file',native_dicke_depth_measured=False)
   output.append(item)
   print(f'{model:9} p={depth} best={hits}/{v.shots}, feasible={feasible}/{v.shots}, statevector_diff={error:.3g}',flush=True)
 b.write_json(out/'replay_summary.json',output)
 print('PASS: 18 saved circuits were executed and compared to independent NumPy probabilities')
if __name__=='__main__':main()
