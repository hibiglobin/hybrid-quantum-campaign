#!/usr/bin/env python3
"""Repeat independent Stage 3 classical Monte Carlo, based on root canonical probabilities.
Writes to a new output path, never overwriting submitted evidence.
"""
from pathlib import Path
import argparse, json, math
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--shots',type=int,default=8192);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,default=None)
 a=p.parse_args();s=json.loads((a.root/'stage3_result.json').read_text());probs=np.asarray(s['probabilities_q0_to_q5'],dtype=float)
 rng=np.random.default_rng(a.seed)
 successes=int(np.count_nonzero((rng.random((a.shots,6))<probs).sum(axis=1)>=3))
 result={'backend':'Classical NumPy Bernoulli Monte Carlo', 'seed':a.seed,'shots':a.shots,'success_shots':successes,'estimate':successes/a.shots,'exact_probability':s['exact_probability'],'absolute_error':abs(successes/a.shots-s['exact_probability'])}
 print(json.dumps(result,indent=2))
 if a.out:
  a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
