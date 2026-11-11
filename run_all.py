#!/usr/bin/env python3
"""Single-folder submission runner. Python 3.11+ recommended."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parent

def invoke(args):
    print('RUN:', ' '.join(map(str,args)),flush=True)
    return subprocess.call(list(map(str,args)),cwd=ROOT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify','stage1-baseline','stage1-refit','stage1-nested','aer-stages23','aer-p3-p8-optimize','aer-p3-p8-replay'])
    p.add_argument('--shots',type=int,default=8192)
    p.add_argument('--max-depth',type=int,default=8)
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--maxiter',type=int,default=350)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    py=sys.executable
    if a.command=='verify':
        r=invoke([py,ROOT/'verify_submission.py'])
        if r:return r
        return invoke([py,ROOT/'verify_extended.py'])
    if a.command.startswith('stage1-'):
        mode={'stage1-baseline':'baseline','stage1-refit':'prediction','stage1-nested':'nested'}[a.command]
        return invoke([py,ROOT/'code/verify_stage1_retraining.py',ROOT,'--mode',mode])
    if a.command=='aer-stages23':
        return invoke([py,ROOT/'code/run_aer_evidence.py','--input-dir',ROOT/'reproduce_inputs','--output-dir',ROOT/'reproduced/aer_stages23','--shots',a.shots,'--noise-shots',a.shots,'--seed',a.seed])
    if a.command=='aer-p3-p8-optimize':
        cmd=[py,ROOT/'experiments/p3_p8/code/compare_p3_p8.py','--submission',ROOT,
          '--checkpoints',ROOT/'experiments/p3_p8/checkpoints',
          '--baseline-csv',ROOT/'experiments/p3_p8/checkpoints/p1_p3_baseline.csv',
          '--out',ROOT/'reproduced/p3_p8_optimized','--shots',a.shots,'--seed',a.seed,
          '--max-depth',a.max_depth,'--starts',a.starts,'--maxiter',a.maxiter,'--aer']
        return invoke(cmd)
    if a.command=='aer-p3-p8-replay':
        return invoke([py,ROOT/'experiments/p3_p8/code/replay_saved.py','--root',ROOT,
           '--shots',a.shots,'--seed',a.seed])
    return 2
if __name__=='__main__':raise SystemExit(main())
