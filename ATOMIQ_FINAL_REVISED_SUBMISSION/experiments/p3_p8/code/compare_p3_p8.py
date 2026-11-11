#!/usr/bin/env python3
"""Stage 2: fair p=3..8 Dicke+XY / Dicke+X / Plus+X comparison.

Independent exact enumeration; ideal NumPy/SciPy optimization, optional Qiskit
Statevector parity and Aer sampling. Does NOT claim quantum speedup.

Run: python code/compare_p3_p8.py --submission ATOMIQ_TEAM_COMPLETE_SUBMISSION \
       --out p3_p8_results --checkpoints ATOMIQ_P3_P8_FAIR_BENCHMARK/checkpoints \
       --starts 3 --maxiter 350 --shots 8192 --aer
"""
import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

N=10
DIM=1<<N
K=6
ORDER=list(range(0,9,2))+list(range(1,9,2))
MODELS=('dicke_xy','dicke_x','plus_x')


def read_data(root):
    with (root/'stage1_result.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==100 and len({r['customer_id'] for r in rows})==100
    preds={r['customer_id']:float(r['predicted_probability']) for r in rows}
    ids=sorted(preds,key=lambda c:(-preds[c],c))[:N]
    with (root/'reproduce_inputs'/'N17_top10_economics.csv').open(newline='') as f:
        econ={r['customer_id']:r for r in csv.DictReader(f)}
    assert all(c in econ for c in ids)
    cost=np.array([float(econ[c]['contact_cost']) for c in ids])
    capacity=np.array([float(econ[c]['capacity_units']) for c in ids])
    p=np.array([preds[c] for c in ids])
    deposits=np.array([float(econ[c]['expected_deposit_value']) for c in ids])
    return ids,p,cost,capacity,p*deposits-cost


def prepare(cost,capacity,value):
    masks=np.arange(DIM,dtype=np.int32)
    bits=((masks[:,None] >> np.arange(N)) & 1)
    weight=bits.sum(axis=1)
    budget=bits@cost
    cap=bits@capacity
    profit=bits@value
    valid=(weight==K)&(budget<=25)&(cap<=22)
    assert int(valid.sum())==math.comb(N,K)==210, 'Check this instance before using penalty-free XY'
    optimum=int(np.argmax(np.where(valid,profit,-np.inf)))
    assert format(optimum,'010b')=='0101001111'
    # Same penalty convention as benchmark_stage2.py; strict infeasible upper-bound penalty.
    penalty=2*float(np.abs(value).sum())+1
    # Penalize any infeasible six-subset as in original experiment (none for these inputs).
    diagonal_x=profit-penalty*(weight-K)**2-penalty*((~valid)&(weight==K))
    active=np.flatnonzero(weight==K)
    lookup={int(mask):idx for idx,mask in enumerate(active)}
    pairs=[]
    for j in ORDER:
        left=[];right=[]
        for mask in active:
            mask=int(mask)
            if not(mask>>j)&1 and (mask>>(j+1))&1:
                left.append(lookup[mask]);right.append(lookup[mask^(1<<j)^(1<<(j+1))])
        pairs.append((np.array(left,dtype=int),np.array(right,dtype=int)))
    x_pairs=[]
    for j in range(N):
        a=masks[(masks & (1<<j))==0]
        x_pairs.append((a,a^(1<<j)))
    return dict(masks=masks,weight=weight,budget=budget,capacity=cap,profit=profit,
                valid=valid,optimum=optimum,penalty=penalty,diagonal_x=diagonal_x,
                active=active,xy_pairs=pairs,x_pairs=x_pairs)


def simulate(model,params,data):
    if model=='dicke_xy':
        psi=np.full(len(data['active']),1/math.sqrt(210),dtype=complex)
        energy=data['profit'][data['active']]
        for gamma,beta in zip(params[::2],params[1::2]):
            psi*=np.exp(1j*gamma*energy)
            c=np.cos(2*beta);s=-1j*np.sin(2*beta)
            for a,b in data['xy_pairs']:
                va=psi[a].copy();vb=psi[b].copy()
                psi[a]=c*va+s*vb;psi[b]=s*va+c*vb
        full=np.zeros(DIM,dtype=complex)
        full[data['active']]=psi
        return full
    psi=(data['weight']==K).astype(complex)/math.sqrt(210) if model=='dicke_x' else np.full(DIM,1/math.sqrt(DIM),dtype=complex)
    for gamma,beta in zip(params[::2],params[1::2]):
        psi*=np.exp(1j*gamma*data['diagonal_x'])
        c=np.cos(beta);s=-1j*np.sin(beta)
        for a,b in data['x_pairs']:
            va=psi[a].copy();vb=psi[b].copy()
            psi[a]=c*va+s*vb;psi[b]=s*va+c*vb
    return psi


def make_circuit(model,params,value,data):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    circuit=QuantumCircuit(N)
    if model.startswith('dicke'):
        init=np.zeros(DIM,complex)
        init[data['active']]=1/math.sqrt(210)
        circuit.initialize(init,range(N))
    else:
        circuit.h(range(N))
    for gamma,beta in zip(params[::2],params[1::2]):
        if model=='dicke_xy':
            # exp(+i gamma sum v_i x_i) implemented up to a global phase.
            for j,v in enumerate(value): circuit.rz(float(gamma*v),j)
            for edge in ORDER:
                circuit.rxx(2*float(beta),edge,edge+1)
                circuit.ryy(2*float(beta),edge,edge+1)
        else:
            circuit.append(DiagonalGate(np.exp(1j*gamma*data['diagonal_x'])),range(N))
            for j in range(N):circuit.rx(2*float(beta),j)
    return circuit


def evaluate(model,params,data):
    probs=np.abs(simulate(model,params,data))**2
    assert abs(float(probs.sum())-1)<1e-9
    return {'expected_raw_value':float(probs@data['profit']),
            'expected_penalized_value':float(probs@(data['profit'] if model=='dicke_xy' else data['diagonal_x'])),
            'feasible_probability':float(probs[data['valid']].sum()),
            'optimum_probability':float(probs[data['optimum']])},probs


def optimize_one(model,depth,data,rng,starts,maxiter,prior,candidate_checkpoint=None):
    candidates=[]
    def objective(params):
        ev,_=evaluate(model,params,data)
        return -ev['expected_penalized_value']
    def add(label,z,do_opt=True):
        if do_opt:
            r=minimize(objective,z,method='COBYLA',options={'maxiter':maxiter,'rhobeg':.25,'tol':1e-7})
            z=np.asarray(r.x,dtype=float); nfev=int(r.nfev)
        else:nfev=1
        ev,_=evaluate(model,z,data)
        candidates.append((ev['expected_penalized_value'],label,z,nfev))
    if prior is not None:
        z=np.concatenate((prior,[0.,0.]))
        add('warm_identity',z,False)
        add('warm_optimized',z,True)
        for extra in ((.12,.15),(-.2,.25)):
            zz=z.copy();zz[-2:]=extra;add('warm_perturbed',zz,True)
    if candidate_checkpoint is not None:
        z=np.array(candidate_checkpoint,dtype=float)
        if len(z)!=2*depth:raise ValueError('Checkpoint parameter dimension mismatch')
        add('previous_checkpoint',z,False)
        add('checkpoint_refine',z,True)
    for s in range(starts):
        z=np.empty(2*depth)
        z[::2]=rng.uniform(-1.3,1.3,depth)
        z[1::2]=rng.uniform(-.8,.8,depth)
        add(f'random_{s}',z,True)
    best=max(candidates,key=lambda t:t[0])
    return best[2],best[1],sum(t[3] for t in candidates)


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def run(args):
    root=Path(args.submission).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cks=Path(args.checkpoints).resolve() if args.checkpoints else None
    ids,p,cost,capacity,value=read_data(root)
    data=prepare(cost,capacity,value)
    optimum=data['optimum']
    write_json(out/'exact_reference.json',{
        'customer_ids_q0_to_q9':ids,'probabilities':p.tolist(),'net_values':value.tolist(),
        'costs':cost.tolist(),'capacities':capacity.tolist(),
        'exact_optimum':float(data['profit'][optimum]),
        'optimal_bitstring_q9_to_q0':format(optimum,'010b'),
        'optimal_ids':[ids[i] for i in range(N) if (optimum>>i)&1],
        'feasible_6_subsets':int(data['valid'].sum()),'total_6_subsets':math.comb(N,K),
        'x_mixer_penalty':float(data['penalty'])})
    if args.aer:
        from qiskit import transpile
        from qiskit.quantum_info import Statevector
        from qiskit_aer import AerSimulator
        import qiskit,qiskit_aer
        backend=AerSimulator(method='statevector',seed_simulator=args.seed)
        write_json(out/'aer_versions.json',{'qiskit':qiskit.__version__,'qiskit_aer':qiskit_aer.__version__,'seed':args.seed,'shots':args.shots})
    rows=[]
    for model in MODELS:
        prior=None
        for depth in range(3,args.max_depth+1):
            checkpoint=None
            ck=cks/f'p{depth}_result.json' if cks else None
            if model=='dicke_xy' and ck is not None and ck.exists():
                checkpoint=json.loads(ck.read_text())['optimized_gamma_beta_interleaved']
            if depth==3 and model!='dicke_xy' and args.baseline_csv:
                with Path(args.baseline_csv).open(newline='') as f:
                    for r in csv.DictReader(f):
                        if r['model']==model and int(r['depth_p'])==3:
                            checkpoint=json.loads(r['gamma_beta_interleaved_json'])
                            break
            rng=np.random.default_rng(args.seed+1000*depth+{'dicke_xy':1,'dicke_x':2,'plus_x':3}[model])
            t0=time.perf_counter()
            params,chosen,nfev=optimize_one(model,depth,data,rng,args.starts,args.maxiter,prior,checkpoint)
            result,probabilities=evaluate(model,params,data)
            row={'model':model,'depth_p':depth,'optimized_gamma_beta_interleaved':json.dumps(params.tolist()),
                 'selected_start':chosen,'optimization_evaluations':nfev,
                 'optimization_seconds':round(time.perf_counter()-t0,3),
                 'classical_exact_optimum':float(data['profit'][optimum]),
                 'optimal_bitstring':format(optimum,'010b'),
                 'two_qubit_rotations_excluding_stateprep':18*depth if model=='dicke_xy' else 0,
                 **result}
            if args.aer:
                circuit=make_circuit(model,params,value,data)
                state=np.asarray(Statevector.from_instruction(circuit).data)
                discrepancy=float(np.max(np.abs(state)**2-probabilities).__abs__())
                if discrepancy>1e-8:
                    raise RuntimeError(f'Qiskit-vs-NumPy mismatch model={model} p={depth}: {discrepancy:.6g}')
                measured=circuit.copy();measured.measure_all()
                transpiled=transpile(measured,backend,seed_transpiler=args.seed,optimization_level=1)
                counts={str(k).replace(' ',''):int(v) for k,v in backend.run(transpiled,shots=args.shots,seed_simulator=args.seed).result().get_counts().items()}
                assert sum(counts.values())==args.shots
                feasible_shots=sum(cnt for bit,cnt in counts.items() if data['valid'][int(bit,2)])
                best_hits=counts.get(format(optimum,'010b'),0)
                row.update({'aer_shots':args.shots,'aer_best_hits':best_hits,
                            'aer_feasible_shots':feasible_shots,
                            'aer_best_frequency':best_hits/args.shots,
                            'aer_feasible_frequency':feasible_shots/args.shots,
                            'qiskit_numpy_max_probability_discrepancy':discrepancy,
                            'aer_transpiled_depth_initialize_atomic':transpiled.depth(),
                            'aer_transpiled_size_initialize_atomic':transpiled.size(),
                            'aer_transpiled_gates':json.dumps({str(k):int(v) for k,v in transpiled.count_ops().items()})})
                write_json(out/f'{model}_p{depth}_aer_counts.json',counts)
            rows.append(row)
            write_json(out/f'{model}_p{depth}_parameters.json',{'model':model,'p':depth,'parameters':params.tolist(),**result})
            print(f"{model:9s} p={depth} E={result['expected_raw_value']:.5f} feas={result['feasible_probability']:.3%} optimal={result['optimum_probability']:.3%}"+(f" Aer hits={row['aer_best_hits']}" if args.aer else ' NumPy only'),flush=True)
            prior=params
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (out/'comparison_p3_p8.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    write_json(out/'comparison_p3_p8.json',rows)
    print('COMPLETE:',out,'conditions:',len(rows),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission',required=True)
    ap.add_argument('--out',default='p3_p8_results')
    ap.add_argument('--checkpoints',default=None)
    ap.add_argument('--baseline-csv',default=None)
    ap.add_argument('--max-depth',type=int,default=8)
    ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--maxiter',type=int,default=350)
    ap.add_argument('--shots',type=int,default=8192)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--aer',action='store_true')
    args=ap.parse_args()
    if not 3<=args.max_depth<=8:ap.error('--max-depth must be 3..8')
    run(args)
