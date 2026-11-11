#!/usr/bin/env python3
"""N7-N9: 14q Ring specialist for poutcome=unknown, routed with fold-trained F2.

CLI:
 python atomiq_unknown_14q_N7_N9.py --train Stage1_Train.tsv --folds 5 --seeds 42 179 316
 python atomiq_unknown_14q_N7_N9.py --self-test

Dependencies: numpy, pandas, scipy, scikit-learn, qiskit, and adjacent scripts
 atomiq_stage1_AB_optimization.py, atomiq_stage1_15q_plus_6q_feature_fusion.py,
 atomiq_unknown_14q_N1_N3.py, atomiq_unknown_14q_N4_N6.py

N6 reference: 14q ring, strength 0.7, 25 train-selected connected pairs.
N7: N6 Ring readouts + classical features (also all 91 + classical).
N8: Ring with top K=5,10,15,25,40,60,91 connected pairs.
N9: Ring entangling strengths 0.35/0.7/1.05 and depths 1/2,
    25 readout pairs chosen on each outer-training fold.

All scalers, feature screening, and models are fitted within OUTER TRAIN folds.
Non-unknown baseline F2 is trained on the full outer TRAIN (both strata) and
applied only to non-unknown OUTER VALID rows. The F2 global baseline is also
scored on all rows. Unknown specialists train only on unknown rows.

Optional --friend-oof file: post-hoc comparison to 0.7*F2+0.3*friend LR
for the non-unknown branch only. This is DIAGNOSTIC, not leakage-safe nested
CV, because the friend's OOF predictions use different folds.

Computation is heavy: statevectors for 14q and 15q and six N9 circuits.
Use --self-test before real runs; --seeds 42 for a first real run.
"""
from __future__ import annotations
import argparse
import math
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import atomiq_stage1_AB_optimization as base
import atomiq_stage1_15q_plus_6q_feature_fusion as fusion
import atomiq_unknown_14q_N1_N3 as old
import atomiq_unknown_14q_N4_N6 as prev

COLS = old.COLUMNS
NQ = len(COLS)
PAIRS = old.PAIRS
Q6 = fusion.Q6_COLS
K_VALUES = (5, 10, 15, 25, 40, 60, 91)
N9_CONFIGS = ((0.35,1),(0.70,1),(1.05,1),(0.35,2),(0.70,2),(1.05,2))
RING = [(i, (i + 1) % NQ) for i in range(NQ)]


def compute_ring(angles, strength, depth):
    """Qiskit 14q statevector, 14 Z and all 91 connected ZZ expectations."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    angles = np.asarray(angles, dtype=float).copy()
    angles[:, COLS.index('housing')] = 0.25 + 0.5*angles[:, COLS.index('housing')]
    out = np.empty((len(angles), 14+91), dtype=np.float64)
    factor = float(strength) / math.sqrt(len(RING)) / int(depth)
    for r, row in enumerate(angles):
        qc = QuantumCircuit(NQ)
        for i, v in enumerate(row): qc.ry(np.pi*float(v), i)
        for _ in range(depth):
            for i,j in RING: qc.rzz(np.pi*factor*float(row[i]*row[j]),i,j)
            for i in range(NQ): qc.rx(0.5/depth, i)
        p = np.abs(np.asarray(Statevector.from_instruction(qc).data))**2
        for bit in range(NQ):
            d = 1<<bit
            a = p.reshape(-1,2,d)
            p0=a[:,0,:].copy(); p1=a[:,1,:].copy()
            a[:,0,:]=p0+p1
            a[:,1,:]=p0-p1
        single = np.asarray([p[1<<i] for i in range(NQ)])
        out[r,:NQ]=single
        out[r,NQ:]=[p[(1<<i)|(1<<j)] - single[i]*single[j] for i,j in PAIRS]
    return out


def fake_ring(angles,strength,depth):
    a=np.asarray(angles, dtype=float)
    v = a @ np.linspace(0.3,1.2,a.shape[1])
    f=np.arange(105)[None,:]
    return np.sin(v[:,None]*(0.04+f*0.023)+float(strength)*0.2+depth*0.11)


def make_f2_outer(tr,va,ytr,seed,fake=False):
    """Outer-fold F2 predictions: no outer validation labels used."""
    if fake:
        # Only for pipeline self-test, never for real model evaluation.
        c=base.classical_preprocessor()
        a=np.asarray(c.fit_transform(tr[base.ORDER]),dtype=float)
        b=np.asarray(c.transform(va[base.ORDER]),dtype=float)
        return prev.fit_probs(a,ytr,b,seed)
    q15t,q15v=fusion.make_quantum_features(tr,va,base.ORDER,True)
    q6t,q6v=fusion.make_quantum_features(tr,va,Q6,False)
    return fusion.fit_predict(tr,ytr,va,q15t,q15v,q6t,q6v,variant=2)


def score(y,p):
    return prev.scoring(y,p)


def run(train='Stage1_Train.tsv',outdir='unknown_N7_N9_results',folds=5,
        seeds=(42,179,316),friend_oof=None,fake=False,verbose=True):
    df=pd.read_csv(train,sep='\t')
    needed={'customer_id','y','poutcome'}|set(base.ORDER)
    if missing:=needed-set(df): raise ValueError(f'Missing columns: {missing}')
    if df.customer_id.duplicated().any(): raise ValueError('Duplicate IDs')
    y=df.y.astype(str).str.lower().eq('yes').astype(int).to_numpy()
    mask=df.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    strata=2*y+mask.astype(int)
    if np.bincount(strata).min()<folds:
        raise ValueError('Too few members of a label/history stratum for folds')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    friend=None
    if friend_oof:
        tab=pd.read_csv(friend_oof)
        required={'customer_id','lr_compact'}
        if required-set(tab):raise ValueError('friend OOF needs customer_id and lr_compact')
        if tab.customer_id.duplicated().any():raise ValueError('Duplicate friend IDs')
        aligned=df[['customer_id']].merge(tab[['customer_id','lr_compact']],on='customer_id',how='left',validate='one_to_one')
        if aligned.lr_compact.isna().any():raise ValueError('Missing customer IDs in friend OOF')
        friend=np.asarray(aligned.lr_compact,dtype=float)
        print('WARNING: friend OOF uses DIFFERENT folds: 70:30 comparison is exploratory only.',flush=True)
    all_preds=[];by_seed=[];by_fold=[];selrows=[]
    for seed in seeds:
        cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=int(seed))
        preds={}; foldid=np.zeros(len(df),dtype=int)
        for fold,(itr,iva) in enumerate(cv.split(df,strata),1):
            tr=df.iloc[itr].reset_index(drop=True)
            va=df.iloc[iva].reset_index(drop=True)
            yt=y[itr]
            utrain=tr.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
            uval=va.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
            ut=tr.loc[utrain].reset_index(drop=True)
            uv=va.loc[uval].reset_index(drop=True)
            uy=yt[utrain]
            if verbose:print(f'seed={seed} fold={fold}/{folds}, unknown={len(ut)} train/{len(uv)} valid',flush=True)
            # Globally trained F2, applied to both nonunknown and all-customer baseline.
            pf2=make_f2_outer(tr,va,yt,int(seed),fake=fake)
            matrices={}; chosen={}
            enc=base.FoldAngleEncoder(COLS).fit(ut)
            at=enc.transform(ut); av=enc.transform(uv)
            prep=old.classical_preprocessor()
            ct=np.asarray(prep.fit_transform(ut[COLS]),dtype=float)
            cvmat=np.asarray(prep.transform(uv[COLS]),dtype=float)
            angles=np.vstack([at,av])
            for strength,depth in N9_CONFIGS:
                if verbose:print(f'  ring: strength={strength:g}, depth={depth}',flush=True)
                q= fake_ring(angles,strength,depth) if fake else compute_ring(angles,strength,depth)
                qt,qv=q[:len(ut)],q[len(ut):]
                for k in ([*K_VALUES] if (strength,depth)==(0.7,1) else [25]):
                    idx=old.choose_pairs_train_only(qt,uy,k)
                    name=(f'N8_ring_top{k}' if (strength,depth)==(0.7,1)
                          else f'N9_ring_s{strength:g}_L{depth}')
                    mtrain=np.c_[qt[:,:NQ],qt[:,NQ:][:,idx]]
                    mvalid=np.c_[qv[:,:NQ],qv[:,NQ:][:,idx]]
                    matrices[name]=(mtrain,mvalid)
                    for j in idx:
                        i1,i2=PAIRS[int(j)]
                        selrows.append(dict(seed=seed,fold=fold,model=name,
                            feature_a=COLS[i1],feature_b=COLS[i2],pair_index=int(j)))
                    if (strength,depth)==(0.7,1) and k==25:
                        matrices['N7_ring25_plus_classical']=(np.c_[ct,mtrain],np.c_[cvmat,mvalid])
                    if (strength,depth)==(0.7,1) and k==91:
                        matrices['N7_ring91_plus_classical']=(np.c_[ct,mtrain],np.c_[cvmat,mvalid])
            cur={'Baseline_F2':np.asarray(pf2,dtype=float).copy()}
            for name,(xt,xv) in matrices.items():
                p=pf2.copy()
                p[uval]=prev.fit_probs(xt,uy,xv,int(seed))
                cur[name]=p
            if friend is not None:
                pnon=0.7*pf2+0.3*friend[iva]
                cur['Baseline_F2_friend30_DIAGNOSTIC']=pnon
                for name,(xt,xv) in matrices.items():
                    p=pnon.copy()
                    p[uval]=prev.fit_probs(xt,uy,xv,int(seed))
                    cur[f'{name}__friend30_DIAGNOSTIC']=p
            for name, p in cur.items():
                if name not in preds:preds[name]=np.full(len(df),np.nan)
                preds[name][iva]=p
                # Fold scores are useful for diagnosis but top10 per fold is not global top10.
                for subset,submask in [('all',np.ones(len(iva),dtype=bool)),('unknown',uval),('nonunknown',~uval)]:
                    if submask.sum() and np.unique(y[iva][submask]).size==2:
                        by_fold.append(dict(seed=seed,fold=fold,model=name,subset=subset,**score(y[iva][submask],p[submask])))
            foldid[iva]=fold
        for name,p in preds.items():
            if not np.isfinite(p).all():raise RuntimeError('Incomplete OOF: '+name)
            for subset,submask in [('all',np.ones(len(df),dtype=bool)),('unknown',mask),('nonunknown',~mask)]:
                by_seed.append(dict(seed=seed,model=name,subset=subset,**score(y[submask],p[submask])))
        for i,id_ in enumerate(df.customer_id):
            all_preds.append(dict(seed=int(seed),customer_id=id_,poutcome=df.poutcome.iloc[i],label=int(y[i]),fold=int(foldid[i]),
                **{f'p_{name}':float(p[i]) for name,p in preds.items()}))
    metrics=pd.DataFrame(by_seed)
    summary=metrics.groupby(['subset','model'],as_index=False).agg(
        PR_AUC_mean=('PR_AUC','mean'),ROC_AUC_mean=('ROC_AUC','mean'),
        Brier_mean=('Brier','mean'),LogLoss_mean=('LogLoss','mean'),
        hits_mean=('top10_hits','mean'),hits_min=('top10_hits','min'),
        PR_AUC_std=('PR_AUC','std'))
    pd.DataFrame(all_preds).to_csv(out/'all_oof.csv',index=False)
    metrics.to_csv(out/'by_seed.csv',index=False)
    pd.DataFrame(by_fold).to_csv(out/'by_fold.csv',index=False)
    pd.DataFrame(selrows).to_csv(out/'selected_pairs.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    if verbose:
        for subset in ('unknown','all'):
            print('\n###',subset)
            print(summary[(summary.subset==subset)&(~summary.model.str.contains('DIAGNOSTIC'))]
                  .sort_values('PR_AUC_mean',ascending=False)
                  [['model','PR_AUC_mean','Brier_mean','hits_mean','hits_min']].to_string(index=False))
        print('Saved to',out.resolve())
    return summary


def self_test():
    with tempfile.TemporaryDirectory() as d:
        s=run(Path(__file__).parent/'Stage1_Train.tsv',d,folds=3,seeds=(42,),fake=True,verbose=False)
        names=set(s.model)
        assert {'N7_ring25_plus_classical','N8_ring_top91','N9_ring_s1.05_L2','Baseline_F2'}<=names
        assert set(s.subset)=={'all','unknown','nonunknown'}
        p=pd.read_csv(Path(d)/'all_oof.csv')
        assert len(p)==300 and p.customer_id.is_unique
        assert p.filter(like='p_').notna().all().all()
        print('SELF-TEST PASS (synthetic quantum data; not real quantum validation)')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',default='Stage1_Train.tsv')
    ap.add_argument('--outdir',default='unknown_N7_N9_results')
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,179,316])
    ap.add_argument('--friend-oof',default=None)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:self_test()
    else:run(a.train,a.outdir,a.folds,a.seeds,a.friend_oof)
