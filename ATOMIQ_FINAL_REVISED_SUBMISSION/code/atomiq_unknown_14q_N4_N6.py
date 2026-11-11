#!/usr/bin/env python3
"""N4-N6: unknown-only quantum specialists + a SEPARATE non-unknown model.

Runs outer stratified CV on ALL 300 customers. The 14q specialist is fitted
ONLY on the unknown portion of each training fold; the non-unknown branch is
an independent classical logistic model trained on all outer training rows
(no validation labels are used). Predictions are joined for global top-10.

N4: 14q singles + train-selected 25 of 91 connected 14q pairs + all 15
    connected 6q pairs from the original F2 six-attribute mapping.
N5: 14q singles + train-selected k of 91 pairs; k swept 5,10,15,25,40,60,91.
N6: 14q singles + 25 train-selected pair readouts from a DIFFERENT
    entanglement topology: ring / age-and-balance-centered / train-learned
    25-edge graph (the last uses only outer training labels).

N1/N2/N3 are refitted on the SAME outer folds for fair comparison. Outputs
both unknown-only and overall metrics. Graph selection and readout selection
always use training labels only. Hyperparameter selection on these OOF results
is exploratory, not unbiased external validation.

CLI:
  python atomiq_unknown_14q_N4_N6.py --train Stage1_Train.tsv --folds 5 --seeds 42 179 316
  python atomiq_unknown_14q_N4_N6.py --self-test
Requires numpy, pandas, scikit-learn, qiskit and adjacent:
  atomiq_stage1_AB_optimization.py
  atomiq_unknown_14q_N1_N3.py
"""
from __future__ import annotations
import argparse
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import atomiq_stage1_AB_optimization as base
import atomiq_unknown_14q_N1_N3 as old

COLS = old.COLUMNS
N = len(COLS)
PAIRS = old.PAIRS
K_VALUES = (5, 10, 15, 25, 40, 60, 91)
Q6_COLS = ('age','balance','campaign','poutcome','day','previous')


def model(seed):
    return Pipeline([('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='saga', penalty='elasticnet', C=0.1,
             l1_ratio=0.5, random_state=int(seed), max_iter=6000, tol=1e-3))])


def fit_probs(xtr, ytr, xva, seed):
    if len(set(ytr)) < 2:
        return np.full(len(xva), float(np.mean(ytr)))
    clf = model(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        clf.fit(xtr, ytr)
    return clf.predict_proba(xva)[:, 1]


def scoring(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1-1e-10)
    order = np.argsort(-p, kind='stable')[:min(10, len(y))]
    return dict(n=int(len(y)), positives=int(y.sum()),
                PR_AUC=float(average_precision_score(y,p)) if y.sum() else np.nan,
                ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                Brier=float(brier_score_loss(y,p)),
                LogLoss=float(log_loss(y,p,labels=[0,1])),
                top10_hits=int(y[order].sum()),
                precision_at_10=float(y[order].mean()),
                recall_at_10=float(y[order].sum()/y.sum()) if y.sum() else np.nan)


def sparse_quantum(angles, edges, strength=0.7):
    """14q angle encoding and RZZ edges, returns 14 singles + 91 connected pairs.

    Readouts always include ALL 91 Cij, even if an edge is not used in circuit.
    Hence N6 isolates circuit topology from readout selection.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    angles = np.asarray(angles, dtype=float).copy()
    hi = COLS.index('housing')
    angles[:,hi] = .25 + .5*angles[:,hi]  # match base shift=True
    edges = list(dict.fromkeys(tuple(sorted(e)) for e in edges))
    result = np.empty((len(angles), N+len(PAIRS)))
    factor = strength / math.sqrt(max(1, len(edges)))
    for r, row in enumerate(angles):
        qc=QuantumCircuit(N)
        for i, val in enumerate(row): qc.ry(np.pi*float(val), i)
        for i,j in edges: qc.rzz(factor*np.pi*float(row[i]*row[j]), i,j)
        for i in range(N): qc.rx(.5, i)
        prob=np.abs(np.asarray(Statevector.from_instruction(qc).data))**2
        # Fast Walsh-Hadamard transform for Z-product observables.
        for bit in range(N):
            d=1<<bit
            b=prob.reshape(-1, 2, d)
            a=b[:,0,:].copy(); c=b[:,1,:].copy()
            b[:,0,:]=a+c; b[:,1,:]=a-c
        singles=np.array([prob[1<<i] for i in range(N)])
        result[r,:N]=singles
        result[r,N:]=[prob[(1<<i)|(1<<j)]-singles[i]*singles[j] for i,j in PAIRS]
    return result


def graph_edges(name, qtrain, ytrain):
    if name == 'ring':
        return [(i, (i+1)%N) for i in range(N)]
    if name == 'age_balance_hubs':
        hubs = (COLS.index('age'), COLS.index('balance'))
        return [(i,j) for i,j in PAIRS if i in hubs or j in hubs]
    if name == 'learned_edges':
        chosen=old.choose_pairs_train_only(qtrain, ytrain, 25)
        return [PAIRS[int(j)] for j in chosen]
    raise ValueError(name)


def calculate_quantum(tr, va, ytr, enabled_graphs, fake=False):
    enc14=base.FoldAngleEncoder(COLS).fit(tr)
    a14tr=enc14.transform(tr); a14va=enc14.transform(va)
    if fake:
        def synthetic(a, width, shift):
            raw=a @ np.linspace(.3, 1.2, a.shape[1])
            return np.sin(raw[:,None]*np.linspace(.04,2.5,width)[None,:]+shift)
        qtr=synthetic(a14tr,105,.1); qva=synthetic(a14va,105,.1)
        q6tr=synthetic(a14tr,21,.3); q6va=synthetic(a14va,21,.3)
    else:
        qall=base.get_quantum_features(np.vstack([a14tr,a14va]), COLS,'full',True,.7)
        qtr,qva=qall[:len(tr)],qall[len(tr):]
        enc6=base.FoldAngleEncoder(Q6_COLS).fit(tr)
        a6=np.vstack([enc6.transform(tr),enc6.transform(va)])
        q6=base.get_quantum_features(a6,Q6_COLS,'full',False,.7)
        q6tr,q6va=q6[:len(tr)],q6[len(tr):]
    graphs={}
    for graph in enabled_graphs:
        edges=graph_edges(graph, qtr, ytr)
        if fake:
            qs_tr=synthetic(a14tr,105,.55+len(edges)*.002)
            qs_va=synthetic(a14va,105,.55+len(edges)*.002)
        else:
            combined=sparse_quantum(np.vstack([a14tr,a14va]),edges)
            qs_tr,qs_va=combined[:len(tr)],combined[len(tr):]
        graphs[graph]=(qs_tr,qs_va,edges)
    return qtr,qva,q6tr,q6va,graphs


def build_inputs(tr,va,ytr,qtr,qva,q6tr,q6va,graphs):
    prep=old.classical_preprocessor()
    ctr=np.asarray(prep.fit_transform(tr[COLS]),dtype=float)
    cva=np.asarray(prep.transform(va[COLS]),dtype=float)
    k25=old.choose_pairs_train_only(qtr,ytr,25)
    out={
        'N1_all_91':(qtr,qva),
        'N2_top25':(np.c_[qtr[:,:N],qtr[:,N:][:,k25]],
                     np.c_[qva[:,:N],qva[:,N:][:,k25]]),
        'N3_quantum_classical':(np.c_[ctr,qtr],np.c_[cva,qva]),
        'N4_14q25_plus_6q15':(
            np.c_[qtr[:,:N],qtr[:,N:][:,k25],q6tr[:,6:]],
            np.c_[qva[:,:N],qva[:,N:][:,k25],q6va[:,6:]])
    }
    for k in K_VALUES:
        indices=old.choose_pairs_train_only(qtr,ytr,k)
        out[f'N5_top{k}']=(np.c_[qtr[:,:N],qtr[:,N:][:,indices]],
                           np.c_[qva[:,:N],qva[:,N:][:,indices]])
    for name,(gs_tr,gs_va,_) in graphs.items():
        sel=old.choose_pairs_train_only(gs_tr,ytr,25)
        out[f'N6_{name}']=(np.c_[gs_tr[:,:N],gs_tr[:,N:][:,sel]],
                            np.c_[gs_va[:,:N],gs_va[:,N:][:,sel]])
    return out,k25


def run(train_path='Stage1_Train.tsv', outdir='unknown_N4_N6_results',
        seeds=(42,179,316), folds=5, graphs=('ring','age_balance_hubs'),
        fake=False, verbose=True):
    data=pd.read_csv(train_path,sep='\t')
    if data.customer_id.duplicated().any(): raise ValueError('Duplicate customer IDs')
    if set(data.y.unique()) != {'yes','no'}: raise ValueError('Expected yes/no in y')
    y=data.y.eq('yes').astype(int).to_numpy()
    is_unknown=data.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    if min(np.bincount(y[is_unknown]))<folds: raise ValueError('Too few unknown positive rows for folds')
    # Split global rows, stratifying both label and unknown routing.
    strata=y.astype(int)*2+is_unknown.astype(int)
    if np.min(np.bincount(strata)) < folds:
        raise ValueError('A (label, unknown-group) stratum has fewer than folds rows')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    oof_rows=[]; fold_metrics=[]; full_metrics=[]; selected=[]; edgelist=[]
    for seed in seeds:
        cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=int(seed))
        pred={}; fold_assignment=np.full(len(data),-1)
        for fold,(itr,iva) in enumerate(cv.split(data,strata),start=1):
            tra=data.iloc[itr]; val=data.iloc[iva]
            ytr=y[itr]
            # Independently trained global classical reference: applied ONLY to
            # nonunknown validation rows in routed candidates.
            classical=old.classical_preprocessor()
            cxtr=np.asarray(classical.fit_transform(tra[COLS]),dtype=float)
            cxva=np.asarray(classical.transform(val[COLS]),dtype=float)
            nonunknown_p=fit_probs(cxtr,ytr,cxva,int(seed))
            # The unknown specialist NEVER trains on success/failure/other.
            utrain=tra.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
            uvalid=val.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
            utra=tra.loc[utrain].reset_index(drop=True)
            uval=val.loc[uvalid].reset_index(drop=True)
            uy=ytr[utrain]
            if verbose:
                print(f'[seed {seed}, fold {fold}/{folds}] unknown train={len(utra)} valid={len(uval)}; quantum 14q',flush=True)
            qt,qv,q6t,q6v,graphs_q=calculate_quantum(utra,uval,uy,graphs,fake=fake)
            matrices,k25=build_inputs(utra,uval,uy,qt,qv,q6t,q6v,graphs_q)
            selected.extend(dict(seed=seed,fold=fold,model='N2/N4',pair_index=int(j),
                                 feature_a=COLS[PAIRS[int(j)][0]],feature_b=COLS[PAIRS[int(j)][1]]) for j in k25)
            for name,(_,_,edges) in graphs_q.items():
                edgelist.extend(dict(seed=seed,fold=fold,graph=name,feature_a=COLS[i],feature_b=COLS[j]) for i,j in edges)
            names=['Classical_global_only']+list(matrices)
            for name in names:
                if name not in pred: pred[name]=np.full(len(data),np.nan)
                pp=nonunknown_p.copy()
                if name!='Classical_global_only':
                    xtr,xva=matrices[name]
                    pp[uvalid]=fit_probs(xtr,uy,xva,int(seed))
                pred[name][iva]=pp
                fold_metrics.append(dict(seed=seed,fold=fold,model=name,subset='all',**scoring(y[iva],pp)))
                fold_metrics.append(dict(seed=seed,fold=fold,model=name,subset='unknown',**scoring(y[iva][uvalid],pp[uvalid])))
                if (~uvalid).any():
                    fold_metrics.append(dict(seed=seed,fold=fold,model=name,subset='nonunknown',**scoring(y[iva][~uvalid],pp[~uvalid])))
            fold_assignment[iva]=fold
        for name,pp in pred.items():
            if not np.isfinite(pp).all(): raise RuntimeError(f'Incomplete OOF: {name}')
            for subset,mask in [('all',np.ones(len(y),dtype=bool)),('unknown',is_unknown),('nonunknown',~is_unknown)]:
                full_metrics.append(dict(seed=seed,model=name,subset=subset,**scoring(y[mask],pp[mask])))
        for ix,row in data.iterrows():
            oof_rows.append(dict(seed=seed,customer_id=row.customer_id,poutcome=row.poutcome,
                label=int(y[ix]),fold=int(fold_assignment[ix]),**{f'p_{name}':float(pp[ix]) for name,pp in pred.items()}))
    metrics=pd.DataFrame(full_metrics)
    summary=(metrics.groupby(['subset','model'],as_index=False)
             .agg(**{f'{col}_mean':(col,'mean') for col in ['PR_AUC','ROC_AUC','Brier','LogLoss','top10_hits','precision_at_10','recall_at_10']},
                  PR_AUC_std=('PR_AUC','std'),top10_hits_std=('top10_hits','std')))
    pd.DataFrame(oof_rows).to_csv(out/'all_oof.csv',index=False)
    metrics.to_csv(out/'by_seed.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'by_fold.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'selected_pairs.csv',index=False)
    pd.DataFrame(edgelist).to_csv(out/'topology_edges.csv',index=False)
    if verbose:
        for subset in ['unknown','all']:
            print(f'\n=== {subset.upper()} ===')
            cols=['model','PR_AUC_mean','Brier_mean','top10_hits_mean']
            print(summary.loc[summary.subset==subset,cols].sort_values('PR_AUC_mean',ascending=False).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
        print('Saved:',out.resolve())
    return dict(summary=summary,by_seed=metrics)


def self_test():
    import tempfile
    path=Path(__file__).parent/'Stage1_Train.tsv'
    if not path.exists(): raise FileNotFoundError('Stage1_Train.tsv needed for self-test')
    with tempfile.TemporaryDirectory() as tmp:
        result=run(path,outdir=tmp,seeds=(42,),folds=3,graphs=('ring','age_balance_hubs','learned_edges'),fake=True,verbose=False)
        metrics=result['summary']
        assert not metrics.empty and set(metrics.subset)=={'all','unknown','nonunknown'}
        assert 'N4_14q25_plus_6q15' in set(metrics.model)
        assert 'N5_top91' in set(metrics.model)
        assert 'N6_learned_edges' in set(metrics.model)
        df=pd.read_csv(Path(tmp)/'all_oof.csv')
        assert len(df)==300 and df.customer_id.is_unique
        print('SELF-TEST PASS: synthetic features, not a quantum performance test')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default='Stage1_Train.tsv')
    parser.add_argument('--outdir',default='unknown_N4_N6_results')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,179,316])
    parser.add_argument('--graphs',nargs='+',choices=['ring','age_balance_hubs','learned_edges'],default=['ring','age_balance_hubs'])
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test()
    else: run(args.train,args.outdir,args.seeds,args.folds,args.graphs)
