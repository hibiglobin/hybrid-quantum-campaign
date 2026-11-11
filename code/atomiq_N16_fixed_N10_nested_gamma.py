#!/usr/bin/env python3
"""N16: fixed N10 / F2+Friend and nested tuning of ONE history coefficient.

UNKNOWN: p = 0.5*p_F2 + 0.5*p_N8_Ring40
NON-UNKNOWN: p0 = 0.7*p_F2 + 0.3*p_FriendLR
  p(gamma) = sigmoid(logit(p0) + gamma*shrunk_history_log_odds_delta)

F2, N8 and FriendLR are fitted strictly on the current training fold.
A history delta is estimated strictly from the training fold's labels, with
40 prior pseudo-observations shrunk towards the training-fold overall rate.
Inner CV picks gamma only, and outer CV evaluates the result. Compare
no history (gamma=0), fixed gamma=0.20/0.35, and nested-chosen gamma.

Requires: Stage1_Train.tsv, plus all local dependencies of
atomiq_N15_strict_nested_history.py (N14, N7, N4, N1, F2 and AB scripts).

Usage:
  python atomiq_N16_fixed_N10_nested_gamma.py --self-test
  python atomiq_N16_fixed_N10_nested_gamma.py --train Stage1_Train.tsv --seeds 42
  python atomiq_N16_fixed_N10_nested_gamma.py --train Stage1_Train.tsv --seeds 42 179 316

WARNING: full run recomputes quantum statevectors in outer and inner folds;
Qiskit must be installed in the *active* Python environment.
"""
from __future__ import annotations
import argparse
import tempfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

import atomiq_N15_strict_nested_history as n15

EPS=1e-8

def probabilities(src, poutcome, gamma):
    group=np.asarray(poutcome).astype(str)
    unknown=np.char.lower(group)=='unknown'
    f2=np.asarray(src['f2'],float)
    ring=np.asarray(src['ring'],float)
    friend=np.asarray(src['friend'],float)
    delta=np.asarray(src['delta'],float)
    if not (len(f2)==len(ring)==len(friend)==len(delta)==len(group)):
        raise ValueError('Source arrays have inconsistent lengths')
    pred=0.5*f2+0.5*ring
    nonbase=0.7*f2+0.3*friend
    pred[~unknown]=n15.n14.sigmoid(n15.n14.logit(nonbase[~unknown])+gamma*delta[~unknown])
    return np.clip(pred,EPS,1-EPS)

def metrics(y,p,k=10):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),EPS,1-EPS)
    if len(y)==0: return None
    order=np.argsort(-p,kind='stable')[:min(k,len(y))]
    hits=int(y[order].sum())
    return dict(n=len(y),positives=int(y.sum()),
        PR_AUC=float(average_precision_score(y,p)) if y.sum() else np.nan,
        ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
        Brier=float(brier_score_loss(y,p)),
        LogLoss=float(log_loss(y,p,labels=[0,1])),
        hits_at_10=hits,precision_at_10=hits/len(order),
        recall_at_10=hits/int(y.sum()) if y.sum() else np.nan)

def objective(y,p,rule):
    m=metrics(y,p)
    if rule=='brier':return m['Brier']
    if rule=='brier_ap':return m['Brier']-0.03*m['PR_AUC']
    if rule=='logloss':return m['LogLoss']
    raise ValueError(rule)

def run(train, outdir, outer_folds, inner_folds, seeds, gamma_grid,
        fixed_gammas, select_rule, fake=False):
    df=pd.read_csv(train,sep='\t')
    must={'customer_id','poutcome','y'}
    if not must.issubset(df.columns):raise ValueError(f'Missing columns: {must-set(df.columns)}')
    if df.customer_id.duplicated().any():raise ValueError('Duplicate customer ID')
    y=df.y.astype(str).str.lower().str.strip().eq('yes').to_numpy(int)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    labels=['nested_gamma','no_history']+[f'fixed_gamma_{g:g}' for g in fixed_gammas]
    oof=[];choices=[];innertable=[];foldtable=[];friendtable=[];priorrows=[]
    for seed in seeds:
        strata=n15.strata_for(df,y,outer_folds)
        outer=StratifiedKFold(outer_folds,shuffle=True,random_state=seed)
        allpred={name:np.full(len(df),np.nan) for name in labels}
        fold_for=np.full(len(df),-1,int)
        for f,(tri,vai) in enumerate(outer.split(df,strata),1):
            tr=df.iloc[tri].reset_index(drop=True)
            va=df.iloc[vai].reset_index(drop=True)
            yt=y[tri]
            ys=n15.strata_for(tr,yt,inner_folds)
            inner=StratifiedKFold(inner_folds,shuffle=True,random_state=seed+1000+f)
            p_inner={name:np.full(len(tr),np.nan) for name in ['f2','ring','friend','delta']}
            print(f'seed={seed} outer={f}/{outer_folds}: inner CV',flush=True)
            for j,(it,iv) in enumerate(inner.split(tr,ys),1):
                src=n15.fit_sources(tr.iloc[it],tr.iloc[iv],yt[it],seed+101*f+j,fake=fake)
                for name in p_inner:p_inner[name][iv]=src[name]
                friendtable.append(dict(seed=seed,outer_fold=f,inner_fold=j,stage='inner',
                    best_k=src['friend_choice'][0],best_C=src['friend_choice'][1]))
            if not all(np.isfinite(a).all() for a in p_inner.values()):
                raise RuntimeError('Incomplete inner OOF predictions')
            candidates=[]
            for gamma in gamma_grid:
                p=probabilities(p_inner,tr.poutcome,gamma)
                val=objective(yt,p,select_rule);m=metrics(yt,p)
                candidates.append((val,gamma,m))
                innertable.append(dict(seed=seed,outer_fold=f,gamma=gamma,
                    inner_objective=val,**m))
            best=min(candidates,key=lambda it:(it[0],it[1]))
            gamma=best[1]
            choices.append(dict(seed=seed,outer_fold=f,chosen_gamma=gamma,
                inner_objective=best[0],inner_PR_AUC=best[2]['PR_AUC'],
                inner_Brier=best[2]['Brier']))
            print(f'  selected gamma={gamma:.3g}; refitting on outer training',flush=True)
            src=n15.fit_sources(tr,va,yt,seed+10001*f,fake=fake)
            friendtable.append(dict(seed=seed,outer_fold=f,inner_fold=-1,stage='outer',
                best_k=src['friend_choice'][0],best_C=src['friend_choice'][1]))
            for g,param in src['history_params'].items():
                priorrows.append(dict(seed=seed,outer_fold=f,history_group=g,**param))
            calculated={'nested_gamma':gamma,'no_history':0.,
                        **{f'fixed_gamma_{g:g}':g for g in fixed_gammas}}
            for name,g in calculated.items():
                preds=probabilities(src,va.poutcome,g)
                allpred[name][vai]=preds
                foldtable.append(dict(seed=seed,outer_fold=f,model=name,subset='all',
                    **metrics(y[vai],preds)))
                is_un=va.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
                for subset,mask in [('unknown',is_un),('nonunknown',~is_un)]:
                    foldtable.append(dict(seed=seed,outer_fold=f,model=name,subset=subset,
                        **metrics(y[vai][mask],preds[mask])))
            fold_for[vai]=f
        if (fold_for<0).any() or any(~np.isfinite(p).all() for p in allpred.values()):
            raise RuntimeError('Incomplete outer OOF predictions')
        for i in range(len(df)):
            oof.append(dict(seed=seed,fold=int(fold_for[i]),customer_id=df.customer_id.iloc[i],
                poutcome=df.poutcome.iloc[i],label=int(y[i]),
                **{f'p_{name}':float(p[i]) for name,p in allpred.items()}))
    predframe=pd.DataFrame(oof)
    byseed=[];top10=[]
    for (seed),d in predframe.groupby('seed'):
        yt=d.label.to_numpy(int)
        unknown=d.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
        for model in labels:
            p=d[f'p_{model}'].to_numpy(float)
            for subset,mask in [('all',np.ones(len(d),bool)),('unknown',unknown),('nonunknown',~unknown)]:
                byseed.append(dict(seed=seed,model=model,subset=subset,**metrics(yt[mask],p[mask])))
            ranked=d.assign(probability=p).sort_values(['probability','customer_id'],ascending=[False,True]).head(10)
            for rank,(_,row) in enumerate(ranked.iterrows(),1):
                top10.append(dict(seed=seed,model=model,rank=rank,customer_id=row.customer_id,
                                  probability=row.probability,label=row.label,poutcome=row.poutcome))
    agg=pd.DataFrame(byseed).groupby(['model','subset'],as_index=False).agg(
        PR_AUC_mean=('PR_AUC','mean'),PR_AUC_sd=('PR_AUC','std'),
        ROC_AUC_mean=('ROC_AUC','mean'),Brier_mean=('Brier','mean'),
        LogLoss_mean=('LogLoss','mean'),hits_mean=('hits_at_10','mean'),
        hits_min=('hits_at_10','min'),hits_max=('hits_at_10','max'))
    files={'nested_oof.csv':predframe,'chosen_gamma_by_outer_fold.csv':pd.DataFrame(choices),
        'inner_gamma_scores.csv':pd.DataFrame(innertable),'by_fold.csv':pd.DataFrame(foldtable),
        'by_seed.csv':pd.DataFrame(byseed),'summary.csv':agg,
        'top10.csv':pd.DataFrame(top10),'friend_refits.csv':pd.DataFrame(friendtable),
        'history_priors.csv':pd.DataFrame(priorrows)}
    for name,data in files.items():data.to_csv(out/name,index=False)
    for subset in ('all','unknown','nonunknown'):
        print('\n'+subset.upper())
        print(agg.loc[agg.subset==subset].sort_values('PR_AUC_mean',ascending=False).to_string(index=False))
    print('\nSaved:',out.resolve())
    return agg

def self_test():
    src={'f2':np.array([.2,.8,.3,.6]),'ring':np.array([.1,.9,.4,.2]),
         'friend':np.array([.2,.5,.5,.7]),'delta':np.array([.2,-.1,0,0])}
    groups=['success','failure','unknown','other']
    p0=probabilities(src,groups,0)
    p1=probabilities(src,groups,.4)
    assert np.isclose(p0[2],(.3+.4)/2)
    assert np.isclose(p0[0],.7*.2+.3*.2)
    assert np.isclose(p0[3],.7*.6+.3*.7)
    assert not np.isclose(p0[0],p1[0])
    assert np.isclose(p0[2],p1[2])
    print('Quick formula self-test OK')
    # Full fake nested test is optional and deliberately separate due to runtime.

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='Stage1_Train.tsv')
    p.add_argument('--outdir',default='N16_fixed_N10_nested_gamma_results')
    p.add_argument('--outer-folds',type=int,default=5)
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,179,316])
    p.add_argument('--gamma-grid',type=float,nargs='+',default=[0,.05,.1,.2,.35,.5])
    p.add_argument('--fixed-gammas',type=float,nargs='+',default=[.2,.35])
    p.add_argument('--selection-metric',choices=['brier','brier_ap','logloss'],default='brier_ap')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--fake',action='store_true',help='Synthetic quantum features for pipeline smoke testing ONLY')
    a=p.parse_args()
    if a.self_test:self_test();return
    for gam in a.gamma_grid+a.fixed_gammas:
        if not 0<=gam<=1:raise ValueError('gamma must be between 0 and 1')
    run(a.train,a.outdir,a.outer_folds,a.inner_folds,a.seeds,a.gamma_grid,
        a.fixed_gammas,a.selection_metric,a.fake)
if __name__=='__main__':cli()
