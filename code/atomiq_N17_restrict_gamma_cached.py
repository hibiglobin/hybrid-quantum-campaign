#!/usr/bin/env python3
"""Re-score N16/N17/FriendHybrid on IDENTICAL outer-fold model fits.

Given the FULL OOF outputs of the previous wide-gamma N17 Nested CV,
remove each previously selected gamma's history correction and apply
fixed gamma = 0, .2, .35, .5 or a constrained inner-CV gamma choice.
This does not retrain any quantum/classical models. It respects outer
train-fold history offsets, and picks constrained gamma solely from
existing inner-fold objective scores.

The three candidate architectures were previously selected/developed using
this same dataset: this is a CONDITIONAL comparison, not an entirely
independent estimate of the entire model development process.

Usage:
python atomiq_N17_restrict_gamma_cached.py \\
  --train Stage1_Train.tsv \\
  --oof 'nested_oof(3).csv' \\
  --choices 'chosen_by_outer_fold(2).csv' \\
  --inner 'inner_grid_scores(2).csv' \\
  --outdir N17_gamma_restricted
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss

CANDIDATES=('N16','N17','N17_FriendHybrid')
EPS=1e-8

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    return np.log(p/(1-p))

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(np.asarray(z,dtype=float),-100,100)))

def history_delta(train,val,pseudo=40.):
    """Exact 40-pseudocount, train-fold only history adjustment from N15."""
    base=train['y'].astype(str).str.strip().str.lower().eq('yes').mean()
    if not (0<base<1): raise ValueError('Training-fold prevalence degenerate')
    out=np.zeros(len(val))
    train_group=train['poutcome'].astype(str).str.lower().to_numpy()
    val_group=val['poutcome'].astype(str).str.lower().to_numpy()
    for group in ('success','failure','other'):
        s=train_group==group
        n=s.sum()
        pos=train.loc[s,'y'].astype(str).str.strip().str.lower().eq('yes').sum()
        post=(pos+pseudo*base)/(n+pseudo)
        delta=logit(post)-logit(base)
        out[val_group==group]=delta
    return out

def evaluate(y,p):
    y=np.asarray(y,dtype=int)
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    top=np.argsort(-p,kind='stable')[:min(10,len(p))]
    hits=int(y[top].sum())
    return {'n':len(y),'positives':int(y.sum()),
            'PR_AUC':float(average_precision_score(y,p)) if y.sum() else float('nan'),
            'ROC_AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'Brier':float(brier_score_loss(y,p)),
            'LogLoss':float(log_loss(y,p,labels=[0,1])),
            'hits_at_10':hits,'precision_at_10':hits/len(top),
            'recall_at_10':hits/y.sum() if y.sum() else float('nan')}

def load_data(train_path,oof_path,choices_path,inner_path):
    train=pd.read_csv(train_path,sep='\t')
    oof=pd.read_csv(oof_path)
    choices=pd.read_csv(choices_path)
    inner=pd.read_csv(inner_path)
    for tab,cols in [(train,('customer_id','poutcome','y')),
                     (oof,('seed','fold','customer_id','poutcome','label',
                           'p_N16','p_N17','p_N17_FriendHybrid')),
                     (choices,('seed','outer_fold','gamma_N16','gamma_N17','gamma_N17_FriendHybrid')),
                     (inner,('seed','outer_fold','candidate','gamma','objective'))]:
        missing=set(cols)-set(tab.columns)
        if missing: raise ValueError(f'Missing columns: {missing}')
    if train.customer_id.duplicated().any() or oof.duplicated(['seed','customer_id']).any():
        raise ValueError('Duplicate customer ID')
    if len(oof)!=len(train)*oof.seed.nunique():
        raise ValueError('OOF must contain every customer once per seed')
    for seed,sd in oof.groupby('seed'):
        if set(sd.customer_id)!=set(train.customer_id):
            raise ValueError(f'Customer IDs mismatch in seed {seed}')
    return train,oof,choices,inner

def run(train_path,oof_path,choices_path,inner_path,outdir,gammas=(0.,.2,.35,.5)):
    train,oof,choices,inner=load_data(train_path,oof_path,choices_path,inner_path)
    gammas=tuple(float(g) for g in gammas)
    if not gammas or len(set(gammas))!=len(gammas):raise ValueError('Duplicate or no gammas')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    results=[];selected=[];winners=[];quality=[]
    for seed,dfseed in oof.groupby('seed',sort=True):
        dfseed=dfseed.copy()
        for fold,group in dfseed.groupby('fold',sort=True):
            subgroup=group.sort_values('customer_id')
            valid=train.set_index('customer_id').loc[subgroup.customer_id].reset_index()
            trainfold=train.loc[~train.customer_id.isin(subgroup.customer_id)]
            delta=history_delta(trainfold,valid,pseudo=40.)
            outside=valid.poutcome.astype(str).str.lower().ne('unknown').to_numpy()
            c=choices[(choices.seed==seed)&(choices.outer_fold==fold)]
            if len(c)!=1:raise ValueError(f'Missing choices for seed={seed}, fold={fold}')
            c=c.iloc[0]
            local=inner[(inner.seed==seed)&(inner.outer_fold==fold)&inner.gamma.isin(gammas)]
            if len(local)!=len(gammas)*len(CANDIDATES):
                raise ValueError(f'Incomplete inner gamma grid for seed={seed}, fold={fold}')
            bests={}
            for model in CANDIDATES:
                g0=float(c['gamma_'+model])
                original=subgroup['p_'+model].to_numpy(float)
                base=original.copy()
                # Recover the original, unadjusted probability on non-unknown.
                base[outside]=sigmoid(logit(original[outside])-g0*delta[outside])
                # Round-trip verifies the inverse exactly (apart from numerical precision).
                rebuilt=base.copy()
                rebuilt[outside]=sigmoid(logit(base[outside])+g0*delta[outside])
                if not np.allclose(rebuilt,original,rtol=0,atol=1e-8):
                    raise AssertionError('Gamma inverse round-trip failed')
                qual=local[local.candidate==model].sort_values(['objective','gamma'],kind='stable')
                best=float(qual.iloc[0].gamma)
                bests[model]=(float(qual.iloc[0].objective),best)
                for gamma in gammas:
                    p=base.copy()
                    p[outside]=sigmoid(logit(base[outside])+gamma*delta[outside])
                    for cid,yy,history,pval in zip(subgroup.customer_id,subgroup.label,subgroup.poutcome,p):
                        results.append({'seed':seed,'fold':fold,'customer_id':cid,
                                        'poutcome':history,'label':int(yy),
                                        'model':model,'setting':f'fixed_gamma_{gamma:g}',
                                        'gamma':gamma,'probability':float(pval)})
                    if abs(gamma-best)<1e-10:
                        bestpred=p.copy()
                selected.append({'seed':seed,'outer_fold':fold,'model':model,'selected_gamma':best,
                                 'inner_objective':bests[model][0],'wide_gamma':g0})
                for cid,yy,history,pval in zip(subgroup.customer_id,subgroup.label,subgroup.poutcome,bestpred):
                    results.append({'seed':seed,'fold':fold,'customer_id':cid,
                                    'poutcome':history,'label':int(yy),
                                    'model':model,'setting':'restricted_nested',
                                    'gamma':best,'probability':float(pval)})
            winner=min(CANDIDATES,key=lambda model:(bests[model][0],CANDIDATES.index(model)))
            winners.append({'seed':seed,'outer_fold':fold,'selected_model':winner,
                            'gamma':bests[winner][1]})
    raw=pd.DataFrame(results)
    # Model-selection version: inner chose both architecture and gamma.
    combo=raw[raw.setting=='restricted_nested'].merge(pd.DataFrame(winners),
                left_on=['seed','fold','model'],right_on=['seed','outer_fold','selected_model'],
                suffixes=('','_selected'))
    combo=combo.drop(columns=['outer_fold','selected_model','gamma_selected'])
    combo['model']='INNER_SELECTED';raw=pd.concat([raw,combo],ignore_index=True)
    if raw.duplicated(['seed','customer_id','model','setting']).any():raise AssertionError('Duplicate scored rows')
    # Save predictions in long format for every model and gamma.
    raw.to_csv(out/'all_gamma_oof.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'restricted_chosen_by_outer_fold.csv',index=False)
    pd.DataFrame(winners).to_csv(out/'restricted_model_selection.csv',index=False)
    metrics=[];tops=[]
    for (seed,model,setting),rows in raw.groupby(['seed','model','setting'],sort=True):
        if len(rows)!=len(train):raise ValueError(f'Missing OOF row for {(seed,model,setting)}')
        rows=rows.sort_values('customer_id')
        history=rows.poutcome.astype(str).str.lower().to_numpy()
        y=rows.label.to_numpy(int);p=rows.probability.to_numpy(float)
        for subset,mask in [('all',np.ones(len(rows),bool)),('unknown',history=='unknown'),
                            ('nonunknown',history!='unknown')]:
            metrics.append({'seed':seed,'model':model,'setting':setting,'subset':subset,
                            **evaluate(y[mask],p[mask])})
        ix=np.argsort(-p,kind='stable')[:10]
        for rank,j in enumerate(ix,1):
            tops.append({'seed':seed,'model':model,'setting':setting,'rank':rank,
                         'customer_id':rows.iloc[j].customer_id,'probability':p[j],
                         'label':int(y[j]),'poutcome':history[j]})
    bm=pd.DataFrame(metrics);bm.to_csv(out/'by_seed.csv',index=False)
    pd.DataFrame(tops).to_csv(out/'top10.csv',index=False)
    summ=bm.groupby(['model','setting','subset'],as_index=False).agg(
        PR_AUC_mean=('PR_AUC','mean'),PR_AUC_sd=('PR_AUC','std'),
        ROC_AUC_mean=('ROC_AUC','mean'),Brier_mean=('Brier','mean'),
        LogLoss_mean=('LogLoss','mean'),hits_mean=('hits_at_10','mean'),
        hits_min=('hits_at_10','min'))
    summ.to_csv(out/'summary.csv',index=False)
    print('\nWHOLE POPULATION')
    print(summ[summ.subset=='all'].sort_values(['model','setting']).to_string(index=False))
    print('\nFiles saved to',out.resolve())
    return summ

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--train',default='Stage1_Train.tsv')
    pa.add_argument('--oof',default='nested_oof(3).csv')
    pa.add_argument('--choices',default='chosen_by_outer_fold(2).csv')
    pa.add_argument('--inner',default='inner_grid_scores(2).csv')
    pa.add_argument('--outdir',default='N17_gamma_restricted')
    pa.add_argument('--gammas',type=float,nargs='+',default=[0.,.2,.35,.5])
    a=pa.parse_args()
    run(a.train,a.oof,a.choices,a.inner,a.outdir,a.gammas)
if __name__=='__main__':main()
