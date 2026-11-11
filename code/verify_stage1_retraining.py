#!/usr/bin/env python3
"""Independent Stage 1 checks, classical baselines, final N17 refit, and nested-CV rerun.

Run from anywhere:
 python verify_stage1_retraining.py /path/to/ATOMIQ_TEAM_COMPLETE_SUBMISSION --mode preflight
 python verify_stage1_retraining.py /path/to/ATOMIQ_TEAM_COMPLETE_SUBMISSION --mode baseline
 python verify_stage1_retraining.py /path/to/ATOMIQ_TEAM_COMPLETE_SUBMISSION --mode prediction
 python verify_stage1_retraining.py /path/to/ATOMIQ_TEAM_COMPLETE_SUBMISSION --mode nested
 python verify_stage1_retraining.py /path/to/ATOMIQ_TEAM_COMPLETE_SUBMISSION --mode all

Output is saved to <root>/stage1_reverification/ and no official submissions are overwritten.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd


def read_csv(path):
    return pd.read_csv(path)


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    from sklearn.metrics import precision_score
    y=np.asarray(y, dtype=int)
    p=np.asarray(p, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Probability data invalid')
    n=min(10, len(y)); indices=np.argsort(-p, kind='stable')[:n]
    return dict(ROC_AUC=float(roc_auc_score(y,p)),PR_AUC=float(average_precision_score(y,p)),
                Brier=float(brier_score_loss(y,p)),LogLoss=float(log_loss(y,np.clip(p,1e-10,1-1e-10))),
                hits_at_10=int(y[indices].sum()),precision_at_10=float(y[indices].mean()))


def make_paths(root, output):
    return dict(root=root, code=root/'code', friend=root/'code'/'friend_atomiq',
                train=root/'data'/'Stage1_Train.tsv', test=root/'data'/'Stage1_Test.tsv',
                submitted=root/'stage1_result.csv', oldcv=root/'results'/'stage1_cv'/'n17_aligned_3fold_summary.csv',
                oldfriend=root/'code'/'friend_atomiq'/'results'/'stage1_cv_metrics.csv',out=output)


def preflight(paths):
    from sklearn.model_selection import StratifiedKFold
    found={k:str(path) for k,path in paths.items() if k not in ('root','code','friend','out')}
    missing=[name for name,path in paths.items() if name not in ('out',) and not path.exists()]
    if missing: raise FileNotFoundError('Required paths absent: '+', '.join(missing))
    train=pd.read_csv(paths['train'],sep='\t');test=pd.read_csv(paths['test'],sep='\t')
    submission=read_csv(paths['submitted'])
    y=train.y.astype(str).str.strip().str.lower().eq('yes').to_numpy(int)
    assert len(train)==300 and len(test)==100 and len(submission)==100
    assert train.customer_id.is_unique and test.customer_id.is_unique and submission.customer_id.is_unique
    assert not set(train.customer_id)&set(test.customer_id)
    assert set(test.customer_id)==set(submission.customer_id)
    assert int(submission.top10_flag.sum())==10
    s=submission.sort_values(['predicted_probability','customer_id'],ascending=[False,True])
    assert list(s.customer_id)==list(submission.sort_values('rank').customer_id)
    assert np.array_equal(submission.sort_values('rank')['rank'].to_numpy(),np.arange(1,101))
    oldcv=pd.read_csv(paths['oldcv']); friendcv=pd.read_csv(paths['oldfriend'])
    assert {'N16','N17','N17_FriendHybrid'} <= set(oldcv.model)
    assert {'lr_compact','hybrid_ensemble','vqc','qk_klr'} <= set(friendcv.model)
    print('PREFLIGHT OK: 300 training, 100 test, 100 submitted; stored CV data exists', flush=True)
    return dict(status='PASS',python=sys.version.split()[0],
                modules={n:bool(importlib.util.find_spec(n)) for n in ['numpy','pandas','sklearn','qiskit','qiskit_aer','qiskit_machine_learning']},
                training_rows=len(train),test_rows=len(test),train_positives=int(y.sum()),
                submitted_top10=list(submission.nsmallest(10,'rank').customer_id),
                stored_aligned_cv_models=sorted(oldcv.model.unique().tolist()),
                stored_teammate_cv_models=sorted(friendcv.model.unique().tolist()))


def do_baseline(paths, seed=42, folds=3):
    from sklearn.model_selection import StratifiedKFold
    import sys
    friend_src=str((paths['friend']/'src').resolve())
    if friend_src not in sys.path: sys.path.insert(0,friend_src)
    code_dir=str(paths['code'].resolve())
    if code_dir not in sys.path:sys.path.insert(0,code_dir)
    from qcampaign.data import FEATURES
    from qcampaign.models.classical import LRCompact, LROneHot, PriorModel
    tr=pd.read_csv(paths['train'],sep='\t')
    y=tr.y.astype(str).str.lower().eq('yes').to_numpy(int)
    X=tr[list(FEATURES)].copy()
    history_unknown=tr.poutcome.astype(str).str.lower().eq('unknown').to_numpy()
    strat=2*y+history_unknown.astype(int)
    if np.bincount(strat,minlength=4).min()<folds: strat=y
    if np.bincount(strat).min()<folds:raise ValueError('Too few labels for chosen folds')
    splitter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    oof={name:np.full(len(tr),np.nan) for name in ('prior','lr_onehot','lr_compact')}
    records=[]
    for fold,(it,iv) in enumerate(splitter.split(tr,strat),start=1):
        print(f'[BASELINE] fold {fold}/{folds} - training on {len(it)} records',flush=True)
        for name,cls in [('prior',PriorModel),('lr_onehot',LROneHot),('lr_compact',LRCompact)]:
            start=time.monotonic()
            model=cls().fit(X.iloc[it].reset_index(drop=True),y[it])
            oof[name][iv]=model.predict_proba(X.iloc[iv].reset_index(drop=True))[:,1]
            records.append(dict(model=name,fold=fold,seconds=round(time.monotonic()-start,3),
                                parameters=json.dumps(getattr(model,'best_params_',{}),default=str)))
    merged=pd.DataFrame(dict(customer_id=tr.customer_id,y=y,**{f'p_{name}':pred for name,pred in oof.items()}))
    merged.to_csv(paths['out']/f'classical_baseline_oof_{folds}fold_seed{seed}.csv',index=False,float_format='%.12g')
    report=[]
    for name,pred in oof.items():
        if not np.isfinite(pred).all():raise RuntimeError(f'Incomplete predictions for {name}')
        report.append(dict(model=name,protocol=f'{folds}-fold x 1 seed={seed}, direct outer CV, internal LR tuning',**metrics(y,pred)))
    pd.DataFrame(report).to_csv(paths['out']/f'classical_baseline_metrics_{folds}fold_seed{seed}.csv',index=False)
    pd.DataFrame(records).to_csv(paths['out']/f'classical_baseline_fit_details_{folds}fold_seed{seed}.csv',index=False)
    print(pd.DataFrame(report).to_string(index=False),flush=True)
    return dict(status='PASS',protocol=f'{folds}-fold seed {seed}',results=report,
                caveat='Not statistically paired to old nested-CV N17; no quantum classifier trained in this mode.')


def run_command(command, cwd, logfile, env=None):
    print('[RUN]', ' '.join(map(str,command)),flush=True)
    print('[LOG]',logfile,flush=True)
    start=time.monotonic()
    with open(logfile,'w',encoding='utf-8') as out:
        out.write('COMMAND: '+' '.join(map(str,command))+'\nCWD: '+str(cwd)+'\n\n')
        out.flush()
        p=subprocess.run(command,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT)
    sec=time.monotonic()-start
    print(f'[DONE] exit={p.returncode}, elapsed={sec:.1f}s',flush=True)
    if p.returncode:
        tail=logfile.read_text(encoding='utf-8',errors='replace').splitlines()[-35:]
        print('\n'.join(tail),file=sys.stderr)
        raise RuntimeError(f'Command failed exit={p.returncode}; inspect {logfile}')
    return sec


def do_prediction(paths, tol=1e-5):
    out=paths['out']/'full_refit';out.mkdir(exist_ok=True,parents=True)
    cmd=[sys.executable,'-u',str(paths['code']/'atomiq_N17_final_test_top10.py'),
         '--train',str(paths['train']),'--test',str(paths['test']),
         '--friend-repo',str(paths['friend']),'--gamma','0.2','--seed','42','--outdir',str(out)]
    elapsed=run_command(cmd,paths['root'],paths['out']/'full_refit.log')
    predicted=pd.read_csv(out/'stage1_N17_FriendHybrid.csv')
    submitted=pd.read_csv(paths['submitted'])
    joined=predicted[['customer_id','predicted_probability']].merge(
        submitted[['customer_id','predicted_probability']],on='customer_id',validate='one_to_one',suffixes=('_rerun','_submitted'))
    assert len(joined)==100
    joined['absolute_error']=(joined.predicted_probability_rerun-joined.predicted_probability_submitted).abs()
    joined=joined.sort_values('absolute_error',ascending=False)
    joined.to_csv(paths['out']/'prediction_difference_by_customer.csv',index=False,float_format='%.12g')
    observed_max=float(joined.absolute_error.max())
    sorted_expected=list(submitted.nsmallest(10,'rank').customer_id)
    sorted_actual=list(predicted.nsmallest(10,'rank').customer_id)
    report={'status':'PASS' if observed_max<=tol and sorted_expected==sorted_actual else 'FAIL',
            'max_abs_probability_difference':observed_max,'tolerance':tol,
            'top10_exact_match':sorted_expected==sorted_actual,
            'expected_top10':sorted_expected,'retrained_top10':sorted_actual,
            'seconds':elapsed}
    print('[PREDICTION]',json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def do_nested(paths, outer=3,inner=2,seed=42):
    out=paths['out']/f'nested_{outer}x{inner}_seed{seed}';out.mkdir(exist_ok=True,parents=True)
    cmd=[sys.executable,'-u',str(paths['code']/'atomiq_N17_same_nested_three_models.py'),
         '--train',str(paths['train']),'--friend-repo',str(paths['friend']),
         '--outdir',str(out),'--outer-folds',str(outer),'--inner-folds',str(inner),
         '--seeds',str(seed),'--objective','brier_ap']
    elapsed=run_command(cmd,paths['root'],paths['out']/'nested_cv.log')
    summary=pd.read_csv(out/'summary.csv')
    required={'N16','N17','N17_FriendHybrid','INNER_SELECTED'}
    assert required<=set(summary.model)
    relevant=summary[summary.subset=='all'].sort_values('PR_AUC_mean',ascending=False)
    relevant.to_csv(paths['out']/'nested_cv_comparison.csv',index=False)
    previous=pd.read_csv(paths['oldcv'])
    old=previous[previous.subset=='all'].set_index('model')
    new=relevant.set_index('model')
    comparison=[]
    for name in sorted(set(old.index)&set(new.index)):
        comparison.append({'model':name,'saved_PR_AUC':float(old.loc[name,'PR_AUC_mean']),
                           'rerun_PR_AUC':float(new.loc[name,'PR_AUC_mean']),
                           'difference':float(new.loc[name,'PR_AUC_mean']-old.loc[name,'PR_AUC_mean'])})
    pd.DataFrame(comparison).to_csv(paths['out']/'nested_vs_saved_reference.csv',index=False)
    # Baseline saved run's inner-fold count cannot be inferred from summary alone.
    # Therefore do NOT call a discrepancy a reproduction failure unless protocols are proven identical.
    return {'status':'PASS','seconds':elapsed,'protocol':f'outer={outer}, inner={inner}, seed={seed}',
            'new_results':relevant.to_dict(orient='records'),
            'saved_reference_comparison':comparison,
            "IMPORTANT":"Saved N17 CV appears from a prior aligned gamma procedure; score equality is NOT guaranteed by this script's protocol."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission_folder',type=Path)
    parser.add_argument('--mode',choices=['preflight','baseline','prediction','nested','all'],default='preflight')
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--tol',type=float,default=1e-5)
    parser.add_argument('--outer-folds',type=int,default=3)
    parser.add_argument('--inner-folds',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    root=args.submission_folder.expanduser().resolve()
    out=(args.out.expanduser().resolve() if args.out else root/'stage1_reverification')
    out.mkdir(parents=True,exist_ok=True)
    paths=make_paths(root,out)
    report={'submission_folder':str(root),'requested_mode':args.mode,
            'timestamp_utc':pd.Timestamp.now(tz='UTC').isoformat(),'checks':{},'failures':{}}
    tasks=[('preflight',lambda:preflight(paths))]
    if args.mode in ('baseline','all'):tasks.append(('baseline',lambda:do_baseline(paths,seed=args.seed,folds=args.outer_folds)))
    if args.mode in ('prediction','all'):tasks.append(('prediction',lambda:do_prediction(paths,tol=args.tol)))
    if args.mode in ('nested','all'):tasks.append(('nested',lambda:do_nested(paths,args.outer_folds,args.inner_folds,args.seed)))
    for label,run in tasks:
        try:
            report['checks'][label]=run()
            if report['checks'][label].get('status')=='FAIL':
                report['failures'][label]='Numerical verification did not match; inspect CSV.'
        except Exception as exc:
            traceback.print_exc()
            report['failures'][label]=str(exc)
            if label=='preflight':break
    target=out/'STAGE1_VERIFICATION_REPORT.json'
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    print('\nREPORT:',target,flush=True)
    print('OVERALL:', 'FAIL' if report['failures'] else 'PASS',flush=True)
    return int(bool(report['failures']))

if __name__=='__main__':raise SystemExit(main())
