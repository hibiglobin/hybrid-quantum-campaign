"""Final Stage 1 inference: A0 fixed-6q, F2 15q+6q-pairs, friend predictions.

Requires Stage1_Train.tsv, a separate Stage1_Test.tsv containing the *100*
unlabelled customers with all 15 feature columns, stage1_predictions.csv
containing friend predictions, and the existing modules:
    atomiq_stage1_AB_optimization.py
    atomiq_stage1_15q_plus_6q_feature_fusion.py

Notebook:
    from atomiq_stage1_final_A0_F2_friend import main
    main('Stage1_Train.tsv', 'Stage1_Test.tsv', 'stage1_predictions.csv')

Requires: qiskit, numpy, pandas, scikit-learn, scipy, catboost.
Only a prediction file is produced; no held-out labels are available.
No model-selection claims are inferred from the 100 unlabelled rows.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import atomiq_stage1_AB_optimization as base
import atomiq_stage1_15q_plus_6q_feature_fusion as fusion

A0_COLS=('age','balance','campaign','poutcome','day','previous')

def main(train_path='Stage1_Train.tsv', test_path='Stage1_Test.tsv',
         friend_path='stage1_predictions.csv', outdir='.'):
    train=pd.read_csv(train_path,sep='\t')
    test=pd.read_csv(test_path,sep='\t')
    for title,data in [('Train',train),('Test',test)]:
        missing={'customer_id',*base.ORDER}-set(data.columns)
        if missing:raise ValueError(f'{title}: missing columns {sorted(missing)}')
        if data.customer_id.isna().any() or data.customer_id.duplicated().any():
            raise ValueError(f'{title}: customer IDs must be unique/nonmissing')
    if 'y' not in train:raise ValueError('Train needs y labels yes/no')
    y0=train.y.astype(str).str.lower().str.strip()
    if not set(y0.unique())<={'yes','no'}:raise ValueError('Train y must be yes/no')
    y=(y0=='yes').astype(int).to_numpy()
    if set(train.customer_id)&set(test.customer_id):
        raise ValueError('Train/test IDs overlap. Verify your test file!')
    if len(test)!=100:raise ValueError(f'Expected exactly 100 test rows, got {len(test)}')

    print('Computing 15q quantum features...',flush=True)
    q15tr,q15te=fusion.make_quantum_features(train,test,base.ORDER,True)
    print('Computing 6q quantum features (shared with A0 and F2)...',flush=True)
    q6tr,q6te=fusion.make_quantum_features(train,test,A0_COLS,False)

    # F2: all data-dependent transforms and 15q pair screening fit only on 300 labels.
    print('Training F2 on all labelled data...',flush=True)
    pf2, selected=fusion.fit_predict(train,y,test,q15tr,q15te,q6tr,q6te,
                                    variant=2,return_indices=True)

    # A0: matches fixed A0 used in E1--E6 / matched A0+F2 OOF comparisons.
    print('Training A0 on all labelled data...',flush=True)
    a0=Pipeline([
        ('scaler',StandardScaler()),
        ('logreg',LogisticRegression(C=0.01,penalty='l2',solver='lbfgs',
                                   max_iter=2500,random_state=42))
    ])
    a0.fit(q6tr,y)
    pa0=a0.predict_proba(q6te)[:,1]

    friend=pd.read_csv(friend_path)
    if not {'customer_id','predicted_probability'}.issubset(friend.columns):
        raise ValueError('Friend CSV needs customer_id and predicted_probability')
    if friend.customer_id.duplicated().any():raise ValueError('Friend IDs duplicated')
    result=pd.DataFrame({'customer_id':test.customer_id.astype(str),
                         'p_A0':pa0,'p_F2':pf2})
    result=result.merge(friend[['customer_id','predicted_probability']]
                        .rename(columns={'predicted_probability':'p_friend'}),
                        on='customer_id',how='left',validate='one_to_one')
    if result.p_friend.isna().any() or len(friend)!=len(test):
        missing=result.loc[result.p_friend.isna(),'customer_id'].tolist()
        raise ValueError(f'Friend/test mismatch, missing friend IDs: {missing[:10]}')
    for c in ['p_A0','p_F2','p_friend']:
        if not np.isfinite(result[c]).all() or not result[c].between(0,1).all():
            raise ValueError(f'Invalid probability column {c}')

    # Fixed illustrative weights; diagnostic only. Scores remain probabilities.
    result['p_F2_90_A0_10']=.90*result.p_F2+.10*result.p_A0
    result['p_F2_80_A0_20']=.80*result.p_F2+.20*result.p_A0
    result['p_F2_90_friend_10']=.90*result.p_F2+.10*result.p_friend
    result['p_F2_80_A0_10_friend_10']=.80*result.p_F2+.10*result.p_A0+.10*result.p_friend
    probcols=[c for c in result.columns if c.startswith('p_')]
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    result.sort_values('customer_id').to_csv(out/'stage1_final_three_models_all100.csv',index=False)
    rows=[]
    for col in probcols:
        top=result.sort_values([col,'customer_id'],ascending=[False,True]).head(10)
        for rank,(_,r) in enumerate(top.iterrows(),1):
            rows.append({'model':col,'rank':rank,'customer_id':r.customer_id,
                         'predicted_probability':float(r[col])})
    top10=pd.DataFrame(rows)
    top10.to_csv(out/'stage1_final_three_models_top10.csv',index=False)
    print('\nTop 10 by F2:\n',top10[top10.model=='p_F2'].to_string(index=False))
    print('\nSaved predictions and candidate lists. F2 selected 15q pair indices:',selected.tolist())
    return {'all100':result,'top10':top10,'selected_pairs':selected}

if __name__=='__main__':
    import sys
    main(*(sys.argv[1:4]))
