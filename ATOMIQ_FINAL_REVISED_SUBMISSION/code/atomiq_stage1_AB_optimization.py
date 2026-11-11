"""Atomiq Stage 1: experiment A (6q tuning) + B (15q hybrid feature selection).

Notebook:
  from atomiq_stage1_AB_optimization import main
  results = main("Stage1_Train.tsv")

Requires numpy pandas scipy scikit-learn qiskit catboost.
The 100-person hidden test set is not used.
"""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier

import hashlib

ORDER = ['age','balance','campaign','poutcome','day','pdays','previous',
         'housing','job','marital','education','default','loan','contact','month']
NUMERIC = ['age','balance','day','campaign','pdays','previous']
CATEGORICAL = [c for c in ORDER if c not in NUMERIC]

def numeric_transform(values, name):
    x = np.asarray(values, dtype=float)
    if name=='balance':return np.sign(x)*np.log1p(np.abs(x))
    if name in ('campaign','previous'):return np.log1p(np.maximum(x,0))
    if name=='pdays':return np.where(x<0,-1,np.log1p(np.maximum(x,0)))
    return x

def nominal_angle(col, val):
    if pd.isna(val):val='unknown'
    h=hashlib.sha256((col+':'+str(val)).encode()).digest()
    return 1.8 * int.from_bytes(h[:8],'big') / 2**64 - 0.9

class FoldAngleEncoder:
    def __init__(self, columns):
        self.columns=list(columns); self.params={}
    def fit(self, frame):
        for col in self.columns:
            if col in NUMERIC:
                v=numeric_transform(frame[col].to_numpy(),col)
                finite=v[np.isfinite(v)]
                med=float(np.median(finite)) if len(finite) else 0.
                v=np.where(np.isfinite(v),v,med)
                self.params[col]=(StandardScaler().fit(v.reshape(-1,1)),med)
        return self
    def transform(self,frame):
        out=np.zeros((len(frame),len(self.columns)),dtype=float)
        for j,col in enumerate(self.columns):
            if col in NUMERIC:
                v=numeric_transform(frame[col].to_numpy(),col)
                scaler,med=self.params[col]
                v=np.where(np.isfinite(v),v,med)
                out[:,j]=np.tanh(scaler.transform(v.reshape(-1,1))[:,0])
            elif col=='poutcome':
                out[:,j]=(frame[col].fillna('unknown').astype(str).to_numpy()=='success').astype(float)
            elif col=='housing':
                out[:,j]=(frame[col].fillna('unknown').astype(str).to_numpy()=='yes').astype(float)
            else:
                out[:,j]=[nominal_angle(col,v) for v in frame[col]]
        return out


SEED = 42
BINARY = ('poutcome','housing')
C_VALUES = (0.001, 0.01, 0.1, 1.0)
# Small, deliberately predeclared attribute alternatives; no label-based exhaustive search.
FEATURE_SETS = {
  5: (
    ('age','balance','campaign','poutcome','day'),
    ('age','balance','campaign','poutcome','previous'),
    ('age','balance','campaign','poutcome','housing'),
  ),
  6: (
    ('age','balance','campaign','poutcome','day','pdays'),
    ('age','balance','campaign','poutcome','previous','housing'),
    ('age','balance','campaign','poutcome','day','previous'),
  ),
  15: (tuple(ORDER),),
}


def score(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    y = np.asarray(y, dtype=int)
    r = {'ROC_AUC':float(roc_auc_score(y,p)),
         'PR_AP':float(average_precision_score(y,p)),
         'Brier':float(brier_score_loss(y,p)),
         'LogLoss':float(log_loss(y,p, labels=[0,1]))}
    ix=np.argsort(-p,kind='stable')[:min(10,len(y))]
    r['Precision_at_10']=float(y[ix].mean())
    return r


def cat_data(frame):
    df=frame.loc[:,ORDER].copy()
    for c in CATEGORICAL:
        df[c]=df[c].astype('string').fillna('__missing__').astype(str)
    for c in NUMERIC:
        df[c]=pd.to_numeric(df[c],errors='coerce').astype(float)
    return df


def cat_model():
    return CatBoostClassifier(iterations=180,depth=3,learning_rate=0.04,
       l2_leaf_reg=5,loss_function='Logloss',verbose=False,
       cat_features=CATEGORICAL,random_seed=SEED,thread_count=2,
       allow_writing_files=False)


def classical_preprocessor():
    return ColumnTransformer([
       ('n',Pipeline([('imp',SimpleImputer(strategy='median')),
                      ('scale',StandardScaler())]),NUMERIC),
       ('c',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                      ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL),
       ],sparse_threshold=0)


def get_quantum_features(angles, columns, topology, shift, strength):
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as e:
        raise ImportError('Install Qiskit in your notebook kernel: pip install qiskit') from e
    n=len(columns)
    edges = ([(i,(i+1)%n) for i in range(n)] if topology == 'ring'
             else list(itertools.combinations(range(n),2)))
    if topology not in ('ring','full'):
        raise ValueError(topology)
    factor = strength if topology=='ring' else strength / math.sqrt(n-1)
    x=np.asarray(angles,dtype=float).copy()
    if shift:
        for c in BINARY:
            if c in columns:
                j=columns.index(c)
                x[:,j]=0.25+0.5*x[:,j]
    pairs=list(itertools.combinations(range(n),2))
    out=np.empty((len(x),n+len(pairs)),dtype=float)
    for row_i,row in enumerate(x):
        qc=QuantumCircuit(n)
        for i in range(n): qc.ry(np.pi*float(row[i]),i)
        for i,j in edges: qc.rzz(factor*np.pi*float(row[i]*row[j]),i,j)
        for i in range(n): qc.rx(0.5,i)
        state=Statevector.from_instruction(qc)
        m=np.abs(np.asarray(state.data))**2
        for bit_idx in range(n):
            bit=1<<bit_idx
            blocks=m.reshape(-1,2,bit)
            a=blocks[:,0,:].copy(); b=blocks[:,1,:].copy()
            blocks[:,0,:]=a+b
            blocks[:,1,:]=a-b
        singles=np.array([m[1<<i] for i in range(n)])
        out[row_i,:n]=singles
        out[row_i,n:]=[m[(1<<i)|(1<<j)]-singles[i]*singles[j] for i,j in pairs]
    return out


def inner_indices(y):
    return list(StratifiedKFold(3,shuffle=True,random_state=SEED+101).split(np.zeros(len(y)),y))


def logistic_oof_and_fit(xtrain,ytrain,xval,cv, cs=C_VALUES):
    """Select C with inner CV; return train-inner-OOF, outer prediction, selected C."""
    best_loss=np.inf
    best_c=None
    best_oof=None
    for c in cs:
        inner_oof=np.full(len(ytrain),np.nan)
        for tr,va in cv:
            pipe=Pipeline([('s',StandardScaler()),('m',LogisticRegression(
                  C=c,max_iter=3000,random_state=SEED))])
            pipe.fit(xtrain[tr],ytrain[tr]); inner_oof[va]=pipe.predict_proba(xtrain[va])[:,1]
        loss=log_loss(ytrain,np.clip(inner_oof,1e-12,1-1e-12),labels=[0,1])
        if loss<best_loss:
            best_loss=loss;best_c=c;best_oof=inner_oof
    final=Pipeline([('s',StandardScaler()),('m',LogisticRegression(
          C=best_c,max_iter=3000,random_state=SEED))])
    final.fit(xtrain,ytrain)
    pval=final.predict_proba(xval)[:,1]
    ptr=final.predict_proba(xtrain)[:,1]
    return best_loss,best_c,best_oof,pval,ptr


def cat_oof(train_df,ytrain,val_df,cv):
    oof=np.full(len(ytrain),np.nan)
    for tr,va in cv:
        model=cat_model();model.fit(train_df.iloc[tr],ytrain[tr])
        oof[va]=model.predict_proba(train_df.iloc[va])[:,1]
    model=cat_model();model.fit(train_df,ytrain)
    return oof,model.predict_proba(val_df)[:,1],model.predict_proba(train_df)[:,1]


def fit_ensemble(oof_matrix,y,outer_matrix):
    """Inner-OOF weight tuning; small fixed simplex and regularized shrinkage."""
    n=oof_matrix.shape[1]
    choices=[]
    for denom in (4,):
        for pieces in itertools.product(range(denom+1),repeat=n-1):
            last=denom-sum(pieces)
            if last>=0: choices.append(np.array((*pieces,last),dtype=float)/denom)
    # include equal weights
    choices.append(np.full(n,1/n))
    best=(np.inf,None)
    for w in choices:
        # shrink OOF weights towards equally weighted ensemble to curb variance
        ww=0.8*w+0.2*np.full(n,1/n)
        p=np.clip(oof_matrix@ww,1e-10,1-1e-10)
        loss=log_loss(y,p,labels=[0,1])
        if loss<best[0]:best=(loss,ww)
    return outer_matrix@best[1],best[1],best[0]




# --- A/B follow-up experiment -----------------------------------------------
# A: 6q attribute subsets, RZZ strength, topology, binary RY encoding.
# B: 15q + original classical features, with supervised selection of quantum
#    connected correlations fitted WITHIN every inner training split.
A_SETS = (
    ('age','balance','campaign','poutcome','day','pdays'),
    ('age','balance','campaign','poutcome','day','previous'),
    ('age','balance','campaign','poutcome','previous','housing'),
    ('age','balance','campaign','poutcome','day','contact'),
    ('age','balance','campaign','poutcome','day','month'),
)


def build_b_features(classic, quantum, yfit, k):
    """Retain all classical + 15 single-Z; select top-k of 105 connected Cij.

    yfit must be the labels of ONLY the split used to train the classifier.
    Use deterministic point-biserial screening for the connected correlations.
    """
    n_singles = 15
    qsingle = quantum[:, :n_singles]
    cor = quantum[:, n_singles:]
    if k >= cor.shape[1]:
        chosen = np.arange(cor.shape[1])
    elif k == 0:
        chosen = np.array([], dtype=int)
    else:
        y0 = np.asarray(yfit, dtype=float) - np.mean(yfit)
        xc = cor - np.mean(cor, axis=0)
        den = np.sqrt((xc*xc).sum(axis=0) * (y0*y0).sum())
        with np.errstate(divide='ignore', invalid='ignore'):
            associations = np.divide(abs(xc.T@y0), den, out=np.zeros(cor.shape[1]), where=den>1e-12)
        chosen = np.sort(np.argsort(-associations, kind='stable')[:k])
    return np.column_stack((classic, qsingle, cor[:, chosen])), chosen


def make_regularized_model(penalty, c):
    if penalty == 'elasticnet':
        clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=c,
                                 solver='saga', max_iter=4000, tol=1e-3,
                                 random_state=SEED)
    else:
        clf = LogisticRegression(penalty=penalty, C=c, solver='liblinear',
                                 max_iter=4000, random_state=SEED)
    return Pipeline([('scale',StandardScaler()), ('clf',clf)])


def fit_b_candidate(df_train, ytrain, qtrain, train_indices, valid_indices, k, penalty, c):
    """Inner split: ALL classical preprocessing and Cij screening fit on train_indices."""
    subtr=df_train.iloc[train_indices]
    subva=df_train.iloc[valid_indices]
    pre=classical_preprocessor().fit(subtr.loc[:,ORDER])
    ctr=pre.transform(subtr.loc[:,ORDER]); cva=pre.transform(subva.loc[:,ORDER])
    xtr, selected=build_b_features(ctr,qtrain[train_indices],ytrain[train_indices],k)
    xva=np.column_stack((cva,qtrain[valid_indices,:15],qtrain[valid_indices,15:][:,selected]))
    model=make_regularized_model(penalty,c)
    model.fit(xtr,ytrain[train_indices])
    return model.predict_proba(xva)[:,1]


def tune_b(df_train, ytrain, qtrain, df_valid, qvalid, cv, exhaustive=False):
    ks=(0,10,25,50,105) if not exhaustive else (0,5,10,15,25,35,50,75,105)
    regularization=[('l2',c) for c in (0.001,0.01,0.1)]
    regularization += [('l1',c) for c in (0.01,0.1,1.0)]
    regularization += [('elasticnet',c) for c in (0.01,0.1,1.0)]
    rows=[];best=None
    for k in ks:
        for penalty,c in regularization:
            pred=np.empty(len(ytrain))
            for it,iv in cv:
                pred[iv]=fit_b_candidate(df_train,ytrain,qtrain,it,iv,k,penalty,c)
            loss=log_loss(ytrain,np.clip(pred,1e-12,1-1e-12),labels=[0,1])
            rows.append(dict(k=k,penalty=penalty,C=c,inner_LogLoss=loss))
            if best is None or loss<best['inner_LogLoss']:
                best=dict(k=k,penalty=penalty,C=c,inner_LogLoss=loss)
    pre=classical_preprocessor().fit(df_train.loc[:,ORDER])
    ctr=pre.transform(df_train.loc[:,ORDER]);cva=pre.transform(df_valid.loc[:,ORDER])
    xtr,chosen=build_b_features(ctr,qtrain,ytrain,best['k'])
    xva=np.column_stack((cva,qvalid[:,:15],qvalid[:,15:][:,chosen]))
    model=make_regularized_model(best['penalty'],best['C'])
    model.fit(xtr,ytrain)
    best['n_final_features']=xtr.shape[1]
    best['selected_pairs']= ';'.join(f'{ORDER[i]}:{ORDER[j]}'
         for p,(i,j) in enumerate(itertools.combinations(range(15),2)) if p in set(chosen))
    best['train_LogLoss']=score(ytrain,model.predict_proba(xtr)[:,1])['LogLoss']
    return best, model.predict_proba(xva)[:,1], rows


def main(path='Stage1_Train.tsv', thorough=False):
    source=pd.read_csv(path,sep='\t')
    missing=set(ORDER+['customer_id','y'])-set(source.columns)
    if missing: raise ValueError(f'Missing columns: {missing}')
    if not set(source.y.astype(str).str.lower().unique()) <= {'yes','no'}:
        raise ValueError('y must contain yes/no labels')
    if source.customer_id.duplicated().any(): raise ValueError('Duplicate customer_id')
    y=(source.y.astype(str).str.lower()=='yes').to_numpy(dtype=int)
    if y.sum()<5 or (1-y).sum()<5:raise ValueError('At least 5 cases of each label needed')
    names=['A_6q_tuned','B_15q_fusion_selected','classical_CatBoost']
    oof={name:np.full(len(y),np.nan) for name in names}
    folds=[]; selections=[];search=[];training=[]
    strengths=(0.4,0.7,1.0) if not thorough else (0.2,0.4,0.7,1.0,1.4)
    sets=A_SETS if thorough else A_SETS[:4]
    configs=[(cols,top,shift,lam) for cols in sets
          for top in ('ring','full') for shift in (False,True) for lam in strengths]
    print('A: 6q candidates per outer Fold:',len(configs),flush=True)
    print('B: 15q full with shifted binary encoding, lambda=.7;'
          ' 5-fold outer / 3-fold inner, LogLoss tuning.',flush=True)
    outer=StratifiedKFold(5,shuffle=True,random_state=SEED)
    for fold,(it,iv) in enumerate(outer.split(source,y),start=1):
        print(f'Fold {fold}/5: A (6q) candidate search...',flush=True)
        tr=source.iloc[it].reset_index(drop=True)
        va=source.iloc[iv].reset_index(drop=True)
        yt=y[it]
        cv=inner_indices(yt)
        angle_cache={}
        best_a=None
        for candidate_index,(cols,top,shift,lam) in enumerate(configs,1):
            cols=tuple(cols)
            if cols not in angle_cache:
                enc=FoldAngleEncoder(cols).fit(tr)
                angle_cache[cols]=(enc.transform(tr),enc.transform(va))
            xa,xb=angle_cache[cols]
            qm=get_quantum_features(np.vstack([xa,xb]),list(cols),top,shift,lam)
            qtr,qva=qm[:len(it)],qm[len(it):]
            loss,c,ioof,pp,p_train=logistic_oof_and_fit(qtr,yt,qva,cv)
            search.append(dict(fold=fold,experiment='A',columns=';'.join(cols),
                               topology=top,shift_binary=shift,strength=lam,C=c,
                               inner_LogLoss=loss))
            if best_a is None or loss<best_a['inner_LogLoss']:
                best_a=dict(fold=fold,columns=';'.join(cols),topology=top,
                            shift_binary=shift,strength=lam,C=c,inner_LogLoss=loss,
                            outer_pred=pp,train_pred=p_train)
            if candidate_index%12==0:
                print(f'  A candidate {candidate_index}/{len(configs)}',flush=True)
        oof['A_6q_tuned'][iv]=best_a.pop('outer_pred')
        train_a=best_a.pop('train_pred')
        training.append(dict(fold=fold,model='A_6q_tuned',
                             train_LogLoss=score(yt,train_a)['LogLoss']))
        selections.append(dict(experiment='A',**best_a))
        print('  A best:',best_a['columns'],best_a['topology'],
              'shift=',best_a['shift_binary'],'lambda=',best_a['strength'],flush=True)

        # B: fixed 15q quantum feature map, all classical attributes; tune
        # supervised correlation screening and classifier inside inner CV.
        print(f'Fold {fold}/5: B (15q) quantum features...',flush=True)
        enc15=FoldAngleEncoder(ORDER).fit(tr)
        qa=enc15.transform(tr); qb=enc15.transform(va)
        qm15=get_quantum_features(np.vstack([qa,qb]),ORDER,'full',True,0.7)
        qtr15,qva15=qm15[:len(it)],qm15[len(it):]
        print(f'Fold {fold}/5: B inner feature-selection / regularization search...',flush=True)
        best_b,pb,rows=tune_b(tr,yt,qtr15,va,qva15,cv,exhaustive=thorough)
        oof['B_15q_fusion_selected'][iv]=pb
        for r in rows:search.append(dict(fold=fold,experiment='B',**r))
        selections.append(dict(fold=fold,experiment='B',**best_b))
        training.append(dict(fold=fold,model='B_15q_fusion_selected',
                             train_LogLoss=best_b['train_LogLoss']))
        print(f'  B best: k={best_b["k"]} penalty={best_b["penalty"]} '
              f'C={best_b["C"]} innerLL={best_b["inner_LogLoss"]:.4f}',flush=True)

        # Classical reference evaluated on the identical outer split.
        cat_tr=cat_data(tr);cat_va=cat_data(va)
        cat=cat_model();cat.fit(cat_tr,yt)
        oof['classical_CatBoost'][iv]=cat.predict_proba(cat_va)[:,1]
        training.append(dict(fold=fold,model='classical_CatBoost',
            train_LogLoss=score(yt,cat.predict_proba(cat_tr)[:,1])['LogLoss']))
        for name in names:
            folds.append(dict(fold=fold,model=name,**score(y[iv],oof[name][iv])))
        print(f'Fold {fold}/5 complete.',flush=True)

    metrics=pd.DataFrame([dict(model=name,**score(y,p)) for name,p in oof.items()])
    oo=pd.DataFrame({'customer_id':source.customer_id,'label':y,
                      **{'p_'+name:p for name,p in oof.items()}})
    outdir=Path(path).resolve().parent
    for tag,df in [('metrics',metrics),('folds',pd.DataFrame(folds)),
                   ('oof',oo),('selected',pd.DataFrame(selections)),
                   ('search',pd.DataFrame(search)),('training',pd.DataFrame(training))]:
        df.to_csv(outdir/f'stage1_AB_{tag}.csv',index=False)
    print('\n=== A/B OUTER OOF RESULTS (development only) ===')
    print(metrics.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('Saved 6 CSV files to',outdir)
    print('Warning: multiple previous experiments used this development set; '
          'the outer OOF is NOT an untouched final test of the overall search process.')
    return metrics


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'Stage1_Train.tsv',
         thorough='--thorough' in sys.argv)
