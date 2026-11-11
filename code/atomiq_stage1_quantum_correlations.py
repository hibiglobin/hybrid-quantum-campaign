"""Atomiq Stage 1: leakage-safe out-of-fold comparison of classical and Pauli-Z features.
Usage: python atomiq_stage1_quantum_correlations.py TRAIN_TSV
Training data ONLY. Never tune using the Stage1 test file.
"""
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SEED=42
NUMERIC=['age','balance','day','campaign','pdays','previous']
CATEGORICAL=['job','marital','education','default','housing','loan','contact','month','poutcome']
QNUM=['age','balance','campaign']

def signed_log(x):
    return np.sign(x)*np.log1p(np.abs(x))

# q0, q1, q2, q3  -> age, balance, campaign, previous campaign succeeded
# Qiskit Pauli labels are reversed (leftmost label acts on q3).
strings=[]
for weight in range(1,5):
    for indices in combinations(range(4),weight):
        chars=['I']*4
        for i in indices:
            chars[3-i]='Z'
        strings.append((''.join(chars),weight))
PAULIS=[Pauli(x[0]) for x in strings]

def make_angles(frame, scaler=None, fit=False):
    raw=frame[QNUM].to_numpy(dtype=float).copy()
    raw[:,1]=signed_log(raw[:,1])
    raw[:,2]=np.log1p(np.clip(raw[:,2],0,None))
    if fit:
        scaler=StandardScaler().fit(raw)
    normalized=np.tanh(scaler.transform(raw))
    flag=(frame['poutcome'].astype(str).to_numpy()=='success').astype(float)
    angles=np.column_stack((normalized,flag))
    return angles,scaler

def quantum_features(angle_row):
    qc=QuantumCircuit(4)
    for i,z in enumerate(angle_row):
        qc.ry(np.pi*z,i)
    # Data-dependent ZZ couplings: state may acquire nontrivial entangling phases.
    for i,j in [(0,1),(1,2),(2,3),(0,3)]:
        qc.rzz(0.7*np.pi*angle_row[i]*angle_row[j],i,j)
    for i in range(4):
        qc.rx(0.5,i)
    state=Statevector.from_instruction(qc)
    return np.array([state.expectation_value(p).real for p in PAULIS])

def classical_pipeline():
    pre=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)
    ])
    return Pipeline([('preprocess',pre),('model',LogisticRegression(max_iter=2000,C=1.0))])

def quantum_pipeline():
    return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,C=1.0))])

def main(path):
    df=pd.read_csv(path,sep='\t')
    assert len(df)==300, f'Expected 300 training rows, got {len(df)}'
    assert df['customer_id'].is_unique
    assert set(df['y'])=={'no','yes'}
    y=(df['y']=='yes').astype(int).to_numpy()
    print(f'Train: {len(df)} customers, yes={sum(y)} ({np.mean(y):.1%})')
    for name in NUMERIC+CATEGORICAL:
        assert name in df.columns, name
    pred={name:np.zeros(len(df)) for name in ['classical_all','Q1','Q1+Q2','Q1+Q2+Q3','Q1+Q2+Q3+Q4']}
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for fold,(tr,va) in enumerate(splitter.split(df,y),1):
        train,valid=df.iloc[tr],df.iloc[va]
        model=classical_pipeline()
        model.fit(train,y[tr])
        pred['classical_all'][va]=model.predict_proba(valid)[:,1]
        atr,scaler=make_angles(train,fit=True)
        ava,_=make_angles(valid,scaler=scaler)
        ftr=np.array([quantum_features(row) for row in atr]);fva=np.array([quantum_features(row) for row in ava])
        for name,n in [('Q1',4),('Q1+Q2',10),('Q1+Q2+Q3',14),('Q1+Q2+Q3+Q4',15)]:
            qmodel=quantum_pipeline()
            qmodel.fit(ftr[:,:n],y[tr])
            pred[name][va]=qmodel.predict_proba(fva[:,:n])[:,1]
        print(f'Fold {fold}/5 completed')
    out=[]
    for name,p in pred.items():
        # OOF@10 shown only as descriptive diagnostics; it is not a fair held-out top-10 metric.
        out.append({'model':name,'ROC_AUC':roc_auc_score(y,p),
                    'PR_AP':average_precision_score(y,p),
                    'Brier':brier_score_loss(y,p),
                    'LogLoss':log_loss(y,np.clip(p,1e-12,1-1e-12))})
    result=pd.DataFrame(out)
    stem=Path(path).with_suffix('')
    result.to_csv(stem.parent/'stage1_cv_metrics.csv',index=False)
    pd.DataFrame({'customer_id':df['customer_id'],'label':y,**{f'p_{k}':v for k,v in pred.items()}}).to_csv(stem.parent/'stage1_oof_predictions.csv',index=False)
    print('\nOut-of-fold metrics (training data only):')
    print(result.round(4).to_string(index=False))
    print('\nFeatures:',[(s,n) for s,n in strings])
    print('Saved: stage1_cv_metrics.csv, stage1_oof_predictions.csv')

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python atomiq_stage1_quantum_correlations.py <Stage1_Train.tsv>')
    main(sys.argv[1])
