"""Atomiq Stage 1: 15-qubit circuit comparison + classical model search.

Run in the same directory as Stage1_Train.tsv:
    from atomiq_stage1_15q_circuits_and_classical import main
    results = main('Stage1_Train.tsv')

Models evaluated:
  quantum_ring_connected          15 singles + all 105 connected pairs
  quantum_full_connected          same, with all-to-all RZZ entanglers
  quantum_layered_connected       same, with two ring entangling layers
  classical_logistic              one-hot + regularized logistic regression
  classical_numeric_interactions  above + degree-2 numeric interactions
  classical_extra_trees           one-hot + ExtraTreesClassifier
  classical_hist_gradient_boost   one-hot + HistGradientBoostingClassifier

- Outer five-fold stratified CV computes unbiased-for-fixed-procedure OOF results.
- Inside each outer training fold, inner three-fold stratified CV chooses
  regularization/hyperparameters using negative log loss.
- No Stage 1 hidden test labels are read. All quantum statevectors are ideal,
  noiseless *classical simulations*, not a demonstration of quantum advantage.
- Comparing circuit geometries also changes gate count and (intentionally)
  entangling strengths; performance differences are not uniquely attributable
  to topology. Small 300-person development set: selection optimism remains.

Install (same environment as previous scripts):
  pip install numpy pandas scikit-learn qiskit
"""
from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

SEED = 42
OUTER_FOLDS, INNER_FOLDS = 5, 3
ORDER = ['age', 'balance', 'campaign', 'poutcome', 'day', 'pdays',
         'previous', 'housing', 'job', 'marital', 'education', 'default',
         'loan', 'contact', 'month']
NUMERIC = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CATEGORICAL = [c for c in ORDER if c not in NUMERIC]
NQ = 15
PAIRS = list(itertools.combinations(range(NQ), 2))
CIRCUITS = ('ring', 'full', 'layered')
Q_NAMES = ['quantum_' + c + '_connected' for c in CIRCUITS]
CLASSICAL_NAMES = ['classical_logistic', 'classical_numeric_interactions',
                   'classical_extra_trees', 'classical_hist_gradient_boost']
ALL_NAMES = CLASSICAL_NAMES + Q_NAMES
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def numeric_transform(values, name):
    x = np.asarray(values, dtype=float)
    if name == 'balance':
        return np.sign(x) * np.log1p(np.abs(x))
    if name in ('campaign', 'previous'):
        return np.log1p(np.maximum(x, 0))
    if name == 'pdays':
        return np.where(x < 0, -1.0, np.log1p(np.maximum(x, 0)))
    return x


def stable_nominal_angle(column, value):
    if pd.isna(value):
        value = 'unknown'
    h = hashlib.sha256((column + ':' + str(value)).encode('utf-8')).digest()
    r = int.from_bytes(h[:8], 'big') / (2**64)
    return 1.8 * r - 0.9


class FoldAngleEncoder:
    def __init__(self, columns):
        self.columns = list(columns)
        self.scalers = {}

    def fit(self, frame):
        for name in self.columns:
            if name in NUMERIC:
                v = numeric_transform(frame[name].to_numpy(), name)
                finite = v[np.isfinite(v)]
                fill = float(np.median(finite)) if len(finite) else 0.0
                v = np.where(np.isfinite(v), v, fill)
                self.scalers[name] = (StandardScaler().fit(v.reshape(-1, 1)), fill)
        return self

    def transform(self, frame):
        x = np.zeros((len(frame), len(self.columns)), dtype=float)
        for i, name in enumerate(self.columns):
            if name in NUMERIC:
                v = numeric_transform(frame[name].to_numpy(), name)
                scaler, fill = self.scalers[name]
                v = np.where(np.isfinite(v), v, fill)
                x[:, i] = np.tanh(scaler.transform(v.reshape(-1, 1))[:, 0])
            elif name == 'poutcome':
                x[:, i] = (frame[name].fillna('unknown').astype(str).to_numpy() == 'success').astype(float)
            elif name == 'housing':
                x[:, i] = (frame[name].fillna('unknown').astype(str).to_numpy() == 'yes').astype(float)
            else:
                x[:, i] = [stable_nominal_angle(name, val) for val in frame[name]]
        return x


def apply_ring(qc, x, factor):
    for i in range(NQ):
        j = (i + 1) % NQ
        qc.rzz(factor * np.pi * float(x[i] * x[j]), i, j)


def make_state(x, topology):
    qc = QuantumCircuit(NQ)
    for i in range(NQ):
        qc.ry(np.pi * float(x[i]), i)
    if topology == 'ring':
        # Exact match to previous 15q A/B/C circuit.
        apply_ring(qc, x, 0.7)
        for i in range(NQ):
            qc.rx(0.5, i)
    elif topology == 'full':
        # All 105 RZZ pairs. Scale angles per edge to avoid enormous total phases.
        for i, j in PAIRS:
            qc.rzz(0.7 / np.sqrt(NQ - 1) * np.pi * float(x[i] * x[j]), i, j)
        for i in range(NQ):
            qc.rx(0.5, i)
    elif topology == 'layered':
        # Rotation between noncommuting RZZ layers allows richer correlations.
        apply_ring(qc, x, 0.7)
        for i in range(NQ):
            qc.rx(0.5, i)
            qc.ry(0.35 * np.pi * float(x[i]), i)
        apply_ring(qc, x, 0.4)
        for i in range(NQ):
            qc.rx(0.3, i)
    else:
        raise ValueError('Unknown topology: ' + topology)
    return Statevector.from_instruction(qc)


def fwht_z_moments(probs):
    m = np.asarray(probs, dtype=float).copy()
    if m.shape != (1 << NQ,):
        raise ValueError('Unexpected statevector size')
    for i in range(NQ):
        bit = 1 << i
        blocks = m.reshape(-1, 2, bit)
        a, b = blocks[:, 0, :].copy(), blocks[:, 1, :].copy()
        blocks[:, 0, :] = a + b
        blocks[:, 1, :] = a - b
    return m


def connected_features(angle_rows, topology):
    out = np.empty((len(angle_rows), NQ + len(PAIRS)), dtype=np.float64)
    for k, row in enumerate(angle_rows):
        sv = make_state(row, topology)
        mom = fwht_z_moments(np.abs(np.asarray(sv.data))**2)
        single = np.array([mom[1 << i] for i in range(NQ)])
        pairs = np.array([mom[(1 << i) | (1 << j)] - single[i] * single[j]
                          for i, j in PAIRS])
        out[k, :NQ] = single
        out[k, NQ:] = pairs
    return out


def preprocessing(include_poly=False):
    # All imputers/scalers/one-hot vocabulary are fitted only within training folds.
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if include_poly:
        num_steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True,
                                                     include_bias=False)))
    num_steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(num_steps), NUMERIC),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ], sparse_threshold=0)


def classical_estimators():
    return {
        'classical_logistic': (
            Pipeline([('prep', preprocessing()),
                      ('clf', LogisticRegression(max_iter=4000, random_state=SEED))]),
            {'clf__C': C_GRID}),
        'classical_numeric_interactions': (
            Pipeline([('prep', preprocessing(include_poly=True)),
                      ('clf', LogisticRegression(max_iter=4000, random_state=SEED))]),
            {'clf__C': C_GRID}),
        'classical_extra_trees': (
            Pipeline([('prep', preprocessing()),
                      ('clf', ExtraTreesClassifier(n_estimators=250, random_state=SEED,
                                                   n_jobs=-1))]),
            {'clf__min_samples_leaf': [2, 5, 10], 'clf__max_features': [0.7, 1.0]}),
        'classical_hist_gradient_boost': (
            Pipeline([('prep', preprocessing()),
                      ('clf', HistGradientBoostingClassifier(random_state=SEED,
                                                           max_iter=100, l2_regularization=1.0))]),
            {'clf__max_leaf_nodes': [3, 7], 'clf__learning_rate': [0.03, 0.1]})
    }


def scores(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ix = np.argsort(-p, kind='stable')[:min(10, len(y))]
    return {'ROC_AUC': roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            'PR_AP': average_precision_score(y, p),
            'Brier': brier_score_loss(y, p),
            'LogLoss': log_loss(y, np.clip(p, 1e-12, 1-1e-12), labels=[0, 1]),
            'Precision_at_10': y[ix].mean(),
            'Recall_at_10': y[ix].sum() / max(int(y.sum()), 1)}


def choose_and_fit(estimator, grid, x, y):
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + 101)
    search = GridSearchCV(estimator, grid, cv=inner, scoring='neg_log_loss',
                          n_jobs=1, refit=True, error_score='raise')
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, -search.best_score_


def main(path='Stage1_Train.tsv'):
    path = Path(path)
    df = pd.read_csv(path, sep='\t')
    missing = set(ORDER + ['customer_id', 'y']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(df) != 300 or not df.customer_id.is_unique:
        raise ValueError('Expected 300 customers with unique customer_id')
    if not set(df.y) <= {'yes', 'no'}:
        raise ValueError('Unexpected y labels')
    y = (df.y == 'yes').to_numpy(dtype=int)
    oof = {name: np.full(len(df), np.nan) for name in ALL_NAMES}
    fold_rows, train_rows, tuning_rows, pair_rows = [], [], [], []
    cv = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(cv.split(df, y), start=1):
        train, valid = df.iloc[tr], df.iloc[va]
        for name, (est, grid) in classical_estimators().items():
            clf, params, inner_loss = choose_and_fit(est, grid, train, y[tr])
            p_tr = clf.predict_proba(train)[:, 1]
            p_va = clf.predict_proba(valid)[:, 1]
            oof[name][va] = p_va
            fold_rows.append({'fold': fold, 'model': name, **scores(y[va], p_va)})
            train_rows.append({'fold': fold, 'model': name, **scores(y[tr], p_tr)})
            tuning_rows.append({'fold': fold, 'model': name,
                                'params': str(params), 'inner_LogLoss': inner_loss})
        print(f'Fold {fold}/{OUTER_FOLDS}: classical models done', flush=True)

        encoder = FoldAngleEncoder(ORDER).fit(train)
        angles = encoder.transform(pd.concat([train, valid]))
        for topology in CIRCUITS:
            name = 'quantum_' + topology + '_connected'
            feats = connected_features(angles, topology)
            ntr = len(tr)
            xtr, xva = feats[:ntr], feats[ntr:]
            clf, params, inner_loss = choose_and_fit(
                Pipeline([('scale', StandardScaler()),
                          ('clf', LogisticRegression(max_iter=4000, random_state=SEED))]),
                {'clf__C': C_GRID}, xtr, y[tr])
            p_tr, p_va = clf.predict_proba(xtr)[:, 1], clf.predict_proba(xva)[:, 1]
            oof[name][va] = p_va
            fold_rows.append({'fold': fold, 'model': name, **scores(y[va], p_va)})
            train_rows.append({'fold': fold, 'model': name, **scores(y[tr], p_tr)})
            tuning_rows.append({'fold': fold, 'model': name,
                                'params': str(params), 'inner_LogLoss': inner_loss})
            abs_connected = np.abs(xtr[:, NQ:])
            for j, (i, k) in enumerate(PAIRS):
                pair_rows.append({'fold': fold, 'circuit': topology,
                                  'feature_i': ORDER[i], 'feature_j': ORDER[k],
                                  'mean_abs_Cij': float(abs_connected[:, j].mean()),
                                  'max_abs_Cij': float(abs_connected[:, j].max()),
                                  'std_Cij': float(xtr[:, NQ + j].std()),
                                  'approximately_zero': bool(abs_connected[:, j].max() < 1e-10)})
            nzero = int((abs_connected.max(axis=0) < 1e-10).sum())
            print(f'Fold {fold}/{OUTER_FOLDS}: {topology} done; '
                  f'near-zero pairs {nzero}/105; C={params["clf__C"]}', flush=True)

    if any(np.isnan(v).any() for v in oof.values()):
        raise RuntimeError('Unfilled OOF predictions')
    metrics = []
    for name in ALL_NAMES:
        s = scores(y, oof[name])
        metrics.append({'model': name,
                        'n_features': 120 if name in Q_NAMES else np.nan,
                        'ROC_AUC': s['ROC_AUC'], 'PR_AP': s['PR_AP'],
                        'Brier': s['Brier'], 'LogLoss': s['LogLoss'],
                        'OOF_Precision_at_10_diagnostic': s['Precision_at_10'],
                        'OOF_Recall_at_10_diagnostic': s['Recall_at_10']})
    metrics = pd.DataFrame(metrics)
    out = path.resolve().parent
    metrics.to_csv(out / 'stage1_15q_circuits_classical_metrics.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'stage1_15q_circuits_classical_folds.csv', index=False)
    pd.DataFrame(train_rows).to_csv(out / 'stage1_15q_circuits_classical_train.csv', index=False)
    pd.DataFrame(tuning_rows).to_csv(out / 'stage1_15q_circuits_classical_tuning.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(out / 'stage1_15q_circuits_classical_pairs.csv', index=False)
    pd.DataFrame({'customer_id': df.customer_id, 'label': y,
                  **{'p_' + name: p for name, p in oof.items()}}).to_csv(
                      out / 'stage1_15q_circuits_classical_oof.csv', index=False)
    print('\n=== POOLED OUTER OOF METRICS (DEVELOPMENT ONLY) ===')
    print(metrics.round(5).to_string(index=False))
    pair_table = pd.DataFrame(pair_rows)
    print('\n=== MEAN NEAR-ZERO CONNECTED PAIRS BY CIRCUIT ===')
    print(pair_table.groupby(['fold', 'circuit']).approximately_zero.sum()
          .groupby('circuit').mean().round(1).to_string())
    print('\n=== MEAN TRAIN VS VALID LOG LOSS (per fold) ===')
    train_loss = pd.DataFrame(train_rows).groupby('model').LogLoss.mean().rename('train')
    val_loss = pd.DataFrame(fold_rows).groupby('model').LogLoss.mean().rename('valid')
    print(pd.concat([train_loss, val_loss], axis=1).round(4).to_string())
    print('\nSaved 6 CSV files in:', out)
    print('Note: all models tune hyperparameters with inner CV; these OOF results'
          ' assess the fixed candidate procedures, but choosing a final winner'
          ' from these results still introduces selection optimism.')
    return metrics


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Stage1_Train.tsv')
