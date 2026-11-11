#!/usr/bin/env python3
"""Atomiq Stage 1: 14-qubit, poutcome-excluded specialists (N1/N2/N3).

Usage:
  python atomiq_unknown_14q_N1_N3.py --train Stage1_Train.tsv --seeds 42 179 316 --folds 5
  python atomiq_unknown_14q_N1_N3.py --self-test

Requires: numpy, pandas, scikit-learn, qiskit; and
          atomiq_stage1_AB_optimization.py in the same directory.
The outer evaluation is on poutcome='unknown' customers only. All fitting,
quantum angle scaling and N2 pair selection use OUTER TRAINING rows only.
Within every fold, N1/N2/N3 use the SAME 14q quantum circuit features.
No resampling or hyperparameter tuning on the outer validation set.
"""
from __future__ import annotations
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import atomiq_stage1_AB_optimization as base
except ImportError as e:
    raise ImportError('Place atomiq_stage1_AB_optimization.py next to this script') from e

COLUMNS = [c for c in base.ORDER if c != 'poutcome']
NQ = len(COLUMNS)
PAIRS = list(itertools.combinations(range(NQ), 2))
N_PAIRS = len(PAIRS)
MODELS = ['N0_classical_reference', 'N1_14q_all_91_pairs',
          'N2_14q_top25_pairs', 'N3_14q_plus_classical']
assert NQ == 14 and N_PAIRS == 91


def classifier(seed):
    return Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(
            penalty='elasticnet', solver='saga', C=0.1, l1_ratio=0.5,
            max_iter=6000, tol=1e-3, random_state=seed)),
    ])


def classical_preprocessor():
    numeric = [c for c in base.NUMERIC if c in COLUMNS]
    categorical = [c for c in base.CATEGORICAL if c in COLUMNS]
    return ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)


def choose_pairs_train_only(q_train, y_train, k=25):
    """Univariate absolute Pearson screening using outer TRAINING labels only."""
    a = np.asarray(q_train[:, NQ:], dtype=float)
    yc = np.asarray(y_train, dtype=float) - np.mean(y_train)
    xc = a - a.mean(axis=0)
    denom = np.sqrt((xc * xc).sum(axis=0) * np.dot(yc, yc))
    corr = np.divide(np.abs(xc.T @ yc), denom,
                     out=np.zeros(a.shape[1]), where=denom > 1e-12)
    chosen = np.argsort(-corr, kind='stable')[:k]
    return np.sort(chosen)


def quantum_feature_fold(train, valid):
    """Fit angle scaler on TRAIN only; identical circuit to previous 15q runs."""
    angle = base.FoldAngleEncoder(COLUMNS).fit(train)
    encoded = np.vstack((angle.transform(train), angle.transform(valid)))
    q = base.get_quantum_features(encoded, COLUMNS, 'full', True, 0.7)
    assert q.shape == (len(train) + len(valid), NQ + N_PAIRS), q.shape
    if not np.isfinite(q).all():
        raise ValueError('Non-finite quantum features')
    return q[:len(train)], q[len(train):]


def model_matrices(train, valid, qtr, qva, ytr):
    """N0 is a classical sanity-check; N1-N3 are requested 14q variants."""
    prep = classical_preprocessor()
    ctr = np.asarray(prep.fit_transform(train[COLUMNS]), dtype=float)
    cva = np.asarray(prep.transform(valid[COLUMNS]), dtype=float)
    sel = choose_pairs_train_only(qtr, ytr, 25)
    matrices = {
        MODELS[0]: (ctr, cva),
        MODELS[1]: (qtr, qva),
        MODELS[2]: (
            np.column_stack((qtr[:, :NQ], qtr[:, NQ:][:, sel])),
            np.column_stack((qva[:, :NQ], qva[:, NQ:][:, sel]))),
        MODELS[3]: (np.column_stack((ctr, qtr)),
                    np.column_stack((cva, qva))),
    }
    return matrices, sel


def metrics(y, pred, k=10):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(pred, dtype=float), 1e-10, 1-1e-10)
    top = np.argsort(-p, kind='stable')[:min(k, len(y))]
    return {
        'PR_AUC': float(average_precision_score(y, p)),
        'ROC_AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        'Brier': float(brier_score_loss(y, p)),
        'LogLoss': float(log_loss(y, p, labels=[0, 1])),
        'top10_hits_unknown': int(y[top].sum()),
        'precision_at_10_unknown': float(y[top].sum() / len(top)),
        'recall_at_10_unknown': float(y[top].sum() / y.sum()) if y.sum() else np.nan,
    }


def run(data, seeds=(42, 179, 316), folds=5, outdir='unknown_14q_results',
        verbose=True, quantum_fn=quantum_feature_fold):
    required = {'customer_id', 'y', 'poutcome'} | set(COLUMNS)
    if required - set(data):
        raise ValueError(f'Missing columns: {sorted(required-set(data))}')
    if data['customer_id'].duplicated().any():
        raise ValueError('customer_id must be unique')
    unique_y = set(data['y'].astype(str).str.strip().str.lower())
    if unique_y != {'yes', 'no'}:
        raise ValueError(f'Expected both yes/no labels; got {unique_y}')
    unknown = data.loc[data.poutcome.astype(str).str.strip().str.lower().eq('unknown')].copy().reset_index(drop=True)
    y = unknown.y.astype(str).str.strip().str.lower().eq('yes').astype(int).to_numpy()
    if min(np.bincount(y)) < folds:
        raise ValueError('Too few positives/negatives in unknown subgroup for requested folds')
    if verbose:
        print(f'Unknown subgroup: {len(y)} people, {y.sum()} subscribers; 14q, 91 Cij', flush=True)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    oof_rows, metric_rows, feature_rows, fold_rows = [], [], [], []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        predictions = np.full((len(y), len(MODELS)), np.nan)
        fold_assignment = np.full(len(y), -1, dtype=int)
        for fold_id, (itr, iva) in enumerate(cv.split(unknown, y), start=1):
            if verbose:
                print(f'[seed={seed} fold={fold_id}/{folds}] generating 14q features', flush=True)
            tr = unknown.iloc[itr].reset_index(drop=True)
            va = unknown.iloc[iva].reset_index(drop=True)
            qtr, qva = quantum_fn(tr, va)
            if qtr.shape != (len(itr), 105) or qva.shape != (len(iva), 105):
                raise ValueError('Quantum generator must return 105 features (14 singles + 91 pairs)')
            matrices, selected = model_matrices(tr, va, qtr, qva, y[itr])
            fold_assignment[iva] = fold_id
            for name, (xtr, xva) in matrices.items():
                model = classifier(int(seed))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    model.fit(xtr, y[itr])
                predictions[iva, MODELS.index(name)] = model.predict_proba(xva)[:, 1]
            feature_rows.extend([{
                'seed': int(seed), 'fold': fold_id,
                'pair_index': int(j), 'feature_a': COLUMNS[PAIRS[j][0]],
                'feature_b': COLUMNS[PAIRS[j][1]],
            } for j in selected])
            for i, name in enumerate(MODELS):
                fold_rows.append({'seed': int(seed), 'fold': fold_id, 'model': name,
                                  **metrics(y[iva], predictions[iva, i], k=min(10, len(iva)))})
        if not np.isfinite(predictions).all():
            raise RuntimeError('Missing OOF predictions')
        for i, name in enumerate(MODELS):
            metric_rows.append({'seed': int(seed), 'model': name,
                                **metrics(y, predictions[:, i])})
        for idx, customer_id in enumerate(unknown.customer_id):
            oof_rows.append({'seed': int(seed), 'customer_id': customer_id,
                             'poutcome': 'unknown', 'label': int(y[idx]),
                             'fold': int(fold_assignment[idx]),
                             **{f'p_{name}': float(predictions[idx, i])
                                for i, name in enumerate(MODELS)}})

    by_seed = pd.DataFrame(metric_rows)
    summary = by_seed.groupby('model', sort=False).agg(
        **{f'{field}_mean': (field, 'mean') for field in
           ['PR_AUC', 'ROC_AUC', 'Brier', 'LogLoss', 'top10_hits_unknown',
            'precision_at_10_unknown', 'recall_at_10_unknown']},
        **{f'{field}_std': (field, 'std') for field in
           ['PR_AUC', 'top10_hits_unknown']},
    ).reset_index()
    oof = pd.DataFrame(oof_rows)
    chosen = pd.DataFrame(feature_rows)
    summary.to_csv(out / 'summary.csv', index=False)
    by_seed.to_csv(out / 'by_seed.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'by_fold.csv', index=False)
    oof.to_csv(out / 'all_oof.csv', index=False)
    chosen.to_csv(out / 'selected_N2_pairs.csv', index=False)
    freq = (chosen.groupby(['feature_a', 'feature_b'], as_index=False)
            .size().sort_values('size', ascending=False)
            .rename(columns={'size': 'selected_folds'}))
    freq.to_csv(out / 'N2_pair_frequency.csv', index=False)
    if verbose:
        print('\n=== UNKNOWN-ONLY OOF SUMMARY ===')
        print(summary.to_string(index=False, float_format=lambda f: f'{f:.4f}'))
        print('Saved to', out.resolve())
    return {'summary': summary, 'by_seed': by_seed, 'all_oof': oof,
            'selected_pairs': chosen, 'pair_frequency': freq}


def main(train_path='Stage1_Train.tsv', seeds=(42, 179, 316), folds=5,
         outdir='unknown_14q_results', verbose=True):
    df = pd.read_csv(train_path, sep='\t')
    return run(df, seeds=seeds, folds=folds, outdir=outdir, verbose=verbose)


def self_test():
    """Verify end-to-end CV/selection/output WITHOUT pretending to run Qiskit."""
    rng = np.random.default_rng(4321)
    data = pd.read_csv(Path(__file__).parent / 'Stage1_Train.tsv', sep='\t')
    data = data.head(140).copy()
    # Synthetic quantum features exercise data plumbing, not measure ML performance.
    def fake_quantum(tr, va):
        def feature(x):
            z = np.asarray(x.age, dtype=float)[:, None]
            angle = np.sin(z * (np.arange(105)[None, :] + 1) * .003)
            return angle + .01 * rng.normal(size=angle.shape)
        return feature(tr), feature(va)
    data = data.assign(poutcome='unknown')
    result = run(data, seeds=(42,), folds=3, outdir='/tmp/atomiq14q_selftest',
                 verbose=False, quantum_fn=fake_quantum)
    assert len(result['summary']) == 4
    assert result['all_oof'].groupby('customer_id').size().eq(1).all()
    print('SELF-TEST PASSED (synthetic quantum features; NOT a performance benchmark)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='Stage1_Train.tsv')
    parser.add_argument('--outdir', default='unknown_14q_results')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 179, 316])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(args.train, args.seeds, args.folds, args.outdir)
