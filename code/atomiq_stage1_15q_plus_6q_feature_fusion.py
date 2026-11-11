"""Atomiq Stage 1: test whether A0 6q quantum features help B0 15q hybrid.

Files required alongside this script:
    Stage1_Train.tsv
    atomiq_stage1_AB_optimization.py

Notebook:
    from atomiq_stage1_15q_plus_6q_feature_fusion import main
    results = main('Stage1_Train.tsv', repeats=3)

CLI:
    python atomiq_stage1_15q_plus_6q_feature_fusion.py Stage1_Train.tsv --repeats 3

Output CSV files (prefix stage1_15q6q): _summary, _by_seed, _by_fold,
    _all_oof, _selected_features, _inner_choices.

Compared on IDENTICAL outer folds:
 B0_15q             : classical one-hot + 15 singles + best 25 15q Cij
 F1_plus_6q_singles : B0 + six 6q single Z expectations
 F2_plus_6q_pairs   : B0 + fifteen 6q connected Cij
 F3_plus_6q_all     : B0 + all 21 6q quantum features
 F4_plus_6q_tuned   : inner-CV choose among F1/F2/F3 or B0 (extra check)

B0 fixed settings match atomiq_stage1_B0_B4_improved.py:
 15q: base.ORDER, full RZZ, shift_binary=True, strength=0.7,
 train-only selection of 25 best connected pairs by Pearson association,
 classical full 15 attributes, ElasticNet C=.1, l1_ratio=.5.
6q features match E1's fixed A6 quantum map:
 age,balance,campaign,poutcome,day,previous; full RZZ,
 shift_binary=False, strength=0.7. These features are NOT A0 predictions:
 they are directly concatenated into B0's classifier.

IMPORTANT:
- Every outer validation row/label is excluded from all fitting, scaling,
  correlation selection, and selection of F4 configuration.
- Inner q-angle scaler uses the OUTER training rows to avoid repeating costly
  statevector simulation; inner validation *covariates* influence this unsupervised
  scaling but never their labels. Outer CV remains leakage-safe; inner scores
  are approximate, used only for F4 choice.  Baseline F0..F3 do not tune.
- Repeated pooled OOF top-10 results are descriptive, not independent test.
- This does NOT prove quantum advantage; exact statevectors are simulated classically.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import atomiq_stage1_AB_optimization as base
except ImportError as exc:
    raise ImportError('Put atomiq_stage1_AB_optimization.py alongside this script') from exc

Q6_COLS = ('age', 'balance', 'campaign', 'poutcome', 'day', 'previous')
PAIR_K = 25
MODELS = (
    'B0_15q',
    'F1_plus_6q_singles',
    'F2_plus_6q_pairs',
    'F3_plus_6q_all',
    'F4_inner_choose',
)


def score(y, pred, top_k=10):
    y = np.asarray(y, dtype=int)
    pred = np.clip(np.asarray(pred, dtype=float), 1.e-10, 1.-1.e-10)
    ix = np.argsort(-pred, kind='stable')[:min(top_k, len(y))]
    return {
        'ROC_AUC': float(roc_auc_score(y, pred)) if len(set(y)) == 2 else np.nan,
        'PR_AP': float(average_precision_score(y, pred)) if len(set(y)) == 2 else np.nan,
        'LogLoss': float(log_loss(y, pred, labels=[0, 1])),
        'Brier': float(brier_score_loss(y, pred)),
        'top10_hits': int(y[ix].sum()),
    }


def choose_pairs(q15, ytrain, k=PAIR_K):
    """Train-only selection of connected pairs (columns 15:120)."""
    c = np.asarray(q15[:, 15:], dtype=float)
    yy = np.asarray(ytrain, dtype=float)
    xc = c - c.mean(axis=0)
    yc = yy - yy.mean()
    den = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    assoc = np.divide(np.abs(xc.T @ yc), den,
                      out=np.zeros(c.shape[1]), where=den > 1.e-12)
    return np.sort(np.argsort(-assoc, kind='stable')[:min(k, c.shape[1])])


def classifier():
    return Pipeline([
        ('standardize', StandardScaler()),
        ('elasticnet', LogisticRegression(
            penalty='elasticnet', solver='saga', C=0.1, l1_ratio=0.5,
            max_iter=4000, tol=1.e-3, random_state=42,
        )),
    ])


def create_matrix(classical, q15, q6, pair_indices, variant):
    """Construct features; q15/q6 are (q singles, then q choose 2 pairs)."""
    if q15.shape[1] != 120 or q6.shape[1] != 21:
        raise ValueError(f'Unexpected quantum shape: 15q={q15.shape}, 6q={q6.shape}')
    blocks = [np.asarray(classical, dtype=float), q15[:, :15], q15[:, 15:][:, pair_indices]]
    if variant in (1, 3):
        blocks.append(q6[:, :6])
    if variant in (2, 3):
        blocks.append(q6[:, 6:])
    return np.column_stack(blocks)


def fit_predict(train, y, test, q15_train, q15_test, q6_train, q6_test, variant,
                return_indices=False):
    """All fitting/screening is performed on train alone."""
    pre = base.classical_preprocessor()
    ct = np.asarray(pre.fit_transform(train.loc[:, base.ORDER]), dtype=float)
    cv = np.asarray(pre.transform(test.loc[:, base.ORDER]), dtype=float)
    pair_indices = choose_pairs(q15_train, y)
    xtr = create_matrix(ct, q15_train, q6_train, pair_indices, variant)
    xva = create_matrix(cv, q15_test, q6_test, pair_indices, variant)
    model = classifier()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        model.fit(xtr, y)
    prob = np.clip(model.predict_proba(xva)[:, 1], 1.e-10, 1.-1.e-10)
    return (prob, pair_indices) if return_indices else prob


def make_quantum_features(train, test, cols, shift):
    encoder = base.FoldAngleEncoder(list(cols)).fit(train)
    angles = np.vstack((encoder.transform(train), encoder.transform(test)))
    quantum = base.get_quantum_features(angles, list(cols), 'full', shift, 0.7)
    return quantum[:len(train)], quantum[len(train):]


def select_fusion_inner(train, y, q15, q6, seed):
    """Choose a variant from 0..3 solely by inner 3-fold LogLoss."""
    split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(split.split(train, y))
    losses = {}
    for variant in range(4):
        oof = np.full(len(train), np.nan)
        for itr, iva in splits:
            oof[iva] = fit_predict(
                train.iloc[itr], y[itr], train.iloc[iva],
                q15[itr], q15[iva], q6[itr], q6[iva], variant,
            )
        losses[variant] = float(log_loss(y, oof, labels=[0, 1]))
    chosen = min(losses, key=lambda k: (losses[k], k))
    return chosen, losses


def main(train_path='Stage1_Train.tsv', repeats=3, outdir='.',
         seed_start=42, seed_step=137, verbose=True):
    df = pd.read_csv(train_path, sep='\t')
    required = {'customer_id', 'y'} | set(base.ORDER)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df.customer_id.isna().any() or df.customer_id.duplicated().any():
        raise ValueError('customer_id must be non-null and unique')
    vals = df.y.astype(str).str.lower().str.strip()
    if not set(vals.unique()) <= {'yes', 'no'}:
        raise ValueError('Expected y values yes/no')
    y = vals.eq('yes').astype(int).to_numpy()
    if min(np.bincount(y)) < 5:
        raise ValueError('Need at least 5 customers of each class')

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    by_fold, by_seed, all_oof, pairs_rows, inner_rows = [], [], [], [], []

    for rep in range(repeats):
        seed = seed_start + seed_step * rep
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        predictions = np.full((len(df), len(MODELS)), np.nan)
        fold_col = np.full(len(df), -1)
        for fold, (itr, iva) in enumerate(outer.split(df, y), 1):
            train = df.iloc[itr].reset_index(drop=True)
            valid = df.iloc[iva].reset_index(drop=True)
            yt = y[itr]
            if verbose:
                print(f'\nRepeat {rep+1}/{repeats}, seed={seed}, fold={fold}/5', flush=True)

            q15t, q15v = make_quantum_features(train, valid, base.ORDER, True)
            q6t, q6v = make_quantum_features(train, valid, Q6_COLS, False)

            for kind in range(4):
                pv, selected = fit_predict(
                    train, yt, valid, q15t, q15v, q6t, q6v, kind,
                    return_indices=True,
                )
                predictions[iva, kind] = pv
                pairs_rows.append({
                    'seed': seed, 'repeat': rep+1, 'fold': fold,
                    'model': MODELS[kind],
                    'selected_15q_pair_indices': ';'.join(map(str, selected)),
                    'n_15q_pairs': len(selected),
                    'n_6q_added': [0, 6, 15, 21][kind],
                })
                if verbose:
                    print(f'  {MODELS[kind]}: predicted range '
                          f'{pv.min():.3f} .. {pv.max():.3f}', flush=True)

            chosen, losses = select_fusion_inner(train, yt, q15t, q6t,
                                                  seed=seed+1000+fold)
            predictions[iva, 4] = predictions[iva, chosen]
            inner_rows.append({
                'seed': seed, 'repeat': rep+1, 'fold': fold,
                'selected_variant': MODELS[chosen],
                **{f'inner_LogLoss_{MODELS[v]}': losses[v] for v in range(4)},
            })
            fold_col[iva] = fold
            for i, name in enumerate(MODELS):
                by_fold.append({
                    'seed': seed, 'repeat': rep+1, 'fold': fold,
                    'model': name, **score(y[iva], predictions[iva, i]),
                })
            if verbose:
                print(f'  F4 chose {MODELS[chosen]}', flush=True)

        if not np.isfinite(predictions).all() or (fold_col < 0).any():
            raise RuntimeError('Missing OOF predictions')
        for i, name in enumerate(MODELS):
            by_seed.append({
                'seed': seed, 'repeat': rep+1, 'model': name,
                **score(y, predictions[:, i]),
            })
        temp = pd.DataFrame(predictions, columns=[f'p_{m}' for m in MODELS])
        temp.insert(0, 'customer_id', df.customer_id.astype(str).values)
        temp['label'] = y
        temp['seed'] = seed
        temp['repeat'] = rep+1
        temp['fold'] = fold_col
        all_oof.append(temp)
        if verbose:
            print(f'\n=== Seed {seed}: pooled 300-customer OOF diagnostics ===', flush=True)
            for row in by_seed[-len(MODELS):]:
                print(f"  {row['model']}: PR {row['PR_AP']:.4f}, "
                      f"ROC {row['ROC_AUC']:.4f}, "
                      f"LogLoss {row['LogLoss']:.4f}, "
                      f"hits {row['top10_hits']}/10", flush=True)

    byseed = pd.DataFrame(by_seed)
    summary = byseed.groupby('model', sort=False).agg(
        repeats=('seed', 'count'),
        ROC_AUC_mean=('ROC_AUC', 'mean'),
        ROC_AUC_sd=('ROC_AUC', 'std'),
        PR_AP_mean=('PR_AP', 'mean'),
        PR_AP_sd=('PR_AP', 'std'),
        LogLoss_mean=('LogLoss', 'mean'),
        Brier_mean=('Brier', 'mean'),
        top10_hits_mean=('top10_hits', 'mean'),
        top10_hits_min=('top10_hits', 'min'),
        top10_hits_max=('top10_hits', 'max'),
    ).reset_index()
    summary = summary.sort_values(['PR_AP_mean', 'top10_hits_mean'],
                                  ascending=False, kind='stable')
    prefix = output / 'stage1_15q6q'
    files = {
        'summary': str(prefix)+'_summary.csv',
        'by_seed': str(prefix)+'_by_seed.csv',
        'by_fold': str(prefix)+'_by_fold.csv',
        'all_oof': str(prefix)+'_all_oof.csv',
        'selected_features': str(prefix)+'_selected_features.csv',
        'inner_choices': str(prefix)+'_inner_choices.csv',
    }
    summary.to_csv(files['summary'], index=False)
    byseed.to_csv(files['by_seed'], index=False)
    pd.DataFrame(by_fold).to_csv(files['by_fold'], index=False)
    pd.concat(all_oof, ignore_index=True).to_csv(files['all_oof'], index=False)
    pd.DataFrame(pairs_rows).to_csv(files['selected_features'], index=False)
    pd.DataFrame(inner_rows).to_csv(files['inner_choices'], index=False)
    print('\n=== Summary (repeated 5-fold CV; diagnostic) ===', flush=True)
    print(summary.to_string(index=False, float_format=lambda z: f'{z:.4f}'), flush=True)
    print('\nSaved results:', json.dumps(files, indent=2), flush=True)
    return {'summary': summary, 'by_seed': byseed,
            'by_fold': pd.DataFrame(by_fold),
            'all_oof': pd.concat(all_oof, ignore_index=True),
            'inner_choices': pd.DataFrame(inner_rows),
            'files': files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_path', nargs='?', default='Stage1_Train.tsv')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()
    main(args.train_path, repeats=args.repeats, outdir=args.outdir)
