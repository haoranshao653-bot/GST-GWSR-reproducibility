"""Random inner five-fold selection; fixed full-local objective and best-ever GP.

Run this file, not original_bisquare.py. No GST. The 205 outer test responses
are evaluated only after bandwidth selection is saved. Folds run sequentially;
independent local predictions run in Loky workers (one GP thread per worker).
"""
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import original_bisquare as base
import paper_gp

STATION_FILE = base.STATION_FILE
K_VALUES = [55]
SEEDS = [42, 123, 456]
OUTER_SEED = 42
CV_SEED = 42
N_JOBS = -1
PROGRESS_EVERY = 20
RESUME = True
OUTPUT_ROOT = Path(__file__).resolve().parent / 'results'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_csv(path, frame):
    temp = path.with_suffix('.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, path)


def prepare_training(raw_train, query):
    """No query responses used for cleaning, transformations or distances."""
    q1, q3, iqr, lower, upper = base.iqr_bounds(raw_train[base.TARGET], 3.0)
    keep = raw_train[base.TARGET].between(lower, upper)
    train = raw_train.loc[keep].copy()
    xs, ys = StandardScaler(), StandardScaler()
    xtrain = xs.fit_transform(train[base.FEATURES].to_numpy())
    xquery = xs.transform(query[base.FEATURES].to_numpy())
    ytrain = ys.fit_transform(np.log(train[base.TARGET].to_numpy()).reshape(-1, 1)).ravel()
    distances = cdist(query[base.COORDS].to_numpy(), train[base.COORDS].to_numpy(),
                      metric='euclidean')
    info = dict(raw_train_count=len(raw_train), retained_train_count=len(train),
                removed_train_count=int((~keep).sum()), lower=lower, upper=upper,
                q1=q1, q3=q3, iqr=iqr,
                retained_row_ids=train['row_id'].astype(int).tolist(),
                removed_row_ids=raw_train.loc[~keep, 'row_id'].astype(int).tolist(),
                x_mean=xs.mean_.tolist(), x_scale=xs.scale_.tolist(),
                log_y_mean=ys.mean_.tolist(), log_y_scale=ys.scale_.tolist())
    return xtrain, ytrain, xquery, distances, ys, info


def execute_point(i, xtrain, ytrain, xquery, distances, k, seed):
    try:
        idx, pred, expression, fit_r2, info = paper_gp.predict_one_point(
            i, xtrain, ytrain, xquery, distances, k, seed)
        return idx, pred, expression, fit_r2, os.getpid(), info
    except Exception as exc:
        raise RuntimeError(f'GP task i={i}, k={k}, seed={seed} failed: {exc}') from exc


def metric_row(frame, label):
    row = base.metrics(frame.q_obs.to_numpy(), frame.q_pred.to_numpy(), label)
    expressions = frame.expression.fillna('').astype(str)
    row.update(n=len(frame), failed_count=int(expressions.str.startswith('FAILED(').sum()),
               protected_count=int(expressions.str.startswith('CLIPPED(').sum()),
               mean_local_fit_R2=float(frame.local_fit_R2.mean()))
    return row


def show_metrics(frame, label):
    row = metric_row(frame, label)
    print(f'[{label}] n={row["n"]} R2={row["R2"]:.6g} RMSE={row["RMSE"]:.6g} '
          f'MAE={row["MAE"]:.6g} MAPE={row["MAPE"]:.4g}% '
          f'FAILED={row["failed_count"]} PROTECTED={row["protected_count"]} '
          f'mean_local_fit_R2(log,unweighted)={row["mean_local_fit_R2"]:.6g}', flush=True)
    return row


def predict_batch(prepared, query, k, seed, label, path):
    xtrain, ytrain, xquery, distances, ys, info = prepared
    if k > len(xtrain):
        raise ValueError('k exceeds retained training count; do not silently reduce k.')
    meta_path = path.with_suffix('.json')
    if RESUME and path.exists() and meta_path.exists():
        table = pd.read_csv(path)
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('csv_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()
                or table.row_id.tolist() != query.row_id.tolist()
                or not np.isfinite(table.q_pred).all()):
            raise ValueError(f'Checkpoint validation failed: {path}')
        print(f'RESUME {label}: verified saved predictions', flush=True)
        return table, meta['elapsed_seconds']
    started = time.perf_counter()
    finished = []
    observed = query[base.TARGET].to_numpy()  # reporting only, never passed to GP
    print(f'START {label}: train={len(xtrain)}, validation/test={len(query)}', flush=True)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        generated = Parallel(n_jobs=N_JOBS, backend='loky', return_as='generator_unordered')(
            delayed(execute_point)(i, xtrain, ytrain, xquery[i], distances[i], k, seed)
            for i in range(len(query)))
        for result in generated:
            finished.append(result)
            if len(finished) % PROGRESS_EVERY == 0 or len(finished) == len(query):
                ids = np.array([r[0] for r in finished], dtype=int)
                try:
                    pred = paper_gp.inverse_transform(np.array([r[1] for r in finished]), ys)
                except ValueError:
                    save_json(path.with_suffix('.failure.json'), dict(label=label,
                              completed_tasks=[dict(task_id=int(r[0]),row_id=int(query.iloc[r[0]].row_id),
                              prediction_scaled=float(r[1]),expression=r[5]['expression_exact'])
                              for r in finished]))
                    raise
                score = base.metrics(observed[ids], pred, label)
                failed = sum(r[2].startswith('FAILED(') for r in finished)
                protected = sum(r[2].startswith('CLIPPED(') for r in finished)
                print(f'  {label} {len(finished)}/{len(query)} '
                      f'PARTIAL_R2={score["R2"]:.6g} RMSE={score["RMSE"]:.6g} '
                      f'last_local_fit_R2={result[3]:.6g} FAILED={failed} '
                      f'PROTECTED={protected}', flush=True)
    finished.sort(key=lambda r: r[0])
    scaled = np.array([r[1] for r in finished])
    pred = paper_gp.inverse_transform(scaled, ys)
    log_pred = ys.inverse_transform(scaled.reshape(-1, 1)).ravel()
    table = query[['row_id', *base.COORDS]].reset_index(drop=True).copy()
    table['q_obs'] = observed
    table['q_pred'] = pred
    table['residual'] = observed-pred
    table['abs_err'] = np.abs(observed-pred)
    table['local_fit_R2'] = [r[3] for r in finished]
    table['expression'] = [r[2] for r in finished]
    table['worker_pid'] = [r[4] for r in finished]
    table['prediction_scaled'] = scaled
    table['prediction_log'] = log_pred
    table['inverse_clip_used'] = (log_pred < -5) | (log_pred > 10)
    for key in finished[0][5]:
        table[key] = [json.dumps(r[5][key]) if isinstance(r[5][key],list)
                      else r[5][key] for r in finished]
    table['k'], table['seed'] = k, seed
    elapsed = time.perf_counter()-started
    save_csv(path, table)
    save_json(meta_path, dict(elapsed_seconds=elapsed,
                             csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **info))
    print(f'DONE {label}: {elapsed/60:.2f} min, worker PIDs={table.worker_pid.nunique()}', flush=True)
    if table.expression.str.startswith('FAILED(').any():
        print('WARNING: GP training/prediction exception used original mean fallback. '
              'Inspect expression and failure_reason columns.', flush=True)
    return table, elapsed


def main():
    source = Path(__file__).resolve()
    df = pd.read_csv(STATION_FILE)
    required = [*base.FEATURES, *base.COORDS, base.TARGET]
    if len(df) != 1024:
        raise ValueError(f'Expected 1024 observations, found {len(df)}. Check input file.')
    if not np.isfinite(df[required].to_numpy(dtype=float)).all() or (df[base.TARGET] <= 0).any():
        raise ValueError('Input contains missing/nonfinite values or nonpositive heat flow.')
    if min(K_VALUES) < 20:
        raise ValueError('Original code uses min_samples=20; candidate k must be >=20.')
    df['row_id'] = np.arange(len(df))
    train_ids, test_ids = train_test_split(np.arange(len(df)), test_size=.2, random_state=OUTER_SEED)
    assert len(train_ids) == 819 and len(test_ids) == 205 and not set(train_ids) & set(test_ids)
    outer_train = df.iloc[train_ids].copy().reset_index(drop=True)
    # Do not build/evaluate outer test predictions until selection is frozen.
    folds = list(KFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(outer_train))
    manifest = dict(k_values=K_VALUES, seeds=SEEDS, outer_seed=OUTER_SEED,
                    cv_seed=CV_SEED, n_splits=5, cv='random KFold',
                    selection='mean across seeds of pooled OOF RMSE; exact tie -> smaller k',
                    station_sha256=hashlib.sha256(STATION_FILE.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    original_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                    engine_sha256=hashlib.sha256(Path(paper_gp.__file__).read_bytes()).hexdigest(),
                    local_objective='normalized full-local weighted SSE + 0.01 * tree_nodes',
                    final_expression='minimum objective over all evaluated generations',
                    max_samples=1.0, prediction_substitution=True, inverse_clip=True,
                    prediction_scaled_threshold=5, inverse_log_clip=[-5,10],
                    python=platform.python_version(),
                    packages={p: version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','gplearn']})
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = OUTPUT_ROOT / fingerprint
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'config.json', manifest)
    split_table = df[['row_id', *base.COORDS]].copy()
    split_table['outer_role'] = 'test'
    split_table.loc[train_ids, 'outer_role'] = 'train'
    split_table['inner_fold'] = -1
    for fold, (_, valid_idx) in enumerate(folds, 1):
        split_table.loc[train_ids[valid_idx], 'inner_fold'] = fold
    save_csv(out/'shared_data_split.csv', split_table)
    print(f'CPUs={os.cpu_count()}, local workers={effective_n_jobs(N_JOBS)}, '
          'concurrent folds=1; no nested parallelism.', flush=True)
    print(f'Outer: 819 train / 205 test. Random 5-fold selection, k={K_VALUES}, seeds={SEEDS}', flush=True)
    print(f'Output: {out}', flush=True)
    prepared_folds = []
    for fold, (fit_idx, valid_idx) in enumerate(folds, 1):
        assert not set(fit_idx) & set(valid_idx)
        train = outer_train.iloc[fit_idx].copy()
        query = outer_train.iloc[valid_idx].copy()
        prepared = prepare_training(train, query.drop(columns=[base.TARGET]))
        save_json(out/f'fold{fold}_training_preprocessing.json', prepared[-1])
        prepared_folds.append((prepared, query))
    fold_rows, seed_rows, k_rows = [], [], []
    for k in K_VALUES:
        this_k = []
        for seed in SEEDS:
            tables = []
            for fold, (prepared, query) in enumerate(prepared_folds, 1):
                label = f'CV k={k} seed={seed} fold={fold}/5'
                table, elapsed = predict_batch(prepared, query, k, seed, label,
                                              out/f'cv_k{k}_seed{seed}_fold{fold}.csv')
                table['fold'] = fold
                tables.append(table)
                row = show_metrics(table, label)
                row.update(k=k, seed=seed, fold=fold, seconds=elapsed)
                fold_rows.append(row)
                save_csv(out/'cv_fold_metrics.csv', pd.DataFrame(fold_rows))
            oof = pd.concat(tables, ignore_index=True).sort_values('row_id')
            assert len(oof) == 819 and oof.row_id.nunique() == 819
            assert set(oof.row_id) == set(train_ids)
            save_csv(out/f'cv_oof_k{k}_seed{seed}.csv', oof)
            row = show_metrics(oof, f'CV POOLED k={k} seed={seed}')
            row.update(k=k, seed=seed)
            this_k.append(row)
            seed_rows.append(row)
            save_csv(out/'cv_seed_metrics.csv', pd.DataFrame(seed_rows))
        summary = dict(k=k, mean_CV_RMSE=float(np.mean([r['RMSE'] for r in this_k])),
                       std_CV_RMSE=float(np.std([r['RMSE'] for r in this_k], ddof=1)),
                       mean_CV_R2=float(np.mean([r['R2'] for r in this_k])),
                       mean_CV_MAE=float(np.mean([r['MAE'] for r in this_k])),
                       failed_count=sum(r['failed_count'] for r in this_k),
                       protected_count=sum(r['protected_count'] for r in this_k))
        k_rows.append(summary)
        save_csv(out/'cv_k_summary.csv', pd.DataFrame(k_rows).sort_values(['mean_CV_RMSE','k']))
        print(f'K COMPLETE: k={k} mean_CV_R2={summary["mean_CV_R2"]:.6g} '
              f'mean_CV_RMSE={summary["mean_CV_RMSE"]:.6g} '
              f'std_RMSE={summary["std_CV_RMSE"]:.6g}', flush=True)
    best = min(k_rows, key=lambda r: (r['mean_CV_RMSE'], r['k']))
    best_k = int(best['k'])
    save_json(out/'selected_k_before_test.json', dict(selected_k=best_k, cv_summary=best,
              outer_test_used_for_selection=False, selection_rule=manifest['selection']))
    print(f'SELECTION FROZEN: k={best_k}. Now evaluate outer test, without further tuning.', flush=True)
    outer_test = df.iloc[test_ids].copy().reset_index(drop=True)
    prepared = prepare_training(outer_train, outer_test.drop(columns=[base.TARGET]))
    save_json(out/'final_training_preprocessing.json', prepared[-1])
    test_tables, test_rows = [], []
    for seed in SEEDS:
        label = f'TEST selected_k={best_k} seed={seed}'
        table, elapsed = predict_batch(prepared, outer_test, best_k, seed, label,
                                      out/f'test_k{best_k}_seed{seed}.csv')
        test_tables.append(table)
        row = show_metrics(table, label)
        row.update(k=best_k, seed=seed, type='single_seed', seconds=elapsed)
        test_rows.append(row)
        save_csv(out/'test_metrics.csv', pd.DataFrame(test_rows))
    stack = np.stack([t.q_pred.to_numpy() for t in test_tables])
    # Predeclared descriptive ensembles, never used to select k or a test winner.
    for name, values in [('mean_ensemble',stack.mean(axis=0)),
                         ('median_ensemble',np.median(stack,axis=0))]:
        table = outer_test[['row_id', *base.COORDS]].copy()
        table['q_obs'], table['q_pred'] = outer_test[base.TARGET].to_numpy(), values
        table['residual'] = table.q_obs-table.q_pred
        save_csv(out/f'test_k{best_k}_{name}.csv', table)
        row = base.metrics(table.q_obs.to_numpy(), values, name)
        row.update(k=best_k, seed=-1, type=name)
        test_rows.append(row)
        print(f'TEST {name}: R2={row["R2"]:.6g}, RMSE={row["RMSE"]:.6g}, '
              f'MAE={row["MAE"]:.6g}', flush=True)
    save_csv(out/'test_metrics.csv', pd.DataFrame(test_rows))
    singles = [r for r in test_rows if r['type']=='single_seed']
    summary = {f'{stat}_{metric}':float(func([r[metric] for r in singles]))
               for metric in ['R2','RMSE','MAE']
               for stat,func in [('mean',np.mean),('std',lambda a:np.std(a,ddof=1))]}
    save_json(out/'test_seed_summary.json', summary)
    print(f'FINAL single-seed mean R2={summary["mean_R2"]:.6g}, '
          f'mean RMSE={summary["mean_RMSE"]:.6g}. No test-based winner selected.', flush=True)
    print(f'All results: {out}', flush=True)


if __name__ == '__main__':
    main()
