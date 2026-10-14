"""Training-only selection of GST distance/feature weights and threshold.

This script is deliberately separate from the older GST prototypes.  It uses
the same cleaned-data, standardized-log-response, adaptive-bisquare local
objective as the strict GWSR protocol, with k fixed at 45.  It selects one
(alpha, tau) pair using only five-fold CV inside the 819-station outer
training set.  The outer 205 responses are never loaded by the selection code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.genetic import _parallel_evolve
from gplearn.fitness import _fitness_map
from gplearn.functions import _function_map
from joblib import Parallel, delayed, parallel_config
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
SECTION = Path(__file__).resolve().parent
if str(SECTION) not in sys.path:
    sys.path.insert(0, str(SECTION))
import original_bisquare as base
import paper_gp

INPUT = Path(os.environ.get("GST_GWSR_STATION_FILE", str(base.STATION_FILE)))
ROOT = Path(os.environ.get(
    "GST_GWSR_ALPHA_TAU_OUTPUT",
    str(REPO_ROOT / "run_outputs" / "alpha_tau_selection"),
))
K = 45
SEEDS = (42, 123, 456)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TAUS = (0.5, 0.7, 0.9)
OUTER_SEED = 42
INNER_SEED = 42
N_FOLDS = 5
POPULATION = 1000
FULL_GENERATIONS = 20
SHORT_GENERATIONS = 5
N_CHAINS = 24
N_JOBS = int(os.environ.get("GST_GWSR_N_JOBS", "1"))
RANDOM_TRANSFER_FRACTION = 0.20
IQR_MULTIPLIER = 3.0
EPS = 1e-12


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    return dict(
        R2=float(r2_score(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        MAE=float(mean_absolute_error(y, p)),
    )


def haversine(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    lon_a = np.radians(coords_a[:, 0])[:, None]
    lat_a = np.radians(coords_a[:, 1])[:, None]
    lon_b = np.radians(coords_b[:, 0])[None, :]
    lat_b = np.radians(coords_b[:, 1])[None, :]
    dlon = lon_a - lon_b
    dlat = lat_a - lat_b
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def prepare_fold(train: pd.DataFrame, query: pd.DataFrame) -> dict:
    q1, q3 = train[base.TARGET].quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    lower, upper = float(q1 - IQR_MULTIPLIER * iqr), float(q3 + IQR_MULTIPLIER * iqr)
    keep = train[base.TARGET].between(lower, upper)
    clean = train.loc[keep].copy().reset_index(drop=True)
    xs, ys = StandardScaler(), StandardScaler()
    xfit = xs.fit_transform(clean[base.FEATURES].to_numpy(dtype=float))
    xquery = xs.transform(query[base.FEATURES].to_numpy(dtype=float))
    yfit = ys.fit_transform(np.log(clean[base.TARGET].to_numpy(dtype=float)).reshape(-1, 1)).ravel()
    # Local GWSR distance is deliberately the same Euclidean lon/lat distance
    # as the strict independent-GWSR protocol.
    d = cdist(query[base.COORDS].to_numpy(dtype=float), clean[base.COORDS].to_numpy(dtype=float))
    n = len(query)
    positions = np.empty((n, K), dtype=int)
    weights = np.empty((n, K), dtype=float)
    descriptors = np.empty((n, 2 * xfit.shape[1]), dtype=float)
    for i in range(n):
        order = np.argsort(d[i])[:K]
        dl = d[i, order]
        h = float(dl[-1]) + 1e-8
        w = np.where(dl < h, (1.0 - (dl / h) ** 2) ** 2, 0.0)
        w /= max(float(w.sum()), EPS)
        xl = xfit[order]
        mu = np.average(xl, axis=0, weights=w)
        var = np.average((xl - mu) ** 2, axis=0, weights=w)
        positions[i] = order
        weights[i] = w
        descriptors[i] = np.concatenate([mu, np.sqrt(np.maximum(var, 0.0))])
    return dict(
        xfit=xfit, yfit=yfit, xquery=xquery, distances=d, positions=positions,
        weights=weights, descriptors=descriptors, ys=ys,
        query=query.reset_index(drop=True), train=clean,
        qmin=float(np.exp(ys.inverse_transform(yfit.reshape(-1, 1)).min())),
        qmax=float(np.exp(ys.inverse_transform(yfit.reshape(-1, 1)).max())),
        removed=int((~keep).sum()),
    )


def similarity_matrices(prepared: dict) -> tuple[np.ndarray, np.ndarray]:
    coords = prepared["query"][base.COORDS].to_numpy(dtype=float)
    dg = haversine(coords, coords)
    positive_d = dg[np.triu_indices_from(dg, k=1)]
    positive_d = positive_d[positive_d > 0]
    rho_g = float(np.median(positive_d)) if len(positive_d) else 1.0
    sg = np.exp(-dg / max(rho_g, EPS))
    df = cdist(prepared["descriptors"], prepared["descriptors"], metric="euclidean")
    positive_f = df[df > 0]
    rho_x = float(np.median(positive_f)) if len(positive_f) else 1.0
    sx = np.exp(-df / max(rho_x, EPS))
    np.fill_diagonal(sg, 1.0)
    np.fill_diagonal(sx, 1.0)
    prepared["rho_g_km"], prepared["rho_x"] = rho_g, rho_x
    return sg, sx


def projected(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(float(np.mean(coords[:, 1])))
    return np.column_stack([coords[:, 0] * 111.320 * np.cos(lat), coords[:, 1] * 110.574])


def build_order(sim: np.ndarray, coords: np.ndarray, tau: float) -> list[dict]:
    n = len(coords)
    nc = min(N_CHAINS, n)
    labels = KMeans(n_clusters=nc, random_state=INNER_SEED, n_init=20).fit_predict(projected(coords))
    rows: list[dict] = []
    for cid in range(nc):
        members = np.flatnonzero(labels == cid).tolist()
        if not members:
            continue
        if len(members) == 1:
            anchor = members[0]
        else:
            within = sim[np.ix_(members, members)].copy()
            np.fill_diagonal(within, 0.0)
            anchor = members[int(np.argmax(within.mean(axis=1)))]
        processed = [anchor]
        pending = set(members) - {anchor}
        rows.append(dict(chain_id=cid, order=0, query_id=anchor, parent=-1,
                         is_anchor=True, eligible=False, score=np.nan))
        order = 1
        while pending:
            bc = bp = None
            bs = -np.inf
            for child in sorted(pending):
                pp = processed[int(np.argmax(sim[child, processed]))]
                ss = float(sim[child, pp])
                if ss > bs:
                    bc, bp, bs = child, pp, ss
            assert bc is not None and bp is not None
            rows.append(dict(chain_id=cid, order=order, query_id=bc, parent=bp,
                             is_anchor=False, eligible=bool(bs >= tau), score=bs))
            processed.append(bc)
            pending.remove(bc)
            order += 1
    return rows


def objective(program, X, y, w, penalty):
    positive = w > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pred = program.execute(X)
        if not np.isfinite(pred[positive]).all():
            return np.inf
        loss = float(np.dot(w[positive], (y[positive] - pred[positive]) ** 2))
        value = loss + penalty * program.length_
    return float(value) if np.isfinite(value) else np.inf


def search_population(X, y, w, seed, generations, initial=None):
    model = base.build_sr_model(seed)
    params = model.get_params().copy()
    params["generations"] = int(generations)
    params["population_size"] = POPULATION
    funcs = [_function_map[name] for name in params["function_set"]]
    arities = {}
    for fn in funcs:
        arities.setdefault(fn.arity, []).append(fn)
    params.update(function_set=funcs, arities=arities, _metric=_fitness_map["mse"],
                  _transformer=None, max_samples=1.0,
                  method_probs=np.cumsum([params["p_crossover"], params["p_subtree_mutation"],
                                           params["p_hoist_mutation"], params["p_point_mutation"]]))
    rng = np.random.RandomState(seed)
    penalty = float(params["parsimony_coefficient"])
    parents = None if initial is None else [copy.deepcopy(x) for x in initial]
    best = None
    best_j = np.inf
    best_info = None
    if parents is not None:
        for p in parents:
            p.fitness_ = objective(p, X, y, w, penalty)
            p.raw_fitness_ = p.fitness_
            if p.fitness_ < best_j:
                best, best_j = copy.deepcopy(p), float(p.fitness_)
    for gen in range(int(generations)):
        seeds = rng.randint(np.iinfo(np.int32).max, size=POPULATION)
        population = _parallel_evolve(POPULATION, parents, X, y, w, seeds, params)
        finite = False
        for p in population:
            j = objective(p, X, y, w, penalty)
            p.fitness_, p.raw_fitness_ = j, j
            if np.isfinite(j):
                finite = True
            if j < best_j:
                best, best_j = copy.deepcopy(p), float(j)
        if not finite:
            raise RuntimeError("No finite local objective in a GP generation")
        parents = population
    if best is None:
        raise RuntimeError("No valid local symbolic expression")
    return best, parents


def run_task(prepared: dict, task_id: int, seed: int, initial, generations: int) -> dict:
    pos = prepared["positions"][task_id]
    X = prepared["xfit"][pos]
    y = prepared["yfit"][pos]
    w = prepared["weights"][task_id]
    try:
        best, final_population = search_population(X, y, w, seed + task_id, generations, initial)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled_pred = float(best.execute(prepared["xquery"][task_id].reshape(1, -1))[0])
            log_pred = float(prepared["ys"].inverse_transform(np.asarray([[scaled_pred]]))[0, 0])
            raw_pred = float(np.exp(np.clip(log_pred, -5.0, 10.0)))
        # Match the protected GWSR rule: guard the standardized prediction
        # before inverse transformation and fall back to the local weighted mean.
        protected = (not np.isfinite(scaled_pred)) or abs(scaled_pred) >= 5.0
        if protected:
            scaled_pred = float(np.average(y, weights=w))
            log_pred = float(prepared["ys"].inverse_transform(np.asarray([[scaled_pred]]))[0, 0])
            raw_pred = float(np.exp(np.clip(log_pred, -5.0, 10.0)))
            expression = f"CLIPPED({paper_gp.exact_expression(best)[:80]})"
        else:
            expression = paper_gp.exact_expression(best)
        return dict(task_id=task_id, prediction=raw_pred, protected=bool(protected),
                    expression=expression, population=final_population)
    except Exception as exc:
        scaled_pred = float(np.average(y, weights=w))
        log_pred = float(prepared["ys"].inverse_transform(np.asarray([[scaled_pred]]))[0, 0])
        raw_pred = float(np.exp(np.clip(log_pred, -5.0, 10.0)))
        return dict(task_id=task_id, prediction=raw_pred, protected=True,
                    expression=f"FAILED({str(exc)[:80]})", population=[])


def run_chain(prepared: dict, rows: list[dict], seed: int) -> list[dict]:
    populations: dict[int, list] = {}
    out = []
    for row in rows:
        tid = int(row["query_id"])
        parent = int(row["parent"])
        initial = None
        generations = FULL_GENERATIONS
        strategy = "complete_search"
        if bool(row["eligible"]) and parent in populations:
            # Keep 80% of the parent population and 20% freshly initialized
            # expressions, preserving exploration while reusing structure.
            fresh_model = base.build_sr_model(seed + 100000 + tid)
            params = fresh_model.get_params().copy()
            params.update(function_set=[_function_map[n] for n in params["function_set"]])
            arities = {}
            for fn in params["function_set"]:
                arities.setdefault(fn.arity, []).append(fn)
            params.update(arities=arities, _metric=_fitness_map["mse"], _transformer=None,
                          max_samples=1.0,
                          method_probs=np.cumsum([params["p_crossover"], params["p_subtree_mutation"],
                                                   params["p_hoist_mutation"], params["p_point_mutation"]]))
            rng = np.random.RandomState(seed + 200000 + tid)
            fresh = _parallel_evolve(POPULATION, None,
                                     prepared["xfit"][prepared["positions"][tid]],
                                     prepared["yfit"][prepared["positions"][tid]],
                                     prepared["weights"][tid],
                                     rng.randint(np.iinfo(np.int32).max, size=POPULATION), params)
            take = int(POPULATION * RANDOM_TRANSFER_FRACTION)
            initial = [copy.deepcopy(p) for p in populations[parent][:-take]] + [copy.deepcopy(p) for p in fresh[:take]]
            generations = SHORT_GENERATIONS
            strategy = "transfer_adaptation"
        result = run_task(prepared, tid, seed, initial, generations)
        populations[tid] = result.pop("population")
        result.update(strategy=strategy, parent=parent, similarity=float(row["score"]))
        out.append(result)
    return out


def run_independent(prepared: dict, seed: int, path: Path) -> tuple[pd.DataFrame, float]:
    if path.exists():
        return pd.read_csv(path), 0.0
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(run_task)(prepared, i, seed, None, FULL_GENERATIONS)
            for i in range(len(prepared["query"]))
        )
    y = prepared["query"][base.TARGET].to_numpy(dtype=float)
    frame = prepared["query"][["row_id", *base.COORDS]].copy()
    frame["q_obs"], frame["q_pred"] = y, [r["prediction"] for r in rows]
    frame["protected"], frame["expression"] = [r["protected"] for r in rows], [r["expression"] for r in rows]
    frame["seed"] = seed
    elapsed = time.perf_counter() - started
    save_csv(path, frame)
    return frame, elapsed


def run_gst(prepared: dict, seed: int, alpha: float, tau: float, path: Path) -> tuple[pd.DataFrame, float, dict]:
    if path.exists() and path.with_suffix(".json").exists():
        return pd.read_csv(path), float(json.loads(path.with_suffix(".json").read_text()).get("runtime", 0.0)), {}
    started = time.perf_counter()
    sg, sx = similarity_matrices(prepared)
    sim = alpha * sg + (1.0 - alpha) * sx
    coords = prepared["query"][base.COORDS].to_numpy(dtype=float)
    rows = build_order(sim, coords, tau)
    by_chain = {}
    for r in rows:
        by_chain.setdefault(int(r["chain_id"]), []).append(r)
    # Chains are independent after their own anchor is chosen; parallelizing
    # them preserves the within-chain parent order while avoiding a long
    # serial sweep over all local tasks.
    chain_results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_chain)(prepared, by_chain[cid], seed)
        for cid in sorted(by_chain)
    )
    outputs = [item for chunk in chain_results for item in chunk]
    outputs.sort(key=lambda x: x["task_id"])
    y = prepared["query"][base.TARGET].to_numpy(dtype=float)
    frame = prepared["query"][["row_id", *base.COORDS]].copy()
    frame["q_obs"], frame["q_pred"] = y, [r["prediction"] for r in outputs]
    frame["protected"] = [r["protected"] for r in outputs]
    frame["expression"] = [r["expression"] for r in outputs]
    frame["strategy"] = [r["strategy"] for r in outputs]
    frame["parent"] = [r["parent"] for r in outputs]
    frame["similarity"] = [r["similarity"] for r in outputs]
    frame["seed"], frame["alpha"], frame["tau"] = seed, alpha, tau
    elapsed = time.perf_counter() - started
    save_csv(path, frame)
    save_json(path.with_suffix(".json"), dict(runtime=elapsed, alpha=alpha, tau=tau,
                                               transfer_count=int((frame.strategy == "transfer_adaptation").sum()),
                                               protected_count=int(frame.protected.sum())))
    return frame, elapsed, dict(transfer_count=int((frame.strategy == "transfer_adaptation").sum()),
                                protected_count=int(frame.protected.sum()))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["row_id"] = np.arange(len(df))
    ids_train, ids_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=OUTER_SEED)
    outer_train = df.iloc[ids_train].copy().reset_index(drop=True)
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=INNER_SEED).split(outer_train))
    save_json(ROOT / "config.json", dict(protocol="training-only GST alpha/tau selection", k=K,
        alphas=ALPHAS, taus=TAUS, seeds=SEEDS, folds=N_FOLDS, outer_train=len(outer_train),
        outer_test=len(ids_test), kernel="adaptive_bisquare", similarity="distance + feature",
        population=POPULATION, full_generations=FULL_GENERATIONS, short_generations=SHORT_GENERATIONS,
        test_response_used_for_selection=False))
    baseline_rows = []
    prepared_folds = []
    for fold, (fit_idx, val_idx) in enumerate(folds, 1):
        prep = prepare_fold(outer_train.iloc[fit_idx].copy(), outer_train.iloc[val_idx].copy())
        prepared_folds.append(prep)
    for seed in SEEDS:
        seed_frames = []
        seed_runtime = 0.0
        for fold, prep in enumerate(prepared_folds, 1):
            path = ROOT / "baseline" / f"fold{fold}_seed{seed}.csv"
            frame, elapsed = run_independent(prep, seed, path)
            seed_frames.append(frame)
            seed_runtime += elapsed
            fold_row = metrics(frame.q_obs.to_numpy(), frame.q_pred.to_numpy())
            fold_row.update(fold=fold, seed=seed, runtime=elapsed,
                            protected=int(frame.protected.sum()))
            baseline_rows.append(fold_row)
        pooled = pd.concat(seed_frames, ignore_index=True).sort_values("row_id")
        save_csv(ROOT / "baseline" / f"oof_seed{seed}.csv", pooled)
        pooled_row = metrics(pooled.q_obs.to_numpy(), pooled.q_pred.to_numpy())
        pooled_row.update(seed=seed, runtime=seed_runtime,
                          protected=int(pooled.protected.sum()))
        save_json(ROOT / "baseline" / f"oof_seed{seed}.json", pooled_row)
        print(f"BASELINE seed={seed} pooled OOF complete", flush=True)
    save_csv(ROOT / "baseline_fold_metrics.csv", pd.DataFrame(baseline_rows))
    summary_rows = []
    all_detail = []
    for alpha in ALPHAS:
        for tau in TAUS:
            cfg_rows = []
            for seed in SEEDS:
                fold_frames = []
                runtime = 0.0
                transfers = protected = 0
                for fold, prep in enumerate(prepared_folds, 1):
                    path = ROOT / f"alpha_{alpha:g}_tau_{tau:g}" / f"fold{fold}_seed{seed}.csv"
                    frame, elapsed, info = run_gst(prep, seed, alpha, tau, path)
                    fold_frames.append(frame)
                    runtime += elapsed
                    transfers += info.get("transfer_count", int((frame.get("strategy", pd.Series(dtype=str)) == "transfer_adaptation").sum()))
                    protected += info.get("protected_count", int(frame.protected.sum()))
                pooled = pd.concat(fold_frames, ignore_index=True).sort_values("row_id")
                save_csv(ROOT / f"alpha_{alpha:g}_tau_{tau:g}" / f"oof_seed{seed}.csv", pooled)
                row = metrics(pooled.q_obs.to_numpy(), pooled.q_pred.to_numpy())
                row.update(alpha=alpha, tau=tau, seed=seed, runtime=runtime,
                           transfer_count=transfers, protected_count=protected)
                cfg_rows.append(row)
                all_detail.append(row)
            b = pd.DataFrame([json.loads((ROOT / "baseline" / f"oof_seed{s}.json").read_text()) for s in SEEDS])
            g = pd.DataFrame(cfg_rows)
            summary_rows.append(dict(alpha=alpha, tau=tau,
                mean_CV_RMSE=float(g.RMSE.mean()), sd_CV_RMSE=float(g.RMSE.std(ddof=1)),
                mean_CV_R2=float(g.R2.mean()), sd_CV_R2=float(g.R2.std(ddof=1)),
                mean_runtime=float(g.runtime.mean()), transfer_count=int(g.transfer_count.sum()),
                protected_count=int(g.protected_count.sum()),
                baseline_mean_CV_RMSE=float(b.RMSE.mean()), baseline_mean_CV_R2=float(b.R2.mean()),
                runtime_reduction=float(1.0 - g.runtime.mean() / b.runtime.mean()),
                rmse_not_worse=bool(g.RMSE.mean() <= b.RMSE.mean())))
            save_csv(ROOT / "config_summary_running.csv", pd.DataFrame(summary_rows))
            print(f"CONFIG alpha={alpha:g} tau={tau:g} complete", flush=True)
    summary = pd.DataFrame(summary_rows).sort_values(["mean_CV_RMSE", "alpha", "tau"])
    save_csv(ROOT / "gst_alpha_tau_summary.csv", summary)
    # The locked manuscript protocol selects the minimum mean training-only
    # CV RMSE. Runtime is reported as an efficiency result and is not used to
    # override the accuracy-based parameter choice.
    selected = summary.sort_values(["mean_CV_RMSE", "alpha", "tau"], ascending=[True, True, True]).iloc[0]
    selection_rule = "minimum mean training-only CV RMSE; alpha then tau are deterministic tie-breaks"
    save_json(ROOT / "selected_alpha_tau.json", dict(selected=selected.to_dict(), rule=selection_rule,
        training_only=True, k=K, alphas=ALPHAS, taus=TAUS, seeds=SEEDS))
    print(summary.to_string(index=False), flush=True)
    print("SELECTED", selected.to_dict(), flush=True)


if __name__ == "__main__":
    main()
