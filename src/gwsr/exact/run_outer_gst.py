from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import train_test_split
import select_alpha_tau_exact as exp

ROOT = Path(os.environ.get(
    "GST_GWSR_ALPHA_TAU_OUTPUT",
    str(Path(__file__).resolve().parents[3] / "run_outputs" / "alpha_tau_selection"),
))
OUT = ROOT / "outer_test_selected"
SEEDS = exp.SEEDS

def main() -> None:
    selected_path = ROOT / "selected_alpha_tau.json"
    if not selected_path.exists():
        selected_path = ROOT / "selected_alpha_tau_final_accuracy.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    alpha, tau = float(selected["alpha"]), float(selected["tau"])
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(exp.INPUT); frame["row_id"] = np.arange(len(frame))
    train_ids, test_ids = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=exp.OUTER_SEED)
    outer_train = frame.iloc[train_ids].copy().reset_index(drop=True)
    outer_test = frame.iloc[test_ids].copy().reset_index(drop=True)
    query = outer_test.drop(columns=[exp.base.TARGET]).copy()
    prepared = exp.prepare_fold(outer_train, query)
    sg, sx = exp.similarity_matrices(prepared); sim = alpha * sg + (1.0 - alpha) * sx
    order = exp.build_order(sim, query[exp.base.COORDS].to_numpy(dtype=float), tau)
    by_chain = {}
    for row in order: by_chain.setdefault(int(row["chain_id"]), []).append(row)
    rows, tables = [], []
    for seed in SEEDS:
        started = time.perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            chunks = Parallel(n_jobs=exp.N_JOBS, backend="loky")(
                delayed(exp.run_chain)(prepared, by_chain[cid], seed) for cid in sorted(by_chain))
        outputs = [x for chunk in chunks for x in chunk]; outputs.sort(key=lambda x: x["task_id"])
        obs = outer_test[exp.base.TARGET].to_numpy(dtype=float)
        table = outer_test[["row_id", *exp.base.COORDS]].copy(); table["q_obs"] = obs
        table["q_pred"] = [x["prediction"] for x in outputs]; table["residual"] = obs-table.q_pred
        table["protected"] = [x["protected"] for x in outputs]; table["expression"] = [x["expression"] for x in outputs]
        table["strategy"] = [x["strategy"] for x in outputs]; table["parent"] = [x["parent"] for x in outputs]
        table["similarity"] = [x["similarity"] for x in outputs]; table["seed"] = seed
        table.to_csv(OUT / f"predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
        row = exp.metrics(obs, table.q_pred.to_numpy()); row.update(seed=seed, runtime_seconds=time.perf_counter()-started,
            transfer_count=int((table.strategy == "transfer_adaptation").sum()), protected_count=int(table.protected.sum()))
        rows.append(row); tables.append(table); print(json.dumps(row, ensure_ascii=False), flush=True)
    stack = np.stack([t.q_pred.to_numpy(dtype=float) for t in tables]); obs = tables[0].q_obs.to_numpy(dtype=float)
    ensembles = {}
    for name, pred in [("mean_ensemble", stack.mean(axis=0)), ("median_ensemble", np.median(stack, axis=0))]:
        row = exp.metrics(obs, pred); row.update(seed=-1); ensembles[name] = row
        pd.DataFrame({"row_id": tables[0].row_id, "q_obs": obs, "q_pred": pred}).to_csv(OUT/f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    manifest = dict(protocol="strict outer test with training-only alpha/tau selection", n_original=len(frame),
        n_outer_train_raw=len(outer_train), n_outer_train_clean=len(prepared["train"]), n_outer_test=len(outer_test),
        split_seed=exp.OUTER_SEED, k=exp.K, seeds=list(SEEDS), kernel="adaptive_bisquare", similarity="distance + feature",
        alpha=alpha, transfer_threshold=tau, population=exp.POPULATION, full_generations=exp.FULL_GENERATIONS,
        short_generations=exp.SHORT_GENERATIONS, test_response_used_for_selection=False,
        test_response_attached_after_search=True,
        protection_rule="nonfinite or absolute standardized prediction >= 5; local weighted-mean fallback; log clip [-5,10]",
        selected_source=str(selected_path), metrics=rows, ensembles=ensembles)
    (OUT/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT/"metrics_by_seed.csv", index=False); (OUT/"ensemble_metrics.json").write_text(json.dumps(ensembles, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
