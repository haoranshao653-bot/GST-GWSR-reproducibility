"""Training-only spatial CV: GST coarse screening, pure GWSR confirmation.

Run this file in PyCharm. No final held-out predictions are made by this script.
Dependencies: accompanying gst_v2.py, gst_gwsr_core.py and requirements.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_config, cpu_count, effective_n_jobs

import gst_v2 as gst
import gst_gwsr_core as core

FEATURES = ['topo','bouguer','sediment','moho','lab','cpd','lithos_km','volcano']
MASTER_K = [20,25,30,35,40,45,50,60,80,100,125,150]
COARSE_K = [20,30,40,50,60,80,100,150]
CONFIRM_SEEDS = [42,123,456]
COARSE_GP_SEED = 42
N_SPLITS, SPATIAL_FOLD_SEED = 5, 2026
EPS, SST_ATOL, SST_RTOL = 1e-12, 1e-12, 1e-10


def parallel_resources():
    return dict(logical_cpu_count=os.cpu_count(),joblib_available_cpu_count=cpu_count(),
        stageA_concurrent_folds=min(N_SPLITS,effective_n_jobs(-1)),
        stageB_worker_capacity=effective_n_jobs(-1),inner_native_threads=1)


def observed_concurrency(starts, ends):
    events=[(float(t),1) for t in starts]+[(float(t),-1) for t in ends]
    current=peak=0
    for _,delta in sorted(events):
        current+=delta; peak=max(peak,current)
    return peak


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    # 先写临时文件再替换，checkpoint的JSON是最后完成标记。
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def write_csv(path, frame):
    temp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp,index=False,encoding='utf-8-sig',float_format='%.17g',na_rep='NaN')
    temp.replace(path)


def build_bisquare_tasks(X_train, train_coords, X_target, target_coords, config, original_rows=None):
    if len(X_train)<config.adaptive_k:
        raise ValueError('k exceeds fold inner-training sample count.')
    distances=core.haversine_distance_km(target_coords,train_coords)
    h=np.partition(distances,config.adaptive_k-1,axis=1)[:,config.adaptive_k-1]
    # 严格双平方：h外权重为0，第k邻居恰处边界；零带宽视作明确失败。
    if not np.isfinite(h).all() or np.any(h<=EPS):
        raise ValueError('Zero/nonfinite adaptive bandwidth: inspect duplicate coordinates.')
    ratio=distances/h[:,None]
    u=np.zeros_like(ratio)
    inside=ratio<1
    u[inside]=(1-ratio[inside]**2)**2
    totals=u.sum(axis=1)
    if not np.isfinite(u).all() or np.any(totals<=EPS):
        raise ValueError('Invalid/zero bisquare weight sum; no uniform-weight fallback allowed.')
    weights=u/totals[:,None]
    if np.any(weights<0) or not np.allclose(weights.sum(axis=1),1):
        raise ValueError('Normalized weights failed validation.')
    tasks=[]
    for i,w in enumerate(weights):
        mu=w@X_train
        std=np.sqrt(np.maximum(np.sum(w[:,None]*(X_train-mu)**2,axis=0),0))
        tasks.append(gst.LocalTask(i,target_coords[i].copy(),X_target[i].copy(),float(h[i]),
            w,mu,std,np.r_[mu,std],i if original_rows is None else int(original_rows[i])))
    return tasks,distances,weights


class StrictEngine(gst.GPEngine):
    """原生GP算子不变；增加可核对种群签名、计数及显式重试。"""
    def __init__(self,*args,inverse,**kwargs):
        self.audit={}
        self.inverse=inverse
        self.search_active=False
        self.capture_initial=False
        super().__init__(*args,**kwargs)

    def begin_task(self,task):
        if getattr(self,'task',None) is not None and self.task.task_id in self.audit:
            self.audit[self.task.task_id]['task_finished_epoch']=time.time()
        super().begin_task(task)
        if task is not None:
            self.audit[task.task_id]=dict(gp_executed=False,generations_executed=0,
                full_retry_count=0,initial_population_signature=None,initial_expressions=[],
                retry_reason='',search_failure_count=0,worker_pid=os.getpid(),
                task_started_epoch=time.time(),task_finished_epoch=None)
            if hasattr(self,'progress_label') and (task.task_id%10==0):
                print(f'{self.progress_label}: local task {task.task_id+1}/{self.progress_total} '
                    f'(sequential GST pool, PID={os.getpid()})',flush=True)

    def generation(self,parents,rng,size):
        pop=super().generation(parents,rng,size)
        if self.search_active:
            info=self.audit[self.task.task_id]
            info['gp_executed']=True
            info['generations_executed']+=1
            if self.capture_initial and parents is None and info['initial_population_signature'] is None:
                texts=[str(p) for p in pop[:10]]
                info['initial_expressions']=texts
                info['initial_population_signature']=hashlib.md5('||'.join(texts).encode()).hexdigest()
        if not any(np.isfinite(p.fitness_) for p in pop):
            raise ValueError('Entire generated population invalid.')
        return pop

    def search(self,*args,**kwargs):
        self.search_active=True
        try:
            result=super().search(*args,**kwargs)
            if gst.safety_reason(result,np.inf) is not None:
                raise ValueError('Invalid GP expression/fitness/prediction.')
            train_prediction=result.best_individual.execute(self.X)
            if not np.isfinite(train_prediction).all():
                raise ValueError('Nonfinite training prediction from best expression.')
            # 不进行旧代码的clip或加权均值替换；逆变换溢出也明确失败。
            if not np.isfinite(self.inverse(np.array([result.prediction]))).all():
                raise ValueError('Nonfinite prediction after inverse transform.')
            return result
        finally:
            self.search_active=False

    def run_full_search(self,task,seed):
        self.capture_initial=True
        info=self.audit[task.task_id]
        try:
            for attempt in range(2):
                used_seed=seed if attempt==0 else gst.task_seed(seed,task.task_id,99)
                try:
                    return self.search(task,self.config.full_generations,used_seed)
                except (MemoryError,KeyboardInterrupt):
                    raise
                except Exception as exc:
                    info['search_failure_count']+=1
                    info['retry_reason']=str(exc)
                    if attempt==0:
                        info['full_retry_count']+=1
                    else:
                        raise RuntimeError(f'Task {task.task_id}: full GP failed twice: {exc}') from exc
        finally:
            self.capture_initial=False

    def run_short_search(self,task,initial_population,seed):
        try:
            return self.search(task,self.config.short_generations,seed,initial_population)
        except (MemoryError,KeyboardInterrupt):
            raise
        except Exception as exc:
            self.audit[task.task_id]['search_failure_count']+=1
            # Runner明确捕获ValueError后独立完整搜索，不返回常数预测。
            raise ValueError(f'Short GP failed: {exc}') from exc

    def prepare_transfer_population(self,*args,**kwargs):
        try:
            return super().prepare_transfer_population(*args,**kwargs)
        except (MemoryError,KeyboardInterrupt):
            raise
        except Exception as exc:
            raise ValueError(f'Transfer preparation failed: {exc}') from exc


def load_clean_training(path):
    frame=pd.read_csv(path)
    if len(frame)!=1024:
        raise ValueError(f'Expected original 1024 stations, got {len(frame)}. Do not silently change split.')
    outer_ids,heldout_ids=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42)
    if (len(outer_ids),len(heldout_ids))!=(819,205):
        raise ValueError('Unexpected outer split.')
    # 只按行号划分；此后不读取205行的响应、坐标、特征，不交给选k函数。
    raw=frame.iloc[outer_ids][['lon','lat',*FEATURES,'heat_flow']].copy()
    raw.insert(0,'original_index',outer_ids)
    del frame,heldout_ids
    values=raw[['lon','lat',*FEATURES,'heat_flow']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite outer-training data; explicit cleaning policy required.')
    q1,q3=raw.heat_flow.quantile([.25,.75]); iqr=q3-q1
    clean=raw[raw.heat_flow.between(q1-3*iqr,q3+3*iqr)].reset_index(drop=True)
    if len(clean)!=815 or (clean.heat_flow<=0).any():
        raise ValueError(f'Expected 815 positive-target cleaned training stations, got {len(clean)}.')
    return clean


def spatial_folds(coords):
    # 经纬度转单位球三维坐标再聚类，不把经纬度直接当平面距离。
    lon,lat=np.radians(coords[:,0]),np.radians(coords[:,1])
    sphere=np.c_[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]
    folds=KMeans(n_clusters=N_SPLITS,random_state=SPATIAL_FOLD_SEED,n_init=20).fit_predict(sphere)
    if len(np.unique(folds))!=N_SPLITS:
        raise ValueError('Spatial clustering did not yield five nonempty folds.')
    return folds


def pooled_metrics(y,pred):
    if len(y)!=len(pred) or not np.isfinite(pred).all():
        raise ValueError('Missing/nonfinite OOF predictions.')
    base=core.metric_summary(np.asarray(y),np.asarray(pred))
    return dict(pooled_CV_R2=base['R2'],pooled_CV_RMSE=base['RMSE'],
        pooled_CV_NRMSE=base['RMSE']/float(np.mean(y)),pooled_CV_MAE=base['MAE'],
        pooled_CV_MAPE=base['MAPE_percent'])


def pure_task_worker(task,xfit,yfit,cfg,inverse,guard_z=2.5):
    """Loky进程中的单个独立GWSR任务，不启动内层进程或共享GST pool。"""
    started=time.time()
    engine=StrictEngine(xfit,yfit,cfg,feature_names=FEATURES,inverse=inverse)
    engine.begin_task(task)
    full,transfer,probe=[gst.task_seed(cfg.master_seed,task.task_id,i) for i in range(3)]
    record=engine.run_full_search(task,full)
    record.full_seed,record.transfer_seed,record.probe_seed=full,transfer,probe
    record.n_fitness_evaluations=engine.evaluations
    record.n_fitness_requests=engine.requests
    record.n_tree_executions=engine.executions
    local_mean=float(task.weights @ yfit)
    unsafe=(not np.isfinite(record.prediction)) or abs(float(record.prediction))>guard_z
    if unsafe:
        record.prediction=float(local_mean)
        record.fallback_used=True
        record.fallback_reason=f'prediction_guard_abs_z>{guard_z}'
    record.prediction_original=float(inverse(np.array([record.prediction]))[0])
    record.runtime_seconds=time.time()-started
    audit=engine.audit[task.task_id]
    audit['task_started_epoch']=started
    audit['task_finished_epoch']=time.time()
    # Pure任务不迁移种群；避免把1000个体通过IPC回传，仅保留最佳个体诊断。
    record.final_population=[]
    return record,audit


def make_fold_predictions(X,y,coords,original_ids,folds,fold,k,seed,stage,settings,limit=None):
    fold_started=time.time()
    fit_ids=np.flatnonzero(folds!=fold); query_ids=np.flatnonzero(folds==fold)
    if limit is not None:
        query_ids=query_ids[:limit]
    if set(fit_ids)&set(query_ids):
        raise RuntimeError('Inner train/validation overlap.')
    xs,ys=StandardScaler(),StandardScaler()
    xfit=xs.fit_transform(X[fit_ids]); xquery=xs.transform(X[query_ids])
    yfit=ys.fit_transform(np.log(y[fit_ids]).reshape(-1,1)).ravel()
    def inverse(z):
        with np.errstate(over='ignore',invalid='ignore'):
            return np.exp(ys.inverse_transform(np.asarray(z).reshape(-1,1)).ravel())
    # 原始seed+k+fold派生，Runner再按task_id/search_type派生；无全局seed覆盖。
    derived=int(np.random.SeedSequence([seed,k,int(fold)]).generate_state(1)[0])
    cfg=gst.GSTConfig(adaptive_k=k,alpha=settings['alpha'],
        transfer_threshold=settings['threshold'],population_size=settings['population'],
        full_generations=settings['full_generations'],short_generations=settings['short_generations'],
        master_seed=derived,distance_metric='haversine_km',use_expression_cache=True)
    print(f'Stage {stage} k={k} seed={seed} fold={fold+1}/5 START: '
        f'{len(query_ids)} locations, coordinator PID={os.getpid()}',flush=True)
    if stage=='A':
        # 外层独立fold由Loky并行；本fold内必须维持任务顺序，不再调用Parallel。
        engine=StrictEngine(xfit,yfit,cfg,feature_names=FEATURES,inverse=inverse)
        engine.progress_label=f'A k={k} fold={fold+1}'
        engine.progress_total=len(query_ids)
        runner=gst.GSTGWSRRunner(xfit,yfit,coords[fit_ids],cfg,gp_engine=engine,
            y_transformer=inverse,task_builder=build_bisquare_tasks)
        try:
            result=runner.fit_predict(coords[query_ids],xquery,
                original_row_indices=original_ids[query_ids],use_gst=True)
        except Exception as exc:
            exc.task_audit=pd.DataFrame([dict(stage=stage,k=k,fold=int(fold),seed=seed,
                task_id=i,original_index=int(original_ids[query_ids[i]]),**info)
                for i,info in engine.audit.items()])
            raise
        audits=engine.audit
        audits[len(query_ids)-1]['task_finished_epoch']=time.time()
    else:
        # Stage B的调用者只逐折执行；仅此处的validation locations并行。
        tasks,_,_=build_bisquare_tasks(xfit,coords[fit_ids],xquery,coords[query_ids],cfg,
            original_ids[query_ids])
        records=[]; audits={}
        with parallel_config(backend='loky',inner_max_num_threads=1):
            generated=Parallel(n_jobs=-1,backend='loky',return_as='generator_unordered')(
                delayed(pure_task_worker)(task,xfit,yfit,cfg,inverse,settings['guard_z']) for task in tasks)
            for record,audit in generated:
                records.append(record); audits[record.task_id]=audit
                if len(records)%10==0 or len(records)==len(tasks):
                    print(f'B k={k} seed={seed} fold={fold+1}: '
                        f'{len(records)}/{len(tasks)} locations finished',flush=True)
        records.sort(key=lambda r:r.task_id)
        result=gst.RunResults(records,tasks,{})
    rows=[]
    for record,task,query_id in zip(result.records,result.tasks,query_ids):
        # Apply the same training-only prediction gate to GST and pure-GWSR
        # validation.  The gate uses only the current task's weighted mean in
        # transformed training-response space; validation responses are still
        # unavailable at this point.
        if (not np.isfinite(record.prediction)) or abs(float(record.prediction))>settings['guard_z']:
            record.prediction=float(task.weights @ yfit)
            record.fallback_used=True
            record.fallback_reason=f'prediction_guard_abs_z>{settings["guard_z"]}'
            record.prediction_original=float(inverse(np.array([record.prediction]))[0])
        pred_fit=record.best_individual.execute(xfit)
        w=task.weights
        mean=float(w@yfit); sst=float(w@((yfit-mean)**2)); sse=float(w@((yfit-pred_fit)**2))
        tol=SST_ATOL+SST_RTOL*float(np.var(yfit))
        unstable=sst<=tol
        local_r2=np.nan if unstable else 1-sse/sst
        if not np.isfinite(local_r2) and not unstable:
            unstable=True; local_r2=np.nan
        std=float(np.std(pred_fit)); constant=bool(std<=1e-10)
        weighted_mean_flag=bool(constant and np.allclose(pred_fit,mean,rtol=1e-8,atol=1e-10))
        audit=audits[task.task_id]
        if not audit['gp_executed'] or audit['generations_executed']<1 or record.n_fitness_evaluations<1:
            raise RuntimeError('GP execution/fitness count assertion failed.')
        if stage=='B' and (record.attempted_transfer or record.used_transfer):
            raise RuntimeError('Transfer is forbidden in Stage B.')
        rows.append(dict(stage=stage,k=k,fold=int(fold),seed=seed,task_id=task.task_id,
            training_row=int(query_id),original_index=int(original_ids[query_id]),
            bandwidth_km=task.bandwidth,nonzero_weight_count=int(np.count_nonzero(w)),
            ESS=float(1/(w@w)),weight_sum=float(w.sum()),weight_max=float(w.max()),
            weighted_SST=sst,weighted_SSE=sse,weighted_RMSE=float(np.sqrt(sse)),
            weighted_MAE=float(w@np.abs(yfit-pred_fit)),local_R2=local_r2,
            local_R2_unstable=unstable,SST_TOL=tol,gp_executed=True,
            generations_executed=audit['generations_executed'],fitness_evaluations=record.n_fitness_evaluations,
            fitness_requests=record.n_fitness_requests,tree_executions=record.n_tree_executions,
            fallback_used=record.fallback_used,fallback_reason=record.fallback_reason,
            attempted_transfer=record.attempted_transfer,used_transfer=record.used_transfer,
            parent_task_id=record.parent_task_id,W=record.W,F=record.F,S=record.S,
            constant_expression=constant,prediction_std=std,weighted_mean_prediction_flag=weighted_mean_flag,
            best_fitness=record.best_fitness,best_expression=record.best_expression,
            best_expression_exact=json.dumps(gst.expression_key(record.best_individual)),
            prediction=record.prediction_original,prediction_transformed=record.prediction,
            runtime=record.runtime_seconds,full_seed=record.full_seed,transfer_seed=record.transfer_seed,
            worker_pid=audit['worker_pid'],task_started_epoch=audit['task_started_epoch'],
            task_finished_epoch=audit['task_finished_epoch'],
            probe_seed=record.probe_seed,retry_count=audit['full_retry_count'],retry_reason=audit['retry_reason'],
            search_failure_count=audit['search_failure_count'],silent_baseline_fallback=False,
            initial_population_signature=audit['initial_population_signature'],
            initial_expressions=json.dumps(audit['initial_expressions'])))
    table=pd.DataFrame(rows)
    # 直到预测和安全判定全部结束，才引入本折验证响应。
    table['observed']=y[query_ids]
    table['fold_started_epoch']=fold_started
    table['fold_finished_epoch']=time.time()
    table['fold_runtime_seconds']=table.fold_finished_epoch-fold_started
    table['fold_coordinator_pid']=os.getpid()
    table['parallel_unit']='fold' if stage=='A' else 'local_task'
    table['configured_worker_capacity']=(parallel_resources()['stageA_concurrent_folds']
        if stage=='A' else effective_n_jobs(-1))
    if table.weighted_mean_prediction_flag.mean()>=.2:
        warnings.warn(f'{stage} k={k} fold={fold} seed={seed}: >=20% GP constants match weighted mean; inspect diagnostics.')
    print(f'Stage {stage} k={k} seed={seed} fold={fold+1}/5 DONE: '
        f'{table.fold_runtime_seconds.iloc[0]:.1f}s, '
        f'worker PIDs={table.worker_pid.nunique()}',flush=True)
    return table


def validate_oof(table,n,folds,stage,k,seed):
    if len(table)!=n or sorted(table.training_row.tolist())!=list(range(n)):
        raise RuntimeError('OOF must contain every cleaned training row exactly once.')
    ordered=table.sort_values('training_row')
    if not np.array_equal(ordered.fold.to_numpy(),folds):
        raise RuntimeError('OOF fold assignment mismatch.')
    if not ((table.stage==stage)&(table.k==k)&(table.seed==seed)).all():
        raise RuntimeError('Cross-stage/k/seed checkpoint reuse detected.')
    if not table.gp_executed.all() or table.silent_baseline_fallback.any():
        raise RuntimeError('GP execution or silent fallback violation.')
    for col in ['prediction','best_fitness','bandwidth_km','weight_sum']:
        if not np.isfinite(table[col].to_numpy(float)).all():
            raise RuntimeError(f'Nonfinite {col}.')
    if not np.allclose(table.weight_sum,1) or table.best_expression.isna().any():
        raise RuntimeError('Weight/expression audit failed.')
    return ordered.reset_index(drop=True)


class Experiment:
    def __init__(self,clean,settings,out,resume):
        self.X=clean[FEATURES].to_numpy(float); self.y=clean.heat_flow.to_numpy(float)
        self.coords=clean[['lon','lat']].to_numpy(float); self.ids=clean.original_index.to_numpy(int)
        self.folds=spatial_folds(self.coords); self.settings=settings; self.out=out; self.resume=resume
        self.n=len(clean)
        if any(np.sum(self.folds!=f)<max(MASTER_K) for f in range(N_SPLITS)):
            raise ValueError('A spatial fold has too few training samples for MASTER_K.')
        source_dir = Path(__file__).resolve().parent
        sources={p:hashlib.sha256((source_dir / p).read_bytes()).hexdigest()
            for p in ['select_k_exact.py','gst_v2.py','gst_gwsr_core.py']}
        metadata=dict(data_hash=hashlib.sha256(clean.to_csv(index=False).encode()).hexdigest(),
            fold_assignment_hash=digest(self.folds.tolist()),settings=settings,
            gp_fixed=dict(population=settings['population'],full_generations=settings['full_generations'],
                short_generations=settings['short_generations'],max_samples=1.0,lambda_=0.01,
                crossover=.7,subtree_mutation=.1,hoist_mutation=.05,point_mutation=.1,
                init_depth=[2,5],tournament=20,operators=list(core.FUNCTION_SET),
                final_selection='minimum penalized Q in final generation',early_stopping=False),
            master_k=MASTER_K,coarse_k=COARSE_K,seeds=CONFIRM_SEEDS,kernel='adaptive_bisquare',
            spatial_folds=5,spatial_seed=2026,distance='haversine_km',code_hashes=sources,
            versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','gplearn','joblib']},
            python=platform.python_version())
        metadata['parallel_execution']=parallel_resources()
        metadata['parallel_policy']='A: folds only; B: sequential folds with loky local tasks; native threads=1'
        metadata['gp_config_hash']=digest(metadata['gp_fixed'])
        self.fingerprint=digest(metadata)
        manifest=out/'experiment_manifest.json'
        if manifest.exists():
            old=json.loads(manifest.read_text(encoding='utf-8'))
            if not resume or old['fingerprint']!=self.fingerprint:
                raise RuntimeError('Output exists or configuration/data/code changed. Use a new output directory.')
        elif out.exists() and any(out.iterdir()):
            raise RuntimeError('Nonempty output without valid manifest.')
        out.mkdir(parents=True,exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)
        if not manifest.exists():
            write_json(manifest,dict(fingerprint=self.fingerprint,**metadata))
        folds_frame=clean[['original_index','lon','lat']].copy(); folds_frame['spatial_fold']=self.folds
        write_csv(out/'training_spatial_folds.csv',folds_frame)
        self.tables={}

    def fold(self,stage,k,seed,f,limit=None):
        table=make_fold_predictions(self.X,self.y,self.coords,self.ids,self.folds,
            f,k,seed,stage,self.settings,limit)
        return table

    def self_test(self):
        tables=[self.fold('B',40,s,0,2) for s in CONFIRM_SEEDS]
        signatures=[str(t.iloc[0].initial_population_signature) for t in tables]
        if len(set(signatures))!=3 or any(s in ('None','nan','') for s in signatures):
            raise RuntimeError('SELF TEST: initial seed populations are not distinct.')
        for t in tables:
            if len(t)!=2 or not t.gp_executed.all() or not np.isfinite(t.prediction).all():
                raise RuntimeError('SELF TEST: execution/prediction check failed.')
            if not np.allclose(t.weight_sum,1) or not np.isfinite(t.best_fitness).all():
                raise RuntimeError('SELF TEST: weights/fitness failed.')
        write_csv(self.out/'self_test_diagnostics.csv',pd.concat(tables,ignore_index=True))
        print('SELF TEST PASSED',flush=True)

    def run_setting(self,stage,k,seed):
        stem=f'stage{stage}_k{k}'+(f'_seed{seed}' if stage=='B' else '')
        csv_path=self.out/'checkpoints'/f'{stem}.csv'; json_path=csv_path.with_suffix('.json')
        started=time.perf_counter()
        if self.resume and csv_path.exists() and json_path.exists():
            meta=json.loads(json_path.read_text(encoding='utf-8'))
            if (meta['fingerprint']!=self.fingerprint or meta['k']!=k or meta['seed']!=seed
                or meta['stage']!=stage or meta['csv_hash']!=hashlib.sha256(csv_path.read_bytes()).hexdigest()):
                raise RuntimeError('Checkpoint hash/key mismatch; refusing reuse.')
            table=pd.read_csv(csv_path,float_precision='round_trip')
            table=validate_oof(table,self.n,self.folds,stage,k,seed)
            runtime=meta['runtime_seconds']
            print(f'Resume verified {stem}',flush=True)
        else:
            pieces=[]
            try:
                if stage=='A':
                    # 五个fold各持有独立GST pool；禁止内层并行。
                    with parallel_config(backend='loky',inner_max_num_threads=1):
                        pieces=Parallel(n_jobs=parallel_resources()['stageA_concurrent_folds'],
                            backend='loky',verbose=10)(
                            delayed(make_fold_predictions)(self.X,self.y,self.coords,self.ids,
                                self.folds,f,k,seed,stage,self.settings) for f in range(N_SPLITS))
                else:
                    # 这里必须是普通for，折内由make_fold_predictions启动位置并行。
                    for f in range(N_SPLITS):
                        pieces.append(self.fold(stage,k,seed,f))
                table=validate_oof(pd.concat(pieces,ignore_index=True),self.n,self.folds,stage,k,seed)
            except Exception as exc:
                if hasattr(exc,'task_audit'):
                    write_csv(self.out/'checkpoints'/f'{stem}_failed_fold_audit.csv',exc.task_audit)
                if pieces:
                    write_csv(self.out/'checkpoints'/f'{stem}_partial_invalid.csv',pd.concat(pieces))
                write_json(self.out/'checkpoints'/f'{stem}_invalid.json',dict(
                    stage=stage,k=k,seed=seed,valid=False,invalid_reason=str(exc),fingerprint=self.fingerprint))
                raise RuntimeError(f'{stem} invalid; experiment stopped, no fake predictions: {exc}') from exc
            runtime=time.perf_counter()-started
            write_csv(csv_path,table)
            write_json(json_path,dict(fingerprint=self.fingerprint,k=k,seed=seed,stage=stage,
                kernel='adaptive_bisquare',runtime_seconds=runtime,valid=True,
                csv_hash=hashlib.sha256(csv_path.read_bytes()).hexdigest()))
        self.tables[(stage,k,seed)]=table
        self.save_parallel_audit()
        score=pooled_metrics(self.y,table.prediction.to_numpy())
        stable=table.loc[~table.local_R2_unstable,'local_R2'].median()
        print(f'k={k} seed={seed}: CV RMSE={score["pooled_CV_RMSE"]:.5f}, '
            f'R2={score["pooled_CV_R2"]:.5f}, MAE={score["pooled_CV_MAE"]:.5f}, '
            f'ESS={table.ESS.mean():.2f}, search failures={table.search_failure_count.sum()}, '
            f'time={runtime:.1f}s, median stable local R2={stable:.4g}, '
            f'unstable={table.local_R2_unstable.sum()}',flush=True)
        attempts=int(table.attempted_transfer.sum())
        summary=dict(k=k,seed=seed,**score,mean_nonzero_count=float(table.nonzero_weight_count.mean()),
            mean_ESS=float(table.ESS.mean()),min_ESS=float(table.ESS.min()),
            transfer_attempt_rate=float(table.attempted_transfer.mean()),
            transfer_accept_rate=float(table.used_transfer.sum()/attempts) if attempts else 0.,
            fallback_rate=float(table.fallback_used.sum()/attempts) if attempts else 0.,
            total_fitness_evaluations=int(table.fitness_evaluations.sum()),runtime_seconds=runtime,
            valid=True,invalid_reason='')
        return table,summary

    def save_parallel_audit(self):
        rows=[]
        for (stage,k,seed),table in self.tables.items():
            for fold,group in table.groupby('fold'):
                peak=observed_concurrency(group.task_started_epoch,group.task_finished_epoch)
                rows.append(dict(stage=stage,k=k,seed=seed,fold=int(fold),
                    fold_runtime_seconds=float(group.fold_runtime_seconds.iloc[0]),
                    fold_started_epoch=float(group.fold_started_epoch.iloc[0]),
                    fold_finished_epoch=float(group.fold_finished_epoch.iloc[0]),
                    fold_coordinator_pid=int(group.fold_coordinator_pid.iloc[0]),
                    local_task_count=len(group),
                    serial_local_task_count=len(group) if stage=='A' else 0,
                    parallel_submitted_local_task_count=len(group) if stage=='B' else 0,
                    observed_local_worker_count=int(group.worker_pid.nunique()),
                    observed_peak_local_tasks=peak,
                    configured_worker_capacity=int(group.configured_worker_capacity.iloc[0]),
                    worker_pids=','.join(map(str,sorted(group.worker_pid.unique())))))
                if stage=='A' and (peak>1 or group.worker_pid.nunique()!=1):
                    raise RuntimeError('Stage A local tasks unexpectedly parallelized.')
        if not rows:
            return
        audit=pd.DataFrame(rows)
        write_csv(self.out/'parallel_fold_audit.csv',audit)
        summary=[]
        for (stage,k,seed),group in audit.groupby(['stage','k','seed']):
            peak=observed_concurrency(group.fold_started_epoch,group.fold_finished_epoch)
            if stage=='B' and peak>1:
                raise RuntimeError('Stage B folds overlapped: nested parallelism violation.')
            summary.append(dict(stage=stage,k=int(k),seed=int(seed),
                observed_peak_concurrent_folds=peak,
                observed_fold_process_count=int(group.fold_coordinator_pid.nunique()),
                serial_local_task_count=int(group.serial_local_task_count.sum()),
                parallel_submitted_local_task_count=int(group.parallel_submitted_local_task_count.sum()),
                peak_local_tasks=int(group.observed_peak_local_tasks.max()),
                total_fold_runtime_seconds=float(group.fold_runtime_seconds.sum())))
        write_csv(self.out/'parallel_execution_summary.csv',pd.DataFrame(summary))

    def save_diagnostics(self,stage):
        tables=[t for (st,_,_),t in self.tables.items() if st==stage]
        if tables:
            combined=pd.concat(tables,ignore_index=True)
            write_csv(self.out/f'stage{stage}_task_diagnostics.csv',combined)
            write_csv(self.out/('stageA_gst_coarse_oof.csv' if stage=='A' else 'stageB_gwsr_confirm_oof.csv'),
                combined[['stage','k','fold','seed','task_id','original_index','observed','prediction']])

    def confirm(self,k):
        tables=[]; stats=[]
        for seed in CONFIRM_SEEDS:
            table,summary=self.run_setting('B',k,seed); tables.append(table); stats.append(summary)
            self.save_diagnostics('B')
        for f in range(N_SPLITS):
            sig=[str(t[t.fold==f].sort_values('task_id').iloc[0].initial_population_signature) for t in tables]
            if len(set(sig))!=3 or any(s in ('nan','None','') for s in sig):
                raise RuntimeError(f'Seed signatures not distinct at k={k}, fold={f}.')
        same_prediction=all(np.max(np.abs(tables[0].prediction.to_numpy()-t.prediction.to_numpy()))<1e-12
            for t in tables[1:])
        same_expression=all(np.array_equal(tables[0].best_expression_exact.to_numpy(),
            t.best_expression_exact.to_numpy()) for t in tables[1:])
        allrows=pd.concat(tables,ignore_index=True)
        result=dict(k=k,valid=True,mean_ESS=float(allrows.ESS.mean()),min_ESS=float(allrows.ESS.min()),
            mean_nonzero_count=float(allrows.nonzero_weight_count.mean()),n_gp_tasks=len(allrows),
            n_gp_failures=int(allrows.search_failure_count.sum()),
            n_constant_expressions=int(allrows.constant_expression.sum()),
            n_unstable_local_R2=int(allrows.local_R2_unstable.sum()),
            mean_fitness_evaluations=float(allrows.fitness_evaluations.mean()),
            total_fitness_evaluations=int(allrows.fitness_evaluations.sum()),
            runtime_seconds=float(sum(s['runtime_seconds'] for s in stats)),
            identical_oof_predictions=same_prediction,
            converged_to_same_solution=bool(same_prediction and same_expression),
            identical_predictions_different_expressions=bool(same_prediction and not same_expression))
        for metric in ['RMSE','R2','MAE','MAPE','NRMSE']:
            vals=[s['pooled_CV_'+metric] for s in stats]
            result['mean_CV_'+metric]=float(np.mean(vals)); result['std_CV_'+metric]=float(np.std(vals,ddof=1))
            for seed,val in zip(CONFIRM_SEEDS,vals):
                result[f'CV_{metric}_seed{seed}']=val
        return result


def choose_best(rows):
    minimum=min(r['mean_CV_RMSE'] for r in rows)
    ties=[r for r in rows if np.isclose(r['mean_CV_RMSE'],minimum,rtol=1e-4,atol=1e-6)]
    return max(ties,key=lambda r:r['k'])['k']


def run_selection(clean,settings,out,resume=False):
    """Only cleaned outer-training dataframe is accepted; no outer test arrays."""
    experiment=Experiment(clean,settings,out,resume)
    experiment.self_test()
    coarse=[]
    for k in COARSE_K:
        _,row=experiment.run_setting('A',k,COARSE_GP_SEED); coarse.append(row)
        write_csv(out/'stageA_gst_coarse_summary.csv',pd.DataFrame(coarse).sort_values('pooled_CV_RMSE'))
        experiment.save_diagnostics('A')
    winners=pd.DataFrame(coarse).sort_values(['pooled_CV_RMSE','k'],ascending=[True,False]).head(2).k.tolist()
    candidates=set()
    for k in winners:
        i=MASTER_K.index(k); candidates.update(MASTER_K[max(0,i-1):min(len(MASTER_K),i+2)])
    write_json(out/'stageB_initial_candidates.json',dict(coarse_top_two=winners,candidates=sorted(candidates)))
    confirmed={}; expansions=[]
    while True:
        for k in sorted(candidates-set(confirmed)):
            confirmed[k]=experiment.confirm(k)
            table=pd.DataFrame(confirmed.values()).sort_values('mean_CV_RMSE')
            table['rank']=np.arange(1,len(table)+1)
            write_csv(out/'stageB_gwsr_confirm_summary.csv',table)
        best=choose_best(list(confirmed.values())); index=MASTER_K.index(best); extra=None
        if best==min(candidates) and index>0:
            extra=MASTER_K[index-1]
        elif best==max(candidates) and index<len(MASTER_K)-1:
            extra=MASTER_K[index+1]
        if extra is None:
            break
        expansions.append(dict(boundary_best=best,added_k=extra)); candidates.add(extra)
        write_json(out/'boundary_expansion.json',expansions)
    table=pd.DataFrame(confirmed.values()).sort_values('mean_CV_RMSE')
    table['rank']=np.arange(1,len(table)+1); table['selected_by_tie_rule']=table.k==best
    write_csv(out/'stageB_gwsr_confirm_summary.csv',table)
    write_json(out/'selected_bisquare_k.json',dict(model='GWSR',kernel='adaptive_bisquare',selected_k=best,
        selection_stage='pure_GWSR_confirmation',selection_metric='mean pooled spatial-CV RMSE across three GP seeds',
        cv_folds=5,gp_seeds=CONFIRM_SEEDS,test_set_used=False,clean_training_count=len(clean),
        coarse_top_two=winners,confirmed_candidates=sorted(candidates),boundary_expansions=expansions,
        at_master_grid_boundary=best in [MASTER_K[0],MASTER_K[-1]],
        tie_rule='rtol=1e-4, atol=1e-6; prefer larger k',fingerprint=experiment.fingerprint,
        note='Best among confirmed candidates only; GST screening does not guarantee global best on MASTER_K.'))
    print(f'Selected k={best}. Selection finished; NO outer-test predictions executed.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument('--input',type=Path,default=repo_root / 'data' / 'processed' / 'heat_flow_stations.csv')
    p.add_argument('--output',type=Path,default=repo_root / 'run_outputs' / 'k_selection')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--self-test-only',action='store_true')
    p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--threshold',type=float,default=.7)
    p.add_argument('--population',type=int,default=1000)
    p.add_argument('--full-generations',type=int,default=20)
    p.add_argument('--short-generations',type=int,default=5)
    p.add_argument('--guard-z',type=float,default=2.5,
        help='Training-only transformed-response prediction safety threshold.')
    p.add_argument('--direct-confirm',action='store_true',
        help='Skip GST coarse screening and confirm the supplied k values with pure GWSR only.')
    p.add_argument('--direct-k',type=int,nargs='+',default=None,
        help='Candidate k values for --direct-confirm (default: 20 25 30 35 40 45 50 60).')
    args=p.parse_args()
    resources=parallel_resources()
    print(f'Logical CPUs={resources["logical_cpu_count"]}; '
        f'joblib available CPUs={resources["joblib_available_cpu_count"]}; '
        f'Stage A concurrent folds={resources["stageA_concurrent_folds"]}; '
        f'Stage B worker capacity={resources["stageB_worker_capacity"]}; '
        'native threads per worker=1. Actual worker PIDs/concurrency recorded in audit CSVs.',flush=True)
    if version('gplearn')!='0.4.3':
        raise RuntimeError('This adapter requires gplearn==0.4.3.')
    settings=dict(alpha=args.alpha,threshold=args.threshold,population=args.population,
        full_generations=args.full_generations,short_generations=args.short_generations,
        guard_z=args.guard_z)
    gst.GSTConfig(alpha=args.alpha,transfer_threshold=args.threshold,population_size=args.population,
        full_generations=args.full_generations,short_generations=args.short_generations)
    clean=load_clean_training(args.input)
    if args.self_test_only:
        Experiment(clean,settings,args.output,args.resume).self_test()
    elif args.direct_confirm:
        # This branch is deliberately pure GWSR: no proxy score and no GST
        # screening can influence the selected neighbourhood size.
        exp=Experiment(clean,settings,args.output,args.resume)
        exp.self_test()
        candidates=sorted(set(args.direct_k or [20,25,30,35,40,45,50,60]))
        rows=[]
        for k in candidates:
            row=exp.confirm(k)
            rows.append(row)
            table=pd.DataFrame(rows).sort_values('mean_CV_RMSE')
            table['rank']=np.arange(1,len(table)+1)
            write_csv(args.output/'direct_gwsr_confirmation_summary.csv',table)
            exp.save_diagnostics('B')
        best=choose_best(rows)
        write_json(args.output/'selected_bisquare_k_direct.json',dict(
            model='GWSR',kernel='adaptive_bisquare',selected_k=best,
            selection_stage='pure_GWSR_training_only_spatial_CV',
            selection_metric='mean pooled spatial-CV RMSE across three GP seeds',
            cv_folds=N_SPLITS,gp_seeds=CONFIRM_SEEDS,test_set_used=False,
            candidates=candidates,tie_rule='rtol=1e-4, atol=1e-6; prefer larger k',
            fingerprint=exp.fingerprint))
        print(f'Direct pure-GWSR confirmation selected k={best}; outer test was not read.',flush=True)
    else:
        run_selection(clean,settings,args.output,args.resume)


if __name__=='__main__':
    main()
