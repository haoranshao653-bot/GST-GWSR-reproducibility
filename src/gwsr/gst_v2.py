"""GST_GWSR_ALL_GAUSSIAN_K25: fixed-order, single-parent population transfer.

Reuses gplearn's existing representation and genetic operators through the
project's gst_gwsr_core model configuration. No evaluation targets enter Runner.
"""
from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from gplearn._program import _Program
from gplearn.functions import _function_map
from gplearn.fitness import _fitness_map
from gplearn.genetic import _parallel_evolve
import gst_gwsr_core as core


@dataclass(frozen=True)
class GSTConfig:
    adaptive_k: int = 25
    weight_eps: float = 1e-12
    alpha: float = .5                 # 示例设置，不是已验证最优值
    transfer_threshold: float = .7    # 示例设置，不是已验证最优值
    population_size: int = 1000
    full_generations: int = 20
    short_generations: int = 5
    probe_fraction: float = .1
    min_probe_size: int = 10
    use_expression_cache: bool = True
    max_cache_size: int = 10000
    master_seed: int = 42
    distance_metric: str = 'euclidean_degrees'
    deduplicate_transfer: bool = True

    def __post_init__(self):
        if not 0 <= self.alpha <= 1 or not 0 <= self.transfer_threshold <= 1:
            raise ValueError('alpha and transfer_threshold must be in [0,1].')
        if min(self.adaptive_k, self.population_size, self.full_generations,
               self.short_generations, self.min_probe_size, self.max_cache_size) < 1:
            raise ValueError('Counts must be positive.')
        if self.short_generations >= self.full_generations:
            raise ValueError('short_generations must be less than full_generations.')
        if not 0 < self.probe_fraction <= 1 or not self.weight_eps > 0:
            raise ValueError('Invalid probe_fraction or weight_eps.')
        if self.master_seed < 0:
            raise ValueError('master_seed must be nonnegative.')


@dataclass
class LocalTask:
    task_id: int
    coord: np.ndarray
    x_target: np.ndarray
    bandwidth: float
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    descriptor: np.ndarray
    original_row_index: Any = None


@dataclass
class SearchResult:
    task_id: int
    best_individual: Any
    best_expression: str
    best_fitness: float
    prediction: float                  # transformed-space prediction
    final_population: list
    used_transfer: bool = False
    attempted_transfer: bool = False
    parent_task_id: Optional[int] = None
    W: Optional[float] = None
    F: Optional[float] = None
    S: Optional[float] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    n_generations: int = 0             # 包括失败短程搜索与回退的实际生成代数
    n_fitness_evaluations: int = 0     # 实际重新计算加权误差的次数
    n_fitness_requests: int = 0        # 包括同任务缓存命中
    n_tree_executions: int = 0         # 仅训练数据上的树执行次数
    runtime_seconds: float = 0
    seed: int = 0
    full_seed: int = 0
    transfer_seed: int = 0
    probe_seed: int = 0
    probe_best_fitness: Optional[float] = None
    transfer_best_fitness: Optional[float] = None
    transfer_expression: Optional[str] = None
    prediction_original: float = np.nan


def task_seed(master, task_id, search_type):
    return int(np.random.SeedSequence([master, task_id, search_type]).generate_state(1)[0])


def build_tasks(X_train, train_coords, X_target, target_coords, config, original_rows=None):
    """X_train/X_target均已用训练参数标准化；任务不复制训练矩阵。"""
    if len(X_train) < config.adaptive_k:
        raise ValueError('Training count is smaller than adaptive_k.')
    distances = core.local_distance_matrix(target_coords, train_coords, config.distance_metric)
    # 第25邻居只决定带宽，以下运算保留全部n_train列，绝不截取25行拟合。
    bandwidths = np.maximum(np.partition(distances, config.adaptive_k-1, axis=1)
                           [:, config.adaptive_k-1], config.weight_eps)
    weights = np.exp(-.5 * (distances / bandwidths[:, None])**2)
    weights /= weights.sum(axis=1, keepdims=True)
    assert weights.shape == (len(target_coords), len(X_train))
    assert np.allclose(weights.sum(axis=1), 1.)
    tasks = []
    for i, w in enumerate(weights):
        mu = np.sum(w[:, None] * X_train, axis=0)
        std = np.sqrt(np.maximum(np.sum(w[:, None] * (X_train-mu)**2, axis=0), 0))
        tasks.append(LocalTask(i, target_coords[i].copy(), X_target[i].copy(),
            float(bandwidths[i]), w, mu, std, np.r_[mu, std],
            i if original_rows is None else original_rows[i]))
    return tasks, distances, weights


def similarities(parent_weights, parent_descriptors, child, alpha):
    W = np.clip(np.minimum(parent_weights, child.weights[None, :]).sum(axis=1), 0, 1)
    d_f = np.sqrt(np.mean((parent_descriptors-child.descriptor[None, :])**2, axis=1))
    F = 1/(1+d_f)
    return W, F, alpha*W+(1-alpha)*F


def select_best_parent(child, completed_results, tasks, config):
    ids = list(completed_results)  # 仅已完成任务；平分时选择最早完成者
    if not ids:
        return None, None, None, None
    W, F, S = similarities(np.stack([tasks[i].weights for i in ids]),
        np.stack([tasks[i].descriptor for i in ids]), child, config.alpha)
    i = int(np.argmax(S))
    return ids[i], float(W[i]), float(F[i]), float(S[i])


def expression_key(ind):
    # gplearn的str会四舍五入常数；精确常数键避免不同表达式错误命中缓存。
    return tuple(('op', x.name, x.arity) if hasattr(x, 'arity') else
                 ('var', int(x)) if isinstance(x, (int, np.integer)) else
                 ('const', float(x).hex()) for x in ind.program)


class ExpressionPredictionCache:
    def __init__(self, enabled=True, max_size=10000):
        self.enabled, self.max_size = enabled, max_size
        self.values = OrderedDict()

    def get(self, key):
        if self.enabled and key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        return None

    def put(self, key, value):
        if self.enabled:
            self.values[key] = value.copy()  # float64避免缓存引入精度变化
            self.values.move_to_end(key)
            while len(self.values) > self.max_size:
                self.values.popitem(last=False)


_EVALUATION_LOCK = threading.RLock()


class GPEngine:
    """复用gplearn原生随机树、竞赛选择、交叉、变异和复制；只适配评价。

    私有API要求gplearn==0.4.3。本引擎顺序使用；不得在同进程另行并发fit。
    """
    def __init__(self, X_train, y_train, config, feature_names=None):
        self.X = np.array(X_train, dtype=float, copy=True)
        self.y = np.array(y_train, dtype=float, copy=True)
        self.config = config
        names = tuple(feature_names or [f'X{i}' for i in range(self.X.shape[1])])
        old = core.RunConfig('y', 'lon', 'lat', names, population_size=config.population_size)
        model = core.build_symbolic_regressor(old, config.full_generations, False)
        self.params = model.get_params()
        functions = [_function_map[x] for x in model.function_set]
        arities = {}
        for f in functions:
            arities.setdefault(f.arity, []).append(f)
        self.params.update(function_set=functions, arities=arities,
            _metric=_fitness_map['mse'], _transformer=None, max_samples=1.0,
            method_probs=np.cumsum([model.p_crossover, model.p_subtree_mutation,
                model.p_hoist_mutation, model.p_point_mutation]))
        self.cache = ExpressionPredictionCache(config.use_expression_cache, config.max_cache_size)
        self.begin_task(None)

    def begin_task(self, task):
        self.task = task
        self.task_fitness = {}         # 切换任务必须清空，fitness不能跨任务迁移
        self.requests = self.evaluations = self.executions = 0

    def raw_fitness(self, ind, X, y, weights):
        if len(X) != len(self.X) or not np.array_equal(weights, self.task.weights):
            raise ValueError('Fitness must use all training rows and current child weights.')
        self.requests += 1
        key = expression_key(ind)
        if key in self.task_fitness:
            return self.task_fitness[key]
        prediction = self.cache.get(key)
        if prediction is None:
            self.executions += 1
            with np.errstate(all='ignore'):
                prediction = np.asarray(ind.execute(self.X), dtype=float)
            self.cache.put(key, prediction)
        self.evaluations += 1
        with np.errstate(all='ignore'):
            loss = float(ind.metric(self.y, prediction, weights))
        if not np.isfinite(prediction).all() or not np.isfinite(loss):
            loss = float('inf')
        self.task_fitness[key] = loss
        return loss

    @contextmanager
    def evaluation_adapter(self):
        # gplearn内部调用原生_Program.raw_fitness；作用域结束必定恢复。
        # 不重写任何遗传算子，也不改变MSE+原复杂度惩罚的目标。
        with _EVALUATION_LOCK:
            original = _Program.raw_fitness
            engine = self
            def adapted(ind, X, y, sample_weight):
                return engine.raw_fitness(ind, X, y, sample_weight)
            _Program.raw_fitness = adapted
            try:
                yield
            finally:
                _Program.raw_fitness = original

    def evaluate_population(self, population):
        for ind in population:
            ind.raw_fitness_ = self.raw_fitness(ind, self.X, self.y, self.task.weights)
            ind.fitness_ = float(ind.fitness())  # 原生节点数复杂度惩罚，lambda=0.01
        return population

    def generation(self, parents, rng, size):
        seeds = rng.randint(np.iinfo(np.int32).max, size=size)
        with self.evaluation_adapter():
            pop = _parallel_evolve(size, parents, self.X, self.y, self.task.weights, seeds, self.params)
        for ind in pop:
            ind.fitness_ = float(ind.fitness())
        return pop

    def create_random_population(self, size, rng):
        return self.generation(None, rng, size)

    @staticmethod
    def invalidate_fitness(population):
        for ind in population:
            for attr in ('fitness_', 'raw_fitness_', 'oob_fitness_'):
                if hasattr(ind, attr):
                    delattr(ind, attr)

    def prepare_transfer_population(self, parent_population, rng):
        population, seen = [], set()
        for original in parent_population:
            if original is None or not original.validate_program():
                continue
            ind = copy.deepcopy(original)
            self.invalidate_fitness([ind])  # 绝不将父fitness用于子任务选择
            key = expression_key(ind)
            if self.config.deduplicate_transfer and key in seen:
                continue
            seen.add(key)
            self.evaluate_population([ind])
            if np.isfinite(ind.fitness_):
                population.append(ind)
            if len(population) == self.config.population_size:
                break
        missing = self.config.population_size-len(population)
        if missing:
            population.extend(self.create_random_population(missing, rng))
        return population

    def search(self, task, generations, seed, initial_population=None):
        rng = np.random.RandomState(seed)
        population = initial_population
        done = 0
        for _ in range(generations):
            population = self.generation(population, rng, self.config.population_size)
            done += 1
        # 原库回归器按末代raw MSE选输出；这里按用户要求的Q选择，
        # Pure与GST使用完全相同规则，不新增elitism或符号化简。
        valid = [p for p in population if np.isfinite(p.fitness_)]
        best = min(valid, key=lambda p: p.fitness_) if valid else None
        with np.errstate(all='ignore'):
            prediction = float(best.execute(task.x_target[None, :])[0]) if best is not None else np.nan
        return SearchResult(task.task_id, best, str(best) if best is not None else '',
            float(best.fitness_) if best is not None else float('inf'), prediction,
            population, n_generations=done, seed=seed)

    def run_full_search(self, task, seed):
        # 真正从None随机初始化；回退与独立基线使用同一个full_seed。
        return self.search(task, self.config.full_generations, seed)

    def run_short_search(self, task, initial_population, seed):
        return self.search(task, self.config.short_generations, seed, initial_population)


def safety_reason(result, probe_best):
    if result.best_individual is None:
        return 'no_valid_expression'
    if not np.isfinite(result.best_fitness):
        return 'non_finite_fitness'
    if not np.isfinite(result.prediction):
        return 'non_finite_prediction'
    if not any(p is not None and np.isfinite(getattr(p, 'fitness_', np.inf))
               for p in result.final_population):
        return 'no_valid_population'
    if result.best_fitness > probe_best:
        return 'worse_than_cold_probe'
    return None


@dataclass
class RunResults:
    records: list[SearchResult]
    tasks: list[LocalTask]
    summary: dict
    @property
    def prediction_original(self):
        return np.array([r.prediction_original for r in self.records])

    def dataframe(self):
        rows = []
        for r, t in zip(self.records, self.tasks):
            row = {k: v for k, v in vars(r).items() if k not in ('best_individual','final_population')}
            row.update(longitude=t.coord[0], latitude=t.coord[1], bandwidth=t.bandwidth,
                original_row_index=t.original_row_index, expression_complexity=(
                r.best_individual.length_ if r.best_individual is not None else None),
                full_or_short='short' if r.used_transfer else 'full',
                prediction_transformed=r.prediction,
                expression_transformed_space=r.best_expression)
            rows.append(row)
        return pd.DataFrame(rows)


class GSTGWSRRunner:
    def __init__(self, X_train, y_train, train_coords, config=None, gp_engine=None,
                 x_scaler=None, y_transformer=None, feature_names=None, task_builder=None):
        """输入X/y已完成预处理；x_scaler仅保留元数据，不二次transform。

        y_transformer为接收一维预测并返回原单位预测的callable。
        不传时预测保持输入y的单位。
        """
        self.config = config or GSTConfig()
        self.X, self.y = np.asarray(X_train, float), np.asarray(y_train, float)
        self.coords = np.asarray(train_coords, float)
        if self.X.ndim != 2 or self.y.shape != (len(self.X),) or self.coords.shape != (len(self.X),2):
            raise ValueError('Invalid training shapes.')
        if not all(np.isfinite(a).all() for a in (self.X,self.y,self.coords)):
            raise ValueError('Training arrays must be finite.')
        self.engine = gp_engine or GPEngine(self.X, self.y, self.config, feature_names)
        self.x_scaler, self.y_transformer = x_scaler, y_transformer
        self.task_builder = task_builder or build_tasks

    def fit_predict(self, target_coords, X_target, original_row_indices=None, use_gst=True):
        # 接口不接受y_target：真实评价响应不能进入搜索、相似度或安全检查。
        started = time.perf_counter()
        coords, X = np.asarray(target_coords,float), np.asarray(X_target,float)
        if X.shape != (len(coords),self.X.shape[1]) or coords.shape != (len(X),2) or len(X)==0:
            raise ValueError('Invalid query shapes.')
        if not np.isfinite(X).all() or not np.isfinite(coords).all():
            raise ValueError('Query arrays must be finite.')
        if original_row_indices is not None and len(original_row_indices) != len(X):
            raise ValueError('original_row_indices length mismatch.')
        tasks, self.distance_matrix, self.weights_matrix = self.task_builder(
            self.X,self.coords,X,coords,self.config,original_row_indices)
        self.engine.cache.values.clear()  # 每次运行独立计时，基线不为GST预热缓存
        completed = {}
        for task in tasks:              # 输入row order固定，不聚类、不重新排序
            tick = time.perf_counter()
            self.engine.begin_task(task)
            full, transfer, probe = [task_seed(self.config.master_seed,task.task_id,i) for i in range(3)]
            parent,W,F,S = select_best_parent(task,completed,tasks,self.config) if use_gst else (None,)*4
            attempted = parent is not None and S >= self.config.transfer_threshold
            reason, probe_best, candidate, short_done = None, None, None, 0
            if attempted:
                try:
                    population = self.engine.prepare_transfer_population(
                        completed[parent].final_population,np.random.RandomState(transfer))
                    probe_size = max(self.config.min_probe_size,
                        int(np.ceil(self.config.probe_fraction*self.config.population_size)))
                    probes = self.engine.create_random_population(probe_size,np.random.RandomState(probe))
                    probe_best = min((p.fitness_ for p in probes),default=np.inf)
                    candidate = self.engine.run_short_search(task,population,transfer)
                    short_done = candidate.n_generations
                    reason = safety_reason(candidate,probe_best)
                except (FloatingPointError, ValueError, OverflowError) as exc:
                    reason = f'transfer_error: {exc}'
                if reason is None:
                    result = candidate
                    result.used_transfer = True
                else:
                    result = self.engine.run_full_search(task,full)
                    result.n_generations += short_done
                    result.fallback_used, result.fallback_reason = True, reason
            else:
                result = self.engine.run_full_search(task,full)
            if safety_reason(result,np.inf) is not None:
                raise RuntimeError(f'Task {task.task_id}: full search has no valid result.')
            result.attempted_transfer = attempted
            result.parent_task_id,result.W,result.F,result.S = parent,W,F,S
            result.full_seed,result.transfer_seed,result.probe_seed = full,transfer,probe
            result.probe_best_fitness = float(probe_best) if probe_best is not None else None
            if candidate is not None:
                result.transfer_best_fitness = candidate.best_fitness
                result.transfer_expression = candidate.best_expression
            result.n_fitness_evaluations = self.engine.evaluations
            result.n_fitness_requests = self.engine.requests
            result.n_tree_executions = self.engine.executions
            result.prediction_original = float(self.y_transformer(np.array([result.prediction]))[0]
                if self.y_transformer is not None else result.prediction)
            if not np.isfinite(result.prediction_original):
                raise RuntimeError('Nonfinite prediction after inverse target transform.')
            result.runtime_seconds = time.perf_counter()-tick
            completed[task.task_id] = result
        records = list(completed.values())
        attempts = sum(r.attempted_transfer for r in records)
        accepted = sum(r.used_transfer for r in records)
        fallbacks = sum(r.fallback_used for r in records)
        means = {f'mean_{k}': float(np.mean([getattr(r,k) for r in records if getattr(r,k) is not None]))
                 if any(getattr(r,k) is not None for r in records) else None for k in ('W','F','S')}
        summary = dict(total_tasks=len(records),independent_tasks=len(records)-attempts,
            full_search_tasks=len(records)-accepted,transfer_attempted_tasks=attempts,
            transfer_accepted_tasks=accepted,fallback_tasks=fallbacks,
            transfer_acceptance_rate=accepted/attempts if attempts else 0.,
            fallback_rate=fallbacks/attempts if attempts else 0.,
            total_runtime=time.perf_counter()-started,
            total_fitness_evaluations=sum(r.n_fitness_evaluations for r in records),
            total_fitness_requests=sum(r.n_fitness_requests for r in records),
            total_tree_executions=sum(r.n_tree_executions for r in records),
            total_generations=sum(r.n_generations for r in records),**means)
        summary['mean_parent_similarity'] = summary['mean_S']
        return RunResults(records,tasks,summary)


def evaluate_predictions(y_true, prediction_original):
    return core.metric_summary(np.asarray(y_true,float),np.asarray(prediction_original,float))
