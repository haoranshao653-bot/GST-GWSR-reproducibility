"""Fixed local objective and best-ever GP archive for manuscript sections 2.1/2.2.

Uses gplearn 0.4.3 native genetic operators, but explicitly manages selection
fitness and the historical best expression. Original prediction safeguards retained.
"""
import copy
from importlib.metadata import version

import numpy as np
from gplearn.genetic import _parallel_evolve
from gplearn.functions import _function_map
from gplearn.fitness import _fitness_map
from sklearn.metrics import r2_score

import original_bisquare as base


def exact_expression(program):
    """Full precision constants, unlike gplearn's display rounded to 3 decimals."""
    items = iter(program.program)
    def take():
        node = next(items)
        if hasattr(node, 'arity'):
            return node.name + '(' + ', '.join(take() for _ in range(node.arity)) + ')'
        if isinstance(node, (int, np.integer)):
            return base.FEATURES[int(node)]
        return repr(float(node))
    return take()


def local_objective(program, X, y, weights, penalty):
    positive = weights > 0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        pred = program.execute(X)
        if not np.isfinite(pred[positive]).all():
            return np.inf, np.inf
        loss = float(np.dot(weights[positive], (y[positive]-pred[positive])**2))
        objective = loss + penalty * program.length_
    if not np.isfinite(objective):
        return np.inf, np.inf
    return loss, objective


def search(X, y, weights, seed, model=None):
    if version('gplearn') != '0.4.3':
        raise RuntimeError('This engine requires gplearn==0.4.3 (private operator API).')
    model = base.build_sr_model(seed) if model is None else model
    X, y, weights = np.asarray(X), np.asarray(y), np.asarray(weights, dtype=float).copy()
    if (not np.isfinite(X).all() or not np.isfinite(y).all()
            or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Invalid local training data or weights.')
    weights /= weights.sum()
    params = model.get_params().copy()
    penalty = float(params['parsimony_coefficient'])
    if penalty < 0:
        raise ValueError('Complexity penalty must be nonnegative.')
    functions = [_function_map[name] for name in params['function_set']]
    arities = {}
    for fn in functions:
        arities.setdefault(fn.arity, []).append(fn)
    params.update(function_set=functions, arities=arities,
                  _metric=_fitness_map['mse'], _transformer=None,
                  max_samples=1.0,  # fixed full local loss, not candidate-specific subsamples
                  method_probs=np.cumsum([params['p_crossover'], params['p_subtree_mutation'],
                                          params['p_hoist_mutation'], params['p_point_mutation']]))
    rng = np.random.RandomState(seed)
    parents, best = None, None
    best_J, best_loss, best_gen = np.inf, np.inf, None
    generation_minima, archive_history = [], []
    invalid_count = 0
    for gen in range(int(params['generations'])):
        seeds = rng.randint(np.iinfo(np.int32).max, size=int(params['population_size']))
        population = _parallel_evolve(int(params['population_size']), parents, X, y,
                                      weights, seeds, params)
        values = []
        for program in population:
            loss, J = local_objective(program, X, y, weights, penalty)
            program.raw_fitness_ = loss
            program.fitness_ = J  # next generation tournaments use exactly this J
            values.append(J)
            invalid_count += int(not np.isfinite(J))
            if J < best_J:
                best, best_J, best_loss, best_gen = copy.deepcopy(program), J, loss, gen+1
        if not np.isfinite(values).any():
            raise RuntimeError(f'Generation {gen+1}: no finite local candidate objective.')
        generation_minima.append(float(np.min(values)))
        archive_history.append(best_J)
        parents = population
    if best is None:
        raise RuntimeError('No valid expression found; no mean fallback is permitted.')
    info = dict(best_objective=best_J, weighted_loss=best_loss,
                complexity_nodes=int(best.length_), complexity_penalty=penalty*best.length_,
                best_generation=best_gen, generations_executed=len(generation_minima),
                evaluated_candidates=len(generation_minima)*int(params['population_size']),
                invalid_candidate_count=invalid_count,
                generation_min_objectives=generation_minima,
                best_so_far_objectives=archive_history,
                expression_exact=exact_expression(best))
    return best, info


def predict_one_point(i, xtrain, ytrain, xquery, distances, k, seed):
    X, y, weights = base.get_local_sample(-1, distances, xtrain, ytrain, k)
    info = dict(best_objective=np.nan, weighted_loss=np.nan, complexity_nodes=np.nan,
                complexity_penalty=np.nan, best_generation=np.nan, generations_executed=np.nan,
                evaluated_candidates=np.nan, invalid_candidate_count=np.nan,
                generation_min_objectives=[], best_so_far_objectives=[], expression_exact='',
                raw_prediction_scaled=np.nan, prediction_protected=False, failure_reason='')
    try:
        best, search_info = search(X, y, weights, seed+i)
        info.update(search_info)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            pred = float(best.execute(np.asarray(xquery).reshape(1,-1))[0])
            fitted = best.execute(X)
        expression = str(best)
        info['raw_prediction_scaled'] = pred
        unweighted = float(r2_score(y, fitted))
        if not np.isfinite(pred) or abs(pred) >= 5:
            pred = float(np.average(y, weights=weights))
            expression = f'CLIPPED({expression[:80]})'
            info['prediction_protected'] = True
    except Exception as exc:
        pred = float(np.average(y, weights=weights))
        expression = f'FAILED({str(exc)[:80]})'
        unweighted = 0.0
        info['prediction_protected'] = True
        info['failure_reason'] = str(exc)
    mean = float(weights @ y)
    sst = float(weights @ ((y-mean)**2))
    tol = 1e-12 + 1e-10*float(np.var(y))
    info['weighted_local_fit_R2'] = float(1-info['weighted_loss']/sst) if sst>tol else np.nan
    info['weighted_RMSE_log_scaled'] = float(np.sqrt(info['weighted_loss']))
    info['weighted_SST'] = sst
    return i, pred, expression, unweighted, info


def inverse_transform(pred, scaler):
    """Exact original inverse, including clipping log predictions to [-5,10]."""
    return base.inverse_transform(np.asarray(pred), scaler)
