"""Small dependency-free NSGA-II mechanics with SBX and polynomial mutation."""

from __future__ import annotations

import numpy as np


def constraint_violation(constraints: np.ndarray) -> np.ndarray:
    constraints = np.asarray(constraints, dtype=float)
    return np.sum(np.maximum(constraints, 0.0), axis=1)


def dominates(a: np.ndarray, b: np.ndarray, a_cv: float = 0.0, b_cv: float = 0.0) -> bool:
    if a_cv == 0.0 and b_cv > 0.0:
        return True
    if a_cv > 0.0 and b_cv == 0.0:
        return False
    if a_cv != b_cv:
        return a_cv < b_cv
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(values: np.ndarray, constraints: np.ndarray | None = None) -> list[np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    cv = np.zeros(n) if constraints is None else constraint_violation(constraints)
    domination_sets: list[list[int]] = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(values[p], values[q], cv[p], cv[q]):
                domination_sets[p].append(q)
            elif dominates(values[q], values[p], cv[q], cv[p]):
                counts[p] += 1
        if counts[p] == 0:
            fronts[0].append(p)
    index = 0
    while index < len(fronts) and fronts[index]:
        next_front: list[int] = []
        for p in fronts[index]:
            for q in domination_sets[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    next_front.append(q)
        index += 1
        if next_front:
            fronts.append(next_front)
    return [np.asarray(front, dtype=int) for front in fronts if front]


def crowding_distance(values: np.ndarray, front: np.ndarray) -> np.ndarray:
    distances = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        return np.full(len(front), np.inf)
    selected = values[front]
    for objective in range(values.shape[1]):
        order = np.argsort(selected[:, objective])
        distances[order[0]] = distances[order[-1]] = np.inf
        span = selected[order[-1], objective] - selected[order[0], objective]
        if span > 0:
            distances[order[1:-1]] += (selected[order[2:], objective] - selected[order[:-2], objective]) / span
    return distances


def rank_and_crowding(values: np.ndarray, constraints: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    fronts = fast_non_dominated_sort(values, constraints)
    rank = np.empty(len(values), dtype=int)
    crowding = np.zeros(len(values), dtype=float)
    for level, front in enumerate(fronts):
        rank[front] = level
        crowding[front] = crowding_distance(values, front)
    return rank, crowding


def tournament(indices: np.ndarray, rank: np.ndarray, crowding: np.ndarray, rng: np.random.Generator) -> int:
    i, j = rng.choice(indices, size=2, replace=False)
    if rank[i] < rank[j] or (rank[i] == rank[j] and crowding[i] > crowding[j]):
        return int(i)
    return int(j)


def sbx_pair(parent_a: np.ndarray, parent_b: np.ndarray, lower: np.ndarray, upper: np.ndarray, probability: float, eta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a, b = parent_a.copy(), parent_b.copy()
    if rng.random() > probability:
        return a, b
    for idx in range(len(a)):
        if rng.random() > .5 or abs(a[idx] - b[idx]) < 1e-14:
            continue
        y1, y2 = sorted((a[idx], b[idx]))
        rand = rng.random()
        beta = 1.0 + 2.0 * (y1 - lower[idx]) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = .5 * ((y1 + y2) - betaq * (y2 - y1))
        beta = 1.0 + 2.0 * (upper[idx] - y2) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child2 = .5 * ((y1 + y2) + betaq * (y2 - y1))
        if rng.random() <= .5:
            a[idx], b[idx] = child2, child1
        else:
            a[idx], b[idx] = child1, child2
    return np.clip(a, lower, upper), np.clip(b, lower, upper)


def polynomial_mutation(x: np.ndarray, lower: np.ndarray, upper: np.ndarray, probability: float, eta: float, rng: np.random.Generator) -> np.ndarray:
    result = x.copy()
    for idx in range(len(result)):
        if rng.random() > probability:
            continue
        delta1 = (result[idx] - lower[idx]) / (upper[idx] - lower[idx])
        delta2 = (upper[idx] - result[idx]) / (upper[idx] - lower[idx])
        rand = rng.random()
        mutation_power = 1.0 / (eta + 1.0)
        if rand <= .5:
            xy = 1.0 - delta1
            value = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
            deltaq = value ** mutation_power - 1.0
        else:
            xy = 1.0 - delta2
            value = 2.0 * (1.0 - rand) + 2.0 * (rand - .5) * xy ** (eta + 1.0)
            deltaq = 1.0 - value ** mutation_power
        result[idx] += deltaq * (upper[idx] - lower[idx])
    return np.clip(result, lower, upper)


def environmental_selection(x: np.ndarray, objectives: np.ndarray, constraints: np.ndarray, n_keep: int) -> np.ndarray:
    selected: list[int] = []
    for front in fast_non_dominated_sort(objectives, constraints):
        if len(selected) + len(front) <= n_keep:
            selected.extend(front.tolist())
            continue
        distances = crowding_distance(objectives, front)
        selected.extend(front[np.argsort(-distances)[: n_keep - len(selected)]].tolist())
        break
    return np.asarray(selected, dtype=int)
