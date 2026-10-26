"""
herdlock.market — seeded, non-LLM simulator of the "Music Lab" social-influence
design (Salganik, Dodds & Watts 2006) used by Hashemi, Ji, Chen, Macy,
arXiv:2610.30769. Agents are parametric (sigmoid + Gaussian noise), a
mechanism-level stand-in for LLM agents, not a reproduction of their numbers.
Full formulas and rationale: see README.md.

K items get a fixed appeal a_k ~ Normal(0,1). Agents arrive sequentially and:

1. SELECT `budget` items, weighted by a mix of uniform browsing and the
   popularity share of visible counts (susceptibility `s`); "strong" also
   sorts by count and applies a position-decay bias (gamma ** position).
2. RECOMMEND yes/no per selected item: P(yes) = sigmoid(taste + social),
   taste = a_k + noise, social = s * log(K * smoothed popularity share). A
   "yes" updates the count immediately, so later agents see it.

Counts are invisible (uniform share, zero social term, s has no effect) when
condition == "independent" or the agent index is below `blind_first`.

Also implements the paper's metrics: market share, Jeffreys-smoothed KL,
Gini, Spearman/Pearson correlation (ties averaged), and unpredictability
(mean pairwise |share difference| across W parallel worlds).
"""

from __future__ import annotations

import numpy as np

TAU = 1.0          # taste-noise standard deviation
GAMMA = 0.98       # strong-condition position-decay base


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smoothed_share(disp: np.ndarray) -> np.ndarray:
    """Smoothed popularity share sh_k = (disp_k + 1) / (sum(disp) + K).

    Equals 1/K for every item when all disp_k are equal (in particular at
    the all-zero start of a world), so `log(K * sh_k)` is exactly 0 there.
    """
    disp = np.asarray(disp, dtype=float)
    K = disp.shape[0]
    return (disp + 1.0) / (disp.sum() + K)


def appeal(K: int = 48, seed: int = 0) -> np.ndarray:
    """Intrinsic appeal logits for K items, fixed per market by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=K)


def run_world(
    appeal: np.ndarray,
    n_agents: int,
    condition: str,
    susceptibility: float,
    budget: int = 4,
    seed: int = 0,
    blind_first: int = 0,
    show: str = "raw",
) -> dict:
    """Simulate one sequential world of `n_agents` over the given appeal.

    condition: "independent" (no counts shown), "weak" (counts shown, random
    order), or "strong" (counts shown, sorted by count, position-decay bias).
    show: "raw" or "log" (log1p-dampened counts, a mitigation).
    blind_first: number of leading agents who see no counts regardless of
    condition (a "blind seed round" mitigation).
    """
    if condition not in ("independent", "weak", "strong"):
        raise ValueError(f"unknown condition: {condition!r}")

    rng = np.random.default_rng(seed)
    K = appeal.shape[0]
    half = n_agents // 2

    counts = np.zeros(K)
    counts_first30 = np.zeros(K)
    counts_after30 = np.zeros(K)
    counts_first_half = np.zeros(K)
    counts_second_half = np.zeros(K)
    selections = np.full((n_agents, budget), -1, dtype=int)

    uniform = np.full(K, 1.0 / K)

    for i in range(n_agents):
        visible = (condition != "independent") and (i >= blind_first)

        if visible:
            disp = np.log1p(counts) if show == "log" else counts
            share = smoothed_share(disp)

            if condition == "strong":
                tie_break = rng.random(K) * 1e-9
                order = np.argsort(-(counts + tie_break))
                position = np.empty(K, dtype=int)
                position[order] = np.arange(K)
                pos_mult = GAMMA ** position
            else:
                pos_mult = np.ones(K)

            weight = ((1 - susceptibility) * uniform + susceptibility * share) * pos_mult
        else:
            share = uniform
            weight = uniform

        weight = weight / weight.sum()
        chosen = rng.choice(K, size=budget, replace=False, p=weight)
        selections[i] = chosen

        taste = appeal[chosen] + rng.normal(0.0, TAU, size=budget)
        if visible:
            social = susceptibility * np.log(K * share[chosen])
        else:
            social = np.zeros(budget)
        p_yes = sigmoid(taste + social)
        yes = rng.random(budget) < p_yes
        yes_items = chosen[yes]

        counts[yes_items] += 1
        if i < 30:
            counts_first30[yes_items] += 1
        else:
            counts_after30[yes_items] += 1
        if i < half:
            counts_first_half[yes_items] += 1
        else:
            counts_second_half[yes_items] += 1

    return dict(
        counts=counts,
        counts_first30=counts_first30,
        counts_after30=counts_after30,
        counts_first_half=counts_first_half,
        counts_second_half=counts_second_half,
        selections=selections,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def market_share(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return counts / counts.sum()


def kl_divergence(counts_p: np.ndarray, counts_q: np.ndarray) -> float:
    """KL(P || Q) over items, Jeffreys-smoothed (+0.5 per item)."""
    counts_p = np.asarray(counts_p, dtype=float)
    counts_q = np.asarray(counts_q, dtype=float)
    K = counts_p.shape[0]
    p = (counts_p + 0.5) / (counts_p.sum() + 0.5 * K)
    q = (counts_q + 0.5) / (counts_q.sum() + 0.5 * K)
    return float(np.sum(p * np.log(p / q)))


def gini(shares: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = equal, (n-1)/n = one-hot)."""
    x = np.sort(np.asarray(shares, dtype=float))
    n = x.size
    s = x.sum()
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1) / n)


def rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties set to the average rank of the tied group."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks_sorted[i : j + 1] = (ranks_sorted[i] + ranks_sorted[j]) / 2.0
        i = j + 1
    ranks = np.empty(n)
    ranks[order] = ranks_sorted
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata_avg(x), rankdata_avg(y))


def unpredictability(shares_matrix: np.ndarray) -> float:
    """Mean over items of the mean pairwise |share_j - share_k| over all
    pairs of worlds. `shares_matrix` has shape (W, K)."""
    shares_matrix = np.asarray(shares_matrix, dtype=float)
    W = shares_matrix.shape[0]
    diffs = []
    for a in range(W):
        for b in range(a + 1, W):
            diffs.append(np.abs(shares_matrix[a] - shares_matrix[b]))
    diffs = np.stack(diffs)
    return float(diffs.mean(axis=0).mean())
