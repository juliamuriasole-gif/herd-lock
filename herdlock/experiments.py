"""
herdlock.experiments

Builds the three result tables (condition comparison, susceptibility sweep,
mitigation comparison) on top of herdlock.market.run_world, using a fixed
item set (one `appeal` vector) shared across every world so that per-item
shares are comparable across worlds and conditions, as in the paper's design
(1 independent world + W weak + W strong worlds per model).
"""

from __future__ import annotations

import numpy as np

from .market import (
    appeal as make_appeal,
    run_world,
    sigmoid,
    market_share,
    kl_divergence,
    gini,
    pearson,
    spearman,
    unpredictability,
)

K = 48
N_AGENTS = 720
BUDGET = 4
SUSCEPTIBILITY = 0.2
W = 8
SEED = 12345

SWEEP_VALUES = (0.0, 0.1, 0.2, 0.3, 0.5)


def _seed(tag_id: int, variant_id: int, w: int) -> int:
    """Deterministic, collision-free per-(tag, variant, world) seed."""
    return SEED + tag_id * 1_000_000 + variant_id * 10_000 + w


def _safe_share(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.full(counts.shape[0], 1.0 / counts.shape[0])
    return counts / total


def run_many(appeal, tag_id, variant_id, n_agents, condition, susceptibility,
             budget=BUDGET, w=W, blind_first=0, show="raw"):
    return [
        run_world(
            appeal, n_agents, condition, susceptibility,
            budget=budget, seed=_seed(tag_id, variant_id, world),
            blind_first=blind_first, show=show,
        )
        for world in range(w)
    ]


def independent_baseline(appeal, n_agents=N_AGENTS):
    """One larger independent world (2x agents), used as the ground-truth
    reference for KL / correlation metrics, as in the paper."""
    return run_world(appeal, n_agents * 2, "independent", 0.0,
                      budget=BUDGET, seed=_seed(99, 0, 0))


def summarize(results, indep_counts, true_shares, indep_shares):
    shares = np.stack([market_share(r["counts"]) for r in results])
    first30 = np.stack([_safe_share(r["counts_first30"]) for r in results])
    after30 = np.stack([_safe_share(r["counts_after30"]) for r in results])

    kl = float(np.mean([kl_divergence(r["counts"], indep_counts) for r in results]))
    gini_val = float(np.mean([gini(s) for s in shares]))
    u = unpredictability(shares)
    mean_yes_rate = float(np.mean([r["counts"].sum() / r["selections"].size for r in results]))
    corr_appeal = pearson(shares.flatten(), np.tile(true_shares, len(results)))
    corr_indep = pearson(shares.flatten(), np.tile(indep_shares, len(results)))
    early_corr = pearson(first30.flatten(), after30.flatten())
    delta_rhos = [
        spearman(r["counts_second_half"], indep_counts)
        - spearman(r["counts_first_half"], indep_counts)
        for r in results
    ]
    delta_rho = float(np.mean(delta_rhos))

    return dict(
        KL=kl, U=u, Gini=gini_val,
        corr_appeal=corr_appeal, corr_independent=corr_indep,
        early_mover_corr=early_corr, delta_rho=delta_rho,
        mean_yes_rate=mean_yes_rate,
    )


def condition_table(appeal, indep_baseline_result, n_agents=N_AGENTS,
                     budget=BUDGET, susceptibility=SUSCEPTIBILITY, w=W):
    indep_counts = indep_baseline_result["counts"]
    indep_shares = market_share(indep_counts)
    true_shares = _safe_share(sigmoid(appeal))

    rows = []
    mean_shares = {}
    for tag_id, condition in ((0, "independent"), (1, "weak"), (2, "strong")):
        results = run_many(appeal, tag_id, 0, n_agents, condition, susceptibility,
                            budget=budget, w=w)
        row = dict(condition=condition)
        row.update(summarize(results, indep_counts, true_shares, indep_shares))
        rows.append(row)
        shares = np.stack([market_share(r["counts"]) for r in results])
        mean_shares[condition] = shares.mean(axis=0)
    return rows, mean_shares


def susceptibility_sweep(appeal, indep_baseline_result, n_agents=N_AGENTS,
                          budget=BUDGET, w=W, values=SWEEP_VALUES):
    indep_counts = indep_baseline_result["counts"]
    indep_shares = market_share(indep_counts)
    true_shares = _safe_share(sigmoid(appeal))

    rows = []
    for variant_id, s in enumerate(values):
        results = run_many(appeal, 3, variant_id, n_agents, "strong", s,
                            budget=budget, w=w)
        row = dict(susceptibility=s)
        row.update(summarize(results, indep_counts, true_shares, indep_shares))
        rows.append(row)
    return rows


def mitigation_table(appeal, indep_baseline_result, n_agents=N_AGENTS,
                      susceptibility=SUSCEPTIBILITY, w=W):
    indep_counts = indep_baseline_result["counts"]
    indep_shares = market_share(indep_counts)
    true_shares = _safe_share(sigmoid(appeal))

    variants = [
        ("none", dict(condition="strong", budget=BUDGET, blind_first=0, show="raw")),
        ("blind_first=100", dict(condition="strong", budget=BUDGET, blind_first=100, show="raw")),
        ("show=log", dict(condition="strong", budget=BUDGET, blind_first=0, show="log")),
        ("budget=12", dict(condition="strong", budget=12, blind_first=0, show="raw")),
        ("independent", dict(condition="independent", budget=BUDGET, blind_first=0, show="raw")),
    ]

    rows = []
    for variant_id, (name, kw) in enumerate(variants):
        condition = kw.pop("condition")
        results = run_many(appeal, 4, variant_id, n_agents, condition, susceptibility,
                            budget=kw["budget"], w=w, blind_first=kw["blind_first"],
                            show=kw["show"])
        row = dict(mitigation=name)
        row.update(summarize(results, indep_counts, true_shares, indep_shares))
        rows.append(row)
    return rows


ROBUSTNESS_SEEDS = (1, 2, 3, 42, 777)


def seed_robustness_check(seeds=ROBUSTNESS_SEEDS, n_agents=200, w=4,
                           susceptibility=SUSCEPTIBILITY, budget=BUDGET):
    """Cheap re-run of the condition table at a handful of other appeal
    seeds (smaller n_agents/w to keep it fast), checking whether the
    independent < weak < strong ordering on KL and Gini holds at each one.
    Replaces an "informal check" with a number: how many of `seeds` it
    held for.
    """
    per_seed = []
    for s in seeds:
        a = make_appeal(K=K, seed=s)
        baseline = independent_baseline(a, n_agents=n_agents)
        rows, _ = condition_table(a, baseline, n_agents=n_agents, budget=budget,
                                   susceptibility=susceptibility, w=w)
        by_cond = {r["condition"]: r for r in rows}
        holds = (by_cond["independent"]["KL"] < by_cond["weak"]["KL"] < by_cond["strong"]["KL"]
                 and by_cond["independent"]["Gini"] < by_cond["weak"]["Gini"] < by_cond["strong"]["Gini"])
        per_seed.append(dict(seed=s, ordering_holds=bool(holds)))
    n_holds = sum(r["ordering_holds"] for r in per_seed)
    return dict(seeds=list(seeds), n_agents=n_agents, w=w,
                per_seed=per_seed, n_holds=n_holds, n_total=len(seeds))


def run_all(seed=SEED, n_agents=N_AGENTS, w=W):
    appeal = make_appeal(K=K, seed=seed)
    baseline = independent_baseline(appeal, n_agents=n_agents)
    condition_rows, chart_shares = condition_table(appeal, baseline, n_agents=n_agents, w=w)
    tables = dict(
        condition=condition_rows,
        sweep=susceptibility_sweep(appeal, baseline, n_agents=n_agents, w=w),
        mitigation=mitigation_table(appeal, baseline, n_agents=n_agents, w=w),
    )
    robustness = seed_robustness_check()
    return appeal, baseline, tables, chart_shares, robustness
