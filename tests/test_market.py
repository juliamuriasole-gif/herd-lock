import numpy as np

from herdlock.market import appeal, run_world, smoothed_share


K = 24
N_AGENTS = 150


def make_appeal(seed=1):
    return appeal(K=K, seed=seed)


def test_determinism_by_seed():
    a = make_appeal()
    r1 = run_world(a, N_AGENTS, "strong", 0.3, budget=4, seed=42)
    r2 = run_world(a, N_AGENTS, "strong", 0.3, budget=4, seed=42)
    assert np.array_equal(r1["counts"], r2["counts"])
    assert np.array_equal(r1["selections"], r2["selections"])


def test_different_seed_differs():
    a = make_appeal()
    r1 = run_world(a, N_AGENTS, "strong", 0.3, budget=4, seed=42)
    r2 = run_world(a, N_AGENTS, "strong", 0.3, budget=4, seed=43)
    assert not np.array_equal(r1["counts"], r2["counts"])


def test_independent_ignores_susceptibility():
    a = make_appeal()
    r_low = run_world(a, N_AGENTS, "independent", 0.0, budget=4, seed=7)
    r_high = run_world(a, N_AGENTS, "independent", 0.9, budget=4, seed=7)
    assert np.array_equal(r_low["counts"], r_high["counts"])
    assert np.array_equal(r_low["selections"], r_high["selections"])


def test_weak_zero_susceptibility_matches_independent():
    a = make_appeal()
    r_indep = run_world(a, N_AGENTS, "independent", 0.0, budget=4, seed=11)
    r_weak = run_world(a, N_AGENTS, "weak", 0.0, budget=4, seed=11)
    assert np.array_equal(r_indep["counts"], r_weak["counts"])
    assert np.array_equal(r_indep["selections"], r_weak["selections"])


def test_strong_zero_susceptibility_differs_from_independent():
    # strong still applies a position-decay bias even at susceptibility=0
    a = make_appeal()
    r_indep = run_world(a, N_AGENTS, "independent", 0.0, budget=4, seed=11)
    r_strong = run_world(a, N_AGENTS, "strong", 0.0, budget=4, seed=11)
    assert not np.array_equal(r_indep["counts"], r_strong["counts"])


def test_budget_respected():
    a = make_appeal()
    for budget in (4, 12):
        r = run_world(a, N_AGENTS, "strong", 0.5, budget=budget, seed=5)
        assert r["selections"].shape == (N_AGENTS, budget)
        for row in r["selections"]:
            assert len(set(row.tolist())) == budget


def test_social_term_zero_at_uniform_state():
    # social_k = s * log(K * smoothed_share_k); at the uniform state (all
    # disp_k equal, in particular all-zero at the start of a world) this
    # must be exactly 0 regardless of susceptibility or the common count.
    for disp in (np.zeros(K), np.full(K, 7.0)):
        share = smoothed_share(disp)
        social = np.log(K * share)
        assert np.allclose(social, 0.0)


def test_showing_counts_does_not_shift_mean_yes_rate_much():
    # with the neutral (zero-at-uniform) social term, showing counts should
    # not mechanically suppress or inflate the average yes-rate the way the
    # old logit-based term did (it crushed it by ~30-40%, see README). Check
    # it stays within a loose absolute tolerance, averaged over several
    # worlds to damp single-world noise.
    a = make_appeal()
    tol = 0.08
    rates = {}
    for condition in ("independent", "weak", "strong"):
        per_world = [
            run_world(a, N_AGENTS, condition, 0.2, budget=4, seed=100 + w)["counts"].sum()
            / (N_AGENTS * 4)
            for w in range(8)
        ]
        rates[condition] = np.mean(per_world)
    assert abs(rates["weak"] - rates["independent"]) < tol
    assert abs(rates["strong"] - rates["independent"]) < tol


def test_blind_first_matches_independent_for_leading_agents():
    a = make_appeal()
    r_indep = run_world(a, N_AGENTS, "independent", 0.5, budget=4, seed=3)
    r_blind = run_world(a, N_AGENTS, "strong", 0.5, budget=4, seed=3, blind_first=N_AGENTS)
    # with the whole world blind, strong degenerates to the position-bias-only
    # case; selections for the leading (here: all) agents use uniform shares,
    # same as independent, since blind_first == n_agents disables counts
    # entirely for every agent -> no position bias ever triggers either,
    # because position bias is only computed when counts are visible.
    assert np.array_equal(r_indep["counts"], r_blind["counts"])
