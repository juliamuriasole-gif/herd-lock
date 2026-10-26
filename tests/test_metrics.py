import numpy as np

from herdlock.market import (
    market_share, kl_divergence, gini, spearman, pearson,
    rankdata_avg, unpredictability,
)


def test_kl_zero_for_identical_counts():
    counts = np.array([5.0, 10.0, 0.0, 3.0])
    assert kl_divergence(counts, counts) == 0.0


def test_gini_zero_for_equal_shares():
    shares = np.full(8, 1.0 / 8)
    assert abs(gini(shares)) < 1e-12


def test_gini_one_hot():
    K = 10
    shares = np.zeros(K)
    shares[3] = 1.0
    assert abs(gini(shares) - (K - 1) / K) < 1e-12


def test_unpredictability_zero_for_identical_worlds():
    shares_matrix = np.tile(np.array([0.5, 0.3, 0.2]), (8, 1))
    assert unpredictability(shares_matrix) == 0.0


def test_unpredictability_positive_for_differing_worlds():
    shares_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert unpredictability(shares_matrix) == 1.0


def test_spearman_hand_computed():
    # x ranks: [1,2,3,4]; y = [2,1,4,3] ranks: [2,1,4,3]
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([20.0, 10.0, 40.0, 30.0])
    # Pearson correlation of rank vectors [1,2,3,4] vs [2,1,4,3]
    rx = np.array([1, 2, 3, 4], dtype=float)
    ry = np.array([2, 1, 4, 3], dtype=float)
    expected = pearson(rx, ry)
    assert abs(spearman(x, y) - expected) < 1e-9


def test_spearman_ties_averaged():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    ranks = rankdata_avg(x)
    assert np.allclose(ranks, [1.5, 1.5, 3.0, 4.0])


def test_spearman_perfect_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert abs(spearman(x, y) - 1.0) < 1e-9


def test_market_share_sums_to_one():
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    shares = market_share(counts)
    assert abs(shares.sum() - 1.0) < 1e-12
