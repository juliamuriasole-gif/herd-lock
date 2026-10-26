from herdlock.experiments import (
    make_appeal, independent_baseline, condition_table, mitigation_table, K,
    seed_robustness_check,
)

N_AGENTS = 250
WORLDS = 4


def _setup(seed=999):
    appeal = make_appeal(K=K, seed=seed)
    baseline = independent_baseline(appeal, n_agents=N_AGENTS)
    return appeal, baseline


def test_strong_exceeds_independent_on_kl_and_unpredictability():
    appeal, baseline = _setup()
    rows, _ = condition_table(appeal, baseline, n_agents=N_AGENTS, w=WORLDS)
    by_condition = {r["condition"]: r for r in rows}
    assert by_condition["strong"]["KL"] > by_condition["independent"]["KL"]
    assert by_condition["strong"]["U"] > by_condition["independent"]["U"]


def test_at_least_one_mitigation_reduces_strong_kl():
    appeal, baseline = _setup()
    rows = mitigation_table(appeal, baseline, n_agents=N_AGENTS, w=WORLDS)
    by_mitigation = {r["mitigation"]: r for r in rows}
    none_kl = by_mitigation["none"]["KL"]
    reduced = [name for name, r in by_mitigation.items()
               if name != "none" and r["KL"] < none_kl]
    assert reduced, "expected at least one mitigation to reduce KL vs 'none'"


def test_condition_table_deterministic():
    appeal, baseline = _setup(seed=321)
    rows1, _ = condition_table(appeal, baseline, n_agents=N_AGENTS, w=WORLDS)
    rows2, _ = condition_table(appeal, baseline, n_agents=N_AGENTS, w=WORLDS)
    assert rows1 == rows2


def test_seed_robustness_check_runs_and_reports_counts():
    result = seed_robustness_check(seeds=(1, 2), n_agents=120, w=2)
    assert result["n_total"] == 2
    assert 0 <= result["n_holds"] <= 2
    assert len(result["per_seed"]) == 2
