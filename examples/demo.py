"""Prints a short headline summary: strong herding vs independent, and the
best cheap mitigation found under strong influence."""

from herdlock import experiments as exp


def main():
    appeal, baseline, tables, _, _ = exp.run_all()

    cond = {row["condition"]: row for row in tables["condition"]}
    indep, strong = cond["independent"], cond["strong"]

    mitig_rows = [r for r in tables["mitigation"] if r["mitigation"] != "independent"]
    best = min(mitig_rows, key=lambda r: r["KL"])
    none_row = next(r for r in tables["mitigation"] if r["mitigation"] == "none")

    print(f"herdlock: {exp.W} worlds x {exp.N_AGENTS} agents, susceptibility={exp.SUSCEPTIBILITY}")
    print(f"independent: KL={indep['KL']:.3f} Gini={indep['Gini']:.3f} corr_with_appeal={indep['corr_appeal']:.3f}")
    print(f"strong influence: KL={strong['KL']:.3f} Gini={strong['Gini']:.3f} corr_with_appeal={strong['corr_appeal']:.3f}")
    print(f"strong influence raises KL by {strong['KL'] - indep['KL']:.3f} nats and Gini by "
          f"{strong['Gini'] - indep['Gini']:.3f} over independent choice.")
    print(f"early-mover correlation under strong influence: {strong['early_mover_corr']:.3f} "
          f"(independent: {indep['early_mover_corr']:.3f})")
    print(f"best cheap mitigation under strong influence (excl. hiding counts entirely): "
          f"'{best['mitigation']}', cutting KL from {none_row['KL']:.3f} to {best['KL']:.3f}")


if __name__ == "__main__":
    main()
