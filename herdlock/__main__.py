"""
CLI entry point: `python -m herdlock --out results`

Runs the condition / susceptibility-sweep / mitigation tables, prints them,
and writes results/summary.json, results/*.csv, and results/herding.png.
Deterministic given the fixed seed in herdlock.experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import experiments as exp

COLOR_INDEPENDENT = "#2a78d6"   # categorical slot 1 (blue)
COLOR_STRONG = "#eb6834"        # categorical slot 2 (orange)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


def print_table(title, rows):
    print(f"\n== {title} ==")
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = [max(len(c), *(len(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    for r in rows:
        cells = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        print("  ".join(cell.ljust(w) for cell, w in zip(cells, widths)))


def markdown_table(title, rows):
    """Render `rows` as a GFM table, rounding the full-precision floats to
    3 decimals directly (no intermediate 4-decimal display step), so README
    tables built from this output can't pick up a double-rounding error."""
    lines = [f"### {title}", ""]
    if not rows:
        return "\n".join(lines)
    cols = list(rows[0].keys())
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = [f"{round(r[c], 3)}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_chart(path, chart_shares):
    independent_sorted = np.sort(chart_shares["independent"])[::-1]
    strong_sorted = np.sort(chart_shares["strong"])[::-1]
    rank = np.arange(1, independent_sorted.size + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(rank, independent_sorted, color=COLOR_INDEPENDENT, linewidth=2,
             solid_capstyle="round", label="independent (hide counts)")
    ax.plot(rank, strong_sorted, color=COLOR_STRONG, linewidth=2,
             solid_capstyle="round", label="strong influence (sorted + counts)")

    ax.set_xlabel("item rank (sorted by share, descending)", color=INK_SECONDARY)
    ax.set_ylabel("market share", color=INK_SECONDARY)
    ax.set_title("Herding concentrates market share on fewer items",
                 color=INK_PRIMARY, fontsize=12, loc="left")

    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    legend = ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="herdlock")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--markdown", action="store_true",
                         help="print GFM markdown tables (round(value, 3) on the "
                              "full-precision results) for pasting into README.md")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    appeal, baseline, tables, chart_shares, robustness = exp.run_all()

    for name, rows in tables.items():
        print_table(name, rows)
        write_csv(os.path.join(args.out, f"{name}.csv"), rows)

    write_chart(os.path.join(args.out, "herding.png"), chart_shares)

    summary = dict(
        params=dict(
            K=exp.K, n_agents=exp.N_AGENTS, budget=exp.BUDGET,
            susceptibility_default=exp.SUSCEPTIBILITY, W=exp.W, seed=exp.SEED,
            sweep_values=list(exp.SWEEP_VALUES),
        ),
        tables=tables,
        seed_robustness=robustness,
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nwrote {args.out}/summary.json, {args.out}/*.csv, {args.out}/herding.png")

    if args.markdown:
        print("\n" + "=" * 60)
        print("markdown (paste verbatim into README.md):")
        print("=" * 60)
        for name, rows in tables.items():
            print()
            print(markdown_table(name, rows))
        print(f"\nseed robustness: ordering held for {robustness['n_holds']}/{robustness['n_total']} "
              f"seeds {robustness['seeds']} (n_agents={robustness['n_agents']}, w={robustness['w']})")


if __name__ == "__main__":
    main()
