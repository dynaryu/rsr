#!/usr/bin/env python
"""Overlay baseline vs. coverage-aware RSR convergence from metrics.json.

Produces the A/B figure for the Reviewer-3 (Comments 3.1/3.2) experiment:
how fast the unclassified probability p^u falls as a function of the *cost*
that actually scales on large networks —

  (a) cumulative system-function (sfun) calls  — the honest cost axis, since
      coverage-aware spends more calls per reference (n_orders) but should
      need far fewer references;
  (b) total number of reference states          — the scaling wall itself
      (e.g. IEEE-118 hitting R_max with ~16% still unclassified).

If coverage-aware reaches a given p^u with fewer references (panel b) at
comparable or lower sfun budget (panel a), that is the result that turns the
Section-5 "limitation" into a positive scalability claim.

Each variant is a directory that is either
  * a single run     : contains metrics.json, or
  * a multi-run base : contains run_00/metrics.json, run_01/, ...
(as written by `run_demo.py` for --runs 1 and --runs N respectively).

Usage:
    python plot_coverage_ab.py --base out/base --cov out/cov
    python plot_coverage_ab.py --base out/base --cov out/cov --out ab.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito blue / vermillion — CVD-safe, validated against a white surface.
# Series identity is also carried by marker shape + a direct end-label, so the
# distinction never rests on colour alone.
# `dy` staggers the direct end-label vertically so the two series never
# overprint when they terminate at the same point (both hit the p^u floor).
STYLE = {
    "baseline":       dict(color="#0072B2", marker="o", label="Baseline", dy=7),
    "coverage-aware": dict(color="#D55E00", marker="s", label="Coverage-aware", dy=-9),
}
FLOOR = 1e-12   # p_unknown reported as 0 (fully classified) is plotted at this floor


def load_runs(variant_dir: Path) -> List[List[Dict]]:
    """Return a list of runs; each run is the list of per-round metric dicts."""
    variant_dir = Path(variant_dir)
    run_dirs = sorted(variant_dir.glob("run_*"))
    metrics_files = ([d / "metrics.json" for d in run_dirs]
                     if run_dirs else [variant_dir / "metrics.json"])
    runs: List[List[Dict]] = []
    for mf in metrics_files:
        if not mf.exists():
            print(f"  warn: {mf} not found, skipping")
            continue
        rounds = [json.loads(ln) for ln in mf.read_text().splitlines() if ln.strip()]
        if rounds:
            runs.append(rounds)
    if not runs:
        raise FileNotFoundError(f"No metrics.json found under {variant_dir}")
    return runs


def run_curves(rounds: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract monotone cost axes and p_unknown for one run.

    n_sfun_upper/lower are per-round counts -> cumulative sum gives the running
    total sfun budget. n_refs_upper/lower are already absolute (running) counts.
    p_unknown is the round's unclassified-probability estimate.
    """
    sfun_round = np.array([r.get("n_sfun_upper", 0) + r.get("n_sfun_lower", 0)
                           for r in rounds], dtype=float)
    cum_sfun = np.cumsum(sfun_round)
    n_refs = np.array([r.get("n_refs_upper", 0) + r.get("n_refs_lower", 0)
                       for r in rounds], dtype=float)
    p_unk = np.array([r.get("p_unknown", np.nan) for r in rounds], dtype=float)
    p_unk = np.where(p_unk > 0, p_unk, FLOOR)
    return {"cum_sfun": cum_sfun, "n_refs": n_refs, "p_unknown": p_unk}


def median_band(runs: List[List[Dict]], xkey: str, n_grid: int = 200):
    """Median p_unknown vs a shared log-spaced cost grid, with min/max band.

    Each run's p_unknown(x) is interpolated (in log space) onto a common grid
    spanning the overlap of all runs, so the median line is well defined even
    though runs stop at different costs.
    """
    curves = [run_curves(r) for r in runs]
    xs = [c[xkey] for c in curves]
    xmin = max(float(x[x > 0].min()) if np.any(x > 0) else 1.0 for x in xs)
    xmax = min(float(x.max()) for x in xs)
    if not (xmax > xmin):
        return None
    grid = np.logspace(np.log10(xmin), np.log10(xmax), n_grid)
    stacked = []
    for c in curves:
        x, y = c[xkey], c["p_unknown"]
        order = np.argsort(x)
        # interpolate log10(p) against log10(x); x is (weakly) increasing
        logy = np.interp(np.log10(grid), np.log10(np.maximum(x[order], 1e-30)),
                         np.log10(y[order]))
        stacked.append(logy)
    logy = np.vstack(stacked)
    return grid, 10 ** np.median(logy, 0), 10 ** logy.min(0), 10 ** logy.max(0)


def draw_panel(ax, variants: Dict[str, List[List[Dict]]], xkey: str, xlabel: str):
    for name, runs in variants.items():
        st = STYLE[name]
        # thin per-run traces for honesty about run-to-run spread
        for r in runs:
            c = run_curves(r)
            ax.plot(np.maximum(c[xkey], FLOOR), c["p_unknown"],
                    color=st["color"], lw=0.8, alpha=0.25, zorder=1)
        band = median_band(runs, xkey) if len(runs) > 1 else None
        if band is not None:
            grid, med, lo, hi = band
            ax.fill_between(grid, lo, hi, color=st["color"], alpha=0.12, lw=0, zorder=2)
            ax.plot(grid, med, color=st["color"], lw=2.0, marker=st["marker"],
                    markevery=[-1], ms=7, zorder=3, label=st["label"])
            ax.annotate(st["label"], (grid[-1], med[-1]), color=st["color"],
                        fontsize=8, fontweight="bold", xytext=(4, st["dy"]),
                        textcoords="offset points", va="center")
        else:
            c = run_curves(runs[0])
            ax.plot(np.maximum(c[xkey], FLOOR), c["p_unknown"], color=st["color"],
                    lw=2.0, marker=st["marker"], markevery=[-1], ms=7,
                    zorder=3, label=st["label"])
            ax.annotate(st["label"], (max(c[xkey][-1], FLOOR), c["p_unknown"][-1]),
                        color=st["color"], fontsize=8, fontweight="bold",
                        xytext=(4, st["dy"]), textcoords="offset points", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="both", ls=":", lw=0.5, color="#c8c8c8", alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path, help="baseline output dir")
    ap.add_argument("--cov", required=True, type=Path, help="coverage-aware output dir")
    ap.add_argument("--out", type=Path, default=Path("coverage_ab.png"))
    ap.add_argument("--title", default="RSR convergence: baseline vs. coverage-aware")
    args = ap.parse_args()

    variants = {
        "baseline": load_runs(args.base),
        "coverage-aware": load_runs(args.cov),
    }
    for name, runs in variants.items():
        print(f"{name}: {len(runs)} run(s)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    draw_panel(ax1, variants, "cum_sfun", "Cumulative system-function calls")
    draw_panel(ax2, variants, "n_refs", "Number of reference states")
    ax1.set_ylabel(r"Unclassified probability  $p^{\,u}$")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
