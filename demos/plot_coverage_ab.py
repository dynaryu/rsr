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

# Okabe-Ito blue / vermillion / bluish-green / purple — CVD-safe, validated
# against a white surface. Series identity is also carried by marker shape + a
# direct end-label, so the distinction never rests on colour alone. `dy`
# staggers the end-labels so variants ending at the same point never overprint.
PALETTE = [
    dict(color="#0072B2", marker="o", dy=7),
    dict(color="#D55E00", marker="s", dy=-9),
    dict(color="#009E73", marker="^", dy=17),
    dict(color="#CC79A7", marker="D", dy=-19),
]
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

    n_sfun_upper/lower and time_sec are per-round -> cumulative sum gives the
    running total sfun budget / wall-clock. n_refs_upper/lower are already
    absolute (running) counts. p_unknown is the round's estimate.
    """
    sfun_round = np.array([r.get("n_sfun_upper", 0) + r.get("n_sfun_lower", 0)
                           for r in rounds], dtype=float)
    cum_sfun = np.cumsum(sfun_round)
    time_round = np.array([r.get("time_sec", 0.0) for r in rounds], dtype=float)
    cum_min = np.cumsum(time_round) / 60.0     # cumulative wall-clock, minutes
    n_refs = np.array([r.get("n_refs_upper", 0) + r.get("n_refs_lower", 0)
                       for r in rounds], dtype=float)
    p_unk = np.array([r.get("p_unknown", np.nan) for r in rounds], dtype=float)
    p_unk = np.where(p_unk > 0, p_unk, FLOOR)
    return {"cum_sfun": cum_sfun, "cum_min": cum_min, "n_refs": n_refs,
            "p_unknown": p_unk}


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


def draw_panel(ax, variants: List[Dict], xkey: str, xlabel: str):
    """variants: ordered list of dicts with keys label, runs, and a style
    (color/marker/dy) taken from PALETTE by position."""
    for v in variants:
        st, runs, label = v["style"], v["runs"], v["label"]
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
                    markevery=[-1], ms=7, zorder=3, label=label)
            xend, yend = grid[-1], med[-1]
        else:
            c = run_curves(runs[0])
            ax.plot(np.maximum(c[xkey], FLOOR), c["p_unknown"], color=st["color"],
                    lw=2.0, marker=st["marker"], markevery=[-1], ms=7,
                    zorder=3, label=label)
            xend, yend = max(c[xkey][-1], FLOOR), c["p_unknown"][-1]
        ax.annotate(label, (xend, yend), color=st["color"], fontsize=8,
                    fontweight="bold", xytext=(4, st["dy"]),
                    textcoords="offset points", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="both", ls=":", lw=0.5, color="#c8c8c8", alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="baseline output dir (labelled 'Baseline')")
    ap.add_argument("--cov", type=Path, help="coverage-aware output dir (labelled 'Coverage-aware')")
    ap.add_argument("--variant", action="append", default=[], metavar="LABEL=DIR",
                    help="extra variant as 'Label=path'; repeatable. Order sets colour.")
    ap.add_argument("--out", type=Path, default=Path("coverage_ab.png"))
    ap.add_argument("--title", default="RSR convergence: baseline vs. coverage-aware")
    args = ap.parse_args()

    # Assemble ordered (label, dir) list from the convenience flags + --variant.
    spec: List[tuple] = []
    if args.base is not None:
        spec.append(("Baseline", args.base))
    if args.cov is not None:
        spec.append(("Coverage-aware", args.cov))
    for item in args.variant:
        if "=" not in item:
            ap.error(f"--variant must be LABEL=DIR, got {item!r}")
        label, _, d = item.rpartition("=")   # split on last '=' (labels may contain '=')
        spec.append((label, Path(d)))
    if not spec:
        ap.error("provide at least one of --base / --cov / --variant")

    variants = []
    for i, (label, d) in enumerate(spec):
        runs = load_runs(d)
        print(f"{label}: {len(runs)} run(s)  <- {d}")
        variants.append({"label": label, "runs": runs,
                         "style": PALETTE[i % len(PALETTE)]})

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    draw_panel(ax1, variants, "cum_min", "Cumulative wall-clock time (minutes)")
    draw_panel(ax2, variants, "cum_sfun", "Cumulative system-function calls")
    draw_panel(ax3, variants, "n_refs", "Number of reference states")
    ax1.set_ylabel(r"Unclassified probability  $p^{\,u}$")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
