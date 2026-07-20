"""Estimate P(blackout) by Subset Simulation for a blackout dataset.

Thin CLI around rsr.subset_sim.subset_sim_estimate — the tractable estimator
where the certificate rules cover ~0% of the space and the residual is the
rare failure tail (e.g. ACTIVSg2000), so the plain-MC hybrid is hopeless.

It runs several independent SuS repetitions and reports the mean p_f with an
empirical 95% CI. NOTE: this is *basic* Subset Simulation (fixed p0,
component-wise MH); on IEEE-14 it lands within ~1.5x of the reference
1.1e-4 — a good order-of-magnitude estimate, not a precise one. Precision at
this scale needs the paper's aE-SuS (adaptive levels + a better MCMC kernel).

Usage:
    python sus_estimate.py --dataset ieee14 --threshold 54.8
    python sus_estimate.py --dataset ACTIVSg2000 --threshold 4.7 --n-runs 5
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blackout_lib import build_model               # noqa: E402
from rsr.subset_sim import subset_sim_estimate      # noqa: E402

_ND = HERE.parent.parent / "network-datasets/datasets"
DEFAULT_THRESHOLD = {"ieee14": 54.8, "ieee118": 13.8, "ieee300": 26.1,
                     "ACTIVSg2000": 4.7}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ACTIVSg2000")
    ap.add_argument("--threshold", type=float, default=None,
                    help="blackout %% failure threshold (default: per-dataset)")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--n-runs", type=int, default=6)
    ap.add_argument("--n-per-level", type=int, default=1000)
    ap.add_argument("--p0", type=float, default=0.1)
    ap.add_argument("--max-levels", type=int, default=15)
    ap.add_argument("--n-workers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full result dict (incl. per-run) here")
    args = ap.parse_args()

    import multiprocessing as mp
    n_workers = mp.cpu_count() if args.n_workers < 0 else args.n_workers
    threshold = args.threshold if args.threshold is not None \
        else DEFAULT_THRESHOLD.get(args.dataset, 10.0)
    ds = _ND / args.dataset / "v1"

    probs, row_names, n_state, sfun, mi = build_model(ds, "cpu", threshold, args.alpha)
    print(f"\nSubset-Simulation estimate: {args.n_runs} runs x "
          f"{args.n_per_level}/level, p0={args.p0}, threshold={threshold}%\n")
    t0 = time.time()
    res = subset_sim_estimate(
        probs, mi["dcopt"], row_names, sys_surv_st=1,
        n_runs=args.n_runs, n_per_level=args.n_per_level, p0=args.p0,
        max_levels=args.max_levels, severity_sign=+1,
        n_workers=n_workers, seed=args.seed, verbose=True)

    print("\n" + "=" * 60)
    if res["p_fail"] > 0:
        lo, hi = res["ci95"]
        print(f"  P(blackout) = {res['p_fail']:.3e}   "
              f"95% CI [{lo:.3e}, {hi:.3e}]")
        print(f"  single-run c.o.v. = {res['cov_single_run']:.2f}   "
              f"({res['n_sfun']:,} sfun over {args.n_runs} runs, "
              f"{time.time() - t0:.0f}s)")
    else:
        print("  No failures reached — raise --max-levels or --n-per-level.")
    if res["n_runs_no_failure"]:
        print(f"  ({res['n_runs_no_failure']} run(s) reached no failures)")

    if args.json is not None:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        meta = {"dataset": args.dataset, "threshold": threshold,
                "alpha": args.alpha, "n_per_level": args.n_per_level,
                "p0": args.p0, "seed": args.seed, **res}
        json.dump(meta, open(args.json, "w"), indent=2, default=float)
        print(f"  saved {args.json}")


if __name__ == "__main__":
    main()
