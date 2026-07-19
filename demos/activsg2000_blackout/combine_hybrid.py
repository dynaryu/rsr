"""Pool the sharded hybrid estimates from a PBS array job into one estimate.

Each array task wrote out_cert_nci/hybrid_shards/task_<i>/hybrid.json from an
independent seed. Because the hybrid estimator is a sum of iid Bernoulli
trials, the shards combine by simply summing failures and samples; the pooled
95% CI is one Wilson interval on the totals (identical to running one big job).

Usage:
    python combine_hybrid.py [--base out_cert_nci]
Writes <base>/hybrid_combined.json and prints the pooled estimate.
"""
import argparse
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# additive fields across shards
SUM_KEYS = ["n_sample", "n_fail", "n_fail_certified", "n_fail_sfun",
            "n_upper_certified", "n_sfun"]


def wilson(k, n, z=1.959964):
    """Wilson score interval (matches rsr.rsr._wilson_interval)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "out_cert_nci"))
    args = ap.parse_args()
    base = Path(args.base)

    shards = sorted(glob.glob(str(base / "hybrid_shards" / "task_*" / "hybrid.json")))
    if not shards:
        raise SystemExit(f"no shards under {base}/hybrid_shards/task_*/hybrid.json")

    tot = {k: 0 for k in SUM_KEYS}
    seeds, per_shard = [], []
    for s in shards:
        d = json.load(open(s))
        for k in SUM_KEYS:
            tot[k] += d[k]
        seeds.append(d.get("seed"))
        per_shard.append((Path(s).parent.name, d["n_sample"], d["n_fail"]))

    if len(set(seeds)) != len(seeds):
        print(f"WARNING: shard seeds are not all distinct ({seeds}) — the "
              "shards are correlated and the CI is optimistic.")

    lo, hi = wilson(tot["n_fail"], tot["n_sample"])
    p = tot["n_fail"] / tot["n_sample"]
    result = {
        "n_shards": len(shards),
        "p_fail": p,
        "ci95": [lo, hi],
        "rel_halfwidth": (hi - lo) / 2 / p if p > 0 else None,
        **tot,
        "free_frac": 1.0 - tot["n_sfun"] / tot["n_sample"],
        "seeds": seeds,
    }
    json.dump(result, open(base / "hybrid_combined.json", "w"), indent=2)

    print(f"combined {len(shards)} shards")
    for name, n, nf in per_shard:
        print(f"  {name}: {nf:>5} fails / {n:,} samples")
    print(f"\n  P(blackout) = {p:.3e}   95% CI [{lo:.3e}, {hi:.3e}]  "
          f"(+-{result['rel_halfwidth'] * 100:.0f}%)" if p > 0 else
          f"\n  P(blackout) = 0   95% CI [{lo:.3e}, {hi:.3e}]")
    print(f"  {tot['n_fail']:,} failures ({tot['n_fail_certified']:,} "
          f"cut-certified + {tot['n_fail_sfun']:,} solver) / "
          f"{tot['n_sample']:,} samples")
    print(f"  free classification: {result['free_frac'] * 100:.2f}%  "
          f"({tot['n_sfun']:,} solver calls)")
    print(f"  saved {base / 'hybrid_combined.json'}")


if __name__ == "__main__":
    main()
