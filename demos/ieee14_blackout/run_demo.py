"""
RSR demo: IEEE 14-bus DC-OPF blackout reliability.

Estimates the probability of a system-wide blackout for the IEEE 14-bus power
system and identifies the critical failure modes (minimal cut-sets) that drive
it, using RSR's branch-and-bound rule extraction.

The model is the multi-state DC-OPF blackout case from Chan et al. (2024):
  - 34 failable components: 5 generator buses (4 states), 9 ordinary buses
    (2 states) and 20 branches (2 states);
  - each component is failed when its state index is low and operational when
    high, so the system function is coherent (monotone) in the state indices;
  - the system FAILS when the DC-OPF blackout size exceeds 54.8% of demand
    (Scenario 1 in the reference), otherwise it SURVIVES.

RSR discovers, by Monte Carlo sampling + minimisation, two reusable rule sets:
  - survival rules (sys >= 1): minimal operational states that guarantee no
    blackout;
  - failure rules  (sys <= 0): minimal cut-sets whose joint degradation forces
    a blackout.
These bound P(blackout) from below and above (the gap is the "unknown"
probability) and the failure rules ARE the minimal cut-sets, from which the
per-component criticality ranking is read off.

Data is read from the `ieee14` network dataset (edges/probs + MATPOWER case)
and the system function is the dataset's pure-Python DC-OPF sfun (scipy linprog,
no MATLAB dependency).

Usage:
    python run_demo.py                       # defaults: ~/Projects/network-datasets/.../ieee14/v1
    python run_demo.py --unk-thres 1e-5      # tighter bound (smaller unknown gap)
    python run_demo.py --dataset /path/to/ieee14/v1 --device cpu
    python run_demo.py --devices cuda:0,cuda:1
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# Reference from the paper (Chan et al. 2024, Table 2, Scenario 1)
REF_PF = 1.1e-4

HERE = Path(__file__).resolve().parent
# repo root (…/rsr) so `import rsr.rsr` works when run from anywhere
sys.path.insert(0, str(HERE.parents[1]))

import rsr.rsr as rsr  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="RSR IEEE 14-bus blackout demo")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / "Projects/network-datasets/datasets/ieee14/v1",
        help="Path to the ieee14 dataset version dir (contains data/ and scripts/)",
    )
    p.add_argument("--threshold", type=float, default=54.8,
                   help="Blackout size (%%) above which the system fails")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Branch capacity scaling factor for the DC-OPF case")
    p.add_argument("--unk-thres", type=float, default=1e-6,
                   help="Convergence threshold on the unknown (bound-gap) probability")
    p.add_argument("--unk-opt", choices=["abs", "rel"], default="abs",
                   help="Interpret --unk-thres as absolute or relative to P(failure)")
    p.add_argument("--max_rounds", type=int, default=500_000,
                   help="max no. of round")
    p.add_argument("--n-sample", type=int, default=10_000_000,
                   help="Samples per probability/search round")
    p.add_argument("--batch", type=int, default=100_000, help="Sample batch size")
    p.add_argument("--device", type=str, default="",
                   help="Single torch device, e.g. 'cpu' or 'cuda' (default: auto)")
    p.add_argument("--devices", type=str, default="",
                   help="Comma-separated multi-GPU devices, e.g. 'cuda:0,cuda:1'")
    p.add_argument("--out", type=Path, default=HERE / "out",
                   help="Output directory for rules, reliability and criticality")
    return p.parse_args()


def load_dataset(dataset: Path):
    """Load the ieee14 edges/probs and return the DC-OPF system function factory."""
    data_dir = dataset / "data"
    scripts_dir = dataset / "scripts"
    if not (data_dir / "probs.json").exists():
        sys.exit(f"Dataset not found under {dataset} (missing data/probs.json)")
    # the dataset ships its own DC-OPF sfun; import it directly
    sys.path.insert(0, str(scripts_dir))
    from sfun_dcopt import make_dcopt_sfun  # noqa: E402

    probs_dict = json.load(open(data_dir / "probs.json"))
    return probs_dict, make_dcopt_sfun, str(data_dir / "ieee14.m")


def build_probs_tensor(probs_dict, device):
    """(n_comp, n_state) probability tensor, padded with zeros to the max state count."""
    row_names = list(probs_dict.keys())
    n_state = max(len(v) for v in probs_dict.values())
    rows = []
    for name in row_names:
        p = probs_dict[name]
        rows.append([p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)])
    return torch.tensor(rows, dtype=torch.float32, device=device), row_names, n_state


def cutsets_from_failure_rules(rules_lower):
    """Minimal cut-sets = failure rules without the bookkeeping 'sys' key.

    Each cut-set is a list of (component, max_failing_state) — the components fail
    the system when every one is at or below its listed state index.
    """
    cutsets = []
    for r in rules_lower:
        comps = {k: v[1] for k, v in r.items() if k != "sys"}
        if comps:
            cutsets.append(comps)
    return cutsets


def criticality(cutsets):
    """Rank components by how they contribute to failure.

    Single-component cut-sets are single points of failure; otherwise rank by the
    number of cut-sets a component appears in and the smallest cut-set it is in.
    """
    spof = {next(iter(cs)) for cs in cutsets if len(cs) == 1}
    freq = Counter(c for cs in cutsets for c in cs)
    ranking = []
    for comp, f in freq.items():
        ranking.append({
            "component": comp,
            "single_point_of_failure": comp in spof,
            "appears_in_n_cutsets": f,
            "min_cutset_size": min(len(cs) for cs in cutsets if comp in cs),
        })
    ranking.sort(key=lambda r: (not r["single_point_of_failure"],
                                r["min_cutset_size"],
                                -r["appears_in_n_cutsets"]))
    return ranking


def main():
    args = parse_args()
    device_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    multi_devices = device_list if len(device_list) > 1 else None
    if args.device:
        device = torch.device(args.device)
    elif device_list:
        device = torch.device(device_list[0])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("RSR demo — IEEE 14-bus DC-OPF blackout reliability")
    print("=" * 64)

    # 1. Load dataset -------------------------------------------------------
    probs_dict, make_dcopt_sfun, case_path = load_dataset(args.dataset)
    probs, row_names, n_state = build_probs_tensor(probs_dict, device)

    n_gen = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 4)
    n_ord = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 2)
    n_br = sum(1 for n in row_names if n.startswith("br"))
    print(f"\n  Dataset:     {args.dataset}")
    print(f"  Components:  {len(row_names)} "
          f"({n_gen} generator buses, {n_ord} ordinary buses, {n_br} branches)")
    print(f"  Max states:  {n_state}   Device: {device}")
    print(f"  Threshold:   {args.threshold}% blackout → system failure")

    # 2. Build the DC-OPF system function ----------------------------------
    print("\nInitialising DC-OPF system function...")
    dcopt = make_dcopt_sfun(case_path=case_path,
                            blackout_threshold=args.threshold, alpha=args.alpha)

    # RSR needs sfun(comps_st) -> (sys_value, sys_state, min_comps_st|None).
    # Here the system value is just the binary system state (0 fail / 1 survive).
    def sfun(comps_st):
        _blackout, sys_st, _ = dcopt(comps_st)
        return sys_st, sys_st, None

    # Sanity: fully operational vs fully failed (highest / lowest state per comp)
    all_ok = {n: max(int(s) for s in probs_dict[n]) for n in row_names}
    all_bad = {n: 0 for n in row_names}
    b_ok, st_ok, _ = dcopt(all_ok)
    b_bad, st_bad, _ = dcopt(all_bad)
    print(f"  All operational: blackout={b_ok:6.2f}%  sys_st={st_ok}")
    print(f"  All failed:      blackout={b_bad:6.2f}%  sys_st={st_bad}")

    # 3. RSR rule extraction ------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting survival / failure rules (unknown gap < {args.unk_thres:g} "
          f"[{args.unk_opt}])...\n", flush=True)
    t0 = time.time()
    res = rsr.run_rule_extraction_by_mcs(
        sfun=sfun,
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        sys_upper_st=1,               # system states: 0 = blackout, 1 = survive
        rules_upper=[],
        rules_lower=[],
        unk_prob_thres=args.unk_thres,
        unk_prob_opt=args.unk_opt,
        n_sample=args.n_sample,
        max_rounds=args.max_rounds,
        sample_batch_size=args.batch,
        devices=multi_devices,
        output_dir=str(args.out),
    )
    elapsed = time.time() - t0

    # 4. Read back rules and summarise -------------------------------------
    rules_surv = json.load(open(res["rules_upper_path"]))
    rules_fail = json.load(open(res["rules_lower_path"]))
    last = res["metrics_log"][-1]
    p_fail = last["p_lower"]          # lower system state = blackout
    p_surv = last["p_upper"]
    p_unk = last["p_unknown"]

    cutsets = cutsets_from_failure_rules(rules_fail)
    crit = criticality(cutsets)
    spofs = [c["component"] for c in crit if c["single_point_of_failure"]]

    # order cut-sets smallest-first for reporting
    cutsets_sorted = sorted(cutsets, key=len)

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  Runtime:            {elapsed:.1f}s   "
          f"({len(rules_surv)} survival rules, {len(rules_fail)} failure rules)")
    print(f"  P(blackout):        {p_fail:.3e}   "
          f"(bounds: [{p_fail:.3e}, {p_fail + p_unk:.3e}], gap {p_unk:.1e})")
    print(f"  P(survival):        {p_surv:.4f}")
    print(f"  Reference (paper):  p_f ~ {REF_PF:.1e}")
    print(f"\n  {len(cutsets)} minimal cut-sets (failure modes).")
    print(f"  Single points of failure: {spofs or 'none'}")
    print("  Smallest failure modes:")
    for cs in cutsets_sorted[:8]:
        conds = ", ".join(f"{k}≤{v}" for k, v in cs.items())
        print(f"    - {{{conds}}}")
    print("\n  Most critical components:")
    for r in crit[:8]:
        tag = " (single point of failure)" if r["single_point_of_failure"] else ""
        print(f"    {r['component']:8s} in {r['appears_in_n_cutsets']:2d} cut-sets "
              f"(min size {r['min_cutset_size']}){tag}")

    # 5. Save artifacts -----------------------------------------------------
    reliability = {
        "p_blackout": p_fail,
        "p_survival": p_surv,
        "unknown_gap": p_unk,
        "p_blackout_bounds": [p_fail, p_fail + p_unk],
        "reference_p_failure": REF_PF,
        "blackout_threshold_pct": args.threshold,
        "runtime_sec": elapsed,
        "n_survival_rules": len(rules_surv),
        "n_failure_rules": len(rules_fail),
        "n_cutsets": len(cutsets),
        "single_points_of_failure": spofs,
        "smallest_cutsets": cutsets_sorted[:20],
    }
    with open(args.out / "reliability.json", "w") as f:
        json.dump(reliability, f, indent=2)
    with open(args.out / "critical_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "single_point_of_failure",
                                          "appears_in_n_cutsets", "min_cutset_size"])
        w.writeheader()
        w.writerows(crit)

    print(f"\n  Saved: {args.out/'reliability.json'}")
    print(f"         {args.out/'critical_components.csv'}")
    print(f"         {Path(res['rules_upper_path']).name} / "
          f"{Path(res['rules_lower_path']).name} (rule sets)")


if __name__ == "__main__":
    main()
