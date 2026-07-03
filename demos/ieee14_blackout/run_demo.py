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
    # single run: full report (bounds, cut-sets, criticality)
    python run_demo.py
    python run_demo.py --unk-thres 1e-5 --device cpu
    python run_demo.py --devices cuda:0,cuda:1

    # repeat N times and summarise the run-to-run spread from metrics.json
    python run_demo.py --runs 10
    python run_demo.py --runs 5 --unk-thres 1e-5 --n-sample 1000000
"""

import csv
import json
import os
import statistics as st
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import torch
import typer
import pdb

# Reference from the paper (Chan et al. 2024, Table 2, Scenario 1)
REF_PF = 1.1e-4

HERE = Path(__file__).resolve().parent
# repo root (…/rsr) so `import rsr.rsr` works when run from anywhere
sys.path.insert(0, str(HERE.parents[1]))

import rsr.rsr as rsr  # noqa: E402

DEFAULT_DATASET = Path.home() / "Projects/network-datasets/datasets/ieee14/v1"

app = typer.Typer(add_completion=False, help=__doc__)


# ----------------------------------------------------------------------------
# Data / model
# ----------------------------------------------------------------------------
def load_dataset(dataset: Path):
    """Load the ieee14 probs and return the DC-OPF system function factory."""
    data_dir = dataset / "data"
    scripts_dir = dataset / "scripts"
    if not (data_dir / "probs.json").exists():
        raise typer.BadParameter(f"Dataset not found under {dataset} (missing data/probs.json)")
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


def build_model(dataset, device, threshold, alpha):
    """Load the dataset once and build the probs tensor + RSR system function.

    Returns everything the extraction loop needs, so a multi-run only pays the
    dataset/DC-OPF setup cost once.
    """
    probs_dict, make_dcopt_sfun, case_path = load_dataset(dataset)
    probs, row_names, n_state = build_probs_tensor(probs_dict, device)

    n_gen = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 4)
    n_ord = sum(1 for n in row_names if n.startswith("vbus") and len(probs_dict[n]) == 2)
    n_br = sum(1 for n in row_names if n.startswith("br"))
    print(f"  Dataset:     {dataset}")
    print(f"  Components:  {len(row_names)} "
          f"({n_gen} generator buses, {n_ord} ordinary buses, {n_br} branches)")
    print(f"  Max states:  {n_state}   Device: {device}")
    print(f"  Threshold:   {threshold}% blackout → system failure")

    dcopt = make_dcopt_sfun(case_path=case_path, blackout_threshold=threshold, alpha=alpha)

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

    return probs, row_names, n_state, sfun


def extract(sfun, probs, row_names, n_state, out: Path, *, unk_thres, unk_opt,
            max_rounds, n_sample, batch, multi_devices, quiet):
    """Run one RSR rule extraction into `out`. Returns (res, wall_seconds)."""
    out.mkdir(parents=True, exist_ok=True)
    # metrics.json is appended to by RSR; drop any stale copy from a prior run
    stale = out / "metrics.json"
    if stale.exists():
        stale.unlink()

    def _go():
        return rsr.run_ref_extraction_by_mcs(
            sfun=sfun, probs=probs, row_names=row_names, n_state=n_state,
            sys_upper_st=1,               # system states: 0 = blackout, 1 = survive
            refs_upper=[], refs_lower=[],
            unk_prob_thres=unk_thres, unk_prob_opt=unk_opt,
            n_sample=n_sample, max_rounds=max_rounds, sample_batch_size=batch,
            devices=multi_devices, output_dir=str(out),
        )

    t0 = time.time()
    if quiet:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            res = _go()
    else:
        res = _go()
    return res, time.time() - t0


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------
def cutsets_from_failure_rules(rules_lower):
    """Minimal cut-sets = failure rules without the bookkeeping 'sys' key.

    Each cut-set is a dict {component: max_failing_state} — the components fail
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


def analyse(res, threshold, elapsed):
    """Turn an extraction result into the reliability dict + criticality ranking."""
    rules_surv = json.load(open(res["refs_upper_path"]))
    rules_fail = json.load(open(res["refs_lower_path"]))
    last = res["metrics_log"][-1]
    p_fail, p_surv, p_unk = last["p_lower"], last["p_upper"], last["p_unknown"]

    cutsets = sorted(cutsets_from_failure_rules(rules_fail), key=len)
    crit = criticality(cutsets)
    spofs = [c["component"] for c in crit if c["single_point_of_failure"]]

    reliability = {
        "p_blackout": p_fail,
        "p_survival": p_surv,
        "unknown_gap": p_unk,
        "p_blackout_bounds": [p_fail, p_fail + p_unk],
        "reference_p_failure": REF_PF,
        "blackout_threshold_pct": threshold,
        "runtime_sec": elapsed,
        "n_survival_rules": len(rules_surv),
        "n_failure_rules": len(rules_fail),
        "n_cutsets": len(cutsets),
        "single_points_of_failure": spofs,
        "smallest_cutsets": cutsets[:20],
    }
    return reliability, crit, cutsets


def save_artifacts(out: Path, reliability, crit):
    with open(out / "reliability.json", "w") as f:
        json.dump(reliability, f, indent=2)
    with open(out / "critical_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "single_point_of_failure",
                                          "appears_in_n_cutsets", "min_cutset_size"])
        w.writeheader()
        w.writerows(crit)


def print_report(res, reliability, crit, cutsets, out: Path):
    """Detailed single-run report."""
    r = reliability
    p_fail, p_unk = r["p_blackout"], r["unknown_gap"]
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  Runtime:            {r['runtime_sec']:.1f}s   "
          f"({r['n_survival_rules']} survival rules, {r['n_failure_rules']} failure rules)")
    print(f"  P(blackout):        {p_fail:.3e}   "
          f"(bounds: [{p_fail:.3e}, {p_fail + p_unk:.3e}], gap {p_unk:.1e})")
    print(f"  P(survival):        {r['p_survival']:.4f}")
    print(f"  Reference (paper):  p_f ~ {REF_PF:.1e}")
    print(f"\n  {r['n_cutsets']} minimal cut-sets (failure modes).")
    print(f"  Single points of failure: {r['single_points_of_failure'] or 'none'}")
    print("  Smallest failure modes:")
    for cs in cutsets[:8]:
        conds = ", ".join(f"{k}≤{v}" for k, v in cs.items())
        print(f"    - {{{conds}}}")
    print("\n  Most critical components:")
    for c in crit[:8]:
        tag = " (single point of failure)" if c["single_point_of_failure"] else ""
        print(f"    {c['component']:8s} in {c['appears_in_n_cutsets']:2d} cut-sets "
              f"(min size {c['min_cutset_size']}){tag}")
    print(f"\n  Saved: {out/'reliability.json'}")
    print(f"         {out/'critical_components.csv'}")
    print(f"         {Path(res['rules_upper_path']).name} / "
          f"{Path(res['rules_lower_path']).name} (rule sets)")


# ----------------------------------------------------------------------------
# Multi-run summary
# ----------------------------------------------------------------------------
def read_metrics(metrics_path: Path, runtime_s: float):
    """Summarise one run's metrics.json (JSON-lines, one entry per round)."""
    rows = [json.loads(ln) for ln in metrics_path.read_text().splitlines() if ln.strip()]
    last = rows[-1]                                  # final independent estimate
    return {
        "p_blackout": last["p_lower"],
        "p_survival": last["p_upper"],
        "gap": last["p_unknown"],
        "n_surv_rules": last["n_refs_upper"],
        "n_fail_rules": last["n_refs_lower"],
        "rounds": max(r.get("round", 0) for r in rows),
        "runtime_s": runtime_s,
    }


def agg(values):
    """mean / std / min / max / coefficient-of-variation for a list of numbers."""
    mean = st.fmean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": sd, "min": min(values), "max": max(values),
            "cv": (sd / mean) if mean else 0.0}


def summarise(runs, out: Path):
    """Per-run table + aggregate stats across repeated runs."""
    n = len(runs)
    print("\n" + "=" * 74)
    print(f"SUMMARY over {n} runs")
    print("=" * 74)
    hdr = (f"{'run':>4} {'rounds':>7} {'runtime_s':>10} {'P(blackout)':>13} "
           f"{'gap':>10} {'surv':>6} {'fail':>6}")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(runs):
        print(f"{i:>4} {m['rounds']:>7} {m['runtime_s']:>10.1f} "
              f"{m['p_blackout']:>13.3e} {m['gap']:>10.1e} "
              f"{m['n_surv_rules']:>6} {m['n_fail_rules']:>6}")

    fields = {
        "P(blackout)": [m["p_blackout"] for m in runs],
        "bound gap":   [m["gap"] for m in runs],
        "surv rules":  [m["n_surv_rules"] for m in runs],
        "fail rules":  [m["n_fail_rules"] for m in runs],
        "rounds":      [m["rounds"] for m in runs],
        "runtime_s":   [m["runtime_s"] for m in runs],
    }
    print("\n" + "-" * 74)
    print(f"{'metric':>12} {'mean':>12} {'std':>12} {'min':>12} {'max':>12} {'cv%':>7}")
    print("-" * 74)
    summary = {}
    for name, vals in fields.items():
        a = agg(vals)
        summary[name] = a
        fmt = ("{:>12.3e}" if max(abs(v) for v in vals) < 1
               and any(v != int(v) for v in vals) else "{:>12.2f}")
        print(f"{name:>12} " + " ".join(fmt.format(a[k]) for k in ("mean", "std", "min", "max"))
              + f" {a['cv'] * 100:>6.1f}")

    mean_pf = summary["P(blackout)"]["mean"]
    print("-" * 74)
    print(f"\n  Mean P(blackout) = {mean_pf:.3e}   "
          f"(reference p_f ~ {REF_PF:.1e}, ratio {mean_pf / REF_PF:.2f})")

    out_json = out / "summary.json"
    with open(out_json, "w") as f:
        json.dump({"runs": runs, "aggregate": summary,
                   "mean_p_blackout": mean_pf, "reference_p_failure": REF_PF},
                  f, indent=2)
    print(f"  Saved: {out_json}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
@app.command()
def main(
    runs: int = typer.Option(1, help="Repetitions; 1 = full report, >1 = summarise metrics.json"),
    dataset: Path = typer.Option(DEFAULT_DATASET, help="ieee14 dataset version dir (data/ + scripts/)"),
    threshold: float = typer.Option(54.8, help="Blackout size (%) above which the system fails"),
    alpha: float = typer.Option(2.0, help="Branch capacity scaling factor for the DC-OPF case"),
    unk_thres: float = typer.Option(1e-6, help="Convergence threshold on the unknown (bound-gap) probability"),
    unk_opt: str = typer.Option("abs", help="Interpret --unk-thres as 'abs' or 'rel' to P(failure)"),
    max_rounds: int = typer.Option(500_000, help="Max extraction rounds"),
    n_sample: int = typer.Option(10_000_000, help="Samples per probability/search round"),
    batch: int = typer.Option(100_000, help="Sample batch size"),
    device: str = typer.Option("", help="Single torch device, e.g. 'cpu' or 'cuda' (default: auto)"),
    devices: str = typer.Option("", help="Comma-separated multi-GPU devices, e.g. 'cuda:0,cuda:1'"),
    out: Path = typer.Option(HERE / "out", help="Output dir (single run) / base dir (multi run)"),
    verbose: bool = typer.Option(False, help="Show RSR's per-round log during multi runs"),
):
    """Estimate P(blackout) and the critical failure modes for the IEEE 14-bus grid."""
    device_list = [d.strip() for d in devices.split(",") if d.strip()]
    multi_devices = device_list if len(device_list) > 1 else None
    if device:
        dev = torch.device(device)
    elif device_list:
        dev = torch.device(device_list[0])
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("RSR demo — IEEE 14-bus DC-OPF blackout reliability")
    print("=" * 64)

    # Build the model once (shared across all repetitions).
    probs, row_names, n_state, sfun = build_model(dataset, dev, threshold, alpha)

    common = dict(unk_thres=unk_thres, unk_opt=unk_opt, max_rounds=max_rounds,
                  n_sample=n_sample, batch=batch, multi_devices=multi_devices)

    if runs <= 1:
        # ---- single run: full report ----
        print(f"\nExtracting survival / failure rules "
              f"(unknown gap < {unk_thres:g} [{unk_opt}])...\n", flush=True)
        res, elapsed = extract(sfun, probs, row_names, n_state, out, quiet=False, **common)
        reliability, crit, cutsets = analyse(res, threshold, elapsed)
        save_artifacts(out, reliability, crit)
        print_report(res, reliability, crit, cutsets, out)
        return

    # ---- multi run: repeat and summarise metrics.json ----
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nRepeating extraction x{runs} "
          f"(unknown gap < {unk_thres:g} [{unk_opt}], samples/round {n_sample:,})\n", flush=True)
    run_metrics = []
    all_metrics = []          # full per-round metrics.json of every run, aggregated
    for i in range(runs):
        run_out = out / f"run_{i:02d}"
        print(f"[{i + 1}/{runs}] {run_out.name} ...", end="", flush=True)
        res, elapsed = extract(sfun, probs, row_names, n_state, run_out,
                               quiet=not verbose, **common)
        # keep per-run artifacts too (reliability + criticality)
        reliability, crit, _ = analyse(res, threshold, elapsed)
        save_artifacts(run_out, reliability, crit)
        metrics_path = run_out / "metrics.json"
        m = read_metrics(metrics_path, elapsed)
        run_metrics.append(m)
        # collect the full per-round metrics so all runs live in one file
        rounds = [json.loads(ln) for ln in metrics_path.read_text().splitlines() if ln.strip()]
        all_metrics.append({"run": i, "metrics": rounds})
        print(f" P(blackout)={m['p_blackout']:.3e}  gap={m['gap']:.1e}  "
              f"fail_rules={m['n_fail_rules']}  {elapsed:.1f}s")

    all_metrics_path = out / "all_metrics.json"
    with open(all_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved: {all_metrics_path}  (per-round metrics of all {runs} runs)")

    summarise(run_metrics, out)


if __name__ == "__main__":
    app()
