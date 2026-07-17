"""
Shared library for the IEEE DC-OPF blackout RSR demos (ieee14/30/57/118).

Each `ieeeNN_blackout/run_demo.py` is a thin wrapper that supplies the
network-specific config (title, reference p_f, dataset path, blackout threshold)
and calls `build_app(...)` here. All the actual work — loading the dataset,
building the DC-OPF system function, running RSR rule extraction, extracting
cut-sets / criticality, reporting, and the multi-run summary — lives in this
module so it is written once and reused across every network.

The model (Chan et al. 2024, Scenario 1): components are bus / branch elements
that are healthy at high state indices and failed at low ones (coherent system
function); the system fails when the DC-OPF blackout size exceeds the network's
threshold. RSR extracts survival rules (sys >= 1) and failure rules / minimal
cut-sets (sys <= 0) that bound P(blackout) from below and above.
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

# repo root (…/rsr) so `import rsr.rsr` works when a demo is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsr.rsr as rsr  # noqa: E402


# ----------------------------------------------------------------------------
# Data / model
# ----------------------------------------------------------------------------
def load_dataset(dataset: Path):
    """Load the network probs and return the DC-OPF system function factory.

    The MATPOWER case file is discovered by glob, so this is network-agnostic.
    """
    data_dir = dataset / "data"
    scripts_dir = dataset / "scripts"
    if not (data_dir / "probs.json").exists():
        raise typer.BadParameter(f"Dataset not found under {dataset} (missing data/probs.json)")
    case_files = sorted(data_dir.glob("*.m"))
    if not case_files:
        raise typer.BadParameter(f"No MATPOWER .m case file under {data_dir}")
    # the dataset ships its own DC-OPF sfun; import it directly
    sys.path.insert(0, str(scripts_dir))
    from sfun_dcopt import make_dcopt_sfun  # noqa: E402

    probs_dict = json.load(open(data_dir / "probs.json"))
    return probs_dict, make_dcopt_sfun, str(case_files[0])


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

    model_info = {"case_path": case_path, "probs_dict": probs_dict, "dcopt": dcopt}
    return probs, row_names, n_state, sfun, model_info


def load_checkpoint(out: Path, device, row_names=None):
    """Load the references + cuts last checkpointed in `out` by a previous run
    (refs_up_1.json/.pt, refs_low_0.json/.pt, failure_cuts_0.pkl).

    Returns a dict with refs_upper/refs_lower (rule dicts), refs_mat_upper/
    refs_mat_lower (tensors on `device`, None when absent) and cuts (list).
    `row_names` enables rebuilding the rule dicts from the binary tensors
    when the JSON is missing or stale (intra-run checkpoints are
    binary-only; JSON is written once at the end of a run).
    """
    ck = {"refs_upper": [], "refs_lower": [],
          "refs_mat_upper": None, "refs_mat_lower": None, "cuts": []}
    cuts_pkl = out / "failure_cuts_0.pkl"
    if cuts_pkl.exists():
        import pickle
        with open(cuts_pkl, "rb") as f:
            ck["cuts"] = pickle.load(f)

    def _load_dicts(p):  # json stores (op, state) as [op, state]; restore tuples
        with open(p) as f:
            return [{k: (tuple(v) if isinstance(v, list) else v)
                     for k, v in d.items()} for d in json.load(f)]

    def _load_side(json_path, pt_path, op, sys_st):
        """(dicts, mat). Intra-run checkpoints only write the binary .pt
        (JSON lands once at the end of a run), so when the JSON is missing
        or stale — e.g. resuming a walltime-killed run — the dicts are
        rebuilt from the matrix."""
        if not pt_path.exists():
            return [], None
        mat = torch.load(pt_path, weights_only=True).to(device)
        dicts = _load_dicts(json_path) if json_path.exists() else []
        if len(dicts) != len(mat):
            if row_names is None:
                raise ValueError(
                    f"{pt_path.name} has {len(mat)} rules but the JSON has "
                    f"{len(dicts)}; pass row_names to load_checkpoint so the "
                    "rule dicts can be rebuilt from the binary tensor")
            dicts = rsr.refs_dicts_from_mat(mat, row_names, op, sys_st)
            print(f"  checkpoint {pt_path.name}: rebuilt {len(dicts)} rule "
                  f"dicts from the binary tensor (JSON absent or stale)")
        return dicts, mat

    up_json, up_pt = out / "refs_up_1.json", out / "refs_up_1.pt"
    low_json, low_pt = out / "refs_low_0.json", out / "refs_low_0.pt"
    if up_pt.exists():
        ck["refs_upper"], ck["refs_mat_upper"] = _load_side(up_json, up_pt, '>=', 1)
        ck["refs_lower"], ck["refs_mat_lower"] = _load_side(low_json, low_pt, '<=', 0)
    return ck


def extract(sfun, probs, row_names, n_state, out: Path, *, unk_thres, unk_opt,
            max_search_loops, max_rounds, max_refs, n_sample, batch, multi_devices, n_workers, quiet,
            coverage_aware=False, ca_n_seeds=8, ca_n_orders=4, ca_max_add=4, ca_failure_beta=0.0,
            ref_generator=None, cut_generator=None, initial_cuts=None,
            resume=False):
    """Run one RSR rule extraction into `out`. Returns (res, wall_seconds).

    `multi_devices` (list of GPUs) parallelises the sampling/classification;
    `n_workers` (CPU processes) parallelises the sfun evaluation + minimisation.
    The two are independent and can be combined on an HPC GPU node.

    `resume=True` warm-starts from the references last saved in `out`
    (refs_up_1.pt/.json + refs_low_0.pt/.json, written every `save_every`
    rounds), continuing where a killed run left off and appending to
    metrics.json. Up to `save_every` rounds since the last checkpoint are lost.
    """
    out.mkdir(parents=True, exist_ok=True)

    refs_upper, refs_lower = [], []
    refs_mat_upper, refs_mat_lower = None, None
    lower_cuts = list(initial_cuts) if initial_cuts else None
    if resume:
        ck = load_checkpoint(out, probs.device, row_names)
        if ck["cuts"]:
            lower_cuts = ck["cuts"] + (lower_cuts or [])
            print(f"Resuming with {len(ck['cuts'])} failure cuts from {out / 'failure_cuts_0.pkl'}")
        if ck["refs_mat_upper"] is not None:
            refs_upper, refs_mat_upper = ck["refs_upper"], ck["refs_mat_upper"]
            refs_lower, refs_mat_lower = ck["refs_lower"], ck["refs_mat_lower"]
            print(f"Resuming from checkpoint in {out}: "
                  f"{len(refs_upper)} survival + {len(refs_lower)} failure references")
        else:
            print(f"--resume set but no checkpoint found in {out}; starting fresh")
    else:
        # fresh run: metrics.json is appended to by RSR; drop any stale copy
        stale = out / "metrics.json"
        if stale.exists():
            stale.unlink()

    def _go():
        return rsr.run_ref_extraction_by_mcs(
            sfun=sfun, probs=probs, row_names=row_names, n_state=n_state,
            sys_upper_st=1,               # system states: 0 = blackout, 1 = survive
            refs_upper=refs_upper, refs_lower=refs_lower,
            refs_mat_upper=refs_mat_upper, refs_mat_lower=refs_mat_lower,
            unk_prob_thres=unk_thres, unk_prob_opt=unk_opt,
            n_sample=n_sample, max_search_loops=max_search_loops, max_rounds=max_rounds,
            max_refs=max_refs, sample_batch_size=batch,
            devices=multi_devices, n_workers=n_workers, output_dir=str(out),
            coverage_aware=coverage_aware,
            ca_n_seeds=ca_n_seeds, ca_n_orders=ca_n_orders, ca_max_add=ca_max_add,
            ca_failure_beta=ca_failure_beta,
            ref_generator=ref_generator, cut_generator=cut_generator,
            lower_cuts=lower_cuts,
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


def analyse(res, threshold, elapsed, ref_pf):
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
        "reference_p_failure": ref_pf,
        "blackout_threshold_pct": threshold,
        "runtime_sec": elapsed,
        "n_survival_rules": len(rules_surv),
        "n_failure_rules": len(rules_fail),
        "n_failure_cuts": len(res.get("lower_cuts", [])),
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


def print_report(res, reliability, crit, cutsets, out: Path, ref_pf):
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
    print(f"  Reference (paper):  p_f ~ {ref_pf:.1e}")
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


def hybrid_estimate(sfun, probs, row_names, n_state, out: Path, *,
                    n_sample, batch, multi_devices, n_workers, ref_pf, seed=None):
    """Hybrid MC phase: the checkpointed rules classify samples for free,
    sfun evaluates only the residual unknowns -> unbiased P(blackout) + 95% CI.

    Complements the rigorous-but-wide bounds from rule extraction; writes
    hybrid.json next to reliability.json in `out`.
    """
    ck = load_checkpoint(out, probs.device, row_names)
    if ck["refs_mat_upper"] is None and not ck["cuts"]:
        raise typer.BadParameter(
            f"--hybrid needs a rule checkpoint in {out} (run an extraction first)")
    n_up = 0 if ck["refs_mat_upper"] is None else len(ck["refs_mat_upper"])
    n_low = 0 if ck["refs_mat_lower"] is None else len(ck["refs_mat_lower"])
    print(f"\nHybrid estimate: {n_sample:,} MC samples against "
          f"{n_up} survival + {n_low} failure rules + {len(ck['cuts'])} cuts ...\n",
          flush=True)
    res = rsr.run_hybrid_estimate(
        sfun=sfun, probs=probs, row_names=row_names, n_state=n_state,
        sys_upper_st=1,
        refs_mat_upper=ck["refs_mat_upper"], refs_mat_lower=ck["refs_mat_lower"],
        lower_cuts=ck["cuts"],
        n_sample=n_sample, sample_batch_size=batch,
        n_workers=n_workers, devices=multi_devices,
        seed=seed, output_dir=str(out))
    lo, hi = res["ci95"]
    print("\n" + "=" * 64)
    print("HYBRID ESTIMATE (rules + Monte Carlo on the residual)")
    print("=" * 64)
    print(f"  P(blackout):        {res['p_fail']:.3e}   "
          f"(95% CI [{lo:.3e}, {hi:.3e}], {res['n_fail']:,} failures)")
    print(f"  Reference (paper):  p_f ~ {ref_pf:.1e}")
    print(f"  Free classification: {res['free_frac'] * 100:.1f}% of samples "
          f"({res['n_sfun']:,} sfun calls for the rest)")
    print(f"  Failures:           {res['n_fail_certified']:,} rule-certified "
          f"+ {res['n_fail_sfun']:,} from sfun")
    print(f"  Runtime:            {res['elapsed_sec']:.1f}s "
          f"(sfun {res['t_sfun_sec']:.1f}s)")
    print(f"\n  Saved: {out/'hybrid.json'}")
    return res
    print(f"         {Path(res['refs_upper_path']).name} / "
          f"{Path(res['refs_lower_path']).name} (rule sets)")


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


def summarise(runs, out: Path, ref_pf):
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
          f"(reference p_f ~ {ref_pf:.1e}, ratio {mean_pf / ref_pf:.2f})")

    out_json = out / "summary.json"
    with open(out_json, "w") as f:
        json.dump({"runs": runs, "aggregate": summary,
                   "mean_p_blackout": mean_pf, "reference_p_failure": ref_pf},
                  f, indent=2)
    print(f"  Saved: {out_json}")


# ----------------------------------------------------------------------------
# HPC / NCI resource detection
# ----------------------------------------------------------------------------
def detect_gpus():
    """List of visible CUDA devices as ['cuda:0', 'cuda:1', ...].

    Honours CUDA_VISIBLE_DEVICES (which PBS sets from the -l ngpus request), so
    it returns exactly the GPUs allocated to the job.
    """
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def detect_cpus():
    """Number of CPUs available to this process.

    On NCI/PBS the job's core count is in PBS_NCPUS; otherwise fall back to the
    scheduling affinity (respects cpuset/cgroup limits) then os.cpu_count().
    """
    n = os.environ.get("PBS_NCPUS")
    if n and n.isdigit():
        return int(n)
    try:
        return len(os.sched_getaffinity(0))       # Linux: cores this job may use
    except AttributeError:
        return os.cpu_count() or 1


def resolve_devices(device: str, devices: str):
    """Turn --device / --devices (incl. the 'auto' sentinel) into (dev, multi_devices, list)."""
    if devices.strip().lower() == "auto":
        device_list = detect_gpus()
    else:
        device_list = [d.strip() for d in devices.split(",") if d.strip()]

    multi_devices = device_list if len(device_list) > 1 else None
    if device:
        dev = torch.device(device)
    elif device_list:
        dev = torch.device(device_list[0])
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev, multi_devices, device_list


def resolve_workers(n_workers: int):
    """--n-workers: -1/0 -> auto (all available CPUs), else the given count."""
    if n_workers <= 0:
        return max(detect_cpus(), 1)
    return n_workers


# ----------------------------------------------------------------------------
# Orchestration + CLI factory
# ----------------------------------------------------------------------------
def run(*, title, ref_pf, dataset, threshold, alpha, unk_thres, unk_opt,
        max_search_loops, max_rounds, max_refs, n_sample, batch, device, devices, n_workers, out, verbose, runs,
        coverage_aware=False, ca_n_seeds=8, ca_n_orders=4, ca_max_add=4, ca_failure_beta=0.0,
        cert=False, sus_cuts=0, sus_runs=1, sus_p0=0.1, resume=False,
        hybrid=0, hybrid_only=False, hybrid_seed=None):
    """Full demo run: single detailed report (runs<=1) or multi-run summary."""
    dev, multi_devices, _ = resolve_devices(device, devices)
    n_workers = resolve_workers(n_workers)

    print("=" * 64)
    print(f"RSR demo — {title}")
    print("=" * 64)
    gpu_str = ",".join(multi_devices) if multi_devices else str(dev)
    print(f"  Resources:   GPUs={gpu_str}  CPU workers={n_workers}  "
          f"(visible: {len(detect_gpus())} GPU, {detect_cpus()} CPU)")

    # Build the model once (shared across all repetitions).
    probs, row_names, n_state, sfun, model_info = build_model(dataset, dev, threshold, alpha)

    if hybrid_only:
        # Skip extraction: run the hybrid MC estimate on the rules last
        # checkpointed in `out` (e.g. by an earlier HPC extraction).
        if hybrid <= 0:
            raise typer.BadParameter("--hybrid-only requires --hybrid N")
        hybrid_estimate(sfun, probs, row_names, n_state, out,
                        n_sample=hybrid, batch=batch, multi_devices=multi_devices,
                        n_workers=n_workers, ref_pf=ref_pf, seed=hybrid_seed)
        return

    # LP-certificate generators: survival boxes + failure half-space cuts read
    # off single DC-OPF solves (see cert_lib.py); rsr verifies each box corner
    # with one sfun call and falls back to the minimiser when a generator
    # declines.
    ref_generator = cut_generator = None
    initial_cuts = None
    if cert:
        from cert_lib import CertModel, sus_failure_cuts, refined_cut_generator  # lazy: needs the dataset's scripts on sys.path
        cm = CertModel(model_info["case_path"], model_info["probs_dict"],
                       threshold, alpha)
        ref_generator = lambda seed: cm.extract(seed, variant="cert")  # noqa: E731
        cut_generator = refined_cut_generator(cm, model_info["dcopt"])
        print("  Certificates: LP survival boxes + dual failure cuts (cert_lib)")

        if sus_cuts > 0:
            # SuS -> cut pipeline: walk to the failure boundary with Subset
            # Simulation and convert every distinct failed state found into a
            # dual cut, so p_lower accumulates from round one.
            print(f"\nSuS -> cut pipeline: {sus_runs} run(s), "
                  f"{sus_cuts} samples/level, p0={sus_p0} ...", flush=True)
            initial_cuts, _sus_stats = sus_failure_cuts(
                cm, model_info["dcopt"], probs, row_names,
                sys_surv_st=1, n_per_level=sus_cuts, p0=sus_p0,
                n_runs=sus_runs, n_workers=n_workers, verbose=True)

    common = dict(unk_thres=unk_thres, unk_opt=unk_opt, max_search_loops=max_search_loops,
                  max_rounds=max_rounds, max_refs=max_refs,
                  n_sample=n_sample, batch=batch, multi_devices=multi_devices,
                  n_workers=n_workers, coverage_aware=coverage_aware,
                  ca_n_seeds=ca_n_seeds, ca_n_orders=ca_n_orders, ca_max_add=ca_max_add,
                  ca_failure_beta=ca_failure_beta,
                  ref_generator=ref_generator, cut_generator=cut_generator,
                  initial_cuts=initial_cuts,
                  resume=resume)

    if runs <= 1:
        # ---- single run: full report ----
        print(f"\nExtracting survival / failure rules "
              f"(unknown gap < {unk_thres:g} [{unk_opt}])...\n", flush=True)
        res, elapsed = extract(sfun, probs, row_names, n_state, out, quiet=False, **common)
        reliability, crit, cutsets = analyse(res, threshold, elapsed, ref_pf)
        save_artifacts(out, reliability, crit)
        print_report(res, reliability, crit, cutsets, out, ref_pf)
        if hybrid > 0:
            hybrid_estimate(sfun, probs, row_names, n_state, out,
                            n_sample=hybrid, batch=batch, multi_devices=multi_devices,
                            n_workers=n_workers, ref_pf=ref_pf, seed=hybrid_seed)
        return

    # ---- multi run: repeat and summarise metrics.json ----
    if hybrid > 0:
        print("  Note: --hybrid only applies to single runs (--runs 1); skipping it.")
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
        reliability, crit, _ = analyse(res, threshold, elapsed, ref_pf)
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

    summarise(run_metrics, out, ref_pf)


def build_app(*, title, ref_pf, default_dataset, default_out, default_threshold, help_doc=None):
    """Build a Typer app for one network. Each run_demo.py calls this and app()."""
    app = typer.Typer(add_completion=False, help=help_doc)

    @app.command()
    def main(
        runs: int = typer.Option(1, help="Repetitions; 1 = full report, >1 = summarise metrics.json"),
        dataset: Path = typer.Option(default_dataset, help="dataset version dir (data/ + scripts/)"),
        threshold: float = typer.Option(default_threshold, help="Blackout size (%) above which the system fails"),
        alpha: float = typer.Option(2.0, help="Branch capacity scaling factor for the DC-OPF case"),
        unk_thres: float = typer.Option(1e-6, help="Convergence threshold on the unknown (bound-gap) probability"),
        unk_opt: str = typer.Option("abs", help="Interpret --unk-thres as 'abs' or 'rel' to P(failure)"),
        max_search_loops: int = typer.Option(500_000, help="Max batches per round"),
        max_rounds: int = typer.Option(100_000, help="Max extraction rounds"),
        max_refs: int = typer.Option(0, help="Stop once #rules (survival+failure) reaches this; 0 = disabled"),
        n_sample: int = typer.Option(10_000_000, help="Samples per probability/search round"),
        batch: int = typer.Option(100_000, help="Sample batch size"),
        device: str = typer.Option("", help="Single torch device, e.g. 'cpu' or 'cuda' (default: auto)"),
        devices: str = typer.Option("", help="Multi-GPU sampling: 'cuda:0,cuda:1' or 'auto' (all visible GPUs)"),
        n_workers: int = typer.Option(1, help="CPU worker processes for sfun + minimisation; -1 = all available CPUs"),
        out: Path = typer.Option(default_out, help="Output dir (single run) / base dir (multi run)"),
        verbose: bool = typer.Option(False, help="Show RSR's per-round log during multi runs"),
        coverage_aware: bool = typer.Option(False, "--coverage-aware", help="Coverage-aware Stage-1 boundary search (Reviewer 3, Comments 3.1/3.2)"),
        ca_n_seeds: int = typer.Option(8, help="Coverage-aware: unclassified seeds examined per round"),
        ca_n_orders: int = typer.Option(4, help="Coverage-aware: coordinate orders tried per seed (set 1 to isolate greedy multi-seed selection at ~baseline cost)"),
        ca_max_add: int = typer.Option(4, help="Coverage-aware: references committed per round (greedy)"),
        ca_failure_beta: float = typer.Option(0.0, help="Coverage-aware: >0 biases seeds+coverage toward the failure boundary (try 2-5); 0 = uniform/mass-optimal"),
        cert: bool = typer.Option(False, "--cert", help="LP-certificate references: survival boxes + dual failure cuts from single DC-OPF solves (cert_lib.py)"),
        sus_cuts: int = typer.Option(0, help="With --cert: Subset-Simulation samples per level for the SuS->cut failure pipeline (0 = off; try 1000)"),
        sus_runs: int = typer.Option(1, help="Independent SuS repetitions (diversifies the failure modes found)"),
        sus_p0: float = typer.Option(0.1, help="SuS intermediate conditional probability"),
        resume: bool = typer.Option(False, "--resume", help="Warm-start from references last checkpointed in --out (continue a run killed by walltime)"),
        hybrid: int = typer.Option(0, "--hybrid", help="After extraction: unbiased MC estimate of P(blackout) with this many samples — the rules classify most samples for free, sfun evaluates only the residual unknowns; reports a 95% CI (0 = off)"),
        hybrid_only: bool = typer.Option(False, "--hybrid-only", help="Skip extraction; run the --hybrid estimate directly on the rules last checkpointed in --out"),
        hybrid_seed: int = typer.Option(-1, help="RNG seed for the hybrid estimate (-1 = don't seed)"),
    ):
        """Estimate P(blackout) and the critical failure modes for this grid."""
        run(title=title, ref_pf=ref_pf, dataset=dataset, threshold=threshold, alpha=alpha,
            unk_thres=unk_thres, unk_opt=unk_opt, max_search_loops=max_search_loops,
            max_rounds=max_rounds, max_refs=max_refs, n_sample=n_sample,
            batch=batch, device=device, devices=devices, n_workers=n_workers, out=out,
            verbose=verbose, runs=runs, coverage_aware=coverage_aware,
            ca_n_seeds=ca_n_seeds, ca_n_orders=ca_n_orders, ca_max_add=ca_max_add,
            ca_failure_beta=ca_failure_beta, cert=cert,
            sus_cuts=sus_cuts, sus_runs=sus_runs, sus_p0=sus_p0, resume=resume,
            hybrid=hybrid, hybrid_only=hybrid_only,
            hybrid_seed=None if hybrid_seed < 0 else hybrid_seed)

    return app
