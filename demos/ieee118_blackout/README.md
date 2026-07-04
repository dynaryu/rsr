# IEEE 118-bus DC-OPF blackout reliability (RSR demo)

Estimate the probability of a system-wide **blackout** on the IEEE 118-bus power
system and identify the **critical failure modes** (minimal cut-sets) that drive
it — using RSR's branch-and-bound rule extraction.

This is the multi-state DC-OPF blackout model of **Chan et al. (2024)** (a.k.a.
Byun, Ryu & Straub, *Branch-and-bound algorithm for efficient reliability
analysis of general coherent systems*, arXiv:2410.22363), Scenario 1. It is the
**largest** of the IEEE blackout demos (`ieee14/30/57/118`) — 304 components — so
it is genuinely HPC-scale: expect many rounds and rules, and run it on a GPU node
(see `nci_gadi.pbs`) for a tight bound.

## The model

304 failable components:

| Component type | Count | States |
|---|---|---|
| Generator buses (`vbus1,4,6,8,…`) | 54 | 4-state: 0 = removed, 1 = 40 %, 2 = 80 %, 3 = full capacity |
| Ordinary buses (`vbus…`) | 64 | 2-state: 0 = failed, 1 = operational |
| Branches (`br1…br186`) | 186 | 2-state: 0 = failed, 1 = operational |

Higher state index = healthier, so the system function is **coherent** (monotone).
The system **fails** when the DC-OPF blackout size exceeds **13.8 %** of demand,
otherwise it **survives**. The DC-OPF is solved in pure Python (scipy `linprog`,
no MATLAB dependency) by the `ieee118` dataset's own system function.

## What RSR computes

RSR draws Monte Carlo samples of component states from their true
probabilities, evaluates the DC-OPF system function on any sample not yet
covered by a rule, and minimises that outcome into two reusable rule sets:

- **survival rules** (`sys ≥ 1`) — minimal healthy states that guarantee no blackout;
- **failure rules** (`sys ≤ 0`) — the **minimal cut-sets**: smallest joint degradations that force a blackout.

Together they **bound** P(blackout) from below and above; the gap is the
*unknown* probability — the sampled mass covered by neither rule set — which
shrinks as rules accumulate. The rare-event efficiency comes from this
rule-covering (branch-and-bound), not from a biased sampler: only samples in
the shrinking unknown region trigger an expensive system-function evaluation.
The failure rules are the cut-sets, and the per-component **criticality**
ranking is read off from them (single points of failure first, then frequency
across cut-sets).

## Shared code

All four IEEE blackout demos share one implementation:
[`demos/blackout_lib.py`](../blackout_lib.py). This folder's `run_demo.py` is a
thin wrapper that only sets the network-specific config (title, reference `p_f`,
dataset path, blackout threshold) and calls `build_app(...)`. To port the demo
to another DC-OPF blackout dataset, copy this folder and edit those four values.

## Data source

Reads the `ieee118` network dataset (edges, per-component probabilities, MATPOWER
case) at:

```
~/Projects/network-datasets/datasets/ieee118/v1/
    data/{edges,probs}.json, data/ieee118.m
    scripts/sfun_dcopt.py, func_dcopt_py.py   # the DC-OPF system function
```

Point elsewhere with `--dataset /path/to/ieee118/v1`.

## Run

```bash
# from the repo root (…/rsr)
python demos/ieee118_blackout/run_demo.py --unk-opt rel --unk-thres 0.5 \
    --n-sample 2000000 --n-workers -1

# multi-GPU + all CPUs on an HPC node
python demos/ieee118_blackout/run_demo.py --devices auto --n-workers -1 \
    --unk-opt rel --unk-thres 0.2 --n-sample 4000000
```

Needs `torch`, `numpy`, `scipy`, `typer`. Uses CUDA automatically when available.
With 304 components this is by far the hardest IEEE demo: the default
`--unk-thres 1e-6` (absolute gap) is impractical on a workstation — use a
**relative** stopping rule (`--unk-opt rel`) and **many CPU workers**
(`--n-workers -1`) to parallelise the minimisations, and prefer a GPU node.
Tighten `--unk-thres` toward 0 for a narrower bound.

## Running on HPC / NCI Gadi (multi-GPU + multi-CPU)

The demo exposes two independent parallelism knobs:

| Flag | Parallelises | Value |
|---|---|---|
| `--devices` | GPU sampling / classification | `cuda:0,cuda:1,…` or **`auto`** (all visible GPUs) |
| `--n-workers` | CPU sfun evaluation + rule minimisation | integer, or **`-1`** (all available CPUs) |

`auto` reads `CUDA_VISIBLE_DEVICES` and `-1` reads `PBS_NCPUS` (falling back to
the process's CPU affinity), so on a scheduler the job uses exactly what it was
allocated. The GPUs do the sampling while the CPU workers evaluate the DC-OPF
system function in parallel.

A **PBS job script for NCI Gadi** is included: `nci_gadi.pbs` (project `n74`,
`gpuvolta`, 2× V100 + 24 CPU). Check the venv path, project and storage mounts,
then `qsub nci_gadi.pbs`. For a 304-component network a full node
(`ngpus=4 / ncpus=48 / mem=382GB`) and generous walltime are recommended.
The CPU workers do CPU-only work (scipy `linprog` + minimisation), so combining
them with GPU sampling after CUDA init is safe. The venv must have `typer`.

## Repeat runs and summarise (`--runs`)

Pass `--runs N` to run the extraction N times (each into `out/run_NN/`) and
report the run-to-run distribution of P(blackout), the bound gap, rule counts,
rounds and per-run runtime. Writes `out/summary.json` and `out/all_metrics.json`
(the full per-round metrics of every run, aggregated into one file). Per-run RSR
logs are hidden unless you pass `--verbose`.

```bash
python demos/ieee118_blackout/run_demo.py --runs 10 --unk-opt rel --unk-thres 0.5 \
    --n-sample 2000000 --n-workers -1
```

## Example output

```
  Components:  304 (54 generator buses, 64 ordinary buses, 186 branches)
  All operational: blackout=  0.00%  sys_st=1
  All failed:      blackout=100.00%  sys_st=0

  P(blackout):        <HPC-scale — fill from a converged run on a GPU node>
  Reference (paper):  p_f ~ 1.0e-04
```

RSR brackets the paper's `p_f ≈ 1.0e-4`. As the largest network here, failure
samples are rare and the many minimisations are expensive, so a tight bound
needs a GPU node with many CPU workers; on a workstation expect a wide bracket
that narrows as rounds accumulate. Beyond the number, RSR returns the explicit
cut-sets and the generator buses that dominate the failure modes.

## Outputs (`out/`)

| File | Contents |
|---|---|
| `reliability.json` | P(blackout), P(survival), bounds, rule/cut-set counts, single points of failure, smallest cut-sets |
| `critical_components.csv` | per-component criticality ranking |
| `refs_up_1.json` | survival rules |
| `refs_low_0.json` | failure rules = minimal cut-sets |
| `metrics.json` | per-round convergence log |
| `summary.json` / `all_metrics.json` | multi-run only: aggregate stats / all runs' per-round metrics |

The two rule sets are **probability-independent** properties of the topology and
threshold: extract them once, then re-score P(blackout) for any new set of
component probabilities (different hazard level) by resampling against the saved
rules — no re-extraction needed.
