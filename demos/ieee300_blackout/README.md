# IEEE 300-bus DC-OPF blackout reliability (RSR demo)

Estimate the probability of a system-wide **blackout** on the IEEE 300-bus power
system and identify the **critical failure modes** (minimal cut-sets) that drive
it — using RSR's branch-and-bound rule extraction.

This is the multi-state DC-OPF blackout model of **Chan et al. (2024)** (a.k.a.
Byun, Ryu & Straub, *Branch-and-bound algorithm for efficient reliability
analysis of general coherent systems*, arXiv:2410.22363), Scenario 1. It is the
**largest** of the IEEE blackout demos (`ieee14/30/57/118/300`) — 711 components
— so it is firmly HPC-scale: run it on a GPU node with many CPU workers (see
`nci_gadi.pbs`), and expect the round/rule count, not convergence, to dominate.

## The model

711 failable components:

| Component type | Count | States |
|---|---|---|
| Generator buses (`vbus…`) | 69 | 4-state: 0 = removed, 1 = 40 %, 2 = 80 %, 3 = full capacity |
| Ordinary buses (`vbus…`) | 231 | 2-state: 0 = failed, 1 = operational |
| Branches (`br1…br411`) | 411 | 2-state: 0 = failed, 1 = operational |

Higher state index = healthier, so the system function is **coherent** (monotone).
The system **fails** when the DC-OPF blackout size exceeds **26.1 %** of demand,
otherwise it **survives**. The DC-OPF is solved in pure Python (scipy `linprog`,
no MATLAB dependency) by the `ieee300` dataset's own system function.

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

All IEEE blackout demos share one implementation:
[`demos/blackout_lib.py`](../blackout_lib.py). This folder's `run_demo.py` is a
thin wrapper that only sets the network-specific config (title, reference `p_f`,
dataset path, blackout threshold) and calls `build_app(...)`.

## Data source

Reads the `ieee300` network dataset (edges, per-component probabilities, MATPOWER
case) at:

```
~/Projects/network-datasets/datasets/ieee300/v1/
    data/{edges,probs}.json, data/ieee300.m
    scripts/sfun_dcopt.py, func_dcopt_py.py   # the DC-OPF system function
```

Point elsewhere with `--dataset /path/to/ieee300/v1`.

## Run

```bash
# from the repo root (…/rsr) — use a relative stopping rule and many CPU workers
python demos/ieee300_blackout/run_demo.py --unk-opt rel --unk-thres 0.5 \
    --n-sample 2000000 --n-workers -1

# multi-GPU + all CPUs on an HPC node
python demos/ieee300_blackout/run_demo.py --devices auto --n-workers -1 \
    --unk-opt rel --unk-thres 0.2 --n-sample 4000000
```

Needs `torch`, `numpy`, `scipy`, `typer`. Uses CUDA automatically when available.
With 711 components this is the hardest IEEE demo: the default `--unk-thres 1e-6`
(absolute gap) is impractical on a workstation. Prefer a **relative** stopping
rule (`--unk-opt rel`) and **many CPU workers** (`--n-workers -1`), and cap the
work explicitly with **`--max-refs`** (stop once that many rules are found) or
`--max-rounds`. Prefer a GPU node.

## Stopping controls

Extraction stops at **whichever triggers first**:

| Flag | Stops when |
|---|---|
| `--unk-thres` (+ `--unk-opt abs\|rel`) | the unknown bound-gap falls below the threshold (convergence) |
| `--max-rounds` | that many extraction rounds have run |
| `--max-refs` | that many rules (survival + failure) have been found; `0` = disabled |
| `--max-search-loops` | caps the sample batches per round (not a global stop) |

`--max-refs` is the practical budget for a network this size (it may overshoot by
up to `--n-workers`, since a round adds several rules at once).

## Running on HPC / NCI Gadi (multi-GPU + multi-CPU)

`--devices auto` uses every GPU the job was allocated (`CUDA_VISIBLE_DEVICES`);
`--n-workers -1` uses `PBS_NCPUS`. The GPUs do the sampling while the CPU workers
evaluate the DC-OPF system function in parallel — for these demos the run is
CPU-bound, so the CPU workers are what move the wall-clock.

A **PBS job script for NCI Gadi** is included: `nci_gadi.pbs` (project `n74`,
`gpuvolta`). Check the venv path, project and storage mounts, then
`qsub nci_gadi.pbs`. For 711 components a full node
(`ngpus=4 / ncpus=48 / mem=382GB`) and generous walltime are recommended; the
bundled script uses `--max-refs` to bound the run. The venv must have `typer`.

## Repeat runs and summarise (`--runs`)

Pass `--runs N` to run the extraction N times (each into `out/run_NN/`) and
report the run-to-run distribution of P(blackout), the bound gap, rule counts,
rounds and per-run runtime. Writes `out/summary.json` and `out/all_metrics.json`.
Per-run RSR logs are hidden unless you pass `--verbose`.

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
