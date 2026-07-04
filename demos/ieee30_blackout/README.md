# IEEE 30-bus DC-OPF blackout reliability (RSR demo)

Estimate the probability of a system-wide **blackout** on the IEEE 30-bus power
system and identify the **critical failure modes** (minimal cut-sets) that drive
it — using RSR's branch-and-bound rule extraction.

This is the multi-state DC-OPF blackout model of **Chan et al. (2024)** (a.k.a.
Byun, Ryu & Straub, *Branch-and-bound algorithm for efficient reliability
analysis of general coherent systems*, arXiv:2410.22363), Scenario 1. It is the
larger sibling of the `ieee14_blackout` demo — same model, 71 components instead
of 34 — so RSR needs more rounds and more rules to converge.

## The model

71 failable components:

| Component type | Count | States |
|---|---|---|
| Generator buses (`vbus1,2,13,22,23,27`) | 6 | 4-state: 0 = removed, 1 = 40 %, 2 = 80 %, 3 = full capacity |
| Ordinary buses (`vbus…`) | 24 | 2-state: 0 = failed, 1 = operational |
| Branches (`br1…br41`) | 41 | 2-state: 0 = failed, 1 = operational |

Higher state index = healthier, so the system function is **coherent** (monotone).
The system **fails** when the DC-OPF blackout size exceeds **40.2 %** of demand,
otherwise it **survives**. The DC-OPF is solved in pure Python (scipy `linprog`,
no MATLAB dependency) by the `ieee30` dataset's own system function.

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

## Data source

Reads the `ieee30` network dataset (edges, per-component probabilities, MATPOWER
case) at:

```
~/Projects/network-datasets/datasets/ieee30/v1/
    data/{edges,probs}.json, data/ieee30.m
    scripts/sfun_dcopt.py, func_dcopt_py.py   # the DC-OPF system function
```

Point elsewhere with `--dataset /path/to/ieee30/v1`.

## Run

```bash
# from the repo root (…/rsr)
python demos/ieee30_blackout/run_demo.py

# relative stopping rule (gap < 50% of P(blackout)), force CPU, or multi-GPU
python demos/ieee30_blackout/run_demo.py --unk-opt rel --unk-thres 0.5 --n-sample 2000000
python demos/ieee30_blackout/run_demo.py --device cpu
python demos/ieee30_blackout/run_demo.py --devices cuda:0,cuda:1
```

Needs `torch`, `numpy`, `scipy`, `typer`. Uses CUDA automatically when available.
With 71 components this network is markedly harder than the 14-bus case. The
default `--unk-thres 1e-6` (absolute gap) may run for a very long time, so prefer
a **relative** stopping rule — `--unk-opt rel --unk-thres 0.5` stops once the
bound gap falls below 50 % of P(blackout), giving a bracketed estimate in a
practical number of rounds. Tighten `--unk-thres` toward 0 for a narrower bound.

## Running on HPC / NCI Gadi (multi-GPU + multi-CPU)

The demo exposes two independent parallelism knobs:

| Flag | Parallelises | Value |
|---|---|---|
| `--devices` | GPU sampling / classification | `cuda:0,cuda:1,…` or **`auto`** (all visible GPUs) |
| `--n-workers` | CPU sfun evaluation + rule minimisation | integer, or **`-1`** (all available CPUs) |

`auto` reads `CUDA_VISIBLE_DEVICES` and `-1` reads `PBS_NCPUS` (falling back to
the process's CPU affinity), so on a scheduler the job uses exactly what it was
allocated — no hand-editing device lists. The GPUs do the sampling while the CPU
workers evaluate the DC-OPF system function in parallel; they compose, so a
GPU node is used end-to-end. A resource line is printed at startup:

```
  Resources:   GPUs=cuda:0,cuda:1,cuda:2,cuda:3  CPU workers=48  (visible: 4 GPU, 48 CPU)
Parallel mode: 48 CPU workers for sfun + minimization
```

```bash
# on a GPU node, use everything allocated to the job
python demos/ieee30_blackout/run_demo.py --devices auto --n-workers -1 \
    --unk-opt rel --unk-thres 0.2 --n-sample 4000000
```

A **PBS job script for NCI Gadi** is included: `nci_gadi.pbs`, following the
conventions of `run_rsr.sh` in this folder (project `n74`, `gpuvolta`, 2× V100 +
24 CPU, `python3/3.12.1` + `cuda` modules, venv on `/g/data`). Check the venv
path, project and storage mounts, then `qsub nci_gadi.pbs`. Notes:

- Gadi's `gpuvolta` gives 12 CPU + ~95.5 GB per GPU, so `ngpus=2 → ncpus=24 /
  mem=191GB` (half node); a full node is `ngpus=4 / ncpus=48 / mem=382GB`.
- With `-l other=hyperthread` you can **oversubscribe** `--n-workers` past
  `ncpus` (the working `run_rsr.sh` used 96 on `ncpus=24`); the script uses 48 =
  one worker per logical thread. `--n-workers -1` would instead take
  `PBS_NCPUS` (24), which under-provisions when hyperthreading.
- The CPU workers do CPU-only work (scipy `linprog` + minimisation), so
  combining them with GPU sampling after CUDA init is safe.
- The venv must have **`typer`** (the older `tsum` venv may not) — `pip install
  typer` into it if the job errors on import.

## Repeat runs and summarise (`--runs`)

RSR is stochastic, so repeated runs give slightly different bounds and rule
counts. Pass `--runs N` to the same script: it builds the model once, runs the
extraction N times (each into `out/run_NN/`), reads the last line of every run's
`metrics.json`, and reports the run-to-run distribution of P(blackout), the
bound gap, rule counts, rounds and per-run runtime. Per-run RSR logs are hidden
unless you pass `--verbose`. Writes `out/summary.json` and `out/all_metrics.json`
(the full per-round metrics of every run, aggregated into one file).

```bash
python demos/ieee30_blackout/run_demo.py --runs 5 --unk-thres 1e-5 --n-sample 1000000
```

## Example output (single GPU, `--unk-opt rel --unk-thres 0.5 --n-sample 2000000`, 877 rounds)

```
  Components:  71 (6 generator buses, 24 ordinary buses, 41 branches)
  All operational: blackout=  0.00%  sys_st=1
  All failed:      blackout=100.00%  sys_st=0

  P(blackout):        6.500e-05   (bounds: [6.500e-05, 1.615e-04], gap 9.7e-05)
  P(survival):        0.9998
  Reference (paper):  p_f ~ 1.0e-04          ← inside the bound

  177 minimal cut-sets (failure modes).  Single points of failure: none
    - {vbus2≤0, vbus6≤0, vbus27≤0}
    - {vbus1≤0, vbus2≤0, vbus12≤0, vbus27≤1}
    - {vbus8≤0, vbus10≤0, vbus15≤0, br29≤0}
    ...
  Most critical components:
    vbus2    in 166 cut-sets (min size 3)     ← generator bus 2 drives most failure modes
    vbus27   in 122 cut-sets (min size 3)
    vbus22   in 127 cut-sets (min size 4)
```

The reference `p_f ≈ 1.0e-4` sits **inside** RSR's bracket `[6.5e-5, 1.6e-4]`.
The bound is looser than the 14-bus case: with 71 components there are far more
failure modes (177 cut-sets vs 7), so covering the failure region tightly needs
many more rounds — the point estimate is the guaranteed lower bound and shrinking
the gap is just a matter of budget. Beyond the number, RSR returns the **explicit
cut-sets** and shows generator buses **2 and 27** dominate the failure modes,
while the grid has **no single point of failure**.

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
