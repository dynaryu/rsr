# IEEE 14-bus DC-OPF blackout reliability (RSR demo)

Estimate the probability of a system-wide **blackout** on the IEEE 14-bus power
system and identify the **critical failure modes** (minimal cut-sets) that drive
it — using RSR's branch-and-bound rule extraction.

This is the multi-state DC-OPF blackout model of **Chan et al. (2024)** (a.k.a.
Byun, Ryu & Straub, *Branch-and-bound algorithm for efficient reliability
analysis of general coherent systems*, arXiv:2410.22363), Scenario 1.

## The model

34 failable components:

| Component type | Count | States |
|---|---|---|
| Generator buses (`vbus1,2,3,6,8`) | 5 | 4-state: 0 = removed, 1 = 40 %, 2 = 80 %, 3 = full capacity |
| Ordinary buses (`vbus…`) | 9 | 2-state: 0 = failed, 1 = operational |
| Branches (`br1…br20`) | 20 | 2-state: 0 = failed, 1 = operational |

Higher state index = healthier, so the system function is **coherent** (monotone).
The system **fails** when the DC-OPF blackout size exceeds **54.8 %** of demand,
otherwise it **survives**. The DC-OPF is solved in pure Python (scipy `linprog`,
no MATLAB dependency) by the `ieee14` dataset's own system function.

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

Reads the `ieee14` network dataset (edges, per-component probabilities, MATPOWER
case) at:

```
~/Projects/network-datasets/datasets/ieee14/v1/
    data/{edges,probs}.json, data/ieee14.m
    scripts/sfun_dcopt.py, func_dcopt_py.py   # the DC-OPF system function
```

Point elsewhere with `--dataset /path/to/ieee14/v1`.

## Run

```bash
# from the repo root (…/rsr)
python demos/ieee14_blackout/run_demo.py

# tighter bound / more samples, or force CPU, or multi-GPU
python demos/ieee14_blackout/run_demo.py --unk-thres 1e-6 --n-sample 2000000
python demos/ieee14_blackout/run_demo.py --device cpu
python demos/ieee14_blackout/run_demo.py --devices cuda:0,cuda:1
```

Needs `torch`, `numpy`, `scipy`. Uses CUDA automatically when available.

## Example output (single GPU, ~3 s)

```
  Components:  34 (5 generator buses, 9 ordinary buses, 20 branches)
  All operational: blackout=  0.00%  sys_st=1
  All failed:      blackout=100.00%  sys_st=0

  P(blackout):        1.090e-04   (bounds: [1.090e-04, 1.380e-04], gap 2.9e-05)
  P(survival):        0.9999
  Reference (paper):  p_f ~ 1.1e-04          ← reproduced

  7 minimal cut-sets (failure modes).  Single points of failure: none
    - {vbus3≤0, vbus4≤0}
    - {vbus3≤0, vbus6≤0, vbus9≤0}
    - {vbus2≤0, vbus3≤1, br7≤0, br14≤0}
    ...
  Most critical components:
    vbus3    in  7 cut-sets (min size 2)      ← generator bus 3 drives every failure mode
    vbus6    in  4 cut-sets (min size 3)
    vbus2    in  4 cut-sets (min size 3)
```

RSR reproduces the paper's `p_f ≈ 1.1e-4` and, beyond the point estimate, returns
**bounds** and the **explicit cut-sets** — showing generator bus 3 (`vbus3`) is
the dominant contributor (it appears in every failure mode) while the grid has
no single point of failure.

## Outputs (`out/`)

| File | Contents |
|---|---|
| `reliability.json` | P(blackout), P(survival), bounds, rule/cut-set counts, single points of failure, smallest cut-sets |
| `critical_components.csv` | per-component criticality ranking |
| `rules_geq_1.json` | survival rules |
| `rules_leq_0.json` | failure rules = minimal cut-sets |
| `metrics.json` | per-round convergence log |

The two rule sets are **probability-independent** properties of the topology and
threshold: extract them once, then re-score P(blackout) for any new set of
component probabilities (different hazard level) by resampling against the saved
rules — no re-extraction needed.
