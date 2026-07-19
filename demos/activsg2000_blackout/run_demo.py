"""
RSR demo: ACTIVSg2000 DC-OPF blackout reliability.

Synthetic 2000-bus Texas grid (Birchfield et al., ACTIVSg2000) under the
multi-state DC-OPF blackout model of Chan et al. (2024), Scenario 1 — the
same model as the ieee14/118/300 demos, at ~7x the size:

  5206 failable components:
    - 485 generator buses  (4-state: 0 removed, 1 40%, 2 80%, 3 full)
    - 1515 ordinary buses  (2-state: 0 failed, 1 operational)
    - 3206 branches        (2-state)

The system FAILS when the DC-OPF blackout size exceeds the threshold below.
Chan et al., "Adaptive Monte Carlo methods for estimating rare events in
power grids" (2024) — the benchmark's own authors — compute the full
blackout CDF for this grid: the rare-event regime is ~2-6% blackout, with
the p_f ~ 1e-4 level (matching the other demos' Scenario 1) at ~6%. We use
6% as the default; raise it for deeper tails, lower it (2-3%, p_f ~ 1e-2 to
1e-3) for quicker pipeline tests. Their aE-SuS reaches p_f ~ 1e-4 in ~9,200
system-function calls, so estimate the residual with Subset Simulation, not
plain MC (each DC-OPF solve here is ~0.9 s).

The dataset (aggregated case + probs.json) is built by
network-datasets/datasets/ACTIVSg2000/v1/scripts/build_dataset.py.

At 5206 components each DC-OPF solve is ~0.9 s (vs ~15 ms on IEEE-118), and
the one-hot sample tensor is ~7x larger, so on an 8 GB GPU the default
100k batch OOMs — pass a smaller --batch (10000 fits 8 GB; scale up on
bigger cards) or --device cpu.

Usage:
    python run_demo.py --batch 10000                     # single run, full report
    python run_demo.py --cert --max-refs 200 --batch 10000 --n-workers -1   # cert smoke test
    python run_demo.py --cert --sus-cuts 1000 --hybrid 1000000 --devices auto --batch 20000
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # demos/ dir, for blackout_lib
from blackout_lib import build_app            # noqa: E402

app = build_app(
    title="ACTIVSg2000 DC-OPF blackout reliability",
    ref_pf=1.0e-4,                            # Chan et al. 2024, ~6% blackout level
    default_dataset=HERE.joinpath("../../../network-datasets/datasets/ACTIVSg2000/v1"),
    default_out=HERE / "out",
    default_threshold=6.0,                     # Chan et al. 2024; see module docstring
    help_doc=__doc__,
)

if __name__ == "__main__":
    app()
