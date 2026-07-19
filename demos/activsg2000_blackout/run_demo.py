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
Threshold calibrated with calibrate_threshold.py (SuS tail-CDF): in THIS
model p_f ~ 1e-4 (the other demos' Scenario-1 level) is at ~4.7% blackout,
1e-2 at ~3%, 1e-5 at ~4.9%. (Chan et al. 2024 report p_f ~ 1e-4 nearer 6%,
so our DC-OPF differs from theirs — likely the alpha=2 branch-capacity
scaling or load handling; recalibrate if alpha changes.) The CDF is steep
around 4.5-5%, so p_f is sensitive to the threshold there; use ~3% (p_f ~
1e-2) for quicker, more robust pipeline tests.

Each DC-OPF solve here is ~0.9 s and rules cover ~0% of this grid, so the
certificate cuts and the plain-MC hybrid both struggle (see the demo run
notes). Chan et al.'s aE-SuS reaches p_f ~ 1e-4 in ~9,200 solves — Subset
Simulation as an ESTIMATOR is the tractable route at this scale.

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
    ref_pf=1.0e-4,                            # p_f ~ 1e-4 at ~4.7% in this model
    default_dataset=HERE.joinpath("../../../network-datasets/datasets/ACTIVSg2000/v1"),
    default_out=HERE / "out",
    default_threshold=4.7,                     # calibrate_threshold.py; see docstring
    help_doc=__doc__,
)

if __name__ == "__main__":
    app()
