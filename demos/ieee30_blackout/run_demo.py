"""
RSR demo: IEEE 30-bus DC-OPF blackout reliability.

Estimates the probability of a system-wide blackout for the IEEE 30-bus power
system and identifies the critical failure modes (minimal cut-sets), using RSR's
branch-and-bound rule extraction. Multi-state DC-OPF blackout case from Chan
et al. (2024), Scenario 1:
  - 71 failable components: 6 generator buses (4 states), 24 ordinary buses
    (2 states) and 41 branches (2 states);
  - the system FAILS when the DC-OPF blackout size exceeds 40.2% of demand.

All logic lives in demos/blackout_lib.py; this file only supplies the
network-specific config. See the module docstring there and the README here.

Usage:
    python run_demo.py                                   # single run, full report
    python run_demo.py --unk-opt rel --unk-thres 0.5 --n-sample 2000000
    python run_demo.py --devices auto --n-workers -1     # HPC GPU node
    python run_demo.py --runs 10                          # run-to-run summary
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # demos/ dir, for blackout_lib
from blackout_lib import build_app            # noqa: E402

app = build_app(
    title="IEEE 30-bus DC-OPF blackout reliability",
    ref_pf=1.0e-4,                            # Chan et al. 2024, Table 2, Scenario 1
    default_dataset=HERE.joinpath("../../../network-datasets/datasets/ieee30/v1"),
    default_out=HERE / "out",
    default_threshold=40.2,
    help_doc=__doc__,
)

if __name__ == "__main__":
    app()
