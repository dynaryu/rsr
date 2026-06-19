# Proposal: RSR as a system-reliability SystemPerformance application for R2D

**To:** NHERI SimCenter (R2D / SimCenterBackendApplications maintainers)
**Re:** Adding network system-reliability and critical-failure-mode analysis to R2D
**Status:** Working proof of concept (Tier-1 post-processor + Tier-2 workflow app), verified end-to-end on synthetic and R2D-format data.

---

## 1. Summary

R2D's `SystemPerformance` stage has applications for transportation (ResidualDemand) and water
(REWET), but **none for power**, and none of the existing tools return **system-reliability with
explicit failure modes**. We propose adding **RSR** (Tensor-based System Uncertainty Methods, an
open-source system-reliability engine) as a `SystemPerformance` application. Given the per-asset
damage probabilities R2D already produces and a network topology, RSR returns:

- the probability the network fails (e.g. source -> load disconnection), **with rigorous bounds**, and
- the **minimal cut-sets** -- which components, in which states, drive system failure, i.e. the
  single points of failure and critical combinations.

The cut-set output is the distinctive value: it turns a damage assessment into a **mitigation-
prioritisation** result, which the simulation-based tools do not provide.

## 2. The gap (from R2D's own files)

- In `R2DExamples/E16ElectricPowerEarthquake/input.json`, the `SystemPerformance` block is
  `{"PowerNetwork": {"Application": "None"}}` -- the stage is wired but **empty for power**.
- E16's power inventory (`input_powerdata.geojson`) is **point/polygon damage assets with no
  connectivity** -- so power stops at asset damage, with no system-level consequence.
- The existing SystemPerformance apps are simulation/functionality based (ResidualDemand:
  traffic residual demand; REWET: restoration simulation). Neither computes a system-failure
  probability with bounds, nor identifies critical cut-sets.

## 3. What RSR is, and what it uniquely adds

RSR consumes **component state probabilities + a system function** and computes (or bounds) the
probability of each system state, extracting the dominant cut-sets/path-sets by branch-and-bound
in PyTorch tensors -- without brute-forcing the combinatorial state space. It ships igraph-based
**network system functions** (origin-destination connectivity, global k-connectivity, max-flow,
DC-OPF blackout) and a subset-simulation fallback for rare events. Method: Byun, Ryu & Straub
(2024), *Branch-and-bound algorithm for efficient reliability analysis of general coherent
systems*.

Relative to R2D's existing tools it adds:

1. **Explicit critical failure modes (cut-sets)** -- the single points of failure and critical
   component combinations; directly actionable for mitigation.
2. **Rigorous bounds** on the system-failure probability (not just a point estimate).
3. **Rare-event efficiency** -- where Monte Carlo needs exponentially many samples.
4. **Hazard-agnostic** -- consumes damage probabilities, so it works for any R2D hazard.

## 4. Where it fits, and the data handshake

It runs at the `SystemPerformance` stage, after DL (pelicun):

```
Hazard -> Asset -> ... -> DL (pelicun) -> [ SystemPerformance: RSR ] -> results
                          Results_<rlz>.json   P(disconnect) + bounds
                          + node/edge geojson  + minimal cut-sets (criticality map layer)
```

| RSR input/output | R2D artifact | adapter |
|---|---|---|
| component fail probabilities | `Results_<rlz>.json`, field `R2Dres_MostLikelyCriticalDamageState`; P(fail) = fraction of realizations with DS >= threshold | `load_damage_probs()` |
| system function | node geojson (`nodeID`) + edge geojson (`StartNode`/`EndNode`); substations node-split so each failable asset is an edge | `load_network()` -> igraph connectivity sfun |
| result | `R2D_results.geojson` criticality layer + reliability JSON/CSV | app writes them |

The damage field, `Results_<rlz>.json` structure, realization selection, app CLI, and output
file were all matched to the existing **ResidualDemand** backend
(`modules/systemPerformance/ResidualDemand/run_residual_demand.py`); node/edge fields to
R2DExamples **E14**.

## 5. Proof of concept (what is built and verified)

A self-contained package, `r2d_addon/`:

- **Tier-1** `r2d_performance.py` -- standalone post-processor reading R2D outputs. Verified
  on a synthetic redundant network and on an R2D-format fixture (`Results_<rlz>.json` + node/edge
  geojson); correctly returns P(disconnected) with bounds and identifies the series single-points-
  of-failure vs the redundant parallel paths.
- **Tier-2** `applications/systemPerformance/RSR2SP/run_rsr_performance.py` -- an rWHALE-style
  SystemPerformance app with the same CLI contract as ResidualDemand
  (`--nodeFile --edgeFile --configFile --r2dRunDir`), honouring the `DamageInput` realization
  selection (`SpecificRealization` / `SampleFromRealizations`) and writing `R2D_results.geojson`.
  Plus the registry entry (`WorkflowApplications_RSR_entry.json`) and the E16 `SystemPerformance`
  config block.
- **E16 runnable example** `applications/systemPerformance/RSR2SP/example_e16/` -- 35 substations
  taken from E16's real inventory with a synthesised partially-meshed topology; the app runs end-
  to-end and identifies 9 critical substations out of 35.

## 6. Scaling to regional networks

Exact rule extraction is intractable for very large networks. We have demonstrated this directly:
on a 373-component electric-power network (SIRA "EPN Yilgarn" model) whole-network extraction did
not finish, while a **hierarchical decomposition** -- per-asset reliability curve -> reduced
network -- produced the full-detail result in minutes. That hierarchical pattern maps directly
onto R2D (per-asset DL already exists; the reduced network is the SystemPerformance step). For
networks where even that is hard, RSR's **subset-simulation** path applies. So the recommended
production strategy is: exact for small/medium systems, hierarchical or subset-simulation for
regional scale.

## 7. Implementation status and remaining work

| Item | Status |
|---|---|
| Tier-1 adapters locked to R2D schema | done, verified |
| Tier-2 app + registry entry + E16 config block | done, verified on fixture |
| E16-derived runnable topology | done (synthetic edges) |
| Test inside a live rWHALE checkout | **needed** -- to confirm rWHALE's `ApplicationData`->CLI argument wiring (mirrors ResidualDemand; may need a 1-2 line tweak) |
| Real E16 line topology + a real DL run | **needed** -- E16 ships no connectivity and no outputs |
| `torch` made an optional/CPU dependency | recommended |

## 8. Dependencies and licensing

RSR is open source (Python; numpy, torch, networkx, igraph). It installs via pip and fits R2D's
Python backend. `torch` is the only heavy dependency and can be CPU-only / optional. No change to
existing R2D apps is required -- this is purely additive.

## 9. Proposed next steps

1. SimCenter review of the design and the SystemPerformance app contract fit.
2. Joint test inside a SimCenterBackendApplications checkout to finalise the rWHALE argument wiring.
3. Build a real power topology layer for E16 (or a new power example) and run a full DL -> RSR
   pass, producing a connectivity-reliability + criticality result alongside the existing damage map.
4. Contribute the app to `modules/systemPerformance/RSR2SP/` with the registry entry and example.

## Appendix -- package manifest (`r2d_addon/`)

```
r2d_performance.py                         Tier-1 post-processor (adapters + RSR core + demo)
README.md                                       integration overview
PROPOSAL.md                                     this note
applications/systemPerformance/RSR2SP/
    run__performance.py                     Tier-2 rWHALE SystemPerformance app
    WorkflowApplications_RSR_entry.json        registry entry to add
    E16_SystemPerformance_block.json            input.json block enabling RSR for E16
    example_e16/                                runnable E16-derived network + synthetic DL + outputs
```

**References.** Byun, J.-E., Ryu, H., & Straub, D. (2024). Branch-and-bound algorithm for efficient
reliability analysis of general coherent systems. RSR source: `~/Projects/rsr`. Schema verified
against NHERI-SimCenter `SimCenterBackendApplications` and `R2DExamples` (GitHub, master).
