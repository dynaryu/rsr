# RSR as an R2D add-on (Tier-1 proof of concept)

A post-processor that adds **system reliability + critical failure modes** to NHERI SimCenter's
**R2D** workflow, using RSR. R2D's existing infrastructure tools (E14 transportation, E15/REWET
water, E16 power) are simulation/functionality based and stop at *asset damage* or
*functionality-over-time*; none return the **probability the network fails** with **explicit
cut-sets** (which components drive failure). That is what RSR adds.

## Where it fits in R2D -- the gap is concrete

R2D already has a `SystemPerformance` workflow stage with apps for transportation (ResidualDemand)
and water (REWET). For the **power** example (E16), `SystemPerformance` is literally
`{"PowerNetwork": {"Application": "None"}}` -- the slot is wired but **empty**. RSR fills it.

```
Hazard -> Asset -> ... -> DL (pelicun) -> [ SystemPerformance: RSR ] -> results
                          Results_<rlz>.json   P(disconnect) + bounds
                          + node/edge geojson  + minimal cut-sets (criticality)
```

This is **Tier-1**: a standalone post-processor reading R2D outputs. **Tier-2** registers the same
logic as a `SystemPerformance` application in `WorkflowApplications.json` -- its CLI already mirrors
the ResidualDemand app (`--nodeFile`, `--edgeFile`, run dir).

## The data handshake (schema verified against SimCenterBackendApplications)

| RSR needs | R2D provides | how |
|---|---|---|
| `probs` (per-asset P(fail)) | `Results_<rlz>.json`, field `R2Dres_MostLikelyCriticalDamageState` per asset; P(fail) = fraction of realizations with DS >= threshold | `load_damage_probs()` |
| `sfun` (system function) | node geojson (`nodeID`) + edge geojson (`StartNode`/`EndNode`); substations node-split so each failable asset is an edge | `load_network()` -> `make_igraph_sfun_conn` |
| (output) reliability + cut-sets | written as a metric + a criticality table | `rsr_*.json/.csv` |

The damage field name and the `Results_<rlz>.json` structure were verified against
`modules/systemPerformance/ResidualDemand/run_residual_demand.py` (the transportation
SystemPerformance backend), and the node/edge geojson fields against R2DExamples E14. The two
adapter functions are the only R2D-version-specific code and are isolated at the top of the script.

> Note: R2D's E16 power inventory is point/polygon *damage* assets with **no connectivity**, so a
> power SystemPerformance run needs a node/edge topology supplied (the ResidualDemand format) --
> this is the missing piece that RSR both needs and motivates.

Substations (node failures) are **node-split** into an internal edge, so every failable asset
becomes an edge -- the representation RSR's connectivity sfun handles, covering substation and
line failures uniformly.

## Run

```bash
# self-contained synthetic power-network demo (no R2D data needed)
~/Projects/rsr/.venv/bin/python r2d_rsr_performance.py --demo --out out/

# against real R2D output (Results_<rlz>.json in --runDir; node/edge geojson; topology source/sink)
r2d_rsr_performance.py --runDir <R2D_results_dir> \
    --nodeFile nodes.geojson --edgeFile edges.geojson --source <nodeID> --sink <nodeID> --out out/
```

Demo output (GEN -> {SUB_A | SUB_B} -> BUS -> LOAD):

```
P(disconnected) in [0.2651, 0.2651]  (unknown gap 0.0000)
11 minimal cut-sets.  Single points of failure: ['BUS', 'L_BUS_LOAD']
```

Correctly: BUS and the final line are series single-points-of-failure; the two parallel paths only
fail together (size-2 cut-sets). Outputs: `rsr_system_reliability.json` (P + bounds + top cut-sets),
`rsr_critical_components.csv` (criticality ranking -> an R2D map layer).

## Tier-2: registered SystemPerformance application

`applications/systemPerformance/RSR2SP/` packages the same logic as an rWHALE-registered app,
mirroring the ResidualDemand layout:

| File | Purpose |
|---|---|
| `run_rsr_performance.py` | the app: CLI `--nodeFile --edgeFile --configFile --r2dRunDir [--rsrRunDir]`, reuses the verified Tier-1 adapters, writes `R2D_results.geojson` + reliability JSON/CSV |
| `WorkflowApplications_RSR_entry.json` | the registry object to add under `SystemPerformanceApplications.Applications` in `WorkflowApplications.json` |
| `E16_SystemPerformance_block.json` | the `SystemPerformance` block to drop into E16's `input.json` (replaces `Application: "None"`) |

To install into a SimCenter checkout:
1. copy `applications/systemPerformance/RSR2SP/` into `SimCenterBackendApplications/applications/systemPerformance/`,
2. add the registry object to `modules/Workflow/WorkflowApplications.json`,
3. set E16's `input.json` `SystemPerformance` block (and supply a `power_nodes/edges.geojson` topology -- E16 ships none),
4. run via `rWHALE.py` as usual; the app reads `Results_<rlz>.json`, honours the `DamageInput`
   realization selection (`SpecificRealization` / `SampleFromRealizations`), and writes
   `R2D_results.geojson` for R2D to visualise.

The app contract (CLI args, `Results_<rlz>.json` input, `R2D_results.geojson` output, realization
selection) was matched to `run_residual_demand.py`; verified end-to-end on a fixture.

## What RSR uniquely adds vs existing R2D tools

- **Explicit critical failure modes (cut-sets)** -- which components drive system failure; the
  single-points-of-failure are the mitigation priorities. No R2D simulation tool produces this.
- **Bounds** on the system-failure probability (the unknown gap), not just a point estimate.
- **Rare-event efficiency** -- branch-and-bound, not crude Monte Carlo.
- **Hazard-agnostic** -- consumes damage probabilities, so works for any R2D hazard.

## Scale

Exact rule extraction is intractable for large regional networks (demonstrated on the SIRA EPN
Yilgarn case: a 373-component network did not finish). For R2D scale, use either RSR's
**subset-simulation** (`subset_sim.py`) or the **hierarchical decomposition** (per-asset fragility
curve -> reduced network), which maps directly onto R2D's per-asset DL step. The SIRA->RSR
pipeline in `../epn_yilgarn_caseB/` is a worked prototype of that hierarchical pattern.
