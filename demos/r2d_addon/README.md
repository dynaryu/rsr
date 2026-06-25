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
                          Results_<rlz>.json   pipeline 1  reference model (rules + cut-sets)
                          + node/edge geojson   pipeline 2  P(disconnect) + bounds + criticality
```

This is **Tier-1**: a standalone post-processor reading R2D outputs. **Tier-2** registers the same
logic as a `SystemPerformance` application in `WorkflowApplications.json` -- its CLI already mirrors
the ResidualDemand app (`--nodeFile`, `--edgeFile`, run dir).

## The data handshake (schema verified against SimCenterBackendApplications)

| RSR needs | R2D provides | how |
|---|---|---|
| `probs` (per-asset P(fail)) | `Results_<rlz>.json`, field `R2Dres_MostLikelyCriticalDamageState` per asset; P(fail) = fraction of realizations with DS >= threshold | `load_damage_probs()` |
| `sfun` (system function) | node geojson (`nodeID`) + edge geojson (`StartNode`/`EndNode`); substations node-split so each failable asset is an edge | `load_network()` -> `make_igraph_sfun_conn` |
| (output) reference model + reliability | reusable rule set + cut-sets (pipeline 1), then P(fail) with bounds (pipeline 2) | `rsr_reference_model.json` / `rsr_critical_components.csv` / `rsr_system_reliability.json` |

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

## Two pipelines (`--mode`)

The analysis splits into two stages so the expensive part is done once and reused:

| | Pipeline 1 -- **reference state** | Pipeline 2 -- **application** |
|---|---|---|
| Purpose | *model the system*: extract the reusable survival/cut-set **rules** for a topology | *apply the modelled system*: evaluate reliability for one damage scenario |
| Depends on | topology + failable-asset inventory (the rule set itself is probability-independent) | a saved reference model + per-asset `p_fail` |
| Cost | expensive (RSR rule extraction) -- run **once** per network | cheap (sample + classify, no extraction) -- run **per hazard / realization** |
| Function | `extract_reference()` | `apply_reference()` |
| Writes | `rsr_reference_model.json` + `rsr_critical_components.csv` | `rsr_system_reliability.json` |

The conceptual split: **cut-sets and criticality are a property of the reference state** (the
topology), while the **probability bounds are a property of the application** (the damage
scenario). Each pipeline writes its own artifact accordingly.

`--mode` selects which to run:

- `reference` -- pipeline 1 only: extract the rule set and save the reference model. Reads
  `--runDir` to learn the failable-asset inventory (which assets can fail); the damage
  probabilities there only steer RSR's sampling, so any representative run dir works. (The
  synthetic `--demo` hard-codes failability, so it alone needs no run dir.)
- `apply` -- pipeline 2 only: load a saved model and evaluate reliability for `--runDir` damage probs.
- `both` *(default)* -- pipeline 1 then 2 in one shot (the original single-call behaviour).

## Run

```bash
# self-contained synthetic power-network demo (no R2D data needed; runs both pipelines)
~/Projects/rsr/.venv/bin/python r2d_rsr_performance.py --demo --out out/

# --- split workflow: model the system once, then apply to many scenarios ---

# pipeline 1: extract the reusable reference model (any representative --runDir gives the
# failable-asset inventory; its probabilities only steer sampling, the rules are reusable)
r2d_rsr_performance.py --mode reference --runDir <any_R2D_results_dir> \
    --nodeFile nodes.geojson --edgeFile edges.geojson --source <nodeID> --sink <nodeID> \
    --out model/                       # writes model/rsr_reference_model.json (+ criticality csv)

# pipeline 2: apply that model to each R2D result dir -- cheap, no re-extraction
r2d_rsr_performance.py --mode apply \
    --model model/rsr_reference_model.json --runDir <R2D_results_dir> --out out_M7/

# --- or both at once (topology + damage probs in a single call) ---
r2d_rsr_performance.py --runDir <R2D_results_dir> \
    --nodeFile nodes.geojson --edgeFile edges.geojson --source <nodeID> --sink <nodeID> --out out/
```

> `--model` defaults to `<out>/rsr_reference_model.json`, so a bare `--mode reference` then
> `--mode apply` sharing the same `--out` will hand the model off automatically.
>
> The reference model is **probability-independent** -- extract it once and reuse it across every
> hazard level, return period, and realization. `extract_reference()` accepts representative
> damage probs only to steer RSR's importance sampling (tighter rule set for that regime); the
> rules remain valid for any probabilities. A real-network `reference` run still passes `--runDir`
> to enumerate the *failable-asset inventory* (which assets can fail), independent of the
> probability values -- only the synthetic `--demo` (failability hard-coded) needs no run dir.

Demo output (GEN -> {SUB_A | SUB_B} -> BUS -> LOAD), `--mode both`:

```
[reference] 11 minimal cut-sets.  Single points of failure: ['BUS', 'L_BUS_LOAD']
[apply] P(disconnected) in [0.2674, 0.2674]  (unknown gap 0.0000)
```

Correctly: BUS and the final line are series single-points-of-failure; the two parallel paths only
fail together (size-2 cut-sets). Outputs: `rsr_reference_model.json` (rules + cut-sets +
criticality, from pipeline 1), `rsr_critical_components.csv` (criticality ranking -> an R2D map
layer, from pipeline 1), `rsr_system_reliability.json` (P + bounds + top cut-sets, from pipeline 2).

## Tier-2: registered SystemPerformance application

`applications/systemPerformance/RSR2SP/` packages the same logic as an rWHALE-registered app,
mirroring the ResidualDemand layout:

| File | Purpose |
|---|---|
| `run_rsr_performance.py` | the app: CLI `--nodeFile --edgeFile --configFile --r2dRunDir [--rsrRunDir] [--mode reference\|apply\|both] [--model …]`, reuses the verified Tier-1 adapters, writes the reference model + `R2D_results.geojson` + reliability JSON/CSV |
| `WorkflowApplications_RSR_entry.json` | the registry object to add under `SystemPerformanceApplications.Applications` in `WorkflowApplications.json` |
| `E16_SystemPerformance_block.json` | the `SystemPerformance` block to drop into E16's `input.json` (replaces `Application: "None"`) |

The app carries the same **two pipelines** as Tier-1 via `--mode` (default `both`, the original
single-shot behaviour). To extract the reusable rule set once and apply it per hazard run:

```bash
run_rsr_performance.py --mode reference --nodeFile … --edgeFile … --configFile cfg.json \
    --r2dRunDir run_M6/ --rsrRunDir model/        # writes model/rsr_reference_model.json
run_rsr_performance.py --mode apply --model model/rsr_reference_model.json \
    --nodeFile … --edgeFile … --configFile cfg.json --r2dRunDir run_M7/ --rsrRunDir out_M7/
```

`Source`/`Sink` (from `configFile`) drive the `reference` pipeline; `FailureThresholdDS` and the
`DamageInput` realization selection drive both. A `reference` run still reads `--r2dRunDir` to
enumerate the **failable asset inventory** (which nodes are damageable substations) — the damage
probabilities there only steer RSR's importance sampling, so any representative run dir works.

Inside rWHALE there are no CLI flags to set, so the pipeline is driven from the `ApplicationData`
block instead: the app reads `Mode` (`reference`|`apply`|`both`, default `both`) and an optional
`Model` path from `configFile`, exactly as it reads `Source`/`Sink`. The CLI `--mode`/`--model`
override these for standalone use. To extract once and reuse across hazard runs, set
`"Mode": "reference"` in the first run's block, then `"Mode": "apply"` with `"Model"` pointing at
the saved `rsr_reference_model.json` in later runs (`model` is also declared as a path input in the
registry entry so rWHALE can stage that file). With no `Mode` key the app runs `both`, so existing
E16 blocks keep working unchanged.

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
