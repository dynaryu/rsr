# E16 power network topology for the RSR SystemPerformance app

A runnable network for R2D's **E16 Electric Power Earthquake** example. E16 ships only
point/polygon power *damage* assets with **no connectivity**, so this provides the node/edge
topology the RSR SystemPerformance app (and any connectivity analysis) needs.

## Files

| File | What | Source |
|---|---|---|
| `power_nodes.geojson` | 35 substations: `nodeID` (topology) + `AIM_id` (original E16 inventory index, for damage linking) | subset of E16 `input_powerdata.geojson` |
| `power_edges.geojson` | 41 transmission lines: `StartNode`/`EndNode` (nodeIDs) + geometry | **synthesised**: minimum spanning tree over the substations + 7 shortest redundant lines (partial mesh) |
| `rsr_config.json` | the `ApplicationData` block: Source/Sink nodeIDs, FailureThresholdDS, DamageInput | — |
| `Results_0..4.json` | **synthesised** pelicun-style DL output (`R2Dres_MostLikelyCriticalDamageState` per substation) | stand-in for a real E16 DL run |

## How the topology was synthesised

E16 has no real connectivity, so edges were generated from substation geometry: a **minimum
spanning tree** (a connected radial backbone) **plus the 7 shortest non-tree lines** to add
redundancy (a realistic partially-meshed transmission layout). Substations keep their **original
E16 inventory index as `AIM_id`**, so they link straight to pelicun damage keyed by asset id.
Replace `power_edges.geojson` with a real Western-Power-style line layer when available; nothing
else changes.

## Run

```bash
~/Projects/rsr/.venv/bin/python ../run_rsr_performance.py \
    --nodeFile power_nodes.geojson --edgeFile power_edges.geojson \
    --configFile rsr_config.json --r2dRunDir . --rsrRunDir out
```

Result (with the synthetic 5-realization damage): `P(disconnected) ~ 0.90`, **9 critical
substations** (single points of failure on the source->sink routes) identified out of 35 -- the
redundant lines spare the rest. Outputs in `out/`: `R2D_results.geojson` (per-substation
criticality map layer), `rsr_system_reliability.json`, `rsr_critical_components.csv`.

> The high disconnection probability reflects E16's default **5 realizations** (coarse `p_fail`
> in steps of 0.2) over a long source->sink route with several articulation points; a production
> run with more realizations gives finer probabilities.
