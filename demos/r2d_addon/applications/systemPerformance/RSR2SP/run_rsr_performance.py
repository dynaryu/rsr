"""
RSR network-reliability SystemPerformance application for the NHERI SimCenter rWHALE workflow.

Conforms to the same contract as the ResidualDemand SystemPerformance app
(modules/systemPerformance/ResidualDemand/run_residual_demand.py):

    run_rsr_performance.py --nodeFile <nodes.geojson> --edgeFile <edges.geojson> \
        --configFile <ApplicationData.json> --r2dRunDir <dir with Results_<rlz>.json> \
        [--rsrRunDir <out>] [--mode reference|apply|both] [--model <reference_model.json>]

Two pipelines (see the Tier-1 r2d_rsr_performance.py for the rationale), selected with --mode:

    reference  pipeline 1 -- extract the reusable cut-set / survival rule set for the network
               topology and save it as a reference model (rsr_reference_model.json). Topology-
               dependent and probability-independent, so it is done ONCE per network and reused.
               (Still reads --r2dRunDir to enumerate the failable asset inventory; the damage
               probabilities only steer RSR's importance sampling.)
    apply      pipeline 2 -- load a saved reference model and evaluate P(disconnected) with bounds
               for this run's damage probabilities. Cheap (sample + classify, no extraction); run
               once per hazard level / realization batch.
    both       (default) pipeline 1 then 2 in one rWHALE invocation -- the original behaviour.

It reads R2D's per-realization DL output (`Results_<rlz>.json`, field
`R2Dres_MostLikelyCriticalDamageState`) and writes, into the run dir:

    rsr_reference_model.json     the reusable rule set + cut-sets (pipeline 1)
    rsr_critical_components.csv   component criticality ranking (pipeline 1)
    rsr_system_reliability.json  P(disconnected) with bounds + top cut-sets (pipeline 2)
    R2D_results.geojson           criticality + p_fail layer per failable asset for R2D (pipeline 2)

configFile is the SystemPerformance ApplicationData block, e.g.:
    {"Source": "0", "Sink": "12", "FailureThresholdDS": 2,
     "DamageInput": {"Type": "SampleFromRealizations", "Parameters": {"SampleSize": 5, "SampleSeed": 1}}}
Source/Sink are used by the reference pipeline; FailureThresholdDS and DamageInput by both.
"""
import argparse, csv, json, sys
from pathlib import Path

# reuse the verified Tier-1 adapters + the two-pipeline RSR core
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from r2d_rsr_performance import (load_damage_probs, load_network,  # noqa: E402
                                 extract_reference, apply_reference,
                                 save_reference, load_reference, criticality)


def select_realizations(run_dir: Path, damage_input):
    """Mirror ResidualDemand.select_realizations_to_run: choose which Results_<rlz> to use."""
    avail = sorted(int(p.stem.split("_", 1)[1]) for p in run_dir.iterdir()
                   if p.name.startswith("Results_") and p.suffix == ".json")
    if not damage_input:
        return avail
    t = damage_input.get("Type")
    if t == "SpecificRealization":
        flt = damage_input["Parameters"]["Filter"]
        sel = []
        for tok in str(flt).split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-"); sel += list(range(int(a), int(b) + 1))
            elif tok:
                sel.append(int(tok))
        return [r for r in sel if r in avail]
    if t == "SampleFromRealizations":
        import numpy as np
        p = damage_input["Parameters"]
        rng = np.random.default_rng(p.get("SampleSeed", 0))
        n = min(int(p["SampleSize"]), len(avail))
        return sorted(rng.choice(avail, n, replace=False).tolist())
    return avail


def write_results_geojson(node_file, edge_file, p_fail, crit_rows, out_path):
    """Attach criticality to each failable asset's geometry -> a layer R2D can map."""
    crit = {r["component"]: r for r in crit_rows}

    def asset_id(props, idx):
        for k in ("AIM_id", "assetID", "asset_id", "id"):
            if props.get(k) is not None:
                return str(props[k])
        return str(idx)

    feats = []
    for fpath in (node_file, edge_file):
        gj = json.loads(Path(fpath).read_text())
        for i, f in enumerate(gj["features"]):
            aid = asset_id(f["properties"], i)
            if aid not in p_fail:
                continue
            c = crit.get(aid, {})
            props = {"asset_id": aid, "p_fail": round(p_fail[aid], 4),
                     "single_point_of_failure": bool(c.get("single_point_of_failure", False)),
                     "appears_in_n_cutsets": int(c.get("appears_in_n_cutsets", 0)),
                     "min_cutset_size": int(c.get("min_cutset_size", 0))}
            feats.append({"type": "Feature", "geometry": f.get("geometry"), "properties": props})
    Path(out_path).write_text(json.dumps(
        {"type": "FeatureCollection", "metadata": {"WorkflowType": "RSR2SP"},
         "features": feats}, indent=2))


def write_criticality_csv(crit_rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "single_point_of_failure",
                                          "appears_in_n_cutsets", "min_cutset_size"])
        w.writeheader(); w.writerows(crit_rows)


def main():
    ap = argparse.ArgumentParser("RSR network-reliability SystemPerformance app", allow_abbrev=False)
    ap.add_argument("--nodeFile", required=True)
    ap.add_argument("--edgeFile", required=True)
    ap.add_argument("--configFile", required=True)
    ap.add_argument("--r2dRunDir", required=True)
    ap.add_argument("--rsrRunDir", default=None)
    ap.add_argument("--mode", choices=["reference", "apply", "both"], default=None,
                    help="reference: extract+save the reusable rule set; apply: evaluate "
                         "reliability from a saved model; both: extract then apply. "
                         "Defaults to the configFile's \"Mode\" (else \"both\").")
    ap.add_argument("--model", default=None,
                    help="reference-model JSON: written by reference, read by apply. "
                         "Defaults to the configFile's \"Model\" (else "
                         "<rsrRunDir|r2dRunDir>/rsr_reference_model.json).")
    ap.add_argument("--input", default=None)   # rWHALE passes this; unused here
    a = ap.parse_args()

    cfg = json.loads(Path(a.configFile).read_text())
    thr = int(cfg.get("FailureThresholdDS", 2))
    run_dir = Path(a.r2dRunDir)
    out = Path(a.rsrRunDir or run_dir); out.mkdir(parents=True, exist_ok=True)
    # Mode/Model: CLI overrides, else the ApplicationData block (configFile), else defaults --
    # so rWHALE drives the pipeline split via the E16 ApplicationData, like Source/Sink.
    mode = a.mode or str(cfg.get("Mode", "both"))
    if mode not in ("reference", "apply", "both"):
        ap.error(f'Mode must be reference|apply|both, got "{mode}"')
    model_path = (Path(a.model) if a.model
                  else Path(cfg["Model"]) if cfg.get("Model")
                  else out / "rsr_reference_model.json")

    # damage probabilities for the selected realizations: the failable asset inventory in every
    # mode, the actual scenario probabilities in apply/both, the sampling guide in reference.
    rlz = select_realizations(run_dir, cfg.get("DamageInput"))
    p_fail = load_damage_probs(run_dir, thr, rlz_filter=rlz)

    # ---- pipeline 1: reference state (model the system) ----
    reference = None
    if mode in ("reference", "both"):
        source, sink = str(cfg["Source"]), str(cfg["Sink"])
        G, comps = load_network(Path(a.nodeFile), Path(a.edgeFile), p_fail)
        reference = extract_reference(G, comps, source, sink, p_fail_guide=p_fail)
        save_reference(reference, model_path)
        write_criticality_csv(reference["criticality"], out / "rsr_critical_components.csv")
        print(f"RSR [reference]: {reference['n_cutsets']} cut-sets; SPOF="
              f"{[r['component'] for r in reference['criticality'] if r['single_point_of_failure']] or 'none'}; "
              f"wrote {model_path.name}, rsr_critical_components.csv")

    # ---- pipeline 2: application (apply the modelled system) ----
    if mode in ("apply", "both"):
        if reference is None:                      # apply-only: load the pre-built model
            if not model_path.exists():
                ap.error(f"--mode apply needs a reference model; not found: {model_path}")
            reference = load_reference(model_path)
            write_criticality_csv(reference["criticality"], out / "rsr_critical_components.csv")
        crit = reference["criticality"]
        rel = apply_reference(reference, p_fail)

        (out / "rsr_system_reliability.json").write_text(json.dumps(
            rel | {"source": reference["source"], "sink": reference["sink"],
                   "n_cutsets": reference["n_cutsets"],
                   "top_cutsets": reference["cutsets"][:20],
                   "realizations_used": rlz}, indent=2))
        write_results_geojson(a.nodeFile, a.edgeFile, p_fail, crit, out / "R2D_results.geojson")

        print(f"RSR [apply]: P(disconnected) in "
              f"[{rel['p_disconnected_lower']:.4f}, {rel['p_disconnected_upper']:.4f}]; "
              f"{reference['n_cutsets']} cut-sets; wrote rsr_system_reliability.json, R2D_results.geojson")


if __name__ == "__main__":
    main()
