"""
RSR network-reliability SystemPerformance application for the NHERI SimCenter rWHALE workflow.

Conforms to the same contract as the ResidualDemand SystemPerformance app
(modules/systemPerformance/ResidualDemand/run_residual_demand.py):

    run_rsr_performance.py --nodeFile <nodes.geojson> --edgeFile <edges.geojson> \
        --configFile <ApplicationData.json> --r2dRunDir <dir with Results_<rlz>.json> \
        [--rsrRunDir <out>]

It reads R2D's per-realization DL output (`Results_<rlz>.json`, field
`R2Dres_MostLikelyCriticalDamageState`), runs RSR source->sink connectivity over the supplied
node/edge network, and writes, into the run dir:

    R2D_results.geojson           criticality layer (per failable asset) for R2D visualisation
    rsr_system_reliability.json  P(disconnected) with bounds + top cut-sets
    rsr_critical_components.csv  component criticality ranking

configFile is the SystemPerformance ApplicationData block, e.g.:
    {"Source": "0", "Sink": "12", "FailureThresholdDS": 2,
     "DamageInput": {"Type": "SampleFromRealizations", "Parameters": {"SampleSize": 5, "SampleSeed": 1}}}
"""
import argparse, json, sys
from pathlib import Path

# reuse the verified Tier-1 adapters + RSR core
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from r2d_rsr_performance import (load_damage_probs, load_network,  # noqa: E402
                                  run_rsr_connectivity, criticality)


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


def main():
    ap = argparse.ArgumentParser("RSR network-reliability SystemPerformance app", allow_abbrev=False)
    ap.add_argument("--nodeFile", required=True)
    ap.add_argument("--edgeFile", required=True)
    ap.add_argument("--configFile", required=True)
    ap.add_argument("--r2dRunDir", required=True)
    ap.add_argument("--rsrRunDir", default=None)
    ap.add_argument("--input", default=None)   # rWHALE passes this; unused here
    a = ap.parse_args()

    cfg = json.loads(Path(a.configFile).read_text())
    source, sink = str(cfg["Source"]), str(cfg["Sink"])
    thr = int(cfg.get("FailureThresholdDS", 2))
    run_dir = Path(a.r2dRunDir)
    out = Path(a.rsrRunDir or run_dir); out.mkdir(parents=True, exist_ok=True)

    rlz = select_realizations(run_dir, cfg.get("DamageInput"))
    p_fail = load_damage_probs(run_dir, thr, rlz_filter=rlz)
    G, comps = load_network(Path(a.nodeFile), Path(a.edgeFile), p_fail)
    res = run_rsr_connectivity(G, comps, p_fail, source, sink)
    crit = criticality(res["cutsets"])

    (out / "rsr_system_reliability.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "cutsets"} | {"top_cutsets": res["cutsets"][:20],
         "realizations_used": rlz}, indent=2))
    import csv
    with open(out / "rsr_critical_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "single_point_of_failure",
                                          "appears_in_n_cutsets", "min_cutset_size"])
        w.writeheader(); w.writerows(crit)
    write_results_geojson(a.nodeFile, a.edgeFile, p_fail, crit, out / "R2D_results.geojson")

    print(f"RSR SystemPerformance: P(disconnected) in "
          f"[{res['p_disconnected_lower']:.4f}, {res['p_disconnected_upper']:.4f}]; "
          f"{res['n_cutsets']} cut-sets; SPOF="
          f"{[r['component'] for r in crit if r['single_point_of_failure']] or 'none'}; "
          f"wrote R2D_results.geojson")


if __name__ == "__main__":
    main()
