"""
Tier-1 R2D add-on: RSR system-reliability + critical failure modes as a post-processor.

R2D (NHERI SimCenter) runs Hazard -> Asset -> ... -> DL (pelicun damage & loss), producing
per-asset damage-state probabilities. Its existing infrastructure tools (E14 transportation,
E15/REWET water, E16 power) are simulation/functionality based and do NOT return system
reliability with explicit failure modes. This tool fills that gap: it consumes pelicun damage
probabilities + a network topology and uses RSR to compute

  * the probability the network is disconnected (source -> load), with bounds, and
  * the minimal cut-sets -- the critical components whose joint failure disconnects the system.

It runs at R2D's "Performance" stage (here as a standalone post-processor / Tier-1 integration;
Tier-2 would register it as a workflow application). RSR's connectivity system function handles
BOTH node failures (substations) and edge failures (lines).

Self-contained demo (synthetic power network, no R2D data needed):
    ~/Projects/rsr/.venv/bin/python r2d_performance.py --demo

Against real R2D output (verify the adapter field names against your R2D/pelicun version):
    r2d_performance.py --damage <pelicun_DL.csv> --network <inventory.geojson> \
        --source GEN_1 --sink LOAD_5 --out results/

Outputs: rsr_system_reliability.json, rsr_critical_components.csv
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import torch
import networkx as nx

from rsr.rsr import run_rule_extraction_by_mcs, from_rule_dict_to_mat
from rsr.igraph_sfun import make_igraph_sfun_conn


# ===========================================================================================
# R2D / pelicun ADAPTERS  -- the only R2D-version-specific code. Verify field names against
# your pelicun output; everything below these two functions is generic.
# ===========================================================================================
def load_damage_probs(run_dir: Path, fail_from_ds: int = 2, rlz_filter=None) -> dict[str, float]:
    """
    Map R2D regional DL output -> P(component failed) per asset.

    Reads the per-realization result files written by R2D's DL stage (pelicun): `Results_<rlz>.json`
    in `run_dir`, the exact files R2D's own SystemPerformance apps (ResidualDemand, REWET) consume.
    Each has structure
        res[asset_type][asset_subtype][str(asset_id)]['R2Dres']['R2Dres_MostLikelyCriticalDamageState']
    (an integer damage state, 0 = undamaged, increasing = worse). A component is FAILED if its
    damage state >= `fail_from_ds`. P(fail) = fraction of realizations in which it is failed.

    Returns {asset_id (str): p_fail}.  (Schema verified against SimCenterBackendApplications
    modules/systemPerformance/ResidualDemand/run_residual_demand.py.)
    """
    from collections import defaultdict
    rlz_files = sorted(p for p in run_dir.iterdir()
                       if p.name.startswith("Results_") and p.suffix == ".json")
    if rlz_filter is not None:                 # keep only selected realization indices
        keep = {str(i) for i in rlz_filter}
        rlz_files = [p for p in rlz_files if p.stem.split("_", 1)[1] in keep]
    if not rlz_files:
        raise FileNotFoundError(f"no Results_<rlz>.json in {run_dir}")
    cnt = defaultdict(lambda: [0, 0])     # asset_id -> [n_failed, n_total]
    for rf in rlz_files:
        res = json.loads(rf.read_text())
        for _atype, subtypes in res.items():
            if not isinstance(subtypes, dict):
                continue
            for _sub, assets in subtypes.items():
                if not isinstance(assets, dict):
                    continue
                for aid, a in assets.items():
                    ds = (a.get("R2Dres", {}) or {}).get("R2Dres_MostLikelyCriticalDamageState")
                    if ds is None:
                        continue
                    cnt[str(aid)][1] += 1
                    if int(ds) >= fail_from_ds:
                        cnt[str(aid)][0] += 1
    return {aid: (c[0] / c[1] if c[1] else 0.0) for aid, c in cnt.items()}


def load_network(nodes_geojson: Path, edges_geojson: Path, p_fail: dict):
    """
    Read R2D node + edge GeoJSON (the ResidualDemand format) -> (nx.Graph, edge_components).

    Nodes GeoJSON: features with `properties.nodeID` (topology id). A node that is also a
    damageable asset (its asset id is in `p_fail`) becomes a failable substation. Edges GeoJSON:
    features with `properties.StartNode`/`EndNode` (referencing nodeIDs); an edge whose asset id
    is in `p_fail` is a failable line. Asset id is read from the first present of
    AIM_id / assetID / asset_id / id (else the feature index, matching R2D's 0-based asset ids).

    Failable nodes are node-split so every failable asset is an edge whose component id equals its
    R2D asset id -- linking straight to load_damage_probs(). Returns (G, edge_components).
    """
    def asset_id(props, idx):
        for k in ("AIM_id", "assetID", "asset_id", "id"):
            if props.get(k) is not None:
                return str(props[k])
        return str(idx)

    nodes_spec, edges_spec = [], []
    for i, f in enumerate(json.loads(nodes_geojson.read_text())["features"]):
        p = f["properties"]; aid = asset_id(p, i)
        nodes_spec.append({"id": str(p["nodeID"]), "comp_id": aid, "failable": aid in p_fail})
    for i, f in enumerate(json.loads(edges_geojson.read_text())["features"]):
        p = f["properties"]; aid = asset_id(p, i)
        # All structural edges are components so the connectivity graph stays intact;
        # non-damageable lines (aid not in p_fail) get p_fail=0 downstream -> always present.
        edges_spec.append({"from": str(p["StartNode"]), "to": str(p["EndNode"]),
                           "id": aid, "failable": True})
    return build_split_graph(nodes_spec, edges_spec)


def build_split_graph(nodes, edges):
    """
    Build a node-split nx.Graph; return (G, ordered list of failable component ids).
    A node has a topology 'id' and optional 'comp_id' (damage asset id, used as the eid when the
    node is split). An edge has 'from'/'to' (topology ids) and 'id' (component id).
    """
    G = nx.Graph()
    split = {}   # topology node id -> (in_name, out_name) for failable substations
    comps = []
    for n in nodes:
        cid = str(n.get("comp_id", n["id"]))
        if n.get("failable", False):
            i, o = f"{n['id']}__in", f"{n['id']}__out"
            G.add_edge(i, o, eid=cid); split[n["id"]] = (i, o); comps.append(cid)
        else:
            G.add_node(n["id"])
    def port(nid, end):  # line into a substation's in/out port, or a plain node
        return split[nid][0 if end == "to" else 1] if nid in split else nid
    for e in edges:
        G.add_edge(port(e["from"], "from"), port(e["to"], "to"), eid=str(e["id"]))
        if e.get("failable", True):
            comps.append(str(e["id"]))
    return G, comps


# ===========================================================================================
# RSR core (generic)
# ===========================================================================================
def run_rsr_connectivity(G, failable, p_fail, source, sink, device="cpu",
                          n_sample=500_000, unk_thres=0.02):
    """Build the connectivity sfun, run RSR rule extraction, return reliability + cut-sets."""
    # Resolve source/sink: if the terminal substation was node-split it no longer exists as a
    # plain node -- inject at its "__in" port, draw from the sink's "__out" port, so the
    # terminal substations' own failures are on the path.
    def _port(n, prefer):
        n = str(n)
        if n in G.nodes:
            return n
        cand = f"{n}__{prefer}"
        return cand if cand in G.nodes else n
    source, sink = _port(source, "in"), _port(sink, "out")
    row_names = list(failable)
    n_state = 2                                   # binary: 0 = failed, 1 = functional
    # probs in RSR order: [P(state0=failed), P(state1=up)]
    probs = torch.tensor([[p_fail.get(c, 0.0), 1.0 - p_fail.get(c, 0.0)] for c in row_names],
                         dtype=torch.float64, device=device)

    _base = make_igraph_sfun_conn(G, source, sink)    # sys_state 1 = connected (survival)
    def sfun(cs):                                      # wrap: numeric value, drop info dict
        _, sys_st, _ = _base(cs)
        return float(sys_st), sys_st, None

    res = run_rule_extraction_by_mcs(
        sfun=sfun, probs=probs, row_names=row_names, n_state=n_state, sys_upper_st=1,
        rules_upper=[], rules_lower=[], unk_prob_thres=unk_thres, unk_prob_opt="rel",
        prob_update_every=100, n_sample=n_sample, sample_batch_size=100_000, max_rounds=3000,
        rule_update_verbose=False, output_dir=tempfile.mkdtemp())
    rules_fail = json.load(open(res["rules_lower_path"]))
    # Final probabilities live in the metrics log; the unknown gap brackets the answer.
    # In RSR's naming the "lower" rule set is sys <= 0 (disconnected/failure).
    metrics = res.get("metrics_log") or []
    last = next((m for m in reversed(metrics) if m.get("p_lower") is not None), {})
    p_disc = float(last.get("p_lower", np.nan))        # lower bound on P(disconnected)
    p_unk = float(last.get("p_unknown", 0.0))          # the unknown gap

    # minimal cut-sets: components driving disconnection (drop the bookkeeping 'sys' key)
    cutsets = []
    for r in rules_fail:
        comps = sorted(c for c in r if c != "sys")
        if comps:
            cutsets.append(comps)
    return {"p_disconnected_lower": p_disc, "p_disconnected_upper": p_disc + p_unk,
            "unknown_gap": p_unk,
            "p_connected": (1.0 - p_disc - p_unk) if np.isfinite(p_disc) else np.nan,
            "n_cutsets": len(cutsets), "cutsets": cutsets}


def criticality(cutsets):
    """Rank components: single-component cut-sets (single points of failure) first, then by
    how many cut-sets they appear in."""
    from collections import Counter
    spof = {c[0] for c in cutsets if len(c) == 1}
    freq = Counter(c for cs in cutsets for c in cs)
    rows = []
    for comp, f in freq.most_common():
        rows.append({"component": comp, "single_point_of_failure": comp in spof,
                     "appears_in_n_cutsets": f,
                     "min_cutset_size": min(len(cs) for cs in cutsets if comp in cs)})
    return rows


# ===========================================================================================
def demo():
    """Synthetic redundant power network: GEN -> two parallel substation paths -> BUS -> LOAD."""
    nodes = [{"id": "GEN"}, {"id": "LOAD"},
             {"id": "SUB_A", "failable": True}, {"id": "SUB_B", "failable": True},
             {"id": "BUS", "failable": True}]
    edges = [{"id": "L_G_A", "from": "GEN", "to": "SUB_A"}, {"id": "L_G_B", "from": "GEN", "to": "SUB_B"},
             {"id": "L_A_BUS", "from": "SUB_A", "to": "BUS"}, {"id": "L_B_BUS", "from": "SUB_B", "to": "BUS"},
             {"id": "L_BUS_LOAD", "from": "BUS", "to": "LOAD"}]
    G, comps = build_split_graph(nodes, edges)
    # synthetic P(fail) at one hazard level (what pelicun would supply per asset)
    p_fail = {"SUB_A": 0.30, "SUB_B": 0.30, "BUS": 0.05,
              "L_G_A": 0.10, "L_G_B": 0.10, "L_A_BUS": 0.10, "L_B_BUS": 0.10, "L_BUS_LOAD": 0.05}
    print("DEMO: synthetic power network (GEN -> {SUB_A | SUB_B} -> BUS -> LOAD)")
    return G, run_rsr_connectivity(G, comps, p_fail, "GEN", "LOAD")


def main():
    ap = argparse.ArgumentParser(description="RSR system-reliability add-on for R2D")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--runDir", type=Path, help="R2D results dir holding Results_<rlz>.json")
    ap.add_argument("--nodeFile", type=Path, help="Nodes geojson (R2D format)")
    ap.add_argument("--edgeFile", type=Path, help="Edges geojson (R2D format)")
    ap.add_argument("--source", type=str); ap.add_argument("--sink", type=str)
    ap.add_argument("--fail-from-ds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()

    if a.demo:
        _, out = demo()
    else:
        if not (a.runDir and a.nodeFile and a.edgeFile and a.source and a.sink):
            ap.error("need --runDir, --nodeFile, --edgeFile, --source, --sink (or --demo)")
        p_fail = load_damage_probs(a.runDir, a.fail_from_ds)
        G, failable = load_network(a.nodeFile, a.edgeFile, p_fail)
        out = run_rsr_connectivity(G, failable, p_fail, a.source, a.sink)

    crit = criticality(out["cutsets"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rsr_system_reliability.json").write_text(json.dumps(
        {k: v for k, v in out.items() if k != "cutsets"} | {"top_cutsets": out["cutsets"][:20]},
        indent=2))
    import csv
    with open(a.out / "rsr_critical_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "single_point_of_failure",
                                          "appears_in_n_cutsets", "min_cutset_size"])
        w.writeheader(); w.writerows(crit)

    print(f"\nP(disconnected) in [{out['p_disconnected_lower']:.4f}, "
          f"{out['p_disconnected_upper']:.4f}]  (unknown gap {out['unknown_gap']:.4f})")
    print(f"{out['n_cutsets']} minimal cut-sets.  Single points of failure: "
          f"{[r['component'] for r in crit if r['single_point_of_failure']] or 'none'}")
    print("Top critical components:")
    for r in crit[:6]:
        print(f"  {r['component']:12s} SPOF={r['single_point_of_failure']!s:5s} "
              f"in {r['appears_in_n_cutsets']} cut-sets (min size {r['min_cutset_size']})")
    print(f"\nWrote: {a.out/'rsr_system_reliability.json'}, {a.out/'rsr_critical_components.csv'}")


if __name__ == "__main__":
    main()
