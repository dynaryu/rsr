"""
Tier-1 R2D add-on: RSR system-reliability + critical failure modes as a post-processor.

R2D (NHERI SimCenter) runs Hazard -> Asset -> ... -> DL (pelicun damage & loss), producing
per-asset damage-state probabilities. Its existing infrastructure tools (E14 transportation,
E15/REWET water, E16 power) are simulation/functionality based and do NOT return system
reliability with explicit failure modes. This tool fills that gap: it consumes pelicun damage
probabilities + a network topology and uses RSR to compute

  * the probability the network is disconnected (source -> load), with bounds, and
  * the minimal cut-sets -- the critical components whose joint failure disconnects the system.

Two pipelines (run either or both, --mode):

  1. REFERENCE-STATE ("model the system")  --mode reference
     Extract the reusable cut-set / survival rule set for a network topology with RSR rule
     extraction. Topology-dependent and probability-independent (the only expensive step), so
     it is done once per system and saved as a reference model (rsr_reference_model.json).
     The minimal cut-sets and component criticality are properties of this reference state.

  2. APPLICATION ("apply the modelled system")  --mode apply
     Given a saved reference model + a specific set of pelicun damage probabilities, evaluate
     P(disconnected) with bounds by sampling and classifying against the saved rules -- cheap,
     no re-extraction. Run this per hazard level / per realization.

  --mode both (default) runs 1 then 2 in one shot (the original single-call behaviour).

Self-contained demo (synthetic power network, no R2D data needed):
    ~/Projects/rsr/.venv/bin/python r2d_rsr_performance.py --demo

Reference model once, then apply to many R2D result dirs:
    r2d_rsr_performance.py --mode reference --nodeFile nodes.geojson --edgeFile edges.geojson \
        --source GEN_1 --sink LOAD_5 --out model/
    r2d_rsr_performance.py --mode apply --model model/rsr_reference_model.json \
        --runDir results_M7/ --out results_M7/

Outputs: rsr_reference_model.json (+ rsr_critical_components.csv) from pipeline 1,
         rsr_system_reliability.json from pipeline 2.
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import torch
import networkx as nx

from rsr.rsr import (run_rule_extraction_by_mcs, from_rule_dict_to_mat,
                     sample_categorical, classify_samples)
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

    NOTE: the topology (and thus the failable component set) defines the reference state, so the
    same nodes/edges geojson must back both pipelines. `p_fail` is used here only to decide which
    assets are failable; pass the union of all assets you may damage (an empty dict makes every
    structural element failable, which is the safe default for a reference-only run).
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
# PIPELINE 1 -- REFERENCE STATE: extract the reusable rule set (cut-sets) for a topology.
# Topology-dependent, probability-independent: run once per system, save, reuse.
# ===========================================================================================
def _resolve_ports(G, source, sink):
    """If a terminal substation was node-split it no longer exists as a plain node -- inject at
    its "__in" port and draw from the sink's "__out" port, so the terminal substations' own
    failures sit on the path."""
    def _port(n, prefer):
        n = str(n)
        if n in G.nodes:
            return n
        cand = f"{n}__{prefer}"
        return cand if cand in G.nodes else n
    return _port(source, "in"), _port(sink, "out")


def extract_reference(G, failable, source, sink, p_fail_guide=None, device="cpu",
                      n_sample=500_000, unk_thres=0.02):
    """
    Pipeline 1. Extract the reference state: the survival + failure (cut-set) rule sets for the
    source->sink connectivity of `G`. Returns a reference-model dict (JSON-serialisable) that
    apply_reference() consumes -- it carries the rules, the component order, and the
    topology-only cut-sets/criticality.

    `p_fail_guide` only steers RSR's importance sampling and the termination gap; the extracted
    rules are valid for ANY probabilities. Pass representative damage probabilities when you have
    them (tighter rule set for that regime); a uniform nominal 0.1 is used per component otherwise.
    """
    source, sink = _resolve_ports(G, source, sink)
    row_names = list(failable)
    n_state = 2                                   # binary: 0 = failed, 1 = functional
    guide = p_fail_guide or {}
    # probs in RSR order: [P(state0=failed), P(state1=up)]
    probs = torch.tensor([[guide.get(c, 0.1), 1.0 - guide.get(c, 0.1)] for c in row_names],
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
    # RSR naming: "upper" rule set is sys >= 1 (survival/connected), "lower" is sys <= 0 (failure).
    rules_upper = json.load(open(res["rules_upper_path"]))
    rules_lower = json.load(open(res["rules_lower_path"]))
    cutsets = cutsets_from_rules(rules_lower)
    return {"n_state": n_state, "source": source, "sink": sink, "row_names": row_names,
            "rules_upper": rules_upper, "rules_lower": rules_lower,
            "n_cutsets": len(cutsets), "cutsets": cutsets,
            "criticality": criticality(cutsets)}


def cutsets_from_rules(rules_lower):
    """Minimal cut-sets = the failure rules' component sets (drop the bookkeeping 'sys' key)."""
    cutsets = []
    for r in rules_lower:
        comps = sorted(c for c in r if c != "sys")
        if comps:
            cutsets.append(comps)
    return cutsets


def criticality(cutsets):
    """Rank components: single-component cut-sets (single points of failure) first, then by
    how many cut-sets they appear in. A property of the reference state (topology), not of any
    particular damage scenario."""
    from collections import Counter
    spof = {c[0] for c in cutsets if len(c) == 1}
    freq = Counter(c for cs in cutsets for c in cs)
    rows = []
    for comp, f in freq.most_common():
        rows.append({"component": comp, "single_point_of_failure": comp in spof,
                     "appears_in_n_cutsets": f,
                     "min_cutset_size": min(len(cs) for cs in cutsets if comp in cs)})
    return rows


def save_reference(reference, path):
    Path(path).write_text(json.dumps(reference, indent=2))


def load_reference(path):
    return json.loads(Path(path).read_text())


# ===========================================================================================
# PIPELINE 2 -- APPLICATION: evaluate reliability for given damage probabilities by classifying
# samples against a pre-extracted reference model. Cheap; no rule extraction.
# ===========================================================================================
def apply_reference(reference, p_fail, device="cpu", n_sample=500_000, sample_batch_size=100_000):
    """
    Pipeline 2. Apply a reference model (from extract_reference / load_reference) to a specific
    set of per-component failure probabilities `p_fail` -> P(disconnected) with bounds.

    Monte-Carlo: draw component states from `p_fail`, classify each against the saved survival
    and failure rules. Proven-failed fraction is the lower bound on P(disconnected); the
    unclassified ('unknown') fraction is the gap that brackets the answer.
    """
    row_names = reference["row_names"]
    n_state = int(reference.get("n_state", 2))
    probs = torch.tensor([[p_fail.get(c, 0.0), 1.0 - p_fail.get(c, 0.0)] for c in row_names],
                         dtype=torch.float64, device=device)

    def _mats(rule_list):
        if not rule_list:
            return torch.zeros((0,), device=device)
        return torch.stack([from_rule_dict_to_mat(r, row_names, n_state, device=device)
                            for r in rule_list])
    up_mat = _mats(reference.get("rules_upper", []))
    lo_mat = _mats(reference.get("rules_lower", []))

    counts = {"upper": 0, "lower": 0, "unknown": 0}
    drawn = 0
    while drawn < n_sample:
        b = min(sample_batch_size, n_sample - drawn)
        c = classify_samples(sample_categorical(probs, b), up_mat, lo_mat)
        for k in counts:
            counts[k] += c[k]
        drawn += b

    p_disc = counts["lower"] / n_sample           # lower bound on P(disconnected)
    p_unk = counts["unknown"] / n_sample          # the unknown gap
    p_conn = counts["upper"] / n_sample           # lower bound on P(connected)
    return {"p_disconnected_lower": p_disc, "p_disconnected_upper": p_disc + p_unk,
            "unknown_gap": p_unk, "p_connected": p_conn}


# ===========================================================================================
# Convenience: both pipelines in one call (original single-shot interface; used by the Tier-2
# run_rsr_performance.py app). Extract the reference at `p_fail`, then apply at the same `p_fail`.
# ===========================================================================================
def run_rsr_connectivity(G, failable, p_fail, source, sink, device="cpu",
                         n_sample=500_000, unk_thres=0.02):
    reference = extract_reference(G, failable, source, sink, p_fail_guide=p_fail, device=device,
                                  n_sample=n_sample, unk_thres=unk_thres)
    rel = apply_reference(reference, p_fail, device=device, n_sample=n_sample)
    return {**rel, "n_cutsets": reference["n_cutsets"], "cutsets": reference["cutsets"]}


# ===========================================================================================
def _demo_inputs():
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
    return G, comps, p_fail, "GEN", "LOAD"


def _write_criticality(reference, out: Path):
    import csv
    with open(out / "rsr_critical_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "single_point_of_failure",
                                          "appears_in_n_cutsets", "min_cutset_size"])
        w.writeheader(); w.writerows(reference["criticality"])


def _write_reliability(rel, reference, out: Path):
    (out / "rsr_system_reliability.json").write_text(json.dumps(
        rel | {"source": reference["source"], "sink": reference["sink"],
               "n_cutsets": reference["n_cutsets"], "top_cutsets": reference["cutsets"][:20]},
        indent=2))


def _print_reference(reference, model_path: Path):
    crit = reference["criticality"]
    print(f"[reference] {reference['n_cutsets']} minimal cut-sets.  Single points of failure: "
          f"{[r['component'] for r in crit if r['single_point_of_failure']] or 'none'}")
    print("[reference] top critical components:")
    for r in crit[:6]:
        print(f"  {r['component']:12s} SPOF={r['single_point_of_failure']!s:5s} "
              f"in {r['appears_in_n_cutsets']} cut-sets (min size {r['min_cutset_size']})")
    print(f"[reference] wrote model: {model_path}")


def _print_reliability(rel):
    print(f"[apply] P(disconnected) in [{rel['p_disconnected_lower']:.4f}, "
          f"{rel['p_disconnected_upper']:.4f}]  (unknown gap {rel['unknown_gap']:.4f})")


def main():
    ap = argparse.ArgumentParser(description="RSR system-reliability add-on for R2D (two pipelines)")
    ap.add_argument("--mode", choices=["reference", "apply", "both"], default="both",
                    help="reference: extract the reusable cut-set model from a topology; "
                         "apply: evaluate reliability for damage probs using a saved model; "
                         "both: extract then apply in one shot (default)")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic power network")
    ap.add_argument("--runDir", type=Path, help="R2D results dir holding Results_<rlz>.json (damage probs)")
    ap.add_argument("--nodeFile", type=Path, help="Nodes geojson (R2D format)")
    ap.add_argument("--edgeFile", type=Path, help="Edges geojson (R2D format)")
    ap.add_argument("--source", type=str); ap.add_argument("--sink", type=str)
    ap.add_argument("--fail-from-ds", type=int, default=2)
    ap.add_argument("--model", type=Path,
                    help="reference-model JSON: written by --mode reference, read by --mode apply "
                         "(default <out>/rsr_reference_model.json)")
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model_path = a.model or (a.out / "rsr_reference_model.json")

    # ---- assemble inputs ----
    G = failable = None
    p_fail = {}
    if a.demo:
        G, failable, p_fail, source, sink = _demo_inputs()
        print("DEMO: synthetic power network (GEN -> {SUB_A | SUB_B} -> BUS -> LOAD)")
    else:
        source, sink = a.source, a.sink
        if a.runDir:
            p_fail = load_damage_probs(a.runDir, a.fail_from_ds)
        if a.mode in ("reference", "both"):
            if not (a.nodeFile and a.edgeFile and source and sink):
                ap.error("--mode reference/both need --nodeFile --edgeFile --source --sink")
            if not a.runDir:
                # load_network marks a node failable only if its asset id is in p_fail, so
                # without --runDir the failable-asset inventory is empty and only lines would
                # fail -- a silently wrong reference model. Require the inventory explicitly.
                ap.error("--mode reference/both need --runDir to enumerate the failable-asset "
                         "inventory (its damage probs only steer sampling); use --demo for the "
                         "synthetic network")
            G, failable = load_network(a.nodeFile, a.edgeFile, p_fail)

    # ---- pipeline 1: reference state (model the system) ----
    reference = None
    if a.mode in ("reference", "both"):
        reference = extract_reference(G, failable, source, sink, p_fail_guide=p_fail or None)
        save_reference(reference, model_path)
        _write_criticality(reference, a.out)
        _print_reference(reference, model_path)

    # ---- pipeline 2: application (apply the modelled system) ----
    if a.mode in ("apply", "both"):
        if reference is None:                     # apply-only: load the pre-built model
            if not model_path.exists():
                ap.error(f"--mode apply needs a reference model; not found: {model_path}")
            reference = load_reference(model_path)
            _write_criticality(reference, a.out)
        if not a.demo and not a.runDir:
            ap.error("--mode apply/both need damage probs via --runDir (or --demo)")
        rel = apply_reference(reference, p_fail)
        _write_reliability(rel, reference, a.out)
        _print_reliability(rel)
        print(f"[apply] wrote: {a.out/'rsr_system_reliability.json'}")


if __name__ == "__main__":
    main()
