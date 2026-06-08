#!/usr/bin/env python3
"""Assemble final deliverables for the CodegenResidue parametric-chaos bundle."""
import json
import glob
import os

OUT = "/work/work/tensilelite-characterization/parametric-chaos/CodegenResidue"
FRAGS = OUT + "/_frags"
TESTDIR = "/work/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/CodegenResidue"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_census():
    census = {}
    with open(OUT + "/branch_census.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            bid = r.get("branch_id")
            if bid:
                census[bid] = r
    return census


def load_phase(phase):
    frags = {}
    for fp in sorted(glob.glob(FRAGS + "/" + phase + "/*.json")):
        d = _load(fp)
        if isinstance(d, dict) and d.get("branch_id"):
            frags[d["branch_id"]] = d
    return frags


def extract_site(frag, cen):
    if not frag:
        if cen:
            loc = cen.get("location", {})
            if isinstance(loc, dict):
                return {
                    "file": cen.get("file", "?"),
                    "line": loc.get("line", "?"),
                }, cen.get("predicate_source", "")
        return {}, ""
    site = frag.get("site", frag.get("source_location", {}))
    if isinstance(site, str):
        site = {}
    if not site.get("file"):
        pn = frag.get("predicate_normalized", {})
        if isinstance(pn, dict):
            site_str = pn.get("site", "")
            if site_str and ":" in site_str:
                parts = site_str.split(":")
                try:
                    site = {"file": parts[0], "line": int(parts[1])}
                except Exception:
                    pass
    if not site.get("file") and cen:
        loc = cen.get("location", {})
        if isinstance(loc, dict):
            site = {"file": cen.get("file", "?"), "line": loc.get("line", "?")}
    pred = frag.get("predicate_raw", frag.get("predicate_text", ""))
    if not pred:
        pn = frag.get("predicate_normalized", {})
        if isinstance(pn, dict):
            op = pn.get("op", "")
            lhs = str(
                pn.get("lhs", pn.get("var", pn.get("operand", pn.get("arg", ""))))
            )
            pred = f"{op} {lhs}"
    return site, str(pred)[:200]


def build_hypergraph(census, slices, all_bids):
    nodes = []
    edges = []
    mapped = 0
    for bid in sorted(all_bids):
        cen = census.get(bid, {})
        sf = slices.get(bid)
        site, pred = extract_site(sf, cen)
        pubs = sf.get("public_inputs", []) if sf else []
        ext = sf.get("external_state", []) if sf else []
        edge_src = "slice-agent" if sf else "census-defuse"
        if pubs:
            mapped += 1
        nodes.append(
            {
                "branch_id": bid,
                "file": site.get("file", cen.get("file", "?")),
                "line": site.get("line", "?"),
                "function": cen.get("function", "?"),
                "branch_kind": cen.get("branch_kind", "?"),
                "predicate_source": pred,
                "referenced_symbols": cen.get("referenced_symbols", []),
            }
        )
        edges.append(
            {
                "branch_id": bid,
                "predicate_source": pred,
                "public_inputs": pubs,
                "external_state": ext,
                "edge_source": edge_src,
            }
        )
    hypergraph = {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "edges_with_public_inputs": mapped,
            "slice_frag_coverage": len(slices),
            "census_defuse_fallback": len(
                [e for e in edges if e["edge_source"] == "census-defuse"]
            ),
        },
        "note": "Nodes are branch records from the 20-branch inventoried work-list. Edges join each node to its public_inputs from Slice frags. edge_source=census-defuse means no Slice frag was persisted.",
    }
    return hypergraph, mapped


def build_domain_model():
    domains = {}
    for sub in ("Domain", "Solve"):
        for fp in sorted(glob.glob(FRAGS + "/" + sub + "/*.json")):
            d = _load(fp)
            if not isinstance(d, dict):
                continue
            bid = d.get("branch_id")
            dom = d.get("domains")
            if bid and dom and bid not in domains:
                domains[bid] = dom
    return domains


def build_catalog(census, slices, solves, verifies, reifies, all_bids):
    records = []
    for bid in sorted(all_bids):
        cen = census.get(bid, {})
        sf = slices.get(bid)
        sol = solves.get(bid)
        ver = verifies.get(bid)
        rei = reifies.get(bid)
        site, pred = extract_site(sf, cen)
        test_paths = []
        if rei:
            tp = rei.get("test_paths", rei.get("test_file"))
            if isinstance(tp, list):
                test_paths = tp
            elif tp:
                test_paths = [tp]
        rec = {
            "branch_id": bid,
            "file": site.get("file", cen.get("file", "?")),
            "line": site.get("line", "?"),
            "function": cen.get("function", "?"),
            "branch_kind": cen.get("branch_kind", "?"),
            "predicate_source": pred,
            "public_inputs": sf.get("public_inputs", []) if sf else [],
            "derived_symbols": sf.get("derived_symbols", []) if sf else [],
            "external_state": sf.get("external_state", []) if sf else [],
            "domains": sol.get("domains", {}) if sol else {},
            "solver": sol.get("solver", "?") if sol else "?",
            "solver_status": (
                (sol.get("solver_status") or sol.get("status") or "?") if sol else "?"
            ),
            "classification": sol.get("classification", "?") if sol else "?",
            "verdict_status": ver.get("status", "?") if ver else "?",
            "confirmed": ver.get("confirmed", False) if ver else False,
            "reified": rei.get("reified", False) if rei else False,
            "test_paths": test_paths,
            "tests_passed": rei.get("passed", False) if rei else False,
        }
        records.append(rec)
    return records


def main():
    print("Loading census...")
    census = load_census()
    print(f"  {len(census)} entries")

    print("Loading phase frags...")
    slices = load_phase("Slice")
    solves = load_phase("Solve")
    verifies = load_phase("Verify")
    reifies = load_phase("Reify")
    print(
        f"  Slice={len(slices)} Solve={len(solves)} Verify={len(verifies)} Reify={len(reifies)}"
    )

    all_bids = (
        set(slices.keys())
        | set(solves.keys())
        | set(verifies.keys())
        | set(reifies.keys())
    )
    print(f"  Total inventoried branch_ids: {len(all_bids)}")

    # 1. branch_parameter_hypergraph.json
    print("Building hypergraph...")
    hypergraph, mapped = build_hypergraph(census, slices, all_bids)
    with open(OUT + "/branch_parameter_hypergraph.json", "w") as f:
        json.dump(hypergraph, f, indent=2)
    print(f"  OK: {len(hypergraph['nodes'])} nodes, {mapped} with public_inputs")

    # 2. domain_model.json
    print("Building domain model...")
    domains = build_domain_model()
    with open(OUT + "/domain_model.json", "w") as f:
        json.dump(domains, f, indent=2)
    print(f"  OK: {len(domains)} branch domains")

    # 3. characterization_catalog.jsonl
    print("Building characterization catalog...")
    catalog = build_catalog(census, slices, solves, verifies, reifies, all_bids)
    with open(OUT + "/characterization_catalog.jsonl", "w") as f:
        for rec in catalog:
            f.write(json.dumps(rec) + "\n")
    print(f"  OK: {len(catalog)} records")

    # 4. scorecard.json
    print("Building scorecard...")
    sat = unsat = unknown = confirmed = 0
    for ver in verifies.values():
        st = (ver.get("status") or "").upper()
        if st == "SAT":
            sat += 1
            if ver.get("confirmed"):
                confirmed += 1
        elif st == "UNSAT":
            unsat += 1
            if ver.get("confirmed"):
                confirmed += 1
        else:
            unknown += 1

    constraints = 0
    op_surface = 0
    ch_path = OUT + "/constraints_harvested.jsonl"
    if os.path.exists(ch_path):
        with open(ch_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("kind") == "constraint":
                        constraints += 1
                    elif r.get("kind") == "op-surface":
                        op_surface = len(r.get("supported_node_types", []))
                except Exception:
                    pass

    ca_model = _load(OUT + "/covering_array/model.json")
    ca_rows = ca_model.get("case_count", 0) if ca_model else 0
    ca_params = len(ca_model.get("parameters", {})) if ca_model else 0
    test_count = len(glob.glob(TESTDIR + "/test_*.py"))

    scorecard = {
        "branchesInventoried": len(all_bids),
        "totalBranches": len(census),
        "constraintsHarvested": constraints,
        "opSurfaceSize": op_surface,
        "publicInputsMapped": mapped,
        "satCount": sat,
        "unsatCount": unsat,
        "unknownCount": unknown,
        "witnessesConfirmed": confirmed,
        "coveringArrayParameters": ca_params,
        "coveringArrayRows": ca_rows,
        "testsReified": test_count,
        "note": (
            "Counts computed from ground-truth fragments. "
            "branchesInventoried = unique branch_ids across all phase frags. "
            "totalBranches = full census from branch_extractor. "
            "testsReified = test_pchaos_*.py files under CodegenResidue/."
        ),
    }
    with open(OUT + "/scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)
    print("  Scorecard summary:")
    for k, v in scorecard.items():
        if k != "note":
            print(f"    {k}: {v}")

    print("Done.")


if __name__ == "__main__":
    main()
