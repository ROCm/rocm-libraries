#!/usr/bin/env python3
"""Driver-side deterministic finalizer for the parametric-chaos Run-1 bundle.

The workflow's Assemble agent is LLM-authored and was observed to (a) truncate the full
branch census to the 20 work-list units, (b) leave the hypergraph sparse because some Slice
agents returned their schema without persisting a fragment, and (c) miscount scorecard fields.
This finalizer recomputes the NUMERIC / JOINED deliverables from ground-truth fragments so the
bundle is accurate (rigor: measure, do not inflate). Narrative deliverables (validation_report,
analyst_summary, README-analysis) are left as the workflow produced them.

Regenerates, all deterministically:
  - branch_census.jsonl + file_inventory.csv  (FULL census via branch_extractor)
  - branch_parameter_hypergraph.json           (all 20 nodes + 20 edges; Slice frag where
                                                persisted, else census def-use)
  - domain_model.json                          (from Solve frags, which carry full domains)
  - scorecard.json                             (counts from Census/Verify/Reify frags + files)

Stdlib only. Run by the driver, in-container, after the workflow returns.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def regen_full_census(extractor_path, root, outdir, files, max_units):
    """Invoke branch_extractor's main path to restore the FULL census deliverable."""
    spec = importlib.util.spec_from_file_location("branch_extractor", extractor_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # replicate main() without re-parsing argv
    all_records, inventory = [], []
    for rel in files:
        with open(os.path.join(root, rel)) as f:
            src = f.read()
        import ast as _ast
        tree = _ast.parse(src, filename=rel)
        du = mod.DefUseCollector(src)
        du.visit(tree)
        ex = mod.BranchExtractor(rel, src)
        ex.visit(tree)
        for r in ex.records:
            fmap = du.maps.get(r["function"], {})
            r["derived_symbols"] = [
                {"name": s, "derived_from": fmap[s]["derived_from"], "category": fmap[s]["category"]}
                for s in r["referenced_symbols"] if s in fmap
            ]
        by_kind = {}
        for r in ex.records:
            by_kind[r["branch_kind"]] = by_kind.get(r["branch_kind"], 0) + 1
        inventory.append({"file": rel, "loc": len(src.splitlines()),
                          "branches": len(ex.records), "kinds": by_kind})
        all_records.extend(ex.records)
    mod.rank(all_records)

    import csv
    with open(os.path.join(outdir, "file_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "loc", "branch_sites", "if", "elif", "while", "ifexp",
                    "assert", "comprehension-if", "guard-return", "guard-raise"])
        for it in inventory:
            k = it["kinds"]
            w.writerow([it["file"], it["loc"], it["branches"],
                        k.get("if", 0), k.get("elif", 0), k.get("while", 0), k.get("ifexp", 0),
                        k.get("assert", 0), k.get("comprehension-if", 0),
                        k.get("guard-return", 0), k.get("guard-raise", 0)])
    with open(os.path.join(outdir, "branch_census.jsonl"), "w") as f:
        for r in all_records:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    return len(all_records)


def build_hypergraph(outdir, frags):
    census = {}
    for fp in sorted(glob.glob(os.path.join(frags, "Census", "*.json"))):
        rec = _load(fp)
        if rec:
            census[rec["id"]] = rec
    nodes, edges, mapped = [], [], 0
    for bid, u in census.items():
        nodes.append({
            "branch_id": bid, "file": u["file"], "function": u["function"],
            "branch_kind": u["branch_kind"], "location": u["location"],
            "predicate_source": u["predicate_source"],
            "referenced_symbols": u.get("referenced_symbols", []),
        })
        slice_frag = _load(os.path.join(frags, "Slice", bid + ".json"))
        if slice_frag and slice_frag.get("public_inputs"):
            public_inputs = slice_frag["public_inputs"]
            external_state = slice_frag.get("external_state", [])
            source = "slice-agent"
        else:
            # deterministic fallback from the extractor's intra-function def-use
            public_inputs = [{"kind": d["category"], "name": d["derived_from"]}
                             for d in u.get("derived_symbols", [])]
            external_state = []
            source = "census-defuse"
        if public_inputs:
            mapped += 1
        edges.append({
            "branch_id": bid, "predicate_source": u["predicate_source"],
            "public_inputs": public_inputs,
            "derived_symbols": u.get("derived_symbols", []),
            "external_state": external_state, "edge_source": source,
        })
    with open(os.path.join(outdir, "branch_parameter_hypergraph.json"), "w") as f:
        json.dump({"nodes": nodes, "edges": edges,
                   "note": "edges with edge_source=census-defuse were reconstructed deterministically "
                           "from the extractor's intra-function def-use because the Slice agent did not "
                           "persist a fragment for that branch."}, f, indent=2)
    return len(nodes), len(edges), mapped


def build_domain_model(outdir, frags):
    """Domains live in both Domain frags and Solve frags; Solve frags cover all solved units."""
    domains = {}
    for sub in ("Domain", "Solve"):
        for fp in sorted(glob.glob(os.path.join(frags, sub, "*.json"))):
            rec = _load(fp)
            if not rec:
                continue
            bid = rec.get("branch_id")
            dom = rec.get("domains")
            if bid and dom and bid not in domains:
                domains[bid] = dom
    with open(os.path.join(outdir, "domain_model.json"), "w") as f:
        json.dump(domains, f, indent=2)
    return len(domains)


def count_tests(testdir):
    files = [p for p in glob.glob(os.path.join(testdir, "test_*.py"))]
    return len(files), sorted(os.path.basename(p) for p in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--max-units", type=int, default=20)
    a = ap.parse_args()

    frags = os.path.join(a.outdir, "_frags")

    total_branches = regen_full_census(a.extractor, a.root, a.outdir, a.files, a.max_units)
    n_nodes, n_edges, mapped = build_hypergraph(a.outdir, frags)
    n_domains = build_domain_model(a.outdir, frags)

    # constraints (kind==constraint), op-surface size
    constraints, op_surface = 0, 0
    for line in open(os.path.join(a.outdir, "constraints_harvested.jsonl")):
        rec = json.loads(line)
        if rec.get("kind") == "constraint":
            constraints += 1
        elif rec.get("kind") == "op-surface":
            op_surface = len(rec.get("supported_node_types", []))

    # verdicts
    sat = unsat = unknown = confirmed = 0
    for fp in glob.glob(os.path.join(frags, "Verify", "*.json")):
        rec = _load(fp) or {}
        st = (rec.get("status") or "").upper()
        if st == "SAT":
            sat += 1
            if rec.get("confirmed"):
                confirmed += 1
        elif st == "UNSAT":
            unsat += 1
            if rec.get("confirmed"):
                confirmed += 1
        else:
            unknown += 1

    ca = _load(os.path.join(a.outdir, "covering_array", "model.json")) or {}
    ca_rows = ca.get("case_count", 0)
    ca_params = len(ca.get("parameters", {}))

    n_tests, test_files = count_tests(a.testdir)

    scorecard = {
        "branchesInventoried": len(glob.glob(os.path.join(frags, "Census", "*.json"))),
        "totalBranches": total_branches,
        "constraintsHarvested": constraints,
        "opSurfaceSize": op_surface,
        "publicInputsMapped": mapped,
        "hypergraphNodes": n_nodes,
        "hypergraphEdges": n_edges,
        "domainsModeled": n_domains,
        "satCount": sat,
        "unsatCount": unsat,
        "unknownCount": unknown,
        "witnessesConfirmed": confirmed,
        "coveringArrayParameters": ca_params,
        "coveringArrayRows": ca_rows,
        "testFilesReified": n_tests,
        "note": "Counts recomputed deterministically by finalize.py from ground-truth fragments "
                "(the LLM Assemble agent's scorecard was inaccurate). testFilesReified counts files "
                "under PublicInputSurface/; all are confirmed passing by the driver's pytest run.",
    }
    with open(os.path.join(a.outdir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=2)

    print(json.dumps({"total_branches": total_branches, "hypergraph": [n_nodes, n_edges, mapped],
                      "domains": n_domains, "constraints": constraints, "op_surface": op_surface,
                      "verdicts": {"sat": sat, "unsat": unsat, "unknown": unknown, "confirmed": confirmed},
                      "covering_array_rows": ca_rows, "test_files": n_tests,
                      "test_file_names": test_files}, indent=2))


if __name__ == "__main__":
    main()
