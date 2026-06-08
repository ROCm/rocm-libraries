#!/usr/bin/env python3
"""
Assemble Run-1 parametric-chaos deliverables from all six fragment directories.
Run in container: python3 /work/work/tensilelite-characterization/parametric-chaos/assemble_run1.py
"""
import json, glob, os, sys, csv
from pathlib import Path
from datetime import datetime

FRAGS = (
    "/work/work/tensilelite-characterization/parametric-chaos/PublicInputSurface/_frags"
)
OUT = "/work/work/tensilelite-characterization/parametric-chaos/PublicInputSurface"

os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/covering_array", exist_ok=True)

# ── 1. Load all fragment types ────────────────────────────────────────────────


def load_frags(subdir, id_key="branch_id"):
    result = {}
    for f in sorted(glob.glob(f"{FRAGS}/{subdir}/*.json")):
        with open(f) as fh:
            d = json.load(fh)
        bid = d.get(id_key) or d.get("branch_id") or d.get("id")
        if bid:
            result[bid] = d
    return result


census_data = load_frags("Census", id_key="id")
slice_data = load_frags("Slice")
domain_data = load_frags("Domain")
solve_data = load_frags("Solve")
verify_data = load_frags("Verify")
reify_data = load_frags("Reify")

all_branch_ids = sorted(
    set(
        list(census_data)
        + list(slice_data)
        + list(domain_data)
        + list(solve_data)
        + list(verify_data)
        + list(reify_data)
    )
)

print(
    f"Census:{len(census_data)} Slice:{len(slice_data)} Domain:{len(domain_data)} "
    f"Solve:{len(solve_data)} Verify:{len(verify_data)} Reify:{len(reify_data)} "
    f"Total:{len(all_branch_ids)}",
    file=sys.stderr,
)

# ── 2. Helper functions ───────────────────────────────────────────────────────


def get_census(bid):
    return census_data.get(bid, {})


def get_file_line(bid):
    c = get_census(bid)
    loc = c.get("location", {})
    f = c.get(
        "file", slice_data.get(bid, {}).get("source_location", {}).get("file", "?")
    )
    l = loc.get(
        "line", slice_data.get(bid, {}).get("source_location", {}).get("line", "?")
    )
    return f, l


def get_public_inputs(bid):
    return slice_data.get(bid, {}).get("public_inputs", [])


def get_external_state(bid):
    return slice_data.get(bid, {}).get("external_state", [])


def get_predicate_text(bid):
    c = get_census(bid)
    if c.get("predicate_source"):
        return c["predicate_source"]
    sl = slice_data.get(bid, {})
    if "predicate_text" in sl:
        return sl["predicate_text"]
    pn = sl.get("predicate_normalized", {})
    if isinstance(pn, dict):
        return pn.get("raw_source") or pn.get("semantics", "")
    return ""


def get_solver_status(bid):
    sv = solve_data.get(bid, {})
    vf = verify_data.get(bid, {})
    # Prefer Verify status (it may have downgraded)
    v_status = vf.get("status", "")
    if v_status:
        return v_status.upper()
    sv_status = sv.get("solver_status", "")
    if sv_status:
        return sv_status.upper().split("-")[0]  # sat-bounded -> SAT
    return "UNKNOWN"


def is_confirmed(bid):
    vf = verify_data.get(bid, {})
    return bool(vf.get("confirmed", False))


def get_classification(bid):
    sv = solve_data.get(bid, {})
    cls = sv.get("classification", "")
    if cls:
        return cls
    # Derive from public inputs
    pubs = get_public_inputs(bid)
    ext = get_external_state(bid)
    if not pubs and not ext:
        return "derived-local"
    kinds = {p.get("kind", "?") for p in pubs}
    if ext:
        return "runtime-dependent"
    if "os" in kinds or "filesystem" in kinds:
        return "runtime-dependent"
    if "yaml" in kinds:
        return "fully-static"
    if "cli" in kinds:
        return "fully-static"
    return "runtime-dependent"


# ── 3. branch_census.jsonl ────────────────────────────────────────────────────

census_records = []
for bid in all_branch_ids:
    f, l = get_file_line(bid)
    c = get_census(bid)
    re = reify_data.get(bid, {})
    pubs = get_public_inputs(bid)
    rec = {
        "branch_id": bid,
        "file": f,
        "line": l,
        "function": c.get("function", "?"),
        "branch_kind": c.get("branch_kind", "?"),
        "predicate_source": get_predicate_text(bid),
        "classification": get_classification(bid),
        "public_input_count": len(pubs),
        "solver_status": get_solver_status(bid),
        "confirmed": is_confirmed(bid),
        "reified": re.get("reified", False),
        "test_passed": re.get("passed", False),
        "test_count": re.get("test_count", 0) or len(re.get("test_paths", [])),
        "rank": c.get("rank", None),
    }
    census_records.append(rec)

path = f"{OUT}/branch_census.jsonl"
with open(path, "w") as fh:
    for r in census_records:
        fh.write(json.dumps(r) + "\n")
print(f"Written branch_census.jsonl: {len(census_records)} records")

# ── 4. constraints_harvested.jsonl ────────────────────────────────────────────

constraints_records = []
for bid in all_branch_ids:
    f, l = get_file_line(bid)
    pubs = get_public_inputs(bid)
    ext = get_external_state(bid)
    for pub in pubs:
        constraints_records.append(
            {
                "branch_id": bid,
                "constraint_kind": "public_input",
                "input_name": pub.get("name", "?"),
                "input_kind": pub.get("kind", "?"),
                "input_type": pub.get("type", pub.get("python_type", "?")),
                "source_file": f,
                "source_line": l,
                "derived_from": pub.get("derived_from", ""),
            }
        )
    for es in ext:
        constraints_records.append(
            {
                "branch_id": bid,
                "constraint_kind": "external_state",
                "input_name": str(es)[:120],
                "input_kind": "external",
                "input_type": "runtime",
                "source_file": f,
                "source_line": l,
                "derived_from": "",
            }
        )
    # Also harvest solve-phase constraints
    sv = solve_data.get(bid, {})
    enc = sv.get("encoding", {})
    for vname, vspec in (enc.get("vars", {}) if isinstance(enc, dict) else {}).items():
        if not any(
            r["input_name"] == vname and r["branch_id"] == bid
            for r in constraints_records
        ):
            constraints_records.append(
                {
                    "branch_id": bid,
                    "constraint_kind": "solver_var",
                    "input_name": vname,
                    "input_kind": vspec.get("z3_sort", "?"),
                    "input_type": vspec.get("z3_sort", "?"),
                    "source_file": f,
                    "source_line": l,
                    "derived_from": vspec.get("origin", vspec.get("meaning", "")),
                }
            )

path = f"{OUT}/constraints_harvested.jsonl"
with open(path, "w") as fh:
    for r in constraints_records:
        fh.write(json.dumps(r) + "\n")
print(f"Written constraints_harvested.jsonl: {len(constraints_records)} records")

# ── 5. branch_parameter_hypergraph.json ───────────────────────────────────────

nodes = []
for bid in all_branch_ids:
    f, l = get_file_line(bid)
    pubs = get_public_inputs(bid)
    nodes.append(
        {
            "branch_id": bid,
            "file": f,
            "line": l,
            "predicate": get_predicate_text(bid),
            "classification": get_classification(bid),
            "solver_status": get_solver_status(bid),
            "confirmed": is_confirmed(bid),
            "public_input_names": [p.get("name", "?") for p in pubs],
        }
    )

edges = []
for bid in all_branch_ids:
    pubs = get_public_inputs(bid)
    if pubs:
        edges.append(
            {
                "branch_id": bid,
                "public_inputs": [
                    {
                        "name": p.get("name", "?"),
                        "kind": p.get("kind", "?"),
                        "type": p.get("type", p.get("python_type", "?")),
                    }
                    for p in pubs
                ],
            }
        )

path = f"{OUT}/branch_parameter_hypergraph.json"
with open(path, "w") as fh:
    json.dump({"nodes": nodes, "edges": edges}, fh, indent=2)
print(
    f"Written branch_parameter_hypergraph.json: {len(nodes)} nodes, {len(edges)} edges"
)

# ── 6. domain_model.json ─────────────────────────────────────────────────────

domain_model = {}
for bid, dm in domain_data.items():
    domain_model[bid] = dm.get("domains", {})

path = f"{OUT}/domain_model.json"
with open(path, "w") as fh:
    json.dump(domain_model, fh, indent=2)
print(f"Written domain_model.json: {len(domain_model)} branches")

# ── 7. characterization_catalog.jsonl ────────────────────────────────────────

catalog_records = []
for bid in all_branch_ids:
    f, l = get_file_line(bid)
    sl = slice_data.get(bid, {})
    dm = domain_data.get(bid, {})
    sv = solve_data.get(bid, {})
    vf = verify_data.get(bid, {})
    re = reify_data.get(bid, {})
    pubs = get_public_inputs(bid)
    test_paths = re.get("test_paths", [])
    catalog_records.append(
        {
            "branch_id": bid,
            # census
            "file": f,
            "line": l,
            "function": get_census(bid).get("function", "?"),
            "branch_kind": get_census(bid).get("branch_kind", "?"),
            "predicate_source": get_predicate_text(bid),
            "classification": get_classification(bid),
            # slice
            "public_inputs": pubs,
            "external_state": get_external_state(bid),
            # domain
            "domains": dm.get("domains", {}),
            # solver
            "solver_status": get_solver_status(bid),
            "solver_classification": sv.get("classification", ""),
            "z3_predicate": (
                sv.get("encoding", {}).get("z3_predicate", "")
                if isinstance(sv.get("encoding"), dict)
                else ""
            ),
            "true_examples": sv.get("true_examples", []),
            "false_examples": sv.get("false_examples", []),
            "crosshair_ran": (
                sv.get("crosshair", {}).get("ran", False)
                if isinstance(sv.get("crosshair"), dict)
                else False
            ),
            # verify
            "confirmed": is_confirmed(bid),
            "downgraded_to": vf.get("downgraded_to"),
            "verify_method": vf.get("method", "")[:200] if vf.get("method") else "",
            # reify
            "reified": re.get("reified", False),
            "test_paths": test_paths,
            "test_passed": re.get("passed", False),
            "test_count": re.get("test_count", 0) or len(test_paths),
        }
    )

path = f"{OUT}/characterization_catalog.jsonl"
with open(path, "w") as fh:
    for r in catalog_records:
        fh.write(json.dumps(r) + "\n")
print(f"Written characterization_catalog.jsonl: {len(catalog_records)} records")

# ── 8. covering_array/model.json and cases.csv ───────────────────────────────

ca_branches = [
    bid for bid in all_branch_ids if bid in domain_data and get_public_inputs(bid)
]
ca_params = []
for bid in ca_branches:
    dm = domain_data.get(bid, {})
    for param_name, param_domain in dm.get("domains", {}).items():
        ca_params.append(
            {
                "branch_id": bid,
                "parameter": param_name,
                "domain": param_domain,
            }
        )

ca_model = {
    "run": 1,
    "strategy": "pairwise-2-way",
    "description": "One entry per (branch, parameter) pair from Domain fragments. Values bounded by fragment analysis.",
    "branches_with_domains": len(ca_branches),
    "parameters": ca_params,
}

path = f"{OUT}/covering_array/model.json"
with open(path, "w") as fh:
    json.dump(ca_model, fh, indent=2)
print(
    f"Written covering_array/model.json: {len(ca_params)} params across {len(ca_branches)} branches"
)

ca_cases = []
row_id = 1
for bid in ca_branches:
    dm = domain_data.get(bid, {})
    f, l = get_file_line(bid)
    for param_name, param_domain in dm.get("domains", {}).items():
        values = param_domain.get("values", [])
        for v in values[:4]:
            ca_cases.append(
                {
                    "row_id": row_id,
                    "branch_id": bid,
                    "source": f"{f}:{l}",
                    "parameter": param_name,
                    "value": str(v),
                    "classification": get_classification(bid),
                }
            )
            row_id += 1

path = f"{OUT}/covering_array/cases.csv"
with open(path, "w", newline="") as fh:
    if ca_cases:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "row_id",
                "branch_id",
                "source",
                "parameter",
                "value",
                "classification",
            ],
        )
        w.writeheader()
        w.writerows(ca_cases)
print(f"Written covering_array/cases.csv: {len(ca_cases)} rows")

# ── 9. Scorecard ─────────────────────────────────────────────────────────────

sat_count = sum(1 for r in catalog_records if r["solver_status"] == "SAT")
unsat_count = sum(1 for r in catalog_records if r["solver_status"] == "UNSAT")
unknown_count = sum(1 for r in catalog_records if r["solver_status"] == "UNKNOWN")
witnesses_confirmed = sum(1 for r in catalog_records if r["confirmed"])
tests_reified = sum(1 for r in catalog_records if r["reified"])
total_constraints = len(constraints_records)
public_inputs_mapped = len([bid for bid in all_branch_ids if get_public_inputs(bid)])

scorecard = {
    "branchesInventoried": len(all_branch_ids),
    "totalBranches": len(all_branch_ids),
    "constraintsHarvested": total_constraints,
    "publicInputsMapped": public_inputs_mapped,
    "satCount": sat_count,
    "unsatCount": unsat_count,
    "unknownCount": unknown_count,
    "witnessesConfirmed": witnesses_confirmed,
    "coveringArrayRows": len(ca_cases),
    "testsReified": tests_reified,
}

path = f"{OUT}/scorecard.json"
with open(path, "w") as fh:
    json.dump(scorecard, fh, indent=2)
print(f"Written scorecard.json: {scorecard}")

# ── 10. preflight.json ────────────────────────────────────────────────────────

preflight = {
    "run": 1,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "python_version": sys.version,
    "fragment_counts": {
        "census": len(census_data),
        "slice": len(slice_data),
        "domain": len(domain_data),
        "solve": len(solve_data),
        "verify": len(verify_data),
        "reify": len(reify_data),
    },
    "total_branch_ids": len(all_branch_ids),
    "deliverables": [
        "preflight.json",
        "file_inventory.csv",
        "branch_census.jsonl",
        "constraints_harvested.jsonl",
        "branch_parameter_hypergraph.json",
        "domain_model.json",
        "characterization_catalog.jsonl",
        "covering_array/model.json",
        "covering_array/cases.csv",
        "validation_report.md",
        "analyst_summary.md",
        "README-analysis.md",
        "scorecard.json",
    ],
    "status": "assembled",
}

path = f"{OUT}/preflight.json"
with open(path, "w") as fh:
    json.dump(preflight, fh, indent=2)
print(f"Written preflight.json")

# ── 11. file_inventory.csv ────────────────────────────────────────────────────

inventory_rows = []
for fname in sorted(os.listdir(OUT)):
    fpath = os.path.join(OUT, fname)
    if fname.startswith("_"):
        continue
    if os.path.isfile(fpath):
        inventory_rows.append(
            {"path": fname, "size_bytes": os.path.getsize(fpath), "type": "file"}
        )
    elif os.path.isdir(fpath):
        for sub in sorted(os.listdir(fpath)):
            sp = os.path.join(fpath, sub)
            if os.path.isfile(sp):
                inventory_rows.append(
                    {
                        "path": f"{fname}/{sub}",
                        "size_bytes": os.path.getsize(sp),
                        "type": "file",
                    }
                )

path = f"{OUT}/file_inventory.csv"
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "type"])
    w.writeheader()
    w.writerows(inventory_rows)
print(f"Written file_inventory.csv: {len(inventory_rows)} entries")

print("\nPhase 1 DONE (JSON/JSONL/CSV deliverables).")
print(
    f"Scorecard summary: SAT={sat_count} UNSAT={unsat_count} UNKNOWN={unknown_count} confirmed={witnesses_confirmed}"
)

# Save state for markdown phase
import pickle

with open("/tmp/run1_state.pkl", "wb") as fh:
    pickle.dump(
        {
            "all_branch_ids": all_branch_ids,
            "census_data": census_data,
            "slice_data": slice_data,
            "domain_data": domain_data,
            "solve_data": solve_data,
            "verify_data": verify_data,
            "reify_data": reify_data,
            "catalog_records": catalog_records,
            "scorecard": scorecard,
            "census_records": census_records,
            "ca_cases": ca_cases,
        },
        fh,
    )
print("State pickled to /tmp/run1_state.pkl")
