#!/usr/bin/env python3
"""
PICT cross-check: verify PICT output covers all 2-way pairs from stdlib cases.csv.
Usage: python3 crosscheck.py <cases_csv> <pict_tsv> <model_json> <receipt_json>
"""
import csv
import json
import sys
from itertools import combinations

# ─── token map: stdlib raw cell → PICT token ────────────────────────────────
# PICT tokens must be valid identifiers (no spaces, special chars)
# We use these deterministic tokens that match model.pict.
# csv.reader preserves surrounding quotes for JSON-string values (e.g. '"Module"').
TOKEN_MAP = {
    # nodeType
    '"Module"': "Module",
    '"Expr"': "Expr",
    '"Attribute"': "Attribute",
    '"Name"': "Name",
    # expressionStr
    '"a == 1"': "a_eq_1",
    '"a + b"': "a_plus_b",
    '"Program.Network.PORT >= MaxPort"': "Program_Network_PORT_ge_MaxPort",
    '"BenchmarkTaskSize > 0"': "BenchmarkTaskSize_gt_0",
    # Parameter
    '"Tensile.Configuration.Parameter"': "Tensile_Configuration_Parameter",
    '"<class Tensile.Configuration.Parameter>"': "class_Tensile_Configuration_Parameter",
    # bool/null (JSON literals stored without quotes in CSV)
    "false": "FALSE",
    "true": "TRUE",
    "null": "NULL",
    # RestoreLog
    '"/path/to/restore.log"': "path_to_restore_log",
    # __name__
    '"__main__"': "__main__",
    '"Tensile.Tensile"': "Tensile_Tensile",
    # args.platform / configPaths (numeric strings → as-is)
    "0": "0",
    "1": "1",
    "3": "3",
}


def normalize_stdlib_value(raw: str) -> str:
    """Strip extra CSV quoting from a stdlib cell value, then map to PICT token."""
    # The CSV uses "" for embedded quotes; after csv.reader they come back as
    # "Module" (with the outer quotes part of the value if not stripped).
    # csv.reader already un-doubles internal quotes, but the outer quotes from
    # the double-quoting scheme remain as literal chars in some cells.
    stripped = raw.strip()
    if stripped in TOKEN_MAP:
        return TOKEN_MAP[stripped]
    # Try stripping one layer of surrounding quotes (for strings stored as ""x"")
    if stripped.startswith('"') and stripped.endswith('"'):
        inner = stripped[1:-1]
        if inner in TOKEN_MAP:
            return TOKEN_MAP[inner]
    return stripped  # fallback: keep raw (will likely cause a mismatch)


def load_stdlib_pairs(cases_csv: str):
    """Return set of frozensets {(param_i, val_i), (param_j, val_j)} from stdlib csv."""
    pairs = set()
    rows = []
    with open(cases_csv, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            normalized = {}
            for h in headers:
                if h == "case_id":
                    continue
                normalized[h] = normalize_stdlib_value(row[h])
            rows.append(normalized)
    param_names = [h for h in headers if h != "case_id"]
    for row in rows:
        for p1, p2 in combinations(param_names, 2):
            pair = frozenset([(p1, row[p1]), (p2, row[p2])])
            pairs.add(pair)
    return pairs, rows, param_names


def load_pict_pairs(pict_tsv: str):
    """Return set of frozensets from PICT tab-separated output."""
    pairs = set()
    rows = []
    with open(pict_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames
        for row in reader:
            normalized = {h: row[h].strip() for h in headers}
            rows.append(normalized)
    param_names = list(headers)
    for row in rows:
        for p1, p2 in combinations(param_names, 2):
            pair = frozenset([(p1, row[p1]), (p2, row[p2])])
            pairs.add(pair)
    return pairs, rows, param_names


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: "
            + sys.argv[0]
            + " <cases_csv> <pict_tsv> <model_json> <receipt_json>"
        )
        sys.exit(1)

    cases_csv, pict_tsv, model_json, receipt_json = sys.argv[1:]

    print(f"Loading stdlib pairs from {cases_csv} ...")
    stdlib_pairs, stdlib_rows, stdlib_params = load_stdlib_pairs(cases_csv)
    print(
        f"  stdlib rows: {len(stdlib_rows)}, params: {len(stdlib_params)}, pairs: {len(stdlib_pairs)}"
    )

    print(f"Loading PICT pairs from {pict_tsv} ...")
    pict_pairs, pict_rows, pict_params = load_pict_pairs(pict_tsv)
    print(
        f"  PICT rows: {len(pict_rows)}, params: {len(pict_params)}, pairs: {len(pict_pairs)}"
    )

    # Verify param name alignment (PICT uses our token names, stdlib has original names)
    # Map stdlib param names to PICT param names (they should be the same set for our model)
    # PICT uses param names from model.pict header (which may differ for 'args.platform' → 'args_platform')
    PARAM_NAME_MAP = {
        "args.platform": "args_platform",
    }

    # Re-compute stdlib pairs using PICT param names
    stdlib_pairs_mapped = set()
    for pair in stdlib_pairs:
        items = list(pair)
        mapped = []
        for pname, pval in items:
            mapped_name = PARAM_NAME_MAP.get(pname, pname)
            mapped.append((mapped_name, pval))
        stdlib_pairs_mapped.add(frozenset(mapped))

    missing = stdlib_pairs_mapped - pict_pairs
    extra = pict_pairs - stdlib_pairs_mapped

    print(f"\nResults:")
    print(f"  stdlib pairs (mapped): {len(stdlib_pairs_mapped)}")
    print(f"  PICT pairs:            {len(pict_pairs)}")
    print(f"  missing from PICT:     {len(missing)}")
    print(f"  extra in PICT:         {len(extra)} (expected; PICT has more rows)")

    passed = len(missing) == 0

    if missing:
        print("\nMISSING PAIRS (stdlib pairs not covered by PICT):")
        for pair in sorted(str(p) for p in missing):
            print(f"  {pair}")
    else:
        print("\nAll stdlib 2-way pairs are covered by PICT output.")

    detail_str = (
        f"PICT superset check: stdlib_pairs={len(stdlib_pairs_mapped)}, "
        f"pict_pairs={len(pict_pairs)}, missing={len(missing)}, "
        f"stdlib_rows={len(stdlib_rows)}, pict_rows={len(pict_rows)}"
    )
    if missing:
        detail_str += f"; MISSING: {sorted(str(p) for p in list(missing)[:5])}"

    receipt = {
        "tool": "pict",
        "available": True,
        "version": None,  # filled by caller
        "smoke_ran": True,
        "real_input": model_json,
        "output_path": pict_tsv,
        "crosscheck": {
            "kind": "pict_superset_of_stdlib_pairs",
            "fallback_ref": cases_csv,
            "passed": passed,
            "detail": detail_str,
        },
        "receipt_path": receipt_json,
    }

    with open(receipt_json, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt written to {receipt_json}")
    print(f"PASSED: {passed}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
