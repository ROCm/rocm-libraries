# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LibraryLogic YAML filter and search tool."""

import argparse
import fnmatch
import glob as glob_module
import sys
from pathlib import Path
from typing import Iterator

try:
    import yaml
    _loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
except ImportError:
    yaml = None
    _loader = None

# source of truth: tensilelite/Tensile/Common/DataType.py:52
_dtypeAliases = {
    # numeric string aliases
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14,
    "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22,
    # name aliases
    "f32": 0, "f64": 1, "f16": 4, "i32": 6, "bf16": 7, "fp8_fnuz": 11,
    "fp8": 15, "bf8": 16, "fp4": 21, "e8": 22,
    # HIP _r aliases
    "f32_r": 0, "f64_r": 1, "f16_r": 4, "i32_r": 6, "bf16_r": 7,
    "f8_r": 15, "bf8_r": 16, "f4_r": 21,
}

# reverse map for display
_dtypeNames = {
    0: "f32", 1: "f64", 4: "f16", 6: "i32", 7: "bf16", 11: "fp8_fnuz",
    15: "fp8", 16: "bf8", 21: "fp4", 22: "e8",
}


def resolveAlias(alias):
    """Resolve a data-type alias string to its numeric code."""
    key = str(alias).lower()
    if key not in _dtypeAliases:
        raise ValueError(f"unknown data type alias: {alias!r}")
    return _dtypeAliases[key]


def _loadYaml(path):
    """Load a YAML file, returning the parsed object or None on failure."""
    try:
        with open(path, "r") as f:
            return yaml.load(f, Loader=_loader)
    except Exception:
        return None


def _normalizeDict(data, path):
    """Normalize a dict-form LibraryLogic YAML to a standard record."""
    arch = data.get("ArchitectureName", "")
    if isinstance(arch, dict):
        arch = arch.get("Architecture", "")
    return {
        "path": path,
        "scheduleName": str(data.get("ScheduleName", "")),
        "arch": str(arch),
        "problemType": data.get("ProblemType", {}),
        "solutions": data.get("Solutions", []),
    }


def _normalizeList(data, path):
    """Normalize a list-form LibraryLogic YAML to a standard record."""
    if len(data) < 6:
        return None
    arch = data[2]
    if isinstance(arch, dict):
        arch = arch.get("Architecture", "")
    problemType = data[4] if isinstance(data[4], dict) else {}
    solutions = data[5] if isinstance(data[5], list) else []
    return {
        "path": path,
        "scheduleName": str(data[1]) if data[1] is not None else "",
        "arch": str(arch),
        "problemType": problemType,
        "solutions": solutions,
    }


def loadLogicHeader(path):
    """Load and normalize a LibraryLogic YAML file.

    Returns a record dict with keys path, scheduleName, arch, problemType,
    solutions, or None on any parse failure.
    """
    if yaml is None:
        return None
    path = Path(path)
    data = _loadYaml(path)
    if data is None:
        return None
    if isinstance(data, dict):
        return _normalizeDict(data, path)
    if isinstance(data, list):
        return _normalizeList(data, path)
    return None


def _coerceToMatch(userVal, storedVal):
    """Coerce a user-provided string to match the stored value's type."""
    if isinstance(storedVal, bool):
        return userVal.lower() in ("true", "1")
    if isinstance(storedVal, int):
        try:
            return int(userVal)
        except ValueError:
            return str(userVal)
    return str(userVal)


def _inputTypePred(code, side):
    """Return a predicate checking DataTypeA and/or DataTypeB."""
    def pred(record):
        pt = record["problemType"]
        if side in ("a", "both") and pt.get("DataTypeA") == code:
            return True
        if side in ("b", "both") and pt.get("DataTypeB") == code:
            return True
        return False
    return pred


def _fieldPred(key, userVal):
    """Return a predicate checking a ProblemType field by key=value."""
    def pred(record):
        stored = record["problemType"].get(key)
        if stored is None:
            return False
        return stored == _coerceToMatch(userVal, stored)
    return pred


def _solutionFieldPred(key, userVal):
    """Return a predicate checking any solution for a field by key=value."""
    def pred(record):
        for sol in record["solutions"]:
            stored = sol.get(key)
            if stored is None:
                continue
            if stored == _coerceToMatch(userVal, stored):
                return True
        return False
    return pred


def _buildTypePredicates(args):
    """Build data-type filter predicates from args."""
    preds = []
    if args.input_type is not None:
        preds.append(_inputTypePred(resolveAlias(args.input_type), "both"))
    if args.input_type_a is not None:
        preds.append(_inputTypePred(resolveAlias(args.input_type_a), "a"))
    if args.input_type_b is not None:
        preds.append(_inputTypePred(resolveAlias(args.input_type_b), "b"))
    if args.dest_type is not None:
        code = resolveAlias(args.dest_type)
        preds.append(lambda rec, c=code: rec["problemType"].get("DestDataType") == c)
    return preds


def _buildRmsPredicates(args):
    """Build partial-RMS filter predicates from args."""
    preds = []
    if args.partial_rms:
        preds.append(lambda rec: any(
            rec["problemType"].get(k) for k in (
                "PartialRMSResidualAdd", "PartialRMSQuant", "PartialRMSStoreBf16D")))
    if args.residual_add:
        preds.append(lambda rec: bool(rec["problemType"].get("PartialRMSResidualAdd")))
    if args.residual_out:
        preds.append(lambda rec: bool(rec["problemType"].get("PartialRMSStoreBf16D")))
    return preds


def buildPredicates(args):
    """Build all filter predicates from parsed CLI args, combined with AND."""
    preds = []

    if args.subtile is not None:
        want = args.subtile
        preds.append(lambda rec, w=want:
            any(s.get("UseSubtileImpl") for s in rec["solutions"]) == w)

    preds.extend(_buildTypePredicates(args))
    preds.extend(_buildRmsPredicates(args))

    if args.dquant_type is not None:
        val = args.dquant_type.lower()
        preds.append(lambda rec, v=val:
            rec["problemType"].get("DQuantType", "None").lower() == v)

    if args.arch is not None:
        pat = args.arch
        preds.append(lambda rec, p=pat:
            fnmatch.fnmatch(rec["arch"], p) or p.lower() in rec["arch"].lower())

    if args.schedule_name is not None:
        pat = args.schedule_name
        preds.append(lambda rec, p=pat:
            fnmatch.fnmatch(rec["scheduleName"], p) or p.lower() in rec["scheduleName"].lower())

    for kv in (args.field or []):
        key, _, val = kv.partition("=")
        preds.append(_fieldPred(key, val))

    for kv in (args.solution_field or []):
        key, _, val = kv.partition("=")
        preds.append(_solutionFieldPred(key, val))

    return preds


def iterLogicFiles(inputs):
    """Yield Path objects for YAML files found in the given inputs.

    Inputs may be directories (recursed), explicit files, or glob patterns.
    Non-YAML results are skipped silently.
    """
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            yield from sorted(p.rglob("*.yaml"))
            continue
        if p.is_file():
            if p.suffix.lower() == ".yaml":
                yield p
            continue
        # Treat as glob pattern.
        for match in sorted(glob_module.glob(str(inp), recursive=True)):
            mp = Path(match)
            if mp.is_file() and mp.suffix.lower() == ".yaml":
                yield mp


def _printSummary(matches):
    """Print an aligned table of matching files."""
    if not matches:
        return
    pathWidth = max(len("path"), max(len(str(r["path"])) for r in matches))
    nameWidth = max(len("ScheduleName"), max(len(r["scheduleName"]) for r in matches))
    header = f"{'path':<{pathWidth}} | {'ScheduleName':<{nameWidth}} | Arch"
    print(header)
    print("-" * len(header))
    for r in matches:
        print(f"{str(r['path']):<{pathWidth}} | {r['scheduleName']:<{nameWidth}} | {r['arch']}")


def _buildArgParser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Filter and search LibraryLogic YAML files.")
    parser.add_argument("paths", nargs="+",
        help="Paths to search: files, directories, or glob patterns.")

    subtile_group = parser.add_mutually_exclusive_group()
    subtile_group.add_argument("--subtile", dest="subtile",
        action="store_const", const=True,
        help="Match files with at least one UseSubtileImpl solution.")
    subtile_group.add_argument("--no-subtile", dest="subtile",
        action="store_const", const=False,
        help="Match files with no UseSubtileImpl solution.")
    parser.set_defaults(subtile=None)

    parser.add_argument("--input-type", metavar="CODE",
        help="Match DataTypeA or DataTypeB (alias or numeric).")
    parser.add_argument("--input-type-a", metavar="CODE",
        help="Match DataTypeA exactly.")
    parser.add_argument("--input-type-b", metavar="CODE",
        help="Match DataTypeB exactly.")
    parser.add_argument("--dest-type", metavar="CODE",
        help="Match DestDataType.")
    parser.add_argument("--dquant-type", metavar="VALUE",
        help="Match DQuantType (case-insensitive: None, Tile, MXFP8).")
    parser.add_argument("--partial-rms", action="store_true",
        help="Match files with any PartialRMS field set.")
    parser.add_argument("--residual-add", action="store_true",
        help="Match files with PartialRMSResidualAdd set.")
    parser.add_argument("--residual-out", action="store_true",
        help="Match files with PartialRMSStoreBf16D set.")
    parser.add_argument("--arch", metavar="PATTERN",
        help="Match architecture name (fnmatch glob or substring).")
    parser.add_argument("--schedule-name", metavar="PATTERN",
        help="Match schedule name (fnmatch glob or substring).")
    parser.add_argument("--field", metavar="KEY=VALUE", action="append",
        help="Match a ProblemType field (repeatable).")
    parser.add_argument("--solution-field", metavar="KEY=VALUE", action="append",
        help="Match a solution field in any solution (repeatable).")

    parser.add_argument("--summary", action="store_true",
        help="Print an aligned table of path, ScheduleName, and Arch.")
    parser.add_argument("--count", action="store_true",
        help="Print only the count of matching files.")
    parser.add_argument("--verbose", action="store_true",
        help="Warn on stderr for unparseable files.")

    return parser


def main():
    """Entry point for the TensileLogicFilter CLI."""
    parser = _buildArgParser()
    args = parser.parse_args()

    try:
        preds = buildPredicates(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    matches = []
    for path in iterLogicFiles(args.paths):
        record = loadLogicHeader(path)
        if record is None:
            if args.verbose:
                print(f"warning: skipping unparseable file {path}", file=sys.stderr)
            continue
        if all(p(record) for p in preds):
            matches.append(record)

    if args.count:
        print(len(matches))
    elif args.summary:
        _printSummary(matches)
    else:
        for r in matches:
            print(r["path"])

    sys.exit(0 if matches else 1)
