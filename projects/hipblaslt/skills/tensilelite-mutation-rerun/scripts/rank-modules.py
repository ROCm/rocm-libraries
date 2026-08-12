#!/usr/bin/env python3
"""rank-modules.py - Issue 17: Phase-0-global ranking + disposition.

Read-only analysis. Implements the plan's ranking formula and emits the
disposition table covering every characterization dir. It does NOT run mutation
testing and does NOT edit source.

Ranking formula (PLAN-MUTATION-COMPLETION.md), all inputs min-max normalized to
[0,1] across candidates (NO log10(LOC); NO subtraction of no_test_fraction):

    score = 0.40*importers_norm + 0.25*cyclomatic_norm
          + 0.15*LOC_norm       + 0.20*no_test_fraction_norm

Metric availability in THIS environment (read-only, no container/tools):
  - LOC              : COMPUTED (line count of the module file[s]).
  - importers        : COMPUTED (grep-style import in-degree over the Tensile tree).
  - cyclomatic       : PENDING  (needs `lizard`; not installed here).
  - no_test_fraction : PENDING  (needs `pytest --cov=<module>` in `tl-mut`).
Because 2 of 4 inputs are PENDING, per-module `score` is PENDING (not faked). The
formula + normalization are implemented and unit-tested (rank-modules-selftest.py);
fill the two pending inputs and re-run to publish the computed order.

Every discovered characterization dir gets exactly one disposition:
certified | scheduled | deferred-high-value | deferred-coverage-gap | out-of-scope
grounded in the plan; dirs the plan has not explicitly placed default to
deferred-high-value marked PROVISIONAL (pending the ranking above).

Python 3.8 compatible.
"""

import argparse
import os
import sys

W_IMPORTERS = 0.40
W_CYCLOMATIC = 0.25
W_LOC = 0.15
W_NO_TEST = 0.20

SRC_REL = "projects/hipblaslt/tensilelite"
CHAR_REL = "Tensile/Tests/unit/characterization"


# --------------------------------------------------------------- formula
def minmax_normalize(values):
    """Min-max normalize to [0,1]. If all values equal (zero range), returns 0.0
    for every element (no discriminating signal), documented behavior."""
    vals = list(values)
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def score(importers_n, cyclomatic_n, loc_n, no_test_n):
    """Weighted score from already-normalized [0,1] inputs."""
    return (
        W_IMPORTERS * importers_n
        + W_CYCLOMATIC * cyclomatic_n
        + W_LOC * loc_n
        + W_NO_TEST * no_test_n
    )


# --------------------------------------------------------------- disposition map (plan-grounded)
# dir -> (disposition, module_paths(list, for LOC/importers or []), reason, evidence)
CERTIFIED = "certified"
SCHEDULED = "scheduled"
DEF_HV = "deferred-high-value"
DEF_CG = "deferred-coverage-gap"
OOS = "out-of-scope"

DISPOSITIONS = {
    "CommonUtilities": (
        CERTIFIED,
        ["Tensile/Common/Utilities.py"],
        "slice-1 certified (survivors triaged, fresh-rerun certified)",
        "PILOT-BASELINE.md; PLAN slice 1",
    ),
    "TensileLogic": (
        CERTIFIED,
        [
            "Tensile/TensileLogic/ValidChipId.py",
            "Tensile/TensileLogic/ValidMatrixInstruction.py",
            "Tensile/TensileLogic/ValidWorkGroup.py",
            "Tensile/TensileLogic/ValidWorkGroupMappingXCC.py",
        ],
        "slice-1 certified for the 4 TensileLogic validators ONLY; the plan defers the REMAINING TensileLogic modules (deferred-high-value, ~85 char tests) - char-dir granularity cannot split them",
        "PILOT-BASELINE.md; PLAN slice 1 + deferred list ('remaining TensileLogic (85)')",
    ),
    "LibraryIO": (
        SCHEDULED,
        ["Tensile/LibraryIO.py"],
        "slice 2 (serialization/round-trip)",
        "PLAN slice table row 2",
    ),
    "ValidParameters": (
        SCHEDULED,
        ["Tensile/Common/ValidParameters.py"],
        "slice 3 (parameter-validation gate)",
        "PLAN slice table row 3",
    ),
    "Configuration": (
        SCHEDULED,
        ["Tensile/Configuration.py"],
        "slice 4 (config resolution)",
        "PLAN slice table row 4",
    ),
    "Naming": (
        SCHEDULED,
        ["Tensile/SolutionStructs/Naming.py"],
        "slice 5 (kernel naming contracts)",
        "PLAN slice table row 5",
    ),
    "SolutionStructsUtils": (
        SCHEDULED,
        [
            "Tensile/SolutionStructs/Utilities.py",
            "Tensile/SolutionStructs/LdsPadding.py",
        ],
        "slice 6 (grouped: Utilities+LdsPadding)",
        "PLAN slice table row 6",
    ),
    "CustomYamlLoader": (
        SCHEDULED,
        ["Tensile/CustomYamlLoader.py"],
        "slice 7 (leaf I/O)",
        "PLAN slice table row 7",
    ),
    "SolutionClass": (
        SCHEDULED,
        ["Tensile/SolutionStructs/Solution.py"],
        "slice 10 (Solution.py covering dir; sub-sliced)",
        "PLAN slice table row 10",
    ),
    "SolutionDerivationSweep": (
        SCHEDULED,
        ["Tensile/SolutionStructs/Solution.py"],
        "slice 10 (Solution.py covering dir; sub-sliced)",
        "PLAN slice table row 10",
    ),
    "Activation": (
        DEF_HV,
        ["Tensile/Activation.py"],
        "deferred high-value next tranche (~156 char tests; plan number is a def-test count, NOT LOC)",
        "PLAN deferred list",
    ),
    "TensileCreateLibraryRun": (
        DEF_HV,
        [],
        "deferred high-value next tranche (~90 char tests)",
        "PLAN deferred list",
    ),
    "LibraryLogic": (
        DEF_HV,
        [],
        "deferred high-value next tranche (~56 char tests)",
        "PLAN deferred list",
    ),
    "ProblemType": (
        DEF_HV,
        [],
        "deferred high-value next tranche (~52 char tests); NOTE: this dir imports SolutionStructs/Problem.py (slice 9 target) - its scheduled-vs-deferred status must be resolved in Phase-0 covering-set analysis for slice 9",
        "PLAN deferred list + slice 9 (Problem.py)",
    ),
    "Validators": (
        DEF_HV,
        [],
        "deferred high-value next tranche (~41 char tests)",
        "PLAN deferred list",
    ),
    "SolutionArms": (
        DEF_HV,
        [],
        "deferred high-value next tranche (~38 char tests)",
        "PLAN deferred list",
    ),
    "CodegenResidue": (
        DEF_HV,
        [],
        "deferred high-value next tranche (_codegen family, ~242 char tests)",
        "PLAN deferred list (_codegen)",
    ),
    "_codegen": (
        DEF_HV,
        [],
        "deferred high-value next tranche (~242 char tests)",
        "PLAN deferred list",
    ),
}

DEFAULT_DISP = DEF_HV
DEFAULT_REASON = "PROVISIONAL: in-scope universe, not explicitly placed by the plan; scheduled/deferred cut pending Phase-0 ranking (cyclomatic + no_test_fraction)"
DEFAULT_EVIDENCE = (
    "PLAN: ranking will draw the scheduled/deferred cut once metrics are computed"
)


# --------------------------------------------------------------- metrics (read-only)
def _root():
    here = os.path.dirname(os.path.abspath(__file__))
    # committed at <repo>/projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/;
    # the repo root is that in-tree suffix stripped off.
    suffix = os.path.join(SRC_REL, "Tensile", "Tests", "unit", "mutation")
    if here.endswith(suffix):
        return here[: -(len(suffix) + 1)]
    return os.path.abspath(os.path.join(here, *([".."] * 7)))


def discover_char_dirs(root):
    base = os.path.join(root, SRC_REL, CHAR_REL)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def loc_of(root, module_paths):
    total = 0
    found_any = False
    for m in module_paths:
        p = os.path.join(root, SRC_REL, m)
        if os.path.isfile(p):
            found_any = True
            with open(p, "r", errors="replace") as fh:
                total += sum(1 for _ in fh)
    return total if found_any else None


def _dotted(module_path):
    return (
        module_path[:-3].replace("/", ".")
        if module_path.endswith(".py")
        else module_path.replace("/", ".")
    )


def importers_of(root, module_paths, py_files):
    """Count .py files under Tensile/ that import any of the module_paths (by dotted
    path). Uses a single pre-read file list. None if module_paths is empty."""
    if not module_paths:
        return None
    dotted = [_dotted(m) for m in module_paths]
    count = 0
    for text in py_files:
        if any(d in text for d in dotted):
            count += 1
    return count


def _load_py_files(root):
    base = os.path.join(root, SRC_REL, "Tensile")
    out = []
    for dirpath, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".py"):
                try:
                    with open(os.path.join(dirpath, f), "r", errors="replace") as fh:
                        out.append(fh.read())
                except OSError:
                    pass
    return out


# --------------------------------------------------------------- emit
def _dir_has_python(root, d):
    base = os.path.join(root, SRC_REL, CHAR_REL, d)
    for _dp, _ds, files in os.walk(base):
        if any(f.endswith(".py") for f in files):
            return True
    return False


def build_rows(root, metrics=None):
    """metrics: optional dict {dir: {"cyclomatic": int, "no_test_fraction": float}}
    to fill the two PENDING inputs; when all four inputs are present for the
    candidate set, per-row `score` is computed (min-max normalized)."""
    metrics = metrics or {}
    dirs = discover_char_dirs(root)
    py_files = _load_py_files(root)
    rows = []
    for d in dirs:
        if d in DISPOSITIONS:
            disp, mods, reason, ev = DISPOSITIONS[d]
        elif not _dir_has_python(root, d):
            disp, mods, reason, ev = (
                OOS,
                [],
                "docs/meta only - no Python under the char dir (not a mutation target)",
                "read-only scan: no .py files in the dir",
            )
        else:
            disp, mods, reason, ev = (
                DEFAULT_DISP,
                [],
                DEFAULT_REASON,
                DEFAULT_EVIDENCE,
            )
        loc = loc_of(root, mods) if mods else None
        imp = importers_of(root, mods, py_files) if mods else None
        m = metrics.get(d, {})
        rows.append(
            {
                "dir": d,
                "disposition": disp,
                "reason": reason,
                "evidence": ev,
                "modules": mods,
                "loc": loc,
                "importers": imp,
                "cyclomatic": m.get("cyclomatic"),
                "no_test_fraction": m.get("no_test_fraction"),
                "score": None,
            }
        )
    _fill_scores(rows)
    return rows


def _fill_scores(rows):
    """Compute score for rows that have ALL FOUR raw inputs, min-max normalized
    across exactly that scored set. Rows missing any input stay PENDING."""
    scored = [
        r
        for r in rows
        if None
        not in (r["importers"], r["cyclomatic"], r["loc"], r["no_test_fraction"])
    ]
    if not scored:
        return
    imp_n = minmax_normalize([r["importers"] for r in scored])
    cyc_n = minmax_normalize([r["cyclomatic"] for r in scored])
    loc_n = minmax_normalize([r["loc"] for r in scored])
    nt_n = minmax_normalize([r["no_test_fraction"] for r in scored])
    for i, r in enumerate(scored):
        r["score"] = round(score(imp_n[i], cyc_n[i], loc_n[i], nt_n[i]), 4)


def _cell(v):
    return "PENDING" if v is None else str(v)


def render(rows):
    L = []
    L.append("# Phase-0-global ranking + disposition\n")
    L.append(
        "Read-only Phase-0 analysis (generated by `rank-modules.py`). No mutation"
    )
    L.append("testing was run; no production source edited.\n")
    L.append("## Ranking formula\n")
    L.append(
        "All raw inputs are **min-max normalized to [0,1]** across candidate modules"
    )
    L.append("(NO `log10(LOC)`; NO subtraction of `no_test_fraction`):\n")
    L.append("```")
    L.append("score = 0.40*importers_norm + 0.25*cyclomatic_norm")
    L.append("      + 0.15*LOC_norm       + 0.20*no_test_fraction_norm")
    L.append("```")
    L.append("")
    L.append("Inputs: **importers**, **cyclomatic**, **LOC**, **no_test_fraction**.\n")
    L.append("Metric availability (this read-only environment):")
    L.append("- LOC: COMPUTED (line count of the module file[s]).")
    L.append("- importers: COMPUTED as a **textual grep FLOOR** - count of `.py` files")
    L.append(
        "  under `Tensile/` (incl. tests) that mention the module's dotted path. This"
    )
    L.append("  over-counts (test files, docstrings, mock strings) and under-counts")
    L.append("  `from <pkg> import <name>` forms; it is a floor/proxy, NOT a true")
    L.append("  production import-graph in-degree (use `pydeps`/`grimp` to refine).")
    L.append("- cyclomatic: **PENDING** - run `lizard` per module, e.g.")
    L.append(
        "  `docker exec -w /work/%s tl-mut lizard <module.py>` (CCN sum)." % SRC_REL
    )
    L.append("- no_test_fraction: **PENDING** - `(1 - line_coverage)` from")
    L.append(
        "  `docker exec -w /work/%s tl-mut pytest --cov=<module> --cov-report=term-missing <covering tests>`."
        % SRC_REL
    )
    L.append("")
    L.append("Because cyclomatic + no_test_fraction are PENDING, per-module `score` is")
    L.append(
        "**PENDING** (not faked). Fill them and re-run `rank-modules.py` to publish the"
    )
    L.append(
        "computed order. The **slice-2..10 order is therefore interim/hand-ranked**"
    )
    L.append("(from the plan) and marked pending metric completion.\n")
    L.append(
        "## Disposition table (every characterization dir; exactly one disposition)\n"
    )
    L.append(
        "| char dir | disposition | LOC | importers | cyclomatic | no_test_frac | score | reason | evidence |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r["dir"],
                r["disposition"],
                _cell(r["loc"]),
                _cell(r["importers"]),
                _cell(r["cyclomatic"]),
                _cell(r["no_test_fraction"]),
                _cell(r["score"]),
                r["reason"],
                r["evidence"],
            )
        )
    L.append("")
    # disposition counts
    counts = {}
    for r in rows:
        counts[r["disposition"]] = counts.get(r["disposition"], 0) + 1
    L.append("## Disposition counts\n")
    for k in (CERTIFIED, SCHEDULED, DEF_HV, DEF_CG, OOS):
        L.append("- %s: %d" % (k, counts.get(k, 0)))
    L.append("")
    L.append("Total char dirs: %d\n" % len(rows))
    L.append("Notes:")
    L.append(
        "- `deferred-coverage-gap` is assigned only from measured `no_test_fraction`"
    )
    L.append("  (PENDING), so no dir is placed there yet - it is defined and reserved.")
    L.append(
        "- Dirs the plan did not explicitly place, that DO contain Python tests, are"
    )
    L.append(
        "  `deferred-high-value` marked PROVISIONAL until the ranking is computed (the"
    )
    L.append("  'high-value' label is the plan's tranche name, not a computed score).")
    L.append("- Dirs with NO Python (docs/meta only, e.g. `adr`) are `out-of-scope`.")
    L.append(
        "- Slices **8 (`BenchmarkSplitter.py`)** and **9 (`Problem.py`)** are Phase-2"
    )
    L.append("  MODULE slices without a dedicated characterization dir (slice 8 uses")
    L.append(
        "  `test_BenchmarkSplitter.py`; slice 9's covering set incl. the `ProblemType`"
    )
    L.append(
        "  dir is derived in Phase 0), so they do NOT appear as `scheduled` char-dir"
    )
    L.append("  rows - tracked at module granularity in the plan slice table instead.")
    L.append(
        "- The deferred-list counts in reasons (e.g. Activation ~156) are the plan's"
    )
    L.append(
        "  def-TEST counts, NOT LOC (the LOC column is the module's real line count)."
    )
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Phase-0 ranking + disposition generator (read-only)"
    )
    p.add_argument("--out", default="-", help="output markdown path ('-' = stdout)")
    p.add_argument(
        "--metrics",
        default=None,
        help="optional JSON {dir: {cyclomatic, no_test_fraction}} to fill the "
        "PENDING inputs and compute scores",
    )
    args = p.parse_args(argv)
    root = _root()
    metrics = None
    if args.metrics:
        import json

        with open(args.metrics, "r") as fh:
            metrics = json.load(fh)
    rows = build_rows(root, metrics=metrics)
    text = render(rows)
    if args.out and args.out != "-":
        with open(args.out, "w") as fh:
            fh.write(text)
        sys.stderr.write("wrote %s (%d char dirs)\n" % (args.out, len(rows)))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
