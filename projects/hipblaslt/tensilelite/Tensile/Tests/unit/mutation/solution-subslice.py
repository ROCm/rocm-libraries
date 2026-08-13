#!/usr/bin/env python3
"""Repeatable Solution.py mutation sub-slicing.

Provides a region model, validator, and slice-record emitter. It does NOT run
mutmut or edit production source. The emitted record is consumed by the
tensilelite-mutation-rerun skill.

The region model enforces these constraints:
  - mutmut 3.6.0 only_mutate is whole-file (path glob), so a region config's
    only_mutate is the WHOLE Solution.py; the region is realized later by
    # pragma: no mutate fencing, NOT by only_mutate.
  - pragma fencing is only safe when both fence boundaries are same-block-level
    siblings (verified live against the real PragmaVisitor); each region carries a
    same_level_fence flag, and a MANDATORY libcst dry-parse gate must run before any
    live mutmut run (emitted as a fence_gate hook in the region config).

Subcommands:
  validate    --regions <regions.json>
  emit-config --regions <regions.json> --region <id> --out <file>

Python 3.8 compatible (no tomllib, no match, no PEP 585 subscripted builtins).
"""

import argparse
import json
import sys

MODULE_DEFAULT = "Tensile/SolutionStructs/Solution.py"
SRC_REL_DEFAULT = "projects/hipblaslt/tensilelite"
CONTAINER_DEFAULT = "tl-mut"
# plan sizing floor (PLAN-MUTATION-COMPLETION.md): ~0.78 mutants/LOC, ~20% survive.
MUTANTS_PER_LOC_FLOOR = 0.78
SURVIVOR_RATE_FLOOR = 0.20


def _load(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _regions(doc):
    r = doc.get("regions")
    if not isinstance(r, list):
        raise ValueError("regions.json must have a 'regions' array")
    return r


def validate_doc(doc):
    """Return a list of error strings (empty == valid)."""
    errs = []
    if not isinstance(doc, dict):
        return ["regions.json must be a JSON object"]
    for key in ("module", "src_rel", "regions"):
        if key not in doc:
            errs.append("missing top-level key: %s" % key)
    try:
        regions = _regions(doc)
    except ValueError as e:
        return errs + [str(e)]
    if not regions:
        errs.append("regions array is empty")

    seen_ids = set()
    prev_end = 0
    prev_start = 0
    for i, reg in enumerate(regions):
        tag = "region[%d]" % i
        if not isinstance(reg, dict):
            errs.append("%s must be an object" % tag)
            continue
        rid = reg.get("id")
        if not rid or not isinstance(rid, str):
            errs.append("%s missing string id" % tag)
        else:
            tag = "region %s" % rid
            if rid in seen_ids:
                errs.append("%s duplicate id" % tag)
            seen_ids.add(rid)
        start = reg.get("start_line")
        end = reg.get("end_line")
        # start/end are integers
        if not isinstance(start, int) or isinstance(start, bool):
            errs.append("%s start_line must be an integer" % tag)
            continue
        if not isinstance(end, int) or isinstance(end, bool):
            errs.append("%s end_line must be an integer" % tag)
            continue
        if start < 1:
            errs.append("%s start_line must be >= 1" % tag)
        if end < start:
            errs.append("%s end_line (%d) < start_line (%d)" % (tag, end, start))
        # region LOC positive and consistent
        loc = reg.get("loc")
        expected_loc = end - start + 1
        if not isinstance(loc, int) or isinstance(loc, bool) or loc <= 0:
            errs.append("%s loc must be a positive integer" % tag)
        elif loc != expected_loc:
            errs.append("%s loc (%d) != end-start+1 (%d)" % (tag, loc, expected_loc))
        # same-level fence requirement represented
        if "same_level_fence" not in reg or not isinstance(
            reg["same_level_fence"], bool
        ):
            errs.append("%s must carry a boolean same_level_fence flag" % tag)
        # covering-test selection present (list); config generation needs it non-empty
        cts = reg.get("covering_test_selection")
        if not isinstance(cts, list):
            errs.append("%s covering_test_selection must be a list" % tag)
        elif not cts:
            errs.append(
                "%s covering_test_selection is empty (config emit would be invalid)"
                % tag
            )
        elif any((not isinstance(x, str)) or (not x.strip()) for x in cts):
            errs.append(
                "%s covering_test_selection has a non-string or empty member" % tag
            )
        # ordered by start_line and non-overlapping
        if i > 0:
            if start <= prev_start:
                errs.append(
                    "%s not ordered: start_line %d <= previous start_line %d"
                    % (tag, start, prev_start)
                )
            if start <= prev_end:
                errs.append(
                    "%s overlaps previous region (start %d <= previous end %d)"
                    % (tag, start, prev_end)
                )
        prev_start = start
        prev_end = end
        # config generation must be possible for this region
        try:
            build_region_config(doc, reg)
        except Exception as e:  # noqa: BLE001 - report as a validation error
            errs.append("%s config generation failed: %s" % (tag, e))
    return errs


def build_region_config(doc, reg):
    """Build a mutation-rerun slice record for one region.

    only_mutate is the WHOLE file (mutmut has no sub-file targeting); the region is
    realized by pragma fencing, recorded here as metadata + a mandatory fence gate.
    """
    module = doc.get("module", MODULE_DEFAULT)
    src_rel = doc.get("src_rel", SRC_REL_DEFAULT)
    container = doc.get("container", CONTAINER_DEFAULT)
    rid = reg["id"]
    start = reg["start_line"]
    end = reg["end_line"]
    cts = reg.get("covering_test_selection") or []
    if not cts:
        raise ValueError(
            "covering_test_selection is empty; cannot emit a valid test_selection"
        )
    bad_members = [x for x in cts if (not isinstance(x, str)) or (not x.strip())]
    if bad_members:
        raise ValueError(
            "covering_test_selection has non-string/empty members: %r" % bad_members
        )
    cfg = {
        "slice_id": "10-%s" % rid,
        "only_mutate": [module],
        "test_selection": list(cts),
        "out": "work/mutation/slices/10-solution/regions/%s"
        % rid,
        "src_rel": src_rel,
        "container": container,
        "group_by": "module_function",
        "test_file_owner": "one_file_per_function",
        # --- region + safety metadata consumed by the rerun workflow/fencer ---
        "region": {
            "id": rid,
            "start_line": start,
            "end_line": end,
            "loc": end - start + 1,
            "label": reg.get("label", ""),
            "scope": reg.get("scope", ""),
            "same_level_fence": reg.get("same_level_fence"),
        },
        "fence_gate": (
            "MANDATORY before any mutmut run: fence all-but-[%d..%d] at same-level "
            "statement boundaries, then parse with mutmut's PragmaVisitor and assert "
            "(a) no PragmaParseError and (b) no_mutate_lines == complement([%d..%d]). "
            "Only proceed if same_level_fence is true and the gate passes."
        )
        % (start, end, start, end),
        "source_safety": (
            "Single serial actor. Backup Solution.py, trap 'restore' on EXIT INT TERM, "
            "clear mutmut mutants/ between regions, and require "
            "`git -C %s diff --quiet -- %s` before the next region and before any commit."
        )
        % (src_rel, module),
    }
    if doc.get("src_commit"):
        cfg["src_commit"] = doc["src_commit"]
    return cfg


def _find_region(doc, rid):
    for reg in _regions(doc):
        if reg.get("id") == rid:
            return reg
    raise KeyError("region id not found: %s" % rid)


def cmd_validate(args):
    doc = _load(args.regions)
    errs = validate_doc(doc)
    if errs:
        sys.stderr.write("INVALID regions.json (%d error(s)):\n" % len(errs))
        for e in errs:
            sys.stderr.write("  - %s\n" % e)
        return 1
    regions = _regions(doc)
    print("OK: %d region(s) valid" % len(regions))
    unsafe = [r["id"] for r in regions if not r.get("same_level_fence")]
    if unsafe:
        print(
            "NOTE: %d region(s) NOT same-level-fenceable (need structural handling): %s"
            % (len(unsafe), ", ".join(unsafe))
        )
    return 0


def cmd_emit_config(args):
    doc = _load(args.regions)
    errs = validate_doc(doc)
    if errs:
        sys.stderr.write(
            "refusing to emit: regions.json is invalid (%d error(s)); run `validate` first:\n"
            % len(errs)
        )
        for e in errs:
            sys.stderr.write("  - %s\n" % e)
        return 1
    try:
        reg = _find_region(doc, args.region)
    except KeyError as e:
        sys.stderr.write("emit-config: %s\n" % e)
        return 1
    cfg = build_region_config(doc, reg)
    text = json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    if args.out and args.out != "-":
        with open(args.out, "w") as fh:
            fh.write(text)
        print(
            "wrote %s (region %s, lines %d..%d)"
            % (args.out, reg["id"], reg["start_line"], reg["end_line"])
        )
    else:
        sys.stdout.write(text)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Solution.py sub-slicing region model + config emitter"
    )
    sub = p.add_subparsers(dest="cmd")
    pv = sub.add_parser("validate", help="validate a regions.json")
    pv.add_argument("--regions", required=True)
    pe = sub.add_parser(
        "emit-config", help="emit a mutation-rerun slice record for one region"
    )
    pe.add_argument("--regions", required=True)
    pe.add_argument("--region", required=True)
    pe.add_argument("--out", default="-")
    args = p.parse_args(argv)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "emit-config":
        return cmd_emit_config(args)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
