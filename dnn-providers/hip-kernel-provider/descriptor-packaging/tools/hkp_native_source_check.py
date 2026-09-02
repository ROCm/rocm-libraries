"""CTest driver: cross-check a packed rocKE attention_dense pack's descriptor
JSON against the C++ source it dispatches into, via
`hipdnn_validate_descriptors --native-source`.

hipdnn_validate_descriptors is a pure JSON/regex/filesystem tool -- it loads
descriptors, registers no-op stub symbols, and pattern-matches the
`register<Name>Symbols` function in the given `.cpp` for its
`constexpr std::string_view` symbol constants. No HIP call, no device. This
script runs it and turns its JSON report into a pass/fail/skip verdict.

Exits 0 when every native-source check is clean, 1 when a symbol the C++
declares is not the symbol the descriptor JSON names (or the reverse), and 77
(ctest's SKIP_RETURN_CODE) when either half of the comparison is absent by
build configuration rather than by defect:

* the arch's pack was never staged into the build tree -- GPU_TARGETS controls
  which arches the rocKE producer lowers from kind: rocke (authored) to
  kind: kpack (loadable), and an arch outside it is never staged; or
* the pack's Native.cpp does not exist in this checkout -- the arch table in
  HkpPackaging.cmake is a fixed list, so it names packs that a given branch
  may not carry.

Both are the same fact in two places: there is no pair to cross-check, which is
not the same as a pair that disagrees. Reporting absence as FAIL would make the
test red on every branch that does not happen to ship every arch in the table.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SKIP = 77  # ctest SKIP_RETURN_CODE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", required=True, help="bare gfx arch this pack targets, e.g. gfx950"
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="the pack's staged descriptor directory",
    )
    parser.add_argument(
        "--validator",
        required=True,
        type=Path,
        help="path to the hipdnn_validate_descriptors binary",
    )
    parser.add_argument(
        "--expect-engine", required=True, help="engine name the pack must resolve to"
    )
    parser.add_argument(
        "--native-source", required=True, type=Path, help="the pack's Native.cpp"
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not args.root.is_dir():
        print(
            f"SKIP: nothing was packaged for this device ({args.arch}): {args.root} "
            "does not exist. GPU_TARGETS controls which arches the rocKE producer "
            "lowers to kind: kpack; an arch outside it stays authored (kind: rocke) "
            "and is never staged here."
        )
        return SKIP

    if not args.native_source.is_file():
        print(
            f"SKIP: no native source for this pack ({args.arch}): "
            f"{args.native_source} does not exist. The arch table in "
            "HkpPackaging.cmake is a fixed list and names packs a given branch "
            "may not carry; with no C++ half there is nothing to cross-check "
            "against, which is a build-configuration fact rather than a "
            "descriptor defect."
        )
        return SKIP

    result = subprocess.run(
        [
            str(args.validator),
            str(args.root),
            "--expect-engine",
            args.expect_engine,
            "--native-source",
            str(args.native_source),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(
            f"FAIL: {args.validator} produced non-JSON output (exit {result.returncode}):"
        )
        print(result.stdout)
        print(result.stderr)
        return 1

    for diagnostic in report.get("diagnostics", []):
        if diagnostic["severity"] in ("ERROR", "FATAL"):
            print(f"VIOLATION: {diagnostic['message']}")

    for check in report.get("native_source_checks", []):
        if check["clean"]:
            continue
        if check["parse_error"]:
            print(
                f"VIOLATION: native-source parse error in '{check['source_file']}': "
                f"{check['parse_error_message']}"
            )
            continue
        for symbol in check["in_source_not_in_descriptors"]:
            print(
                f"VIOLATION: native-source '{check['source_file']}' declares symbol "
                f"'{symbol}' that no descriptor names"
            )

    for symbol in report.get("descriptor_symbols_no_source_declares", []):
        print(
            f"VIOLATION: descriptor names symbol '{symbol}', which "
            f"'{args.native_source}' does not declare"
        )

    if not report.get("success", False):
        print(
            f"FAIL: {args.expect_engine} native-source cross-check against "
            f"{args.root} is not clean"
        )
        return 1

    print(
        f"OK: {args.expect_engine} native-source cross-check clean "
        f"({len(report.get('diagnostics', []))} diagnostics, 0 errors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
