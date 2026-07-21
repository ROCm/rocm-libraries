import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

import json
import os
import subprocess
from pathlib import Path

# The pure union math (pattern splitting/overlap + subtractive intersection) is shared
# with the TheRock runner; dapper_union.py is the single source of truth for it.
from dapper_union import compute_union_filter


def abort_missing_shards(missing, total):
    """Report every shard that produced no output and abort with a non-zero exit.

    A shard with neither its .xml nor .json means the gtest process exited before
    writing results (typically a crash). Dapper does NOT continue with a partial set
    of shards -- a partial analysis is misleading -- so it lists exactly which shards
    failed and fails the whole run.
    """
    bar = "=" * 72
    print(bar, file=sys.stderr)
    print(
        f"DAPPER FATAL: {len(missing)} of {total} gtest shard(s) produced no output "
        "(the shard's gtest process exited before writing its XML, e.g. it crashed).",
        file=sys.stderr,
    )
    print("Failed shard(s):", file=sys.stderr)
    for shard in missing:
        p = Path(shard)
        print(
            f"  - {p.stem}: no {p.with_suffix('.json').name} or {p.name} in {p.parent}",
            file=sys.stderr,
        )
    print(
        "Aborting: dapper will not produce a partial analysis from an incomplete "
        "set of shards.",
        file=sys.stderr,
    )
    print(bar, file=sys.stderr)
    sys.exit(1)


def _convert_xml_shards(json_data):
    """Convert XML shard paths to JSON, preferring an existing .json over the .xml source.

    If any shard has neither output, ALL missing shards are reported and the run is
    aborted (see abort_missing_shards) -- no partial analysis.
    """
    from selective_test_filter import _xml_to_gtest_json

    shards = json_data.get("gtest_shards", [])
    converted = []
    missing = []
    changed = False
    for shard in shards:
        p = Path(shard)
        if p.suffix.lower() == ".xml":
            json_path = p.with_suffix(".json")
            if json_path.exists():
                print(
                    f"Using existing JSON shard {json_path} (skipping XML conversion)."
                )
                converted.append(str(json_path))
                changed = True
            elif p.exists():
                data = _xml_to_gtest_json(p)
                json_path.write_text(json.dumps(data, indent=2))
                converted.append(str(json_path))
                changed = True
            else:
                missing.append(shard)
                converted.append(shard)
        else:
            converted.append(shard)
    if missing:
        abort_missing_shards(missing, len(shards))
    if changed:
        json_data["gtest_shards"] = converted


def calc_union_filter(gtest_filter_json: str, category_name: str, category_filter: str):
    """Native (validate-mode) union: convert the shard XML, compute the subtractive
    union via the shared dapper_union helper, and record it back into the shards JSON.

    The union math itself lives in dapper_union.compute_union_filter (shared with the
    TheRock runner); here we only own the shard-JSON I/O and the annotations that
    dapper_diff reads back.
    """
    with open(gtest_filter_json, "r") as f:
        json_data = json.load(f)
    _convert_xml_shards(json_data)
    dapper_filter = json_data.get("dapper_filter", "")

    json_data["category_name"] = category_name
    category_filter_name = (
        f"category_{category_name}_filter" if category_name else "category_filter"
    )
    json_data[category_filter_name] = category_filter

    union_filter = compute_union_filter(dapper_filter, category_filter)
    json_data["union_filter"] = union_filter

    with open(gtest_filter_json, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"================= calc_union_filter: union_filter={union_filter}")
    return union_filter


def run_gtest(gtest_executable: str, gtest_filter: str):
    print(f"Running {gtest_executable} with filter: {gtest_filter}", flush=True)
    subprocess.run([gtest_executable, f"--gtest_filter={gtest_filter}"], check=True)


def main():
    gtest_executable = sys.argv[1]
    gtest_filter_json = sys.argv[2]
    category_name = "none"
    if len(sys.argv) > 3:
        category_name = sys.argv[3]
    category_filter = "*"
    if len(sys.argv) > 4:
        category_filter = sys.argv[4]

    gtest_filter = calc_union_filter(gtest_filter_json, category_name, category_filter)
    run_gtest(gtest_executable, gtest_filter)


if __name__ == "__main__":
    main()
