import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

import fnmatch
import json
import os
import subprocess
from pathlib import Path


def split_gtest_filter_includes(filter_str):
    """
    Splits a --gtest_filter style string into positive and negative filter lists.

    Example:
        "ABC.*:DEF.*:-XYZ.*:-123.*"
        -> (['ABC.*', 'DEF.*'], ['XYZ.*', '123.*'])
    """
    if not filter_str:
        return [], []

    # Split into positive and negative parts
    if "-" in filter_str:
        positive_part, *negative_part = filter_str.split("-")
        positives = [p for p in positive_part.split(":") if p]
        negatives = negative_part
    else:
        positives = [p for p in filter_str.split(":") if p]
        negatives = []

    # If filter is negative-only, gtest includes all tests
    if not positives:
        positives = ["*"]

    return positives, negatives


def matches_any_filter(s, filters):
    """
    Checks if a string 's' matches any of the wildcard patterns in 'filters'.
    """
    return any(fnmatch.fnmatch(s, pattern) for pattern in filters)


def _fixed_prefix(pattern):
    """
    Return the literal portion of a wildcard pattern up to the first wildcard
    metacharacter ('*', '?', '['). Dapper patterns only wildcard at the end, so
    for them this is the full fixture name; category patterns may wildcard
    anywhere, so this is just their leading literal.
    """
    for i, ch in enumerate(pattern):
        if ch in "*?[":
            return pattern[:i]
    return pattern


def patterns_overlap(dapper_pattern, category_pattern):
    """
    Return True if a dapper (prefix-style) pattern and a category (arbitrary
    wildcard) pattern could match a common gtest fixture name.

    fnmatch needs a concrete string on one side and a pattern on the other, so a
    single stripped comparison is asymmetric and misses real overlaps. We test
    both directions: the dapper pattern's literal prefix against the category
    glob, and the category pattern's literal prefix against the dapper glob.
    Either match means gtest would run at least one shared fixture, so the dapper
    pattern belongs in the union.
    """
    return fnmatch.fnmatch(
        _fixed_prefix(dapper_pattern), category_pattern
    ) or fnmatch.fnmatch(_fixed_prefix(category_pattern), dapper_pattern)


def _convert_xml_shards(json_data):
    """Convert XML shard paths to JSON, preferring an existing .json over the .xml source."""
    from selective_test_filter import _xml_to_gtest_json

    shards = json_data.get("gtest_shards", [])
    converted = []
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
                print(
                    f"Error: shard '{p.stem}' is missing both its .json and .xml outputs."
                )
                print(
                    "Either run the tests to generate shard outputs, or copy valid shard"
                )
                print(f"files ({json_path.name} or {p.name}) into: {p.parent}")
                sys.exit(1)
        else:
            converted.append(shard)
    if changed:
        json_data["gtest_shards"] = converted


def calc_union_filter(gtest_filter_json: str, category_name: str, category_filter: str):
    with open(gtest_filter_json, "r") as f:
        json_data = json.load(f)
    _convert_xml_shards(json_data)
    # super-minimal default test if there's nothing to do:
    default_filter = "CPU_HandleHipDevice_NONE*"
    dapper_filter = default_filter
    if "dapper_filter" in json_data:
        dapper_filter = json_data["dapper_filter"]

    json_data["category_name"] = category_name
    category_filter_name = (
        f"category_{category_name}_filter" if category_name else "category_filter"
    )
    json_data[category_filter_name] = category_filter

    # The category filter can contain wildcards anywhere, but dapper only does at the
    # end of each fixture, so it's easy to compare each dapper item for a category match.
    # Also, dapper does not define negatives, so enforce this by ignoring them.
    dapper_positives, _ = split_gtest_filter_includes(dapper_filter)
    category_positives, category_exclude = split_gtest_filter_includes(category_filter)

    union_positives = [
        df
        for df in dapper_positives
        if any(patterns_overlap(df, cp) for cp in category_positives)
    ]
    deduped = list(dict.fromkeys(union_positives))
    duplicates_removed = len(union_positives) - len(deduped)
    if duplicates_removed:
        print(f"Removed {duplicates_removed} duplicate entries from union_positives")
    union_positives = deduped

    # If the Dapper filter and the category filter share no fixtures, there is
    # nothing meaningful to run. Fall back to the super-minimal default test and
    # warn that coverage may be missing. This is a PASS and COMPLIANT situation,
    # not a failure -- an empty positive filter would otherwise make gtest run
    # everything, which is the opposite of what's intended.
    if not union_positives:
        print(
            "WARNING: no overlap between the Dapper filter and category filter "
            "'{0}'; falling back to super-minimal default test "
            "'{1}'. Testing may be missing for this category, but "
            "this is a PASS and COMPLIANT.".format(category_name, default_filter)
        )
        union_positives = [default_filter]

    union_filter = ":".join(union_positives)
    if category_exclude:
        category_exclude_filter = ":".join(category_exclude)
        union_filter = union_filter + "-" + category_exclude_filter

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
