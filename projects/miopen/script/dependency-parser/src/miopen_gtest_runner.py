import fnmatch
import json
import subprocess
import sys


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
    else:
        positives = [p for p in filter_str.split(":") if p]
        negatives = []

    return positives, negatives


def matches_any_filter(s, filters):
    """
    Checks if a string 's' matches any of the wildcard patterns in 'filters'.
    """
    return any(fnmatch.fnmatch(s, pattern) for pattern in filters)


def calc_union_filter(gtest_filter_json: str, category_name: str, category_filter: str):
    with open(gtest_filter_json, "r") as f:
        json_data = json.load(f)
    # super-minimal default test if there's nothing to do:
    dapper_filter = "CPU_HandleHipDevice_NONE*"
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
        if matches_any_filter(df.strip("*"), category_positives)
    ]
    union_filter = ":".join(union_positives)
    if category_exclude:
        union_filter = union_filter + "-" + category_exclude

    json_data["union_filter"] = union_filter

    with open(gtest_filter_json, "w") as f:
        json.dump(json_data, f, indent=2)

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
