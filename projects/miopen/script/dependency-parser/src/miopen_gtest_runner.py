import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

import json
import os
import subprocess
from pathlib import Path

# The pure union math (pattern splitting/overlap + subtractive intersection) is shared
# with the TheRock runner; dapper_union.py is the single source of truth for it.
from dapper_union import (
    compute_union_filter,
    split_gtest_filter_includes,
    patterns_overlap,
)

# Secondary domains: independent gtest invocations that run outside the primary shards
# (e.g. HipGraphExist runs serially because it OOMs when shards share a GPU). Each is
# analyzed as its own Dapper "domain" -- own shard file(s), own filters, its own result.
# 'name' is the label used in the shard-file grouping, the shard filenames, and the
# one-line report; 'filter' is the gtest filter that invocation runs.
SECONDARY_DOMAINS = [
    {"name": "HipGraphExist", "filter": "*HipGraphExist*"},
]


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


def _convert_shards(shards):
    """Convert a domain's XML shard paths to JSON, preferring an existing .json over the
    .xml source. If any shard has neither output, ALL missing shards are reported and the
    run is aborted (see abort_missing_shards) -- no partial analysis. Returns the converted
    list (json paths where conversion happened, originals otherwise)."""
    from selective_test_filter import _xml_to_gtest_json

    converted = []
    missing = []
    for shard in shards:
        p = Path(shard)
        if p.suffix.lower() == ".xml":
            json_path = p.with_suffix(".json")
            if json_path.exists():
                converted.append(str(json_path))
            elif p.exists():
                data = _xml_to_gtest_json(p)
                json_path.write_text(json.dumps(data, indent=2))
                converted.append(str(json_path))
            else:
                missing.append(shard)
                converted.append(shard)
        else:
            converted.append(shard)
    if missing:
        abort_missing_shards(missing, len(shards))
    return converted


def _join_filter(positives, negatives):
    result = ":".join(positives)
    if negatives:
        result = result + "-" + ":".join(negatives)
    return result


def _domain_filter(own_filter, other_positive_filters):
    """Return own_filter with every other domain's positives added to its negative side,
    making domains mutually exclusive. The primary is the catch-all: its positives are the
    ones NOT passed here to secondaries."""
    pos, neg = split_gtest_filter_includes(own_filter)
    neg = list(neg)
    for other in other_positive_filters:
        for p in split_gtest_filter_includes(other)[0]:
            if p not in neg:
                neg.append(p)
    return _join_filter(pos, neg)


def _classify_impact(dapper_positives):
    """Assign each global-impact positive to a domain: the first secondary whose positive
    it overlaps, else 'primary' (the catch-all). Returns {domain_name: [positives]}."""
    assigned = {"primary": []}
    sec_positives = {}
    for sec in SECONDARY_DOMAINS:
        assigned[sec["name"]] = []
        sec_positives[sec["name"]] = split_gtest_filter_includes(sec["filter"])[0]
    for p in dapper_positives:
        placed = "primary"
        for sec in SECONDARY_DOMAINS:
            if any(patterns_overlap(p, sp) for sp in sec_positives[sec["name"]]):
                placed = sec["name"]
                break
        assigned[placed].append(p)
    return assigned


def build_domains(json_data, shard_groups, category_filter):
    """Build the per-domain analysis records from the global impact + the shard grouping.

    Each domain is completely independent: its own shard files, its own dapper_filter
    (the slice of the global impact assigned to it) and union_filter. 'primary' is the
    single catch-all domain (its positives are never subtracted from secondaries);
    secondaries are the out-of-shard invocations declared in SECONDARY_DOMAINS.
    """
    dapper_positives = split_gtest_filter_includes(json_data.get("dapper_filter", ""))[0]
    assigned = _classify_impact(dapper_positives)
    sec_filter_by_name = {s["name"]: s["filter"] for s in SECONDARY_DOMAINS}
    all_sec_filters = [s["filter"] for s in SECONDARY_DOMAINS]

    domains = []
    primary_union = ""
    for group in shard_groups:
        name = group["name"]
        if name == "primary":
            dtype = "primary"
            # primary = catch-all: exclude every secondary's positives, keep nothing else.
            dfilter = _domain_filter(category_filter, all_sec_filters)
        else:
            dtype = "secondary"
            own = sec_filter_by_name.get(name, name)
            others = [f for n, f in sec_filter_by_name.items() if n != name]
            dfilter = _domain_filter(own, others)
        domain_dapper = ":".join(assigned.get(name, []))
        union = compute_union_filter(domain_dapper, dfilter)
        domains.append(
            {
                "type": dtype,
                "name": name,
                "dapper_filter": domain_dapper,
                "union_filter": union,
                "shards": _convert_shards(group.get("shards", [])),
            }
        )
        if dtype == "primary":
            primary_union = union
    return domains, primary_union


def calc_union_filter(gtest_filter_json: str, category_name: str, category_filter: str):
    """Native (validate-mode) domain assembly: read the impact + the shard grouping
    (miopen_gtest_shards.txt, one entry per domain), slice the impact per domain, compute
    each domain's subtractive union, convert its shard XML, and record domains[] back into
    the tests JSON for dapper_diff to analyze. Returns the primary domain's union filter
    (for the standalone run_gtest path)."""
    with open(gtest_filter_json, "r") as f:
        json_data = json.load(f)

    shards_path = os.path.join(
        os.path.dirname(os.path.abspath(gtest_filter_json)), "miopen_gtest_shards.txt"
    )
    with open(shards_path, "r") as f:
        shard_groups = json.load(f)

    json_data["category_name"] = category_name
    domains, primary_union = build_domains(json_data, shard_groups, category_filter)
    json_data["domains"] = domains

    with open(gtest_filter_json, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"================= calc_union_filter: built {len(domains)} domain(s)")
    return primary_union


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
