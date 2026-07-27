import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

import json
import os
import re
from miopen_gtest_runner import calc_union_filter, abort_missing_shards


def fixture_filter_to_regex(filter):
    filter = re.sub(r"\.\*$", "*", filter.strip())
    return filter.replace("*", ".*")


def parse_gtest_filter(filt):
    positives = set()
    negatives = set()
    if not filt:
        return positives, negatives
    parts = filt.split("-")
    if parts[0]:
        for p in parts[0].split(":"):
            p = fixture_filter_to_regex(p)
            if p:
                positives.add(re.compile(f"^.*{p}$"))
    if len(parts) > 1 and parts[1]:
        for p in parts[1].split(":"):
            p = fixture_filter_to_regex(p)
            if p:
                negatives.add(re.compile(f"^{p}$"))
    return positives, negatives


def matches_any(test_name, pattern_set):
    return any(p.match(test_name) for p in pattern_set)


def analyze_domain(domain):
    """Analyze one independent Dapper domain from its own shard files + filters.

    Domains are self-contained: 'dapper_filter'/'union_filter'/'shards' all belong to this
    domain only. 'type' does not affect the analysis -- it only selects the report format
    later (primary -> full report, secondary -> one-liner). Returns a result dict.
    """
    name = domain.get("name", "")
    dtype = domain.get("type", "primary")
    dapper_filter = domain.get("dapper_filter", "")
    union_filter = domain.get("union_filter", "")
    shard_log_files = domain.get("shards", [])

    absent = [s for s in shard_log_files if not os.path.exists(s)]
    if absent:
        abort_missing_shards(absent, len(shard_log_files))

    dapper_pos, _ = parse_gtest_filter(dapper_filter)
    union_pos, union_neg = parse_gtest_filter(union_filter)

    def is_in_dapper(fixture_name):
        return any(p.match(fixture_name) for p in dapper_pos)

    total_passes = 0
    total_failures = 0
    total_skips = 0
    total_time = 0.0
    dapper_fixtures_ran = {}
    other_fixtures = {}

    for log_file in shard_log_files:
        with open(log_file, "r") as f:
            data = json.load(f)
        total_time += float(data.get("time", "0s").replace("s", ""))
        for test_suite in data.get("testsuites", []):
            suite_name = test_suite.get("name")
            fixtures = dapper_fixtures_ran if is_in_dapper(suite_name) else other_fixtures
            if suite_name not in fixtures:
                fixtures[suite_name] = {"passes": 0, "failures": 0, "skips": 0, "time": 0.0}
            fixtures[suite_name]["time"] += float(
                test_suite.get("time", "0s").replace("s", "")
            )
            for test_case in test_suite.get("testsuite", []):
                status = test_case.get("status")
                result = test_case.get("result")
                if status == "NOTRUN" or result == "SKIPPED":
                    total_skips += 1
                    fixtures[suite_name]["skips"] += 1
                elif result == "COMPLETED" and test_case.get("failures"):
                    total_failures += 1
                    fixtures[suite_name]["failures"] += 1
                elif result == "COMPLETED":
                    total_passes += 1
                    fixtures[suite_name]["passes"] += 1

    dapper_time = sum(f["time"] for f in dapper_fixtures_ran.values())
    dapper_time_savings = total_time - dapper_time
    dapper_time_pct_saved = (
        (dapper_time_savings / total_time * 100) if total_time > 0 else 0.0
    )

    dapper_failures = 0
    covered_dapper_patterns = set()
    covered_union_patterns = set()
    negated_union_patterns = set()
    for suite, data in dapper_fixtures_ran.items():
        if data["failures"] > 0:
            dapper_failures += 1
        for p in dapper_pos:
            if p.match(suite):
                covered_dapper_patterns.add(p.pattern)
        for p in union_pos:
            if p.match(suite):
                covered_union_patterns.add(p.pattern)
                if matches_any(suite, union_neg):
                    negated_union_patterns.add(p.pattern)

    missing_in_union = len(dapper_pos) - len(covered_dapper_patterns)
    negated_in_union = len(negated_union_patterns)
    net_covered_union = len(covered_union_patterns) - negated_in_union
    forward = len(covered_dapper_patterns)
    validation_ok = net_covered_union == forward

    # 'test:' is the domain's test outcome (any test failed in its shards); it is reported
    # for information only -- a failed test already reports itself and never fails dapper.
    test_result = "FAIL" if total_failures > 0 else "PASS"
    # Compliance is about coverage/validation, independent of test pass/fail.
    if not validation_ok:
        compliance = "FAIL"
    elif missing_in_union > 0:
        compliance = "NOT VIABLE"
    else:
        compliance = "COMPLIANT"

    return {
        "name": name,
        "type": dtype,
        "compliance": compliance,
        "validation_ok": validation_ok,
        "test_result": test_result,
        "total_time": total_time,
        "dapper_time": dapper_time,
        "dapper_time_savings": dapper_time_savings,
        "dapper_time_savings_pct": dapper_time_pct_saved,
        "missing_in_union": missing_in_union,
        "negated_in_union": negated_in_union,
        "forward": forward,
        "reverse": net_covered_union,
        "dapper_failures": dapper_failures,
        "total_failures": total_failures,
    }


# Width of the label column in the full report; secondary one-liners pad their
# description to the same width so every ':' lines up in a single column.
LABEL_WIDTH = 42


def _rline(label, value):
    return f"{label:<{LABEL_WIDTH}} : {value}"


def print_full_report(r):
    print("========== Dapper Gtest Sharded Analysis ========================")
    print(_rline("Total Test Time", f"{r['total_time']:.3f}s"))
    print(_rline("Dapper Time", f"{r['dapper_time']:.3f}s"))
    print(
        _rline(
            "Time Dapper would have saved",
            f"{r['dapper_time_savings']:.3f}s ({r['dapper_time_savings_pct']:.3f}%)",
        )
    )
    print(_rline("Dapper fixtures not in category filter", r["missing_in_union"]))
    print(_rline("Dapper fixtures negated by category filter", r["negated_in_union"]))
    print(
        _rline("Covered dapper fixture (forward|reverse)", f"{r['forward']}|{r['reverse']}")
    )
    print(_rline("Dapper Compliance", r["compliance"]))
    print(_rline("Validation Result", "VALID" if r["validation_ok"] else "FAIL"))
    print(_rline("Test Result", r["test_result"]))


def main():
    input_file = "miopen_dapper_tests.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    # When the tests json, category name, and category filter are given, (re)build the
    # per-domain records (union filters, shard conversion) before analyzing.
    if len(sys.argv) > 3:
        calc_union_filter(sys.argv[1], sys.argv[2], sys.argv[3])

    with open(input_file, "r") as f:
        config = json.load(f)
    domains = config.get("domains", [])
    if not domains:
        print(f"Warning: no domains found in {input_file} (json key=domains)")

    reports = [analyze_domain(d) for d in domains]

    for r in reports:
        if r["type"] == "primary":
            print_full_report(r)

    secondaries = [r for r in reports if r["type"] == "secondary"]
    if secondaries:
        # Pad the description to LABEL_WIDTH so the ':' aligns with the primary report's
        # column; pad the result so 'test:' aligns across multiple secondaries.
        res_w = max(len(r["compliance"]) for r in secondaries)
        for r in secondaries:
            print(
                f"{r['name']:<{LABEL_WIDTH}} : {r['compliance']:<{res_w}}  test: {r['test_result']}"
            )

    with open("dapper_results.json", "w") as out_f:
        json.dump({"domains": reports}, out_f, indent=2)

    # dapper_diff fails (non-zero) only on a self-validation failure of any domain.
    if any(not r["validation_ok"] for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
