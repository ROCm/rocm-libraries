import json
import os
import re
import sys
from miopen_gtest_runner import calc_union_filter


def fixture_filter_to_regex(filter):
    filter = re.sub(r"\.\*$", "*", filter.strip())
    return filter.replace("*", ".*")


def analyze_sharded_gtest(input_file):
    # 1. Read input configuration file
    with open(input_file, "r") as f:
        config = json.load(f)

    dapper_filter = config.get("dapper_filter", "")
    union_filter = config.get("union_filter", "")
    shard_log_files = config.get("gtest_shards", [])
    if not shard_log_files:
        print(f"Warning: No shard logs found in {input_file} (json key=gtest_shards)")

    def parse_gtest_filter(filt):
        positives = set()
        negatives = set()
        if not filt:
            return positives, negatives

        parts = filt.split("-")
        if parts[0]:
            # Positive patterns
            for p in parts[0].split(":"):
                p = fixture_filter_to_regex(p)
                if p:
                    positives.add(re.compile(f"^.*{p}$"))
        if len(parts) > 1 and parts[1]:
            # Negative patterns
            for p in parts[1].split(":"):
                p = fixture_filter_to_regex(p)
                if p:
                    negatives.add(re.compile(f"^{p}$"))
        return positives, negatives

    dapper_pos, _ = parse_gtest_filter(dapper_filter)
    union_pos, union_neg = parse_gtest_filter(union_filter)

    def matches_any(test_name, pattern_set):
        return any(p.match(test_name) for p in pattern_set)

    def is_in_dapper(fixture_name):
        return any(p.match(fixture_name) for p in dapper_pos)

    # 2. Load all shard logs and aggregate
    total_passes = 0
    total_failures = 0
    total_skips = 0
    total_time = 0.0

    dapper_fixtures_ran = {}
    other_fixtures = {}

    for log_file in shard_log_files:
        if not os.path.exists(log_file):
            print(f"Warning: Shard json file {log_file} not found. Skipping.")
            continue

        print(f"Parsing log file {log_file}..")
        with open(log_file, "r") as f:
            data = json.load(f)

        # Accumulate total time from shard metadata or calculated sums
        total_time += float(data.get("time", "0s").replace("s", ""))

        for test_suite in data.get("testsuites", []):
            suite_name = test_suite.get("name")
            in_dapper = is_in_dapper(suite_name)
            fixtures = dapper_fixtures_ran if in_dapper else other_fixtures
            if suite_name not in fixtures:
                fixtures[suite_name] = {
                    "passes": 0,
                    "failures": 0,
                    "skips": 0,
                    "tests": [],
                    "time": 0.0,
                }

            fixtures[suite_name]["time"] += float(
                test_suite.get("time", "0s").replace("s", "")
            )

            for test_case in test_suite.get("testsuite", []):
                test_name = f"{suite_name}.{test_case.get('name')}"
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

    # 3. Calculate execution times
    dapper_time = sum(
        f_data["time"]
        for f_name, f_data in dapper_fixtures_ran.items()
        if is_in_dapper(f_name)
    )
    dapper_time_savings = total_time - dapper_time
    dapper_time_pct_saved = (
        (dapper_time_savings / total_time * 100) if total_time > 0 else 0.0
    )

    # 4. Compare test results with gtest filters, and collect sets for validation.
    dapper_failures = 0
    other_failures = 0
    covered_dapper_patterns = set()  # dapper_pos patterns matched by a ran suite
    covered_union_patterns = set()  # union_pos patterns matched by a ran suite
    negated_union_patterns = set()  # union_pos patterns whose suite was also negated

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

    for data in other_fixtures.values():
        if data["failures"] > 0:
            other_failures += 1

    dapper_fixtures_set = {
        p.pattern.replace("^", "").replace("$", "").replace(".*", "*")
        for p in dapper_pos
    }

    # 5. Determine overall and Dapper results
    actual_test_result = "FAIL" if total_failures > 0 else "PASS"
    dapper_test_result = "FAIL" if dapper_failures > 0 else "PASS"
    missing_in_union = len(dapper_pos) - len(covered_dapper_patterns)
    negated_in_union = len(negated_union_patterns)
    dapper_compliance_viable = missing_in_union == 0
    dapper_compliance_result = (
        "COMPLIANT"
        if dapper_compliance_viable and dapper_test_result == "PASS"
        else "FAIL" if dapper_test_result == "FAIL" else "NOT VIABLE"
    )

    # 6. Self-validation: forward count must equal reverse count.
    # Forward: len(dapper_pos) - missing_in_union = len(covered_dapper_patterns)
    # Reverse: union patterns covered minus those negated
    net_covered_union = len(covered_union_patterns) - len(negated_union_patterns)
    expected_covered = len(covered_dapper_patterns)
    validation_ok = net_covered_union == expected_covered

    # 7. Build report & output
    report = {
        "summary": {
            "total_passes": total_passes,
            "total_failures": total_failures,
            "total_skips": total_skips,
            "total_time": total_time,
            "dapper_time": dapper_time,
            "dapper_time_savings": dapper_time_savings,
            "dapper_time_savings_pct": dapper_time_pct_saved,
        },
        "dapper_compliance": {
            "dapper_fixtures_count": len(dapper_fixtures_set),
            "dapper_fixtures_ran": len(dapper_fixtures_ran),
            "fixtures_negated_in_union": negated_in_union,
            "fixtures_missing_in_union": missing_in_union,
            "dapper_compliance_viable": dapper_compliance_viable,
            "dapper_compliance_result": dapper_compliance_result,
        },
        "failures": {
            "dapper_fixtures_with_failures": dapper_failures,
            "other_fixtures_with_failures": other_failures,
        },
        "results": {
            "actual_test_result": actual_test_result,
            "dapper_test_result": dapper_test_result,
        },
        "validation": {
            "covered_dapper_patterns": len(covered_dapper_patterns),
            "covered_union_patterns": net_covered_union,
            "validation_ok": validation_ok,
        },
        "dapper_fixtures_ran": dapper_fixtures_ran,
        "other_fixtures": other_fixtures,
    }

    print("========== Dapper Gtest Sharded Analysis ========================")
    print(f"Total Test Time                            : {total_time:.3f}s")
    print(f"Dapper Time                                : {dapper_time:.3f}s")
    print(
        f"Time Dapper would have saved               : {dapper_time_savings:.3f}s ({dapper_time_pct_saved:.3f}%)"
    )
    print(f"Dapper fixtures not in category filter     : {missing_in_union}")
    print(f"Dapper fixtures negated by category filter : {negated_in_union}")
    print(f"Overall Test Result                        : {actual_test_result}")
    print(f"Dapper Test Result                         : {dapper_test_result}")
    print(
        f"Covered dapper fixture (forward|reverse)   : {expected_covered}|{net_covered_union}"
    )
    print(
        f"Minimal Compliance Achieved?               : {dapper_compliance_result == "COMPLIANT"}"
    )
    print(f"Dapper Compliance                          : {dapper_compliance_result}")
    print(
        f"Validation Result                          : {'VALID' if validation_ok else 'FAIL'}"
    )

    # Write to dapper_results.json
    with open("dapper_results.json", "w") as out_f:
        json.dump(report, out_f, indent=2)

    return report


def main():
    input_file = "miopen_dapper_tests.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    # If dapper test json file, category name, and filter are all given, calculate and record the union
    if len(sys.argv) > 3:
        calc_union_filter(sys.argv[1], sys.argv[2], sys.argv[3])

    report = analyze_sharded_gtest(input_file)
    if not report["validation"]["validation_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
