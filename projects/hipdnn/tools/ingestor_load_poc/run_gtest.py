# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Run one ctest entry directly, with ctest's own command line and environment.

For a gtest binary, a --gtest_filter is appended. hipDNN registers one ctest entry per
binary (or per tier), so `ctest -R <suite>` cannot select suites, and both an empty ctest
selection and a gtest filter matching nothing exit 0: the counts printed here are the
check, not the exit code. An empty filter runs the entry as registered, which is how the
hkp_pack pytest entries run with the kpack/hipcc environment their conftest reads.

Usage: python run_gtest.py <ctest dir> <ctest test name> <gtest filter or ""> <log file>
Prints one JSON line with gtest counts, or pytest counts for a pytest entry, plus rc.
"""
import json
import os
import re
import subprocess
import sys


def main(build_dir, name, gtest_filter, log_path):
    show = subprocess.run(
        ["ctest", "--test-dir", build_dir, "--show-only=json-v1", "-R", f"^{name}$"],
        capture_output=True,
        text=True,
        check=True,
    )
    tests = json.loads(show.stdout)["tests"]
    if len(tests) != 1:
        sys.exit(f"expected one ctest entry named {name}, found {len(tests)}")
    test = tests[0]
    env = dict(os.environ)
    cwd = None
    for prop in test.get("properties", []):
        if prop["name"] == "ENVIRONMENT":
            for item in prop["value"]:
                key, _, value = item.partition("=")
                env[key] = value
        elif prop["name"] == "ENVIRONMENT_MODIFICATION":
            for item in prop["value"]:
                # ctest spells each entry VAR=op:value.
                key, _, rest = item.partition("=")
                op, _, value = rest.partition(":")
                if op == "set":
                    env[key] = value
                elif op == "path_list_prepend":
                    env[key] = value + (os.pathsep + env[key] if env.get(key) else "")
                elif op == "path_list_append":
                    env[key] = (env[key] + os.pathsep if env.get(key) else "") + value
                elif op == "unset":
                    env.pop(key, None)
        elif prop["name"] == "WORKING_DIRECTORY":
            cwd = prop["value"]
    command = test["command"] + (
        [f"--gtest_filter={gtest_filter}"] if gtest_filter else []
    )
    with open(log_path, "w") as log:
        rc = subprocess.run(
            command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    text = open(log_path, errors="replace").read()

    def count(pattern):
        m = re.search(pattern, text, re.M)
        return int(m.group(1)) if m else 0

    result = {"test": name, "filter": gtest_filter, "rc": rc}
    summary = re.findall(r"^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s", text, re.M)
    if summary:
        # pytest: "12 failed, 300 passed, 1 skipped, 3 errors in 41.95s"
        for n, what in re.findall(r"(\d+) (\w+)", summary[-1]):
            result[what] = int(n)
    else:
        result.update(
            {
                "ran": count(r"^\[==========\] (\d+) tests? from \d+ test suites? ran"),
                "passed": count(r"^\[  PASSED  \] (\d+) tests?"),
                "failed": count(r"^\[  FAILED  \] (\d+) tests?, listed below"),
                "skipped": count(r"^\[  SKIPPED \] (\d+) tests?, listed below"),
            }
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main(*sys.argv[1:5])
