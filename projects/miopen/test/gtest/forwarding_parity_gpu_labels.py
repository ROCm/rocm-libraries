#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Print the ex_gpu_* labels the shared test-category parser puts on an enabled test.

    forwarding_parity_gpu_labels.py <parse_test_categories.py> <test_categories.yaml>

Prints the labels as one CMake list on stdout. Exits non-zero, with the parser's
error on stderr, if the parser fails.

The runner selects an architecture by label, so this list must match the parser's
exactly. An extra label makes the parity entry the only test on that architecture;
a missing one leaves the architecture without parity coverage. Running the parser,
rather than reading the YAML a second way, keeps its rules in one place.
"""

import subprocess
import sys


def enabled_ex_gpu_labels(parser_output):
    """Return the sorted ex_gpu_* labels on non-DISABLED tests in `parser_output`.

    A suite whose patterns exclude everything is emitted DISABLED. It still carries
    its label, but nothing runs there, so the parity entry must not run there either.
    """
    labels = set()
    block = None
    for line in parser_output.splitlines():
        stripped = line.strip()
        if line.startswith("set_tests_properties("):
            block = {"labels": [], "disabled": False}
        elif block is None:
            continue
        elif stripped.startswith("LABELS "):
            block["labels"] = stripped[len("LABELS ") :].strip('"').split(";")
        elif stripped == "DISABLED TRUE":
            block["disabled"] = True
        elif stripped == ")":
            if not block["disabled"]:
                labels.update(l for l in block["labels"] if l.startswith("ex_gpu_"))
            block = None
    return sorted(labels)


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            f"usage: {argv[0]} <parse_test_categories.py> <test_categories.yaml>\n"
        )
        return 2
    parser_script, yaml_file = argv[1:]
    # The target name and working directory only appear in the generated commands,
    # which are discarded here.
    out = subprocess.run(
        [sys.executable, parser_script, yaml_file, "miopen_gtest", "."],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        return out.returncode or 1
    print(";".join(enabled_ex_gpu_labels(out.stdout)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
