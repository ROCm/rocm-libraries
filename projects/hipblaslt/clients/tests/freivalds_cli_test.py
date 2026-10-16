#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise validation through the installed client interfaces on a supported GPU."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True, help="hipblaslt-test executable")
    parser.add_argument("--bench", type=Path, required=True, help="hipblaslt-bench executable")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--large", action="store_true", help="also check a 4099 x 4093 x 4097 GEMM")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def run(name, command, *, failure=False, contains=(), absent=()):
        command = [str(value) for value in command]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=600)
        (args.log_dir / (name + ".log")).write_text(result.stdout)
        records.append({"name": name, "command": command, "exit": result.returncode})
        (args.log_dir / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        # Crashes and a run that selected/skipped all tests are not successful refusals.
        assert result.returncode > 0 if failure else result.returncode == 0, (name, result.stdout)
        for text in contains:
            assert text in result.stdout, (name, text, result.stdout)
        for text in absent:
            assert text not in result.stdout, (name, text, result.stdout)
        print(name + ": passed", flush=True)
        return result.stdout

    base = [args.bench, "--precision", "f32_r", "--initialization", "rand_int",
            "-m", "131", "-n", "67", "-k", "200", "--lda", "137", "--ldb", "211",
            "--ldc", "139", "--ldd", "143", "--batch_count", "3", "--alpha", "2",
            "--beta", "-2", "--iters", "1", "--cold_iters", "0"]
    run("bench-freivalds", base + ["--freivalds", "--validation-iters", "3"],
        contains=("checked_launches=3, checked_elements=78993", "full_output_fallback=0"),
        absent=("Bounds check:", "norm_error"))
    run("bench-crosscheck", base + ["--freivalds", "--verify", "--bounds-check"],
        contains=("Freivalds:", "Bounds check:", "norm_error"))
    floating = base.copy()
    floating[floating.index("rand_int")] = "trig_float"
    run("bench-floating-bounds", floating + ["--verify", "--bounds-check", "--validation-iters", "2"],
        contains=("Bounds check: checked_launches=2", "norm_error"), absent=("Freivalds:",))
    run("bench-bounds-only", floating + ["--bounds-check"],
        contains=("Bounds check:",), absent=("Freivalds:", "norm_error"))
    run("bench-fractional-refused", floating + ["--freivalds"], failure=True,
        contains=("Freivalds:", "non-integer"))
    for method in ("--freivalds", "--verify"):
        run("bench-injection-" + method[2:],
            base + [method, "--validation-iters", "4", "--validation-inject", "2"],
            failure=True, contains=("iteration 2 of 4",))
    run("bench-invalid-iterations", base + ["--freivalds", "--validation-iters=0"],
        failure=True, contains=("validation_iters must be at least 1",))
    empty = base.copy()
    empty[empty.index("--batch_count") + 1] = "0"
    run("bench-empty-injection", empty + ["--freivalds", "--validation-inject=0"],
        failure=True, contains=("validation_inject requires a nonempty output",))
    run("bench-verifier-option-order", base + ["--verify", "--freivalds"],
        contains=("Freivalds:", "norm_error"))

    workload = Path(__file__).with_name("data") / "freivalds_cli.yaml"
    with tempfile.TemporaryDirectory(prefix="hipblaslt-validation-") as directory:
        directory = Path(directory)
        data = directory / "workload.data"
        run("generate-data", [sys.executable, args.test.parent / "hipblaslt_gentest.py",
                              "--template", args.test.parent / "hipblaslt_template.yaml",
                              workload, "-o", data])
        for option, path in (("--yaml", workload), ("--data", data)):
            for client in (args.test, args.bench):
                command = [client, option, path, "--freivalds", "--bounds-check",
                           "--validation-iters", "3"]
                if client == args.test:
                    command += ["--gtest_filter=*freivalds_cli*"]
                run(client.name + option, command,
                    contains=("checked_launches=3, checked_elements=78993", "Bounds check:"),
                    absent=("norm_error", "[  SKIPPED ]", "Running 0 tests"))
        run("test-injection", [args.test, "--data", data, "--freivalds", "--bounds-check=0",
                               "--validation-iters=4", "--validation-inject=2",
                               "--gtest_filter=*freivalds_cli*"], failure=True,
            contains=("iteration 2 of 4", "batch 2, row 65, col 33"))
        run("bench-yaml-unit-check", [args.bench, "--data", data, "--validation-inject=0"],
            failure=True, contains=("reference validation", "iteration 0 of 1"))
        run("test-unknown-option", [args.test, "--data", data, "--unknown-verifier"],
            failure=True, contains=("Unknown option:",))
        invalid = directory / "invalid.yaml"
        invalid.write_text(workload.read_text().replace("  freivalds: 0", "  unknown_verifier: 1"))
        run("unknown-yaml-field", [args.test, "--yaml", invalid], failure=True,
            contains=("Unknown test argument(s)", "unknown_verifier"))

    if args.large:
        large = [args.test, "--yaml", workload.with_name("freivalds_large_cli.yaml"),
                 "--freivalds", "--gtest_filter=*freivalds_large_cli*"]
        run("large-freivalds", large,
            contains=("checked_launches=16, checked_elements=536870624",
                      "recomputed_elements=0", "full_output_fallback=0"))
        output = run("large-injection", large + ["--validation-inject", "2"], failure=True,
                     contains=("checked_launches=16, checked_elements=536870624",
                               "recomputed_elements=0", "full_output_fallback=0",
                               "batch 1, row 2049, col 2046"))
        assert output.count("iteration 2 of 4") == 4, output


if __name__ == "__main__":
    main()
