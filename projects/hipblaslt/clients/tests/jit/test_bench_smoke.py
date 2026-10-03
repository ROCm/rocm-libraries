# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check HIPBLASLT_JIT through hipblaslt-bench on a real GPU.

forced runs one FP16 GEMM with HIPBLASLT_JIT=2 on what the test backend replays
and checks it against the CPU reference; library-reuse runs it again from the
published solution, in a process where generation fails. With --jit-off, a
build without HIPBLASLT_ENABLE_JIT must ignore HIPBLASLT_JIT with one warning.
Every run uses an empty HIPBLASLT_TENSILE_LIBPATH and its own JIT library.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess

IGNORED = (
    "hipblaslt warning: HIPBLASLT_JIT=2 is ignored: "
    "hipBLASLt was built without HIPBLASLT_ENABLE_JIT"
)
GEMM = [
    "--api_method", "c", "-m", "256", "-n", "128", "-k", "512",
    "-r", "f16_r", "--compute_type", "f32_r", "--alpha", "1.25", "--beta", "0.5",
    "--verify", "--iters", "3", "--cold_iters", "1", "--print_kernel_info",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rows(stdout):
    """The benchmark's correctness CSV rows, as dictionaries."""
    lines = stdout.splitlines()
    found = []
    for index, line in enumerate(lines[:-1]):
        if "norm_error" not in line or ",atol" not in line:
            continue
        fields = next(csv.reader([re.sub(r"^\s*\[\d+\]:", "", line).strip()]))
        values = next(csv.reader([lines[index + 1].strip()]))
        require(len(fields) == len(values), "Malformed benchmark result row")
        found.append(dict(zip(fields, values)))
    return found


def check_numerics(stdout):
    found = rows(stdout)
    require(found, "No benchmark correctness row")
    for row in found:
        error = float(row["norm_error"])
        require(math.isfinite(error) and error <= 0.01, f"FP16 norm error {error}")


def jit_reports(stderr):
    return re.findall(r"^hipblaslt (?:error|warning): JIT .*$", stderr, re.MULTILINE)


def entries(library):
    return list(library.rglob("TensileLibrary_JIT_*.dat"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench", type=Path)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument(
        "--replay", action="append", type=Path, help="a bundle for the test backend to replay"
    )
    parser.add_argument("--jit-off", action="store_true", help="the build has no JIT")
    args = parser.parse_args()
    if args.jit_off == bool(args.replay):
        parser.error("give either --jit-off or --replay")
    bench = args.bench.resolve(strict=True)
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    empty = output / "empty-device-library"
    empty.mkdir()
    (output / "tmp").mkdir()
    library = output / "lib"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HIPBLASLT_JIT", "HIPBLASLT_TUNING", "AMD_COMGR_"))
    }
    env.update(
        HIPBLASLT_JIT="2",
        HIPBLASLT_TENSILE_LIBPATH=str(empty),
        HIPBLASLT_JIT_LIBRARY_PATH=str(library),
        TMPDIR=str(output / "tmp"),
    )

    def run(name, **overrides):
        command = [str(bench), *GEMM]
        result = subprocess.run(
            command, env=dict(env, **overrides), text=True, capture_output=True, timeout=900
        )
        (output / f"{name}.command.json").write_text(
            json.dumps(dict(command=command, environment=overrides), indent=2)
        )
        (output / f"{name}.stdout").write_text(result.stdout)
        (output / f"{name}.stderr").write_text(result.stderr)
        return result

    if args.jit_off:
        result = run("jit-off")
        require(result.stderr.count(IGNORED) == 1, "HIPBLASLT_JIT=2 was not reported once")
        require(not library.exists(), "A build without JIT created a JIT library")
        print("PASS bench-jit-off: HIPBLASLT_JIT ignored with one warning")
        return

    env["HIPBLASLT_JIT_TEST_REPLAY"] = os.pathsep.join(
        str(path.resolve(strict=True)) for path in args.replay
    )
    kernels = [
        json.loads((path / "manifest.json").read_text())["main_kernel"]["name"]
        for path in args.replay
    ]

    result = run("forced")
    require(result.returncode == 0, f"forced exited with {result.returncode}; see {output}")
    check_numerics(result.stdout)
    require(not jit_reports(result.stderr), "forced reported a JIT problem")
    ran = [kernel for kernel in kernels if kernel in result.stdout]
    require(len(ran) == 1, f"forced did not run exactly one replayed kernel: {ran}")
    require(len(entries(library)) == 1, "forced did not publish one solution")
    print("PASS bench-forced: a JIT solution passed the CPU reference")

    record = output / "record.txt"
    result = run(
        "library-reuse", HIPBLASLT_JIT_TEST_FAULT="record", HIPBLASLT_JIT_TEST_RECORD=str(record)
    )
    require(result.returncode == 0, f"library-reuse exited with {result.returncode}")
    check_numerics(result.stdout)
    require(not record.exists(), f"library-reuse generated: {record}")
    require(not jit_reports(result.stderr), "library-reuse reported a JIT problem")
    require(ran[0] in result.stdout, "library-reuse ran a different kernel")
    print("PASS bench-library-reuse: the published solution ran without generation")


if __name__ == "__main__":
    main()
