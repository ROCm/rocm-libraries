#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate Q27B prefill kernels using the shipped Equality grid's tuning.

Run with the TensileLite toolchain on PATH. The output is an isolated device
library, suitable for hipblaslt-bench via HIPBLASLT_TENSILE_LIBPATH. No custom
assembly is used for the two prefill solutions.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root.parent / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_Q27B.yaml")
    logic = yaml.safe_load(source.read_text())
    rows = [row for row in logic["ExactLogic"] if row[0][1] == 2048]
    indices = sorted({row[1][0] for row in rows})
    remap = {old: new for new, old in enumerate(indices)}
    solutions = []
    for index in indices:
        solution = deepcopy(next(s for s in logic["Solutions"] if s["SolutionIndex"] == index))
        solution["CustomKernelName"] = ""
        solution["SolutionIndex"] = remap[index]
        solutions.append(solution)
    for row in rows:
        row[1][0] = remap[row[1][0]]
    logic["Solutions"] = solutions
    logic["ExactLogic"] = rows
    output = args.output.resolve()
    directory = output / "logic"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / source.name).write_text(yaml.safe_dump(logic, sort_keys=False))
    subprocess.run([
        sys.executable, "-m", "Tensile.TensileCreateLibrary",
        "--architecture=gfx1151", "--logic-filter=*", "--no-compress",
        "--keep-build-tmp", f"--jobs={args.jobs}", str(directory), str(output / "device"), "HIP",
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
