#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pin the Linux runner used by rocJITsu jobs.

rocJITsu does not follow select_build_runner()'s default CPU pool. Those
jobs run on the sanitizer scale set below.
"""

import os

ROCJITSU_RUNS_ON = "az-linux-scale-rocm-asan-prod-2"


def select_linux_build_runner() -> str:
    return ROCJITSU_RUNS_ON


def main() -> None:
    label = select_linux_build_runner()
    print(f"build_runs_on={label}")
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"build_runs_on={label}\n")


if __name__ == "__main__":
    main()
