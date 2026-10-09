#!/usr/bin/env python3
################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""List stable set-cover selectors for a Tensile BenchmarkProblems YAML."""

import argparse

from config_harness import benchmark_problem_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Tensile configuration YAML")
    args = parser.parse_args()
    for index, fingerprint in benchmark_problem_fingerprints(args.config):
        print(f"{index}: {fingerprint}")


if __name__ == "__main__":
    main()
