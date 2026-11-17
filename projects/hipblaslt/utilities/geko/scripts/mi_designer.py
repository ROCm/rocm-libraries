#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI entry point for the MI designer.

Usage:
    python3 mi_designer.py config.yaml -o ./MatInst | tee configGen.log
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from geko.config_generator.load_input_config import load_prepared_config_from_yaml
from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.sizes import get_sizes


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MI designer."""
    argParser = argparse.ArgumentParser(
        description="Generate MatrixInstruction configurations for GEMM tuning",
    )

    argParser.add_argument(
        "config",
        type=str,
        default='./config.yaml',
        help="path to the input config file",
    )
    argParser.add_argument(
        "--outputfile", "-o",
        type=str,
        default='./MatInst',
        help="Output file/directory. It can take absolute/relative path.",
    )
    argParser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="enable debug logging",
    )

    return argParser.parse_args()


def main() -> None:
    """CLI entry point: load config, iterate over sizes, call generate_for_size."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    config = load_prepared_config_from_yaml(args.config)

    mi_finder_log_path = os.path.join(args.outputfile, "MI_finder_log")
    Path(mi_finder_log_path).mkdir(parents=True, exist_ok=True)

    mi_designer = MIDesign(mi_finder_log_path, config)

    sizes = get_sizes(config)
    for size in sizes:
        M, N, B, K = size
        groups = mi_designer.generate_for_size(tuple(size))
        print(f"Size ({M}, {N}, {B}, {K}): {len(groups)} MI groups generated")


if __name__ == "__main__":
    main()
