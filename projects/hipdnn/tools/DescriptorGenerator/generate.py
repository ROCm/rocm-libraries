#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI entry point for the hipDNN descriptor code generator."""

import argparse
import sys
from pathlib import Path

from codegen.config_loader import ConfigError, load_config
from codegen.generator import DescriptorGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate hipDNN operation descriptor boilerplate from YAML configs."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config file (e.g., configs/convolution_fwd.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory (e.g., ../../ to write relative to hipdnn project root)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded config for operation: {config.name}")
    print(f"  Class: {config.class_name}")
    print(f"  FBS table: {config.fbs_table}")
    print(f"  Tensors: {[f.name for f in config.tensor_fields]}")
    print(f"  Data fields: {[f.name for f in config.data_fields]}")

    template_dir = Path(__file__).parent / "templates"
    generator = DescriptorGenerator(template_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        written = generator.render(config, args.output_dir)
    except Exception as e:
        print(f"Template rendering error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nGenerated {len(written)} files:")
    for f in written:
        print(f"  {f}")

    print("\nDone. See CLAUDE.md for post-generation integration steps.")


if __name__ == "__main__":
    main()
