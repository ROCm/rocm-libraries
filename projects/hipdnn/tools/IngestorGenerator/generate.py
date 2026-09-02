#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI entry point for the hipDNN generic-kernel-ingestor descriptor generator."""

import argparse
import sys
from pathlib import Path

from codegen.config_loader import ConfigError, load_config
from codegen.generator import IngestorGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a hipDNN generic-kernel-ingestor descriptor bundle "
        "from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config file (e.g., configs/conv_fwd.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the generated bundle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be generated without writing them or "
        "creating the output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required to overwrite an existing, non-empty output directory. "
        "Without it, an existing non-empty output directory is an error -- the "
        "extend flow points this tool at a live descriptor directory, and an "
        "unconditional overwrite would silently clobber hand-filled content "
        "(e.g. a completed graph_match body).",
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

    print(f"Loaded config for engine: {config.engine.name}")
    print(f"  Packs: {[p.name for p in config.packs]}")
    print(f"  Knobs: {config.engine.knobs}")

    template_dir = Path(__file__).parent / "templates"
    generator = IngestorGenerator(template_dir)

    if args.dry_run:
        files = generator.preview_files(config)
        print(
            f"\nDry run -- would generate {len(files)} files (output dir not created):"
        )
        for f in files:
            print(f"  {f}")
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        print(
            f"Error: Output directory '{args.output_dir}' already exists and is "
            f"non-empty. Pass --force to overwrite (only safe when you intend to "
            f"regenerate over hand-filled content you have already reviewed).",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        written = generator.render(config, args.output_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Template rendering error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nGenerated {len(written)} files:")
    for f in written:
        print(f"  {f}")

    print(
        "\nDone. See README.md for the CMake/registration splice points and the "
        "validate-descriptors round trip."
    )


if __name__ == "__main__":
    main()
