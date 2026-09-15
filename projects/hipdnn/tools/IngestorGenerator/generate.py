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
    parser.add_argument(
        "--check-placeholders",
        action="store_true",
        help="Do not generate. Report unfilled stub markers in the files this "
        "config's engine emits, searched under --emitted-root (default: "
        "--output-dir), and exit 1 if any remain or if any of them cannot be "
        "located. It derives the file set from the config, so it cannot miss a "
        "file a hand-written glob forgot.",
    )
    parser.add_argument(
        "--emitted-root",
        action="append",
        dest="emitted_roots",
        default=[],
        type=Path,
        help="A root the provider spliced this engine's emitted files into. "
        "Repeatable, because the provider SPLITS the bundle: packs land in the "
        "engine directory and the generated test stubs under "
        "src/tests/engines/.../packs/, which may share no ancestor worth "
        "scanning. Each root is searched only at the engine-specific relative "
        "paths this generator's own CMake fragments splice to, and a file found "
        "at the same relative path under two roots is an error rather than a "
        "pick. Omit to search --output-dir alone. A root that does not exist is "
        "an error, never an empty search.",
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

    if args.check_placeholders:
        expected = generator.preview_files(config)
        # No --emitted-root means the output dir IS the one root, so the existing
        # single-tree invocation keeps meaning exactly what it says.
        roots = args.emitted_roots or [args.output_dir]
        shown = ", ".join(f"'{root}'" for root in roots)
        try:
            located, missing, ambiguous = generator.locate_emitted(roots, expected)
            unfilled = generator.unfilled_placeholders(roots, expected)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        total_shippable = len(located) + len(missing) + len(ambiguous)
        print(
            f"\nLocated {len(located)} of this engine's {total_shippable} shippable "
            f"files under {shown}."
        )
        failed = False
        if ambiguous:
            # One relative path resolving twice: the scan cannot tell which is
            # yours, and picking one made the verdict depend on directory order --
            # a filled stale copy would report green while the real file kept its
            # markers. Roots are unioned first, so a second root holding the same
            # relative path is caught here rather than silently preferred.
            print(f"{len(ambiguous)} file(s) matched in more than one place:")
            for rel, paths in ambiguous.items():
                print(f"        {rel}")
                for p in paths:
                    print(f"            {p}")
            print(
                "  Narrow the roots so each file resolves once (a build tree or a "
                "stale copy under one of them is the usual cause)."
            )
            failed = True
        if missing:
            # A file the engine ships but nobody can find is an UNFINISHED SPLICE,
            # not a pass. "No unfilled placeholders" across a set the scan never
            # located is the exact false green this gate exists to prevent, so a
            # located set that does not cover every shippable file fails here even
            # when every file it DID read was clean.
            print(
                f"{len(missing)} expected file(s) not found under any of those roots:"
            )
            for rel in missing:
                print(f"        {rel}")
            print(
                "  Add an --emitted-root for the tree holding them (the provider "
                "splits the engine and its tests; the cmake_test_sources fragment "
                "says where), or finish the splice."
            )
            failed = True
        if unfilled:
            total = sum(unfilled.values())
            print(f"{total} unfilled placeholder(s) across {len(unfilled)} file(s):")
            for rel, count in unfilled.items():
                print(f"  {count:4}  {rel}")
            failed = True
        if failed:
            sys.exit(1)
        print(f"No unfilled placeholders across all {total_shippable} located files.")
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

    # A fresh generation wrote everything into one tree, so that tree is the root.
    unfilled = generator.unfilled_placeholders([args.output_dir], written)
    if unfilled:
        total = sum(unfilled.values())
        print(
            f"\n{total} unfilled placeholder(s) across {len(unfilled)} file(s) -- "
            "each one is a hook body you owe:"
        )
        for rel, count in unfilled.items():
            print(f"  {count:4}  {rel}")
        print(
            "  Re-run this command after implementing them; zero here is the "
            "step-6 gate. Note the generated tests/ stubs count too."
        )

    print(
        "\nDone. See README.md for the CMake/registration splice points and the "
        "validate-descriptors round trip."
    )


if __name__ == "__main__":
    main()
