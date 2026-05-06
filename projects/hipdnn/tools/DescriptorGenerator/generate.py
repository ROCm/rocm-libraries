#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI entry point for the hipDNN descriptor code generator."""

import argparse
import sys
from pathlib import Path

from codegen.config_loader import ConfigError, load_config, validate_for_mode
from codegen.generator import DescriptorGenerator

# Generation modes
MODE_BACKEND = "backend"
MODE_FRONTEND = "frontend"
MODE_FULL = "full"
MODE_LIFT_ONLY = "lift-only"
VALID_MODES = (MODE_BACKEND, MODE_FRONTEND, MODE_FULL, MODE_LIFT_ONLY)

# Modes that require frontend config fields
FRONTEND_MODES = (MODE_FRONTEND, MODE_FULL)


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
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=MODE_BACKEND,
        help="Generation mode: 'backend' (default, current behavior), "
        "'frontend' (frontend-only), 'full' (everything), "
        "'lift-only' (lifting files only)",
    )
    parser.add_argument(
        "--lift-only",
        action="store_true",
        help="(Deprecated) Alias for --mode lift-only. "
        "Generate only lifting-related files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be generated without writing them.",
    )
    args = parser.parse_args()

    # Handle deprecated --lift-only flag
    if args.lift_only:
        if args.mode != MODE_BACKEND:
            print(
                "Error: --lift-only cannot be combined with --mode. "
                "Use --mode lift-only instead.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            "Warning: --lift-only is deprecated. Use --mode lift-only instead.",
            file=sys.stderr,
        )
        args.mode = MODE_LIFT_ONLY

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate frontend fields when mode requires frontend generation
    if args.mode in FRONTEND_MODES:
        try:
            validate_for_mode(config, args.mode)
        except ConfigError as e:
            print(f"Config error: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded config for operation: {config.name}")
    print(f"  Class: {config.class_name}")
    print(f"  FBS table: {config.fbs_table}")
    print(f"  Tensors: {[f.name for f in config.tensor_fields]}")
    print(f"  Data fields: {[f.name for f in config.data_fields]}")
    print(f"  Mode: {args.mode}")

    if args.mode in FRONTEND_MODES:
        print(f"  Frontend inputs: " f"{[t.name for t in config.frontend.inputs]}")
        print(f"  Frontend outputs: " f"{[t.name for t in config.frontend.outputs]}")

    template_dir = Path(__file__).parent / "templates"
    generator = DescriptorGenerator(template_dir)

    if args.dry_run:
        files = _preview_files(config, args.mode)
        print(f"\nDry run: {len(files)} files would be generated:")
        for f in files:
            print(f"  {f}")
        print("\nNo files were written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        written = generator.render(config, args.output_dir, args.mode)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Template rendering error: {e}", file=sys.stderr)
        sys.exit(1)

    mode_label = {
        MODE_BACKEND: "backend",
        MODE_FRONTEND: "frontend",
        MODE_FULL: "full",
        MODE_LIFT_ONLY: "lifting",
    }.get(args.mode, "")
    print(f"\nGenerated {len(written)} {mode_label} files:")
    for f in written:
        print(f"  {f}")

    print("\nDone. See CLAUDE.md for post-generation integration steps.")


def _preview_files(config, mode: str) -> list[str]:
    """Return the list of file paths that would be generated for the given mode."""
    files = []

    # Backend files
    backend_files = [
        f"backend/src/descriptors/{config.header_filename}",
        f"backend/src/descriptors/{config.source_filename}",
        f"frontend/include/hipdnn_frontend/detail/{config.packer_filename}",
        f"frontend/include/hipdnn_frontend/detail/{config.unpacker_filename}",
        f"backend/tests/descriptors/{config.test_descriptor_filename}",
        f"backend/tests/descriptors/{config.test_graph_filename}",
        f"backend/tests/descriptors/{config.test_from_node_filename}",
        f"tests/frontend/{config.test_integration_filename}",
        f"tests/frontend/{config.test_integration_lifting_filename}",
    ]
    backend_fragments = [
        "fragments/attribute_enum_block.txt",
        "fragments/descriptor_type_enum.txt",
        "fragments/string_utils_block.txt",
        "fragments/factory_case.txt",
        "fragments/cmake_entries.txt",
        "fragments/node_factory_case.txt",
        "fragments/operation_unpacker_case.txt",
        "fragments/operation_unpacker_test.txt",
        "fragments/operation_type_enum.txt",
        "fragments/node_unpack_override.txt",
    ]

    # Frontend files
    frontend_files = [
        f"frontend/include/hipdnn_frontend/attributes/{config.attributes_header_filename}",
        f"frontend/include/hipdnn_frontend/node/{config.node_header_filename}",
    ]
    frontend_test_files = [
        f"frontend/tests/{config.test_attributes_filename}",
        f"frontend/tests/{config.test_node_filename}",
        f"frontend/tests/{config.test_frontend_graph_filename}",
    ]
    frontend_fragments = [
        "fragments/graph_method.txt",
        "fragments/graph_includes.txt",
        "fragments/frontend_cmake_entries.txt",
        "fragments/node_type_enum.txt",
    ]

    # Lift-only files
    lift_files = [
        f"frontend/include/hipdnn_frontend/detail/{config.unpacker_filename}",
        f"backend/tests/descriptors/{config.test_from_node_filename}",
        f"tests/frontend/{config.test_integration_lifting_filename}",
    ]
    lift_fragments = [
        "fragments/node_factory_case.txt",
        "fragments/operation_unpacker_case.txt",
        "fragments/operation_unpacker_test.txt",
        "fragments/operation_type_enum.txt",
        "fragments/node_unpack_override.txt",
        "fragments/descriptor_lifting_additions.txt",
        "fragments/packer_name_addition.txt",
    ]

    if mode == MODE_BACKEND:
        files = backend_files + backend_fragments
    elif mode == MODE_FRONTEND:
        files = frontend_files + frontend_test_files + frontend_fragments
    elif mode == MODE_FULL:
        files = (
            backend_files
            + backend_fragments
            + frontend_files
            + frontend_test_files
            + frontend_fragments
        )
    elif mode == MODE_LIFT_ONLY:
        files = lift_files + lift_fragments

    # Mode enum files (dynamic based on config)
    if mode in (MODE_BACKEND, MODE_FULL):
        for df in config.generatable_mode_fields:
            files.append(f"backend/include/{df.enum_def.backend_header}")
            files.append(f"fragments/mode_backend_plumbing_{df.name}.txt")
            files.append(f"fragments/mode_frontend_plumbing_{df.name}.txt")
            files.append(f"fragments/mode_frontend_tests_{df.name}.txt")

    # Constants file (only when no pre-existing constants header)
    if not config.test_data.constants_include:
        constants_path = f"test_sdk/include/hipdnn_test_sdk/constants/{config.effective_constants_include}.hpp"
        if mode in (MODE_BACKEND, MODE_FULL, MODE_LIFT_ONLY):
            files.append(constants_path)

    return files


if __name__ == "__main__":
    main()
