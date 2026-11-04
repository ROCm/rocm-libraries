#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Render the combined engine support matrix from committed claim sidecars.

Implements the renderer described by RFC 0015 §11
(``projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md``): it discovers every
support sidecar under the bundle tree, joins each to the graph it claims, and
projects the result.

The matrix is a *view* of the committed claims, never a second state to
maintain. Generation is hardware-free: only ``.json`` / ``.support.json`` files
are read, so a plain checkout with no GPU, no plugin, and no DVC pull can
regenerate it.

The work is split three ways, so each part can be read and tested alone:

- ``support_matrix_units.py`` -- collection: walks the bundle tree (through the
  shared discovery mirror in ``bundle_discovery.py``), joins each sidecar to the
  graph it claims, and returns one ``ClaimUnit`` per graph.
- ``support_matrix_markdown.py`` -- rendering: projects the units into the
  markdown document. It never reads the bundle tree, so another output format
  is another module over the same units.
- this file -- the command line: paths, arguments, ``--check``, and refusing
  to write anything past a bundle defect.

The full document, ``SUPPORT_MATRIX.md``, is **generated, not committed** --
see ``dnn-providers/integration-tests/.gitignore`` for why, and RFC 0015 §11.
So is the compact ``--overview-only`` render,
``SUPPORT_MATRIX_GRAPH_OVERVIEW.md``. The committed overview,
``SUPPORT_MATRIX_OVERVIEW.md``, counts API options rather than graphs and is
rendered by ``support_features/feature_matrix.py``.

Usage:

    python3 dnn-providers/integration-tests/scripts/render_support_matrix.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bundle_discovery import DEFAULT_BUNDLES_DIR  # noqa: E402
from support_matrix_markdown import (  # noqa: E402
    REGEN_COMMAND,
    REGEN_OVERVIEW_COMMAND,
    render_markdown,
)
from support_matrix_units import collect_units, warn, write_stdout  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_PATH = pathlib.Path(__file__).resolve()
INTEGRATION_TESTS_DIR = SCRIPT_PATH.parent.parent
DEFAULT_MATRIX_PATH = INTEGRATION_TESTS_DIR / "SUPPORT_MATRIX.md"
DEFAULT_OVERVIEW_PATH = INTEGRATION_TESTS_DIR / "SUPPORT_MATRIX_GRAPH_OVERVIEW.md"

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the combined engine support matrix from committed "
            "support-claim sidecars (RFC 0015 §11)."
        )
    )
    parser.add_argument(
        "--bundles-dir",
        type=pathlib.Path,
        default=DEFAULT_BUNDLES_DIR,
        help=f"Bundle tree to scan (default: {DEFAULT_BUNDLES_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_MATRIX_PATH,
        help=(
            f"Write here (default: {DEFAULT_MATRIX_PATH}, or "
            f"{DEFAULT_OVERVIEW_PATH} with --overview-only). Use '-' for stdout. "
            "The default is a file rather than stdout because the document is "
            "large, and producing it is the point of running this."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write. Re-render and compare against an existing file "
            "(--output, or the default output path); exit 1 if they differ or "
            "it is missing. Neither render is committed, so this only keeps "
            "a local copy in sync."
        ),
    )
    parser.add_argument(
        "--max-case-ids",
        type=int,
        default=0,
        help=(
            "Inline this many sweep case ids per bundle in the markdown "
            "traceability comments (0 = none, the default; negative = all). "
            "The bundle path is always emitted."
        ),
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help=(
            "Emit only the per-target overview tables (no per-family detail "
            "sections, no collapsible wrappers)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.overview_only and args.output == DEFAULT_MATRIX_PATH:
        args.output = DEFAULT_OVERVIEW_PATH

    root = args.bundles_dir
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    errors: list[str] = []
    units = collect_units(root, errors)
    if errors:
        # Each was printed as found. Writing (or blessing) a matrix past them
        # would publish claims the C++ harness refuses to load.
        print(
            f"error: {len(errors)} bundle defect(s); nothing rendered",
            file=sys.stderr,
        )
        return 1
    if not units:
        warn(f"{root}: no claim-bearing bundles found")

    document = render_markdown(
        units, args.max_case_ids, overview_only=args.overview_only
    )

    if args.check:
        target = args.output
        regen_cmd = REGEN_OVERVIEW_COMMAND if args.overview_only else REGEN_COMMAND
        if str(target) == "-":
            print("error: --check needs a file to compare against", file=sys.stderr)
            return 2
        try:
            committed = target.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {target} ({exc})", file=sys.stderr)
            print(f"       regenerate it with: {regen_cmd}", file=sys.stderr)
            return 1
        if committed != document:
            print(
                f"error: {target} is stale -- it does not match a re-render of "
                f"the current sidecars",
                file=sys.stderr,
            )
            print(f"       regenerate it with: {regen_cmd}", file=sys.stderr)
            return 1
        return 0

    if str(args.output) == "-":
        write_stdout(document)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    # To stderr, so `--output -` piping stays clean and this stays out of the
    # document. The file is git-ignored, so saying where it went is the only
    # way the reader learns it exists.
    print(
        f"wrote {args.output} ({len(document.encode('utf-8')):,} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
