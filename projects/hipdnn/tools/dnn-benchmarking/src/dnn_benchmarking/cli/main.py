# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import os
import sys

# Redirect ROCm/tool caches away from the network home directory before any
# ROCm library is imported. MIOpen, comgr, pip, and torch all default to
# ~/.cache/ or ~/.miopen/, which is a network filesystem on AMD dev machines.
# Only set each variable if the user has not already overridden it.
_LOCAL_CACHE_DEFAULTS = {
    "XDG_CACHE_HOME": "/workspace/cache",  # pip, torch, black, all XDG-aware tools
    "MIOPEN_USER_DB_PATH": "/workspace/miopen_cache",  # MIOpen user kernel DB
    "MIOPEN_CUSTOM_CACHE_DIR": "/workspace/miopen_cache",  # MIOpen find-db / system DB
    "AMD_COMGR_CACHE_DIR": "/workspace/comgr_cache",  # ROCm compiler cache
}
for _var, _default in _LOCAL_CACHE_DEFAULTS.items():
    os.environ.setdefault(_var, _default)
from pathlib import Path

from ..common.exceptions import GraphLoadError
from ..reporting.reporter import Reporter
from .ab_runner_cli import run_ab_cli
from .parser import create_parser
from .pytorch_runner_cli import run_pytorch_cli
from .suite_runner_cli import run_suite_cli
from .gpu_check import gpu_is_available


def _resolve_graphs(args, reporter: Reporter):
    """Resolve --graph args to file paths. Returns (tmpdirs, files, tarball_source)."""
    from ..graph.resolver import is_tarball as _is_tarball, resolve_graph_files_multi

    if len(args.graph) == 1 and _is_tarball(args.graph[0]):
        reporter.print_extracting(args.graph[0])

    try:
        tmpdirs, files, tarball_source = resolve_graph_files_multi(args.graph)
    except GraphLoadError as e:
        reporter.print_error(str(e))
        return None, None, None

    if tmpdirs:
        reporter.print_extracted_count(len(files), str(args.graph))

    if not files:
        for td in tmpdirs:
            td.cleanup()
        reporter.print_no_graphs_found(str(args.graph))
        return tmpdirs, None, None

    return tmpdirs, files, tarball_source


def main() -> int:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    reporter = Reporter()

    tmpdirs, resolved_files, tarball_source = _resolve_graphs(args, reporter)
    if resolved_files is None:
        return 1

    if not gpu_is_available():
        reporter.print_error(
            "No GPU detected. A GPU with ROCm or CUDA support is required."
        )
        for td in tmpdirs:
            td.cleanup()
        return 1

    try:
        if args.AId is not None or args.BId is not None:
            if len(resolved_files) > 1:
                reporter.print_error(
                    "A/B testing requires a single graph file, not a glob pattern"
                )
                return 1
            return run_ab_cli(args, Path(resolved_files[0]), reporter)

        elif args.backend == "pytorch":
            if len(resolved_files) > 1:
                reporter.print_error(
                    "Suite mode is not supported with --backend pytorch"
                )
                return 1
            return run_pytorch_cli(args, Path(resolved_files[0]), reporter)

        else:
            return run_suite_cli(
                args,
                graph_paths=[Path(p) for p in resolved_files],
                reporter=reporter,
                tarball_source=tarball_source,
            )
    finally:
        if tmpdirs:
            for td in tmpdirs:
                td.cleanup()


if __name__ == "__main__":
    sys.exit(main())
