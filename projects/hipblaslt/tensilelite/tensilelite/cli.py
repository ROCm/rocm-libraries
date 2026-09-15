# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Command dispatcher for the supported TensileLite command surface."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import sys


_COMMAND_HELP = {
    "benchmark-cluster": "run the distributed benchmark workflow",
    "create-library": "generate device libraries from library logic",
    "generate-summations": "generate summation models for library logic",
    "logic": "validate library logic",
    "logic-to-yaml": "convert library logic to a Tensile configuration",
    "merge-library": "merge tuned library logic without regressions",
    "retune-library": "retune an existing library logic file",
    "run": "run the benchmark and tuning workflow",
    "update-library": "update library logic files",
}


def _invoke_sys_argv_main(
    entry_point: Callable[[], int | None], command: str, argv: Sequence[str] | None
) -> int | None:
    """Forward dispatcher arguments to an entry point that reads ``sys.argv``."""
    previous_argv = sys.argv
    try:
        sys.argv = [f"tensilelite {command}", *(argv or [])]
        return entry_point()
    finally:
        sys.argv = previous_argv


def _handler(command: str) -> Callable[[Sequence[str] | None], int | None]:
    if command == "benchmark-cluster":
        from .benchmark_cluster import main

        return lambda argv: _invoke_sys_argv_main(main, command, argv)
    if command == "create-library":
        from .tensilelite_create_library.run import run

        return run
    if command == "generate-summations":
        from .GenerateSummations import GenerateSummations

        return GenerateSummations
    if command == "logic":
        from .tensilelite_logic.run import main

        return main
    if command == "logic-to-yaml":
        from .lib_logic_to_yaml import main

        return lambda argv: _invoke_sys_argv_main(main, command, argv)
    if command == "merge-library":
        from .merge_library import main

        return lambda argv: _invoke_sys_argv_main(main, command, argv)
    if command == "retune-library":
        from .retune_library import main

        return lambda argv: _invoke_sys_argv_main(main, command, argv)
    if command == "run":
        from .tensilelite import main

        return main
    if command == "update-library":
        from .update_library import main

        return lambda argv: _invoke_sys_argv_main(main, command, argv)
    raise AssertionError(f"unhandled command: {command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensilelite",
        description="TensileLite generator, validation, and tuning tools.",
    )
    parser.add_argument("command", nargs="?", choices=tuple(_COMMAND_HELP), help="command to run")
    return parser


def _print_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print("\ncommands:")
    width = max(map(len, _COMMAND_HELP))
    for name, description in _COMMAND_HELP.items():
        print(f"  {name:<{width}}  {description}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not args or args[0] in {"-h", "--help"}:
        _print_help(parser)
        return 0
    if args[0] == "--version":
        from . import __version__

        print(__version__)
        return 0
    command = args.pop(0)
    if command not in _COMMAND_HELP:
        parser.error(f"invalid command: {command!r}")
    result = _handler(command)(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
