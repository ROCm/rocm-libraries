# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Legacy executable wrappers for the bounded TensileLite migration window."""

from __future__ import annotations

from pathlib import Path
import sys


_warned = False


def _warn(command: str) -> None:
    global _warned
    if _warned:
        return
    command = Path(command).name
    print(
        f"DEPRECATED: {command} is deprecated; use the 'tensilelite' command family. "
        "Legacy aliases will be removed at ROCm 9.0.",
        file=sys.stderr,
    )
    _warned = True


def _canonical(command: str) -> int:
    _warn(sys.argv[0])
    from tensilelite.cli import main

    return main([command, *sys.argv[1:]])


def tensile() -> int:
    return _canonical("run")


def create_library() -> int:
    return _canonical("create-library")


def logic() -> int:
    return _canonical("logic")


def benchmark_cluster():
    _warn(sys.argv[0])
    from tensilelite.benchmark_cluster import main

    return main()


def generate_summations():
    _warn(sys.argv[0])
    from tensilelite.GenerateSummations import GenerateSummations

    return GenerateSummations(sys.argv[1:])


def logic_to_yaml():
    _warn(sys.argv[0])
    from tensilelite.lib_logic_to_yaml import main

    return main()


def merge_library():
    _warn(sys.argv[0])
    from tensilelite.merge_library import main

    return main()


def retune_library():
    _warn(sys.argv[0])
    from tensilelite.retune_library import main

    return main()


def update_library():
    _warn(sys.argv[0])
    from tensilelite.update_library import main

    return main()


def verify_stinky_elf() -> int:
    _warn(sys.argv[0])
    from tensilelite.verify_stinky_comment_vs_elf_text import main

    return main(sys.argv[1:])


def get_path() -> int:
    _warn(sys.argv[0])
    import tensilelite

    print(Path(tensilelite.__file__).resolve().parent, end="")
    return 0
