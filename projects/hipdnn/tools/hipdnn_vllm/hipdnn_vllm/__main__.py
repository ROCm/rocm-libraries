# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``hipdnn-vllm <command>`` / ``python -m hipdnn_vllm <command>``.
Commands: ``doctor``, ``probe``."""

from __future__ import annotations

import sys

USAGE = (
    "usage: hipdnn-vllm doctor [--device N] [--json]\n"
    "       hipdnn-vllm probe --model <hf-id> [--dtype D] [--device N] [--json]"
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "doctor":
        from .doctor import main as doctor_main

        return doctor_main(rest)
    if cmd == "probe":
        from .probe import main as probe_main

        return probe_main(rest)
    print(f"unknown command {cmd!r}; available: doctor, probe", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
