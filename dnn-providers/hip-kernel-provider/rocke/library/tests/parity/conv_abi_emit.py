#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_abi_emit.py -- Python reference emitter for the conv
# kernarg ABI parity family. Prints one ABI list (or launch signature) per
# config index, one "name kind" line per entry, so run_diff.py can byte-compare
# it with conv_abi_emit.c. The conv builders emit their params from these
# lists, so the kernel families already cover the variants they build; this
# family also covers the ones no builder reaches yet (3-D dgrad) and the
# rejected inputs.
#
# Config index map:
#   0  -- conv_arg_names("fwd")
#   1  -- conv_arg_names("fwd", is_3d)
#   2  -- conv_arg_names("wgrad")
#   3  -- conv_arg_names("wgrad", is_3d)
#   4  -- conv_arg_names("wgrad", two_stage)
#   5  -- conv_arg_names("wgrad", is_3d, two_stage)
#   6  -- conv_arg_names("dgrad")
#   7  -- conv_arg_names("dgrad", is_3d)
#   8  -- conv_direct_arg_names("fwd")
#   9  -- conv_direct_arg_names("dgrad")
#   10 -- conv_fwd_problem_block()
#   11 -- conv_fwd_problem_block(is_3d)
#   12 -- conv_arg_names("fwd", two_stage)       (rejected)
#   13 -- conv_arg_names("bwd")                  (rejected)
#   14 -- conv_direct_arg_names("wgrad")
#   15 -- conv_direct_arg_names("bwd")           (rejected)
import sys

from kernels.common.conv_abi import (
    conv_arg_names,
    conv_direct_arg_names,
    conv_fwd_problem_block,
)


_CONFIGS = {
    0: lambda: conv_arg_names(direction="fwd"),
    1: lambda: conv_arg_names(direction="fwd", is_3d=True),
    2: lambda: conv_arg_names(direction="wgrad"),
    3: lambda: conv_arg_names(direction="wgrad", is_3d=True),
    4: lambda: conv_arg_names(direction="wgrad", two_stage=True),
    5: lambda: conv_arg_names(direction="wgrad", is_3d=True, two_stage=True),
    6: lambda: conv_arg_names(direction="dgrad"),
    7: lambda: conv_arg_names(direction="dgrad", is_3d=True),
    8: lambda: conv_direct_arg_names(direction="fwd"),
    9: lambda: conv_direct_arg_names(direction="dgrad"),
    10: lambda: conv_fwd_problem_block(),
    11: lambda: conv_fwd_problem_block(is_3d=True),
    12: lambda: conv_arg_names(direction="fwd", two_stage=True),
    13: lambda: conv_arg_names(direction="bwd"),
    14: lambda: conv_direct_arg_names(direction="wgrad"),
    15: lambda: conv_direct_arg_names(direction="bwd"),
}


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: conv_abi_emit.py <config_index> [ll]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode != "ll":
        # The lists are not IR; there is nothing to serialize or verify.
        sys.stderr.write(f"unknown mode {mode!r}\n")
        return 2
    if idx not in _CONFIGS:
        sys.stderr.write(f"unknown config index {idx}\n")
        return 2
    try:
        items = _CONFIGS[idx]()
    except ValueError:
        sys.stdout.write("REJECTED\n")
        return 0
    sys.stdout.write("".join(f"{name} {kind}\n" for name, kind in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
