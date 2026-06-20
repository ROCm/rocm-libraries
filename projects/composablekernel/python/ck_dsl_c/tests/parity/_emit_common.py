# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/_emit_common.py -- shared driver for the Python reference
# emitters. Every <family>_emit.py parses the same argv (config index in
# argv[1], optional mode in argv[2] defaulting to "ll"), then dispatches on the
# mode: "ll" prints lower_kernel_to_llvm, "ir" prints the ck.dsl.ir/v1
# serialization, "verify" prints verifier diagnostics, and any other mode is a
# usage error. run_emit() centralizes that boilerplate so each emitter only has
# to provide its config selector and kernel builder.
import sys

from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def run_emit(spec_fn, build_fn, *, usage=None, arch="gfx950"):
    """Drive one parity emitter and return its process exit code.

    spec_fn(idx) returns either a spec or a (spec, arch) tuple; a bare spec uses
    the `arch` argument (default "gfx950"), while a tuple supplies its own arch.
    build_fn(spec, arch=arch) returns the kernel. The selected mode (argv[2], default
    "ll") chooses lower_kernel_to_llvm, serialize, or verifier-diagnostic
    output; an unrecognized mode is rejected.
    """
    if len(sys.argv) < 2:
        sys.stderr.write(usage or "usage: <config_index> [ll|ir|verify]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    selected = spec_fn(idx)
    if isinstance(selected, tuple):
        spec, arch = selected
    else:
        spec = selected
    kernel = build_fn(spec, arch=arch)
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch=arch)
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0
