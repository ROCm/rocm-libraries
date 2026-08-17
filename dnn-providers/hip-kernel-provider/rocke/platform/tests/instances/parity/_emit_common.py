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

from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify

# The "ll" mode must emit the NATIVE Python engine's .ll regardless of the
# package-default backend (this driver is the python reference oracle). On
# trees that expose the native lowerer directly (post-flip), use it; on older
# reference trees that lack it, the public lower_kernel_to_llvm IS the native
# lowerer (no backend dispatch there), so fall back to it.
try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower


def run_emit(spec_fn, build_fn, *, usage=None, arch="gfx950"):
    """Drive one parity emitter and return its process exit code.

    spec_fn(idx) returns either a spec or a (spec, arch) tuple; a bare spec uses
    the `arch` argument (default "gfx950"), while a tuple supplies its own arch.
    build_fn(spec, arch=arch) returns the kernel. The selected mode (argv[2], default
    "ll") chooses the native Python lowerer, serialize, or verifier-diagnostic
    output; an unrecognized mode is rejected.

    The "ll" mode deliberately uses the NATIVE Python lowerer
    (``_lower_kernel_to_llvm_python``) rather than the backend-dispatching
    ``lower_kernel_to_llvm``: this driver is the differential gate's PYTHON
    REFERENCE oracle, so it must produce the Python engine's .ll regardless of
    the package-default backend (which may now be the C++ engine). Pinning the
    native lowerer keeps the gate a true python-vs-cpp comparison after the
    default flip.
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
        # A config either uses the default arch or pins its own (an arch-specific
        # kernel). Byte-identity is a property of this exact (spec, arch): both
        # engines must agree, including agreeing to reject a (spec, arch) that
        # the target does not support. There is no global arch override.
        spec, arch = selected
    else:
        spec = selected
    kernel = build_fn(spec, arch=arch)
    if mode == "ll":
        text = _native_lower(kernel, arch=arch)
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0
