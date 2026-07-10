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
#
# BATCH MODE (argv[1] == "--batch"): the reference oracle is otherwise re-spawned
# once per config, paying a fresh interpreter + `import rocke.core` (~90 ms) for
# every ~3 ms of real emit work. Batch mode imports once and streams every
# config of a family through the SAME `_emit_one` core the per-config path uses,
# emitting a framed manifest the differential harness parses. The per-config path
# is left byte-for-byte identical so it can serve as the harness's isolated
# reference (`--isolated`) and the guard-test oracle.
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


# Default config-enumeration cap for batch mode when the harness does not pass
# one. Mirrors run_diff.MAX_CFG; the harness passes its own value as argv[3] so
# that constant stays single-sourced there.
_BATCH_CAP_DEFAULT = 128


def _emit_one(spec_fn, build_fn, idx, mode, default_arch):
    """Compute the emitted text for one (idx, mode) -- the single definition of
    what a config emits, shared by the per-config and batch paths so both are
    byte-identical by construction.

    spec_fn(idx) returns either a spec or a (spec, arch) tuple; a bare spec uses
    ``default_arch``, a tuple supplies its own arch. spec_fn raises SystemExit
    with an "unknown config" message when idx is past the sampled range; that
    propagates to the caller, which treats it as end-of-range.

    The "ll" mode deliberately uses the NATIVE Python lowerer
    (``_lower_kernel_to_llvm_python``) rather than the backend-dispatching
    ``lower_kernel_to_llvm``: this driver is the differential gate's PYTHON
    REFERENCE oracle, so it must produce the Python engine's .ll regardless of
    the package-default backend (which may now be the C++ engine). Pinning the
    native lowerer keeps the gate a true python-vs-cpp comparison after the
    default flip.
    """
    selected = spec_fn(idx)
    if isinstance(selected, tuple):
        # A config either uses the default arch or pins its own (an arch-specific
        # kernel). Byte-identity is a property of this exact (spec, arch): both
        # engines must agree, including agreeing to reject a (spec, arch) that
        # the target does not support. There is no global arch override.
        spec, arch = selected
    else:
        spec, arch = selected, default_arch
    kernel = build_fn(spec, arch=arch)
    if mode == "ll":
        return _native_lower(kernel, arch=arch)
    if mode == "ir":
        return serialize(kernel)
    return "".join(str(d) + "\n" for d in verify(kernel))  # verify


def _write_frame(out, idx, rc, data):
    """Emit one manifest frame: an ASCII header line then the raw output bytes
    then a separator newline, so arbitrary (binary-safe) output round-trips."""
    out.write(b"IDX %d %d %d\n" % (idx, rc, len(data)))
    out.write(data)
    out.write(b"\n")


def _run_batch(spec_fn, build_fn, mode, default_arch, cap):
    """Stream a framed manifest for every config of one family, importing the
    engine once. Each config is run through the same ``_emit_one`` core as the
    per-config path, so the emitted bytes are identical; only the number of
    interpreter starts differs.

    A config that raises unknown-config ends the family (an "END <idx>" line,
    matching the per-config end-of-range sentinel). Any other failure is a
    per-config rejection -- nonzero rc, empty output -- exactly as an isolated
    process would report it, and enumeration continues (a rejected config does
    not stop the family; only end-of-range does).

    Crash isolation is intentionally NOT provided here: the Python engine is the
    pure reference oracle (it does not exercise the UB the harness hunts for on
    the C++ side), so batching it loses no CRASH signal. The C side stays
    per-config precisely so its SIGSEGVs are still caught as CRASH.
    """
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    out = sys.stdout.buffer
    for idx in range(cap):
        try:
            text = _emit_one(spec_fn, build_fn, idx, mode, default_arch)
        except SystemExit as e:
            msg = "" if e.code is None else str(e.code)
            if "unknown config" in msg.lower():
                out.write(b"END %d\n" % idx)
                out.flush()
                return 0
            rc = e.code if isinstance(e.code, int) else 1
            _write_frame(out, idx, rc, b"")
            continue
        except Exception:  # noqa: BLE001 - a rejection, faithful to per-config
            _write_frame(out, idx, 1, b"")
            continue
        _write_frame(out, idx, 0, text.encode("utf-8"))
    # Reached the cap without end-of-range: the per-config harness likewise never
    # probes past `cap`, so this END is beyond what it will read (harmless).
    out.write(b"END %d\n" % cap)
    out.flush()
    return 0


def run_emit(spec_fn, build_fn, *, usage=None, arch="gfx950"):
    """Drive one parity emitter and return its process exit code.

    Per-config invocation: argv[1] is the config index, argv[2] the optional mode
    (default "ll"). Batch invocation: argv[1] == "--batch", argv[2] the mode,
    argv[3] the optional enumeration cap -- emits a framed manifest for all
    configs (see ``_run_batch``).
    """
    if len(sys.argv) < 2:
        sys.stderr.write(usage or "usage: <config_index> [ll|ir|verify]\n")
        return 2
    if sys.argv[1] == "--batch":
        mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
        cap = int(sys.argv[3]) if len(sys.argv) > 3 else _BATCH_CAP_DEFAULT
        return _run_batch(spec_fn, build_fn, mode, arch, cap)
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    sys.stdout.write(_emit_one(spec_fn, build_fn, idx, mode, arch))
    return 0
