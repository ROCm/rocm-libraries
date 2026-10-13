# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric gate for the emulated TF32 atoms (BF16x3).

Runs one wave of each emulated gfx950 atom and answers two questions the host
tests cannot:

1. **Is the expansion faithful?** The emulated ``tile.mma`` is compared, bit
   for bit, against a hand-written kernel that issues the same three BF16
   MFMAs over host-split operands in the same order. Agreement means the
   lowering's split, term order and accumulator threading are what the recipe
   says they are -- on real silicon, not in an IR text check.

2. **Is it accurate?** Against an absolute reference, with the emulation
   landing near fp32 and a plain BF16 MMA of the same operands landing far
   short of it.

Both checks are **layout-free by construction**, which is what makes them
trustworthy without a hand-derived MFMA lane map:

* the differential feeds identical flat fragment buffers to two kernels that
  share the same physical atom, so whatever the lane layout is, both see it;
* the absolute check fills each operand with a single constant, so every
  logical element of A is the same value and every logical element of B is the
  same value. ``D = K * alpha * beta`` at every output position for *any*
  layout -- and ``alpha``/``beta`` are chosen with significand bits below
  bf16's reach, so the split is genuinely exercised rather than trivially
  exact. They also carry bits below a native XF32 instruction's 10-bit
  mantissa, which that instruction would drop: landing within tolerance of the
  exact product is what pins the contract that the emulation drops none.

Usage::

    PYTHONPATH=python python python/rocke/examples/gfx950/xf32/bf16x3_parity.py

Device I/O goes through ``DeviceMem`` + ``Runtime.memcpy_*`` over numpy host
arrays, so the gate needs nothing beyond rocke's own hard dependency. torch is
not installed on every gfx950 node, and a numeric gate that skips on the one
machine that can run it is not a gate.

Exits non-zero if any check fails, so it doubles as a CI gate. It runs on HIP
device 0, the one the launcher uses; on a mixed node, point it at the gfx950
with ``HIP_VISIBLE_DEVICES=<index>``. If device 0 is not a gfx950 it prints
``SKIPPED`` and exits 0 -- the emulated atoms are gfx950-only, and a skip must
not be mistaken for a pass. The test suite runs it through
``tests/core/test_xf32_bf16x3_device.py``, which finds a gfx950 at any index
and sets that variable itself.

The complementary host-side half (what the recipe is worth against fp32,
TF32 and plain bf16, with no GPU) is ``tests/core/test_xf32_bf16x3_numerics.py``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F32, TF32, IRBuilder, KernelDef, PtrType, VectorType
from rocke.core.lower_llvm import lower_kernel_to_llvm

_ARCH = "gfx950"

_EMULATED = ("mfma_f32_16x16x32_xf32", "mfma_f32_32x32x16_xf32")

# Operands for the absolute check. Both round to 1.0 in bf16, so the leading
# term of the split carries no information at all and the answer depends
# entirely on the two residual terms -- which is precisely the part of the
# recipe worth gating. A plain BF16 MMA computes K*1*1 here and is wrong by
# about 2^-11.
_ALPHA = float(np.float32(1.0 + 2.0**-12))
_BETA = float(np.float32(1.0 - 2.0**-11 + 2.0**-19))

# The emulation should land within ~2^-16 of the truth; a single BF16 MMA of
# the same operands should miss by more than 2^-13. The gap between the two
# bounds is wide enough that neither is a coin flip.
_TOL_EMULATED = 2.0**-16
_MIN_BF16_ERROR = 2.0**-13

# The order the lowering uses. Reproduced here rather than imported so that a
# silent change to the lowering shows up as a device mismatch instead of being
# tracked by the reference.
_TERMS = (("a_small", "b_big"), ("a_big", "b_small"), ("a_big", "b_big"))


def _fragment_base(b: IRBuilder, tid, frag_len: int):
    """Where this lane's fragment starts: a flat ``tid * frag_len``.

    Deliberately not a layout map. The host never interprets these offsets --
    it only needs both kernels to use the same ones.
    """
    return b.mul(tid, b.const_i32(frag_len))


def _store_accumulator(b: IRBuilder, d, out_ptr, tid, c_frag_len: int) -> None:
    """Every accumulator lane, element-wise.

    Element-wise rather than one wide store because the 32x32 accumulator is
    16 slots, past what ``global_store_vN`` accepts for f32.
    """
    base = _fragment_base(b, tid, c_frag_len)
    for slot in range(c_frag_len):
        b.store(b.vec_extract(d, slot), out_ptr, b.add(base, b.const_i32(slot)))


def _emulated_kernel(atom) -> KernelDef:
    """One emulated ``tile.mma`` over TF32 operands: the thing under test.

    The fp32 operand buffers are loaded as-is and reinterpreted as the logical
    TF32 type, exactly as they would be for a native XF32 atom.
    """
    b = IRBuilder("xf32_emulated")
    b.kernel.attrs["max_workgroup_size"] = atom.wave_size
    a_ptr = b.param("a", PtrType(F32, "global"))
    b_ptr = b.param("b", PtrType(F32, "global"))
    out_ptr = b.param("out", PtrType(F32, "global"))
    tid = b.thread_id_x()
    a = b.global_load_vN(
        a_ptr, _fragment_base(b, tid, atom.a_frag_len), F32, atom.a_frag_len
    )
    bv = b.global_load_vN(
        b_ptr, _fragment_base(b, tid, atom.b_frag_len), F32, atom.b_frag_len
    )
    a = b.bitcast(a, VectorType(TF32, atom.a_frag_len))
    bv = b.bitcast(bv, VectorType(TF32, atom.b_frag_len))
    c = b.vector_splat(b.const_f32(0.0), atom.c_frag_len)
    _store_accumulator(b, b.mma(atom, a, bv, c), out_ptr, tid, atom.c_frag_len)
    b.ret()
    return b.kernel


def _handwritten_kernel(phys, terms) -> KernelDef:
    """The same recipe, spelled out: N BF16 MFMAs over pre-split operands.

    With ``terms`` set to all three this is the reference the emulation must
    match bit for bit. With just ``(("a_big", "b_big"),)`` it is the plain
    BF16 MMA the emulation is supposed to beat -- same kernel, so the two
    differ only in how many terms they issue.
    """
    b = IRBuilder("xf32_handwritten")
    b.kernel.attrs["max_workgroup_size"] = phys.wave_size
    ptrs = {
        name: b.param(name, PtrType(BF16, "global"))
        for name in ("a_big", "a_small", "b_big", "b_small")
    }
    out_ptr = b.param("out", PtrType(F32, "global"))
    tid = b.thread_id_x()
    frag_len = {"a": phys.a_frag_len, "b": phys.b_frag_len}
    loaded = {
        name: b.global_load_vN(
            ptr, _fragment_base(b, tid, frag_len[name[0]]), BF16, frag_len[name[0]]
        )
        for name, ptr in ptrs.items()
    }
    acc = b.vector_splat(b.const_f32(0.0), phys.c_frag_len)
    for a_name, b_name in terms:
        acc = b.mma(phys, loaded[a_name], loaded[b_name], acc)
    _store_accumulator(b, acc, out_ptr, tid, phys.c_frag_len)
    b.ret()
    return b.kernel


def _to_bf16(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit patterns (``uint16``), round-to-nearest-even.

    Hand-rolled because stock numpy has no ``bfloat16`` and this path is
    deliberately torch- and ``ml_dtypes``-free (see ``runtime/host_buffers``).
    RNE is what the ISA's fp32->bf16 conversion does, so the host split must
    match it or the differential would compare two different recipes. The
    carry into the exponent is handled by the add, which is why this is a
    bias-and-shift rather than a truncate. Infinities and NaNs are out of
    scope -- every operand here is finite by construction.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def _from_bf16(u16: np.ndarray) -> np.ndarray:
    """bf16 bit patterns back to fp32 -- exact, bf16 being a truncated fp32."""
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _split(x: np.ndarray):
    """The host mirror of the lowering's Dekker split, as bf16 bit patterns."""
    big = _to_bf16(x)
    small = _to_bf16(x - _from_bf16(big))
    return big, small


def _launch(kernel: KernelDef, values, *, wave_size: int):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.launcher import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    target = ArchTarget.from_gfx(_ARCH)
    hsaco, _ = build_hsaco_from_llvm_ir(
        lower_kernel_to_llvm(kernel, arch=_ARCH), isa=target.isa_triple
    )
    signature = [{"name": p.name, "type": p.type.name} for p in kernel.params]
    launcher = KernelLauncher(hsaco=hsaco, kernel_name=kernel.name, signature=signature)
    launcher(values, config=LaunchConfig(grid=(1, 1, 1), block=(wave_size, 1, 1)))
    synchronize_and_release()


def _run_atom(op_id: str) -> list[str]:
    """Every failure found for one emulated atom, as human-readable lines."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem

    target = ArchTarget.from_gfx(_ARCH)
    atom = target.mma.by_op_id(op_id)
    phys = target.mma.by_op_id(atom.emulation.physical_op_id)
    failures: list[str] = []
    rt = Runtime()
    # Every device buffer stays alive until the readback: DeviceMem frees on
    # collection, and the kernel is still reading it at launch time.
    owned: list[DeviceMem] = []

    def upload(host: np.ndarray) -> DeviceMem:
        dev = DeviceMem(host.nbytes)
        rt.memcpy_h2d(dev.ptr(), as_u8_buffer(host), host.nbytes)
        owned.append(dev)
        return dev

    def run(kernel, values, *, wave_size: int) -> np.ndarray:
        """Launch over uploaded operands and return the accumulator, as fp32."""
        out = np.zeros(atom.wave_size * atom.c_frag_len, dtype=np.float32)
        out_dev = DeviceMem(out.nbytes)
        rt.memset(out_dev.ptr(), 0, out.nbytes)
        _launch(kernel, dict(values, out=out_dev), wave_size=wave_size)
        rt.memcpy_d2h(as_u8_buffer(out), out_dev.ptr(), out.nbytes)
        return out

    def emulated(a_f32, b_f32):
        return run(
            _emulated_kernel(atom),
            {"a": upload(a_f32), "b": upload(b_f32)},
            wave_size=atom.wave_size,
        )

    def handwritten(a_f32, b_f32, terms):
        """The same recipe spelled out, over operands split on the host."""
        a_big, a_small = _split(a_f32)
        b_big, b_small = _split(b_f32)
        return run(
            _handwritten_kernel(phys, terms),
            {
                "a_big": upload(a_big),
                "a_small": upload(a_small),
                "b_big": upload(b_big),
                "b_small": upload(b_small),
            },
            wave_size=phys.wave_size,
        )

    a_len = atom.wave_size * atom.a_frag_len
    b_len = atom.wave_size * atom.b_frag_len

    # (1) Faithfulness, on random operands. Bitwise, not tolerant: both
    # kernels issue the same three MFMAs over the same bits in the same
    # order, so anything other than equality is a real difference in what
    # the lowering built.
    rng = np.random.default_rng(2208)
    a = rng.standard_normal(a_len, dtype=np.float32)
    b = rng.standard_normal(b_len, dtype=np.float32)
    mismatched = int(np.count_nonzero(emulated(a, b) != handwritten(a, b, _TERMS)))
    print(f"  [{op_id}] expansion vs hand-written BF16x3: {mismatched} slots differ")
    if mismatched:
        failures.append(f"{op_id}: expansion does not match the hand-written recipe")

    # (2) Accuracy, on constant operands, where the truth is K*alpha*beta at
    # every output position regardless of layout.
    a = np.full(a_len, _ALPHA, dtype=np.float32)
    b = np.full(b_len, _BETA, dtype=np.float32)
    got = emulated(a, b)
    bf16_only = handwritten(a, b, (("a_big", "b_big"),))
    truth = float(atom.k) * _ALPHA * _BETA
    emulated_err = float(np.abs(got.astype(np.float64) - truth).max()) / abs(truth)
    bf16_err = float(np.abs(bf16_only.astype(np.float64) - truth).max()) / abs(truth)
    print(
        f"  [{op_id}] vs K*alpha*beta: emulated rel={emulated_err:.3e} "
        f"bf16-only rel={bf16_err:.3e}"
    )
    if not emulated_err < _TOL_EMULATED:
        failures.append(f"{op_id}: emulated error {emulated_err:.3e} exceeds tolerance")
    # The BF16 baseline must actually be bad here, or the check above proves
    # nothing -- a kernel that silently dropped the residual terms would
    # otherwise pass by landing wherever bf16 lands.
    if not bf16_err > _MIN_BF16_ERROR:
        failures.append(
            f"{op_id}: the BF16 baseline was too accurate ({bf16_err:.3e}); "
            "the operands no longer exercise the split"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-device",
        action="store_true",
        help="fail instead of skipping when no gfx950 device is present",
    )
    args = parser.parse_args()

    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch(0)
    except (ImportError, OSError, RuntimeError) as exc:
        # No HIP runtime, no driver, or no device at all -- all of which mean
        # the same thing here: there is nothing to run the gate on.
        arch = f"<unavailable: {exc}>"
    if not str(arch).startswith(_ARCH):
        message = f"no {_ARCH} device (found {arch}); emulated atoms are {_ARCH}-only"
        if args.require_device:
            print(f"FAILED: {message}")
            return 2
        print(f"SKIPPED: {message}")
        return 0

    failures: list[str] = []
    for op_id in _EMULATED:
        failures.extend(_run_atom(op_id))
    if failures:
        for line in failures:
            print(f"FAILED: {line}")
        return 2
    print(f"PASSED: BF16x3 emulation verified on {arch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
