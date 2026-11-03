# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric gate for the emulated TF32 atoms (BF16x3).

Runs one wave of each emulated gfx950 atom and answers three questions the host
tests cannot:

1. **Is the expansion faithful?** The emulated ``tile.mma`` is compared, bit
   for bit, against a hand-written kernel that issues the same three BF16
   MFMAs over host-split operands in the same order. Agreement means the
   lowering's split, term order and accumulator threading are what the recipe
   says they are -- on real silicon, not in an IR text check. The comparison is
   on the ``uint32`` bit patterns, so it distinguishes ``+0`` from ``-0`` and
   one NaN from another, which a float compare would not.

2. **Is it accurate?** Against an absolute reference, with the emulation
   landing near fp32 and a plain BF16 MMA of the same operands landing far
   short of it.

3. **Does it hold at the edges of the payload domain?** The top 2^-9 of the
   finite range, where bf16 rounding overflows to infinity and an uncapped split
   makes a finite product NaN; and infinities and NaNs, which must make every
   output they feed non-finite -- and no other.

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

# Payloads at the top of the finite range, where bf16 round-to-nearest-even
# overflows to infinity (0x7f7f8000 and up). 0x7f7fe000 is TF32's largest finite
# value and 0x7f7fffff is fp32's. The lowering caps the leading bf16 at its
# largest finite value so these still split into finite parts.
_TOP_BINADE = (0x7F7F8000, 0x7F7FA000, 0x7F7FC000, 0x7F7FE000, 0x7F7FFFFF)
_SMALLEST_NORMAL = 0x00800000

# The edge cases run on constant operands, so the answer is K * alpha * beta at
# every output position whatever the layout. Each is (label, alpha bits, beta
# bits, expectation); expectations are "finite" (within the emulated
# tolerance of the exact product), "zero", "+inf", "-inf", "nan", or
# "non-finite" (infinity or NaN, which one is not promised).
_EDGE_CASES = (
    ("top binade x smallest normal", 0x7F7FE000, _SMALLEST_NORMAL, "finite"),
    ("-top binade x smallest normal", 0xFF7FE000, _SMALLEST_NORMAL, "finite"),
    (
        "first overflowing payload x smallest normal",
        0x7F7F8000,
        _SMALLEST_NORMAL,
        "finite",
    ),
    ("largest finite x smallest normal", 0x7F7FFFFF, _SMALLEST_NORMAL, "finite"),
    (
        "just below the threshold x smallest normal",
        0x7F7F7FFF,
        _SMALLEST_NORMAL,
        "finite",
    ),
    ("product overflows", 0x7F7FE000, 0x7F7FE000, "+inf"),
    ("product overflows, negative", 0x7F7FE000, 0xFF7FE000, "-inf"),
    ("+inf x 1", 0x7F800000, 0x3F800000, "+inf"),
    ("-inf x 1", 0xFF800000, 0x3F800000, "-inf"),
    ("1 x +inf", 0x3F800000, 0x7F800000, "+inf"),
    ("+inf x +inf", 0x7F800000, 0x7F800000, "+inf"),
    ("+inf x 0", 0x7F800000, 0x00000000, "nan"),
    ("NaN x 1", 0x7FC00000, 0x3F800000, "nan"),
    ("1 x NaN", 0x3F800000, 0x7FC00000, "nan"),
    ("NaN held in the low bits x 1", 0x7F800001, 0x3F800000, "nan"),
    # A signed zero accumulates into the +0 the accumulator starts at.
    ("-0 x 1", 0x80000000, 0x3F800000, "zero"),
    # Subnormal payloads. bf16 reaches down to 2^-133, so a subnormal payload is
    # split to an absolute error of 2^-134 rather than a relative one. These
    # are exact or nearly so, against a partner that makes the product normal
    # and against one that leaves it subnormal.
    ("subnormal 2^-131 x 2^100", 0x00040000, 0x71800000, "finite"),
    ("largest subnormal x 2^100", 0x007FFFFF, 0x71800000, "finite"),
    ("-subnormal 2^-127 x 1", 0x80200000, 0x3F800000, "finite"),
    # An infinity against a partner large enough that its own residual overflows
    # against the cap: infinity or NaN, but never a finite number.
    ("+inf x large partner", 0x7F800000, 0x442BA000, "non-finite"),
    ("+inf x large negative partner", 0x7F800000, 0xC42BA000, "non-finite"),
)

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
    bias-and-shift rather than a truncate. A payload at the top of the range
    carries all the way into an infinity, as the hardware narrowing does; the
    cap that holds it back is in ``_split``. Infinities and NaNs never reach this
    differential -- the operands it compares are finite by construction -- and
    the non-finite cases are checked against closed-form expectations instead.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def _from_bf16(u16: np.ndarray) -> np.ndarray:
    """bf16 bit patterns back to fp32 -- exact, bf16 being a truncated fp32."""
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _cap(bf16: np.ndarray) -> np.ndarray:
    """Hold bf16 bit patterns at the largest finite value, keeping the sign.

    The lowering's cap, on the same patterns: an unsigned compare of each
    against ``0x7f7f`` with its own sign bit attached.
    """
    cap = (bf16 & np.uint16(0x8000)) | np.uint16(0x7F7F)
    return np.where(bf16 > cap, cap, bf16).astype(np.uint16)


def _split(x: np.ndarray):
    """The host mirror of the lowering's Dekker split, as bf16 bit patterns."""
    big = _cap(_to_bf16(x))
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


def _differ(label: str, op_id: str, got: np.ndarray, want: np.ndarray) -> list[str]:
    """Report, and return failures for, any slot where two results differ in bits.

    The ``uint32`` views are what is compared. A float compare treats ``+0`` and
    ``-0`` as equal and a NaN as unequal to itself, so it would pass a sign
    flip on a zero and fail a result that is bit-for-bit right.
    """
    mismatched = int(
        np.count_nonzero(
            np.ascontiguousarray(got, dtype=np.float32).view(np.uint32)
            != np.ascontiguousarray(want, dtype=np.float32).view(np.uint32)
        )
    )
    print(
        f"  [{op_id}] expansion vs hand-written BF16x3, {label}: {mismatched} slots differ"
    )
    if mismatched:
        return [f"{op_id}: expansion does not match the hand-written recipe ({label})"]
    return []


def _edge_failures(op_id: str, atom, emulated, a_len: int, b_len: int) -> list[str]:
    """The edges of the payload domain, on constant operands (see ``_EDGE_CASES``)."""
    failures: list[str] = []
    k = float(atom.k)
    for label, alpha_bits, beta_bits, expect in _EDGE_CASES:
        alpha = np.array([alpha_bits], dtype=np.uint32).view(np.float32)[0]
        beta = np.array([beta_bits], dtype=np.uint32).view(np.float32)[0]
        got = emulated(
            np.full(a_len, alpha, dtype=np.float32),
            np.full(b_len, beta, dtype=np.float32),
        )
        if expect == "finite":
            truth = k * float(alpha) * float(beta)
            err = float(np.abs(got.astype(np.float64) - truth).max()) / abs(truth)
            ok = bool(np.isfinite(got).all()) and err < _TOL_EMULATED
            shown = f"rel={err:.3e}"
        elif expect in ("+inf", "-inf"):
            ok = bool((got == (np.inf if expect == "+inf" else -np.inf)).all())
            shown = f"{got[0]}"
        elif expect == "zero":
            ok = bool((got == 0).all())
            shown = f"{got[0]}"
        elif expect == "nan":
            ok = bool(np.isnan(got).all())
            shown = f"{got[0]}"
        else:
            ok = bool((~np.isfinite(got)).all())
            shown = f"{got[0]}"
        print(f"  [{op_id}] {label}: expect {expect}, got {shown}")
        if not ok:
            failures.append(f"{op_id}: {label}: expected {expect}, got {shown}")
    return failures


def _propagation_failures(
    op_id: str, atom, emulated, a: np.ndarray, b: np.ndarray
) -> list[str]:
    """A non-finite payload reaches exactly the outputs that consume it.

    One element of A feeds one row of the result -- ``n`` outputs -- and one
    element of B one column -- ``m`` outputs -- whatever the lane layout, which
    is what keeps this check layout-free. Every other output must come back
    bit-identical to the run without the infinity, so a poisoned wave or a
    result that smeared across lanes is caught as well as one that went missing.
    """
    failures: list[str] = []
    m, n, _ = atom.shape
    clean = emulated(a, b)
    for name, bad in (("+inf", np.inf), ("-inf", -np.inf), ("NaN", np.nan)):
        for operand, fed in (("A", n), ("B", m)):
            poisoned_a, poisoned_b = a.copy(), b.copy()
            if operand == "A":
                poisoned_a[a.size // 3] = bad
            else:
                poisoned_b[b.size // 3] = bad
            got = emulated(poisoned_a, poisoned_b)
            hit = ~np.isfinite(got)
            same = np.array_equal(
                got[~hit].view(np.uint32), clean[~hit].view(np.uint32)
            )
            print(
                f"  [{op_id}] {name} in one element of {operand}: "
                f"{int(hit.sum())} non-finite outputs (expected {fed}), "
                f"the rest {'unchanged' if same else 'CHANGED'}"
            )
            if int(hit.sum()) != fed:
                failures.append(
                    f"{op_id}: {name} in {operand} made {int(hit.sum())} outputs "
                    f"non-finite, expected exactly {fed}"
                )
            if not same:
                failures.append(
                    f"{op_id}: {name} in {operand} changed an output it does not feed"
                )
    return failures


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
    failures.extend(
        _differ("random operands", op_id, emulated(a, b), handwritten(a, b, _TERMS))
    )

    # The same differential with the top of the finite range mixed in, which is
    # where the cap does its work. The partner is scaled down so that no sum of
    # products reaches infinity: this one is about the split, and an overflowing
    # sum would only add NaN payload bits to compare.
    edge_a = a.copy()
    top = np.array(_TOP_BINADE, dtype=np.uint32)
    top = np.concatenate([top, top | np.uint32(0x80000000)]).view(np.float32)
    edge_a[::7] = np.resize(top, edge_a[::7].shape)
    edge_b = (b * np.float32(2.0**-8)).astype(np.float32)
    got, want = emulated(edge_a, edge_b), handwritten(edge_a, edge_b, _TERMS)
    failures.extend(_differ("top-binade operands", op_id, got, want))
    if not np.isfinite(got).all():
        failures.append(f"{op_id}: a top-binade payload produced a non-finite output")

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

    # (3) The edges of the payload domain.
    failures.extend(_edge_failures(op_id, atom, emulated, a_len, b_len))
    rng = np.random.default_rng(2209)
    failures.extend(
        _propagation_failures(
            op_id,
            atom,
            emulated,
            rng.standard_normal(a_len, dtype=np.float32),
            rng.standard_normal(b_len, dtype=np.float32),
        )
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
