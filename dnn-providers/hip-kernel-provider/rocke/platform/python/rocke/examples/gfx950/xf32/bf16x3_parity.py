# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric gate for the XF32 / BF16x3 emulation (AICK-2208, step 9).

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
  exact.

Usage::

    PYTHONPATH=python python python/rocke/examples/gfx950/xf32/bf16x3_parity.py

Exits non-zero if any check fails, so it doubles as a CI gate. On a host with
no gfx950 device it prints ``SKIPPED`` and exits 0 -- the emulated atoms are
gfx950-only, and a skip must not be mistaken for a pass. The test suite runs it
through ``tests/core/test_xf32_bf16x3_device.py``.

The complementary host-side half (what the recipe is worth against fp32,
TF32 and plain bf16, with no GPU) is ``tests/core/test_xf32_bf16x3_numerics.py``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F32, IRBuilder, KernelDef, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm

_ARCH = "gfx950"

_EMULATED = ("xf32_bf16x3_f32_16x16x32", "xf32_bf16x3_f32_32x32x16")

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
    """One emulated ``tile.mma`` over fp32 operands: the thing under test."""
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


def _split(x):
    """The host mirror of the lowering's split, in torch (RNE, as the ISA is)."""
    import torch

    big = x.to(torch.bfloat16)
    small = (x - big.to(torch.float32)).to(torch.bfloat16)
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
    import torch

    target = ArchTarget.from_gfx(_ARCH)
    atom = target.mma.by_op_id(op_id)
    phys = target.mma.by_op_id(atom.emulation.physical_op_id)
    failures: list[str] = []

    def out_buffer():
        return torch.zeros(
            atom.wave_size * atom.c_frag_len, dtype=torch.float32, device="cuda"
        )

    def emulated(a_f32, b_f32):
        out = out_buffer()
        _launch(
            _emulated_kernel(atom),
            {"a": a_f32, "b": b_f32, "out": out},
            wave_size=atom.wave_size,
        )
        return out

    def handwritten(a_f32, b_f32, terms):
        """The same recipe spelled out, over operands split on the host."""
        a_big, a_small = _split(a_f32)
        b_big, b_small = _split(b_f32)
        out = out_buffer()
        _launch(
            _handwritten_kernel(phys, terms),
            {
                "a_big": a_big,
                "a_small": a_small,
                "b_big": b_big,
                "b_small": b_small,
                "out": out,
            },
            wave_size=phys.wave_size,
        )
        return out

    a_len = atom.wave_size * atom.a_frag_len
    b_len = atom.wave_size * atom.b_frag_len

    # (1) Faithfulness, on random operands. Bitwise, not tolerant: both
    # kernels issue the same three MFMAs over the same bits in the same
    # order, so anything other than equality is a real difference in what
    # the lowering built.
    torch.manual_seed(2208)
    a = torch.randn(a_len, dtype=torch.float32, device="cuda")
    b = torch.randn(b_len, dtype=torch.float32, device="cuda")
    mismatched = int((emulated(a, b) != handwritten(a, b, _TERMS)).sum().item())
    print(f"  [{op_id}] expansion vs hand-written BF16x3: {mismatched} slots differ")
    if mismatched:
        failures.append(f"{op_id}: expansion does not match the hand-written recipe")

    # (2) Accuracy, on constant operands, where the truth is K*alpha*beta at
    # every output position regardless of layout.
    a = torch.full((a_len,), _ALPHA, dtype=torch.float32, device="cuda")
    b = torch.full((b_len,), _BETA, dtype=torch.float32, device="cuda")
    got = emulated(a, b)
    bf16_only = handwritten(a, b, (("a_big", "b_big"),))
    truth = float(atom.k) * _ALPHA * _BETA
    emulated_err = float((got.double() - truth).abs().max().item()) / abs(truth)
    bf16_err = float((bf16_only.double() - truth).abs().max().item()) / abs(truth)
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
