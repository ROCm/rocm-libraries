# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Typed AMDGPU inline-asm helpers (NEW, additive — golden-safe).

This module wraps :meth:`IRBuilder.inline_asm` with named, typed helpers
for machine instructions whose operand *register classes* must be pinned
explicitly — something the typed LLVM intrinsics do not expose. The
motivating case is the dense, unscaled ``v_mfma_f32_16x16x128_f8f6f4``
with **AGPR srcA/srcB + VGPR accumulator** (the aiter staging layout):
the only LLVM intrinsic for K=128 fp8 is the *scaled* one
(``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4``), and even pinning its
scales to 0 leaves srcA/srcB placement to the register allocator. The
``a`` inline-asm constraint forces them into the AGPR file.

AMDGPU inline-asm constraint cheatsheet:
  ``v`` = VGPR input, ``a`` = AGPR input, ``s`` = SGPR input;
  ``=v`` / ``=a`` / ``=s`` = outputs; a digit (``0``) ties an input to
  that output (read+write in place, required for the MFMA accumulator).

Nothing here changes any existing op/atom: it adds new helper functions
that only the mega-kernel selects, so the golden shared-kernel digest is
unaffected.
"""

from __future__ import annotations


from ..core.ir import I32, IRBuilder, Value, VectorType


# The f8f6f4 MFMA srcA/srcB operand type the AMDGPU backend accepts under the
# inline-asm ``a`` (AGPR) constraint: a 256-bit ``<8 x i32>`` (8 AGPRs). The
# backend register allocator REJECTS an ``<32 x i8>`` operand on this asm
# (comgr CODEGEN_BC_TO_RELOCATABLE error), so a fragment arriving as
# ``<32 x fp8e4m3>`` (the mega-kernel's native load type) MUST be bitcast to
# ``<8 x i32>`` first. This mirrors the typed-intrinsic hero lowering, which
# also bitcasts ``<32 x i8>`` -> ``<8 x i32>`` before the call.
_MFMA_F8F6F4_SRC_TY = VectorType(I32, 8)


def _as_src_v8i32(b: IRBuilder, v: Value) -> Value:
    """Bitcast a 256-bit operand to ``<8 x i32>`` (no-op if already that type)."""
    if isinstance(v.type, VectorType) and v.type == _MFMA_F8F6F4_SRC_TY:
        return v
    return b.vec_bitcast(v, _MFMA_F8F6F4_SRC_TY)


# gfx950 16x16x128 f8f6f4 MFMA result/source-hazard wait-states.
#
# The LLVM AMDGPU GCNHazardRecognizer inserts the required ``s_nop``s around
# *intrinsic* MFMAs automatically — covering both (1) the long result latency
# before a non-MFMA consumer reads ``vdst``, and (2) the source-read window
# before a later instruction may overwrite ``srcA``/``srcB``. An **inline-asm**
# MFMA is opaque to that recognizer (it only sees an asm blob with ``a``/``v``
# operands, not an MFMA), so NO waits are inserted and the result is silently
# wrong (the accumulator is consumed / the source AGPRs are clobbered before the
# MFMA retires).
#
# Empirically on MI355X (asm_mfma_parity.py): a bare asm MFMA returns the
# unmodified accumulator (A·B contributes 0); a single trailing ``s_nop 3``
# fixes the isolated case, and a 4-deep accumulation chain (the real K-loop
# shape, where the next K-step's load reuses the same source AGPRs) becomes
# byte-identical to the intrinsic at ``s_nop 8`` (smaller values still corrupt).
# We bake in ``s_nop 8`` so the helper is correct for ANY consumer/chain. This
# is conservative for throughput (it serialises the MFMA latency the AGPR lever
# is meant to hide) — closing that is the next perf iteration (interleave
# independent loads / sched_group_barrier into the window instead of a blanket
# nop), NOT a correctness concern.
_MFMA_F8F6F4_HAZARD_NOP = 8


def mfma_f8f6f4_agpr(
    b: IRBuilder,
    a: Value,
    bb: Value,
    acc: Value,
    *,
    convergent: bool = True,
    hazard_nop: int = _MFMA_F8F6F4_HAZARD_NOP,
) -> Value:
    """Dense **unscaled** ``v_mfma_f32_16x16x128_f8f6f4`` with AGPR srcA/srcB.

    Emits (per ASM_HELPERS_PLAN §2, plus the hazard ``s_nop`` proven required
    for inline-asm MFMAs — see :data:`_MFMA_F8F6F4_HAZARD_NOP`)::

        %acc = call <4 x float> asm sideeffect
               "v_mfma_f32_16x16x128_f8f6f4 $0, $2, $3, $1\n\ts_nop 8",
               "=v,0,a,a"(<4 x float> %c, <8 x i32> %a8, <8 x i32> %b8)

    Operand / ``$``-mapping (LLVM: ``$0`` = output, then inputs in listed
    order):
      - ``$0`` ``=v`` : output accumulator, VGPR (matches aiter ``v[..]``)
      - ``$1`` ``0``  : tied input accumulator (ties to ``$0`` — read+write
        in place; this is what keeps the running K-accumulation correct)
      - ``$2`` ``a``  : srcA, **AGPR**, ``<8 x i32>``
      - ``$3`` ``a``  : srcB, **AGPR**, ``<8 x i32>``

    The template ``$0, $2, $3, $1`` maps to ``vdst, srcA, srcB, srcC(acc)``.
    The trailing assembler operand carries the implicit ``cbsz=0/blgp=0``
    (fp8 e4m3 for BOTH A and B) — no ``cbsz:``/``blgp:`` modifiers, no
    ``_scale`` suffix, so the math is identical to the scaled intrinsic
    with all scales pinned to 0 (factor 1.0). Verified byte-identical to the
    ``mfma_f32_16x16x128_fp8`` atom (single MFMA and 4-deep chain) in
    ``asm_mfma_parity.py``.

    Args:
      a, bb: the A and B operand tiles, each ``<8 x i32>`` (256 bits =
        32 fp8 e4m3 bytes per lane). Caller must bitcast to ``<8 x i32>``
        before this call (the asm constraint type must match exactly).
      acc: the incoming ``<4 x float>`` accumulator (tied; returned updated).
      convergent: mark the asm convergent (default True) — currently advisory;
        ``sideeffect`` already blocks DCE/dup/reorder (the textual
        ``convergent`` keyword is not valid on an LLVM inline-asm call, so the
        emitter does not render it).
      hazard_nop: number for the trailing ``s_nop`` (default 8, the proven
        minimum for chained AGPR-source correctness). Set to 0 ONLY if the
        caller guarantees the LLVM recognizer can otherwise see the hazard.

    Returns the updated ``<4 x float>`` accumulator.
    """
    # The backend accepts only ``<8 x i32>`` for the ``a``-constrained MFMA
    # sources; bitcast the native fp8 fragment (``<32 x fp8e4m3>``) if needed.
    a = _as_src_v8i32(b, a)
    bb = _as_src_v8i32(b, bb)
    # NOTE: the AMDGPU assembler treats ``;`` as a COMMENT, so additional
    # statements MUST be separated by a NEWLINE (``\n``), not ``;`` — otherwise
    # the trailing ``s_nop`` is silently dropped and the hazard returns.
    template = "v_mfma_f32_16x16x128_f8f6f4 $0, $2, $3, $1"
    if hazard_nop:
        template += f"\n\ts_nop {int(hazard_nop)}"
    return b.inline_asm(
        template,
        "=v,0,a,a",
        [acc, a, bb],
        result_type=acc.type,
        sideeffect=True,
        convergent=convergent,
        result_name_hint="mfma",
    )


def mfma_f8f6f4_agpr_cluster(
    b: IRBuilder,
    accs,
    srcs,
    *,
    tail_nop: int = _MFMA_F8F6F4_HAZARD_NOP,
    inter_nop: int = 0,
    convergent: bool = True,
):
    """Cluster of N back-to-back **unscaled** ``v_mfma_f32_16x16x128_f8f6f4``
    MFMAs (AGPR srcA/srcB + VGPR acc) emitted as ONE inline-asm block.

    This is the *nuclear* form: instead of one ``sideeffect`` asm per MFMA
    (which fragments the schedule — every MFMA becomes its own opaque
    scheduling fence, the documented +25–31% per-op regression), the whole
    gate/up (or any) MFMA burst of one K-group is a SINGLE asm block whose
    internal instruction stream is the hand-schedule, matching aiter/pyisa's
    ``cl_gemm0_XmulG_gemm1_XmulU`` body (back-to-back AGPR-source MFMAs into
    N independent VGPR accumulators). The surrounding loads stay DSL-issued
    so the machine scheduler can still overlap them around the cluster (it
    only sees ONE asm node, not N fences).

    Hazard handling: the N MFMAs write N DISTINCT accumulators and read N
    DISTINCT source pairs, so they pipeline freely — the only hazard is the
    long MFMA *result* latency before a non-MFMA consumer (the dequant FMA)
    reads ``vdst``. A single trailing ``s_nop`` (``tail_nop``) after the last
    MFMA covers that for the whole cluster (one nop amortised over N MFMAs,
    vs the per-op helper's N nops). ``inter_nop`` (default 0) optionally
    spaces each MFMA — only needed if a later MFMA in the cluster reuses an
    earlier one's source AGPRs (it does not here; each gets its own ds_read).

    Args:
      accs: list of N incoming ``<4 x float>`` accumulators (tied in/out).
      srcs: list of N ``(a, bb)`` operand-tile pairs; each is bitcast to
        ``<8 x i32>`` (the AGPR-constrained MFMA source type) as needed.
      tail_nop: ``s_nop`` count after the last MFMA (default 8 — the proven
        result-latency wait; the dequant consumer reads vdst right after).
      inter_nop: ``s_nop`` between consecutive MFMAs (default 0).

    Returns the list of N updated ``<4 x float>`` accumulators (same order).
    """
    n = len(accs)
    assert n == len(srcs) and n >= 1
    # Bitcast all sources to <8 x i32>.
    src_v = []
    for a, bb in srcs:
        src_v.append((_as_src_v8i32(b, a), _as_src_v8i32(b, bb)))

    # Operand order (after the N outputs): N tied accs, then 2N sources.
    #   $0..$(n-1)      : outputs        (=v)
    #   $n..$(2n-1)     : tied accs      (0,1,..,n-1)  -> $(n+i) is acc i
    #   $2n..$(4n-1)    : sources (a)    -> srcA_i = $(2n+2i), srcB_i = $(2n+2i+1)
    lines = []
    for i in range(n):
        vdst = f"${i}"
        srcA = f"${2 * n + 2 * i}"
        srcB = f"${2 * n + 2 * i + 1}"
        vsrc = f"${n + i}"
        lines.append(f"v_mfma_f32_16x16x128_f8f6f4 {vdst}, {srcA}, {srcB}, {vsrc}")
        if inter_nop and i + 1 < n:
            lines.append(f"s_nop {int(inter_nop)}")
    if tail_nop:
        lines.append(f"s_nop {int(tail_nop)}")
    template = "\n\t".join(lines)

    out_constraints = ",".join(["=v"] * n)
    tied_constraints = ",".join(str(i) for i in range(n))
    src_constraints = ",".join(["a"] * (2 * n))
    constraints = f"{out_constraints},{tied_constraints},{src_constraints}"

    operands = list(accs)
    for a, bb in src_v:
        operands.append(a)
        operands.append(bb)

    return b.inline_asm_multi(
        template,
        constraints,
        operands,
        result_types=[acc.type for acc in accs],
        sideeffect=True,
        convergent=convergent,
        result_name_hint="mfmacl",
    )


def s_nop(b: IRBuilder, n: int = 0) -> None:
    """Emit ``s_nop <n>`` (no result). Handy as a scheduling spacer."""
    b.inline_asm(f"s_nop {int(n)}", "", [])
