# SPDX-License-Identifier: Apache-2.0
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""``six``-style polyfill for six small FlyDSL helpers.

The kernels in this tree use ``ssel``, ``smin``, ``smax``, ``sdiv_rd_pow2``,
``vector_elem_type`` and ``wmma_f32_16x16x16``, which the pinned FlyDSL
(0.3.4) does not provide. This module is **authored**, not vendored. Each
function is written as ``try: from flydsl.kernels.common...`` followed by a
local body in the ``except ImportError`` arm, so a FlyDSL that did ship the
helper under that path would be used in preference to the local copy. None of
those paths exists in FlyDSL 0.3.4 (``flydsl.kernels`` is not a package), so
today the local bodies are the only code that runs. If a later FlyDSL provides
a helper at a different path, change the probe path or drop the local body.

*Polyfill*, not *shim* or *compat*: this repository's ``shim`` already means
something specific (the generated C++ dispatch layer), so reusing the word
here would mislead.

**This module must import with no ``flydsl`` installed.** ``DualwaveSwpTraits``
below is reached by the *generator*, through ``gfx950_standalone``'s fallback,
and the generator has no flydsl by design. So every ``flydsl`` import here is
function-local -- the same laziness applied to torch in
``fmha_abi_gfx1201.py``, and for the same reason. The two ``ir.Type``
annotations are quoted for the same reason: an annotation on a ``def`` is
evaluated when the ``def`` executes.
"""

from dataclasses import dataclass, fields

__all__ = [
    "ssel",
    "smin",
    "smax",
    "sdiv_rd_pow2",
    "wmma_f32_16x16x16",
    "vector_elem_type",
    "DualwaveSwpTraits",
]


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _pow2_shift(value: int) -> int:
    assert _is_pow2(value)
    return value.bit_length() - 1


try:
    from flydsl.kernels.common.utils import ssel
except ImportError:

    def ssel(pred, a, b):
        """``pred ? a : b`` as an ``fx.Int32``.

        ``pred`` may be an ``fx.Boolean`` (what a comparison returns), a raw
        i1 ``ir.Value``, or an ``ArithValue`` -- ``fx.Boolean`` accepts all
        three.

        Operands are *not* coerced. Pass i32; passing an ``fx.Index`` gets
        you a 64-bit unsigned select, which is a silent bug wherever the
        value can be negative.
        """
        import flydsl.expr as fx

        return fx.Int32(fx.Boolean(pred).select(a, b))


try:
    from flydsl.kernels.common.utils import smin
except ImportError:

    def smin(a, b):
        """Signed minimum of two i32 values. Same operand contract as `ssel`."""
        return ssel((a < b), a, b)


try:
    from flydsl.kernels.common.utils import smax
except ImportError:

    def smax(a, b):
        """Signed maximum of two i32 values. See `smin`."""
        return ssel((a > b), a, b)


try:
    from flydsl.kernels.common.utils import sdiv_rd_pow2
except ImportError:

    def sdiv_rd_pow2(value, divisor: int):
        """``floor(value / divisor)`` for a *signed* i32 and a power-of-two divisor.

        The signed counterpart to ``udiv_pow2``: an arithmetic right shift
        rounds toward negative infinity, which is what a floor division
        needs and what ``arith.divsi``'s truncation does not give on
        negative input.
        """
        import flydsl.expr as fx

        assert _is_pow2(
            divisor
        ), f"sdiv_rd_pow2 needs a power-of-two divisor, got {divisor}"
        return fx.Int32(value) >> fx.Int32(_pow2_shift(divisor))


try:
    from flydsl.kernels.common.mma.wmma_ops import vector_elem_type
except ImportError:

    def vector_elem_type(value) -> "ir.Type":
        """Element type of a vector-typed value, raw or DSL-wrapped.

        ``ir.VectorType.isinstance`` does not exist in this binding; Python
        ``isinstance`` against the downcast class is what works.
        """
        from flydsl._mlir import ir
        from flydsl.expr.utils.arith import _to_raw as _raw

        ty = _raw(value).type
        if not isinstance(ty, ir.VectorType):
            raise TypeError(f"expected a vector value, got {ty}")
        return ir.VectorType(ty).element_type


try:
    from flydsl.kernels.common.mma.wmma_ops import wmma_f32_16x16x16
except ImportError:

    def wmma_f32_16x16x16(a, b, acc, acc_type: "ir.Type | None" = None):
        """One 16x16x16 WMMA into an f32 accumulator: ``acc += a @ b``.

        ``a`` and ``b`` are 8-lane f16 or bf16 vectors and must agree;
        ``acc`` is an 8-lane f32 vector. ``acc_type`` defaults to ``acc``'s
        own type and only needs passing when ``acc`` is a raw value whose
        type the caller wants to override.

        bf16 operands are bitcast to ``i16`` because that is the ABI the
        intrinsic takes -- a reinterpretation, not a conversion.
        """
        import flydsl.expr as fx
        from flydsl._mlir import ir
        from flydsl.expr import rocdl
        from flydsl.expr.typing import Vector
        from flydsl.expr.utils.arith import _to_raw as _raw

        a_ty, b_ty = vector_elem_type(a), vector_elem_type(b)
        if a_ty != b_ty:
            raise TypeError(
                f"WMMA operands must share an element type, got {a_ty} and {b_ty}"
            )
        res_ty = acc_type if acc_type is not None else _raw(acc).type

        if isinstance(a_ty, ir.BF16Type):
            a16 = _raw(Vector(_raw(a)).bitcast(fx.Int16))
            b16 = _raw(Vector(_raw(b)).bitcast(fx.Int16))
            return rocdl.wmma_f32_16x16x16_bf16(res_ty, a16, b16, _raw(acc)).result
        if isinstance(a_ty, ir.F16Type):
            return rocdl.wmma_f32_16x16x16_f16(
                res_ty, _raw(a), _raw(b), _raw(acc)
            ).result
        raise TypeError(f"wmma_f32_16x16x16 supports f16 and bf16 operands, got {a_ty}")


# ---------------------------------------------------------------------------
# ``DualwaveSwpTraits`` -- a verbatim copy, and the check that keeps it one
#
# Unlike the six helpers above, this class comes from the vendored
# ``kernels/attention_cdna/flash_attn_utils.py``. It is copied anyway because
# of *where* it lives: that module's scope imports ``flydsl.compiler``. The
# generator reaches this class -- ``fmha_traits_gfx950.ParityDualwaveTraits``
# subclasses it, and ``fmha_tuning_*_gfx950.resolve()`` constructs one to
# validate a configuration -- and the generator must never import flydsl.
#
# The copy can go when ``fmha_traits_gfx950`` no longer needs a
# flydsl-bearing module for its base class -- when ``DualwaveSwpTraits``
# moves to a module importing no flydsl, or when the generator stops
# constructing traits.
#
# Copied verbatim from the vendored ``DualwaveSwpTraits``. Do not reformat,
# rename or reorder it: the safety
# argument below is that it is an exact copy, and
# ``assert_dualwave_swp_traits_equivalent`` is what proves that claim at
# build time rather than asserting it in a comment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualwaveSwpTraits:
    """Pure compile-time tile/layout constants for gfx950 DUALWAVE_SWP."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_N_OUT: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    HEAD_DIM_V: int
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEP: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    PAGED: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    KV_CACHE_LAYOUT: str
    KV_VECTORIZED: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DMA_BYTES: int
    BF16_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    SMEM_LINEAR_WAVE: int
    SMEM_N_PER_WAVE: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_PAD: int
    SMEM_V_PAD: int
    SMEM_K_LINE_STRIDE: int
    SMEM_V_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    SMEM_V_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    K_LDS_TO_REG_N_STRIP_STRIDE: int
    K_LDS_TO_REG_KSTEP_INNER_STRIDE: int
    K_LDS_TO_REG_KSTEP_OUTER_STRIDE: int
    V_LDS_TO_REG_HALF_WAVE_STRIDE: int
    V_LDS_TO_REG_LANE_QUAD_STRIDE: int
    V_LDS_TO_REG_N_GROUP_STRIDE: int
    V_LDS_TO_REG_LANE_IN_QUAD_STRIDE: int
    V_LDS_TO_REG_K_SUBSTEP_STRIDE: int
    V_LDS_TO_REG_DCHUNK_PAIR_STRIDE: int
    V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE: int
    V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE: int
    PAGED_BT_LDS_SIZE: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    KV_VEC_SIZE: int
    VEC_V_ROW_STRIDE: int
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int
    RETURN_LSE: bool = False
    XCD_SWIZZLE: bool = False

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.PAGED,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.KV_CACHE_LAYOUT,
            self.KV_VECTORIZED,
            self.RETURN_LSE,
            self.XCD_SWIZZLE,
        )


def assert_dualwave_swp_traits_equivalent(upstream, local=DualwaveSwpTraits) -> None:
    """Fail loudly if the copy above has drifted from ``upstream``'s class.

    Compares ``dataclasses.fields()`` -- **names, types and order**. For a
    79-field dataclass whose every field is supplied by keyword at every
    construction (``fmha_traits_gfx950.make_traits`` passes every one,
    including the two that carry defaults), that comparison is *total*, not a
    sample: any upstream add, remove, rename, reorder or retype lands here.

    Defaults are deliberately **not** compared. Both defaulted fields are
    always passed explicitly, so a changed default cannot reach a kernel, and
    checking it would turn a harmless upstream edit into a failed build.

    Same philosophy as ``flyc_compile._verify_elf``: check the assumption
    where it can be checked, and stop the build when it moves.

    **The caller is ``gfx950_standalone``, and deliberately not this module's
    own scope.** The obvious spelling -- a module-scope
    ``try: from kernels.attention_cdna.flash_attn_utils import DualwaveSwpTraits``
    here, matching the six helpers above -- would run this comparison during
    a **gfx1201** build too, because the gfx1201 kernels alias this module
    wholesale as ``common_utils``/``wmma_ops``, and the class is not stable
    across FlyDSL versions. That spelling would let a gfx950 concern fail a
    gfx1201 build over a class gfx1201 never touches. Calling from
    ``gfx950_standalone`` runs the check on exactly the builds that use the
    class, and no others.
    """
    if upstream is local:
        return
    got = [(f.name, f.type) for f in fields(upstream)]
    want = [(f.name, f.type) for f in fields(local)]
    if got == want:
        return
    got_names = [n for n, _ in got]
    want_names = [n for n, _ in want]
    detail = []
    for name in want_names:
        if name not in got_names:
            detail.append(f"  removed upstream: {name}")
    for name in got_names:
        if name not in want_names:
            detail.append(f"  added upstream:   {name}")
    for (gn, gt), (wn, wt) in zip(got, want):
        if gn == wn and gt != wt:
            detail.append(f"  retyped:          {gn}: {wt!r} -> {gt!r}")
    if got_names != want_names and sorted(got_names) == sorted(want_names):
        detail.append("  field ORDER changed")
    raise AssertionError(
        "flyc_polyfill.DualwaveSwpTraits has drifted from "
        f"{upstream.__module__}.{upstream.__qualname__} "
        f"({len(want)} fields copied, {len(got)} upstream).\n"
        + "\n".join(detail or ["  (no per-field difference isolated)"])
        + "\n\nRe-copy the class verbatim from the vendored "
        "kernels/attention_cdna/flash_attn_utils.py."
    )
