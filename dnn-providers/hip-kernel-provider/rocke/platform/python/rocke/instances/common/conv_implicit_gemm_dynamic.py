# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime-generic (fully dynamic) implicit-GEMM forward convolution.

This is the *dynamic-shape* sibling of :mod:`conv_implicit_gemm`. It mirrors the
static→dynamic split used by :mod:`rmsnorm2d_dynamic` / :mod:`layernorm2d_dynamic`:
instead of baking the convolution geometry (``N, C, K, Hi, Wi, R, S, stride,
pad, dilation``) into the kernel as ``const_i32`` operands, every geometry value
is read from a **runtime i32 scalar argument**, and the derived GEMM extents
(``Ho, Wo, M, N_gemm, K_gemm``) plus the tensor row-major strides are computed
in-kernel as SSA. The only thing baked into each ``.co`` is the *tile/perf
config* (tile shape, warp grid, MMA atom, pipeline, epilogue, vector widths).

Net effect: **one ``.co`` per tile config serves any 2-D forward-conv shape.**
The tuner picks the winning tile config per problem; the catalog selector keys
on ``dtype`` / ``groups`` / vector-width alignment only (see the family README).

Genericity comes from **partial-tile boundary masking**, which is already
present in the shared conv body and merely references runtime SSA here:

* load side  — the hardware buffer-resource OOB clamp (offset > ``*_bytes`` →
  reads 0), already fed the runtime ``A_bytes`` / ``B_bytes`` scalars, PLUS the
  transform-DAG validity predicates (``embed`` halo ``0 <= hi < Hi`` and ``pad``
  partial-K ``y < R`` / ``x < S``), which now test runtime ``Hi`` / ``Wi`` / ``R``
  / ``S``;
* store side — the epilogue ``m < M`` / ``n < N_gemm`` predication, which now
  tests the runtime ``M`` / ``N_gemm`` SSA.

So a conv whose spatial / channel / filter extents do **not** divide the tile is
computed correctly: the partial rows/cols/K-tail are masked, not mis-addressed.

Authoring surface::

    geom = DynamicConvGeometry()
    kernel = build_implicit_gemm_conv_dynamic(
        ImplicitGemmConvSpec(
            problem=DYNAMIC_CONV_PLACEHOLDER,   # shape is runtime; see below
            name="conv_igemm_fprop_dyn",
            tile_m=64, tile_n=64, tile_k=64,
            warp_m=2, warp_n=2,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            wave_size=32,                        # gfx1151 WMMA
            pipeline="mem", epilogue="default",
            vector_size_a=8, vector_size_b=8, vector_size_c=1,
        ),
        arch="gfx1151",
        geom=geom,
    )

The ``problem`` field still carries a :class:`ConvProblem`, but only its
*non-shape* facets are consulted on the dynamic path (dtype via ``spec.data``,
tile/warp/wave via ``spec``). Its shape numbers are placeholders — use
:data:`DYNAMIC_CONV_PLACEHOLDER`, which is deliberately **not** pointwise so the
shared body takes the general descriptor path (never the flat-GEMM fast path).

Runtime divisor note: the ``m -> n, ho, wo`` and ``k -> y, x, c`` coordinate
unpack divides by the runtime ``Ho*Wo`` / ``Wo`` / ``S*C`` / ``C``. v1 emits
plain :class:`UnmergeDynamic` (hardware ``sdiv`` / ``srem``); all operands are
non-negative. The documented perf fast-follow swaps these for host-precomputed
magic ``(multiplier, shift)`` scalar args fed to ``do_magic_division`` — the ABI
is superset-friendly, so that adds args without a break.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ...core.ir import I32, IRBuilder, KernelDef, Value
from ...helpers.transforms import (
    TensorDescriptor,
    embed_dynamic,
    pad_dynamic,
    unmerge_dynamic,
)
from ._conv_implicit_gemm_common import ConvProblem, _ir_dtype
from .conv_implicit_gemm import ImplicitGemmConvSpec, build_implicit_gemm_conv

# A non-pointwise placeholder shape for the ``spec.problem`` slot on the dynamic
# path. Its numbers are never used to address memory (all geometry is runtime);
# they only need to (a) keep ``ConvProblem.__post_init__`` happy, (b) be
# *non-pointwise* (Y=X=3) so the shared body takes the general descriptor path,
# and (c) leave ``C`` / ``K`` as benign multiples so the (unused) auto
# vector-size heuristic never trips. Always pass explicit ``vector_size_*`` on
# the spec so the placeholder ``C`` / ``K`` are truly irrelevant.
DYNAMIC_CONV_PLACEHOLDER = ConvProblem(
    N=1, Hi=8, Wi=8, C=8, K=8, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
)


@dataclass
class DynamicConvGeometry:
    """Runtime convolution geometry: declares the i32 scalar ABI + derives SSA.

    A single instance is threaded into :func:`build_implicit_gemm_conv` via its
    ``dyn`` hook. :meth:`bind` declares the runtime scalar parameters (in the
    fixed ABI order the C++ ``ConvFpropAdapter`` emits) and computes the derived
    GEMM extents / tensor strides as SSA. The three descriptor builders then
    compose the *same* coordinate-transform DAG as the static conv, but with the
    dynamic (runtime-valued) transforms.

    All fields below are populated by :meth:`bind`; before that they are ``None``.
    """

    # ---- raw runtime scalar params (declared by bind, in ABI order) ----
    N: Optional[Value] = field(default=None, init=False)
    C: Optional[Value] = field(default=None, init=False)
    K: Optional[Value] = field(default=None, init=False)
    Hi: Optional[Value] = field(default=None, init=False)
    Wi: Optional[Value] = field(default=None, init=False)
    R: Optional[Value] = field(default=None, init=False)  # filter height (== Y)
    S: Optional[Value] = field(default=None, init=False)  # filter width  (== X)
    sH: Optional[Value] = field(default=None, init=False)
    sW: Optional[Value] = field(default=None, init=False)
    pH: Optional[Value] = field(default=None, init=False)
    pW: Optional[Value] = field(default=None, init=False)
    dH: Optional[Value] = field(default=None, init=False)
    dW: Optional[Value] = field(default=None, init=False)

    # ---- derived GEMM extents (SSA) ----
    Ho: Optional[Value] = field(default=None, init=False)
    Wo: Optional[Value] = field(default=None, init=False)
    M: Optional[Value] = field(default=None, init=False)
    N_gemm: Optional[Value] = field(default=None, init=False)
    K_gemm: Optional[Value] = field(default=None, init=False)

    # ---- derived row-major tensor strides (SSA) ----
    _WiC: Optional[Value] = field(default=None, init=False)
    _HiWiC: Optional[Value] = field(default=None, init=False)
    _SC: Optional[Value] = field(default=None, init=False)
    _WoK: Optional[Value] = field(default=None, init=False)
    _HoWoK: Optional[Value] = field(default=None, init=False)
    _neg_pH: Optional[Value] = field(default=None, init=False)
    _neg_pW: Optional[Value] = field(default=None, init=False)

    _bound: bool = field(default=False, init=False)

    # Order in which the geometry scalars are declared as kernel params. This is
    # the ABI contract the C++ ``ConvFpropAdapter.buildBindings`` and the family
    # ``args_signature`` must match (after the A/B/D pointers + A/B/D_bytes).
    PARAM_ORDER = (
        "N",
        "C",
        "K",
        "Hi",
        "Wi",
        "R",
        "S",
        "sH",
        "sW",
        "pH",
        "pW",
        "dH",
        "dW",
    )

    def bind(self, b: IRBuilder) -> None:
        """Declare the runtime i32 params and derive the GEMM extents / strides.

        Must be called exactly once, after the ``A/B/D`` pointers and the
        ``A_bytes/B_bytes/D_bytes`` scalars have been declared, so the packed
        kernarg layout matches the adapter's binding order.
        """
        if self._bound:
            raise RuntimeError("DynamicConvGeometry.bind called twice")

        for name in self.PARAM_ORDER:
            setattr(self, name, b.param(name, I32))

        c0 = b.const_i32(0)
        c1 = b.const_i32(1)
        c2 = b.const_i32(2)

        def out_extent(
            inp: Value, pad: Value, dil: Value, flt: Value, stride: Value
        ) -> Value:
            # (inp + 2*pad - dil*(flt-1) - 1) // stride + 1
            num = b.sub(
                b.sub(b.add(inp, b.mul(c2, pad)), b.mul(dil, b.sub(flt, c1))),
                c1,
            )
            return b.add(b.div(num, stride), c1)

        self.Ho = out_extent(self.Hi, self.pH, self.dH, self.R, self.sH)
        self.Wo = out_extent(self.Wi, self.pW, self.dW, self.S, self.sW)

        # GEMM extents. groups == 1 on this dynamic path, so N_gemm == K and
        # K_gemm == R*S*C (no per-group slabbing).
        self.M = b.mul(b.mul(self.N, self.Ho), self.Wo)
        self.N_gemm = self.K
        self.K_gemm = b.mul(b.mul(self.R, self.S), self.C)

        # Row-major strides. Innermost stride (C for A/B, K for D) is 1.
        self._WiC = b.mul(self.Wi, self.C)
        self._HiWiC = b.mul(self.Hi, self._WiC)
        self._SC = b.mul(self.S, self.C)
        self._WoK = b.mul(self.Wo, self.K)
        self._HoWoK = b.mul(self.Ho, self._WoK)
        self._neg_pH = b.sub(c0, self.pH)
        self._neg_pW = b.sub(c0, self.pW)

        self._bound = True

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError("DynamicConvGeometry used before bind()")

    # ------------------------------------------------------------------
    # Descriptor builders — same DAG shape as the static make_*_descriptor,
    # with the numeric params lifted to runtime SSA.
    # ------------------------------------------------------------------

    def a_descriptor(self, dtype: str = "fp16") -> TensorDescriptor:
        """(m, k) -> NHWC input offset, all geometry runtime.

        Mirrors :func:`conv_implicit_gemm.make_a_descriptor` (2-D branch):
        ``unmerge m -> n, ho, wo`` ; ``embed (ho, y) -> hi`` ;
        ``embed (wo, x) -> wi`` ; ``unmerge k -> y, x, c`` ; ``pad y`` ; ``pad x``.
        """
        self._require_bound()
        return TensorDescriptor.naive_runtime(
            "A_nhwc",
            coord_names=["n", "hi", "wi", "c"],
            strides=[self._HiWiC, self._WiC, self.C, 1],
            dtype=_ir_dtype(dtype),
        ).transform(
            unmerge_dynamic(
                "m", into=["n", "ho", "wo"], dims=[self.N, self.Ho, self.Wo]
            ),
            embed_dynamic(
                ["ho", "y"],
                "hi",
                strides=[self.sH, self.dH],
                offset=self._neg_pH,
                lo=0,
                hi=self.Hi,
            ),
            embed_dynamic(
                ["wo", "x"],
                "wi",
                strides=[self.sW, self.dW],
                offset=self._neg_pW,
                lo=0,
                hi=self.Wi,
            ),
            unmerge_dynamic("k", into=["y", "x", "c"], dims=[self.R, self.S, self.C]),
            pad_dynamic("y", lo=0, hi=self.R),
            pad_dynamic("x", lo=0, hi=self.S),
        )

    def b_descriptor(self, dtype: str = "fp16") -> TensorDescriptor:
        """(k_out, k_gemm) -> KYXC weight offset, all geometry runtime.

        Mirrors :func:`conv_implicit_gemm.make_b_descriptor` (2-D branch):
        ``unmerge k_gemm -> y, x, c`` ; ``pad y`` ; ``pad x``.
        """
        self._require_bound()
        return TensorDescriptor.naive_runtime(
            "B_kyxc",
            coord_names=["k_out", "y", "x", "c"],
            strides=[self.K_gemm, self._SC, self.C, 1],
            dtype=_ir_dtype(dtype),
        ).transform(
            unmerge_dynamic(
                "k_gemm", into=["y", "x", "c"], dims=[self.R, self.S, self.C]
            ),
            pad_dynamic("y", lo=0, hi=self.R),
            pad_dynamic("x", lo=0, hi=self.S),
        )

    def d_descriptor(self, dtype: str = "fp16") -> TensorDescriptor:
        """(m, k_out) -> NHWK output offset, all geometry runtime.

        Mirrors :func:`conv_implicit_gemm.make_d_descriptor` (2-D branch):
        ``unmerge m -> n, ho, wo``.
        """
        self._require_bound()
        return TensorDescriptor.naive_runtime(
            "D_nhwk",
            coord_names=["n", "ho", "wo", "k_out"],
            strides=[self._HoWoK, self._WoK, self.K, 1],
            dtype=_ir_dtype(dtype),
        ).transform(
            unmerge_dynamic(
                "m", into=["n", "ho", "wo"], dims=[self.N, self.Ho, self.Wo]
            ),
        )

    def kernel_name(self, spec: ImplicitGemmConvSpec) -> str:
        """Shape-free kernel name (tile config only) — one binary per config.

        Parity with ``rmsnorm2d_dynamic``: the kernel name carries the perf
        config but NOT the problem shape, so the same symbol serves every shape.
        """
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            spec.name,
            spec.data.dtype_a,
            f"t{spec.tile_m}x{spec.tile_n}x{spec.tile_k}",
            f"w{spec.warp_m}x{spec.warp_n}",
            f"a{spec.warp_tile_m}x{spec.warp_tile_n}x{spec.warp_tile_k}",
            f"{spec.pipeline}_{spec.epilogue}",
            spec.acc_epilogue.tag(),
        )


def build_implicit_gemm_conv_dynamic(
    spec: ImplicitGemmConvSpec,
    arch: str = "gfx1151",
    geom: Optional[DynamicConvGeometry] = None,
) -> KernelDef:
    """Build a runtime-generic forward-conv kernel.

    Thin wrapper over :func:`build_implicit_gemm_conv` with a
    :class:`DynamicConvGeometry` injected via the ``dyn`` hook. All of the
    pipeline / LDS / MMA / epilogue machinery is reused unchanged; only the
    descriptor addressing and the M / N_gemm / K_gemm bounds become runtime SSA.

    The dynamic path requires the plain ``scf_for_iter`` K-loop, so ``spec`` must
    have ``async_dma=False``, ``unroll_k=False``, ``pipeline != "basic"``, and
    ``chiplet_swizzle=False`` (grid tile counts are unknown at build time). These
    are validated inside :func:`build_implicit_gemm_conv`.
    """
    if geom is None:
        geom = DynamicConvGeometry()
    return build_implicit_gemm_conv(spec, arch=arch, dyn=geom)
