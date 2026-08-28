# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Implicit-GEMM backward-weight convolution (wgrad) kernel instance.

Computes the weight gradient of a 2-D (or 3-D) convolution:

    dW[k, y, x, c] = sum_{n, ho, wo} dY[n, ho, wo, k] * X[n, hi, wi, c]
    where  hi = ho*sH - pH + y*dH,  wi = wo*sW - pW + x*dW

This is an implicit-GEMM of shape:

    M     = K             (output-channel / weight row dimension)
    N_wg  = Y*X*C         (weight column dimension — filter spatial × input channel)
    K_wg  = N*Ho*Wo       (reduction dimension — over output spatial positions)

So:
    A operand: dY (output gradient), layout NHWK  →  GEMM A: (K_wg, M)ᵀ = (N*Ho*Wo, K)
    B operand: X  (input activations), layout NHWC →  GEMM B: (K_wg, N_wg) = (N*Ho*Wo, Y*X*C)
    D operand: dW (weight gradient),   layout KYXC →  GEMM D: (M, N_wg)   = (K, Y*X*C)

The B descriptor reuses :func:`make_a_descriptor` from
:mod:`._conv_implicit_gemm_common`: the convolution address map for the input
tensor X is exactly the same as in the forward pass, with ``k_wg`` playing the
role of the K tile column (it unpacks via the same ``unmerge('k' → y,x,c)``
chain).  The A descriptor (dY) is a simple NHWK naive tensor unmerged over the
``k_wg`` reduction axis.

Authoring style (what the kernel writer types)::

    spec = WgradConvSpec(
        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3,
                            sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),
        tile_m=64, tile_n=64, tile_k=64,
        warp_m=2, warp_n=2,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    )
    kernel = build_implicit_gemm_conv_wgrad(spec)

GEMM dimension mapping
----------------------
Forward::

    GEMM-M   = N*Ho*Wo      (output spatial positions)
    GEMM-N   = K            (output channels)
    GEMM-K   = Y*X*C        (filter × input channel)

Wgrad::

    GEMM-M   = K            (output channels, weight rows)
    GEMM-N   = Y*X*C        (filter spatial × input channel, weight cols)
    GEMM-K   = N*Ho*Wo      (output spatial positions, reduction)

Pipeline and epilogue options match the forward builder; see
:class:`WgradConvSpec` for the field descriptions.

Split-K
-------
Wgrad is reduction-heavy: K_wg = N*Ho*Wo can be orders of magnitude larger
than the M*N tile area (e.g. K_wg = 25,088 vs M*N = 64*576 for the bake-off
shape).  When the M×N grid is too small to saturate the device, split-K
partitions K_wg into ``split_k`` equal slices along the Z grid dimension and
atomic-adds each CTA's partial f32 accumulator directly into ``dW`` via
``global_atomic_add``.

Supported output dtypes for split-K:
  - ``fp32``: scalar ``global_atomic_add`` (f32 atomicrmw fadd, gfx940+).
  - ``bf16``: packed ``global_atomic_add_pk_bf16`` (<2 x bfloat>, gfx940+).
  - ``fp16``: packed ``global_atomic_add_pk_f16`` (<2 x half>, gfx940+).

The ``dW`` pointer type and the kernel ABI are identical between
split_k=1 and split_k>1 — no extra parameters.

Caller contract (``split_k > 1``):
  1. **Zero-initialise the ``dW`` buffer before every launch.**  The kernel only
     issues atomic-adds, never a direct store, so any non-zero initial content
     accumulates into the result.  Forgetting this step produces silently wrong
     gradients with no runtime error.
  2. Launch with grid ``(ceil(wg_N/tile_n), ceil(wg_M/tile_m), split_k)``.

When ``split_k == 1`` the kernel writes ``dW`` normally (no atomics).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import List, Optional, Sequence, Tuple

from ...core.ir import (
    BF16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ...helpers.atoms import MfmaAtom, mfma_atom
from ...helpers.epilogues import CShuffleEpilogue, DirectEpilogue
from ...helpers.geometry import WarpGrid
from ...helpers.layouts import LdsLayout
from ...helpers.loads import AsyncTileLoader, CoalescedTileLoader
from ...helpers.mfma_gemm_inner import decode_mfma_lanes
from ...helpers.pipeline import SoftwarePipeline
from ...helpers.schedule import SchedulePolicy
from ...helpers.spec import kernel_name_join
from ...helpers.tensor_view import make_buffer_resource
from ...helpers.transforms import TensorDescriptor, embed, pad, unmerge_magic
from ._conv_implicit_gemm_common import (
    ConvAccumulatorEpilogue,
    ConvDataSpec,
    ConvProblem,
    _apply_accumulator_epilogue,
    _emit_frag_smem_load,
    _emit_mfma,
    _emit_smem_load,
    _ir_dtype,
    make_a_descriptor,
)


# ---------------------------------------------------------------------
# Wgrad-specific GEMM dimension helpers
# ---------------------------------------------------------------------


def _wg_M(p: ConvProblem) -> int:
    """Wgrad GEMM-M: output channels (per group)."""
    return p.kpg


def _wg_N(p: ConvProblem) -> int:
    """Wgrad GEMM-N: filter spatial × input channels (per group)."""
    z = p.Z if p.is_3d else 1
    return z * p.Y * p.X * p.cpg


def _wg_K(p: ConvProblem) -> int:
    """Wgrad GEMM-K (reduction): output spatial positions."""
    base = p.N * p.Ho * p.Wo
    return base * p.Do if p.is_3d else base


# ---------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------


def make_dy_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (k_wg, m_wg) -> N[D]HWK offset descriptor for dY (output gradient).

    dY is stored in NHWK layout.  In the wgrad GEMM:
      - the M dimension indexes output channels  ``k_out ∈ [0, K)``
      - the K_wg reduction dimension indexes output positions  ``m_fwd ∈ [0, N*Ho*Wo)``

    The descriptor unpacks ``k_wg = m_fwd`` back into ``(n, ho, wo)`` (or
    ``(n, do, ho, wo)`` for 3-D) via :func:`~._conv_implicit_gemm_common.unmerge_magic`
    so each thread can compute its NHWK byte offset directly.

    2-D::

        naive(NHWK):         (n, ho, wo, k_out)
        unmerge('k_wg' → n, ho, wo)   →  user-facing: (k_wg, k_out=m_wg)

    3-D::

        naive(NDHWK):        (n, do, ho, wo, k_out)
        unmerge('k_wg' → n, do, ho, wo)  →  user-facing: (k_wg, k_out=m_wg)
    """
    if p.is_3d:
        return TensorDescriptor.naive(
            "dY_ndhwk",
            lengths=[p.N, p.Do, p.Ho, p.Wo, p.K],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "do", "ho", "wo", "k_out"],
        ).transform(
            unmerge_magic(
                "k_wg", into=["n", "do", "ho", "wo"], dims=[p.N, p.Do, p.Ho, p.Wo]
            )
        )
    return TensorDescriptor.naive(
        "dY_nhwk",
        lengths=[p.N, p.Ho, p.Wo, p.K],
        dtype=_ir_dtype(dtype),
        coord_names=["n", "ho", "wo", "k_out"],
    ).transform(unmerge_magic("k_wg", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo]))


def make_x_wgrad_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (k_wg, n_wg) -> N[D]HWC offset descriptor for X (input activations).

    In the wgrad GEMM, X is the B operand:
      - the K_wg reduction dimension indexes output positions ``k_wg = m_fwd``
      - the N_wg dimension indexes filter+channel positions ``n_wg ∈ [0, Y*X*C)``

    The descriptor is the same coordinate-transform DAG as the forward A
    descriptor (see :func:`~._conv_implicit_gemm_common.make_a_descriptor`),
    with the ``k_wg`` role replacing ``m`` and ``n_wg`` replacing ``k``.
    Reusing ``make_a_descriptor`` with ``decompose_m=True`` gives exactly this:
    it maps ``(m=k_wg, k=n_wg)`` → NHWC offset with the convolution embed +
    boundary-check pad chain.
    """
    return make_a_descriptor(p, decompose_m=True, dtype=dtype)


def make_dw_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (m_wg, n_wg) -> K[Z]YXC offset descriptor for dW (weight gradient).

    dW is stored in KYXC (2-D) or KZYXC (3-D) layout:
      - m_wg indexes output channels ``k_out ∈ [0, K)``
      - n_wg indexes filter+channel positions ``∈ [0, Y*X*C)``

    This mirrors :func:`~.conv_implicit_gemm.make_b_descriptor` from the
    forward pass — the weight tensor has the same layout for both reading
    (forward B) and writing (wgrad D).

    2-D::

        naive(KYXC):          (k_out, y, x, c)
        unmerge('n_wg' → y, x, c)   →  user-facing: (k_out=m_wg, n_wg)
        pad('y'), pad('x')           →  partial-tile boundary guard

    3-D::

        naive(KZYXC):         (k_out, z, y, x, c)
        unmerge('n_wg' → z, y, x, c)
        pad('z'), pad('y'), pad('x')
    """
    # Grouped (groups>1): the weight-gradient tensor is PACKED per group —
    # dW is [K, [Z,] Y, X, cpg] (PyTorch grouped weight [K, C/groups, kH, kW]),
    # NOT dense over the full C.  The output channel ``k_out`` (∈ [0, K)) already
    # encodes the group (g = k_out // kpg); the caller passes the group-absolute
    # ``k_out = group*kpg + k_out_in_group``.  The n_wg reduction spans only
    # ``Y*X*cpg``, so the C dim is simply ``cpg`` and there is no group→c embed
    # (the group's channel slab is the whole packed C dim of this tensor).
    cdim = p.cpg if p.groups > 1 else p.C
    if p.is_3d:
        return TensorDescriptor.naive(
            "dW_kzyxc",
            lengths=[p.K, p.Z, p.Y, p.X, cdim],
            dtype=_ir_dtype(dtype),
            coord_names=["k_out", "z", "y", "x", "c"],
        ).transform(
            unmerge_magic(
                "n_wg", into=["z", "y", "x", "c"], dims=[p.Z, p.Y, p.X, cdim]
            ),
            pad("z", lo=0, hi=p.Z),
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        )
    return TensorDescriptor.naive(
        "dW_kyxc",
        lengths=[p.K, p.Y, p.X, cdim],
        dtype=_ir_dtype(dtype),
        coord_names=["k_out", "y", "x", "c"],
    ).transform(
        unmerge_magic("n_wg", into=["y", "x", "c"], dims=[p.Y, p.X, cdim]),
        pad("y", lo=0, hi=p.Y),
        pad("x", lo=0, hi=p.X),
    )


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class WgradConvSpec:
    """One concrete implicit-GEMM backward-weight convolution configuration.

    GEMM orientation:
      M     = K             (output channels — weight rows)
      N_wg  = Y*X*C         (filter spatial × input channel — weight cols)
      K_wg  = N*Ho*Wo       (output positions — reduction)

    Pipeline, epilogue, and async-DMA options are the same as
    :class:`~.conv_implicit_gemm.ImplicitGemmConvSpec`.  Grouped convolution
    (``groups > 1``) is supported via grid-per-group (the group index rides on
    block_id_z) and is orthogonal to the epilogue and split-K: every epilogue
    (direct, cshuffle, wmma-direct, and the split-K atomic path) threads the
    per-group ``k_out += group*kpg`` fold.  Only grouped pointwise (1x1) wgrad
    remains a follow-on.
    """

    problem: ConvProblem
    name: str = "conv_igemm_wgrad"
    data: ConvDataSpec = field(default_factory=ConvDataSpec)

    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 64

    warp_m: int = 2
    warp_n: int = 2

    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    wave_size: int = 64

    pipeline: str = "mem"
    epilogue: str = "default"
    async_dma: bool = False
    unroll_k: bool = False
    lds_k_pad: Optional[int] = None

    vector_size_a: Optional[int] = None
    vector_size_b: Optional[int] = None
    vector_size_c: Optional[int] = None

    lds_layout: Optional[LdsLayout] = None

    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64

    waves_per_eu: Optional[int] = None
    acc_epilogue: ConvAccumulatorEpilogue = field(
        default_factory=ConvAccumulatorEpilogue
    )
    # Split-K: partition K_wg into this many equal slices along block_id_z.
    # -1 = auto (resolved by build_implicit_gemm_conv_wgrad via the CK formula).
    #  0 = runtime atomic: kernel accepts ``ks`` as an i32 argument; the caller
    #      supplies the slice width at launch.  Only one kernel is compiled; any
    #      split-K degree is achievable at runtime by varying grid-Z and ``ks``.
    #      Caller must zero-init dW before every launch.
    #  1 = disabled (default, normal store).
    # >1 = fixed atomic: the exact degree is baked into the kernel name/IR.
    #      Caller must zero-init dW before launch.
    # ABI for 0 and >1: dW is not writeonly; K_wg is padded as needed.
    split_k: int = 1

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.wave_size

    @property
    def k_atoms_per_tile_k(self) -> int:
        return self.tile_k // self.warp_tile_k

    @property
    def mfmas_per_warp_m(self) -> int:
        return self.tile_m // (self.warp_m * self.warp_tile_m)

    @property
    def mfmas_per_warp_n(self) -> int:
        return self.tile_n // (self.warp_n * self.warp_tile_n)

    @property
    def atom(self) -> MfmaAtom:
        return mfma_atom(
            self.data.dtype_a, self.warp_tile_m, self.warp_tile_n, self.warp_tile_k
        )

    # ---- wgrad GEMM dimensions ----

    @property
    def wg_M(self) -> int:
        # GEMM-M: per-group output channels kpg (== K when groups==1).
        return _wg_M(self.problem)

    @property
    def wg_N(self) -> int:
        # GEMM-N: per-group Y*X*cpg (== Y*X*C when groups==1).
        return _wg_N(self.problem)

    @property
    def wg_K(self) -> int:
        return _wg_K(self.problem)

    def wg_K_padded(self, split_k: Optional[int] = None) -> int:
        """K_wg rounded up to the nearest multiple of ``tile_k * split_k``.

        When ``split_k=0`` (runtime) the caller must pass the concrete degree
        to use for padding (e.g. the degree that will be used at launch time).
        """
        sk = split_k if split_k is not None else self.split_k
        if sk <= 0:
            raise ValueError(
                "wg_K_padded requires a concrete split_k degree (>= 1); "
                f"got {sk!r}. Pass the launch-time degree explicitly."
            )
        stride = self.tile_k * sk
        k = _wg_K(self.problem)
        return ((k + stride - 1) // stride) * stride

    def kernel_name(self) -> str:
        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warp_m}x{self.warp_n}",
            f"a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"{self.pipeline}_{self.epilogue}",
            self.acc_epilogue.tag(),
            flags={
                "async": self.async_dma,
                f"spk{self.split_k}": self.split_k > 1,
                "spkauto": self.split_k == -1,
                "spkrt": self.split_k == 0,
            },
        )

    def validate(self) -> None:
        if self.tile_m % (self.warp_m * self.warp_tile_m) != 0:
            raise ValueError(
                f"tile_m {self.tile_m} not divisible by warp_m * warp_tile_m "
                f"({self.warp_m} * {self.warp_tile_m})"
            )
        if self.tile_n % (self.warp_n * self.warp_tile_n) != 0:
            raise ValueError(
                f"tile_n {self.tile_n} not divisible by warp_n * warp_tile_n "
                f"({self.warp_n} * {self.warp_tile_n})"
            )
        if self.tile_k % self.warp_tile_k != 0:
            raise ValueError(
                f"tile_k {self.tile_k} not divisible by warp_tile_k {self.warp_tile_k}"
            )
        if self.block_size > 1024:
            raise ValueError(f"block_size {self.block_size} > 1024")
        if self.split_k < -1:
            raise ValueError(
                f"split_k must be -1 (auto), 0 (runtime atomic), 1 (disabled), "
                f"or >1 (fixed); got {self.split_k}"
            )
        if self.split_k == 0 or self.split_k > 1:
            if self.data.dtype_d not in ("fp32", "bf16", "fp16"):
                raise ValueError(
                    f"split_k atomic requires dtype_d in fp32/bf16/fp16 "
                    f"(got {self.data.dtype_d!r})"
                )
            if self.data.dtype_d in ("bf16", "fp16") and self.problem.cpg % 2 != 0:
                raise ValueError(
                    f"split_k atomic with dtype_d={self.data.dtype_d!r} requires even cpg "
                    f"(packed <2 x dtype> atomic pairs must stay within one filter "
                    f"position; the packed dW inner dim is cpg=C/groups); "
                    f"got cpg={self.problem.cpg} (C={self.problem.C}, groups={self.problem.groups})"
                )
        if self.data.dtype_d in ("bf16", "fp16") and self.epilogue == "default":
            raise ValueError(
                f"split_k atomic with dtype_d={self.data.dtype_d!r} requires "
                f"epilogue='cshuffle' (default emits zero-fill packed atomics with "
                f"scattered MFMA layout; cshuffle produces contiguous pairs)"
            )
        layout = self.effective_lds_layout()
        if self.async_dma:
            layout.validate_for_async()
        if self.async_dma and self.lds_k_pad not in (None, 0):
            raise ValueError(
                "async_dma requires lds_k_pad to be 0/None because "
                "raw_ptr_buffer_load_lds writes a packed lane-contiguous tile"
            )
        if (
            self.acc_epilogue.clamp_min is not None
            and self.acc_epilogue.clamp_max is not None
            and self.acc_epilogue.clamp_min > self.acc_epilogue.clamp_max
        ):
            raise ValueError(
                "acc_epilogue clamp_min must be <= clamp_max "
                f"(got {self.acc_epilogue.clamp_min} > {self.acc_epilogue.clamp_max})"
            )

    def effective_lds_layout(self) -> LdsLayout:
        if self.lds_layout is not None:
            layout = self.lds_layout
        elif self.lds_k_pad is not None:
            layout = LdsLayout.padded_k(self.tile_k, self.lds_k_pad)
        elif self.async_dma:
            layout = LdsLayout.packed_async(self.tile_k)
        else:
            layout = LdsLayout.padded_k(self.tile_k, 8 if self.tile_k >= 16 else 0)
        layout.validate()
        return layout

    @staticmethod
    def default_vector_sizes(
        C: int, K: int, dtype: str, split_k: int = 1
    ) -> "Tuple[int, int, int]":
        """Return ``(vec_a, vec_b, vec_c)`` for a wgrad problem.

        Wgrad memory layout:
          A (dY):  NHWK → last dim K → vec_a
          B (X):   NHWC → last dim C → vec_b
          D (dW):  KYXC → last dim C → vec_c

        When ``split_k != 1`` (including ``split_k == 0`` for runtime selection)
        the epilogue is ``default`` (direct scalar store), which does not support
        vec_c > 1, so vec_c is forced to 1.
        """
        sizes = [8, 4, 2, 1] if dtype != "fp32" else [4, 2, 1]

        def _vec(n: int) -> int:
            return next(v for v in sizes if n % v == 0)

        vec_c = 1 if split_k != 1 else _vec(C)
        return _vec(K), _vec(C), vec_c


# ---------------------------------------------------------------------
# Arch-aware spec validation
# ---------------------------------------------------------------------


def is_valid_wgrad_spec(spec: WgradConvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    Checks geometry divisibility, block-size cap, MMA-atom availability, and
    LDS budget — the same gates as the forward :func:`~.conv_implicit_gemm.is_valid_spec`
    but applied to the wgrad GEMM dimensions (M=K, N=Y*X*C, K_red=N*Ho*Wo).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.tile_m % (spec.warp_m * spec.warp_tile_m):
        return False, "tile_m not divisible by warp_m * warp_tile_m"
    if spec.tile_n % (spec.warp_n * spec.warp_tile_n):
        return False, "tile_n not divisible by warp_n * warp_tile_n"
    if spec.tile_k % spec.warp_tile_k:
        return False, "tile_k not divisible by warp_tile_k"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > {target.max_threads_per_block} "
            f"(hardware cap) on {arch}"
        )
    if (
        spec.vector_size_c is not None
        and spec.vector_size_c > 1
        and spec.epilogue == "default"
    ):
        return False, (
            f"default epilogue is not supported with vector size c: {spec.vector_size_c}"
        )

    family = "wmma" if target.wave_size == 32 else "mma"
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    sk = spec.split_k
    if sk < -1:
        return False, f"split_k must be -1 (auto), 0 (runtime), 1, or >1 (got {sk})"
    # -1 = auto: resolved at build time; always valid at the spec-check stage.
    # 0 = runtime atomic; validate constraints identically to >1 without a degree.
    _is_atomic = sk == 0 or sk > 1
    if _is_atomic and spec.data.dtype_d not in ("fp32", "bf16", "fp16"):
        return False, (
            f"split_k atomic requires dtype_d in fp32/bf16/fp16 "
            f"(got {spec.data.dtype_d!r})"
        )
    if (
        _is_atomic
        and spec.data.dtype_d in ("bf16", "fp16")
        and spec.problem.cpg % 2 != 0
    ):
        return False, (
            f"split_k atomic with dtype_d={spec.data.dtype_d!r} requires even cpg "
            f"(packed <2 x dtype> atomic pairs must stay within one filter "
            f"position; the packed dW inner dim is cpg=C/groups); "
            f"got cpg={spec.problem.cpg}"
        )
    if spec.data.dtype_d in ("bf16", "fp16") and spec.epilogue == "default":
        return False, (
            f"split_k atomic with dtype_d={spec.data.dtype_d!r} requires "
            f"epilogue='cshuffle' (default emits zero-fill packed atomics with "
            f"scattered MFMA layout; cshuffle produces contiguous pairs)"
        )

    atom = (spec.warp_tile_m, spec.warp_tile_n, spec.warp_tile_k)
    if not target.mma.has_shape(
        family=family,
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    ):
        return False, f"unsupported {spec.data.dtype_a} warp_tile {atom} on {arch}"

    # Grouped wgrad (groups>1): grid-per-group, the group index on block_id_z.
    # Grouping is orthogonal to the epilogue/vec/split_k -- it only shifts the dW
    # output-channel slab, and every epilogue (direct, cshuffle, wmma-direct, and
    # the split-K atomic path) threads the per-group fold ``k_out += group*kpg``.
    # When split_k>1 the group and the K-slice share the z axis
    # (z = groups*split_k; see the kernel body).  Only grouped pointwise (1x1)
    # remains a follow-on.
    if spec.problem.groups > 1:
        if spec.problem.is_pointwise:
            return False, "grouped pointwise (1x1) wgrad is not yet supported"

    _ab_dtype_bytes = 4 if spec.data.dtype_a in ("fp32",) else 2
    _lds_layout = spec.effective_lds_layout()
    _a_shape = _lds_layout.storage_shape(spec.tile_m)
    _b_shape = _lds_layout.storage_shape(spec.tile_n)
    _ab_bytes = (
        _a_shape[0] * _a_shape[1] + _b_shape[0] * _b_shape[1]
    ) * _ab_dtype_bytes
    _double = spec.pipeline == "compv4" or spec.async_dma or spec.unroll_k
    _ab_lds = _ab_bytes * (2 if _double else 1)
    _c_dtype_bytes = 4 if spec.data.dtype_d == "fp32" else 2
    _c_lds = (
        spec.tile_m * spec.tile_n * _c_dtype_bytes if spec.epilogue == "cshuffle" else 0
    )
    _total_lds = _ab_lds + _c_lds
    if not target.fits_lds(_total_lds):
        return False, (
            f"LDS budget {_total_lds} bytes "
            f"(A/B={'x2 ' if _double else ''}{_ab_bytes}, C={_c_lds}) "
            f"> {target.lds_capacity_bytes} cap on {arch}"
        )

    if family == "wmma":
        # 16x16x16 is the RDNA WMMA hero (gfx1151/gfx1201); 16x16x32 is the
        # gfx1250 hero (its only fp16/bf16 WMMA atom -- there is no 16x16x16 on
        # gfx1250). Both feed the same emit_wmma_phase fragment machinery
        # (a_frag_len 8 vs 16, assembled from 8-wide ds_read chunks).
        if atom not in ((16, 16, 16), (16, 16, 32)):
            return False, (
                f"WMMA wgrad supports only 16x16x16 or 16x16x32 (got {atom}) on {arch}"
            )
        if spec.pipeline != "mem":
            return False, (
                f"WMMA wgrad supports only the 'mem' pipeline "
                f"(got {spec.pipeline!r}) on {arch}"
            )
        if spec.epilogue != "default":
            return False, (
                f"WMMA wgrad supports only the 'default' epilogue "
                f"(got {spec.epilogue!r}) on {arch}"
            )
        for flag, label in (
            (spec.async_dma, "async_dma"),
            (spec.unroll_k, "unroll_k"),
            (spec.chiplet_swizzle, "chiplet_swizzle"),
        ):
            if flag:
                return False, f"WMMA wgrad does not support {label} on {arch}"

    return True, "ok"


def _wgrad_mma_family(arch: str) -> str:
    from ...core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_wgrad_op(spec: WgradConvSpec, arch: str):
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    op = target.mma.op_for_shape(
        family=_wgrad_mma_family(arch),
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    )
    if op is None:
        raise ValueError(
            f"no MMA atom for wgrad warp_tile "
            f"({spec.warp_tile_m},{spec.warp_tile_n},{spec.warp_tile_k}) on {arch}"
        )
    return op


# ---------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------


def build_implicit_gemm_conv_wgrad(
    spec: WgradConvSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the IR for one implicit-GEMM backward-weight conv kernel.

    GEMM shape:
        M     = K             (output channels)
        N_wg  = Y*X*C         (filter spatial × input channel)
        K_wg  = N*Ho*Wo       (output spatial positions, reduction)

    Operands:
        A (dY): output-gradient tensor, NHWK layout.
                GEMM role: A (M=K rows × K_wg=N*Ho*Wo cols after transpose).
        B (X):  input-activation tensor, NHWC layout.
                GEMM role: B (K_wg=N*Ho*Wo rows × N_wg=Y*X*C cols).
                Uses the same coordinate-transform DAG as the forward A
                descriptor (convolution affine embed + boundary pad).
        D (dW): weight-gradient output, KYXC layout.
                GEMM role: D (M=K rows × N_wg=Y*X*C cols).

    Pipeline and epilogue options mirror the forward builder.

    When ``spec.split_k == -1`` the split-K degree is chosen automatically
    using the CK formula (``helpers.split_k.select_split_k_wgrad``):
    ``floor((waves_per_cu * num_cus) / base_grid)`` clamped to ``[1, wg_K]``.
    """
    # Resolve split_k=-1 (auto) before validate() so validation sees the real value.
    if spec.split_k == -1:
        from ...helpers.split_k import select_split_k_wgrad
        from dataclasses import replace as _dc_replace

        decision = select_split_k_wgrad(
            wg_M=_wg_M(spec.problem),
            wg_N=_wg_N(spec.problem),
            wg_K=_wg_K(spec.problem),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=arch,
        )
        spec = _dc_replace(spec, split_k=decision.split_k)

    spec.validate()
    ok, why = is_valid_wgrad_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid wgrad spec for {arch}: {why}")

    p = spec.problem
    # Grouped wgrad is grid-per-group: one workgroup per conv group, the group
    # index riding on block_id_z.  The load side (dY, X, GEMM dims, grid) is built
    # straight from the per-group problem.
    p_load = p
    ir_dtype_a = _ir_dtype(spec.data.dtype_a)
    ir_dtype_b = _ir_dtype(spec.data.dtype_b)
    ir_dtype_d = _ir_dtype(spec.data.dtype_d)

    _is_split_k = spec.split_k > 1 or spec.split_k == 0
    _split_k_runtime = spec.split_k == 0  # ks passed as kernel arg at launch
    _grouped = p_load.groups > 1

    b = IRBuilder(spec.kernel_name())
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    # dY (output gradient): A operand in the GEMM sense (K rows, K_wg reduction)
    dY = b.param(
        "dY", PtrType(ir_dtype_a, "global"), noalias=True, readonly=True, align=16
    )
    # X (input activations): B operand in the GEMM sense (K_wg reduction, N_wg cols)
    X = b.param(
        "X", PtrType(ir_dtype_b, "global"), noalias=True, readonly=True, align=16
    )
    # dW (weight gradient): output D.
    # split_k=1: normal writeonly store.
    # split_k>1: atomic-add into caller-zero-init dW; writeonly dropped because
    #            atomicrmw is read+modify+write (dtype_d in fp32/bf16/fp16).
    _dw_writeonly = not _is_split_k
    dW = b.param(
        "dW",
        PtrType(ir_dtype_d, "global"),
        noalias=True,
        writeonly=_dw_writeonly,
        align=16,
    )
    dY_bytes = b.param("dY_bytes", I32)
    X_bytes = b.param("X_bytes", I32)
    dW_bytes = b.param("dW_bytes", I32)
    # Runtime split-K: ks = slice width per CTA, computed and passed by the launcher.
    # Only present when split_k == 0; fixed-degree kernels bake ks as a constant.
    # ks_count = number of slices; only needed when grouped+runtime so the kernel
    # can decode (group, slice) from block_id_z = group*ks_count + slice.
    _ks_param = b.param("ks", I32) if _split_k_runtime else None
    _ks_count_param = (
        b.param("ks_count", I32) if (_split_k_runtime and _grouped) else None
    )

    op = _resolve_wgrad_op(spec, arch)
    atom = spec.atom if op.family == "mma" else None
    a_per_lane = op.a_frag_len
    b_per_lane = op.b_frag_len
    _smem_dtype: Optional[Type] = (
        BF16 if op.a_dtype == "bf16" else F32 if op.a_dtype == "fp32" else None
    )
    c_per_lane = op.c_frag_len

    # Wgrad GEMM dims (per group): kpg × Y*X*cpg, reduction N*Ho*Wo.
    wg_M = _wg_M(p_load)  # kpg  (K when groups==1)
    wg_N = _wg_N(p_load)  # Y*X*cpg  (Y*X*C when groups==1)
    wg_K = _wg_K(p_load)  # N*Ho*Wo (reduction; group-independent)

    block_m, block_n, block_k = spec.tile_m, spec.tile_n, spec.tile_k

    grid = WarpGrid.from_atom(
        op,
        tile_m=block_m,
        tile_n=block_n,
        tile_k=block_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        wave_size=spec.wave_size,
    ).bind(b, block_m_axis="y", block_n_axis="x")
    tid = grid.tid
    lane = grid.lane
    warp_id = grid.warp_id
    warp_m_idx = grid.warp_m_idx
    warp_n_idx = grid.warp_n_idx

    c0 = b.const_i32(0)
    c_block_k = b.const_i32(block_k)
    c_wg_K = b.const_i32(wg_K)

    # Grouped wgrad: the group index rides on ``block_id_z`` (grid-per-group,
    # matching forward).  When split_k>1 the group and the K-slice SHARE the z
    # axis: the grid launches z = groups*split_k and every CTA decodes
    # ``group = block_id_z // split_k`` and ``slice = block_id_z % split_k``.
    # split_k==0 (runtime): same scheme but split_k degree is runtime; the kernel
    # receives ``ks_count`` (the degree) as an extra arg alongside ``ks`` (slice
    # width) so both div and mod can be emitted.  Grid: z = groups * ks_count.
    # ``group_v`` stays None on the ungrouped path so all groups==1 IR is
    # byte-identical to the pre-grouped kernel.
    grouped = _grouped
    group_v = None
    if grouped:
        c_kpg = b.const_i32(p_load.kpg)  # kpg: dY output-channel slab stride
        if not _is_split_k:
            group_v = b.to_sgpr_u32(b.block_id_z())

    # Split-K K-slice bounds.  K_wg is padded so every slice is exactly ks
    # elements wide and ks % tile_k == 0.  OOB loads return 0 via buffer clamp.
    # split_k == 1: k_lo = 0, k_hi = None (loop runs to c_wg_K, unpadded).
    # split_k == 0: ks/ks_count are runtime kernel args; k_lo/k_hi computed at runtime.
    # split_k > 1: ks is a compile-time constant.
    if _is_split_k:
        if _split_k_runtime:
            c_ks = _ks_param  # runtime i32 from kernel arg
            if grouped:
                # z = group * ks_count + slice; decode both at runtime.
                z_id = b.block_id_z()
                group_v = b.to_sgpr_u32(b.div(z_id, _ks_count_param))
                slice_id = b.mod(z_id, _ks_count_param)
                k_lo = b.to_sgpr_u32(b.mul(slice_id, c_ks))
            else:
                k_lo = b.to_sgpr_u32(b.mul(b.block_id_z(), c_ks))
        else:
            wg_K_padded = spec.wg_K_padded()
            c_ks = b.const_i32(wg_K_padded // spec.split_k)
            if grouped:
                # z = group*split_k + slice: recover both from block_id_z.
                c_split_k = b.const_i32(spec.split_k)
                z_id = b.block_id_z()
                group_v = b.to_sgpr_u32(b.div(z_id, c_split_k))
                slice_id = b.mod(z_id, c_split_k)
                k_lo = b.to_sgpr_u32(b.mul(slice_id, c_ks))
            else:
                k_lo = b.to_sgpr_u32(b.mul(b.block_id_z(), c_ks))
        k_hi = b.to_sgpr_u32(b.add(k_lo, c_ks))
    else:
        k_lo = c0
        k_hi = None

    # Chiplet swizzle (same logic as forward; tile counts from wgrad GEMM dims)
    if spec.chiplet_swizzle:
        from ...helpers.grid import chiplet_aware_super_tile

        num_pid_m = (wg_M + block_m - 1) // block_m
        num_pid_n = (wg_N + block_n - 1) // block_n
        c_num_pid_n = b.const_i32(num_pid_n)
        wgid_flat = b.add(b.mul(b.block_id_y(), c_num_pid_n), b.block_id_x())
        swz = chiplet_aware_super_tile(
            b,
            wgid_flat,
            num_pid_m=num_pid_m,
            num_pid_n=num_pid_n,
            wgm=spec.chiplet_wgm,
            num_xcds=spec.chiplet_num_xcds,
            chunk_size=spec.chiplet_chunk_size,
        )
        block_m_off_v = b.mul(swz.row, b.const_i32(block_m))
        block_n_off_v = b.mul(swz.col, b.const_i32(block_n))
        grid = dc_replace(grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v)
    else:
        block_m_off_v = grid.block_m_off
        block_n_off_v = grid.block_n_off

    lds_layout = spec.effective_lds_layout()
    if spec.async_dma:
        lds_layout.validate_for_async()
    A_smem = b.smem_alloc(
        ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem"
    )
    B_smem = b.smem_alloc(
        ir_dtype_b, lds_layout.storage_shape(block_n), name_hint="B_smem"
    )
    double_buffer = spec.pipeline == "compv4" or spec.async_dma or spec.unroll_k
    if double_buffer:
        A_smem2 = b.smem_alloc(
            ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem2"
        )
        B_smem2 = b.smem_alloc(
            ir_dtype_b, lds_layout.storage_shape(block_n), name_hint="B_smem2"
        )
    else:
        A_smem2 = A_smem
        B_smem2 = B_smem

    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n
    k_atoms = spec.k_atoms_per_tile_k

    acc_init = b.zero_vec_f32(c_per_lane)
    accs = [
        (f"acc_m{mi}_n{ni}", acc_init) for mi in range(mfmas_m) for ni in range(mfmas_n)
    ]

    threads = spec.block_size
    # A (dY, NHWK) and B (X, NHWC) have their GEMM reduction axis K_wg = N*Ho*Wo,
    # which is NOT the stride-1 tensor axis: in NHWK the stride between adjacent
    # K_wg positions is K, in NHWC it is C.  A naive vectorised load along K_wg
    # would read consecutive *channel* values at one spatial position instead of
    # the next spatial position (wrong data) -- hence the historical vec=1.
    #
    # The stride-1 axis is instead the GEMM *free* axis: k_out (= M) for dY and
    # the inner C of N_wg for X.  So vectorise the global load along that free
    # (row) axis and transpose it into the row-major (M/N, K) LDS tile on store
    # (CoalescedTileLoader vector_axis="row").  The transpose-on-store fills the
    # SAME row-major (M/N, K) LDS tile the scalar path produced (identical LDS
    # contents; only the load/store instructions differ), so the MMA consumer --
    # MFMA or WMMA -- reads it unchanged.
    #
    # Enabled for every sync path (MFMA and WMMA); only the async
    # (raw_ptr_buffer_load_lds) path is excluded, because it writes
    # lane-contiguous LDS and cannot host a transpose-on-store.  Correctness
    # requires the free-axis vector to stay within one stride-1 run -- vec_a | K
    # (dY's k_out is contiguous over the whole K dim) and vec_b | C (an X vector
    # must not cross a (y,x) filter boundary) -- and choose_vec additionally
    # enforces even tile distribution.  When a width > 1 is not admissible the
    # loader falls back to the scalar vector_axis="col" path, keeping those
    # configs byte-identical.
    axis_a = axis_b = "col"
    load_vec_a = 1
    load_vec_b = 1
    if not spec.async_dma:

        def _free_axis_vec(chan: int, dtype: str) -> int:
            widths = (8, 4, 2, 1) if dtype != "fp32" else (4, 2, 1)
            return next(v for v in widths if chan % v == 0)

        va = CoalescedTileLoader.choose_vec(
            tile_rows=block_m,
            tile_cols=block_k,
            block_size=threads,
            max_vec=_free_axis_vec(p_load.kpg, spec.data.dtype_a),
            vector_axis="row",
        )
        vb = CoalescedTileLoader.choose_vec(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            max_vec=_free_axis_vec(p_load.cpg, spec.data.dtype_b),
            vector_axis="row",
        )
        if va > 1:
            load_vec_a, axis_a = va, "row"
        if vb > 1:
            load_vec_b, axis_b = vb, "row"

    # For pointwise (Y=X=1, stride 1, pad 0) all descriptors collapse to flat
    # 2-D address arithmetic — no unmerge/embed/pad, just multiply+add:
    #   dY: offset = k_wg * K + k_out   (NHWK with M=N*Ho*Wo pre-multiplied)
    #   X:  offset = k_wg * C + n_wg    (NHWC, n_wg == c directly)
    #   dW: offset = k_out * C + n_wg   (KC, n_wg == c directly)
    if p.is_pointwise:
        dY_desc = None
        X_desc = None
        _c_K_ir = b.const_i32(p.kpg)
        _c_C_ir = b.const_i32(p.cpg)
        _c_wgM_ir = b.const_i32(wg_M)
        _c_wgN_ir = b.const_i32(wg_N)
        _c_wgK_ir = b.const_i32(wg_K)
    else:
        dY_desc = make_dy_descriptor(p, dtype=spec.data.dtype_a)
        # X's channel decode spans the group's cpg slab, selected by the group
        # index (grid-per-group).
        X_desc = make_x_wgrad_descriptor(p_load, dtype=spec.data.dtype_b)
        _c_K_ir = _c_C_ir = _c_wgM_ir = _c_wgN_ir = _c_wgK_ir = None

    dy_buf_rsrc = make_buffer_resource(b, dY, num_bytes=dY_bytes)
    x_buf_rsrc = make_buffer_resource(b, X, num_bytes=X_bytes)
    dw_buf_rsrc = make_buffer_resource(b, dW, num_bytes=dW_bytes)
    dy_rsrc = dy_buf_rsrc.rsrc
    x_rsrc = x_buf_rsrc.rsrc
    dw_rsrc = dw_buf_rsrc.rsrc

    k_off_capture: List[Optional[Value]] = [None]

    def dy_descriptor(b_: IRBuilder, row: Value, col: Value):
        k_out = b_.add(block_m_off_v, row)
        if grouped:
            # dY is NHWK with the full K; group g owns the output-channel slab
            # [g*kpg, (g+1)*kpg).  Fold the group base into the (stride-1) k_out
            # coord — no descriptor change needed.
            k_out = b_.add(k_out, b_.mul(group_v, c_kpg))
        k_wg_red = b_.add(k_off_capture[0], col)
        if p.is_pointwise:
            # Flat: offset = k_wg_red * K + k_out
            off = b_.add(b_.mul(k_wg_red, _c_K_ir), k_out)
            kred_ok = b_.cmp_lt(k_wg_red, _c_wgK_ir)
            kout_ok = b_.cmp_lt(k_out, _c_wgM_ir)
            return off, b_.land(kred_ok, kout_ok)
        return dY_desc.offset(b_, k_wg=k_wg_red, k_out=k_out)

    def x_descriptor(b_: IRBuilder, row: Value, col: Value):
        k_val = b_.add(block_n_off_v, row)  # N_wg: filter+channel position
        m_val = b_.add(k_off_capture[0], col)  # K_wg: output spatial position
        if p.is_pointwise:
            # Flat: offset = k_wg * C + n_wg (n_wg == c for 1x1)
            off = b_.add(b_.mul(m_val, _c_C_ir), k_val)
            kwg_ok = b_.cmp_lt(m_val, _c_wgK_ir)
            nwg_ok = b_.cmp_lt(k_val, _c_wgN_ir)
            return off, b_.land(kwg_ok, nwg_ok)
        if grouped:
            # X (NHWC, full C): the grouped make_a_descriptor selects the group's
            # channel slab via the ``group`` embed (c = group*cpg + c_in_group).
            return X_desc.offset(b_, m=m_val, k=k_val, group=group_v)
        return X_desc.offset(b_, m=m_val, k=k_val)

    if spec.async_dma:
        a_loader = AsyncTileLoader.from_tile(
            tile_rows=block_m,
            tile_cols=block_k,
            block_size=threads,
            wave_size=spec.wave_size,
            elem_dtype=ir_dtype_a,
        )
        b_loader = AsyncTileLoader.from_tile(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            wave_size=spec.wave_size,
            elem_dtype=ir_dtype_b,
        )
        a_sync_loader = None
        b_sync_loader = None
    else:
        a_loader = None
        b_loader = None
        a_sync_loader = CoalescedTileLoader(
            tile_rows=block_m,
            tile_cols=block_k,
            block_size=threads,
            load_vec=load_vec_a,
            elem_dtype=ir_dtype_a,
            vector_axis=axis_a,
        )
        b_sync_loader = CoalescedTileLoader(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            load_vec=load_vec_b,
            elem_dtype=ir_dtype_b,
            vector_axis=axis_b,
        )

    schedule = SchedulePolicy.for_pipeline(
        "async_dma" if spec.async_dma else spec.pipeline
    )
    schedule.emit_prologue(b)

    def emit_load_phase(k_off: Value, A_dst: Value, B_dst: Value) -> None:
        k_off_capture[0] = k_off

        if spec.async_dma:
            from ...core.ir import CACHE_STREAM

            a_slot = a_loader.bind(b, smem_dst=A_dst, wave_id=warp_id)
            a_slot.issue(
                b,
                tid=tid,
                rsrc=dy_rsrc,
                descriptor=dy_descriptor,
                coherency=CACHE_STREAM,
            )
            b_slot = b_loader.bind(b, smem_dst=B_dst, wave_id=warp_id)
            b_slot.issue(
                b, tid=tid, rsrc=x_rsrc, descriptor=x_descriptor, coherency=CACHE_STREAM
            )
            return

        a_sync_loader.load(
            b, tid=tid, smem_dst=A_dst, descriptor=dy_descriptor, rsrc=dy_rsrc
        )
        b_sync_loader.load(
            b, tid=tid, smem_dst=B_dst, descriptor=x_descriptor, rsrc=x_rsrc
        )

    def emit_wmma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        a_map = op.a_layout()
        b_map = op.b_layout()
        a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
        b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        new_accs: List[Value] = list(iter_vars)
        for kk in range(k_atoms):
            k_tile_base = b.const_i32(kk * spec.warp_tile_k)
            a_rows = []
            for mi in range(mfmas_m):
                atom_row = b.add(warp_m_off, b.const_i32(mi * spec.warp_tile_m))
                a_rows.append(
                    _emit_frag_smem_load(
                        b,
                        A_src,
                        a_row_in_atom,
                        a_k_in_atom,
                        atom_row,
                        k_tile_base,
                        a_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                atom_row = b.add(warp_n_off, b.const_i32(ni * spec.warp_tile_n))
                b_cols.append(
                    _emit_frag_smem_load(
                        b,
                        B_src,
                        b_col_in_atom,
                        b_k_in_atom,
                        atom_row,
                        k_tile_base,
                        b_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                )
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], new_accs[flat])
                    flat += 1
        return new_accs

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        if op.family == "wmma":
            return emit_wmma_phase(A_src, B_src, iter_vars)

        decoded = decode_mfma_lanes(b, atom, lane)
        m_in_atom = decoded.m_in_atom
        n_in_atom = decoded.n_in_atom
        k_blk = decoded.k_blk

        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        new_accs: List[Value] = list(iter_vars)

        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)),
                b.const_i32(kk * spec.warp_tile_k),
            )
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off, b.add(b.const_i32(mi * spec.warp_tile_m), m_in_atom)
                )
                a_rows.append(
                    _emit_smem_load(
                        b, A_src, a_row, col_base, a_per_lane, smem_dtype=_smem_dtype
                    )
                )

            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * spec.warp_tile_n), n_in_atom)
                )
                b_cols.append(
                    _emit_smem_load(
                        b, B_src, b_row, col_base, b_per_lane, smem_dtype=_smem_dtype
                    )
                )

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    acc = _emit_mfma(b, atom, a_rows[mi], b_cols[ni], new_accs[flat])
                    new_accs[flat] = acc
                    flat += 1

            schedule.emit_after_mfma_step(
                b,
                ds_read_count=mfmas_m + mfmas_n,
                mfma_count=mfmas_m * mfmas_n,
            )

        return new_accs

    # ---- K loop ----
    # k_lo / k_hi select the slice this CTA processes:
    #   split_k == 1: k_lo=0, k_hi=None → full [0, wg_K)
    #   split_k >  1: k_lo=z*ks, k_hi=k_lo+ks (SGPR-pinned, scalar arith)
    _k_upper = c_wg_K if k_hi is None else k_hi

    if spec.unroll_k:
        slice_k = wg_K if k_hi is None else (spec.wg_K_padded() // spec.split_k)
        K_iters = (slice_k + block_k - 1) // block_k
        current_accs = [v for _, v in accs]
        bufs = [(A_smem, B_smem), (A_smem2, B_smem2)]

        emit_load_phase(k_lo, bufs[0][0], bufs[0][1])
        b.sync()

        for it in range(K_iters):
            cur = bufs[it % 2]
            if it + 1 < K_iters:
                nxt = bufs[(it + 1) % 2]
                emit_load_phase(
                    b.add(k_lo, b.const_i32((it + 1) * block_k)), nxt[0], nxt[1]
                )
            k_off_capture[0] = b.add(k_lo, b.const_i32(it * block_k))
            current_accs = emit_mfma_phase(cur[0], cur[1], current_accs)
            b.sync()

        final_accs = current_accs
    elif not spec.async_dma:
        for_op = b.scf_for_iter(k_lo, _k_upper, c_block_k, accs, iv_name="k0")
        with for_op as (k0, iter_vars):
            emit_load_phase(k0, A_smem, B_smem)
            b.sync()
            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)
            b.sync()
            b.scf_yield(*new_accs)
        final_accs = for_op.results
    else:
        slice_k = wg_K if k_hi is None else (spec.wg_K_padded() // spec.split_k)
        K_iters = (slice_k + block_k - 1) // block_k
        bufs = [(A_smem, B_smem), (A_smem2, B_smem2)]

        pipeline = SoftwarePipeline(
            num_iters=K_iters,
            double_buffer=double_buffer,
            wait_vmcnt=True,
            sync_after_wait=True,
            sync_before_issue=True,
            overlap_vmcnt=True,
        )

        def issue_load(it: int, buf_pair):
            emit_load_phase(
                b.add(k_lo, b.const_i32(it * block_k)), buf_pair[0], buf_pair[1]
            )

        def compute(_, buf_pair, state):
            return emit_mfma_phase(buf_pair[0], buf_pair[1], state)

        final_accs = pipeline.run_ping_pong(
            b,
            buffers=bufs,
            initial_state=[v for _, v in accs],
            issue_load=issue_load,
            compute=compute,
            schedule=schedule,
        )

    # ---- epilogue ----
    final_accs = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs)

    if _is_split_k and op.family == "wmma":
        # WMMA split-K: atomic-add via the WMMA C-fragment layout (fp32/bf16/fp16).
        _emit_wgrad_split_k_epilogue_wmma(
            b,
            spec,
            op,
            final_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off_v,
            block_n_off_v,
            dW,
            group=group_v,
        )
    elif _is_split_k and spec.epilogue == "cshuffle":
        # MFMA split-K + cshuffle: scatter to LDS then paired pk_atomic from smem.
        # Avoids the zero-fill workaround of the direct split-K path for bf16/fp16.
        _emit_wgrad_split_k_cshuffle_epilogue(
            b,
            spec,
            atom,
            final_accs,
            grid,
            dW,
            group=group_v,
        )
    elif _is_split_k:
        # MFMA split-K: supports fp32, bf16, fp16 via packed atomics.
        _emit_wgrad_split_k_epilogue(
            b,
            spec,
            atom,
            final_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off_v,
            block_n_off_v,
            dW,
            c_per_lane,
            group=group_v,
        )
    elif spec.epilogue == "cshuffle":
        _emit_wgrad_cshuffle_epilogue(b, spec, final_accs, grid, dw_rsrc, group=group_v)
    elif op.family == "wmma":
        _emit_wgrad_direct_epilogue_wmma(
            b,
            spec,
            op,
            final_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off_v,
            block_n_off_v,
            dw_rsrc,
            c0,
            group=group_v,
        )
    else:
        _emit_wgrad_direct_epilogue(b, spec, final_accs, grid, dw_rsrc, group=group_v)

    return b.kernel


# ---------------------------------------------------------------------
# Epilogues
# ---------------------------------------------------------------------


def _emit_wgrad_split_k_epilogue(
    b: IRBuilder,
    spec: WgradConvSpec,
    atom: MfmaAtom,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    dw_ptr: Value,
    c_per_lane: int,
    group: Optional[Value] = None,
) -> None:
    """Atomic-add partial accumulator directly into dW for all split-K slices.

    Each CTA owns one K-slice.  Its MFMA accumulator (f32) holds the partial
    sum over that slice.  We scatter every per-lane slot to its
    ``(k_out * wg_N + n_wg)`` element index in dW and issue an atomic-add
    so all ``split_k`` CTAs converge without a second reduction pass.
    The caller must zero-init dW before launch.

    Dispatch by dtype_d:
      fp32 — scalar ``global_atomic_add`` (atomicrmw fadd f32, gfx940+).
      bf16 — packed ``global_atomic_add_pk_bf16`` (<2 x bfloat>, gfx940+);
              f32 accumulators are truncated to bf16 and paired per even index.
      fp16 — packed ``global_atomic_add_pk_f16`` (<2 x half>, gfx940+);
              f32 accumulators are truncated to fp16 and paired per even index.

    For bf16/fp16, slots are processed in pairs (i, i+1).  Within each pair
    the even slot's element index is used as the base; the odd slot's value
    occupies the high element of the <2 x dtype> vector.  Both slots must
    share the same row (c_m) — for the standard MFMA C-fragment layout
    this is guaranteed because consecutive ``c_per_lane`` indices within
    one ``(mi, ni)`` atom iterate along the N axis (same row, adjacent
    columns) in groups of ``kc_m1`` which is always ≥ 2 for the atoms
    we support.  If ``c_per_lane`` is odd the last element falls back to
    a scalar f32 atomic-add to avoid reading out of bounds.
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding
    from ...helpers.distribution import make_static_tile_distribution

    dtype_d = spec.data.dtype_d
    p = spec.problem
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    _, __, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))

    c_nlane = b.const_i32(kc_nlane)
    n_in_atom = b.mod(lane, c_nlane)
    m_blk = b.div(lane, c_nlane)
    p_lane = [m_blk, n_in_atom]

    # Decode all (row, col) pairs for the c_per_lane accumulator slots.
    rows: List[Value] = []
    cols: List[Value] = []
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        rows.append(x_row)
        cols.append(x_col)

    wg_M_v = b.const_i32(_wg_M(p))
    wg_N_v = b.const_i32(_wg_N(p))

    # Grouped: the accumulator row ``c_m`` is per-group ([0, kpg)); the atomic
    # address must land in the group's absolute dW output-channel slab
    # (k_out = group*kpg + c_m).  Bounds checks stay on the per-group ``c_m``.
    # Ungrouped (group is None) leaves every address byte-identical.
    _c_kpg = b.const_i32(p.kpg) if group is not None else None

    def _row_addr(c_m: Value) -> Value:
        if group is None:
            return c_m
        return b.add(c_m, b.mul(group, _c_kpg))

    def _to_dtype(v_f32: Value) -> Value:
        """Convert f32 accumulator value to dtype_d."""
        if dtype_d == "fp32":
            return v_f32
        if dtype_d == "bf16":
            return b.trunc_f32_to_bf16(v_f32)
        return b.trunc_f32_to_f16(v_f32)

    def _emit_single_packed_atomic(c_m: Value, c_n: Value, val_f32: Value) -> None:
        """OOB-guarded single-element atomic for bf16/fp16.

        The packed intrinsic requires a 32-bit-aligned pair; to update only
        one element we need to know whether c_n is even or odd and pair it
        with a zero in the appropriate slot.  We check parity at build time
        if c_n is a constant, or emit a runtime branch otherwise.
        This emits one ``<2 x dtype>`` atomic per element, with the unused
        slot holding zero — the hardware adds zero to its slot, which is
        the identity and is safe.
        """
        zero = _to_dtype(b.const_f32(0.0))
        val = _to_dtype(val_f32)
        m_ok = b.cmp_lt(c_m, wg_M_v)
        n_ok = b.cmp_lt(c_n, wg_N_v)
        with b.scf_if(b.land(m_ok, n_ok)):
            # Make c_n even: if c_n is odd, use c_n-1 as base and put val in slot 1.
            c_n_is_odd = b.mod(c_n, b.const_i32(2))
            is_odd = b.cmp_ne(c_n_is_odd, b.const_i32(0))
            c_n_even = b.sub(c_n, c_n_is_odd)  # c_n - (c_n % 2)
            c_off_even = b.add(b.mul(_row_addr(c_m), wg_N_v), c_n_even)
            # Slot 0 = even position, slot 1 = odd position.
            v_even = b.select(is_odd, zero, val)
            v_odd = b.select(is_odd, val, zero)
            vec = b.vec_pack([v_even, v_odd], val.type)
            if dtype_d == "bf16":
                b.global_atomic_add_pk_bf16(dw_ptr, c_off_even, vec)
            else:
                b.global_atomic_add_pk_f16(dw_ptr, c_off_even, vec)

    def _emit_scalar_atomic(c_m: Value, c_n: Value, val_f32: Value) -> None:
        """OOB-guarded scalar atomic-add, dispatching on dtype_d."""
        if dtype_d == "fp32":
            c_off = b.add(b.mul(_row_addr(c_m), wg_N_v), c_n)
            with b.scf_if(b.land(b.cmp_lt(c_m, wg_M_v), b.cmp_lt(c_n, wg_N_v))):
                b.global_atomic_add(dw_ptr, c_off, val_f32)
        else:
            _emit_single_packed_atomic(c_m, c_n, val_f32)

    flat = 0
    for mi in range(mfmas_m):
        atom_m_base = b.add(block_warp_m_off, b.const_i32(mi * spec.warp_tile_m))
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            atom_n_base = b.add(block_warp_n_off, b.const_i32(ni * spec.warp_tile_n))

            if dtype_d == "fp32":
                # rows[i] / cols[i] are the per-slot (row, col) within the atom,
                # decoded directly from the MFMA C-fragment distribution.
                for i in range(c_per_lane):
                    c_m = b.add(atom_m_base, rows[i])
                    c_n = b.add(atom_n_base, cols[i])
                    _emit_scalar_atomic(c_m, c_n, b.vec_extract(acc, i))
            else:
                # bf16 / fp16: one _emit_single_packed_atomic per acc slot.
                # A packed <2 x dtype> atomic requires the two elements to be at
                # the same row (same c_m) AND adjacent N-columns.  For MFMA atoms
                # the C-fragment layout assigns one *row* position per slot
                # (rows[i] = m_blk * kc_m1 + i%kc_m1), so consecutive slots i and
                # i+1 always have rows[i] != rows[i+1].  Attempting to pair them
                # under the same c_m silently writes both values to the wrong row
                # (the row of slot i only).  Use one atomic per slot instead.
                for i in range(c_per_lane):
                    c_m_i = b.add(atom_m_base, rows[i])
                    c_n_i = b.add(atom_n_base, cols[i])
                    _emit_single_packed_atomic(c_m_i, c_n_i, b.vec_extract(acc, i))


def _emit_wgrad_split_k_epilogue_wmma(
    b: IRBuilder,
    spec: WgradConvSpec,
    op,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    dw_ptr: Value,
    group: Optional[Value] = None,
) -> None:
    """Atomic-add partial WMMA accumulator directly into dW for split-K.

    Mirrors :func:`_emit_wgrad_split_k_epilogue` but uses the WMMA C-fragment
    layout (``op.c_layout()``) instead of the MFMA distribution decoder.

    Dispatch by dtype_d:
      fp32 — scalar ``global_atomic_add`` (atomicrmw fadd f32).
      bf16 — packed ``global_atomic_add_pk_bf16`` (<2 x bfloat>); requires C%2==0.
      fp16 — packed ``global_atomic_add_pk_f16`` (<2 x half>); requires C%2==0.

    For bf16/fp16 each slot is issued as a single packed atomic with zero in the
    unused lane (same approach as the MFMA split-K epilogue).  The parity of c_n
    is checked at runtime to place the value in the correct half of the pair.
    ``is_valid_wgrad_spec`` enforces C%2==0 so the 32-bit-aligned base address
    assumption always holds within a filter position.

    ``group`` (grouped wgrad only): the accumulator row ``c_m`` is per-group
    ([0, kpg)); fold ``group*kpg`` into the atomic address so each group writes
    to its own output-channel slab in dW.  Ungrouped (group is None) leaves
    every address byte-identical to the pre-grouped path.

    The caller must zero-initialise dW before launch.
    """
    p = spec.problem
    dtype_d = spec.data.dtype_d
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    wg_M_v = b.const_i32(_wg_M(p))
    wg_N_v = b.const_i32(_wg_N(p))

    _c_kpg = b.const_i32(p.kpg) if group is not None else None

    def _row_addr(c_m: Value) -> Value:
        if group is None:
            return c_m
        return b.add(c_m, b.mul(group, _c_kpg))

    c_map = op.c_layout()

    def _to_dtype(v_f32: Value) -> Value:
        if dtype_d == "fp32":
            return v_f32
        if dtype_d == "bf16":
            return b.trunc_f32_to_bf16(v_f32)
        return b.trunc_f32_to_f16(v_f32)

    def _emit_single_packed_atomic(c_m: Value, c_n: Value, val_f32: Value) -> None:
        zero = _to_dtype(b.const_f32(0.0))
        val = _to_dtype(val_f32)
        m_ok = b.cmp_lt(c_m, wg_M_v)
        n_ok = b.cmp_lt(c_n, wg_N_v)
        with b.scf_if(b.land(m_ok, n_ok)):
            c_n_is_odd = b.mod(c_n, b.const_i32(2))
            is_odd = b.cmp_ne(c_n_is_odd, b.const_i32(0))
            c_n_even = b.sub(c_n, c_n_is_odd)
            c_off_even = b.add(b.mul(_row_addr(c_m), wg_N_v), c_n_even)
            v_even = b.select(is_odd, zero, val)
            v_odd = b.select(is_odd, val, zero)
            vec = b.vec_pack([v_even, v_odd], val.type)
            if dtype_d == "bf16":
                b.global_atomic_add_pk_bf16(dw_ptr, c_off_even, vec)
            else:
                b.global_atomic_add_pk_f16(dw_ptr, c_off_even, vec)

    flat = 0
    for mi in range(mfmas_m):
        atom_m_base = b.add(block_warp_m_off, b.const_i32(mi * spec.warp_tile_m))
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            atom_n_base = b.add(block_warp_n_off, b.const_i32(ni * spec.warp_tile_n))
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, lane, i)
                c_m = b.add(atom_m_base, row_off)
                c_n = b.add(atom_n_base, col_off)
                if dtype_d == "fp32":
                    m_ok = b.cmp_lt(c_m, wg_M_v)
                    n_ok = b.cmp_lt(c_n, wg_N_v)
                    c_off = b.add(b.mul(_row_addr(c_m), wg_N_v), c_n)
                    with b.scf_if(b.land(m_ok, n_ok)):
                        b.global_atomic_add(dw_ptr, c_off, b.vec_extract(acc, i))
                else:
                    _emit_single_packed_atomic(c_m, c_n, b.vec_extract(acc, i))


def _emit_wgrad_split_k_cshuffle_epilogue(
    b: IRBuilder,
    spec: WgradConvSpec,
    atom: MfmaAtom,
    accs: Sequence[Value],
    grid,
    dW: Value,
    group: Optional[Value] = None,
) -> None:
    """Split-K epilogue via cshuffle + packed atomic-adds into dW.

    After the GEMM loop each CTA holds a partial dW tile.  This epilogue:
      1. Scatters the MFMA accumulator to an LDS staging buffer in row-major
         order (identical to the non-atomic cshuffle epilogue).
      2. Issues a barrier.
      3. Reads back ``sv``-wide chunks per thread and issues atomic-adds:
           fp32 — scalar ``global_atomic_add`` per element.
           bf16 — ``global_atomic_add_pk_bf16`` (<2 x bfloat>) per pair.
           fp16 — ``global_atomic_add_pk_f16`` (<2 x half>) per pair.

    For bf16/fp16: adjacent elements in the sv-wide chunk share the same row
    and consecutive N-positions after the shuffle, so they form a genuine
    <2 x dtype> pair — no zero-fill required (contrast with the direct
    split-K epilogue's ``_emit_single_packed_atomic``).

    ``group`` (grouped wgrad only): fold ``group*kpg`` into the atomic address
    so each group writes to its own output-channel slab in dW.
    """
    p = spec.problem
    dtype_d = spec.data.dtype_d

    _cshuffle_kwargs: dict = {"out_dtype": dtype_d}
    if spec.vector_size_c is not None:
        _cshuffle_kwargs["max_store_vec"] = spec.vector_size_c
    else:
        _vc_C = p.cpg if group is not None else p.C
        _vc_K = p.kpg if group is not None else p.K
        # Pass split_k=1 so default_vector_sizes does not force vec_c=1; the
        # cshuffle atomic path reads from smem in sv-wide chunks and issues paired
        # pk_atomics, so a wide store_vec is beneficial (not contraindicated).
        _, __, vec_c = WgradConvSpec.default_vector_sizes(
            _vc_C, _vc_K, dtype_d, split_k=1
        )
        _cshuffle_kwargs["max_store_vec"] = vec_c

    wg_M_v = b.const_i32(_wg_M(p))
    wg_N_v = b.const_i32(_wg_N(p))

    # Grouped: shift block_m_off by group*kpg so the atomic addresses land in
    # the group's absolute output-channel slab.  N axis is unaffected.
    if group is not None:
        c_kpg = b.const_i32(p.kpg)
        eff_grid = dc_replace(
            grid, block_m_off=b.add(grid.block_m_off, b.mul(group, c_kpg))
        )
    else:
        eff_grid = grid

    CShuffleEpilogue.from_grid(
        atom=atom, grid=eff_grid, **_cshuffle_kwargs
    ).atomic_store(
        b,
        accs=accs,
        dw_ptr=dW,
        wg_N=wg_N_v,
        bounds=(wg_M_v, wg_N_v),
    )


def _emit_wgrad_direct_epilogue(
    b: IRBuilder,
    spec: WgradConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    dw_rsrc: Value,
    group: Optional[Value] = None,
) -> None:
    """Per-lane scalar store to dW via the weight-gradient descriptor.

    Delegates to :class:`rocke.helpers.epilogues.DirectEpilogue`.
    The address function maps ``(m_val=k_out, n_val=n_wg)`` to the KYXC
    linear byte offset via :func:`make_dw_descriptor`.

    ``group`` (grouped wgrad only): the M-tile row ``m_val`` is per-group
    (``[0, kpg)``); fold ``group*kpg`` into ``k_out`` for the group's output-
    channel slab and pass ``group`` for the channel embed ``c = group*cpg + …``.
    """
    p = spec.problem
    if p.is_pointwise:
        _c_N = b.const_i32(_wg_N(p))

        def dw_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            return b_.add(b_.mul(m_val, _c_N), n_val), b.const_i32(1)

    elif group is not None:
        # Grouped: dW packed [K,Y,X,cpg]; the group rides on the global k_out
        # only (m_val is the per-group [0,kpg) row).
        dW_desc = make_dw_descriptor(p, dtype=spec.data.dtype_d)
        _c_kpg = b.const_i32(p.kpg)

        def dw_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            m_g = b_.add(m_val, b_.mul(group, _c_kpg))
            return dW_desc.offset(b_, k_out=m_g, n_wg=n_val)

    else:
        dW_desc = make_dw_descriptor(p, dtype=spec.data.dtype_d)

        def dw_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            return dW_desc.offset(b_, k_out=m_val, n_wg=n_val)

    DirectEpilogue(atom=spec.atom, grid=grid, out_dtype=spec.data.dtype_d).store(
        b,
        accs=accs,
        addr_fn=dw_addr,
        d_rsrc=dw_rsrc,
        bounds=(b.const_i32(_wg_M(p)), b.const_i32(_wg_N(p))),
    )


def _emit_wgrad_direct_epilogue_wmma(
    b: IRBuilder,
    spec: WgradConvSpec,
    op,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    dw_rsrc: Value,
    c0: Value,
    group: Optional[Value] = None,
) -> None:
    """Per-lane store for the WMMA (gfx1151/gfx1250) accumulator layout into dW.

    ``group`` (grouped wgrad only): fold ``group*kpg`` into the per-group M row
    and pass ``group`` for the dW channel-slab embed.
    """
    p = spec.problem
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    c_M = b.const_i32(_wg_M(p))
    c_N = b.const_i32(_wg_N(p))
    _c_wgN_wmma = b.const_i32(_wg_N(p)) if p.is_pointwise else None
    _c_kpg = b.const_i32(p.kpg) if group is not None else None
    dW_desc = None if p.is_pointwise else make_dw_descriptor(p, dtype=spec.data.dtype_d)
    c_map = op.c_layout()
    _fp32_out = spec.data.dtype_d == "fp32"
    _bf16_out = spec.data.dtype_d == "bf16"
    _elem_bytes = 4 if _fp32_out else 2

    flat = 0
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            atom_m_off = b.add(
                b.add(block_m_off, warp_m_off),
                b.const_i32(mi * spec.warp_tile_m),
            )
            atom_n_off = b.add(
                b.add(block_n_off, warp_n_off),
                b.const_i32(ni * spec.warp_tile_n),
            )
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, lane, i)
                m_val = b.add(atom_m_off, row_off)
                n_val = b.add(atom_n_off, col_off)
                m_ok = b.cmp_lt(m_val, c_M)
                n_ok = b.cmp_lt(n_val, c_N)
                ok = b.land(m_ok, n_ok)

                v_f32 = b.vec_extract(acc, i)
                if p.is_pointwise:
                    dw_off_elems = b.add(b.mul(m_val, _c_wgN_wmma), n_val)
                elif group is not None:
                    m_g = b.add(m_val, b.mul(group, _c_kpg))
                    dw_off_elems, _ = dW_desc.offset(b, k_out=m_g, n_wg=n_val)
                else:
                    dw_off_elems, _ = dW_desc.offset(b, k_out=m_val, n_wg=n_val)
                dw_off_bytes = b.mul(dw_off_elems, b.const_i32(_elem_bytes))
                safe_off = b.select(ok, dw_off_bytes, b.const_i32((1 << 31) - 1))
                if _fp32_out:
                    b.buffer_store_f32(dw_rsrc, safe_off, c0, v_f32)
                elif _bf16_out:
                    b.buffer_store_bf16(
                        dw_rsrc, safe_off, c0, b.trunc_f32_to_bf16(v_f32)
                    )
                else:
                    b.buffer_store_f16(dw_rsrc, safe_off, c0, b.trunc_f32_to_f16(v_f32))


def _emit_wgrad_cshuffle_epilogue(
    b: IRBuilder,
    spec: WgradConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    dw_rsrc: Value,
    group: Optional[Value] = None,
) -> None:
    """LDS-staged cshuffle epilogue writing to dW (KYXC layout).

    Delegates to :class:`rocke.helpers.epilogues.CShuffleEpilogue`.
    The address function maps ``(m_val=k_out, n_val=n_wg)`` to KYXC offset
    via :func:`make_dw_descriptor`.

    ``group`` (grouped wgrad only): the M row ``m_val`` is per-group ([0, kpg));
    fold ``group*kpg`` into ``k_out`` so the staged store lands in the group's
    dW output-channel slab, and derive the store vector width from cpg/kpg (the
    packed dW inner dim), not the full C/K.
    """
    p = spec.problem
    if p.is_pointwise:
        _c_N = b.const_i32(_wg_N(p))

        def dw_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            return b_.add(b_.mul(m_val, _c_N), n_val), b.const_i32(1)

    elif group is not None:
        dW_desc = make_dw_descriptor(p, dtype=spec.data.dtype_d)
        _c_kpg = b.const_i32(p.kpg)

        def dw_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            m_g = b_.add(m_val, b_.mul(group, _c_kpg))
            return dW_desc.offset(b_, k_out=m_g, n_wg=n_val)

    else:
        dW_desc = make_dw_descriptor(p, dtype=spec.data.dtype_d)

        def dw_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            return dW_desc.offset(b_, k_out=m_val, n_wg=n_val)

    _cshuffle_kwargs: dict = {"out_dtype": spec.data.dtype_d}
    if spec.vector_size_c is not None:
        _cshuffle_kwargs["max_store_vec"] = spec.vector_size_c
    else:
        # Grouped: the packed dW inner (store) dim is cpg, so bound the store
        # vector width by cpg/kpg rather than the full C/K.
        _vc_C = p.cpg if group is not None else spec.problem.C
        _vc_K = p.kpg if group is not None else spec.problem.K
        _, __, vec_c = WgradConvSpec.default_vector_sizes(
            _vc_C, _vc_K, spec.data.dtype_d, split_k=spec.split_k
        )
        _cshuffle_kwargs["max_store_vec"] = vec_c
    CShuffleEpilogue.from_grid(atom=spec.atom, grid=grid, **_cshuffle_kwargs).store(
        b,
        accs=accs,
        addr_fn=dw_addr,
        d_rsrc=dw_rsrc,
        bounds=(b.const_i32(_wg_M(p)), b.const_i32(_wg_N(p))),
    )
