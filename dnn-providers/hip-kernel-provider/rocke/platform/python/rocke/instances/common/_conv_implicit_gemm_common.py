# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared dataclasses, descriptors, and MFMA helpers for implicit-GEMM conv.

Internal module (underscore-prefixed).  External callers should import from
:mod:`rocke.instances.common.conv_implicit_gemm` (forward) or
:mod:`rocke.instances.common.conv_implicit_gemm_wgrad` (backward-weight) —
both re-export everything they consume from here.

What lives here
---------------
* :class:`ConvDataSpec`, :class:`ConvProblem`, :class:`ConvAccumulatorEpilogue`
  — problem / dtype / epilogue dataclasses shared by both directions.
* :func:`make_a_descriptor` — coordinate-transform DAG for the NHWC input
  tensor; identical for forward and wgrad (both read the same activation
  tensor).
* ``_ir_dtype``, ``_DTYPE_TO_IR`` — dtype helpers.
* MFMA-body helpers: ``_emit_mfma``, ``_emit_smem_load``,
  ``_emit_frag_smem_load``, ``_apply_accumulator_epilogue``,
  ``_choose_load_vec_for`` — the LDS-load + MFMA emission plumbing reused by
  both the forward and wgrad K-loop bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ...core.ir import (
    BF16,
    F16,
    F32,
    IRBuilder,
    Type,
    Value,
)
from ...helpers.spec import choose_load_vec
from ...helpers.transforms import TensorDescriptor, embed, pad, unmerge_magic


# ---------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------

_DTYPE_TO_IR: dict = {"f16": F16, "fp16": F16, "bf16": BF16, "fp32": F32}


def _ir_dtype(dtype: str) -> Type:
    """Map a dtype string (``"fp16"``, ``"bf16"``, ``"fp32"``) to an IR ``Type``."""
    t = _DTYPE_TO_IR.get(dtype)
    if t is None:
        raise ValueError(
            f"unsupported conv dtype {dtype!r}; choose fp16, bf16, or fp32"
        )
    return t


# ---------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ConvDataSpec:
    """Element / accumulator dtype choice for the conv kernel.

    Layouts:
      A: NHWC, dtype_a (input activations)
      B: KYXC, dtype_b (weights)
      D: NHWK, dtype_d (output)
      accumulator: dtype_acc (always fp32)
    """

    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_d: str = "fp16"
    dtype_acc: str = "fp32"


@dataclass(frozen=True)
class ConvProblem:
    """2-D or 3-D convolution shape parameters.

    2-D (default — ``Di`` / ``Z`` are ``None``):
      A: NHWC,  shape ``[N, Hi, Wi, C]``
      B: KYXC, shape ``[K, Y, X, C]``
      D: NHWK,  shape ``[N, Ho, Wo, K]``
      M = N*Ho*Wo,  N_gemm = K,  K_gemm = Y*X*C

     3-D (set ``Di`` and ``Z``):
       A: NDHWC, shape ``[N, Di, Hi, Wi, C]``
       B: KZYXC, shape ``[K, Z, Y, X, C]``
       D: NDHWK, shape ``[N, Do, Ho, Wo, K]``
       M = N*Do*Ho*Wo,  N_gemm = K/groups,  K_gemm = Z*Y*X*(C/groups)

    ``C`` and ``K`` are always the *total* channel counts across all groups.
    Use ``cpg`` / ``kpg`` for per-group counts.
    """

    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    Y: int
    X: int
    sH: int = 1
    sW: int = 1
    pH: int = 0
    pW: int = 0
    dH: int = 1
    dW: int = 1
    groups: int = 1
    # 3-D-only fields; leave as None for 2-D convolutions.
    Di: Optional[int] = None
    Z: Optional[int] = None
    sD: Optional[int] = None
    pD: Optional[int] = None
    dD: Optional[int] = None

    def __post_init__(self) -> None:
        depth = (self.Di, self.Z, self.sD, self.pD, self.dD)
        any_set = any(v is not None for v in depth)
        all_set = all(v is not None for v in depth)
        if any_set and not all_set:
            raise ValueError(
                "3-D ConvProblem requires Di, Z, sD, pD, dD (set all or leave all as None)"
            )
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {self.groups}")
        if self.C % self.groups != 0:
            raise ValueError(f"C={self.C} is not divisible by groups={self.groups}")
        if self.K % self.groups != 0:
            raise ValueError(f"K={self.K} is not divisible by groups={self.groups}")

    @property
    def is_3d(self) -> bool:
        return self.Di is not None

    # ---- depth spatial output (3-D only) ----
    @property
    def Do(self) -> Optional[int]:
        if not self.is_3d:
            return None
        return (self.Di + 2 * self.pD - self.dD * (self.Z - 1) - 1) // self.sD + 1

    @property
    def Ho(self) -> int:
        return (self.Hi + 2 * self.pH - self.dH * (self.Y - 1) - 1) // self.sH + 1

    @property
    def Wo(self) -> int:
        return (self.Wi + 2 * self.pW - self.dW * (self.X - 1) - 1) // self.sW + 1

    @property
    def M(self) -> int:
        base = self.N * self.Ho * self.Wo
        return base * self.Do if self.is_3d else base

    @property
    def cpg(self) -> int:
        """Input channels per group (C / groups)."""
        return self.C // self.groups

    @property
    def kpg(self) -> int:
        """Output channels per group (K / groups)."""
        return self.K // self.groups

    @property
    def N_gemm(self) -> int:
        return self.kpg

    @property
    def K_gemm(self) -> int:
        z = self.Z if self.is_3d else 1
        return z * self.Y * self.X * self.cpg

    @property
    def is_pointwise(self) -> bool:
        """True when the conv reduces to an explicit GEMM (1x1 kernel, stride 1, no pad)."""
        hw = (
            self.Y == 1
            and self.X == 1
            and self.sH == 1
            and self.sW == 1
            and self.pH == 0
            and self.pW == 0
        )
        if not self.is_3d:
            return hw
        return hw and self.Z == 1 and self.sD == 1 and self.pD == 0

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N_gemm * self.K_gemm * self.groups

    def short(self) -> str:
        g = f"G{self.groups}" if self.groups > 1 else ""
        if self.is_3d:
            return (
                f"N{self.N}D{self.Di}H{self.Hi}W{self.Wi}C{self.C}"
                f"_K{self.K}Z{self.Z}Y{self.Y}X{self.X}{g}"
            )
        return f"N{self.N}H{self.Hi}W{self.Wi}C{self.C}_K{self.K}Y{self.Y}X{self.X}{g}"


@dataclass(frozen=True)
class ConvAccumulatorEpilogue:
    """Static fp32 accumulator transform applied before the conv store.

    This is intentionally narrower than ``helpers.fuse.FusedEpilogue``:
    it runs directly on MFMA accumulator fragments inside the hand-authored
    conv instance. The default is identity, preserving the historical conv IR.
    """

    bias: float = 0.0
    scale: float = 1.0
    relu: bool = False
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    def is_identity(self) -> bool:
        return (
            self.bias == 0.0
            and self.scale == 1.0
            and not self.relu
            and self.clamp_min is None
            and self.clamp_max is None
        )

    def tag(self) -> str:
        if self.is_identity():
            return ""
        pieces: List[str] = []
        if self.bias != 0.0:
            pieces.append(f"bias{self.bias:g}")
        if self.scale != 1.0:
            pieces.append(f"scale{self.scale:g}")
        if self.relu:
            pieces.append("relu")
        if self.clamp_min is not None or self.clamp_max is not None:
            lo = "-inf" if self.clamp_min is None else f"{self.clamp_min:g}"
            hi = "inf" if self.clamp_max is None else f"{self.clamp_max:g}"
            pieces.append(f"clamp{lo}to{hi}")
        return "epi_" + "_".join(pieces)


# ---------------------------------------------------------------------
# Descriptor builders
# ---------------------------------------------------------------------


def _a_channel_decode(p: ConvProblem, *, is_3d: bool) -> List:
    """Channel-decode transforms for A's ``k_gemm`` unmerge.

    For ``groups == 1`` this is the historical single ``unmerge`` of the
    contraction index ``k`` into the spatial-filter coords plus the full input
    channel ``c`` (``dims=[.., Y, X, C]``) — byte-identical to the pre-groups
    kernel.

    For ``groups > 1`` (grouped / cardinality-grouped conv) the contraction
    index only spans the *per-group* channels ``cpg = C / groups``, so ``k``
    unmerges into ``c_in_group`` (``dims=[.., Y, X, cpg]``) and an affine
    ``embed`` recovers the absolute NHWC channel
    ``c = group * cpg + c_in_group`` from the extra ``group`` upper coord
    threaded in by the caller's block-z index. This mirrors CK Tile's
    grouped-conv per-group ``C_`` slab: the descriptor stays NHWC
    (``C = total``) and the group selects the ``[g*cpg, (g+1)*cpg)`` slab.
    ``group`` is supplied at ``A_desc.offset(..., group=g)`` for the grouped
    path only, so ``upper_names`` gains ``group`` exactly when ``groups > 1``.

    ``embed`` (not ``merge``) is used deliberately: it is one of the transform
    kinds already mirrored in the C++ engine, so the grouped descriptor lowers
    byte-identically in both engines without adding a new C++ transform kind.
    The ``0 <= c < C`` bound it emits is always true here (``group < groups`` and
    ``c_in_group < cpg``) and costs only a redundant compare on the grouped path.
    """
    if p.groups > 1:
        if is_3d:
            unmerge_into = ["z", "y", "x", "c_in_group"]
            unmerge_dims = [p.Z, p.Y, p.X, p.cpg]
        else:
            unmerge_into = ["y", "x", "c_in_group"]
            unmerge_dims = [p.Y, p.X, p.cpg]
        return [
            unmerge_magic("k", into=unmerge_into, dims=unmerge_dims),
            embed(
                upper=["group", "c_in_group"],
                into="c",
                strides=[p.cpg, 1],
                offset=0,
                lo=0,
                hi=p.C,
            ),
        ]
    if is_3d:
        return [
            unmerge_magic("k", into=["z", "y", "x", "c"], dims=[p.Z, p.Y, p.X, p.C])
        ]
    return [unmerge_magic("k", into=["y", "x", "c"], dims=[p.Y, p.X, p.C])]


def make_a_descriptor(
    p: ConvProblem, decompose_m: bool = True, dtype: str = "fp16"
) -> TensorDescriptor:
    """Build the (m, k) -> N[D]HWC linear-offset descriptor for the input.

    2-D DAG  (``p.is_3d`` is False):
      naive(NHWC):                       (n, hi, wi, c)
      + unmerge('m' -> n, ho, wo):       (hi, wi, c, m, ho, wo) intermediate
      + embed((ho, y) -> hi):            (wi, c, m, y, wo)      intermediate
      + embed((wo, x) -> wi):            (c, m, y, x)           intermediate
      + unmerge('k' -> y, x, c):         (m, k)                 user-facing
      + pad('y' lo=0 hi=Y):              boundary check
      + pad('x' lo=0 hi=X):              boundary check

    3-D DAG  (``p.is_3d`` is True):
      naive(NDHWC):                      (n, di, hi, wi, c)
      + unmerge('m' -> n, do, ho, wo)
      + embed((do, z) -> di)
      + embed((ho, y) -> hi)
      + embed((wo, x) -> wi)
      + unmerge('k' -> z, y, x, c)
      + pad('z'), pad('y'), pad('x')

    When ``decompose_m`` is ``False`` the leading ``unmerge('m' -> ...)``
    is dropped and the user-facing upper coords become ``(n, ho, wo, k)``
    directly. This is a strict win for callers that already hold ``(ho, wo)``
    cheaply (e.g. computed via shift/mask from the tile row): the default
    chain would re-decompose ``m = ho*Wo + wo`` back into ``(n, ho, wo)`` via
    two magic divisions (~10 VALU per A coord) — a pure round-trip. Feeding
    ``(n, ho, wo)`` straight in produces a bit-identical offset while skipping
    both the caller-side flatten and the descriptor-side magic unmerge.

    The ``embed`` transforms encode the convolution affine maps
    ``hi = ho*sH - pH + y*dH`` and ``wi = wo*sW - pW + x*dW``, with the
    convolution boundary check baked into the descriptor's validity predicate.
    The ``pad`` transforms add per-coord bound checks on ``y`` and ``x``: when
    ``K_gemm`` is not divisible by the block ``tile_k``, the K-loop loads past
    ``K_gemm-1`` and the unmerge produces ``y >= Y`` or ``x >= X``. Without
    these ``pad`` transforms the kernel would read valid-looking offsets that
    *cross* into adjacent weight rows and blend wrong weights into the
    accumulator.
    """
    transforms = []
    if p.is_3d:
        if decompose_m:
            transforms.append(
                unmerge_magic(
                    "m", into=["n", "do", "ho", "wo"], dims=[p.N, p.Do, p.Ho, p.Wo]
                )
            )
        transforms += [
            embed(
                upper=["do", "z"],
                into="di",
                strides=[p.sD, p.dD],
                offset=-p.pD,
                lo=0,
                hi=p.Di,
            ),
            embed(
                upper=["ho", "y"],
                into="hi",
                strides=[p.sH, p.dH],
                offset=-p.pH,
                lo=0,
                hi=p.Hi,
            ),
            embed(
                upper=["wo", "x"],
                into="wi",
                strides=[p.sW, p.dW],
                offset=-p.pW,
                lo=0,
                hi=p.Wi,
            ),
            *_a_channel_decode(p, is_3d=True),
            pad("z", lo=0, hi=p.Z),
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        ]
        return TensorDescriptor.naive(
            "A_ndhwc",
            lengths=[p.N, p.Di, p.Hi, p.Wi, p.C],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "di", "hi", "wi", "c"],
        ).transform(*transforms)
    else:
        if decompose_m:
            transforms.append(
                unmerge_magic(upper="m", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo])
            )
        transforms += [
            embed(
                upper=["ho", "y"],
                into="hi",
                strides=[p.sH, p.dH],
                offset=-p.pH,
                lo=0,
                hi=p.Hi,
            ),
            embed(
                upper=["wo", "x"],
                into="wi",
                strides=[p.sW, p.dW],
                offset=-p.pW,
                lo=0,
                hi=p.Wi,
            ),
            *_a_channel_decode(p, is_3d=False),
            # pad('y'/'x'): guard against partial K-tile overruns into adjacent weight rows.
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        ]
        return TensorDescriptor.naive(
            "A_nhwc",
            lengths=[p.N, p.Hi, p.Wi, p.C],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "hi", "wi", "c"],
        ).transform(*transforms)


# ---------------------------------------------------------------------
# MFMA body helpers (shared by forward and wgrad K-loop bodies)
# ---------------------------------------------------------------------


def _emit_mfma(b: IRBuilder, atom, a: Value, bv: Value, c: Value) -> Value:
    return atom.emit(b, a, bv, c)


def _emit_smem_load(
    b: IRBuilder,
    smem: Value,
    row: Value,
    col: Value,
    n: int,
    *,
    smem_dtype: Optional[Type] = None,
) -> Value:
    if smem_dtype is not None and smem_dtype is not F16:
        vec = b.smem_load_vN(smem, row, col, dtype=smem_dtype, n=n)
        return b.vec_extract(vec, 0) if (smem_dtype is F32 and n == 1) else vec
    if n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN_f16(smem, row, col, n=n)


def _apply_accumulator_epilogue(
    b: IRBuilder,
    epilogue: ConvAccumulatorEpilogue,
    accs: Sequence[Value],
) -> List[Value]:
    """Apply a static fp32 epilogue to each accumulator fragment.

    The transform is scalar per accumulator lane, then packed back into the
    original vector width so the existing direct/cshuffle epilogues can consume
    the result unchanged.
    """
    if epilogue.is_identity():
        return list(accs)

    out: List[Value] = []
    c_zero = b.const_f32(0.0)
    c_bias = b.const_f32(epilogue.bias) if epilogue.bias != 0.0 else None
    c_scale = b.const_f32(epilogue.scale) if epilogue.scale != 1.0 else None
    c_clamp_min = (
        b.const_f32(epilogue.clamp_min) if epilogue.clamp_min is not None else None
    )
    c_clamp_max = (
        b.const_f32(epilogue.clamp_max) if epilogue.clamp_max is not None else None
    )

    for acc in accs:
        elems: List[Value] = []
        for i in range(acc.type.count):
            v = b.vec_extract(acc, i)
            if c_bias is not None:
                v = b.fadd(v, c_bias)
            if c_scale is not None:
                v = b.fmul(v, c_scale)
            if epilogue.relu:
                v = b.fmax(v, c_zero)
            if c_clamp_min is not None:
                v = b.fmax(v, c_clamp_min)
            if c_clamp_max is not None:
                v = b.fmin(v, c_clamp_max)
            elems.append(v)
        out.append(b.vec_pack(elems, elems[0].type))
    return out


def _emit_frag_smem_load(
    b: IRBuilder,
    src: Value,
    mn_in_atom: Value,
    k_in_atom: Value,
    atom_mn_base: Value,
    k_tile_base: Value,
    frag_len: int,
    *,
    smem_dtype: Optional[Type] = None,
) -> Value:
    """Load one ``frag_len``-wide operand fragment from a row-major LDS tile.

    Both the A LDS tile ``(block_m, block_k)`` and the B LDS tile
    ``(block_n, block_k)`` are row-major with the M/N index as the row and K as
    the column. One lane's fragment occupies a single tile row
    (``atom_mn_base + mn_in_atom``) and ``frag_len`` contiguous K columns from
    ``k_tile_base + k_in_atom`` — true for both the MFMA and WMMA layout maps,
    whose A/B fragment slots are K-contiguous. fp16 smem loads cap at 8 lanes,
    so a wider fragment (WMMA ``<16 x half>``) is assembled from 8-wide chunks.
    """
    lds_row = b.add(atom_mn_base, mn_in_atom)
    lds_col = b.add(k_tile_base, k_in_atom)
    if frag_len <= 8:
        return _emit_smem_load(
            b, src, lds_row, lds_col, frag_len, smem_dtype=smem_dtype
        )
    frag = None
    for off in range(0, frag_len, 8):
        chunk = _emit_smem_load(
            b, src, lds_row, b.add(lds_col, b.const_i32(off)), 8, smem_dtype=smem_dtype
        )
        frag = chunk if frag is None else b.vec_concat(frag, chunk)
    return frag


def _choose_load_vec_for(
    tile_m: int, tile_n: int, tile_k: int, block_size: int, dtype_a: str
) -> int:
    """Pick the widest load-vector width for the given tile geometry.

    Thin adapter over the shared :func:`rocke.helpers.spec.choose_load_vec`."""
    elem_bytes = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype_a, 2)
    return choose_load_vec(tile_m, tile_n, tile_k, block_size, elem_bytes=elem_bytes)
