# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused MoE forward pipeline (CK Tile ``15_fused_moe`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/15_fused_moe``. The
fused-MoE forward is a six-stage pipeline; the CK Tile reference
executes it as a small handful of kernel launches that share workspace
buffers between them. This module implements the three *MoE-specific*
kernels (the ones with no plain-GEMM analogue) plus a launcher class
that documents how to compose them with the existing GEMM and
quantisation kernels in :mod:`ck_dsl.instances`.

Pipeline overview (token=T, expert=E, topk=K, hidden=H, intermediate=I,
quantised activation dtype=Q, expert-weight dtype=W)::

 (0) router -> TopkIds[T, K], TopkWeights[T, K] (caller-provided)
 (1) moe sort -> Offsets[E], Counts[E], (3 launches, )
 SortedTokenIds[T*K], SortedTopkIds[T*K],
 SortedWeights[T*K]
 (2) gather -> GroupedInput[T*K, H] (build_moe_gather)
 (2b) quant -> QGroupedInput[T*K, H], QScale[T*K] (moe_smoothquant, )
 (3) gate gemm : per-expert (Counts[e], I, H) GEMM (block_scale_gemm, )
 QGroupedInput @ W_gate[e]^T -> GateOut[T*K, I]
 (3b) up gemm : per-expert (Counts[e], I, H) GEMM (block_scale_gemm, )
 QGroupedInput @ W_up[e]^T -> UpOut[T*K, I]
 (4) silu_mul -> Hidden[T*K, I] (build_moe_silu_mul)
 (4b) quant -> QHidden[T*K, I], HScale[T*K] (moe_smoothquant, )
 (5) down gemm : per-expert (Counts[e], H, I) GEMM (block_scale_gemm, )
 QHidden @ W_down[e]^T -> DownOut[T*K, H]
 (6) topk reduce -> Y[T, H] (build_moe_topk_weighted_reduce)

The launcher class :class:`FusedMoeLauncher` packages the workspace
sizes, the per-stage grids, and the orchestration loop. Users plug in
a per-expert GEMM dispatcher (a callable that takes
``(expert_idx, a_ptr, b_ptr, c_ptr, m, n, k)``); on the production
side that's typically a thin wrapper around
:func:`ck_dsl.instances.build_block_scale_gemm` (FP8 path) or
:func:`ck_dsl.instances.build_universal_gemm` (FP16/BF16 path).

What this module *ships in IR*:

* :func:`build_moe_gather` -- gather kernel (the
 indirect ``GroupedInput[b, h] = X[sorted_token_ids[b], h]`` step).
* :func:`build_moe_silu_mul` -- SwiGLU activation fusion
 (``Hidden[b, i] = silu(GateOut[b, i]) * UpOut[b, i]``).
* :func:`build_moe_topk_weighted_reduce` -- the final atomic-accumulate
 pass (``atomic_add(Y[token_id, h], weight * DownOut[b, h])``).

Each kernel has a matching ``*_grid`` / ``*_signature`` helper for
use with :class:`ck_dsl.runtime.launcher.KernelLauncher`.

Limitations of v1 (tracked in the wave plan):

* The per-expert GEMM dispatch is left to the caller. The launcher's
 ``run`` is currently a documentation stub; the production launcher
 (with per-expert grid sizing and workspace reuse) is a v2 follow-on.
* Bias add inside the gate/up/down projections is not exposed --
 that's a flag on the underlying GEMM kernel; pass it through your
 ``expert_gemm_fn`` if needed.
* Dropout / expert-load-balancing telemetry are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.gather_scatter import (
    gather_row_offset,
    load_sorted_token_id,
    load_sorted_topk_weight,
)
from ..helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from ..helpers.spec import (
    SignatureBuilder,
    kernel_name_join,
)


__all__ = [
    "FusedMoeLauncher",
    "FusedMoeSpec",
    "build_moe_gather",
    "build_moe_silu_mul",
    "build_moe_topk_weighted_reduce",
    "is_valid_spec",
    "moe_gather_grid",
    "moe_gather_signature",
    "moe_silu_mul_grid",
    "moe_silu_mul_signature",
    "moe_topk_weighted_reduce_grid",
    "moe_topk_weighted_reduce_signature",
    "moe_fused_workspace_bytes",
]


DType = Literal["f16", "bf16", "fp16"]


@dataclass(frozen=True)
class FusedMoeSpec:
    """One concrete fused-MoE configuration.

    Captures the shapes + dtypes shared across every kernel in the
    pipeline. Compile-time sizes (``tokens``, ``experts``, ``topk``,
    ``hidden``, ``intermediate``) become kernel-name suffixes and let
    the IR builders bake in vector widths; runtime values like
    ``Counts[e]`` per-expert sizes are passed as kernel args.

    Attributes
    ----------
    tokens
    Number of input rows in ``X`` (post-attention residual).
    experts
    Number of experts ``E`` in the MoE layer.
    topk
    Routing depth ``K`` (number of experts per token).
    hidden
    Hidden / embedding dimension ``H``.
    intermediate
    Per-expert MLP inner dimension ``I``. For SwiGLU the down
    projection's K dim is ``I``.
    dtype
    Activation dtype shared across gather/silu_mul/reduce
    (``"f16"`` or ``"bf16"``). The per-expert GEMM may use
    a quantised dtype internally; that's its own concern.
    block_size
    Workgroup size for the streaming kernels.
    vec
    Vector width for global loads/stores (``2``/``4``/``8``).
    Drives the per-thread element count.
    name
    Kernel-name prefix; phase tag is appended per kernel.
    """

    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    dtype: DType = "f16"
    block_size: int = 256
    vec: int = 4
    name: str = "ck_dsl_fused_moe"

    @property
    def total_pairs(self) -> int:
        """``tokens * topk`` -- one bucket per ``(token, k_topk)`` pair."""
        return self.tokens * self.topk

    @property
    def elems_per_thread_hidden(self) -> int:
        return self.hidden // self.block_size

    @property
    def elems_per_thread_inter(self) -> int:
        return self.intermediate // self.block_size

    def kernel_name(self, phase: str) -> str:
        return kernel_name_join(
            self.name,
            phase,
            f"T{self.tokens}",
            f"E{self.experts}",
            f"K{self.topk}",
            f"H{self.hidden}",
            f"I{self.intermediate}",
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
        )


def is_valid_spec(spec: FusedMoeSpec) -> Tuple[bool, str]:
    if spec.tokens <= 0 or spec.experts <= 0 or spec.topk <= 0:
        return False, (
            f"tokens / experts / topk must be > 0 (got {spec.tokens}, "
            f"{spec.experts}, {spec.topk})"
        )
    if spec.hidden <= 0 or spec.intermediate <= 0:
        return False, (
            f"hidden / intermediate must be > 0 "
            f"(got {spec.hidden}, {spec.intermediate})"
        )
    if spec.topk > spec.experts:
        return False, f"topk ({spec.topk}) > experts ({spec.experts})"
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"block_size {spec.block_size} not in {{64..1024}}"
    if spec.vec not in (2, 4, 8):
        return False, f"vec {spec.vec} not in {{2, 4, 8}}"
    if spec.dtype not in ("f16", "fp16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.hidden % spec.vec != 0:
        return False, f"hidden {spec.hidden} not divisible by vec {spec.vec}"
    if spec.intermediate % spec.vec != 0:
        return False, (
            f"intermediate {spec.intermediate} not divisible by vec {spec.vec}"
        )
    if spec.hidden % spec.block_size != 0:
        return False, (
            f"hidden {spec.hidden} not divisible by block_size {spec.block_size}; "
            "v1 requires one CTA per bucket row to cover the full hidden vector"
        )
    if spec.intermediate % spec.block_size != 0:
        return False, (
            f"intermediate {spec.intermediate} not divisible by block_size "
            f"{spec.block_size}; same one-CTA-per-row constraint as hidden"
        )
    return True, "ok"


# ---------------------------------------------------------------------
# Stage (2): gather. ``GroupedInput[b, h] = X[sorted_token_ids[b], h]``.
# ---------------------------------------------------------------------


def build_moe_gather(spec: FusedMoeSpec) -> KernelDef:
    """Pre-sort -> per-bucket gather of input rows.

    For each bucket index ``b in [0, tokens*topk)``:

    token_id = SortedTokenIds[b] # i32 indirect
    for h_col in 0..hidden:
    GroupedInput[b, h_col] = X[token_id, h_col] # native dtype

    Kernel signature::

    (X: ptr<dtype, global>, # (tokens, hidden)
    SortedTokenIds: ptr<i32, global>, # (tokens*topk,) from moe_sort
    GroupedInput: ptr<dtype, global>, # (tokens*topk, hidden)
    tokens: i32, hidden: i32)

    Grid: ``(tokens * topk, 1, 1)``. Each CTA handles one bucket row;
    each thread within the CTA streams ``elems_per_thread`` consecutive
    hidden columns through a vectorised load / store.

    Notes
    -----
    The indirect ``X[token_id]`` is a per-CTA invariant -- once
    ``token_id`` is loaded by lane 0 and broadcast (via
    :func:`load_sorted_token_id` + the implicit SGPR promotion the
    compiler does for wave-uniform integers), every lane derives its
    own per-column address with a scalar add and a runtime mul. The
    code below relies on the compiler to lift the
    ``token_id * hidden`` partial product into scalar registers.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    H = spec.hidden
    BS = spec.block_size
    EPT = spec.elems_per_thread_hidden
    dtype = spec.dtype

    b = IRBuilder(spec.kernel_name("gather"))
    b.kernel.attrs["max_workgroup_size"] = BS

    ty = io_ir_type(dtype)

    X = b.param("X", PtrType(ty, "global"), noalias=True, readonly=True, align=16)
    SortedTokenIds = b.param(
        "SortedTokenIds",
        PtrType(I32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    GroupedInput = b.param(
        "GroupedInput",
        PtrType(ty, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    _tokens = b.param("tokens", I32)  # noqa: F841 - ABI matches CK Tile
    _hidden = b.param("hidden", I32)  # noqa: F841 - ABI matches CK Tile

    bid = b.block_id_x()
    tid = b.thread_id_x()

    token_id = load_sorted_token_id(b, SortedTokenIds, bid)
    # Token ids out of range silently produce a zero gather. Mirrors
    # the CK Tile reference's behaviour for unused bucket slots when
    # the router emits sentinel -1 ids.
    valid_token = b.cmp_ge(token_id, b.const_i32(0))
    bucket_base = b.mul(bid, b.const_i32(H))

    for k in range(EPT):
        h_col = b.add(b.mul(tid, b.const_i32(EPT)), b.const_i32(k))
        in_h = b.cmp_lt(h_col, b.const_i32(H))
        with b.scf_if(b.land(valid_token, in_h)):
            src_off = gather_row_offset(b, SortedTokenIds, bid, hidden=H, col=h_col)
            v = load_scalar_as_f32(b, X, src_off, dtype=dtype)
            dst_off = b.add(bucket_base, h_col)
            store_scalar_from_f32(b, GroupedInput, dst_off, v, dtype=dtype)

    return b.kernel


# ---------------------------------------------------------------------
# Stage (4): silu_mul (SwiGLU activation fusion).
# ---------------------------------------------------------------------


def build_moe_silu_mul(spec: FusedMoeSpec) -> KernelDef:
    """SwiGLU activation fusion across the gate / up MLP output.

    ``Hidden[b, i] = silu(GateOut[b, i]) * UpOut[b, i]``

    where ``silu(x) = x * sigmoid(x)``. Used between the gate / up
    GEMM and the down GEMM in every fused-MoE forward; matches the
    activation that LLaMA-style MoE layers run.

    Kernel signature::

    (GateOut: ptr<dtype, global>, # (tokens*topk, intermediate)
    UpOut: ptr<dtype, global>, # (tokens*topk, intermediate)
    Hidden: ptr<dtype, global>, # (tokens*topk, intermediate)
    total_pairs: i32, intermediate: i32)

    Grid: ``(total_pairs, 1, 1)``; one CTA per bucket row, each thread
    handling ``elems_per_thread_inter`` consecutive intermediate cols.

    Implementation notes:

    * The sigmoid is computed via ``exp2(-x * log2(e))`` (matches the
    formula used by :class:`ck_dsl.helpers.fuse.SiLU` and the
    ``elementwise`` instance), so the AMDGPU backend lowers it to
    one ``v_exp_f32`` + ``v_rcp_f32`` per element.
    * All compute is done in f32 then truncated back to the activation
    dtype on store.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    I_DIM = spec.intermediate
    BS = spec.block_size
    EPT = spec.elems_per_thread_inter
    dtype = spec.dtype

    b = IRBuilder(spec.kernel_name("silu_mul"))
    b.kernel.attrs["max_workgroup_size"] = BS

    ty = io_ir_type(dtype)
    GateOut = b.param(
        "GateOut", PtrType(ty, "global"), noalias=True, readonly=True, align=16
    )
    UpOut = b.param(
        "UpOut", PtrType(ty, "global"), noalias=True, readonly=True, align=16
    )
    Hidden = b.param(
        "Hidden", PtrType(ty, "global"), noalias=True, writeonly=True, align=16
    )
    _total_pairs = b.param("total_pairs", I32)  # noqa: F841 - ABI
    _inter = b.param("intermediate", I32)  # noqa: F841 - ABI

    bid = b.block_id_x()
    tid = b.thread_id_x()
    row_base = b.mul(bid, b.const_i32(I_DIM))

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)

    for k in range(EPT):
        i_col = b.add(b.mul(tid, b.const_i32(EPT)), b.const_i32(k))
        in_i = b.cmp_lt(i_col, b.const_i32(I_DIM))
        with b.scf_if(in_i):
            off = b.add(row_base, i_col)
            g = load_scalar_as_f32(b, GateOut, off, dtype=dtype)
            u = load_scalar_as_f32(b, UpOut, off, dtype=dtype)
            sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
            silu = b.fmul(g, sig)
            h = b.fmul(silu, u)
            store_scalar_from_f32(b, Hidden, off, h, dtype=dtype)

    return b.kernel


# ---------------------------------------------------------------------
# Stage (6): topk-weighted reduce. The MoE-specific final accumulate.
# ---------------------------------------------------------------------


def build_moe_topk_weighted_reduce(spec: FusedMoeSpec) -> KernelDef:
    """Atomic topk-weighted scatter back into the per-token output.

    For each bucket ``b in [0, tokens*topk)``:

    token_id = SortedTokenIds[b] # i32 indirect
    w = SortedWeights[b] # f32 (router weight)
    for h_col in 0..hidden:
    atomic_add(Y[token_id, h_col], w * DownOut[b, h_col])

    The accumulator dtype is f32 -- ``Y`` must be a pre-cleared f32
    tensor of shape ``(tokens, hidden)``. The caller is responsible
    for the f32->dtype cast in a follow-on kernel (we keep this one
    atomic-safe and dtype-agnostic).

    Kernel signature::

    (DownOut: ptr<dtype, global>, # (tokens*topk, hidden)
    SortedTokenIds: ptr<i32, global>, # (tokens*topk,) from moe_sort
    SortedWeights: ptr<f32, global>, # (tokens*topk,) from moe_sort
    Y: ptr<f32, global>, # (tokens, hidden) f32 accumulator
    total_pairs: i32, hidden: i32, tokens: i32)

    Grid: ``(total_pairs, 1, 1)``; one CTA per bucket row, each thread
    handling ``elems_per_thread_hidden`` consecutive hidden cols.

    The atomic_add uses the default ``monotonic`` ordering -- that
    matches CK Tile's split-K and MoE-reduce loops (and is correct
    because every contributor sees the same destination shape, no
    cross-element ordering invariants).
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    H = spec.hidden
    BS = spec.block_size
    EPT = spec.elems_per_thread_hidden
    dtype = spec.dtype

    b = IRBuilder(spec.kernel_name("reduce"))
    b.kernel.attrs["max_workgroup_size"] = BS

    ty = io_ir_type(dtype)
    DownOut = b.param(
        "DownOut", PtrType(ty, "global"), noalias=True, readonly=True, align=16
    )
    SortedTokenIds = b.param(
        "SortedTokenIds",
        PtrType(I32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    SortedWeights = b.param(
        "SortedWeights",
        PtrType(F32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    Y = b.param("Y", PtrType(F32, "global"), align=4)
    _total_pairs = b.param("total_pairs", I32)  # noqa: F841 - ABI
    _hidden = b.param("hidden", I32)  # noqa: F841 - ABI
    _tokens = b.param("tokens", I32)  # noqa: F841 - ABI

    bid = b.block_id_x()
    tid = b.thread_id_x()

    token_id = load_sorted_token_id(b, SortedTokenIds, bid)
    weight = load_sorted_topk_weight(b, SortedWeights, bid)
    valid_token = b.cmp_ge(token_id, b.const_i32(0))
    bucket_base = b.mul(bid, b.const_i32(H))
    y_row_base = b.mul(token_id, b.const_i32(H))

    for k in range(EPT):
        h_col = b.add(b.mul(tid, b.const_i32(EPT)), b.const_i32(k))
        in_h = b.cmp_lt(h_col, b.const_i32(H))
        with b.scf_if(b.land(valid_token, in_h)):
            src_off = b.add(bucket_base, h_col)
            v = load_scalar_as_f32(b, DownOut, src_off, dtype=dtype)
            contrib = b.fmul(weight, v)
            dst_off = b.add(y_row_base, h_col)
            b.global_atomic_add(Y, dst_off, contrib)

    return b.kernel


# ---------------------------------------------------------------------
# Launch helpers (host-side glue).
# ---------------------------------------------------------------------


def moe_gather_grid(spec: FusedMoeSpec) -> Tuple[int, int, int]:
    return (spec.total_pairs, 1, 1)


def moe_silu_mul_grid(spec: FusedMoeSpec) -> Tuple[int, int, int]:
    return (spec.total_pairs, 1, 1)


def moe_topk_weighted_reduce_grid(spec: FusedMoeSpec) -> Tuple[int, int, int]:
    return (spec.total_pairs, 1, 1)


def moe_gather_signature(spec: FusedMoeSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("SortedTokenIds", "i32")
        .ptr("GroupedInput", spec.dtype)
        .scalar("tokens", "i32")
        .scalar("hidden", "i32")
        .build()
    )


def moe_silu_mul_signature(spec: FusedMoeSpec):
    return (
        SignatureBuilder()
        .ptr("GateOut", spec.dtype)
        .ptr("UpOut", spec.dtype)
        .ptr("Hidden", spec.dtype)
        .scalar("total_pairs", "i32")
        .scalar("intermediate", "i32")
        .build()
    )


def moe_topk_weighted_reduce_signature(spec: FusedMoeSpec):
    return (
        SignatureBuilder()
        .ptr("DownOut", spec.dtype)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("Y", "f32")
        .scalar("total_pairs", "i32")
        .scalar("hidden", "i32")
        .scalar("tokens", "i32")
        .build()
    )


def moe_fused_workspace_bytes(spec: FusedMoeSpec) -> int:
    """Aggregate scratch budget for one fused-MoE forward.

    Bytes = ``GroupedInput`` + ``GateOut`` + ``UpOut`` + ``Hidden`` +
    ``DownOut`` (all ``tokens*topk * {hidden|intermediate}`` in
    ``dtype``), plus the moe-sort workspace (handled by
    :func:`ck_dsl.instances.moe_sorting_workspace_bytes`).

    The caller is responsible for clearing ``Y`` (a f32 ``(tokens,
    hidden)`` accumulator) and the moe-sort histogram before the
    pipeline runs.
    """
    elem_bytes = 2  # f16 / bf16
    grouped = spec.total_pairs * spec.hidden * elem_bytes
    gate = spec.total_pairs * spec.intermediate * elem_bytes
    up = spec.total_pairs * spec.intermediate * elem_bytes
    hidden_buf = spec.total_pairs * spec.intermediate * elem_bytes
    down = spec.total_pairs * spec.hidden * elem_bytes
    return grouped + gate + up + hidden_buf + down


# ---------------------------------------------------------------------
# Orchestration scaffold.
# ---------------------------------------------------------------------


@dataclass
class FusedMoeLauncher:
    """Documentation-grade launcher for the full fused-MoE forward.

    This class does not own any GPU memory or launch primitives; it
    enumerates the *call graph* that a runtime driver should issue
    to execute :class:`FusedMoeSpec` end-to-end. The intended use::

    spec = FusedMoeSpec(tokens=..., experts=..., topk=...,
    hidden=..., intermediate=..., dtype="f16")
    launcher = FusedMoeLauncher(spec)
    for phase, kernel_def, grid, signature in launcher.plan():
    # ... dispatch via your KernelLauncher of choice ...
    pass

    The ``plan`` method yields the MoE-specific phases (sort, gather,
    silu_mul, reduce). Per-expert GEMMs are *not* iterated by the
    launcher (their grid depends on per-expert ``Counts[e]`` known
    only after the sort completes); supply a separate
    ``expert_gemm_fn(expert_idx, ...)`` callback at runtime and call
    it inside the gather -> gemm -> silu loop.

    For a production launcher with per-expert dispatch + workspace
    reuse, see the planned follow-on (the v1 launcher here is
    intentionally a callgraph descriptor, not a runtime loop, to keep
    the IR module pure-Python and import-time-safe).
    """

    spec: FusedMoeSpec
    name_prefix: str = "fused_moe"

    def plan(self) -> "list[tuple[str, KernelDef, Tuple[int, int, int], object]]":
        """Return ``[(phase_name, kernel_def, grid, signature), ...]``
        for the MoE-specific stages.

        Per-expert GEMMs (gate, up, down) and the smoothquant passes
        are NOT included; the caller composes them via the standard
        instance builders (``build_block_scale_gemm``,
        ``build_universal_gemm``, ``build_moe_smoothquant``).
        """
        s = self.spec
        return [
            (
                "gather",
                build_moe_gather(s),
                moe_gather_grid(s),
                moe_gather_signature(s),
            ),
            (
                "silu_mul",
                build_moe_silu_mul(s),
                moe_silu_mul_grid(s),
                moe_silu_mul_signature(s),
            ),
            (
                "topk_reduce",
                build_moe_topk_weighted_reduce(s),
                moe_topk_weighted_reduce_grid(s),
                moe_topk_weighted_reduce_signature(s),
            ),
        ]

    def expert_gemm_shape(
        self, *, stage: str, expert_count: int
    ) -> Tuple[int, int, int]:
        """Per-expert GEMM problem size for a given stage.

        Stages: ``"gate"`` / ``"up"`` (M=count[e], N=intermediate,
        K=hidden), ``"down"`` (M=count[e], N=hidden, K=intermediate).
        Used by the caller's ``expert_gemm_fn`` to pick a tile config
        appropriate for the per-expert M (which varies wildly across
        experts after the route).
        """
        s = self.spec
        if stage in ("gate", "up"):
            return (expert_count, s.intermediate, s.hidden)
        if stage == "down":
            return (expert_count, s.hidden, s.intermediate)
        raise ValueError(f"unknown stage {stage!r}; expected 'gate' / 'up' / 'down'")

    def workspace_bytes(self) -> int:
        return moe_fused_workspace_bytes(self.spec)
