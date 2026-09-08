# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused MoE forward pipeline (CK Tile ``15_fused_moe`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/15_fused_moe``. The
fused-MoE forward is a six-stage pipeline; the CK Tile reference
executes it as a small handful of kernel launches that share workspace
buffers between them. This module implements the three *MoE-specific*
kernels (the ones with no plain-GEMM analogue) plus a launcher class
that documents how to compose them with the existing GEMM and
quantisation kernels in :mod:`rocke.instances`.

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
:func:`rocke.instances.build_block_scale_gemm` (FP8 path) or
:func:`rocke.instances.build_universal_gemm` (FP16/BF16 path).

What this module *ships in IR*:

* :func:`build_moe_gather` -- gather kernel (the
 indirect ``GroupedInput[b, h] = X[sorted_token_ids[b], h]`` step).
* :func:`build_moe_silu_mul` -- SwiGLU activation fusion
 (``Hidden[b, i] = silu(GateOut[b, i]) * UpOut[b, i]``).
* :func:`build_moe_topk_weighted_reduce` -- the final atomic-accumulate
 pass (``atomic_add(Y[token_id, h], weight * DownOut[b, h])``).

Each kernel has a matching ``*_grid`` / ``*_signature`` helper for
use with :class:`rocke.runtime.launcher.KernelLauncher`.

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
from typing import Any, Callable, Dict, List, Literal, Mapping, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.distribution import (
    TileDistributionEncoding,
    make_static_distributed_tensor,
    make_static_tile_distribution,
)
from ...helpers.gather_scatter import (
    load_sorted_token_id,
    load_sorted_topk_weight,
)
from ...helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from ...helpers.spec import (
    SignatureBuilder,
    kernel_name_join,
)
from ...helpers.tensor_view import make_global_view, make_tile_window


def _resolve_launch_arch(arch: "str | None") -> str:
    """Resolve a launch-path target arch.

    Explicit ``arch`` wins; otherwise probe the running device via
    :func:`rocke.runtime.hip_module.get_device_arch` and fall back to
    ``"gfx950"`` when no HIP device is visible (static IR / cross-compile
    test environments). Kept tolerant of import / probe failure so the
    module stays import-time-safe without a HIP runtime.
    """
    if arch is not None:
        return arch
    try:
        from ...runtime.hip_module import get_device_arch

        dev = get_device_arch()
        if dev:
            return dev
    except Exception:
        pass
    return "gfx950"


def _effective_vec(spec_vec: int, block_size: int, n: int) -> int:
    """Largest power-of-two vec width in ``{1, 2, 4, 8}`` not exceeding
    ``spec_vec`` such that ``n`` is divisible by ``block_size * vec``.

    The streaming MoE kernels use the "interleaved chunks" pattern that
    :func:`rocke.helpers.sweep.sweep_row_chunks` encodes for the norm
    / elementwise families: each CTA covers ``n`` columns of one row,
    processed in chunks of ``block_size * vec`` consecutive elements;
    thread ``tid`` in chunk ``k`` handles
    ``[k*BS*vec + tid*vec, k*BS*vec + tid*vec + vec)``. The starting
    offset is always vec-aligned, which lets the AMDGPU backend emit
    ``global_load_dwordx{N}`` / ``ds_read_b128`` etc.

    Returns ``1`` only when none of the larger vec widths divide ``n``;
    the kernel body branches on ``VEC == 1`` and falls back to the
    scalar load/store path so callers do not need to special-case
    awkward shapes (e.g. ``H == block_size``).
    """
    ev = min(spec_vec, 8)
    while ev > 1 and (n % (block_size * ev) != 0):
        ev //= 2
    return ev


def _chunk_distribution(block_size: int, vec: int):
    """CK Tile distribution of one ``block_size * vec`` interleaved chunk.

    The MoE streaming kernels process one row per CTA over
    ``chunks = EPT // VEC`` consecutive ``block_size * vec`` slabs of the
    row. Each slab is the same fast tile the elementwise instance uses: a
    single 1D X dim split as ``Hs = ((block_size, vec),)`` with

      * level 0 (``block_size``) the lane axis -> P0
      * level 1 (``vec``)        the per-thread vector -> Y0

    so ``calculate_x`` reconstructs the in-chunk column ``tid*vec + y`` --
    exactly the ``k*BS*VEC + tid*VEC + i`` decode the hand-rolled chunk
    loop emits, now expressed through ``TileDistribution`` +
    ``StaticDistributedTensor`` (the same surface ``elementwise.py``
    drives). One ``load_tile`` / ``store_tile`` per chunk from a window
    anchored at ``row_base + k*BS*VEC`` keeps the emitted addresses
    identical.
    """
    encoding = TileDistributionEncoding(
        Hs=((block_size, vec),),
        Ps2RHs_major=((1,),),
        Ps2RHs_minor=((0,),),
        Ys2RHs_major=(1,),
        Ys2RHs_minor=(1,),
    )
    return make_static_tile_distribution(encoding)


def _silu_mul_f32(b, g, u, *, one_f32, c_neg_log2e):
    """Emit the f32 SwiGLU chain ``silu(g) * u`` (sigmoid via exp2).

    Same op order as the inline silu_mul sites:
    ``sig = rcp(1 + exp2(-x*log2e))``, ``silu = g*sig``, ``out = silu*u``.
    Constants are caller-supplied so the emitted SSA matches the existing
    inline order exactly. Kept module-local per the instance-file
    convention (see report 03 §4); a sibling copy lives in
    ``moe_gemm_fused.py``.
    """
    sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
    silu = b.fmul(g, sig)
    return b.fmul(silu, u)


__all__ = [
    "FusedMoeLauncher",
    "FusedMoeSpec",
    "build_moe_gather",
    "build_moe_silu_mul",
    "build_moe_silu_mul_packed",
    "build_moe_static_scatter_gather",
    "build_moe_topk_weighted_reduce",
    "is_valid_spec",
    "moe_gather_grid",
    "moe_gather_signature",
    "moe_silu_mul_grid",
    "moe_silu_mul_packed_grid",
    "moe_silu_mul_packed_signature",
    "moe_silu_mul_signature",
    "moe_static_scatter_gather_grid",
    "moe_static_scatter_gather_signature",
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
    name: str = "rocke_fused_moe"
    # P72: when True, ``build_moe_topk_weighted_reduce`` emits the
    # bf16-accumulator path that uses
    # :meth:`IRBuilder.global_atomic_add_pk_bf16` for halved atomic
    # contention vs the f32 atomic. The output ``Y`` becomes bf16
    # instead of f32. Real numerical change; callers gate on parity.
    bf16_accumulator: bool = False

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
    threads within the CTA stream the row through wide vector loads
    using the "interleaved chunks" layout the elementwise / norm
    kernels use. Each iteration covers ``BS * vec`` consecutive
    hidden columns, with thread ``tid`` reading ``vec`` consecutive
    elements starting at ``k*BS*vec + tid*vec``. The vec width is
    chosen to be the largest power-of-two in ``{1, 2, 4, 8}`` that
    ``BS * vec`` divides ``hidden`` (see :func:`_effective_vec`);
    ``vec == 1`` routes through ``global_load_f16`` for shapes where
    even vec=2 doesn't divide.

    Notes
    -----
    The indirect ``X[token_id]`` is a per-CTA invariant -- once
    ``token_id`` is loaded by lane 0 and pinned into an SGPR via
    :func:`to_sgpr_u32`, every lane derives its own per-column
    address with a scalar add and the ``token_id * hidden`` base
    stays in scalar registers across every chunk. Each lane issues
    one ``global_load_dwordx{N}`` per chunk (matching CK Tile's
    ``moe_sorting_kernel.hpp`` per-row gather and AITER's
    ``ck_moe`` token-permute kernels: load source row in vec-aligned
    chunks, store directly to bucket slot). Sentinel
    ``token_id < 0`` rows produce a zero gather -- the
    ``valid_token`` test is hoisted out of the inner loop so the
    branch is wave-uniform.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    H = spec.hidden
    BS = spec.block_size
    EPT = spec.elems_per_thread_hidden
    VEC = _effective_vec(spec.vec, BS, H)
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

    # Wave-uniform: every lane in the CTA pulls the same token_id +
    # validity. Pinning into an SGPR keeps the source row base
    # ``token_id * hidden`` in scalar registers across every chunk
    # iteration; without it the register allocator may copy the
    # base into VGPRs and bloat per-lane address arithmetic.
    token_id = b.to_sgpr_u32(load_sorted_token_id(b, SortedTokenIds, bid))
    valid_token = b.cmp_ge(token_id, b.const_i32(0))
    bucket_base = b.mul(bid, b.const_i32(H))
    src_row_base = b.mul(token_id, b.const_i32(H))

    chunks = EPT // VEC
    c_vec = b.const_i32(VEC)
    with b.scf_if(valid_token):
        for k in range(chunks):
            # Interleaved-chunk layout: chunk k covers
            # ``[k*BS*VEC, (k+1)*BS*VEC)`` along the hidden axis;
            # thread tid in the chunk reads ``VEC`` consecutive
            # elements starting at ``k*BS*VEC + tid*VEC``. The base
            # offset is always VEC-aligned, which is the precondition
            # for the AMDGPU backend to emit ``global_load_dwordx{N}``
            # / ``global_store_dwordx{N}``.
            h_col = b.add(b.const_i32(k * BS * VEC), b.mul(tid, c_vec))
            src_off = b.add(src_row_base, h_col)
            dst_off = b.add(bucket_base, h_col)
            if VEC == 1:
                # Fallback for shapes where BS * 2 > hidden (e.g.
                # H=BS). Keep the value in its native dtype -- gather
                # is a pure copy, no f32 round-trip needed.
                if dtype in ("f16", "fp16"):
                    v = b.global_load_f16(X, src_off)
                else:
                    v = b.global_load_bf16(X, src_off)
                b.global_store(GroupedInput, dst_off, v)
            else:
                v = b.global_load_vN(X, src_off, ty, VEC)
                b.global_store_vN(GroupedInput, dst_off, v, VEC)

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

    Grid: ``(total_pairs, 1, 1)``; one CTA per bucket row, threads in
    the CTA stream ``BS * vec`` consecutive intermediate columns per
    chunk using the same interleaved-chunks layout as
    :func:`build_moe_gather`. The effective vec is picked by
    :func:`_effective_vec` to be the largest power-of-two that
    ``BS * vec`` divides ``intermediate``.

    Implementation notes:

    * Per chunk, each lane issues one ``global_load_dwordx{N}`` for
      ``GateOut`` and one for ``UpOut``, computes the SwiGLU per
      lane in f32 (matching CK Tile's
      ``fused_moegemm_pipeline_flatmm_ex`` activation tile and
      AITER's ``moe_silu_mul`` epilogue), then issues one
      ``global_store_dwordx{N}`` for ``Hidden``. Halves the
      instruction count vs the old scalar path on the I/O side and
      gives the AMDGPU backend room to overlap the two reads.
    * The sigmoid is computed via ``exp2(-x * log2(e))`` (matches the
      formula used by :class:`rocke.helpers.fuse.SiLU` and the
      ``elementwise`` instance), so the AMDGPU backend lowers it to
      one ``v_exp_f32`` + ``v_rcp_f32`` per element. Bit-equivalent
      to the prior scalar path -- parity tests stay green at the
      same 5e-3 tolerance.
    * All compute is done in f32 then truncated back to the activation
      dtype on store.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    I_DIM = spec.intermediate
    BS = spec.block_size
    EPT = spec.elems_per_thread_inter
    VEC = _effective_vec(spec.vec, BS, I_DIM)
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
    c_vec = b.const_i32(VEC)

    chunks = EPT // VEC
    if VEC == 1:
        # Fallback for shapes where BS * 2 > intermediate: scalar
        # load / op / store per column (LoadStoreTraits would also
        # pick scalar here, but the inline path avoids the tile
        # plumbing on a degenerate vec width).
        for k in range(chunks):
            i_col = b.add(b.const_i32(k * BS * VEC), b.mul(tid, c_vec))
            off = b.add(row_base, i_col)
            g = load_scalar_as_f32(b, GateOut, off, dtype=dtype)
            u = load_scalar_as_f32(b, UpOut, off, dtype=dtype)
            h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e)
            store_scalar_from_f32(b, Hidden, off, h, dtype=dtype)
        return b.kernel

    # CK Tile distribution path: one (BS, VEC) chunk per load_tile /
    # store_tile, the lane P feeding level 0 of the H decomposition.
    distribution = _chunk_distribution(BS, VEC)
    chunk_elems = BS * VEC
    gate_view = make_global_view(GateOut, shape=(chunk_elems,), dtype=ty)
    up_view = make_global_view(UpOut, shape=(chunk_elems,), dtype=ty)
    out_view = make_global_view(Hidden, shape=(chunk_elems,), dtype=ty)
    ps = [[tid]]
    for k in range(chunks):
        chunk_origin = (b.add(row_base, b.const_i32(k * BS * VEC)),)
        gate_tile = make_tile_window(
            gate_view, lengths=(chunk_elems,), origin=chunk_origin
        )
        up_tile = make_tile_window(up_view, lengths=(chunk_elems,), origin=chunk_origin)
        out_tile = make_tile_window(
            out_view, lengths=(chunk_elems,), origin=chunk_origin
        )
        g_dt = gate_tile.load(b, distribution=distribution, ps=ps)
        u_dt = up_tile.load(b, distribution=distribution, ps=ps)
        out_dt = make_static_distributed_tensor(distribution, dtype=ty)
        for y in distribution.iterate_ys():
            out_dt.set(
                y,
                _silu_mul_f32(
                    b,
                    g_dt.get(y),
                    u_dt.get(y),
                    one_f32=one_f32,
                    c_neg_log2e=c_neg_log2e,
                ),
            )
        out_tile.store(b, out_dt, ps=ps)

    return b.kernel


# ---------------------------------------------------------------------
# Stage (4'): silu_mul packed variant.
# Mirrors AITER's "G1U1" gate-up packing: one (M, 2*I) input buffer
# instead of two separate (M, I) buffers. Halves the global-memory
# traffic vs the unpacked path (one B-tensor in the gate+up GEMM
# instead of two; one fused output buffer here instead of two
# separate gate / up buffers); pairs with a single batched GEMM
# whose N axis = 2*I and W = concat(W_gate, W_up, dim=1).
# ---------------------------------------------------------------------


def build_moe_silu_mul_packed(spec: FusedMoeSpec) -> KernelDef:
    """SwiGLU activation reading from a packed gate+up buffer.

    ``Hidden[b, i] = silu(GateUp[b, i]) * GateUp[b, I + i]`` where
    ``GateUp`` has shape ``(M, 2*I)`` row-major. The "G1U1" weight
    packing convention: column slab ``[0, I)`` holds the gate output,
    ``[I, 2*I)`` holds the up output. This kernel is the activation
    half of the gate+up fusion -- the matching GEMM is one batched
    GEMM with ``N = 2*I`` and ``W_gate_up = torch.cat([W_gate, W_up],
    dim=1)`` (shape ``(E, 2*I, H)``).

    Kernel signature::

        (GateUp:    ptr<dtype, global>,   # (tokens*topk, 2*intermediate)
         Hidden:    ptr<dtype, global>,   # (tokens*topk, intermediate)
         total_pairs: i32,
         intermediate: i32)

    Grid: ``(total_pairs, 1, 1)``; one CTA per bucket row, each
    thread handling ``elems_per_thread_inter`` consecutive
    intermediate columns. Identical numerics to
    :func:`build_moe_silu_mul` (sigmoid via
    ``exp2(-x * log2(e))``, all compute in f32, truncate to dtype on
    store) so parity tests can compare the two paths.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    I_DIM = spec.intermediate
    BS = spec.block_size
    EPT = spec.elems_per_thread_inter
    VEC = _effective_vec(spec.vec, BS, I_DIM)
    dtype = spec.dtype

    b = IRBuilder(spec.kernel_name("silu_mul_packed"))
    b.kernel.attrs["max_workgroup_size"] = BS

    ty = io_ir_type(dtype)
    GateUp = b.param(
        "GateUp", PtrType(ty, "global"), noalias=True, readonly=True, align=16
    )
    Hidden = b.param(
        "Hidden", PtrType(ty, "global"), noalias=True, writeonly=True, align=16
    )
    _total_pairs = b.param("total_pairs", I32)  # noqa: F841 - ABI
    _inter = b.param("intermediate", I32)  # noqa: F841 - ABI

    bid = b.block_id_x()
    tid = b.thread_id_x()
    # Row stride for the packed buffer is 2*I; row b's gate slab
    # starts at b * 2*I, up slab starts at b * 2*I + I. Output row
    # stride is I (Hidden is unpacked).
    two_i = b.const_i32(2 * I_DIM)
    i_const = b.const_i32(I_DIM)
    gate_base = b.mul(bid, two_i)
    up_base = b.add(gate_base, i_const)
    out_base = b.mul(bid, i_const)

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    c_vec = b.const_i32(VEC)

    # Interleaved-chunks layout (matches build_moe_silu_mul): per
    # chunk, each lane reads one VEC of the gate slab + one VEC of
    # the up slab, stores one VEC into Hidden. With the G1U1 packing
    # the gate and up slabs are at fixed offsets I_DIM apart, so a
    # single base pointer + two adds let the AMDGPU backend coalesce
    # both reads into one ``s_load_dwordx{N}`` + a pair of paired
    # VMEM transactions.
    chunks = EPT // VEC
    if VEC == 1:
        # Scalar fallback (BS * 2 > intermediate): read gate / up from
        # their fixed-offset slabs of the packed buffer per column.
        for k in range(chunks):
            i_col = b.add(b.const_i32(k * BS * VEC), b.mul(tid, c_vec))
            g_off = b.add(gate_base, i_col)
            u_off = b.add(up_base, i_col)
            o_off = b.add(out_base, i_col)
            g = load_scalar_as_f32(b, GateUp, g_off, dtype=dtype)
            u = load_scalar_as_f32(b, GateUp, u_off, dtype=dtype)
            h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e)
            store_scalar_from_f32(b, Hidden, o_off, h, dtype=dtype)
        return b.kernel

    # CK Tile distribution path: gate and up windows index the same
    # GateUp view at the two G1U1 slab origins (gate_base, up_base);
    # one load_tile per slab per chunk, one store_tile into Hidden.
    distribution = _chunk_distribution(BS, VEC)
    chunk_elems = BS * VEC
    gateup_view = make_global_view(GateUp, shape=(chunk_elems,), dtype=ty)
    out_view = make_global_view(Hidden, shape=(chunk_elems,), dtype=ty)
    ps = [[tid]]
    for k in range(chunks):
        col_off = b.const_i32(k * BS * VEC)
        gate_origin = (b.add(gate_base, col_off),)
        up_origin = (b.add(up_base, col_off),)
        out_origin = (b.add(out_base, col_off),)
        gate_tile = make_tile_window(
            gateup_view, lengths=(chunk_elems,), origin=gate_origin
        )
        up_tile = make_tile_window(
            gateup_view, lengths=(chunk_elems,), origin=up_origin
        )
        out_tile = make_tile_window(out_view, lengths=(chunk_elems,), origin=out_origin)
        g_dt = gate_tile.load(b, distribution=distribution, ps=ps)
        u_dt = up_tile.load(b, distribution=distribution, ps=ps)
        out_dt = make_static_distributed_tensor(distribution, dtype=ty)
        for y in distribution.iterate_ys():
            out_dt.set(
                y,
                _silu_mul_f32(
                    b,
                    g_dt.get(y),
                    u_dt.get(y),
                    one_f32=one_f32,
                    c_neg_log2e=c_neg_log2e,
                ),
            )
        out_tile.store(b, out_dt, ps=ps)

    return b.kernel


def moe_silu_mul_packed_grid(spec: FusedMoeSpec) -> Tuple[int, int, int]:
    return (spec.total_pairs, 1, 1)


def moe_silu_mul_packed_signature(spec: FusedMoeSpec):
    return (
        SignatureBuilder()
        .ptr("GateUp", spec.dtype)
        .ptr("Hidden", spec.dtype)
        .scalar("total_pairs", "i32")
        .scalar("intermediate", "i32")
        .build()
    )


# ---------------------------------------------------------------------
# Static-mode scatter+gather fusion.
# ---------------------------------------------------------------------


def build_moe_static_scatter_gather(spec: FusedMoeSpec) -> KernelDef:
    """Static-offset scatter and gather in one kernel.

    This is the static-offset fast path's replacement for the separate
    ``moe_sort_scatter`` and ``moe_gather`` kernels:

    * read ``TopkIds[t, k]`` and ``TopkWeights[t, k]``;
    * claim the next slot in expert ``eid`` via
      ``atomic_add(Counter[eid], 1)``;
    * write ``SortedTokenIds[slot] = t`` and
      ``SortedWeights[slot] = weight``;
    * copy ``X[t, :]`` directly into ``GroupedInput[slot, :]``.

    Static slot layout is ``slot = eid * slot_size + local_off``. Padded
    rows are initialized by the caller (``SortedTokenIds=-1``,
    ``GroupedInput=0``) and untouched by this kernel.

    Kernel signature::

        (TopkIds: ptr<i32>, TopkWeights: ptr<f32>, Counter: ptr<i32>,
         X: ptr<dtype>, SortedTokenIds: ptr<i32>, SortedWeights: ptr<f32>,
         GroupedInput: ptr<dtype>,
         tokens: i32, topk: i32, num_experts: i32,
         hidden: i32, slot_size: i32)

    Grid: ``(tokens * topk, 1, 1)``; one CTA per routed pair. Threads in
    the CTA cooperatively copy the hidden row.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fused_moe spec: {why}")

    H = spec.hidden
    BS = spec.block_size
    EPT = spec.elems_per_thread_hidden
    VEC = _effective_vec(spec.vec, BS, H)
    dtype = spec.dtype

    b = IRBuilder(spec.kernel_name("static_scatter_gather"))
    b.kernel.attrs["max_workgroup_size"] = BS

    ty = io_ir_type(dtype)
    TopkIds = b.param(
        "TopkIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    TopkWeights = b.param(
        "TopkWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    Counter = b.param("Counter", PtrType(I32, "global"), align=4)
    X = b.param("X", PtrType(ty, "global"), noalias=True, readonly=True, align=16)
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), writeonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), writeonly=True, align=4
    )
    GroupedInput = b.param(
        "GroupedInput", PtrType(ty, "global"), writeonly=True, align=16
    )
    tokens = b.param("tokens", I32)
    topk = b.param("topk", I32)
    num_experts = b.param("num_experts", I32)
    _hidden = b.param("hidden", I32)  # noqa: F841 - ABI compatibility
    slot_size = b.param("slot_size", I32)

    bid = b.block_id_x()
    tid = b.thread_id_x()
    out_row_slot = b.smem_alloc(I32, [1], name_hint="sg_out_row")
    num_pairs = b.mul(tokens, topk)
    in_bounds = b.cmp_lt(bid, num_pairs)

    c_vec = b.const_i32(VEC)
    chunks = EPT // VEC

    with b.scf_if(in_bounds):
        eid = b.global_load_i32(TopkIds, bid)
        valid_e = b.land(b.cmp_ge(eid, b.const_i32(0)), b.cmp_lt(eid, num_experts))
        with b.scf_if(valid_e):
            t_idx = b.div(bid, topk)
            is_lead = b.cmp_eq(tid, b.const_i32(0))
            with b.scf_if(is_lead):
                local = b.global_atomic_add(Counter, eid, b.const_i32(1))
                base = b.mul(eid, slot_size)
                out_row_lead = b.add(base, local)
                b.smem_store_vN(out_row_slot, [b.const_i32(0)], out_row_lead, n=1)

                w = b.global_load_f32(TopkWeights, bid)
                b.global_store(SortedTokenIds, out_row_lead, t_idx, align=4)
                b.global_store(SortedWeights, out_row_lead, w, align=4)
            b.sync()
            # Wave-uniform broadcast (single ds_read + LDS-to-SGPR
            # promotion). Lift the destination row base into an SGPR
            # so the per-chunk address arithmetic stays scalar.
            out_row = b.to_sgpr_u32(
                b.vec_extract(
                    b.smem_load_vN(out_row_slot, b.const_i32(0), dtype=I32, n=1),
                    0,
                )
            )
            src_row_base = b.mul(t_idx, b.const_i32(H))
            dst_row_base = b.mul(out_row, b.const_i32(H))

            # Copy X[t_idx, :] -> GroupedInput[out_row, :] in
            # interleaved-chunk vec loads (matches build_moe_gather).
            for k in range(chunks):
                h_col = b.add(b.const_i32(k * BS * VEC), b.mul(tid, c_vec))
                src = b.add(src_row_base, h_col)
                dst = b.add(dst_row_base, h_col)
                if VEC == 1:
                    if dtype in ("f16", "fp16"):
                        v = b.global_load_f16(X, src)
                    else:
                        v = b.global_load_bf16(X, src)
                    b.global_store(GroupedInput, dst, v)
                else:
                    v = b.global_load_vN(X, src, ty, VEC)
                    b.global_store_vN(GroupedInput, dst, v, VEC)

    return b.kernel


def moe_static_scatter_gather_grid(spec: FusedMoeSpec) -> Tuple[int, int, int]:
    return (spec.total_pairs, 1, 1)


def moe_static_scatter_gather_signature(spec: FusedMoeSpec):
    return (
        SignatureBuilder()
        .ptr("TopkIds", "i32")
        .ptr("TopkWeights", "f32")
        .ptr("Counter", "i32")
        .ptr("X", spec.dtype)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("GroupedInput", spec.dtype)
        .scalar("tokens", "i32")
        .scalar("topk", "i32")
        .scalar("num_experts", "i32")
        .scalar("hidden", "i32")
        .scalar("slot_size", "i32")
        .build()
    )


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
    VEC = _effective_vec(spec.vec, BS, H)
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

    # Per-CTA invariants pinned into SGPRs so per-chunk address
    # arithmetic stays scalar. token_id + weight are wave-uniform;
    # ``to_sgpr_u32`` makes that explicit to the backend (one
    # ``s_load_dword`` + readfirstlane vs N redundant VMEM loads).
    token_id = b.to_sgpr_u32(load_sorted_token_id(b, SortedTokenIds, bid))
    weight = load_sorted_topk_weight(b, SortedWeights, bid)
    valid_token = b.cmp_ge(token_id, b.const_i32(0))
    bucket_base = b.mul(bid, b.const_i32(H))
    y_row_base = b.mul(token_id, b.const_i32(H))
    # Block-partitioned layout (kept from the original scalar code):
    # each lane owns a contiguous run of ``EPT`` hidden columns;
    # adjacent lanes' chunks are ``EPT`` apart. Critically for the
    # f32 atomic_add into ``Y``, this means adjacent lanes target
    # DIFFERENT cachelines per iteration (no intra-wave atomic
    # contention) -- the interleaved layout that gather / silu_mul
    # use bunches all 64 lanes onto the same 4 cachelines and
    # measured ~16% slower on high-contention shapes (T=128, K=2
    # prefill: many CTAs share token_id rows). See the report's
    # benchmark section for the layout-vs-VEC sweep.
    #
    # The hidden axis is still vec-loaded from ``DownOut`` (the
    # bandwidth-dominant read), but per-lane: each lane issues
    # ``EPT / VEC`` wide loads of its own chunk, then ``VEC``
    # scalar atomic_adds into its own chunk. AMDGPU has no
    # packed-f32 atomic on gfx9/gfx94x; the only packed global
    # atomic is ``v2bf16`` via ``global_atomic_add_pk_bf16`` which
    # would forfeit the f32 accumulator precision. The win is in
    # halving the DownOut load instruction count without changing
    # the atomic pattern.
    chunks = EPT // VEC
    lane_chunk_base = b.mul(tid, b.const_i32(EPT))
    with b.scf_if(valid_token):
        for k in range(chunks):
            h_col = b.add(lane_chunk_base, b.const_i32(k * VEC))
            src_off = b.add(bucket_base, h_col)
            dst_off = b.add(y_row_base, h_col)
            if VEC == 1:
                v = load_scalar_as_f32(b, DownOut, src_off, dtype=dtype)
                contrib = b.fmul(weight, v)
                b.global_atomic_add(Y, dst_off, contrib)
            else:
                v_vec = b.global_load_vN(DownOut, src_off, ty, VEC)
                for i in range(VEC):
                    v = b.cast_to_f32(b.vec_extract(v_vec, i))
                    contrib = b.fmul(weight, v)
                    b.global_atomic_add(Y, b.add(dst_off, b.const_i32(i)), contrib)

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
    :func:`rocke.instances.moe_sorting_workspace_bytes`).

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


_PHASE_ORDER: Tuple[str, str, str] = ("gather", "silu_mul", "topk_reduce")


@dataclass
class FusedMoeLauncher:
    """Launcher for the MoE-specific phases of the fused-MoE forward.

    Drives the three MoE-specific kernels (gather, SwiGLU activation,
    top-k weighted reduce) as a CK-Tile-style chained launch on a
    single HIP stream. The call graph mirrors the C++ reference at
    ``example/ck_tile/15_fused_moe`` but uses the
    :func:`~rocke.runtime.launcher.launch_kernel` /
    :func:`~rocke.runtime.launcher.make_kernel` primitive instead of
    a hand-rolled launch loop. Same-stream FIFO ordering on HIP is
    the only correctness primitive between phases -- no events, no
    intermediate fences -- which matches both the CK Tile semantic
    and the
    ``ck_tile::launch_kernel(s, MOE_SORTING_MP_0, MOE_SORTING_MP_1,
    MOE_SORTING_MP_23)`` macro pattern in the fused-MoE example
    sorting dispatcher.

    Production usage::

        spec = FusedMoeSpec(tokens=..., experts=..., topk=...,
                            hidden=..., intermediate=..., dtype="f16")
        launcher = FusedMoeLauncher(spec)
        ms = launcher.run(
            {
                "gather":      {"X": X, "SortedTokenIds": sids,
                                "GroupedInput": grouped, "tokens": T,
                                "hidden": H},
                "silu_mul":    {"GateOut": gate, "UpOut": up,
                                "Hidden": hidden, "total_pairs": T*K,
                                "intermediate": I},
                "topk_reduce": {"DownOut": down, "SortedTokenIds": sids,
                                "SortedWeights": weights, "Y": Y,
                                "total_pairs": T*K, "hidden": H,
                                "tokens": T},
            },
            stream=stream,
        )

    The launcher is the v2 production runtime. It compiles each of
    the three IR kernels lazily on first :meth:`run` /
    :meth:`make_callables` call, caches the resulting
    :class:`~rocke.runtime.launcher.KernelLauncher` instances on
    ``self`` (one HSACO + module + function per phase), and reuses
    them across every subsequent dispatch. Construct one
    :class:`FusedMoeLauncher` per :class:`FusedMoeSpec` and reuse it
    for the lifetime of the problem.

    Scope notes
    -----------
    The MoE forward also requires per-expert gate / up / down GEMMs
    (block-scale or universal) and an optional smoothquant pass --
    these are *not* iterated by the launcher because their per-expert
    grids depend on ``Counts[e]`` (only known after the sort). The
    caller composes them around the gather + silu_mul + topk_reduce
    chain using the standard instance builders
    (:func:`build_block_scale_gemm`, :func:`build_universal_gemm`,
    :func:`build_moe_smoothquant`) and an
    ``expert_gemm_fn(expert_idx, a_ptr, b_ptr, c_ptr, m, n, k)``
    callback. The MoE sort phases (histogram / scan / scatter) live
    in :mod:`rocke.instances.common.moe_sorting` and chain analogously
    through :func:`launch_kernel`.

    The legacy :meth:`plan` method is preserved as a pure descriptor
    -- it returns the list of ``(phase, kernel_def, grid, signature)``
    tuples without compiling anything, for callers that want to
    inspect the call graph or drive their own dispatch.
    """

    spec: FusedMoeSpec
    name_prefix: str = "fused_moe"
    # Target GPU for the lazy ``compile_kernel`` calls. ``None`` resolves
    # to the running device (``runtime.hip_module.get_device_arch()``)
    # and falls back to ``"gfx950"`` when no device is visible (static
    # IR / cross-compile test environments). The MoE streaming kernels
    # here (gather / silu_mul / topk_reduce) emit no MFMA, so they build
    # identically on gfx942 and gfx950 -- threading ``arch`` only keeps
    # the lowered ISA matched to the launch target.
    arch: "str | None" = None

    def __post_init__(self) -> None:
        # Lazy KernelLauncher cache: built on first :meth:`run` /
        # :meth:`make_callables` call, reused thereafter. Holds one
        # HSACO + HIP module + kernel function handle per phase.
        # Stored via ``object.__setattr__`` so dataclass-generated
        # ``__repr__`` stays stable (the launchers are not part of
        # the spec identity).
        object.__setattr__(self, "_launchers", None)

    def plan(self) -> "list[tuple[str, KernelDef, Tuple[int, int, int], object]]":
        """Return ``[(phase_name, kernel_def, grid, signature), ...]``
        for the MoE-specific stages.

        Per-expert GEMMs (gate, up, down) and the smoothquant passes
        are NOT included; the caller composes them via the standard
        instance builders (``build_block_scale_gemm``,
        ``build_universal_gemm``, ``build_moe_smoothquant``).

        Pure descriptor: does not compile anything, does not allocate
        :class:`KernelLauncher` instances. Use :meth:`make_callables`
        or :meth:`run` for the runtime path.
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

    def _ensure_launchers(self) -> Dict[str, Any]:
        """Compile the 3 MoE-specific kernels on first call and cache
        one :class:`~rocke.runtime.launcher.KernelLauncher` per
        phase. Subsequent calls return the cached dict directly.

        Imports the runtime helpers lazily to keep
        :mod:`rocke.instances.common.fused_moe` import-time-safe (the
        module is exercised by static IR tests in environments
        without a HIP runtime).
        """
        if self._launchers is not None:  # type: ignore[has-type]
            return self._launchers  # type: ignore[has-type]
        from ...helpers.compile import compile_kernel
        from ...runtime.launcher import KernelLauncher

        arch = _resolve_launch_arch(self.arch)
        s = self.spec
        out: Dict[str, Any] = {}
        for phase, kernel_def, _grid, signature in self.plan():
            artifact = compile_kernel(kernel_def, arch=arch, capture_ir_text=False)
            out[phase] = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=signature,
                cache_key=("fused_moe", phase, s.kernel_name(phase)),
            )
        object.__setattr__(self, "_launchers", out)
        return out

    def make_callables(
        self,
        values: Mapping[str, Mapping[str, Any]],
    ) -> List[Callable[[Any], None]]:
        """Bake per-phase ``(values, grid, block, lds_bytes=0)`` into
        three :func:`~rocke.runtime.launcher.make_kernel` closures,
        ready for :func:`~rocke.runtime.launcher.launch_kernel`.

        ``values`` is a phase-keyed mapping with entries for each of
        ``"gather"``, ``"silu_mul"``, ``"topk_reduce"`` matching the
        per-phase signature returned by :func:`moe_gather_signature`,
        :func:`moe_silu_mul_signature`, and
        :func:`moe_topk_weighted_reduce_signature`. All three phases
        use ``block=(spec.block_size, 1, 1)`` and ``lds_bytes=0``.

        Returns the closures in declaration order (gather, silu_mul,
        topk_reduce). Use this entry point if you want to interleave
        :func:`make_kernel` closures with arbitrary host lambdas
        (for example, a ``maybe_clear_workspace`` callable that
        zeroes the output ``Y`` accumulator before topk_reduce
        starts) before passing the full list to
        :func:`launch_kernel`. For the simple case of "just run all
        three on one stream", :meth:`run` is the one-call shortcut.
        """
        from ...runtime.launcher import make_kernel

        missing = [p for p in _PHASE_ORDER if p not in values]
        if missing:
            raise KeyError(
                f"FusedMoeLauncher.make_callables: missing values for phase(s) "
                f"{missing!r}; expected keys {list(_PHASE_ORDER)!r}"
            )
        s = self.spec
        block = (s.block_size, 1, 1)
        launchers = self._ensure_launchers()
        return [
            make_kernel(
                launchers["gather"],
                values["gather"],
                moe_gather_grid(s),
                block,
            ),
            make_kernel(
                launchers["silu_mul"],
                values["silu_mul"],
                moe_silu_mul_grid(s),
                block,
            ),
            make_kernel(
                launchers["topk_reduce"],
                values["topk_reduce"],
                moe_topk_weighted_reduce_grid(s),
                block,
            ),
        ]

    def run(
        self,
        values: Mapping[str, Mapping[str, Any]],
        *,
        stream: int = 0,
        time_kernel: bool = False,
        cold_niters: int = 3,
        nrepeat: int = 10,
        is_gpu_timer: bool = True,
    ) -> float:
        """Drive gather -> silu_mul -> topk_reduce as a CK-Tile-style
        chained launch on ``stream``.

        Returns ``0.0`` when ``time_kernel=False`` (production
        dispatch) and the average per-iteration wall time in
        milliseconds when ``time_kernel=True`` (benchmark loop with
        ``cold_niters`` warmup iters + ``nrepeat`` timed iters
        wrapping the whole 3-callable group; see
        :func:`launch_kernel` for the full timing contract).

        Production callers that read the output tensors on the host
        immediately after :meth:`run` returns should add an explicit
        :func:`~rocke.runtime.launcher.wait_stream_and_release`
        (or rely on torch's stream-aware next-read sync). The
        primitive does not implicitly fence at the end of the
        non-timing path; the timing path's outer
        :class:`~rocke.runtime.hip_module.Event` synchronize is the
        only sync point.
        """
        from ...runtime.launcher import StreamConfig, launch_kernel

        callables = self.make_callables(values)
        return launch_kernel(
            StreamConfig(
                stream_id=int(stream),
                time_kernel=bool(time_kernel),
                cold_niters=int(cold_niters),
                nrepeat=int(nrepeat),
                is_gpu_timer=bool(is_gpu_timer),
            ),
            *callables,
        )
