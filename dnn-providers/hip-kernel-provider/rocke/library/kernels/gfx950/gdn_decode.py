# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN/KDA single-token decode kernel instance builder.

For each active sequence and value head, advance one linear-attention decode
step over a fixed-size recurrent state ``S`` (a ``head_v_dim x head_k_dim``
key->value matrix), per the gated delta rule:

    q_hat = l2norm(q) * head_k_dim**-0.5
    k_hat = l2norm(k)
    log_decay_gdn    = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
    log_decay_kda[d] = lower_bound * sigmoid(exp(A_log[h]) * (a[h,d] + dt_bias[h,d]))
    decay = exp(log_decay)
    beta  = sigmoid(b[h])
    S     = S * decay
    v_new = (v - S @ k_hat) * beta
    out   = S @ q_hat + v_new * dot(k_hat, q_hat)
    S     = S + outer(v_new, k_hat)

GDN uses one scalar decay per value head. KDA uses one decay per K channel and
therefore scales the columns of ``S``. Only gate production and the fade differ;
the remaining recurrence and serving infrastructure are shared.

Only pages named by ``read_indices`` / ``write_indices`` are touched. The
contract is that ``-1`` is the ONLY valid skip sentinel and every active lane
owns a unique write index. Repeated read pages are allowed, and one sequence
may update the page it read; but a sequence must not write a page that
ANOTHER active sequence reads in the same launch (different workgroups would
race on it, in the state pool and in the fused conv-state pool alike). The
host guard does not check that last rule; it does reject any negative index
other than ``-1`` before launch. This code is a lenient superset of that contract: it skips
on ``read < 0 or write < 0``, so on a path that bypasses the host guard
(``validate_indices=False``) a stray ``-7`` is silently treated as a skip rather
than caught. Callers should treat ``-1`` as the only sentinel; the wider test
here is defence, not licence.

Tensors are row-major and contiguous, except the two fused-mode rows noted
below, matching the packed linear-attention decode contract:

    query, key   : [B, 1, num_k_heads, head_k_dim]   dtype
    value, out   : [B, 1, num_v_heads, head_v_dim]   dtype
    a (GDN)      : [B, 1, num_v_heads]               dtype
    a (KDA)      : [B, 1, num_v_heads, head_k_dim]   dtype
    b            : [B, 1, num_v_heads]               dtype
    dt_bias GDN  : [num_v_heads]                     dtype
    dt_bias KDA  : [num_v_heads, head_k_dim]         f32
    A_log        : [num_v_heads]                     f32
    read/write_indices : [B]                         i32
    state        : [pool, num_v_heads, head_v_dim, head_k_dim]  state_dtype (f16/bf16/f32)

The optional fusions (``fuse_conv``, ``fuse_out_norm``; ALGORITHM.md section
4.9) add or replace tensors. ``conv_dim = 2*Hk*Dk + Hv*Dv`` (the spec's
``conv_dim`` property). ``mixed_qkv`` and ``out_gate`` may be row slices of a
wider buffer: their rows are ``qkv_stride`` / ``og_stride`` elements apart and
each row is contiguous; everything else is contiguous:

    mixed_qkv    : [B, >= conv_dim], row stride qkv_stride  dtype (replaces q/k/v)
    conv_state   : [pool, conv_dim, CONV_WIDTH - 1]         dtype
    conv_weight  : [conv_dim, CONV_WIDTH]                   f32
    out_gate     : [B, >= Hv*Dv], row stride og_stride      dtype
    norm_weight  : [head_v_dim]                             f32

Every pointer argument must start at the byte alignment that
``gdn_decode_pointer_alignment`` names for it; the emitted vector accesses
assume it and the host launch path checks it.

**Two emitters, one contract**, selected by ``GdnDecodeSpec.simple``.

``simple=False`` -- the default, and the only path dispatch can reach -- is
**warp-tiled**: ``num_warps * wave_size`` threads per workgroup and
``blocks_per_v_dim`` workgroups per ``(sequence, value_head)``. Each warp splits
the ``head_k_dim`` reduction across ``warp_threads_k`` lanes and recombines with
an XOR butterfly (``quad_perm`` at offsets 1-2, ``ds_swizzle`` wider; with
``dpp_reduce`` offsets 4 and 8 use ``tile.dpp_xor`` instead), so the
unfused kernel allocates no LDS (``fuse_out_norm`` adds a ``num_warps``-float
LDS reduction when ``num_warps > 1``, ALGORITHM.md section 4.9; the
``conv_once`` and ``out_lds`` knobs add LDS blocks of their own, section
4.10). Dispatch selects this tile from the gfx950 GDN registry: `auto` prefers
one tile per request (``dispatch.gdn.gfx950.auto_tile``: GDN's static default,
KDA's work-keyed table, or the shared fused default for a fused request), while
an explicit `spec_id` pins a tile.

``simple=True`` is the v1 reference: one workgroup per ``(sequence, value_head)``,
``head_v_dim`` threads, thread ``t`` owning state row ``t`` (the full
``head_k_dim``-wide key vector for value-dim ``t``) in registers, so every dot
product is thread-local and needs no cross-thread reduction. Q/K L2 norms and
``dot(k,q)`` are recomputed per thread -- redundant but simple -- which makes it
VGPR-heavy by construction. It is **not reachable through dispatch**; it exists
as the correctness baseline the warp-tiled path is validated against, and is
selected only by naming the spec directly (see ``ALGORITHM.md`` section 4.7).

Built for gfx950 (wave64) and placed alongside the KDA chunkwise kernel in
``kernels/gfx950/``; the ``arch`` argument is a validation/target hook, not a
portability claim -- a new arch adds its own tuned specs here rather than
importing across folders.
"""

from __future__ import annotations

import math
import numbers
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, Iterator, Literal, Optional, Tuple, get_args

from rocke.helpers.activations import LN2, LOG2E, SOFTPLUS_THRESHOLD
from rocke.core.ir import (
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    TemporalHint,
    Value,
)
from rocke.helpers.grid import NUM_XCDS_MI350X
from rocke.helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from rocke.helpers.reduction import block_lds_reduce_with_wave_prologue, tree_reduce
from rocke.helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join

__all__ = [
    "GdnDecodeSpec",
    "is_valid_spec",
    "build_gdn_decode",
    "gdn_decode_grid",
    "gdn_decode_signature",
    "GDN_DTYPES",
    "STATE_DTYPES",
    "fuse_conv_heads_error",
    "CONV_WIDTH",
    "OUT_GATE_ACTIVATION",
    "gdn_decode_pointer_alignment",
    "STATE_HINTS",
    "dpp_reduce_applies",
    "xcd_remap_applies",
    "stream_rows_applies",
    "interleave_cols_applies",
]

DType = Literal["f16", "bf16"]
# The dtypes this kernel can emit, derived from the type rather than restated
# beside it. Dispatch RE-EXPORTS this: a copy drifts in the direction that
# fails silently -- a prefilter rejecting a shape the kernel has since learned
# to run, or admitting one it cannot.
GDN_DTYPES = get_args(DType)
# The recurrent state may also be f32: a serving stack may keep the decode
# state in f32, and prefill can hand it over in f32. Only the state widens;
# q/k/v, the gates and the output stay in ``dtype``. The kernel already does
# every state update in f32 registers, so f32 changes only the state loads and
# stores.
StateDType = Literal["f16", "bf16", "f32"]
STATE_DTYPES = get_args(StateDType)


def fuse_conv_heads_error(num_k_heads: int, num_v_heads: int) -> str | None:
    """Why ``fuse_conv`` cannot run with these head counts, or ``None``.

    One rule for ``is_valid_spec`` and dispatch's ``request_errors``, which
    re-exports it so the two cannot drift when the emitter learns grouped heads.
    """
    if num_k_heads == num_v_heads:
        return None
    return (
        "NOT_YET_IMPLEMENTED: fuse_conv requires num_k_heads == num_v_heads "
        "(the in-place shift of shared q/k conv channels would race), got "
        f"{num_k_heads}/{num_v_heads}"
    )


# Eps of the q/k L2 normalisation (compile-time). The fused output norm's eps
# is the separate runtime ``norm_eps`` argument.
NORM_EPS = 1e-6
EXP2_CLAMP = 126.0  # f32 exp2 argument range; keeps exp2_fast inside its contract
# State elements per vector access: 16 B in f16/bf16, 32 B in f32 (the backend
# splits that into two 16 B accesses).
STATE_VEC = 8
# Element size in bytes of every dtype this kernel stores; is_valid_spec bars
# any I/O or state dtype but these.
_ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4}
# Byte alignment the 16 B vector accesses assume of their base pointer. The
# params that carry them declare it; an f32 vec8 access (32 B payload) must
# not claim more than the param promises.
_VEC_ALIGN = 16
# The KDA ``dt_bias`` param declares no alignment, and its f32 vec8 loads have
# always claimed their 32 B payload size. The host checks it (see
# gdn_decode_pointer_alignment) so the claim is backed, and the emitted IR of
# every KDA kernel stays what it was.
_KDA_DT_BIAS_ALIGN = 32
# Causal conv1d width of the fused conv (fuse_conv): the current token plus
# CONV_WIDTH - 1 history taps per channel. Both target models use 4
# (ALGORITHM.md section 1.4).
CONV_WIDTH = 4
CONV_TAPS = CONV_WIDTH - 1
# Gate activation of the fused output RMSNorm, per gate kind. It is a property
# of the model each gate kind serves (ALGORITHM.md section 1.4): the GDN layer
# (Qwen3-Next) gates its RMSNorm with SiLU, the KDA layer (Kimi Linear) with
# sigmoid. Derived rather than a separate spec field so a spec cannot pair a
# gate kind with the other model's norm by omission; gate_kind is already in
# the kernel name, so the cache key stays faithful.
OUT_GATE_ACTIVATION = {"gdn": "silu", "kda": "sigmoid"}
# Cache policy of the recurrent-state loads and stores (the ``state_load_hint``
# / ``state_store_hint`` knobs). "default" is the policy every kernel had
# before the knobs existed; "streaming" marks the accesses LLVM
# ``!nontemporal``: the state is read once and written once per call, so
# caching it may gain nothing.
StateHint = Literal["default", "streaming"]
STATE_HINTS = get_args(StateHint)
_TEMPORAL = {"default": TemporalHint.DEFAULT, "streaming": TemporalHint.STREAMING}
# kernel_name() token of a non-default hint ("lhnt" / "shnt").
_HINT_TOKEN = {"streaming": "nt"}
# Highest amdgpu-waves-per-eu value: a gfx950 SIMD holds at most 8 waves.
_MAX_WAVES_PER_EU = 8


def _state_ir_type(state_dtype: str):
    return F32 if state_dtype == "f32" else io_ir_type(state_dtype)


def _load_f32_vec(
    b: IRBuilder,
    ptr,
    idx,
    n: int,
    *,
    align: int,
    temporal_hint: TemporalHint = TemporalHint.DEFAULT,
):
    """Load ``n`` f32 values as one vector access and extract each lane."""
    v = b.global_load_vN(ptr, idx, F32, n, align=align, temporal_hint=temporal_hint)
    return [b.vec_extract(v, i) for i in range(n)]


def _load_state_f32(b: IRBuilder, ptr, idx, *, state_dtype: str, n: int, hint: str):
    """Load ``n`` state elements as f32 values with cache policy ``hint``.

    The 16-bit states go through the same helper as before f32 existed, and
    ``hint="default"`` adds no metadata, so their emitted IR is unchanged.
    """
    if state_dtype == "f32":
        return _load_f32_vec(
            b, ptr, idx, n, align=_VEC_ALIGN, temporal_hint=_TEMPORAL[hint]
        )
    return load_vec_as_f32(
        b, ptr, idx, dtype=state_dtype, n=n, temporal_hint=_TEMPORAL[hint]
    )


def _pack_state(b: IRBuilder, values, *, state_dtype: str):
    """Pack f32 ``values`` into a state-dtype vector (no conversion for f32)."""
    if state_dtype == "f32":
        return b.vec_pack(values, F32)
    return pack_f32_to(b, values, dtype=state_dtype)


def _store_state(
    b: IRBuilder, ptr, idx, vec, *, state_dtype: str, n: int, hint: str
) -> None:
    """Store a packed state vector with cache policy ``hint``."""
    if state_dtype == "f32":
        b.global_store_vN(
            ptr, idx, vec, n, align=_VEC_ALIGN, temporal_hint=_TEMPORAL[hint]
        )
        return
    store_vec(b, ptr, idx, vec, n=n, temporal_hint=_TEMPORAL[hint])


# Where a tile-dependent knob changes emitted code. A knob that is on where its
# predicate is False emits the knob-off kernel, so kernel_name() adds its token
# only where the predicate holds: one name per distinct kernel. The fused-only
# knobs (conv_once, norm_gate_once, out_lds) need no predicate: is_valid_spec
# rejects them wherever they would be inert. The state hints and waves_per_eu
# change code on both emitters and every tile.
def dpp_reduce_applies(spec: "GdnDecodeSpec") -> bool:
    """``dpp_reduce``: the warp-tiled in-group sum has an xor-4 stage
    (``warp_threads_k >= 8``)."""
    return not spec.simple and spec.warp_threads_k >= 8


def xcd_remap_applies(spec: "GdnDecodeSpec") -> bool:
    """``xcd_remap``: the warp-tiled path (the reference path keeps the
    hardware workgroup order)."""
    return not spec.simple


def stream_rows_applies(spec: "GdnDecodeSpec") -> bool:
    """``stream_rows``: a warp-tiled lane owns more than one state row, so
    there is a row loop to stream."""
    return not spec.simple and spec.rows_per_lane > 1


def interleave_cols_applies(spec: "GdnDecodeSpec") -> bool:
    """``interleave_cols``: the warp-tiled path with an f32 state and more
    than one K lane. A 2-byte state already moves one 16 B run per lane, and a
    single K lane's runs are already contiguous."""
    return not spec.simple and spec.state_dtype == "f32" and spec.warp_threads_k > 1


def late_row_inputs(spec: "GdnDecodeSpec") -> bool:
    """Whether the warp-tiled emitter loads each row's late inputs where the
    parent order did instead of first (ALGORITHM.md §4.3, "One exception"):
    each row's ``v`` conv inputs in the row loop, each state chunk at its
    decay, and the per-lane ``out_gate`` / ``norm_weight`` at the norm.

    Only a single wave with ``fuse_conv`` takes the exception. It owns every
    V row of the head, and hoisting their ``v`` conv inputs together with the
    state exceeds the VGPR budget. ``conv_once`` reads ``v`` from LDS, which
    leaves only the state and the norm inputs to hoist; at ``DK = DV = 128``
    with ``WTK`` 4 or 8 and no ``waves_per_eu`` floor hoisting them spills no
    more than the late order (test_gdn_decode_fused.py pins it), so those
    tiles load first. Every other single-wave ``fuse_conv`` tile keeps the
    late order: at the other ``WTK``, at ``DK = DV = 256``, under a
    ``waves_per_eu`` floor and at the other head dims no test shows hoisting
    spill-free. That set (``DK == DV == 128``, ``WTK`` 4 or 8, no
    ``waves_per_eu`` floor) is the tested spill envelope, not a workload
    choice: widen it only together with the spill test.
    """
    if not (spec.fuse_conv and spec.num_warps == 1):
        return False
    hoists = (
        spec.conv_once
        and spec.waves_per_eu == 0
        and spec.head_k_dim == spec.head_v_dim == 128
        and spec.warp_threads_k in (4, 8)
    )
    return not hoists


def _vec_param(b: IRBuilder, name: str, ty, **access):
    """A global pointer param whose base the 16 B vector accesses rely on.
    Attr order is noalias, access (readonly / writeonly), align."""
    return b.param(
        name, PtrType(ty, "global"), noalias=True, **access, align=_VEC_ALIGN
    )


def _gate_and_index_params(b: IRBuilder, spec: "GdnDecodeSpec", io_ty):
    """The gate inputs and the pool-index params, shared by both emitters and
    declared between q/k/v and the state."""

    def ro(name, ty):
        return b.param(name, PtrType(ty, "global"), noalias=True, readonly=True)

    Ag = ro("a", io_ty)
    Bg = ro("b", io_ty)
    # f32 in KDA mode: the per-channel gate is evaluated in f32 and KDA prefill
    # declares the same tensor f32, so the two families share one contract.
    DTB = ro("dt_bias", F32 if spec.gate_kind == "kda" else io_ty)
    ALOG = ro("A_log", F32)
    RIDX = ro("read_indices", I32)
    WIDX = ro("write_indices", I32)
    return Ag, Bg, DTB, ALOG, RIDX, WIDX


def _pool_slots(b: IRBuilder, RIDX, WIDX, b_i):
    """Read and write pool slots of sequence ``b_i`` and the predicate that
    skips continuous-batching padding (negative sentinel index)."""
    read_pool = b.global_load_i32(RIDX, b_i)
    write_pool = b.global_load_i32(WIDX, b_i)
    active = b.land(
        b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
    )
    return read_pool, write_pool, active


def _slot_ptr(b: IRBuilder, base, pool, slot_bytes: int):
    """Pointer to pool slot ``pool``. The slot base overflows i32 for large
    pools, so the pointer advances by a 64-bit byte offset and in-slot
    indices stay i32."""
    return b.global_ptr_add(base, b.mul(b.sext(pool, I64), b.const_i64(slot_bytes)))


def _head_base(b: IRBuilder, b_i, h_i, heads: int, dim: int):
    """Element offset of head ``h_i`` of token ``b_i`` in a contiguous
    ``[B, 1, heads, dim]`` tensor."""
    return b.add(b.mul(b_i, b.const_i32(heads * dim)), b.mul(h_i, b.const_i32(dim)))


def _state_row(b: IRBuilder, hv_i, row, spec: "GdnDecodeSpec"):
    """In-slot element offset of state row ``row`` of value head ``hv_i``
    (state layout ``[slot, HV, DV, DK]``)."""
    DK = spec.head_k_dim
    return b.add(
        b.mul(hv_i, b.const_i32(spec.head_v_dim * DK)), b.mul(row, b.const_i32(DK))
    )


def _exp_f32(b: IRBuilder, x):
    """``exp(x)`` composed from the hardware exp2, with a clamped argument.

    exp2_fast emits no range guard, so its contract is that the *caller*
    bounds the argument. Every other exp2_fast in this repo is a softmax,
    whose argument is <= 0 by construction; a gate argument is not -- a,
    dt_bias, A_log and b come straight from caller tensors and nothing
    bounds them. We meet the contract explicitly instead, with the same
    fmin/fmax clamp the KDA emitter uses (``ex2``, kda_chunkwise.py).

    The two lowerings differ only in the f32 denormal window: unclamped
    exp2_fast and the guarded b.exp2 agree on the whole positive range
    (both saturate to +inf past ~88), and where they part, exp2_fast
    flushes to 0 while exp2 returns the denormal. A denormal cannot
    survive rounding into a 16-bit state, and in an f32 state it is far
    below any tolerance the result is held to, so this clamp buys contract
    compliance rather than accuracy -- which is why it is a 2-op clamp
    and not the ~5-op guarded lowering.
    """
    arg = b.fmul(x, b.const_f32(LOG2E))
    arg = b.fmin(b.fmax(arg, b.const_f32(-EXP2_CLAMP)), b.const_f32(EXP2_CLAMP))
    return b.exp2_fast(arg)


def _sigmoid(b: IRBuilder, x):
    """``1 / (1 + exp(-x))`` through the clamped :func:`_exp_f32`."""
    return b.rcp_fast(b.fadd(b.const_f32(1.0), _exp_f32(b, b.fneg(x))))


def _silu(b: IRBuilder, x):
    """``x * sigmoid(x)``."""
    return b.fmul(x, _sigmoid(b, x))


def _log1p(b: IRBuilder, x):
    """``ln(1 + x)`` from the hardware log2."""
    return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))


def _gdn_gate_inputs(
    b: IRBuilder, spec: "GdnDecodeSpec", Ag, Bg, DTB, ALOG, a_idx, hv_i
):
    """The GDN gate's loads as f32: ``(a, b, dt_bias, A_log)``."""
    ra = load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype)
    rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
    rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
    ral = b.global_load_f32(ALOG, hv_i)  # A_log is fp32
    return ra, rb, rdt, ral


def _gdn_gate_math(b: IRBuilder, ra, rb, rdt, ral):
    """GDN gate of one value head from its loaded inputs: ``(decay, beta)``,
    both scalars. decay = exp(-exp(A_log) * softplus(a + dt_bias)),
    beta = sigmoid(b)."""
    x = b.fadd(ra, rdt)
    sp = b.select(
        b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
        x,
        _log1p(b, _exp_f32(b, x)),
    )
    decay = _exp_f32(b, b.fneg(b.fmul(_exp_f32(b, ral), sp)))
    beta = _sigmoid(b, rb)
    return decay, beta


def _gdn_gate(b: IRBuilder, spec: "GdnDecodeSpec", Ag, Bg, DTB, ALOG, a_idx, hv_i):
    """GDN gate of one value head, its loads right before its math: the
    reference path's order. The warp-tiled emitter calls the two halves apart,
    loads first (ALGORITHM.md §4.3)."""
    return _gdn_gate_math(b, *_gdn_gate_inputs(b, spec, Ag, Bg, DTB, ALOG, a_idx, hv_i))


def _kda_head_inputs(b: IRBuilder, spec: "GdnDecodeSpec", Bg, ALOG, a_idx, hv_i):
    """The per-head KDA gate loads as f32: ``(b, A_log)``."""
    rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
    ral = b.global_load_f32(ALOG, hv_i)  # per head in both gate kinds
    return rb, ral


def _kda_head_math(b: IRBuilder, rb, ral):
    """``(beta, exp(A_log))``. exp(A_log[h]) is per head, so it is hoisted
    out of the channel loop."""
    return _sigmoid(b, rb), _exp_f32(b, ral)


def _kda_gate_rows(b: IRBuilder, spec: "GdnDecodeSpec", b_i, hv_i):
    """Offsets of this head's DK-channel gate and dt_bias rows."""
    DK = spec.head_k_dim
    g_row = _head_base(b, b_i, hv_i, spec.num_v_heads, DK)
    dtb_row = b.mul(hv_i, b.const_i32(DK))
    return g_row, dtb_row


def _kda_gate_head(b: IRBuilder, spec: "GdnDecodeSpec", Bg, ALOG, a_idx, b_i, hv_i):
    """Per-head part of the KDA gate: ``(beta, exp(A_log), g_row, dtb_row)``,
    loads, math, then the row offsets: the reference path's order."""
    rb, ral = _kda_head_inputs(b, spec, Bg, ALOG, a_idx, hv_i)
    beta, exp_alog = _kda_head_math(b, rb, ral)
    return (beta, exp_alog, *_kda_gate_rows(b, spec, b_i, hv_i))


def _kda_channel_decay(b: IRBuilder, spec: "GdnDecodeSpec", exp_alog, g, dt):
    """Decay of one K channel:
    ``exp(lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias)))``.

    ``dt`` is a thunk returning the channel's dt_bias, so a caller that
    extracts it lazily keeps its op order. With ``fuse_gate=False``, ``g``
    already carries the natural-log-domain decay: only the exponential
    remains and ``dt`` is never called.
    """
    if spec.fuse_gate:
        inner = b.fmul(exp_alog, b.fadd(g, dt()))
        sig = _sigmoid(b, inner)
        log_decay = b.fmul(b.const_f32(spec.lower_bound), sig)
    else:
        log_decay = g
    return _exp_f32(b, log_decay)


def _kda_head_decay(
    b: IRBuilder, spec: "GdnDecodeSpec", Ag, DTB, exp_alog, g_row, dtb_row
):
    """The reference path's KDA decay: one per K channel over the full DK
    extent (each thread owns a whole state row), loaded and computed one
    STATE_VEC chunk at a time."""
    decay = []
    for c in range(0, spec.head_k_dim, STATE_VEC):
        gv = load_vec_as_f32(
            b, Ag, b.add(g_row, b.const_i32(c)), dtype=spec.dtype, n=STATE_VEC
        )
        # dt_bias is f32, so load_vec_as_f32 (a 16-bit ingest helper) does
        # not apply: one f32 vector load instead of STATE_VEC scalar ones.
        dtv = _load_f32_vec(
            b, DTB, b.add(dtb_row, b.const_i32(c)), STATE_VEC, align=_KDA_DT_BIAS_ALIGN
        )
        decay += [
            _kda_channel_decay(b, spec, exp_alog, g, lambda d=d: d)
            for g, d in zip(gv, dtv)
        ]
    return decay


def _dot(b: IRBuilder, xs, ys, reduce):
    """``reduce`` of the element-wise products ``xs[i] * ys[i]``."""
    return reduce([b.fmul(x, y) for x, y in zip(xs, ys)])


def _scale_qk(b: IRBuilder, qv, kv, *, use_l2norm: bool, scale: float, reduce):
    """Scale q by ``1/sqrt(DK)``, after an L2 norm of q and k when
    ``use_l2norm``. ``qv``/``kv`` are flat lists of this thread's channels;
    ``reduce`` sums a list across every channel of the head."""
    if not use_l2norm:
        return [b.fmul(x, b.const_f32(scale)) for x in qv], kv
    sum_q2 = _dot(b, qv, qv, reduce)
    sum_k2 = _dot(b, kv, kv, reduce)
    inv_q = b.rsqrt(b.fadd(sum_q2, b.const_f32(NORM_EPS)))
    inv_k = b.rsqrt(b.fadd(sum_k2, b.const_f32(NORM_EPS)))
    sq = b.fmul(inv_q, b.const_f32(scale))
    return [b.fmul(x, sq) for x in qv], [b.fmul(x, inv_k) for x in kv]


def _forget(b: IRBuilder, spec: "GdnDecodeSpec", state, decay):
    """Gated forget of state values. A GDN decay is one scalar that
    broadcasts; a KDA decay is a list zipped with ``state`` (both indexed by
    K channel, in the same order)."""
    if spec.gate_kind == "gdn":
        return [b.fmul(s, decay) for s in state]
    return [b.fmul(s, d) for s, d in zip(state, decay)]


def _norm_gate(b: IRBuilder, o, rstd, *factors):
    """One output row of the gated RMSNorm: ``o * rstd``, then times each of
    ``factors`` in order -- ``(norm_weight, act(out_gate))`` per lane, or the
    one ``act(out_gate) * norm_weight`` product of ``norm_gate_once``. Every
    store of the norm's output calls it, so the knob-off ``out_lds`` store and
    the per-lane store round alike."""
    y = b.fmul(o, rstd)
    for f in factors:
        y = b.fmul(y, f)
    return y


@dataclass(frozen=True)
class GdnDecodeSpec:
    """One GDN single-token decode instance."""

    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    dtype: DType = "bf16"
    state_dtype: StateDType = "bf16"
    use_qk_l2norm: bool = True
    # Forget-gate granularity. "gdn" applies one scalar decay per head; "kda"
    # applies a per-channel DK-vector decay. GDN is the special case of KDA in
    # which every channel shares a value, so the general kernel serves both --
    # but only the general one can express the vector, which is why this is a
    # kernel field and not a dispatch detail.
    gate_kind: Literal["gdn", "kda"] = "gdn"
    # KDA gate lower bound: log-decay = lower_bound * sigmoid(...), so the gate
    # is bounded in (lower_bound, 0). Unread when gate_kind == "gdn", whose
    # softplus gate is unbounded below.
    lower_bound: float = -5.0
    # True: the kernel computes the decay from raw logits (the shipping path,
    # one launch). False: `a` carries a precomputed NATURAL-LOG-domain decay and
    # the kernel only exponentiates and multiplies. The False mode is never
    # dispatched; it exists so this kernel can be timed against a competitor
    # recurrence-only kernel at an identical work boundary.
    fuse_gate: bool = True
    wave_size: int = 64
    # GDN's dispatcher default uses these values whenever they are legal.
    # Direct callers still get a valid general-purpose configuration.
    num_warps: int = 2
    warp_threads_k: int = 16
    blocks_per_v_dim: int = (
        8  # split a head's V-dim across this many CTAs (small-B fill)
    )
    simple: bool = False  # True => v1 one-thread-per-row reference path
    # Optional fusions of a hybrid layer's neighbours (ALGORITHM.md §4.9).
    # Both need one workgroup per head (blocks_per_v_dim == 1); fuse_conv
    # also needs Hk == Hv, since shared q/k conv channels would race. Both
    # change the kernel ABI, so is_valid_spec accepts only a real bool.
    fuse_conv: bool = False  # causal conv1d + SiLU on q/k/v, in-place tap shift
    fuse_out_norm: bool = (
        False  # o * rsqrt(mean(o^2)+norm_eps) * norm_weight * act(gate)
    )
    # Performance knobs (ALGORITHM.md §4.10). Every default is the code the
    # kernel emitted before the knob existed, so a knob-off spec keeps its name
    # and its IR byte for byte.
    #
    # Cache policy of the state loads / stores ("streaming" = !nontemporal).
    state_load_hint: StateHint = "default"
    state_store_hint: StateHint = "default"
    # In-group sum (wsum): run the xor-4 / xor-8 stages on tile.dpp_xor with
    # mask 7 / 15 (DPP row_half_mirror / row_mirror on gfx9 CDNA) instead of
    # ds_swizzle. The sum is the same: the stages run in ascending order, so
    # when one runs every lane of a quad (half-row) already holds the quad
    # (half-row) sum. Stages above 8 keep ds_swizzle.
    dpp_reduce: bool = False
    # Renumber workgroups so each of the 8 XCDs (which receive workgroups
    # round-robin, pid % 8) runs one contiguous range of (sequence, head,
    # v-block) work instead of every 8th one. A bijection for any grid size.
    # It derives the grid size from the batch_size argument (unread
    # otherwise), so the launch grid must be gdn_decode_grid(batch_size, spec),
    # as the builder's prepare() makes it.
    xcd_remap: bool = False
    # Consume the state row by row: each owned row's decay, dot products,
    # update and store run in its own iteration, so the first math waits only
    # for the rows loaded before it. Off, every row is decayed up front.
    stream_rows: bool = False
    # f32 state only: map K columns to lanes in 16 B runs interleaved across
    # the WTK-lane group (run p of lane l at chunk column p * WTK * 4 + l * 4)
    # instead of VPT consecutive columns per lane, so one 16 B state access
    # across the group covers one contiguous WTK * 16 B span. Every
    # per-column operand (q, k, the KDA decays, the conv taps and weights)
    # follows the same map.
    interleave_cols: bool = False
    # fuse_conv only: compute each conv channel (and each KDA decay channel)
    # once per workgroup, one thread per channel, and hand the results to the
    # lanes through LDS. Off, every lane recomputes the channels it consumes.
    conv_once: bool = False
    # fuse_out_norm only (and, not yet implemented without it, conv_once,
    # whose LDS block holds the products): compute act(out_gate) *
    # norm_weight once per output row into LDS, instead of every lane loading
    # its rows' gate and weight.
    norm_gate_once: bool = False
    # fuse_out_norm only: each row's pre-norm output goes to LDS ahead of the
    # norm's cross-wave barrier; after it, the first DV / OUT_W lanes each
    # normalize OUT_W consecutive rows and store them with one packed store,
    # instead of one 2-byte store per row from its k-lane-0 owner.
    out_lds: bool = False
    # Minimum waves per SIMD the compiler must fit (amdgpu-waves-per-eu "N,8");
    # it caps registers per lane accordingly. 0 leaves occupancy to the
    # compiler and emits no attribute.
    waves_per_eu: int = 0
    name: str = "rocke_gdn_decode"

    @property
    def block_size(self) -> int:
        return self.head_v_dim if self.simple else self.num_warps * self.wave_size

    @property
    def v_per_k_head(self) -> int:
        return self.num_v_heads // self.num_k_heads

    @property
    def conv_dim(self) -> int:
        """Channels of the packed ``[q | k | v]`` row that ``fuse_conv`` reads."""
        return (
            2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        )

    @property
    def out_gate_activation(self) -> str:
        """``"silu"`` or ``"sigmoid"``: the fused output norm's gate activation."""
        return OUT_GATE_ACTIVATION[self.gate_kind]

    @property
    def rows_per_lane(self) -> int:
        """State rows each lane owns in the warp-tiled body (WTV_ITERS)."""
        lanes_v = (self.wave_size // self.warp_threads_k) * self.num_warps
        return (self.head_v_dim // self.blocks_per_v_dim) // lanes_v

    # One definition of "tile knob X changes code": X and X_applies. Both
    # kernel_name and the emitter read these, so the cache key follows the IR.
    @property
    def dpp_reduce_on(self) -> bool:
        return self.dpp_reduce and dpp_reduce_applies(self)

    @property
    def xcd_remap_on(self) -> bool:
        return self.xcd_remap and xcd_remap_applies(self)

    @property
    def stream_rows_on(self) -> bool:
        return self.stream_rows and stream_rows_applies(self)

    @property
    def interleave_cols_on(self) -> bool:
        return self.interleave_cols and interleave_cols_applies(self)

    def kernel_name(self) -> str:
        # Every field that changes emitted code MUST appear here: this name is the
        # compile/launcher cache key, so two specs sharing a name means one of them
        # silently runs the other's kernel. Fields whose value equals the default
        # are folded into the deviation-only suffixes below to keep names stable.
        parts = (
            self.dtype,
            f"kh{self.num_k_heads}",
            f"vh{self.num_v_heads}",
            f"dk{self.head_k_dim}",
            f"dv{self.head_v_dim}",
            f"w{self.num_warps}k{self.warp_threads_k}b{self.blocks_per_v_dim}",
        )
        if self.state_dtype != self.dtype:
            parts += (f"st{self.state_dtype}",)
        # Deviation-only, so the KDA gate kind is additive: a default-gate spec
        # keeps the exact name it had before this field existed, and every
        # pinned GDN golden hash stays valid. lower_bound is nested because the
        # GDN gate never reads it -- letting it reach the name there would give
        # two names to two byte-identical kernels.
        if self.gate_kind != "gdn":
            parts += (self.gate_kind,)
            if self.fuse_gate:
                if self.lower_bound != -5.0:
                    parts += (f"lb{self.lower_bound:g}",)
            else:
                parts += ("nofg",)
        if self.wave_size != 64:
            parts += (f"ws{self.wave_size}",)
        # The fusion flags and the knobs are deviation-only too: off (or
        # inert, per their *_applies predicate), they add nothing, so every
        # name recorded before they existed is unchanged.
        if self.state_load_hint != "default":
            parts += (
                f"lh{_HINT_TOKEN.get(self.state_load_hint, self.state_load_hint)}",
            )
        if self.state_store_hint != "default":
            parts += (
                f"sh{_HINT_TOKEN.get(self.state_store_hint, self.state_store_hint)}",
            )
        if self.waves_per_eu:
            parts += (f"wpe{self.waves_per_eu}",)
        return kernel_name_join(
            self.name,
            *parts,
            flags={
                "l2": self.use_qk_l2norm,
                "s": self.simple,
                "cv": self.fuse_conv,
                "rn": self.fuse_out_norm,
                "dppr": self.dpp_reduce_on,
                "xcd": self.xcd_remap_on,
                "sr": self.stream_rows_on,
                "ic": self.interleave_cols_on,
                "co": self.conv_once,
                "ngo": self.norm_gate_once,
                "olds": self.out_lds,
            },
        )


def _field_type_error(spec: GdnDecodeSpec) -> Optional[str]:
    """The first flag, integer or real field of the wrong type, or None.

    Every other rule branches on these fields' truth or arithmetic, so a
    wrong type would be coerced into a DIFFERENT valid spec
    (``fuse_conv="false"`` is truthy and swaps in the fused ABI) rather than
    rejected.
    """
    for _field in (
        "use_qk_l2norm",
        "fuse_gate",
        "simple",
        "fuse_conv",
        "fuse_out_norm",
        "dpp_reduce",
        "xcd_remap",
        "stream_rows",
        "interleave_cols",
        "conv_once",
        "norm_gate_once",
        "out_lds",
    ):
        _value = getattr(spec, _field)
        if type(_value) is not bool:
            return f"{_field} must be a bool, got {_value!r}"
    for _field in (
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "wave_size",
        "num_warps",
        "warp_threads_k",
        "blocks_per_v_dim",
        "waves_per_eu",
    ):
        _value = getattr(spec, _field)
        if isinstance(_value, bool) or not isinstance(_value, numbers.Integral):
            return f"{_field} must be an integer, got {_value!r}"
    if isinstance(spec.lower_bound, bool) or not isinstance(
        spec.lower_bound, numbers.Real
    ):
        return f"lower_bound must be a real number, got {spec.lower_bound!r}"
    return None


def _knob_error(spec: GdnDecodeSpec) -> Optional[str]:
    """The first performance-knob rule ``spec`` breaks, or None.

    Each rule is functional -- a value the emitter cannot build, or a knob
    that reads a tensor or result its spec does not have -- except the one
    marked NOT_YET_IMPLEMENTED. A tile-dependent knob that is merely inert on
    this tile is legal and emits the knob-off kernel (see the *_applies
    predicates). The knob flags are type-checked by :func:`_field_type_error`.
    """
    for _field in ("state_load_hint", "state_store_hint"):
        _value = getattr(spec, _field)
        if _value not in STATE_HINTS:
            return f"{_field} must be one of {STATE_HINTS}, got {_value!r}"
    if not 0 <= spec.waves_per_eu <= _MAX_WAVES_PER_EU:
        return (
            f"waves_per_eu must be in [0, {_MAX_WAVES_PER_EU}], "
            f"got {spec.waves_per_eu!r}"
        )
    if spec.conv_once and not spec.fuse_conv:
        return "conv_once requires fuse_conv"
    if spec.norm_gate_once and not spec.fuse_out_norm:
        return "norm_gate_once requires fuse_out_norm"
    if spec.out_lds and not spec.fuse_out_norm:
        return "out_lds requires fuse_out_norm"
    # An emitter coupling, not a functional limit (ALGORITHM.md §8, follow-up
    # 8): the products live in conv_once's LDS block, behind its barrier.
    if spec.norm_gate_once and not spec.conv_once:
        return (
            "NOT_YET_IMPLEMENTED: norm_gate_once requires conv_once (it shares "
            "conv_once's LDS block)"
        )
    return None


def is_valid_spec(spec: GdnDecodeSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Reject impossible/unsupported GDN decode configs before IR is built."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    # Type checks first (see _field_type_error).
    why = _field_type_error(spec)
    if why:
        return False, why

    if spec.wave_size != target.wave_size:
        # The wave_size rules further down check INTERNAL consistency
        # (wave_size % warp_threads_k). This one checks agreement with the
        # hardware: the lane layout and the xor butterfly both take the wave
        # width as given, so a wave32 target built with a wave64 spec emits IR
        # whose cross-lane arithmetic is simply wrong -- and silently, since
        # nothing downstream re-derives it.
        return (
            False,
            f"spec.wave_size {spec.wave_size} != {arch} wave size "
            f"{target.wave_size}",
        )
    if spec.dtype not in GDN_DTYPES or spec.state_dtype not in STATE_DTYPES:
        return False, f"unsupported dtype {spec.dtype}/{spec.state_dtype}"
    if spec.gate_kind not in ("gdn", "kda"):
        return False, f"gate_kind must be 'gdn' or 'kda' (got {spec.gate_kind!r})"
    if spec.gate_kind == "gdn" and not spec.fuse_gate:
        return False, "gate_kind='gdn' requires fuse_gate=True"
    why = _knob_error(spec)
    if why:
        return False, why
    # The three fusion rules below are limits of this emitter's in-place,
    # one-workgroup-per-head design, not of the hardware (ALGORITHM.md §4.9,
    # follow-up 7), hence the family's NOT_YET_IMPLEMENTED marker.
    if spec.fuse_conv or spec.fuse_out_norm:
        if spec.simple:
            return False, (
                "NOT_YET_IMPLEMENTED: fuse_conv / fuse_out_norm require the "
                "warp-tiled path (simple=False)"
            )
        if spec.blocks_per_v_dim != 1:
            return False, (
                "NOT_YET_IMPLEMENTED: fuse_conv / fuse_out_norm require "
                "blocks_per_v_dim == 1 (one workgroup per head), got "
                f"{spec.blocks_per_v_dim}"
            )
    if spec.fuse_conv:
        why = fuse_conv_heads_error(spec.num_k_heads, spec.num_v_heads)
        if why is not None:
            return False, why
    if (
        spec.gate_kind == "kda"
        and spec.fuse_gate
        and (not math.isfinite(spec.lower_bound) or spec.lower_bound >= 0.0)
    ):
        return False, (
            "lower_bound must be finite negative for the fused KDA gate, "
            f"got {spec.lower_bound}"
        )
    for _field, _value in (
        ("num_k_heads", spec.num_k_heads),
        ("num_v_heads", spec.num_v_heads),
        ("head_k_dim", spec.head_k_dim),
        ("head_v_dim", spec.head_v_dim),
    ):
        if _value <= 0:
            return False, f"{_field} must be positive, got {_value}"
    if spec.num_v_heads % spec.num_k_heads:
        return False, "num_v_heads must be divisible by num_k_heads"
    if spec.head_k_dim % STATE_VEC or spec.head_v_dim % STATE_VEC:
        return False, f"head dims must be multiples of {STATE_VEC}"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    if not spec.simple:
        for _field, _value in (
            ("num_warps", spec.num_warps),
            ("warp_threads_k", spec.warp_threads_k),
            ("blocks_per_v_dim", spec.blocks_per_v_dim),
            ("wave_size", spec.wave_size),
        ):
            if _value <= 0:
                return False, f"{_field} must be positive, got {_value}"
        if spec.wave_size % spec.warp_threads_k:
            return False, "wave_size must be divisible by warp_threads_k"
        vpt = STATE_VEC
        warp_tile_k = spec.warp_threads_k * vpt
        wgroup_v = spec.num_warps * (spec.wave_size // spec.warp_threads_k)
        if spec.head_k_dim % warp_tile_k:
            return False, f"head_k_dim must be a multiple of {warp_tile_k}"
        if spec.head_v_dim % spec.blocks_per_v_dim:
            return False, "head_v_dim must be a multiple of blocks_per_v_dim"
        tile_v = spec.head_v_dim // spec.blocks_per_v_dim
        if tile_v % wgroup_v:
            return (
                False,
                f"head_v_dim/blocks_per_v_dim must be a multiple of {wgroup_v}",
            )
    return True, ""


def build_gdn_decode(spec: GdnDecodeSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one GDN single-token decode instance (dispatch)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gdn_decode spec for {arch}: {why}")
    return _build_simple(spec) if spec.simple else _build_warp_tiled(spec)


def _build_simple(spec: GdnDecodeSpec) -> KernelDef:
    """v1 reference: one thread per value-dim (state row); no cross-thread reduce."""

    HK, HV = spec.num_k_heads, spec.num_v_heads
    DK, DV = spec.head_k_dim, spec.head_v_dim
    S_SLOT = HV * DV * DK * _ELEM_BYTES[spec.state_dtype]  # bytes per pool slot

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, _MAX_WAVES_PER_EU)
    reduce = partial(tree_reduce, b, b.fadd)

    io_ty = io_ir_type(spec.dtype)
    Q = _vec_param(b, "query", io_ty, readonly=True)
    K = _vec_param(b, "key", io_ty, readonly=True)
    Vv = _vec_param(b, "value", io_ty, readonly=True)
    Ag, Bg, DTB, ALOG, RIDX, WIDX = _gate_and_index_params(b, spec, io_ty)
    STATE = _vec_param(b, "state", _state_ir_type(spec.state_dtype))
    OUT = _vec_param(b, "out", io_ty, writeonly=True)
    _ = b.param("batch_size", I32)  # noqa: F841

    tid = b.thread_id_x()  # value-dim row this thread owns (0..DV-1)
    bidx = b.block_id_x()
    b_i = b.div(bidx, b.const_i32(HV))
    hv_i = b.mod(bidx, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(spec.v_per_k_head))

    read_pool, write_pool, active = _pool_slots(b, RIDX, WIDX, b_i)
    with b.scf_if(active):
        # ---- load q,k rows [DK] -> f32 registers ----
        q_base = _head_base(b, b_i, hk_i, HK, DK)
        qv, kv = [], []
        for c in range(0, DK, STATE_VEC):
            off = b.add(q_base, b.const_i32(c))
            qv += load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=STATE_VEC)
            kv += load_vec_as_f32(b, K, off, dtype=spec.dtype, n=STATE_VEC)
        qn, kn = _scale_qk(
            b,
            qv,
            kv,
            use_l2norm=spec.use_qk_l2norm,
            scale=1.0 / math.sqrt(DK),
            reduce=reduce,
        )

        # ---- gates (per value head) ----
        a_idx = b.add(b.mul(b_i, b.const_i32(HV)), hv_i)
        if spec.gate_kind == "gdn":
            decay, beta = _gdn_gate(b, spec, Ag, Bg, DTB, ALOG, a_idx, hv_i)
        else:
            # KDA: one decay per K channel.
            beta, exp_alog, g_row, dtb_row = _kda_gate_head(
                b, spec, Bg, ALOG, a_idx, b_i, hv_i
            )
            decay = _kda_head_decay(b, spec, Ag, DTB, exp_alog, g_row, dtb_row)

        # ---- dot(k_hat, q_hat) (scalar, redundant per thread) ----
        dot_kq = _dot(b, kn, qn, reduce)

        # ---- load state row t=tid: state[read_pool, hv, tid, 0:DK] ----
        state_r = _slot_ptr(b, STATE, read_pool, S_SLOT)
        rs_base = _state_row(b, hv_i, tid, spec)
        sv = []
        for c in range(0, DK, STATE_VEC):
            off = b.add(rs_base, b.const_i32(c))
            sv += _load_state_f32(
                b,
                state_r,
                off,
                state_dtype=spec.state_dtype,
                n=STATE_VEC,
                hint=spec.state_load_hint,
            )
        sv = _forget(b, spec, sv, decay)

        # ---- S_row . k_hat  and  S_row . q_hat ----
        sum_hk = _dot(b, sv, kn, reduce)
        sum_hq = _dot(b, sv, qn, reduce)

        # ---- delta value + read-out for this value-dim ----
        v_idx = b.add(_head_base(b, b_i, hv_i, HV, DV), tid)
        rv = load_scalar_as_f32(b, Vv, v_idx, dtype=spec.dtype)
        v_new = b.fmul(b.fsub(rv, sum_hk), beta)
        out_val = b.fadd(sum_hq, b.fmul(v_new, dot_kq))
        store_scalar_from_f32(b, OUT, v_idx, out_val, dtype=spec.dtype)

        # ---- rank-1 state write: S_row += k_hat * v_new ----
        state_w = _slot_ptr(b, STATE, write_pool, S_SLOT)
        ws_base = _state_row(b, hv_i, tid, spec)
        new_s = [b.fma(k, v_new, s) for k, s in zip(kn, sv)]
        for c in range(0, DK, STATE_VEC):
            vec = _pack_state(b, new_s[c : c + STATE_VEC], state_dtype=spec.state_dtype)
            off = b.add(ws_base, b.const_i32(c))
            _store_state(
                b,
                state_w,
                off,
                vec,
                state_dtype=spec.state_dtype,
                n=STATE_VEC,
                hint=spec.state_store_hint,
            )

    return b.kernel


def _xcd_remap(b: IRBuilder, pid, npid):
    """Logical workgroup id for hardware id ``pid`` of ``npid`` (xcd_remap).

    The hardware hands workgroup ``pid`` to XCD ``x = pid % 8``. XCD ``x``
    gets the contiguous range starting at ``x * ceil(npid / 8)``, minus one
    slot for each earlier XCD that received one fewer workgroup when ``npid``
    is not a multiple of 8. A bijection on ``[0, npid)``.
    """
    # Not helpers.grid.chiplet_transform_chunked: that maps fixed-size chunks
    # and leaves a tail at identity; this gives each XCD one ceil(npid / 8)
    # range, so no workgroup keeps its hardware order.
    n = b.const_i32(NUM_XCDS_MI350X)
    xcd = b.mod(pid, n)
    tall = b.add(b.mod(b.sub(npid, b.const_i32(1)), n), b.const_i32(1))
    per = b.div(b.add(npid, b.const_i32(NUM_XCDS_MI350X - 1)), n)
    short = b.smax(b.sub(xcd, tall), b.const_i32(0))
    return b.sub(b.add(b.div(pid, n), b.mul(xcd, per)), short)


@dataclass(frozen=True)
class _Tile:
    """Geometry of the warp-tiled body, in plain ints (no IR).

    A workgroup of NW waves covers TILE_V of a head's DV rows. A wave is WTV
    V lanes x WTK K lanes; a lane owns WTV_ITERS V rows (WGROUP_V apart) and,
    of each, WTK_ITERS K chunks of VPT channels (WARP_TILE_K apart).

    K-column map: each lane owns K_PIECES runs of K_RUN consecutive columns
    per K chunk; run p of lane l starts at chunk column p * WTK * K_RUN +
    l * K_RUN, so slot i of a lane's chunk is always column
    ``k_offs[ki][i // K_RUN] + i % K_RUN``, for the state and every operand.
    Default: one run of VPT. interleave_cols: 16 B runs of the f32 state.
    """

    HK: int
    HV: int
    DK: int
    DV: int
    WAVE: int
    WTK: int
    WTV: int
    NW: int
    VPT: int
    BS: int
    WARP_TILE_K: int
    WTK_ITERS: int
    WGROUP_V: int
    BPV: int
    TILE_V: int
    WTV_ITERS: int
    K_RUN: int
    K_PIECES: int
    S_SLOT: int  # bytes per state pool slot
    CST_SLOT: int  # bytes per conv-state pool slot; it carries the I/O dtype
    V_CHAN0: int  # first v channel of the packed [q | k | v] row: 2 * Hk * DK
    LATE_ROW_INPUTS: bool  # late_row_inputs(spec)
    # the per-lane norm: neither out_lds nor norm_gate_once supplies the gate
    # and weight
    LANE_NORM: bool
    # out_lds: OUT_W consecutive output rows per storing lane (2, 4 or 8, so
    # the DV / OUT_W storing lanes fit one wave up to DV = 8 * WAVE), in
    # OUT_ROUNDS passes over the workgroup.
    OUT_W: int
    N_OUT: int
    OUT_ROUNDS: int
    # conv_once's LDS block: the post-SiLU f32 conv outputs [q DK | k DK |
    # v DV] (N_CONV; the segments start at slots Q0, K0 and V0), then (KDA)
    # the DK per-channel decays (N_DEC), then (norm_gate_once) from GW0 the DV
    # per-row products gate(out_gate) * norm_weight (N_GW); one slot per
    # channel, written once, read by all.
    Q0: int
    K0: int
    V0: int
    N_CONV: int
    N_DEC: int
    N_GW: int
    GW0: int
    N_ONCE: int

    @classmethod
    def of(cls, spec: GdnDecodeSpec) -> "_Tile":
        WAVE, WTK, NW = spec.wave_size, spec.warp_threads_k, spec.num_warps
        DK, DV = spec.head_k_dim, spec.head_v_dim
        WARP_TILE_K = WTK * STATE_VEC
        K_RUN = STATE_VEC
        if spec.interleave_cols_on:
            K_RUN = _VEC_ALIGN // _ELEM_BYTES[spec.state_dtype]
        OUT_W = 2 if DV <= 2 * WAVE else 4 if DV <= 4 * WAVE else 8
        N_CONV = 2 * DK + DV
        N_DEC = DK if spec.gate_kind == "kda" else 0
        N_GW = DV if spec.norm_gate_once else 0
        return cls(
            HK=spec.num_k_heads,
            HV=spec.num_v_heads,
            DK=DK,
            DV=DV,
            WAVE=WAVE,
            WTK=WTK,
            WTV=WAVE // WTK,
            NW=NW,
            VPT=STATE_VEC,
            BS=NW * WAVE,
            WARP_TILE_K=WARP_TILE_K,
            WTK_ITERS=DK // WARP_TILE_K,
            WGROUP_V=NW * (WAVE // WTK),
            BPV=spec.blocks_per_v_dim,
            TILE_V=DV // spec.blocks_per_v_dim,
            WTV_ITERS=spec.rows_per_lane,
            K_RUN=K_RUN,
            K_PIECES=STATE_VEC // K_RUN,
            S_SLOT=spec.num_v_heads * DV * DK * _ELEM_BYTES[spec.state_dtype],
            CST_SLOT=spec.conv_dim * CONV_TAPS * _ELEM_BYTES[spec.dtype],
            V_CHAN0=2 * spec.num_k_heads * DK,
            LATE_ROW_INPUTS=late_row_inputs(spec),
            LANE_NORM=spec.fuse_out_norm
            and not spec.out_lds
            and not spec.norm_gate_once,
            OUT_W=OUT_W,
            N_OUT=DV // OUT_W,
            OUT_ROUNDS=-(-(DV // OUT_W) // (NW * WAVE)),
            Q0=0,
            K0=DK,
            V0=2 * DK,
            N_CONV=N_CONV,
            N_DEC=N_DEC,
            N_GW=N_GW,
            GW0=N_CONV + N_DEC,
            N_ONCE=N_CONV + N_DEC + N_GW,
        )


@dataclass(frozen=True)
class _Args:
    """The warp-tiled kernel's arguments and LDS blocks; a fusion's or knob's
    are None while it is off. Under fuse_conv, Q, K and Vv are the one
    packed ``mixed_qkv`` row."""

    Q: Value
    K: Value
    Vv: Value
    Ag: Value
    Bg: Value
    DTB: Value
    ALOG: Value
    RIDX: Value
    WIDX: Value
    STATE: Value
    OUT: Value
    batch_size: Value
    CST: Optional[Value] = None  # fuse_conv
    CW: Optional[Value] = None
    qkv_stride: Optional[Value] = None
    OG: Optional[Value] = None  # fuse_out_norm
    NWT: Optional[Value] = None
    og_stride: Optional[Value] = None
    norm_eps: Optional[Value] = None
    lds_rn: Optional[Value] = None  # fuse_out_norm with NW > 1
    lds_once: Optional[Value] = None  # conv_once
    lds_out: Optional[Value] = None  # out_lds


@dataclass(frozen=True)
class _Lane:
    """One lane of the warp-tiled body: the builder, spec, geometry and
    arguments every phase shares, and this lane's coordinates.

    ``k_offs[ki][p]`` (the first column of this lane's run ``p`` in K chunk
    ``ki``), ``v_rows[vi]`` and ``s_rows[vi]`` (the in-slot base of state row
    ``v_rows[vi]``) are computed once and shared by every load and store, so
    a K channel or a V row has one index expression in the whole kernel:
    slot i of each chunk is the same K channel for the gate, q/k, the state
    and the conv taps. Per-lane K values (q, k, KDA decay, each state row)
    are flat lists of WTK_ITERS * VPT entries in (K chunk, slot) order;
    :meth:`chunk` cuts chunk ``ki`` out of one.
    """

    b: IRBuilder
    spec: GdnDecodeSpec
    t: _Tile
    a: _Args
    tid: Value
    b_i: Value
    hv_i: Value
    hk_i: Value
    k_lane: Value
    gv_start: Value
    read_pool: Value
    write_pool: Value
    k_offs: list
    v_rows: list
    s_rows: list

    def wsum(self, v):
        """xor-butterfly sum over the WTK-lane group (broadcast in-group)."""
        # xor 1/2 via quad_perm (VALU DPP: no LDS crossbar / no lgkmcnt(0) stall);
        # xor 4 crosses the 4-lane quad so it stays on ds_swizzle, unless
        # dpp_reduce: then xor 4 / xor 8 use tile.dpp_xor with mask 7 / 15
        # (DPP row_half_mirror / row_mirror on gfx9 CDNA). Those pair lane
        # i with i ^ 7 (i ^ 15), not i ^ 4 (i ^ 8), but the stages run in
        # ascending order: by the time one runs, every lane of a quad
        # (half-row) holds that quad's (half-row's) sum, so both pairings
        # add the same two values.
        #
        # This split is NOT applied at the four sibling sites that make the
        # same choice -- they are all still on plain warp_shuffle_xor, and
        # each would take the same win for masks 1 and 2:
        #   helpers/reduction.py `_warp_xor_reduce` (the generic form of
        #     this loop, in a module this kernel already imports)
        #   cpp/helpers/attention.cpp `rocke_warp_xor_reduce_sum` (its twin)
        #   kernels/gfx950/kda_chunkwise.py `_reduce16_fadd`
        #   attention_tiled_2d.py + its gfx950 twin, register transpose
        # Sweeping them touches two engines and three families, so it is
        # deliberately not done here; see the follow-up ticket. Anyone
        # editing this loop should consider whether the sibling is still
        # waiting.
        b, dpp = self.b, self.spec.dpp_reduce_on
        shfl = [1 << i for i in range(self.t.WTK.bit_length() - 1)]  # [1,2,4]: WTK=8
        for off in shfl:
            if off <= 2:
                v = b.fadd(v, b.warp_shuffle_xor_quad(v, off))
            elif dpp and off <= 8:
                v = b.fadd(v, b.warp_shuffle_xor_dpp(v, 2 * off - 1))
            else:
                v = b.fadd(v, b.warp_shuffle_xor(v, off))
        return v

    def head_sum(self, xs):
        """Sum over every K channel of the head: this lane's values, then
        the WTK-lane group."""
        return self.wsum(tree_reduce(self.b, self.b.fadd, xs))

    def chunk(self, xs, ki):
        return xs[ki * self.t.VPT : (ki + 1) * self.t.VPT]

    def is_k_lane0(self):
        return self.b.cmp_eq(self.k_lane, self.b.const_i32(0))

    # conv_state layout [slot, channel, tap]: cst_chan and CST_SLOT are the
    # only places that know it.
    def cst_chan(self, ch):
        return self.b.mul(ch, self.b.const_i32(CONV_TAPS))

    def q_chan(self, koff=None):
        """q conv channel of the K run at ``koff``: hk*DK + koff, or the
        head's first q channel hk*DK without ``koff``."""
        first = self.b.mul(self.hk_i, self.b.const_i32(self.t.DK))
        return first if koff is None else self.b.add(first, koff)

    def k_chan(self, qch, k_span=None):
        """k conv channel of q channel ``qch``: one q-span (Hk*DK) later,
        as in the packed row. ``k_span`` is an optional prebuilt constant
        for Hk*DK (the conv_once prologue folds its slot origin into it),
        so each call site keeps its op order (golden IR)."""
        if k_span is None:
            k_span = self.b.const_i32(self.t.HK * self.t.DK)
        return self.b.add(qch, k_span)

    def v_span_chan(self, row):
        """hv*DV + row: the conv channel of this head's V row ``row``,
        counted from the start of the v span (V_CHAN0 into the packed
        row). An int ``row`` becomes a constant built after hv*DV."""
        b = self.b
        first = b.mul(self.hv_i, b.const_i32(self.t.DV))
        return b.add(first, b.const_i32(row) if isinstance(row, int) else row)

    def og_head_row(self):
        """First element of this head's [DV] out_gate row."""
        b = self.b
        return b.add(
            b.mul(self.b_i, self.a.og_stride), b.mul(self.hv_i, b.const_i32(self.t.DV))
        )


def _warp_tiled_args(b: IRBuilder, spec: GdnDecodeSpec, t: _Tile) -> _Args:
    """Declare the warp-tiled kernel's params in ABI order
    (:func:`gdn_decode_signature`), then its LDS blocks: the fused norm's
    partials, conv_once's channels and out_lds's rows."""
    CONV, RN = spec.fuse_conv, spec.fuse_out_norm
    io_ty = io_ir_type(spec.dtype)
    fused = {}
    if CONV:
        Q = K = Vv = _vec_param(b, "mixed_qkv", io_ty, readonly=True)
    else:
        Q = _vec_param(b, "query", io_ty, readonly=True)
        K = _vec_param(b, "key", io_ty, readonly=True)
        Vv = _vec_param(b, "value", io_ty, readonly=True)
    Ag, Bg, DTB, ALOG, RIDX, WIDX = _gate_and_index_params(b, spec, io_ty)
    STATE = _vec_param(b, "state", _state_ir_type(spec.state_dtype))
    OUT = _vec_param(b, "out", io_ty, writeonly=True)
    if CONV:
        fused["CST"] = _vec_param(b, "conv_state", io_ty)
        fused["CW"] = _vec_param(b, "conv_weight", F32, readonly=True)
    if RN:
        fused["OG"] = b.param(
            "out_gate", PtrType(io_ty, "global"), noalias=True, readonly=True
        )
        fused["NWT"] = b.param(
            "norm_weight", PtrType(F32, "global"), noalias=True, readonly=True
        )
    batch_size = b.param("batch_size", I32)
    if CONV:
        fused["qkv_stride"] = b.param("qkv_stride", I32)
    if RN:
        fused["og_stride"] = b.param("og_stride", I32)
        fused["norm_eps"] = b.param("norm_eps", F32)
        # one f32 partial per wave for the cross-wave sum of squares
        if t.NW > 1:
            fused["lds_rn"] = b.smem_alloc_f32([t.NW], name_hint="rn_partials")
    if spec.conv_once:
        fused["lds_once"] = b.smem_alloc_f32([t.N_ONCE], name_hint="conv_once")
    if spec.out_lds:
        # each row's pre-norm f32 output, written by its k-lane-0 owner
        fused["lds_out"] = b.smem_alloc_f32([t.DV], name_hint="out_rows")
    return _Args(
        Q, K, Vv, Ag, Bg, DTB, ALOG, RIDX, WIDX, STATE, OUT, batch_size, **fused
    )


@contextmanager
def _active_lane(
    b: IRBuilder, spec: GdnDecodeSpec, t: _Tile, a: _Args
) -> Iterator[_Lane]:
    """This lane's coordinates, then the body's guard: yields the
    :class:`_Lane` inside ``scf_if(active)``, so a padded sequence (negative
    pool index) runs nothing."""
    tid = b.thread_id_x()
    bidx = b.block_id_x()
    if spec.xcd_remap_on:
        bidx = _xcd_remap(b, bidx, b.mul(a.batch_size, b.const_i32(t.HV * t.BPV)))
    b_hv_i = b.div(bidx, b.const_i32(t.BPV))
    tile_v_start = b.mul(b.mod(bidx, b.const_i32(t.BPV)), b.const_i32(t.TILE_V))
    b_i = b.div(b_hv_i, b.const_i32(t.HV))
    hv_i = b.mod(b_hv_i, b.const_i32(t.HV))
    hk_i = b.div(hv_i, b.const_i32(spec.v_per_k_head))
    w_tid = b.mod(tid, b.const_i32(t.WAVE))
    wid = b.div(tid, b.const_i32(t.WAVE))
    k_lane = b.mod(w_tid, b.const_i32(t.WTK))
    v_lane = b.div(w_tid, b.const_i32(t.WTK))
    warp_k_start = b.mul(k_lane, b.const_i32(t.K_RUN))
    gv_start = b.add(b.mul(wid, b.const_i32(t.WTV)), v_lane)

    read_pool, write_pool, active = _pool_slots(b, a.RIDX, a.WIDX, b_i)
    with b.scf_if(active):
        k_offs = [
            [
                b.add(
                    warp_k_start,
                    b.const_i32(ki * t.WARP_TILE_K + p * t.WTK * t.K_RUN),
                )
                for p in range(t.K_PIECES)
            ]
            for ki in range(t.WTK_ITERS)
        ]
        v_rows = [
            b.add(tile_v_start, b.add(gv_start, b.const_i32(vi * t.WGROUP_V)))
            for vi in range(t.WTV_ITERS)
        ]
        s_rows = [_state_row(b, hv_i, v_row, spec) for v_row in v_rows]
        yield _Lane(
            b=b,
            spec=spec,
            t=t,
            a=a,
            tid=tid,
            b_i=b_i,
            hv_i=hv_i,
            hk_i=hk_i,
            k_lane=k_lane,
            gv_start=gv_start,
            read_pool=read_pool,
            write_pool=write_pool,
            k_offs=k_offs,
            v_rows=v_rows,
            s_rows=s_rows,
        )


def _build_warp_tiled(spec: GdnDecodeSpec) -> KernelDef:
    """v2: warp-tiled. One CTA per (seq, value-head); the
    ``head_v_dim x head_k_dim`` state is distributed across the block's warps
    (WTV x WTK lanes, VPT values/lane), and the K-reductions (L2 norms,
    dot(k,q), S.k, S.q) are folded once per WTK-lane group via shuffle-xor,
    eliminating v1's per-thread redundant compute and register pressure.

    The body runs in phases: every global load (gate, q/k and their conv
    inputs or conv_once's channels, state, v, norm inputs), then the gate
    math, the conv (per lane, or conv_once's prologue), the recurrence over
    this lane's rows, the fused norm and the conv-tap shift.
    """
    t = _Tile.of(spec)
    CONV, RN, ONCE = spec.fuse_conv, spec.fuse_out_norm, spec.conv_once
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = t.BS
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, _MAX_WAVES_PER_EU)
    a = _warp_tiled_args(b, spec, t)

    with _active_lane(b, spec, t, a) as ln:
        # Every global load is issued before any math; ALGORITHM.md §4.3 ("Why
        # loads first") gives the reason and the register cost.
        gate_in = _load_gate_inputs(ln)
        q_raw, k_raw, conv = _load_qk(ln, gate_in)
        st = _load_state(ln)
        v = _load_v(ln, conv)
        out_lds_in = _load_out_lds_inputs(ln) if spec.out_lds else None
        r_norm = _norm_inputs(ln) if t.LANE_NORM and not t.LATE_ROW_INPUTS else None

        decay, beta, exp_alog = _gate_math(ln, gate_in)
        if ONCE:
            qn, kn, rv, decay = _conv_once(ln, conv, decay, exp_alog)
        elif CONV:
            qn, kn, rv = _conv_qkv(ln, q_raw, k_raw, conv, v)
        else:
            qn, kn, rv = q_raw, k_raw, v.rv
        outs, v = _recurrence(ln, st, decay, beta, qn, kn, rv, v, conv)
        if RN:
            _out_norm(ln, outs, v.v_idx, r_norm, out_lds_in)
        if CONV and not ONCE:
            _shift_conv_taps(ln, q_raw, k_raw, conv, v)

    return b.kernel


@dataclass(frozen=True)
class _GateIn:
    """The gate's loaded inputs: the per-head ones (GDN's a, b, dt_bias and
    A_log; KDA's b and A_log) and, for KDA, this lane's gate slice and f32
    dt_bias runs per K chunk (none under conv_once, whose prologue loads one
    channel per thread from the gate and dt_bias rows)."""

    head: tuple
    gvs: list
    dtvecs: list
    g_row: Optional[Value] = None  # KDA: this head's gate row
    dtb_row: Optional[Value] = None  # KDA: this head's dt_bias row


def _load_gate_inputs(ln: _Lane) -> _GateIn:
    """The gate's global loads. Each koff is also the state load's offset,
    so slot i of a KDA gate slice is the same K channel as slot i of the
    state."""
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    a_idx = b.add(b.mul(ln.b_i, b.const_i32(t.HV)), ln.hv_i)
    if spec.gate_kind == "gdn":
        gdn_in = _gdn_gate_inputs(b, spec, a.Ag, a.Bg, a.DTB, a.ALOG, a_idx, ln.hv_i)
        return _GateIn(gdn_in, [], [])
    kda_in = _kda_head_inputs(b, spec, a.Bg, a.ALOG, a_idx, ln.hv_i)
    g_row, dtb_row = _kda_gate_rows(b, spec, ln.b_i, ln.hv_i)
    gvs, dtvecs = [], []
    if not spec.conv_once:
        # A K run of 8 f32 starts on a 32 B boundary of dt_bias, one of 4
        # (interleave_cols) on a 16 B boundary.
        dt_align = min(_KDA_DT_BIAS_ALIGN, t.K_RUN * 4)
        for runs in ln.k_offs:
            gv, dts = [], []
            for koff in runs:
                gv += load_vec_as_f32(
                    b, a.Ag, b.add(g_row, koff), dtype=spec.dtype, n=t.K_RUN
                )
                dts.append(
                    b.global_load_vN(
                        a.DTB, b.add(dtb_row, koff), F32, t.K_RUN, align=dt_align
                    )
                )
            gvs.append(gv)
            dtvecs.append(dts)
    return _GateIn(kda_in, gvs, dtvecs, g_row, dtb_row)


def _conv_taps(ln: _Lane, slot, ch, n):
    """CONV_TAPS taps x n channels from [slot, channel, tap]: CONV_TAPS
    contiguous n-element vectors."""
    b = ln.b
    taps = []
    for j in range(CONV_TAPS):
        taps += load_vec_as_f32(
            b,
            slot,
            b.add(ln.cst_chan(ch), b.const_i32(n * j)),
            dtype=ln.spec.dtype,
            n=n,
        )
    return [taps[CONV_TAPS * i : CONV_TAPS * (i + 1)] for i in range(n)]


def _conv_weights(ln: _Lane, ch, n):
    """CONV_WIDTH weights x n channels from [channel, tap]: CONV_WIDTH
    contiguous n-float vectors. The param promises 16 B, so a 32 B load
    claims 16 B."""
    b = ln.b
    ws = []
    for j in range(CONV_WIDTH):
        ws += _load_f32_vec(
            b,
            ln.a.CW,
            b.add(b.mul(ch, b.const_i32(CONV_WIDTH)), b.const_i32(n * j)),
            n,
            align=_VEC_ALIGN,
        )
    return [ws[CONV_WIDTH * i : CONV_WIDTH * (i + 1)] for i in range(n)]


@dataclass(frozen=True)
class _ConvQk:
    """fuse_conv: this token's packed-row and conv-state read-slot handles,
    and either the conv inputs of this lane's q/k chunks, indexed
    ``[ki][i]``, or (conv_once) the write slot and the prologue's work
    items."""

    qkv_row: Value  # first element of this token's packed [q | k | v] row
    cst_r: Value  # conv-state read slot
    qchs: Optional[list] = None  # q conv channel of each K run, [ki][p]
    q_taps: Optional[list] = None  # CONV_TAPS taps per channel, oldest first
    k_taps: Optional[list] = None
    q_w: Optional[list] = None  # CONV_WIDTH weights per channel
    k_w: Optional[list] = None
    cst_w: Optional[Value] = None  # conv_once: conv-state write slot
    once_items: Optional[list] = None  # conv_once: see _load_conv_once


def _load_qk(ln: _Lane, gate_in: _GateIn):
    """This lane's raw q and k K chunks, one VPT list per chunk (K_PIECES
    runs each), and under fuse_conv their conv inputs: ``(q, k, conv)``.
    Under conv_once the lanes load no q/k: the prologue's work items are
    loaded here instead (:func:`_load_conv_once`)."""
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    qn, kn = [], []
    if not spec.fuse_conv:
        qk_base = _head_base(b, ln.b_i, ln.hk_i, t.HK, t.DK)
        for runs in ln.k_offs:
            q_c, k_c = [], []
            for koff in runs:
                off = b.add(qk_base, koff)
                q_c += load_vec_as_f32(b, a.Q, off, dtype=spec.dtype, n=t.K_RUN)
                k_c += load_vec_as_f32(b, a.K, off, dtype=spec.dtype, n=t.K_RUN)
            qn.append(q_c)
            kn.append(k_c)
        return qn, kn, None
    # q/k/v come from one packed row: q at channel hk*DK, k one q-span
    # later, v after both. The conv taps live at the read slot.
    qkv_row = b.mul(ln.b_i, a.qkv_stride)
    qk_base = b.add(qkv_row, ln.q_chan())
    k_shift = b.const_i32(t.HK * t.DK)
    cst_r = _slot_ptr(b, a.CST, ln.read_pool, t.CST_SLOT)
    if spec.conv_once:
        cst_w, items = _load_conv_once(ln, qkv_row, cst_r, gate_in)
        return qn, kn, _ConvQk(qkv_row, cst_r, cst_w=cst_w, once_items=items)
    q_taps, k_taps, q_w, k_w = [], [], [], []
    qchs = [[ln.q_chan(koff) for koff in runs] for runs in ln.k_offs]
    for ki in range(t.WTK_ITERS):
        q_c, k_c, qt, qw, kt, kw = [], [], [], [], [], []
        for p in range(t.K_PIECES):
            off = b.add(qk_base, ln.k_offs[ki][p])
            q_c += load_vec_as_f32(b, a.Q, off, dtype=spec.dtype, n=t.K_RUN)
            k_c += load_vec_as_f32(
                b, a.K, b.add(off, k_shift), dtype=spec.dtype, n=t.K_RUN
            )
            kch = ln.k_chan(qchs[ki][p], k_shift)
            qt += _conv_taps(ln, cst_r, qchs[ki][p], t.K_RUN)
            qw += _conv_weights(ln, qchs[ki][p], t.K_RUN)
            kt += _conv_taps(ln, cst_r, kch, t.K_RUN)
            kw += _conv_weights(ln, kch, t.K_RUN)
        qn.append(q_c)
        kn.append(k_c)
        q_taps.append(qt)
        q_w.append(qw)
        k_taps.append(kt)
        k_w.append(kw)
    return qn, kn, _ConvQk(qkv_row, cst_r, qchs, q_taps, k_taps, q_w, k_w)


def _load_conv_once(ln: _Lane, qkv_row, cst_r, gate_in: _GateIn):
    """conv_once's loads: one work item per LDS slot ``f = tid + it * BS``,
    the conv channels [q | k | v] first, then the KDA decay channels, then
    the norm_gate_once rows. All loads are issued here, ahead of the state
    loads, with the index clamped into range so no load sits behind a
    branch; only the writes are guarded. Returns ``(cst_w, items)``."""
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    DK, N_CONV, GW0, K0, V0 = t.DK, t.N_CONV, t.GW0, t.K0, t.V0
    cst_w = _slot_ptr(b, a.CST, ln.write_pool, t.CST_SLOT)
    # (first slot, end slot, channel minus slot) per conv segment: slot f of
    # segment s holds channel f + origin, from the same channel maps as the
    # per-lane path.
    segs = (
        (t.Q0, K0, ln.q_chan()),
        (K0, V0, ln.k_chan(ln.q_chan(), b.const_i32(t.HK * DK - K0))),
        (V0, N_CONV, ln.v_span_chan(t.V_CHAN0 - V0)),
    )
    items = []
    for it in range(-(-t.N_ONCE // t.BS)):
        lo, hi = it * t.BS, (it + 1) * t.BS  # static range of f this round
        f = b.add(ln.tid, b.const_i32(lo)) if lo else ln.tid
        item = {"f": f, "hi": hi}
        if lo < N_CONV:
            fc = b.smin(f, b.const_i32(N_CONV - 1)) if hi > N_CONV else f
            live = [s for s in segs if s[0] < hi and s[1] > lo]
            ch_off = live[-1][2]
            for s in reversed(live[:-1]):
                ch_off = b.select(b.cmp_lt(fc, b.const_i32(s[1])), s[2], ch_off)
            ch = b.add(fc, ch_off)
            item["ch"] = ch
            item["x"] = _chan_x(ln, qkv_row, ch)
            item["taps"] = _chan_taps(ln, cst_r, ch, ln.cst_chan(ch))
            item["w"] = _chan_w(ln, ch)
        if t.N_DEC and hi > N_CONV and lo < GW0:
            d = b.sub(f, b.const_i32(N_CONV))
            if lo < N_CONV:
                d = b.smax(d, b.const_i32(0))
            if hi > GW0:
                d = b.smin(d, b.const_i32(t.N_DEC - 1))
            item["a"] = load_scalar_as_f32(
                b, a.Ag, b.add(gate_in.g_row, d), dtype=spec.dtype
            )
            item["dt"] = b.global_load_f32(a.DTB, b.add(gate_in.dtb_row, d))
        if t.N_GW and hi > GW0:
            g = b.sub(f, b.const_i32(GW0))
            if lo < GW0:
                g = b.smax(g, b.const_i32(0))
            if hi > t.N_ONCE:
                g = b.smin(g, b.const_i32(t.N_GW - 1))
            item["og"] = load_scalar_as_f32(
                b, a.OG, b.add(ln.og_head_row(), g), dtype=spec.dtype
            )
            item["nw"] = b.global_load_f32(a.NWT, g)
        items.append(item)
    return cst_w, items


@dataclass(frozen=True)
class _StateIn:
    """The state read slot and, unless LATE_ROW_INPUTS, every raw state
    chunk this lane owns, keyed (vi, ki)."""

    state_r: Value
    s_raw: Optional[dict]


def _state_chunk(ln: _Lane, state_r, vi, ki):
    """This lane's chunk ``ki`` of state row ``vi``, loaded as f32 one K run
    at a time with the state load hint."""
    vals = []
    for koff in ln.k_offs[ki]:
        vals += _load_state_f32(
            ln.b,
            state_r,
            ln.b.add(ln.s_rows[vi], koff),
            state_dtype=ln.spec.state_dtype,
            n=ln.t.K_RUN,
            hint=ln.spec.state_load_hint,
        )
    return vals


def _load_state(ln: _Lane) -> _StateIn:
    """The raw state chunks; the LATE_ROW_INPUTS tile loads each one at its
    decay instead (:func:`_decay_row`)."""
    t = ln.t
    state_r = _slot_ptr(ln.b, ln.a.STATE, ln.read_pool, t.S_SLOT)
    if t.LATE_ROW_INPUTS:
        return _StateIn(state_r, None)
    s_raw = {
        (vi, ki): _state_chunk(ln, state_r, vi, ki)
        for vi in range(t.WTV_ITERS)
        for ki in range(t.WTK_ITERS)
    }
    return _StateIn(state_r, s_raw)


# One conv channel at a time (each owned v row, and conv_once's one channel
# per thread): its newest value from the packed row, its taps at the read
# slot, its weights, and the shifted taps at the write slot.
def _chan_x(ln: _Lane, qkv_row, c):
    return load_scalar_as_f32(ln.b, ln.a.Vv, ln.b.add(qkv_row, c), dtype=ln.spec.dtype)


def _chan_taps(ln: _Lane, cst_r, c, tap0=None):
    """The CONV_TAPS taps of channel ``c``, oldest first. ``tap0`` is an
    optional prebuilt cst_chan(c), so each call site keeps its op order
    (golden IR)."""
    b = ln.b
    return [
        load_scalar_as_f32(
            b,
            cst_r,
            b.add(ln.cst_chan(c) if tap0 is None else tap0, b.const_i32(j)),
            dtype=ln.spec.dtype,
        )
        for j in range(CONV_TAPS)
    ]


def _chan_w(ln: _Lane, c):
    b = ln.b
    return _load_f32_vec(
        b, ln.a.CW, b.mul(c, b.const_i32(CONV_WIDTH)), CONV_WIDTH, align=_VEC_ALIGN
    )


def _store_shifted_taps(ln: _Lane, cst_w, c, taps, x):
    """Drop channel ``c``'s oldest tap and append its newest value ``x``."""
    b = ln.b
    base = ln.cst_chan(c)
    for j, val in enumerate(list(taps[1:]) + [x]):
        store_scalar_from_f32(
            b, cst_w, b.add(base, b.const_i32(j)), val, dtype=ln.spec.dtype
        )


@dataclass(frozen=True)
class _VRows:
    """This lane's value inputs, one entry per owned V row. On the
    LATE_ROW_INPUTS tile the conv inputs stay None here: :func:`_recurrence`
    loads them in its row loop and returns a copy that carries them."""

    v_idx: list  # element offset of the row in value / out
    rv: Optional[list] = None  # unfused: the row's v
    vch: Optional[list] = None  # fuse_conv: the row's v conv channel,
    v_raw: Optional[list] = None  # its newest value, taps and weights
    v_taps: Optional[list] = None
    v_w: Optional[list] = None


def _load_v(ln: _Lane, conv: Optional[_ConvQk]) -> _VRows:
    """This lane's v, one per V row: loaded, or under fuse_conv each row's
    conv channel and (but on the LATE_ROW_INPUTS tile) its conv inputs.
    conv_once loads none: its v comes from LDS after the prologue."""
    b, spec, t = ln.b, ln.spec, ln.t
    v_idx = [
        b.add(_head_base(b, ln.b_i, ln.hv_i, t.HV, t.DV), v_row) for v_row in ln.v_rows
    ]
    if not spec.fuse_conv:
        rv = [load_scalar_as_f32(b, ln.a.Vv, i, dtype=spec.dtype) for i in v_idx]
        return _VRows(v_idx, rv=rv)
    if spec.conv_once:
        return _VRows(v_idx)
    # v channel of each owned row in the packed row / conv state
    vch = [b.add(b.const_i32(t.V_CHAN0), ln.v_span_chan(v_row)) for v_row in ln.v_rows]
    if t.LATE_ROW_INPUTS:
        return _VRows(v_idx, vch=vch)
    return _VRows(
        v_idx,
        vch=vch,
        v_raw=[_chan_x(ln, conv.qkv_row, c) for c in vch],
        v_taps=[_chan_taps(ln, conv.cst_r, c) for c in vch],
        v_w=[_chan_w(ln, c) for c in vch],
    )


def _load_out_lds_inputs(ln: _Lane):
    """out_lds's storing lanes and their norm inputs: ``(out_j, o_og,
    o_nw)``.

    Lane j of round r stores rows [OUT_W * j, OUT_W * (j + 1)), j = tid + r *
    BS; an index past the last group is clamped so its loads stay in range
    (it stores nothing). Without the norm_gate_once LDS slots they load their
    rows' out_gate and norm_weight here, with the other loads.
    """
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    out_j = []
    for r in range(t.OUT_ROUNDS):
        j = b.add(ln.tid, b.const_i32(r * t.BS)) if r else ln.tid
        guard = (r + 1) * t.BS > t.N_OUT
        jc = b.smin(j, b.const_i32(t.N_OUT - 1)) if guard else j
        out_j.append((j, jc, guard))
    if spec.norm_gate_once:
        return out_j, None, None
    og_row = ln.og_head_row()
    o_og, o_nw = [], []
    for _, jc, _ in out_j:
        row0 = b.mul(jc, b.const_i32(t.OUT_W))
        o_og.append(
            [
                load_scalar_as_f32(
                    b,
                    a.OG,
                    b.add(og_row, b.add(row0, b.const_i32(e))),
                    dtype=spec.dtype,
                )
                for e in range(t.OUT_W)
            ]
        )
        o_nw.append(_load_f32_vec(b, a.NWT, row0, t.OUT_W, align=4))
    return out_j, o_og, o_nw


def _norm_inputs(ln: _Lane):
    """Each owned row's out_gate and norm_weight (the per-lane norm):
    ``(r_og, r_nw)``."""
    b, a = ln.b, ln.a
    og_row = ln.og_head_row()
    r_og = [
        load_scalar_as_f32(b, a.OG, b.add(og_row, v_row), dtype=ln.spec.dtype)
        for v_row in ln.v_rows
    ]
    return r_og, [b.global_load_f32(a.NWT, v_row) for v_row in ln.v_rows]


def _gate_math(ln: _Lane, gate_in: _GateIn):
    """``(decay, beta, exp_alog)`` from the loaded gate inputs. GDN's decay
    is one scalar (and exp_alog None); KDA's is a flat list over this lane's
    K channels, empty under conv_once, whose prologue computes the decays
    from exp_alog."""
    b, spec, t = ln.b, ln.spec, ln.t
    if spec.gate_kind == "gdn":
        decay, beta = _gdn_gate_math(b, *gate_in.head)
        return decay, beta, None
    # KDA: one decay per K channel, for the slice THIS lane owns.
    #
    # The state tile is keyed (vi, ki) -- V row and K chunk -- but a
    # channel's decay does not depend on which V row is being faded, so
    # the decay is computed once per K channel and reused across all
    # WTV_ITERS rows. That is what holds the extra register cost to
    # WTK_ITERS*VPT values instead of multiplying with the state tile.
    beta, exp_alog = _kda_head_math(b, *gate_in.head)
    decay = []
    for gv, dts in zip(gate_in.gvs, gate_in.dtvecs):
        # dt_bias lanes are extracted lazily, next to their use.
        decay += [
            _kda_channel_decay(
                b,
                spec,
                exp_alog,
                gv[i],
                lambda i=i: b.vec_extract(dts[i // t.K_RUN], i % t.K_RUN),
            )
            for i in range(t.VPT)
        ]
    return decay, beta, exp_alog


def _conv_silu(b: IRBuilder, h, w, x):
    """conv1d + SiLU of one channel. h: CONV_TAPS taps oldest-first,
    w: CONV_WIDTH weights, x: the newest (current) value."""
    acc = tree_reduce(b, b.fadd, [b.fmul(t, wt) for t, wt in zip(list(h) + [x], w)])
    return _silu(b, acc)


def _out_gate_act(spec: GdnDecodeSpec):
    """The fused RMSNorm's output gate activation (OUT_GATE_ACTIVATION)."""
    return _silu if spec.out_gate_activation == "silu" else _sigmoid


def _lds_f32(b: IRBuilder, buf, idx, n):
    """``n`` contiguous f32 LDS slots from ``idx``, up to 4 per load."""
    m = min(n, 4)
    vecs = [
        b.smem_load_vN_f32(buf, b.add(idx, b.const_i32(m * q)) if q else idx, n=m)
        for q in range(n // m)
    ]
    return [b.vec_extract(v, i) for v in vecs for i in range(m)]


def _conv_once(ln: _Lane, conv: _ConvQk, decay, exp_alog):
    """conv_once's prologue math, then this lane's operands from LDS:
    ``(qn, kn, rv, decay)``; under KDA ``decay`` is a new list, the given one
    followed by this lane's decays read from LDS.

    Each work item (:func:`_load_conv_once`) computes its slot's conv1d +
    SiLU, KDA decay or norm_gate_once product and stores it; then one LDS
    barrier, and each lane reads its q/k (and KDA decay) runs and its rows'
    v back.
    """
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    N_CONV, GW0, N_ONCE, lds = t.N_CONV, t.GW0, t.N_ONCE, a.lds_once
    K0, V0 = t.K0, t.V0
    gate_act = _out_gate_act(spec)
    for item in conv.once_items:
        f, hi = item["f"], item["hi"]
        val = None
        if "ch" in item:
            val = _conv_silu(b, item["taps"], item["w"], item["x"])
        if "a" in item:
            dec = _kda_channel_decay(
                b, spec, exp_alog, item["a"], lambda item=item: item["dt"]
            )
            val = (
                dec
                if val is None
                else b.select(b.cmp_lt(f, b.const_i32(N_CONV)), val, dec)
            )
        if "og" in item:
            gw = b.fmul(gate_act(b, item["og"]), item["nw"])
            val = (
                gw if val is None else b.select(b.cmp_lt(f, b.const_i32(GW0)), val, gw)
            )
        if hi > N_ONCE:
            with b.scf_if(b.cmp_lt(f, b.const_i32(N_ONCE))):
                b.smem_store_vN_f32(lds, [f], val, 1)
        else:
            b.smem_store_vN_f32(lds, [f], val, 1)
        if "ch" in item:
            # Single writer: the thread that read this channel's taps
            # shifts them, so read slot == write slot needs no barrier.
            shift_taps = partial(
                _store_shifted_taps, ln, conv.cst_w, item["ch"], item["taps"], item["x"]
            )
            if hi > N_CONV:
                with b.scf_if(b.cmp_lt(f, b.const_i32(N_CONV))):
                    shift_taps()
            else:
                shift_taps()
    # LDS-only drain: the state loads stay in flight across it.
    b.sync_lds_only()

    qn, kn, lds_decay = [], [], []
    for runs in ln.k_offs:
        q_c, k_c, d_c = [], [], []
        for koff in runs:
            q_c += _lds_f32(b, lds, koff, t.K_RUN)  # the q segment starts at Q0 = 0
            k_c += _lds_f32(b, lds, b.add(koff, b.const_i32(K0)), t.K_RUN)
            if t.N_DEC:
                d_c += _lds_f32(b, lds, b.add(koff, b.const_i32(N_CONV)), t.K_RUN)
        qn.append(q_c)
        kn.append(k_c)
        lds_decay += d_c
    rv = [_lds_f32(b, lds, b.add(v_row, b.const_i32(V0)), 1)[0] for v_row in ln.v_rows]
    return qn, kn, rv, (decay + lds_decay if t.N_DEC else decay)


def _conv_qkv(ln: _Lane, q_raw, k_raw, conv: _ConvQk, v: _VRows):
    """conv1d + SiLU on this lane's raw q/k chunks and v rows: ``(qn, kn,
    rv)``. The raw values stay with the caller: they become the newest conv
    tap. ``rv`` is None on the LATE_ROW_INPUTS tile, whose v conv runs in
    :func:`_recurrence`'s row loop."""
    b, t = ln.b, ln.t
    qn = [
        [
            _conv_silu(b, conv.q_taps[ki][i], conv.q_w[ki][i], q_raw[ki][i])
            for i in range(t.VPT)
        ]
        for ki in range(t.WTK_ITERS)
    ]
    kn = [
        [
            _conv_silu(b, conv.k_taps[ki][i], conv.k_w[ki][i], k_raw[ki][i])
            for i in range(t.VPT)
        ]
        for ki in range(t.WTK_ITERS)
    ]
    if t.LATE_ROW_INPUTS:
        return qn, kn, None
    rv = [_conv_silu(b, h, w, x) for h, w, x in zip(v.v_taps, v.v_w, v.v_raw)]
    return qn, kn, rv


def _decay_row(ln: _Lane, st: _StateIn, decay, vi):
    """This lane's decayed K values of state row ``vi``. A KDA decay chunk
    covers the same K channels as state chunk ki, in the same order, and is
    reused across every vi. The LATE_ROW_INPUTS tile loads each chunk here,
    as the parent order did."""
    spec, t = ln.spec, ln.t
    row = []
    for ki in range(t.WTK_ITERS):
        if t.LATE_ROW_INPUTS:
            vec = _state_chunk(ln, st.state_r, vi, ki)
        else:
            vec = st.s_raw[(vi, ki)]
        row += _forget(
            ln.b, spec, vec, decay if spec.gate_kind == "gdn" else ln.chunk(decay, ki)
        )
    return row


def _recurrence(ln: _Lane, st: _StateIn, decay, beta, qn, kn, rv, v, conv):
    """The gated delta rule on this lane's rows: scale q/k, dot(k, q), fade
    every state row, then per row S.k, S.q, the delta value, the output and
    the rank-1 update stored to the write slot: ``(outs, v)``. Unfused, each
    row's k-lane-0 owner stores its output here; under fuse_out_norm
    ``outs`` holds the outputs for :func:`_out_norm`. The LATE_ROW_INPUTS
    tile (but under conv_once, whose v comes from LDS) loads and convolves
    each row's v here, and ``v`` is returned with those conv inputs for the
    tap shift."""
    b, spec, t = ln.b, ln.spec, ln.t
    qn, kn = _scale_qk(
        b,
        [x for c in qn for x in c],
        [x for c in kn for x in c],
        use_l2norm=spec.use_qk_l2norm,
        scale=1.0 / math.sqrt(t.DK),
        reduce=ln.head_sum,
    )
    dot_kq = _dot(b, kn, qn, ln.head_sum)

    # With stream_rows each row is decayed in its own iteration below
    # instead of up front; with one row per lane there is nothing to stream,
    # and the emitted code is the default's.
    stream = spec.stream_rows_on
    sv = [] if stream else [_decay_row(ln, st, decay, vi) for vi in range(t.WTV_ITERS)]
    state_w = _slot_ptr(b, ln.a.STATE, ln.write_pool, t.S_SLOT)
    outs = []  # per-row outputs the fused norm needs
    late = t.LATE_ROW_INPUTS and not spec.conv_once
    if late:
        rv, v_raw, v_taps = [], [], []
    for vi in range(t.WTV_ITERS):
        if stream:
            sv.append(_decay_row(ln, st, decay, vi))
        phk = _dot(b, sv[vi], kn, ln.head_sum)
        phq = _dot(b, sv[vi], qn, ln.head_sum)
        if late:
            v_raw.append(_chan_x(ln, conv.qkv_row, v.vch[vi]))
            v_taps.append(_chan_taps(ln, conv.cst_r, v.vch[vi]))
            rv.append(_conv_silu(b, v_taps[vi], _chan_w(ln, v.vch[vi]), v_raw[vi]))
        # v_new is in-group uniform (rv, broadcast phk, beta) - no bcast.
        v_new = b.fmul(b.fsub(rv[vi], phk), beta)
        out_val = b.fadd(phq, b.fmul(v_new, dot_kq))
        if spec.fuse_out_norm:
            outs.append(out_val)  # stored after the cross-wave norm
        else:
            with b.scf_if(ln.is_k_lane0()):
                store_scalar_from_f32(
                    b, ln.a.OUT, v.v_idx[vi], out_val, dtype=spec.dtype
                )
        for ki in range(t.WTK_ITERS):
            kc, sc = ln.chunk(kn, ki), ln.chunk(sv[vi], ki)
            for p in range(t.K_PIECES):
                run = slice(p * t.K_RUN, (p + 1) * t.K_RUN)
                new = [b.fma(k, v_new, s) for k, s in zip(kc[run], sc[run])]
                vec = _pack_state(b, new, state_dtype=spec.state_dtype)
                off = b.add(ln.s_rows[vi], ln.k_offs[ki][p])
                _store_state(
                    b,
                    state_w,
                    off,
                    vec,
                    state_dtype=spec.state_dtype,
                    n=t.K_RUN,
                    hint=spec.state_store_hint,
                )
    if late:
        v = replace(v, v_raw=v_raw, v_taps=v_taps)
    return outs, v


def _out_norm(ln: _Lane, outs, v_idx, r_norm, out_lds_in):
    """Gated RMSNorm over the head's DV outputs, stored by each row's
    k-lane-0 owner, or under out_lds by :func:`_store_out_lds`.

    The gate activation follows the gate kind (OUT_GATE_ACTIVATION). Every
    k-lane of a row holds the same out_val, so only k-lane 0 contributes to
    the sum of squares. For NW > 1 the cross-wave reduction issues a
    workgroup barrier after every wave has written its partial, i.e. after
    every wave's tap reads (taps -> outs -> partial).
    """
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    if spec.out_lds:
        # Each row's pre-norm output to LDS ahead of the reduction's
        # barrier, which then also publishes it to the storing lanes.
        with b.scf_if(ln.is_k_lane0()):
            for vi in range(t.WTV_ITERS):
                b.smem_store_vN_f32(a.lds_out, [ln.v_rows[vi]], outs[vi], 1)
    if t.LANE_NORM and t.LATE_ROW_INPUTS:
        r_norm = _norm_inputs(ln)
    ss = tree_reduce(b, b.fadd, [b.fmul(o, o) for o in outs])
    contrib = b.select(ln.is_k_lane0(), ss, b.const_f32(0.0))
    total = block_lds_reduce_with_wave_prologue(
        b, contrib, a.lds_rn, ln.tid, block_size=t.BS, wave_size=t.WAVE
    )
    rstd = b.rsqrt(b.fadd(b.fmul(total, b.const_f32(1.0 / t.DV)), a.norm_eps))
    if spec.out_lds:
        _store_out_lds(ln, rstd, out_lds_in)
        return
    gate_act = _out_gate_act(spec)
    for vi in range(t.WTV_ITERS):
        if spec.norm_gate_once:
            gw_idx = b.add(ln.v_rows[vi], b.const_i32(t.GW0))
            (gw,) = _lds_f32(b, a.lds_once, gw_idx, 1)
            y = _norm_gate(b, outs[vi], rstd, gw)
        else:
            r_og, r_nw = r_norm
            gate = gate_act(b, r_og[vi])
            y = _norm_gate(b, outs[vi], rstd, r_nw[vi], gate)
        with b.scf_if(ln.is_k_lane0()):
            store_scalar_from_f32(b, a.OUT, v_idx[vi], y, dtype=spec.dtype)


def _store_out_lds(ln: _Lane, rstd, out_lds_in):
    """out_lds's stores: storing lane j normalizes the OUT_W consecutive
    pre-norm rows it reads back from LDS and stores them with one packed
    store (:func:`_load_out_lds_inputs`)."""
    b, spec, t, a = ln.b, ln.spec, ln.t, ln.a
    out_j, o_og, o_nw = out_lds_in
    if t.NW == 1:
        # one wave: the reduction has no barrier to reuse
        b.sync_lds_only()
    gate_act = _out_gate_act(spec)
    out_row0 = _head_base(b, ln.b_i, ln.hv_i, t.HV, t.DV)
    for r, (j, _, guard) in enumerate(out_j):

        def store_group(r=r, j=j):
            row0 = b.mul(j, b.const_i32(t.OUT_W))
            o = _lds_f32(b, a.lds_out, row0, t.OUT_W)
            if spec.norm_gate_once:
                gw = _lds_f32(b, a.lds_once, b.add(row0, b.const_i32(t.GW0)), t.OUT_W)
                ys = [_norm_gate(b, o[e], rstd, gw[e]) for e in range(t.OUT_W)]
            else:
                ys = []
                for e in range(t.OUT_W):
                    gate = gate_act(b, o_og[r][e])
                    ys.append(_norm_gate(b, o[e], rstd, o_nw[r][e], gate))
            store_vec(
                b,
                a.OUT,
                b.add(out_row0, row0),
                pack_f32_to(b, ys, dtype=spec.dtype),
                n=t.OUT_W,
            )

        if guard:
            with b.scf_if(b.cmp_lt(j, b.const_i32(t.N_OUT))):
                store_group()
        else:
            store_group()


def _shift_conv_taps(ln: _Lane, q_raw, k_raw, conv: _ConvQk, v: _VRows):
    """Shift the conv taps in place: drop the oldest, append the raw
    current token. (conv_once shifts each channel's taps in its prologue
    instead.)

    Other waves read the same q/k taps, so every wave must be past its tap
    reads before any tap is overwritten. With the norm on, its reduction
    barrier already orders that; otherwise one barrier is emitted here. One
    wave needs none.
    """
    b, spec, t = ln.b, ln.spec, ln.t
    if t.NW > 1 and not spec.fuse_out_norm:
        b.sync()
    cst_w = _slot_ptr(b, ln.a.CST, ln.write_pool, t.CST_SLOT)
    # q/k channels: one writer group (wave 0, v-lane 0), each k-lane
    # writing its VPT channels x CONV_TAPS shifted taps, one K run at a time.
    R = t.K_RUN
    with b.scf_if(b.cmp_eq(ln.gv_start, b.const_i32(0))):
        for ki in range(t.WTK_ITERS):
            for p in range(t.K_PIECES):
                qch = conv.qchs[ki][p]
                kch = ln.k_chan(qch)
                for ch, taps, raw in (
                    (qch, conv.q_taps, q_raw),
                    (kch, conv.k_taps, k_raw),
                ):
                    shifted = []
                    for i in range(p * R, (p + 1) * R):
                        shifted += list(taps[ki][i][1:]) + [raw[ki][i]]
                    for j in range(CONV_TAPS):
                        store_vec(
                            b,
                            cst_w,
                            b.add(ln.cst_chan(ch), b.const_i32(R * j)),
                            pack_f32_to(
                                b, shifted[R * j : R * (j + 1)], dtype=spec.dtype
                            ),
                            n=R,
                        )
    # v channels: the k-lane-0 owner of each row
    with b.scf_if(ln.is_k_lane0()):
        for vi in range(t.WTV_ITERS):
            _store_shifted_taps(ln, cst_w, v.vch[vi], v.v_taps[vi], v.v_raw[vi])


def gdn_decode_pointer_alignment(spec: GdnDecodeSpec) -> Dict[str, int]:
    """Byte alignment the emitted code assumes of each pointer argument's base.

    A param's ``align`` and every vector access's ``align`` are promises to
    LLVM that nothing on the device checks; a misaligned base (for example a
    contiguous view that starts one element into its storage) breaks them
    silently. The host launch path checks every tensor named here. Every
    in-slot / in-row offset the kernel adds is a multiple of the same
    alignment, given ``is_valid_spec`` (head dims multiples of 8) and the
    host's row-stride check for ``mixed_qkv``.
    """
    names = ["state", "out"]
    names += (
        ["mixed_qkv", "conv_state", "conv_weight"]
        if spec.fuse_conv
        else ["query", "key", "value"]
    )
    align = {name: _VEC_ALIGN for name in names}
    if spec.gate_kind == "kda":
        # Both emitters read the per-channel gate as 16 B vectors and the f32
        # dt_bias as 32 B ones (_KDA_DT_BIAS_ALIGN).
        align["a"] = _VEC_ALIGN
        align["dt_bias"] = _KDA_DT_BIAS_ALIGN
    return align


def gdn_decode_grid(batch: int, spec: GdnDecodeSpec) -> Tuple[int, int, int]:
    """One workgroup per (sequence, value head, v-sub-block)."""
    bpv = 1 if spec.simple else spec.blocks_per_v_dim
    return ceil_div_grid((batch * spec.num_v_heads * bpv, 1))


def gdn_decode_signature(spec: GdnDecodeSpec):
    """Kernel ABI. The fused modes only add (or, for fuse_conv, replace q/k/v
    with one packed row) parameters, so the unfused ABI is unchanged."""
    sig = SignatureBuilder()
    if spec.fuse_conv:
        sig = sig.ptr("mixed_qkv", spec.dtype)
    else:
        sig = (
            sig.ptr("query", spec.dtype).ptr("key", spec.dtype).ptr("value", spec.dtype)
        )
    sig = (
        sig.ptr("a", spec.dtype)
        .ptr("b", spec.dtype)
        .ptr("dt_bias", "f32" if spec.gate_kind == "kda" else spec.dtype)
        .ptr("A_log", "f32")
        .ptr("read_indices", "i32")
        .ptr("write_indices", "i32")
        .ptr("state", spec.state_dtype)
        .ptr("out", spec.dtype)
    )
    if spec.fuse_conv:
        sig = sig.ptr("conv_state", spec.dtype).ptr("conv_weight", "f32")
    if spec.fuse_out_norm:
        sig = sig.ptr("out_gate", spec.dtype).ptr("norm_weight", "f32")
    sig = sig.scalar("batch_size", "i32")
    if spec.fuse_conv:
        sig = sig.scalar("qkv_stride", "i32")
    if spec.fuse_out_norm:
        sig = sig.scalar("og_stride", "i32").scalar("norm_eps", "f32")
    return sig.build()
