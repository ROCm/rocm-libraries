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
on ``read < 0 or write < 0`` (``num_tokens > 1``: the whole chain on
``read < 0``, and token ``t``'s snapshot store alone on
``write_indices[b, t] < 0``), so on a path that bypasses the host guard
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
    out_gate     : [B*T, >= Hv*Dv], row stride og_stride    dtype
                   (row b*T + t is token t of sequence b; T = num_tokens)
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
``dpp_reduce`` offsets 4 and 8 use DPP row mirrors instead), so the
unfused kernel allocates no LDS (``fuse_out_norm`` adds a ``num_warps``-float
LDS reduction when ``num_warps > 1``, ``num_tokens * num_warps`` floats for a
chain; ALGORITHM.md section 4.9). Dispatch
selects this tile from the gfx950 GDN registry: `auto` runs one static tile
and knob stack per case (``dispatch.gdn.axes.AUTO_DEFAULTS``: gate kind x
fused x state width), while an explicit `spec_id` pins a tile.

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
from dataclasses import dataclass
from functools import partial
from typing import Dict, Literal, Tuple, get_args

from rocke.helpers.activations import LN2, LOG2E, SOFTPLUS_THRESHOLD
from rocke.core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType, TemporalHint
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
    "MAX_TOKENS",
    "NORM_GATE_ACTS",
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
# The ``norm_gate_act`` override of that default: a model whose GDN layer gates
# its RMSNorm with sigmoid (Qwen3.8-Flash-Next's ``output_gate_type``) sets it
# explicitly. "auto" is the per-gate-kind default above.
NORM_GATE_ACTS = ("auto", "silu", "sigmoid")
_NORM_GATE_TOKEN = {"silu": "ngsilu", "sigmoid": "ngsig"}
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
# Most tokens per sequence one launch chains (num_tokens): the state stays in
# registers across tokens and every token's q/k/v and gates load up front, so
# the bound keeps that footprint finite. MTP verify needs draft count + 1.
MAX_TOKENS = 8


def _state_ir_type(state_dtype: str):
    return F32 if state_dtype == "f32" else io_ir_type(state_dtype)


def _load_state_f32(b: IRBuilder, ptr, idx, *, state_dtype: str, n: int, hint: str):
    """Load ``n`` state elements as f32 values with cache policy ``hint``.

    The 16-bit states go through the same helper as before f32 existed, and
    ``hint="default"`` adds no metadata, so their emitted IR is unchanged.
    """
    if state_dtype == "f32":
        v = b.global_load_vN(
            ptr, idx, F32, n, align=_VEC_ALIGN, temporal_hint=_TEMPORAL[hint]
        )
        return [b.vec_extract(v, i) for i in range(n)]
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
    with ``WTK`` 4 or 8 and no ``waves_per_eu`` floor they fit without a new
    spill, so those tiles load first. Elsewhere the exception stays: at the
    other ``WTK``, at ``DK = DV = 256`` and under a ``waves_per_eu`` cap
    hoisting adds spills, and the other head dims are not characterized.
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
    # In-group sum (wsum): run the xor-4 / xor-8 stages on DPP row_half_mirror
    # / row_mirror instead of ds_swizzle. The sum is the same: the stages run
    # in ascending order, so when a mirror runs every lane of a quad (half-row)
    # already holds the quad (half-row) sum. Stages above 8 keep ds_swizzle.
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
    # Tokens per sequence in one launch (the MTP verify step scores T tokens
    # of each sequence at once). Token t starts from token t - 1's state,
    # carried in registers; each token's output goes to out[b, t] and its
    # post-state to its own pool slot write_indices[b, t] (the per-token
    # snapshots a speculative step rolls back to). The q/k/v, a and b
    # tensors gain the token axis after the batch axis. 1 is single-token
    # decode, whose ABI and emitted code are unchanged.
    num_tokens: int = 1
    # fuse_out_norm's gate activation: "auto" follows the gate kind
    # (OUT_GATE_ACTIVATION); "silu" / "sigmoid" override it. Inert without
    # fuse_out_norm, and an override equal to the default emits the default
    # kernel under the default name.
    norm_gate_act: str = "auto"
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
        """``"silu"`` or ``"sigmoid"``: the fused output norm's gate activation
        (``norm_gate_act``, or the gate kind's default when it is "auto")."""
        if self.norm_gate_act != "auto":
            return self.norm_gate_act
        return OUT_GATE_ACTIVATION[self.gate_kind]

    @property
    def rows_per_lane(self) -> int:
        """State rows each lane owns in the warp-tiled body (WTV_ITERS)."""
        lanes_v = (self.wave_size // self.warp_threads_k) * self.num_warps
        return (self.head_v_dim // self.blocks_per_v_dim) // lanes_v

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
        if self.num_tokens != 1:
            parts += (f"t{self.num_tokens}",)
        if (
            self.fuse_out_norm
            and self.out_gate_activation != OUT_GATE_ACTIVATION[self.gate_kind]
        ):
            act = self.out_gate_activation
            # .get: a malformed value still names (is_valid_spec rejects it)
            parts += (_NORM_GATE_TOKEN.get(act, f"ng{act}"),)
        return kernel_name_join(
            self.name,
            *parts,
            flags={
                "l2": self.use_qk_l2norm,
                "s": self.simple,
                "cv": self.fuse_conv,
                "rn": self.fuse_out_norm,
                "dppr": self.dpp_reduce and dpp_reduce_applies(self),
                "xcd": self.xcd_remap and xcd_remap_applies(self),
                "sr": self.stream_rows and stream_rows_applies(self),
                "ic": self.interleave_cols and interleave_cols_applies(self),
                "co": self.conv_once,
                "ngo": self.norm_gate_once,
                "olds": self.out_lds,
            },
        )


def is_valid_spec(spec: GdnDecodeSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Reject impossible/unsupported GDN decode configs before IR is built."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    # Type checks first: every rule below branches on these fields' truth or
    # arithmetic, so a wrong type would be coerced into a DIFFERENT valid spec
    # (``fuse_conv="false"`` is truthy and swaps in the fused ABI) rather than
    # rejected.
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
            return False, f"{_field} must be a bool, got {_value!r}"
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
        "num_tokens",
    ):
        _value = getattr(spec, _field)
        if isinstance(_value, bool) or not isinstance(_value, numbers.Integral):
            return False, f"{_field} must be an integer, got {_value!r}"
    if isinstance(spec.lower_bound, bool) or not isinstance(
        spec.lower_bound, numbers.Real
    ):
        return False, f"lower_bound must be a real number, got {spec.lower_bound!r}"

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
    # Performance knobs (their flags are type-checked above). Each rule is
    # functional -- a value the emitter cannot build, or a knob that reads a
    # tensor or result its spec does not have -- except the one marked
    # NOT_YET_IMPLEMENTED. A tile-dependent knob that is merely inert on this
    # tile is legal and emits the knob-off kernel (see the *_applies
    # predicates).
    for _field in ("state_load_hint", "state_store_hint"):
        _value = getattr(spec, _field)
        if _value not in STATE_HINTS:
            return False, f"{_field} must be one of {STATE_HINTS}, got {_value!r}"
    if not 0 <= spec.waves_per_eu <= _MAX_WAVES_PER_EU:
        return False, (
            f"waves_per_eu must be in [0, {_MAX_WAVES_PER_EU}], "
            f"got {spec.waves_per_eu!r}"
        )
    if spec.norm_gate_act not in NORM_GATE_ACTS:
        return False, (
            f"norm_gate_act must be one of {NORM_GATE_ACTS}, got {spec.norm_gate_act!r}"
        )
    if not 1 <= spec.num_tokens <= MAX_TOKENS:
        return False, (
            f"num_tokens must be in [1, {MAX_TOKENS}], got {spec.num_tokens!r}"
        )
    # fuse_out_norm chains (each token's output normalized on its own); the
    # conv shift is a one-token in-place update, so fuse_conv does not.
    if spec.num_tokens > 1 and (spec.simple or spec.fuse_conv):
        return False, (
            "NOT_YET_IMPLEMENTED: num_tokens > 1 requires the warp-tiled path "
            "without fuse_conv (simple and fuse_conv off)"
        )
    if spec.conv_once and not spec.fuse_conv:
        return False, "conv_once requires fuse_conv"
    if spec.norm_gate_once and not spec.fuse_out_norm:
        return False, "norm_gate_once requires fuse_out_norm"
    if spec.out_lds and not spec.fuse_out_norm:
        return False, "out_lds requires fuse_out_norm"
    # An emitter coupling, not a functional limit (ALGORITHM.md §8, follow-up
    # 8): the products live in conv_once's LDS block, behind its barrier.
    if spec.norm_gate_once and not spec.conv_once:
        return False, (
            "NOT_YET_IMPLEMENTED: norm_gate_once requires conv_once (it shares "
            "conv_once's LDS block)"
        )
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
    G = spec.v_per_k_head
    BS = spec.block_size
    scale = 1.0 / math.sqrt(DK)

    # Contiguous row-major strides (element counts).
    Q_HN, Q_HK = HK * DK, DK  # query/key: [B,1,HK,DK]
    V_HN, V_HK = HV * DV, DV  # value/out: [B,1,HV,DV]
    S_POOL, S_HV, S_VR = HV * DV * DK, DV * DK, DK  # state: [pool,HV,DV,DK]
    ST_BYTES = _ELEM_BYTES[spec.state_dtype]

    io_ty = io_ir_type(spec.dtype)
    st_ty = _state_ir_type(spec.state_dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, _MAX_WAVES_PER_EU)

    Q = b.param(
        "query", PtrType(io_ty, "global"), noalias=True, readonly=True, align=_VEC_ALIGN
    )
    K = b.param(
        "key", PtrType(io_ty, "global"), noalias=True, readonly=True, align=_VEC_ALIGN
    )
    Vv = b.param(
        "value", PtrType(io_ty, "global"), noalias=True, readonly=True, align=_VEC_ALIGN
    )
    Ag = b.param("a", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Bg = b.param("b", PtrType(io_ty, "global"), noalias=True, readonly=True)
    # f32 in KDA mode: the per-channel gate is evaluated in f32 and KDA prefill
    # declares the same tensor f32, so the two families share one contract.
    DTB = b.param(
        "dt_bias",
        PtrType(F32 if spec.gate_kind == "kda" else io_ty, "global"),
        noalias=True,
        readonly=True,
    )
    ALOG = b.param("A_log", PtrType(F32, "global"), noalias=True, readonly=True)
    RIDX = b.param("read_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    WIDX = b.param("write_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    STATE = b.param("state", PtrType(st_ty, "global"), noalias=True, align=_VEC_ALIGN)
    OUT = b.param(
        "out", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=_VEC_ALIGN
    )
    _ = b.param("batch_size", I32)  # noqa: F841

    tid = b.thread_id_x()  # value-dim row this thread owns (0..DV-1)
    bidx = b.block_id_x()
    b_i = b.div(bidx, b.const_i32(HV))
    hv_i = b.mod(bidx, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(G))

    read_pool = b.global_load_i32(RIDX, b_i)
    write_pool = b.global_load_i32(WIDX, b_i)

    # Skip continuous-batching padding lanes (negative sentinel index).
    active = b.land(
        b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
    )
    with b.scf_if(active):
        # exp/sigmoid from exp2/log2/rcp; see _exp_f32 for the clamp.
        exp_f32 = partial(_exp_f32, b)
        sigmoid = partial(_sigmoid, b)

        def log1p_f32(x):
            return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))

        # ---- load q,k rows [DK] -> f32 registers ----
        q_base = b.add(b.mul(b_i, b.const_i32(Q_HN)), b.mul(hk_i, b.const_i32(Q_HK)))
        qv, kv = [], []
        for c in range(0, DK, STATE_VEC):
            off = b.add(q_base, b.const_i32(c))
            qv += load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=STATE_VEC)
            kv += load_vec_as_f32(b, K, off, dtype=spec.dtype, n=STATE_VEC)

        # ---- L2 normalize q (and *scale), k ----
        if spec.use_qk_l2norm:
            sum_q2 = tree_reduce(b, b.fadd, [b.fmul(x, x) for x in qv])
            sum_k2 = tree_reduce(b, b.fadd, [b.fmul(x, x) for x in kv])
            inv_q = b.rsqrt(b.fadd(sum_q2, b.const_f32(NORM_EPS)))
            inv_k = b.rsqrt(b.fadd(sum_k2, b.const_f32(NORM_EPS)))
            sq = b.fmul(inv_q, b.const_f32(scale))
            qn = [b.fmul(x, sq) for x in qv]
            kn = [b.fmul(x, inv_k) for x in kv]
        else:
            qn = [b.fmul(x, b.const_f32(scale)) for x in qv]
            kn = kv

        # ---- gates (per value head) ----
        # The GDN arm below is the original emission, verbatim and in its
        # original order -- moved into a branch, not rewritten. Order matters:
        # these calls append ops to the IR, so hoisting even a shared load out
        # of the arm would reorder GDN's instructions and move every golden
        # hash pinned against it. Duplicating two loads across the arms is the
        # cheap side of that trade.
        a_idx = b.add(b.mul(b_i, b.const_i32(HV)), hv_i)

        if spec.gate_kind == "gdn":
            ra = load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype)
            rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
            rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
            ral = b.global_load_f32(ALOG, hv_i)  # A_log is fp32

            x = b.fadd(ra, rdt)
            sp = b.select(
                b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
                x,
                log1p_f32(exp_f32(x)),
            )
            decay = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
            beta = sigmoid(rb)
        else:
            # KDA: one decay per K channel.
            #   log_decay[d] = lower_bound * sigmoid(exp(A_log[h]) * (g[d] + dt_bias[h,d]))
            # exp(A_log[h]) is per head, so it is hoisted out of the channel loop.
            # This thread owns a whole state row, so it needs the full DK extent.
            rb = load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype)
            ral = b.global_load_f32(ALOG, hv_i)  # per head in both gate kinds
            beta = sigmoid(rb)
            exp_alog = exp_f32(ral)
            g_base = b.add(
                b.mul(b_i, b.const_i32(HV * DK)), b.mul(hv_i, b.const_i32(DK))
            )
            dtb_base = b.mul(hv_i, b.const_i32(DK))
            decay = []
            for c in range(0, DK, STATE_VEC):
                gv = load_vec_as_f32(
                    b, Ag, b.add(g_base, b.const_i32(c)), dtype=spec.dtype, n=STATE_VEC
                )
                # dt_bias is f32, so load_vec_as_f32 (a 16-bit ingest helper)
                # does not apply; global_load_vN lowers STATE_VEC f32 values to
                # one vector load rather than STATE_VEC scalar ones.
                dtvec = b.global_load_vN(
                    DTB,
                    b.add(dtb_base, b.const_i32(c)),
                    F32,
                    STATE_VEC,
                    align=_KDA_DT_BIAS_ALIGN,
                )
                dtv = [b.vec_extract(dtvec, j) for j in range(STATE_VEC)]
                for j in range(STATE_VEC):
                    if spec.fuse_gate:
                        inner = b.fmul(exp_alog, b.fadd(gv[j], dtv[j]))
                        sig = sigmoid(inner)
                        log_decay = b.fmul(b.const_f32(spec.lower_bound), sig)
                    else:
                        # `a` already carries natural-log-domain decay; only the
                        # exponential remains.
                        log_decay = gv[j]
                    decay.append(exp_f32(log_decay))

        # ---- dot(k_hat, q_hat) (scalar, redundant per thread) ----
        dot_kq = tree_reduce(b, b.fadd, [b.fmul(kn[j], qn[j]) for j in range(DK)])

        # ---- load state row t=tid: state[read_pool, hv, tid, 0:DK] ----
        # The pool base (read_pool * S_POOL) overflows i32 once the pool holds
        # >=4096 slots, so advance the pointer by a 64-bit byte offset and keep
        # the in-slot index (< S_POOL) in i32.
        state_r = b.global_ptr_add(
            STATE, b.mul(b.sext(read_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        rs_base = b.add(b.mul(hv_i, b.const_i32(S_HV)), b.mul(tid, b.const_i32(S_VR)))
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
        # Gated forget. A scalar decay broadcasts over the row; a per-channel
        # decay zips with it -- `sv` and `decay` are both indexed by K channel,
        # in the same order, so position i of each is the same channel.
        if spec.gate_kind == "gdn":
            sv = [b.fmul(s, decay) for s in sv]
        else:
            sv = [b.fmul(s, d) for s, d in zip(sv, decay)]

        # ---- S_row . k_hat  and  S_row . q_hat ----
        sum_hk = tree_reduce(b, b.fadd, [b.fmul(sv[j], kn[j]) for j in range(DK)])
        sum_hq = tree_reduce(b, b.fadd, [b.fmul(sv[j], qn[j]) for j in range(DK)])

        # ---- delta value + read-out for this value-dim ----
        v_idx = b.add(
            b.add(b.mul(b_i, b.const_i32(V_HN)), b.mul(hv_i, b.const_i32(V_HK))), tid
        )
        rv = load_scalar_as_f32(b, Vv, v_idx, dtype=spec.dtype)
        v_new = b.fmul(b.fsub(rv, sum_hk), beta)
        out_val = b.fadd(sum_hq, b.fmul(v_new, dot_kq))
        store_scalar_from_f32(b, OUT, v_idx, out_val, dtype=spec.dtype)

        # ---- rank-1 state write: S_row += k_hat * v_new ----
        state_w = b.global_ptr_add(
            STATE, b.mul(b.sext(write_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        ws_base = b.add(b.mul(hv_i, b.const_i32(S_HV)), b.mul(tid, b.const_i32(S_VR)))
        new_s = [b.fma(kn[j], v_new, sv[j]) for j in range(DK)]
        for c in range(0, DK, STATE_VEC):
            vec = _pack_state(b, new_s[c : c + STATE_VEC], state_dtype=spec.state_dtype)
            _store_state(
                b,
                state_w,
                b.add(ws_base, b.const_i32(c)),
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


def _build_warp_tiled(spec: GdnDecodeSpec) -> KernelDef:
    """v2: warp-tiled. One CTA per (seq, value-head); the
    ``head_v_dim x head_k_dim`` state is distributed across the block's warps
    (WTV x WTK lanes, VPT values/lane), and the K-reductions (L2 norms,
    dot(k,q), S.k, S.q) are folded once per WTK-lane group via shuffle-xor,
    eliminating v1's per-thread redundant compute and register pressure.
    """
    HK, HV = spec.num_k_heads, spec.num_v_heads
    DK, DV = spec.head_k_dim, spec.head_v_dim
    G = spec.v_per_k_head
    WAVE = spec.wave_size
    WTK = spec.warp_threads_k
    WTV = WAVE // WTK
    NW = spec.num_warps
    VPT = STATE_VEC
    BS = NW * WAVE
    WARP_TILE_K = WTK * VPT
    WTK_ITERS = DK // WARP_TILE_K
    WGROUP_V = NW * WTV
    BPV = spec.blocks_per_v_dim
    TILE_V = DV // BPV
    WTV_ITERS = spec.rows_per_lane  # TILE_V // WGROUP_V
    scale = 1.0 / math.sqrt(DK)
    shfl = [1 << i for i in range(WTK.bit_length() - 1)]  # [1,2,4] for WTK=8
    DPP = spec.dpp_reduce and dpp_reduce_applies(spec)

    Q_HN, Q_HK = HK * DK, DK
    V_HN, V_HK = HV * DV, DV
    S_POOL, S_HV, S_VR = HV * DV * DK, DV * DK, DK
    ST_BYTES = _ELEM_BYTES[spec.state_dtype]
    CONV, RN = spec.fuse_conv, spec.fuse_out_norm
    # A single wave with fuse_conv may keep some inputs late to avoid a spill
    # (late_row_inputs; ALGORITHM.md §4.3, "One exception").
    LATE_ROW_INPUTS = late_row_inputs(spec)
    CONV_DIM = spec.conv_dim  # packed [q | k | v] channels
    IO_BYTES = _ELEM_BYTES[spec.dtype]  # the conv state carries the I/O dtype
    CST_SLOT_BYTES = CONV_DIM * CONV_TAPS * IO_BYTES  # one conv-state pool slot
    ONCE = spec.conv_once
    # K-column map. Each lane owns K_PIECES runs of K_RUN consecutive columns
    # per K chunk; run p of lane l starts at chunk column p * WTK * K_RUN +
    # l * K_RUN, so slot i of a lane's chunk is always column
    # k_offs[ki][i // K_RUN] + i % K_RUN, for the state and every operand.
    # Default: one run of VPT. interleave_cols: 16 B runs of the f32 state.
    K_RUN = VPT
    if spec.interleave_cols and interleave_cols_applies(spec):
        K_RUN = _VEC_ALIGN // ST_BYTES
    K_PIECES = VPT // K_RUN
    # out_lds: OUT_W consecutive output rows per storing lane (2, 4 or 8, so
    # the DV / OUT_W storing lanes fit one wave up to DV = 8 * WAVE), in
    # OUT_ROUNDS passes over the workgroup.
    OLDS = spec.out_lds
    OUT_W = 2 if DV <= 2 * WAVE else 4 if DV <= 4 * WAVE else 8
    N_OUT = DV // OUT_W
    OUT_ROUNDS = -(-N_OUT // BS)

    io_ty = io_ir_type(spec.dtype)
    st_ty = _state_ir_type(spec.state_dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, _MAX_WAVES_PER_EU)

    def vec_param(name, ty, **kw):
        """A pointer param whose base the 16 B vector accesses rely on. Attr
        order is noalias, access, align, as before this helper existed."""
        return b.param(
            name, PtrType(ty, "global"), noalias=True, **kw, align=_VEC_ALIGN
        )

    if CONV:
        QKV = vec_param("mixed_qkv", io_ty, readonly=True)
        Q = K = Vv = QKV
    else:
        Q = vec_param("query", io_ty, readonly=True)
        K = vec_param("key", io_ty, readonly=True)
        Vv = vec_param("value", io_ty, readonly=True)
    Ag = b.param("a", PtrType(io_ty, "global"), noalias=True, readonly=True)
    Bg = b.param("b", PtrType(io_ty, "global"), noalias=True, readonly=True)
    DTB = b.param(
        "dt_bias",
        PtrType(F32 if spec.gate_kind == "kda" else io_ty, "global"),
        noalias=True,
        readonly=True,
    )
    ALOG = b.param("A_log", PtrType(F32, "global"), noalias=True, readonly=True)
    RIDX = b.param("read_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    WIDX = b.param("write_indices", PtrType(I32, "global"), noalias=True, readonly=True)
    STATE = vec_param("state", st_ty)
    OUT = vec_param("out", io_ty, writeonly=True)
    if CONV:
        CST = vec_param("conv_state", io_ty)
        CW = vec_param("conv_weight", F32, readonly=True)
    if RN:
        OG = b.param("out_gate", PtrType(io_ty, "global"), noalias=True, readonly=True)
        NWT = b.param(
            "norm_weight", PtrType(F32, "global"), noalias=True, readonly=True
        )
    batch_size = b.param("batch_size", I32)
    if CONV:
        qkv_stride = b.param("qkv_stride", I32)
    if RN:
        og_stride = b.param("og_stride", I32)
        norm_eps = b.param("norm_eps", F32)
        # one f32 partial per wave (and per token) for the cross-wave sum of
        # squares
        lds_rn = (
            b.smem_alloc_f32([NW * spec.num_tokens], name_hint="rn_partials")
            if NW > 1
            else None
        )
    if ONCE:
        # Post-SiLU f32 conv outputs [q DK | k DK | v DV], then (KDA) the DK
        # per-channel decays, then (norm_gate_once) the DV per-row products
        # gate(out_gate) * norm_weight; one slot per channel, written once,
        # read by all.
        N_CONV = 2 * DK + DV
        N_DEC = DK if spec.gate_kind == "kda" else 0
        N_GW = DV if spec.norm_gate_once else 0
        GW0 = N_CONV + N_DEC
        N_ONCE = GW0 + N_GW
        lds_once = b.smem_alloc_f32([N_ONCE], name_hint="conv_once")
    if OLDS:
        # each row's pre-norm f32 output (per token), written by its k-lane-0
        # owner
        lds_out = b.smem_alloc_f32([DV * spec.num_tokens], name_hint="out_rows")

    tid = b.thread_id_x()
    bidx = b.block_id_x()
    if spec.xcd_remap and xcd_remap_applies(spec):
        bidx = _xcd_remap(b, bidx, b.mul(batch_size, b.const_i32(HV * BPV)))
    b_hv_i = b.div(bidx, b.const_i32(BPV))
    tile_v_start = b.mul(b.mod(bidx, b.const_i32(BPV)), b.const_i32(TILE_V))
    b_i = b.div(b_hv_i, b.const_i32(HV))
    hv_i = b.mod(b_hv_i, b.const_i32(HV))
    hk_i = b.div(hv_i, b.const_i32(G))
    w_tid = b.mod(tid, b.const_i32(WAVE))
    wid = b.div(tid, b.const_i32(WAVE))
    k_lane = b.mod(w_tid, b.const_i32(WTK))
    v_lane = b.div(w_tid, b.const_i32(WTK))
    warp_k_start = b.mul(k_lane, b.const_i32(K_RUN))
    gv_start = b.add(b.mul(wid, b.const_i32(WTV)), v_lane)

    # num_tokens: write_indices is [B, T]; token t's post-state goes to slot
    # write_pools[t]. T == 1 emits the single-token code unchanged.
    T = spec.num_tokens
    read_pool = b.global_load_i32(RIDX, b_i)
    if T == 1:
        write_pools = [b.global_load_i32(WIDX, b_i)]
    else:
        w0 = b.mul(b_i, b.const_i32(T))
        write_pools = [
            b.global_load_i32(WIDX, b.add(w0, b.const_i32(t)) if t else w0)
            for t in range(T)
        ]
    write_pool = write_pools[0]
    if T == 1:
        active = b.land(
            b.cmp_ge(read_pool, b.const_i32(0)), b.cmp_ge(write_pool, b.const_i32(0))
        )
    else:
        # A chain is padding only when its read slot is: every token's
        # snapshot store (token 0's included) is gated on its own slot, so a
        # -1 at write_indices[b, t] skips that one store and nothing else.
        active = b.cmp_ge(read_pool, b.const_i32(0))
    with b.scf_if(active):

        exp_f32 = partial(_exp_f32, b)
        sigmoid = partial(_sigmoid, b)
        silu = partial(_silu, b)

        def log1p_f32(x):
            return b.fmul(b.log2(b.fadd(b.const_f32(1.0), x)), b.const_f32(LN2))

        def tok_row(t):
            """Row of token ``t`` of this sequence in a ``[B, T, ...]`` tensor;
            the sequence row itself (no op emitted) when ``T == 1``."""
            if T == 1:
                return b_i
            return b.add(b.mul(b_i, b.const_i32(T)), b.const_i32(t))

        def wsum(v):  # xor-butterfly sum over the WTK-lane group (broadcast in-group)
            # xor 1/2 via quad_perm (VALU DPP: no LDS crossbar / no lgkmcnt(0) stall);
            # xor 4 crosses the 4-lane quad so it stays on ds_swizzle, unless
            # dpp_reduce: then xor 4 / xor 8 use DPP row_half_mirror /
            # row_mirror. A mirror pairs lane i with 7 - i (15 - i), not i ^ 4
            # (i ^ 8), but the stages run in ascending order: by the time a
            # mirror runs, every lane of a quad (half-row) holds that quad's
            # (half-row's) sum, so both pairings add the same two values.
            # On gfx9 these are also exactly b.dpp_xor(v, 7) / b.dpp_xor(v, 15)
            # (same llvm.amdgcn.update.dpp control word and masks; checked at
            # llvm20/22/23, the lowered IR differs only in the SSA name hint).
            # Switching to dpp_xor would re-name values in every dpp_reduce
            # golden, so it is left for the upstreaming decision on
            # tile.dpp_row_mirror.
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
            for off in shfl:
                if off <= 2:
                    v = b.fadd(v, b.warp_shuffle_xor_quad(v, off))
                elif DPP and off <= 8:
                    m = b.dpp_row_mirror(b.bitcast(v, I32), half=off == 4)
                    v = b.fadd(v, b.bitcast(m, F32))
                else:
                    v = b.fadd(v, b.warp_shuffle_xor(v, off))
            return v

        # Fusion helpers (fuse_conv / fuse_out_norm). They emit nothing unless
        # called, and only the fused paths below call them.
        def conv(h, w, x):  # h: CONV_TAPS taps oldest-first, w: CONV_WIDTH weights
            return tree_reduce(
                b, b.fadd, [b.fmul(t, wt) for t, wt in zip(list(h) + [x], w)]
            )

        def conv_taps(ch, n):
            """CONV_TAPS taps x n channels from [slot, channel, tap]: CONV_TAPS
            contiguous n-element vectors."""
            taps = []
            for j in range(CONV_TAPS):
                taps += load_vec_as_f32(
                    b,
                    cst_r,
                    b.add(b.mul(ch, b.const_i32(CONV_TAPS)), b.const_i32(n * j)),
                    dtype=spec.dtype,
                    n=n,
                )
            return [taps[CONV_TAPS * i : CONV_TAPS * (i + 1)] for i in range(n)]

        def conv_weights(ch, n):
            """CONV_WIDTH weights x n channels from [channel, tap]: CONV_WIDTH
            contiguous n-float vectors. The param promises 16 B, so a 32 B
            load claims 16 B."""
            ws = []
            for j in range(CONV_WIDTH):
                wv = b.global_load_vN(
                    CW,
                    b.add(b.mul(ch, b.const_i32(CONV_WIDTH)), b.const_i32(n * j)),
                    F32,
                    n,
                    align=_VEC_ALIGN,
                )
                ws += [b.vec_extract(wv, i) for i in range(n)]
            return [ws[CONV_WIDTH * i : CONV_WIDTH * (i + 1)] for i in range(n)]

        # Per-head bases and the LDS reader, one helper each. They emit their
        # instructions at each call, so every caller sees the same expression.
        def out_head_row(t=0):  # first element of this head's [DV] out / value row
            return b.add(
                b.mul(tok_row(t), b.const_i32(V_HN)), b.mul(hv_i, b.const_i32(V_HK))
            )

        def og_head_row(t=0):  # first element of this head's [DV] out_gate row
            return b.add(b.mul(tok_row(t), og_stride), b.mul(hv_i, b.const_i32(DV)))

        def conv_slot(pool):  # conv-state slot ``pool`` ([C_conv, CONV_TAPS])
            return b.global_ptr_add(
                CST, b.mul(b.sext(pool, I64), b.const_i64(CST_SLOT_BYTES))
            )

        def lds_f32(buf, idx, n):  # n contiguous f32 slots, up to 4 per load
            m = min(n, 4)
            vecs = [
                b.smem_load_vN_f32(
                    buf, b.add(idx, b.const_i32(m * q)) if q else idx, n=m
                )
                for q in range(n // m)
            ]
            return [b.vec_extract(v, i) for v in vecs for i in range(m)]

        # Every global load is issued before any math; ALGORITHM.md §4.3 ("Why
        # loads first") gives the reason and the register cost. The K-chunk
        # lane offsets, V rows and state row bases are computed once here and
        # shared by every load and store, and each per-head base above is one
        # helper, so a K channel or a V row has one index expression in the
        # whole kernel. k_offs[ki][p] is the first column of this lane's run p
        # in K chunk ki (one run unless interleave_cols).
        k_offs = [
            [
                b.add(warp_k_start, b.const_i32(ki * WARP_TILE_K + p * WTK * K_RUN))
                for p in range(K_PIECES)
            ]
            for ki in range(WTK_ITERS)
        ]
        v_rows = [
            b.add(tile_v_start, b.add(gv_start, b.const_i32(vi * WGROUP_V)))
            for vi in range(WTV_ITERS)
        ]
        s_rows = [
            b.add(b.mul(hv_i, b.const_i32(S_HV)), b.mul(v_row, b.const_i32(S_VR)))
            for v_row in v_rows
        ]

        # Per token t: ras[t] / rbs[t] (and the KDA gvss[t]); dt_bias and A_log
        # are per head and load once, with token 0's gate inputs.
        ras, rbs, gvss = [], [], []
        for t in range(T):
            a_idx = b.add(b.mul(tok_row(t), b.const_i32(HV)), hv_i)
            if spec.gate_kind == "gdn":
                ras.append(load_scalar_as_f32(b, Ag, a_idx, dtype=spec.dtype))
                rbs.append(load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype))
                if t == 0:
                    rdt = load_scalar_as_f32(b, DTB, hv_i, dtype=spec.dtype)
                    ral = b.global_load_f32(ALOG, hv_i)
                continue
            rbs.append(load_scalar_as_f32(b, Bg, a_idx, dtype=spec.dtype))
            if t == 0:
                ral = b.global_load_f32(ALOG, hv_i)  # per head in both gate kinds
            g_row = b.add(
                b.mul(tok_row(t), b.const_i32(HV * DK)), b.mul(hv_i, b.const_i32(DK))
            )
            if t == 0:
                dtb_row = b.mul(hv_i, b.const_i32(DK))
            if not ONCE:  # conv_once computes the decays in its prologue
                # A K run of 8 f32 starts on a 32 B boundary of dt_bias, one of
                # 4 (interleave_cols) on a 16 B boundary.
                dt_align = min(_KDA_DT_BIAS_ALIGN, K_RUN * 4)
                gvs = []
                if t == 0:
                    dtvecs = []
                for ki in range(WTK_ITERS):
                    # k_offs[ki] is also the state load's offset, so slot i of
                    # this slice is the same K channel as slot i of the state.
                    gv, dts = [], []
                    for koff in k_offs[ki]:
                        gv += load_vec_as_f32(
                            b, Ag, b.add(g_row, koff), dtype=spec.dtype, n=K_RUN
                        )
                        if t == 0:
                            dts.append(
                                b.global_load_vN(
                                    DTB,
                                    b.add(dtb_row, koff),
                                    F32,
                                    K_RUN,
                                    align=dt_align,
                                )
                            )
                    gvs.append(gv)
                    if t == 0:
                        dtvecs.append(dts)
                gvss.append(gvs)
        if T == 1:
            ra = ras[0] if spec.gate_kind == "gdn" else None
            rb = rbs[0]
            if spec.gate_kind == "kda" and not ONCE:
                gvs = gvss[0]

        # this lane's q,k K-chunks -> f32
        qn = [None] * WTK_ITERS
        kn = [None] * WTK_ITERS
        if CONV:
            # q/k/v come from one packed row: q at channel hk*DK, k one q-span
            # later, v after both. The conv taps live at the read slot.
            qkv_row = b.mul(b_i, qkv_stride)
            qk_base = b.add(qkv_row, b.mul(hk_i, b.const_i32(DK)))
            k_shift = b.const_i32(HK * DK)
            cst_r = conv_slot(read_pool)
        if CONV and not ONCE:
            q_taps, k_taps, q_w, k_w = {}, {}, {}, {}
            # q conv channel of each K run; the k channel is one q-span later
            qchs = [
                [b.add(b.mul(hk_i, b.const_i32(DK)), koff) for koff in k_offs[ki]]
                for ki in range(WTK_ITERS)
            ]
            for ki in range(WTK_ITERS):
                qn[ki], kn[ki] = [], []
                q_taps[ki], q_w[ki], k_taps[ki], k_w[ki] = [], [], [], []
                for p in range(K_PIECES):
                    off = b.add(qk_base, k_offs[ki][p])
                    qn[ki] += load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=K_RUN)
                    kn[ki] += load_vec_as_f32(
                        b, K, b.add(off, k_shift), dtype=spec.dtype, n=K_RUN
                    )
                    qch = qchs[ki][p]
                    kch = b.add(qch, k_shift)
                    q_taps[ki] += conv_taps(qch, K_RUN)
                    q_w[ki] += conv_weights(qch, K_RUN)
                    k_taps[ki] += conv_taps(kch, K_RUN)
                    k_w[ki] += conv_weights(kch, K_RUN)
        elif ONCE:
            # One work item per LDS slot f = tid + it * BS: the conv channels
            # [q | k | v] first, then the KDA decay channels, then the
            # norm_gate_once rows. All loads are issued here, ahead of the state
            # loads, with the index clamped into range so no load sits behind
            # a branch; only the writes are guarded.
            cst_w = conv_slot(write_pool)
            # (first slot, end slot, channel minus slot) per conv segment
            segs = (
                (0, DK, b.mul(hk_i, b.const_i32(DK))),
                (
                    DK,
                    2 * DK,
                    b.add(b.mul(hk_i, b.const_i32(DK)), b.const_i32(HK * DK - DK)),
                ),
                (
                    2 * DK,
                    N_CONV,
                    b.add(
                        b.mul(hv_i, b.const_i32(DV)), b.const_i32(2 * HK * DK - 2 * DK)
                    ),
                ),
            )
            once_items = []
            for it in range(-(-N_ONCE // BS)):
                lo, hi = it * BS, (it + 1) * BS  # static range of f this round
                f = b.add(tid, b.const_i32(lo)) if lo else tid
                item = {"f": f, "hi": hi}
                if lo < N_CONV:
                    fc = b.smin(f, b.const_i32(N_CONV - 1)) if hi > N_CONV else f
                    live = [s for s in segs if s[0] < hi and s[1] > lo]
                    ch_off = live[-1][2]
                    for s in reversed(live[:-1]):
                        ch_off = b.select(b.cmp_lt(fc, b.const_i32(s[1])), s[2], ch_off)
                    ch = b.add(fc, ch_off)
                    item["ch"] = ch
                    item["x"] = load_scalar_as_f32(
                        b, QKV, b.add(qkv_row, ch), dtype=spec.dtype
                    )
                    tap0 = b.mul(ch, b.const_i32(CONV_TAPS))
                    item["taps"] = [
                        load_scalar_as_f32(
                            b, cst_r, b.add(tap0, b.const_i32(t)), dtype=spec.dtype
                        )
                        for t in range(CONV_TAPS)
                    ]
                    wv = b.global_load_vN(
                        CW,
                        b.mul(ch, b.const_i32(CONV_WIDTH)),
                        F32,
                        CONV_WIDTH,
                        align=_VEC_ALIGN,
                    )
                    item["w"] = [b.vec_extract(wv, i) for i in range(CONV_WIDTH)]
                if N_DEC and hi > N_CONV and lo < GW0:
                    d = b.sub(f, b.const_i32(N_CONV))
                    if lo < N_CONV:
                        d = b.smax(d, b.const_i32(0))
                    if hi > GW0:
                        d = b.smin(d, b.const_i32(N_DEC - 1))
                    item["a"] = load_scalar_as_f32(
                        b, Ag, b.add(g_row, d), dtype=spec.dtype
                    )
                    item["dt"] = b.global_load_f32(DTB, b.add(dtb_row, d))
                if N_GW and hi > GW0:
                    g = b.sub(f, b.const_i32(GW0))
                    if lo < GW0:
                        g = b.smax(g, b.const_i32(0))
                    if hi > N_ONCE:
                        g = b.smin(g, b.const_i32(N_GW - 1))
                    og_row0 = og_head_row()
                    item["og"] = load_scalar_as_f32(
                        b, OG, b.add(og_row0, g), dtype=spec.dtype
                    )
                    item["nw"] = b.global_load_f32(NWT, g)
                once_items.append(item)
        else:
            qns, kns = [], []
            for t in range(T):
                qk_base = b.add(
                    b.mul(tok_row(t), b.const_i32(Q_HN)),
                    b.mul(hk_i, b.const_i32(Q_HK)),
                )
                qn, kn = [None] * WTK_ITERS, [None] * WTK_ITERS
                for ki in range(WTK_ITERS):
                    qn[ki], kn[ki] = [], []
                    for koff in k_offs[ki]:
                        off = b.add(qk_base, koff)
                        qn[ki] += load_vec_as_f32(b, Q, off, dtype=spec.dtype, n=K_RUN)
                        kn[ki] += load_vec_as_f32(b, K, off, dtype=spec.dtype, n=K_RUN)
                qns.append(qn)
                kns.append(kn)
            qn, kn = qns[0], kns[0]

        # raw state tiles. The pool base overflows i32 for large pools, so
        # advance the pointer by a 64-bit byte offset once and keep the in-slot
        # index in i32.
        state_r = b.global_ptr_add(
            STATE, b.mul(b.sext(read_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )

        def state_tile(vi, ki):
            vals = []
            for koff in k_offs[ki]:
                off = b.add(s_rows[vi], koff)
                vals += _load_state_f32(
                    b,
                    state_r,
                    off,
                    state_dtype=spec.state_dtype,
                    n=K_RUN,
                    hint=spec.state_load_hint,
                )
            return vals

        if not LATE_ROW_INPUTS:
            s_raw = {
                (vi, ki): state_tile(vi, ki)
                for vi in range(WTV_ITERS)
                for ki in range(WTK_ITERS)
            }

        # this lane's v, one per V row (and per token)
        v_idxs = [[b.add(out_head_row(t), v_row) for v_row in v_rows] for t in range(T)]
        v_idx = v_idxs[0]
        if ONCE:
            pass  # v comes from LDS after the conv_once prologue
        elif CONV:
            # v channel of each owned row in the packed row / conv state
            vch = [
                b.add(
                    b.const_i32(2 * HK * DK),
                    b.add(b.mul(hv_i, b.const_i32(DV)), v_row),
                )
                for v_row in v_rows
            ]

            def v_raw_of(c):
                return load_scalar_as_f32(b, Vv, b.add(qkv_row, c), dtype=spec.dtype)

            def v_taps_of(c):
                return [
                    load_scalar_as_f32(
                        b,
                        cst_r,
                        b.add(b.mul(c, b.const_i32(CONV_TAPS)), b.const_i32(t)),
                        dtype=spec.dtype,
                    )
                    for t in range(CONV_TAPS)
                ]

            def v_w_of(c):
                wv = b.global_load_vN(
                    CW,
                    b.mul(c, b.const_i32(CONV_WIDTH)),
                    F32,
                    CONV_WIDTH,
                    align=_VEC_ALIGN,
                )
                return [b.vec_extract(wv, i) for i in range(CONV_WIDTH)]

            if LATE_ROW_INPUTS:
                v_raw, v_taps, v_w = [], [], []
            else:
                v_raw = [v_raw_of(c) for c in vch]
                v_taps = [v_taps_of(c) for c in vch]
                v_w = [v_w_of(c) for c in vch]
        else:
            rvs = [
                [load_scalar_as_f32(b, Vv, i, dtype=spec.dtype) for i in v_idxs[t]]
                for t in range(T)
            ]
            rv = rvs[0]
        if OLDS:
            # The storing lanes: lane j of round r stores rows
            # [OUT_W * j, OUT_W * (j + 1)), j = tid + r * BS; an index past the
            # last group is clamped so its loads stay in range (it stores
            # nothing). Without the norm_gate_once LDS slots they load their
            # rows' out_gate and norm_weight here, with the other loads.
            out_j = []
            for r in range(OUT_ROUNDS):
                j = b.add(tid, b.const_i32(r * BS)) if r else tid
                guard = (r + 1) * BS > N_OUT
                jc = b.smin(j, b.const_i32(N_OUT - 1)) if guard else j
                out_j.append((j, jc, guard))
            if not spec.norm_gate_once:
                # o_ogs[t]: token t's gates (T == 1: the single token's)
                og_rows = [og_head_row(t) for t in range(T)]
                o_ogs, o_nw = [[] for _ in range(T)], []
                for _, jc, _ in out_j:
                    row0 = b.mul(jc, b.const_i32(OUT_W))
                    for t in range(T):
                        o_ogs[t].append(
                            [
                                load_scalar_as_f32(
                                    b,
                                    OG,
                                    b.add(og_rows[t], b.add(row0, b.const_i32(e))),
                                    dtype=spec.dtype,
                                )
                                for e in range(OUT_W)
                            ]
                        )
                    nwv = b.global_load_vN(NWT, row0, F32, OUT_W, align=4)
                    o_nw.append([b.vec_extract(nwv, e) for e in range(OUT_W)])
                o_og = o_ogs[0]

        def norm_inputs():
            """Each owned row's out_gate and norm_weight (the per-lane norm)."""
            og_row = og_head_row()
            r_og = [
                load_scalar_as_f32(b, OG, b.add(og_row, v_row), dtype=spec.dtype)
                for v_row in v_rows
            ]
            return r_og, [b.global_load_f32(NWT, v_row) for v_row in v_rows]

        def norm_gates(t):
            """Token ``t``'s out_gate of each owned row (num_tokens > 1)."""
            og_row = og_head_row(t)
            return [
                load_scalar_as_f32(b, OG, b.add(og_row, v_row), dtype=spec.dtype)
                for v_row in v_rows
            ]

        # the per-lane norm: neither out_lds nor norm_gate_once supplies the
        # gate and weight
        LANE_NORM = RN and not OLDS and not spec.norm_gate_once
        if LANE_NORM and not LATE_ROW_INPUTS:
            r_og, r_nw = norm_inputs()
            # later tokens' gates (the weight is per row, shared by tokens)
            r_ogs = [r_og] + [norm_gates(t) for t in range(1, T)]

        def _chain_out_norm(outs_t):
            """fuse_out_norm for ``num_tokens > 1``: token ``t``'s gated RMSNorm
            over its DV outputs ``outs_t[t]`` (one value per owned row), then
            its stores. The T sums of squares share one cross-wave exchange:
            each wave's T partials go to LDS ahead of one barrier (with
            out_lds, the pre-norm rows go to LDS ahead of the same barrier).
            Every k-lane of a row holds the same output, so only k-lane 0
            contributes to the sums."""
            lane0 = b.cmp_eq(k_lane, b.const_i32(0))
            gate_act = silu if spec.out_gate_activation == "silu" else sigmoid
            if OLDS:
                with b.scf_if(lane0):
                    for t in range(T):
                        for vi in range(WTV_ITERS):
                            row = b.add(v_rows[vi], b.const_i32(t * DV))
                            b.smem_store_vN_f32(lds_out, [row], outs_t[t][vi], 1)
            partials = []
            for t in range(T):
                ss = tree_reduce(b, b.fadd, [b.fmul(o, o) for o in outs_t[t]])
                contrib = b.select(lane0, ss, b.const_f32(0.0))
                # block_size == wave_size: the in-wave butterfly only
                partials.append(
                    block_lds_reduce_with_wave_prologue(
                        b, contrib, lds_rn, tid, block_size=WAVE, wave_size=WAVE
                    )
                )
            if NW > 1:
                with b.scf_if(b.cmp_eq(w_tid, b.const_i32(0))):
                    for t in range(T):
                        slot = b.add(wid, b.const_i32(t * NW)) if t else wid
                        b.smem_store_vN_f32(lds_rn, [slot], partials[t], 1)
                b.sync()
                totals = [
                    tree_reduce(b, b.fadd, lds_f32(lds_rn, b.const_i32(t * NW), NW))
                    for t in range(T)
                ]
            else:
                totals = partials
                if OLDS:
                    b.sync_lds_only()  # one wave: no barrier to reuse
            rstds = [
                b.rsqrt(b.fadd(b.fmul(tot, b.const_f32(1.0 / DV)), norm_eps))
                for tot in totals
            ]
            for t in range(T):
                if OLDS:
                    out_row0 = out_head_row(t)
                    for r, (j, _, guard) in enumerate(out_j):

                        def store_group(r=r, j=j, t=t, out_row0=out_row0):
                            row0 = b.mul(j, b.const_i32(OUT_W))
                            o = lds_f32(
                                lds_out, b.add(row0, b.const_i32(t * DV)), OUT_W
                            )
                            ys = [
                                b.fmul(
                                    b.fmul(b.fmul(o[e], rstds[t]), o_nw[r][e]),
                                    gate_act(o_ogs[t][r][e]),
                                )
                                for e in range(OUT_W)
                            ]
                            store_vec(
                                b,
                                OUT,
                                b.add(out_row0, row0),
                                pack_f32_to(b, ys, dtype=spec.dtype),
                                n=OUT_W,
                            )

                        if guard:
                            with b.scf_if(b.cmp_lt(j, b.const_i32(N_OUT))):
                                store_group()
                        else:
                            store_group()
                else:
                    for vi in range(WTV_ITERS):
                        gate = gate_act(r_ogs[t][vi])
                        y = b.fmul(
                            b.fmul(b.fmul(outs_t[t][vi], rstds[t]), r_nw[vi]), gate
                        )
                        with b.scf_if(lane0):
                            store_scalar_from_f32(
                                b, OUT, v_idxs[t][vi], y, dtype=spec.dtype
                            )

        if T > 1:
            # num_tokens: the chain of T recurrence steps on one sequence. The
            # state of every owned (row, K chunk) stays in registers between
            # tokens; each token stores its output and its post-state snapshot.
            # No fuse_conv (is_valid_spec), so only the unfused q/k/v loads
            # exist. With fuse_out_norm each token's outputs are kept until
            # the chain ends and then normalized per token (one cross-wave
            # reduction for all T tokens).
            stream = spec.stream_rows and stream_rows_applies(spec)
            state = None  # (vi, ki) -> VPT f32 values after the previous token
            outs_t = [[] for _ in range(T)]  # fuse_out_norm: pre-norm outputs
            for t in range(T):
                if spec.gate_kind == "gdn":
                    x = b.fadd(ras[t], rdt)
                    sp = b.select(
                        b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
                        x,
                        log1p_f32(exp_f32(x)),
                    )
                    decay_t = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
                    beta_t = sigmoid(rbs[t])
                else:
                    beta_t = sigmoid(rbs[t])
                    exp_alog_t = exp_f32(ral)
                    decay_t = {}
                    for ki in range(WTK_ITERS):
                        dec = []
                        for i in range(VPT):
                            if spec.fuse_gate:
                                dt = b.vec_extract(dtvecs[ki][i // K_RUN], i % K_RUN)
                                inner = b.fmul(exp_alog_t, b.fadd(gvss[t][ki][i], dt))
                                log_decay = b.fmul(
                                    b.const_f32(spec.lower_bound), sigmoid(inner)
                                )
                            else:
                                log_decay = gvss[t][ki][i]
                            dec.append(exp_f32(log_decay))
                        decay_t[ki] = dec
                q_t, k_t = qns[t], kns[t]
                if spec.use_qk_l2norm:
                    pq = wsum(
                        tree_reduce(
                            b, b.fadd, [b.fmul(x, x) for row in q_t for x in row]
                        )
                    )
                    pk = wsum(
                        tree_reduce(
                            b, b.fadd, [b.fmul(x, x) for row in k_t for x in row]
                        )
                    )
                    sq = b.fmul(
                        b.rsqrt(b.fadd(pq, b.const_f32(NORM_EPS))), b.const_f32(scale)
                    )
                    inv_k = b.rsqrt(b.fadd(pk, b.const_f32(NORM_EPS)))
                    q_t = [[b.fmul(x, sq) for x in row] for row in q_t]
                    k_t = [[b.fmul(x, inv_k) for x in row] for row in k_t]
                else:
                    q_t = [[b.fmul(x, b.const_f32(scale)) for x in row] for row in q_t]
                dot_t = wsum(
                    tree_reduce(
                        b,
                        b.fadd,
                        [
                            b.fmul(k_t[ki][i], q_t[ki][i])
                            for ki in range(WTK_ITERS)
                            for i in range(VPT)
                        ],
                    )
                )
                sv_t = {}

                def decay_tok_row(vi, t=t, decay_t=decay_t, sv_t=sv_t, state=state):
                    for ki in range(WTK_ITERS):
                        vec = s_raw[(vi, ki)] if t == 0 else state[(vi, ki)]
                        if spec.gate_kind == "gdn":
                            sv_t[(vi, ki)] = [b.fmul(s, decay_t) for s in vec]
                        else:
                            sv_t[(vi, ki)] = [
                                b.fmul(s, dd) for s, dd in zip(vec, decay_t[ki])
                            ]

                if not stream:
                    for vi in range(WTV_ITERS):
                        decay_tok_row(vi)
                state_w_t = b.global_ptr_add(
                    STATE,
                    b.mul(b.sext(write_pools[t], I64), b.const_i64(S_POOL * ST_BYTES)),
                )
                # every token's snapshot store is gated on its own slot (the
                # chain itself is gated on the read slot only)
                w_ok = b.cmp_ge(write_pools[t], b.const_i32(0))
                new_state = {}
                for vi in range(WTV_ITERS):
                    if stream:
                        decay_tok_row(vi)
                    phk = wsum(
                        tree_reduce(
                            b,
                            b.fadd,
                            [
                                b.fmul(sv_t[(vi, ki)][i], k_t[ki][i])
                                for ki in range(WTK_ITERS)
                                for i in range(VPT)
                            ],
                        )
                    )
                    phq = wsum(
                        tree_reduce(
                            b,
                            b.fadd,
                            [
                                b.fmul(sv_t[(vi, ki)][i], q_t[ki][i])
                                for ki in range(WTK_ITERS)
                                for i in range(VPT)
                            ],
                        )
                    )
                    v_new = b.fmul(b.fsub(rvs[t][vi], phk), beta_t)
                    out_val = b.fadd(phq, b.fmul(v_new, dot_t))
                    if RN:
                        outs_t[t].append(out_val)  # normalized after the chain
                    else:
                        with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                            store_scalar_from_f32(
                                b, OUT, v_idxs[t][vi], out_val, dtype=spec.dtype
                            )
                    for ki in range(WTK_ITERS):
                        new_state[(vi, ki)] = [
                            b.fma(k_t[ki][i], v_new, sv_t[(vi, ki)][i])
                            for i in range(VPT)
                        ]

                    def store_row(vi=vi, new_state=new_state, state_w_t=state_w_t):
                        for ki in range(WTK_ITERS):
                            for p in range(K_PIECES):
                                run = new_state[(vi, ki)][p * K_RUN : (p + 1) * K_RUN]
                                _store_state(
                                    b,
                                    state_w_t,
                                    b.add(s_rows[vi], k_offs[ki][p]),
                                    _pack_state(b, run, state_dtype=spec.state_dtype),
                                    state_dtype=spec.state_dtype,
                                    n=K_RUN,
                                    hint=spec.state_store_hint,
                                )

                    if w_ok is None:
                        store_row()
                    else:
                        with b.scf_if(w_ok):
                            store_row()
                state = new_state
            if RN:
                _chain_out_norm(outs_t)
            return b.kernel

        # gates (per value head)
        if spec.gate_kind == "gdn":
            x = b.fadd(ra, rdt)
            sp = b.select(
                b.fcmp("ogt", x, b.const_f32(SOFTPLUS_THRESHOLD)),
                x,
                log1p_f32(exp_f32(x)),
            )
            decay = exp_f32(b.fneg(b.fmul(exp_f32(ral), sp)))
            beta = sigmoid(rb)
        else:
            # KDA: one decay per K channel, for the slice THIS lane owns.
            #
            # The state tile is keyed (vi, ki) -- V row and K chunk -- but a
            # channel's decay does not depend on which V row is being faded, so
            # the decay is keyed by ki alone and reused across all WTV_ITERS
            # rows. That is what holds the extra register cost to WTK_ITERS*VPT
            # values instead of multiplying with the state tile.
            beta = sigmoid(rb)
            exp_alog = exp_f32(ral)

            def kda_decay(a, dt):  # one channel's decay from its gate inputs
                if spec.fuse_gate:
                    inner = b.fmul(exp_alog, b.fadd(a, dt))
                    sig = sigmoid(inner)
                    log_decay = b.fmul(b.const_f32(spec.lower_bound), sig)
                else:
                    log_decay = a
                return exp_f32(log_decay)

            decay = {}
            if not ONCE:
                for ki in range(WTK_ITERS):
                    gv, dts = gvs[ki], dtvecs[ki]
                    # dt is extracted only on the fused gate, as before
                    decay[ki] = [
                        kda_decay(
                            gv[i],
                            (
                                b.vec_extract(dts[i // K_RUN], i % K_RUN)
                                if spec.fuse_gate
                                else None
                            ),
                        )
                        for i in range(VPT)
                    ]

        # fused RMSNorm output gate, per gate kind (OUT_GATE_ACTIVATION)
        norm_gate = silu if spec.out_gate_activation == "silu" else sigmoid

        if ONCE:
            # Prologue math: one work item per LDS slot, then one LDS barrier.
            for item in once_items:
                f, hi = item["f"], item["hi"]
                val = None
                if "ch" in item:
                    val = silu(conv(item["taps"], item["w"], item["x"]))
                if "a" in item:
                    dec = kda_decay(item["a"], item["dt"])
                    val = (
                        dec
                        if val is None
                        else b.select(b.cmp_lt(f, b.const_i32(N_CONV)), val, dec)
                    )
                if "og" in item:
                    gw = b.fmul(norm_gate(item["og"]), item["nw"])
                    val = (
                        gw
                        if val is None
                        else b.select(b.cmp_lt(f, b.const_i32(GW0)), val, gw)
                    )
                if hi > N_ONCE:
                    with b.scf_if(b.cmp_lt(f, b.const_i32(N_ONCE))):
                        b.smem_store_vN_f32(lds_once, [f], val, 1)
                else:
                    b.smem_store_vN_f32(lds_once, [f], val, 1)
                if "ch" in item:
                    # Single writer: the thread that read this channel's taps
                    # shifts them, so read slot == write slot needs no barrier.
                    def shift_taps(item=item):
                        base = b.mul(item["ch"], b.const_i32(CONV_TAPS))
                        shifted = list(item["taps"][1:]) + [item["x"]]
                        for t, tv in enumerate(shifted):
                            store_scalar_from_f32(
                                b,
                                cst_w,
                                b.add(base, b.const_i32(t)),
                                tv,
                                dtype=spec.dtype,
                            )

                    if hi > N_CONV:
                        with b.scf_if(b.cmp_lt(f, b.const_i32(N_CONV))):
                            shift_taps()
                    else:
                        shift_taps()
            # LDS-only drain: the state loads stay in flight across it.
            b.sync_lds_only()

            for ki in range(WTK_ITERS):
                qn[ki], kn[ki] = [], []
                if N_DEC:
                    decay[ki] = []
                for koff in k_offs[ki]:
                    qn[ki] += lds_f32(lds_once, koff, K_RUN)
                    kn[ki] += lds_f32(lds_once, b.add(koff, b.const_i32(DK)), K_RUN)
                    if N_DEC:
                        decay[ki] += lds_f32(
                            lds_once, b.add(koff, b.const_i32(N_CONV)), K_RUN
                        )
            rv = [
                lds_f32(lds_once, b.add(v_rows[vi], b.const_i32(2 * DK)), 1)[0]
                for vi in range(WTV_ITERS)
            ]

        if CONV and not ONCE:
            # conv1d + SiLU on the raw q/k/v; the raw values are kept, since
            # they become the newest conv tap.
            q_raw, k_raw = qn, kn
            qn = [
                [
                    silu(conv(q_taps[ki][i], q_w[ki][i], q_raw[ki][i]))
                    for i in range(VPT)
                ]
                for ki in range(WTK_ITERS)
            ]
            kn = [
                [
                    silu(conv(k_taps[ki][i], k_w[ki][i], k_raw[ki][i]))
                    for i in range(VPT)
                ]
                for ki in range(WTK_ITERS)
            ]
            rv = [
                silu(conv(v_taps[vi], v_w[vi], v_raw[vi])) for vi in range(len(v_raw))
            ]

        if spec.use_qk_l2norm:
            pq = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(qn[ki][i], qn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            pk = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(kn[ki][i], kn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            inv_q = b.rsqrt(b.fadd(pq, b.const_f32(NORM_EPS)))
            inv_k = b.rsqrt(b.fadd(pk, b.const_f32(NORM_EPS)))
            sq = b.fmul(inv_q, b.const_f32(scale))
            qn = [
                [b.fmul(qn[ki][i], sq) for i in range(VPT)] for ki in range(WTK_ITERS)
            ]
            kn = [
                [b.fmul(kn[ki][i], inv_k) for i in range(VPT)]
                for ki in range(WTK_ITERS)
            ]
        else:
            qn = [
                [b.fmul(qn[ki][i], b.const_f32(scale)) for i in range(VPT)]
                for ki in range(WTK_ITERS)
            ]

        dot_kq = wsum(
            tree_reduce(
                b,
                b.fadd,
                [
                    b.fmul(kn[ki][i], qn[ki][i])
                    for ki in range(WTK_ITERS)
                    for i in range(VPT)
                ],
            )
        )

        # decay the state tiles. decay[ki] covers the same K channels as state
        # chunk ki, in the same order, and is reused across every vi. The
        # LATE_ROW_INPUTS tile loads each chunk here, as the parent order did.
        # With stream_rows each row is decayed in its own iteration below
        # instead; with one row per lane there is nothing to stream, and the
        # emitted code is the default's.
        stream = spec.stream_rows and stream_rows_applies(spec)
        sv = {}

        def decay_row(vi):
            for ki in range(WTK_ITERS):
                vec = state_tile(vi, ki) if LATE_ROW_INPUTS else s_raw[(vi, ki)]
                if spec.gate_kind == "gdn":
                    sv[(vi, ki)] = [b.fmul(s, decay) for s in vec]
                else:
                    sv[(vi, ki)] = [b.fmul(s, d) for s, d in zip(vec, decay[ki])]

        if not stream:
            for vi in range(WTV_ITERS):
                decay_row(vi)

        state_w = b.global_ptr_add(
            STATE, b.mul(b.sext(write_pool, I64), b.const_i64(S_POOL * ST_BYTES))
        )
        outs = []  # per-row outputs the fused norm below needs
        for vi in range(WTV_ITERS):
            if stream:
                decay_row(vi)
            phk = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(sv[(vi, ki)][i], kn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            phq = wsum(
                tree_reduce(
                    b,
                    b.fadd,
                    [
                        b.fmul(sv[(vi, ki)][i], qn[ki][i])
                        for ki in range(WTK_ITERS)
                        for i in range(VPT)
                    ],
                )
            )
            if LATE_ROW_INPUTS and not ONCE:
                v_raw.append(v_raw_of(vch[vi]))
                v_taps.append(v_taps_of(vch[vi]))
                rv.append(silu(conv(v_taps[vi], v_w_of(vch[vi]), v_raw[vi])))

            # v_new is in-group uniform (rv, broadcast phk, beta) - no bcast.
            v_new = b.fmul(b.fsub(rv[vi], phk), beta)
            out_val = b.fadd(phq, b.fmul(v_new, dot_kq))
            if RN:
                outs.append(out_val)  # stored after the cross-wave norm below
            else:
                with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                    store_scalar_from_f32(b, OUT, v_idx[vi], out_val, dtype=spec.dtype)
            for ki in range(WTK_ITERS):
                for p in range(K_PIECES):
                    run = range(p * K_RUN, (p + 1) * K_RUN)
                    new = [b.fma(kn[ki][i], v_new, sv[(vi, ki)][i]) for i in run]
                    vec = _pack_state(b, new, state_dtype=spec.state_dtype)
                    off = b.add(s_rows[vi], k_offs[ki][p])
                    _store_state(
                        b,
                        state_w,
                        off,
                        vec,
                        state_dtype=spec.state_dtype,
                        n=K_RUN,
                        hint=spec.state_store_hint,
                    )

        if RN:
            if OLDS:
                # Each row's pre-norm output to LDS ahead of the reduction's
                # barrier, which then also publishes it to the storing lanes.
                with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                    for vi in range(WTV_ITERS):
                        b.smem_store_vN_f32(lds_out, [v_rows[vi]], outs[vi], 1)
            # Gated RMSNorm over the head's DV outputs; the gate activation
            # follows the gate kind (OUT_GATE_ACTIVATION, norm_gate). Every
            # k-lane of a row holds the same out_val, so only k-lane 0
            # contributes to the sum of squares. For NW > 1 the cross-wave
            # reduction issues a workgroup barrier after every wave has written
            # its partial, i.e. after every wave's tap reads (taps -> outs ->
            # partial).
            if LANE_NORM and LATE_ROW_INPUTS:
                r_og, r_nw = norm_inputs()
            ss = tree_reduce(b, b.fadd, [b.fmul(o, o) for o in outs])
            contrib = b.select(b.cmp_eq(k_lane, b.const_i32(0)), ss, b.const_f32(0.0))
            total = block_lds_reduce_with_wave_prologue(
                b, contrib, lds_rn, tid, block_size=BS, wave_size=WAVE
            )
            rstd = b.rsqrt(b.fadd(b.fmul(total, b.const_f32(1.0 / DV)), norm_eps))
            if OLDS:
                if NW == 1:
                    # one wave: the reduction has no barrier to reuse
                    b.sync_lds_only()

                out_row0 = out_head_row()
                for r, (j, _, guard) in enumerate(out_j):

                    def store_group(r=r, j=j):
                        row0 = b.mul(j, b.const_i32(OUT_W))
                        o = lds_f32(lds_out, row0, OUT_W)
                        if spec.norm_gate_once:
                            gw = lds_f32(lds_once, b.add(row0, b.const_i32(GW0)), OUT_W)
                            ys = [
                                b.fmul(b.fmul(o[e], rstd), gw[e]) for e in range(OUT_W)
                            ]
                        else:
                            ys = []
                            for e in range(OUT_W):
                                gate = norm_gate(o_og[r][e])
                                ys.append(
                                    b.fmul(b.fmul(b.fmul(o[e], rstd), o_nw[r][e]), gate)
                                )
                        store_vec(
                            b,
                            OUT,
                            b.add(out_row0, row0),
                            pack_f32_to(b, ys, dtype=spec.dtype),
                            n=OUT_W,
                        )

                    if guard:
                        with b.scf_if(b.cmp_lt(j, b.const_i32(N_OUT))):
                            store_group()
                    else:
                        store_group()
            else:
                for vi in range(WTV_ITERS):
                    if spec.norm_gate_once:
                        gw_idx = b.add(v_rows[vi], b.const_i32(GW0))
                        (gw,) = lds_f32(lds_once, gw_idx, 1)
                        y = b.fmul(b.fmul(outs[vi], rstd), gw)
                    else:
                        gate = norm_gate(r_og[vi])
                        y = b.fmul(b.fmul(b.fmul(outs[vi], rstd), r_nw[vi]), gate)
                    with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                        store_scalar_from_f32(b, OUT, v_idx[vi], y, dtype=spec.dtype)

        if CONV and not ONCE:
            # Shift the conv taps in place: drop the oldest, append the raw
            # current token. Other waves read the same q/k taps, so every wave
            # must be past its tap reads before any tap is overwritten. With
            # the norm on, its reduction barrier (above) already orders that;
            # otherwise one barrier is emitted here. One wave needs none.
            # (conv_once shifts each channel's taps in its prologue instead.)
            if NW > 1 and not RN:
                b.sync()
            cst_w = conv_slot(write_pool)
            # q/k channels: one writer group (wave 0, v-lane 0), each k-lane
            # writing its VPT channels x CONV_TAPS shifted taps, one K run at
            # a time.
            with b.scf_if(b.cmp_eq(gv_start, b.const_i32(0))):
                for ki in range(WTK_ITERS):
                    for p in range(K_PIECES):
                        qch = qchs[ki][p]
                        for ch, taps, raw in (
                            (qch, q_taps, q_raw),
                            (b.add(qch, b.const_i32(HK * DK)), k_taps, k_raw),
                        ):
                            shifted = []
                            for i in range(p * K_RUN, (p + 1) * K_RUN):
                                shifted += list(taps[ki][i][1:]) + [raw[ki][i]]
                            for j in range(CONV_TAPS):
                                store_vec(
                                    b,
                                    cst_w,
                                    b.add(
                                        b.mul(ch, b.const_i32(CONV_TAPS)),
                                        b.const_i32(K_RUN * j),
                                    ),
                                    pack_f32_to(
                                        b,
                                        shifted[K_RUN * j : K_RUN * (j + 1)],
                                        dtype=spec.dtype,
                                    ),
                                    n=K_RUN,
                                )
            # v channels: the k-lane-0 owner of each row
            with b.scf_if(b.cmp_eq(k_lane, b.const_i32(0))):
                for vi in range(WTV_ITERS):
                    base = b.mul(vch[vi], b.const_i32(CONV_TAPS))
                    for t, val in enumerate(list(v_taps[vi][1:]) + [v_raw[vi]]):
                        store_scalar_from_f32(
                            b, cst_w, b.add(base, b.const_i32(t)), val, dtype=spec.dtype
                        )

    return b.kernel


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
