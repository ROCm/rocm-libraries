# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/aotriton
#   commit:   b5e8cfb909e554c62081f56055b11d6ba3fd6bfa
#   path:     modules/flash/flyc/fmha_abi_gfx1201.py
#
# Reaches this package through the SDPA backward sweep. Differs from `../attention_rdna4/fmha_abi_gfx1201.py` in the package imports
# (`attention_bwd_cdna`) and in `prep_tensors`, which has no `v_heads_own` option (V's heads equal K's).


"""Host side of the gfx1201 attention ABI: the wire spec, and marshalling to it.

Everything the *caller* has to get right, in one place. Four kernels now share
this ABI -- the forward and the three backward ones -- and before this module
existed each carried its own copy of all of it. That is not merely repetitive:
`VarlenBits` is a **wire encoding**, and three copies of a wire encoding are
three chances for it to drift while every test still passes, because each
kernel is validated against its own copy.

**Not `fmha_common_gfx1201.py`.** That module is device-side: it emits IR, and
its central constraint is what the AST rewriter will and will not do to a
module-level function. Nothing here emits IR. These are plain Python functions
that run on the host before a launch, turning torch tensors and Python
arguments into the scalars and pointers the kernel signature wants. The two
concerns fail differently and are read at different times.

The three functions that take a build knob as their first argument
(`resolve_window`, `dropout_args`, `varlen_args`) are the only ones that are
not pure. They were closures over the builder's `const_expr` state; making the
dependency an argument is what let them move, and it also makes visible which
host behaviour a knob reaches -- something the closure form hid.
"""

from __future__ import annotations

import contextlib
import math
import weakref

from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha
from kernels.attention_bwd_cdna.philox import dropout_threshold

# `torch` and `torch.float32` are imported lazily, inside the three functions
# that need them, because the build venv must never have torch and neither use
# is reached by the AOT compile driver, which passes philox_seed=None.

import flydsl.compiler as flyc
import flydsl.expr as fx

__all__ = [
    "VARLEN_STACKED",
    "VARLEN_LENGTH_MAX",
    "VARLEN_LENGTH_CUMULATIVE",
    "VARLEN_LENGTH_INDIVIDUAL",
    "VARLEN_POSITION_IMPLIED",
    "VARLEN_POSITION_REUSE",
    "VARLEN_POSITION_ARRAY",
    "VARLEN_LSE_LAYOUT_HT",
    "VARLEN_LSE_LAYOUT_TH",
    "VARLEN_DENSE",
    "VARLEN_COMPACT_SIDE",
    "CAUSAL_SENTINEL",
    "varlen_bits",
    "varlen_compact",
    "varlen_padded",
    "varlen_strided",
    "varlen_seqused_k",
    "NULL_PTR",
    "ptr_arg",
    "strides_of",
    "row_tensor_arg",
    "resolve_scale",
    "prep_tensors",
    "lse_args",
    "resolve_window",
    "bias_args",
    "dropout_args",
    "dropout_outputs",
    "u64_scalar",
    "varlen_args",
    "run_compiled",
    "new_compiled_cache",
    "dtype_to_elem_type",
]


# Resolving in one place removes the class of bug rather than the
# instance.
CAUSAL_SENTINEL = {
    1: fmha.WINDOW_TOPLEFT,  # j <= i
    2: fmha.WINDOW_BOTRIGHT,  # j <= i + (seqlen_k - seqlen_q)
}

# ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
#
# One byte per side, decoded by the same kernel-side function twice, plus
# the LSE layout in byte 2. `0` is BHSD / MAX / IMPLIED on both sides with
# an (H, T) logsumexp -- the conventional dense case, and the default.
VARLEN_STACKED = 1
VARLEN_LENGTH_MAX = 0 << 1  # noqa: F841  (the table is the wire spec)
VARLEN_LENGTH_CUMULATIVE = 1 << 1
VARLEN_LENGTH_INDIVIDUAL = 2 << 1
VARLEN_POSITION_IMPLIED = 0 << 3
VARLEN_POSITION_REUSE = 1 << 3
VARLEN_POSITION_ARRAY = 2 << 3
# `_HT` is AOTriton's and this kernel's default: shape (H, T), T
# contiguous. `_TH` is Transformer Engine's (T, H).
VARLEN_LSE_LAYOUT_HT = 0 << 16
VARLEN_LSE_LAYOUT_TH = 1 << 16  # noqa: F841  (ditto)


def varlen_bits(q_side=0, k_side=0, lse_layout=VARLEN_LSE_LAYOUT_HT):
    """Assemble VarlenBits from per-side bytes."""
    for name, b in (("q_side", q_side), ("k_side", k_side)):
        if not 0 <= b <= 0xFF:
            raise ValueError(f"{name} must fit in a byte, got {b:#x}")
        if (b >> 3) & 3 == 1 and (b >> 1) & 3 != 1:
            # REUSE takes a *position* out of the length array, which is
            # only a position when the lengths are cumulative.
            raise ValueError(
                f"{name}={b:#04x}: POSITION=REUSE requires "
                f"LENGTH=CUMULATIVE (plan section 1, axis C)"
            )
        if (b >> 1) & 3 == 3 or (b >> 3) & 3 == 3:
            raise ValueError(f"{name}={b:#04x} uses a reserved code")
    return q_side | (k_side << 8) | lse_layout


VARLEN_DENSE = 0  # noqa: F841  (ditto)
VARLEN_COMPACT_SIDE = (
    VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_REUSE
)  # 0x0B
VARLEN_PADDED_SIDE = VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_IMPLIED
# 0x02

VARLEN_STRIDED_SIDE = (
    VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_ARRAY
)  # 0x13
VARLEN_SEQUSED_PACKED_SIDE = (
    VARLEN_STACKED | VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_ARRAY
)  # 0x15
VARLEN_SEQUSED_CACHE_SIDE = VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_IMPLIED  # 0x04


def varlen_compact(
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Classical packed varlen: 1THD tensors, `cu_seqlens` for both roles.

    `seqinfo_?1` is deliberately **not** passed: `POSITION = REUSE` takes
    the position out of the cumulative length value already loaded, so
    this configuration reads no position array at all (plan section 1.2).
    """
    return dict(
        bits=varlen_bits(VARLEN_COMPACT_SIDE, VARLEN_COMPACT_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_padded(
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """BHSD tensors whose sequences are short: lengths only, no positions."""
    return dict(
        bits=varlen_bits(VARLEN_PADDED_SIDE, VARLEN_PADDED_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_strided(
    cu_seqlens_q,
    cu_seqlens_k,
    seq_strides_q,
    seq_strides_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Packed tensors with padding *between* sequences (TE's layout).

    Differs from `varlen_compact` in one thing only: positions come from a
    second array instead of being reused from the length array. That is
    the whole of AOTriton's `StridedVarlen`, and the reason the two roles
    must never be swapped -- `seq_strides` differences are *padded*
    extents, not lengths.
    """
    return dict(
        bits=varlen_bits(VARLEN_STRIDED_SIDE, VARLEN_STRIDED_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=seq_strides_q,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=seq_strides_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_seqused_k(
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    k_is_cache=False,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Packed Q against a KV cache with per-sequence *used* lengths.

    `torch.nn.attention.varlen`'s `seqused_k`, and the configuration no
    `VarlenType` can express: the K side takes its **length** from an
    individual array and its **position** from a cumulative one, so the
    two axes read different tensors.

    `k_is_cache=True` is the rectangular variant -- a BHSD cache with no
    `cu_seqlens_k` at all, where the position is implied by the batch
    index.
    """
    k_side = VARLEN_SEQUSED_CACHE_SIDE if k_is_cache else VARLEN_SEQUSED_PACKED_SIDE
    return dict(
        bits=varlen_bits(VARLEN_COMPACT_SIDE, k_side, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=seqused_k,
        seqinfo_k1=None if k_is_cache else cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


NULL_PTR = flyc.from_c_void_p(fx.Uint8, 0)


def ptr_arg(t):
    if hasattr(t, "data_ptr"):
        type_name = type(t).__name__
        module_name = type(t).__module__
        ptr = (
            0
            if type_name == "FakeTensor" or "fake_tensor" in module_name
            else t.data_ptr()
        )
        return flyc.from_c_void_p(fx.Uint8, ptr)
    return t


def strides_of(t, name):
    """`(batch, head, seq)` strides of a rank-4 BHSD tensor, in elements.

    **Shape and layout are different things**, and this reads the second.
    *Shape* is `t.shape`, and the kernel requires it to be BHSD --
    `(batch, num_heads, seq_len, head_dim)` -- because that is what fixes
    which axis each of the three returned slots describes. *Layout* is how
    the data actually sits in memory, and any permutation is accepted so
    long as D is innermost, which is what `stride(3) == 1` checks. A BSHD
    *layout* is therefore fine; pass it as `t.transpose(1, 2)`, which has
    BHSD shape.

    The order is the ABI. AOTriton dispatches the compiled hsaco directly
    rather than through the Python wrapper, so these three slots are the
    contract, and a caller that fills them from a BSHD shape swaps head
    with sequence -- reading heads where the kernel means tokens, which
    produces finite garbage and never faults.
    """
    if not hasattr(t, "stride"):
        raise TypeError(
            f"{name} must be a rank-4 tensor so its strides can be read, "
            f"got {type(t).__name__}"
        )
    if t.dim() != 4:
        raise ValueError(f"{name} must be rank 4, got shape {tuple(t.shape)}")
    if t.stride(3) != 1:
        raise ValueError(
            f"{name} must have a contiguous last dimension, got "
            f"stride(3)={t.stride(3)}"
        )
    # **The D-axis pitch must cover a whole 8-element chunk.**
    #
    # Every 16-bit access on this axis is 8 columns wide and the guard is
    # `col < head_dim`, so the chunk containing the last live column runs to
    # `ceil8(head_dim)`. That is a real address bound and not a "discard the
    # surplus", but only while the allocation reaches that far: at a tighter
    # pitch the chunk lands in the next row, and on the final row it leaves the
    # tensor. The store side is the sharp one -- dQ, dK, dV and O *write* those
    # columns, so a tight pitch corrupts the next row rather than merely
    # reading it. Measured at head_dim 20 with a packed dQ: 4 elements past the
    # allocation and 41% error in the last row.
    #
    # Checked here rather than in each interface because there were three
    # private copies of this rule and each had a different hole -- dQ's omitted
    # `dq` itself, and none of them ran on the builder path that every test and
    # both `tooling/` probes use. `strides_of` is on every path by construction.
    #
    # Stated against `shape[3]` rather than the compiled tile: the tile is not
    # visible here, and it is not the bound anyway. Vacuous whenever head_dim
    # is itself a multiple of 8, which every ladder rung is.
    _pitch_needed = (t.shape[3] + 7) // 8 * 8
    if t.stride(2) < _pitch_needed:
        raise ValueError(
            f"{name} has a D-axis pitch of {t.stride(2)} elements, under the "
            f"{_pitch_needed} that head_dim {t.shape[3]} needs. Accesses on this axis are 8 "
            f"columns wide, so the chunk holding the last column runs to ceil8({t.shape[3]})"
            f"={_pitch_needed} and would otherwise read -- and for an output, write -- past "
            f"the row. Allocate the last dimension padded to {_pitch_needed} and slice."
        )
    return t.stride(0), t.stride(1), t.stride(2)


def prep_tensors(named, *, q_heads=(), k_heads=()):
    """`(pointers, (nhq, nhk, hqk, hvo), strides)` in launch order.

    `named` is `[("Q", Q), ("K", K), ...]` -- order is the launch order for
    both the pointers and the three-per-tensor strides, and the names are what
    the diagnostics quote. "Q", "K" and "V" must be present; they are what the
    head counts and head dims are read from.

    `q_heads` / `k_heads` name the *other* tensors that must carry Q's head
    count or K's. That is the only thing that varied between the four kernels'
    copies of this: the forward checks O against Q, dQ checks dO and dQ, dK/dV
    checks dO against Q and dK/dV against K. Everything else -- the stride
    sweep, the GQA divisibility rule, reading head_dim from Q and head_dim_v
    from V -- was identical three times over.

    Deliberately does **not** flatten: `t.reshape(-1)` materialises a copy for
    any non-contiguous tensor, which would silently defeat the whole point of
    reading strides. `data_ptr()` is the tensor's base either way.
    """
    by_name = dict(named)
    for req in ("Q", "K", "V"):
        if req not in by_name:
            raise ValueError(
                f"prep_tensors needs a {req!r} entry, got {[n for n, _ in named]}"
            )
    st = []
    for name, t in named:
        st.extend(strides_of(t, name))
    q, k, v = by_name["Q"], by_name["K"], by_name["V"]
    # BHSD: axis 1 is heads, axis 3 the head dim. Read rather than assumed --
    # under MQA/GQA K and V carry fewer heads than Q.
    nhq, nhk = q.shape[1], k.shape[1]
    hqk, hvo = q.shape[3], v.shape[3]
    if v.shape[1] != nhk:
        raise ValueError(f"K and V must share num_heads, got {nhk} and {v.shape[1]}")
    for name in q_heads:
        if by_name[name].shape[1] != nhq:
            raise ValueError(
                f"{name} must carry num_heads_q ({nhq}), got {by_name[name].shape[1]}"
            )
    for name in k_heads:
        if by_name[name].shape[1] != nhk:
            raise ValueError(
                f"{name} must carry num_heads_k ({nhk}), got {by_name[name].shape[1]}"
            )
    if nhq % nhk:
        raise ValueError(
            f"num_heads_q ({nhq}) must be divisible by num_heads_k ({nhk})"
        )
    return [ptr_arg(t) for _, t in named], (nhq, nhk, hqk, hvo), st


def resolve_scale(q, scale, padded_head, default_scale):
    """`sm_scale` from the tensor's *real* head dim, not the compiled tile.

    A builder can only default to `1/sqrt(BLOCK_DMODEL)`, since it has no idea
    what `hdim_qk` will be -- and under a padded head that is the wrong number.
    Deriving it from `q.shape[3]` is right in both cases, and identical to the
    builder's default whenever the two coincide.
    """
    if scale is not None:
        return float(scale)
    if padded_head and hasattr(q, "shape"):
        return 1.0 / math.sqrt(q.shape[3])
    return float(default_scale)


def row_tensor_arg(t, name, num_head_q, seq_len, varlen):
    """Check a rank-2 f32 row tensor -- logsumexp, or delta -- and take its pointer.

    The two share one offset computation in the kernel, so they must share a
    layout, and checking them the same way here is what makes that safe.
    Unlike Q/K/V the kernel derives their pitches from VarlenBits rather than
    reading strides, so contiguity is required rather than merely convenient --
    and the host is the only place the caller's actual layout can be verified.
    """
    from torch import float32 as torch_f32  # lazy: the build venv has no torch

    if t is None:
        raise ValueError(f"{name} is required")
    if t.dtype != torch_f32:
        raise ValueError(f"{name} must be float32, got {t.dtype}")
    if t.dim() != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous: the kernel derives its pitches from VarlenBits"
        )
    layout = 0 if varlen is None else (int(varlen["bits"]) >> 16) & 3
    if layout == 0:
        want_last = int(seq_len) if varlen is None else varlen.get("lse_tokens")
        if want_last is not None and t.shape[1] != int(want_last):
            raise ValueError(
                f"{name} with LSE_LAYOUT_HT wants (*, {int(want_last)}), got {tuple(t.shape)}"
            )
    elif t.shape[1] != num_head_q:
        raise ValueError(
            f"{name} with LSE_LAYOUT_TH wants (*, {num_head_q}), got {tuple(t.shape)}"
        )
    return ptr_arg(t)


def lse_args(lse, seq_len, varlen, num_head_q):
    """The logsumexp pointer, and a check that its layout matches the bits.

    Returns only a pointer: unlike Q/K/V/O this tensor is always compact,
    so the kernel derives both pitches from `LSE_LAYOUT`, `num_head_q` and
    the token count (sdpa-varlen-plan.md section 4.2). Inferring strides
    here as well would give one fact two sources.

    What the host can do instead -- and could not while it was inferring --
    is verify the caller's tensor actually has the declared layout.
    """
    from torch import float32 as torch_f32  # lazy: the build venv has no torch

    if lse is None:
        return NULL_PTR
    if lse.dtype != torch_f32:
        raise ValueError(f"logsumexp must be float32, got {lse.dtype}")
    if lse.dim() != 2:
        raise ValueError(f"logsumexp must be rank 2, got shape {tuple(lse.shape)}")
    if not lse.is_contiguous():
        raise ValueError(
            "logsumexp must be contiguous: the kernel derives its pitches "
            "from VarlenBits rather than reading strides"
        )
    _layout = 0 if varlen is None else (int(varlen["bits"]) >> 16) & 3
    if _layout == 0:
        # The token pitch the kernel will derive. Only checkable when the
        # caller supplies it: under a stacked layout it lives in
        # `seqinfo[N]` on the device, and reading that back would cost a
        # sync for a validation.
        want_last = int(seq_len) if varlen is None else varlen.get("lse_tokens")
        if want_last is not None and lse.shape[1] != int(want_last):
            raise ValueError(
                f"VARLEN_LSE_LAYOUT_HT wants (*, {int(want_last)}), got "
                f"{tuple(lse.shape)}"
            )
    else:
        if lse.shape[1] != num_head_q:
            raise ValueError(
                f"VARLEN_LSE_LAYOUT_TH wants (*, {num_head_q}), got "
                f"{tuple(lse.shape)}"
            )
    return ptr_arg(lse)


def resolve_window(causal_type, host_causal_type, window, seqlen_q, seqlen_k):
    """(window_left, window_right), signed, as the kernel wants them.

    Non-causal still forwards a pair so both arms share one ABI and stay
    directly comparable -- the same reason the strides are always passed
    even under strides_constexpr.

    Causal alignments forward a *sentinel*, not a bound: the kernel
    resolves it per sequence, which is the only correct thing to do once
    there is more than one sequence.
    """
    if causal_type == 0:
        if window is not None:
            # Silently dropping it would return dense attention that is
            # the right shape, finite, and wrong -- and a window is only
            # ever passed by a caller who believes it is being applied.
            # The non-causal arm has no left-masked region to apply one
            # with, so this is a build-time choice, not a runtime one.
            raise ValueError(
                "window= requires a causal build; this one has "
                "causal=False. Pass causal=True, causal_type=3 for "
                "generalized sliding-window attention"
            )
        return 0, 0
    if host_causal_type in CAUSAL_SENTINEL:
        if window is not None:
            raise ValueError(
                f"causal_type={host_causal_type} already fixes the window; "
                "pass causal_type=3 to choose one"
            )
        _s = CAUSAL_SENTINEL[host_causal_type]
        wl, wr = _s, _s
    else:
        if window is None:
            raise ValueError(
                "causal_type=3 is generalized sliding-window attention and "
                "requires window=(left, right); "
                f"pass ({seqlen_q}, 0) for top-left causal or "
                f"({seqlen_q}, {seqlen_k - seqlen_q}) for bottom-right"
            )
        wl, wr = window
    return int(wl), int(wr)


def u64_scalar(value, device, stream=None):
    """A one-element device u64 holding `value`, or `value` if already one.

    `None` stays `None` (the null case). A tensor is taken as-is and *not*
    copied, which is the point: under graph capture the caller owns a counter
    the graph re-reads, and copying it here would freeze it again.

    An int is materialised, which is what AOTriton's host does with its own
    `DEFAULT_PHILOX_SEED`. `torch.int64` and not a uint64: torch has no
    unsigned 64-bit dtype, and the kernel reads the same eight bytes either
    way. Allocated on `stream` when one is given, because the kernel that
    reads it runs there.

    **The returned tensor must stay referenced until the launch is enqueued.**
    Only its raw pointer reaches the kernel, so nothing else keeps it alive --
    callers bind it to a local that outlives the launch call.
    """
    import torch  # lazy: only reached for a plain int seed; AOT passes None

    if value is None or hasattr(value, "data_ptr"):
        if value is not None and value.numel() < 1:
            raise ValueError("a philox scalar tensor must hold at least one element")
        if value is not None and value.element_size() != 8:
            raise ValueError(
                f"a philox scalar tensor must be 8 bytes per element, got {value.dtype}"
            )
        return value
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        return torch.tensor([int(value)], dtype=torch.int64, device=device)


def dropout_outputs(enable_dropout, seed_output, offset_output):
    """(seed_output, offset_output) in launch order -- the forward's write-back.

    The forward records the `(seed, offset)` it actually drew from so the
    backward can be handed them rather than re-deriving them. That matters
    only under graph capture, where the effective offset is a sum formed on
    the device out of a counter the host cannot read without synchronising.
    Either may be `None`, meaning the caller does not want the value.
    """
    if not enable_dropout:
        return NULL_PTR, NULL_PTR
    return (
        NULL_PTR if seed_output is None else ptr_arg(seed_output),
        NULL_PTR if offset_output is None else ptr_arg(offset_output),
    )


def bias_args(bias_type, emit_db, bias, dbias, like):
    """`(B, DB, b_strides, db_strides)`, each stride set a `(batch, head, seq_q)`
    triple in launch order.

    **Two triples, not one.** B and DB are separate tensors and may be laid out
    differently -- `dbias` is the caller's buffer, and nothing obliges it to
    match the bias it is the gradient of. Deriving one from the other happens
    to work whenever both are contiguous, which is why it survived a first
    pass, and writes to the wrong addresses the moment either is a view.

    The bias is `(B, H, Sq, Sk)`, so its three leading strides are batch, head
    and *query row* -- the last axis is the KV column and is the contiguous one
    the kernel's inner loop walks. That is the forward's layout, unchanged, so
    a caller can hand the backward the very tensor it gave the forward.

    `emit_db` is whether the *kernel* has a DB slot at all -- true for dQ,
    false for dK/dV and the fused kernel, which take a bias but do not write
    its gradient. It is a fact about the signature, not a knob: a dQ build with
    a bias always emits dB, since dB is the `dS` it already forms.

    Both pointers are always in the kernarg signature and are `NULL_PTR` when
    the bias axis is off, which is how the dropout arguments behave: a kernel
    argument that exists unconditionally is one fewer ABI variant to keep in
    step, and `const_expr` means the null is never dereferenced.

    `dbias` is checked against `bias` for shape rather than being inferred,
    because a dB that is silently the wrong shape writes out of bounds.
    """
    if not bias_type:
        if bias is not None:
            raise ValueError(
                "bias= was passed but this build has bias=False; rebuild with bias=True"
            )
        return NULL_PTR, NULL_PTR, (0, 0, 0), (0, 0, 0)
    if bias is None:
        raise ValueError("this build has bias=True and requires bias=")
    if bias.dim() != 4:
        raise ValueError(f"bias must be (B, H, Sq, Sk), got {tuple(bias.shape)}")
    if bias.dtype != like.dtype:
        raise ValueError(f"bias dtype {bias.dtype} must match q's {like.dtype}")
    if bias.stride(3) != 1:
        # The kernel loads eight adjacent KV columns per accumulator group in
        # one v8, exactly as the forward does; the same check lives there.
        raise ValueError(
            f"bias must have a contiguous last (Sk) dimension, got stride(3)={bias.stride(3)}"
        )
    db = NULL_PTR
    if emit_db:
        if dbias is None:
            raise ValueError(
                "a dQ build with bias=True always writes dB and requires dbias="
            )
        if tuple(dbias.shape) != tuple(bias.shape):
            raise ValueError(
                f"dbias shape {tuple(dbias.shape)} must equal bias's {tuple(bias.shape)}"
            )
        if dbias.dtype != bias.dtype:
            raise ValueError(
                f"dbias dtype {dbias.dtype} must equal bias's {bias.dtype}"
            )
        if dbias.stride(3) != 1:
            raise ValueError(
                f"dbias must have a contiguous last (Sk) dimension, got stride(3)={dbias.stride(3)}"
            )
        db = ptr_arg(dbias)
    elif dbias is not None:
        raise ValueError(
            "dbias= was passed to a kernel that does not write dB; that is the dQ kernel's"
        )
    sb = bias.stride()[:3]
    sdb = dbias.stride()[:3] if emit_db else (0, 0, 0)
    return ptr_arg(bias), db, sb, sdb


def dropout_args(
    enable_dropout, dropout_p, seed, offset1, offset2, device=None, stream=None
):
    """(seed, offset1, offset2, threshold, 1/(1-p), keepalive) in launch order.

    The trailing `keepalive` is not a kernel argument. It is whatever tensor
    `u64_scalar` had to materialise for an int seed, returned so the caller
    can hold it across the launch; see that function.

    The counter is the pair torch splits it into, not one pre-summed scalar:
    `offset1` is a device pointer the kernel adds in when non-null, `offset2`
    an immediate. See `fmha.philox_offset_base` for why a captured CUDA graph
    needs the pointer half, and `philox_offset1`/`philox_offset2` in AOTriton
    for the ABI this matches.

    `offset1` may be a tensor holding one u64, or None for the uncaptured
    case. A one-element `torch.int64` tensor is the same 8 bytes as the `u64`
    the kernel reads; torch has no unsigned 64-bit dtype to spell it with, and
    the counter is far from the sign bit.

    The threshold and the scale are computed here, once per call, rather
    than per element in the kernel -- `philox.dropout_threshold` turns the
    probability into an i32 the raw random can be compared against, so the
    hot path never converts a random to a float.
    """
    if not enable_dropout:
        return NULL_PTR, NULL_PTR, 0, 0, 1.0, None
    if dropout_p is None:
        raise ValueError("this build has dropout=True and requires dropout_p=")
    p = float(dropout_p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {p}")
    seed_t = u64_scalar(seed, device, stream)
    off1_t = u64_scalar(offset1, device, stream)
    return (
        NULL_PTR if seed_t is None else ptr_arg(seed_t),
        NULL_PTR if off1_t is None else ptr_arg(off1_t),
        int(offset2),
        dropout_threshold(p),
        1.0 / (1.0 - p),
        (seed_t, off1_t),
    )


def varlen_args(
    strides_constexpr, varlen, seqlen_q, seqlen_k, q, batch_size, num_seqlens
):
    """(bits, q0, q1, k0, k1, max_q, max_k) in launch order, plus two checks.

    `varlen` is None for the dense case, else a dict with `bits` and
    whichever `seqinfo_*` tensors that configuration reads. Unread slots
    stay **null**, which is safe because the kernel's decode branches
    rather than selects -- see the prologue.

    **`batch_size` and `num_seqlens` are different quantities and never share
    a variable on the host.** `batch_size` is `q.size(0)`, always, whatever the
    layout. `num_seqlens` is how many sequences are packed into a 1HTD tensor,
    and is 0 when nothing is packed. For a dense BHSD call they are `(B, 0)`;
    for a packed `(1, H, T, D)` call holding N sequences they are `(1, N)` --
    genuinely different numbers, which is why one variable cannot serve.

    Neither is returned: both are already the caller's, and this function's job
    for them is to *check*, because each has a second, independent source of
    truth. `batch_size` must be `q.size(0)`, and a packed `num_seqlens` must be
    `len(cu_seqlens_q) - 1`. Passing N where `batch_size` belongs is the
    specific mistake -- it launches N programs over a tensor whose batch axis
    is 1, and every one of them addresses a plausible row.
    """
    if int(batch_size) != int(q.shape[0]):
        raise ValueError(
            f"batch_size={int(batch_size)} but q.size(0)={int(q.shape[0])}. batch_size is the "
            f"tensor's batch extent whatever the layout; a packed 1HTD tensor has 1, and its "
            f"sequence count goes in num_seqlens."
        )
    if varlen is None:
        if int(num_seqlens):
            raise ValueError(
                f"num_seqlens={int(num_seqlens)} without varlen=; a dense call packs no sequences"
            )
        return (0, NULL_PTR, NULL_PTR, NULL_PTR, NULL_PTR, int(seqlen_q), int(seqlen_k))
    if strides_constexpr:
        raise ValueError(
            "strides_constexpr derives the layout from the shape, which "
            "varlen invalidates; it is a dense-only diagnostic arm"
        )
    # No implemented-subset gate: every encodable side byte now decodes,
    # since the decoder is one function covering all three axis values.
    # `varlen_bits` rejects the combinations that are not *meaningful*
    # (reserved codes, REUSE without cumulative lengths).
    bits = int(varlen["bits"])
    # A STACKED Q side is the packed one, and the only one the kernel reads
    # `num_seqlens` for (`lse_token_pitch`, to reach slot [N] of the array
    # holding the batch total). A non-stacked varlen side -- `varlen_padded`,
    # BHSD tensors with short sequences -- has a real batch axis and packs
    # nothing, so its count is 0 like the dense case.
    if bits & VARLEN_STACKED:
        if varlen.get("seqinfo_q0") is None:
            raise ValueError(
                "a STACKED Q side needs seqinfo_q0 (cu_seqlens_q) to count its sequences"
            )
        packed = int(varlen["seqinfo_q0"].numel()) - 1
        if int(num_seqlens) != packed:
            raise ValueError(
                f"num_seqlens={int(num_seqlens)} but cu_seqlens_q describes {packed} packed sequences"
            )
    elif int(num_seqlens):
        raise ValueError(
            f"num_seqlens={int(num_seqlens)} but the Q side is not STACKED, so nothing is packed "
            f"and the batch axis is real; this configuration wants num_seqlens=0"
        )
    got = tuple(
        ptr_arg(varlen[k]) if varlen.get(k) is not None else NULL_PTR
        for k in ("seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1")
    )
    return (bits,) + got + (int(varlen["max_seqlen_q"]), int(varlen["max_seqlen_k"]))


def run_compiled(cache, exe, *args):
    """First call compiles and runs; later calls fast-dispatch the compiled fn.

    `cache` is the caller's `weakref.WeakKeyDictionary`, not a module-level one:
    four kernels share this function and a single dict keyed by launcher would
    work, but passing it keeps the lifetime obviously the caller's.

    `flyc.compile` caches internally (`JitFunction._mem_cache`, keyed on the
    full argument signature), so this is not about avoiding recompilation --
    it is about dispatch overhead, measured at **157 us per call** at
    (head_dim 64, N 512), where the whole call is 63 us with the cache and
    220 us without. The gap closes as the kernel grows.

    A `WeakKeyDictionary` rather than an `exe._cf` attribute: that attribute is
    not one FlyDSL defines, so writing it mutates a FlyDSL-owned object. Weak
    keys so an entry dies with its launcher instead of pinning one compiled
    artifact -- and the GPU code object it owns -- per configuration for the
    life of the process.
    """
    cf = cache.get(exe)
    if cf is None:
        cache[exe] = flyc.compile(exe, *args)
    else:
        cf(*args)


def new_compiled_cache():
    """A cache for `run_compiled`. One per kernel module."""
    return weakref.WeakKeyDictionary()


def dtype_to_elem_type(dtype_str):
    """`"f16"` / `"bf16"` -> the FlyDSL Numeric class the kernel builds with."""
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
