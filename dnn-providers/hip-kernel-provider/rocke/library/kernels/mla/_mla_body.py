# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared emitters of the MLA kernel family.

The arch-neutral core the MLA kernels build on: the MFMA atom geometry and LDS
layout pads, lane math, the kernel-argument table (one table drives both
``b.param`` and the host launch signature), lane ids, and the transposed-frame
mask + online softmax. Prefill-only and gfx942-only emitters live in
:mod:`kernels.mla._mla_prefill_gfx942_body`.

**Contract.** The names in ``__all__`` are relied on by more than one MLA
kernel, and every helper emits its ops in a fixed order that the MLA goldens
pin: the IR builder is side-effecting (an unused ``b.const_i32`` still emits
an op), so reordering two calls inside a helper is a codegen change for every
caller. Change one only with a deliberate golden re-bless.

**Conventions.**

* *Atom.* Every GEMM is a ``16x16x16`` MFMA (``MFMA_M == MFMA_N == MFMA_K ==
  16``); operand reads are four elements wide at a K step of 16, so these
  helpers do not fit a wider-K atom.
* *Lane ids.* ``lane_row = lane % 16`` is the MFMA operand row and
  ``lane_kq = 4 * (lane // 16)`` the operand K sub-offset (:func:`lane_ids`).
* *Transposed frame.* The softmax helpers work on Sᵀ (``m = key``,
  ``n = query``): lane ``l`` holds query ``l % 16`` and keys
  ``kt * 16 + lane_kq + r`` for ``r`` in ``[0, C_PER_LANE)``, so a query's
  row reduce is in-lane plus two cross-lane stages (:func:`query_reduce`).
* *Loops.* :func:`run_k_loops` walks key tiles from tile 0: an unmasked run,
  then a masked tail.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType, Type, Value
from rocke.helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from rocke.helpers.attention import mfma_16x16x16_for_dtype as mfma_16x16x16
from rocke.helpers.attention_fwd_ext import LOG2E
from rocke.helpers.distribution import make_static_tile_distribution
from rocke.helpers.spec import SignatureBuilder

from kernels.common._bf16_rne import bf16x4_rne_finite

__all__ = [  # noqa: RUF022 -- grouped by section, as in the module
    "MFMA_M",
    "MFMA_N",
    "MFMA_K",
    "WAVE_SIZE",
    "C_PER_LANE",
    "NEG_INF",
    "WT_PAD",
    "V8_PAD",
    "C_STAGE_W",
    "mfma_16x16x16",
    "shift",
    "query_reduce",
    "mfma_16x16_c_row",
    "on_wave",
    "wave_tile_row",
    "pick4",
    "half4",
    "zero_tiles",
    "KernelArg",
    "KernelArgs",
    "tail_args",
    "kernel_signature",
    "open_kernel",
    "Lanes",
    "lane_ids",
    "SoftmaxConsts",
    "softmax_consts",
    "softmax_iter_args",
    "acc_iter_args",
    "KeyMask",
    "key_mask",
    "SoftmaxRow",
    "online_softmax_row",
    "scale_frag",
    "pack_p",
    "rescale_if_any",
    "split_softmax_state",
    "run_k_loops",
]

MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
WAVE_SIZE = 64

# C registers each lane holds for one 16x16 MFMA tile.
C_PER_LANE = 4

# Unused trailing columns on an LDS buffer, in bf16 elements. Purely an LDS-layout
# knob: it shifts the row stride off the 32-dword bank width so that a strided
# walk down the row index spreads across banks instead of serialising on one.
# Must stay even -- ``smem_load_vN(..., n=4)`` reads 8 B and an odd element pad
# would misalign every row. 0 restores the natural (bank-conflicting) layout.
WT_PAD = 4

# Same idea as ``WT_PAD``, for buffers that are *written* with 8-wide (16 B) bf16
# vector stores. ``smem_store_vN`` declares ``align = n * elem_bytes``, so an
# 8-wide bf16 store promises 16 B alignment; a pad of 4 would leave odd rows only
# 8 B-aligned and the promise would be a lie. 8 keeps every row start at 16 B.
V8_PAD = 8

# Width, in bf16 elements, of one lane's slice when staging ``c_kv`` into
# ``kv_lds``. This is 4, not 8, and the narrower store is what lets ``kv_lds``
# carry ``WT_PAD`` instead of ``V8_PAD`` -- which is the whole point.
#
# ``kv_lds`` is the B operand of the score GEMM, and that read is the hottest
# LDS access in the kernel (``SCORE_K_ITERS * K_TILES`` of them per k-tile).
# Lane ``l`` of a 16-lane group reads row ``a = l % 16``, so the bank it lands
# on is ``(stride_dwords * a + const) % 32``. With ``V8_PAD`` the stride is 292
# dwords == 4 mod 32, whose period over ``a`` is only 8: lanes ``a`` and
# ``a + 8`` collide on every read, 2-way, for the whole loop. ``WT_PAD`` puts
# the stride at 290 == 2 mod 32, period 16, so all sixteen lanes land on
# distinct banks and the read goes conflict-free.
#
# The pad cannot simply be lowered on its own: a 4-element pad leaves odd rows
# at 8 mod 16 B, and an 8-wide store's ``align = 16`` declaration would become
# a lie -- a silent miscompile, not an assert. So the store width has to come
# down with it. An XOR swizzle was considered instead and is structurally
# blocked: a column bit XORed at bit ``k`` displaces the bank by ``2**(k-1)``
# dwords, and only a displacement of 2 mod 4 breaks a stride of 4 mod 32, which
# pins ``k == 2`` -- exactly the column bit an 8-wide store must preserve.
# Either route therefore requires the narrower store, and given that, the pad
# is the simpler of the two and costs no swizzle arithmetic.
#
# Narrowing is not a concession on either side of the copy. The global load
# stays fully coalesced (lanes read 8 B at an 8 B stride == contiguous), and
# the ``kv_lds`` *store* improves too: at width 8 its column term is ``4a``,
# the same 2-way pattern as the read, while at width 4 it is ``2a`` and goes
# conflict-free as well.
C_STAGE_W = 4

# Running-max sentinel of the online softmax. Finite on purpose: ``exp2`` of a
# difference of two sentinels is 1, not NaN, and the masks zero those terms.
NEG_INF = -1e30

_C16_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.bf16_16x16x16())
)

# ---------------------------------------------------------------------------
# Lane math
# ---------------------------------------------------------------------------


def shift(b: IRBuilder, offset: int, value: Value) -> Value:
    """``value + offset``, skipping the add when ``offset`` is 0.

    The IR builder is side-effecting: ``b.const_i32(0)`` emits an op whether or
    not its handle is used. Unrolled tile loops that degenerate to a single
    iteration would otherwise leave a trail of dead constants in the IR.
    """
    return value if offset == 0 else b.add(b.const_i32(offset), value)


def query_reduce(b: IRBuilder, scalar: Value, *, combine: str) -> Value:
    """Reduce ``scalar`` across the four 16-lane groups of a wave.

    In the transposed score fragment (Sᵀ, ``m = key``, ``n = query``) lane ``l``
    holds query ``l % 16`` and keys ``4 * (l // 16) + r``. After the in-lane
    fold over ``r``, a query's partials sit in lanes ``l``, ``l ^ 16``,
    ``l ^ 32`` and ``l ^ 48``: two XOR stages leave the result in all four. XOR
    16 stays inside a 32-lane half and runs as ``ds_swizzle``; XOR 32 crosses
    halves, which ``ds_swizzle`` cannot, and runs as ``ds_bpermute``.
    """
    fold = b.fadd if combine == "sum" else b.fmax
    v = fold(scalar, b.warp_shuffle_xor(scalar, 16))
    return fold(v, b.warp_shuffle_xor(v, 32))


def mfma_16x16_c_row(b: IRBuilder, lane: Value, reg: int) -> Value:
    """MFMA-local output row for a ``16x16`` C element ``reg`` (0..3).

    ``lane`` must be the *wave*-local lane id, not the workgroup thread id:
    every wave decodes its own accumulator independently.
    """
    if not (0 <= reg < 4):
        raise ValueError(f"mfma_16x16 reg must be 0..3, got {reg}")
    m_blk = b.div(lane, b.const_i32(16))
    n = b.mod(lane, b.const_i32(16))
    row, _col = _C16_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(reg)], ps=[[m_blk, n]]
    )
    return row


def on_wave(b: IRBuilder, wave: Value, w: int) -> AbstractContextManager:
    """Region run by wave ``w`` only. It must hold no barrier and carry no
    results out (``scf.if`` yields none)."""
    return b.scf_if(b.cmp_eq(wave, b.const_i32(w)))


def wave_tile_row(
    b: IRBuilder, wave: Value, per_wave: int, t: int, lane_off: Value
) -> Value:
    """Row ``lane_off`` of the wave's ``t``-th 16-wide tile.

    Wave ``w`` owns tiles ``[w * per_wave, (w + 1) * per_wave)``.
    """
    n_tile = b.add(b.mul(wave, b.const_i32(per_wave)), b.const_i32(t))
    return b.add(b.mul(n_tile, b.const_i32(MFMA_N)), lane_off)


def pick4(b: IRBuilder, xs: Sequence[Value], e: int, dtype: Type) -> Value:
    """Element ``e`` of each of four vectors, packed into one four-wide vector."""
    return b.vec_pack([b.vec_extract(x, e) for x in xs], dtype)


def half4(b: IRBuilder, v8: Value, h: int, dtype: Type) -> Value:
    """Elements ``[4 h, 4 h + 4)`` of an eight-wide vector, as a four-wide one."""
    return b.vec_pack([b.vec_extract(v8, 4 * h + i) for i in range(4)], dtype)


def zero_tiles(b: IRBuilder, *shape: int) -> list:
    """Nested lists of zeroed f32x4 C fragments, ``shape`` outermost first."""
    if len(shape) == 1:
        return [b.zero_vec_f32(4) for _ in range(shape[0])]
    return [zero_tiles(b, *shape[1:]) for _ in range(shape[0])]


# ---------------------------------------------------------------------------
# Kernel arguments
# ---------------------------------------------------------------------------

_IR_DTYPES = {"bf16": BF16, "f16": F16, "f32": F32, "i32": I32}
_ROLES = ("out", "in", "index", "scalar")


@dataclass(frozen=True)
class KernelArg:
    """One kernel argument.

    ``role`` fixes the LLVM attributes: ``out`` is a write-only, ``in`` a
    read-only tensor (both ``noalias``, 16 B aligned); ``index`` a read-only
    i32 table (4 B aligned); ``scalar`` a by-value scalar. ``key`` is the
    attribute the declared value is bound to on :class:`KernelArgs`.
    """

    key: str
    name: str
    dtype: str
    role: str

    def __post_init__(self) -> None:
        # A typo here would split the device and host ABI silently: open_kernel
        # and kernel_signature each branch on these fields.
        if self.role not in _ROLES:
            raise ValueError(
                f"KernelArg {self.name}: role={self.role!r}, expected one of {_ROLES}"
            )
        if self.dtype not in _IR_DTYPES:
            raise ValueError(
                f"KernelArg {self.name}: dtype={self.dtype!r}, expected one of "
                f"{tuple(_IR_DTYPES)}"
            )
        if self.key not in KernelArgs.__slots__:
            raise ValueError(
                f"KernelArg {self.name}: key={self.key!r}, expected a KernelArgs field"
            )


def tail_args() -> tuple[KernelArg, ...]:
    """The index tables and scalars every MLA prefill kernel declares last."""
    return (
        KernelArg("cu_q", "cu_seqlens_q_ptr", "i32", "index"),
        KernelArg("cu_k", "cu_seqlens_k_ptr", "i32", "index"),
        KernelArg("block_table", "block_table_ptr", "i32", "index"),
        KernelArg("scale", "scale", "f32", "scalar"),
        KernelArg("num_seqs", "num_seqs", "i32", "scalar"),
        KernelArg("bt_stride", "block_table_stride", "i32", "scalar"),
        KernelArg("max_k", "max_k", "i32", "scalar"),
    )


def kernel_signature(args: Sequence[KernelArg]) -> list[dict]:
    """Host launch signature of ``args``."""
    sig = SignatureBuilder()
    for a in args:
        if a.role == "scalar":
            sig.scalar(a.name, a.dtype)
        else:
            sig.ptr(a.name, a.dtype)
    return sig.build()


@dataclass(slots=True)
class KernelArgs:
    """The declared argument values, by :attr:`KernelArg.key`."""

    out: Value | None = None
    lse: Value | None = None
    scores: Value | None = None
    q: Value | None = None
    c_kv: Value | None = None
    k_rope: Value | None = None
    w_uk: Value | None = None
    cu_q: Value | None = None
    cu_k: Value | None = None
    block_table: Value | None = None
    scale: Value | None = None
    num_seqs: Value | None = None
    bt_stride: Value | None = None
    max_k: Value | None = None


def open_kernel(
    name: str,
    threads: int,
    args: Sequence[KernelArg],
    *,
    agpr_alloc: tuple[int, int] | None = None,
    index_noalias: bool = False,
) -> tuple[IRBuilder, KernelArgs]:
    """New kernel builder with its attributes set and ``args`` declared.

    ``index_noalias`` also declares the index tables ``noalias``. They are
    never written, and the promise lets the backend fetch them with scalar
    loads, off the in-order vmcnt that vector loads share.
    """
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = threads
    if agpr_alloc is not None:
        b.kernel.attrs["agpr_alloc"] = agpr_alloc
    vals = KernelArgs()
    for a in args:
        t = _IR_DTYPES[a.dtype]
        if a.role == "out":
            v = b.param(
                a.name, PtrType(t, "global"), noalias=True, writeonly=True, align=16
            )
        elif a.role == "in":
            v = b.param(
                a.name, PtrType(t, "global"), noalias=True, readonly=True, align=16
            )
        elif a.role == "index":
            na = {"noalias": True} if index_noalias else {}
            v = b.param(a.name, PtrType(t, "global"), readonly=True, align=4, **na)
        elif a.role == "scalar":
            v = b.param(a.name, t)
        else:  # pragma: no cover - KernelArg validates role
            raise ValueError(f"unknown role {a.role!r}")
        setattr(vals, a.key, v)
    return b, vals


# ---------------------------------------------------------------------------
# Lane ids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lanes:
    """Per-lane ids every MLA kernel derives from the thread id."""

    tid: Value
    lane: Value
    wave: Value
    # MFMA operand row (lane % 16) and K sub-offset (4 * (lane // 16)).
    lane_row: Value
    lane_kq: Value


def lane_ids(b: IRBuilder) -> Lanes:
    """``tid``, ``lane``, ``wave``, ``lane_row`` and ``lane_kq``, in that emission order."""
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE_SIZE))
    wave = b.div(tid, b.const_i32(WAVE_SIZE))
    lane_row = b.mod(lane, b.const_i32(16))
    lane_kq = b.mul(b.div(lane, b.const_i32(16)), b.const_i32(4))
    return Lanes(tid, lane, wave, lane_row, lane_kq)


# ---------------------------------------------------------------------------
# Mask + online softmax
# ---------------------------------------------------------------------------
#
# The k-loop works in the transposed frame: it computes Sᵀ (key x query) and
# accᵀ (latent x query), so each lane owns one query of each M tile --
# ``lane % 16`` -- and its softmax state is one (m, l) pair per M tile. In the
# Sᵀ fragment a lane holds query ``lane % 16`` and keys
# ``kt*16 + 4*(lane/16) + r``, so a query's row reduce is in-lane over its
# 4*K_TILES keys plus two cross-lane stages over the four 16-lane groups.


@dataclass(frozen=True)
class SoftmaxConsts:
    """Constants of the online softmax: the log2-domain scale and the sentinels."""

    scale_log2: Value
    neg_inf: Value
    zero: Value


def softmax_consts(b: IRBuilder, scale: Value) -> SoftmaxConsts:
    """Fold ``log2(e)`` into the raw ABI scale, next to the ``exp2`` that
    consumes it, so the log2 domain never escapes into the ABI."""
    scale_log2 = b.fmul(scale, b.const_f32(LOG2E))
    return SoftmaxConsts(scale_log2, b.const_f32(NEG_INF), b.const_f32(0.0))


def softmax_iter_args(c: SoftmaxConsts, m_tiles: int) -> list[tuple[str, Value]]:
    """``(m, l)`` loop-carried state, one pair per M tile."""
    return [(f"m{mt}", c.neg_inf) for mt in range(m_tiles)] + [
        (f"l{mt}", c.zero) for mt in range(m_tiles)
    ]


def acc_iter_args(b: IRBuilder, m_tiles: int, n_tiles: int) -> list[tuple[str, Value]]:
    """Zeroed f32x4 accumulator tiles as loop-carried state, M-tile major."""
    return [
        (f"acc{mt}_{t}", b.zero_vec_f32(4))
        for mt in range(m_tiles)
        for t in range(n_tiles)
    ]


@dataclass(frozen=True)
class KeyMask:
    """Per-element key bounds of one key tile.

    Two bounds, and both land on ``p`` rather than only on ``s``: the ragged
    key bound (the last tile overhangs S_k, where page-granular staging
    supplies real but meaningless bytes) and the bottom-right causal bound.
    Masking ``s`` alone is not enough -- with the -1e30 sentinel a fully
    masked row gets m_new == -1e30, so every element would yield
    exp2(s - m_new) == 1 and ``l`` would come out BK, not 0. l == 0 is what
    makes ``safe_inv_l`` store exact zeros and the LSE collapse to a true -inf.
    """

    k_abs: list
    in_k: list
    context_off: Value


def key_mask(
    b: IRBuilder,
    kb_start: Value,
    lane_kq: Value,
    s_k: Value,
    context_off: Value,
    *,
    k_tiles: int,
) -> KeyMask:
    """Key bounds of the key tile starting at ``kb_start``: keys below ``s_k``,
    and the bottom-right causal offset ``context_off`` for :class:`KeyMask`."""
    k_abs = [
        [b.add(kb_start, shift(b, kt * MFMA_N + r, lane_kq)) for r in range(C_PER_LANE)]
        for kt in range(k_tiles)
    ]
    in_k = [[b.cmp_lt(k, s_k) for k in row] for row in k_abs]
    return KeyMask(k_abs, in_k, context_off)


@dataclass(frozen=True)
class SoftmaxRow:
    """One M tile's softmax update: new ``(m, l)``, the rescale, and P."""

    m: Value
    l: Value
    alpha: Value
    p: list[Value]
    # Lazy rescale only: whether any lane of the wave moved its running max
    # (wave-uniform i32, nonzero when one did).
    any_need: Value | None


def online_softmax_row(
    b: IRBuilder,
    s_tiles: Sequence[Value],
    m_old: Value,
    l_old: Value,
    c: SoftmaxConsts,
    *,
    mask: KeyMask | None = None,
    q_abs: Value | None = None,
    lazy_tau: float | None = None,
) -> SoftmaxRow:
    """One M tile's online-softmax update over a key tile's Sᵀ fragments.

    ``mask`` (with this lane's query row ``q_abs``) applies the per-element
    key bounds; ``None`` is a tile every query row can fully see.

    ``lazy_tau`` keeps the running max in use unless the new one exceeds it by
    more than ``lazy_tau`` (log2 units); ``any_need`` then reports whether any
    lane of the wave moved it, so the caller can skip the accumulator rescale
    when none did. Lanes that did not move have ``alpha == 1``.
    """
    keep: list[Value] = []
    s_v: list[Value] = []
    if mask is not None:
        causal_hi = b.add(q_abs, mask.context_off)
    for kt, frag in enumerate(s_tiles):
        for r in range(C_PER_LANE):
            s_scaled = b.fmul(b.vec_extract(frag, r), c.scale_log2)
            if mask is not None:
                keep.append(
                    b.land(mask.in_k[kt][r], b.cmp_le(mask.k_abs[kt][r], causal_hi))
                )
                s_v.append(b.select(keep[-1], s_scaled, c.neg_inf))
            else:
                s_v.append(s_scaled)
    local_max = s_v[0]
    for v in s_v[1:]:
        local_max = b.fmax(local_max, v)
    m_new = b.fmax(m_old, query_reduce(b, local_max, combine="max"))
    need = None
    if lazy_tau is not None:
        if not 0 <= lazy_tau < 128:
            raise ValueError(f"lazy_tau={lazy_tau}, expected 0 <= lazy_tau < 128")
        need = b.fcmp("ogt", m_new, b.fadd(m_old, b.const_f32(lazy_tau)))
        m_new = b.select(need, m_new, m_old)
    # exp2_fast: one v_exp_f32, no range-reduction guard. alpha's argument is
    # <= 0. p's argument is <= 0 without lazy_tau and <= lazy_tau with it (the
    # stale max may sit up to lazy_tau below a score), far below exp2's f32
    # overflow at 128. An underflow to 0 is the right answer for a weight.
    alpha = b.exp2_fast(b.fsub(m_old, m_new))
    if mask is not None:
        p_v = [
            b.select(k, b.exp2_fast(b.fsub(v, m_new)), c.zero)
            for k, v in zip(keep, s_v, strict=True)
        ]
    else:
        p_v = [b.exp2_fast(b.fsub(v, m_new)) for v in s_v]
    local_sum = p_v[0]
    for v in p_v[1:]:
        local_sum = b.fadd(local_sum, v)
    l_new = b.fadd(b.fmul(l_old, alpha), query_reduce(b, local_sum, combine="sum"))
    any_need = None
    if need is not None:
        any_need = b.wave_any(b.select(need, b.const_i32(1), b.const_i32(0)))
    return SoftmaxRow(m_new, l_new, alpha, p_v, any_need)


def scale_frag(b: IRBuilder, frag: Value, alpha: Value) -> Value:
    """``frag * alpha``, element by element."""
    for r in range(C_PER_LANE):
        frag = b.vec_insert(frag, b.fmul(b.vec_extract(frag, r), alpha), r)
    return frag


def pack_p(b: IRBuilder, p: Sequence[Value], k_tiles: int, *, rne: bool) -> list[Value]:
    """P as bf16 B fragments, one per key tile.

    The Sᵀ fragment is already the PV GEMM's B operand layout (k = key,
    n = query), so P never leaves registers. ``rne`` rounds with
    :func:`bf16x4_rne_finite` instead of ``fptrunc``.
    """
    out = []
    for kt in range(k_tiles):
        vals = p[kt * C_PER_LANE : (kt + 1) * C_PER_LANE]
        if rne:
            out.append(bf16x4_rne_finite(b, vals))
        else:
            out.append(b.vec_trunc_f32_to_bf16(b.vec_pack(vals, F32)))
    return out


def run_k_loops(
    b: IRBuilder,
    iter_args: list[tuple[str, Value]],
    *,
    n_full: Value,
    tail: tuple[Value, Value],
    step: int,
    body: Callable[[Value, list, bool], list],
) -> list[Value]:
    """The unmasked loop over ``[0, n_full)``, then the masked tail.

    ``body(k_tile, state, masked)`` returns the next state. The run-ahead
    stage and the softmax state flow from the first loop into the second
    through the loop results. Returns the tail loop's results.
    """
    main_loop = b.scf_for_iter(
        b.const_i32(0), n_full, b.const_i32(step), iter_args=iter_args, iv_name="k_main"
    )
    with main_loop as (k_tile, state):
        b.scf_yield(*body(k_tile, state, False))
    tail_loop = b.scf_for_iter(
        tail[0],
        tail[1],
        b.const_i32(step),
        iter_args=[
            (f"{name}_t", v)
            for (name, _), v in zip(iter_args, main_loop.results, strict=True)
        ],
        iv_name="k_tail",
    )
    with tail_loop as (k_tile, state):
        b.scf_yield(*body(k_tile, state, True))
    return tail_loop.results


def split_softmax_state(
    state: Sequence[Value], m_tiles: int, n_acc: int
) -> tuple[list[Value], list[Value], list[list[Value]]]:
    """``(ms, ls, accs)`` from a loop state laid out as :func:`softmax_iter_args`
    followed by :func:`acc_iter_args` (``n_acc`` tiles per M tile)."""
    ms = list(state[:m_tiles])
    ls = list(state[m_tiles : 2 * m_tiles])
    base = 2 * m_tiles
    accs = [
        list(state[base + mt * n_acc : base + (mt + 1) * n_acc])
        for mt in range(m_tiles)
    ]
    return ms, ls, accs


def rescale_if_any(
    b: IRBuilder, frags: Sequence[Value], alpha: Value, any_need: Value, *, tag: str
) -> list[Value]:
    """``frags * alpha`` when ``any_need`` is set, else ``frags`` unchanged.

    A zero- or one-trip loop on the wave-uniform flag, which carries the
    accumulator through as its iteration arguments.
    """
    rs_loop = b.scf_for_iter(
        b.const_i32(0),
        b.select(b.cmp_ne(any_need, b.const_i32(0)), b.const_i32(1), b.const_i32(0)),
        b.const_i32(1),
        iter_args=[(f"rs{tag}_{t}", v) for t, v in enumerate(frags)],
        iv_name=f"rescale{tag}",
    )
    with rs_loop as (_, rs_state):
        b.scf_yield(*[scale_frag(b, v, alpha) for v in rs_state])
    return list(rs_loop.results)
