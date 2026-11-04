# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Case table, materialisation and oracle of the unified (paged) SDPA backward.

A unified case is a set of THD query spans over a paged KV cache: per
sequence ``s`` a query span of ``q_lens[s]`` tokens and a kv length of
``kv_lens[s]`` keys stored in pages of ``page`` tokens, addressed through an
int32 block table. Every case id starts with ``bwd_unified_``.

Materialisation (:func:`materialize_unified`) builds the physical caches
first and gathers each sequence's contiguous K / V through its block table,
so shared (prefix) pages hold the same values for every sequence that
references them. The oracle is
:func:`rocke.numeric.sdpa_reference.sdpa_reference_bwd` on the gathered
contiguous THD problem; the expected gradient caches scatter-add the gathered
dK / dV through the same table (a shared slot sums every referencing
sequence's gradient), so the comparison happens slot by slot.

Canaries (checked by the device harness):

* every slot of every page referenced by a live block is written; the slots
  of the last live page past ``len_kv`` are exact zeros;
* pages no live block references (spare pages and the trap page) keep the
  sentinel in the gradient caches and hold NaN in K / V;
* block-table entries past ``ceil(len_kv / page)`` (and the row-stride
  padding) hold the trap page's id: using one would read NaN K / V or write
  into the trap page;
* K / V slots past ``len_kv`` hold NaN (never read: rows are clamped);
* gaps between the elements of strided tensors hold NaN (inputs) or keep the
  sentinel (outputs).

Tiers: ``smoke`` (a few minutes on one device), ``full``, ``exhaustive``
(cumulative, like the backward table).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from rocke.numeric.sdpa_reference import sdpa_reference, sdpa_reference_bwd

from .bwd_cases import TOLERANCE_BWD, SdpaBwdReference, compare_bwd
from .helpers import TIERS, round_to

__all__ = [
    "MASKS",
    "TABLES",
    "UnifiedBwdCase",
    "UnifiedBwdInputs",
    "compare_unified",
    "get_unified_case",
    "materialize_unified",
    "select_unified",
    "unified_cases",
    "validate_unified_case",
]

# (left_bound, right_bound, top_left) per mask name; ``*_win`` take the case's
# window as the left bound.
MASKS = {
    "none": (-1, -1, True),
    "tl": (-1, 0, True),
    "br": (-1, 0, False),
    "tl_win": (None, 0, True),
    "br_win": (None, 0, False),
}
# Physical page assignment of the live blocks.
TABLES = ("contiguous", "random", "reversed", "scattered")
LAYOUTS = ("packed", "padded", "narrow")
_SEED = 0x5EED_0B4D


@dataclass(frozen=True)
class UnifiedBwdCase:
    """One unified backward problem (see the module docstring).

    ``shared_prefix`` full pages at the start of every sequence map to the
    same physical pages (a non-injective table: atomic page mode only).
    ``spare_blocks`` unreferenced pages are added to the pool; the trap page is
    one more. ``extra_entries`` block-table entries per row beyond the longest
    sequence and ``bt_pad`` row-stride padding entries hold the trap id.
    ``q_slack`` / ``kv_slack`` raise ``max_seqlen_q`` / ``max_seqlen_kv``
    above the longest span. ``layout``: ``"packed"`` (canonical strides),
    ``"padded"`` (token / block strides padded by multiples of 8) or
    ``"narrow"`` (odd strides: the one-element access instances). ``knobs``:
    tuning overrides ``((name, value), ...)`` (spec knobs, ``g_split``, ...).
    ``modes``: the page modes the case runs in.
    """

    id: str
    tier: str
    q_lens: tuple
    kv_lens: tuple
    page: int
    d: int = 64
    dtype: str = "bf16"
    h_q: int = 4
    h_kv: int = 4
    mask: str = "br"
    window: int = 15
    table: str = "random"
    shared_prefix: int = 0
    spare_blocks: int = 2
    extra_entries: int = 1
    bt_pad: int = 0
    q_slack: int = 0
    kv_slack: int = 0
    layout: str = "packed"
    scale: float | None = None
    knobs: tuple = ()
    modes: tuple = ("direct", "atomic")
    tags: tuple = field(default=())

    # -- views shared with the backward table's comparison --------------------

    @property
    def length_mode(self) -> str:
        return "ragged"

    def lens(self) -> tuple[tuple, tuple]:
        return self.q_lens, self.kv_lens

    @property
    def injective(self) -> bool:
        return self.shared_prefix == 0

    @property
    def band(self) -> tuple[int, int, bool]:
        left, right, tl = MASKS[self.mask]
        return (self.window if left is None else left), right, tl

    def knob_dict(self) -> dict:
        return dict(self.knobs)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.q_lens) + self.q_slack

    @property
    def max_seqlen_kv(self) -> int:
        return max(self.kv_lens) + self.kv_slack


def validate_unified_case(case: UnifiedBwdCase) -> list[str]:
    e = []
    if not case.id.startswith("bwd_unified_"):
        e.append("unified case ids start with bwd_unified_")
    if case.tier not in TIERS:
        e.append(f"tier {case.tier}")
    if len(case.q_lens) != len(case.kv_lens) or not case.q_lens:
        e.append("q_lens / kv_lens must be non-empty and of equal length")
    if min(case.q_lens + case.kv_lens) < 0:
        e.append("negative length")
    if case.max_seqlen_q <= 0 or case.max_seqlen_kv <= 0 or sum(case.q_lens) <= 0:
        e.append("an empty problem (declined EMPTY) is not a numeric case")
    if case.page <= 0 or case.page % 16:
        e.append(f"page {case.page} is not a multiple of 16")
    if case.mask not in MASKS:
        e.append(f"mask {case.mask}")
    if case.table not in TABLES:
        e.append(f"table {case.table}")
    if case.layout not in LAYOUTS:
        e.append(f"layout {case.layout}")
    if case.h_q % case.h_kv:
        e.append("h_q must be a multiple of h_kv")
    if case.dtype not in TOLERANCE_BWD:
        e.append(f"dtype {case.dtype}")
    if case.d not in (32, 64, 128):
        e.append(f"head dim {case.d}")
    if case.shared_prefix:
        if any(n < case.shared_prefix * case.page for n in case.kv_lens):
            e.append("a shared prefix needs every kv length to cover its full pages")
        if "direct" in case.modes:
            e.append("a shared (non-injective) table runs in atomic page mode only")
    if not set(case.modes) <= {"direct", "atomic"} or not case.modes:
        e.append(f"modes {case.modes}")
    return e


# ---------------------------------------------------------------------------
# case table
# ---------------------------------------------------------------------------


def _mk(cid, tier, q_lens, kv_lens, page, **kw) -> UnifiedBwdCase:
    return UnifiedBwdCase(
        id=f"bwd_unified_{cid}",
        tier=tier,
        q_lens=tuple(q_lens),
        kv_lens=tuple(kv_lens),
        page=page,
        **kw,
    )


def _page_cases():
    """Every page size of the plan's classes: 16, 32 ("sub" on CDNA), 48, 80
    (the one-wave fallback), 64, 128, 256 ("multi")."""
    out = []
    q = (17, 33, 5, 40)
    kv = (70, 77, 41, 100)
    smoke = {16, 64}
    for page in (16, 32, 48, 64, 80, 128, 256):
        tier = "smoke" if page in smoke else "full"
        out.append(_mk(f"page{page}_br", tier, q, kv, page, tags=("page_size",)))
    # fp16 and the other head dims on a "sub" and a "multi" page
    for page in (32, 128):
        for d, dt in ((32, "fp16"), (128, "bf16"), (64, "fp16")):
            out.append(
                _mk(
                    f"page{page}_d{d}_{dt}", "full", q, kv, page, d=d, dtype=dt,
                    tags=("page_size", "dtype"),
                )  # fmt: skip
            )
    return out


def _class_cases():
    """Each page class forced on pages another class would take."""
    out = []
    q, kv = (21, 9, 48), (90, 33, 128)
    out.append(
        _mk(
            "forced_fallback_page64", "full", q, kv, 64,
            knobs=(("tile", "fallback"),), tags=("page_class",),
        )  # fmt: skip
    )
    out.append(
        _mk(
            "forced_sub64_page64", "full", q, kv, 64,
            knobs=(("tile", "sub64"),), tags=("page_class",),
        )  # fmt: skip
    )
    out.append(
        _mk(
            "forced_fallback_page16_d128", "full", q, kv, 16, d=128,
            knobs=(("tile", "fallback"),), tags=("page_class",),
        )  # fmt: skip
    )
    # a 128-token sub page exists only where the default tile is wider
    # (kN0 = 256: the gfx950 d64 tile); other targets skip the case
    out.append(
        _mk(
            "forced_sub128_page128", "full", (40, 130, 9), (300, 257, 128), 128,
            knobs=(("tile", "sub128"),), tags=("page_class",),
        )  # fmt: skip
    )
    return out


def _table_cases():
    """Random, scattered, reversed and contiguous tables; last pages partial."""
    out = []
    q, kv = (30, 12, 50, 7), (95, 61, 70, 33)
    for table in TABLES:
        tier = "smoke" if table == "scattered" else "full"
        out.append(
            _mk(
                f"table_{table}", tier, q, kv, 32, table=table, spare_blocks=3,
                tags=("table",),
            )  # fmt: skip
        )
    out.append(
        _mk(
            "table_padded_rows", "full", q, kv, 16, table="random", bt_pad=5,
            extra_entries=3, tags=("table",),
        )  # fmt: skip
    )
    return out


def _zero_length_cases():
    """Zero-length query or kv spans, including the last sequence."""
    out = []
    out.append(
        _mk("zero_q_spans", "smoke", (0, 20, 0, 15, 0), (40, 33, 17, 64, 25), 16,
            tags=("zero_length",))  # fmt: skip
    )
    out.append(
        _mk("zero_kv_spans", "full", (10, 20, 31, 5), (0, 33, 0, 0), 32,
            tags=("zero_length",))  # fmt: skip
    )
    out.append(
        _mk("zero_both_last", "full", (12, 40, 0), (50, 81, 0), 64,
            tags=("zero_length",))  # fmt: skip
    )
    out.append(
        _mk("zero_q_last_br_page256", "full", (5, 0), (300, 120), 256,
            d=128, tags=("zero_length",))  # fmt: skip
    )
    return out


def _mask_cases():
    out = []
    q, kv = (30, 64, 9), (90, 64, 70)
    for mask in ("none", "tl", "br", "tl_win", "br_win"):
        tier = "smoke" if mask == "br_win" else "full"
        out.append(_mk(f"mask_{mask}", tier, q, kv, 32, mask=mask, tags=("mask",)))
    out.append(
        _mk("mask_tl_q_gt_kv", "full", (80, 40), (33, 21), 16, mask="tl",
            tags=("mask",))  # fmt: skip
    )
    return out


def _head_cases():
    out = []
    q, kv = (25, 3, 60), (80, 40, 100)
    out.append(_mk("gqa4", "smoke", q, kv, 32, h_q=8, h_kv=2, tags=("heads",)))
    out.append(_mk("mqa", "full", q, kv, 64, h_q=8, h_kv=1, tags=("heads",)))
    out.append(
        _mk("mqa_fp16_d128", "full", q, kv, 128, h_q=4, h_kv=1, d=128,
            dtype="fp16", tags=("heads",))  # fmt: skip
    )
    # short query spans: the plan packs the group's heads on the M rows
    out.append(
        _mk("decode_like_gqa8", "smoke", (1, 1, 2, 1), (90, 17, 300, 64), 16,
            h_q=8, h_kv=1, tags=("heads", "head_pack"))  # fmt: skip
    )
    out.append(
        _mk("short_spans_gqa4_page128", "full", (3, 1, 2), (140, 7, 129), 128,
            h_q=8, h_kv=2, tags=("heads", "head_pack"))  # fmt: skip
    )
    out.append(
        _mk("gsplit2_gqa4", "full", q, kv, 32, h_q=8, h_kv=2,
            knobs=(("g_split", 2),), modes=("atomic",), tags=("heads", "g_split"))  # fmt: skip
    )
    return out


def _shared_cases():
    """Shared-prefix tables (non-injective): atomic page mode only."""
    out = []
    out.append(
        _mk("shared_prefix2", "smoke", (20, 9, 33), (80, 64, 70), 16,
            shared_prefix=2, modes=("atomic",), tags=("shared",))  # fmt: skip
    )
    out.append(
        _mk("shared_prefix1_page64_gqa", "full", (40, 1, 17, 5), (100, 64, 130, 70),
            64, h_q=8, h_kv=2, shared_prefix=1, modes=("atomic",),
            tags=("shared",))  # fmt: skip
    )
    return out


def _canary_cases():
    """Past-length slots, page > kN0 and roundup(len_kv, page) > max_seqlen_kv."""
    out = []
    # len_kv = max_seqlen_kv = 100, page 256, kN0 128 (d128): one CTA covers
    # keys [0, 128); slots 128..255 of the page need the zero-fill tile.
    out.append(
        _mk("canary_page256_len100", "smoke", (37,), (100,), 256, d=128,
            spare_blocks=1, tags=("canary",))  # fmt: skip
    )
    out.append(
        _mk("canary_slack", "full", (20, 33), (45, 17), 128, kv_slack=60, q_slack=9,
            extra_entries=2, tags=("canary",))  # fmt: skip
    )
    return out


def _layout_cases():
    out = []
    q, kv = (21, 40, 6), (64, 90, 33)
    out.append(
        _mk("layout_padded", "full", q, kv, 32, layout="padded", tags=("layout",))
    )
    out.append(
        _mk("layout_narrow", "full", q, kv, 32, layout="narrow", tags=("layout",))
    )
    out.append(
        _mk("layout_narrow_page48", "exhaustive", q, kv, 48, layout="narrow",
            tags=("layout",))  # fmt: skip
    )
    return out


def _knob_cases():
    """Knob variants of the shared tile step on a paged problem."""
    out = []
    q, kv = (33, 70, 12), (100, 70, 64)
    for name, knobs in (
        ("edge_tiles", (("edge_tiles", True),)),
        ("epilogue_direct", (("dkv_epilogue", "direct"),)),
        ("epilogue_pair", (("dkv_epilogue", "lds_staged_pair"),)),
        ("atomic_select", (("dkv_atomic_guard", "select"),)),
        ("lb_reverse", (("lb_order", "reverse"),)),
        ("head_outer", (("step_order", "head_outer"), ("edge_tiles", True))),
        # body levers built apart from the unified shell (the merged barrier,
        # the counted waits and the G4 key slicing are CDNA levers: on RDNA
        # the unified plan resolves them to their defaults)
        ("barrier_merge", (("barrier_merge", "on"),)),
        ("counted", (("loop_sync", "counted"),)),
        ("stage_table_atomics", (("sched", "stage_table_atomics"),)),
        ("xt_lds", (("transpose_source", "xt_lds"),)),
        ("g4_slice_db", (("g4_slicing", "slice_db"), ("block_k4", 32))),
        ("valu", (("bwd_exp2", "fast"), ("stats_fma", True), ("stats_vec", 4))),
    ):
        out.append(_mk(f"knob_{name}", "full", q, kv, 64, h_q=8, h_kv=2, knobs=knobs,
                       tags=("knob",)))  # fmt: skip
    return out


def _many_sequences():
    out = []
    lens = tuple(range(1, 25))
    out.append(
        _mk("many_short", "full", lens, tuple(reversed(lens)), 16, h_q=4, h_kv=2,
            tags=("many",))  # fmt: skip
    )
    out.append(
        _mk("long_mixed_d128", "exhaustive", (300, 1, 513, 77), (700, 1, 513, 1030),
            128, d=128, h_q=4, h_kv=2, table="random", tags=("many",))  # fmt: skip
    )
    return out


_BUILDERS = (
    _page_cases,
    _class_cases,
    _table_cases,
    _zero_length_cases,
    _mask_cases,
    _head_cases,
    _shared_cases,
    _canary_cases,
    _layout_cases,
    _knob_cases,
    _many_sequences,
)
_CACHE = None


def unified_cases() -> tuple[UnifiedBwdCase, ...]:
    global _CACHE
    if _CACHE is None:
        out = []
        for build in _BUILDERS:
            out.extend(build())
        _CACHE = tuple(out)
    return _CACHE


def select_unified(tier: str = "full") -> list[UnifiedBwdCase]:
    """Cases of ``tier`` and every cheaper tier (``smoke`` <= ``full`` <= ``exhaustive``)."""
    rank = TIERS.index(tier)
    return [c for c in unified_cases() if TIERS.index(c.tier) <= rank]


def get_unified_case(case_id: str) -> UnifiedBwdCase:
    for c in unified_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------


@dataclass
class UnifiedBwdInputs:
    """A materialised unified problem.

    ``buffers``: flat float32 storage (values rounded to the dtype; NaN in
    gaps and in never-read slots) of ``q o do k_cache v_cache``; ``lse`` is
    ``[total_q * h_q]`` fp32. ``strides`` / ``dims`` / ``sizes`` describe every
    tensor (gradients included). ``block_table`` is int32 ``[num_seqs,
    bt_stride]``. ``ref`` is the gathered-THD oracle; ``exp_dk`` / ``exp_dv``
    are the expected gradient caches, ``referenced`` marks the pages a live
    block references, ``past_len`` the referenced slots no sequence covers.
    """

    case: UnifiedBwdCase
    cu_seqlens_q: np.ndarray
    seq_lens_kv: np.ndarray
    block_table: np.ndarray
    num_blocks: int
    max_blocks_per_seq: int
    bt_stride: int
    trap: int
    buffers: dict
    strides: dict
    dims: dict
    sizes: dict
    lse: np.ndarray
    ref: SdpaBwdReference
    exp_dk: np.ndarray
    exp_dv: np.ndarray
    referenced: np.ndarray
    past_len: np.ndarray
    scale: float


def _strides_of(case: UnifiedBwdCase, num_blocks: int) -> dict:
    """``(token, head, D)`` q-side and ``(block, token, head, D)`` cache strides."""
    d, hq, hk, page = case.d, case.h_q, case.h_kv, case.page
    pad = {"packed": (0, 0), "padded": (8, 16), "narrow": (3, 7)}[case.layout]
    q_t = hq * d + pad[0]
    tok = hk * d + (5 if case.layout == "narrow" else pad[0])
    blk = page * tok + pad[1]
    out = {n: (q_t, d, 1) for n in ("q", "o", "do", "dq")}
    out.update(
        {n: (blk, tok, d, 1) for n in ("k_cache", "v_cache", "dk_cache", "dv_cache")}
    )
    return out


def _span(dims, strides) -> int:
    return 1 + sum((n - 1) * s for n, s in zip(dims, strides)) if min(dims) > 0 else 0


def _view(buf, dims, strides):
    item = buf.itemsize
    return np.lib.stride_tricks.as_strided(
        buf, shape=dims, strides=tuple(s * item for s in strides), writeable=True
    )


def _assign_pages(case: UnifiedBwdCase, rng) -> tuple:
    """``(block_table rows, num_blocks, trap, live pairs)``.

    Live logical blocks ``(s, j)``, ``j < ceil(kv_len / page)``, get physical
    ids by the case's table kind; the first ``shared_prefix`` blocks of every
    sequence share ids; ``spare_blocks`` ids stay unreferenced and one more is
    the trap page.
    """
    page = case.page
    n_blk = [-(-n // page) for n in case.kv_lens]
    sp = case.shared_prefix
    unique = sp + sum(max(n - sp, 0) for n in n_blk)
    num_blocks = unique + case.spare_blocks + 1
    if case.table == "contiguous":
        pool = list(range(num_blocks))
    elif case.table == "reversed":
        pool = list(range(num_blocks - 1, -1, -1))
    elif case.table == "random":
        pool = list(rng.permutation(num_blocks))
    else:  # scattered: live pages interleaved with the unreferenced ones
        evens = list(range(0, num_blocks, 2))
        odds = list(range(1, num_blocks, 2))
        pool = list(rng.permutation(evens)) + list(rng.permutation(odds))
    live_ids = pool[:unique]
    trap = int(pool[-1])
    shared_ids = live_ids[:sp]
    rest = iter(live_ids[sp:])
    rows = []
    for s, n in enumerate(n_blk):
        row = []
        for j in range(n):
            row.append(int(shared_ids[j]) if j < sp else int(next(rest)))
        rows.append(row)
    return rows, num_blocks, trap, n_blk


def materialize_unified(case: UnifiedBwdCase) -> UnifiedBwdInputs:
    """Deterministic inputs, block table, oracle and expected caches of ``case``."""
    errs = validate_unified_case(case)
    if errs:
        raise ValueError(f"{case.id}: {errs}")
    rng = np.random.default_rng((zlib.crc32(case.id.encode()) ^ _SEED) & 0xFFFFFFFF)
    page, d, hq, hk, dt = case.page, case.d, case.h_q, case.h_kv, case.dtype
    B = len(case.q_lens)
    rows, num_blocks, trap, n_blk = _assign_pages(case, rng)
    need = max(-(-case.max_seqlen_kv // page), max(n_blk + [1]))
    mbps = need + case.extra_entries
    bt_stride = mbps + case.bt_pad
    table = np.full((B, bt_stride), trap, np.int32)
    for s, row in enumerate(rows):
        table[s, : len(row)] = row

    total_q = sum(case.q_lens)
    total_kv = sum(case.kv_lens)
    strides = _strides_of(case, num_blocks)
    cache_dims = (num_blocks, page, hk, d)
    dims = {n: (total_q, hq, d) for n in ("q", "o", "do", "dq")}
    dims.update({n: cache_dims for n in ("k_cache", "v_cache", "dk_cache", "dv_cache")})
    sizes = {n: _span(dims[n], strides[n]) for n in dims}

    # physical caches: random values; NaN in unreferenced pages, past-length
    # slots and stride gaps (none of them may ever be read)
    live = np.zeros((num_blocks, page), bool)
    referenced = np.zeros(num_blocks, bool)
    for s, row in enumerate(rows):
        for j, pid in enumerate(row):
            referenced[pid] = True
            n = min(page, case.kv_lens[s] - j * page)
            live[pid, :n] = True
    buffers = {}
    caches = {}
    for name in ("k_cache", "v_cache"):
        vals = round_to(rng.standard_normal(cache_dims), dt).astype(np.float32)
        vals[~live] = np.nan
        buf = np.full(sizes[name], np.nan, np.float32)
        _view(buf, cache_dims, strides[name])[...] = vals
        buffers[name] = buf
        caches[name] = vals
    # gathered contiguous K / V of every sequence (THD)
    kv_off = np.concatenate([[0], np.cumsum(case.kv_lens)]).astype(np.int64)
    q_off = np.concatenate([[0], np.cumsum(case.q_lens)]).astype(np.int64)
    slot_of = []  # (pid, slot) of every gathered kv token
    for s in range(B):
        for t in range(case.kv_lens[s]):
            slot_of.append((rows[s][t // page], t % page))
    pids = np.array([p for p, _ in slot_of], np.int64).reshape(-1)
    slots = np.array([x for _, x in slot_of], np.int64).reshape(-1)
    kg = caches["k_cache"][pids, slots].astype(np.float64).reshape(total_kv, hk, d)
    vg = caches["v_cache"][pids, slots].astype(np.float64).reshape(total_kv, hk, d)
    q = round_to(rng.standard_normal((total_q, hq, d)), dt).astype(np.float64)
    left, right, tl = case.band
    scale = case.scale if case.scale is not None else 1.0 / math.sqrt(d)
    kw = dict(
        scale=scale, left_bound=left, right_bound=right, top_left=tl,
        q_offsets=q_off, kv_offsets=kv_off,
    )  # fmt: skip
    o_ref, lse_ref = sdpa_reference(q, kg, vg, **kw)
    o = round_to(o_ref, dt).astype(np.float64)
    lse = lse_ref.astype(np.float32)
    do = round_to(rng.standard_normal((total_q, hq, d)), dt).astype(np.float64)
    res = sdpa_reference_bwd(q, kg, vg, o, do, lse.astype(np.float64), **kw)
    for name, vals in (("q", q), ("o", o), ("do", do)):
        buf = np.full(sizes[name], np.nan, np.float32)
        if total_q:
            _view(buf, dims[name], strides[name])[...] = vals
        buffers[name] = buf
    valid_q = np.ones((total_q, hq), bool)
    ref = SdpaBwdReference(
        dq=res.dq,
        dk=res.dk,
        dv=res.dv,
        delta=res.delta,
        valid_q=valid_q,
        valid_kv=np.ones(total_kv, bool),
        dead_q=valid_q & np.isneginf(lse.astype(np.float64)),
    )
    exp_dk = np.zeros(cache_dims, np.float64)
    exp_dv = np.zeros(cache_dims, np.float64)
    if total_kv:
        np.add.at(exp_dk, (pids, slots), res.dk)
        np.add.at(exp_dv, (pids, slots), res.dv)
    past_len = referenced[:, None] & ~live
    return UnifiedBwdInputs(
        case=case,
        cu_seqlens_q=q_off.astype(np.int32),
        seq_lens_kv=np.array(case.kv_lens, np.int32),
        block_table=table,
        num_blocks=num_blocks,
        max_blocks_per_seq=mbps,
        bt_stride=bt_stride,
        trap=trap,
        buffers=buffers,
        strides=strides,
        dims=dims,
        sizes=sizes,
        lse=np.ascontiguousarray(lse.reshape(-1), np.float32),
        ref=ref,
        exp_dk=exp_dk,
        exp_dv=exp_dv,
        referenced=referenced,
        past_len=past_len,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def compare_unified(inp: UnifiedBwdInputs, dq, dk_cache, dv_cache) -> list[str]:
    """Kernel gradients vs the oracle; returns a list of problems.

    ``dq`` is ``[total_q, h_q, D]``; ``dk_cache`` / ``dv_cache`` are the full
    ``[num_blocks, page, h_kv, D]`` caches (decoded). dQ uses the backward
    table's bound (:func:`.bwd_cases.compare_bwd`); dK / dV are compared slot
    by slot over every slot of every referenced page (same norm-relative
    global and per-row bounds); slots of referenced pages that no sequence
    covers must be exact zeros.
    """
    ref = inp.ref
    sel = np.broadcast_to(inp.referenced[:, None], inp.past_len.shape)
    got_k = np.asarray(dk_cache, np.float64)[sel]
    got_v = np.asarray(dv_cache, np.float64)[sel]
    want_k = inp.exp_dk[sel]
    want_v = inp.exp_dv[sel]
    slots = SdpaBwdReference(
        dq=ref.dq,
        dk=want_k,
        dv=want_v,
        delta=ref.delta,
        valid_q=ref.valid_q,
        valid_kv=np.ones(want_k.shape[0], bool),
        dead_q=ref.dead_q,
    )
    probs = compare_bwd(inp.case, dq, got_k, got_v, slots)
    for name, got in (("dk", dk_cache), ("dv", dv_cache)):
        past = np.asarray(got, np.float64)[inp.past_len]
        if past.size and not np.all(past == 0.0):
            bad = int(np.sum(past != 0.0))
            probs.append(
                f"{name}: {bad} values in past-length slots are not exact zeros"
            )
    return probs
