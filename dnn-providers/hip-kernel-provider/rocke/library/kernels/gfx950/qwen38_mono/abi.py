# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The FlyDSL Qwen3.8 mono layer kernel's ABI, mirrored for rocKE stage builds.

A rocKE stage that runs inside (hybrid) or next to (split launch) the FlyDSL
mono layer kernel must read and write the same mailbox regions and weights.
This module mirrors, from the same formulas and without importing the model
code, what the mono kernel's ``kernels/layout.py``, ``kernels/kernel.py`` and
``stages/moe.py`` define:

* the launch arguments (``LAYER_ARGS``: ``step`` i64, ``layer`` i32, then
  41 i64 device addresses, kernarg offset of argument ``k`` = ``16 + 8 k``);
* the scratch mailbox regions (``REGIONS`` in pairs, ``SC`` byte offsets,
  ``SCRATCH_BYTES`` per request row, ``ROW_PAIRS``), the launch tag and the
  stamp region (``ATOM_Q38_STAMPS``);
* the MoE weight blocks (gate_up / down tiles and scale dwords per expert);
* the mono MoE stages' task counts and CTA bases per request-row count
  (``layout.for_rows``), so a split build can keep mono's placement.

The GPU test ``library/tests/test_qwen38_moe_gfx950_numeric.py`` checks every
value here against the model code on a node that has it.
"""

from __future__ import annotations

from types import SimpleNamespace

GRID = 256
THREADS = 512
WAVES = THREADS // 64
LAYER_SLOTS = 64
MAX_ROWS = 4
PAIR = 8

# ---------------------------------------------------------------- geometry
HIDDEN = 2560
HC_COUNT = 4
HC_LOWRANK = 320
HC_WIDTH = HC_COUNT * HIDDEN
HC_WORDS = HC_WIDTH // 2  # bf16 words of one request row of h_in / h_out
DELTA_K_HEADS, DELTA_V_HEADS, DELTA_K_DIM, DELTA_V_DIM = 16, 48, 128, 128
DELTA_KEY = DELTA_K_HEADS * DELTA_K_DIM
DELTA_VALUE = DELTA_V_HEADS * DELTA_V_DIM
DELTA_QKV = 2 * DELTA_KEY + DELTA_VALUE
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 2, 256
INDEX_HEADS, INDEX_DIM, INDEX_BUDGET, INDEX_COMPRESS = 4, 128, 2048, 4
QSA_PROJ = 2 * Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM + INDEX_HEADS * INDEX_DIM + INDEX_DIM
EXPERTS = 512  # the model's; the CI fixture builds 16
TOP_K = 10
MOE_INTER = 640

LAYER_ARGS = (
    "scratch", "ple", "h_in", "hc_norm_w", "w_down", "s_down", "w_up", "s_up", "w_inj",
    "w_qkvz", "s_qkvz", "conv_w", "conv_in", "conv_out", "ba_w", "a_log", "dt_bias",
    "norm_w", "rec_in", "rec_out", "w_qsa", "s_qsa", "qsa", "w_out", "s_out", "m_norm_w",
    "m_w_down", "m_s_down", "m_w_up", "m_s_up", "m_w_inj", "r_w", "e_w_gu", "e_s_gu",
    "sh_w_gu", "sh_s_gu", "e_w_dn", "e_s_dn", "sh_w_dn", "sh_s_dn", "h_out",
)  # fmt: skip

# ---------------------------------------------------------------- mailboxes
_TASKS_B3 = DELTA_V_HEADS * 4
_TASKS_C1 = HIDDEN // 16
_SCORE_TILE = 32
_QSA_LIST = 4 * (INDEX_BUDGET // INDEX_COMPRESS) + 3
_SS_ROW = _SCORE_TILE * (-(-_QSA_LIST // _SCORE_TILE))
_TASKS_Q5 = KV_HEADS * (-(-_QSA_LIST // _SCORE_TILE))
QSA_BLOCKS = 32768

REGIONS = {
    "lr8": HC_LOWRANK // 4, "lre": HC_LOWRANK // 32, "x8": HIDDEN // 4, "xe": HIDDEN // 32,
    "xbf": HIDDEN // 2, "inj": HC_COUNT, "qkvz": (DELTA_QKV + DELTA_VALUE) // 2,
    "core": _TASKS_B3 * 32, "d8": DELTA_VALUE // 4, "de": DELTA_VALUE // 32,
    "w": HC_WIDTH // 2, "whs": HC_COUNT * _TASKS_C1, "mlr8": HC_LOWRANK // 4,
    "mlre": HC_LOWRANK // 32, "mx8": HIDDEN // 4, "mxe": HIDDEN // 32, "mxbf": HIDDEN // 2,
    "minj": HC_COUNT, "lg": EXPERTS // 2, "sg": 1,
    "dn": (TOP_K + 1) * (MOE_INTER // 4 + MOE_INTER // 32),
    "qp": QSA_PROJ // 2, "qn": (Q_HEADS + KV_HEADS) * HEAD_DIM // 2, "isc": QSA_BLOCKS,
    "sel": INDEX_BUDGET // INDEX_COMPRESS, "selh": 1, "ss": Q_HEADS * _SS_ROW,
    "ssh": _TASKS_Q5, "pp": Q_HEADS * _SS_ROW // 2, "pph": Q_HEADS, "pk": HC_WIDTH // 2,
    "pv": HIDDEN // 2, "ph": HC_WIDTH // 2, "phs": HC_COUNT * _TASKS_C1, "inv": HC_COUNT,
    "minv": HC_COUNT,
}  # fmt: skip
DN_X8 = MOE_INTER // 4  # a dn row: 160 fp8 words, then the exponents
DN_ROW = DN_X8 + MOE_INTER // 32  # 180 pairs

SC: dict[str, int] = {}
_off = 0
for _name, _pairs in REGIONS.items():
    SC[_name] = _off
    _off += (_pairs * PAIR + 255) // 256 * 256
SCRATCH_BYTES = _off
ROW_PAIRS = SCRATCH_BYTES // PAIR


def mailbox_bytes(rows: int) -> int:
    """``layout.mailbox_bytes``: every row's regions (MAX_ROWS at least), twice."""
    return 2 * max(rows, MAX_ROWS) * SCRATCH_BYTES


# ``ATOM_Q38_STAMPS`` stage stamps (``kernels/stamps.py``): per CTA ``POINTS``
# int64 s_memrealtime ticks -- point 0 the entry, per stage i enter (1 + 10 i),
# exit (2 + 10 i), ready of wave w (3 + 10 i + w) -- written at the end to
# ``scratch + STAMP_BASE + ((layer * GRID + cta) * POINTS + point) * 8``.
STAMP_POINTS = 160
STAMP_PER_STAGE = 2 + WAVES
STAMP_BASE = mailbox_bytes(MAX_ROWS)
STAMP_REGION_BYTES = LAYER_SLOTS * GRID * STAMP_POINTS * 8


def pair_ix(i: int, row: int = 0) -> int:
    """Pair ``i`` of request row ``row`` of a region (``common.mb_ix``)."""
    return i + row * ROW_PAIRS


# ---------------------------------------------------------------- MoE weights
def k_iters(k: int) -> int:
    """``p2.k_iters``: 128-K steps of a stored MXFP4 weight (K padded to 256)."""
    return 2 * ((k + 255) // 256)


GU_KI = k_iters(HIDDEN)  # 20
GU_K1 = GU_KI // 2  # scale dwords per lane of a gate/up block pair
GU_PAIRS = k_iters(MOE_INTER) * 128 // 16  # 48 (40 used; the intermediate is padded to 768)
GU_W_BYTES = 2 * GU_PAIRS * GU_KI * 1024
GU_S_BYTES = GU_PAIRS * GU_K1 * 64 * 4
GU_TILES = MOE_INTER // 64  # 10 slices of 64 columns
DN_KI = k_iters(MOE_INTER)  # 6 (k_iter 5 is zero)
DN_STEPS = MOE_INTER // 128  # 5
DN_K1 = (DN_STEPS + 1) // 2  # 3
DN_W_BYTES = HIDDEN // 16 * DN_KI * 1024
DN_S_BYTES = HIDDEN // 32 * (DN_KI // 2) * 64 * 4

# Router (M3): a weight row is ``HIDDEN`` bf16; the 16-part K split.
ROUTER_PARTS = tuple((192 * p, 192) if p < 8 else (128 * p + 512, 128) for p in range(16))
RW_PART = 100
RW_ROW = len(ROUTER_PARTS) * RW_PART
RX_PART = 196
ROW_WORDS = HIDDEN // 2

# ---------------------------------------------------------------- placement
_TASKS = {"m1": HC_LOWRANK // 32 + HC_COUNT, "m2": HIDDEN // 32}
_BASE = {"m1": 96, "m2": 110, "m3": 190, "m4": 156, "m5": 0}


def mono_moe_layout(rows: int, experts: int = EXPERTS) -> SimpleNamespace:
    """The mono MoE stages' task counts and bases at ``rows`` request rows
    (``layout.for_rows``): ``tasks``/``base`` for m3, m4, m5, ``m3_task_rows``,
    ``m4_slices``. Task counts follow the build's expert count (``m3``: one
    task per 8 router rows, then the shared gate's)."""
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"rows must be in 1..{MAX_ROWS}, got {rows}")
    m3_one = experts // 8 + 1
    m3_full = EXPERTS // 8 + 1  # placement uses the model's count
    base = dict(_BASE)
    if rows == 1:
        m3_task_rows, m4_slices = 1, 1
    else:
        m1 = rows * _TASKS["m1"]
        base["m2"] = base["m1"] + m1
        base["m3"] = (base["m2"] + _TASKS["m2"]) % GRID
        m3_task_rows = next(
            d for d in range(1, rows + 1) if rows % d == 0 and rows // d * m3_full <= GRID
        )
        m4_slices = 1 if (rows * TOP_K + 1) * GU_TILES <= GRID else 2
    tasks = {
        "m3": rows // m3_task_rows * m3_one,
        "m4": (rows * TOP_K + 1) * (GU_TILES // m4_slices),
        "m5": HIDDEN // 16,
    }
    return SimpleNamespace(
        rows=rows, tasks=tasks, base={k: base[k] for k in ("m3", "m4", "m5")},
        m3_task_rows=m3_task_rows, m4_slices=m4_slices,
    )  # fmt: skip


__all__ = [
    "DN_K1", "DN_KI", "DN_ROW", "DN_S_BYTES", "DN_STEPS", "DN_W_BYTES", "DN_X8", "EXPERTS",
    "GRID", "GU_K1", "GU_KI", "GU_PAIRS", "GU_S_BYTES", "GU_TILES", "GU_W_BYTES", "HC_COUNT",
    "HC_WORDS", "HIDDEN", "LAYER_ARGS", "LAYER_SLOTS", "MAX_ROWS", "MOE_INTER", "PAIR",
    "REGIONS", "ROUTER_PARTS", "ROW_PAIRS", "ROW_WORDS", "RW_PART", "RW_ROW", "RX_PART", "SC",
    "SCRATCH_BYTES", "STAMP_BASE", "STAMP_PER_STAGE", "STAMP_POINTS", "STAMP_REGION_BYTES",
    "THREADS", "TOP_K", "WAVES", "k_iters", "mailbox_bytes", "mono_moe_layout", "pair_ix",
]  # fmt: skip
