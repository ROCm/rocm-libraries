# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ATOM mono layer kernel's schedule and scratch layout, as rocKE data
(``mailbox_layout = atom_compat``).

Mirrors ``ATOM/atom/models/qwen4_exp_mono/kernels/layout.py`` (rows = 1):
``TASKS``, ``BASE``, ``ORDERS``, ``REGIONS`` and the derived ``SC`` offsets,
``SCRATCH_BYTES``, ``MAX_ROWS``; the numbers are computed from the same
Qwen3.8-Flash-Next geometry with the same formulas. A test compares them
with the ATOM tree when it is available (``Q38_ATOM_DIR``). rocKE never
imports ATOM.

On top of ATOM's data, each stage's dataflow (the mailbox regions it reads
and writes, and how its tasks depend on the producers': full = all-to-all,
``covers`` = every producer task read by someone) is written down from the
ATOM stage sources, so :meth:`TaskTable.validate` can run its dataflow and
cross-layer write-after-read checks on the real schedule (roadmap 3.2: a
multi-layer launch is safe today only through A1's all-to-all wait on M5;
any overlap lever must keep, or replace, that order).
"""

from __future__ import annotations

from rocke.helpers.task_schedule import Dep, Stage, TaskTable

from .framework import ScratchLayout

__all__ = [
    "BASE",
    "GEOMETRY",
    "MAX_ROWS",
    "ORDERS",
    "PHASES",
    "REGIONS",
    "SC",
    "SCRATCH_BYTES",
    "TASKS",
    "atom_layout",
    "atom_stack",
    "atom_table",
]

GRID = 256
GEOMETRY = dict(
    hidden=2560, hc_count=4, hc_lowrank=320, delta_k_heads=16, delta_v_heads=48,
    delta_k_dim=128, delta_v_dim=128, q_heads=24, kv_heads=2, head_dim=256,
    index_heads=4, index_dim=128, index_budget=2048, index_compress=4,
    experts=512, top_k=10, moe_inter=640,
)  # fmt: skip
_C = GEOMETRY
_DELTA_KEY = _C["delta_k_heads"] * _C["delta_k_dim"]  # 2048
_DELTA_VALUE = _C["delta_v_heads"] * _C["delta_v_dim"]  # 6144
_DELTA_QKV = 2 * _DELTA_KEY + _DELTA_VALUE  # 10240
_HC_WIDTH = _C["hc_count"] * _C["hidden"]  # 10240
_QSA_PROJ = (
    _C["index_heads"] * _C["index_dim"] + _C["index_dim"]  # index q, index k
    + _C["q_heads"] * _C["head_dim"] * 2  # q and gate per head
    + 2 * _C["kv_heads"] * _C["head_dim"]  # k, v
)  # fmt: skip

_HC_IN = ("a1", "a2")
_OUT_MLP = ("c1", "m1", "m2", "m3", "m4", "m5")
ORDERS = {
    "delta": _HC_IN + ("b1", "b3") + _OUT_MLP,
    "qsa": _HC_IN + ("q1", "q2", "q3", "q4", "q5", "q6", "q7") + _OUT_MLP,
}
ORDERS["delta_ple"] = ("p1", "p2") + ORDERS["delta"]

QSA_LIST = 4 * (_C["index_budget"] // _C["index_compress"]) + 3
SCORE_TILE = 32
SCORE_TILES = -(-QSA_LIST // SCORE_TILE)  # 65
SS_ROW = SCORE_TILE * SCORE_TILES
TASKS = {
    "a1": _C["hc_lowrank"] // 32 + _C["hc_count"],
    "a2": _C["hidden"] // 32,
    "b1": (_DELTA_QKV + _DELTA_VALUE) // 64,
    "b3": _C["delta_v_heads"] * 4,
    "c1": _C["hidden"] // 16,
    "m1": _C["hc_lowrank"] // 32 + _C["hc_count"],
    "m2": _C["hidden"] // 32,
    "m3": _C["experts"] // 8 + 1,
    "m4": (_C["top_k"] + 1) * (_C["moe_inter"] // 64),
    "m5": _C["hidden"] // 16,
    "q1": _QSA_PROJ // 64,
    "q2": _C["q_heads"] + _C["kv_heads"],
    "q3": GRID,
    "q4": 9,
    "q5": _C["kv_heads"] * SCORE_TILES,
    "q6": _C["q_heads"],
    "q7": _C["kv_heads"] * 3 * 8,
    "p1": _C["hidden"] // 16,
    "p2": _C["hidden"] // 32,
}
BASE = {
    "a1": 0, "a2": 176, "b1": 0, "b3": 0, "c1": 192,
    "m1": 96, "m2": 110, "m3": 190, "m4": 156, "m5": 0,
    "q1": 0, "q2": 218, "q3": 0, "q4": 74, "q5": 128, "q6": 2, "q7": 26,
    "p1": 0, "p2": 160,
}  # fmt: skip
QSA_BLOCKS = 32768
REGIONS = {
    "lr8": _C["hc_lowrank"] // 4,
    "lre": _C["hc_lowrank"] // 32,
    "x8": _C["hidden"] // 4,
    "xe": _C["hidden"] // 32,
    "xbf": _C["hidden"] // 2,
    "inj": _C["hc_count"],
    "qkvz": (_DELTA_QKV + _DELTA_VALUE) // 2,
    "core": TASKS["b3"] * 32,
    "d8": _DELTA_VALUE // 4,
    "de": _DELTA_VALUE // 32,
    "w": _HC_WIDTH // 2,
    "whs": _C["hc_count"] * TASKS["c1"],
    "mlr8": _C["hc_lowrank"] // 4,
    "mlre": _C["hc_lowrank"] // 32,
    "mx8": _C["hidden"] // 4,
    "mxe": _C["hidden"] // 32,
    "mxbf": _C["hidden"] // 2,
    "minj": _C["hc_count"],
    "lg": _C["experts"] // 2,
    "sg": 1,
    "dn": (_C["top_k"] + 1) * (_C["moe_inter"] // 4 + _C["moe_inter"] // 32),
    "qp": _QSA_PROJ // 2,
    "qn": (_C["q_heads"] + _C["kv_heads"]) * _C["head_dim"] // 2,
    "isc": QSA_BLOCKS,
    "sel": _C["index_budget"] // _C["index_compress"],
    "selh": 1,
    "ss": _C["q_heads"] * SS_ROW,
    "ssh": TASKS["q5"],
    "pp": _C["q_heads"] * SS_ROW // 2,
    "pph": _C["q_heads"],
    "pk": _HC_WIDTH // 2,
    "pv": _C["hidden"] // 2,
    "ph": _HC_WIDTH // 2,
    "phs": _C["hc_count"] * TASKS["c1"],
    "inv": _C["hc_count"],
    "minv": _C["hc_count"],
}
MAX_ROWS = 4
PAIR = 8
SC: dict[str, int] = {}
_off = 0
for _n, _p in REGIONS.items():
    SC[_n] = _off
    _off += (_p * PAIR + 255) // 256 * 256
SCRATCH_BYTES = _off


def atom_layout(parity_slots: int = 1, rows: int = MAX_ROWS) -> ScratchLayout:
    """ATOM's scratch regions (row 0 of every region at ``SC``; rows
    ``SCRATCH_BYTES`` apart). ``parity_slots=2`` is the native-mode double
    buffer of a multi-layer launch (not ATOM's layout)."""
    return ScratchLayout.of(REGIONS, rows=rows, parity_slots=parity_slots)


def _full(*names: str) -> tuple[Dep, ...]:
    return tuple(Dep(n) for n in names)


def _cov(name: str) -> Dep:
    return Dep(name, full=False, covers=True)


def _part(name: str) -> Dep:
    return Dep(name, full=False, covers=False)


# stage -> (deps, reads, writes); from the ATOM stage sources (c.mb_* uses)
_FLOW = {
    "a1": ((), ("h",), ("lr8", "lre", "inj", "inv")),
    "a2": (_full("a1"), ("lr8", "lre", "inv"), ("x8", "xe", "xbf")),
    "b1": (_full("a2"), ("x8", "xe"), ("qkvz",)),
    "b3": ((_cov("b1"), Dep("a2")), ("qkvz", "xbf"), ("core", "d8", "de")),
    "c1": ((Dep("b3"), _part("a1")), ("d8", "de", "inj", "h"), ("w", "whs")),
    "m1": (_full("c1"), ("w", "whs"), ("mlr8", "mlre", "minj", "minv")),
    "m2": (_full("m1"), ("mlr8", "mlre", "minv"), ("mx8", "mxe", "mxbf")),
    "m3": (_full("m2"), ("mxbf",), ("lg", "sg")),
    "m4": (_full("m3", "m2"), ("lg", "mx8", "mxe"), ("dn",)),
    "m5": ((Dep("m4"), Dep("m3"), _part("m1"), _cov("c1")),
           ("dn", "lg", "sg", "minj", "w"), ("hout",)),  # fmt: skip
    "q1": (_full("a2"), ("x8", "xe"), ("qp",)),
    "q2": ((_cov("q1"),), ("qp",), ("qn",)),
    # Q3 / Q4 (qsa_fast): scores in isc plus the fast path's hint regions
    # (fisch, fch, fcand, past the mailbox regions); Q4's 8 chunk tasks
    # feed its final task, which every Q5 task polls (covers, transitively)
    "q3": ((_part("q1"),), ("qp",), ("isc", "fisch")),
    "q4": (_full("q3"), ("isc", "fisch", "fch", "fcand"), ("fch", "fcand", "sel", "selh")),
    "q5": ((_cov("q4"), _cov("q2")), ("sel", "selh", "qn"), ("ss", "ssh")),
    "q6": ((_cov("q5"),), ("ss", "ssh"), ("pp", "pph")),
    "q7": ((_cov("q6"), _part("q4")), ("pp", "pph", "sel"), ("d8", "de")),
    "p1": ((), ("ple_in",), ("pk", "pv")),
    # P2 hands rows to the next row through isc (ple.py)
    "p2": (_full("p1"), ("pk", "pv", "h", "isc"), ("ph", "phs", "isc")),
}
_QSA_C1 = ((Dep("q7"), _part("a1")), ("d8", "de", "inj", "h"), ("w", "whs"))
_PLE_A1 = (_full("p2"), ("ph", "phs"), ("lr8", "lre", "inj", "inv"))
_PLE_C1 = ((Dep("b3"), _part("a1"), _cov("p2")), ("d8", "de", "inj", "ph"), ("w", "whs"))


# LDS overlay phases (ATOM layout: _ATTN, _MOE, _QSA, _ATTN_QSA, _PLE tables)
PHASES = {
    **{n: "attn" for n in ("a1", "a2", "b1", "b3", "c1", "m1", "m2", "q1", "q2")},
    **{n: "moe" for n in ("m3", "m4", "m5")},
    **{n: "qsa" for n in ("q3", "q4")},
    **{n: "attn_qsa" for n in ("q5", "q6", "q7")},
    **{n: "ple" for n in ("p1", "p2")},
}


def atom_table(kind: str, *, stack: bool = False, parity: tuple[str, ...] = ()) -> TaskTable:
    """ATOM's rows = 1 schedule of ``kind`` as a :class:`TaskTable`.

    ``stack``: model a multi-layer launch: the first stage also depends,
    through the previous layer, on M5 (the layer output ``h`` it polls), as
    a ``build_stack`` would make it."""
    stages = []
    for n in ORDERS[kind]:
        deps, reads, writes = _FLOW[n]
        if kind == "qsa" and n == "c1":
            deps, reads, writes = _QSA_C1
        if kind == "delta_ple" and n == "a1":
            deps, reads, writes = _PLE_A1
        if kind == "delta_ple" and n == "c1":
            deps, reads, writes = _PLE_C1
        deps = tuple(deps)
        if stack and n == ORDERS[kind][0]:
            deps = deps + (Dep("m5", layer=-1),)
            reads = tuple(reads) + ("hout",)
        stages.append(Stage(n, TASKS[n], base=BASE[n], deps=deps, reads=tuple(reads),
                            writes=tuple(writes), lds_phase=PHASES[n]))  # fmt: skip
    inputs = ("h", "ple_in") + (() if stack else ("hout",))
    return TaskTable(tuple(stages), grid=GRID, inputs=inputs, parity=parity, kind=kind)


def atom_stack(kinds, *, parity: tuple[str, ...] = ()) -> list[TaskTable]:
    """Tables of a multi-layer launch running layers of ``kinds`` in order
    (``validate_stack`` input), e.g. the model's layers 0..3:
    ``("delta", "delta_ple", "delta", "qsa")``."""
    return [atom_table(k, stack=True, parity=parity) for k in kinds]
