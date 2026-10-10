# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fragment helpers for the attention backward: lane-map facts, the checked
C-to-A register relabel, the K-permuted operand load and the LDS fragment
loader.

Everything here is driven by the catalog lane maps of
:class:`rocke.core.arch.target.MmaOp` (``a_layout``, ``b_layout``,
``c_layout``). The maps are evaluated twice:

* on an *integer* builder (:class:`IntLaneBuilder`) for every lane and slot, at
  build time, to prove facts (contiguity, affinity, the relabel) and to derive
  the per-slot coordinate deltas;
* on the real :class:`~rocke.core.ir.IRBuilder` to emit the lane-dependent base
  coordinate of slot 0.

C-to-A relabel
--------------
The backward computes ``S = Q K^T`` (C fragment, rows = queries, cols = keys)
and then needs ``P^T`` as the *A* operand of ``dV += P^T dO`` (rows = keys,
K = queries); likewise ``dS^T`` for ``dK += dS^T Q``. :func:`plan_c_to_a`
finds how a consumer A fragment is assembled from ``c_tiles`` producer C
fragments stacked along the consumer K, and proves it on every lane and slot:

* ``"direct"``: each lane already holds every value its A fragment needs; the
  fragment is a cast of the C registers (MFMA 16x16 atoms, gfx12 WMMA). For a
  consumer atom with K = 32 (gfx950 ``16x16x32``) the two concatenated C tiles
  hold the 32 K values in a *permuted* order; the plan records that permutation
  ``k_perm`` (catalog K index -> logical K index) and the other operand must be
  loaded with the same permutation (:func:`kperm_b_coords`) so that the sum
  over K is unchanged.
* ``"xlane16"``: half of the values live in the ``lane ^ 16`` partner (gfx11
  WMMA). The fragment is built with one ``permlanex16`` and two ``perm_b32``
  per packed dword; :func:`simulate_relabel` replays that exact program on
  coordinates and the plan is accepted only if it reproduces the A map.

Anything else raises :class:`ValueError` (the caller then routes the hand-off
through LDS). No ``ds_read_tr16_b128`` is ever emitted here.

Transposed operand reads
------------------------
:class:`LdsView` is a 2-D window into one flat LDS allocation (the backward
places every buffer of a CTA inside one allocation at the phase planner's
offsets); :func:`load_frag_lds` accepts it in place of a 2-D allocation.
:func:`load_frag_lds_tr16` reads a K-outer operand with the gfx950
``ds_read_tr16_b64`` (64-bit form only): each lane passes its own address
(:func:`tr16_lane_address`), and the delivered coordinates
(:func:`tr16_model_coords`) are proved equal to the catalog operand map at
build time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from math import gcd

from rocke.core.arch.target import LayoutMap, MmaOp
from rocke.core.ir import BF16, F16, I32, IRBuilder, Value, VectorType

__all__ = [
    "AtomFacts",
    "FragCoords",
    "IntLaneBuilder",
    "LdsView",
    "RelabelPlan",
    "atom_facts",
    "check_tr16_coords",
    "emit_c_to_a",
    "eval_layout",
    "ir_type",
    "kperm_b_coords",
    "load_frag_lds",
    "load_frag_lds_tr16",
    "operand_coords",
    "plan_c_to_a",
    "simulate_relabel",
    "tr16_lane_address",
    "tr16_model_coords",
]


# ---------------------------------------------------------------------------
# Integer evaluation of the catalog lane maps
# ---------------------------------------------------------------------------


class IntLaneBuilder:
    """Duck-typed stand-in for ``IRBuilder`` that evaluates index math on ints.

    The catalog lane maps only use ``const_i32``, ``add``, ``sub``, ``mul``,
    ``div`` and ``mod``; the bit operations are provided for helper formulas.
    """

    def const_i32(self, v: int) -> int:
        return int(v)

    def add(self, a: int, b: int) -> int:
        return a + b

    def sub(self, a: int, b: int) -> int:
        return a - b

    def mul(self, a: int, b: int) -> int:
        return a * b

    def div(self, a: int, b: int) -> int:
        return a // b

    def mod(self, a: int, b: int) -> int:
        return a % b

    def xor(self, a: int, b: int) -> int:
        return a ^ b

    def shl(self, a: int, b: int) -> int:
        return a << b

    def lshr(self, a: int, b: int) -> int:
        return a >> b


_INT = IntLaneBuilder()


def eval_layout(layout: LayoutMap) -> tuple[tuple[tuple[int, int], ...], ...]:
    """``[lane][slot] -> (coord0, coord1)`` for every lane and slot of ``layout``."""
    return tuple(
        tuple(
            tuple(int(c) for c in layout.coord(_INT, lane, s))
            for s in range(layout.frag_len)
        )
        for lane in range(layout.wave_size)
    )


def _mn_k(role: str, coord: tuple[int, int]) -> tuple[int, int]:
    """Normalise a coordinate to ``(m-or-n, k)``: A is ``(row, k)``, B ``(k, col)``."""
    return (coord[0], coord[1]) if role == "a" else (coord[1], coord[0])


def _layout(op: MmaOp, role: str) -> LayoutMap:
    return {"a": op.a_layout, "b": op.b_layout, "c": op.c_layout}[role]()


@cache
def _eval_cached(layout: LayoutMap):
    # Keyed on the LayoutMap (its lane function is part of the hash), never on
    # the MmaOp: MmaOp equality ignores the maps, so two ops with different maps
    # would otherwise share one evaluation.
    return eval_layout(layout)


def _eval(op: MmaOp, role: str):
    return _eval_cached(_layout(op, role))


# ---------------------------------------------------------------------------
# Atom facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomFacts:
    """Build-time lane-map facts of one atom (all lanes and slots evaluated)."""

    op_id: str
    a_contig: bool  # a lane's A slots: one row, consecutive K
    b_contig: bool  # a lane's B slots: one column, consecutive K
    affine: bool  # every role: slot coord = slot-0 coord + lane-independent delta
    c_is_aT: bool  # one C tile is the A fragment of the transposed product
    ab_same_k: bool  # A and B carry the same K index in every lane and slot


def _contig(op: MmaOp, role: str) -> bool:
    for lane_coords in _eval(op, role):
        mn0, k0 = _mn_k(role, lane_coords[0])
        for s, c in enumerate(lane_coords):
            if _mn_k(role, c) != (mn0, k0 + s):
                return False
    return True


def _affine(op: MmaOp, role: str) -> bool:
    ev = _eval(op, role)
    base_d = None
    for lane_coords in ev:
        c0 = lane_coords[0]
        d = tuple((c[0] - c0[0], c[1] - c0[1]) for c in lane_coords)
        if base_d is None:
            base_d = d
        elif d != base_d:
            return False
    return True


def atom_facts(op: MmaOp) -> AtomFacts:
    a, b_, c = _eval(op, "a"), _eval(op, "b"), _eval(op, "c")
    c_is_aT = op.c_frag_len == op.a_frag_len and all(
        c[lane][s] == (a[lane][s][1], a[lane][s][0])
        for lane in range(op.wave_size)
        for s in range(op.a_frag_len)
    )
    ab_same_k = op.a_frag_len == op.b_frag_len and all(
        a[lane][s][1] == b_[lane][s][0]
        for lane in range(op.wave_size)
        for s in range(op.a_frag_len)
    )
    return AtomFacts(
        op_id=op.op_id,
        a_contig=_contig(op, "a"),
        b_contig=_contig(op, "b"),
        affine=all(_affine(op, r) for r in ("a", "b", "c")),
        c_is_aT=c_is_aT,
        ab_same_k=ab_same_k,
    )


# ---------------------------------------------------------------------------
# C-to-A relabel plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelabelPlan:
    """How a consumer A fragment is assembled from producer C fragments.

    ``slot_src[s] = (tile, cslot)`` for ``kind == "direct"``.  ``xlane[m] =
    (j, sel_lo_half, sel_hi_half)`` for ``kind == "xlane16"``: packed A dword
    ``m`` is ``perm_b32(own_j, partner_j, sel)`` with ``sel`` chosen by the lane
    half. ``k_perm[k_catalog] = k_logical`` (identity when no permutation).
    """

    c_op_id: str
    a_op_id: str
    kind: str  # "direct" | "xlane16"
    c_tiles: int
    c_tile_rows: int
    slot_src: tuple[tuple[int, int], ...] = ()
    xlane: tuple[tuple[int, int, int], ...] = ()
    k_perm: tuple[int, ...] = ()

    @property
    def needs_k_perm(self) -> bool:
        return self.k_perm != tuple(range(len(self.k_perm)))


def _logical_a(c_coord: tuple[int, int], tile: int, rows: int) -> tuple[int, int]:
    """C value at (row, col) of tile ``tile`` is consumer A element (col, tile*rows+row)."""
    return (c_coord[1], tile * rows + c_coord[0])


def _try_direct(c_op: MmaOp, a_op: MmaOp, tiles: int) -> RelabelPlan | None:
    cf = c_op.c_frag_len
    if a_op.a_frag_len != tiles * cf:
        return None
    a, c = _eval(a_op, "a"), _eval(c_op, "c")
    slot_src = tuple((s // cf, s % cf) for s in range(a_op.a_frag_len))
    perm: dict[int, int] = {}
    for lane in range(a_op.wave_size):
        for s, (t, cs) in enumerate(slot_src):
            row, k_log = _logical_a(c[lane][cs], t, c_op.m)
            a_row, a_k = a[lane][s]
            if row != a_row:
                return None
            if perm.setdefault(a_k, k_log) != k_log:
                return None
    if sorted(perm) != list(range(a_op.k)) or sorted(perm.values()) != list(
        range(a_op.k)
    ):
        return None
    return RelabelPlan(
        c_op_id=c_op.op_id,
        a_op_id=a_op.op_id,
        kind="direct",
        c_tiles=tiles,
        c_tile_rows=c_op.m,
        slot_src=slot_src,
        k_perm=tuple(perm[k] for k in range(a_op.k)),
    )


def _try_xlane16(c_op: MmaOp, a_op: MmaOp, tiles: int) -> RelabelPlan | None:
    ws = a_op.wave_size
    if ws != 32 or tiles != 1 or a_op.a_frag_len != 2 * c_op.c_frag_len:
        return None
    if a_op.a_frag_len % 2 or c_op.c_frag_len % 2:
        return None
    a, c = _eval(a_op, "a"), _eval(c_op, "c")
    # where[lane][(row, k)] -> cslot for the values a lane holds
    held = [{_logical_a(c[lane][cs], 0, c_op.m): cs for cs in range(c_op.c_frag_len)}
            for lane in range(ws)]  # fmt: skip
    # per half: A slot -> (from_partner, cslot); must be uniform over the half
    per_half: list[list[tuple[int, int]] | None] = [None, None]
    for lane in range(ws):
        h = lane // 16
        srcs = []
        for s in range(a_op.a_frag_len):
            need = a[lane][s]
            if need in held[lane]:
                srcs.append((0, held[lane][need]))
            elif need in held[lane ^ 16]:
                srcs.append((1, held[lane ^ 16][need]))
            else:
                return None
        if per_half[h] is None:
            per_half[h] = srcs
        elif per_half[h] != srcs:
            return None
    xlane = []
    for m in range(a_op.a_frag_len // 2):
        js, sels = set(), []
        for h in (0, 1):
            sel = 0
            for pos, s in enumerate((2 * m, 2 * m + 1)):
                partner, cs = per_half[h][s]
                js.add(cs // 2)
                byte0 = (0 if partner else 4) + 2 * (cs % 2)
                sel |= byte0 << (16 * pos)
                sel |= (byte0 + 1) << (16 * pos + 8)
            sels.append(sel)
        if len(js) != 1:
            return None
        xlane.append((js.pop(), sels[0], sels[1]))
    plan = RelabelPlan(
        c_op_id=c_op.op_id,
        a_op_id=a_op.op_id,
        kind="xlane16",
        c_tiles=1,
        c_tile_rows=c_op.m,
        xlane=tuple(xlane),
        k_perm=tuple(range(a_op.k)),
    )
    if not _proved(plan, c_op, a_op):
        return None
    return plan


def _check_pair(c_op: MmaOp, a_op: MmaOp) -> int:
    if c_op.wave_size != a_op.wave_size:
        raise ValueError("producer and consumer atoms have different wave sizes")
    if c_op.n != a_op.m:
        raise ValueError(
            f"C columns ({c_op.n}) must equal the consumer A rows ({a_op.m})"
        )
    if a_op.k % c_op.m:
        raise ValueError(
            f"consumer K ({a_op.k}) is not a multiple of the C tile rows ({c_op.m})"
        )
    return a_op.k // c_op.m


def plan_c_to_a(c_op: MmaOp, a_op: MmaOp) -> RelabelPlan:
    """Prove and return the register relabel from ``c_op`` C to ``a_op`` A.

    Raises :class:`ValueError` when neither the direct relabel nor the
    ``lane ^ 16`` exchange reproduces the consumer A map on every lane and slot.
    """
    tiles = _check_pair(c_op, a_op)
    plan = _try_direct(c_op, a_op, tiles) or _try_xlane16(c_op, a_op, tiles)
    if plan is None or not _proved(plan, c_op, a_op):
        raise ValueError(
            f"no register C-to-A relabel from {c_op.op_id} to {a_op.op_id}: the "
            f"lane maps do not match; route the hand-off through LDS"
        )
    return plan


# ---------------------------------------------------------------------------
# Simulation (the proof): replay the emission program on coordinates
# ---------------------------------------------------------------------------


def _perm_sim(src0: tuple, src1: tuple, sel: int) -> tuple:
    """``v_perm_b32`` on byte-labelled dwords: bytes 0..3 from src1, 4..7 from src0."""
    pool = tuple(src1) + tuple(src0)
    out = []
    for i in range(4):
        idx = (sel >> (8 * i)) & 0xFF
        if idx > 7:
            raise ValueError(f"selector byte {idx:#x} is not a plain byte select")
        out.append(pool[idx])
    return tuple(out)


def simulate_relabel(plan: RelabelPlan, c_op: MmaOp, a_op: MmaOp):
    """Replay the emission program of ``plan`` on logical coordinates.

    Returns ``[lane][a_slot] -> (consumer_row, logical_k)``: the element of the
    transposed product each A register ends up holding.
    """
    c = _eval(c_op, "c")
    ws = a_op.wave_size
    out = []
    if plan.kind == "direct":
        for lane in range(ws):
            out.append(
                tuple(
                    _logical_a(c[lane][cs], t, plan.c_tile_rows)
                    for t, cs in plan.slot_src
                )
            )
        return tuple(out)
    if plan.kind != "xlane16":
        raise ValueError(f"unknown relabel kind {plan.kind!r}")

    # bytes: each 16-bit element contributes (elem, 0) and (elem, 1)
    def dwords(lane):
        vals = [_logical_a(c[lane][cs], 0, plan.c_tile_rows) for cs in range(len(c[0]))]
        return [
            ((vals[2 * j], 0), (vals[2 * j], 1), (vals[2 * j + 1], 0), (vals[2 * j + 1], 1))
            for j in range(len(vals) // 2)
        ]  # fmt: skip

    own = [dwords(lane) for lane in range(ws)]
    for lane in range(ws):
        partner = own[lane ^ 16]  # permlanex16: source lane = lane ^ 16
        slots = []
        for j, sel0, sel1 in plan.xlane:
            sel = sel0 if lane // 16 == 0 else sel1
            d = _perm_sim(own[lane][j], partner[j], sel)
            for half in (0, 1):
                lo, hi = d[2 * half], d[2 * half + 1]
                if lo[0] != hi[0] or (lo[1], hi[1]) != (0, 1):
                    raise ValueError("relabel program splits a 16-bit element")
                slots.append(lo[0])
        out.append(tuple(slots))
    return tuple(out)


def _proved(plan: RelabelPlan, c_op: MmaOp, a_op: MmaOp) -> bool:
    a = _eval(a_op, "a")
    try:
        got = simulate_relabel(plan, c_op, a_op)
    except ValueError:
        return False
    for lane in range(a_op.wave_size):
        for s in range(a_op.a_frag_len):
            row, k = a[lane][s]
            if got[lane][s] != (row, plan.k_perm[k]):
                return False
    return True


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def ir_type(dtype: str):
    if dtype in ("fp16", "f16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"operand dtype must be fp16 or bf16, got {dtype!r}")


def _concat_all(b: IRBuilder, parts: list[Value]) -> Value:
    while len(parts) > 1:
        nxt = [
            b.vec_concat(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)
        ]
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


def _assemble(b: IRBuilder, chunks: list[Value], elem) -> Value:
    """Concatenate vector chunks (in slot order) into one fragment."""
    if len(chunks) == 1:
        return chunks[0]
    sizes = {ch.type.count for ch in chunks}
    if len(sizes) == 1 and (len(chunks) & (len(chunks) - 1)) == 0:
        return _concat_all(b, list(chunks))
    elems = [b.vec_extract(ch, i) for ch in chunks for i in range(ch.type.count)]
    return b.vec_pack(elems, elem)


def emit_c_to_a(
    b: IRBuilder,
    plan: RelabelPlan,
    c_tiles: Sequence[Value],
    *,
    dtype: str,
    lane: Value | None = None,
) -> Value:
    """Emit the consumer A fragment from ``plan.c_tiles`` C fragments (f32).

    The C values are rounded to ``dtype`` (fp16 / bf16). ``lane`` (the lane id
    in the wave) is needed only for the ``"xlane16"`` exchange.
    """
    if len(c_tiles) != plan.c_tiles:
        raise ValueError(f"plan needs {plan.c_tiles} C tiles, got {len(c_tiles)}")
    elem = ir_type(dtype)
    cast = [b.vec_cast_f32_to(t, elem) for t in c_tiles]
    if plan.kind == "direct":
        cf = cast[0].type.count
        if plan.slot_src == tuple((s // cf, s % cf) for s in range(len(plan.slot_src))):
            return _assemble(b, cast, elem)
        elems = [b.vec_extract(cast[t], cs) for t, cs in plan.slot_src]
        return b.vec_pack(elems, elem)
    if plan.kind != "xlane16":
        raise ValueError(f"unknown relabel kind {plan.kind!r}")
    if lane is None:
        raise ValueError("the lane ^ 16 relabel needs the lane id")
    v = cast[0]
    n_dw = v.type.count // 2
    words = b.vec_bitcast(v, VectorType(I32, n_dw))
    own = [b.vec_extract(words, j) for j in range(n_dw)]
    partner: dict[int, Value] = {}
    is_lo = b.cmp_lt(lane, b.const_i32(16))
    sels: dict[tuple[int, int], Value] = {}
    out = []
    for j, s0, s1 in plan.xlane:
        if j not in partner:
            partner[j] = b.permlanex16(own[j])
        key = (s0, s1)
        if key not in sels:
            sels[key] = b.select(is_lo, b.const_i32(_i32(s0)), b.const_i32(_i32(s1)))
        out.append(b.perm_b32(own[j], partner[j], sels[key]))
    packed = b.vec_pack(out, I32)
    return b.vec_bitcast(packed, VectorType(elem, 2 * len(out)))


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


# ---------------------------------------------------------------------------
# Operand coordinates and the LDS fragment loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FragCoords:
    """Per-lane operand coordinates: slot-0 base (emitted) plus slot deltas.

    Coordinates are ``(mn, k)``: ``mn`` is the A row or the B column.
    """

    role: str
    frag_len: int
    wave_size: int
    deltas: tuple[tuple[int, int], ...]
    base: Callable = field(repr=False, compare=False)
    kperm: bool = False

    def emit_base(self, b, lane):
        return self.base(b, lane)

    def int_coords(self, lane: int) -> tuple[tuple[int, int], ...]:
        mn, k = self.base(_INT, lane)
        return tuple((mn + dm, k + dk) for dm, dk in self.deltas)


def _deltas_of(per_lane: Sequence[Sequence[tuple[int, int]]]):
    d0 = None
    for coords in per_lane:
        m0, k0 = coords[0]
        d = tuple((m - m0, k - k0) for m, k in coords)
        if d0 is None:
            d0 = d
        elif d != d0:
            raise ValueError("operand lane map is not affine in the slot index")
    return d0


def operand_coords(op: MmaOp, role: str) -> FragCoords:
    """Catalog coordinates of operand ``role`` (``"a"`` or ``"b"``) of ``op``."""
    if role not in ("a", "b"):
        raise ValueError(f"role must be 'a' or 'b', got {role!r}")
    layout = _layout(op, role)
    per_lane = [[_mn_k(role, c) for c in lane] for lane in _eval(op, role)]
    deltas = _deltas_of(per_lane)

    def base(b, lane, _layout=layout, _role=role):
        return _mn_k(_role, _layout.coord(b, lane, 0))

    return FragCoords(
        role=role,
        frag_len=layout.frag_len,
        wave_size=layout.wave_size,
        deltas=deltas,
        base=base,
    )


def kperm_b_coords(plan: RelabelPlan, c_op: MmaOp, b_op: MmaOp) -> FragCoords:
    """B-operand coordinates with the K permutation of ``plan`` applied.

    B slot ``s`` of lane ``l`` holds logical K ``k_perm[k_B(l, s)]``. This is
    legal only when the consumer A and B maps carry the same K in every lane
    and slot (``ab_same_k``); then the logical K of slot ``s`` is the producer
    C row of ``plan.slot_src[s]``, which is how the base is emitted.
    """
    if plan.kind != "direct":
        raise ValueError("a K-permuted load exists only for the direct relabel")
    if b_op.op_id != plan.a_op_id:
        raise ValueError(f"plan consumer is {plan.a_op_id}, not {b_op.op_id}")
    if not atom_facts(b_op).ab_same_k:
        raise ValueError(f"{b_op.op_id}: A and B K maps differ; K permutation unsafe")
    bl = _eval(b_op, "b")
    per_lane = [
        [(c[1], plan.k_perm[c[0]]) for c in bl[lane]] for lane in range(b_op.wave_size)
    ]
    deltas = _deltas_of(per_lane)
    c_layout = c_op.c_layout()
    b_layout = b_op.b_layout()
    t0, cs0 = plan.slot_src[0]
    rows = plan.c_tile_rows

    def base(b, lane):
        k_row, _ = c_layout.coord(b, lane, cs0)
        k = b.add(k_row, b.const_i32(t0 * rows)) if t0 else k_row
        _, col = b_layout.coord(b, lane, 0)
        return col, k

    fc = FragCoords(
        role="b",
        frag_len=b_op.b_frag_len,
        wave_size=b_op.wave_size,
        deltas=deltas,
        base=base,
        kperm=True,
    )
    for lane in range(b_op.wave_size):  # the emitted base must equal the proof
        if fc.int_coords(lane) != tuple(per_lane[lane]):
            raise ValueError("K-permuted B base does not reproduce the permutation")
    return fc


_WIDTHS = (8, 4, 2, 1)


def _runs(deltas, along: int):
    """Split slots into runs contiguous along coordinate ``along`` (0 mn, 1 k)."""
    other = 1 - along
    runs: list[list[int]] = []
    for s, d in enumerate(deltas):
        if runs:
            p = deltas[runs[-1][-1]]
            if d[other] == p[other] and d[along] == p[along] + 1:
                runs[-1].append(s)
                continue
        runs.append([s])
    return runs


def load_frag_lds(
    b: IRBuilder,
    coords: FragCoords,
    smem: Value,
    lane: Value,
    *,
    dtype: str,
    mn0: Value | int = 0,
    k0: Value | int = 0,
    storage: str = "k_inner",
    align: int = 8,
) -> Value:
    """Load one operand fragment from a 2-D LDS tile.

    ``storage = "k_inner"`` indexes ``smem[mn][k]`` (K contiguous: vector
    reads along K); ``"k_outer"`` indexes ``smem[k][mn]``. ``mn0``/``k0`` are
    the tile origin; the caller guarantees that the contiguous dimension's
    pitch and origin are multiples of ``align`` elements (vector widths never
    exceed what that and the lane bases allow).
    """
    if storage not in ("k_inner", "k_outer"):
        raise ValueError(f"storage must be 'k_inner' or 'k_outer', got {storage!r}")
    if isinstance(smem, LdsView):
        align = gcd(align, smem.align_elems)
    elem = ir_type(dtype)
    along = 1 if storage == "k_inner" else 0
    # alignment of the contiguous coordinate over every lane's run start
    g = align
    runs = _runs(coords.deltas, along)
    for lane_i in range(coords.wave_size):
        cc = coords.int_coords(lane_i)
        for r in runs:
            g = gcd(g, cc[r[0]][along])
    mn, k = coords.emit_base(b, lane)
    if not (isinstance(mn0, int) and mn0 == 0):
        mn = b.add(mn, b.const_i32(mn0) if isinstance(mn0, int) else mn0)
    if not (isinstance(k0, int) and k0 == 0):
        k = b.add(k, b.const_i32(k0) if isinstance(k0, int) else k0)
    chunks: list[Value] = []
    for r in runs:
        pos = 0
        while pos < len(r):
            rem = len(r) - pos
            w = next(w for w in _WIDTHS if w <= rem and g % w == 0 and rem % w == 0)
            dm, dk = coords.deltas[r[pos]]
            im = b.add(mn, b.const_i32(dm)) if dm else mn
            ik = b.add(k, b.const_i32(dk)) if dk else k
            idx = (im, ik) if storage == "k_inner" else (ik, im)
            if isinstance(smem, LdsView):
                chunks.append(smem.load(b, *idx, dtype=elem, n=w))
            else:
                chunks.append(b.smem_load_vN(smem, *idx, dtype=elem, n=w))
            pos += w
    return _assemble(b, chunks, elem)


# ---------------------------------------------------------------------------
# Views into one flat LDS allocation and the gfx950 transpose-read loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LdsView:
    """A ``[rows][pitch]`` 2-D window into one flat 1-D LDS allocation.

    ``smem`` is a 1-D ``smem_alloc`` of 16-bit elements; ``base`` is the
    window origin in those elements and ``pitch`` the row pitch in *view*
    elements. ``scale`` is the number of allocation elements per view element
    (1 for 16-bit views, 2 for an fp32 view of the same allocation). Every
    access becomes ``smem[base + (row * pitch + col) * scale]``, so the phase
    planner's byte offsets map one to one onto the IR.

    ``align_elems`` is the largest power of two (in view elements, at most 8)
    that divides both the origin and the pitch; vector accesses never claim a
    larger alignment.
    """

    smem: Value
    base: int
    pitch: int
    rows: int
    scale: int = 1

    def __post_init__(self) -> None:
        if self.base < 0 or self.pitch < 1 or self.rows < 1 or self.scale not in (1, 2):
            raise ValueError(f"illegal LDS view {self}")
        if self.base % self.scale:
            raise ValueError("an fp32 view must start on a 4-byte boundary")

    @property
    def align_elems(self) -> int:
        a = 8
        while a > 1 and ((self.base // self.scale) % a or self.pitch % a):
            a //= 2
        return a

    def index(self, b, row, col) -> Value:
        """Flat allocation index of view element ``(row, col)`` (i32)."""
        row_off = row * self.pitch if isinstance(row, int) else None
        if row_off is None:
            r = b.mul(row, b.const_i32(self.pitch)) if self.pitch != 1 else row
        else:
            r = b.const_i32(row_off)
        lin = r if (isinstance(col, int) and col == 0) else b.add(r, _c(b, col))
        if self.scale != 1:
            lin = b.mul(lin, b.const_i32(self.scale))
        return b.add(lin, b.const_i32(self.base)) if self.base else lin

    def load(self, b, row, col, *, dtype, n: int) -> Value:
        """``n`` elements from ``(row, col)`` (``col`` a multiple of ``n``).

        A vector wider than :attr:`align_elems` (a padded row pitch) is split
        into aligned pieces, so no access claims more alignment than it has.
        """
        a = self.align_elems
        if n <= a:
            return b.smem_load_vN(self.smem, self.index(b, row, col), dtype=dtype, n=n)
        parts = [
            b.smem_load_vN(
                self.smem, self.index(b, row, _offset(b, col, k)), dtype=dtype, n=a
            )
            for k in range(0, n, a)
        ]
        return b.vec_pack([b.vec_extract(p, i) for p in parts for i in range(a)], dtype)

    def store(self, b, row, col, value: Value, n: int) -> None:
        """Store ``n`` elements at ``(row, col)``; split like :meth:`load`."""
        a = self.align_elems
        if n <= a:
            b.smem_store_vN(self.smem, [self.index(b, row, col)], value, n)
            return
        elem = value.type.elem
        for k in range(0, n, a):
            piece = [b.vec_extract(value, k + i) for i in range(a)]
            part = piece[0] if a == 1 else b.vec_pack(piece, elem)
            idx = self.index(b, row, _offset(b, col, k))
            b.smem_store_vN(self.smem, [idx], part, a)

    def tr16_load(self, b, row, col, *, dtype) -> Value:
        """``ds_read_tr16_b64`` with this lane's own address ``(row, col)``."""
        return b.ds_read_tr16_b64(self.smem, self.index(b, row, col), dtype=dtype)


def _c(b, x):
    return b.const_i32(x) if isinstance(x, int) else x


def _offset(b, col, k: int):
    """``col + k`` (``col`` an int or an i32 value)."""
    if k == 0:
        return col
    return col + k if isinstance(col, int) else b.add(col, b.const_i32(k))


def tr16_lane_address(lane: int, t: int, kpl: int) -> tuple[int, int]:
    """``(k, mn)`` offset that lane ``lane`` supplies for transpose read ``t``.

    The per-lane map of ``ds_read_tr16_b64`` (validated on the device by the
    transpose probe): lane ``l`` addresses K row ``(l // 16) * kpl +
    (l % 16) // 4 + 4 t`` and column ``(l % 4) * 4`` of a K-outer tile; it is
    *not* the tile origin for every lane.
    """
    return ((lane // 16) * kpl + (lane % 16) // 4 + 4 * t, (lane % 4) * 4)


def tr16_model_coords(lane: int, kpl: int) -> tuple[tuple[int, int], ...]:
    """``(mn, k)`` the transpose reads deliver to ``lane`` (hardware model).

    Within a 16-lane group ``G``, output lane ``16 G + j`` slot ``r`` of read
    ``t`` is element ``j % 4`` of the 4 elements addressed by lane
    ``16 G + 4 r + j // 4``.
    """
    g, j = divmod(lane, 16)
    out = []
    for t in range(kpl // 4):
        for r in range(4):
            src = 16 * g + 4 * r + j // 4
            k, mn = tr16_lane_address(src, t, kpl)
            out.append((mn + j % 4, k))
    return tuple(out)


@cache
def _tr16_proved(coords_key: tuple) -> bool:
    kpl, per_lane = coords_key
    return all(
        tr16_model_coords(lane, kpl) == per_lane[lane] for lane in range(len(per_lane))
    )


def check_tr16_coords(coords: FragCoords) -> int:
    """Prove the transpose read delivers ``coords``; returns elements per lane.

    Raises :class:`ValueError` for a wave size other than 64, a fragment that is
    not a whole number of 4-element reads, or any lane / slot whose model
    coordinate differs from ``coords`` (for example a K-permuted map).
    """
    kpl = coords.frag_len
    if coords.wave_size != 64 or kpl % 4 or kpl not in (4, 8):
        raise ValueError(
            f"ds_read_tr16_b64 needs a wave64 fragment of 4 or 8 elements "
            f"(got {kpl} on wave{coords.wave_size})"
        )
    per_lane = tuple(coords.int_coords(lane) for lane in range(coords.wave_size))
    if not _tr16_proved((kpl, per_lane)):
        raise ValueError(
            "the ds_read_tr16_b64 lane map does not deliver this operand's "
            "coordinates; use the plain K-outer reader"
        )
    return kpl


def load_frag_lds_tr16(
    b: IRBuilder,
    coords: FragCoords,
    smem: "LdsView",
    lane: Value,
    *,
    dtype: str,
    arch: str,
    mn0: Value | int = 0,
    k0: Value | int = 0,
) -> Value:
    """Load one operand fragment from a K-outer tile with ``ds_read_tr16_b64``.

    The tile is stored ``smem[k][mn]`` (mn contiguous). One 64-bit transpose
    read per 4 K values (K = 32 fragments take two reads, concatenated in slot
    order); the 128-bit form is never used. Every lane passes its *own*
    address (:func:`tr16_lane_address`). Raises :class:`ValueError` before any
    IR is emitted when ``arch`` has no 64-bit transpose read, the fragment map
    is not the one the read delivers, or a lane's 8-byte read could be
    misaligned.
    """
    from ._attention_bwd_caps import check_transpose_read

    check_transpose_read(arch, 64)
    if not isinstance(smem, LdsView):
        raise ValueError("the transpose-read loader needs an LdsView")
    kpl = check_tr16_coords(coords)
    if smem.scale != 1 or smem.base % 4 or smem.pitch % 4:
        raise ValueError("transpose reads need a 16-bit view with origin and pitch % 4")
    if isinstance(mn0, int) and mn0 % 4:
        raise ValueError(f"transpose-read column origin {mn0} is not a multiple of 4")
    elem = ir_type(dtype)
    c16 = b.const_i32(16)
    g = b.div(lane, c16)
    j = b.mod(lane, c16)
    row_l = b.add(b.mul(g, b.const_i32(kpl)), b.div(j, b.const_i32(4)))
    col_l = b.mul(b.mod(j, b.const_i32(4)), b.const_i32(4))
    if not (isinstance(k0, int) and k0 == 0):
        row_l = b.add(row_l, _c(b, k0))
    if not (isinstance(mn0, int) and mn0 == 0):
        col_l = b.add(col_l, _c(b, mn0))
    reads = []
    for t in range(kpl // 4):
        row = b.add(row_l, b.const_i32(4 * t)) if t else row_l
        reads.append(smem.tr16_load(b, row, col_l, dtype=elem))
    return _assemble(b, reads, elem)
