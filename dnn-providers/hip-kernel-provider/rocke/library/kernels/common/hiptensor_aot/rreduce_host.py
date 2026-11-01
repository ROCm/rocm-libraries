# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side rules of the tuned reduction classes (:mod:`.rreduce`): knob validity, launch
arguments, grids, workspace, the finalize launch and the untuned knob choice.

hipTensor's rocKE backend implements the same rules in C++ (``rocke_rreduce_rule.hpp``, used by
``rocke_plan.cpp`` and by the C++ pack builder, whose ``--rreduce-picks`` mode rocKE's pack
cross-check test compares with :func:`untuned_choices`); this module is their Python statement,
used by rocKE's tests and tuning tools. Problems are given in hipTensor's canonical form: ``kept`` = ``(extent, A stride,
D stride)`` and ``red`` = ``(extent, A stride)``, each ordered by A's strides, innermost first,
extent-1 modes dropped and modes contiguous in every tensor merged (C has D's strides).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .common import MAX_RANK, as_i32_bits, fnv1a64, magic_u31
from .rreduce import (
    LOAD_VECS,
    SLOT_COUNTS,
    STREAMING_ARCHS,
    RreduceFinalSpec,
    RreduceSpec,
)

I32_LIMIT = 1 << 31
GRID_YZ_MAX = 65535
MAX_GRID_THREADS = (1 << 32) - 1
#: Threads the finalize aims to keep busy when there are few outputs.
FINAL_TARGET = 1 << 20
#: Threads per output of the finalize classes of the pack.
FINAL_LANES = (1, 4, 16, 64)
#: The route table of the classes (a data file of rocKE's dispatch tables, which the C++ engine
#: compiles in): per target the palette (classes of the pack the untuned choice picks from) and
#: the tuned problems (canonical problem -> knobs), see :func:`load_table`.
TABLE_NAME = "hiptensor_rreduce.json"


def table_path() -> str:
    import rocke.dispatch.families as fam

    return os.path.join(os.path.dirname(os.path.abspath(fam.__file__)), TABLE_NAME)


#: Data-type rows of the API's reductions (A = C = D).
ROWS = ("f16", "bf16", "f32", "f64")
ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}


@dataclass(frozen=True)
class Knobs:
    """A class's knobs plus the launch's split (a runtime argument)."""

    mode: str
    block: int
    vec: int
    lanes: int
    unroll: int
    split: int = 1
    rows: int = 0  # col only (0: block // lanes)
    nt: bool = False

    def as_dict(self) -> Dict[str, object]:
        return dict(
            mode=self.mode,
            block=self.block,
            vec=self.vec,
            lanes=self.lanes,
            unroll=self.unroll,
            split=self.split,
            rows=self.rows,
            nt=int(self.nt),
        )


def mode_of(kept, red) -> str:
    """hipTensor's rule: ``row`` when A's innermost mode is reduced."""
    n_out = math.prod(k[0] for k in kept)
    n_red = math.prod(r[0] for r in red)
    if n_red <= 1:
        return "col"
    if n_out == 1:
        return "row"
    return "row" if abs(red[0][1]) < abs(kept[0][1]) else "col"


def col_group(kept) -> int:
    """Number of kept modes (innermost first) contiguous in A from the innermost one on: the run
    a col class tiles."""
    if not kept:
        return 0
    n = 1
    while n < len(kept) and kept[n][1] == kept[n - 1][1] * kept[n - 1][0]:
        n += 1
    return n


def chunk_dims(red, vec: int) -> List[Tuple[int, int]]:
    """Row mode: the reduced slots in ``vec`` chunks (an inner mode of exactly ``vec`` elements
    gives no slot)."""
    d0, s0 = red[0]
    dims = [(d0 // vec, s0 * vec)] + [(e, s) for e, s in red[1:]]
    dims = [x for x in dims if x[0] != 1]
    return dims if dims else [(1, 0)]


def slot_count(n: int) -> Optional[int]:
    for c in SLOT_COUNTS:
        if n <= c:
            return c
    return None


def max_vec(kept, red, a: str, mode: str) -> int:
    if mode == "row":
        n, s = red[0]
    else:
        if not kept:
            return 1
        n, s = kept[0][0], kept[0][1]
    if s != 1:
        return 1
    best = 1
    for v in LOAD_VECS[a]:
        if n % v == 0:
            best = max(best, v)
    return best


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def loop_geometry(kept, red, k: Knobs) -> Tuple[int, int, int, int]:
    """``(n_idx, step, per, n_it)``: flat reduction indices (row: chunks, col: rows), visited
    ``step`` apart, ``per`` per split, ``n_it`` loop trips of ``unroll`` loads."""
    if k.mode == "row":
        n_idx = math.prod(e for e, _ in chunk_dims(red, k.vec))
        step = k.lanes
    else:
        n_idx = math.prod(e for e, _ in red) if red else 1
        step = k.rows if k.rows else k.block // k.lanes
    per = _ceil(n_idx, k.split)
    return n_idx, step, per, _ceil(_ceil(per, step), k.unroll)


def grid_x(kept, red, k: Knobs) -> int:
    n_out = math.prod(x[0] for x in kept)
    if k.mode == "row":
        return _ceil(n_out, k.block // k.lanes)
    ng = col_group(kept)
    m = math.prod(x[0] for x in kept[:ng])
    tiles = _ceil(m, k.lanes * k.vec)
    rt = k.rows if k.rows else k.block // k.lanes
    items = k.block // (k.lanes * rt)
    return _ceil((n_out // m) * tiles, items)


def check_knobs(
    kept, red, a: str, k: Knobs, *, wave: int = 64, arch: str = "gfx950"
) -> Optional[str]:
    """Why ``k`` cannot run the problem (None when it can)."""
    if k.mode not in ("row", "col"):
        return "mode"
    if any(x[1] < 0 or x[2] < 0 for x in kept) or any(x[1] < 0 for x in red):
        return "negative strides"
    if len(kept) > MAX_RANK or len(red) > MAX_RANK:
        return "more than 8 kept or reduced modes"
    if k.mode != mode_of(kept, red):
        return "mode differs from the problem's"
    if k.nt and arch not in STREAMING_ARCHS:
        return "streaming loads need gfx942 / gfx950"
    if not (k.block & (k.block - 1) == 0 and wave <= k.block <= 1024):
        return f"block {k.block}"
    if k.lanes < 1 or k.lanes & (k.lanes - 1) or k.lanes > k.block:
        return f"lanes {k.lanes}"
    if k.vec not in LOAD_VECS[a]:
        return f"vec {k.vec}"
    if k.vec > 1 and k.vec > max_vec(kept, red, a, k.mode):
        return f"vec {k.vec} > {max_vec(kept, red, a, k.mode)}"
    if k.unroll not in (1, 2, 4, 8):
        return f"unroll {k.unroll}"
    if k.split < 1 or k.split > GRID_YZ_MAX:
        return f"split {k.split}"
    n_out = math.prod(x[0] for x in kept)
    if k.mode == "col":
        if k.rows and (k.rows & (k.rows - 1) or k.lanes * k.rows > k.block):
            return f"rows {k.rows}"
        if not kept:
            return "col mode needs a kept mode"
        ng = col_group(kept)
        m = math.prod(x[0] for x in kept[:ng])
        if k.lanes > 1 and (k.lanes // 2) * k.vec >= m:
            return "lanes x vec exceed the run"
        rt = k.rows if k.rows else k.block // k.lanes
        n_rows = math.prod(e for e, _ in red) if red else 1
        if rt > 1 and rt // 2 >= n_rows:
            return "more than half the thread rows idle"
        if k.split > max(1, _ceil(n_rows, rt)):
            return "split exceeds the reduction rows per thread row"
        if len(red) > MAX_RANK:
            return "more than 8 reduced modes"
        items = k.block // (k.lanes * rt)
        tiles = _ceil(m, k.lanes * k.vec)
        if (n_out // m) * tiles + items >= I32_LIMIT:
            return "work items >= 2^31"
    else:
        if k.rows:
            return "rows is a col knob"
        if not red:
            return "row mode needs a reduced mode"
        chunks = math.prod(e for e, _ in chunk_dims(red, k.vec))
        if k.split > max(1, _ceil(chunks, k.lanes)):
            return "split exceeds the chunks per lane"
        if k.lanes > 1 and (k.lanes // 2) >= chunks:
            return "lanes exceed the chunks"
        if len(chunk_dims(red, k.vec)) > MAX_RANK:
            return "more than 8 reduced slots"
    if n_out + k.block * k.vec >= I32_LIMIT:
        return "outputs >= 2^31"
    n_idx, step, per, n_it = loop_geometry(kept, red, k)
    last = (k.split - 1) * per + (step - 1) + (n_it * k.unroll - 1) * step
    if max(last, k.split * per) >= I32_LIMIT:
        return "reduction indices >= 2^31"
    gx = grid_x(kept, red, k)
    if gx * k.block > MAX_GRID_THREADS:
        return "launch grid exceeds the HIP limit"
    return None


def final_lanes(n_out: int, split: int, wave: int = 64) -> int:
    """Threads per output of the finalize: enough to keep about ``FINAL_TARGET`` threads busy
    when there are few outputs, at most a wave and the split count, rounded down to a
    :data:`FINAL_LANES` class."""
    lanes = 1
    while lanes * 2 <= min(wave, split) and n_out * lanes * 2 <= FINAL_TARGET:
        lanes *= 2
    return max(x for x in FINAL_LANES if x <= lanes)


#: hipTensor rounds every workspace size up to this many bytes.
WORKSPACE_ALIGN = 256


def split_workspace(kept, a: str, d: str, split: int) -> int:
    """The workspace bytes of the partial results of a launch with ``split`` (compute-type
    partials, one per output and split index; none without a split), rounded up to
    :data:`WORKSPACE_ALIGN` as hipTensor's planner does."""
    if split <= 1:
        return 0
    cb = 8 if "f64" in (a, d) else 4
    n = math.prod(x[0] for x in kept) * split * cb
    return -(-n // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


def cap_split(kept, a: str, d: str, split: int, limit: Optional[int]) -> int:
    """``split``, or the largest power of two below it whose partial results fit ``limit``
    bytes (1, no partial results, when none does; ``None``: no limit). hipTensor lowers a tuned
    route's split with this rule when the plan's workspace limit is smaller than its need.
    """
    while (
        limit is not None and split > 1 and split_workspace(kept, a, d, split) > limit
    ):
        s = 1
        while s * 2 < split:
            s *= 2
        split = s
    return split


def needs_idx64(kept, red, k: Knobs) -> bool:
    a_span = sum((e - 1) * s for e, s, _ in kept) + sum((e - 1) * s for e, s in red)
    d_span = sum((e - 1) * sd for e, _, sd in kept)
    n_out = math.prod(x[0] for x in kept)
    w_span = (n_out + k.block * k.vec) * k.split if k.split > 1 else 0
    return max(a_span, d_span, w_span) >= I32_LIMIT


def _slots(
    vals: Dict[str, int], prefix: str, dims, n: int, tensors: Sequence[str]
) -> None:
    """Fill a slot set: ``dims`` = (extent, stride per tensor...) innermost first, padded to ``n``
    with extent 1, magic (0, 0) and stride 0."""
    dims = list(dims) + [(1,) + (0,) * len(tensors)] * (n - len(dims))
    for i, dim in enumerate(dims):
        vals[f"{prefix}e{i}"] = dim[0]
        for t, name in enumerate(tensors):
            vals[f"{prefix}{name}{i}"] = dim[1 + t] if dim[0] != 1 else 0
        if i < n - 1:
            m, s = magic_u31(dim[0])
            vals[f"{prefix}m{i}"] = as_i32_bits(m)
            vals[f"{prefix}s{i}"] = s


@dataclass(frozen=True)
class Plan:
    spec: RreduceSpec
    args: Dict[str, int]
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    final: Optional[RreduceFinalSpec]
    final_args: Optional[Dict[str, int]]
    final_grid: Optional[Tuple[int, int, int]]
    workspace: int


def plan(
    kept,
    red,
    a: str,
    d: str,
    op: str,
    k: Knobs,
    *,
    una: bool = False,
    wave: int = 64,
    arch: str = "gfx950",
    nr: Optional[int] = None,
    idx64: Optional[bool] = None,
) -> Plan:
    """The launches of the problem with knobs ``k`` (ValueError when they do not apply).
    ``nr`` / ``idx64``: run it with a class of more reduced slots (padded) or with 64-bit
    offsets although they fit 32 bits (the untuned choice may pick such a class)."""
    why = check_knobs(kept, red, a, k, wave=wave, arch=arch)
    if why is not None:
        raise ValueError(why)
    kept = [tuple(x) for x in kept]
    red = [tuple(x) for x in red]
    need64 = needs_idx64(kept, red, k)
    if idx64 is None:
        idx64 = need64
    elif need64 and not idx64:
        raise ValueError("the problem needs 64-bit offsets")
    n_out = math.prod(x[0] for x in kept)
    n_idx, step, per, n_it = loop_geometry(kept, red, k)
    vals: Dict[str, int] = dict(
        n_out=n_out, n_idx=n_idx, per=per, n_it=n_it, split=k.split
    )
    if k.mode == "row":
        rdims = chunk_dims(red, k.vec)
        nr = _slots_for(len(rdims), nr)
        _slots(vals, "k", [(e, sa, sd) for e, sa, sd in kept], MAX_RANK, ("A", "D"))
        final_kept = kept
    else:
        rdims = list(red) if red else [(1, 0)]
        nr = _slots_for(len(rdims), nr)
        ng = col_group(kept)
        grp, rest = kept[:ng], kept[ng:]
        m = math.prod(x[0] for x in grp)
        tiles = _ceil(m, k.lanes * k.vec)
        rt = k.rows if k.rows else k.block // k.lanes
        items = k.block // (k.lanes * rt)
        tm, ts = magic_u31(tiles)
        vals.update(
            m=m,
            ka=grp[0][1],
            tiles=tiles,
            tm=as_i32_bits(tm),
            ts=ts,
            n_items=(n_out // m) * tiles,
        )
        _slots(vals, "g", [(e, sd) for e, _, sd in grp], MAX_RANK, ("D",))
        _slots(vals, "x", [(e, sa, sd) for e, sa, sd in rest], MAX_RANK, ("A", "D"))
        final_kept = list(grp) + list(rest)
        del items
    _slots(vals, "r", rdims, nr, ("A",))
    spec = RreduceSpec(
        mode=k.mode,
        a=a,
        d=d,
        op=op,
        nr=nr,
        block=k.block,
        vec=k.vec,
        lanes=k.lanes,
        unroll=k.unroll,
        rows=k.rows if k.mode == "col" else 0,
        nt=k.nt,
        una=una,
        idx64=idx64,
        wave=wave,
    )
    grid = (grid_x(kept, red, k), k.split, 1)
    final = final_args = final_grid = None
    ws = 0
    if k.split > 1:
        cb = 8 if spec.cdt == "f64" else 4
        ws = n_out * k.split * cb
        lanes = final_lanes(n_out, k.split, wave)
        final = RreduceFinalSpec(a=a, d=d, op=op, lanes=lanes, idx64=idx64, wave=wave)
        final_args = dict(n_out=n_out, split=k.split)
        _slots(final_args, "k", [(e, sd) for e, _, sd in final_kept], MAX_RANK, ("D",))
        final_grid = (_ceil(n_out, final.block // lanes), 1, 1)
    return Plan(spec, vals, grid, (k.block, 1, 1), final, final_args, final_grid, ws)


def _slots_for(n: int, nr: Optional[int]) -> int:
    need = slot_count(n)
    if need is None:
        raise ValueError("more than 8 reduced slots")
    if nr is None:
        return need
    if nr < need or nr not in SLOT_COUNTS:
        raise ValueError(f"a class of {nr} reduced slots cannot run {n} slots")
    return nr


def needed_slots(kept, red, k: Knobs) -> Optional[int]:
    """The reduced-slot count a class needs for the problem with knobs ``k``."""
    n = len(chunk_dims(red, k.vec)) if k.mode == "row" else max(1, len(red))
    return slot_count(n)


def log_q(x: int) -> int:
    """``x`` on a quarter-octave scale (integers only): ``4 * floor(log2 x)`` plus the next two
    bits of ``x``; 0 for ``x <= 1``."""
    if x <= 1:
        return 0
    e = x.bit_length() - 1
    frac = (x >> (e - 2)) & 3 if e >= 2 else (x << (2 - e)) & 3
    return 4 * e + frac


def features(kept, red) -> Tuple[int, int, int, int, int]:
    """The features the untuned choice compares problems by: outputs, reduced elements and the
    contiguous run of A's innermost mode (reduced mode in row mode, the kept run in col mode) on
    the quarter-octave scale, and the numbers of reduced and kept modes."""
    n_out = math.prod(x[0] for x in kept) if kept else 1
    n_red = math.prod(r[0] for r in red) if red else 1
    if mode_of(kept, red) == "row":
        inner = red[0][0] if red[0][1] == 1 else 1
    else:
        ng = col_group(kept)
        m = math.prod(x[0] for x in kept[:ng])
        inner = m if kept[0][1] == 1 else 1
    return (log_q(n_out), log_q(n_red), log_q(inner), len(red), len(kept))


def feature_distance(f, g) -> int:
    return (
        2 * abs(f[0] - g[0])
        + 2 * abs(f[1] - g[1])
        + abs(f[2] - g[2])
        + 2 * abs(f[3] - g[3])
        + 2 * abs(f[4] - g[4])
    )


@dataclass(frozen=True)
class Prototype:
    """A measured problem of the untuned choice: its group (mode, element bytes, MAX / MIN), its
    features, the palette knobs that ran it fastest and the wave count of its tuned launch on a
    device of ``cus`` compute units."""

    mode: str
    eb: int
    maxmin: bool
    f: Tuple[int, int, int, int, int]
    block: int
    vec: int
    lanes: int
    unroll: int
    rows: int
    nt: bool
    waves: int
    cus: int


def prototypes(table: Optional[dict] = None) -> List[Prototype]:
    table = load_table() if table is None else table
    out = []
    for p in table.get("prototypes", []):
        k = p["knobs"]
        out.append(
            Prototype(
                p["mode"],
                int(p["eb"]),
                bool(p["maxmin"]),
                tuple(int(x) for x in p["f"]),
                int(k["block"]),
                int(k["vec"]),
                int(k["lanes"]),
                int(k["unroll"]),
                int(k.get("rows", 0)),
                bool(k.get("nt", 0)),
                int(p["waves"]),
                int(p["cus"]),
            )
        )
    return out


#: The knobs (block, vec, lanes, unroll, rows) that run every problem of a mode (row: one thread
#: per output; col: one thread per kept element and one reduction row per thread), and the wave
#: count (of a 256-CU device) their split aims for.
UNIVERSAL_KNOBS = {"row": (256, 1, 1, 4, 0), "col": (256, 1, 1, 4, 1)}
UNIVERSAL_WAVES = 4096


def launch_waves(kept, red, k: Knobs) -> int:
    return grid_x(kept, red, k) * k.split * k.block // 64


def untuned_choices(
    kept,
    red,
    a: str,
    d: str,
    op: str,
    classes,
    protos,
    *,
    una: bool = False,
    cus: int = 256,
    arch: str = "gfx950",
    n: int = 3,
    workspace_limit: Optional[int] = None,
):
    """The untuned choice (hipTensor's rule): the prototypes of the problem's mode and element size
    nearest to it (same MAX / MIN class first, then by :func:`feature_distance`, then table
    order); each gives its knobs with the class of the pack that runs them (the fewest reduced
    slots at least the problem needs, 32-bit offsets when they fit) and the smallest power-of-two
    split that reaches the prototype's wave count (scaled to ``cus``) and whose partial results
    fit ``workspace_limit`` bytes (:func:`split_workspace`; ``None``: no limit, 0: no split).
    Up to ``n`` distinct ``(knobs, class)`` pairs, the first the rule's choice."""
    mode = mode_of(kept, red)
    if (mode == "row" and not red) or (mode == "col" and not kept):
        # one element in all (every mode of extent 1): no class has work for it
        return []
    eb = ELEM_BYTES[a]
    mm = op in ("max", "min")
    streaming = arch in STREAMING_ARCHS
    f = features(kept, red)
    order = [
        (p.maxmin != mm, feature_distance(f, p.f), i)
        for i, p in enumerate(protos)
        if p.mode == mode and p.eb == eb
    ]
    order.sort()
    pool = [
        s for s in classes if (s.mode, s.a, s.d, s.op, s.una) == (mode, a, d, op, una)
    ]

    def fits(split: int) -> bool:
        return (
            workspace_limit is None
            or split_workspace(kept, a, d, split) <= workspace_limit
        )

    out, seen = [], set()
    for _, _, i in order:
        p = protos[i]
        nt = p.nt and streaming
        k = Knobs(mode, p.block, p.vec, p.lanes, p.unroll, 1, p.rows, nt)
        if check_knobs(kept, red, a, k, arch=arch) is not None:
            continue
        need = needed_slots(kept, red, k)
        cands = [
            s
            for s in pool
            if (s.block, s.vec, s.lanes, s.unroll, s.rows, s.nt)
            == (k.block, k.vec, k.lanes, k.unroll, k.rows, k.nt)
            and need is not None
            and s.nr >= need
        ]
        if not cands:
            continue
        target = p.waves * cus // p.cus
        split = 1
        while launch_waves(kept, red, replace(k, split=split)) < target:
            nxt = replace(k, split=split * 2)
            if check_knobs(kept, red, a, nxt, arch=arch) is not None or not fits(
                nxt.split
            ):
                break
            split *= 2
        k = replace(k, split=split)
        need64 = needs_idx64(kept, red, k)
        cands = [s for s in cands if s.idx64 or not need64]
        if not cands:
            continue
        cands.sort(key=lambda s: (s.nr, s.idx64))
        s = cands[0]
        key = (s.kernel_name(), split)
        if key in seen:
            continue
        seen.add(key)
        out.append((k, s))
        if len(out) == n:
            break
    if not out:
        # no prototype's knobs run the problem: the universal knobs of the mode (one thread per
        # output or per kept element, scalar loads), which run every problem
        block, vec, lanes, unroll, rows = UNIVERSAL_KNOBS[mode]
        k = Knobs(mode, block, vec, lanes, unroll, 1, rows, streaming)
        need = needed_slots(kept, red, k)
        if need is not None and check_knobs(kept, red, a, k, arch=arch) is None:
            target = UNIVERSAL_WAVES * cus // 256
            while launch_waves(kept, red, k) < target:
                nxt = replace(k, split=k.split * 2)
                if check_knobs(kept, red, a, nxt, arch=arch) is not None or not fits(
                    nxt.split
                ):
                    break
                k = nxt
            need64 = needs_idx64(kept, red, k)
            cands = [
                s
                for s in pool
                if (s.block, s.vec, s.lanes, s.unroll, s.rows, s.nt)
                == (k.block, k.vec, k.lanes, k.unroll, k.rows, k.nt)
                and s.nr >= need
                and (s.idx64 or not need64)
            ]
            cands.sort(key=lambda s: (s.nr, s.idx64))
            if cands:
                out.append((k, cands[0]))
    return out


def with_split(k: Knobs, split: int) -> Knobs:
    return replace(k, split=split)


# ------------------------------------------------------------------------------- route table


def load_table(path: Optional[str] = None) -> dict:
    """The route table: ``{"abi", "version", "palette": [class...], "entries": [entry...]}``.

    A class is an :class:`RreduceSpec` as a dict plus ``"arch"``. An entry is a tuned problem:
    ``arch``, ``a``, ``d``, ``op``, canonical ``kept`` / ``red`` (innermost first), ``knobs``
    (:class:`Knobs` as a dict) and ``class`` (the class those knobs run it with)."""
    with open(path or table_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def spec_of(c: dict) -> RreduceSpec:
    s = RreduceSpec(
        mode=c["mode"],
        a=c["a"],
        d=c["d"],
        op=c["op"],
        nr=int(c["nr"]),
        block=int(c["block"]),
        vec=int(c["vec"]),
        lanes=int(c["lanes"]),
        unroll=int(c["unroll"]),
        rows=int(c.get("rows", 0)),
        nt=bool(c.get("nt", 0)),
        una=bool(c.get("una", 0)),
        idx64=bool(c.get("idx64", 0)),
    )
    why = s.check()
    if why is not None:
        raise ValueError(f"malformed rreduce class {c}: {why}")
    return s


def class_dict(s: RreduceSpec, arch: str) -> dict:
    d = dict(arch=arch)
    for k, v in asdict(s).items():
        if k == "wave":
            continue
        d[k] = int(v) if isinstance(v, bool) else v
    return d


def knobs_of(k: dict) -> Knobs:
    return Knobs(
        mode=k["mode"],
        block=int(k["block"]),
        vec=int(k["vec"]),
        lanes=int(k["lanes"]),
        unroll=int(k["unroll"]),
        split=int(k.get("split", 1)),
        rows=int(k.get("rows", 0)),
        nt=bool(k.get("nt", 0)),
    )


def pack_classes(arch: str, table: Optional[dict] = None) -> List[RreduceSpec]:
    """The classes of the target's pack: its palette, then the classes of its tuned problems
    (first occurrence order, no duplicates)."""
    table = load_table() if table is None else table
    out, seen = [], set()
    items = [c for c in table["palette"] if c["arch"] == arch]
    items += [e["class"] for e in table["entries"] if e["arch"] == arch]
    for c in items:
        s = spec_of(c)
        if s.kernel_name() not in seen:
            seen.add(s.kernel_name())
            out.append(s)
    return out


def final_classes(arch: str) -> List[RreduceFinalSpec]:
    """The finalize classes of every pack: each data-type row, operator, lane count and offset
    width."""
    del arch
    return [
        RreduceFinalSpec(a=a, d=a, op=op, lanes=lanes, idx64=idx64)
        for idx64 in (False, True)
        for a in ROWS
        for op in ("add", "mul", "max", "min")
        for lanes in FINAL_LANES
    ]


def problem_key(a: str, d: str, op: str, kept, red) -> str:
    """The canonical problem of a tuned route record (``reduce.tuned.<fnv1a64 hex>.key``)."""
    k = ",".join(f"{e}:{sa}:{sd}" for e, sa, sd in kept)
    r = ",".join(f"{e}:{sa}" for e, sa in red)
    return f"a={a};d={d};op={op};k={k};r={r}"


def route_record(e: dict) -> str:
    """The launch knobs of a tuned problem as the pack's route value."""
    k, c = e["knobs"], e["class"]
    return (
        f"v1;mode={k['mode']};block={k['block']};vec={k['vec']};lanes={k['lanes']};"
        f"unroll={k['unroll']};split={k.get('split', 1)};rows={k.get('rows', 0)};"
        f"nt={int(bool(k.get('nt', 0)))};nr={c['nr']};idx64={int(bool(c.get('idx64', 0)))}"
    )


def proto_record(p: dict) -> str:
    """A prototype of the untuned choice as the pack's route value."""
    k = p["knobs"]
    f = ",".join(str(int(x)) for x in p["f"])
    return (
        f"v1;mode={p['mode']};eb={int(p['eb'])};mm={int(bool(p['maxmin']))};f={f};"
        f"block={k['block']};vec={k['vec']};lanes={k['lanes']};unroll={k['unroll']};"
        f"rows={k.get('rows', 0)};nt={int(bool(k.get('nt', 0)))};waves={int(p['waves'])};"
        f"cus={int(p['cus'])}"
    )


def routes(arch: str, table: Optional[dict] = None) -> Dict[str, str]:
    """The pack routes of the target's tuned problems (``reduce.tuned.<hash>`` and its ``.key``),
    then the prototypes of the untuned choice (``reduce.proto.<i>``, ``reduce.proto.count``; the
    same for every target, measured on the target the table names per prototype)."""
    table = load_table() if table is None else table
    out: Dict[str, str] = {}
    for e in table["entries"]:
        if e["arch"] != arch:
            continue
        key = problem_key(e["a"], e["d"], e["op"], e["kept"], e["red"])
        name = f"reduce.tuned.{fnv1a64(key):016x}"
        out[name] = route_record(e)
        out[name + ".key"] = key
    protos = table.get("prototypes", [])
    for i, p in enumerate(protos):
        out[f"reduce.proto.{i}"] = proto_record(p)
    out["reduce.proto.count"] = str(len(protos))
    return out


def entry_class(arch: str, a: str, d: str, op: str, kept, red, k: Knobs) -> RreduceSpec:
    """The class a tuned problem runs with (its plan's spec)."""
    return plan(kept, red, a, d, op, k, arch=arch).spec
