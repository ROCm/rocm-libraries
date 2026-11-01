# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned strided reduction classes (hipTensor reduction), the geometry as kernel arguments.

``D = alpha * reduce_op(opA(A)) + beta * opC(C)`` with ``op`` in ADD / MUL / MAX / MIN, every
tensor read through its own strides, accumulation in f32 (f64 for f64 data), one rounding to D's
dtype. MAX and MIN propagate NaN. With ``alpha == 0`` A is not read; with ``beta == 0`` C is not
read.

These are the bodies of rocKE's native strided reduction (vector loads, several threads per
output, unrolled independent accumulators, multi-CTA splits with compute-type partials, streaming
loads) with the problem's canonical geometry as kernel arguments instead of constants, so one
compiled class serves every problem whose canonical form has its reduced-slot count. The class
(:class:`RreduceSpec`) fixes the reduction operator, the data types, the mode and the knobs:

* ``row`` (A's innermost mode is reduced): ``lanes`` threads per output, ``block // lanes``
  outputs per block (grid x). The reduction space is cut into chunks of ``vec`` contiguous
  elements of the innermost reduced mode; chunk ``c`` is split into the ``nr`` reduced slots
  (innermost first) with host-computed magic numbers. Split ``s = blockIdx.y`` owns the chunk
  range ``[s * per, (s + 1) * per)``; lane ``l`` loads chunks ``s * per + l + lanes * j``,
  ``unroll`` of them per loop trip into independent accumulators. The lanes of an output combine
  with XOR shuffles (and an LDS step when ``lanes`` exceeds the wave).
* ``col`` (A's innermost mode is kept): the kept modes that are contiguous in A from the
  innermost one on form the run (extent ``m``), tiled by ``lanes * vec``; a thread owns ``vec``
  consecutive outputs of a tile, the block's ``rows`` thread rows split the reduction rows,
  ``block // (lanes * rows)`` tiles per block. Rows combine with XOR shuffles and an LDS step
  across waves.

MAX / MIN accumulate with plain ``fmax`` / ``fmin`` and track NaN separately (one unordered
compare per pair of loaded values, a per-accumulator flag that turns the result into NaN after
the loop); the cross-lane, LDS and finalize steps use a NaN-propagating combine.

Split: the host passes ``split`` (= grid y). ``split == 1``: the kernel applies alpha / beta / C
and writes D. ``split > 1``: it writes its unrounded partial to ``W[s * n_out + o]`` (``o`` = flat
output index, kept modes innermost first; col: the run first, then the other kept modes) and
:func:`build_rreduce_final` combines the partials and writes D.

Index arithmetic: coordinates are i32, offsets i32 or i64 (``idx64``). Every slot a host fills
has an extent of at least 2 (canonical modes); unused slots are the outermost ones, with extent 1
and magic ``(0, 0)``: a flat index in range is 0 there, which the magic formula maps to the
coordinate 0. Row mode: the innermost reduced slot is counted in ``vec`` chunks; an inner mode of
exactly ``vec`` elements gives no slot.

Kernel ABI ``hiptensor-aot-rreduce/v1``. Row: ``A, C, D, W, alpha, beta, [opA], opC, n_out,
ke0..ke7, km0..km6, ks0..ks6, kA0..kA7, kD0..kD7`` (kept slots; C has D's strides),
``re0.., rm0.., rs0.., rA0..`` (``nr`` reduced slots, magic pairs for the first ``nr - 1``),
``n_idx, per, n_it, split``. Col: ``A, C, D, W, alpha, beta, [opA], opC, n_out, m, ka, tiles,
tm, ts, n_items``, the run's slots ``ge0..ge7, gm0..gm6, gs0..gs6, gD0..gD7``, the other kept
modes ``xe0..xe7, xm0..xm6, xs0..xs6, xA0..xA7, xD0..xD7``, the reduced slots, ``n_idx, per,
n_it, split``. Final: ``W, C, D, alpha, beta, opC, n_out, split, ke0..ke7, km0..km6, ks0..ks6,
kD0..kD7``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType, TemporalHint, Value
from rocke.helpers.magic_division import do_magic_division_u31_dynamic

from .common import DTYPES, MAX_RANK, OP_ADD, OP_MAX, OP_MIN, OP_MUL, Emit, ptr

RREDUCE_ABI = "hiptensor-aot-rreduce/v1"

#: The reduction operators of the classes and their hipTensor operator codes.
OPS = {"add": OP_ADD, "mul": OP_MUL, "max": OP_MAX, "min": OP_MIN}
#: Vector widths of A's loads (16 bytes at most).
LOAD_VECS = {"f16": (1, 2, 4, 8), "bf16": (1, 2, 4, 8), "f32": (1, 2, 4), "f64": (1, 2)}
#: Targets whose lowering accepts streaming (nontemporal) loads.
STREAMING_ARCHS = ("gfx942", "gfx950")
#: Reduced-slot counts of the classes (canonical problems with more reduced modes use 8).
SLOT_COUNTS = (1, 2, 3, 4, 8)


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def _log2(x: int) -> int:
    return x.bit_length() - 1


@dataclass(frozen=True)
class RreduceSpec:
    """One compiled reduction class (see the module docstring)."""

    mode: str  # "row" | "col"
    a: str  # A dtype
    d: str  # C / D dtype
    op: str  # "add" | "mul" | "max" | "min"
    nr: int = 1  # reduced slots
    block: int = 256
    vec: int = 1  # elements per A load
    lanes: int = 64  # row: threads per output; col: threads along the run
    unroll: int = 1  # independent loads (and accumulators) per thread per loop trip
    rows: int = 0  # col: reduction-row threads per tile (0: block // lanes)
    nt: bool = False  # streaming A loads (gfx942 / gfx950)
    una: bool = (
        False  # opA other than IDENTITY (the basic operators, a uniform branch per trip)
    )
    idx64: bool = False
    wave: int = 64

    @property
    def cdt(self) -> str:
        return "f64" if "f64" in (self.a, self.d) else "f32"

    @property
    def col_rows(self) -> int:
        return self.rows if self.rows else self.block // self.lanes

    @property
    def col_items(self) -> int:
        return self.block // (self.lanes * self.col_rows)

    def check(self) -> Optional[str]:
        """Why the class cannot be built (None when it can)."""
        for n in ("nt", "una", "idx64"):
            if type(getattr(self, n)) is not bool:
                return f"{n} must be a bool, got {getattr(self, n)!r}"
        if self.mode not in ("row", "col"):
            return f"mode {self.mode!r}"
        if self.a not in DTYPES or self.d not in DTYPES:
            return "unknown dtype"
        if self.op not in OPS:
            return f"op {self.op!r}"
        if self.nr not in SLOT_COUNTS:
            return f"nr {self.nr}"
        if self.wave not in (32, 64):
            return f"wave {self.wave}"
        if not _pow2(self.block) or not self.wave <= self.block <= 1024:
            return f"block {self.block}"
        if not _pow2(self.lanes) or self.lanes > self.block:
            return f"lanes {self.lanes}"
        if self.vec not in LOAD_VECS[self.a]:
            return f"vec {self.vec}"
        if self.unroll not in (1, 2, 4, 8):
            return f"unroll {self.unroll}"
        if self.rows and (
            self.mode != "col"
            or not _pow2(self.rows)
            or self.lanes * self.rows > self.block
        ):
            return f"rows {self.rows}"
        return None

    def kernel_name(self) -> str:
        rows = f"_w{self.rows}" if self.rows else ""
        return (
            f"hiptensor_aot_rreduce_{self.mode}_{self.op}_{self.a}_{self.d}_r{self.nr}"
            f"_b{self.block}_v{self.vec}_l{self.lanes}{rows}_u{self.unroll}"
            + ("_nt" if self.nt else "")
            + ("_una" if self.una else "")
            + ("_i64" if self.idx64 else "_i32")
            + (f"_wave{self.wave}" if self.wave != 64 else "")
        )


@dataclass(frozen=True)
class RreduceFinalSpec:
    """The split finalize of a class: ``lanes`` threads per output combine its partials."""

    a: str
    d: str
    op: str
    lanes: int = 1
    idx64: bool = False
    block: int = 256
    wave: int = 64

    @property
    def cdt(self) -> str:
        return "f64" if "f64" in (self.a, self.d) else "f32"

    def check(self) -> Optional[str]:
        if type(self.idx64) is not bool:
            return f"idx64 must be a bool, got {self.idx64!r}"
        if self.a not in DTYPES or self.d not in DTYPES:
            return "unknown dtype"
        if self.op not in OPS:
            return f"op {self.op!r}"
        if self.wave not in (32, 64):
            return f"wave {self.wave}"
        if not _pow2(self.block) or not self.wave <= self.block <= 1024:
            return f"block {self.block}"
        if not _pow2(self.lanes) or self.lanes > self.wave or self.block % self.lanes:
            return f"lanes {self.lanes}"
        return None

    def kernel_name(self) -> str:
        return (
            f"hiptensor_aot_rreduce_final_{self.op}_{self.a}_{self.d}_l{self.lanes}_b{self.block}"
            + ("_i64" if self.idx64 else "_i32")
            + (f"_wave{self.wave}" if self.wave != 64 else "")
        )


# ------------------------------------------------------------------------------- helpers


def _identity(e: Emit, op: str) -> Value:
    if op == "add":
        return e.cf(0.0)
    if op == "mul":
        return e.cf(1.0)
    return e.cf(float("-inf") if op == "max" else float("inf"))


def _combine(e: Emit, op: str, a: Value, c: Value) -> Value:
    """``op`` of two compute values; MAX / MIN propagate NaN (a NaN operand gives ``a + c``)."""
    b = e.b
    if op == "add":
        return b.fadd(a, c)
    if op == "mul":
        return b.fmul(a, c)
    m = b.fmax(a, c) if op == "max" else b.fmin(a, c)
    uno = b.fcmp("uno", a, c)
    nan = b.fadd(a, c)
    return b.select(uno, nan, m)


def _loop_combine(e: Emit, op: str, a: Value, c: Value) -> Value:
    """``op`` inside the reduction loop: plain ``fmax`` / ``fmin`` for MAX / MIN (the loop tracks
    NaN with flags)."""
    b = e.b
    if op == "add":
        return b.fadd(a, c)
    if op == "mul":
        return b.fmul(a, c)
    return b.fmax(a, c) if op == "max" else b.fmin(a, c)


def _tree(e: Emit, op: str, vals: Sequence[Value], loop: bool) -> Value:
    vals = list(vals)
    while len(vals) > 1:
        nxt = []
        for i in range(0, len(vals) - 1, 2):
            if loop:
                nxt.append(_loop_combine(e, op, vals[i], vals[i + 1]))
            else:
                nxt.append(_combine(e, op, vals[i], vals[i + 1]))
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def _nan_any(e: Emit, vals: Sequence[Value]) -> Value:
    """i1: any of ``vals`` is NaN (one unordered compare per pair)."""
    b = e.b
    vals = list(vals)
    flags = []
    for i in range(0, len(vals) - 1, 2):
        flags.append(b.fcmp("uno", vals[i], vals[i + 1]))
    if len(vals) % 2:
        flags.append(b.fcmp("uno", vals[-1], vals[-1]))
    f = flags[0]
    for g in flags[1:]:
        f = b.lor(f, g)
    return f


def _divmod(e: Emit, x: Value, d: Value, m: Value, s: Value) -> Tuple[Value, Value]:
    """``(x // d, x % d)`` with the magic pair of ``d`` (exact for ``d >= 2``; for a padding slot,
    ``d == 1`` and ``m == 0``, exact for ``x == 0``)."""
    b = e.b
    q = do_magic_division_u31_dynamic(b, x, m, s, one=False)
    qd = b.mul(q, d)
    r = b.sub(x, qd)
    return q, r


class _Slots:
    """Kernel arguments of a slot set: extents, magic pairs, and strides per tensor."""

    def __init__(self, e: Emit, prefix: str, n: int, tensors: Sequence[str]):
        b = e.b
        self.n = n
        self.ext = [b.param(f"{prefix}e{i}", I32) for i in range(n)]
        self.mag = []
        for i in range(n - 1):
            m = b.param(f"{prefix}m{i}", I32)
            s = b.param(f"{prefix}s{i}", I32)
            self.mag.append((m, s))
        self.strides = {}
        for t in tensors:
            self.strides[t] = [e.stride_param(f"{prefix}{t}{i}") for i in range(n)]

    def coords(self, e: Emit, x: Value) -> List[Value]:
        """Coordinates of flat index ``x`` (innermost slot first)."""
        out = []
        r = x
        for i in range(self.n - 1):
            q, c = _divmod(e, r, self.ext[i], self.mag[i][0], self.mag[i][1])
            out.append(c)
            r = q
        out.append(r)
        return out

    def offset(self, e: Emit, coords: Sequence[Value], t: str) -> Value:
        b = e.b
        o = None
        for c, st in zip(coords, self.strides[t]):
            v = e.off(c, st)
            o = v if o is None else b.add(o, v)
        return o


def _load_vec(
    e: Emit, ptr_: Value, off: Value, dt: str, vec: int, nt: bool
) -> List[Value]:
    """``vec`` elements of dtype ``dt`` at element offset ``off``, in the compute type."""
    b = e.b
    ty, nb = DTYPES[dt]
    if e.idx64:
        byte = b.mul(off, b.const_i64(nb))
        p = b.global_ptr_add(ptr_, byte)
        idx = e.c32(0)
    else:
        p, idx = ptr_, off
    if vec == 1:
        raw = [b.global_load(p, idx, ty, align=nb)]
    else:
        hint = TemporalHint.STREAMING if nt else TemporalHint.DEFAULT
        v = b.global_load_vN(p, idx, ty, vec, align=nb * vec, temporal_hint=hint)
        raw = [b.vec_extract(v, i) for i in range(vec)]
    return [e.to_c(x, dt) for x in raw]


def _store_w(e: Emit, W: Value, off: Value, vals: Sequence[Value]) -> None:
    """Compute-type partials at element offset ``off`` of W (one vector store)."""
    b = e.b
    nb = 8 if e.cdt == "f64" else 4
    if e.idx64:
        byte = b.mul(off, b.const_i64(nb))
        p = b.global_ptr_add(W, byte)
        idx = e.c32(0)
    else:
        p, idx = W, off
    if len(vals) == 1:
        b.global_store(p, idx, vals[0], align=nb)
        return
    v = b.vec_pack(list(vals), e.cty)
    b.global_store_vN(p, idx, v, len(vals), align=nb * len(vals))


def _load_w(e: Emit, W: Value, off: Value) -> Value:
    b = e.b
    nb = 8 if e.cdt == "f64" else 4
    if e.idx64:
        byte = b.mul(off, b.const_i64(nb))
        p = b.global_ptr_add(W, byte)
        idx = e.c32(0)
    else:
        p, idx = W, off
    return b.global_load(p, idx, e.cty, align=nb)


def _epilogue(e: Emit, P, acc: Value, coff: Value, d: str) -> Value:
    """``alpha * acc + beta * opC(C)`` (C read only when beta != 0; a zero alpha drops acc)."""
    b = e.b
    zero = e.cf(0.0)
    anz = b.fcmp("one", P["alpha"], zero)
    aa = b.fmul(P["alpha"], acc)
    v = b.select(anz, aa, zero)
    bnz = b.fcmp("one", P["beta"], zero)
    cv = e.when(bnz, [zero], lambda _c: [e.unary(e.load(P["C"], coff, d)[0], P["opC"])])
    bc = b.fmul(P["beta"], cv[0])
    s = b.fadd(v, bc)
    return b.select(bnz, s, v)


def _common_params(e: Emit, s: RreduceSpec, with_a: bool):
    b = e.b
    P = {}
    if with_a:
        P["A"] = ptr(b, "A", s.a, readonly=True)
    P["C"] = ptr(b, "C", s.d, readonly=True)
    P["D"] = ptr(b, "D", s.d, readonly=False)
    P["W"] = b.param("W", PtrType(e.cty, "global"), align=8)
    P["alpha"] = e.scalar_param("alpha")
    P["beta"] = e.scalar_param("beta")
    if with_a and s.una:
        P["opA"] = b.param("opA", I32)
    P["opC"] = b.param("opC", I32)
    P["n_out"] = b.param("n_out", I32)
    return P


def _loop_params(b: IRBuilder):
    n_idx = b.param("n_idx", I32)
    per = b.param("per", I32)
    n_it = b.param("n_it", I32)
    split = b.param("split", I32)
    return n_idx, per, n_it, split


# ------------------------------------------------------------------------------- the loop


def _red_loop(
    e: Emit,
    s: RreduceSpec,
    loop_args,
    lane: Value,
    step: int,
    sy: Value,
    load_at: Callable[[Value], List[List[Value]]],
    n_vals: int,
    reduce_vals: Callable[[List[Value]], List[Value]],
    opa: Optional[Value],
) -> List[Value]:
    """The per-thread reduction loop. The thread visits flat indices ``sy * per + lane + step * j``
    below ``min((sy + 1) * per, n_idx)``; ``load_at(idx)`` returns, per unrolled load, the loaded
    compute values; ``reduce_vals`` turns one load's values into ``n_vals`` accumulator inputs
    (row: their combine; col: the values). Returns the ``n_vals`` combined accumulators of the
    thread (NaN when a MAX / MIN input was NaN)."""
    b = e.b
    n_idx, per, n_it, _split = loop_args
    U = s.unroll
    op = s.op
    nan_mode = op in ("max", "min")
    sp = b.mul(sy, per)
    start = b.add(sp, lane)
    end = b.add(sp, per)
    hi = b.smin(end, n_idx)
    ident = _identity(e, op)
    init = [ident] * (U * n_vals)
    if nan_mode:
        init = init + [e.cf(0.0)] * n_vals
    k = e.next_uid()
    loop = b.scf_for_iter(
        e.c32(0),
        n_it,
        e.c32(1),
        [(f"racc{k}_{i}", v) for i, v in enumerate(init)],
        iv_name=f"rit{k}",
    )
    with loop as (it, cur):
        stride = e.c32(step * U)
        its = b.mul(it, stride)
        base = b.add(start, its)
        loads, valids = [], []
        for u in range(U):
            if u:
                cu = e.c32(u * step)
                idx = b.add(base, cu)
            else:
                idx = base
            valid = b.cmp_lt(idx, hi)
            safe = b.select(valid, idx, e.c32(0))
            loads.append(load_at(safe))
            valids.append(valid)
        if opa is not None:
            flat = [v for vals in loads for v in vals]
            flat = e.unary_list(flat, opa)
            n = len(loads[0])
            loads = [flat[u * n : (u + 1) * n] for u in range(U)]
        nxt = []
        for u in range(U):
            ins = reduce_vals(loads[u])
            for i in range(n_vals):
                x = b.select(valids[u], ins[i], ident)
                nxt.append(_loop_combine(e, op, cur[u * n_vals + i], x))
        if nan_mode:
            for i in range(n_vals):
                f = None
                for u in range(U):
                    vals = loads[u] if n_vals == 1 else [loads[u][i]]
                    g = _nan_any(e, vals)
                    g = b.land(g, valids[u])
                    f = g if f is None else b.lor(f, g)
                one = e.cf(1.0)
                nxt.append(b.select(f, one, cur[U * n_vals + i]))
        b.scf_yield(*nxt)
    res = list(loop.results)
    accs = []
    for i in range(n_vals):
        acc = _tree(e, op, [res[u * n_vals + i] for u in range(U)], True)
        if nan_mode:
            flag = b.fcmp("one", res[U * n_vals + i], e.cf(0.0))
            nan = b.fmul(e.cf(float("inf")), e.cf(0.0))
            acc = b.select(flag, nan, acc)
        accs.append(acc)
    return accs


# ------------------------------------------------------------------------------- row


def _row(e: Emit, s: RreduceSpec, P) -> None:
    b = e.b
    kept = _Slots(e, "k", MAX_RANK, ("A", "D"))
    red = _Slots(e, "r", s.nr, ("A",))
    loop_args = _loop_params(b)
    split = loop_args[3]
    L = s.lanes
    PB = s.block // L
    tid = b.thread_id_x()
    bx = b.block_id_x()
    sy = b.block_id_y()
    if L > 1:
        lane = b.land(tid, e.c32(L - 1))
        g = b.lshr(tid, e.c32(_log2(L)))
    else:
        lane = e.c32(0)
        g = tid
    if PB > 1:
        bp = b.mul(bx, e.c32(PB))
        o = b.add(bp, g)
    else:
        o = bx
    o_ok = b.cmp_lt(o, P["n_out"])
    last = b.sub(P["n_out"], e.c32(1))
    o_ld = b.select(o_ok, o, last)
    kc = kept.coords(e, o_ld)
    a_base = kept.offset(e, kc, "A")
    d_off = kept.offset(e, kc, "D")

    def load_at(idx: Value) -> List[Value]:
        rc = red.coords(e, idx)
        ro = red.offset(e, rc, "A")
        off = b.add(a_base, ro)
        return _load_vec(e, P["A"], off, s.a, s.vec, s.nt)

    def reduce_vals(vals: List[Value]) -> List[Value]:
        return [_tree(e, s.op, vals, True)]

    anz = b.fcmp("one", P["alpha"], e.cf(0.0))
    ident = _identity(e, s.op)
    acc = e.when(
        anz,
        [ident],
        lambda _c: _red_loop(
            e, s, loop_args, lane, L, sy, load_at, 1, reduce_vals, P.get("opA")
        ),
    )[0]
    # lanes of one output: XOR butterfly inside the wave, then LDS across waves
    in_wave = min(L, s.wave)
    for i in range(_log2(in_wave)):
        sh = b.warp_shuffle_xor(acc, 1 << i)
        acc = _combine(e, s.op, acc, sh)
    if L > s.wave:
        nw = L // s.wave
        lds = b.smem_alloc(e.cty, (s.block // s.wave,), name_hint="rred_lds")
        wid = b.lshr(tid, e.c32(_log2(s.wave)))
        lane_w = b.land(tid, e.c32(s.wave - 1))
        with b.scf_if(b.cmp_eq(lane_w, e.c32(0))):
            b.smem_store_vN(lds, [wid], acc, 1)
        b.sync()
        first = b.land(wid, e.c32(~(nw - 1) & 0x7FFFFFFF))
        parts = []
        for i in range(nw):
            slot = b.add(first, e.c32(i))
            v = b.smem_load_vN(lds, slot, dtype=e.cty, n=1)
            parts.append(b.vec_extract(v, 0))
        acc = _tree(e, s.op, parts, False)
    cond = o_ok
    if L > 1:
        lead = b.cmp_eq(lane, e.c32(0))
        cond = b.land(lead, o_ok)
    with b.scf_if(cond):
        one = b.cmp_eq(split, e.c32(1))
        with b.scf_if(one):
            v = _epilogue(e, P, acc, d_off, s.d)
            e.store(P["D"], d_off, [v], s.d)
        with b.scf_if(b.lnot(one)):
            so = e.off(sy, e.widen(P["n_out"]))
            ow = e.widen(o)
            woff = b.add(so, ow)
            _store_w(e, P["W"], woff, [acc])


# ------------------------------------------------------------------------------- col


def _col(e: Emit, s: RreduceSpec, P) -> None:
    b = e.b
    m = b.param("m", I32)
    ka = e.stride_param("ka")
    tiles = b.param("tiles", I32)
    tm = b.param("tm", I32)
    ts = b.param("ts", I32)
    n_items = b.param("n_items", I32)
    grp = _Slots(e, "g", MAX_RANK, ("D",))
    rest = _Slots(e, "x", MAX_RANK, ("A", "D"))
    red = _Slots(e, "r", s.nr, ("A",))
    loop_args = _loop_params(b)
    split = loop_args[3]
    KT = s.lanes
    RT = s.col_rows
    OB = s.col_items
    vec = s.vec
    tile = KT * vec
    tid = b.thread_id_x()
    bx = b.block_id_x()
    sy = b.block_id_y()
    if KT > 1:
        tk = b.land(tid, e.c32(KT - 1))
        tr_all = b.lshr(tid, e.c32(_log2(KT)))
    else:
        tk = e.c32(0)
        tr_all = tid
    item_ok = None
    if OB > 1:
        if RT > 1:
            tr = b.land(tr_all, e.c32(RT - 1))
            to = b.lshr(tr_all, e.c32(_log2(RT)))
        else:
            tr = e.c32(0)
            to = tr_all
        bo = b.mul(bx, e.c32(OB))
        item = b.add(bo, to)
        item_ok = b.cmp_lt(item, n_items)
        last = b.sub(n_items, e.c32(1))
        item = b.select(item_ok, item, last)
    else:
        tr = tr_all
        item = bx
    orest, kt = e.divmod(item, tiles, tm, ts)
    kt0 = b.mul(kt, e.c32(tile))
    if vec > 1:
        tv = b.mul(tk, e.c32(vec))
        k0 = b.add(kt0, tv)
    else:
        k0 = b.add(kt0, tk)
    k_ok = b.cmp_lt(k0, m)
    k_ld = b.select(k_ok, k0, e.c32(0))
    xc = rest.coords(e, orest)
    a_rest = rest.offset(e, xc, "A")
    d_rest = rest.offset(e, xc, "D")
    ak = e.off(k_ld, ka)
    a_base = b.add(a_rest, ak)
    gc = grp.coords(e, k_ld)
    d_grp = grp.offset(e, gc, "D")

    def load_at(idx: Value) -> List[Value]:
        rc = red.coords(e, idx)
        ro = red.offset(e, rc, "A")
        off = b.add(a_base, ro)
        return _load_vec(e, P["A"], off, s.a, vec, s.nt)

    def reduce_vals(vals: List[Value]) -> List[Value]:
        return list(vals)

    anz = b.fcmp("one", P["alpha"], e.cf(0.0))
    ident = _identity(e, s.op)
    acc = e.when(
        anz,
        [ident] * vec,
        lambda _c: _red_loop(
            e, s, loop_args, tr, RT, sy, load_at, vec, reduce_vals, P.get("opA")
        ),
    )
    # rows inside a wave: lane bits log2(KT) .. log2(min(KT * RT, wave)) - 1
    span = min(KT * RT, s.wave)
    if KT < span:
        for i in range(_log2(KT), _log2(span)):
            nxt = []
            for v in acc:
                sh = b.warp_shuffle_xor(v, 1 << i)
                nxt.append(_combine(e, s.op, v, sh))
            acc = nxt
    # rows across waves (LDS): one leader row per wave
    rows_per_wave = max(1, s.wave // KT)
    n_lead = max(1, RT // rows_per_wave)
    if n_lead > 1:
        lds = b.smem_alloc(e.cty, (OB * n_lead * KT * vec,), name_hint="rred_col_lds")
        if rows_per_wave > 1:
            lead_row = b.lshr(tr, e.c32(_log2(rows_per_wave)))
            is_lead = b.cmp_eq(b.land(tr, e.c32(rows_per_wave - 1)), e.c32(0))
        else:
            lead_row = tr
            is_lead = None
        if OB > 1:
            tn = b.mul(to, e.c32(n_lead))
            row_slot = b.add(tn, lead_row)
        else:
            row_slot = lead_row
        rk = b.mul(row_slot, e.c32(KT))
        rkt = b.add(rk, tk)
        slot = b.mul(rkt, e.c32(vec))

        def wr():
            for i in range(vec):
                si = b.add(slot, e.c32(i))
                b.smem_store_vN(lds, [si], acc[i], 1)

        if is_lead is None:
            wr()
        else:
            with b.scf_if(is_lead):
                wr()
        b.sync()
        base = b.mul(tk, e.c32(vec))
        if OB > 1:
            tb = b.mul(to, e.c32(n_lead * KT * vec))
            base = b.add(base, tb)
        new = []
        for i in range(vec):
            parts = []
            for r in range(n_lead):
                si = b.add(base, e.c32(r * KT * vec + i))
                v = b.smem_load_vN(lds, si, dtype=e.cty, n=1)
                parts.append(b.vec_extract(v, 0))
            new.append(_tree(e, s.op, parts, False))
        acc = new
    cond = k_ok
    if RT > 1:
        lead = b.cmp_eq(tr, e.c32(0))
        cond = b.land(lead, cond)
    if item_ok is not None:
        cond = b.land(cond, item_ok)
    d_off = b.add(d_rest, d_grp)
    with b.scf_if(cond):
        one = b.cmp_eq(split, e.c32(1))
        with b.scf_if(one):
            kd = grp.strides["D"][0]
            outs, doffs = [], []
            for i in range(vec):
                if i:
                    di = e.off(e.c32(i), kd)
                    doff = b.add(d_off, di)
                else:
                    doff = d_off
                outs.append(_epilogue(e, P, acc[i], doff, s.d))
                doffs.append(doff)
            if vec == 1:
                e.store(P["D"], d_off, [outs[0]], s.d)
            else:
                unit = b.cmp_eq(kd, e.one_stride())
                with b.scf_if(unit):
                    _store_d_vec(e, P["D"], d_off, outs, s.d)
                with b.scf_if(b.lnot(unit)):
                    for i in range(vec):
                        e.store(P["D"], doffs[i], [outs[i]], s.d)
        with b.scf_if(b.lnot(one)):
            om = b.mul(orest, m)
            o_flat = b.add(om, k0)
            so = e.off(sy, e.widen(P["n_out"]))
            ow = e.widen(o_flat)
            woff = b.add(so, ow)
            _store_w(e, P["W"], woff, acc)


def _store_d_vec(e: Emit, D: Value, off: Value, vals: Sequence[Value], dt: str) -> None:
    """``len(vals)`` consecutive D elements (one vector store, each value rounded once)."""
    b = e.b
    ty, nb = DTYPES[dt]
    outs = [e.from_c(v, dt) for v in vals]
    if e.idx64:
        byte = b.mul(off, b.const_i64(nb))
        p = b.global_ptr_add(D, byte)
        idx = e.c32(0)
    else:
        p, idx = D, off
    v = b.vec_pack(outs, ty)
    b.global_store_vN(p, idx, v, len(outs), align=nb * len(outs))


# ------------------------------------------------------------------------------- builders


class _RedEmit(Emit):
    """:class:`Emit` with the few offset helpers of these kernels."""

    def widen(self, v: Value) -> Value:
        """An i32 count as an offset-type value."""
        return self.b.sext(v, self.otype) if self.idx64 else v

    def one_stride(self) -> Value:
        return self.b.const_i64(1) if self.idx64 else self.c32(1)

    @property
    def otype(self):
        from rocke.core.ir import I64

        return I64 if self.idx64 else I32

    def next_uid(self) -> int:
        return next(self._uid)


def build_rreduce(spec: RreduceSpec, *, arch: str = "gfx950") -> KernelDef:
    """The class kernel (the IR does not depend on the target; ``arch`` names it for the
    builder contract)."""
    del arch
    s = spec
    why = s.check()
    if why is not None:
        raise ValueError(why)
    b = IRBuilder(s.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = s.block
    e = _RedEmit(b, s.idx64, s.cdt, False)
    P = _common_params(e, s, True)
    if s.mode == "row":
        _row(e, s, P)
    else:
        _col(e, s, P)
    return b.kernel


def build_rreduce_final(spec: RreduceFinalSpec, *, arch: str = "gfx950") -> KernelDef:
    """``D[o] = alpha * combine_s W[s * n_out + o] (+ beta * opC(C[o]))``: ``lanes`` threads per
    output (lane ``l`` combines partials ``l, l + lanes, ...``, then an XOR combine)."""
    del arch
    f = spec
    why = f.check()
    if why is not None:
        raise ValueError(why)
    b = IRBuilder(f.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = f.block
    e = _RedEmit(b, f.idx64, f.cdt, False)
    P = {}
    P["W"] = b.param("W", PtrType(e.cty, "global"), readonly=True, align=8)
    P["C"] = ptr(b, "C", f.d, readonly=True)
    P["D"] = ptr(b, "D", f.d, readonly=False)
    P["alpha"] = e.scalar_param("alpha")
    P["beta"] = e.scalar_param("beta")
    P["opC"] = b.param("opC", I32)
    P["n_out"] = b.param("n_out", I32)
    split = b.param("split", I32)
    kept = _Slots(e, "k", MAX_RANK, ("D",))
    L = f.lanes
    tid = b.thread_id_x()
    bx = b.block_id_x()
    if L > 1:
        bo = b.mul(bx, e.c32(f.block // L))
        g = b.lshr(tid, e.c32(_log2(L)))
        o = b.add(bo, g)
        lane = b.land(tid, e.c32(L - 1))
    else:
        bo = b.mul(bx, e.c32(f.block))
        o = b.add(bo, tid)
        lane = e.c32(0)
    ok = b.cmp_lt(o, P["n_out"])
    last = b.sub(P["n_out"], e.c32(1))
    o_ld = b.select(ok, o, last)
    ident = _identity(e, f.op)
    # trips: ceil(split / lanes)
    if L > 1:
        sl = b.add(split, e.c32(L - 1))
        trips = b.lshr(sl, e.c32(_log2(L)))
    else:
        trips = split
    k = e.next_uid()
    loop = b.scf_for_iter(
        e.c32(0), trips, e.c32(1), [(f"facc{k}", ident)], iv_name=f"fit{k}"
    )
    with loop as (j, cur):
        jl = b.mul(j, e.c32(L))
        si = b.add(jl, lane)
        valid = b.cmp_lt(si, split)
        safe = b.select(valid, si, e.c32(0))
        so = e.off(safe, e.widen(P["n_out"]))
        ow = e.widen(o_ld)
        woff = b.add(so, ow)
        v = _load_w(e, P["W"], woff)
        x = b.select(valid, v, ident)
        b.scf_yield(_combine(e, f.op, cur[0], x))
    acc = loop.results[0]
    for i in range(_log2(L)):
        sh = b.warp_shuffle_xor(acc, 1 << i)
        acc = _combine(e, f.op, acc, sh)
    kc = kept.coords(e, o_ld)
    d_off = kept.offset(e, kc, "D")
    cond = ok
    if L > 1:
        lead = b.cmp_eq(lane, e.c32(0))
        cond = b.land(ok, lead)
    with b.scf_if(cond):
        v = _epilogue(e, P, acc, d_off, f.d)
        e.store(P["D"], d_off, [v], f.d)
    return b.kernel
