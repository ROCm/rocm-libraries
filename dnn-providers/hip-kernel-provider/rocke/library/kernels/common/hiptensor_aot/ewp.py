# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned tiled elementwise kernel for transposing layouts (hipTensor permutations and
elementwise binary / trinary operations whose inputs are not all laid out like the output).

``D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C))`` with the operators, scales,
conversions and zero-scale rules of :mod:`.ew`, on tensors of one element size (2, 4 or 8
bytes), with 16-byte global accesses on both sides of the transposition.

Index space (built by the host from the canonical problem, every extent and stride a kernel
argument): two *runs* and a batch space.

* the ``o`` run: up to two modes (``o0`` first), D's stride-1 mode and, when needed, D's next
  fastest mode. Positions along the run are ``i = i0 + eo * i1``.
* the ``t`` run: up to two modes, the stride-1 mode of the transposed inputs (kind ``T``) and,
  when needed, their next fastest mode.
* the batch space: every other mode, at most :data:`SLOTS_B` digits (one index per loop trip,
  wave-uniform offsets).

Mode ``t`` (``EwpSpec.mode``, element transposition): a CTA computes one ``to x tt`` tile of
the (o, t) plane per batch index. Phase 1 reads each ``T`` input in ``V x V`` blocks (``V``
16-byte loads along ``t``, one per ``o``; ``V`` elements per 16 bytes), transposes each block in
registers and writes 16-byte vectors along ``o`` into an LDS tile in D's order (``[t][o]``,
16-byte chunks XOR-swizzled with ``swz`` / ``swsh``). Phase 2 reads 16-byte vectors along ``o``
from LDS, reads the ``V`` inputs (laid out like D along ``o``) with 16-byte loads, computes and
writes D with 16-byte stores.

Mode ``a`` (atom: every tensor shares its stride-1 mode ``w``, the ``o`` and ``t`` runs are the
modes after it): a tile is ``tw`` 16-byte chunks of ``w`` by ``to`` positions of the ``o`` run
by ``tt`` positions of the ``t`` run. Phase 1 copies the ``T`` inputs' chunks (contiguous along
``w`` then ``t``) into an LDS tile in D's order (``[t][o][w]``), phase 2 reads them back along
``w`` then ``o`` (D's order) and computes as in mode ``t``. Each tensor is read and written in
runs of whole ``w`` atoms on both sides.

Kinds (``EwpSpec.kinds``, one letter per input in ``A [B] C`` order, compile-time): ``T`` read
through LDS, ``V`` read in phase 2 with 16-byte loads along ``o`` (mode ``a``: along ``w``).

``nt``: bit 0 streaming loads, bit 1 streaming stores (gfx942 / gfx950). ``walk``: the tile
coordinate that is fastest across consecutive CTAs (``o`` or ``t``; ``w`` is always fastest).
``tpc``: tiles per CTA (consecutive tile indices, one LDS tile reused).

``vb`` (vector bytes, default 16): every global access, LDS chunk and the ``V x V`` blocks use
``vb``-byte vectors (``V = vb / element size``), so that layouts whose extents or strides are not
multiples of 16 bytes keep vector accesses (e.g. ``vb = 4``: pairs of f16); ``tb`` (mode t, 0:
``vb``) is the vector of the ``T`` inputs' loads along ``t`` (``VT = tb / element size``), so that
phase 1 transposes ``V x VT`` blocks (``V`` loads of ``VT`` elements) when D's side allows only
narrower vectors than the transposed inputs' side. ``rag`` (ragged
classes): a tile may overhang the end of a run (mode ``a``: of the shared mode too). Positions
past an end are clamped to the last vector of the run; such a lane reads and writes the same
vector as the lane that owns it (the same values), so ragged tiles need no masks. The runs of
a ragged class need not hold a whole number of tiles; their inner extents (and the shared mode)
must still be multiples of ``V``.

Kernel ABI ``hiptensor-aot-ewp/v2``: ``A [B] [C] D``, ``alpha [beta] [gamma]``, ``opA [opB]
[opC] [opAB] [opABC]``, then (i32) ``eo, om, os, et, tm, ts`` (inner extents of the runs and
their magic pairs), ``nw, wm, ws, nf, fm, fs`` (tile counts: ``nw`` along ``w``, ``nf`` along the
walk's fast run, with magic pairs), ``ntl`` (tiles per batch index), ``nb, ystep``, ragged
classes only: ``lo, lt`` (the lengths of the o and t runs) and, mode ``a``, ``lw`` (the extent of
the shared mode), then the run strides ``s<T>o0, s<T>o1, s<T>t0, s<T>t1`` per tensor, and the
batch digits ``be<i>, bm<i>, bs<i>, s<T><i>`` (element offsets). (v1: no ``vb`` / ``rag``.)
Launch: block ``(block, 1, 1)``, grid ``(ceil(ntl / tpc), min(nb, 65535), 1)`` with
``ntl = nw * tiles_o * tiles_t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, TemporalHint, Value

from .common import (
    DTYPES,
    OP_ADD,
    OP_CONJ,
    OP_IDENTITY,
    OP_MAX,
    OP_MIN,
    OP_MUL,
    Emit,
    ptr,
)

EWP_ABI = "hiptensor-aot-ewp/v2"
#: batch digits (the modes that are not in the two runs)
SLOTS_B = 8
#: largest number of vectors a thread holds in phase 1
MAX_VECTORS = 32
#: LDS budget of the tiles (gfx950; the pack's classes of the other targets fit their LDS)
LDS_BYTES = 163840


@dataclass(frozen=True)
class EwpSpec:
    arity: int  # 1 (permute), 2 (binary), 3 (trinary)
    a: str
    d: str
    b: Optional[str] = None
    c: Optional[str] = None
    kinds: str = "T"  # one letter per input (A [B] C): T (through LDS) or V (direct)
    mode: str = "t"  # t: element transposition, a: shared stride-1 mode (atoms)
    to: int = 64  # tile along the o run (elements; mode a: positions of the run)
    tt: int = 64  # tile along the t run
    tw: int = 1  # mode a: vectors (vb bytes) of the shared mode per tile
    block: int = 256
    nt: int = 0  # bit 0: streaming loads, bit 1: streaming stores
    swz: int = 0  # XOR group of the LDS chunk index (0 / 1: none)
    swsh: int = 0  # chunk-index bit the swizzle key starts at
    walk: str = "o"  # the tile coordinate fastest across consecutive CTAs: o or t
    tpc: int = 1  # tiles per CTA (consecutive tile indices of one batch index)
    ilp: int = (
        0  # phase-2 units whose loads are issued together (0: all of the thread's)
    )
    vb: int = 16  # bytes per vector access (16, 8, 4 or 2; at least the element size)
    rag: int = 0  # 1: tiles may overhang the ends of the runs (ragged class)
    tb: int = 0  # mode t: bytes per vector of the T inputs along t (0: vb)

    # ------------------------------------------------------------------ derived
    def tensors(self) -> List[str]:
        return {1: ["A", "D"], 2: ["A", "C", "D"], 3: ["A", "B", "C", "D"]}[self.arity]

    def inputs(self) -> List[str]:
        return self.tensors()[:-1]

    def dtypes(self) -> Dict[str, Optional[str]]:
        return {"A": self.a, "B": self.b, "C": self.c, "D": self.d}

    def kind(self, n: str) -> str:
        return self.kinds[self.inputs().index(n)]

    @property
    def cdt(self) -> str:
        dts = [x for x in (self.a, self.b, self.c, self.d) if x]
        return "f64" if "f64" in dts else "f32"

    @property
    def es(self) -> int:
        return DTYPES[self.d][1]

    @property
    def vec(self) -> int:
        return self.vb // self.es

    @property
    def vect(self) -> int:
        """Elements per vector of the T inputs along t (mode t)."""
        return (self.tb or self.vb) // self.es

    def p1_units(self) -> int:
        """Phase-1 units per T input: V x V blocks (mode t) or vectors (mode a)."""
        if self.mode == "a":
            return self.to * self.tt * self.tw
        return (self.to // self.vec) * (self.tt // self.vect)

    def p2_units(self) -> int:
        """Phase-2 units: vectors of D."""
        if self.mode == "a":
            return self.to * self.tt * self.tw
        return (self.to // self.vec) * self.tt

    def tile_bytes(self) -> int:
        if self.mode == "a":
            return self.to * self.tt * self.tw * self.vb
        return self.to * self.tt * self.es

    @property
    def swlo(self) -> int:
        """Lowest chunk-index bit the swizzle flips: above the chunks of one atom (mode a)."""
        return self.tw.bit_length() - 1 if self.mode == "a" else 0

    def chunks(self) -> int:
        """Vector chunks (vb bytes) of one LDS tile."""
        return self.tile_bytes() // self.vb

    def check(self) -> None:
        if self.arity not in (1, 2, 3):
            raise ValueError("arity must be 1, 2 or 3")
        if self.arity >= 2 and not self.c:
            raise ValueError("arity >= 2 needs a C dtype")
        if self.arity == 3 and not self.b:
            raise ValueError("arity 3 needs a B dtype")
        for n in self.tensors():
            if self.dtypes()[n] not in DTYPES:
                raise ValueError(f"{n}: dtype {self.dtypes()[n]!r}")
            if DTYPES[self.dtypes()[n]][1] != self.es:
                raise ValueError("every tensor must have the same element size")
        if len(self.kinds) != self.arity or any(k not in "TV" for k in self.kinds):
            raise ValueError("kinds: one of T / V per input")
        if "T" not in self.kinds:
            raise ValueError("at least one input read through LDS (kind T)")
        if self.mode not in ("t", "a"):
            raise ValueError("mode must be t or a")
        if self.vb not in (2, 4, 8, 16) or self.vb < self.es:
            raise ValueError(
                "vb must be 2, 4, 8 or 16 bytes and at least the element size"
            )
        if self.rag not in (0, 1):
            raise ValueError("rag must be 0 or 1")
        if self.tb and (
            self.tb not in (2, 4, 8, 16) or self.tb < self.es or self.mode != "t"
        ):
            raise ValueError(
                "tb: 2, 4, 8 or 16 bytes, at least the element size, mode t only"
            )
        for x in (self.to, self.tt, self.tw, self.block):
            if x <= 0 or x & (x - 1):
                raise ValueError("tile extents and block must be powers of two")
        if self.mode == "t" and (self.to % self.vec or self.tt % self.vect):
            raise ValueError("mode t: the tile extents must be multiples of the vector")
        if self.mode == "t" and self.tw != 1:
            raise ValueError("mode t: tw must be 1")
        if self.block < 64 or self.block > 1024:
            raise ValueError("block must be 64 .. 1024")
        if self.p1_units() < 1 or self.p2_units() < 1:
            raise ValueError("empty tile")
        held = -(-self.p1_units() // self.block) * (self.vec if self.mode == "t" else 1)
        if held > MAX_VECTORS:
            raise ValueError(f"{held} vectors per thread in phase 1 > {MAX_VECTORS}")
        if self.kinds.count("T") * self.tile_bytes() > LDS_BYTES:
            raise ValueError("the LDS tiles exceed the LDS budget")
        if self.nt not in (0, 1, 2, 3):
            raise ValueError("nt must be 0 .. 3")
        if self.vec == 1 and self.nt & 1:
            raise ValueError(
                "one-element vectors: scalar loads have no streaming hint (nt bit 0)"
            )
        if self.swz not in (0, 1) and (
            self.swz & (self.swz - 1)
            or (1 << self.swsh) < self.swz << self.swlo
            or self.chunks() % (1 << self.swsh)
        ):
            raise ValueError("the swizzle is not a bijection of the tile's chunks")
        if self.walk not in ("o", "t"):
            raise ValueError("walk must be o or t")
        if self.tpc not in (1, 2, 4, 8):
            raise ValueError("tpc must be 1, 2, 4 or 8")
        if self.ilp not in (0, 1, 2, 4, 8):
            raise ValueError("ilp must be 0, 1, 2, 4 or 8")

    def kernel_name(self) -> str:
        dts = "_".join(x for x in (self.a, self.b, self.c, self.d) if x)
        n = (
            f"hiptensor_aot_ewp{self.arity}_{dts}_{self.kinds}_{self.mode}"
            f"_o{self.to}_t{self.tt}"
        )
        if self.mode == "a":
            n += f"_w{self.tw}"
        n += f"_b{self.block}"
        if self.nt:
            n += f"_nt{self.nt}"
        if self.swz > 1:
            n += f"_z{self.swz}s{self.swsh}"
        if self.walk != "o":
            n += "_wt"
        if self.tpc > 1:
            n += f"_x{self.tpc}"
        if self.ilp:
            n += f"_g{self.ilp}"
        if self.vb != 16:
            n += f"_v{self.vb}"
        if self.rag:
            n += "_r"
        if self.tb:
            n += f"_u{self.tb}"
        return n


def default_swizzle(spec: EwpSpec, banks_chunks: int = 16) -> Tuple[int, int]:
    """``(swz, swsh)``: an XOR swizzle that spreads phase 1's LDS writes over the banks (16
    chunks of 16 bytes span the 64 banks of gfx950; 8 the 32 banks of the other targets;
    ``banks_chunks`` counts 16-byte chunks, a class of ``vb``-byte chunks has ``16 / vb`` times
    as many). Mode t: lanes of a wave write the same column chunk of rows ``V`` apart, so the key
    comes from the row bits above ``V`` and flips the column bits. Mode a: lanes write the chunks
    of one atom for consecutive ``t``, rows ``to * tw`` chunks apart: the key comes from the ``t``
    bits and flips the ``o`` bits above the atom's chunks."""
    V = spec.vec
    banks_chunks = banks_chunks * 16 // spec.vb
    if spec.mode == "a":
        lo = spec.swlo
        k = max(1, min(banks_chunks // spec.tw, spec.to))
        sh = (spec.to * spec.tw).bit_length() - 1
    else:
        lo = 0
        k = max(1, min(banks_chunks, spec.to // V * V))
        sh = (spec.to // V).bit_length() - 1 + (spec.vect.bit_length() - 1)
    if k <= 1 or (1 << sh) < (k << lo) or spec.chunks() % (1 << sh):
        return 0, 0
    return k, sh


# ---------------------------------------------------------------------- kernel
def _pow2_div(b: IRBuilder, e: Emit, x: Value, d: int) -> Value:
    if d == 1:
        return x
    return b.lshr(x, e.c32(d.bit_length() - 1))


def _pow2_mod(b: IRBuilder, e: Emit, x: Value, d: int) -> Value:
    if d == 1:
        return e.c32(0)
    return b.land(x, e.c32(d - 1))


def _binary_fixed(b: IRBuilder, a: Value, c: Value, op: int) -> Value:
    """:meth:`.common.Emit.binary` of a known operator (the same value, without the selects of
    the other operators)."""
    if op == OP_ADD:
        return b.fadd(a, c)
    if op == OP_MUL:
        return b.fmul(a, c)
    r = b.fmax(a, c) if op == OP_MAX else b.fmin(a, c)
    uno = b.fcmp("uno", a, c)
    nan = b.fadd(a, c)
    return b.select(uno, nan, r)


def build_ewp(spec: EwpSpec, *, arch: str = "gfx950") -> KernelDef:
    """The kernel of a class (``arch`` is not used: every class is target-neutral IR; the pack
    builds a target's classes for it)."""
    spec.check()
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    e = Emit(b, False, spec.cdt, False)
    names = spec.tensors()
    ins = spec.inputs()
    dts = spec.dtypes()
    V = spec.vec
    VT = spec.vect
    BS = spec.block
    TO, TT, TW = spec.to, spec.tt, spec.tw
    mode_a = spec.mode == "a"
    hint_ld = TemporalHint.STREAMING if spec.nt & 1 else TemporalHint.DEFAULT
    hint_st = TemporalHint.STREAMING if spec.nt & 2 else TemporalHint.DEFAULT
    tins = [n for n in ins if spec.kind(n) == "T"]

    P = {n: ptr(b, n, dts[n], readonly=n != "D") for n in names}
    scal_of = {"A": "alpha", "B": "beta", "C": "gamma"}
    S = {n: e.scalar_param(scal_of[n]) for n in ins}
    OPS = {n: b.param("op" + n, I32) for n in ins}
    if spec.arity >= 2:
        OPS["AB"] = b.param("opAB", I32)
    if spec.arity == 3:
        OPS["ABC"] = b.param("opABC", I32)
    eo = b.param("eo", I32)
    om = b.param("om", I32)
    os_ = b.param("os", I32)
    et = b.param("et", I32)
    tm = b.param("tm", I32)
    ts = b.param("ts", I32)
    nw = b.param("nw", I32)
    wm = b.param("wm", I32)
    ws = b.param("ws", I32)
    nf = b.param("nf", I32)
    fm = b.param("fm", I32)
    fs = b.param("fs", I32)
    ntl = b.param("ntl", I32)
    nb = b.param("nb", I32)
    ystep = b.param("ystep", I32)
    if spec.rag:
        lo = b.param("lo", I32)
        lt = b.param("lt", I32)
        lw = b.param("lw", I32) if mode_a else None
    SO = {n: (b.param(f"s{n}o0", I32), b.param(f"s{n}o1", I32)) for n in names}
    ST = {n: (b.param(f"s{n}t0", I32), b.param(f"s{n}t1", I32)) for n in names}
    bext, bmg = [], []
    bst: Dict[str, List[Value]] = {n: [] for n in names}
    for i in range(SLOTS_B):
        bext.append(b.param(f"be{i}", I32))
        if i < SLOTS_B - 1:
            bmg.append((b.param(f"bm{i}", I32), b.param(f"bs{i}", I32)))
        for n in names:
            bst[n].append(b.param(f"s{n}{i}", I32))

    tid = b.thread_id_x()
    bid = b.block_id_x()
    first = b.mul(bid, e.c32(spec.tpc)) if spec.tpc > 1 else bid
    zero = e.cf(0.0)
    nz = {n: b.fcmp("one", S[n], zero) for n in ins}
    allnz = nz[ins[0]]
    for n in ins[1:]:
        allnz = b.land(allnz, nz[n])
    # the fast path: every scale non-zero and every unary operator the identity (CONJ of a real;
    # hipTensor folds NEG into the scale)
    idop = None
    for n in ins:
        t_ = b.lor(
            b.cmp_eq(OPS[n], e.c32(OP_IDENTITY)), b.cmp_eq(OPS[n], e.c32(OP_CONJ))
        )
        idop = t_ if idop is None else b.land(idop, t_)
    fastp = b.land(allnz, idop)
    if spec.rag:
        # ragged class: the last position a lane may touch along each run (a lane past the end
        # takes the last vector of the run: the same data as the lane that owns it)
        if mode_a:
            last_w = b.sub(lw, e.c32(V))
            last_o1 = b.sub(lo, e.c32(1))
            last_t1 = b.sub(lt, e.c32(1))
        else:
            last_ov = b.sub(lo, e.c32(V))
            last_tv = b.sub(lt, e.c32(VT))
            last_t1 = b.sub(lt, e.c32(1))

    def gload(n: str, off: Value, w: int = V) -> Value:
        """A vector of ``w`` elements of tensor ``n`` (``<w x ty>``, default ``vb`` bytes; w = 1:
        one scalar load, no streaming hint)."""
        ty = DTYPES[dts[n]][0]
        nbytes = w * DTYPES[dts[n]][1]
        if w == 1:
            return b.vec_pack([b.global_load(P[n], off, ty, align=nbytes)], ty)
        return b.global_load_vN(P[n], off, ty, w, align=nbytes, temporal_hint=hint_ld)

    def lstore(n: str, chunk: Value, v: Value) -> None:
        if V == 1:
            b.smem_store_vN(lds[n], [chunk], b.vec_extract(v, 0), 1)
        else:
            b.smem_store_vN(lds[n], [chunk], v, V)

    def run_off(n: str, pos: Value, ext: Value, m: Value, s: Value, st) -> Value:
        """Offset of position ``pos`` of a run (``pos = i0 + ext * i1``) in tensor ``n``."""
        hi, lo = e.divmod(pos, ext, m, s)
        return b.add(b.mul(lo, st[0]), b.mul(hi, st[1]))

    def ooff(n: str, pos: Value) -> Value:
        return run_off(n, pos, eo, om, os_, SO[n])

    def toff(n: str, pos: Value) -> Value:
        return run_off(n, pos, et, tm, ts, ST[n])

    lds = {}
    for n in tins:
        ty = DTYPES[dts[n]][0]
        lds[n] = b.smem_alloc(ty, [spec.chunks() * V], name_hint=f"t{n}")

    def swz(chunk: Value) -> Value:
        if spec.swz <= 1:
            return chunk
        key = b.land(b.lshr(chunk, e.c32(spec.swsh)), e.c32(spec.swz - 1))
        if spec.swlo:
            key = b.shl(key, e.c32(spec.swlo))
        return b.xor(chunk, key)

    def units(total: int) -> List[Value]:
        """This thread's units of a phase (a thread past the last unit redoes the last one:
        identical data, no branch)."""
        out = []
        for it in range(-(-total // BS)):
            u = tid if it == 0 else b.add(tid, e.c32(it * BS))
            if (it + 1) * BS > total:
                u = b.smin(u, e.c32(total - 1))
            out.append(u)
        return out

    def one_tile(tile_id: Value, boff: Dict[str, Value]) -> None:
        # tile coordinates: w fastest, then the walk's fast run, then the other run
        q, tile_w = e.divmod(tile_id, nw, wm, ws)
        slow, fast = e.divmod(q, nf, fm, fs)
        tile_o, tile_t = (fast, slow) if spec.walk == "o" else (slow, fast)
        o0 = b.mul(tile_o, e.c32(TO))
        t0 = b.mul(tile_t, e.c32(TT))
        w0 = b.mul(tile_w, e.c32(TW * V))
        # every wave done with the previous tile's LDS
        b.sync()

        # ---------------------------------------------------------- phase 1
        for n in tins:
            ty = DTYPES[dts[n]][0]
            with b.scf_if(nz[n]):
                loaded = []
                for u in units(spec.p1_units()):
                    if mode_a:
                        c = _pow2_mod(b, e, u, TW)
                        rest = _pow2_div(b, e, u, TW)
                        tl = _pow2_mod(b, e, rest, TT)
                        ol = _pow2_div(b, e, rest, TT)
                        if spec.rag:
                            pw = b.smin(b.add(w0, b.mul(c, e.c32(V))), last_w)
                            po = b.smin(b.add(o0, ol), last_o1)
                            pt = b.smin(b.add(t0, tl), last_t1)
                            c = _pow2_div(b, e, b.sub(pw, w0), V)
                            ol = b.sub(po, o0)
                            tl = b.sub(pt, t0)
                            off = b.add(
                                b.add(boff[n], pw),
                                b.add(ooff(n, po), toff(n, pt)),
                            )
                        else:
                            off = b.add(
                                b.add(boff[n], b.add(w0, b.mul(c, e.c32(V)))),
                                b.add(ooff(n, b.add(o0, ol)), toff(n, b.add(t0, tl))),
                            )
                        v = gload(n, off)
                        loaded.append((ol, tl, c, [v]))
                    else:
                        ub_t = _pow2_mod(b, e, u, TT // VT)
                        ub_o = _pow2_div(b, e, u, TT // VT)
                        ol = b.mul(ub_o, e.c32(V))
                        tl = b.mul(ub_t, e.c32(VT))
                        if spec.rag:
                            po = b.smin(b.add(o0, ol), last_ov)
                            pt = b.smin(b.add(t0, tl), last_tv)
                            ub_o = _pow2_div(b, e, b.sub(po, o0), V)
                            ub_t = _pow2_div(b, e, b.sub(pt, t0), VT)
                            base = b.add(boff[n], b.add(ooff(n, po), toff(n, pt)))
                        else:
                            base = b.add(
                                boff[n],
                                b.add(ooff(n, b.add(o0, ol)), toff(n, b.add(t0, tl))),
                            )
                        rws = []
                        for j in range(V):
                            o_ = (
                                base
                                if j == 0
                                else b.add(base, b.mul(e.c32(j), SO[n][0]))
                            )
                            rws.append(gload(n, o_, VT))
                        loaded.append((ub_o, ub_t, None, rws))
                for ol, tl, c, rws in loaded:
                    if mode_a:
                        # chunk (t, o, w) of the [t][o][w] tile
                        ch = b.add(b.mul(b.add(b.mul(tl, e.c32(TO)), ol), e.c32(TW)), c)
                        lstore(n, b.mul(swz(ch), e.c32(V)), rws[0])
                        continue
                    ub_o, ub_t = ol, tl
                    for i in range(VT):
                        # elements (o .. o + V - 1, t + i): column i of the V x VT block
                        vv = b.vec_pack(
                            [b.vec_extract(rws[j], i) for j in range(V)], ty
                        )
                        trow = (
                            b.add(b.mul(ub_t, e.c32(VT)), e.c32(i))
                            if i
                            else b.mul(ub_t, e.c32(VT))
                        )
                        ch = b.add(b.mul(trow, e.c32(TO // V)), ub_o)
                        lstore(n, b.mul(swz(ch), e.c32(V)), vv)
        b.sync()

        # ---------------------------------------------------------- phase 2
        p2 = []
        for u in units(spec.p2_units()):
            if mode_a:
                c = _pow2_mod(b, e, u, TW)
                rest = _pow2_div(b, e, u, TW)
                ol = _pow2_mod(b, e, rest, TO)
                tl = _pow2_div(b, e, rest, TO)
                if spec.rag:
                    wv = b.smin(b.add(w0, b.mul(c, e.c32(V))), last_w)
                    go = b.smin(b.add(o0, ol), last_o1)
                    gt = b.smin(b.add(t0, tl), last_t1)
                    c = _pow2_div(b, e, b.sub(wv, w0), V)
                    ol = b.sub(go, o0)
                    tl = b.sub(gt, t0)
                ch = b.add(b.mul(b.add(b.mul(tl, e.c32(TO)), ol), e.c32(TW)), c)
                if not spec.rag:
                    wv = b.add(w0, b.mul(c, e.c32(V)))
            else:
                uo = _pow2_mod(b, e, u, TO // V)
                tl = _pow2_div(b, e, u, TO // V)
                ol = b.mul(uo, e.c32(V))
                if spec.rag:
                    go = b.smin(b.add(o0, ol), last_ov)
                    gt = b.smin(b.add(t0, tl), last_t1)
                    uo = _pow2_div(b, e, b.sub(go, o0), V)
                    tl = b.sub(gt, t0)
                ch = b.add(b.mul(tl, e.c32(TO // V)), uo)
                wv = None
            if not spec.rag:
                go = b.add(o0, ol)
                gt = b.add(t0, tl)

            def offs(n: str, go=go, gt=gt, wv=wv) -> Value:
                o_ = b.add(boff[n], b.add(ooff(n, go), toff(n, gt)))
                return o_ if wv is None else b.add(o_, wv)

            p2.append((ch, offs))

        def raw(n: str, unit) -> Value:
            ty = DTYPES[dts[n]][0]
            ch, offs = unit
            if spec.kind(n) == "T":
                return b.smem_load_vN(lds[n], b.mul(swz(ch), e.c32(V)), dtype=ty, n=V)
            return gload(n, offs(n))

        def body(every: bool, op: Optional[int] = None) -> None:
            """``op`` given: the fast path (no operator table; ``op`` the binary operator, 0 for
            a permutation)."""
            g = spec.ilp or len(p2)
            dty = DTYPES[spec.d][0]
            for g0 in range(0, len(p2), g):
                grp = p2[g0 : g0 + g]
                vals = {}
                if every:
                    loads = {n: [raw(n, unit) for unit in grp] for n in ins}
                for n in ins:

                    def get(_c, n=n):
                        vs = [raw(n, unit) for unit in grp]
                        return [
                            e.to_c(b.vec_extract(v, k), dts[n])
                            for v in vs
                            for k in range(V)
                        ]

                    if every:
                        xs = [
                            e.to_c(b.vec_extract(v, k), dts[n])
                            for v in loads[n]
                            for k in range(V)
                        ]
                        if op is not None:
                            vals[n] = [b.fmul(S[n], x) for x in xs]
                        else:
                            vals[n] = [
                                b.fmul(S[n], x) for x in e.unary_list(xs, OPS[n])
                            ]
                    else:
                        xs = e.when(nz[n], [zero] * (V * len(grp)), get)
                        vals[n] = [
                            b.select(nz[n], b.fmul(S[n], x), zero)
                            for x in e.unary_list(xs, OPS[n])
                        ]
                for q, (ch, offs) in enumerate(grp):
                    out = []
                    for k in range(q * V, q * V + V):
                        if spec.arity == 1:
                            out.append(vals["A"][k])
                        elif op is not None:
                            x = vals["A"][k]
                            if spec.arity == 3:
                                x = _binary_fixed(b, x, vals["B"][k], op)
                            out.append(_binary_fixed(b, x, vals["C"][k], op))
                        elif spec.arity == 2:
                            out.append(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]))
                        else:
                            ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"])
                            out.append(e.binary(ab, vals["C"][k], OPS["ABC"]))
                    outs = [e.from_c(x, spec.d) for x in out]
                    b.global_store_vN(
                        P["D"],
                        offs("D"),
                        b.vec_pack(outs, dty),
                        V,
                        align=spec.vb,
                        temporal_hint=hint_st,
                    )

        # the fast path per binary operator (trinary: the same operator twice); then every scale
        # non-zero (straight-line loads, operator table); otherwise skip the inputs of zero scale
        if spec.arity == 1:
            conds = [(fastp, 0)]
        else:
            conds = []
            for bop in (OP_ADD, OP_MUL, OP_MAX, OP_MIN):
                cnd = b.land(fastp, b.cmp_eq(OPS["AB"], e.c32(bop)))
                if spec.arity == 3:
                    cnd = b.land(cnd, b.cmp_eq(OPS["ABC"], e.c32(bop)))
                conds.append((cnd, bop))
        covered = None
        for cnd, bop in conds:
            with b.scf_if(cnd):
                body(True, bop)
            covered = cnd if covered is None else b.lor(covered, cnd)
        with b.scf_if(b.land(allnz, b.lnot(covered))):
            body(True)
        with b.scf_if(b.lnot(allnz)):
            body(False)

    rows = b.scf_for(b.block_id_y(), nb, ystep, iv_name="bt")
    with rows as bi:
        # batch offsets (wave-uniform)
        boff: Dict[str, Optional[Value]] = {n: None for n in names}
        r = bi
        for i in range(SLOTS_B):
            if i < SLOTS_B - 1:
                r, dg = e.divmod(r, bext[i], *bmg[i])
            else:
                dg = r
            for n in names:
                t_ = b.mul(dg, bst[n][i])
                boff[n] = t_ if boff[n] is None else b.add(boff[n], t_)
        for k in range(spec.tpc):
            tile_id = first if k == 0 else b.add(first, e.c32(k))
            if spec.tpc > 1:
                with b.scf_if(b.cmp_lt(tile_id, ntl)):
                    one_tile(tile_id, boff)
            else:
                one_tile(tile_id, boff)
    return b.kernel


# ---------------------------------------------------------------------- host rules
def ewp_runs(
    spec: EwpSpec, dims
) -> Optional[Tuple[List[int], List[int], Optional[int]]]:
    """``(o_run, t_run, w)``: the dims of the two runs (inner first) and the shared stride-1
    dim (mode a), or None when the layout does not fit the class (see :func:`ewp_plan`).

    ``dims``: canonical modes ``(extent, [stride per tensor in spec.tensors() order])``.
    """
    names = spec.tensors()
    nd = len(names) - 1
    kidx = [i for i, n in enumerate(names[:-1]) if spec.kind(n) == "T"]
    ki = kidx[0]

    def unit_dim(k: int, exclude) -> Optional[int]:
        return next(
            (i for i, (_e, s) in enumerate(dims) if s[k] == 1 and i not in exclude),
            None,
        )

    w = None
    excl: List[int] = []
    if spec.mode == "a":
        w = unit_dim(nd, [])
        if w is None or any(dims[w][1][k] != 1 for k in range(len(names))):
            return None
        excl = [w]
        ew_ = dims[w][0]

        def first(k: int, exclude) -> Optional[int]:
            # the dim right after w in tensor k (stride == extent of w)
            return next(
                (
                    i
                    for i, (_e, s) in enumerate(dims)
                    if s[k] == ew_ and i not in exclude
                ),
                None,
            )

        o_first = first(nd, excl)
        t_first = first(ki, excl + ([o_first] if o_first is not None else []))
    else:
        o_first = unit_dim(nd, [])
        t_first = unit_dim(ki, [o_first] if o_first is not None else [])
    if o_first is None or t_first is None or o_first == t_first:
        return None

    def grow(first: int, k: int, tile: int, exclude) -> Optional[List[int]]:
        """The run from ``first``: alone when its extent is a multiple of the tile, else with
        the mode contiguous after it in tensor ``k`` (not in ``exclude``) when the two hold a
        whole number of tiles. A ragged class takes that mode whenever there is one (a longer
        run wastes less of its last tile), else the mode alone."""
        if dims[first][0] % tile == 0:
            return [first]
        want = dims[first][0] * dims[first][1][k]
        nxt = next(
            (
                i
                for i, (_e, s) in enumerate(dims)
                if s[k] == want and i != first and i not in exclude
            ),
            None,
        )
        if spec.rag:
            return [first] if nxt is None else [first, nxt]
        if nxt is None or (dims[first][0] * dims[nxt][0]) % tile:
            return None
        return [first, nxt]

    if spec.rag:
        # both orders (t run first, o run first); the one whose tiles waste less; on a tie the
        # o run first for one or two inputs (D's stores then cover longer contiguous runs), the
        # t run first for three (their loads outweigh the stores)
        def fill(o_run, t_run):
            no = 1
            for i in o_run:
                no *= dims[i][0]
            nt = 1
            for i in t_run:
                nt *= dims[i][0]
            return no * nt, -(-no // spec.to) * spec.to * -(-nt // spec.tt) * spec.tt

        t1 = grow(t_first, ki, spec.tt, excl + [o_first])
        o1 = grow(o_first, nd, spec.to, excl + t1)
        o2 = grow(o_first, nd, spec.to, excl + [t_first])
        t2 = grow(t_first, ki, spec.tt, excl + o2)
        u1, w1 = fill(o1, t1)
        u2, w2 = fill(o2, t2)
        if u2 * w1 > u1 * w2 or (u2 * w1 == u1 * w2 and spec.arity < 3):
            return o2, t2, w
        return o1, t1, w
    # the t run first, then the o run; when that fails (both want the same mode), the other way
    t_run = grow(t_first, ki, spec.tt, excl + [o_first])
    o_run = grow(o_first, nd, spec.to, excl + t_run) if t_run is not None else None
    if o_run is None:
        o_run = grow(o_first, nd, spec.to, excl + [t_first])
        t_run = grow(t_first, ki, spec.tt, excl + o_run) if o_run is not None else None
        if t_run is None:
            return None
    return o_run, t_run, w


def ewp_plan(spec: EwpSpec, dims, limits=(1 << 31) - 1):
    """Arguments and grid of an :mod:`.ewp` launch (the rules hipTensor's C++ planner applies),
    or None when the layout does not fit the class: the runs (:func:`ewp_runs`) must hold a
    whole number of tiles (a ragged class: any length), D (and every ``V`` input) must have
    stride 1 along ``o`` (mode t) and every ``T`` input stride 1 along ``t``, the inner extents
    (mode a: the shared mode) and every other stride must keep the class's vectors aligned, at
    most :data:`SLOTS_B` batch modes, every tensor's offsets below 2^31, no negative stride.
    """
    from .common import as_i32_bits, magic_u31

    spec.check()
    names = spec.tensors()
    nd = len(names) - 1
    V = spec.vec
    rr = ewp_runs(spec, dims)
    if rr is None:
        return None
    o_run, t_run, w = rr
    used = o_run + t_run + ([w] if w is not None else [])
    others = [i for i in range(len(dims)) if i not in used]
    if len(others) > SLOTS_B:
        return None
    for k in range(len(names)):
        if any(s[k] < 0 for _e, s in dims):
            return None
        if sum((e_ - 1) * s[k] for e_, s in dims) > limits:
            return None
    eo0 = dims[o_run[0]][0]
    et0 = dims[t_run[0]][0]
    if spec.mode == "a":
        ew_ = dims[w][0]
        if ew_ % (V if spec.rag else spec.tw * V):
            return None
        # a tile row covers the whole shared mode or at least 128 bytes of it
        if spec.tw * V < ew_ and spec.tw * spec.vb < 128:
            return None
        for k in range(len(names)):
            if any(dims[i][1][k] % V for i in used + others if i != w):
                return None
    else:
        VT = spec.vect
        if eo0 % V or et0 % VT:
            return None
        for k, n in enumerate(names):
            direct = n == "D" or spec.kind(n) == "V"
            unit = o_run[0] if direct else t_run[0]
            vk = V if direct else VT
            if dims[unit][1][k] != 1:
                return None
            if any(dims[i][1][k] % vk for i in used + others if i != unit):
                return None
    vals: Dict[str, int] = {}

    def magic(mn: str, sn: str, d: int) -> None:
        m, s = magic_u31(d)
        vals[mn], vals[sn] = as_i32_bits(m), s

    no = 1
    for i in o_run:
        no *= dims[i][0]
    nt_ = 1
    for i in t_run:
        nt_ *= dims[i][0]
    if spec.rag:
        tiles_o, tiles_t = -(-no // spec.to), -(-nt_ // spec.tt)
        tiles_w = -(-dims[w][0] // (spec.tw * V)) if w is not None else 1
        vals["lo"] = no
        vals["lt"] = nt_
        if w is not None:
            vals["lw"] = dims[w][0]
    else:
        tiles_o, tiles_t = no // spec.to, nt_ // spec.tt
        tiles_w = dims[w][0] // (spec.tw * V) if w is not None else 1
    vals["eo"] = eo0
    magic("om", "os", eo0)
    vals["et"] = et0
    magic("tm", "ts", et0)
    vals["nw"] = tiles_w
    magic("wm", "ws", tiles_w)
    nf = tiles_o if spec.walk == "o" else tiles_t
    vals["nf"] = nf
    magic("fm", "fs", nf)
    nbatch = 1
    for i in others:
        nbatch *= dims[i][0]
    vals["nb"] = nbatch
    vals["ystep"] = min(nbatch, 65535)
    for k, n in enumerate(names):
        for r, run in (("o", o_run), ("t", t_run)):
            vals[f"s{n}{r}0"] = dims[run[0]][1][k]
            vals[f"s{n}{r}1"] = dims[run[1]][1][k] if len(run) > 1 else 0
    for i in range(SLOTS_B):
        e_, s = dims[others[i]] if i < len(others) else (1, [0] * len(names))
        vals[f"be{i}"] = e_
        if i < SLOTS_B - 1:
            magic(f"bm{i}", f"bs{i}", e_)
        for k, n in enumerate(names):
            vals[f"s{n}{i}"] = s[k] if e_ != 1 else 0
    ntl = tiles_w * tiles_o * tiles_t
    if ntl > (1 << 31) - 1:
        return None
    vals["ntl"] = ntl
    return vals, (-(-ntl // spec.tpc), min(nbatch, 65535), 1)


def ewp_kinds(spec_like_names: Sequence[str], dims, mode: str) -> Optional[str]:
    """The kinds of the inputs of a layout (``T``: fastest along another mode than D, ``V``:
    fastest along D's stride-1 mode, or, in mode a, along the mode after the shared one like D),
    or None when an input is neither."""
    nd = len(spec_like_names) - 1
    o = next((i for i, (_e, s) in enumerate(dims) if s[nd] == 1), None)
    if o is None:
        return None
    out = ""
    if mode == "a":
        ew_ = dims[o][0]
        od = next(
            (i for i, (_e, s) in enumerate(dims) if s[nd] == ew_ and i != o), None
        )
        for k in range(nd):
            if dims[o][1][k] != 1:
                return None
            out += "V" if od is not None and dims[od][1][k] == ew_ else "T"
        return out
    for k in range(nd):
        if dims[o][1][k] == 1:
            out += "V"
        elif any(s[k] == 1 for _e, s in dims):
            out += "T"
        else:
            return None
    return out


__all__ = [
    "EWP_ABI",
    "EwpSpec",
    "LDS_BYTES",
    "MAX_VECTORS",
    "SLOTS_B",
    "build_ewp",
    "ewp_kinds",
    "ewp_plan",
    "ewp_runs",
]
