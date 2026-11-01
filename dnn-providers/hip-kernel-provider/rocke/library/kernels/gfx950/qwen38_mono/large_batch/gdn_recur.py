# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L1: the GDN (gated delta rule) recurrence at large batch (rows 8-128, TP1 / TP2), gfx950.

``experiments/qwen38-true-mono/DESIGN-LB.md`` 6.1 and 12 (P-L1). Three kernel families, all over ATOM's state
contract: a per-row pointer table ``tab[layer][row] = (conv_in, conv_out, rec_in, rec_out)`` (i64), each
``rec_*`` pointing at that row's ``[HV][DV][DK]`` state of one layer (``[v, k]``, k contiguous; TP2 ranks hold
their 24 value heads):

* :func:`build_lb_kernel` (``q38lb_gdnr``): the LB recurrence. A unit is one (request, value head); ``waves``
  waves of a 512-thread CTA share a unit (``waves=8``: one unit per CTA at a time), each wave owning
  ``128 / waves`` v-rows. ``lanes`` adjacent lanes split a v-row's 128 k values into interleaved 16-byte
  accesses (``lanes=4``: a group of 4 lanes reads 64 contiguous bytes per instruction), so ``kv = S k`` and
  ``o = S' q`` are in-lane fma chains plus a ``log2(lanes)``-step DPP butterfly: no barriers and no LDS block
  sums inside a unit. The state (and the unit's small inputs) of the next ``depth - 1`` work items are issued
  before the current item's math (a software pipeline carried in registers). The gated RMSNorm needs the sum of
  squares over all 128 v-rows: with ``waves > 1`` the waves of a unit meet through an LDS ring entry
  (o values, per-wave partials, an arrival counter): the LAST arriver finishes the unit (fixed-order sum of the
  partials, so the result does not depend on who arrives last) while the others go on to their next unit; a
  ring entry is reused only after its previous unit's finisher marked it done (a bounded LDS poll that in
  practice never spins). One barrier per launch (LDS init). ``state="bf16"``: bf16 state storage, f32 math
  (DESIGN-LB N11, accuracy-gated later). ``chain=G``: MTP verify, G consecutive rows of one request share one
  state read (``snap="all"``: every row's snapshot written, ATOM's contract today; ``"last"``: only the last,
  the D2 contract).
* :func:`build_ft_kernel` (``q38lb_gdnft``): FlyTape's ``gdn_recur_with`` re-implemented (one 256-thread CTA per
  CU, one (row, head) unit at a time, FlyTape's ``[k, v]`` thread map: 4 value columns x 16 keys per thread,
  12 barriers per unit, LDS block and column sums), for an in-process comparison.
* :func:`build_stream_kernel` (``q38lb_gdnst``): the bytes floor: the same state bytes read and written (1
  read + G or 1 writes per unit for chains) with no math, at any depth, grid and cache policy.

Every kernel takes ``rows`` and ``layers`` at runtime: one launch walks ``layers`` consecutive layers' units as
one stream (the in-kernel stage cost without launch gaps); ``layers=1`` launches give the per-launch cost.
Invalid (tail) work items address a 0-byte buffer resource: their loads return 0 without memory traffic and
their stores are dropped, so the pipelined loop has no tail branches.

Numerics (default builds): fixed-order f32 math (``fma`` chains in element order, DPP butterflies, fixed-order
partial sums); a ``dbg`` build also writes the unit's transcendental results (``v_rsq`` / ``v_exp`` / ``v_rcp``)
so a host emulation of the same order (``gdn_recur_ref.emulate_lb``) can be bitwise. Mutants (must fail):
``nodecay``, ``store_old``, ``rms_own``, ``pipe``, ``nowar`` (with ``stress``).

Python engine only by policy (``buffer_*_cp``, ``int_to_ptr``); kernel names start with ``q38lb_`` (the family
README's cpp-parity exemption). Nothing here is imported by a served build.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from rocke.core.ir import (
    BF16,
    CPOL_NT,
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = [
    "DBG_W",
    "DK",
    "DV",
    "EPS",
    "GRID",
    "THREADS",
    "TP1",
    "TP2",
    "FtKnobs",
    "RecKnobs",
    "RecShape",
    "StreamKnobs",
    "build_ft_kernel",
    "build_lb_kernel",
    "build_stream_kernel",
    "best_knobs",
    "lb_params",
    "ft_params",
    "stream_params",
]

DK = DV = 128
EPS = 1e-6
RSQ128 = 0x3DB504F3  # v_rsq_f32(128.0), the mono's / ATOM's q scale bits
LOG2E = 1.4426950408889634
GRID, THREADS, WAVES = 256, 512, 8
TAB_COLS = 4
REC_IN, REC_OUT = 2, 3
DBG_W = 136  # f32 per (row, head) of a dbg build: inv_q, inv_k, decay, inv_rms, beta, pad, sig(z)[128] at 8
MAXIT = 1 << 22  # bounded LDS poll (x s_sleep 1)


@dataclass(frozen=True)
class RecShape:
    """Per-rank head geometry: TP1 48 value / 16 key heads, TP2 24 / 8."""

    hv: int = 48
    hk: int = 16

    def __post_init__(self) -> None:
        if self.hv % self.hk or self.hv <= 0:
            raise ValueError("hv must be a positive multiple of hk")

    @property
    def gqa(self) -> int:
        return self.hv // self.hk

    @property
    def key(self) -> int:
        return self.hk * DK

    @property
    def val(self) -> int:
        return self.hv * DV

    @property
    def row_elems(self) -> int:
        """bf16 elements per qkvz row: q [key], k [key], v [val], z [val] (ATOM's B1 output order)."""
        return 2 * self.key + 2 * self.val

    @property
    def q_off(self) -> int:
        return 0

    @property
    def k_off(self) -> int:
        return self.key

    @property
    def v_off(self) -> int:
        return 2 * self.key

    @property
    def z_off(self) -> int:
        return 2 * self.key + self.val

    @property
    def name(self) -> str:
        return f"h{self.hv}k{self.hk}"


TP1 = RecShape(48, 16)
TP2 = RecShape(24, 8)

MUTANTS = ("none", "nodecay", "store_old", "rms_own", "pipe")


def _suffix(obj, defaults) -> str:
    """Knob suffix: only the fields that differ from the defaults, in field order."""
    parts = []
    for f in fields(obj):
        v, d = getattr(obj, f.name), getattr(defaults, f.name)
        if v == d or f.name == "tag":
            continue
        if isinstance(v, bool):
            parts.append(f.name if v else f"no{f.name}")
        else:
            parts.append(f"{f.name}{v}")
    tag = getattr(obj, "tag", "")
    if tag:
        parts.append(tag)
    return "_".join(parts)


@dataclass(frozen=True)
class RecKnobs:
    """Knobs of the LB recurrence. Defaults = the P-L1 default arm; the kernel name lists only deviations."""

    waves: int = 8  # waves per unit (8: one unit per CTA at a time)
    lanes: int = 4  # lanes per v-row
    depth: int = 2  # work items (chunks) in flight per wave, incl. the one computed
    st_nt: bool = True  # nt bit on the state stores (single-use stream)
    ld_nt: bool = False  # nt bit on the state loads
    order: str = "row"  # unit order: "row" (row-major (row, head)) or "head" (head-major)
    state: str = "f32"  # "f32" or "bf16" storage (f32 math)
    chain: int = 1  # rows per state read (MTP verify chain)
    snap: str = "all"  # chain snapshots: "all" rows or the "last" one
    ring: int = 2  # LDS ring entries per unit slot (waves > 1)
    map: str = "lane"  # "lane": lanes-per-v-row interleaved 16-byte quads; "row": a wave access = whole v-rows;
    # "rowi": as "row", v-rows interleaved over the waves so one access of all 8 waves = 8 KiB contiguous
    ptr: str = "vec"  # row-table pointers: "vec" (vector loads 2 items ahead) | "smem" (scalar loads 1 ahead)
    stagger: bool = False  # rotate each CTA's wave -> v-row block map by the CTA id (same math, same bits)
    abl: str = "none"  # diagnostic ablation, timing only (wrong results): "copy" (state through, no math), "noepi"
    dbg: bool = False  # write the transcendental intermediates (bitwise emulation)
    stamp: bool = False  # per-CTA s_memrealtime start / end (informational spans; never timed)
    stress: int = 0  # numerics only: the finisher of every 3rd unit sleeps stress x 127 x 64 clocks
    nowar: bool = False  # mutant: no ring WAR poll (fails under stress)
    mutant: str = "none"
    tag: str = ""  # recompiled-control name suffix

    def check(self) -> None:
        if self.waves not in (1, 2, 4, 8):
            raise ValueError("waves in 1, 2, 4, 8")
        if self.lanes not in (2, 4, 8):
            raise ValueError("lanes in 2, 4, 8")
        if not 1 <= self.depth <= 4:
            raise ValueError("depth in 1..4")
        if self.order not in ("row", "head"):
            raise ValueError("order row | head")
        if self.state not in ("f32", "bf16"):
            raise ValueError("state f32 | bf16")
        if not 1 <= self.chain <= 4:
            raise ValueError("chain in 1..4")
        if self.snap not in ("all", "last"):
            raise ValueError("snap all | last")
        if not 1 <= self.ring <= 8:
            raise ValueError("ring in 1..8")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.mutant == "pipe" and self.depth < 2:
            raise ValueError("the pipe mutant needs depth >= 2")
        if self.mutant == "rms_own" and self.waves == 1:
            raise ValueError("rms_own needs waves > 1")
        if (self.nowar or self.stress) and self.waves == 1:
            raise ValueError("nowar / stress need waves > 1")
        if self.map not in ("lane", "row", "rowi"):
            raise ValueError("map lane | row | rowi")
        if self.map == "rowi" and self.waves != 8:
            raise ValueError("map=rowi interleaves the v-rows of one unit over the CTA's 8 waves: waves=8")
        if self.ptr not in ("vec", "smem"):
            raise ValueError("ptr vec | smem")
        if self.map != "lane" and self.lanes != 4:
            raise ValueError("map=row sets its own lanes per v-row (128 B / 16 B accesses): leave lanes at 4")
        if self.abl not in ("none", "copy", "noepi"):
            raise ValueError("abl none | copy | noepi")
        if self.abl != "none" and (self.dbg or self.mutant != "none" or self.chain > 1):
            raise ValueError("abl is a timing-only diagnostic of the chain-1 build")
        if self.chunks < 1:
            raise ValueError(f"waves {self.waves} x lanes {self.lanes}: a wave owns fewer v-rows than one chunk")

    # derived geometry
    @property
    def es(self) -> int:
        return 4 if self.state == "f32" else 2

    @property
    def ve(self) -> int:  # elements per 16-byte access
        return 16 // self.es

    @property
    def lr(self) -> int:  # lanes per v-row
        return DK // self.ve if self.map != "lane" else self.lanes

    @property
    def vpi(self) -> int:  # map=row: v-rows per wave access
        return 64 // self.lr

    @property
    def per_lane(self) -> int:  # map=lane: elements of one v-row per lane
        return DK // self.lanes

    @property
    def nq(self) -> int:  # 16-byte accesses per lane per chunk
        return 16 // self.vpi if self.map != "lane" else self.per_lane // self.ve

    @property
    def chunk_rows(self) -> int:  # v-rows per wave per chunk
        return 16 if self.map != "lane" else 64 // self.lanes

    @property
    def chunks(self) -> int:  # chunks per wave per unit
        return (DV // self.waves) // self.chunk_rows

    @property
    def slots(self) -> int:  # unit slots per CTA
        return WAVES // self.waves

    def suffix(self) -> str:
        return _suffix(self, RecKnobs())


@dataclass(frozen=True)
class FtKnobs:
    """FlyTape's ``gdn_recur`` as rocKE (256 threads, [k, v] state, 12 barriers per unit)."""

    stamp: bool = False
    tag: str = ""

    def suffix(self) -> str:
        return _suffix(self, FtKnobs())


@dataclass(frozen=True)
class StreamKnobs:
    """The bytes floor: read + write the state bytes, no math."""

    depth: int = 2
    st_nt: bool = True
    ld_nt: bool = False
    state: str = "f32"
    chain: int = 1
    snap: str = "all"
    grid: int = 256  # CTAs (256: one per CU, LDS pad; else no pad)
    ptr: str = "vec"  # row-table pointers: "vec" (vector loads 2 items ahead) | "smem" (scalar loads)
    amap: str = "cta"  # access i of the CTA covers one contiguous block ("cta") | wave w owns sub-block w ("wave")
    threads: int = 512  # threads per CTA (256: FlyTape's CTA size)
    stagger: bool = False  # rotate the access order (amap=cta) / the wave -> sub-block map (wave) by the CTA id
    stamp: bool = False
    tag: str = ""

    def check(self) -> None:
        if not 1 <= self.depth <= 4:
            raise ValueError("depth in 1..4")
        if self.state not in ("f32", "bf16"):
            raise ValueError("state f32 | bf16")
        if not 1 <= self.chain <= 4 or self.snap not in ("all", "last"):
            raise ValueError("chain 1..4, snap all | last")
        if self.grid not in (256, 512, 1024):
            raise ValueError("grid in 256, 512, 1024")
        if self.ptr not in ("vec", "smem"):
            raise ValueError("ptr vec | smem")
        if self.amap not in ("cta", "wave"):
            raise ValueError("amap cta | wave")
        if self.threads not in (256, 512):
            raise ValueError("threads 256 | 512")

    def suffix(self) -> str:
        return _suffix(self, StreamKnobs())


# ------------------------------------------------------------------ shared helpers


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _f(b: IRBuilder, v: float) -> Value:
    return b.const_f32(float(v))


def _fbits(b: IRBuilder, bits: int) -> Value:
    return b.bitcast(_c(b, bits), F32)


def _shfl(b: IRBuilder, v: Value, off: int):
    """The butterfly partner of f32 ``v`` at distance ``off``, all on the VALU (no ``ds_swizzle``): 1, 2
    ``quad_perm``; 4 ``row_half_mirror``; 8 ``row_ror:8``; 16 / 32 ``permlane16_swap`` / ``permlane32_swap``
    (these two return the (lo, hi) pair, whose sum is the pair sum). ``row_half_mirror`` pairs lane i with
    7 - i, not i ^ 4: a valid butterfly stage only once the stages below it are done (every lane of a quad
    holds the same value), which holds for every butterfly here (stages applied in increasing order, or
    starting at the group width on values replicated within each group)."""
    x = b.bitcast(v, I32)
    if off in (1, 2, 8):
        r = b.dpp_xor(x, off)
    elif off == 4:
        r = b.dpp_row_mirror(x, True)
    elif off == 16:
        lo, hi = b.permlane16_swap(x, x)
        return b.bitcast(lo, F32), b.bitcast(hi, F32)
    elif off == 32:
        lo, hi = b.permlane32_swap(x, x)
        return b.bitcast(lo, F32), b.bitcast(hi, F32)
    else:
        raise ValueError(off)
    return b.bitcast(r, F32)


def _xadd(b: IRBuilder, v: Value, off: int) -> Value:
    """``v + v[lane ^ off]`` (commutative: every lane of a pair holds the same bits)."""
    p = _shfl(b, v, off)
    if isinstance(p, tuple):
        return b.fadd(p[0], p[1])
    return b.fadd(v, p)


def _butterfly(b: IRBuilder, v: Value, offs) -> Value:
    for off in offs:
        v = _xadd(b, v, off)
    return v


def _ptr(b: IRBuilder, addr: Value) -> Value:
    return b.int_to_ptr(addr, PtrType(I32, "global"))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _pad_lds(b: IRBuilder) -> tuple[Value, Value]:
    """The CTA's LDS arena (more than half a CU's LDS: one CTA per CU) and its byte address."""
    lds = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="gdnr_lds")
    a0 = b.trunc(b.smem_addr_of(lds), I32)
    return lds, a0


def _touch_lds(b: IRBuilder, a0: Value) -> None:
    """An asm write to the last word of the arena: keeps the whole allocation (P4 lesson)."""
    with b.scf_if(D.is_thread0(b)):
        b.inline_asm(
            "ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
            "v,v,~{memory}",
            [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), _c(b, 0)],
            None,
        )


def _lds_wr(b: IRBuilder, addr: Value, v: Value) -> None:
    """Ordered ``ds_write_b32`` (all earlier LDS ops of the wave complete first, this one completes)."""
    b.inline_asm(
        "s_waitcnt lgkmcnt(0)\nds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
        "v,v,~{memory}",
        [addr, b.bitcast(v, I32) if v.type == F32 else v],
        None,
    )


def _lds_add_rtn(b: IRBuilder, addr: Value, v: Value) -> Value:
    """Ordered ``ds_add_rtn_u32``: earlier LDS ops of the wave complete first; returns the old value."""
    return b.inline_asm(
        "s_waitcnt lgkmcnt(0)\nds_add_rtn_u32 $0, $1, $2\ns_waitcnt lgkmcnt(0)",
        "=v,v,v,~{memory}",
        [addr, v],
        I32,
        result_name_hint="ldsadd",
    )


def _lds_rd(b: IRBuilder, addr: Value) -> Value:
    """Ordered ``ds_read_b32`` (f32)."""
    r = b.inline_asm(
        "ds_read_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
        "=v,v,~{memory}",
        [addr],
        I32,
        result_name_hint="ldsrd",
    )
    return b.bitcast(r, F32)


def _lds_poll_eq(b: IRBuilder, addr: Value, expected: Value) -> Value:
    """Spin (``s_sleep 1``) until the LDS word at ``addr`` equals ``expected`` or ``MAXIT`` retries; returns
    the retry count."""
    top = "${:private}gdnw${:uid}_top"
    done = "${:private}gdnw${:uid}_done"
    lines = [
        "s_mov_b32 $2, 0",
        f"{top}:",
        "ds_read_b32 $0, $3",
        "s_waitcnt lgkmcnt(0)",
        "v_cmp_ne_u32_e64 $1, $0, $4",
        "s_and_b64 $1, $1, exec",
        f"s_cbranch_scc0 {done}",
        "s_add_u32 $2, $2, 1",
        "s_cmp_ge_u32 $2, $5",
        f"s_cbranch_scc1 {done}",
        "s_sleep 1",
        f"s_branch {top}",
        f"{done}:",
    ]
    _v, _m, it = b.inline_asm_multi(
        "\n".join(lines),
        "=&v,=&s,=&s,v,v,s,~{vcc},~{scc},~{memory}",
        [addr, expected, _c(b, MAXIT)],
        result_types=[I32, I64, I32],
        result_name_hint="gdnw",
    )
    return it


def _stamp(b: IRBuilder, stamps_a: Value, cta: Value, col: int) -> None:
    with b.scf_if(D.is_thread0(b)):
        p = b.int_to_ptr(stamps_a, PtrType(I64, "global"))
        b.store(D.memrealtime(b), p, b.add(b.mul(cta, _c(b, 2)), _c(b, col)))


def _bf2(b: IRBuilder, w: Value) -> tuple[Value, Value]:
    return M.bf2_f32(b, w)


def _sigmoid(b: IRBuilder, z: Value) -> Value:
    """ATOM ``sigmoid_fast``: ``v_rcp(1 + v_exp((0 - z) log2 e))``."""
    e = b.exp2_fast(b.fmul(b.fsub(_f(b, 0.0), z), _f(b, LOG2E)))
    return b.rcp_fast(b.fadd(_f(b, 1.0), e))


def _udiv_const(b: IRBuilder, x: Value, d: int) -> Value:
    if d & (d - 1) == 0:
        return b.lshr(x, _c(b, d.bit_length() - 1))
    return b.div(x, _c(b, d))


def _umod_const(b: IRBuilder, x: Value, d: int) -> Value:
    if d & (d - 1) == 0:
        return b.land(x, _c(b, d - 1))
    return b.mod(x, _c(b, d))


# ------------------------------------------------------------------ the LB recurrence


def lb_params(k: RecKnobs) -> list[str]:
    """Kernel parameter order (i64 addresses, then i32): the host packs ``<Q...ii``."""
    p = ["tab", "qkvz", "gates", "normw", "out"]
    if k.dbg:
        p.append("dbg")
    if k.stamp:
        p.append("stamps")
    return p + ["rows", "layers"]


def lb_kernel_name(shape: RecShape, k: RecKnobs) -> str:
    s = k.suffix()
    return f"q38lb_gdnr_{shape.name}" + (f"_{s}" if s else "")


class _Item:
    """Uniform scalars of work item (chunk step) ``f`` of this wave."""

    def __init__(self, lb: "_Lb", f: Value):
        b, k, sh = lb.b, lb.k, lb.sh
        C = k.chunks
        self.f = f
        self.n = _udiv_const(b, f, C)
        self.c = _umod_const(b, f, C)
        idx = b.add(lb.slot, b.mul(self.n, _c(b, lb.NS)))
        self.valid = b.cmp_lt(idx, lb.total)
        idx = b.select(self.valid, idx, _c(b, 0))
        self.layer = b.div(idx, lb.U)
        u = b.sub(idx, b.mul(self.layer, lb.U))
        if k.order == "row":
            req = _udiv_const(b, u, sh.hv)
            self.h = b.sub(u, b.mul(req, _c(b, sh.hv)))
        else:
            self.h = b.div(u, lb.RQ)
            req = b.sub(u, b.mul(self.h, lb.RQ))
        self.kh = _udiv_const(b, self.h, sh.gqa)
        self.lr0 = b.add(b.mul(self.layer, lb.rows), b.mul(req, _c(b, k.chain)))  # flat (layer, row) of row 0
        # first v-row of this wave's chunk; map=lane: this lane's v-row in the chunk
        self.vbase = b.add(lb.wave_v0, b.mul(self.c, _c(b, k.chunk_rows)))
        self.vrow = b.add(self.vbase, lb.grp) if k.map == "lane" else None


class _Lb:
    def __init__(self, shape: RecShape, k: RecKnobs):
        k.check()
        self.sh, self.k = shape, k
        self.b = b = IRBuilder(lb_kernel_name(shape, k))
        b.kernel.attrs["max_workgroup_size"] = THREADS
        a = {n: b.param(n, I64) for n in lb_params(k)[:-2]}
        self.a = a
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.layers = b.readfirstlane(b.param("layers", I32))
        self.cta = b.readfirstlane(b.block_id_x())
        if k.stamp:
            _stamp(b, a["stamps"], self.cta, 0)
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        W, G = k.waves, k.chain
        # logical v-row block of this wave (stagger: rotated by the CTA id; the RMS partials are indexed by it,
        # so the finisher's fixed-order sum and every result bit are unchanged)
        self.ws = _umod_const(b, b.add(self.wave, self.cta) if k.stagger else self.wave, W)
        self.slot_local = _udiv_const(b, self.wave, W)
        self.slot = b.add(b.mul(self.cta, _c(b, k.slots)), self.slot_local)
        self.NS = GRID * k.slots
        self.RQ = _udiv_const(b, self.rows, G) if G > 1 else self.rows
        self.U = b.mul(self.RQ, _c(b, shape.hv))
        self.total = b.mul(self.U, self.layers)
        LR = k.lr
        self.grp = _udiv_const(b, self.lane, LR)  # map=lane: v-row group; map=row: vsub
        self.j = _umod_const(b, self.lane, LR)  # map=lane: lane in group; map=row: the lane's 16-byte quad
        self.wave_v0 = b.mul(self.ws, _c(b, DV // W))
        self.HB = DV * DK * k.es  # state bytes per head
        self.VB = DK * k.es  # per v-row
        # LDS layout (f32 words): per-wave q/k staging (map=lane), then the ring
        self.lds, self.a0 = _pad_lds(b)
        self.QKW = 2 * DK * G if k.map == "lane" else 0
        self.ring0 = WAVES * self.QKW
        self.PARTW = ((G * W + 3) // 4) * 4
        self.EW = G * DV + self.PARTW + 4  # entry: o[G][128], part[G][W], cnt, done, pad
        need = self.ring0 + k.slots * k.ring * self.EW
        if need > ONE_CTA_PER_CU_WORDS:
            raise ValueError(f"LDS words {need} > {ONE_CTA_PER_CU_WORDS}")
        _touch_lds(b, self.a0)
        self.qkbase = b.mul(self.wave, _c(b, self.QKW))
        # norm weight pair of this lane (epilogue lanes cover v-rows 2l, 2l+1)
        wrs = _rsrc(b, a["normw"], _c(b, DV * 2))
        self.nw = _bf2(b, b.buffer_load(wrs, b.mul(self.lane, _c(b, 4)), _c(b, 0), I32))
        b.s_waitcnt(vmcnt=0)  # once per launch: no pre-loop load left pending into the pipelined loop
        self.trs = _rsrc(b, a["tab"], _c(b, 0x7FFFFFF0))

    # -- LDS word addresses of the ring
    def entry_word(self, e: Value) -> Value:
        b = self.b
        return b.add(
            _c(self.b, self.ring0),
            b.mul(b.add(b.mul(self.slot_local, _c(b, self.k.ring)), e), _c(b, self.EW)),
        )

    def byte(self, word: Value) -> Value:
        return self.b.add(self.a0, self.b.mul(word, _c(self.b, 4)))

    def snap_rows(self) -> list[int]:
        G = self.k.chain
        return list(range(G)) if self.k.snap == "all" else [G - 1]

    def tab_ptrs(self, it: _Item) -> list[Value]:
        """``ptr="smem"``: rec_in of row 0 and rec_out of every snapshot row (scalar invariant loads)."""
        b = self.b
        tbl = b.int_to_ptr(self.a["tab"], PtrType(I64, "constant"))
        base = b.mul(it.lr0, _c(b, TAB_COLS))
        out = [b.global_load_i64(tbl, b.add(base, _c(b, REC_IN)))]
        for t in self.snap_rows():
            out.append(b.global_load_i64(tbl, b.add(base, _c(b, TAB_COLS * t + REC_OUT))))
        return out

    # -- the pointer pipeline: VECTOR loads of the row table, two items ahead of their use (no scalar loads in
    # the loop: an outstanding s_load makes every LDS wait an lgkmcnt(0) on it)
    def ptr_load(self, it: _Item) -> list[Value]:
        """Raw (rec_in, rec_out of row 0) as one dwordx4, plus rec_out of the other snapshot rows."""
        b = self.b
        base = b.mul(it.lr0, _c(b, TAB_COLS * 8))
        out = [b.buffer_load_vN(self.trs, b.add(base, _c(b, REC_IN * 8)), _c(b, 0), I32, 4)]
        for t in self.snap_rows():
            if t == 0:
                continue
            out.append(b.buffer_load_vN(self.trs, b.add(base, _c(b, (TAB_COLS * t + REC_OUT) * 8)), _c(b, 0), I32, 2))
        return out

    def ptr_vals(self, raw: list[Value]) -> list[Value]:
        """(rec_in, rec_out of each snapshot row) as uniform i64."""
        b = self.b

        def i64(lo, hi):
            return b.lor(b.zext(b.readfirstlane(lo), I64), b.shl(b.zext(b.readfirstlane(hi), I64), b.const_i64(32)))

        v4 = raw[0]
        vals = [i64(b.vec_extract(v4, 0), b.vec_extract(v4, 1))]
        rest = iter(raw[1:])
        for t in self.snap_rows():
            if t == 0:
                vals.append(i64(b.vec_extract(v4, 2), b.vec_extract(v4, 3)))
            else:
                v2 = next(rest)
                vals.append(i64(b.vec_extract(v2, 0), b.vec_extract(v2, 1)))
        return vals

    def nsmall(self) -> int:
        return {"lane": 4, "row": 6, "rowi": 5}[self.k.map]

    def npacket(self) -> int:
        k = self.k
        return k.nq + self.nsmall() * k.chain + len(self.snap_rows())

    def issue(self, it: _Item, ptrs: list[Value]) -> list[Value]:
        """Loads of work item ``it``: the lane's state chunk, then per chain row t the small inputs (map=lane:
        q|k one dwordx2 per lane, v of the lane's v-row; map=row: the lane's q and k quads, the chunk's 16 v),
        the z pair of the epilogue lane, (g, beta); then the snapshot rows' rec_out (uniform i64)."""
        b, k, sh = self.b, self.k, self.sh
        hv_bytes = sh.hv * self.HB
        srs = _rsrc(b, ptrs[0], b.select(it.valid, _c(b, hv_bytes), _c(b, 0)))
        vals = []
        aux = CPOL_NT if k.ld_nt else 0
        for i in range(k.nq):
            vals.append(b.buffer_load_cp(srs, self.state_off(it, i), _c(b, 0), dwords=4, aux=aux))
        RS = sh.row_elems * 2
        for t in range(k.chain):
            row = b.add(it.lr0, _c(b, t))
            qa = b.add(self.a["qkvz"], b.mul(b.zext(row, I64), b.const_i64(RS)))
            qrs = _rsrc(b, qa, b.select(it.valid, _c(b, RS), _c(b, 0)))
            if k.map == "lane":
                hi = b.cmp_ge(self.lane, _c(b, 32))
                l32 = b.land(self.lane, _c(b, 31))
                qk_el = b.add(
                    b.add(b.select(hi, _c(b, sh.k_off), _c(b, sh.q_off)), b.mul(it.kh, _c(b, DK))),
                    b.mul(l32, _c(b, 4)),
                )
                vals.append(b.buffer_load_vN(qrs, b.mul(qk_el, _c(b, 2)), _c(b, 0), I32, 2))
                v_el = b.add(_c(b, sh.v_off), b.add(b.mul(it.h, _c(b, DV)), it.vrow))
                vals.append(b.buffer_load(qrs, b.mul(b.land(v_el, _c(b, -2)), _c(b, 2)), _c(b, 0), I32))
            else:
                nd = k.ve // 2  # dwords of the lane's ve bf16
                kq = b.mul(self.j, _c(b, k.ve))
                for off in (sh.q_off, sh.k_off):
                    el = b.add(_c(b, off), b.add(b.mul(it.kh, _c(b, DK)), kq))
                    vals.append(b.buffer_load_vN(qrs, b.mul(el, _c(b, 2)), _c(b, 0), I32, nd))
                if k.map == "row":
                    v_el = b.add(_c(b, sh.v_off), b.add(b.mul(it.h, _c(b, DV)), it.vbase))
                    for part in range(2):
                        vals.append(b.buffer_load_vN(qrs, b.add(b.mul(v_el, _c(b, 2)), _c(b, 16 * part)), _c(b, 0), I32, 4))
                else:
                    # rowi: lane i < nq loads the vpi v of access i (v-rows i*8*vpi + ws*vpi + 0..vpi-1)
                    li = b.smin(self.lane, _c(b, k.nq - 1))
                    v_el = b.add(
                        _c(b, sh.v_off),
                        b.add(b.mul(it.h, _c(b, DV)), b.add(b.mul(li, _c(b, 8 * k.vpi)), b.mul(self.ws, _c(b, k.vpi)))),
                    )
                    vals.append(b.buffer_load_vN(qrs, b.mul(v_el, _c(b, 2)), _c(b, 0), I32, k.vpi // 2))
            z_el = b.add(_c(b, sh.z_off), b.add(b.mul(it.h, _c(b, DV)), b.mul(self.lane, _c(b, 2))))
            vals.append(b.buffer_load(qrs, b.mul(z_el, _c(b, 2)), _c(b, 0), I32))
            ga = b.add(self.a["gates"], b.mul(b.zext(row, I64), b.const_i64(sh.hv * 8)))
            grs = _rsrc(b, ga, b.select(it.valid, _c(b, sh.hv * 8), _c(b, 0)))
            vals.append(b.buffer_load_vN(grs, b.mul(it.h, _c(b, 8)), _c(b, 0), F32, 2))
        return vals + list(ptrs[1:])

    def vrow_of(self, it: _Item, i: int) -> Value:
        """v-row of this lane's access ``i`` (map=row: v-row ``i * vpi + vsub`` of the chunk)."""
        b, k = self.b, self.k
        if k.map == "lane":
            return it.vrow
        if k.map == "rowi":
            return b.add(b.add(_c(b, i * 8 * k.vpi), b.mul(self.ws, _c(b, k.vpi))), self.grp)
        return b.add(it.vbase, b.add(_c(b, i * k.vpi), self.grp))

    def state_off(self, it: _Item, i: int) -> Value:
        b, k = self.b, self.k
        if k.map == "lane":
            kk = b.mul(b.add(self.j, _c(b, k.lanes * i)), _c(b, k.ve))  # first k of access i
        else:
            kk = b.mul(self.j, _c(b, k.ve))
        return b.add(
            b.add(b.mul(it.h, _c(b, self.HB)), b.mul(self.vrow_of(it, i), _c(b, self.VB))),
            b.mul(kk, _c(b, k.es)),
        )

    # -- compute
    def stage_qk(self, small: list[Value], t: int, dbg_rs, it: _Item) -> tuple[list, list]:
        """q-hat and k-hat at this lane's k positions, in the lane's element order.
        map=lane: lanes 0-31 hold q (4 each), 32-63 k; normalize (32-lane tree), stage in the wave's LDS, read
        back the lane's positions. map=row: the lane loaded its own q and k quads; a tree over the lanes of a
        v-row gives both sums, no LDS."""
        b, k = self.b, self.k
        rsq = _fbits(b, RSQ128)
        if k.map != "lane":
            xs = []
            for w in (small[0], small[1]):
                x = []
                for d in range(k.ve // 2):
                    lo, hi = _bf2(b, b.vec_extract(w, d) if k.ve > 2 else w)
                    x += [lo, hi]
                xs.append(x)
            sums = []
            for x in xs:
                s = b.fmul(x[0], x[0])
                for e in range(1, k.ve):
                    s = b.fma(x[e], x[e], s)
                sums.append(_butterfly(b, s, [1 << s_ for s_ in range(k.lr.bit_length() - 1)]))
            inv_q = b.fmul(b.rsqrt(b.fadd(sums[0], _f(b, EPS))), rsq)
            inv_k = b.rsqrt(b.fadd(sums[1], _f(b, EPS)))
            if dbg_rs is not None:
                with b.scf_if(b.land(self.dbg_lead(it), b.cmp_eq(self.lane, _c(b, 0)))):
                    b.buffer_store_f32(dbg_rs[t], _c(b, 0), _c(b, 0), inv_q)
                    b.buffer_store_f32(dbg_rs[t], _c(b, 4), _c(b, 0), inv_k)
            return [b.fmul(x, inv_q) for x in xs[0]], [b.fmul(x, inv_k) for x in xs[1]]
        qk = small[0]
        x = []
        for d in range(2):
            lo, hi = _bf2(b, b.vec_extract(qk, d))
            x += [lo, hi]
        s = b.fmul(x[0], x[0])
        for e in range(1, 4):
            s = b.fma(x[e], x[e], s)
        s = _butterfly(b, s, (1, 2, 4, 8, 16))
        inv = b.rsqrt(b.fadd(s, _f(b, EPS)))
        is_q = b.cmp_lt(self.lane, _c(b, 32))
        scale = b.select(is_q, b.fmul(inv, rsq), inv)
        if dbg_rs is not None:
            with b.scf_if(b.land(self.dbg_lead(it), b.cmp_eq(self.j, _c(b, 0)))):
                with b.scf_if(b.cmp_eq(b.land(self.lane, _c(b, 31)), _c(b, 0))):
                    # lane 0: inv_q, lane 32: inv_k
                    slot = b.select(is_q, _c(b, 0), _c(b, 1))
                    b.buffer_store_f32(dbg_rs[t], b.mul(slot, _c(b, 4)), _c(b, 0), scale)
        xh = [b.fmul(x[e], scale) for e in range(4)]
        base = b.add(self.qkbase, _c(b, 2 * DK * t))
        b.smem_store_vN(self.lds, [b.add(base, b.mul(self.lane, _c(b, 4)))], b.vec_pack(xh, F32), 4)
        qh, kh = [], []
        for i in range(k.nq):
            kk = b.mul(b.add(self.j, _c(b, k.lanes * i)), _c(b, k.ve))
            for part in range(k.ve // 4):
                for dst, off in ((qh, 0), (kh, DK)):
                    v = b.smem_load_vN(self.lds, b.add(base, b.add(kk, _c(b, off + 4 * part))), dtype=F32, n=4)
                    dst += [b.vec_extract(v, e) for e in range(4)]
        return qh, kh

    def dbg_lead(self, it: _Item) -> Value:
        """The wave that writes a unit's per-unit dbg words: sub-wave 0 at chunk 0, valid items only."""
        b = self.b
        return b.land(b.land(b.cmp_eq(self.ws, _c(b, 0)), b.cmp_eq(it.c, _c(b, 0))), it.valid)

    def unpack_state(self, vecs: list[Value]) -> list[Value]:
        b, k = self.b, self.k
        out = []
        for v in vecs:
            for d in range(4):
                w = b.vec_extract(v, d)
                if k.state == "f32":
                    out.append(b.bitcast(w, F32))
                else:
                    lo, hi = _bf2(b, w)
                    out += [lo, hi]
        return out

    def pack_state(self, vals: list[Value]) -> list[Value]:
        b, k = self.b, self.k
        out = []
        if k.state == "f32":
            for i in range(0, len(vals), 4):
                out.append(b.vec_pack([b.bitcast(v, I32) for v in vals[i : i + 4]], I32))
        else:
            for i in range(0, len(vals), 8):
                ws = [M.bf16_pair(b, vals[i + 2 * d], vals[i + 2 * d + 1]) for d in range(4)]
                out.append(b.vec_pack(ws, I32))
        return out

    def emit(self) -> KernelDef:
        b, k, sh = self.b, self.k, self.sh
        W, G, C, D_ = k.waves, k.chain, k.chunks, k.depth
        # ring init + the launch's one barrier
        if W > 1:
            with b.scf_if(b.cmp_lt(self.tid, _c(b, k.slots * k.ring))):
                sl = _udiv_const(b, self.tid, k.ring)
                e = _umod_const(b, self.tid, k.ring)
                ew = b.add(
                    _c(b, self.ring0), b.mul(b.add(b.mul(sl, _c(b, k.ring)), e), _c(b, self.EW))
                )
                cw = b.add(ew, _c(b, G * DV + self.PARTW))
                _lds_wr(b, self.byte(cw), _c(b, 0))
                _lds_wr(b, self.byte(b.add(cw, _c(b, 1))), b.sub(e, _c(b, k.ring)))
            b.s_barrier_bare(fence="compiler")
        n_items = b.div(b.add(self.total, _c(b, self.NS - 1)), _c(b, self.NS))
        # chunk steps, padded to a multiple of the depth (the loop body is unrolled depth times so every
        # packet keeps its registers: no loop-carried copies of in-flight loads, which would wait for them)
        NF = b.mul(n_items, _c(b, C))
        NFp = b.mul(b.div(b.add(NF, _c(b, D_ - 1)), _c(b, D_)), _c(b, D_))
        # Steady state, chunk step s: [pointer loads of item s+D+1] [packet of item s+D-1, from item s+D-1's
        # pointers] [compute s] [stores s]. Prologue: pointers of items 0, 1 (waited), then the virtual steps
        # s = -(D-1) .. -1 with the stores replaced by as many stores to a 0-byte resource, so the loop entry
        # has the back edge's vmcnt history (LLVM's waits stay exact).
        smem = k.ptr == "smem"
        if smem:
            ptrq = [self.tab_ptrs(_Item(self, _c(b, D_ - 1)))]
        else:
            ptrq = [self.ptr_load(_Item(self, _c(b, 0))), self.ptr_load(_Item(self, _c(b, 1)))]
            b.s_waitcnt(vmcnt=0)
        pks = []
        for s in range(-(D_ - 1), 0):
            if smem:
                it0 = _Item(self, _c(b, s + D_ - 1))
                pk = self.issue(it0, self.tab_ptrs(it0))
            else:
                raw = self.ptr_load(_Item(self, _c(b, s + D_ + 1)))
                pk = self.issue(_Item(self, _c(b, s + D_ - 1)), self.ptr_vals(ptrq[0]))
                ptrq = ptrq[1:] + [raw]
            b.sched_barrier(0)
            self.dummy_stores()
            b.sched_barrier(0)
            pks.append(pk)
        flat_ptrq = [v for r in ptrq for v in r]
        NRAW = len(ptrq[0])
        NQ_ = len(ptrq)
        carried = [v for p in pks for v in p] + flat_ptrq
        Pinit = [_f(b, 0.0) for _ in range(G)] if C > 1 else []
        args = [(f"gc{i}", v) for i, v in enumerate(carried + Pinit)]
        NPK = self.npacket()
        with b.scf_for_iter(_c(b, 0), NFp, _c(b, D_), args, iv_name="gf") as (f0, cv):
            pks = [list(cv[i * NPK : (i + 1) * NPK]) for i in range(D_ - 1)]
            o = (D_ - 1) * NPK
            ptrq = [list(cv[o + q * NRAW : o + (q + 1) * NRAW]) for q in range(NQ_)]
            Pacc = list(cv[o + NQ_ * NRAW :])
            for u in range(D_):
                f = b.add(f0, _c(b, u)) if u else f0
                if smem:
                    new_pk = self.issue(_Item(self, b.add(f, _c(b, D_ - 1))), ptrq[0])
                    ptrq = [self.tab_ptrs(_Item(self, b.add(f, _c(b, D_))))]
                else:
                    raw = self.ptr_load(_Item(self, b.add(f, _c(b, D_ + 1))))
                    new_pk = self.issue(_Item(self, b.add(f, _c(b, D_ - 1))), self.ptr_vals(ptrq[0]))
                    ptrq = ptrq[1:] + [raw]
                b.sched_barrier(0)
                cur = pks[0] if D_ > 1 else new_pk
                nxt_pk = (pks[1] if D_ > 2 else new_pk) if D_ > 1 else None
                Pacc = self.compute(_Item(self, f), cur, nxt_pk, Pacc)
                pks = pks[1:] + [new_pk] if D_ > 1 else []
            b.scf_yield(*([v for p in pks for v in p] + [v for r in ptrq for v in r] + Pacc))
        if k.stamp:
            b.s_barrier_bare(fence="compiler")
            _stamp(b, self.a["stamps"], self.cta, 1)
        b.ret()
        return b.kernel

    def compute(self, it: _Item, pk: list[Value], nxt_pk, Pacc: list[Value]) -> list[Value]:
        b, k, sh = self.b, self.k, self.sh
        W, G, C = k.waves, k.chain, k.chunks
        NS_ = self.nsmall()
        S = self.unpack_state(pk[: k.nq])
        small = [pk[k.nq + NS_ * t : k.nq + NS_ * (t + 1)] for t in range(G)]
        ptrs_out = pk[k.nq + NS_ * G :]
        if k.abl == "copy":
            # diagnostic: the loaded state goes back unchanged (every small load consumed through a 0 * x term
            # LLVM cannot fold, so the vector-memory stream is the full build's)
            x = _f(b, 0.0)
            for v in small[0]:
                w = b.vec_extract(v, 0) if isinstance(v.type, VectorType) else v
                x = b.fadd(x, b.bitcast(w, F32) if w.type == I32 else w)
            Sx = list(S)
            Sx[0] = b.fadd(S[0], b.fmul(_f(b, 0.0), x))
            ors = _rsrc(b, ptrs_out[0], b.select(it.valid, _c(b, sh.hv * self.HB), _c(b, 0)))
            for i, w in enumerate(self.pack_state(Sx)):
                b.buffer_store_cp(w, ors, self.state_off(it, i), _c(b, 0), aux=CPOL_NT if k.st_nt else 0)
            rs0 = _rsrc(b, self.a["out"], _c(b, 0))
            b.buffer_store_cp(_c(b, 0), rs0, _c(b, 0), _c(b, 0))  # the output store's slot in the vm history
            return Pacc
        dbg_rs = None
        if k.dbg:
            dbg_rs = []
            for t in range(G):
                ent = b.add(b.mul(b.add(it.lr0, _c(b, t)), _c(b, sh.hv)), it.h)
                da = b.add(self.a["dbg"], b.mul(b.zext(ent, I64), b.const_i64(DBG_W * 4)))
                dbg_rs.append(_rsrc(b, da, b.select(it.valid, _c(b, DBG_W * 4), _c(b, 0))))
        e_ring = _umod_const(b, it.n, k.ring)
        ew = self.entry_word(e_ring)
        if W > 1 and not k.nowar and k.abl == "none":
            with b.scf_if(b.cmp_eq(it.c, _c(b, 0))):
                _lds_poll_eq(b, self.byte(b.add(ew, _c(b, G * DV + self.PARTW + 1))), b.sub(it.n, _c(b, k.ring)))
        Pnew = []
        hv_bytes = sh.hv * self.HB
        red = [1 << s for s in range(k.lr.bit_length() - 1)]  # the tree over the lanes of a v-row
        for t in range(G):
            if t:
                b.sched_barrier(0)  # chain steps stay in order (else LLVM hoists every step's LDS reads: spills)
            sm = small[t]
            qh, khat = self.stage_qk(sm, t, dbg_rs, it)
            gb = sm[-1]
            g = b.vec_extract(gb, 0)
            beta = b.vec_extract(gb, 1)
            if k.mutant == "pipe":
                beta = b.vec_extract(nxt_pk[k.nq + NS_ * t + NS_ - 1], 1)
            decay = b.exp2_fast(b.fmul(g, _f(b, LOG2E)))
            if dbg_rs is not None:
                with b.scf_if(b.land(self.dbg_lead(it), b.cmp_eq(self.lane, _c(b, 0)))):
                    b.buffer_store_f32(dbg_rs[t], _c(b, 8), _c(b, 0), decay)
                    b.buffer_store_f32(dbg_rs[t], _c(b, 16), _c(b, 0), beta)
            # v of each v-row this lane holds
            if k.map == "lane":
                odd = b.cmp_ne(b.land(it.vrow, _c(b, 1)), _c(b, 0))
                vlo, vhi = _bf2(b, sm[1])
                vs = [b.select(odd, vhi, vlo)]
                groups = [list(range(len(S)))]  # one v-row: every element
            else:
                if k.map == "row":
                    v16 = []
                    for part in (sm[2], sm[3]):
                        for d in range(4):
                            lo, hi = _bf2(b, b.vec_extract(part, d))
                            v16 += [lo, hi]
                else:
                    # rowi: access i's vpi values from lane i (uniform), then this lane's by its vsub
                    v16 = []
                    for i in range(k.nq):
                        for d in range(k.vpi // 2):
                            w = b.vec_extract(sm[2], d)
                            lo, hi = _bf2(b, b.readlane(w, _c(b, i)))
                            v16 += [lo, hi]
                vs = []
                for i in range(k.nq):
                    cand = v16[i * k.vpi : (i + 1) * k.vpi]
                    v = cand[0]
                    for q in range(1, k.vpi):
                        v = b.select(b.cmp_eq(self.grp, _c(b, q)), cand[q], v)
                    vs.append(v)
                groups = [list(range(i * k.ve, (i + 1) * k.ve)) for i in range(k.nq)]
            sv = S if k.mutant == "nodecay" else [b.fmul(s, decay) for s in S]
            Sn = [None] * len(S)
            os_ = []
            for gi, idx in enumerate(groups):
                # map=lane: element m of the chain multiplies k-hat m; map=row: element e of access i
                kk = [khat[m] if k.map == "lane" else khat[m - idx[0]] for m in idx]
                qq = [qh[m] if k.map == "lane" else qh[m - idx[0]] for m in idx]
                kv = b.fmul(sv[idx[0]], kk[0])
                for e in range(1, len(idx)):
                    kv = b.fma(sv[idx[e]], kk[e], kv)
                kv = _butterfly(b, kv, red)
                delta = b.fmul(beta, b.fsub(vs[gi], kv))
                for e, m in enumerate(idx):
                    Sn[m] = b.fma(delta, kk[e], sv[m])
                o = b.fmul(Sn[idx[0]], qq[0])
                for e in range(1, len(idx)):
                    o = b.fma(Sn[idx[e]], qq[e], o)
                os_.append(_butterfly(b, o, red))
            # state snapshot store
            if k.snap == "all" or t == G - 1:
                pi = self.snap_rows().index(t)
                ors = _rsrc(b, ptrs_out[pi], b.select(it.valid, _c(b, hv_bytes), _c(b, 0)))
                vals = Sn
                if k.mutant == "store_old":
                    vals = [b.select(b.cmp_eq(self.ws, _c(b, 0)), S[e], Sn[e]) for e in range(len(Sn))]
                for i, w in enumerate(self.pack_state(vals)):
                    b.buffer_store_cp(w, ors, self.state_off(it, i), _c(b, 0), aux=CPOL_NT if k.st_nt else 0)
            S = Sn
            if k.abl == "noepi":
                continue
            # o to the ring entry, the chunk's partial sum of squares
            if k.map == "lane":
                o = os_[0]
                b.smem_store_vN(self.lds, [b.add(ew, b.add(_c(b, t * DV), it.vrow))], o, 1)
                sq = b.fmul(o, o)
                sq = _butterfly(b, sq, [k.lr << s for s in range(6 - (k.lr.bit_length() - 1))])
            else:
                # lane (quad j, vsub) writes the o of access min(j, nq - 1): v-row min(j, nq-1) * vpi + vsub
                isel = b.smin(self.j, _c(b, k.nq - 1))
                osel = os_[0]
                for i in range(1, k.nq):
                    osel = b.select(b.cmp_eq(isel, _c(b, i)), os_[i], osel)
                if k.map == "rowi":
                    vr = b.add(b.add(b.mul(isel, _c(b, 8 * k.vpi)), b.mul(self.ws, _c(b, k.vpi))), self.grp)
                else:
                    vr = b.add(it.vbase, b.add(b.mul(isel, _c(b, k.vpi)), self.grp))
                b.smem_store_vN(self.lds, [b.add(ew, b.add(_c(b, t * DV), vr))], osel, 1)
                sq = b.fmul(os_[0], os_[0])
                for i in range(1, k.nq):
                    sq = b.fma(os_[i], os_[i], sq)
                sq = _butterfly(b, sq, [k.lr << s for s in range(6 - (k.lr.bit_length() - 1))])
            if C > 1:
                sq = b.select(b.cmp_eq(it.c, _c(b, 0)), sq, b.fadd(Pacc[t], sq))
            Pnew.append(sq)
        if k.abl == "noepi":
            rs0 = _rsrc(b, self.a["out"], _c(b, 0))
            for t in range(G):
                b.buffer_store_cp(_c(b, 0), rs0, _c(b, 4 * t), _c(b, 0))  # the output stores' slots
            return Pacc
        self.epilogue(it, ew, Pnew, small, dbg_rs)
        return Pnew if C > 1 else []

    def epilogue(self, it: _Item, ew: Value, P: list[Value], small, dbg_rs) -> None:
        """Every chunk step of every wave runs the same vector-memory instructions (the output stores go to a
        0-byte resource unless this wave finishes the unit at this step), so every path through the loop has
        the same vmcnt history and LLVM's waits stay exact. LDS: partials, then the arrival (``+1`` at the
        unit's last chunk, ``+0`` before), the finisher = the last arriver."""
        b, k = self.b, self.k
        W, G, C = k.waves, k.chain, k.chunks
        last = b.cmp_eq(it.c, _c(b, C - 1))
        part0 = b.add(ew, _c(b, G * DV))
        cntw = b.add(part0, _c(b, self.PARTW))
        if W == 1:
            fin = last
            tot = P
        else:
            for t in range(G):
                _lds_wr(b, self.byte(b.add(part0, b.add(_c(b, t * W), self.ws))), P[t])
            # one arrival per wave: lane 0 adds (1 at the unit's last chunk), the other lanes add 0; lane 0's
            # return value is the count of waves that arrived before this one
            inc = b.select(b.land(last, b.cmp_eq(self.lane, _c(b, 0))), _c(b, 1), _c(b, 0))
            old = b.readfirstlane(_lds_add_rtn(b, self.byte(cntw), inc))
            fin = b.land(last, b.cmp_eq(old, _c(b, W - 1)))
            with b.scf_if(fin):
                _lds_wr(b, self.byte(cntw), _c(b, 0))
                if k.stress:
                    with b.scf_if(b.cmp_eq(_umod_const(b, it.n, 3), _c(b, 0))):
                        self.uid = getattr(self, "uid", 0) + 1
                        with b.scf_for(_c(b, 0), _c(b, k.stress), _c(b, 1), iv_name=f"gst{self.uid}"):
                            D.s_sleep(b, 127)
            if k.mutant == "rms_own":
                tot = P
            else:
                tot = []
                for t in range(G):
                    s = _lds_rd(b, self.byte(b.add(part0, _c(b, t * W))))
                    for w in range(1, W):
                        s = b.fadd(s, _lds_rd(b, self.byte(b.add(part0, _c(b, t * W + w)))))
                    tot.append(s)
        self.finish(it, ew, tot, small, dbg_rs, fin)
        if W > 1:
            with b.scf_if(fin):
                _lds_wr(b, self.byte(b.add(cntw, _c(b, 1))), it.n)

    def finish(self, it: _Item, ew: Value, tot: list[Value], small, dbg_rs, fin: Value) -> None:
        """The unit's gated RMSNorm and bf16 output (lane l: v-rows 2l, 2l+1); stores land only for the
        finisher of a valid unit."""
        b, k, sh = self.b, self.k, self.sh
        live = b.land(fin, it.valid)
        for t in range(k.chain):
            inv = b.rsqrt(b.fadd(b.fmul(tot[t], _f(b, 1.0 / DV)), _f(b, EPS)))
            ov = b.smem_load_vN(self.lds, b.add(ew, b.add(_c(b, t * DV), b.mul(self.lane, _c(b, 2)))), dtype=F32, n=2)
            z = _bf2(b, small[t][-2])
            ys, sig = [], []
            for e in range(2):
                sg = _sigmoid(b, z[e])
                sig.append(sg)
                ys.append(b.fmul(b.fmul(b.fmul(b.vec_extract(ov, e), inv), self.nw[e]), sg))
            row = b.add(it.lr0, _c(b, t))
            oa = b.add(self.a["out"], b.mul(b.zext(row, I64), b.const_i64(sh.val * 2)))
            ors = _rsrc(b, oa, b.select(live, _c(b, sh.val * 2), _c(b, 0)))
            off = b.mul(b.add(b.mul(it.h, _c(b, DV)), b.mul(self.lane, _c(b, 2))), _c(b, 2))
            b.buffer_store_cp(M.bf16_pair(b, ys[0], ys[1]), ors, off, _c(b, 0))
            if dbg_rs is not None:
                with b.scf_if(live):
                    with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
                        b.buffer_store_f32(dbg_rs[t], _c(b, 12), _c(b, 0), inv)
                    so = b.mul(b.add(_c(b, 8), b.mul(self.lane, _c(b, 2))), _c(b, 4))
                    b.buffer_store_f32(dbg_rs[t], so, _c(b, 0), sig[0])
                    b.buffer_store_f32(dbg_rs[t], b.add(so, _c(b, 4)), _c(b, 0), sig[1])

    def stores_per_item(self) -> int:
        return self.k.nq * len(self.snap_rows()) + self.k.chain

    def dummy_stores(self) -> None:
        """The vector-memory stores of one work item, to a 0-byte resource: the prologue issues them after
        each packet so the loop entry has the same vmcnt history as the loop back edge."""
        b = self.b
        rs = _rsrc(b, self.a["out"], _c(b, 0))
        z4 = b.vec_pack([_c(b, 0)] * 4, I32)
        n = self.k.nq * len(self.snap_rows())
        for i in range(n):  # distinct offsets: identical stores would be merged
            b.buffer_store_cp(z4, rs, _c(b, 16 * i), _c(b, 0))
        for t in range(self.k.chain):
            b.buffer_store_cp(_c(b, 0), rs, _c(b, 16 * (n + t)), _c(b, 0))


def build_lb_kernel(shape: RecShape = TP1, knobs: RecKnobs = RecKnobs()) -> KernelDef:  # noqa: B008
    return _Lb(shape, knobs).emit()


def best_knobs(shape: RecShape, rows: int, *, mode: str = "stream", state: str = "f32", chain: int = 1) -> RecKnobs:
    """P-L1's measured best knobs (``experiments/qwen38-true-mono/large-batch/P-L1.md``; the defaults of
    :class:`RecKnobs` stay DESIGN-LB 6.1's form, a losing lever kept for traceability). ``mode="stream"``: the
    recurrence runs as a stage of a persistent kernel (its items stream through without launch boundaries);
    ``mode="launch"``: one launch per layer (DESIGN-LB's ``layers=1`` bring-up), where at <= 3 units per CTA the
    pipeline prologue does not pay and depth 1 wins. ``chain=4``: MTP verify, one state read per request (lanes=8:
    the lane map at 4 rows spills; ``snap="last"`` needs ATOM's D2 state contract)."""
    if mode not in ("stream", "launch"):
        raise ValueError("mode stream | launch")
    if chain > 1:
        return RecKnobs(chain=chain, lanes=8, state=state)
    units_per_cta = -(-rows * shape.hv // GRID)
    depth = 1 if (mode == "launch" and units_per_cta <= 3) else 2
    return RecKnobs(map="rowi", ld_nt=True, depth=depth, state=state)


# ------------------------------------------------------------------ FlyTape's gdn_recur, re-implemented


def ft_params(k: FtKnobs) -> list[str]:
    p = ["tab", "qkvz", "gates", "normw", "out"]
    if k.stamp:
        p.append("stamps")
    return p + ["rows", "layers"]


def ft_kernel_name(shape: RecShape, k: FtKnobs) -> str:
    s = k.suffix()
    return f"q38lb_gdnft_{shape.name}" + (f"_{s}" if s else "")


def build_ft_kernel(shape: RecShape = TP1, knobs: FtKnobs = FtKnobs()) -> KernelDef:  # noqa: B008
    """FlyTape ``bodies.gdn_recur_with`` (one 256-thread CTA per CU): per (row, head) unit, q / k L2 norms by
    two block sums, decay, [k, v] state (thread: 4 v columns x 16 keys, 16-byte accesses), column sums over
    keys (shuffle + LDS, two barriers each), the gated RMSNorm by a block sum; f32 output. Inputs as the LB
    kernel's (bf16 qkvz, f32 (g, beta)); state per row pointer, ``[HV][DK][DV]`` (k-major) per layer."""
    T, WV = 256, 4
    quads, groups = DV // 4, T // (DV // 4)  # 32, 8
    kper = DK // groups  # 16
    sh, k = shape, knobs
    b = IRBuilder(ft_kernel_name(shape, knobs))
    b.kernel.attrs["max_workgroup_size"] = T
    a = {n: b.param(n, I64) for n in ft_params(k)[:-2]}
    rows = b.readfirstlane(b.param("rows", I32))
    layers = b.readfirstlane(b.param("layers", I32))
    cta = b.readfirstlane(b.block_id_x())
    if k.stamp:
        _stamp(b, a["stamps"], cta, 0)
    tid = b.thread_id_x()
    lane = b.lane_id()
    wave = D.wave_id(b)
    lds, a0 = _pad_lds(b)
    _touch_lds(b, a0)
    # LDS words: work [0, 2 dk) q-hat, k-hat; column sums [2 dk, 2 dk + 4 dk); block sums [6 dk, 6 dk + 4)
    COL0, BS0 = 2 * DK, 6 * DK
    v0 = b.mul(_umod_const(b, tid, quads), _c(b, 4))
    k0 = b.mul(_udiv_const(b, tid, quads), _c(b, kper))
    is_q = b.cmp_lt(tid, _c(b, DK))
    first = b.cmp_lt(tid, _c(b, quads))
    U = b.mul(rows, _c(b, sh.hv))
    total = b.mul(U, layers)
    n_items = b.div(b.add(total, _c(b, GRID - 1)), _c(b, GRID))
    tbl = b.int_to_ptr(a["tab"], PtrType(I64, "constant"))
    wrs = _rsrc(b, a["normw"], _c(b, DV * 2))

    def block_sum(v):
        v = _butterfly(b, v, (1, 2, 4, 8, 16, 32))
        b.sync_lds_only(fence="compiler")
        with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
            b.smem_store_vN(lds, [b.add(_c(b, BS0), wave)], v, 1)
        b.sync_lds_only(fence="compiler")
        t = b.vec_extract(b.smem_load_vN(lds, _c(b, BS0), dtype=F32, n=1), 0)
        for w in range(1, WV):
            t = b.fadd(t, b.vec_extract(b.smem_load_vN(lds, _c(b, BS0 + w), dtype=F32, n=1), 0))
        return t

    def column_sums(parts):
        parts = [_xadd(b, p, 32) for p in parts]
        for e in range(4):
            b.smem_store_vN(lds, [b.add(_c(b, COL0 + e), b.add(b.mul(wave, _c(b, DK)), v0))], parts[e], 1)
        b.sync_lds_only(fence="compiler")
        sums = []
        for e in range(4):
            t = b.vec_extract(b.smem_load_vN(lds, b.add(_c(b, COL0 + e), v0), dtype=F32, n=1), 0)
            for w in range(1, WV):
                t = b.fadd(
                    t, b.vec_extract(b.smem_load_vN(lds, b.add(_c(b, COL0 + w * DK + e), v0), dtype=F32, n=1), 0)
                )
            sums.append(t)
        b.sync_lds_only(fence="compiler")
        return sums

    with b.scf_for(_c(b, 0), n_items, _c(b, 1), iv_name="fu") as n:
        b.sync_lds_only(fence="compiler")
        idx = b.add(cta, b.mul(n, _c(b, GRID)))
        valid = b.cmp_lt(idx, total)
        idx = b.select(valid, idx, _c(b, 0))
        layer = b.div(idx, U)
        u = b.sub(idx, b.mul(layer, U))
        r = _udiv_const(b, u, sh.hv)
        h = b.sub(u, b.mul(r, _c(b, sh.hv)))
        kh = _udiv_const(b, h, sh.gqa)
        lr = b.add(b.mul(layer, rows), r)
        RS = sh.row_elems * 2
        qa = b.add(a["qkvz"], b.mul(b.zext(lr, I64), b.const_i64(RS)))
        qrs = _rsrc(b, qa, b.select(valid, _c(b, RS), _c(b, 0)))
        ga = b.add(a["gates"], b.mul(b.zext(lr, I64), b.const_i64(sh.hv * 8)))
        grs = _rsrc(b, ga, b.select(valid, _c(b, sh.hv * 8), _c(b, 0)))
        gb = b.buffer_load_vN(grs, b.mul(h, _c(b, 8)), _c(b, 0), F32, 2)
        g, beta = b.vec_extract(gb, 0), b.vec_extract(gb, 1)
        src = b.select(
            is_q,
            b.add(_c(b, sh.q_off), b.add(b.mul(kh, _c(b, DK)), tid)),
            b.add(_c(b, sh.k_off), b.add(b.mul(kh, _c(b, DK)), b.sub(tid, _c(b, DK)))),
        )
        val = b.cast_to_f32(b.buffer_load(qrs, b.mul(src, _c(b, 2)), _c(b, 0), BF16))
        sq = b.fmul(val, val)
        sq_q = block_sum(b.select(is_q, sq, _f(b, 0.0)))
        sq_k = block_sum(b.select(is_q, _f(b, 0.0), sq))
        scale = b.select(
            is_q,
            b.fmul(b.rsqrt(b.fadd(sq_q, _f(b, 1e-6))), _f(b, 1.0 / (DK**0.5))),
            b.rsqrt(b.fadd(sq_k, _f(b, 1e-6))),
        )
        b.smem_store_vN(lds, [tid], b.fmul(val, scale), 1)
        b.sync_lds_only(fence="compiler")
        decay = b.exp2_fast(b.fmul(g, _f(b, LOG2E)))
        vw = b.buffer_load_vN(qrs, b.mul(b.add(_c(b, sh.v_off), b.add(b.mul(h, _c(b, DV)), v0)), _c(b, 2)), _c(b, 0), I32, 2)
        vv = list(_bf2(b, b.vec_extract(vw, 0))) + list(_bf2(b, b.vec_extract(vw, 1)))
        qs = [b.vec_extract(b.smem_load_vN(lds, b.add(k0, _c(b, kk)), dtype=F32, n=1), 0) for kk in range(kper)]
        ks = [
            b.vec_extract(b.smem_load_vN(lds, b.add(k0, _c(b, DK + kk)), dtype=F32, n=1), 0) for kk in range(kper)
        ]
        rec_in = b.global_load_i64(tbl, b.add(b.mul(lr, _c(b, TAB_COLS)), _c(b, REC_IN)))
        rec_out = b.global_load_i64(tbl, b.add(b.mul(lr, _c(b, TAB_COLS)), _c(b, REC_OUT)))
        HBYTES = sh.hv * DK * DV * 4
        srs = _rsrc(b, rec_in, b.select(valid, _c(b, HBYTES), _c(b, 0)))
        drs = _rsrc(b, rec_out, b.select(valid, _c(b, HBYTES), _c(b, 0)))
        sbase = b.mul(b.add(b.mul(b.add(b.mul(h, _c(b, DK)), k0), _c(b, DV)), v0), _c(b, 4))
        state = []
        for kk in range(kper):
            q4 = b.buffer_load_vN(srs, b.add(sbase, _c(b, kk * DV * 4)), _c(b, 0), F32, 4)
            state.append([b.fmul(b.vec_extract(q4, e), decay) for e in range(4)])
        kv = [_f(b, 0.0)] * 4
        for kk in range(kper):
            kv = [b.fma(state[kk][e], ks[kk], kv[e]) for e in range(4)]
        kv = column_sums(kv)
        delta = [b.fmul(b.fsub(vv[e], kv[e]), beta) for e in range(4)]
        out = [_f(b, 0.0)] * 4
        for kk in range(kper):
            new = [b.fma(ks[kk], delta[e], state[kk][e]) for e in range(4)]
            out = [b.fma(new[e], qs[kk], out[e]) for e in range(4)]
            b.buffer_store_vN_f32(drs, b.add(sbase, _c(b, kk * DV * 4)), _c(b, 0), b.vec_pack(new, F32), 4)
        o = column_sums(out)
        osq = b.fmul(o[0], o[0])
        for e in range(1, 4):
            osq = b.fma(o[e], o[e], osq)
        osq = block_sum(b.select(first, osq, _f(b, 0.0)))
        rr = b.rsqrt(b.fadd(b.fmul(osq, _f(b, 1.0 / DK)), _f(b, EPS)))
        with b.scf_if(first):
            zw = b.buffer_load_vN(qrs, b.mul(b.add(_c(b, sh.z_off), b.add(b.mul(h, _c(b, DV)), v0)), _c(b, 2)), _c(b, 0), I32, 2)
            z = list(_bf2(b, b.vec_extract(zw, 0))) + list(_bf2(b, b.vec_extract(zw, 1)))
            nw = b.buffer_load_vN(wrs, b.mul(v0, _c(b, 2)), _c(b, 0), I32, 2)
            w = list(_bf2(b, b.vec_extract(nw, 0))) + list(_bf2(b, b.vec_extract(nw, 1)))
            res = [b.fmul(b.fmul(b.fmul(o[e], rr), w[e]), _sigmoid(b, z[e])) for e in range(4)]
            oa = b.add(a["out"], b.mul(b.zext(lr, I64), b.const_i64(sh.val * 4)))
            ors = _rsrc(b, oa, b.select(valid, _c(b, sh.val * 4), _c(b, 0)))
            b.buffer_store_vN_f32(ors, b.mul(b.add(b.mul(h, _c(b, DV)), v0), _c(b, 4)), _c(b, 0), b.vec_pack(res, F32), 4)
    if k.stamp:
        b.s_barrier_bare(fence="compiler")
        _stamp(b, a["stamps"], cta, 1)
    b.ret()
    return b.kernel


# ------------------------------------------------------------------ the bytes floor


def stream_params(k: StreamKnobs) -> list[str]:
    return ["tab"] + (["stamps"] if k.stamp else []) + ["rows", "layers"]


def stream_kernel_name(shape: RecShape, k: StreamKnobs) -> str:
    s = k.suffix()
    return f"q38lb_gdnst_{shape.name}" + (f"_{s}" if s else "")


def build_stream_kernel(shape: RecShape = TP1, knobs: StreamKnobs = StreamKnobs()) -> KernelDef:  # noqa: B008
    """Per unit (request, head): read the head's state (rec_in of the chain's row 0) and write it to the
    snapshot rows' rec_out, 16 bytes per lane per access, the CTA's 512 lanes covering 8 KiB contiguous per
    access, ``depth`` units in flight per CTA."""
    k = knobs
    k.check()
    sh = shape
    G = k.chain
    b = IRBuilder(stream_kernel_name(shape, k))
    TH = k.threads
    b.kernel.attrs["max_workgroup_size"] = TH
    a = {n: b.param(n, I64) for n in stream_params(k)[:-2]}
    rows = b.readfirstlane(b.param("rows", I32))
    layers = b.readfirstlane(b.param("layers", I32))
    cta = b.readfirstlane(b.block_id_x())
    if k.stamp:
        _stamp(b, a["stamps"], cta, 0)
    tid = b.thread_id_x()
    if k.grid == 256:
        _lds, a0 = _pad_lds(b)
        _touch_lds(b, a0)
    es = 4 if k.state == "f32" else 2
    HB = DK * DV * es
    NA = HB // (TH * 16)  # accesses per lane per unit
    NWV = TH // 64
    RQ = _udiv_const(b, rows, G) if G > 1 else rows
    U = b.mul(RQ, _c(b, sh.hv))
    total = b.mul(U, layers)
    NS = k.grid
    n_items = b.div(b.add(total, _c(b, NS - 1)), _c(b, NS))
    tbl = b.int_to_ptr(a["tab"], PtrType(I64, "constant"))
    snaps = list(range(G)) if k.snap == "all" else [G - 1]

    def item(nv):
        idx = b.add(cta, b.mul(nv, _c(b, NS)))
        valid = b.cmp_lt(idx, total)
        idx = b.select(valid, idx, _c(b, 0))
        layer = b.div(idx, U)
        u = b.sub(idx, b.mul(layer, U))
        req = _udiv_const(b, u, sh.hv)
        h = b.sub(u, b.mul(req, _c(b, sh.hv)))
        lr0 = b.add(b.mul(layer, rows), b.mul(req, _c(b, G)))
        return valid, h, lr0

    trs = _rsrc(b, a["tab"], _c(b, 0x7FFFFFF0))

    def ptrs(lr0):
        """ptr="smem": (rec_in, rec_out of the snapshot rows) by scalar invariant loads."""
        base = b.mul(lr0, _c(b, TAB_COLS))
        out = [b.global_load_i64(tbl, b.add(base, _c(b, REC_IN)))]
        for t in snaps:
            out.append(b.global_load_i64(tbl, b.add(base, _c(b, TAB_COLS * t + REC_OUT))))
        return out

    def ptr_load(nv):
        """ptr="vec": raw vector loads (rec_in, rec_out row 0 as dwordx4; other snapshot rows dwordx2)."""
        _v, _h, lr0 = item(nv)
        base = b.mul(lr0, _c(b, TAB_COLS * 8))
        out = [b.buffer_load_vN(trs, b.add(base, _c(b, REC_IN * 8)), _c(b, 0), I32, 4)]
        for t in snaps:
            if t:
                out.append(b.buffer_load_vN(trs, b.add(base, _c(b, (TAB_COLS * t + REC_OUT) * 8)), _c(b, 0), I32, 2))
        return out

    def ptr_vals(raw):
        def i64(lo, hi):
            return b.lor(b.zext(b.readfirstlane(lo), I64), b.shl(b.zext(b.readfirstlane(hi), I64), b.const_i64(32)))

        v4 = raw[0]
        vals = [i64(b.vec_extract(v4, 0), b.vec_extract(v4, 1))]
        rest = iter(raw[1:])
        for t in snaps:
            if t == 0:
                vals.append(i64(b.vec_extract(v4, 2), b.vec_extract(v4, 3)))
            else:
                v2 = next(rest)
                vals.append(i64(b.vec_extract(v2, 0), b.vec_extract(v2, 1)))
        return vals

    wv = b.lshr(tid, _c(b, 6))
    ln = b.land(tid, _c(b, 63))
    if k.stagger:
        wv = b.land(b.add(wv, cta), _c(b, NWV - 1))

    def off(h, i):
        if k.amap == "wave":  # wave w owns sub-block w (HB / 8 bytes), access i = 1 KiB of it
            ii = _c(b, i * 1024)
            return b.add(b.mul(h, _c(b, HB)), b.add(b.add(b.mul(wv, _c(b, HB // NWV)), ii), b.mul(ln, _c(b, 16))))
        if k.stagger:  # rotate the CTA's access order
            ir = b.land(b.add(_c(b, i), cta), _c(b, NA - 1))
            return b.add(b.mul(h, _c(b, HB)), b.add(b.mul(ir, _c(b, TH * 16)), b.mul(tid, _c(b, 16))))
        return b.add(b.mul(h, _c(b, HB)), b.add(_c(b, i * TH * 16), b.mul(tid, _c(b, 16))))

    def issue(nv, p):
        """The item's loads, then its snapshot rows' rec_out (uniform i64) for the stores."""
        valid, h, _lr0 = item(nv)
        rs = _rsrc(b, p[0], b.select(valid, _c(b, sh.hv * HB), _c(b, 0)))
        return [
            b.buffer_load_cp(rs, off(h, i), _c(b, 0), dwords=4, aux=CPOL_NT if k.ld_nt else 0) for i in range(NA)
        ] + list(p[1:])

    def store(nv, pk):
        valid, h, _lr0 = item(nv)
        vals, outs = pk[:NA], pk[NA:]
        for po in outs:
            rs = _rsrc(b, po, b.select(valid, _c(b, sh.hv * HB), _c(b, 0)))
            for i in range(NA):
                b.buffer_store_cp(vals[i], rs, off(h, i), _c(b, 0), aux=CPOL_NT if k.st_nt else 0)

    def dummies():
        nrs = _rsrc(b, a["tab"], _c(b, 0))
        for i in range(NA * len(snaps)):  # distinct offsets: identical stores would be merged
            b.buffer_store_cp(b.vec_pack([_c(b, 0)] * 4, I32), nrs, _c(b, 16 * i), _c(b, 0))

    # steady state, step s: [pointers of item s+D+1 (vec) / of item s+D (smem)] [loads of item s+D-1] [stores
    # of item s]; the prologue's virtual steps store to a 0-byte resource (the loop entry gets the back edge's
    # vmcnt history); the loop is unrolled depth times so packets keep their registers
    Dp = k.depth
    smem = k.ptr == "smem"
    if smem:
        ptrq = [ptrs(item(_c(b, Dp - 1))[2])]
    else:
        ptrq = [ptr_load(_c(b, 0)), ptr_load(_c(b, 1))]
        b.s_waitcnt(vmcnt=0)
    pks = []
    for s_ in range(-(Dp - 1), 0):
        if smem:
            pk = issue(_c(b, s_ + Dp - 1), ptrs(item(_c(b, s_ + Dp - 1))[2]))
        else:
            raw = ptr_load(_c(b, s_ + Dp + 1))
            pk = issue(_c(b, s_ + Dp - 1), ptr_vals(ptrq[0]))
            ptrq = ptrq[1:] + [raw]
        b.sched_barrier(0)
        dummies()
        b.sched_barrier(0)
        pks.append(pk)
    NRAW, NQ_ = len(ptrq[0]), len(ptrq)
    NPK = NA + len(snaps)
    args = [(f"sc{i}", v) for i, v in enumerate([v for p in pks for v in p] + [v for r in ptrq for v in r])]
    nfp = b.mul(b.div(b.add(n_items, _c(b, Dp - 1)), _c(b, Dp)), _c(b, Dp))
    with b.scf_for_iter(_c(b, 0), nfp, _c(b, Dp), args, iv_name="sn") as (n0, cv):
        pks = [list(cv[i * NPK : (i + 1) * NPK]) for i in range(Dp - 1)]
        o = (Dp - 1) * NPK
        ptrq = [list(cv[o + q * NRAW : o + (q + 1) * NRAW]) for q in range(NQ_)]
        for u in range(Dp):
            nv = b.add(n0, _c(b, u)) if u else n0
            if smem:
                new = issue(b.add(nv, _c(b, Dp - 1)), ptrq[0])
                ptrq = [ptrs(item(b.add(nv, _c(b, Dp)))[2])]
            else:
                raw = ptr_load(b.add(nv, _c(b, Dp + 1)))
                new = issue(b.add(nv, _c(b, Dp - 1)), ptr_vals(ptrq[0]))
                ptrq = ptrq[1:] + [raw]
            b.sched_barrier(0)
            cur = pks[0] if Dp > 1 else new
            store(nv, cur)
            pks = pks[1:] + [new] if Dp > 1 else []
        b.scf_yield(*([v for p in pks for v in p] + [v for r in ptrq for v in r]))
    if k.stamp:
        b.s_barrier_bare(fence="compiler")
        _stamp(b, a["stamps"], cta, 1)
    b.ret()
    return b.kernel
