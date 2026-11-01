# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The mono's GEMV-tile family as rocKE stage functions (true-mono WP2b): B1, C1, Q1, P1.

Bit-identical re-implementations of ATOM's FlyDSL stages (``kernels/stages``):

* ``b1``: ``delta.b1`` -- the fused QKVZ projection (MXFP4 weights x the attention HC's MXFP8 mixed row), then for
  the QKV tiles the causal conv + SiLU and the conv-state shift (verify window in seq builds); z tiles publish the
  raw bf16 projection (``qkvz``).
* ``q1``: ``qsa.q1`` -- the fused QSA projection, published as bf16 pairs (``qp``).
* ``c1``: ``out_proj.c1`` -- out_proj over B3's / Q7's MXFP8 output (``d8`` / ``de``), the four waves' chains
  reduced ``((w0 + w1) + w2) + w3``, then the attention HC write ``res + bf16(bf16(proj) * inj)`` per stream,
  published as pairs (``w``) with the 16-column square sums (``whs``). The PLE layer's residual is P2's ``ph``.
* ``p1``: ``ple.ple_proj`` -- the embedding row's MXFP8 pack, the key n-blocks (waves 0-3, one chain) and the value
  n-block (wave 4, four chains reduced in the native order), published as ``pk`` / ``pv`` pairs.

Narrow builds (``rows <= 4``: ``r1..r4``, ``r2s..r4s``; group 0). The device function is linked into the FlyDSL
mono layer kernel (``ATOM_Q38_EXTERN_STAGES``, ``stage_fn`` ABI, ``lds="arg"``): it reads and writes the mono's own
mailboxes and its LDS entries at the mono's offsets (``layout.for_rows``), so a wave of the next stage can start
while a wave of this one still reads its LDS, exactly as in the FlyDSL build.

Every floating-point operation is the mono's, in the mono's order, without fast-math flags (the function is
compiled by the same LLVM in the same module). The scaled MFMA chains keep the mono's step order and accumulator.

Knobs (:class:`GemvKnobs`; the defaults reproduce the FlyDSL stage's instruction choices; every knob below keeps
the numerics bit for bit unless marked):

* protocol: ``mb_impl`` (``asm`` | ``ir`` | ``irf``), ``pause`` (poll retry: -1 none, 0 ``s_nop 0`` = the mono,
  N > 0 ``s_sleep N`` = ATOM's ``pause=N`` lever), ``numa`` (ATOM's ``numa=1`` mailbox area), ``numa_hdr``;
* the FlyDSL catalogue's GEMV findings (``reverse/FLYDSL-STAGES.md`` 1.1, 1.2, 1.4): ``w_cpol`` / ``s_cpol`` (weight
  and weight-scale cache bits; the mono loads both plain), ``oob`` (``load`` = the mono: waves 4-7 (P1: 5-7) issue
  out-of-range loads and skip the chain; ``split`` = the body twice behind a wave-uniform branch, the idle waves
  issue no weight loads), ``asop`` (the A operand's E8M0 scales of four MFMA steps packed
  into one dword and picked with ``op_sel``, built before the chain: no scale VGPR rewrite per MFMA), ``bsop`` (C1:
  the weight scales likewise; the mono shifts one scale dword per MFMA), ``ra`` (LDS operand read-ahead in steps,
  0 = the mono's order), ``sgb`` (``sched_group_barrier`` interleave of the chain), ``mfma`` (``op`` = the
  intrinsic, as the mono; ``asm`` = the same instruction as inline asm with the accumulator tied, ``asm_lead`` /
  ``asm_tail`` hazard wait states);
* ``barrier`` (``wg`` = the mono's workgroup release / acquire fences around ``s_barrier``; ``lds`` = an
  ``lgkmcnt(0)`` + ``s_barrier`` + compiler fence); ``hdr`` (``numa=1``: the NUMA header word by a scalar load,
  ``smem``, or by the B3 / MoE ports' vector load + ``readfirstlane``, ``vmem``, whose ``vmcnt(0)`` drains every load
  in flight at the stage's entry);
* B1: ``host_b3pre`` (the B3 knob spec whose ``pre="dma"`` prologue this B1 emits after its weight loads, as the
  FlyDSL B1 calls ``b3pre``; empty = none), ``conv_wait`` (seq builds: ``vm0`` = the mono's ``vmcnt(0)`` before the
  barrier);
* C1: ``store_out`` (the attention-half kernel's ``h_out`` store);
* tests: ``mutate`` (named mutants that must make the bitwise A/B fail).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from types import SimpleNamespace

from rocke.core.ir import F32, I16, I32, I64, IRBuilder, PtrType, VectorType
from rocke.helpers import mailbox as M

from ..device_function import DeviceFunctionKnobs, lower_device_function, write_device_function

__all__ = [
    "ORDER_ONLY",
    "GemvGeom",
    "GemvKnobs",
    "GemvShape",
    "MUTANTS",
    "STAGES",
    "build_gemv_kernel",
    "geom_from_atom",
    "knobs_from_spec",
    "knobs_spec",
    "lower_gemv_function",
    "write_gemv_functions",
]

STAGES = {"b1": ("delta", "delta_ple"), "c1": ("delta", "delta_ple", "qsa"), "q1": ("qsa",), "p1": ("delta_ple",)}
OOB = 1 << 28  # ATOM ``stages/hc.py`` OOB (int32 elements): a load past the resource returns zeros
RATIO_BITS = 0x3B124925  # ATOM ``p2.RATIO_BITS``: fl(1/448)
LOG2E = 1.4426950408889634  # ATOM ``device_ops.exp``
QNAN = 0x7FC00000  # llvm.maximum.f32's NaN result on AMDGPU
NUMA_PAGE = 1 << 21
NUMA_CHUNK = 8192
POLL_MAX = 12  # ATOM ``mailbox.POLL_MAX``

# Test-only mutants (``GemvKnobs.mutate``): every one must make the in-situ bitwise A/B differ.
#   put_bit:    every put payload's lowest bit flipped (all stages; ATOM's ``mutate.<stage>`` on the port)
#   chain_swap: the MFMA chain's steps 1 and 2 swapped (summation order; all stages; 0 <-> 1 is exact)
#   opsel:      the weight scale's op_sel byte of step 1 off by one (B1, Q1, P1: byte q + 1; C1: the next byte)
#   conv_tap:   B1's conv taps 0 and 1 swapped (QKV tiles)
#   resid:      C1 adds the residual of stream (b + 1) % 4
#   vsum:       P1's value chains reduced ((c1 + c0) + c2) + c3 (a reassociation the bitwise A/B must see)
#   row:        rows > 1: row r's outputs come from row 0's projection (B1 / Q1 epilogue, C1 reduce, P1 key)
#   chain_drop: the chain's step 1 left out (all stages)
#   a_kit:      step 1 multiplies the A operand of step 2 (all stages)
# chain_swap and vsum change the fp32 summation order only: the outputs are bf16, so such a mutant moves an output
# only where a 1-ulp fp32 change crosses a bf16 rounding boundary (rare): they measure the gate's sensitivity to
# order changes and are reported, not required. The order itself is proven by construction (the per-stage IR /
# ISA diff: the same MFMA sequence on one accumulator).
MUTANTS = ("put_bit", "chain_swap", "opsel", "conv_tap", "resid", "vsum", "row", "chain_drop", "a_kit")
ORDER_ONLY = ("chain_swap", "vsum")


# ----------------------------------------------------------------------------- geometry (from ATOM, read-only)


@dataclass(frozen=True)
class GemvGeom:
    """What the four stages need from ATOM's ``config`` / ``layout`` / ``stages.ple`` (read at generation time)."""

    grid: int
    threads: int
    max_rows: int
    row_pairs: int
    hidden: int
    hc_count: int
    qkv: int  # delta_qkv: channels through the causal conv
    qkvz_n: int  # delta_qkv + delta_value
    delta_value: int
    qsa_proj: int
    sc: dict
    tasks: dict
    base: dict
    ple_state: tuple
    lds_rows: dict  # (rows, seq) -> (lds offsets, strides, words)
    lds_groups: dict = field(default_factory=dict)  # (rows, group) -> (lds offsets, strides, words)
    waves: int = 8
    wide_rows: int = 16

    @property
    def hc_width(self) -> int:
        return self.hc_count * self.hidden

    @property
    def hc_words(self) -> int:  # ATOM common.HC_WORDS
        return self.hc_width // 2

    @property
    def row_words(self) -> int:  # ATOM out_proj.ROW_WORDS
        return self.hidden // 2

    def lds(self, rows: int, seq: bool, group: int = 0):
        if group:
            return self.lds_groups[(rows, group)]
        return self.lds_rows[(rows, seq)]


def _k_iters(K: int) -> int:
    return 2 * ((K + 255) // 256)


@lru_cache(maxsize=4)
def geom_from_atom(atom_root: str | None = None) -> GemvGeom:
    from ..mono_layout import _import_atom, _literals

    base = "atom.models.qwen4_exp_mono.kernels"
    cfg = _import_atom(base + ".config", atom_root)
    lay = _import_atom(base + ".layout", atom_root)
    C = cfg.DEFAULT
    if atom_root is None:
        import importlib

        ple = importlib.import_module(base + ".stages.ple")
        ple_state = tuple(ple.STATE)
    else:
        ple_state = tuple(
            _literals(os.path.join(atom_root, *base.split("."), "stages", "ple.py"), ("STATE",))["STATE"]
        )
    lds_rows = {}
    for rows in range(1, lay.MAX_ROWS + 1):
        for seq in (False, True):
            if seq and rows == 1:
                continue
            L = lay.for_rows(rows, seq)
            lds_rows[(rows, seq)] = (dict(L.lds), dict(L.stride), int(L.lds_words))
    lw = _import_atom(base + ".layout_wide", atom_root)
    lds_groups = {}
    for group in (2, 3, 4):
        for rows in range(2 * group, lw.WIDE_ROWS + 1, group):
            L = lw.for_wide(rows, group)
            lds_groups[(rows, group)] = (dict(L.lds), dict(L.stride), int(L.lds_words))
    return GemvGeom(
        grid=cfg.GRID,
        threads=cfg.THREADS,
        max_rows=lay.MAX_ROWS,
        row_pairs=lay.ROW_PAIRS,
        hidden=C.hidden,
        hc_count=C.hc_count,
        qkv=C.delta_qkv,
        qkvz_n=C.delta_qkv + C.delta_value,
        delta_value=C.delta_value,
        qsa_proj=C.qsa_proj,
        sc=dict(lay.SC),
        tasks=dict(lay.TASKS),
        base=dict(lay.BASE),
        ple_state=ple_state,
        lds_rows=lds_rows,
        lds_groups=lds_groups,
        waves=lay.WAVES,
        wide_rows=lw.WIDE_ROWS,
    )


# ----------------------------------------------------------------------------- knobs and shapes


@dataclass(frozen=True)
class GemvShape:
    stage: str
    kind: str
    rows: int = 1
    seq: bool = False
    group: int = 0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}, got {self.stage!r}")
        if self.kind not in STAGES[self.stage]:
            raise ValueError(f"{self.stage} does not run in a {self.kind} layer")
        if self.group:
            if self.seq:
                raise ValueError("a grouped build is r<rows>g<S> (no s)")
            if self.group < 2 or self.rows % self.group or self.rows // self.group < 2 or self.rows > 16:
                raise ValueError("grouped builds: 2+ groups of S >= 2 rows, 16 rows at most")
        elif not 1 <= self.rows <= 4:
            raise ValueError("ungrouped builds: rows 1..4")
        if self.seq and self.rows < 2:
            raise ValueError("a seq build has 2 rows at least")

    @property
    def wide(self) -> bool:
        """More rows than the narrow builds' ``MAX_ROWS``: ATOM's ``*_wide`` stage bodies."""
        return self.rows > 4

    @property
    def variant(self) -> str:
        return f"r{self.rows}" + ("s" if self.seq else "") + (f"g{self.group}" if self.group else "")

    @property
    def stem(self) -> str:
        return f"{self.stage}_{self.kind}_{self.variant}"

    @property
    def symbol(self) -> str:
        return f"rocke_q38_{self.stage}_{self.variant}"


MB_IMPLS = ("asm", "ir", "irf")
BARRIERS = ("wg", "lds")
OOBS = ("load", "split")
CPOLS = (0, 1, 2, 3, 16, 17, 18, 19)
SGBS = ("", "mfma1", "mfma2")
MFMAS = ("op", "asm")


@dataclass(frozen=True)
class GemvKnobs:
    mb_impl: str = "asm"
    pause: int = 0
    numa: int = 0
    numa_hdr: int = 0
    w_cpol: int = 0
    s_cpol: int = 0
    oob: str = "load"
    asop: int = 0
    bsop: int = 0
    ra: int = 0
    sgb: str = ""
    mfma: str = "op"
    asm_lead: int = 2
    asm_tail: int = 16
    barrier: str = "wg"
    hdr: str = "smem"
    pmax: str = "max3"
    epi: str = "lds"
    cprio: int = 0
    host_b3pre: str = ""
    conv_wait: str = "vm0"
    store_out: bool = False
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.mb_impl not in MB_IMPLS:
            raise ValueError(f"mb_impl must be one of {MB_IMPLS}")
        if not -1 <= int(self.pause) <= 127:
            raise ValueError("pause: -1 (none), 0 (s_nop 0) or 1..127 (s_sleep N)")
        if self.numa not in (0, 1):
            raise ValueError("numa must be 0 or 1")
        if self.numa_hdr < 0 or self.numa_hdr % 256 or (self.numa_hdr and not self.numa):
            raise ValueError("numa_hdr: a non-negative multiple of 256, with numa=1")
        if self.w_cpol not in CPOLS or self.s_cpol not in CPOLS:
            raise ValueError(f"w_cpol / s_cpol must be one of {CPOLS}")
        if self.oob not in OOBS:
            raise ValueError(f"oob must be one of {OOBS}")
        if self.asop not in (0, 1) or self.bsop not in (0, 1):
            raise ValueError("asop / bsop: 0 or 1")
        if not 0 <= self.ra <= 4:
            raise ValueError("ra: 0..4")
        if self.sgb not in SGBS:
            raise ValueError(f"sgb must be one of {SGBS}")
        if self.mfma not in MFMAS:
            raise ValueError(f"mfma must be one of {MFMAS}")
        # a VALU write -> XDL read needs 2 wait states, an XDL result -> non-XDL read about 11: lower values would be
        # hazards the compiler cannot see (inline asm), so they are refused rather than searched
        if self.asm_lead not in (-1, 2, 3, 4) or not 11 <= self.asm_tail <= 32:
            raise ValueError("asm_lead: -1 (auto) or 2..4 wait states, asm_tail: 11..32")
        if self.barrier not in BARRIERS:
            raise ValueError(f"barrier must be one of {BARRIERS}")
        if self.hdr not in ("smem", "vmem"):
            raise ValueError("hdr: smem | vmem")
        if self.pmax not in ("max3", "emu"):
            raise ValueError("pmax: max3 | emu")
        if self.epi not in ("lds", "reg"):
            raise ValueError("epi: lds | reg")
        if not 0 <= self.cprio <= 3:
            raise ValueError("cprio: 0..3")
        if self.conv_wait not in ("vm0", "none"):
            raise ValueError("conv_wait: vm0 | none")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate must be one of {MUTANTS}")

    @property
    def header_offset(self) -> int:
        if not self.numa:
            raise ValueError("numa=0 builds do not use the NUMA area")
        if self.numa_hdr:
            return self.numa_hdr
        from ..moe_fused import NUMA_HDR

        return NUMA_HDR

    @property
    def mailbox(self) -> M.MailboxKnobs:
        pause = {-1: "none", 0: "s_nop", 1: "s_sleep1"}.get(self.pause, "s_nop")
        if self.mb_impl != "asm":
            from .. import mailbox_ir  # noqa: F401  (registers "ir" / "irf")
        return M.MailboxKnobs(impl=self.mb_impl, pause=pause)


def knobs_spec(k: GemvKnobs) -> str:
    """``k=v,...`` of the non-default fields (inverse: :func:`knobs_from_spec`)."""
    d = GemvKnobs()
    return ",".join(
        f"{f.name}={getattr(k, f.name)}"
        for f in fields(k)
        if f.name != "host_b3pre" and getattr(k, f.name) != getattr(d, f.name)
    )


def knobs_from_spec(spec: str) -> GemvKnobs:
    kn: dict = {}
    types = {f.name: f.type for f in fields(GemvKnobs)}
    for kv in filter(None, (x.strip() for x in (spec or "").split(","))):
        k, _, v = kv.partition("=")
        if k not in types:
            raise ValueError(f"unknown GEMV knob {k!r}")
        if k == "host_b3pre":
            raise ValueError("host_b3pre holds a B3 spec with commas: set it with replace(), not in a spec string")
        cur = getattr(GemvKnobs(), k)
        if isinstance(cur, bool):
            kn[k] = v.lower() in ("1", "true")
        elif isinstance(cur, int):
            kn[k] = int(v)
        else:
            kn[k] = v
    return GemvKnobs(**kn)


# ----------------------------------------------------------------------------- emitter


class _E:
    """One stage function's emission context (``a``: the ``stage_fn`` ABI values)."""

    def __init__(self, b: IRBuilder, a, g: GemvGeom, shape: GemvShape, k: GemvKnobs, pool_words: int):
        self.b, self.a, self.g, self.s, self.k = b, a, g, shape, k
        self.rows = shape.rows
        self.wide = shape.wide
        lds, stride, words = g.lds(shape.rows, shape.seq, shape.group)
        self.lds_off, self.lds_stride = lds, stride
        # The mono's arena, from word 0 (``lds_base``): entries at the mono's own offsets.
        self.pool = b.smem_alloc(I32, [max(pool_words, 4)], name_hint="mono_arena")
        self.mbk = k.mailbox
        self.scratch = b.readfirstlane(a.scratch)
        self._area_mb = None
        if k.numa:
            self._area()  # at entry: dominates every use

    # -- scalars
    def c(self, v: int):
        return self.b.const_i32(int(v))

    def f(self, v: float):
        return self.b.const_f32(float(v))

    def fbits(self, bits: int):
        return self.b.bitcast(self.c(bits), F32)

    # -- LDS (word indices into the mono arena)
    def ent(self, name: str, idx, row=0):
        """Word index of ``idx`` in row ``row`` of LDS entry ``name`` (ATOM ``row_ix`` on ``c.L``)."""
        b = self.b
        base = self.lds_off[name]
        i = idx if not isinstance(idx, int) else self.c(idx)
        i = b.add(i, self.c(base))
        if isinstance(row, int):
            return i if row == 0 else b.add(i, self.c(row * self.lds_stride[name]))
        return b.add(i, b.mul(row, self.c(self.lds_stride[name])))

    def ldw(self, w, dtype=I32):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, w, dtype=dtype, n=1), 0)

    def ldv(self, w, n: int):
        return self.b.smem_load_vN(self.pool, w, dtype=I32, n=n)

    def stw(self, w, v) -> None:
        self.b.smem_store_vN(self.pool, [w], v, 1)

    def stv(self, w, v) -> None:
        self.b.smem_store_vN(self.pool, [w], v, v.type.count)

    def barrier(self) -> None:
        if self.k.barrier == "wg":
            self.b.s_barrier_bare(fence="workgroup")  # ATOM gpu.barrier: release fence, s_barrier, acquire fence
        else:
            self.b.sync_lds_only(fence="compiler")

    # -- global memory
    def rsrc(self, addr, nbytes: int = 0x7FFFFFF0):
        b = self.b
        ptr = b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(nbytes))

    def load(self, rs, byte_off, dwords: int, cpol: int = 0):
        return self.b.buffer_load_cp(rs, byte_off, self.c(0), dwords=dwords, aux=cpol)

    def store(self, rs, byte_off, data) -> None:
        self.b.buffer_store_cp(data, rs, byte_off, self.c(0), aux=0)

    def table(self, addr):
        """Entry ``i`` (wave-uniform) of the int64 table at ``addr``: a scalar load (ATOM ``table_entry``)."""
        b = self.b
        tbl = b.int_to_ptr(b.readfirstlane(addr), PtrType(I64, "constant"))
        return lambda i: b.readfirstlane(b.global_load_i64(tbl, i if not isinstance(i, int) else self.c(i)))

    # -- mailboxes
    def mb(self, region: str) -> M.Mailbox:
        # Built where used (no cache: a value made inside one runtime region does not dominate another's uses;
        # LLVM folds the repeats). The NUMA area and its selector are made once at entry (``__init__``).
        if self.k.numa:
            return self._area()
        b = self.b
        addr = b.add(self.scratch, b.const_i64(self.g.sc[region]))
        return M.mailbox_from_i64(b, addr, self.c(0x7FFFFFF0), self.mbk)

    def _area(self) -> M.Mailbox:
        """ATOM's NUMA area as one mailbox and this CTA's copy selector (``stage_b3._B3._area``)."""
        if self._area_mb is None:
            b = self.b
            hdr = self.k.header_offset
            a = b.add(self.scratch, b.const_i64(hdr + 256 + NUMA_PAGE - 1))
            area = b.land(a, b.const_i64(-NUMA_PAGE))
            self._area_mb = M.mailbox_from_i64(b, area, self.c(0x7FFFFFF0), self.mbk)
            # ``hdr=smem`` (default): the header word through a constant-space pointer: a scalar load (the host
            # writes it before the launch; no wave writes it), no vmcnt(0) at the stage's entry. ``vmem``: the
            # B3 / MoE ports' vector load + readfirstlane, whose s_waitcnt vmcnt(0) drains every load in flight.
            space = "constant" if self.k.hdr == "smem" else "global"
            hp = b.int_to_ptr(b.add(self.scratch, b.const_i64(hdr)), PtrType(I32, space))
            mask = b.readfirstlane(b.global_load(hp, self.c(0), I32, align=4))
            self._sel = b.readfirstlane(b.xor(mask, b.sub(self.c(0), M.iod_id(b))))
        return self._area_mb

    def _xi(self, region: str, i):
        b = self.b
        x = b.add(self.c(self.g.sc[region]), b.mul(i, self.c(M.PAIR_BYTES)))
        return b.add(x, b.land(x, self.c(-NUMA_CHUNK)))

    def put_ix(self, region: str, i):
        if not self.k.numa:
            return i
        return self.b.lshr(self._xi(region, i), self.c(3))

    def poll_ix(self, region: str, i):
        if not self.k.numa:
            return i
        b = self.b
        self._area()
        xi = self._xi(region, i)
        sel = b.land(b.lshr(self._sel, b.lshr(xi, self.c(21))), self.c(1))
        return b.lshr(b.add(xi, b.shl(sel, self.c(13))), self.c(3))

    def mbix(self, i, r):
        """Pair ``i`` of request row ``r`` (ATOM ``mb_ix``)."""
        b = self.b
        if isinstance(i, int):
            i = self.c(i)
        if isinstance(r, int):
            return i if r == 0 else b.add(i, self.c(r * self.g.row_pairs))
        return b.add(i, b.mul(r, self.c(self.g.row_pairs)))

    def poll(self, specs):
        """``specs``: ``[(region, pair index, npairs)]`` -> one list of payload words per spec, in batches of
        ``POLL_MAX`` as ATOM's ``mailbox.poll``."""
        b = self.b
        out = []
        for first in range(0, len(specs), POLL_MAX):
            batch = [(self.mb(r), self.poll_ix(r, i), n) for r, i, n in specs[first : first + POLL_MAX]]
            if self.k.mb_impl == "asm":
                out += self._poll_asm(batch)
            else:
                out += self._poll_ir(batch)
        return out

    def _pause_text(self) -> str:
        p = self.k.pause
        return "" if p < 0 else ("s_nop 0" if p == 0 else f"s_sleep {p}")

    def _poll_ir(self, specs):
        """``mailbox_ir``'s ``ir`` / ``irf`` poll (a compiler-visible ``tile.tagged_poll`` loop of ``sc1`` buffer
        loads, ``2 n`` dwords per spec: FlyDSL's poll shape) with this build's pause (``s_sleep N`` too)."""
        b = self.b
        tspecs = []
        for mb, i, n in specs:
            off = M.pair_offset(b, i)
            if mb.copy_off is not None:
                off = b.add(off, mb.copy_off)
            tspecs.append((mb.rsrc, off, 2 * n))
        kw = {"reload": "launder"} if self.k.mb_impl == "irf" else {}
        vecs = b.tagged_poll(self.a.tag, tspecs, aux=self.mbk.cpol, pause=self._pause_text(), **kw)
        return [[b.vec_extract(v, 2 * q) for q in range(n)] for v, (_, _, n) in zip(vecs, specs)]

    def _poll_asm(self, specs):
        """``mailbox._poll_batch_asm`` with this build's pause (``s_sleep N`` for N > 1 too)."""
        b = self.b
        bases, base_of = [], []
        for mb, _, _ in specs:
            for j, x in enumerate(bases):
                if x is mb.addr:
                    base_of.append(j)
                    break
            else:
                base_of.append(len(bases))
                bases.append(mb.addr)
        npairs = [n for _, _, n in specs]
        pause = self._pause_text()
        template, cons = M.asm_poll_template(npairs, len(bases), base_of, self.mbk.cpol_asm, pause)
        voffs = [M.pair_offset(b, i) for _, i, _ in specs]
        outs = b.inline_asm_multi(
            template, cons, voffs + bases + [b.zext(self.a.tag, I64)],
            result_types=[VectorType(I32, 2)] * sum(npairs) + [I64, I64], result_name_hint="apoll",
        )  # fmt: skip
        res, q = [], 0
        for n in npairs:
            res.append([b.vec_extract(outs[q + e], 0) for e in range(n)])
            q += n
        return res

    @property
    def put_knobs(self) -> M.MailboxKnobs:
        return replace(self.mbk, numa_copies=2) if self.k.numa else self.mbk

    def put(self, region: str, i, v) -> None:
        b = self.b
        if self.k.mutate == "put_bit":
            v = b.xor(b.bitcast(v, I32) if v.type == F32 else v, self.c(1))
        M.put(b, self.mb(region), self.put_ix(region, i), v, self.a.tag, self.put_knobs)

    # -- lane exchanges (bits only)
    def xshfl1(self, v):
        """``v`` of lane ``lane ^ 1`` (ATOM ``xshfl(v, 1)``: DPP quad_perm [1, 0, 3, 2])."""
        b = self.b
        is_f = v.type == F32
        x = b.bitcast(v, I32) if is_f else v
        r = b.quad_perm(x, (1, 0, 3, 2))
        return b.bitcast(r, F32) if is_f else r

    def xshfl(self, v, off: int):
        """``v`` of lane ``lane ^ off`` (off in 1, 2, 4): ATOM ``xshfl`` (DPP; a permutation moves bits)."""
        b = self.b
        if off == 1:
            return self.xshfl1(v)
        is_f = v.type == F32
        x = b.bitcast(v, I32) if is_f else v
        if off == 2:
            r = b.quad_perm(x, (2, 3, 0, 1))
        elif off == 4:
            # lanes with bit 2 clear read lane + 4 (row_shl), the others lane - 4 (row_shr) (``stage_q2`` dpp2)
            up = b.mov_dpp(x, row_shl=4)
            dn = b.mov_dpp(x, row_shr=4)
            r = b.select(b.cmp_eq(b.land(self.a.lane, self.c(4)), self.c(0)), up, dn)
        else:
            r = b.dpp_xor(x, off)
        return b.bitcast(r, F32) if is_f else r

    # -- numerics helpers (ATOM ``stages/common.py``, ``device_ops.py``, ``p2.py``)
    def bf16r(self, x):
        b = self.b
        return b.cast_to_f32(b.trunc_f32_to_bf16(x))

    def bf16_bits(self, x):
        b = self.b
        return b.zext(b.bitcast(b.trunc_f32_to_bf16(x), I16), I32)

    def bf(self, bits):
        """bf16 bits (low half) -> f32."""
        return self.b.bitcast(self.b.shl(bits, self.c(16)), F32)

    def sigmoid_fast(self, x):
        """``v_rcp(1 + v_exp(-x * log2 e))`` (ATOM ``common.sigmoid_fast``: rcp(1.0 + exp(0.0 - x)))."""
        b = self.b
        neg = b.fsub(self.f(0.0), x)
        e = self._intrinsic("llvm.amdgcn.exp2.f32", b.fmul(neg, self.f(LOG2E)))
        return self._intrinsic("llvm.amdgcn.rcp.f32", b.fadd(self.f(1.0), e))

    def _intrinsic(self, name: str, x):
        b = self.b
        if name == "llvm.amdgcn.exp2.f32":
            return b.exp2_fast(x)
        if name == "llvm.amdgcn.rcp.f32":
            return b.rcp_fast(x)
        raise KeyError(name)

    def maximum(self, x, y):
        """``llvm.maximum.f32`` (FlyDSL ``fx.max`` on f32): NaN if either is NaN, else the larger."""
        b = self.b
        return b.select(b.fcmp("uno", x, y), self.fbits(QNAN), b.fmax(x, y))

    # -- the MXFP4 GEMV unit (ATOM ``common.p2_issue`` / ``p2_mma`` / ``a_operand``)
    def issue(self, rs_w, rs_s, nb, ki: int, g: int, off):
        """Weights (4 x dwordx4) and the scale dword of k_iter group ``g`` of n-block ``nb``; ``off`` int32
        elements (``OOB`` past the resource)."""
        b = self.b
        lane = self.a.lane
        ws = []
        for q in range(4):
            el = b.add(b.mul(b.add(b.mul(b.add(b.mul(nb, self.c(ki)), self.c(g * 4 + q)), self.c(64)), lane), self.c(4)), off)
            ws.append(self.load(rs_w, b.mul(el, self.c(4)), 4, self.k.w_cpol))
        el = b.add(b.add(b.mul(b.add(b.mul(nb, self.c(ki // 4)), self.c(g)), self.c(64)), lane), off)
        sw = self.load(rs_s, b.mul(el, self.c(4)), 1, self.k.s_cpol)
        return ws, sw

    def a_operand(self, xr, kit: int):
        """Lane's fp8 A operand of k_iter ``kit``: two 16-byte halves 64 bytes apart (``xr``: word index)."""
        b = self.b
        off = b.add(xr, b.add(self.c(kit * 32), b.mul(b.lshr(self.a.lane, self.c(4)), self.c(4))))
        a0 = self.ldv(off, 4)
        a1 = self.ldv(b.add(off, self.c(16)), 4)
        return b.vec_concat(a0, a1)

    def a_scale(self, er, kit: int):
        b = self.b
        return self.ldw(b.add(er, b.add(self.c(kit * 4), b.lshr(self.a.lane, self.c(4)))))

    def pack4(self, words):
        """Bytes 0 of four dwords -> one dword (byte q = word q's byte 0): the op_sel scale form."""
        b = self.b
        m = self.c(0xFF)
        out = b.land(words[0], m)
        for q in range(1, 4):
            out = b.lor(out, b.shl(b.land(words[q], m), self.c(8 * q)))
        return out

    def mma(self, acc, a_op, w, sa, sb, opsel_a: int, opsel_b: int, last: bool = False, valu_src: bool = True):
        if self.k.mfma == "op":
            return self.b.mfma_scale_mx(
                a_op, w, acc, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp4", opsel_a=opsel_a, opsel_b=opsel_b
            )
        # ``mfma=asm``: the same instruction with its accumulator TIED (dst = srcC): the dependent chain runs
        # back to back (srcC forwarding) whatever the register allocator would pick for the intrinsic. Inline asm
        # is opaque to the hazard recognizer: ``asm_lead`` wait states before it (a VALU-written source needs 2),
        # ``asm_tail`` after the chain's last step (its result's readers).
        lines = []
        # ``asm_lead=-1`` (auto): 2 wait states only where a source is a VALU result the emitter knows of (the zero
        # accumulator of the first step, a shifted or packed scale); loads and the previous step's tied accumulator
        # need none. A VALU copy the compiler might add is caught by the static hazard audit (``wp2b_isa.py``).
        lead = (2 if valu_src else 0) if self.k.asm_lead < 0 else self.k.asm_lead
        while lead > 0:
            n = min(lead, 16)
            lines.append(f"s_nop {n - 1}")
            lead -= n
        mods = (
            f"op_sel:[{opsel_a & 1},{opsel_b & 1},0] op_sel_hi:[{(opsel_a >> 1) & 1},{(opsel_b >> 1) & 1},0]"
        )
        lines.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $4, $5 {mods} blgp:4")
        if last:
            tail = self.k.asm_tail
            while tail > 0:
                n = min(tail, 16)
                lines.append(f"s_nop {n - 1}")
                tail -= n
        return self.b.inline_asm(
            "\n\t".join(lines), "=v,v,v,0,v,v", [a_op, w, acc, sa, sb], acc.type, sideeffect=False,
            result_name_hint="mxasm",
        )

    def tail_fence(self, accs):
        """``mfma=asm``: one ``asm_tail`` wait after several independent asm chains (P1's value chains), tied
        through every chain's accumulator so no reader of any of them moves above it."""
        b = self.b
        lines, tail = [], self.k.asm_tail
        while tail > 0:
            n = min(tail, 16)
            lines.append(f"s_nop {n - 1}")
            tail -= n
        n = len(accs)
        cons = ",".join(["=v"] * n + [str(i) for i in range(n)])
        outs = b.inline_asm_multi("\n\t".join(lines), cons, list(accs), result_types=[a.type for a in accs],
                                  sideeffect=False, result_name_hint="mxtail")  # fmt: skip
        return list(outs)

    def chain(self, steps, xr, er, acc=None, tail=True):
        """``acc += W @ x`` over ``steps`` = ``[(kit, w, (scale dword, op_sel byte))]`` in order: the mono's single
        accumulator chain. Knobs: ``asop`` (A scales packed four steps per dword before the chain), ``ra`` (LDS
        operand read-ahead), ``sgb``; ``mutate=chain_swap`` swaps the first two steps."""
        b, k = self.b, self.k
        steps = list(steps)
        if k.mutate == "chain_swap":
            # steps 1 and 2 (swapping 0 and 1 is exact: the first step adds to a zero accumulator)
            steps[1], steps[2] = steps[2], steps[1]
        if k.mutate == "opsel":
            kit, w, (sb, sel) = steps[1]
            steps[1] = (kit, w, (sb, (sel + 1) % 4))
        if k.mutate == "a_kit":
            kit, w, sc = steps[1]
            steps[1] = (steps[2][0], w, sc)
        if k.mutate == "chain_drop":
            del steps[1]
        n = len(steps)
        if k.cprio:
            b.s_setprio(k.cprio)
        if acc is None:
            acc = b.zero_vec_f32(4)
        sa_of = {}
        if k.asop:
            # one packed A-scale dword per four consecutive steps (byte q = step 4j + q's exponent)
            for j in range(0, n, 4):
                grp = steps[j : j + 4]
                words = [self.a_scale(er, kit) for kit, _, _ in grp]
                words += [self.c(127)] * (4 - len(words))
                packed = self.pack4(words)
                for q, (kit, _, _) in enumerate(grp):
                    sa_of[j + q] = (packed, q)
        ops = {}

        def fetch(i):
            if i >= n or i in ops:
                return
            kit = steps[i][0]
            sa = sa_of.get(i) or (self.a_scale(er, kit), 0)
            ops[i] = (self.a_operand(xr, kit), sa)

        for i in range(min(k.ra, n)):
            fetch(i)
        for i, (kit, w, (sb, selb)) in enumerate(steps):
            fetch(i)
            fetch(i + k.ra)
            a_op, (sa, sela) = ops.pop(i)
            # VALU sources: the zero accumulator (step 0) and a packed A-scale dword at its first use (step 4 j;
            # the later steps of the group are past that step's lead)
            acc = self.mma(acc, a_op, w, sa, sb, sela, selb, last=(i == n - 1) and tail,
                           valu_src=(i == 0) or (bool(sa_of) and i % 4 == 0))
            if k.sgb == "mfma1":
                b.sched_group_barrier(0x100, 2, 0)  # the next step's two ds_read_b128
                b.sched_group_barrier(0x008, 1, 0)
            elif k.sgb == "mfma2":
                b.sched_group_barrier(0x100, 3, 0)
                b.sched_group_barrier(0x008, 1, 0)
        if k.cprio:
            b.s_setprio(0)
        return acc

    def mfma_operands(self, x8_name: str, xe_name: str):
        """``(xr, er)``: word indices of this lane's A row's fp8 words and exponents (ATOM ``mfma_operands``)."""
        b = self.b
        if self.rows == 1:
            return self.c(self.lds_off[x8_name]), self.c(self.lds_off[xe_name])
        r = b.smin(b.land(self.a.lane, self.c(15)), self.c(self.rows - 1))
        return self.ent(x8_name, 0, r), self.ent(xe_name, 0, r)

    def wave_lt(self, n: int):
        return self.b.cmp_lt(self.a.wave, self.c(n))

    def oob_off(self):
        """Waves 4-7: int32 elements past the resource (ATOM ``off``)."""
        return self.b.select(self.wave_lt(4), self.c(0), self.c(OOB))

    def poll_mixed_wide(self):
        """ATOM ``common_wide.poll_mixed_wide``: wave 0 waits on row 0's 80 exponent pairs, then thread ``i < 320``
        loads its fp8 words of every row and exponents ``i, i + 320, ...`` of rows 1 .. rows - 1."""
        b, a, g = self.b, self.a, self.g
        lane, tid = a.lane, a.tid
        n = g.hidden // 32
        vecs = g.hidden // 8
        ne = (self.rows - 1) * n
        with b.scf_if(b.cmp_eq(a.wave, self.c(0))):
            e = self.poll(
                [("xe", self.mbix(lane, 0), 1), ("xe", self.mbix(b.smin(b.add(lane, self.c(64)), self.c(n - 1)), 0), 1)]
            )
            self.stw(self.ent("xe", lane), e[0][0])
            with b.scf_if(b.cmp_lt(lane, self.c(16))):
                self.stw(self.ent("xe", b.add(lane, self.c(64))), e[1][0])
        self.barrier()
        with b.scf_if(b.cmp_lt(tid, self.c(vecs))):
            nm = -(-ne // vecs)
            ks = [b.smin(b.add(tid, self.c(vecs * m)), self.c(ne - 1)) for m in range(nm)]
            specs = [("x8", self.mbix(b.mul(tid, self.c(2)), r), 2) for r in range(self.rows)]
            specs += [("xe", self.mbix(b.mod(kk, self.c(n)), b.add(self.c(1), b.div(kk, self.c(n)))), 1) for kk in ks]
            v = self.poll(specs)
            for r in range(self.rows):
                self.stw(self.ent("x8", b.mul(tid, self.c(2)), r), v[r][0])
                self.stw(self.ent("x8", b.add(b.mul(tid, self.c(2)), self.c(1)), r), v[r][1])
            for m in range(nm):
                with b.scf_if(b.cmp_lt(b.add(tid, self.c(vecs * m)), self.c(ne))):
                    kk = ks[m]
                    self.stw(self.ent("xe", b.mod(kk, self.c(n)), b.add(self.c(1), b.div(kk, self.c(n)))),
                             v[self.rows + m][0])  # fmt: skip
        self.barrier()

    def poll_mixed(self):
        """ATOM ``common.poll_mixed``: A2's MXFP8 mixed row of every request row into LDS ``x8`` / ``xe``."""
        if self.wide:
            self.poll_mixed_wide()
            return
        b, a, g = self.b, self.a, self.g
        lane, tid = a.lane, a.tid
        n = g.hidden // 32
        with b.scf_if(b.cmp_eq(a.wave, self.c(0))):
            e = self.poll(
                [("xe", self.mbix(lane, 0), 1), ("xe", self.mbix(b.smin(b.add(lane, self.c(64)), self.c(n - 1)), 0), 1)]
            )
            self.stw(self.ent("xe", lane), e[0][0])
            with b.scf_if(b.cmp_lt(lane, self.c(16))):
                self.stw(self.ent("xe", b.add(lane, self.c(64))), e[1][0])
        self.barrier()
        with b.scf_if(b.cmp_lt(tid, self.c(g.hidden // 8))):
            if self.rows == 1:
                v = self.poll([("x8", self.mbix(b.mul(tid, self.c(2)), 0), 2)])[0]
                self.stw(self.ent("x8", b.mul(tid, self.c(2))), v[0])
                self.stw(self.ent("x8", b.add(b.mul(tid, self.c(2)), self.c(1))), v[1])
            else:
                kk = b.smin(tid, self.c((self.rows - 1) * n - 1))
                er = b.add(self.c(1), b.div(kk, self.c(n)))
                ej = b.mod(kk, self.c(n))
                specs = [("x8", self.mbix(b.mul(tid, self.c(2)), r), 2) for r in range(self.rows)]
                v = self.poll(specs + [("xe", self.mbix(ej, er), 1)])
                for r in range(self.rows):
                    self.stw(self.ent("x8", b.mul(tid, self.c(2)), r), v[r][0])
                    self.stw(self.ent("x8", b.add(b.mul(tid, self.c(2)), self.c(1)), r), v[r][1])
                with b.scf_if(b.cmp_lt(tid, self.c((self.rows - 1) * n))):
                    self.stw(self.ent("xe", ej, er), v[self.rows][0])
        self.barrier()


# ----------------------------------------------------------------------------- B1 / Q1


def _proj_chain(e: _E, rs_w, rs_s, nb, ki: int, off):
    """Weights of the wave's n-block (5 groups, 4 dwordx4 + 1 scale dword each), issued now."""
    units = [e.issue(rs_w, rs_s, nb, ki, gi, off) for gi in range(ki // 4)]
    steps = []
    for gi, (ws, sw) in enumerate(units):
        for q in range(4):
            steps.append((gi * 4 + q, ws[q], (sw, q)))
    return steps


def _store_red(e: _E, acc) -> None:
    """Lanes 0-15 of wave ``w`` store row ``r``'s 16 outputs at ``red[w * 16 + lane]`` (row ``r``); wide builds:
    ATOM ``common_wide.store_rows`` (``acc[i]`` of lane ``l`` is request row ``4 (l // 16) + i``)."""
    b, a = e.b, e.a
    if e.wide:
        _store_rows(e, acc, b.add(b.mul(a.wave, e.c(16)), b.land(a.lane, e.c(15))))
        return
    with b.scf_if(b.cmp_lt(a.lane, e.c(16))):
        for r in range(e.rows):
            e.stw(e.ent("red", b.add(b.mul(a.wave, e.c(16)), a.lane), r), b.vec_extract(acc, r))


def _store_rows(e: _E, acc, col) -> None:
    b, a = e.b, e.a
    for i in range(4):
        r = b.add(b.mul(b.lshr(a.lane, e.c(4)), e.c(4)), e.c(i))
        with b.scf_if(b.cmp_lt(r, e.c(e.rows))):
            e.stw(e.ent("red", col, r), b.vec_extract(acc, i))


def _split(e: _E, n_chain: int, body) -> None:
    """``oob=load`` (the mono): one body; waves past ``n_chain`` load out of range and skip the chain.
    ``oob=split``: the body twice behind a wave-uniform branch, so waves past ``n_chain`` issue no weight loads
    and hold no chain code (``body(role)``: ``None`` mono, ``"chain"``, ``"idle"``)."""
    if e.k.oob == "load":
        body(None)
        return
    b = e.b
    with b.scf_if_else(e.wave_lt(n_chain)) as (th, el):
        with th:
            body("chain")
        with el:
            body("idle")


def _role_if(e: _E, role, n_chain: int, fn) -> None:
    """Run ``fn`` on the chain waves: under ``wave < n_chain`` (mono), always (``chain``), never (``idle``)."""
    if role == "idle":
        return
    if role == "chain":
        fn()
        return
    with e.b.scf_if(e.wave_lt(n_chain)):
        fn()


def _b1(e: _E) -> None:
    _split(e, 4, lambda role: _b1_body(e, role))


def _b1_body(e: _E, role) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    lane, wave = a.lane, a.wave
    rows, seq, group = e.rows, e.s.seq, e.s.group
    t = a.task
    QKV = g.qkv
    KI = _k_iters(g.hidden)  # 20
    rw = e.c(0) if rows == 1 else b.smin(wave, e.c(rows - 1))
    ent = e.table(a.conv_in)

    def state(r):
        """Row ``r``'s (conv_in, conv_out) addresses: ATOM ``delta_state``. Seq builds: every row starts from row
        0's state; grouped builds: every row's entry holds its group's input state."""
        if seq:
            return ent(0), ent(b.add(b.mul(r, e.c(4)), e.c(1)))
        return ent(b.mul(r, e.c(4))), ent(b.add(b.mul(r, e.c(4)), e.c(1)))

    conv_in, conv_out = state(rw)

    # Weights and the conv operands do not depend on the mixed row: issued before the wait (the mono's order).
    nb = b.add(b.mul(t, e.c(4)), b.land(wave, e.c(3)))
    steps = None
    if role != "idle":
        rs_w = e.rsrc(a.w_qkvz, g.hidden * g.qkvz_n // 2)
        rs_s = e.rsrc(a.s_qkvz, g.hidden * g.qkvz_n // 32)
        steps = _proj_chain(e, rs_w, rs_s, nb, KI, e.oob_off() if role is None else e.c(0))
    ch = b.smin(b.add(b.mul(t, e.c(64)), lane), e.c(QKV - 1))
    cw = e.load(e.rsrc(a.conv_w), b.mul(ch, e.c(8)), 2)
    half = b.lshr(ch, e.c(1))

    def taps_of(cin):
        rs_ci = e.rsrc(cin)
        return [e.load(rs_ci, b.mul(b.add(e.c(kk * (QKV // 2)), half), e.c(4)), 1) for kk in range(3)]

    if k.epi == "reg":
        _b1_body_reg(e, role, steps, state)
        return
    cs = taps_of(conv_in)
    # wide builds past ``WAVES`` rows: wave ``w`` also runs row ``w + 8 ek``'s epilogue (``delta_group_wide``)
    more = []
    for ek in range(1, -(-rows // g.waves)):
        rq = b.smin(b.add(wave, e.c(g.waves * ek)), e.c(rows - 1))
        cin_q, cout_q = state(rq)
        more.append((rq, cout_q, taps_of(cin_q)))

    if k.host_b3pre:
        _host_b3pre(e)
    e.poll_mixed()

    def mm():
        xr, er = e.mfma_operands("x8", "xe")
        _store_red(e, e.chain(steps, xr, er))

    _role_if(e, role, 4, mm)
    if (seq or group) and k.conv_wait == "vm0":
        # ATOM's verify window: a request's snapshots overwrite its input taps in place (``conv_out[r]`` overlaps
        # ``conv_in``), so every wave's ``cs`` loads land before any wave stores past the barrier
        b.s_waitcnt(vmcnt=0)
    e.barrier()

    epi = b.cmp_eq(wave, e.c(0)) if rows == 1 else e.wave_lt(rows)
    with b.scf_if(epi):
        _b1_epi(e, t, rw, cw, cs, conv_out, half)
    for ek, (rq, cout_q, cs_q) in enumerate(more, start=1):
        with b.scf_if(b.cmp_lt(b.add(wave, e.c(g.waves * ek)), e.c(rows))):
            _b1_epi(e, t, rq, cw, cs_q, cout_q, half)


def _b1_body_reg(e: _E, role, steps, state) -> None:
    """``epi=reg`` (narrow ungrouped builds): no LDS round trip and no CTA barrier after the GEMV. Lanes 0-15 of
    MFMA wave ``w`` hold columns ``16 w ..`` of every request row (``acc[r]``, the 16x16 tile's rows 0-3), so the
    wave runs the epilogue of its 16 channels for every row itself: the same operations per (row, channel) as the
    mono's wave-``r`` epilogue over 64 channels (bitwise). The verify window's in-place snapshots touch only the
    wave's own channels, whose taps it loaded itself (its stores follow those loads' use)."""
    b, a, g, k = e.b, e.a, e.g, e.k
    lane, wave = a.lane, a.wave
    rows, seq = e.rows, e.s.seq
    t = a.task
    QKV = g.qkv
    l16 = b.land(lane, e.c(15))
    chw = b.smin(b.add(b.add(b.mul(t, e.c(64)), b.mul(b.land(wave, e.c(3)), e.c(16))), l16), e.c(QKV - 1))
    cw = e.load(e.rsrc(a.conv_w), b.mul(chw, e.c(8)), 2)
    half = b.lshr(chw, e.c(1))

    def taps(cin):
        rs_ci = e.rsrc(cin)
        return [e.load(rs_ci, b.mul(b.add(e.c(kk * (QKV // 2)), half), e.c(4)), 1) for kk in range(3)]

    # every row's (conv_out, taps): seq builds share row 0's input state
    rows_state = []
    for r in range(rows):
        cin, cout = state(e.c(r))
        rows_state.append((cout, taps(cin) if (r == 0 or not seq) else None))
    if k.host_b3pre:
        _host_b3pre(e)
    e.poll_mixed()

    def mm():
        xr, er = e.mfma_operands("x8", "xe")
        acc = e.chain(steps, xr, er)
        projs = [e.bf16r(b.vec_extract(acc, r)) for r in range(rows)]
        if k.mutate == "row" and rows > 1:
            projs = [projs[0]] * rows
        with b.scf_if(b.cmp_lt(lane, e.c(16))):
            for r in range(rows):
                cout = rows_state[r][0]
                cs = rows_state[0 if seq else r][1]
                with b.scf_if_else(b.cmp_lt(t, e.c(QKV // 64))) as (th, el):
                    with th:
                        o = _b1_conv_reg(e, r, projs, cw, cs, cout, half)
                        _b1_put_reg(e, t, wave, lane, r, o)
                    with el:
                        _b1_put_reg(e, t, wave, lane, r, projs[r])

    _role_if(e, role, 4, mm)


def _b1_conv_reg(e: _E, r: int, projs, cw, cs, conv_out, half):
    """Row ``r``'s conv + SiLU of the lane's channel and its next conv state (``_b1_epi`` at a static row)."""
    b, a, g, k = e.b, e.a, e.g, e.k
    lane = a.lane
    seq = e.s.seq
    QKV = g.qkv
    proj = projs[r]
    w0, w1 = M.bf2_f32(b, b.vec_extract(cw, 0))
    w2, w3 = M.bf2_f32(b, b.vec_extract(cw, 1))
    sh = b.mul(b.land(lane, e.c(1)), e.c(16))
    old = [b.land(b.lshr(w, sh), e.c(0xFFFF)) for w in cs]
    if seq:
        # ATOM delta._seq_conv at a static row: the history is row 0's state, then the rows before r
        bits = old + [e.bf16_bits(v) for v in projs[: e.rows - 1]]
        hist = [e.bf(x) for x in old] + projs[: e.rows - 1]
        taps = [hist[r + kk] for kk in range(3)]
        nxt = [bits[r + 1 + kk] for kk in range(2)] + [e.bf16_bits(proj)]
        words = [b.lor(nn, b.shl(e.xshfl1(nn), e.c(16))) for nn in nxt]
    else:
        taps = [e.bf(x) for x in old]
        words = [cs[1], cs[2], M.bf16_pair(b, proj, e.xshfl1(proj))]
    if k.mutate == "conv_tap":
        taps = [taps[1], taps[0], taps[2]]
    conv = b.fmul(taps[0], w0)
    conv = b.fadd(conv, b.fmul(taps[1], w1))
    conv = b.fadd(conv, b.fmul(taps[2], w2))
    conv = b.fadd(conv, b.fmul(proj, w3))
    o = e.bf16r(b.fmul(conv, e.sigmoid_fast(conv)))
    with b.scf_if(b.cmp_eq(b.land(lane, e.c(1)), e.c(0))):
        rs_co = e.rsrc(conv_out)
        for kk in range(3):
            e.store(rs_co, b.mul(b.add(e.c(kk * (QKV // 2)), half), e.c(4)), words[kk])
    return o


def _b1_put_reg(e: _E, t, wave, lane, r: int, out) -> None:
    b = e.b
    other = e.xshfl1(out)
    with b.scf_if(b.cmp_eq(b.land(lane, e.c(1)), e.c(0))):
        col = b.add(b.mul(b.land(wave, e.c(3)), e.c(16)), lane)
        e.put("qkvz", e.mbix(b.add(b.mul(t, e.c(32)), b.lshr(col, e.c(1))), r), M.bf16_pair(b, out, other))


def _b1_epi(e: _E, t, rw, cw, cs, conv_out, half) -> None:
    """Row ``rw``'s B1 epilogue on one wave: the conv + SiLU of the QKV tiles (and the next conv state), the raw
    projection of the z tiles, published as bf16 pairs (``qkvz``)."""
    b, a, g, k = e.b, e.a, e.g, e.k
    lane = a.lane
    rows, seq, group = e.rows, e.s.seq, e.s.group
    QKV = g.qkv
    prow = e.c(0) if (rows > 1 and k.mutate == "row") else rw
    proj = e.bf16r(e.ldw(e.ent("red", lane, prow), F32))

    def conv_out_value():
        w0, w1 = M.bf2_f32(b, b.vec_extract(cw, 0))
        w2, w3 = M.bf2_f32(b, b.vec_extract(cw, 1))
        sh = b.mul(b.land(lane, e.c(1)), e.c(16))
        old = [b.land(b.lshr(w, sh), e.c(0xFFFF)) for w in cs]
        if group:
            taps, nxt = _group_conv(e, lane, rw, old, proj)
            words = [b.lor(nn, b.shl(e.xshfl1(nn), e.c(16))) for nn in nxt]
        elif seq:
            taps, nxt = _seq_conv(e, lane, rw, old, proj)
            words = [b.lor(nn, b.shl(e.xshfl1(nn), e.c(16))) for nn in nxt]
        else:
            taps = [e.bf(x) for x in old]
            words = [cs[1], cs[2], M.bf16_pair(b, proj, e.xshfl1(proj))]
        if k.mutate == "conv_tap":
            taps = [taps[1], taps[0], taps[2]]
        # conv taps are oldest first; bf16 x bf16 products are exact (the mono's mul + add order)
        conv = b.fmul(taps[0], w0)
        conv = b.fadd(conv, b.fmul(taps[1], w1))
        conv = b.fadd(conv, b.fmul(taps[2], w2))
        conv = b.fadd(conv, b.fmul(proj, w3))
        o = e.bf16r(b.fmul(conv, e.sigmoid_fast(conv)))
        with b.scf_if(b.cmp_eq(b.land(lane, e.c(1)), e.c(0))):
            rs_co = e.rsrc(conv_out)
            for kk in range(3):
                e.store(rs_co, b.mul(b.add(e.c(kk * (QKV // 2)), half), e.c(4)), words[kk])
        return o

    # QKV tiles publish the conv + SiLU output, z tiles the projection (the mono branches on the tile)
    with b.scf_if_else(b.cmp_lt(t, e.c(QKV // 64))) as (th, el):
        with th:
            _b1_put(e, t, lane, rw, conv_out_value())
        with el:
            _b1_put(e, t, lane, rw, proj)


def _group_conv(e: _E, lane, rw, old, proj):
    """ATOM ``delta_group._group_conv``: row ``rw``'s three older taps and next state; row ``r``'s history is its
    group's state ``old`` then the projections of the group's rows before it."""
    rows, group = e.rows, e.s.group
    prj = [e.bf16r(e.ldw(e.ent("red", lane, q), F32)) for q in range(rows - 1)]
    pbits = [e.bf16_bits(v) for v in prj]
    taps, keep = [[], [], []], [[], []]
    for r in range(rows):
        j = r % group
        bits = old + pbits[r - j : r]
        hist = [e.bf(x) for x in old] + prj[r - j : r]
        for kk in range(3):
            taps[kk].append(hist[j + kk])
        for kk in range(2):
            keep[kk].append(bits[j + 1 + kk])
    return [_pick_row(e, rw, v) for v in taps], [_pick_row(e, rw, v) for v in keep] + [e.bf16_bits(proj)]


def _b1_put(e: _E, t, lane, rw, out) -> None:
    b = e.b
    other = e.xshfl1(out)
    with b.scf_if(b.cmp_eq(b.land(lane, e.c(1)), e.c(0))):
        e.put("qkvz", e.mbix(b.add(b.mul(t, e.c(32)), b.lshr(lane, e.c(1))), rw), M.bf16_pair(b, out, other))


def _pick_row(e: _E, r, values):
    b = e.b
    v = values[0]
    for q in range(1, len(values)):
        v = b.select(b.cmp_eq(r, e.c(q)), values[q], v)
    return v


def _seq_conv(e: _E, lane, rw, old, proj):
    """ATOM ``delta._seq_conv``: row ``rw``'s three older taps (f32) and its next state (three bf16 low halves)."""
    b = e.b
    rows = e.rows
    hist = [e.bf16r(e.ldw(e.ent("red", lane, q), F32)) for q in range(rows - 1)]
    bits = old + [e.bf16_bits(v) for v in hist]
    hist = [e.bf(x) for x in old] + hist
    taps = [_pick_row(e, rw, [hist[r + kk] for r in range(rows)]) for kk in range(3)]
    keep = [_pick_row(e, rw, [bits[r + 1 + kk] for r in range(rows)]) for kk in range(2)]
    return taps, keep + [e.bf16_bits(proj)]


def _host_b3pre(e: _E) -> None:
    """B3's ``pre="dma"`` prologue for this CTA's B3 task (the mono's B1 calls ``b3pre`` here). B3's base is B1's
    (``layout.BASE``): the CTA's B3 task is its B1 task."""
    from .. import stage_b3 as SB

    if e.g.base["b3"] != e.g.base["b1"]:
        raise ValueError("host_b3pre assumes B3 and B1 share a base (no place lever)")
    bk = SB.knobs_from_spec(e.k.host_b3pre)
    if bk.pre != "dma":
        raise ValueError("host_b3pre: the B3 knob spec has no pre=dma prologue")
    shape = SB.B3Shape(e.rows, e.s.seq, 0)
    lay = SimpleNamespace(sc=e.g.sc, row_pairs=e.g.row_pairs)
    a = e.a
    pad = SB.MONO_B3_LDS_START[shape.rows]
    arena = SB._arena(shape, bk, pad=pad)
    from rocke.helpers.lds_arena import LdsView

    if arena.words > e.pool_words_total:
        raise ValueError("host_b3pre: B3's arena exceeds this function's arena view")
    lds = LdsView(arena, e.pool)
    io = SB.B3Io(
        b=e.b, tid=a.tid, lane=a.lane, wave=a.wave, task=a.task, tag=a.tag, scratch=e.scratch, lds=lds,
        addr=lambda n: getattr(a, n), sc={n: lay.sc[n] for n in SB.B3_REGIONS}, row_pairs=lay.row_pairs,
    )  # fmt: skip
    b = e.b
    with b.scf_if(b.cmp_lt(a.task, e.c(SB.b3_tasks(bk)))):
        SB.emit_b3_pre(io, shape, bk)


def _q1(e: _E) -> None:
    b, g = e.b, e.g
    with b.scf_if(b.cmp_lt(e.a.task, e.c(g.tasks["q1"]))):
        _split(e, 4, lambda role: _q1_body(e, role))


def _q1_body(e: _E, role) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    lane, wave = a.lane, a.wave
    rows = e.rows
    t = a.task
    KI = _k_iters(g.hidden)
    rw = e.c(0) if rows == 1 else b.smin(wave, e.c(rows - 1))
    nb = b.add(b.mul(t, e.c(4)), b.land(wave, e.c(3)))
    steps = None
    if role != "idle":
        rs_w = e.rsrc(a.w_qsa, g.hidden * g.qsa_proj // 2)
        rs_s = e.rsrc(a.s_qsa, g.hidden * g.qsa_proj // 32)
        steps = _proj_chain(e, rs_w, rs_s, nb, KI, e.oob_off() if role is None else e.c(0))
    e.poll_mixed()
    if k.epi == "reg":
        # ``epi=reg``: MFMA wave w publishes its 16 columns of every row from registers (no LDS, no barrier)
        def mmr():
            xr, er = e.mfma_operands("x8", "xe")
            acc = e.chain(steps, xr, er)
            with b.scf_if(b.cmp_lt(lane, e.c(16))):
                col = b.add(b.mul(b.land(wave, e.c(3)), e.c(16)), lane)
                for r in range(rows):
                    src = 0 if (k.mutate == "row" and rows > 1) else r
                    proj = e.bf16r(b.vec_extract(acc, src))
                    other = e.xshfl1(proj)
                    with b.scf_if(b.cmp_eq(b.land(lane, e.c(1)), e.c(0))):
                        e.put("qp", e.mbix(b.add(b.mul(t, e.c(32)), b.lshr(col, e.c(1))), r),
                              M.bf16_pair(b, proj, other))  # fmt: skip

        _role_if(e, role, 4, mmr)
        return

    def mm():
        xr, er = e.mfma_operands("x8", "xe")
        _store_red(e, e.chain(steps, xr, er))

    _role_if(e, role, 4, mm)
    e.barrier()
    epi = b.cmp_eq(wave, e.c(0)) if rows == 1 else e.wave_lt(rows)

    def q1_epi(r):
        prow = e.c(0) if (rows > 1 and k.mutate == "row") else r
        proj = e.bf16r(e.ldw(e.ent("red", lane, prow), F32))
        other = e.xshfl1(proj)
        with b.scf_if(b.cmp_eq(b.land(lane, e.c(1)), e.c(0))):
            e.put("qp", e.mbix(b.add(b.mul(t, e.c(32)), b.lshr(lane, e.c(1))), r), M.bf16_pair(b, proj, other))

    with b.scf_if(epi):
        q1_epi(rw)
    # wide builds past ``WAVES`` rows (``qsa_wide._q1_out``): wave ``w`` also runs row ``w + 8 ek``
    for ek in range(1, -(-rows // g.waves)):
        q = b.add(wave, e.c(g.waves * ek))
        with b.scf_if(b.cmp_lt(q, e.c(rows))):
            q1_epi(b.smin(q, e.c(rows - 1)))


# ----------------------------------------------------------------------------- C1


def _c1(e: _E) -> None:
    b, g = e.b, e.g
    with b.scf_if(b.cmp_lt(e.a.task, e.c(g.tasks["c1"]))):
        _split(e, 4, lambda role: _c1_body(e, role))


def _c1_body(e: _E, role) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    tid, lane, wave = a.tid, a.lane, a.wave
    rows = e.rows
    t = a.task
    KI = _k_iters(g.delta_value)  # 48
    CH = KI // 4  # 12 k_iters per wave
    if True:
        part = b.land(wave, e.c(3))
        off = e.oob_off() if role is None else e.c(0)
        rs_w = e.rsrc(a.w_out, g.hidden * g.delta_value // 2)
        rs_s = e.rsrc(a.s_out, g.hidden * g.delta_value // 32)

        def weights(off):
            ws = []
            for j in range(CH):
                el = b.add(b.add(b.mul(b.add(b.add(b.mul(t, e.c(KI)), part), e.c(4 * j)), e.c(64)), lane), e.c(0))
                el = b.add(b.mul(el, e.c(4)), off)
                ws.append(e.load(rs_w, b.mul(el, e.c(4)), 4, k.w_cpol))
            ss = []
            for j in range(CH):
                el = b.add(b.add(b.mul(b.add(b.mul(t, e.c(CH)), e.c(j)), e.c(64)), lane), off)
                ss.append(e.load(rs_s, b.mul(el, e.c(4)), 1, k.s_cpol))
            return ws, ss

        wss = weights(off) if role != "idle" else None
        # thread ``tid < 32``: stream ``tid // 8``, bf16 word ``tid % 8`` of the tile; batched: 32 r + i
        if rows == 1:
            bs, rt = b.lshr(b.smin(tid, e.c(31)), e.c(3)), e.c(0)
        else:
            tt = b.smin(tid, e.c(32 * rows - 1))
            bs, rt = b.lshr(b.land(tt, e.c(31)), e.c(3)), b.lshr(tt, e.c(5))
        word = b.add(b.add(b.mul(bs, e.c(g.row_words)), b.mul(t, e.c(8))), b.land(tid, e.c(7)))
        rbs = bs
        if k.mutate == "resid":
            rbs = b.land(b.add(bs, e.c(1)), e.c(3))
        rword = b.add(b.add(b.mul(rbs, e.c(g.row_words)), b.mul(t, e.c(8))), b.land(tid, e.c(7)))
        if e.s.kind == "delta_ple":
            res = e.poll([("ph", e.mbix(rword, rt), 1)])[0][0]
        else:
            rs_h = e.rsrc(a.h_in, g.hc_width * 2 * rows)
            hw = b.add(rword, b.mul(rt, e.c(g.hc_words))) if rows > 1 else rword
            res = e.load(rs_h, b.mul(hw, e.c(4)), 1)
        inj = b.bitcast(e.poll([("inj", e.mbix(bs, rt), 1)])[0][0], F32)

        rw = e.c(0) if rows == 1 else b.smin(wave, e.c(rows - 1))
        mine = b.cmp_eq(wave, e.c(0)) if rows == 1 else e.wave_lt(rows)
        with b.scf_if(mine):
            ex = e.poll([("de", e.mbix(b.add(lane, e.c(64 * m)), rw), 1) for m in range(3)])
            for m in range(3):
                e.stw(e.ent("xe", b.add(lane, e.c(64 * m)), rw), ex[m][0])
        # wide builds past ``WAVES`` rows (``out_proj_wide._c1_exps``): wave ``w`` also fills row ``w + 8 ek``'s
        for ek in range(1, -(-rows // g.waves)):
            q = b.add(wave, e.c(g.waves * ek))
            rq = b.smin(q, e.c(rows - 1))
            with b.scf_if(b.cmp_lt(q, e.c(rows))):
                ex = e.poll([("de", e.mbix(b.add(lane, e.c(64 * m)), rq), 1) for m in range(3)])
                for m in range(3):
                    e.stw(e.ent("xe", b.add(lane, e.c(64 * m)), rq), ex[m][0])
        e.barrier()
        hi = b.add(b.smin(tid, e.c(255)), e.c(512))
        specs = []
        for r in range(rows):
            specs += [("d8", e.mbix(b.mul(tid, e.c(2)), r), 2), ("d8", e.mbix(b.mul(hi, e.c(2)), r), 2)]
        v = e.poll(specs)
        for r in range(rows):
            e.stw(e.ent("x8", b.mul(tid, e.c(2)), r), v[2 * r][0])
            e.stw(e.ent("x8", b.add(b.mul(tid, e.c(2)), e.c(1)), r), v[2 * r][1])
        with b.scf_if(b.cmp_lt(tid, e.c(256))):
            for r in range(rows):
                e.stw(e.ent("x8", b.mul(hi, e.c(2)), r), v[2 * r + 1][0])
                e.stw(e.ent("x8", b.add(b.mul(hi, e.c(2)), e.c(1)), r), v[2 * r + 1][1])
        e.barrier()

        def mm():
            ws, ss = wss
            xr, er = e.mfma_operands("x8", "xe")
            sh = b.mul(part, e.c(8))
            steps = []
            if k.bsop:
                # byte ``part`` of four consecutive scale dwords in one dword: op_sel j % 4 (no per-MFMA rewrite)
                for j0 in range(0, CH, 4):
                    packed = e.pack4([b.lshr(ss[j], sh) for j in range(j0, j0 + 4)])
                    for q in range(4):
                        steps.append((None, ws[j0 + q], (packed, q)))
            else:
                for j in range(CH):
                    steps.append((None, ws[j], (b.lshr(ss[j], sh), 0)))
            # k_iter of step j: part + 4 j (a runtime value: the chain addresses LDS with it)
            acc = _c1_chain(e, steps, xr, er, part)
            _store_red(e, acc)

        _role_if(e, role, 4, mm)
        e.barrier()

        with b.scf_if(b.cmp_lt(tid, e.c(32 * rows))):
            j = b.mul(b.land(tid, e.c(7)), e.c(2))

            rrow = e.c(0) if (rows > 1 and k.mutate == "row") else rt

            def red(i):
                return e.ldw(e.ent("red", i, rrow), F32)

            p0, p1 = red(j), red(b.add(j, e.c(1)))
            for w in range(1, 4):
                p0 = b.fadd(p0, red(b.add(j, e.c(w * 16))))
                p1 = b.fadd(p1, red(b.add(j, e.c(w * 16 + 1))))
            r0, r1 = M.bf2_f32(b, res)
            o0 = b.fadd(r0, e.bf16r(b.fmul(e.bf16r(p0), inj)))
            o1 = b.fadd(r1, e.bf16r(b.fmul(e.bf16r(p1), inj)))
            out = M.bf16_pair(b, o0, o1)
            if k.store_out:
                rs_o = e.rsrc(a.h_out)
                hw = b.add(word, b.mul(rt, e.c(g.hc_words))) if rows > 1 else word
                e.store(rs_o, b.mul(hw, e.c(4)), out)
            e.put("w", e.mbix(word, rt), out)
            v0, v1 = M.bf2_f32(b, out)
            sq = b.fadd(b.fmul(v0, v0), b.fmul(v1, v1))
            for o in (1, 2, 4):
                sq = b.fadd(sq, e.xshfl(sq, o))
            with b.scf_if(b.cmp_eq(b.land(tid, e.c(7)), e.c(0))):
                e.put("whs", e.mbix(b.add(b.mul(bs, e.c(g.tasks["c1"])), t), rt), sq)


def _c1_chain(e: _E, steps, xr, er, part):
    """C1's chain: step ``j`` is k_iter ``part + 4 j`` (wave-uniform runtime ``part``); the A operand and its
    exponent are addressed at run time. Same knobs as :meth:`_E.chain`."""
    b, k = e.b, e.k
    steps = list(steps)
    n = len(steps)
    kits = [b.add(part, e.c(4 * j)) for j in range(n)]
    if k.mutate == "chain_swap":
        steps[1], steps[2] = steps[2], steps[1]
        kits[1], kits[2] = kits[2], kits[1]
    if k.mutate == "opsel":
        _, w, (sb, sel) = steps[1]
        steps[1] = (None, w, (sb, (sel + 1) % 4))
    if k.mutate == "a_kit":
        kits[1] = kits[2]
    if k.mutate == "chain_drop":
        del steps[1]
        del kits[1]
    n = len(steps)

    def a_op(kit):
        off = b.add(xr, b.add(b.mul(kit, e.c(32)), b.mul(b.lshr(e.a.lane, e.c(4)), e.c(4))))
        return b.vec_concat(e.ldv(off, 4), e.ldv(b.add(off, e.c(16)), 4))

    def a_sc(kit):
        return e.ldw(b.add(er, b.add(b.mul(kit, e.c(4)), b.lshr(e.a.lane, e.c(4)))))

    sa_of = {}
    if k.asop:
        for j in range(0, n, 4):
            words = [a_sc(kits[i]) for i in range(j, min(n, j + 4))]
            words += [e.c(127)] * (4 - len(words))
            packed = e.pack4(words)
            for q in range(min(4, n - j)):
                sa_of[j + q] = (packed, q)
    ops = {}

    def fetch(i):
        if i >= n or i in ops:
            return
        ops[i] = (a_op(kits[i]), sa_of.get(i) or (a_sc(kits[i]), 0))

    acc = b.zero_vec_f32(4)
    for i in range(min(k.ra, n)):
        fetch(i)
    for i, (_, w, (sb, selb)) in enumerate(steps):
        fetch(i)
        fetch(i + k.ra)
        aop, (sa, sela) = ops.pop(i)
        # C1's weight scales are VALU results: the mono's shift (every step) or the packed dwords (asop + bsop:
        # first use of each, step 4 j); the zero accumulator at step 0
        packed = bool(k.bsop) and bool(k.asop)
        acc = e.mma(acc, aop, w, sa, sb, sela, selb, last=(i == n - 1), valu_src=(i % 4 == 0) if packed else True)
        if k.sgb == "mfma1":
            b.sched_group_barrier(0x100, 2, 0)
            b.sched_group_barrier(0x008, 1, 0)
        elif k.sgb == "mfma2":
            b.sched_group_barrier(0x100, 3, 0)
            b.sched_group_barrier(0x008, 1, 0)
    return acc


# ----------------------------------------------------------------------------- P1


def _p1(e: _E) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    tid, lane, wave = a.tid, a.lane, a.wave
    rows = e.rows
    t = a.task
    KI = _k_iters(g.hidden)  # 20
    STREAM_VECS = g.hidden // 8  # 320
    KEY_NB = g.hc_width // 16  # 640: out of range
    key_w_bytes = g.hc_width * g.hidden // 2
    key_s_bytes = g.hc_width // 16 * (KI // 4) * 64 * 4
    nst = len(g.ple_state)
    ent = e.table(a.ple)

    def st(name, r=0):
        return ent(r * nst + g.ple_state.index(name))

    s_emb, s_kw, s_ks, s_vw, s_vs = st("emb"), st("key_w"), st("key_s"), st("val_w"), st("val_s")
    xr, er = e.mfma_operands("p_x8", "p_xe")
    key = e.wave_lt(4)
    nb = b.select(key, b.add(b.mul(t, e.c(4)), wave), b.select(b.cmp_eq(wave, e.c(4)), t, e.c(KEY_NB)))
    rs_w = e.rsrc(b.select(key, s_kw, s_vw), key_w_bytes)
    rs_s = e.rsrc(b.select(key, s_ks, s_vs), key_s_bytes)
    r_emb = [e.rsrc(s_emb)] + [e.rsrc(st("emb", r)) for r in range(1, rows)]

    with b.scf_if(b.cmp_lt(t, e.c(g.tasks["p1"]))):
        _split(e, 5, lambda role: _p1_body(e, role, xr, er, key, nb, rs_w, rs_s, r_emb))


def _p1_body(e: _E, role, xr, er, key, nb, rs_w, rs_s, r_emb) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    tid, lane, wave = a.tid, a.lane, a.wave
    rows = e.rows
    KI = _k_iters(g.hidden)
    STREAM_VECS = g.hidden // 8
    if True:
        vec = b.mul(b.smin(tid, e.c(STREAM_VECS - 1)), e.c(4))
        xs = [e.load(r_emb[r], b.mul(vec, e.c(4)), 4) for r in range(rows)]
        b.sched_barrier(0)
        e.barrier()
        # Waves 5-7 read out of range (n-block 640): zeros, no memory traffic (``oob=split``: no loads at all).
        units = [e.issue(rs_w, rs_s, nb, KI, gi, e.c(0)) for gi in range(KI // 4)] if role != "idle" else None
        for rp in range(rows):
            words, ex = _pack_vec(e, xs[rp])
            with b.scf_if(e.wave_lt(5)):
                e.stv(e.ent("p_x8", b.mul(tid, e.c(2)), rp), words)
                with b.scf_if(b.cmp_eq(b.land(lane, e.c(3)), e.c(0))):
                    e.stw(e.ent("p_xe", b.lshr(tid, e.c(2)), rp), ex)
        e.barrier()

        if role == "idle":
            return
        with b.scf_if(key):
            steps = []
            for gi, (ws, sw) in enumerate(units):
                for q in range(4):
                    steps.append((gi * 4 + q, ws[q], (sw, q)))
            acc = e.chain(steps, xr, er)
            for rk in range(rows):
                src = 0 if k.mutate == "row" else rk
                _put_block(e, "pk", nb, b.vec_extract(acc, src % 4), rk)
        with b.scf_if(b.cmp_eq(wave, e.c(4))):
            vs = _value_chains(e, units, xr, er)
            for rv in range(rows):
                _put_block(e, "pv", nb, vs[rv], rv)


def _value_chains(e: _E, units, xr, er):
    """ATOM ``ple._value_chains``: chain ``q`` over k_iters ``q, q + 4, ...``, then ((c0 + c1) + c2) + c3."""
    b, k = e.b, e.k
    accs = []
    for q in range(4):
        steps = [(gi * 4 + q, ws[q], (sw, q)) for gi, (ws, sw) in enumerate(units)]
        accs.append(e.chain(steps, xr, er, tail=(k.mfma == "op")))
    if k.mfma == "asm":
        accs = e.tail_fence(accs)
    out = []
    for r in range(e.rows):
        c = [b.vec_extract(x, r % 4) for x in accs]  # wide: row r is acc[r % 4] of its lanes (``ple_wide``)
        if k.mutate == "vsum":
            out.append(b.fadd(b.fadd(b.fadd(c[0], c[2]), c[1]), c[3]))
        else:
            out.append(b.fadd(b.fadd(b.fadd(c[0], c[1]), c[2]), c[3]))
    return out


def _put_block(e: _E, region: str, nb, v, r) -> None:
    """ATOM ``ple._put_block`` (lanes 0-15 hold the 16 outputs); wide builds ``ple_wide._put_block_wide``
    (request row ``r`` on lanes ``16 (r // 4) ..``)."""
    b, lane = e.b, e.a.lane
    out = e.bf16r(v)
    other = e.xshfl1(out)
    even = b.cmp_eq(b.land(lane, e.c(1)), e.c(0))
    if e.wide:
        ok = b.land(b.cmp_eq(b.lshr(lane, e.c(4)), e.c(r // 4)), even)
        col = b.lshr(b.land(lane, e.c(15)), e.c(1))
    else:
        ok = b.land(b.cmp_lt(lane, e.c(16)), even)
        col = b.lshr(lane, e.c(1))
    with b.scf_if(ok):
        e.put(region, e.mbix(b.add(b.mul(nb, e.c(8)), col), r), M.bf16_pair(b, out, other))


def _max3(e: _E, a, b_, c, absm=(0, 0, 0), nop=False):
    """``v_maximum3_f32`` (IEEE maximum of three: NaN-propagating, -0 < +0; what LLVM selects for
    ``llvm.maximum.f32`` on gfx950) as inline asm; ``nop``: two trailing wait states (a DPP reads the result next,
    and the asm is opaque to the hazard recognizer)."""
    ops = [f"|${i + 1}|" if m else f"${i + 1}" for i, m in enumerate(absm)]
    t = f"v_maximum3_f32 $0, {ops[0]}, {ops[1]}, {ops[2]}" + ("\n\ts_nop 1" if nop else "")
    return e.b.inline_asm(t, "=v,v,v,v", [a, b_, c], F32, sideeffect=False, result_name_hint="max3")


def _pack_vec(e: _E, x):
    """ATOM ``ple._pack_vec``: thread ``tid``'s 8 bf16 values -> (2 fp8 words, the 32-block's E8M0 exponent)."""
    b = e.b
    y = []
    for i in range(4):
        w = b.vec_extract(x, i)
        y += list(M.bf2_f32(b, w))
    if e.k.pmax == "max3":
        # FlyDSL's instructions (``llvm.maximum.f32`` on gfx950): v_maximum3_f32 with |.| source modifiers, the
        # chain max(|y0|, |y1|), then two values per step; the lane maxima as max3(m, x, x)
        amax = _max3(e, y[0], y[1], y[1], absm=(1, 1, 1))
        for i in (2, 4, 6):
            amax = _max3(e, amax, y[i], y[i + 1], absm=(0, 1, 1), nop=(i == 6))
        x = e.xshfl(amax, 1)
        amax = _max3(e, amax, x, x, nop=True)
        x = e.xshfl(amax, 2)
        amax = _max3(e, amax, x, x)
    else:
        amax = b.fabs(y[0])
        for i in range(1, 8):
            amax = e.maximum(amax, b.fabs(y[i]))
        amax = e.maximum(amax, e.xshfl(amax, 1))
        amax = e.maximum(amax, e.xshfl(amax, 2))
    bits = b.bitcast(b.fmul(amax, e.fbits(RATIO_BITS)), I32)
    ex = b.add(
        b.land(b.lshr(bits, e.c(23)), e.c(0xFF)),
        b.select(b.cmp_ne(b.land(bits, e.c(0x7FFFFF)), e.c(0)), e.c(1), e.c(0)),
    )
    ex = b.smin(b.smax(ex, e.c(1)), e.c(254))
    scale = b.bitcast(b.shl(ex, e.c(23)), F32)
    words = []
    for h in range(2):
        wd = e.c(0)
        wd = b.cvt_scalef32_pk_fp8_f32(wd, y[4 * h], y[4 * h + 1], scale, hi=False)
        wd = b.cvt_scalef32_pk_fp8_f32(wd, y[4 * h + 2], y[4 * h + 3], scale, hi=True)
        words.append(wd)
    return b.vec_pack(words, I32), ex


# ----------------------------------------------------------------------------- build / lower / write


def _abi_params():
    from ..stage_fn import ABI_PARAMS

    return ABI_PARAMS


def _pool_words(g: GemvGeom, shape: GemvShape, k: GemvKnobs) -> int:
    lds, stride, words = g.lds(shape.rows, shape.seq, shape.group)
    names = {"b1": ("red", "x8", "xe"), "q1": ("red", "x8", "xe"), "c1": ("red", "x8", "xe"), "p1": ("p_x8", "p_xe")}
    need = max(lds[n] + stride[n] * shape.rows for n in names[shape.stage])
    if shape.stage == "b1" and k.host_b3pre:
        from .. import stage_b3 as SB

        bk = SB.knobs_from_spec(k.host_b3pre)
        arena = SB._arena(SB.B3Shape(shape.rows, shape.seq, 0), bk, pad=SB.MONO_B3_LDS_START[shape.rows])
        need = max(need, arena.words)
    if need > words:
        raise ValueError(f"{shape.stem}: LDS reach {need} words past the mono arena {words}")
    return need


def build_gemv_kernel(shape: GemvShape, knobs: GemvKnobs, g: GemvGeom | None = None):
    """The stage function's ``KernelDef`` (``stage_fn.ABI_PARAMS``, then the body)."""
    g = g or geom_from_atom()
    if knobs.mutate == "conv_tap" and shape.stage != "b1":
        raise ValueError("mutant conv_tap is B1's")
    if knobs.mutate == "resid" and shape.stage != "c1":
        raise ValueError("mutant resid is C1's")
    if knobs.mutate == "vsum" and shape.stage != "p1":
        raise ValueError("mutant vsum is P1's")
    if knobs.mutate == "row" and shape.rows == 1:
        raise ValueError("mutant row needs rows > 1")
    if knobs.host_b3pre and (shape.stage != "b1" or shape.group):
        raise ValueError("host_b3pre is B1's, ungrouped builds (the grouped builds' B3 has no prologue)")
    if knobs.bsop and shape.stage != "c1":
        raise ValueError("bsop is C1's (the other stages' weight scales already use op_sel)")
    if knobs.store_out and shape.stage != "c1":
        raise ValueError("store_out is C1's")
    if knobs.epi == "reg" and (shape.stage not in ("b1", "q1") or shape.group or shape.rows > 4):
        raise ValueError("epi=reg: B1 / Q1, ungrouped builds of up to 4 rows")
    if knobs.cprio and shape.stage == "c1":
        raise ValueError("cprio: B1 / Q1 / P1 chains (C1's chain has its own emitter)")
    b = IRBuilder(shape.symbol)
    b.kernel.attrs["max_workgroup_size"] = g.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in _abi_params():
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer", "l0", "l1", "l2", "l3"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(
        **vals, tid=tid, lane=b.land(tid, b.const_i32(63)), wave=b.readfirstlane(b.lshr(tid, b.const_i32(6)))
    )
    pw = _pool_words(g, shape, knobs)
    e = _E(b, a, g, shape, knobs, pw)
    e.pool_words_total = pw
    {"b1": _b1, "q1": _q1, "c1": _c1, "p1": _p1}[shape.stage](e)
    b.ret()
    return b.kernel


def lower_gemv_function(shape: GemvShape, knobs: GemvKnobs, g: GemvGeom | None = None, *, arch="gfx950",
                        flavor=None, fn_knobs=None):  # fmt: skip
    return lower_device_function(
        build_gemv_kernel(shape, knobs, g),
        arch=arch,
        symbol=shape.symbol,
        llvm_flavor=flavor,
        knobs=fn_knobs or DeviceFunctionKnobs(lds="arg", inline="always"),
    )


def sidecar_extra(shape: GemvShape, knobs: GemvKnobs) -> dict:
    meta = {"stage_knobs": knobs_spec(knobs), "kind": shape.kind, "gemv_port": "WP2b"}
    if knobs.numa:
        meta.update({"mailboxes": "numa1", "numa_hdr": knobs.header_offset})
    if knobs.host_b3pre:
        meta["hosts"] = {"b3pre": knobs.host_b3pre}
    return meta


def write_gemv_functions(out_dir: str, shapes, knobs_for, g: GemvGeom | None = None, flavor=None) -> list:
    """Write ``<stage>_<kind>_<variant>.{ll,json}`` per shape into ``out_dir``. ATOM:
    ``ATOM_Q38_EXTERN_STAGES=<stage>=<out_dir>/<stage>_{kind}_{variant}.ll`` (numa=1 builds: run with the ``numa=1``
    schedule lever, as the B3 / MoE functions)."""
    g = g or geom_from_atom()
    os.makedirs(out_dir, exist_ok=True)
    stems = []
    for shape in shapes:
        knobs = knobs_for(shape)
        fn = lower_gemv_function(shape, knobs, g, flavor=flavor)
        paths = write_device_function(fn, out_dir, stem=shape.stem, bitcode=False)
        with open(paths["json"]) as f:
            meta = json.load(f)
        meta.update(sidecar_extra(shape, knobs))
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        stems.append(shape.stem)
    return stems


def parse_variant(v: str):
    m = re.fullmatch(r"r(\d+)(s?)(?:g(\d+))?", v.strip())
    if not m:
        raise ValueError(f"bad variant {v!r}")
    return int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0)


NARROW = ("r1", "r2", "r3", "r4", "r2s", "r3s", "r4s")
GROUPED = ("r4g2", "r6g2", "r8g2", "r6g3", "r9g3", "r12g3", "r8g4", "r12g4", "r16g4")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE GEMV-family stage functions (B1 / C1 / Q1 / P1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="b1,c1,q1,p1")
    ap.add_argument("--kinds", default="", help="default: every kind the stage runs in")
    ap.add_argument("--variants", default=",".join(NARROW))
    ap.add_argument("--knobs", default="", help="k=v,... (GemvKnobs)")
    ap.add_argument("--b3pre", default="", help="B1: the B3 knob spec whose pre=dma prologue B1 hosts")
    ap.add_argument("--atom-root")
    ap.add_argument("--flavor")
    a = ap.parse_args(argv)
    g = geom_from_atom(a.atom_root)
    knobs = knobs_from_spec(a.knobs)
    shapes = []
    for st in a.stages.split(","):
        kinds = a.kinds.split(",") if a.kinds else STAGES[st]
        for kd in kinds:
            if kd not in STAGES[st]:
                continue
            for v in a.variants.split(","):
                rows, seq, group = parse_variant(v)
                shapes.append(GemvShape(st, kd, rows, seq, group))

    def knobs_for(s):
        return replace(knobs, host_b3pre=a.b3pre) if (s.stage == "b1" and a.b3pre) else knobs

    stems = write_gemv_functions(a.out, shapes, knobs_for, g, flavor=a.flavor)
    print("wrote", len(stems), "functions:", " ".join(stems))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ----------------------------------------------------------------------------- path b: rocKE layer kernels


def framework_factory(stage: str, knobs: GemvKnobs = GemvKnobs(), g: GemvGeom | None = None):  # noqa: B008
    """A ``framework`` stage factory (``framework.build_layer(factories={stage: ...})``) running this module's
    emitter in a rocKE layer kernel (Path B), as ``stage_adapt.stage_fn_factory`` runs a ``stage_fn`` body: every CTA
    calls it once with its round-0 task (``per_cta``), the ABI values from the framework (``stage_adapt._abi``). In a
    kernel build the emitter's arena is its own ``smem_alloc`` (the lowering packs the per-stage pools); entries keep
    ATOM's offsets inside it. ``numa`` must stay 0 there (the NUMA area is ATOM's)."""
    from ..framework import StageSpec as FwStage
    from ..stage_adapt import _abi

    if knobs.numa:
        raise ValueError("framework builds address the plain mailboxes (numa=0)")

    def factory(kind: str, shape, st):
        gg = g or geom_from_atom()
        sh = GemvShape(stage, kind, shape.rows, shape.seq, shape.group)
        pw = _pool_words(gg, sh, knobs)

        def emit(b, ctx, p, t) -> None:
            a = _abi(b, ctx, p, t, [b.const_i32(0)] * 4)
            old = b.fence_barriers
            b.fence_barriers = "compiler"
            try:
                e = _E(b, a, gg, sh, knobs, pw)
                e.pool_words_total = pw
                {"b1": _b1, "q1": _q1, "c1": _c1, "p1": _p1}[stage](e)
            finally:
                b.fence_barriers = old

        return FwStage(stage=st, emit=emit, per_cta=True)

    return factory
