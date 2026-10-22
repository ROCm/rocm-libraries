# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""B3 of the mono DeltaNet layer (gated delta rule decode) as a rocKE stage.

Re-implementation of ATOM ``kernels/stages/delta.py`` ``b3`` (and the grouped
form ``delta_group.b3``) with the same contract, as ONE emitter that serves

1. the standalone isolated-stage kernel (:func:`build_b3_kernel`; the host
   fills the mailboxes B3 reads; mono's launch ABI) used for sweeps and the
   A/B against the mono's own B3,
2. the hybrid device function (``stage_fn`` registry, stage ``"b3"``), linked
   into the FlyDSL mono layer kernel in place of its B3,
3. the rocKE layer kernel (:func:`b3_stage_spec` -> ``framework.StageSpec``).

Contract (mono B3, rows ``R`` <= 4; decode, seq = MTP verify, grouped verify):

* inputs: the attention HC's bf16 mixed row (``xbf``, A2), B1's bf16 qkvz
  projection after conv + SiLU (``qkvz``), the per-row DeltaNet state table
  ``conv_in`` (``[rows][conv_in, conv_out, rec_in, rec_out]`` i64), ``ba_w``
  ``[2 Hv, H]`` bf16, ``a_log`` / ``dt_bias`` ``[Hv]`` f32, ``norm_w`` ``[Dv]``
  bf16;
* per value head ``vh``: ``b``, ``a`` as bf16-rounded 16-part dots of the
  mixed row (the router's part split, fixed order), ``beta = sigmoid(b)``,
  ``decay = exp(-exp(a_log) softplus(a + dt_bias))``; q / k L2 norms (q also
  scaled by ``rsq(128)``); the f32 VK state ``S <- decay S; S += beta (v - S
  k) k^T; core = S q`` per state row with the native 16-lane trees; state
  snapshot per row to ``rec_out`` (seq / grouped: the rows chain, only a
  chain's first row reads ``rec_in``);
* gated RMSNorm over the head's 128 core rows times ``norm_w`` times
  ``sigmoid(z)``, rounded to bf16, then the MXFP8 pack (E4M3 words in ``d8``,
  one E8M0 exponent per 32 rows in ``de``);
* the head's 128 rows are split over ``split`` sibling CTAs (mono: 4); with
  ``split > 1`` the siblings exchange their f32 core rows through ``core``.

Every floating-point operation is the mono's, in the mono's order, without
fast-math flags, so the results are bitwise equal to the mono's B3 when both
are compiled by the same LLVM; lane exchanges use other instructions where
convenient (an exchange moves bits and each butterfly step is the commutative
``v + partner``).

Knobs (:class:`B3Knobs`; every default is the mono's behaviour, the kernel /
symbol name names only deviations). Numerics-preserving: ``split``,
``head_map``, ``wait``, ``stores``, ``state_nt``, ``barrier``, ``shfl4``,
``cvt``, ``mailbox``, ``numa`` (the mailbox layout of ATOM's numa=1 lever). ``exp="hw"`` changes the decay's exp (not bitwise).

Python-engine-only ops: ``math.exp`` (the decay's ``llvm.exp.f32``, the
mono's ``math.exp``) is in every default build, so these builds compile with
``backend="python"`` and carry the ``_pyonly`` name suffix (family README,
"cpp-parity exemption"); ``exp="hw"`` builds without ``state_nt`` / ``ir``
mailboxes / ``cvt="intrinsic"`` use mirrored ops only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from rocke.core.ir import (
    CPOL_NT,
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    Value,
    VectorType,
    uses_python_only_ops,
)
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as mxq
from rocke.helpers.lds_arena import LdsArena, LdsEntry, LdsView

__all__ = [
    "B3_REGIONS",
    "GEOMETRY",
    "TUNED_SPECS",
    "TUNED_SPECS_NUMA",
    "VERIFIED_VARIANTS",
    "B3Knobs",
    "B3Shape",
    "b3_head",
    "b3_kernel_name",
    "b3_lds_entries",
    "b3_stage_spec",
    "b3_tasks",
    "b3_tiles",
    "build_b3_function_kernel",
    "build_b3_kernel",
    "emit_b3",
    "emit_b3_framework",
    "emit_b3_pre",
    "knobs_from_spec",
    "knobs_spec",
    "lower_b3_function",
    "ref_layout",
    "sidecar_numa",
    "tuned_knobs",
    "write_b3_functions",
]

# ------------------------------------------------------------------ geometry
# ATOM ``kernels/config.py`` DEFAULT (Qwen3.8-Flash-Next at TP1).
GEOMETRY = SimpleNamespace(
    hidden=2560,
    k_heads=16,
    v_heads=48,
    k_dim=128,
    v_dim=128,
    eps=1e-6,
)
H = GEOMETRY.hidden
HK, HV, DK, DV = 16, 48, 128, 128
GQA = HV // HK  # 3
KEY = HK * DK  # 2048
VALUE = HV * DV  # 6144
QKV = 2 * KEY + VALUE  # 10240: channels through the conv (z follows)
EPS = 1e-6
GRID, THREADS, WAVES = 256, 512, 8
MONO_SPLIT = 4  # mono's CTAs per value head
B1_TILE_PAIRS = 32  # one B1 task publishes 64 channels = 32 pairs

# The router's 16-part K split (ATOM ``layout.ROUTER_PARTS``); B3 computes b
# and a with it. Parts 0-7 hold 192 values, parts 8-15 128.
ROUTER_PARTS = tuple((192 * p, 192) if p < 8 else (128 * p + 512, 128) for p in range(16))
RW_PART = 100  # words (bf16 pairs) per weight part, padded
RW_ROW = len(ROUTER_PARTS) * RW_PART  # 1600
RX_PART = 196  # f32 per mixed-row part, padded
CHAIN_STEPS = max(n for _, n in ROUTER_PARTS) // 8  # 24
SHORT_STEPS = min(n for _, n in ROUTER_PARTS) // 8  # 16
BIG_FIRST = ROUTER_PARTS[8][0]  # 1536: first k of the short parts

RSQ128 = 0x3DB504F3  # v_rsq_f32(128.0)
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# Mailbox regions B3 touches (ATOM ``layout.SC`` names).
B3_REGIONS = ("xbf", "qkvz", "core", "d8", "de")

SPLITS = (4, 2, 1)
HEAD_MAPS = ("linear", "xcd")
WAITS = ("pre_poll", "barrier")
STORES = ("early", "after_put", "end")
BARRIERS = ("lds", "full")
SHFL4 = ("rowshift", "swizzle")
EXPS = ("llvm", "hw")
CVTS = ("asm", "intrinsic")
STATE_ISSUES = ("after_ba", "entry")
NORMS = ("split", "redundant")
CORE_PUTS = ("lds", "direct")
XCHGS = ("core", "numa2")
TABLES = ("smem", "global")
HOISTS = ("none", "gates", "gates_z")
PRES = ("none", "dma")
DMA_CHUNK = 1024  # bytes one wave moves per LDS-DMA instruction (64 lanes x 16 B)
BAW_RAW_WORDS = 2 * (H // 2) + 32  # b row, a row (bf16 words), padded: steps past a short part read on
# ``xchg="numa2"``: the sibling exchange in a private region of the scratch's
# second half (``layout.mailbox_bytes``: the QSA layers' index scores, unused
# by a DeltaNet launch), 16 KiB blocks of 1024 pairs whose copy sits 8 KiB on
# (``mailbox.NUMA_COPY_BYTES``): every put writes both copies, a consumer
# polls the copy homed at its own IOD (``mailbox.local_copy``, flip 0:
# uncalibrated). Byte offset of the region and the pairs of one request row.
XCHG_OFFSET = 312 * 16384  # first 16 KiB boundary past MAX_ROWS * SCRATCH_BYTES
XCHG_BLOCK_PAIRS = M.NUMA_COPY_BYTES // M.PAIR_BYTES  # 1024
XCHG_ROW_PAIRS = 2 * (DV * HV // XCHG_BLOCK_PAIRS) * XCHG_BLOCK_PAIRS  # 12288
# knob ``numa=1``: ATOM's NUMA area (``kernels/numa.py``): the header past the
# mailbox part and the debug regions (:attr:`B3Knobs.header_offset`; word 0 the
# calibrated page mask), the area at the next 2 MiB boundary past header + 256.
NUMA_PAGE = 1 << 21
NUMA_CHUNK = 8192


@dataclass(frozen=True)
class B3Knobs:
    """Build knobs of B3 (defaults = the mono's B3).

    split: CTAs per value head (4 = mono; 2; 1 = one CTA holds the head's 128
      state rows and needs no sibling exchange, 48 tasks).
    head_map: task -> (value head, split) map: ``linear`` (mono: ``vh = t //
      split``) | ``xcd`` (the siblings of a head sit on CTAs of one XCD, so
      their core exchange stays in that XCD's L2; tasks on CTA ``bid`` with
      the dispatch's round-robin ``bid % 8`` XCD assignment).
    wait: the wait ahead of the qkvz poll: ``pre_poll`` (mono: wave 0 spins on
      the 6 B1 tiles' last pairs, then a barrier, then the payload poll) |
      ``barrier`` (only a barrier: the payload poll spins itself, one round
      trip after B1's stores instead of two).
    stores: when the state snapshots go out: ``early`` (mono: right after the
      recurrence, ahead of the core exchange) | ``after_put`` (after the core
      put) | ``end`` (after the gated output is published; the epilogue's
      poll does not wait behind them).
    state_nt: ``nt`` (streaming) bit on the snapshot stores (Python-only op).
    barrier: CTA barriers that publish LDS writes to other waves: ``lds``
      (``sync_lds_only``: ``lgkmcnt(0)`` then ``s_barrier``; gfx950's back-off
      barrier does not wait for the counters itself) | ``full`` (``sync``:
      also ``vmcnt(0)``, i.e. every outstanding load and store, as a barrier
      with a workgroup-scope fence may compile to).
    shfl4: the xor-4 lane exchange of the 16-lane trees: ``rowshift`` (two
      bank-masked DPP row shifts, inline asm, as the mono's ``xshfl``) |
      ``swizzle`` (``ds_swizzle``).
    exp: the decay's exp: ``llvm`` (``llvm.exp.f32``, the mono's) | ``hw``
      (``v_exp_f32(x log2 e)``; numerics change, not bitwise).
    cvt: the MXFP8 convert route (``rocke.helpers.mx_quant``): ``asm`` |
      ``intrinsic`` (Python-only op).
    state_issue: where the state loads issue: ``after_ba`` (mono: after the
      b / a staging barrier) | ``entry`` (at the stage entry, ahead of the
      b / a weight loads and the mixed row's poll).
    norms: q / k L2 norms of a batched build (rows > 1): ``split`` (mono:
      wave ``2 r + n`` computes one, then a barrier) | ``redundant`` (every
      wave computes all, as the one-row build does: no barrier).
    core_put: the sibling exchange's put (split > 1): ``lds`` (mono: the core
      rows through LDS, a barrier, 16-byte puts of two rows) | ``direct``
      (each row's lane 0 puts its own row from registers, no barrier; the
      epilogue reads the task's own rows from the polled head).
    xchg: where the sibling exchange lives (split > 1): ``core`` (mono: the
      ``core`` region, one copy) | ``numa2`` (a private region in 16 KiB
      blocks with a copy per IOD; :data:`XCHG_OFFSET`).
    table: how the per-row state-address table is read: ``smem`` (mono: scalar
      loads; here one inline-asm ``s_load_dwordx4`` of each row's
      ``rec_in, rec_out`` and one ``lgkmcnt(0)``) | ``global`` (compiler-chosen
      loads: a vector load wherever LLVM cannot prove the table unclobbered,
      and its ``vmcnt`` wait drains every earlier vector access).
    hoist: work moved ahead of the polls it does not depend on: ``none``
      (mono: the gates after the qkvz poll, ``sigmoid(z)`` after the core
      exchange) | ``gates`` (an LDS barrier after the b / a chains, then the
      gates and the chain heads' ``decay * S`` before the qkvz wait) |
      ``gates_z`` (also ``sigmoid(z)`` of the epilogue's rows right after the
      qkvz poll, ahead of the recurrence and the exchange). Same operations,
      same order per value: bitwise unchanged.
    pre: work issued in an earlier stage (hybrid: the mono calls the
      ``rocke_q38_b3pre_<variant>`` function inside B1, after B1's weight
      loads and before its wait for the mixed row): ``none`` (mono) | ``dma``
      (the chain heads' state rows and the b / a weight rows go to LDS by
      buffer->LDS DMA there, so B3 starts with them on chip; the b / a chains
      read the weights unpadded). Same values: bitwise unchanged. Needs the
      LDS for it (:func:`b3_lds_entries`); the isolated kernel runs the
      prologue at its own start.
    mailbox: :class:`rocke.helpers.mailbox.MailboxKnobs` of every hand-off.
    numa: 0 | 1 = the mailbox layout of a mono build with ATOM's ``numa=1``
      schedule lever (``kernels/numa.py``, NUMA dual-copy mailboxes): every
      region B3 touches (``xbf``, ``qkvz``, ``d8``, ``de`` and, with
      ``xchg="core"``, the sibling exchange ``core``) is addressed in ATOM's
      NUMA area: puts store both copies (byte ``x`` of the mailbox part at
      ``x + (x & ~8191)`` and ``+ 8192``), polls read the copy homed at this
      CTA's IOD (the header's calibrated page mask). Hybrid function only (the
      sidecar says ``"mailboxes": "numa1"``; ATOM keeps the lever for the
      build and refuses the build without it). ``xchg="numa2"`` keeps its
      private region in the plain scratch (unused under the lever).
    numa_hdr: byte offset of ATOM's NUMA header (``numa.header_offset()``);
      0 = :data:`moe_fused.NUMA_HDR` (this tree's layout). ``numa=1`` only.
    """

    split: int = MONO_SPLIT
    head_map: str = "linear"
    wait: str = "pre_poll"
    stores: str = "early"
    state_nt: bool = False
    barrier: str = "lds"
    shfl4: str = "rowshift"
    exp: str = "llvm"
    cvt: str = "asm"
    state_issue: str = "after_ba"
    norms: str = "split"
    core_put: str = "lds"
    xchg: str = "core"
    table: str = "smem"
    hoist: str = "none"
    pre: str = "none"
    mailbox: M.MailboxKnobs = field(default_factory=M.MailboxKnobs)
    numa: int = 0
    numa_hdr: int = 0

    def __post_init__(self) -> None:
        if self.numa not in (0, 1):
            raise ValueError(f"numa must be 0 or 1, got {self.numa!r}")
        if self.numa_hdr < 0 or self.numa_hdr % 256 or (self.numa_hdr and not self.numa):
            raise ValueError("numa_hdr must be a non-negative multiple of 256, and needs numa=1")
        for name, allowed in (
            ("split", SPLITS), ("head_map", HEAD_MAPS), ("wait", WAITS),
            ("stores", STORES), ("barrier", BARRIERS), ("shfl4", SHFL4),
            ("exp", EXPS), ("cvt", CVTS), ("state_issue", STATE_ISSUES), ("norms", NORMS),
            ("core_put", CORE_PUTS), ("xchg", XCHGS), ("table", TABLES), ("hoist", HOISTS),
            ("pre", PRES),
        ):  # fmt: skip
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.split == 1 and self.stores == "after_put":
            raise ValueError("stores='after_put' needs split > 1 (a core put)")
        if self.split == 1 and self.core_put != "lds":
            raise ValueError("core_put applies to split > 1 (a core exchange)")
        if self.split == 1 and self.xchg != "core":
            raise ValueError("xchg applies to split > 1 (a core exchange)")

    def suffix(self) -> str:
        d = B3Knobs()
        parts = []
        if self.split != d.split:
            parts.append(f"sp{self.split}")
        if self.head_map != d.head_map:
            parts.append(f"hm{self.head_map}")
        if self.wait != d.wait:
            parts.append(f"w{self.wait}")
        if self.stores != d.stores:
            parts.append(f"st{self.stores}")
        if self.state_nt:
            parts.append("snt")
        if self.barrier != d.barrier:
            parts.append(f"b{self.barrier}")
        if self.shfl4 != d.shfl4:
            parts.append(f"x4{self.shfl4}")
        if self.exp != d.exp:
            parts.append(f"e{self.exp}")
        if self.cvt != d.cvt:
            parts.append(f"cv{self.cvt}")
        if self.state_issue != d.state_issue:
            parts.append(f"si{self.state_issue}")
        if self.norms != d.norms:
            parts.append(f"n{self.norms}")
        if self.core_put != d.core_put:
            parts.append(f"cp{self.core_put}")
        if self.xchg != d.xchg:
            parts.append(f"xc{self.xchg}")
        if self.table != d.table:
            parts.append(f"tb{self.table}")
        if self.hoist != d.hoist:
            parts.append(f"ho{self.hoist}")
        if self.pre != d.pre:
            parts.append(f"pre{self.pre}")
        ms = self.mailbox.suffix()
        if ms:
            parts.append(ms)
        if self.numa:
            parts.append(f"numa{self.numa}")
        if self.numa_hdr:
            parts.append(f"nh{self.numa_hdr}")
        return "_".join(parts)

    @property
    def header_offset(self) -> int:
        """ATOM's NUMA header offset this build addresses (``numa=1``)."""
        if not self.numa:
            raise ValueError("header_offset: numa=0 builds do not use the NUMA area")
        if self.numa_hdr:
            return self.numa_hdr
        from .moe_fused import NUMA_HDR

        return NUMA_HDR

    @property
    def bitwise(self) -> bool:
        """The build computes the mono's bits."""
        return self.exp == "llvm"


@dataclass(frozen=True)
class B3Shape:
    """A build's request rows: ``rows`` (1..4); ``seq`` (MTP verify: the rows
    are consecutive tokens of one request, ``rows >= 2``); ``group`` S (a
    grouped verify of ``rows / S`` requests, ``rows >= 2 S``)."""

    rows: int = 1
    seq: bool = False
    group: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.rows <= 4:
            raise ValueError(f"rows must be in 1..4, got {self.rows}")
        if self.seq and self.rows < 2:
            raise ValueError("a seq build has 2 rows at least")
        if self.group and (
            self.seq or self.group < 2 or self.rows % self.group or self.rows < 2 * self.group
        ):
            raise ValueError(f"bad group {self.group} for rows {self.rows}")

    @property
    def chain(self) -> int:
        """Rows per recurrence chain: 1 (independent rows), rows (seq), S."""
        if self.seq:
            return self.rows
        return self.group or 1

    @property
    def variant(self) -> str:
        return f"r{self.rows}" + ("s" if self.seq else "") + (f"g{self.group}" if self.group else "")


def b3_tasks(knobs: B3Knobs = B3Knobs()) -> int:  # noqa: B008
    return HV * knobs.split


def b3_head(t: int, knobs: B3Knobs = B3Knobs()) -> tuple[int, int]:  # noqa: B008
    """Host model of the task -> ``(value head, split)`` map (``head_map``)."""
    S = knobs.split
    if knobs.head_map == "linear":
        return t // S, t % S
    per = HV * S // 8
    x, j = t % 8, t // 8
    assert j < per
    return x * (per // S) + j // S, j % S


def b3_tiles(kh: int, vh: int, s: int, split: int) -> tuple[int, ...]:
    """The B1 tiles (64 channels each) a task reads: q and k of key head
    ``kh`` (2 each), then the v and z tiles of rows ``s * 128 / split ..`` of
    value head ``vh`` (ATOM ``layout.b3_tiles`` at split 4). Host model."""
    rt = DV // split
    q = 2 * kh
    k = KEY // 64 + 2 * kh
    first = (vh * DV + s * rt) // 64
    n = max(1, rt // 64)
    v = tuple(2 * KEY // 64 + first + i for i in range(n))
    z = tuple(QKV // 64 + first + i for i in range(n))
    return (q, q + 1, k, k + 1, *v, *z)


def b3_lds_entries(shape: B3Shape, knobs: B3Knobs) -> tuple[LdsEntry, ...]:
    """B3's LDS entries (one phase), in the mono's ``_DELTA_TABLE`` order."""
    rt = DV // knobs.split
    ents = [
        LdsEntry("b3_qk", 2 * DK, True),  # q then k of the key head (f32)
        LdsEntry("b3_vz", 2 * rt, True),  # v then z of the task's rows (f32)
        LdsEntry("b3_core", rt, True),  # the task's core rows (f32)
        LdsEntry("b3_sums", 4 * WAVES, True),  # b, a, inv_q, inv_k
    ]
    if knobs.split > 1:
        ents.append(LdsEntry("b3_cores", DV, True))  # the head's 128 core rows
    if knobs.pre == "dma":
        heads = shape.rows // shape.chain
        ents += [
            LdsEntry("b3_bawr", BAW_RAW_WORDS),  # b and a weight rows, unpadded (bf16)
            LdsEntry("b3_spre", heads * rt * DK),  # the chain heads' state rows (f32)
        ]
    else:
        ents.append(LdsEntry("b3_baw", 2 * RW_ROW))  # b and a weight rows by part (bf16)
    ents.append(LdsEntry("b3_bax", len(ROUTER_PARTS) * RX_PART, True))  # mixed row by part
    return tuple(ents)


def b3_kernel_name(shape: B3Shape, knobs: B3Knobs, prefix: str = "rk_q38_b3") -> str:
    parts = [prefix, shape.variant]
    s = knobs.suffix()
    if s:
        parts.append(s)
    return "_".join(parts)


def ref_layout() -> SimpleNamespace:
    """The mono scratch layout B3 uses (``abi.SC``, ``abi.ROW_PAIRS``),
    mirrored from the same formulas as ATOM ``layout.py``."""
    from . import abi

    return SimpleNamespace(
        sc={k: abi.SC[k] for k in B3_REGIONS},
        row_pairs=abi.ROW_PAIRS,
        stamp_base=abi.STAMP_BASE,
    )


# ------------------------------------------------------------------ emitter


@dataclass
class B3Io:
    """What the emitter needs from its host (kernel, device function or the
    layer framework). All scalars are uniform i32 / i64 ``Value``s.

    ``addr(name)``: uniform i64 address of kernel argument ``name``.
    ``poll(specs)``: ``mailbox.poll`` with the layer's tag (and the host's
    ready stamp). ``sc``: region byte offsets, ``row_pairs``: pairs between
    request rows of a region.
    """

    b: IRBuilder
    tid: Value
    lane: Value
    wave: Value
    task: Value
    tag: Value
    scratch: Value
    lds: LdsView
    addr: object
    sc: dict
    row_pairs: int
    poll: object = None
    ready: object = None  # stamps hook after a poll the host does not wrap


class _B3:
    def __init__(self, io: B3Io, shape: B3Shape, knobs: B3Knobs):
        self.io, self.shape, self.k = io, shape, knobs
        self.b = io.b
        self.rows = shape.rows
        self.S = knobs.split
        self.RT = DV // self.S  # task rows
        self.NJ = 4 // self.S  # state rows per thread
        self.NB = self.RT // 32  # MXFP8 blocks per task
        self.NL = 2 * DK // 4 + 2 * (self.RT // 4)  # qkvz loader threads per row
        self.mbk = knobs.mailbox
        self._numa_mb = None  # knob numa=1: the NUMA area (``_area``)
        self._sel = None

    # -- small helpers
    def c(self, v: int) -> Value:
        return self.b.const_i32(int(v))

    def f(self, v: float) -> Value:
        return self.b.const_f32(float(v))

    def fbits(self, bits: int) -> Value:
        return self.b.bitcast(self.c(bits), F32)

    def bf16r(self, x: Value) -> Value:
        b = self.b
        return b.cast_to_f32(b.trunc_f32_to_bf16(x))

    def hexp(self, x: Value) -> Value:
        """ATOM ``device_ops.exp``: ``v_exp_f32(x * log2 e)``."""
        b = self.b
        return b.exp2_fast(b.fmul(x, self.f(LOG2E)))

    def sigmoid(self, x: Value) -> Value:
        """ATOM ``sigmoid_fast``: ``v_rcp(1 + exp(0 - x))``."""
        b = self.b
        return b.rcp_fast(b.fadd(self.f(1.0), self.hexp(b.fsub(self.f(0.0), x))))

    def softplus(self, x: Value) -> Value:
        """ATOM ``softplus_fast``: ``log2(1 + exp(x)) ln 2``, identity above 20."""
        b = self.b
        sp = b.fmul(b.log2(b.fadd(self.f(1.0), self.hexp(x))), self.f(LN2))
        return b.select(b.fcmp("ogt", x, self.f(20.0)), x, sp)

    def exp_decay(self, x: Value) -> Value:
        """The mono's ``fmath.exp`` (``math.exp``)."""
        if self.k.exp == "llvm":
            return self.b.exp(x)
        return self.hexp(x)

    def shfl(self, v: Value, off: int) -> Value:
        """``v`` of lane ``lane ^ off`` (f32)."""
        b = self.b
        x = b.bitcast(v, I32)
        if off in (1, 2, 8):
            r = b.dpp_xor(x, off)
        elif off == 4 and self.k.shfl4 == "rowshift":
            r = b.inline_asm(
                "s_nop 1\n\tv_mov_b32_dpp $0, $1 row_shr:4 row_mask:0xf bank_mask:0xa"
                "\n\tv_mov_b32_dpp $0, $1 row_shl:4 row_mask:0xf bank_mask:0x5",
                "=&v,v",
                [x],
                I32,
                sideeffect=False,
                result_name_hint="xs4",
            )
        elif off == 32:
            lo, hi = b.permlane32_swap(x, x)
            # lanes < 32 get (partner, own), lanes >= 32 (own, partner): one
            # of the two is the partner; as ATOM ``xred`` sum lo + hi below.
            return (b.bitcast(lo, F32), b.bitcast(hi, F32))
        else:
            r = b.warp_shuffle_xor(x, off)
        return b.bitcast(r, F32)

    def xadd(self, v: Value, off: int) -> Value:
        """ATOM ``xred(v, off, add)``."""
        b = self.b
        p = self.shfl(v, off)
        if isinstance(p, tuple):
            return b.fadd(p[0], p[1])
        return b.fadd(v, p)

    def xmax(self, v: Value, off: int) -> Value:
        b = self.b
        p = self.shfl(v, off)
        if isinstance(p, tuple):
            return b.fmax(p[0], p[1])
        return b.fmax(v, p)

    def sum16(self, v: Value) -> Value:
        for off in (8, 4, 2, 1):
            v = self.xadd(v, off)
        return v

    # -- LDS
    def lw(self, name: str, idx, row=0) -> Value:
        return self.io.lds.word(self.b, name, idx, row)

    def ldf(self, name: str, idx, row=0) -> Value:
        b = self.b
        v = b.smem_load_vN(self.io.lds.smem, self.lw(name, idx, row), dtype=F32, n=1)
        return b.vec_extract(v, 0)

    def ldf4(self, name: str, idx, row=0) -> list:
        b = self.b
        v = b.smem_load_vN(self.io.lds.smem, self.lw(name, idx, row), dtype=F32, n=4)
        return [b.vec_extract(v, e) for e in range(4)]

    def ldi4(self, name: str, idx, row=0) -> list:
        b = self.b
        v = b.smem_load_vN(self.io.lds.smem, self.lw(name, idx, row), dtype=I32, n=4)
        return [b.vec_extract(v, e) for e in range(4)]

    def stf(self, name: str, idx, value: Value, row=0) -> None:
        self.b.smem_store_vN(self.io.lds.smem, [self.lw(name, idx, row)], value, 1)

    def stv(self, name: str, idx, vec: Value, row=0) -> None:
        self.b.smem_store_vN(self.io.lds.smem, [self.lw(name, idx, row)], vec, vec.type.count)

    def barrier(self) -> None:
        if self.k.barrier == "full":
            self.b.sync(fence="compiler")
        else:
            self.b.sync_lds_only(fence="compiler")

    # -- memory
    def rsrc(self, addr: Value, nbytes: int) -> Value:
        b = self.b
        ptr = dops.ptr_from_i64(b, b.readfirstlane(addr))
        return b.buffer_rsrc(ptr, self.c(nbytes))

    def mb(self, region: str) -> M.Mailbox:
        b = self.b
        if self.k.numa:
            return self._area()
        a = b.add(self.io.scratch, b.const_i64(self.io.sc[region]))
        return M.mailbox_from_i64(b, a, self.c(0x7FFFFFF0), self.mbk)

    # -- ATOM's NUMA area (knob ``numa=1``): every region B3 touches, two copies
    def _area(self) -> M.Mailbox:
        """The NUMA area as one mailbox (pair indices from :meth:`put_ix` /
        :meth:`poll_ix`), and this CTA's copy selector (the header's page mask
        with the IOD folded in: bit p = the copy homed at this IOD on 2 MiB
        page p of the area, as ATOM ``numa.entry_state``). Emitted once."""
        if self._numa_mb is None:
            b = self.b
            hdr = self.k.header_offset
            a = b.add(self.io.scratch, b.const_i64(hdr + 256 + NUMA_PAGE - 1))
            area = b.land(a, b.const_i64(-NUMA_PAGE))
            self._numa_mb = M.mailbox_from_i64(b, area, self.c(0x7FFFFFF0), self.mbk)
            hp = dops.ptr_from_i64(b, b.add(self.io.scratch, b.const_i64(hdr)), pointee=I32)
            mask = b.readfirstlane(b.global_load(hp, self.c(0), I32, align=4))
            self._sel = b.readfirstlane(b.xor(mask, b.sub(self.c(0), M.iod_id(b))))
        return self._numa_mb

    def _xi(self, region: str, i: Value) -> Value:
        """Area byte offset of copy 0 of pair ``i`` of ``region`` (ATOM ``numa.phys``)."""
        b = self.b
        x = b.add(self.c(self.io.sc[region]), b.mul(i, self.c(M.PAIR_BYTES)))
        return b.add(x, b.land(x, self.c(-NUMA_CHUNK)))

    def put_ix(self, region: str, i: Value) -> Value:
        """Pair index a put of pair ``i`` of ``region`` stores to: ``i`` itself
        (plain), else copy 0 in the area (the put's ``numa_copies=2`` writes
        copy 1, 8 KiB on)."""
        if not self.k.numa:
            return i
        return self.b.lshr(self._xi(region, i), self.c(3))

    def poll_ix(self, region: str, i: Value) -> Value:
        """Pair index a poll of pair ``i`` of ``region`` reads: ``i`` itself
        (plain), else the copy homed at this CTA's IOD."""
        if not self.k.numa:
            return i
        b = self.b
        self._area()
        xi = self._xi(region, i)
        sel = b.land(b.lshr(self._sel, b.lshr(xi, self.c(21))), self.c(1))
        return b.lshr(b.add(xi, b.shl(sel, self.c(13))), self.c(3))

    @property
    def put_knobs(self) -> M.MailboxKnobs:
        """Mailbox knobs of a put (``numa=1``: both copies)."""
        return replace(self.mbk, numa_copies=2) if self.k.numa else self.mbk

    def ix(self, i: Value, r) -> Value:
        """Pair ``i`` of request row ``r`` (ATOM ``mb_ix``)."""
        b = self.b
        if isinstance(r, int):
            return i if r == 0 else b.add(i, self.c(r * self.io.row_pairs))
        return b.add(i, b.mul(r, self.c(self.io.row_pairs)))

    def poll(self, specs, knobs=None) -> list:
        if self.io.poll is not None and knobs is None:
            return self.io.poll(specs)
        out = M.poll(self.b, specs, self.io.tag, knobs or self.mbk)
        if self.io.ready is not None:
            self.io.ready()
        return out

    def bf16x8(self, words) -> list:
        b = self.b
        out = []
        for w in words:
            out += [
                b.bitcast(b.shl(w, self.c(16)), F32),
                b.bitcast(b.land(w, self.c(-65536)), F32),
            ]
        return out

    def sumsq64(self, name: str, base, row, lane: Value) -> Value:
        """ATOM ``sumsq64`` of the 64 f32 at ``name[base ..]`` of row ``row``."""
        b = self.b
        j = b.land(lane, self.c(31))
        if isinstance(base, int):
            jb = j if base == 0 else b.add(j, self.c(base))
        else:
            jb = b.add(j, base)
        x0 = self.ldf(name, jb, row)
        x1 = self.ldf(name, b.add(jb, self.c(32)), row)
        v = b.fma(x0, x0, b.fmul(x1, x1))
        for off in (16, 8, 4, 2, 1):
            v = self.xadd(v, off)
        return v

    # -- the stage
    def head(self, t: Value) -> tuple[Value, Value]:
        """``(vh, s)`` of task ``t``."""
        b, S = self.b, self.S
        if self.k.head_map == "linear":
            if S == 1:
                return t, self.c(0)
            return b.lshr(t, self.c(S.bit_length() - 1)), b.land(t, self.c(S - 1))
        # xcd: task t runs on CTA t (mono B3 base 0), XCD t % 8; the XCD's
        # j-th task (j = t // 8) takes head 8-local j // S, split j % S.
        per = HV * S // 8  # tasks per XCD
        x = b.land(t, self.c(7))
        j = b.lshr(t, self.c(3))
        hx = per // S  # heads per XCD
        if S == 1:
            return b.add(b.mul(x, self.c(hx)), j), self.c(0)
        vh = b.add(b.mul(x, self.c(hx)), b.lshr(j, self.c(S.bit_length() - 1)))
        return vh, b.land(j, self.c(S - 1))

    def emit(self) -> None:
        io, b, k = self.io, self.b, self.k
        c = self.c
        rows, S, RT, NJ, NB, NL = self.rows, self.S, self.RT, self.NJ, self.NB, self.NL
        G = self.shape.chain
        tid, lane, wave = io.tid, io.lane, io.wave
        vh, s = self.head(io.task)
        vh = b.readfirstlane(vh)
        s = b.readfirstlane(s)
        kh = b.readfirstlane(b.div(vh, c(GQA)))
        row0 = b.lshr(tid, c(4))  # 0..31
        kl = b.land(tid, c(15))
        mb_xbf, mb_qkvz = self.mb("xbf"), self.mb("qkvz")
        mb_d8, mb_de = self.mb("d8"), self.mb("de")
        pk = self.put_knobs
        if k.xchg == "numa2":
            xk = xk_poll = replace(self.mbk, numa_copies=2)
            xa = b.add(io.scratch, b.const_i64(XCHG_OFFSET))
            mb_core = M.mailbox_from_i64(b, xa, c(0x7FFFFFF0), xk)
            mb_core_poll = M.local_copy(b, mb_core, 0)

            def xix(p, r):
                q = b.add(b.mul(b.lshr(p, c(10)), c(2 * XCHG_BLOCK_PAIRS)), b.land(p, c(1023)))
                if isinstance(r, int):
                    return q if r == 0 else b.add(q, c(r * XCHG_ROW_PAIRS))
                return b.add(q, b.mul(r, c(XCHG_ROW_PAIRS)))

            xput = xpoll = xix
        else:
            xk, xk_poll = pk, self.mbk
            mb_core = mb_core_poll = self.mb("core")

            def xput(p, r):
                return self.put_ix("core", self.ix(p, r))

            def xpoll(p, r):
                return self.poll_ix("core", self.ix(p, r))

        def load_state():
            # Every row's state addresses, then the chain heads' state.
            nbytes = HV * DV * DK * 4
            rec_in, rec_out = [], []
            if k.table == "smem":
                base = b.readfirstlane(io.addr("conv_in"))
                lines = [f"s_load_dwordx4 ${r}, ${rows}, {(4 * r + 2) * 8}" for r in range(rows)]
                lines.append("s_waitcnt lgkmcnt(0)")
                vt = VectorType(I32, 4)
                outs = b.inline_asm_multi(
                    "\n".join(lines),
                    ",".join(["=&s"] * rows + ["s"]),
                    [base],
                    result_types=[vt] * rows,
                    result_name_hint="stab",
                )
                for r in range(rows):
                    w = [b.vec_extract(outs[r], e) for e in range(4)]
                    a_in = b.lor(b.zext(w[0], I64), b.shl(b.zext(w[1], I64), b.const_i64(32)))
                    a_out = b.lor(b.zext(w[2], I64), b.shl(b.zext(w[3], I64), b.const_i64(32)))
                    rec_out.append(self.rsrc(a_out, nbytes))
                    rec_in.append(self.rsrc(a_in, nbytes) if r % G == 0 else None)
            else:
                tbl = dops.ptr_from_i64(b, b.readfirstlane(io.addr("conv_in")), pointee=I64)
                for r in range(rows):
                    rec_out.append(self.rsrc(b.global_load_i64(tbl, c(4 * r + 3)), nbytes))
                    rec_in.append(
                        self.rsrc(b.global_load_i64(tbl, c(4 * r + 2)), nbytes)
                        if r % G == 0
                        else None
                    )
            hrow = [
                b.add(b.add(b.mul(s, c(RT)), row0), c(32 * j)) for j in range(NJ)
            ]  # head rows of this thread
            sbase = [
                b.add(b.mul(b.add(b.mul(vh, c(DV)), hrow[j]), c(DK)), kl)
                for j in range(NJ)
            ]
            st = {}
            for r in range(rows):
                if r % G:
                    continue
                if k.pre == "dma":
                    # the prologue's copy: head h's task rows, row-major
                    h = r // G
                    st[r] = [
                        [
                            self.ldf(
                                "b3_spre",
                                b.add(
                                    b.mul(b.add(row0, c(32 * j)), c(DK)),
                                    b.add(kl, c(h * RT * DK + 16 * e)),
                                ),
                            )
                            for e in range(8)
                        ]
                        for j in range(NJ)
                    ]
                    continue
                st[r] = [
                    [
                        b.buffer_load(
                            rec_in[r], b.mul(b.add(sbase[j], c(16 * e)), c(4)), c(0), F32
                        )
                        for e in range(8)
                    ]
                    for j in range(NJ)
                ]
            return rec_out, sbase, st

        if k.state_issue == "entry" and k.pre != "dma":
            rec_out, sbase, st = load_state()

        # Prefetch what does not depend on B1: gate parameters, norm weights.
        r_alog = self.rsrc(io.addr("a_log"), HV * 4)
        r_dtb = self.rsrc(io.addr("dt_bias"), HV * 4)
        r_nw = self.rsrc(io.addr("norm_w"), DV * 2)
        r_ba = self.rsrc(io.addr("ba_w"), 2 * HV * H * 2)
        vh4 = b.mul(vh, c(4))
        a_log = b.buffer_load(r_alog, vh4, c(0), F32)
        dt_bias = b.buffer_load(r_dtb, vh4, c(0), F32)
        g16 = b.land(lane, c(15))
        bk = b.smin(b.lshr(g16, c(2)), c(NB - 1))  # epilogue block of this lane
        m4 = b.land(g16, c(3))
        r0 = b.add(b.mul(bk, c(32)), b.mul(m4, c(8)))  # its first task row
        nw_row = b.add(b.mul(s, c(RT)), r0)
        nw = b.buffer_load_vN(r_nw, b.mul(nw_row, c(2)), c(0), I32, 4)

        # b and a: the bf16 mixed row and both weight rows into the router's
        # 16-part LDS layout (threads 128 .. 447), overlapping wave 0's wait.
        bj = b.smin(b.smax(b.sub(tid, c(128)), c(0)), c(H // 8 - 1))
        in_ba = b.land(b.cmp_ge(tid, c(128)), b.cmp_lt(tid, c(128 + H // 8)))
        with b.scf_if(in_ba):
            if k.pre != "dma":
                woff = b.mul(b.add(b.mul(vh, c(H // 2)), b.mul(bj, c(4))), c(4))
                wb = b.buffer_load_vN(r_ba, woff, c(0), I32, 4)
                wa = b.buffer_load_vN(r_ba, b.add(woff, c(HV * H * 2)), c(0), I32, 4)
                b.sched_barrier(0)
            specs = []
            for r in range(rows):
                specs += [
                    (mb_xbf, self.poll_ix("xbf", self.ix(b.mul(bj, c(4)), r)), 2),
                    (mb_xbf, self.poll_ix("xbf", self.ix(b.add(b.mul(bj, c(4)), c(2)), r)), 2),
                ]
            xv = self.poll(specs)
            for r in range(rows):
                for h in range(2):
                    j = b.add(b.mul(bj, c(2)), c(h))
                    x0, x1 = M.bf2_f32(b, xv[2 * r + h][0])
                    x2, x3 = M.bf2_f32(b, xv[2 * r + h][1])
                    part, first = self._part(b.mul(j, c(4)))
                    dst = b.sub(b.add(b.mul(part, c(RX_PART)), b.mul(j, c(4))), first)
                    self.stv("b3_bax", dst, b.vec_pack([x0, x1, x2, x3], F32), r)
            if k.pre != "dma":
                part, first = self._part(b.mul(bj, c(8)))
                off = b.add(
                    b.mul(part, c(RW_PART)), b.lshr(b.sub(b.mul(bj, c(8)), first), c(1))
                )
                self.stv("b3_baw", off, wb)
                self.stv("b3_baw", b.add(off, c(RW_ROW)), wa)
        if k.pre == "dma":
            # every wave's prologue DMAs have landed before the barrier
            b.s_waitcnt(vmcnt=0)
        self.barrier()

        if k.state_issue == "after_ba" or k.pre == "dma":
            rec_out, sbase, st = load_state()

        # b, a chains: wave 1 (b) and wave 2 (a), lane 16 r + q runs part q
        # of request row r: one fma chain from +0, the 16 parts in order.
        in_ch = b.land(
            b.land(b.cmp_ge(wave, c(1)), b.cmp_lt(wave, c(3))),
            b.cmp_lt(lane, c(16 * rows)),
        )
        with b.scf_if(in_ch):
            q = lane if rows == 1 else b.land(lane, c(15))
            rr = c(0) if rows == 1 else b.lshr(lane, c(4))
            big = b.cmp_lt(q, c(8))
            if k.pre == "dma":
                firstq = b.select(big, b.mul(q, c(192)), b.add(b.mul(q, c(128)), c(512)))
                wrow = b.add(b.mul(b.sub(wave, c(1)), c(H // 2)), b.lshr(firstq, c(1)))
                wname = "b3_bawr"
            else:
                wrow = b.add(b.mul(b.sub(wave, c(1)), c(RW_ROW)), b.mul(q, c(RW_PART)))
                wname = "b3_baw"
            xrow = b.mul(q, c(RX_PART))
            acc = self.f(0.0)
            for st_ in range(CHAIN_STEPS):
                wv = self.ldi4(wname, b.add(wrow, c(4 * st_)))
                w8 = self.bf16x8(wv)
                x8 = self.ldf4("b3_bax", b.add(xrow, c(8 * st_)), rr) + self.ldf4(
                    "b3_bax", b.add(xrow, c(8 * st_ + 4)), rr
                )
                new = acc
                for e in range(8):
                    new = b.fma(w8[e], x8[e], new)
                acc = new if st_ < SHORT_STEPS else b.select(big, new, acc)
            bits = b.bitcast(acc, I32)
            for r in range(rows):
                total = b.bitcast(b.readlane(bits, c(16 * r)), F32)
                for qq in range(1, 16):
                    total = b.fadd(total, b.bitcast(b.readlane(bits, c(16 * r + qq)), F32))
                with b.scf_if(b.cmp_eq(lane, c(0))):
                    self.stf("b3_sums", b.sub(wave, c(1)), self.bf16r(total), r)

        R = range(rows)

        def gates():
            decay_, beta_ = [], []
            for r in R:
                bb = self.ldf("b3_sums", 0, r)
                aa = self.ldf("b3_sums", 1, r)
                sp = self.softplus(b.fadd(aa, dt_bias))
                decay_.append(
                    self.exp_decay(b.fmul(sp, b.fsub(self.f(0.0), self.exp_decay(a_log))))
                )
                beta_.append(self.sigmoid(bb))
            # the chain heads' decayed state (the first op of the recurrence)
            sv0_ = {
                (r, j): [b.fmul(st[r][j][e], decay_[r]) for e in range(8)]
                for r in R
                if r % G == 0
                for j in range(NJ)
            }
            return decay_, beta_, sv0_

        if k.hoist != "none":
            self.barrier()  # the b / a sums of waves 1, 2
            decay, beta, sv0 = gates()

        # qkvz: wait for B1's tiles, then loader thread i < NL of row r polls
        # 4 values (2 pairs): q (32 threads), k (32), v, z (RT / 4 each).
        ntile = 4 + 2 * max(1, RT // 64)
        if k.wait == "pre_poll":
            tiles = self._tiles(kh, vh, s)

            def hint(m: Value):
                kk = m if rows == 1 else b.mod(m, c(ntile))
                tile = tiles[-1]
                for i in range(ntile - 2, -1, -1):
                    tile = b.select(b.cmp_eq(kk, c(i)), tiles[i], tile)
                rr_ = 0 if rows == 1 else b.div(m, c(ntile))
                return mb_qkvz, self.poll_ix("qkvz", self.ix(b.add(b.mul(tile, c(B1_TILE_PAIRS)), c(31)), rr_))

            M.pre_poll(b, hint, ntile * rows, io.tag, self.mbk)
        else:
            b.s_barrier_bare(fence="compiler")
        with b.scf_if(b.cmp_lt(tid, c(NL * rows))):
            i = tid if rows == 1 else b.mod(tid, c(NL))
            ri = 0 if rows == 1 else b.div(tid, c(NL))
            nv = RT // 4
            vbase = b.add(b.mul(vh, c(DV // 2)), b.mul(s, c(RT // 2)))
            src = b.select(
                b.cmp_lt(i, c(64 + nv)),
                b.select(
                    b.cmp_lt(i, c(64)),
                    b.select(
                        b.cmp_lt(i, c(32)),
                        b.add(b.mul(kh, c(64)), b.mul(i, c(2))),
                        b.add(c(KEY // 2), b.add(b.mul(kh, c(64)), b.mul(b.sub(i, c(32)), c(2)))),
                    ),
                    b.add(c(KEY), b.add(vbase, b.mul(b.sub(i, c(64)), c(2)))),
                ),
                b.add(c(QKV // 2), b.add(vbase, b.mul(b.sub(i, c(64 + nv)), c(2)))),
            )
            v = self.poll([(mb_qkvz, self.poll_ix("qkvz", self.ix(src, ri)), 2)])[0]
            q0, q1 = M.bf2_f32(b, v[0])
            q2, q3 = M.bf2_f32(b, v[1])
            vec = b.vec_pack([q0, q1, q2, q3], F32)
            is_qk = b.cmp_lt(i, c(64))
            # q/k into b3_qk, v/z into b3_vz (row ri)
            w_qk = self.lw("b3_qk", b.mul(i, c(4)), ri)
            w_vz = self.lw("b3_vz", b.mul(b.sub(i, c(64)), c(4)), ri)
            b.smem_store_vN(io.lds.smem, [b.select(is_qk, w_qk, w_vz)], vec, 4)
        self.barrier()

        if k.hoist == "none":
            # Gates (every thread, identical in all siblings).
            decay, beta, sv0 = gates()
        # sigmoid(z) of the epilogue lane's 8 rows (gates_z: here; else in the
        # epilogue, after the exchange)
        rw = c(0) if rows == 1 else b.smin(wave, c(rows - 1))

        def zgate():
            zv_ = self.ldf4("b3_vz", b.add(r0, c(RT)), rw) + self.ldf4(
                "b3_vz", b.add(r0, c(RT + 4)), rw
            )
            return [self.sigmoid(z) for z in zv_]

        zsig = zgate() if k.hoist == "gates_z" else None

        # L2 norms of q and k.
        rsq128 = self.fbits(RSQ128)
        if rows == 1 or k.norms == "redundant":
            inv_q, inv_k = [], []
            for r in R:
                sq = b.fadd(self.sumsq64("b3_qk", 0, r, lane), self.sumsq64("b3_qk", 64, r, lane))
                sk = b.fadd(
                    self.sumsq64("b3_qk", 128, r, lane), self.sumsq64("b3_qk", 192, r, lane)
                )
                inv_q.append(b.fmul(b.rsqrt(b.fadd(sq, self.f(EPS))), rsq128))
                inv_k.append(b.rsqrt(b.fadd(sk, self.f(EPS))))
        else:
            with b.scf_if(b.cmp_lt(wave, c(2 * rows))):
                nk = b.land(wave, c(1))
                wr = b.lshr(wave, c(1))
                base = b.add(b.mul(nk, c(128)), b.mul(b.lshr(lane, c(5)), c(64)))
                halves = self.sumsq64("b3_qk", base, wr, lane)
                rs = b.rsqrt(b.fadd(self.xadd(halves, 32), self.f(EPS)))
                rs = b.select(b.cmp_eq(nk, c(0)), b.fmul(rs, rsq128), rs)
                with b.scf_if(b.cmp_eq(lane, c(0))):
                    self.stf("b3_sums", b.add(c(2), nk), rs, wr)
            self.barrier()
            inv_q = [self.ldf("b3_sums", 2, r) for r in R]
            inv_k = [self.ldf("b3_sums", 3, r) for r in R]
        qh = [
            [b.fmul(self.ldf("b3_qk", b.add(kl, c(16 * e)), r), inv_q[r]) for e in range(8)]
            for r in R
        ]
        kh_ = [
            [
                b.fmul(self.ldf("b3_qk", b.add(kl, c(128 + 16 * e)), r), inv_k[r])
                for e in range(8)
            ]
            for r in R
        ]

        # Gated delta rule on the thread's rows: S <- decay S;
        # delta = beta (v - S k); S <- S + delta k^T; core = S q.
        sn = {}
        core = {}
        for j in range(NJ):
            trow = b.add(row0, c(32 * j))  # task row
            for r in R:
                prev = st[r][j] if r % G == 0 else sn[(r - 1, j)]
                if r % G == 0:
                    sv = sv0[(r, j)]
                else:
                    sv = [b.fmul(prev[e], decay[r]) for e in range(8)]
                pr = self.f(0.0)
                for e in range(8):
                    pr = b.fma(sv[e], kh_[r][e], pr)
                vv = self.ldf("b3_vz", trow, r)
                delta = b.fmul(beta[r], b.fsub(vv, self.sum16(pr)))
                sn[(r, j)] = [b.fma(delta, kh_[r][e], sv[e]) for e in range(8)]
                cr = self.f(0.0)
                for e in range(8):
                    cr = b.fma(sn[(r, j)][e], qh[r][e], cr)
                core[(r, j)] = self.sum16(cr)

        def store_state():
            for r in R:
                for j in range(NJ):
                    for e in range(8):
                        voff = b.mul(b.add(sbase[j], c(16 * e)), c(4))
                        if k.state_nt:
                            b.buffer_store_cp(
                                b.bitcast(sn[(r, j)][e], I32), rec_out[r], voff, c(0),
                                aux=CPOL_NT,
                            )
                        else:
                            b.buffer_store_f32(rec_out[r], voff, c(0), sn[(r, j)][e])

        if k.stores == "early":
            store_state()
        direct = S > 1 and k.core_put == "direct"
        if direct:
            # lane 0 of each row's 16 lanes puts its row (every lane of the
            # row holds the same sum).
            with b.scf_if(b.cmp_eq(kl, c(0))):
                for r in R:
                    for j in range(NJ):
                        hr = b.add(
                            b.add(b.mul(vh, c(DV)), b.mul(s, c(RT))), b.add(row0, c(32 * j))
                        )
                        M.put(b, mb_core, xput(hr, r), core[(r, j)], io.tag, xk)
            if k.stores == "after_put":
                store_state()
        else:
            with b.scf_if(b.cmp_eq(kl, c(0))):
                for r in R:
                    for j in range(NJ):
                        self.stf("b3_core", b.add(row0, c(32 * j)), core[(r, j)], r)
            self.barrier()

        # Siblings exchange the head's core rows (split > 1).
        if S > 1 and not direct:
            npt = RT // 2  # put2 threads per row
            with b.scf_if(b.cmp_lt(tid, c(npt * rows))):
                jj = tid if rows == 1 else b.mod(tid, c(npt))
                rj = 0 if rows == 1 else b.div(tid, c(npt))
                c0 = self.ldf("b3_core", b.mul(jj, c(2)), rj)
                c1 = self.ldf("b3_core", b.add(b.mul(jj, c(2)), c(1)), rj)
                pi = b.add(b.add(b.mul(vh, c(DV)), b.mul(s, c(RT))), b.mul(jj, c(2)))
                M.put2(b, mb_core, xput(pi, rj), c0, c1, io.tag, xk)
            if k.stores == "after_put":
                store_state()

        # Epilogue: wave r (wave 0 at rows 1) finishes request row r: the
        # head's RMS over its 128 core rows, the gate, the MXFP8 pack.
        epi = b.cmp_eq(wave, c(0)) if rows == 1 else b.cmp_lt(wave, c(rows))
        with b.scf_if(epi):
            if S > 1:
                got = self.poll(
                    [(mb_core_poll, xpoll(b.add(b.mul(vh, c(DV)), b.mul(lane, c(2))), rw), 2)],
                    xk_poll,
                )[0]
                two = b.mul(lane, c(2))
                self.stf("b3_cores", two, b.bitcast(got[0], F32), rw)
                self.stf("b3_cores", b.add(two, c(1)), b.bitcast(got[1], F32), rw)
                cname = "b3_cores"
            else:
                cname = "b3_core"
            tot = b.fadd(
                self.sumsq64(cname, 0, rw, lane), self.sumsq64(cname, 64, rw, lane)
            )
            inv_rms = b.rsqrt(b.fadd(b.fmul(tot, self.f(1.0 / DV)), self.f(EPS)))
            if direct:
                own = b.add(b.mul(s, c(RT)), r0)  # the task's rows in the head
                cv = self.ldf4("b3_cores", own, rw) + self.ldf4(
                    "b3_cores", b.add(own, c(4)), rw
                )
            else:
                cv = self.ldf4("b3_core", r0, rw) + self.ldf4("b3_core", b.add(r0, c(4)), rw)
            if zsig is None:
                zsig = zgate()
            wv = self.bf16x8([b.vec_extract(nw, e) for e in range(4)])
            out = [
                self.bf16r(
                    b.fmul(b.fmul(b.fmul(cv[e], inv_rms), wv[e]), zsig[e])
                )
                for e in range(8)
            ]
            amax = b.fabs(out[0])
            for e in range(1, 8):
                amax = b.fmax(amax, b.fabs(out[e]))
            amax = self.xmax(self.xmax(amax, 1), 2)
            ex, scale = mxq.e8m0_fp8(b, amax)
            cvt = self.k.cvt
            if cvt == "intrinsic":
                from ..mx_quant import intrinsic_cvt

                cvt = intrinsic_cvt
            lo = mxq.pack_fp8x4(b, out[0], out[1], out[2], out[3], scale, cvt=cvt)
            hi = mxq.pack_fp8x4(b, out[4], out[5], out[6], out[7], scale, cvt=cvt)
            blk = b.add(b.mul(s, c(NB)), bk)  # block of the head
            with b.scf_if(b.cmp_lt(lane, c(4 * NB))):
                M.put2(
                    b, mb_d8,
                    self.put_ix("d8", self.ix(b.add(b.add(b.mul(vh, c(32)), b.mul(blk, c(8))), b.mul(m4, c(2))), rw)),
                    lo, hi, io.tag, pk,
                )  # fmt: skip
                with b.scf_if(b.cmp_eq(m4, c(0))):
                    M.put(b, mb_de, self.put_ix("de", self.ix(b.add(b.mul(vh, c(4)), blk), rw)), ex, io.tag, pk)
        if k.stores == "end":
            b.s_barrier_bare(fence="compiler")
            store_state()

    def emit_pre(self) -> None:
        """``pre="dma"``: buffer->LDS DMA of the chain heads' state rows and the
        b / a weight rows into the B3 arena (a later B3 body reads them)."""
        io, b = self.io, self.b
        c = self.c
        rows, RT = self.rows, self.RT
        G = self.shape.chain
        vh, s = self.head(io.task)
        vh = b.readfirstlane(vh)
        s = b.readfirstlane(s)
        wave, lane = io.wave, io.lane
        # The arena may reach past the mono's B3 window into entries an
        # earlier stage of this CTA (A2) read: every wave's LDS reads done.
        b.sync_lds_only(fence="compiler")
        base = b.readfirstlane(io.addr("conv_in"))
        heads = list(range(0, rows, G))
        lines = [f"s_load_dwordx2 ${i}, ${len(heads)}, {(4 * r + 2) * 8}" for i, r in enumerate(heads)]
        lines.append("s_waitcnt lgkmcnt(0)")
        outs = b.inline_asm_multi(
            "\n".join(lines),
            ",".join(["=&s"] * len(heads) + ["s"]),
            [base],
            result_types=[I64] * len(heads),
            result_name_hint="ptab",
        )
        arena = b.smem_addr_of(io.lds.smem)
        lane16 = b.mul(lane, c(16))

        def dma(rsrc, src_byte, lds_word, nchunks):
            for rnd in range((nchunks + WAVES - 1) // WAVES):
                ch = b.add(wave, c(rnd * WAVES))
                with b.scf_if(b.cmp_lt(ch, c(nchunks))):
                    off = b.mul(ch, c(DMA_CHUNK))
                    dst = b.smem_ptr_add(
                        arena, b.zext(b.add(c(4 * lds_word), off), I64)
                    )
                    b.async_buffer_load_lds_addr(
                        rsrc, dst,
                        b.add(b.add(src_byte, off), lane16), c(0), 4,
                    )  # fmt: skip

        nbytes = HV * DV * DK * 4
        for i, r in enumerate(heads):
            rs = self.rsrc(outs[i], nbytes)
            src = b.mul(b.add(b.mul(vh, c(DV)), b.mul(s, c(RT))), c(DK * 4))
            lw = io.lds.arena.offset["b3_spre"] + (r // G) * RT * DK
            dma(rs, src, lw, RT * DK * 4 // DMA_CHUNK)
        r_ba = self.rsrc(io.addr("ba_w"), 2 * HV * H * 2)
        lw = io.lds.arena.offset["b3_bawr"]
        for row, first in ((0, vh), (1, b.add(vh, c(HV)))):
            dma(r_ba, b.mul(first, c(H * 2)), lw + row * (H // 2), H * 2 // DMA_CHUNK)

    def _part(self, k: Value) -> tuple[Value, Value]:
        """ATOM ``moe._part``: the router part holding ``k`` and its first k."""
        b, c = self.b, self.c
        big = b.cmp_lt(k, c(BIG_FIRST))
        p = b.select(big, b.div(k, c(192)), b.div(b.sub(k, c(512)), c(128)))
        first = b.select(big, b.mul(p, c(192)), b.add(b.mul(p, c(128)), c(512)))
        return p, first

    def _tiles(self, kh: Value, vh: Value, s: Value) -> list:
        """Device form of :func:`b3_tiles`."""
        b, c = self.b, self.c
        RT = self.RT
        q = b.mul(kh, c(2))
        kk = b.add(c(KEY // 64), q)
        first = b.lshr(b.add(b.mul(vh, c(DV)), b.mul(s, c(RT))), c(6))
        n = max(1, RT // 64)
        v = [b.add(c(2 * KEY // 64), b.add(first, c(i))) for i in range(n)]
        z = [b.add(c(QKV // 64), b.add(first, c(i))) for i in range(n)]
        return [q, b.add(q, c(1)), kk, b.add(kk, c(1)), *v, *z]


def emit_b3_pre(io: B3Io, shape: B3Shape, knobs: B3Knobs) -> None:
    """Emit B3's prologue for task ``io.task`` (``knobs.pre == "dma"``): run
    in an earlier stage on the CTA that later runs the task. Every thread of
    the CTA must call this."""
    if knobs.pre != "dma":
        raise ValueError("emit_b3_pre needs pre='dma'")
    _B3(io, shape, knobs).emit_pre()


def emit_b3(io: B3Io, shape: B3Shape, knobs: B3Knobs = B3Knobs()) -> None:  # noqa: B008
    """Emit one B3 task (``io.task``, uniform; the caller guards ``task <
    b3_tasks(knobs)``). Every thread of the CTA must call this."""
    _B3(io, shape, knobs).emit()


# ------------------------------------------------------------------ hosts


# Word offset of the mono's first B3 LDS entry (``layout.for_rows(rows).lds["qk"]``,
# the same for the seq / grouped builds of a row count): everything below holds
# B1's and C1's GEMV entries (``red``, ``x8``, ``xe``), which a wave still in
# B1's epilogue reads while another wave starts B3. A B3 device function linked
# into the mono (whose ``lds_base`` is the arena's word 0) places its entries
# from here on. ``test_qwen38_stage_b3`` checks the values against ATOM's
# layout when the ATOM tree is importable.
MONO_B3_LDS_START = {1: 1856, 2: 3808, 3: 5712, 4: 7616}
# The mono's arena size per row count (B3's entries must end below it).
MONO_LDS_WORDS = {1: 22640, 2: 32256, 3: 32384, 4: 38912}


def _arena(shape: B3Shape, knobs: B3Knobs, extra=(), pad: int = 0) -> LdsArena:
    ents = b3_lds_entries(shape, knobs)
    if pad:
        ents = (LdsEntry("b3_pad", pad),) + ents
    return LdsArena({"b3": ents}, rows=shape.rows, pinned=extra)


def build_b3_kernel(
    shape: B3Shape,
    knobs: B3Knobs = B3Knobs(),  # noqa: B008
    *,
    sc: dict | None = None,
    row_pairs: int | None = None,
    stamps: bool = False,
    stamp_base: int | None = None,
    name: str | None = None,
) -> KernelDef:
    """The isolated-stage kernel: the mono's launch ABI (``step`` i64,
    ``layer`` i32, the 41 ``LAYER_ARGS`` i64), grid ``GRID`` x ``THREADS``,
    B3 task ``t`` on CTA ``t``; the host fills ``xbf`` and ``qkvz`` with the
    launch's tag. ``stamps``: ATOM-format stage stamps (one stage) at
    ``stamp_base`` of the scratch (``timeline.py`` decodes them)."""
    from . import abi
    from . import stamps as stp

    if knobs.numa:
        raise ValueError("numa=1 addresses ATOM's NUMA area: hybrid function only (lower_b3_function)")
    lay = ref_layout()
    sc = dict(sc or lay.sc)
    row_pairs = row_pairs or lay.row_pairs
    stamp_base = lay.stamp_base if stamp_base is None else stamp_base
    kname = name or b3_kernel_name(shape, knobs) + ("_tl" if stamps else "")
    b = IRBuilder(kname)
    b.kernel.attrs["max_workgroup_size"] = THREADS
    step = b.param("step", I64)
    layer = b.readfirstlane(b.param("layer", I32))
    args = {n: b.param(n, I64) for n in abi.LAYER_ARGS}
    arena = _arena(shape, knobs, (stp.LDS_ENTRY,) if stamps else ())
    lds = arena.emit(b)
    tid = b.thread_id_x()
    st = stp.Stamps(b, lds, 1) if stamps else None
    if st is not None:
        st.begin()
    sw = b.global_load_i32(dops.ptr_from_i64(b, b.readfirstlane(step)), b.const_i32(0))
    tag = M.launch_tag(b, sw, layer)
    bid = b.readfirstlane(b.block_id_x())

    def poll(specs):
        out = M.poll(b, specs, tag, knobs.mailbox)
        if st is not None:
            st.ready(0)
        return out

    io = B3Io(
        b=b, tid=tid, lane=b.lane_id(), wave=dops.wave_id(b), task=bid, tag=tag,
        scratch=b.readfirstlane(args["scratch"]), lds=lds, addr=lambda n: args[n],
        sc=sc, row_pairs=row_pairs, poll=poll,
        ready=(lambda: st.ready(0)) if st is not None else None,
    )  # fmt: skip
    if st is not None:
        st.enter(0)
    with b.scf_if(b.cmp_lt(bid, b.const_i32(b3_tasks(knobs)))):
        if knobs.pre == "dma":
            emit_b3_pre(io, shape, knobs)
        emit_b3(io, shape, knobs)
    if st is not None:
        st.leave(0)
        st.flush(io.scratch, stamp_base, layer, bid)
    b.ret()
    k = b.kernel
    if uses_python_only_ops(k) and not k.name.endswith("_pyonly"):
        k.name = k.name + "_pyonly"
    return k


def backend_for(kernel: KernelDef) -> str | None:
    return "python" if uses_python_only_ops(kernel) else None


# -- the layer framework (rocKE layer kernel; framework.StageSpec)


def b3_stage_spec(
    shape: B3Shape,
    knobs: B3Knobs = B3Knobs(),  # noqa: B008
    *,
    base: int = 0,
    phase: str = "d3",
    deps: tuple = ("a2", "b1"),
):
    """``framework.StageSpec`` of B3 for a rocKE layer kernel: reads ``xbf``
    (A2) and ``qkvz`` (B1), writes ``core`` (its siblings), ``d8``, ``de``
    (C1)."""
    from rocke.helpers.task_schedule import Stage

    from .framework import StageSpec

    stage = Stage(
        "b3", tasks=b3_tasks(knobs), base=base, deps=deps,
        reads=("xbf", "qkvz", "core"), writes=("core", "d8", "de"), lds_phase=phase,
    )  # fmt: skip

    def emit(b, ctx, p, t):
        emit_b3_framework(b, ctx, p, t, shape, knobs)

    return StageSpec(stage=stage, emit=emit, lds=b3_lds_entries(shape, knobs))


def _framework_ready(ctx):
    st = getattr(ctx, "_stamps", None)
    if st is None:
        return None
    return lambda: st.ready(ctx.stage_i)


def emit_b3_framework(b, ctx, p, t, shape: B3Shape, knobs: B3Knobs) -> None:
    """``emit(b, ctx, p, t)`` of ``framework.StageSpec``."""
    if knobs.numa:
        raise ValueError("numa=1 addresses ATOM's NUMA area: not in the rocKE layer framework")
    lay = ctx.layout
    sc = {n: lay.offsets[n] for n in B3_REGIONS}
    io = B3Io(
        b=b, tid=ctx.tid, lane=ctx.lane, wave=ctx.wave, task=t, tag=ctx.tag,
        scratch=ctx.scratch, lds=ctx.lds, addr=p.addr, sc=sc,
        row_pairs=lay.row_bytes // M.PAIR_BYTES, poll=lambda specs: ctx.poll(specs),
        ready=_framework_ready(ctx),
    )  # fmt: skip
    emit_b3(io, shape, replace(knobs, mailbox=ctx.knobs.mailbox))


# -- the hybrid device function (stage_fn registry, stage "b3")


def _stage_fn_supported(spec, layout) -> bool:
    try:
        B3Shape(spec.rows, spec.seq, spec.group)
    except ValueError:
        return False
    return spec.rows <= layout.max_rows


def _hybrid_body(b, a, layout, shape: B3Shape, knobs: B3Knobs, part: str = "body") -> None:
    """B3 inside the mono: ``a`` holds the ``stage_fn`` ABI values (the mono
    passes ``tag`` and its B3 task index ``task = bid``: mono's B3 base is
    0). The arena starts at the mono's B3 window (:data:`MONO_B3_LDS_START`)."""
    pad = MONO_B3_LDS_START[shape.rows]
    arena = _arena(shape, knobs, pad=pad)
    if arena.words > MONO_LDS_WORDS[shape.rows]:
        raise ValueError(f"B3 arena {arena.words} words exceeds the mono's {MONO_LDS_WORDS[shape.rows]}")
    lds = arena.emit(b)
    io = B3Io(
        b=b, tid=a.tid, lane=a.lane, wave=a.wave, task=a.task, tag=a.tag,
        scratch=b.readfirstlane(a.scratch), lds=lds, addr=lambda n: getattr(a, n),
        sc={n: layout.sc[n] for n in B3_REGIONS}, row_pairs=layout.row_pairs,
    )  # fmt: skip
    with b.scf_if(b.cmp_lt(a.task, b.const_i32(b3_tasks(knobs)))):
        if part == "pre":
            emit_b3_pre(io, shape, knobs)
        else:
            emit_b3(io, shape, knobs)


def _stage_fn_body(b, a, layout, spec) -> None:
    """``stage_fn`` registry body (default :class:`B3Knobs`; the mailbox
    implementation from ``spec.knobs``). Other knob values:
    :func:`lower_b3_function`."""
    shape = B3Shape(spec.rows, spec.seq, spec.group)
    knobs = B3Knobs(mailbox=M.MailboxKnobs(impl=spec.knobs.mailbox_impl))
    _hybrid_body(b, a, layout, shape, knobs)


def build_b3_function_kernel(shape: B3Shape, knobs: B3Knobs, layout, part: str = "body") -> KernelDef:
    """The hybrid function's ``KernelDef`` (``stage_fn.ABI_PARAMS``) for any
    :class:`B3Knobs` (``stage_fn.build_stage_kernel`` builds the registry's
    default-knob body)."""
    from .stage_fn import ABI_PARAMS

    stem = "b3" if part == "body" else "b3pre"
    b = IRBuilder(f"rocke_q38_{stem}_{shape.variant}")
    b.kernel.attrs["max_workgroup_size"] = layout.threads
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(
        **vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))),
    )  # fmt: skip
    _hybrid_body(b, a, layout, shape, knobs, part)
    b.ret()
    return b.kernel


def lower_b3_function(
    shape: B3Shape, knobs: B3Knobs, layout, *, arch="gfx950", flavor=None, fn_knobs=None,
    part: str = "body",
):  # fmt: skip
    """Lower B3 as the hybrid device function ``rocke_q38_b3_<variant>``
    (``part="pre"``: its prologue ``rocke_q38_b3pre_<variant>``, knob
    ``pre="dma"``) (``lds="arg"``: the mono passes its arena's address)."""
    from .device_function import DeviceFunctionKnobs, lower_device_function

    stem = "b3" if part == "body" else "b3pre"
    return lower_device_function(
        build_b3_function_kernel(shape, knobs, layout, part),
        arch=arch,
        symbol=f"rocke_q38_{stem}_{shape.variant}",
        llvm_flavor=flavor,
        knobs=fn_knobs or DeviceFunctionKnobs(lds="arg"),
    )


def _register() -> None:
    from .stage_fn import register_stage

    register_stage("b3", _stage_fn_body, _stage_fn_supported)


# ------------------------------------------------------------------ tuned knobs
# Best knob sets of the in-situ hill climb with lookback (the mono DeltaNet
# layer with this B3 linked in against the plain mono layer, gfx950, bitwise
# outputs; lever ledger ``hackathon-prep/reports/qwen38-rocke/ledger/gdn-b3.md``).
# Opt-in: B3Knobs() stays the mono's B3, ``tuned_knobs`` / the CLI's
# ``--knobs tuned`` pick these. Variants without an entry take the entry of
# the nearest measured one (``_TUNED_FALLBACK``).
_K1 = "split=2,wait=barrier,state_nt=True,state_issue=entry,xchg=numa2,hoist=gates_z,pre=dma,mb_impl=ir"
_K4A = "head_map=xcd,state_nt=True,mb_impl=ir"
# r2s: the signed lookback around K1 (gdn-b3 fixer) found split=1 with the
# prologue better than K1 (same run, 30 rounds, every round won)
_K2S = "split=1,state_nt=True,pre=dma,mb_impl=ir"
# r4s: the signed lookback around K4A found wait=barrier, state_issue=entry
# and no poll pause on top of it better (same run, 30 rounds, 29 won)
_K4S = "head_map=xcd,wait=barrier,state_nt=True,state_issue=entry,mb_impl=ir,mb_pause=none"
TUNED_SPECS = {
    "r1": _K1,
    "r2": _K1,
    "r2s": _K2S,
    "r3s": _K1,
    "r3": _K4A,  # pre="dma" does not fit the LDS of 3-4 independent rows
    "r4": _K4A,
    "r4s": _K4S,
    "r4g2": _K4A,  # grouped verify, 2 requests x 2 rows; pre="dma" does not fit its LDS
}
_TUNED_FALLBACK: dict = {}  # unmeasured narrow variants: "r4s"
# The same variants for a server with ATOM's ``numa=1`` schedule lever (``ATOM_Q38_SCHED=numa=1`` or ``best``):
# knob ``numa=1`` (the body addresses the lever's NUMA area; ATOM keeps the lever for the build), re-tuned on top
# of the lever (hill climb + signed lookback in situ against the numa=1 mono; lever ledger ``gdn-b3.md``,
# "NUMA"). CLI ``--knobs tuned_numa``. r1 / r4s keep their plain sets (no lever or combination beat them on the
# lever); r2s: split 2 (was 1) with wait=barrier and the intrinsic MXFP8 convert, also best at r2 / r3s (same-run
# head-to-head against K1 + numa); r4: wait=barrier, hoist=gates_z, the intrinsic convert and no poll pause,
# WITHOUT head_map=xcd / mb_impl=ir (both neutral-to-worse on the lever); r3 keeps K4A (the r4 set loses there).
_K2S_NUMA = "split=2,wait=barrier,state_nt=True,cvt=intrinsic,pre=dma,mb_impl=ir,numa=1"
_K4_NUMA = "wait=barrier,state_nt=True,cvt=intrinsic,hoist=gates_z,mb_pause=none,numa=1"
TUNED_SPECS_NUMA = {v: spec + ",numa=1" for v, spec in TUNED_SPECS.items()}
TUNED_SPECS_NUMA.update({"r2": _K2S_NUMA, "r2s": _K2S_NUMA, "r3s": _K2S_NUMA, "r4": _K4_NUMA})
# Variants whose rocKE B3 is checked bitwise against the mono's B3 in situ on
# gfx950 (default and tuned knobs, mutant differs; r4g2 against the mono's
# grouped build). The CLI emits only these unless ``--unverified``.
VERIFIED_VARIANTS = frozenset(TUNED_SPECS)


def knobs_from_spec(spec: str) -> B3Knobs:
    """``"k=v,..."`` (``B3Knobs`` fields; ``mb_<f>=v`` sets ``MailboxKnobs.f``)
    -> :class:`B3Knobs`. ``mb_impl=ir`` registers the ``ir`` mailbox."""
    kn: dict = {}
    mb: dict = {}
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        val: object = v
        if v.isdigit():
            val = int(v)
        elif v.lower() in ("true", "false"):
            val = v.lower() == "true"
        if k.startswith("mb_"):
            mb[k[3:]] = val
        else:
            kn[k] = val
    if mb.get("impl") == "ir":
        from . import mailbox_ir  # noqa: F401  (registers MailboxKnobs(impl="ir"))
    knobs = B3Knobs(**kn)
    return replace(knobs, mailbox=M.MailboxKnobs(**mb)) if mb else knobs


def knobs_spec(knobs: B3Knobs) -> str:
    """Inverse of :func:`knobs_from_spec` (non-default fields only)."""
    d, md = B3Knobs(), M.MailboxKnobs()
    parts = [
        f"{f}={getattr(knobs, f)}"
        for f in knobs.__dataclass_fields__
        if f != "mailbox" and getattr(knobs, f) != getattr(d, f)
    ]
    parts += [
        f"mb_{f}={getattr(knobs.mailbox, f)}"
        for f in knobs.mailbox.__dataclass_fields__
        if getattr(knobs.mailbox, f) != getattr(md, f)
    ]
    return ",".join(parts)


def tuned_knobs(shape: B3Shape, numa: bool = False) -> B3Knobs:
    """The tuned knob set of ``shape``'s variant (see :data:`TUNED_SPECS`;
    ``numa``: :data:`TUNED_SPECS_NUMA`, for a server with ATOM's numa=1 lever)."""
    table = TUNED_SPECS_NUMA if numa else TUNED_SPECS
    v = shape.variant
    key = v if v in table else _TUNED_FALLBACK.get(v, "r4s")
    return knobs_from_spec(table[key])


def sidecar_numa(knobs: B3Knobs) -> dict:
    """Sidecar keys of the mailbox layout a build addresses (ATOM
    ``extern_stages``, "NUMA"): ``numa=1`` builds ``{"mailboxes": "numa1",
    "numa_hdr": <offset>}`` (ATOM keeps its numa=1 lever for the build and
    refuses the build without it; the prologue carries its body's keys, ATOM
    keeps the lever only when every linked function is ``numa1``), plain
    builds none (ATOM drops the lever for such a build)."""
    if not knobs.numa:
        return {}
    return {"mailboxes": "numa1", "numa_hdr": knobs.header_offset}


def write_b3_functions(out_dir, shapes, knobs_for, layout, flavor=None) -> list:
    """Write ``b3_<variant>.ll`` / ``.json`` (and ``b3pre_<variant>`` for
    ``pre="dma"``) of every shape into ``out_dir``; ``knobs_for(shape)`` picks
    the knobs. ATOM: ``ATOM_Q38_EXTERN_STAGES=b3=<out_dir>`` (plus
    ``,b3pre=<out_dir>`` when a ``pre="dma"`` build is among them). Returns the
    written stems.

    Sidecars: ``"stage_knobs"`` (:func:`knobs_spec`) on both files and, on
    the body, ``"needs"``: ``["b3pre"]`` for ``pre="dma"`` (it reads the LDS
    the prologue fills), else ``[]``. ATOM's ``extern_stages.resolve`` refuses
    a body without its prologue, drops both under ``MISSING=mono`` when the
    prologue's file is missing, and refuses a pair built with different knobs.
    ``numa=1`` builds: both sidecars carry :func:`sidecar_numa` (``"mailboxes":
    "numa1"``, ``"numa_hdr"``): run them with ``ATOM_Q38_SCHED=numa=1`` (or
    ``best``). A stale ``b3pre_<variant>`` of an earlier ``pre="dma"`` build is removed
    when the new body does not need it."""
    from .device_function import write_device_function

    stems = []
    for shape in shapes:
        knobs = knobs_for(shape)
        spec = knobs_spec(knobs)
        needs = ["b3pre"] if knobs.pre == "dma" else []
        parts = ("body", "pre") if needs else ("body",)
        for part in parts:
            stem = ("b3_" if part == "body" else "b3pre_") + shape.variant
            fn = lower_b3_function(shape, knobs, layout, flavor=flavor, part=part)
            paths = write_device_function(fn, out_dir, stem=stem, bitcode=False)
            with open(paths["json"]) as f:
                meta = json.load(f)
            meta["stage_knobs"] = spec
            meta.update(sidecar_numa(knobs))
            if part == "body":
                meta["needs"] = needs
            else:
                meta["prologue_of"] = "b3"
            with open(paths["json"], "w", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            stems.append(stem)
        if not needs:
            for ext in (".ll", ".bc", ".json"):
                stale = os.path.join(out_dir, "b3pre_" + shape.variant + ext)
                if os.path.isfile(stale):
                    os.remove(stale)
    return stems


def main(argv=None) -> int:
    """``python -m kernels.gfx950.qwen38_mono.stage_b3 --out DIR [--knobs
    default|tuned|<spec>] [--variants r1,r2s,...] [--atom-root PATH]``."""
    import argparse

    from .mono_layout import from_atom

    ap = argparse.ArgumentParser(description="emit the rocKE B3 stage functions for the mono")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--knobs", default="tuned",
        help="default (the mono's B3) | tuned | tuned_numa (TUNED_SPECS_NUMA: run with ATOM_Q38_SCHED=numa=1 or best) "
             "| k=v,...",
    )  # fmt: skip
    ap.add_argument("--variants", default="r1,r2,r3,r4,r2s,r3s,r4s,r4g2")
    ap.add_argument("--atom-root")
    ap.add_argument("--flavor")
    ap.add_argument(
        "--unverified", action="store_true",
        help="allow variants without a GPU numerics check against the mono (grouped: see UNVERIFIED)",
    )  # fmt: skip
    a = ap.parse_args(argv)
    shapes = []
    for v in a.variants.split(","):
        m = re.fullmatch(r"r(\d)(s?)(?:g(\d))?", v.strip())
        if not m:
            raise SystemExit(f"bad variant {v!r}")
        shape = B3Shape(int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0))
        if shape.variant not in VERIFIED_VARIANTS and not a.unverified:
            raise SystemExit(
                f"variant {shape.variant} has no in-situ GPU check against the mono's B3 yet "
                "(VERIFIED_VARIANTS); pass --unverified to emit it anyway"
            )
        shapes.append(shape)
    if a.knobs == "default":
        knobs_for = lambda s: B3Knobs()
    elif a.knobs == "tuned":
        knobs_for = tuned_knobs
    elif a.knobs == "tuned_numa":
        knobs_for = lambda s: tuned_knobs(s, numa=True)
    else:
        fixed = knobs_from_spec(a.knobs)
        knobs_for = lambda s: fixed
    lay = from_atom(a.atom_root)
    stems = write_b3_functions(a.out, shapes, knobs_for, lay, flavor=a.flavor)
    pre = any(s.startswith("b3pre_") for s in stems)
    print("wrote", " ".join(stems))
    print(f"ATOM_Q38_EXTERN_STAGES=b3={a.out}" + (f",b3pre={a.out}" if pre else ""))
    if a.knobs == "tuned_numa" or any(knobs_for(s).numa for s in shapes):
        print("numa=1 bodies: run with ATOM_Q38_SCHED=numa=1 (or best); ATOM keeps the lever for these builds")
    return 0


_register()

if __name__ == "__main__":
    raise SystemExit(main())
