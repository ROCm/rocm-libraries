# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L5: the large-batch QSA core (sparse attention) at rows 8-128, TP1 / TP2, gfx950.

``experiments/qwen38-true-mono/DESIGN-LB.md`` 15.4 / 15.6 (P-L5) and 4.2 (Q-core). One kernel family,
``q38lb_qsa_*``, that first reproduces FlyTape's ``qsa_core`` (``flytape/models/qwen4/bodies_qsa.py`` + the indexer /
scan / select helpers of ``bodies.py``) in rocKE, then exposes the LB levers as knobs.

Per QSA layer and row (``pos`` = the row's position at this step; ``nh`` q heads, ``nkv`` kv heads per rank, head
dim 256, partial RoPE on 64 dims; indexer 4 heads x 128, blocks of 4 tokens, budget 2048 tokens = 512 blocks):

* **heads**: q / k zero-centred RMSNorm (``x rrms (1 + w)``) and RoPE (rotate-half on the first 64 dims), q to
  ``qb`` (bf16), k and v appended to the row's KV cache at ``pos`` (``[ctx, nkv, 2, 256]`` bf16, FlyTape's layout).
* **indexer key**: the raw key goes to the ring (``pos % 4 < 3``) or completes a block: mean of 4, RMSNorm, RoPE at the
  block start, E4M3 codes + a power-of-two scale (``kc`` / ``ks``); past the budget the new block is scored at once.
* **within the budget** (``(pos + 1) / 4 <= 512``, "dense"): the row attends every token; every gang member prepares
  every head itself (the same values, so the duplicate stores agree) and attends its split; no handshake.
* **past the budget** ("sparse"): member 0 computes the f32 indexer query and publishes it; every member scores its
  chunks of the old blocks (``ReLU(q . k)`` summed over the 4 heads, E4M3 x scale / sqrt(128)); the last to arrive
  selects the top 512 blocks (16-bit keys = f32 bits 30..15, ties to the lower block, ascending list) and releases
  the attention; members build their token-position table (the selected blocks' 4 tokens + the incomplete tail).
* **attention** (per kv head): a stage of ``16 x waves`` tokens, 16 per wave; ``S^T = K Q^T`` on
  ``mfma_f32_16x16x32_bf16`` (K the A operand, the kv head's ``nh / nkv`` q heads as the N columns, 12 of 16 real), a
  wave-private running max / sum in the ``exp2`` domain, ``O^T += V^T P^T`` on ``mfma_f32_16x16x16bf16_1k`` with V
  read transposed from LDS (``ds_read_b64_tr_b16``), the next stage's K / V loaded while the current one computes;
  the waves merge through LDS, the member's partial (max, sum, 256 f32 per head) goes to ``part`` / ``ml``, the last
  arriver of each group of 8 members merges them (two levels past 8), multiplies by ``sigmoid(gate)`` and writes the
  bf16 core output.

Units are (layer, row, member), a row's ``gang`` members consecutive; gangs are powers of two that divide the grid
(256 CTAs, one per CU), so a gang is resident together and its members may wait for each other. ``layers`` and ``rows``
are runtime arguments; every per-(layer, row) buffer is indexed by ``layer * rows + row``, so one launch walks
``layers`` independent QSA layers (``walk="stream"``: units flow without a barrier; ``walk="phase"``: a striped grid
sync after every layer, FlyTape's phase structure).

Cross-CTA payloads (the indexer query, scores, the block list, partials, appended K / V and the sparse rows' q) are
stored with ``sc1`` (write-through) and read with ``sc1`` loads after the handshake (P5 / WP-L1 rules); counters are
agent-scope atomics, reset by their last arriver with an atomic add (never a plain store); flags are bounded polls that
set the launch-fatal word on a timeout. ``acq="plain"`` (stale-read twin) reads with plain loads.

Knobs (``QsaKnobs``); every LB lever is default-off: ``waves`` (4 = FlyTape's 256-thread CTA, 8 = 512 threads with a
128-token stage), ``gang`` (0 = FlyTape's ``gang_size``), ``sel`` (``bits``: FlyTape's 16-step bit search;
``hist``: two byte-histogram passes, the same selection), ``kvcp`` (KV load cache bits), ``walk``, ``hz`` (asm hazard
pad, on). Must-fail mutants: ``MUTANTS``. Python engine only (``buffer_*_cp``, ``int_to_ptr``, asm); kernel names start
with ``q38lb_`` (no cpp-parity twin). Nothing here is imported by a served build.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from rocke.core.ir import (
    BF16,
    CPOL_NT,
    CPOL_SC1,
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
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

from ..persist import sync as PS
from . import lbf_hazard

__all__ = [
    "GRID",
    "HD",
    "IH",
    "MUTANTS",
    "NBLK",
    "QsaKnobs",
    "QsaShape",
    "TP1",
    "TP2",
    "build_qsa_kernel",
    "build_floor_kernel",
    "ft_gang",
    "qsa_params",
    "floor_params",
    "lds_words",
    "resolve",
]

GRID = 256
HD = 256  # head dim
ROT = 64  # rotary dims
IH = 128  # indexer head dim
IHN = 4  # indexer heads
RATIO = 4  # indexer compress ratio (tokens per block)
BUDGET = 2048  # indexer budget (tokens)
NBLK = BUDGET // RATIO  # 512 blocks
FAN = 8  # MERGE_FAN
KEY_BITS = 16
EPS = 1e-6
LOG2E = 1.4426950408889634
NEG = -1e30
FP8_FLOOR_BITS = 0x38D1B717  # f32 bits of 1e-4 (qsa_state.FP8_FLOOR)
MAXIT_DEFAULT = 1 << 22

MUTANTS = ("none", "rope", "scale", "merge", "tie", "window", "kvpos", "iqrope", "scan")


@dataclass(frozen=True)
class QsaShape:
    """Per-rank head geometry: TP1 24 q / 2 kv heads, TP2 12 / 1 (the indexer is replicated)."""

    nh: int = 24
    nkv: int = 2

    @property
    def group(self) -> int:
        return self.nh // self.nkv

    @property
    def q_rows(self) -> int:
        return 2 * self.nh * HD

    @property
    def kv_rows(self) -> int:
        return self.nkv * HD

    @property
    def in_rows(self) -> int:
        """bf16 elements per qsa_proj row: q with interleaved gate, k, v, indexer q (4 x 128), indexer k (128)."""
        return self.q_rows + 2 * self.kv_rows + (IHN + 1) * IH

    @property
    def iq_col(self) -> int:
        return self.q_rows + 2 * self.kv_rows

    @property
    def ik_col(self) -> int:
        return self.iq_col + IHN * IH

    @property
    def out_dim(self) -> int:
        return self.nh * HD

    @property
    def name(self) -> str:
        return f"h{self.nh}k{self.nkv}"


TP1 = QsaShape(24, 2)
TP2 = QsaShape(12, 1)


def ft_gang(batch: int, ctxcap: int, grid: int = GRID) -> int:
    """FlyTape ``bodies_qsa.gang_size``: the grid shared by the rows, at most 32 and at most the 64-token rounds a
    row's attended list has at ``ctxcap``, as a power of two that divides the grid."""
    rounds = -(-min(ctxcap, BUDGET + RATIO - 1) // 64)
    gang = max(2, min(32, grid // batch, rounds, grid & -grid))
    return 1 << (gang.bit_length() - 1)


@dataclass(frozen=True)
class QsaKnobs:
    """Knobs of the QSA core. Defaults = FlyTape's structure; the kernel name lists only deviations."""

    batch: int = 8  # rows bucket of the build (FlyTape compiles per batch; it sizes the gang)
    ctxcap: int = 192  # largest position + 1 the build serves (FlyTape's max_ctx; sizes the select's key array)
    waves: int = 4  # waves per CTA: 4 = FlyTape (256 threads, 64-token stage); 8 = 512 threads, 128-token stage
    gang: int = 0  # members per row; 0 = FlyTape's gang_size(batch, ctxcap)
    walk: str = "stream"  # "stream" (no barrier between layers) | "phase" (striped grid sync after every layer)
    kvcp: str = "sc1"  # KV load cache bits: "sc1" | "ntsc1"
    acq: str = "sc1"  # consumer reads of other CTAs' payloads: "sc1" | "plain" (stale-read twin)
    sel: str = "bits"  # select: "bits" (FlyTape's 16-step search) | "hist" (two byte-histogram passes) | "nib" (four
    # 4-bit digit rounds: per-lane packed counters, one wave reduction and one barrier per digit; the same boundary) |
    # "bits2" (eight 2-bit rounds: three candidates counted per round, half the barriers; the same boundary) |
    # "v1" / "v2" (the 1- / 2-bit searches counting on the VALU: per-lane compare-adds and one integer butterfly per
    # candidate instead of a ballot + s_bcnt1 per key; the same boundary)
    pf: int = 2  # K / V of the next stage in flight while the current computes (2, FlyTape) | 1: loaded per stage
    scan_its: int = 16  # scan chunk = (threads / 16) x scan_its blocks (16: FlyTape's 256 per 256-thread CTA)
    score: str = "valu"  # indexer scores: "valu" (FlyTape: f32 FMAs + 16-lane reductions) | "mfma" (keys E4M3 -> bf16
    # exactly, the query rounded to bf16, mfma_f32_16x16x32_bf16 with the 4 heads as columns: tolerance class, DESIGN N10)
    kw16: bool = False  # score=mfma: lane group g loads its block's dims 32 g .. 32 g + 31 as two 16-byte loads and uses
    # them in all four K steps (K order permuted identically in the query operand): half the load instructions
    mpad: bool = False  # waves' merge in LDS with a 260-word row (FlyTape: 256: the 16 head rows of a lane group hit one
    # bank, a 16-way conflict on the 16-byte stores and loads)
    gearly: bool = False  # attention: the output-gate loads are issued before the stage loop (FlyTape: after the merge)
    sela: bool = False  # every member runs the select on all scores (no select-release handshake; the block list in LDS)
    spf: bool = False  # scan: the next chunk's keys in flight while the current chunk is scored (unrolled by 2)
    qloc: bool = False  # q via LDS: the dense path keeps q in LDS only (no global round trip); sparse copies q to LDS once
    qlds: bool = False  # (needs qloc) the S MFMAs read their q operand from LDS every stage (32 fewer VGPRs: waves=8)
    stress: int = 0  # numerics only: member 0 sleeps stress x 64 x 64 clocks before publishing the indexer query, and
    # the other members pretouch its lines before they wait (the stale-read twin's setup)
    hz: bool = True  # s_nop 4 before asm VMEM with SGPR inputs (the inline-asm hazard pad)
    stamp: bool = False  # informational: s_memrealtime per unit phase into ``stp[(layer * rows + row) * gang + member][16]``
    mutant: str = "none"
    tag: str = ""  # recompiled-control name suffix

    def check(self) -> None:
        if self.waves not in (4, 8):
            raise ValueError("waves 4 | 8")
        if self.gang and (self.gang & (self.gang - 1) or not 1 <= self.gang <= 64):
            raise ValueError("gang: a power of two <= 64 (0 = FlyTape's)")
        if self.walk not in ("stream", "phase"):
            raise ValueError("walk stream | phase")
        if self.kvcp not in ("sc1", "ntsc1"):
            raise ValueError("kvcp sc1 | ntsc1")
        if self.acq not in ("sc1", "plain"):
            raise ValueError("acq sc1 | plain")
        if self.sel not in ("bits", "bits2", "hist", "nib", "v1", "v2"):
            raise ValueError("sel bits | bits2 | hist | nib | v1 | v2")
        if self.pf not in (1, 2):
            raise ValueError("pf 1 | 2")
        if self.scan_its not in (2, 4, 8, 16):
            raise ValueError("scan_its 2 | 4 | 8 | 16")
        if self.score not in ("valu", "mfma"):
            raise ValueError("score valu | mfma")
        if self.kw16 and self.score != "mfma":
            raise ValueError("kw16 needs score=mfma")
        if self.qlds and not self.qloc:
            raise ValueError("qlds needs qloc")
        if self.score == "mfma" and self.scan_its < 4:
            raise ValueError("score=mfma needs scan_its >= 4 (a chunk of >= 16 blocks per wave)")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.ctxcap < 8 or self.batch < 1:
            raise ValueError("ctxcap >= 8, batch >= 1")

    @property
    def threads(self) -> int:
        return 64 * self.waves

    @property
    def stage(self) -> int:
        return 16 * self.waves

    @property
    def g(self) -> int:
        """The resolved gang."""
        return self.gang or ft_gang(self.batch, self.ctxcap)

    def suffix(self) -> str:
        parts = []
        d = QsaKnobs()
        for f in fields(self):
            v, dv = getattr(self, f.name), getattr(d, f.name)
            if f.name in ("tag", "batch", "ctxcap") or v == dv:
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
        parts.append(f"b{self.batch}c{self.ctxcap}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def resolve(k: QsaKnobs) -> QsaKnobs:
    k.check()
    return k


PARAMS = ("proj", "rope", "wq", "wk", "wiq", "wik", "kv", "raw", "kc", "ks", "score", "blk", "qb", "iq", "part", "ml",
          "cnt", "rdy", "core", "posb", "err", "sync", "stp")
SCAL = ("rows", "layers", "ctx", "kcap", "sstr", "maxit")


def qsa_params(k: QsaKnobs) -> list[str]:
    """Kernel parameters (i64 addresses, then i32 scalars): the host packs ``<{n}Q{m}i``."""
    return list(PARAMS) + list(SCAL)


def kernel_name(shape: QsaShape, k: QsaKnobs) -> str:
    return f"q38lb_qsa_{shape.name}_{k.suffix()}"


# ------------------------------------------------------------------ LDS map (i32 words)


def _keys_count(k: QsaKnobs) -> int:
    rnd = 8 * k.threads
    blocks = max(1, k.ctxcap // RATIO)
    return -(-blocks // rnd) * rnd


def _pos_rounds(k: QsaKnobs) -> int:
    st = k.stage
    tokens = -(-(BUDGET + RATIO - 1) // (k.g * st)) * st
    return -(-((-(-tokens // st) + 1) * st) // k.threads)


class _Lds:
    def __init__(self, k: QsaKnobs):
        W = k.waves
        self.TOPS = 64
        self.DENS = self.TOPS + 16 * W
        self.CNTS = self.DENS + 16 * W  # [bit][wave] counts of the select (KEY_BITS x W); hist bins reuse BIG
        self.SCAN = self.CNTS + KEY_BITS * W  # 2 x W wave totals of the block scan (nib: 4 digits x W x 8 words below)
        self.SHARED = self.SCAN + 2 * W + 4  # 128 f32: the new block's stored key (indexer unit)
        self.NIB = self.SHARED + IH  # nib select: [digit][wave][8] packed 16-bit bin pairs
        self.BLK = self.NIB + (4 * W * 8 if k.sel == "nib" else 0)  # 512 words: the block list of a sela select
        self.QL = self.BLK + (NBLK if k.sela else 0)  # q heads bf16 [24][256] (qloc)
        self.HIST = self.QL + (24 * HD // 2 if k.qloc else 0)  # 256 bins x W (hist select) + 2 x 256 prefix scratch
        self.POS = self.HIST + (256 * W + 512 if k.sel == "hist" else 0)
        self.POS = (self.POS + 3) // 4 * 4
        self.BIG = self.POS + _pos_rounds(k) * k.threads
        self.BIG = (self.BIG + 63) // 64 * 64
        merge_words = W * 16 * (HD + (4 if k.mpad else 0))  # f32 [wave][head col][dim]; V staging aliases it
        key_words = _keys_count(k) // 2
        self.end = self.BIG + max(merge_words, key_words)
        self.total = max(self.end, ONE_CTA_PER_CU_WORDS)


def lds_words(k: QsaKnobs) -> int:
    return _Lds(resolve(k)).total


# ------------------------------------------------------------------ small helpers


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _f(b: IRBuilder, v: float) -> Value:
    return b.const_f32(float(v))


def _ptr(b: IRBuilder, addr: Value) -> Value:
    return b.int_to_ptr(addr, PtrType(I32, "global"))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b: IRBuilder, v: Value) -> Value:
    return b.zext(v, I64)


def _off(b: IRBuilder, base: Value, idx: Value, unit: int) -> Value:
    """``base + idx * unit`` in i64 (``idx`` i32 >= 0)."""
    return b.add(base, b.mul(_i64(b, idx), b.const_i64(unit)))


def _shfl(b: IRBuilder, v: Value, off: int):
    """VALU butterfly partner of f32 ``v`` (gdn_recur._shfl): 1, 2 quad_perm; 4 row_half_mirror (valid after the
    stages below it); 8 row_ror:8; 16 / 32 permlane swaps (return the (lo, hi) pair)."""
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


def _xop(b: IRBuilder, v: Value, off: int, op) -> Value:
    p = _shfl(b, v, off)
    if isinstance(p, tuple):
        return op(p[0], p[1])
    return op(v, p)


def _wsum(b: IRBuilder, v: Value, offs=(1, 2, 4, 8, 16, 32)) -> Value:
    for o in offs:
        v = _xop(b, v, o, b.fadd)
    return v


def _wmax(b: IRBuilder, v: Value, offs=(1, 2, 4, 8, 16, 32)) -> Value:
    for o in offs:
        v = _xop(b, v, o, b.fmax)
    return v


def _bf4(b: IRBuilder, v2: Value) -> list[Value]:
    """<2 x i32> of 4 bf16 -> 4 f32."""
    out = []
    for i in range(2):
        w = b.vec_extract(v2, i)
        out.append(b.bitcast(b.shl(w, _c(b, 16)), F32))
        out.append(b.bitcast(b.land(w, _c(b, -65536)), F32))
    return out


def _bf2(b: IRBuilder, w: Value) -> list[Value]:
    return [b.bitcast(b.shl(w, _c(b, 16)), F32), b.bitcast(b.land(w, _c(b, -65536)), F32)]


def _pack_bf(b: IRBuilder, xs: list[Value]) -> Value:
    """2k f32 -> <k x i32> of bf16 pairs (RNE)."""
    words = []
    for i in range(0, len(xs), 2):
        h0 = b.zext(b.bitcast(b.trunc_f32_to_bf16(xs[i]), _I16), I32)
        h1 = b.zext(b.bitcast(b.trunc_f32_to_bf16(xs[i + 1]), _I16), I32)
        words.append(b.lor(h0, b.shl(h1, _c(b, 16))))
    if len(words) == 1:
        return words[0]
    return b.vec_pack(words, I32)


from rocke.core.ir import I16 as _I16  # noqa: E402


_IV = [0]


def _iv(prefix: str) -> str:
    """A loop induction-variable name unique in the process (a repeated scf_for iv name in one kernel lowers to a
    duplicate SSA name: alola rule 181)."""
    _IV[0] += 1
    return f"{prefix}{_IV[0]}"


def _ia(args):
    """Iteration-argument names made unique (they become SSA names too)."""
    return [(_iv(nm), v) for nm, v in args]


def _sigmoid(b: IRBuilder, z: Value) -> Value:
    e = b.exp2(b.fmul(b.fsub(_f(b, 0.0), z), _f(b, LOG2E)))
    return b.rcp(b.fadd(_f(b, 1.0), e))


def _popc64(b: IRBuilder, m: Value) -> Value:
    return b.inline_asm("s_bcnt1_i32_b64 $0, $1", "=s,s", [m], I32, result_name_hint="bcnt")


def _wave_count(b: IRBuilder, pred: Value) -> Value:
    """Lanes where ``pred`` (i1) holds, wave-uniform."""
    return _popc64(b, b.wave_ballot(b.zext(pred, I32)))


# ------------------------------------------------------------------ the kernel


class _Qsa:
    def __init__(self, shape: QsaShape, k: QsaKnobs):
        self.sh, self.k = shape, k
        _IV[0] = 0
        self.W, self.T, self.ST, self.G = k.waves, k.threads, k.stage, k.g
        self.L = _Lds(k)
        self.b = b = IRBuilder(kernel_name(shape, k))
        b.kernel.attrs["max_workgroup_size"] = self.T
        self.a = {n: b.param(n, I64) for n in PARAMS}
        self.s = {n: b.readfirstlane(b.param(n, I32)) for n in SCAL}
        self.cta = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = b.readfirstlane(D.wave_id(b))
        self.lds = b.smem_alloc(I32, [self.L.total], name_hint="qsa_lds")
        self.lds0 = b.trunc(b.smem_addr_of(self.lds), I32)
        self.cp_ld = CPOL_SC1 if k.acq == "sc1" else 0
        self.cp_kv = CPOL_SC1 | (CPOL_NT if k.kvcp == "ntsc1" else 0)

    # ---- LDS word access (i32 array)
    def lw(self, idx: Value, v: Value) -> None:
        b = self.b
        b.smem_store_vN(self.lds, [idx], b.bitcast(v, I32) if v.type == F32 else v, 1)

    def lr(self, idx: Value, dtype=I32) -> Value:
        b = self.b
        v = b.vec_extract(b.smem_load_vN(self.lds, idx, dtype=I32, n=1), 0)
        return b.bitcast(v, F32) if dtype == F32 else v

    def bar(self) -> None:
        self.b.s_barrier_bare(fence="compiler")

    def wait_stores(self) -> None:
        self.b.s_waitcnt(vmcnt=0)

    # ---- global helpers
    def ld(self, rs: Value, voff: Value, dwords: int, cp: int) -> Value:
        return self.b.buffer_load_cp(rs, voff, _c(self.b, 0), dwords=dwords, aux=cp)

    def st(self, rs: Value, voff: Value, data: Value, cp: int = CPOL_SC1) -> None:
        self.b.buffer_store_cp(data, rs, voff, _c(self.b, 0), aux=cp)

    def atomic(self, base: Value, voff: Value, v: Value) -> Value:
        """Agent-scope returning add (thread 0 calls it)."""
        return D.atomic_add_scoped(self.b, base, voff, v, scope="agent", returns=True)

    def arrive(self, base: Value, voff: Value) -> Value:
        """The CTA's stores drained, one arrival on the counter: the arrivals before it (uniform)."""
        b = self.b
        self.wait_stores()
        self.bar()
        with b.scf_if(D.is_thread0(b)):
            old = self.atomic(base, voff, _c(b, 1))
            self.lw(_c(b, 8), old)
        self.bar()
        return b.readfirstlane(self.lr(_c(b, 8)))

    def reset(self, base: Value, voff: Value, n: int) -> None:
        """Thread 0 brings a counter back to 0 (atomic: every XCD sees it)."""
        b = self.b
        with b.scf_if(D.is_thread0(b)):
            D.atomic_add_scoped(b, base, voff, _c(b, -n), scope="agent")

    def publish(self, base: Value, voff: Value, value: int) -> None:
        b = self.b
        self.wait_stores()
        self.bar()
        with b.scf_if(D.is_thread0(b)):
            b.inline_asm("s_waitcnt vmcnt(0)\nglobal_store_dword $0, $1, $2 sc1\ns_waitcnt vmcnt(0)", "v,v,s,~{memory}",
                         [voff, _c(b, value), base], None)

    def wait(self, base: Value, voff: Value, expected: int, skip: Value) -> None:
        """Wave 0 lane 0 polls ``base[voff] >= expected`` (sc1, bounded); a timeout sets the launch-fatal word.
        ``skip`` (uniform i1): the CTA that published it passes."""
        b = self.b
        with b.scf_if(b.land(D.is_wave0(b), b.lnot(skip))):
            exp = b.select(b.cmp_eq(self.lane, _c(b, 0)), _c(b, expected), _c(b, 0))
            _seen, it = PS.poll_counter(b, voff, base, exp, self.s["maxit"], "s_sleep1")
            with b.scf_if(b.land(D.is_thread0(b), b.cmp_ge(it, self.s["maxit"]))):
                PS.raise_fatal(b, self.a["err"], "poll_timeout")
                D.atomic_add_scoped(b, self.a["err"], _c(b, 4 * PS.ERR_TIMEOUT), _c(b, 1))
        self.bar()

    # ---- the build
    def build(self) -> KernelDef:
        b, k, s = self.b, self.k, self.s
        with b.scf_if(b.cmp_lt(self.tid, _c(b, 64))):
            self.lw(self.tid, _c(b, 0))
        self.bar()
        G = self.G
        per_layer = b.mul(s["rows"], _c(b, G))
        if k.walk == "stream":
            total = b.mul(per_layer, s["layers"])
            n_rounds = b.div(b.add(total, _c(b, GRID - 1)), _c(b, GRID))
            with b.scf_for(_c(b, 0), n_rounds, _c(b, 1), iv_name=_iv("qu")) as i:
                u = b.add(self.cta, b.mul(i, _c(b, GRID)))
                with b.scf_if(b.cmp_lt(u, total)):
                    layer = b.div(u, per_layer)
                    v = b.sub(u, b.mul(layer, per_layer))
                    self.unit(layer, b.div(v, _c(b, G)), b.land(v, _c(b, G - 1)))
        else:
            gs = PS.GridSync(grid=GRID, stripes=8, maxit=0, pause="s_nop", arrivals="cta", waves=self.W)
            n_rounds = b.div(b.add(per_layer, _c(b, GRID - 1)), _c(b, GRID))
            with b.scf_for(_c(b, 0), s["layers"], _c(b, 1), iv_name=_iv("ql")) as layer:
                with b.scf_for(_c(b, 0), n_rounds, _c(b, 1), iv_name=_iv("qv")) as i:
                    v = b.add(self.cta, b.mul(i, _c(b, GRID)))
                    with b.scf_if(b.cmp_lt(v, per_layer)):
                        self.unit(layer, b.div(v, _c(b, G)), b.land(v, _c(b, G - 1)))
                PS.grid_sync(b, gs, self.a["sync"], self.cta, b.add(layer, _c(b, 1)), self.lds0, self.a["err"],
                             maxit=s["maxit"])
        b.ret()
        kd = b.kernel
        if k.hz:
            lbf_hazard.pad_asm_vmem(kd)
        return kd

    # ---- one unit
    def unit(self, layer: Value, row: Value, member: Value) -> None:
        b, sh, k, s = self.b, self.sh, self.k, self.s
        self.bar()
        lr = b.add(b.mul(layer, s["rows"]), row)
        posb = b.int_to_ptr(self.a["posb"], PtrType(I32, "constant"))
        pos = b.readfirstlane(b.global_load_i32(posb, row))
        if k.mutant == "kvpos":
            pos_app = b.sub(pos, _c(b, 1))
        else:
            pos_app = pos
        blocks = b.lshr(b.add(pos, _c(b, 1)), _c(b, 2))
        above = b.cmp_gt(blocks, _c(b, NBLK))
        old = b.lshr(pos, _c(b, 2))
        u = _Unit(self, layer, row, member, lr, pos, pos_app, blocks, old)
        u.stamp(0)
        with b.scf_if(above):
            u.sparse_prep()
        with b.scf_if(b.lnot(above)):
            u.dense_prep()
            u.stamp(1)
        u.attend_all(above)
        with b.scf_if(above):
            last = b.cmp_eq(self.arrive(u.cnt_base, _c(b, 4 * u.CB)), _c(b, self.G - 1))
            with b.scf_if(last):
                self.reset(u.cnt_base, _c(b, 4 * u.CB), self.G)
                if k.sela:
                    self.reset(u.cnt_base, _c(b, 4 * u.CA), self.G)
                with b.scf_if(D.is_thread0(b)):
                    for f in (0, 1):
                        b.inline_asm("global_store_dword $0, $1, $2 sc1", "v,v,s,~{memory}",
                                     [_c(b, 4 * f), _c(b, 0), u.rdy_base], None)
        u.stamp(6)


class _Unit:
    """One (layer, row, member) unit: uniform bases and the phase bodies."""

    def __init__(self, q: _Qsa, layer, row, member, lr, pos, pos_app, blocks, old):
        self.q, self.b = q, q.b
        b, sh, s, a = q.b, q.sh, q.s, q.a
        self.layer, self.row, self.member, self.lr = layer, row, member, lr
        self.pos, self.pos_app, self.blocks, self.old = pos, pos_app, blocks, old
        self.first = b.cmp_eq(member, _c(b, 0))
        # per-(layer, row) bases (uniform)
        self.proj = _rsrc(b, _off(b, a["proj"], lr, sh.in_rows * 2), _c(b, sh.in_rows * 2))
        kvb = s["ctx"]  # tokens per row's KV region
        kv_row_bytes = sh.nkv * 2 * HD * 2
        self.kv_base = b.readfirstlane(b.add(a["kv"], b.mul(b.mul(_i64(b, lr), _i64(b, kvb)), b.const_i64(kv_row_bytes))))
        self.kv = b.buffer_rsrc(_ptr(b, self.kv_base), b.mul(kvb, _c(b, kv_row_bytes)))
        self.raw = _rsrc(b, _off(b, a["raw"], lr, (RATIO - 1) * IH * 2), _c(b, (RATIO - 1) * IH * 2))
        kcap = s["kcap"]
        self.kc = _rsrc(b, b.add(a["kc"], b.mul(b.mul(_i64(b, lr), _i64(b, kcap)), b.const_i64(IH))), b.mul(kcap, _c(b, IH)))
        self.ks = _rsrc(b, b.add(a["ks"], b.mul(b.mul(_i64(b, lr), _i64(b, kcap)), b.const_i64(4))), b.mul(kcap, _c(b, 4)))
        sstr = s["sstr"]
        self.score = _rsrc(b, b.add(a["score"], b.mul(b.mul(_i64(b, lr), _i64(b, sstr)), b.const_i64(4))),
                           b.mul(sstr, _c(b, 4)))
        self.blk = _rsrc(b, _off(b, a["blk"], lr, NBLK * 4), _c(b, NBLK * 4))
        self.qb = _rsrc(b, _off(b, a["qb"], lr, sh.out_dim * 2), _c(b, sh.out_dim * 2))
        self.iq = _rsrc(b, _off(b, a["iq"], lr, IHN * IH * 4), _c(b, IHN * IH * 4))
        slots = sh.nh * q.G
        self.part = _rsrc(b, _off(b, a["part"], lr, slots * HD * 4), _c(b, slots * HD * 4))
        self.ml = _rsrc(b, _off(b, a["ml"], lr, slots * 8), _c(b, slots * 8))
        self.cnt_base = b.readfirstlane(_off(b, a["cnt"], lr, 256))
        self.rdy_base = b.readfirstlane(_off(b, a["rdy"], lr, 256))
        self.core = _rsrc(b, _off(b, a["core"], lr, sh.out_dim * 2), _c(b, sh.out_dim * 2))
        self.CA = sh.nkv * (FAN + 1)  # score counter word
        self.CB = self.CA + 1  # attention-done counter word
        # rope rows of pos (heads, 4 dims per lane) - loaded where used
        self.rope = b.int_to_ptr(a["rope"], PtrType(F32, "global"))

    def stamp(self, col: int) -> None:
        """Thread 0 records s_memrealtime at stamp column ``col`` (0 start, 1 prep, 2 scan arrived, 3 select released,
        4 positions, 5 kv head 0 attended, 6 end; inside the select (the selecting member): 8 start, 9 keys in LDS,
        10 boundary found, 11 runs scanned, 12 list written; attention of kv head 0: 13 stages done, 14 partial stored
        and arrived, 15 the q operand loaded)."""
        b, q = self.b, self.q
        if not q.k.stamp:
            return
        with b.scf_if(D.is_thread0(b)):
            p = b.int_to_ptr(q.a["stp"], PtrType(I64, "global"))
            idx = b.add(b.mul(b.add(b.mul(self.lr, _c(b, q.G)), self.member), _c(b, 16)), _c(b, col))
            b.store(D.memrealtime(b), p, idx)

    # ------------------------------------------------ heads
    def heads(self, groups: list[Value], glob_q: bool = True) -> None:
        """Heads ``g * W + wave`` for each g: q / k norm + RoPE, q to qb, k / v appended at pos (sc1 stores)."""
        q, b, sh, k = self.q, self.b, self.q.sh, self.q.k
        lane, W = q.lane, q.W
        nh, nkv = sh.nh, sh.nkv
        dims = b.mul(lane, _c(b, 4))
        wq = b.int_to_ptr(q.a["wq"], PtrType(I32, "global"))
        wk = b.int_to_ptr(q.a["wk"], PtrType(I32, "global"))
        lo = b.add(b.mul(self.layer, _c(b, HD // 2)), b.mul(lane, _c(b, 2)))
        qw = [b.global_load_i32(wq, lo), b.global_load_i32(wq, b.add(lo, _c(b, 1)))]
        kw = [b.global_load_i32(wk, lo), b.global_load_i32(wk, b.add(lo, _c(b, 1)))]
        qw = _bf2(b, qw[0]) + _bf2(b, qw[1])
        kw = _bf2(b, kw[0]) + _bf2(b, kw[1])
        rot = []
        for e in range(4):
            fr = b.land(b.add(dims, _c(b, e)), _c(b, ROT // 2 - 1))
            cs = b.global_load_f32(self.rope, b.add(b.mul(self.pos_app, _c(b, ROT)), fr))
            sn = b.global_load_f32(self.rope, b.add(b.mul(self.pos_app, _c(b, ROT)), b.add(fr, _c(b, ROT // 2))))
            rot.append((cs, sn))
        loads, heads, live = [], [], []
        for g in groups:
            head = b.add(b.mul(g, _c(b, W)), q.wave)
            ok = b.cmp_lt(head, _c(b, nh + 2 * nkv))
            head = b.select(ok, head, _c(b, 0))
            src = b.select(b.cmp_lt(head, _c(b, nh)), b.mul(head, _c(b, 2 * HD)),
                           b.add(_c(b, sh.q_rows), b.mul(b.sub(head, _c(b, nh)), _c(b, HD))))
            loads.append(b.buffer_load_cp(self.proj, b.mul(b.add(src, dims), _c(b, 2)), _c(b, 0), dwords=2, aux=0))
            heads.append(head)
            live.append(ok)
        lower = b.cmp_lt(lane, _c(b, 8))  # dims < 32
        rotl = b.cmp_lt(lane, _c(b, 16))  # dims < 64
        for i in range(len(groups)):
            head, ok = heads[i], live[i]
            with b.scf_if(ok):
                x = _bf4(b, loads[i])
                is_q = b.cmp_lt(head, _c(b, nh))
                is_k = b.cmp_lt(head, _c(b, nh + nkv))
                ss = b.fmul(x[0], x[0])
                for e in range(1, 4):
                    ss = b.fma(x[e], x[e], ss)
                ss = _wsum(b, ss)
                rr = b.rsqrt(b.fadd(b.fmul(ss, _f(b, 1.0 / HD)), _f(b, EPS)))
                normed = [b.fmul(b.fmul(x[e], rr), b.fadd(_f(b, 1.0), b.select(is_q, qw[e], kw[e]))) for e in range(4)]
                out = []
                for e in range(4):
                    if k.mutant == "rope":
                        partner = normed[e]
                    else:
                        partner = _shfl(b, normed[e], 8)
                    cs, sn = rot[e]
                    turned = b.select(lower, b.fsub(b.fmul(normed[e], cs), b.fmul(partner, sn)),
                                      b.fadd(b.fmul(normed[e], cs), b.fmul(partner, sn)))
                    out.append(b.select(rotl, turned, normed[e]))
                val = [b.select(is_k, out[e], x[e]) for e in range(4)]
                data = _pack_bf(b, val)
                with b.scf_if(is_q):
                    if glob_q:
                        self.q.st(self.qb, b.mul(b.add(b.mul(head, _c(b, HD)), dims), _c(b, 2)), data)
                    if k.qloc:
                        b.smem_store_vN(q.lds, [b.add(_c(b, q.L.QL), b.add(b.mul(head, _c(b, HD // 2)), b.mul(lane, _c(b, 2))))],
                                        data, 2)
                with b.scf_if(b.lnot(is_q)):
                    kvh = b.select(is_k, b.sub(head, _c(b, nh)), b.sub(head, _c(b, nh + nkv)))
                    which = b.select(is_k, _c(b, 0), _c(b, 1))
                    el = b.mul(b.add(b.mul(b.add(b.mul(self.pos_app, _c(b, nkv)), kvh), _c(b, 2)), which), _c(b, HD))
                    self.q.st(self.kv, b.mul(b.add(el, dims), _c(b, 2)), data)

    # ------------------------------------------------ indexer
    def _norm2(self, vals: list[Value], wbase: Value, w_ptr_name: str) -> list[Value]:
        """Zero-centred RMSNorm of a wave's 128-dim vector (2 dims per lane)."""
        b, q = self.b, self.q
        ss = _wsum(b, b.fma(vals[1], vals[1], b.fmul(vals[0], vals[0])))
        rr = b.rsqrt(b.fadd(b.fmul(ss, _f(b, 1.0 / IH)), _f(b, EPS)))
        wp = b.int_to_ptr(q.a[w_ptr_name], PtrType(I32, "global"))
        w = _bf2(b, b.global_load_i32(wp, b.add(b.mul(self.layer, _c(b, IH // 2)), q.lane)))
        del wbase
        return [b.fmul(b.fmul(vals[e], rr), b.fadd(_f(b, 1.0), w[e])) for e in range(2)]

    def _rope2(self, vals: list[Value], p: Value, mutant: bool = False) -> list[Value]:
        """Partial RoPE of a 128-dim vector held 2 dims per lane: dims < 64 pair with the dims 32 away (lane ^ 16)."""
        b, q = self.b, self.q
        lane = q.lane
        out = []
        low = b.cmp_lt(lane, _c(b, 16))
        rotl = b.cmp_lt(lane, _c(b, 32))
        for e in range(2):
            partner = vals[e] if mutant else b.warp_shuffle_xor(vals[e], 16)
            fr = b.land(b.add(b.mul(lane, _c(b, 2)), _c(b, e)), _c(b, ROT // 2 - 1))
            cs = b.global_load_f32(self.rope, b.add(b.mul(p, _c(b, ROT)), fr))
            sn = b.global_load_f32(self.rope, b.add(b.mul(p, _c(b, ROT)), b.add(fr, _c(b, ROT // 2))))
            r = b.select(low, b.fsub(b.fmul(vals[e], cs), b.fmul(partner, sn)),
                         b.fadd(b.fmul(vals[e], cs), b.fmul(partner, sn)))
            out.append(b.select(rotl, r, vals[e]))
        return out

    def indexer_query(self) -> list[Value]:
        """Wave h (< 4) holds indexer query head h, normed and rotated at pos (2 dims per lane)."""
        b, q, sh = self.b, self.q, self.q.sh
        h = b.select(b.cmp_lt(q.wave, _c(b, IHN)), q.wave, _c(b, 0))
        off = b.mul(b.add(_c(b, sh.iq_col), b.add(b.mul(h, _c(b, IH)), b.mul(q.lane, _c(b, 2)))), _c(b, 2))
        x = _bf2(b, b.buffer_load_cp(self.proj, off, _c(b, 0), dwords=1, aux=0))
        n = self._norm2(x, None, "wiq")
        return self._rope2(n, self.pos, mutant=q.k.mutant == "iqrope")

    def indexer_key(self, query: list[Value] | None) -> None:
        """Wave 0 appends the raw key to the ring or completes a block (codes + scale); past the budget the new
        block is also scored with ``query`` (every wave h < 4 holds head h)."""
        b, q, sh = self.b, self.q, self.q.sh
        lane = q.lane
        slot = b.land(self.pos, _c(b, RATIO - 1))
        ring_off = b.mul(lane, _c(b, 4))  # bytes: 2 bf16 per lane
        complete = b.cmp_eq(slot, _c(b, RATIO - 1))
        with b.scf_if(D.is_wave0(b)):
            kw = b.buffer_load_cp(self.proj, b.mul(b.add(_c(b, sh.ik_col), b.mul(lane, _c(b, 2))), _c(b, 2)), _c(b, 0),
                                  dwords=1, aux=0)
            with b.scf_if(b.lnot(complete)):
                q.st(self.raw, b.add(b.mul(slot, _c(b, IH * 2)), ring_off), kw)
            with b.scf_if(complete):
                tot = _bf2(b, kw)
                for s_ in range(RATIO - 1):
                    pr = _bf2(b, b.buffer_load_cp(self.raw, b.add(_c(b, s_ * IH * 2), ring_off), _c(b, 0), dwords=1,
                                                  aux=q.cp_ld))
                    tot = [b.fadd(tot[e], pr[e]) for e in range(2)]
                mean = [b.fmul(tot[e], _f(b, 1.0 / RATIO)) for e in range(2)]
                normed = self._norm2(mean, None, "wik")
                start = b.sub(self.pos, _c(b, RATIO - 1))
                rotated = self._rope2(normed, start)
                mag = b.fmax(b.fabs(rotated[0]), b.fabs(rotated[1]))
                amax = _wmax(b, mag)
                bits = b.bitcast(amax, I32)
                bits = b.select(b.cmp_gt(bits, _c(b, FP8_FLOOR_BITS)), bits, _c(b, FP8_FLOOR_BITS))
                expo = b.lshr(bits, _c(b, 23))
                up = b.select(b.cmp_gt(b.land(bits, _c(b, 0x7FFFFF)), _c(b, 0x600000)), _c(b, 1), _c(b, 0))
                scale = b.bitcast(b.shl(b.add(b.sub(expo, _c(b, 8)), up), _c(b, 23)), F32)
                inverse = b.bitcast(b.shl(b.sub(b.sub(_c(b, 262), expo), up), _c(b, 23)), F32)
                scaled = [b.fmul(rotated[e], inverse) for e in range(2)]
                # lane 2w: own two codes low, lane 2w + 1's two codes high (dims 4w .. 4w + 3)
                pa = [b.bitcast(b.dpp_xor(b.bitcast(scaled[e], I32), 1), F32) for e in range(2)]
                even = b.cmp_eq(b.land(lane, _c(b, 1)), _c(b, 0))
                four = b.vec_pack([scaled[0], scaled[1], pa[0], pa[1]], F32)
                word = b.bitcast(b.cvt_pk_fp8_f32x4(four), I32)
                block = b.lshr(self.pos, _c(b, 2))
                with b.scf_if(even):
                    q.st(self.kc, b.add(b.mul(block, _c(b, IH)), b.mul(b.lshr(lane, _c(b, 1)), _c(b, 4))), word)
                with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
                    q.st(self.ks, b.mul(block, _c(b, 4)), b.bitcast(scale, I32))
                if query is not None:
                    # the stored key as the scan sees it (codes x scale), dims 2 lane .. 2 lane + 1
                    own = b.cvt_pk_f32_fp8x4(b.cvt_pk_fp8_f32x4(b.vec_pack([scaled[0], scaled[1], scaled[0], scaled[1]], F32)))
                    for e in range(2):
                        q.lw(b.add(_c(b, q.L.SHARED), b.add(b.mul(lane, _c(b, 2)), _c(b, e))),
                             b.fmul(b.vec_extract(own, e), scale))
        if query is not None:
            with b.scf_if(complete):
                q.bar()
                with b.scf_if(b.cmp_lt(q.wave, _c(b, IHN))):
                    k0 = q.lr(b.add(_c(b, q.L.SHARED), b.mul(lane, _c(b, 2))), F32)
                    k1 = q.lr(b.add(_c(b, q.L.SHARED), b.add(b.mul(lane, _c(b, 2)), _c(b, 1))), F32)
                    dot = _wsum(b, b.fma(query[1], k1, b.fmul(query[0], k0)))
                    with b.scf_if(b.cmp_eq(lane, _c(b, 0))):
                        q.lw(b.add(_c(b, q.L.TOPS), q.wave), b.fmax(dot, _f(b, 0.0)))
                q.bar()
                with b.scf_if(D.is_thread0(b)):
                    tot = q.lr(_c(b, q.L.TOPS), F32)
                    for h in range(1, IHN):
                        tot = b.fadd(tot, q.lr(_c(b, q.L.TOPS + h), F32))
                    q.st(self.score, b.mul(b.sub(self.blocks, _c(b, 1)), _c(b, 4)),
                         b.bitcast(b.fmul(tot, _f(b, IH ** -0.5)), I32))
                q.bar()

    # ------------------------------------------------ scan / select
    def _scan_loads(self, chunk: Value) -> list[Value]:
        """The chunk's key loads (and scales), flat (loop-carried by the pipelined scan)."""
        b, q = self.b, self.q
        T = q.T
        CH = (T // 16) * q.k.scan_its
        if q.k.score == "mfma":
            groups = CH // (16 * q.W)
            n = b.land(q.lane, _c(b, 15))
            g = b.lshr(q.lane, _c(b, 4))
            base0 = b.add(b.mul(chunk, _c(b, CH)), b.mul(q.wave, _c(b, 16 * groups)))
            out = []
            for gi in range(groups):
                j = b.add(base0, b.add(_c(b, 16 * gi), n))
                jc = b.select(b.cmp_lt(j, self.old), j, _c(b, 0))
                if q.k.kw16:
                    out += [b.buffer_load_cp(self.kc, b.add(b.mul(jc, _c(b, IH)), b.add(b.mul(g, _c(b, 32)), _c(b, 16 * h))),
                                             _c(b, 0), dwords=4, aux=0) for h in range(2)]
                else:
                    out += [b.buffer_load_cp(self.kc, b.add(b.mul(jc, _c(b, IH)), b.add(_c(b, 32 * c), b.mul(g, _c(b, 8)))),
                                             _c(b, 0), dwords=2, aux=0) for c in range(IH // 32)]
                blk0 = b.add(base0, b.add(_c(b, 16 * gi), b.mul(g, _c(b, 4))))
                out.append(self.q.ld(self.ks, b.mul(blk0, _c(b, 4)), 4, 0))
            return out
        sub = b.land(q.tid, _c(b, 15))
        first = b.add(b.mul(chunk, _c(b, CH)), b.lshr(q.tid, _c(b, 4)))
        keys, scales = [], []
        for it in range(q.k.scan_its):
            j = b.add(first, _c(b, (T // 16) * it))
            jc = b.select(b.cmp_lt(j, self.old), j, _c(b, 0))
            keys.append(b.buffer_load_cp(self.kc, b.add(b.mul(jc, _c(b, IH)), b.mul(sub, _c(b, 8))), _c(b, 0), dwords=2,
                                         aux=0))
            scales.append(b.buffer_load_cp(self.ks, b.mul(jc, _c(b, 4)), _c(b, 0), dwords=1, aux=0))
        return keys + scales

    def _scan_compute(self, chunk: Value, loads: list[Value], qs) -> None:
        b, q = self.b, self.q
        T = q.T
        CH = (T // 16) * q.k.scan_its
        if q.k.score == "mfma":
            groups = CH // (16 * q.W)
            n = b.land(q.lane, _c(b, 15))
            g = b.lshr(q.lane, _c(b, 4))
            base0 = b.add(b.mul(chunk, _c(b, CH)), b.mul(q.wave, _c(b, 16 * groups)))
            nld = 2 if q.k.kw16 else IH // 32
            per = nld + 1
            for gi in range(groups):
                acc = b.zero_vec_f32(4)
                for c in range(IH // 32):
                    if q.k.kw16:
                        w4 = loads[gi * per + c // 2]
                        words = [b.vec_extract(w4, 2 * (c % 2) + w) for w in range(2)]
                    else:
                        w2 = loads[gi * per + c]
                        words = [b.vec_extract(w2, w) for w in range(2)]
                    f = []
                    for wd in words:
                        v = b.cvt_pk_f32_fp8x4(b.vec_bitcast(b.vec_pack([wd], I32), VectorType(_FP8, 4)))
                        f += [b.vec_extract(v, e) for e in range(4)]
                    acc = b.mfma_f32_16x16x32_bf16(b.vec_trunc_f32_to_bf16(b.vec_pack(f, F32)), qs[c], acc)
                sc = loads[gi * per + nld]
                blk0 = b.add(base0, b.add(_c(b, 16 * gi), b.mul(g, _c(b, 4))))
                for e in range(4):
                    x = b.fmax(b.vec_extract(acc, e), _f(b, 0.0))
                    if q.k.mutant == "scan":
                        x = b.select(b.cmp_eq(n, _c(b, 3)), _f(b, 0.0), x)
                    x = b.fadd(x, b.bitcast(b.dpp_xor(b.bitcast(x, I32), 1), F32))
                    x = b.fadd(x, b.bitcast(b.dpp_xor(b.bitcast(x, I32), 2), F32))
                    jb = b.add(blk0, _c(b, e))
                    scale = b.bitcast(b.vec_extract(sc, e), F32)
                    with b.scf_if(b.land(b.cmp_lt(jb, self.old), b.cmp_eq(n, _c(b, 0)))):
                        q.st(self.score, b.mul(jb, _c(b, 4)), b.bitcast(b.fmul(b.fmul(x, scale), _f(b, IH ** -0.5)), I32))
            return
        its = q.k.scan_its
        sub = b.land(q.tid, _c(b, 15))
        first = b.add(b.mul(chunk, _c(b, CH)), b.lshr(q.tid, _c(b, 4)))
        for it in range(its):
            j = b.add(first, _c(b, (T // 16) * it))
            kk = []
            for w in range(2):
                v = b.cvt_pk_f32_fp8x4(b.vec_bitcast(b.vec_pack([b.vec_extract(loads[it], w)], I32), VectorType(_FP8, 4)))
                kk += [b.vec_extract(v, e) for e in range(4)]
            total = _f(b, 0.0)
            for h in range(IHN):
                dot = _f(b, 0.0)
                for e in range(8):
                    dot = b.fma(qs[h][e], kk[e], dot)
                dot = _wsum(b, dot, (1, 2, 4, 8))
                if q.k.mutant == "scan" and h == IHN - 1:
                    continue
                total = b.fadd(total, b.fmax(dot, _f(b, 0.0)))
            with b.scf_if(b.land(b.cmp_lt(j, self.old), b.cmp_eq(sub, _c(b, 0)))):
                q.st(self.score, b.mul(j, _c(b, 4)),
                     b.bitcast(b.fmul(b.fmul(total, b.bitcast(loads[its + it], F32)), _f(b, IH ** -0.5)), I32))

    def scan(self, chunk: Value, qs) -> None:
        """Scores of one chunk of blocks below ``old`` (``_scan_loads`` then ``_scan_compute``)."""
        self._scan_compute(chunk, self._scan_loads(chunk), qs)

    def _key_of(self, w: Value) -> Value:
        return self.b.lshr(w, _c(self.b, 15))

    def select(self, to_lds: bool = False) -> None:
        """The budget's blocks by descending 16-bit key, ties to the lower block, ascending into ``blk``."""
        b, q, k = self.b, self.q, self.q.k
        T, W, L = q.T, q.W, q.L
        tid, lane, wave = q.tid, q.lane, q.wave
        blocks = self.blocks
        R = 8 * T
        kc = _keys_count(k)
        single = kc <= 4096
        rounds_all = kc // R
        rounds = b.div(b.add(blocks, _c(b, R - 1)), _c(b, R))
        self.stamp(8)
        # keys -> LDS (2 per word), 8 per thread per round
        with b.scf_for(_c(b, 0), _c(b, rounds_all) if single else rounds, _c(b, 1), iv_name=_iv("sk")) as r:
            j = b.add(b.mul(r, _c(b, R)), b.mul(tid, _c(b, 8)))
            words = []
            for hh in range(2):
                jj = b.add(j, _c(b, 4 * hh))
                v = self.q.ld(self.score, b.mul(b.select(b.cmp_lt(jj, blocks), jj, _c(b, 0)), _c(b, 4)), 4, q.cp_ld)
                for e in range(4):
                    words.append(b.select(b.cmp_lt(b.add(jj, _c(b, e)), blocks), self._key_of(b.vec_extract(v, e)),
                                          _c(b, 0)))
            packed = [b.lor(words[2 * i], b.shl(words[2 * i + 1], _c(b, 16))) for i in range(4)]
            b.smem_store_vN(q.lds, [b.add(_c(b, L.BIG), b.lshr(j, _c(b, 1)))], b.vec_pack(packed, I32), 4)
        q.bar()
        self.stamp(9)
        if k.sel == "bits":
            boundary = self._bits_boundary(rounds, single, rounds_all)
        elif k.sel == "bits2":
            boundary = self._bits2_boundary(rounds, single, rounds_all)
        elif k.sel in ("v1", "v2"):
            boundary = self._vcnt_boundary(rounds, single, rounds_all, 1 if k.sel == "v1" else 2)
        elif k.sel == "nib":
            boundary = self._nib_boundary(rounds, single, rounds_all)
        else:
            boundary = self._hist_boundary(rounds, single, rounds_all)
        self.stamp(10)
        # per-thread contiguous runs: strict and tied counts, block scan, then the writes
        per = b.mul(b.div(b.add(blocks, _c(b, 8 * T - 1)), _c(b, 8 * T)), _c(b, 8))
        lo = b.mul(tid, per)
        hi = b.smin(b.add(lo, per), blocks)
        lp = b.scf_for_iter(lo, hi, _c(b, 8), _ia([("st0", _c(b, 0)), ("ti0", _c(b, 0))]), iv_name=_iv("sr"))
        with lp as (j, (st0, ti0)):
            ks = self._keys8(j)
            for e in range(8):
                valid = b.cmp_lt(b.add(j, _c(b, e)), hi)
                st0 = b.add(st0, b.select(b.land(valid, b.cmp_gt(ks[e], boundary)), _c(b, 1), _c(b, 0)))
                ti0 = b.add(ti0, b.select(b.land(valid, b.cmp_eq(ks[e], boundary)), _c(b, 1), _c(b, 0)))
            b.scf_yield(st0, ti0)
        strict, ties = lp.results
        sb, tb, stot = self._block_scan2(strict, ties)
        self.stamp(11)
        want = b.sub(_c(b, NBLK), stot)
        take = b.sub(want, tb)
        take = b.select(b.cmp_lt(take, _c(b, 0)), _c(b, 0), b.smin(take, ties))
        if k.mutant == "tie":
            # ties to the HIGHER block: skip the first (ties - take) ties of the run instead of taking the first
            skip = b.sub(ties, take)
        else:
            skip = None
        slot = b.add(sb, b.smin(tb, want))
        with b.scf_for_iter(lo, hi, _c(b, 8), _ia([("sl0", slot), ("se0", _c(b, 0))]), iv_name=_iv("sw")) as (j, (sl0, se0)):
            ks = self._keys8(j)
            for e in range(8):
                valid = b.cmp_lt(b.add(j, _c(b, e)), hi)
                tie = b.land(valid, b.cmp_eq(ks[e], boundary))
                if skip is None:
                    tk = b.land(tie, b.cmp_lt(se0, take))
                else:
                    tk = b.land(tie, b.cmp_ge(se0, skip))
                chosen = b.lor(b.land(valid, b.cmp_gt(ks[e], boundary)), tk)
                with b.scf_if(chosen):
                    if to_lds:
                        q.lw(b.add(_c(b, L.BLK), sl0), b.add(j, _c(b, e)))
                        with b.scf_if(self.first):
                            q.st(self.blk, b.mul(sl0, _c(b, 4)), b.add(j, _c(b, e)))
                    else:
                        q.st(self.blk, b.mul(sl0, _c(b, 4)), b.add(j, _c(b, e)))
                sl0 = b.add(sl0, b.select(chosen, _c(b, 1), _c(b, 0)))
                se0 = b.add(se0, b.select(tie, _c(b, 1), _c(b, 0)))
            b.scf_yield(sl0, se0)
        self.stamp(12)

    def _keys8(self, j: Value) -> list[Value]:
        b, q = self.b, self.q
        v = b.smem_load_vN(q.lds, b.add(_c(b, q.L.BIG), b.lshr(j, _c(b, 1))), dtype=I32, n=4)
        out = []
        for i in range(4):
            w = b.vec_extract(v, i)
            out += [b.land(w, _c(b, 0xFFFF)), b.lshr(w, _c(b, 16))]
        return out

    def _count_ge(self, cand: Value, rounds: Value, single: bool, rounds_all: int, resident) -> Value:
        b, q = self.b, self.q
        R = 8 * q.T
        if single:
            cnt = _c(b, 0)
            for ks in resident:
                for kk in ks:
                    cnt = b.add(cnt, _wave_count(b, b.cmp_ge(kk, cand)))
            return cnt
        lp = b.scf_for_iter(_c(b, 0), rounds, _c(b, 1), _ia([("cg0", _c(b, 0))]), iv_name=_iv("cr"))
        with lp as (r, (cg0,)):
            ks = self._keys8(b.add(b.mul(r, _c(b, R)), b.mul(q.tid, _c(b, 8))))
            for kk in ks:
                cg0 = b.add(cg0, _wave_count(b, b.cmp_ge(kk, cand)))
            b.scf_yield(cg0)
        return lp.results[0]

    def _bits_boundary(self, rounds: Value, single: bool, rounds_all: int) -> Value:
        """FlyTape's search: the largest K with at least 512 keys >= K, bit by bit from the top."""
        b, q = self.b, self.q
        W, L = q.W, q.L
        R = 8 * q.T
        resident = [self._keys8(b.add(_c(b, r * R), b.mul(q.tid, _c(b, 8)))) for r in range(rounds_all)] if single else None
        boundary = _c(b, 0)
        for bit in range(KEY_BITS):
            cand = b.add(boundary, _c(b, 1 << (KEY_BITS - 1 - bit)))
            cnt = self._count_ge(cand, rounds, single, rounds_all, resident)
            with b.scf_if(b.cmp_eq(q.lane, _c(b, 0))):
                q.lw(b.add(_c(b, L.CNTS + bit * W), q.wave), cnt)
            q.bar()
            tot = q.lr(_c(b, L.CNTS + bit * W))
            for w in range(1, W):
                tot = b.add(tot, q.lr(_c(b, L.CNTS + bit * W + w)))
            boundary = b.select(b.cmp_ge(tot, _c(b, NBLK)), cand, boundary)
        return b.readfirstlane(boundary)

    def _vcnt_boundary(self, rounds: Value, single: bool, rounds_all: int, bits: int) -> Value:
        """The bit search with ``bits`` bits per round (``2 ** bits - 1`` candidates), counted on the VALU: each lane
        adds its keys' compares, one integer butterfly per candidate gives the wave count; one LDS exchange + barrier
        per round. The same boundary as ``_bits_boundary``."""
        b, q = self.b, self.q
        W, L = q.W, q.L
        R = 8 * q.T
        nc = (1 << bits) - 1
        resident = [self._keys8(b.add(_c(b, r * R), b.mul(q.tid, _c(b, 8)))) for r in range(rounds_all)] if single else None
        boundary = _c(b, 0)
        one, zero = _c(b, 1), _c(b, 0)
        for rd in range(KEY_BITS // bits):
            sh = KEY_BITS - bits - bits * rd
            cands = [b.add(boundary, _c(b, m << sh)) for m in range(1, nc + 1)]
            if single:
                cnt = [zero] * nc
                for ks in resident:
                    for kk in ks:
                        cnt = [b.add(cnt[m], b.select(b.cmp_ge(kk, cands[m]), one, zero)) for m in range(nc)]
            else:
                lp = b.scf_for_iter(_c(b, 0), rounds, _c(b, 1), _ia([(f"vc{m}", zero) for m in range(nc)]),
                                    iv_name=_iv("vcr"))
                with lp as (r, it):
                    cc = list(it)
                    for kk in self._keys8(b.add(b.mul(r, _c(b, R)), b.mul(q.tid, _c(b, 8)))):
                        cc = [b.add(cc[m], b.select(b.cmp_ge(kk, cands[m]), one, zero)) for m in range(nc)]
                    b.scf_yield(*cc)
                cnt = lp.results
            wsum = []
            for v in cnt:
                for off in (1, 2, 4, 8, 16, 32):
                    pt = _shfl(b, b.bitcast(v, F32), off)
                    if isinstance(pt, tuple):
                        v = b.add(b.bitcast(pt[0], I32), b.bitcast(pt[1], I32))
                    else:
                        v = b.add(v, b.bitcast(pt, I32))
                wsum.append(v)
            base = L.CNTS + (rd % 2) * nc * W
            with b.scf_if(b.cmp_eq(q.lane, _c(b, 0))):
                for m in range(nc):
                    q.lw(b.add(_c(b, base + m * W), q.wave), wsum[m])
            q.bar()
            nb = boundary
            for m in range(nc):
                tot = q.lr(_c(b, base + m * W))
                for w in range(1, W):
                    tot = b.add(tot, q.lr(_c(b, base + m * W + w)))
                nb = b.select(b.cmp_ge(tot, _c(b, NBLK)), cands[m], nb)
            boundary = nb
        return b.readfirstlane(boundary)

    def _bits2_boundary(self, rounds: Value, single: bool, rounds_all: int) -> Value:
        """FlyTape's search two bits per round: the counts of keys >= prefix + {1, 2, 3} << shift in one pass (three
        ballots per key), one LDS exchange + barrier per round; picks the largest candidate whose count reaches 512 -
        the same boundary as the 16 single-bit rounds."""
        b, q = self.b, self.q
        W, L = q.W, q.L
        R = 8 * q.T
        resident = [self._keys8(b.add(_c(b, r * R), b.mul(q.tid, _c(b, 8)))) for r in range(rounds_all)] if single else None
        boundary = _c(b, 0)
        for rd in range(KEY_BITS // 2):
            sh = KEY_BITS - 2 - 2 * rd
            cands = [b.add(boundary, _c(b, m << sh)) for m in (1, 2, 3)]
            if single:
                cnt = [_c(b, 0)] * 3
                for ks in resident:
                    for kk in ks:
                        cnt = [b.add(cnt[m], _wave_count(b, b.cmp_ge(kk, cands[m]))) for m in range(3)]
            else:
                lp = b.scf_for_iter(_c(b, 0), rounds, _c(b, 1), _ia([(f"c2{m}", _c(b, 0)) for m in range(3)]),
                                    iv_name=_iv("c2r"))
                with lp as (r, it):
                    cc = list(it)
                    for kk in self._keys8(b.add(b.mul(r, _c(b, R)), b.mul(q.tid, _c(b, 8)))):
                        cc = [b.add(cc[m], _wave_count(b, b.cmp_ge(kk, cands[m]))) for m in range(3)]
                    b.scf_yield(*cc)
                cnt = lp.results
            base = L.CNTS + (rd % 2) * 3 * W
            with b.scf_if(b.cmp_eq(q.lane, _c(b, 0))):
                for m in range(3):
                    q.lw(b.add(_c(b, base + m * W), q.wave), cnt[m])
            q.bar()
            for m in range(3):
                tot = q.lr(_c(b, base + m * W))
                for w in range(1, W):
                    tot = b.add(tot, q.lr(_c(b, base + m * W + w)))
                cnt[m] = tot
            nb = boundary  # counts fall as the candidate grows: the largest candidate still reaching 512 wins
            for m in range(3):
                nb = b.select(b.cmp_ge(cnt[m], _c(b, NBLK)), cands[m], nb)
            boundary = nb
        return b.readfirstlane(boundary)

    def _nib_boundary(self, rounds: Value, single: bool, rounds_all: int) -> Value:
        """The bit search's boundary in four 4-bit digit rounds. Per round: every lane counts its keys that match the
        prefix above the digit into 16 packed 8-bit bins (4 VGPRs), the wave sums them as 16-bit pairs (DPP / permlane
        butterfly), one LDS exchange + barrier, then every lane picks the largest digit whose suffix count reaches the
        remaining need. The same K as ``_bits_boundary`` (the largest K with >= 512 keys >= K)."""
        b, q = self.b, self.q
        W, L = q.W, q.L
        R = 8 * q.T
        resident = [self._keys8(b.add(_c(b, r * R), b.mul(q.tid, _c(b, 8)))) for r in range(rounds_all)] if single else None
        prefix = _c(b, 0)
        above = _c(b, 0)
        for d in range(4):
            sh = 12 - 4 * d

            def count_into(regs, kk):
                if d == 0:
                    match = None
                else:
                    match = b.cmp_eq(b.lshr(kk, _c(b, sh + 4)), b.lshr(prefix, _c(b, sh + 4)))
                nib = b.land(b.lshr(kk, _c(b, sh)), _c(b, 15))
                one = b.shl(_c(b, 1), b.mul(b.land(nib, _c(b, 3)), _c(b, 8)))
                sel = b.lshr(nib, _c(b, 2))
                out = []
                for i in range(4):
                    hit = b.cmp_eq(sel, _c(b, i))
                    if match is not None:
                        hit = b.land(hit, match)
                    out.append(b.add(regs[i], b.select(hit, one, _c(b, 0))))
                return out

            regs = [_c(b, 0)] * 4
            if single:
                for ks in resident:
                    for kk in ks:
                        regs = count_into(regs, kk)
            else:
                lp = b.scf_for_iter(_c(b, 0), rounds, _c(b, 1), _ia([(f"nb{i}", regs[i]) for i in range(4)]),
                                    iv_name=_iv("nr"))
                with lp as (r, it):
                    rr = list(it)
                    for kk in self._keys8(b.add(b.mul(r, _c(b, R)), b.mul(q.tid, _c(b, 8)))):
                        rr = count_into(rr, kk)
                    b.scf_yield(*rr)
                regs = lp.results
            # 16-bit pairs: word 2i = bins (4i, 4i + 2), word 2i + 1 = bins (4i + 1, 4i + 3); wave sums
            pairs = []
            for i in range(4):
                pairs.append(b.land(regs[i], _c(b, 0x00FF00FF)))
                pairs.append(b.land(b.lshr(regs[i], _c(b, 8)), _c(b, 0x00FF00FF)))
            sums = []
            for v in pairs:
                for off in (1, 2, 4, 8, 16, 32):
                    pt = _shfl(b, b.bitcast(v, F32), off)
                    if isinstance(pt, tuple):
                        v = b.add(b.bitcast(pt[0], I32), b.bitcast(pt[1], I32))
                    else:
                        v = b.add(v, b.bitcast(pt, I32))
                sums.append(v)
            base = L.NIB + d * W * 8
            with b.scf_if(b.cmp_eq(q.lane, _c(b, 0))):
                for i in range(8):
                    q.lw(b.add(_c(b, base + i), b.mul(q.wave, _c(b, 8))), sums[i])
            q.bar()
            tot = [_c(b, 0)] * 8
            for w in range(W):
                for i in range(8):
                    tot[i] = b.add(tot[i], q.lr(_c(b, base + w * 8 + i)))
            hist = [None] * 16
            for i in range(4):
                lo, hi = tot[2 * i], tot[2 * i + 1]
                hist[4 * i + 0] = b.land(lo, _c(b, 0xFFFF))
                hist[4 * i + 2] = b.lshr(lo, _c(b, 16))
                hist[4 * i + 1] = b.land(hi, _c(b, 0xFFFF))
                hist[4 * i + 3] = b.lshr(hi, _c(b, 16))
            need = b.sub(_c(b, NBLK), above)
            cum = _c(b, 0)
            dig = _c(b, 0)
            add_above = _c(b, 0)
            found = None
            for j in range(15, -1, -1):
                ncum = b.add(cum, hist[j])
                hit = b.cmp_ge(ncum, need)
                first_hit = hit if found is None else b.land(hit, b.lnot(found))
                dig = b.select(first_hit, _c(b, j), dig)
                add_above = b.select(first_hit, cum, add_above)
                found = hit if found is None else b.lor(found, hit)
                cum = ncum
            prefix = b.lor(prefix, b.shl(dig, _c(b, sh)))
            above = b.add(above, add_above)
        return b.readfirstlane(prefix)

    def _hist_boundary(self, rounds: Value, single: bool, rounds_all: int) -> Value:
        """Two passes: a 256-bin histogram of the high bytes (LDS atomics), the bin holding the 512th largest key, then
        a histogram of the low bytes inside that bin. The same boundary as the bit search."""
        b, q = self.b, self.q
        T, L = q.T, q.L
        R = 8 * T
        H = L.HIST
        nr = _c(b, rounds_all) if single else rounds

        def clear():
            for i in range(0, 256, T):
                with b.scf_if(b.cmp_lt(b.add(q.tid, _c(b, i)), _c(b, 256))):
                    q.lw(b.add(_c(b, H), b.add(q.tid, _c(b, i))), _c(b, 0))
            q.bar()

        def fill(sel_hi: Value | None):
            with b.scf_for(_c(b, 0), nr, _c(b, 1), iv_name=_iv("hf") if sel_hi is None else "hg") as r:
                ks = self._keys8(b.add(b.mul(r, _c(b, R)), b.mul(q.tid, _c(b, 8))))
                for kk in ks:
                    if sel_hi is None:
                        bin_ = b.lshr(kk, _c(b, 8))
                        cond = None
                    else:
                        bin_ = b.land(kk, _c(b, 255))
                        cond = b.cmp_eq(b.lshr(kk, _c(b, 8)), sel_hi)
                    addr = b.add(q.lds0, b.mul(b.add(_c(b, H), bin_), _c(b, 4)))
                    if cond is None:
                        b.inline_asm("ds_add_u32 $0, $1", "v,v,~{memory}", [addr, _c(b, 1)], None)
                    else:
                        with b.scf_if(cond):
                            b.inline_asm("ds_add_u32 $0, $1", "v,v,~{memory}", [addr, _c(b, 1)], None)
            b.inline_asm("s_waitcnt lgkmcnt(0)", "~{memory}", [], None)
            q.bar()

        def find(need: Value) -> tuple[Value, Value]:
            """Thread 0: the largest bin d with sum(bins >= d) >= need; returns (d, count of keys in bins > d)."""
            with b.scf_if(D.is_thread0(b)):
                lp = b.scf_for_iter(_c(b, 0), _c(b, 256), _c(b, 1),
                                    _ia([("fa", _c(b, 0)), ("fd", _c(b, -1)), ("fx", _c(b, 0))]), iv_name=_iv("hs"))
                with lp as (i, (fa, fd, fx)):
                    d = b.sub(_c(b, 255), i)
                    nfa = b.add(fa, q.lr(b.add(_c(b, H), d)))
                    hit = b.land(b.cmp_lt(fd, _c(b, 0)), b.cmp_ge(nfa, need))
                    fd = b.select(hit, d, fd)
                    fx = b.select(hit, fa, fx)
                    b.scf_yield(nfa, fd, fx)
                _fa, fd, fx = lp.results
                q.lw(_c(b, 12), fd)
                q.lw(_c(b, 13), fx)
            q.bar()
            return b.readfirstlane(q.lr(_c(b, 12))), b.readfirstlane(q.lr(_c(b, 13)))

        clear()
        fill(None)
        hi, above_hi = find(_c(b, NBLK))
        q.bar()
        clear()
        fill(hi)
        lo, _ = find(b.sub(_c(b, NBLK), above_hi))
        q.bar()
        return b.lor(b.shl(hi, _c(b, 8)), lo)

    def _block_scan2(self, a: Value, c: Value) -> tuple[Value, Value, Value]:
        """Exclusive prefix sums over the CTA's threads of two i32 and the first one's total."""
        b, q = self.b, self.q
        lane, W, L = q.lane, q.W, q.L
        xa, xc = a, c
        for st in range(6):
            d = 1 << st
            src = b.mul(b.sub(lane, _c(b, d)), _c(b, 4))
            pa = b.ds_bpermute(src, xa)
            pc = b.ds_bpermute(src, xc)
            ge = b.cmp_ge(lane, _c(b, d))
            xa = b.add(xa, b.select(ge, pa, _c(b, 0)))
            xc = b.add(xc, b.select(ge, pc, _c(b, 0)))
        q.bar()
        with b.scf_if(b.cmp_eq(lane, _c(b, 63))):
            q.lw(b.add(_c(b, L.SCAN), q.wave), xa)
            q.lw(b.add(_c(b, L.SCAN + W), q.wave), xc)
        q.bar()
        ba, bc, ta = _c(b, 0), _c(b, 0), _c(b, 0)
        for w in range(W):
            t_a = q.lr(_c(b, L.SCAN + w))
            t_c = q.lr(_c(b, L.SCAN + W + w))
            before = b.cmp_lt(_c(b, w), q.wave)
            ba = b.add(ba, b.select(before, t_a, _c(b, 0)))
            bc = b.add(bc, b.select(before, t_c, _c(b, 0)))
            ta = b.add(ta, t_a)
        q.bar()
        return b.sub(b.add(ba, xa), a), b.sub(b.add(bc, xc), c), ta

    # ------------------------------------------------ windows / positions
    def window(self):
        """The member's split of the row's attended list: whole stage rounds (FlyTape ``_window``)."""
        b, q = self.b, self.q
        G, ST = q.G, q.ST
        kept = b.mul(b.smin(self.blocks, _c(b, NBLK)), _c(b, RATIO))
        tail = b.land(b.add(self.pos, _c(b, 1)), _c(b, RATIO - 1))
        count = b.add(kept, tail)
        if q.k.mutant == "window":
            count = b.sub(count, _c(b, 1))
        per = b.mul(b.div(b.add(count, _c(b, G * ST - 1)), _c(b, G * ST)), _c(b, ST))
        first = b.mul(self.member, per)
        take = b.sub(count, first)
        take = b.select(b.cmp_lt(take, _c(b, 0)), _c(b, 0), b.smin(take, per))
        return first, take, kept

    def build_positions(self, first: Value, take: Value, kept: Value) -> None:
        b, q, k = self.b, self.q, self.q.k
        T = q.T
        rounds = _pos_rounds(k)
        listed = []
        for r in range(rounds):
            index = b.add(first, b.add(q.tid, _c(b, r * T)))
            blk = b.select(b.cmp_lt(index, kept), b.lshr(index, _c(b, 2)), _c(b, 0))
            blk = b.select(b.cmp_lt(blk, _c(b, NBLK)), blk, _c(b, 0))
            if k.sela:
                listed.append(q.lr(b.add(_c(b, q.L.BLK), blk)))
            else:
                listed.append(self.q.ld(self.blk, b.mul(blk, _c(b, 4)), 1, q.cp_ld))
        for r in range(rounds):
            local = b.add(q.tid, _c(b, r * T))
            index = b.add(first, local)
            p_in = b.add(b.mul(listed[r], _c(b, RATIO)), b.land(index, _c(b, RATIO - 1)))
            p_tail = b.add(b.mul(self.blocks, _c(b, RATIO)), b.sub(index, kept))
            p = b.select(b.cmp_lt(index, kept), p_in, p_tail)
            p = b.select(b.cmp_lt(local, take), p, _c(b, 0))
            q.lw(b.add(_c(b, q.L.POS), local), p)
        q.bar()

    # ------------------------------------------------ phases
    def dense_prep(self) -> None:
        b, q, sh = self.b, self.q, self.q.sh
        with b.scf_if(self.first):
            self.indexer_key(None)
        groups = -(-(sh.nh + 2 * sh.nkv) // q.W)
        self.heads([_c(b, g) for g in range(groups)], glob_q=not q.k.qloc)
        q.wait_stores()
        q.bar()

    def sparse_prep(self) -> None:
        b, q, sh, k = self.b, self.q, self.q.sh, self.q.k
        G = q.G
        groups = -(-(sh.nh + 2 * sh.nkv) // q.W)
        lane = q.lane
        if k.stress:
            with b.scf_if(b.lnot(self.first)):  # pretouch the indexer query's lines (stale-read twin setup)
                t = self.q.ld(self.iq, b.mul(q.tid, _c(b, 4)), 1, 0)
                b.inline_asm("s_waitcnt vmcnt(0)", "v,~{memory}", [t], None)
        with b.scf_if(self.first):
            qv = self.indexer_query()
            with b.scf_if(b.cmp_lt(q.wave, _c(b, IHN))):
                q.st(self.iq, b.mul(b.add(b.mul(q.wave, _c(b, IH)), b.mul(lane, _c(b, 2))), _c(b, 4)),
                     b.vec_pack([b.bitcast(qv[0], I32), b.bitcast(qv[1], I32)], I32))
            self.indexer_key(qv)
            if k.stress:
                with b.scf_for(_c(b, 0), _c(b, k.stress), _c(b, 1), iv_name=_iv("qstr")):
                    D.s_sleep(b, 64)
            q.publish(self.rdy_base, _c(b, 0), 1)
        gl = [b.add(self.member, _c(b, i * G)) for i in range(-(-groups // G))]
        self.heads(gl)
        q.wait(self.rdy_base, _c(b, 0), 1, self.first)
        sub = b.land(q.tid, _c(b, 15))
        qs = []
        if k.score == "mfma":
            n16 = b.land(lane, _c(b, 15))
            g16 = b.lshr(lane, _c(b, 4))
            hq = b.select(b.cmp_lt(n16, _c(b, IHN)), n16, _c(b, 0))
            for c in range(IH // 32):
                if k.kw16:
                    at = b.mul(b.add(b.mul(hq, _c(b, IH)), b.add(b.mul(g16, _c(b, 32)), _c(b, 8 * c))), _c(b, 4))
                else:
                    at = b.mul(b.add(b.mul(hq, _c(b, IH)), b.add(_c(b, 32 * c), b.mul(g16, _c(b, 8)))), _c(b, 4))
                v0 = self.q.ld(self.iq, at, 4, q.cp_ld)
                v1 = self.q.ld(self.iq, b.add(at, _c(b, 16)), 4, q.cp_ld)
                f = [b.select(b.cmp_lt(n16, _c(b, IHN)), b.bitcast(b.vec_extract(v, e), F32), _f(b, 0.0))
                     for v in (v0, v1) for e in range(4)]
                qs.append(b.vec_trunc_f32_to_bf16(b.vec_pack(f, F32)))
        else:
            for h in range(IHN):
                at = b.mul(b.add(_c(b, h * IH), b.mul(sub, _c(b, 8))), _c(b, 4))
                v0 = self.q.ld(self.iq, at, 4, q.cp_ld)
                v1 = self.q.ld(self.iq, b.add(at, _c(b, 16)), 4, q.cp_ld)
                qs.append([b.bitcast(b.vec_extract(v, e), F32) for v in (v0, v1) for e in range(4)])
        CH = (q.T // 16) * k.scan_its
        units = b.div(b.add(self.old, _c(b, CH - 1)), _c(b, CH))
        if k.spf:
            mine = b.select(b.cmp_gt(units, self.member), b.div(b.add(b.sub(units, self.member), _c(b, G - 1)), _c(b, G)),
                            _c(b, 0))
            npairs = b.div(b.add(mine, _c(b, 1)), _c(b, 2))
            cur = self._scan_loads(self.member)
            lp = b.scf_for_iter(_c(b, 0), npairs, _c(b, 1), _ia([(f"sp{i}", v) for i, v in enumerate(cur)]),
                                iv_name=_iv("qsp"))
            with lp as (pi, it):
                c0 = b.add(self.member, b.mul(pi, _c(b, 2 * G)))
                nxt = self._scan_loads(b.add(c0, _c(b, G)))
                self._scan_compute(c0, list(it), qs)
                nxt2 = self._scan_loads(b.add(c0, _c(b, 2 * G)))
                self._scan_compute(b.add(c0, _c(b, G)), nxt, qs)
                b.scf_yield(*nxt2)
        else:
            with b.scf_for(self.member, units, _c(b, G), iv_name=_iv("qsc")) as chunk:
                self.scan(chunk, qs)
        self.stamp(1)
        if k.sela:
            old_a = q.arrive(self.cnt_base, _c(b, 4 * self.CA))
            self.stamp(2)
            q.wait(self.cnt_base, _c(b, 4 * self.CA), G, b.cmp_eq(old_a, _c(b, G - 1)))
            self.select(to_lds=True)
            q.bar()
        else:
            last = b.cmp_eq(q.arrive(self.cnt_base, _c(b, 4 * self.CA)), _c(b, G - 1))
            self.stamp(2)
            with b.scf_if(last):
                self.select()
                q.reset(self.cnt_base, _c(b, 4 * self.CA), G)
                q.publish(self.rdy_base, _c(b, 4), 1)
            q.wait(self.rdy_base, _c(b, 4), 1, last)
        self.stamp(3)
        first, take, kept = self.window()
        self.build_positions(first, take, kept)
        if k.qloc:
            # every q head of the row (written by the members' heads prep, ordered by the scan arrivals) into LDS
            for i in range(-(-(sh.nh * HD // 8) // q.T)):
                e = b.add(q.tid, _c(b, i * q.T))
                with b.scf_if(b.cmp_lt(e, _c(b, sh.nh * HD // 8))):
                    v = q.ld(self.qb, b.mul(e, _c(b, 16)), 4, q.cp_ld)
                    b.smem_store_vN(q.lds, [b.add(_c(b, q.L.QL), b.mul(e, _c(b, 4)))], v, 4)
            q.bar()
        self.stamp(4)

    def attend_all(self, above: Value) -> None:
        b, q, sh = self.b, self.q, self.q.sh
        first, take, _kept = self.window()
        with b.scf_for(_c(b, 0), _c(b, sh.nkv), _c(b, 1), iv_name=_iv("qkh")) as kvh:
            self.attention(kvh, first, take, above)
            q.bar()
            with b.scf_if(b.cmp_eq(kvh, _c(b, 0))):
                self.stamp(5)

    # ------------------------------------------------ attention
    def _kv_loads(self, kvh: Value, stage: Value, first: Value, take: Value, above: Value):
        """K and V of token ``wave * 16 + lane % 16`` of ``stage``: 8 x 16-byte chunks each (dims 8 (lane / 16) + 32 c)."""
        b, q, sh = self.b, self.q, self.q.sh
        n = b.land(q.lane, _c(b, 15))
        g = b.lshr(q.lane, _c(b, 4))
        local = b.add(b.mul(stage, _c(b, q.ST)), b.add(b.mul(q.wave, _c(b, 16)), n))
        tab = q.lr(b.add(_c(b, q.L.POS), b.smin(local, _c(b, _pos_rounds(q.k) * q.T - 1))))
        direct = b.select(b.cmp_lt(local, take), b.add(first, local), _c(b, 0))
        p = b.select(above, tab, direct)
        base = b.mul(b.add(b.mul(b.add(b.mul(p, _c(b, sh.nkv)), kvh), _c(b, 2 * HD)), b.mul(g, _c(b, 8))), _c(b, 2))
        ks = [q.ld(self.kv, b.add(base, _c(b, 64 * c)), 4, q.cp_kv) for c in range(8)]
        vs = [q.ld(self.kv, b.add(base, _c(b, 512 + 64 * c)), 4, q.cp_kv) for c in range(8)]
        return ks + vs

    def attention(self, kvh: Value, first: Value, take: Value, above: Value) -> None:
        b, q, sh, k = self.b, self.q, self.q.sh, self.q.k
        W, ST, G, L = q.W, q.ST, q.G, q.L
        lane, wave = q.lane, q.wave
        n = b.land(lane, _c(b, 15))
        g = b.lshr(lane, _c(b, 4))
        grp = sh.group
        ok = b.cmp_lt(n, _c(b, grp))
        head = b.add(b.mul(kvh, _c(b, grp)), b.select(ok, n, _c(b, grp - 1)))
        V8 = VectorType(BF16, 8)
        V4 = VectorType(BF16, 4)
        F4 = VectorType(F32, 4)
        def q_of(c):
            if k.qloc:
                w = b.add(_c(b, L.QL), b.add(b.mul(head, _c(b, HD // 2)), b.add(_c(b, 16 * c), b.mul(g, _c(b, 4)))))
                return b.vec_bitcast(b.smem_load_vN(q.lds, w, dtype=I32, n=4), V8)
            off = b.mul(b.add(b.mul(head, _c(b, HD)), b.add(_c(b, 32 * c), b.mul(g, _c(b, 8)))), _c(b, 2))
            return b.vec_bitcast(q.ld(self.qb, off, 4, q.cp_ld), V8)

        qs = None if k.qlds else [q_of(c) for c in range(8)]
        if k.stamp and qs is not None:
            with b.scf_if(b.cmp_eq(kvh, _c(b, 0))):
                b.inline_asm("s_waitcnt vmcnt(0) lgkmcnt(0)", "v,v,~{memory}",
                             [b.vec_extract(b.vec_bitcast(qs[0], VectorType(I32, 4)), 0),
                              b.vec_extract(b.vec_bitcast(qs[7], VectorType(I32, 4)), 0)], None)
                self.stamp(15)
        per_w = HD // W
        nmt = per_w // 16
        dims = [b.add(b.mul(wave, _c(b, per_w)), b.add(_c(b, mt * 16), b.mul(g, _c(b, 4)))) for mt in range(nmt)]

        def gate_loads():
            return [b.buffer_load_cp(self.proj, b.mul(b.add(_c(b, HD), b.add(b.mul(head, _c(b, 2 * HD)), d)), _c(b, 2)),
                                     _c(b, 0), dwords=2, aux=0) for d in dims]

        graw = gate_loads() if k.gearly else None
        stages = b.div(b.add(take, _c(b, ST - 1)), _c(b, ST))
        pairs = b.div(b.add(stages, _c(b, 1)), _c(b, 2))
        scl = (LOG2E / 8.0) if k.mutant == "scale" else (LOG2E / 16.0)
        zero4 = b.zero_vec_f32(4)
        pf2 = k.pf == 2
        args = [("tp", _f(b, NEG)), ("dn", _f(b, 0.0))] + [(f"ac{m}", zero4) for m in range(16)]
        if pf2:
            cur = self._kv_loads(kvh, _c(b, 0), first, take, above)
            args += [(f"cu{i}", cur[i]) for i in range(16)]
        vbase = b.mul(wave, _c(b, 32 * 16 * 4))  # words: [wave][32 chunks][16 tokens][8 bf16 = 4 words]

        def stage(s_, top, den, acc, curk):
            if curk is None:
                curk = self._kv_loads(kvh, s_, first, take, above)
                pend = None
            else:
                pend = self._kv_loads(kvh, b.add(s_, _c(b, 1)), first, take, above)
            sc = zero4
            for c in range(8):
                sc = b.mfma_f32_16x16x32_bf16(b.vec_bitcast(curk[c], V8), q_of(c) if qs is None else qs[c], sc)
                valid = b.cmp_lt(b.add(b.mul(s_, _c(b, ST)), b.add(b.mul(wave, _c(b, 16)), n)), take)
                vv = b.select(valid, curk[8 + c], b.zero_vec(I32, 4))
                vw = b.add(_c(b, L.BIG), b.add(vbase, b.mul(b.add(b.mul(b.add(_c(b, 4 * c), g), _c(b, 16)), n), _c(b, 4))))
                b.smem_store_vN(q.lds, [vw], vv, 4)
            tok = b.add(b.mul(s_, _c(b, ST)), b.add(b.mul(wave, _c(b, 16)), b.mul(g, _c(b, 4))))
            scs = []
            for e in range(4):
                x = b.fmul(b.vec_extract(sc, e), _f(b, scl))
                scs.append(b.select(b.cmp_lt(b.add(tok, _c(b, e)), take), x, _f(b, NEG)))
            mx = b.fmax(b.fmax(scs[0], scs[1]), b.fmax(scs[2], scs[3]))
            mx = _xop(b, mx, 16, b.fmax)
            mx = _xop(b, mx, 32, b.fmax)
            new_top = b.fmax(top, mx)
            corr = b.exp2(b.fsub(top, new_top))
            probs = [b.select(b.cmp_lt(b.add(tok, _c(b, e)), take), b.exp2(b.fsub(scs[e], new_top)), _f(b, 0.0))
                     for e in range(4)]
            den = b.fadd(b.fmul(den, corr), b.fadd(b.fadd(probs[0], probs[1]), b.fadd(probs[2], probs[3])))
            p = b.vec_trunc_f32_to_bf16(b.vec_pack(probs, F32))
            nacc = []
            for mt in range(16):
                chunk = b.add(_c(b, 2 * mt), b.lshr(b.land(n, _c(b, 3)), _c(b, 1)))
                tok16 = b.add(b.mul(g, _c(b, 4)), b.lshr(n, _c(b, 2)))
                word = b.add(_c(b, L.BIG), b.add(vbase, b.add(b.mul(b.add(b.mul(chunk, _c(b, 16)), tok16), _c(b, 4)),
                                                               b.mul(b.land(n, _c(b, 1)), _c(b, 2)))))
                v = b.ds_read_tr16_b64(q.lds, word, dtype=BF16)
                a_ = b.vec_pack([b.fmul(b.vec_extract(acc[mt], e), corr) for e in range(4)], F32)
                nacc.append(b.mfma_f32_16x16x16_bf16(v, p, a_))
            return new_top, den, nacc, pend

        if pf2:
            lp = b.scf_for_iter(_c(b, 0), pairs, _c(b, 1), _ia(args), iv_name=_iv("qst"))
            with lp as (pi, it):
                top, den = it[0], it[1]
                acc = list(it[2:18])
                curk = list(it[18:34])
                s0 = b.mul(pi, _c(b, 2))
                top, den, acc, nxt = stage(s0, top, den, acc, curk)
                top, den, acc, nxt2 = stage(b.add(s0, _c(b, 1)), top, den, acc, nxt)
                b.scf_yield(top, den, *acc, *nxt2)
        else:
            lp = b.scf_for_iter(_c(b, 0), stages, _c(b, 1), _ia(args), iv_name=_iv("qs1"))
            with lp as (si, it):
                top, den, acc, _n = stage(si, it[0], it[1], list(it[2:18]), None)
                b.scf_yield(top, den, *acc)
        res = lp.results
        top, den = res[0], res[1]
        if k.stamp:
            with b.scf_if(b.cmp_eq(kvh, _c(b, 0))):
                self.stamp(13)
        acc = res[2:18]
        den = _xop(b, den, 16, b.fadd)
        den = _xop(b, den, 32, b.fadd)
        q.bar()
        with b.scf_if(b.cmp_eq(g, _c(b, 0))):
            q.lw(b.add(_c(b, L.TOPS), b.add(b.mul(wave, _c(b, 16)), n)), top)
            q.lw(b.add(_c(b, L.DENS), b.add(b.mul(wave, _c(b, 16)), n)), den)
        RS = HD + (4 if k.mpad else 0)
        for mt in range(16):
            at = b.add(_c(b, L.BIG), b.add(b.mul(b.add(b.mul(wave, _c(b, 16)), n), _c(b, RS)),
                                           b.add(_c(b, mt * 16), b.mul(g, _c(b, 4)))))
            b.smem_store_vN(q.lds, [at], b.vec_bitcast(acc[mt], VectorType(I32, 4)), 4)
        q.bar()
        tops = [q.lr(b.add(_c(b, L.TOPS + w * 16), n), F32) for w in range(W)]
        mx = tops[0]
        for w in range(1, W):
            mx = b.fmax(mx, tops[w])
        fac = [b.exp2(b.fsub(t, mx)) for t in tops]
        tot = _f(b, 0.0)
        for w in range(W):
            tot = b.fma(fac[w], q.lr(b.add(_c(b, L.DENS + w * 16), n), F32), tot)
        nums = []
        for d in dims:
            acc4 = [_f(b, 0.0)] * 4
            for w in range(W):
                v = b.smem_load_vN(q.lds, b.add(_c(b, L.BIG), b.add(b.mul(b.add(_c(b, w * 16), n), _c(b, RS)), d)),
                                   dtype=F32, n=4)
                acc4 = [b.fma(b.vec_extract(v, e), fac[w], acc4[e]) for e in range(4)]
            nums.append(acc4)
        gates = [_bf4(b, x) for x in (graw if graw is not None else gate_loads())]
        slot_head = b.mul(head, _c(b, G))
        self._store_partial(b.add(slot_head, self.member), ok, g, dims, mx, tot, nums)
        cbase = b.mul(kvh, _c(b, FAN + 1))
        grpi = b.div(self.member, _c(b, FAN))
        firstm = b.mul(grpi, _c(b, FAN))
        members = min(G, FAN)
        last = b.cmp_eq(q.arrive(self.cnt_base, b.mul(b.add(cbase, grpi), _c(b, 4))), _c(b, members - 1))
        if k.stamp:
            with b.scf_if(b.cmp_eq(kvh, _c(b, 0))):
                self.stamp(14)
        with b.scf_if(last):
            q.reset(self.cnt_base, b.mul(b.add(cbase, grpi), _c(b, 4)), members)
            mtop, mden, mnum = self._merge(slot_head, firstm, 1, members, dims)
            if G <= FAN:
                self._finish(head, ok, dims, gates, mnum, mden)
            else:
                self._store_partial(b.add(slot_head, firstm), ok, g, dims, mtop, mden, mnum)
                last2 = b.cmp_eq(q.arrive(self.cnt_base, b.mul(b.add(cbase, _c(b, FAN)), _c(b, 4))),
                                 _c(b, G // FAN - 1))
                with b.scf_if(last2):
                    q.reset(self.cnt_base, b.mul(b.add(cbase, _c(b, FAN)), _c(b, 4)), G // FAN)
                    _t, mden2, mnum2 = self._merge(slot_head, _c(b, 0), FAN, G // FAN, dims)
                    self._finish(head, ok, dims, gates, mnum2, mden2)

    def _store_partial(self, slot: Value, ok: Value, g: Value, dims, top, den, nums) -> None:
        b, q = self.b, self.q
        with b.scf_if(ok):
            for i, d in enumerate(dims):
                q.st(self.part, b.mul(b.add(b.mul(slot, _c(b, HD)), d), _c(b, 4)),
                     b.vec_pack([b.bitcast(x, I32) for x in nums[i]], I32))
            with b.scf_if(b.land(b.cmp_eq(q.wave, _c(b, 0)), b.cmp_eq(g, _c(b, 0)))):
                q.st(self.ml, b.mul(slot, _c(b, 8)), b.vec_pack([b.bitcast(top, I32), b.bitcast(den, I32)], I32))

    def _merge(self, slot_head: Value, first: Value, stride: int, count: int, dims):
        b, q = self.b, self.q
        kn = count
        if q.k.mutant == "merge":
            kn = count - 1
        loads = []
        for i in range(count):
            slot = b.add(slot_head, b.add(first, _c(b, i * stride)))
            ml = q.ld(self.ml, b.mul(slot, _c(b, 8)), 2, q.cp_ld)
            parts = [q.ld(self.part, b.mul(b.add(b.mul(slot, _c(b, HD)), d), _c(b, 4)), 4, q.cp_ld) for d in dims]
            loads.append((ml, parts))
        top = b.bitcast(b.vec_extract(loads[0][0], 0), F32)
        for i in range(1, kn):
            top = b.fmax(top, b.bitcast(b.vec_extract(loads[i][0], 0), F32))
        den = _f(b, 0.0)
        num = [[_f(b, 0.0)] * 4 for _ in dims]
        for i in range(kn):
            ml, parts = loads[i]
            w = b.exp2(b.fsub(b.bitcast(b.vec_extract(ml, 0), F32), top))
            den = b.fma(w, b.bitcast(b.vec_extract(ml, 1), F32), den)
            num = [[b.fma(w, b.bitcast(b.vec_extract(parts[m], e), F32), num[m][e]) for e in range(4)]
                   for m in range(len(dims))]
        return top, den, num

    def _finish(self, head: Value, ok: Value, dims, gates, num, den) -> None:
        b, q = self.b, self.q
        inv = b.rcp(den)
        with b.scf_if(ok):
            for i, d in enumerate(dims):
                out = [b.fmul(b.fmul(num[i][e], inv), _sigmoid(b, gates[i][e])) for e in range(4)]
                q.st(self.core, b.mul(b.add(b.mul(head, _c(b, HD)), d), _c(b, 2)), _pack_bf(b, out))


from rocke.core.ir import FP8E4M3 as _FP8  # noqa: E402


def build_qsa_kernel(shape: QsaShape = TP2, knobs: QsaKnobs = QsaKnobs()) -> KernelDef:  # noqa: B008
    return _Qsa(shape, resolve(knobs)).build()


# ------------------------------------------------------------------ the bytes floor


def floor_params() -> list[str]:
    return ["src", "sink", "nbytes", "reps"]


def build_floor_kernel(threads: int = 512, depth: int = 4) -> KernelDef:
    """A read streamer: 256 CTAs read ``nbytes`` (16 B per lane per access, 8 KiB contiguous per CTA access) ``reps``
    times, nt + sc1 loads (the KV loads' bits), xor-folded into one store per lane to ``sink`` (never read)."""
    b = IRBuilder(f"q38lb_qsafloor_t{threads}_d{depth}")
    b.kernel.attrs["max_workgroup_size"] = threads
    src = b.param("src", I64)
    sink = b.param("sink", I64)
    nbytes = b.readfirstlane(b.param("nbytes", I32))
    reps = b.readfirstlane(b.param("reps", I32))
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    lds = b.smem_alloc(I32, [ONE_CTA_PER_CU_WORDS], name_hint="flr_lds")
    with b.scf_if(D.is_thread0(b)):
        b.smem_store_vN(lds, [_c(b, ONE_CTA_PER_CU_WORDS - 1)], _c(b, 0), 1)
    rs = b.buffer_rsrc(_ptr(b, src), nbytes)
    step = threads * 16 * GRID
    n_it = b.div(b.add(nbytes, _c(b, step * depth - 1)), _c(b, step * depth))
    acc = b.zero_vec(I32, 4)
    outer = b.scf_for_iter(_c(b, 0), reps, _c(b, 1), _ia([("fr", acc)]), iv_name=_iv("fl"))
    with outer as (_r, (fr,)):
        inner = b.scf_for_iter(_c(b, 0), n_it, _c(b, 1), _ia([("fa", fr)]), iv_name=_iv("fi"))
        with inner as (i, (fa,)):
            base = b.add(b.mul(b.add(b.mul(i, _c(b, depth * GRID)), cta), _c(b, threads * 16)), b.mul(tid, _c(b, 16)))
            vs = [b.buffer_load_cp(rs, b.add(base, _c(b, d * GRID * threads * 16)), _c(b, 0), dwords=4,
                                   aux=CPOL_SC1 | CPOL_NT) for d in range(depth)]
            for v in vs:
                fa = b.vec_pack([b.xor(b.vec_extract(fa, e), b.vec_extract(v, e)) for e in range(4)], I32)
            b.scf_yield(fa)
        b.scf_yield(inner.results[0])
    res = outer.results[0]
    sp = b.int_to_ptr(sink, PtrType(I32, "global"))
    w = b.xor(b.xor(b.vec_extract(res, 0), b.vec_extract(res, 1)), b.xor(b.vec_extract(res, 2), b.vec_extract(res, 3)))
    b.store(w, sp, b.add(b.mul(cta, _c(b, threads)), tid))
    b.ret()
    return b.kernel

