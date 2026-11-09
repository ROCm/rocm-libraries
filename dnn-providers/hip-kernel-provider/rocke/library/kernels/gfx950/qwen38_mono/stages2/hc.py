# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The hyper-connection stages A1 / A2 / M1 / M2 of the Qwen3.8 mono layer as rocKE stage functions (true-mono WP2a).

Port of ATOM ``kernels/stages/hc.py`` (``a1``, ``a2``, the P3 spread ``_spread_a1`` / ``_spread_down`` /
``_spread_pack`` / ``_quarter_pack``) with the same contract: same mailboxes (plain or ATOM's ``numa=1`` area), same
LDS arena entries (``layout.for_rows``), same task numbering (the caller passes the CTA's task, which already folds
in ``spread`` / ``place`` / ``dyn`` levers), same arithmetic in the same order, no fast-math flags. So a build linked
into the FlyDSL layer kernel (``ATOM_Q38_EXTERN_STAGES=a1=<dir>/{kind},a2=...``) is bitwise the FlyDSL stage.

* ``a1`` / ``m1`` (HC norm + down + injection): down task ``g < 10`` of a request row: every task normalizes all 4
  streams (square sums of the 32-chunks, inverse RMS through f64, bf16 normalized values, MXFP8 pack), then wave
  ``w`` runs the 20-step scaled-MFMA chain of n-block ``2 g + w // 4`` over ``kit = w % 4 + 4 i``; wave 0 sums the 4
  partials in order, applies ``silu(y / 4)`` and publishes the 32-column MXFP8 pack (``lr8`` / ``lre``). Tasks 10-13
  compute the injection weights (bf16 dot of the normalized row with a weight row, 32 fma chains then an in-order
  lane sum, ``2 sigmoid(y / 4)``). The attention HC reads the layer input from memory (``h_in``), or in the PLE layer
  P2's output (polled, ``ph`` / ``phs``); the MLP HC polls C1's streams (``w`` / ``whs``).
* ``a2`` / ``m2`` (HC up + read + mixed pack): task ``t`` (80): wave ``w`` runs the 3-step up chain of n-block
  ``2 t + 160 (w // 2) + w % 2`` on the low-rank pack, the read epilogue ``bf16(sigma(up) * norm)``; wave ``r``
  publishes row ``r``'s mixed row (fp8 words, bf16 words, exponent).
* ``spread`` (P3; ``HcKnobs.spread`` = the build's ``ATOM_Q38_SCHED`` ``spread[.a1|.m1]`` value): 1 = 80 units (one
  wave's chain each, partials to the ``isc`` mailbox, A2 / M2 sum them), 2 = 20 one-n-block tasks (in-CTA sum, the
  activations to the mailbox), 3 / 4 = 2 / 1 with the rows deduplicated, 10 = 1 with the quarter pack; 5-9, 11 the
  P3 test mutants.

Knobs (:class:`HcKnobs`; defaults = the FlyDSL stage; every non-default value is a numerics-preserving lever unless
marked tolerance class) expose every hard-coded choice and the catalogued FlyDSL inefficiencies
(``experiments/qwen38-true-mono/reverse/FLYDSL-STAGES.md``): LDS operand read-ahead and the MFMA / ``ds_read``
interleave of the scaled-MFMA chains, the A-scale handling (hoisted scale registers, op_sel-packed scales), cache
policy of every weight stream, load issue order, poll implementation / pause / wait form, the entry preload of A1
(``pre="args"``: ATOM-truemono passes ``a1_preload``'s registers), and the tolerance-class N1 hooks
(``acc``: K-split accumulators). Python engine only (the device-function lowering, ``math.exp``, ``llvm.sqrt.f64``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace

from rocke.core.ir import F32, F64, I32, I64, IRBuilder, PtrType, Value
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as Q

from .. import abi as A
from .. import mailbox_ir  # noqa: F401  (registers the ir / irf mailbox implementations)
from ..device_function import DeviceFunctionKnobs, lower_device_function, write_device_function

__all__ = [
    "HC_STAGES",
    "HcKnobs",
    "HcSpec",
    "SPREAD_VALUES",
    "build_hc_kernel",
    "hc_lds",
    "hc_lds_wide",
    "knobs_from_spec",
    "knobs_spec",
    "lower_hc_function",
    "sidecar",
    "spec_lds",
    "supported",
    "write_hc_functions",
]

# ------------------------------------------------------------------ geometry (ATOM config / hc.py / layout.py)
HIDDEN, HC_COUNT, LOWRANK = A.HIDDEN, A.HC_COUNT, A.HC_LOWRANK
HC_WIDTH = HC_COUNT * HIDDEN
THREADS, WAVES, GRID = A.THREADS, A.WAVES, A.GRID
VECS = HC_WIDTH // 8  # 1280 vectors of 8 bf16 over the 4 streams
STREAM_VECS = HIDDEN // 8  # 320
ROUNDS = 3
STREAM_CHUNKS = HIDDEN // 32  # 80
INV_HIDDEN = 0x39CCCCCD  # fl(1/2560)
EPS = 0x358637BD  # fl(1e-6)
DOWN_KI = A.k_iters(HC_WIDTH)  # 80
DOWN_CHAIN = DOWN_KI // 4  # 20
DOWN_SBLOCKS = (DOWN_CHAIN + 15) // 16  # 2
DOWN_GROUPS = LOWRANK // 32  # 10
INJ_TASK = DOWN_GROUPS
INJ_ROW_VECS = HC_WIDTH // 8  # 1280
INJ_LANES = 32
INJ_SEG = HC_WIDTH // 2 // INJ_LANES + 4  # 164
INJ_VECS = INJ_ROW_VECS // INJ_LANES  # 40
INJ_CHUNK = 8
UP_KI = 3
UP_NB = 2 * HIDDEN // 32  # 160
HC_WORDS = A.HC_WORDS
TASKS_HC = DOWN_GROUPS + HC_COUNT  # 14 (a1 / m1, one request row)
TASKS_UP = HIDDEN // 32  # 80
C1_TILES = HIDDEN // 16  # 160 producers of the MLP HC's streams
P2_TASKS = HIDDEN // 32  # 80 producers of the PLE layer's attention HC streams
RECORDS = {"w_down": LOWRANK * HC_WIDTH // 2, "s_down": LOWRANK // 16 * 4 * DOWN_SBLOCKS * 64 * 16}
OOB_BYTES = 1 << 30  # ATOM ``hc.OOB`` (1 << 28 int32 elements): past every record, the load returns zeros
SCALE_ONE = 127
RATIO_BITS = 0x3B124925

# P3 spread (``layout.py``)
SPREAD_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
SPREAD_NB = 2 * DOWN_GROUPS  # 20
SPREAD_UNITS = 4 * SPREAD_NB  # 80
SPREAD_OFF = {"a1": 16384, "m1": 16384 + SPREAD_UNITS * 16}
QUARTER_VECS = DOWN_KI // 4 * 16  # 320

HC_STAGES = ("a1", "a2", "m1", "m2")
SYMBOL_PREFIX = "rocke_q38_"
# NUMA (ATOM ``kernels/numa.py``)
NUMA_PAGE = 1 << 21
NUMA_CHUNK = 8192

# sched_group_barrier masks
SGB_MFMA, SGB_VALU, SGB_DS_READ, SGB_DS_WRITE, SGB_VMEM_READ = 0x8, 0x2, 0x100, 0x200, 0x20


def spread_mode(v: int):
    """(units mode, dedupe) of spread value ``v`` (``layout.spread_mode``)."""
    assert v in SPREAD_VALUES, v
    return v in (1, 4, 7, 8, 10, 11), v in (3, 4, 8, 9)


def spread_down_tasks(v: int, rows: int) -> int:
    units, dedupe = spread_mode(v)
    per_row = SPREAD_UNITS if units else SPREAD_NB
    return per_row if dedupe or rows == 1 else per_row * rows


# ------------------------------------------------------------------ the LDS arena (``layout._arena`` port)
_RW_ROW = A.RW_ROW
_SS_ROW = 32 * (-(-(4 * (2048 // 4) + 3) // 32))  # 2080
_ATTN = (
    ("x8", 6144 // 4), ("xe", 6144 // 32), ("red", WAVES * 16), ("qk", 2 * 128), ("vz", 2 * 32), ("core", 32),
    ("sums", 4 * WAVES), ("cores", 128), ("baw", 2 * _RW_ROW), ("bax", 16 * A.RX_PART),
    ("hsum", HC_WIDTH // 32), ("inv", 4), ("n8", HC_WIDTH // 4), ("ne", HC_WIDTH // 32),
    ("nbf", INJ_LANES * INJ_SEG), ("injw", INJ_LANES * INJ_SEG), ("l8", 3 * 32), ("le", 3 * 4), ("prod", 4 * 32),
)  # fmt: skip
_MOE = (
    ("rw", 8 * _RW_ROW), ("rx", 16 * A.RX_PART), ("lgl", 8), ("top", 96), ("gx8", HIDDEN // 4), ("gxe", HIDDEN // 32),
    ("gred", WAVES * 16), ("dn", (A.TOP_K + 1) * A.DN_ROW), ("ys", (A.TOP_K + 1) * 16),
)  # fmt: skip
_QSA = (("iqn", 4 * 128), ("q4_keys", 5120))
_K_STRIDE = 256 // 2 + 4
_ATTN_QSA = (
    ("q5_q", (24 // 2 + 1) * 256 // 2), ("q5_k", 32 * _K_STRIDE), ("q5_sel", 32 // 4), ("q6_e", 9 * 256),
    ("q6_red", 4 * WAVES), ("q7_sel", 2048 // 4), ("q7_p", 4 * _SS_ROW), ("q7_cur", 16), ("q7_part", WAVES * 128),
)  # fmt: skip
_PLE = (("p_x8", HIDDEN // 4), ("p_xe", HIDDEN // 32), ("p_sig", HC_COUNT), ("p_inv", HC_COUNT))
_PER_ROW = {
    "x8", "xe", "red", "qk", "vz", "core", "sums", "cores", "bax", "hsum", "inv", "n8", "ne", "l8", "le", "prod",
    "rx", "lgl", "top", "gx8", "gxe", "gred", "dn", "ys", "p_x8", "p_xe",
}  # fmt: skip
_ATTN_D = dict(_ATTN)
_HC_TABLE = tuple((n, _ATTN_D[n]) for n in ("red", "hsum", "inv", "n8", "ne", "nbf", "injw", "l8", "le", "prod"))
_DELTA_TABLE = tuple((n, _ATTN_D[n]) for n in ("red", "x8", "xe", "qk", "vz", "core", "sums", "cores", "baw", "bax"))
SEQ_GROW = {"q5_q": 256 // 2, "q7_cur": 16}
LDS_LIMIT = 40960
HC_LDS = ("red", "hsum", "inv", "n8", "ne", "nbf", "injw", "l8", "le", "prod")


def _place(lds, stride, table, rows=1):
    w = 0
    for name, words in table:
        assert lds.get(name, w) == w, name
        lds[name] = w
        stride[name] = (words + 3) // 4 * 4
        if rows > 1 and name in _PER_ROW:
            stride[name] += (16 - stride[name]) % 64
        w += stride[name] * (rows if name in _PER_ROW else 1)
    return w


def _arena(rows=1, split=False, m4_slices=1, seq=False):
    lds, stride = {}, {}
    attn = (_HC_TABLE, _DELTA_TABLE) if split else (_ATTN,)
    moe = tuple((n, w * m4_slices if n == "gred" else w) for n, w in _MOE)
    attn_qsa = _ATTN_QSA
    if seq:
        attn_qsa = tuple((n, w + (rows - 1) * SEQ_GROW.get(n, 0)) for n, w in _ATTN_QSA)
    tables = (*attn, moe, _QSA, attn_qsa, _PLE)
    return lds, stride, max(_place(lds, stride, t, rows) for t in tables)


def hc_lds(rows: int, seq: bool = False) -> SimpleNamespace:
    """The HC stages' LDS entries of a ``rows``-row build (``layout.for_rows``): ``lds`` / ``stride`` (words) of
    :data:`HC_LDS`, the arena's ``words`` and ``split``."""
    if rows == 1:
        lds, stride, words = _arena()
        split = False
    else:
        m4_slices = 1 if (rows * A.TOP_K + 1) * (A.MOE_INTER // 64) <= GRID else 2
        lds, stride, words = _arena(rows, m4_slices=m4_slices, seq=seq)
        split = words > LDS_LIMIT
        if split:
            lds, stride, words = _arena(rows, split=True, m4_slices=m4_slices, seq=seq)
    assert words <= LDS_LIMIT, words
    return SimpleNamespace(
        lds={n: lds[n] for n in HC_LDS}, stride={n: stride[n] for n in HC_LDS}, words=words, split=split
    )


# ------------------------------------------------------------------ the wide builds' arena (``layout_wide.for_wide`` port)
WIDE_ROWS = 16
M4_SLICES_WIDE = 2
INV_ROWS = 12  # ``hc.INV_ROWS``: from 12 rows A1 / M1 hand A2 / M2 the inverse RMS (``inv`` / ``minv`` mailbox)
INV_TID = 256  # ``hc.INV_TID``: A2 / M2's first thread polling it
PRE_POLL_MAX = 64 * 12  # ``common_wide.PRE_POLL_MAX`` (ATOM ``mailbox.POLL_MAX`` = 12): hints per pre-poll


def _place_n(lds, stride, table, rows, n_of):
    """``layout_wide._place``: ``n_of.get(name, rows)`` rows of per-row entry ``name`` (its stride that of ``rows``)."""
    w = 0
    for name, words in table:
        assert lds.get(name, w) == w, name
        lds[name] = w
        stride[name] = (words + 3) // 4 * 4
        if rows > 1 and name in _PER_ROW:
            stride[name] += (16 - stride[name]) % 64
        w += stride[name] * (n_of.get(name, rows) if name in _PER_ROW else 1)
    return w


def hc_lds_wide(rows: int, group: int) -> SimpleNamespace:
    """The HC stages' LDS entries of a grouped build (``layout_wide.for_wide``): up to ``MAX_ROWS`` rows the seq
    build's layout; past it the wide build's phases (each phase's entries from word 0; the HC phase is
    ``_HC_TABLE``; past ``WAVES`` rows ``n8`` / ``ne`` hold one row), ``words`` the largest phase."""
    assert group >= 2 and rows % group == 0 and rows // group >= 2 and rows <= WIDE_ROWS, (rows, group)
    if rows <= A.MAX_ROWS:
        return hc_lds(rows, seq=True)
    m3_task_rows = rows // 2 if rows % 2 == 0 else rows
    n_of = {"n8": 1, "ne": 1} if rows > WAVES else {}
    n_of["rx"] = min(WAVES, m3_task_rows)
    tables = {
        "hc": _HC_TABLE,
        "d1": _DELTA_TABLE[:3],
        "d3": tuple((n, _RW_ROW if n == "bax" else w) for n, w in _DELTA_TABLE[3:]),
        "m3": _MOE[:3],
        "m45": tuple((n, w * M4_SLICES_WIDE if n == "gred" else w) for n, w in _MOE[3:]),
        "q34": _QSA,
        "q57": tuple((n, w + (group - 1) * SEQ_GROW.get(n, 0)) for n, w in _ATTN_QSA),
        "ple": _PLE,
    }
    if _place_n({}, {}, tables["m45"], rows, n_of) > LDS_LIMIT:
        m45 = tables.pop("m45")
        tables["m4"] = tuple(e for e in m45 if e[0] in ("top", "gx8", "gxe", "gred"))
        tables["m5"] = tuple(e for e in m45 if e[0] in ("top", "dn", "ys"))
    sizes, hc = {}, None
    for name, table in tables.items():
        lds, stride = {}, {}
        sizes[name] = _place_n(lds, stride, table, rows, n_of)
        assert sizes[name] <= LDS_LIMIT, (name, sizes[name])
        if name == "hc":
            hc = (lds, stride)
    lds, stride = hc
    return SimpleNamespace(lds={n: lds[n] for n in HC_LDS}, stride={n: stride[n] for n in HC_LDS},
                           words=max(sizes.values()), split=False)  # fmt: skip


def spec_lds(spec) -> SimpleNamespace:
    """The arena of ``spec``'s build: ``hc_lds_wide`` for a grouped build, else ``hc_lds``."""
    return hc_lds_wide(spec.rows, spec.group) if spec.group else hc_lds(spec.rows, spec.seq)


# ------------------------------------------------------------------ knobs
POLL_IMPLS = ("asm", "ir", "irf")
WAITS = ("barrier", "prepoll")
SCALES = ("ldsstep", "hoist", "opsel")
A2_ORDERS = ("mono", "h_first")
AMAXES = ("umax", "fmax")
INVS = ("lds", "regs")
INVMATHS = ("f64", "rsq32")  # the inverse RMS: the reference f64 sqrt + divide | fp32 v_rsq_f32 (TOLERANCE CLASS)
SIGMS = ("acc", "hw")  # sigmoid / SiLU: accurate exp + correctly rounded divide | v_exp_f32 + v_rcp_f32 (TOLERANCE CLASS)
L2E = 1.4426950408889634
SHFL4S = ("rowshift", "select")
STATS = ("streams", "halves")
PRES = ("none", "args")
SRCS = ("mem", "xl")
ORDERS = ("weights", "streams")
XHOLDS = ("none", "layer", "xpre")
M5_TASKS = HIDDEN // 16  # 160: the ``xl`` hand-off's producers (M5's tasks)
MUTANTS = (
    "eps",  # the inverse RMS epsilon times 2^20 (1e-6 -> ~1.05; one epsilon ulp is invisible under the mean square)
    "down_part",  # wave 0 sums partials 0..2 only (drops partial 3)
    "pack_scale",  # the low-rank pack's exponent + 1 (A1 / M1, wave 0)
    "inj_lane",  # the injection's lane sum skips lane 31
    "up_kit",  # A2 / M2: the up chain's k_iter 2 reads k_iter 1's operand
    "mix_avg",  # A2 / M2: the stream sum * 0.5 instead of * 0.25
    "quarter",  # spread 10: pack the wrong quarter (as ATOM's value 11)
)


@dataclass(frozen=True)
class HcKnobs:
    """Build knobs of the HC stage functions (defaults = the FlyDSL stage).

    mailbox: ``rocke.helpers.mailbox`` knobs of the puts and polls (``impl`` asm | ir | irf, ``pause`` s_nop |
      s_sleep1 | none). FlyDSL polls with compiler-visible ``sc1`` buffer loads and ``s_nop 0`` (``ir``).
    pause_sleep: N > 0 replaces the poll pause with ``s_sleep N`` (ATOM's ``pause=N`` lever's form).
    numa: 1 = ATOM's ``numa=1`` mailbox layout (both copies on put, the IOD-local copy on poll; sidecar
      ``"mailboxes": "numa1"``). numa_hdr: its header offset (0 = :data:`moe_fused.NUMA_HDR`).
    spread: the build's P3 spread value of this HC (0 = none). Must equal ATOM's ``spread[.a1|.m1]`` lever of the
      build (the sidecar names it; ATOM-truemono refuses a mismatch).
    pre: A1 of the attention HC in a memory-read build: ``none`` (load ``h`` / ``w'`` in the stage) | ``args``
      (ATOM's ``a1_preload`` registers, issued at the kernel entry, arrive as 24 extra i32 arguments: the FlyDSL
      stage's own form; needs ATOM-truemono's extern ``pre_args`` hook).
    wait: the wait ahead of a polled HC's stream poll (A1/M1) and C1-side polls (A2/M2): ``barrier`` (FlyDSL narrow:
      a CTA barrier, the payload poll spins) | ``prepoll`` (wave 0 spins on one hint pair per producer first).
    ra: LDS operand read-ahead of the scaled-MFMA chains (0 = FlyDSL: each step's ``ds_read``s right before its MFMA;
      1 / 2 = the reads of step ``i + ra`` issued before MFMA ``i``). Bitwise (same MFMA order).
    sgb: 1 = ``sched_group_barrier`` interleave of the chain (``ds_read`` group, then one MFMA per step).
    scale: A-operand scales of the down chain: ``ldsstep`` (FlyDSL: one ``ds_read`` per step right before its MFMA,
      the scale VGPR rewritten between MFMAs) | ``hoist`` (all 20 scale words read before the chain) | ``opsel``
      (hoisted and packed 4 per dword, picked with the MFMA's op_sel: no scale register written in the chain).
    w_cpol: cache-policy bits of the weight / scale streams (``buffer_load_cp`` aux; 0 = plain, 2 = nt, 16 = sc1,
      1 = sc0, sums). wi_cpol: of the injection weight rows; nw_cpol: of ``h`` / ``w'`` (norm loads).
    a2order: A2 of a memory-read HC: ``mono`` (up weights, then the stream words, then the norm rows) |
      ``h_first`` (the norm rows and stream words first: their wait no longer waits for the weights).
    stats: the chunk square sums of a POLLED HC's A1 / M1 (M1; A1 of the PLE layer): ``streams`` (FlyDSL: from the
      polled streams, VALU) | ``halves`` (from the producer's 16-column sums ``whs`` / ``phs``, as A2 / M2 take them:
      the same tree, so the same bits; no VALU tree on the critical path, one more poll).
    inv: the streams' inverse RMS: ``lds`` (FlyDSL: waves 0-3 one stream each, through LDS ``inv``, a barrier) |
      ``regs`` (one-row views: every wave computes all four itself; one barrier and an LDS round trip fewer).
    amax: the block amax of the MXFP8 packs: ``umax`` (unsigned max of the |x| bit patterns: the same bits as
      FlyDSL's ``llvm.maximum`` on these non-negative values, NaN propagated, one VALU per step) | ``fmax``
      (``llvm.maxnum`` with its canonicalizing compares).
    shfl4: the xor-4 lane exchange: ``rowshift`` (FlyDSL's two bank-masked row shifts) | ``select`` (two full row
      shifts and a lane select).
    anchor: 1 = LDS entries past the DS 16-bit byte offset are addressed from an opaque SGPR copy of their base (the
      per-access constants fold into the offset field, as FlyDSL's pointer arithmetic gets) | 0 = folded constants
      (one ``v_mov`` + ``v_mad`` per access of those entries).
    hbar: A1 of a memory-read HC: 1 = FlyDSL's CTA barrier between the ``h`` / ``w'`` loads and the down weights'
      (every wave's norm loads ahead of any weight stream) | 0 = none.
    acc: TOLERANCE CLASS (N1 hook, never a default): K-split accumulators per down chain (1 = FlyDSL; 2 = two
      chains summed at the end: changes the summation order).
    invmath: TOLERANCE CLASS (default-off, e2e fast-math (a)): the streams' inverse RMS ``f64`` (reference: f32 -> f64
      sqrt -> f64 divide -> f32, ~7 ``v_rsq_f64`` / ``v_rcp_f64`` / ``v_div_fixup_f64`` + refinement per build) |
      ``rsq32`` (one ``v_rsq_f32`` on the f32 ``ms + eps``, ~1 ulp: changes the last bit of the inverse).
    sigm: TOLERANCE CLASS (default-off, e2e fast-math (b)): sigmoid / SiLU of the HC (A2 / M2 injection gate, the
      down's activation): ``acc`` (accurate ``llvm.exp.f32`` + correctly rounded divide) | ``hw`` (``exp2_fast`` of
      ``-x log2 e`` = one ``v_exp_f32``, then ``v_rcp_f32``; no range guard: the argument's overflow gives ``inf``,
      whose reciprocal is the correct 0, an underflow gives 0 -> 1).
    src: the attention HC's streams (A1 / A2 of the delta and qsa layers): ``mem`` (FlyDSL: the layer input from
      memory) | ``xl`` (the persistent kernel's cross-layer hand-off, probe P1's ``xl``: the previous layer's M5
      publishes the output words as tagged pairs and their 16-column square sums into this layer's ``ph`` / ``phs``,
      160 producers, hint stride 1; the HC runs its polled path on them). DESIGN WP2a ``xpre`` / D0.8.
    order: a polled A1 / M1's issue order: ``weights`` (FlyDSL: ``w'`` and the down weights, then the stream poll) |
      ``streams`` (P1's ``a1hf``: poll the streams and load ``w'`` first, CTA barrier, then the down weights; for a
      source that is (nearly) there when the stage starts, as ``xl`` behind a hold). Bitwise.
    xhold: the layer-entry hold of an ``xl`` A1 (it runs first in the layer, on every CTA): ``none`` | ``layer``
      (every CTA's wave 0 spins on every M5 producer's last-row sum pair, then a CTA barrier: no next-layer weight
      issue before the hand-off) | ``xpre`` (the same for CTAs without an A1 task; A1's CTAs issue their next-layer
      weights right after their own M5 work and poll the streams themselves: D0.8's default). In a framework kernel
      the default comes from ``PersistKnobs.xhold``. Bitwise (a hold only waits).
    mutate: test-only mutants (:data:`MUTANTS`) the bitwise gate must catch.
    fn: device-function emission (``lds="arg"``, ``inline="always"``).
    """

    mailbox: M.MailboxKnobs = field(default_factory=lambda: M.MailboxKnobs(impl="ir"))
    pause_sleep: int = 0
    numa: int = 0
    numa_hdr: int = 0
    spread: int = 0
    pre: str = "none"
    wait: str = "barrier"
    ra: int = 0
    sgb: int = 0
    scale: str = "ldsstep"
    w_cpol: int = 0
    wi_cpol: int = 0
    nw_cpol: int = 0
    a2order: str = "mono"
    stats: str = "streams"
    hbar: int = 1
    anchor: int = 1
    amax: str = "umax"
    shfl4: str = "rowshift"
    inv: str = "lds"
    acc: int = 1
    invmath: str = "f64"
    sigm: str = "acc"
    src: str = "mem"
    order: str = "weights"
    xhold: str = "none"
    mutate: str = ""
    fn: DeviceFunctionKnobs = field(default_factory=lambda: DeviceFunctionKnobs(lds="arg", inline="always"))

    def __post_init__(self) -> None:
        if self.mailbox.impl not in POLL_IMPLS:
            raise ValueError(f"mailbox impl must be one of {POLL_IMPLS}")
        if self.mailbox.impl in ("ir", "irf"):
            from .. import mailbox_ir  # noqa: F401  (registers ir / irf)
        if self.numa not in (0, 1):
            raise ValueError("numa must be 0 or 1")
        if self.numa_hdr < 0 or self.numa_hdr % 256 or (self.numa_hdr and not self.numa):
            raise ValueError("numa_hdr must be a non-negative multiple of 256, and needs numa=1")
        if self.spread and self.spread not in SPREAD_VALUES:
            raise ValueError(f"spread must be 0 or one of {SPREAD_VALUES}")
        for name, allowed in (("pre", PRES), ("wait", WAITS), ("scale", SCALES), ("a2order", A2_ORDERS),
                              ("stats", STATS), ("amax", AMAXES), ("shfl4", SHFL4S), ("inv", INVS),
                              ("invmath", INVMATHS), ("sigm", SIGMS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.ra not in (0, 1, 2):
            raise ValueError("ra must be 0, 1 or 2")
        if self.sgb not in (0, 1):
            raise ValueError("sgb must be 0 or 1")
        if self.hbar not in (0, 1):
            raise ValueError("hbar must be 0 or 1")
        if self.anchor not in (0, 1):
            raise ValueError("anchor must be 0 or 1")
        if self.acc not in (1, 2):
            raise ValueError("acc must be 1 or 2")
        for name, allowed in (("src", SRCS), ("order", ORDERS), ("xhold", XHOLDS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if not 0 <= self.pause_sleep <= 127:
            raise ValueError("pause_sleep must be in 0..127")
        for name in ("w_cpol", "wi_cpol", "nw_cpol"):
            v = getattr(self, name)
            if v & ~0x13:
                raise ValueError(f"{name}: cache bits sc0 1 | nt 2 | sc1 16 only")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate must be one of {MUTANTS}")

    @property
    def header_offset(self) -> int:
        if not self.numa:
            raise ValueError("header_offset: numa=0 builds do not use the NUMA area")
        if self.numa_hdr:
            return self.numa_hdr
        from ..moe_fused import NUMA_HDR

        return NUMA_HDR

    @property
    def tolerance(self) -> bool:
        """True when a knob changes the numerics (tolerance class: never bitwise)."""
        return self.acc != 1 or self.invmath != "f64" or self.sigm != "acc"


def knobs_spec(k: HcKnobs) -> str:
    """``key=value,...`` of the non-default knobs (``mb_<field>`` for mailbox ones); inverse :func:`knobs_from_spec`."""
    d, md = HcKnobs(), M.MailboxKnobs(impl="ir")
    parts = []
    for f in k.__dataclass_fields__:
        if f in ("mailbox", "fn"):
            continue
        if getattr(k, f) != getattr(d, f):
            parts.append(f"{f}={getattr(k, f)}")
    for f in k.mailbox.__dataclass_fields__:
        if getattr(k.mailbox, f) != getattr(md, f):
            parts.append(f"mb_{f}={getattr(k.mailbox, f)}")
    for f in ("lds", "inline", "convergent"):
        if getattr(k.fn, f) != getattr(d.fn, f):
            parts.append(f"fn_{f}={getattr(k.fn, f)}")
    return ",".join(parts)


def knobs_from_spec(spec: str) -> HcKnobs:
    kn, mb, fn = {}, {}, {}
    ints = {f for f, v in asdict(HcKnobs()).items() if isinstance(v, int) and not isinstance(v, bool)}
    for kv in filter(None, (s.strip() for s in (spec or "").split(","))):
        key, _, v = kv.partition("=")
        if key.startswith("mb_"):
            mb[key[3:]] = int(v) if key[3:] in ("poll_max", "numa_copies") else (v == "True" if v in ("True", "False") else v)
        elif key.startswith("fn_"):
            fn[key[3:]] = v == "True" if v in ("True", "False") else v
        else:
            kn[key] = int(v) if key in ints else v
    base = HcKnobs()
    if mb:
        kn["mailbox"] = replace(base.mailbox, **mb)
    if fn:
        kn["fn"] = replace(base.fn, **fn)
    return HcKnobs(**kn)


@dataclass(frozen=True)
class HcSpec:
    """One stage function: ``stage`` (a1 | a2 | m1 | m2), the layer ``kind`` (delta | delta_ple | qsa: the PLE
    layer's attention HC is polled from P2), ``rows`` / ``seq`` / ``group`` (the build variant), knobs. A grouped
    build of up to ``MAX_ROWS`` rows has the seq build's layout (``layout_wide.for_wide``) and the narrow code; past
    ``MAX_ROWS`` rows (up to :data:`WIDE_ROWS`, grouped only) the wide build's layout and A2 (``hc_wide.a2``)."""

    stage: str
    kind: str = "delta"
    rows: int = 1
    seq: bool = False
    knobs: HcKnobs = field(default_factory=HcKnobs)
    group: int = 0  # grouped verify builds (``r<rows>g<S>``, fix round): rows / S requests of S rows

    @property
    def variant(self) -> str:
        return f"r{self.rows}" + ("s" if self.seq else "") + (f"g{self.group}" if self.group else "")

    @property
    def wide(self) -> bool:
        """More than ``MAX_ROWS`` rows: ATOM's wide build (``layout_wide.for_wide``, ``hc_wide.a2``)."""
        return self.rows > A.MAX_ROWS

    @property
    def symbol(self) -> str:
        return f"{SYMBOL_PREFIX}{self.stage}_{self.variant}"

    @property
    def mlp(self) -> bool:
        return self.stage in ("m1", "m2")

    @property
    def polled(self) -> bool:
        return self.mlp or self.kind == "delta_ple"

    @property
    def down(self) -> str:
        return "m1" if self.mlp else "a1"


def supported(spec: HcSpec) -> str | None:
    """None when ``spec`` is buildable, else the reason."""
    if spec.stage not in HC_STAGES:
        return f"stage {spec.stage!r} is not an HC stage"
    if spec.kind not in ("delta", "delta_ple", "qsa"):
        return f"kind {spec.kind!r}"
    k = spec.knobs
    if spec.group:
        g = spec.group
        if spec.seq:
            return "a grouped build is not a seq build (its layout is the seq one)"
        if g < 2 or spec.rows % g or spec.rows // g < 2 or spec.rows > WIDE_ROWS:
            return f"grouped builds: rows a multiple of the group, two groups at least, {WIDE_ROWS} rows at most"
        if k.spread:
            return "spread: ATOM builds the spread for narrow ungrouped layers only"
    elif not 1 <= spec.rows <= A.MAX_ROWS:
        return "rows 1..4 (ungrouped builds; wider builds are grouped)"
    if spec.seq and spec.rows < 2:
        return "a seq build has 2 rows at least"
    if k.src == "xl" and (spec.mlp or spec.kind == "delta_ple"):
        return "src=xl: the attention HC of the delta / qsa layers (the PLE layer's polls P2; the MLP HC C1)"
    if k.src == "xl" and spec.group:
        return "src=xl: narrow ungrouped builds (P1's M5 hand-off)"
    if k.src == "xl" and k.pre == "args":
        return "src=xl: the streams are polled, there is no a1_preload"
    if k.order == "streams" and not (spec.stage in ("a1", "m1") and (spec.mlp or spec.kind == "delta_ple" or k.src == "xl")):
        return "order=streams: a polled A1 / M1"
    if k.xhold != "none" and not (spec.stage == "a1" and k.src == "xl"):
        return "xhold: the layer-entry hold of an src=xl A1"
    if spec.wide and k.numa:
        return "numa=1: ATOM drops the numa lever above 4 rows"
    if spec.wide and k.pre == "args":
        return "pre=args: a wide build has no a1_preload (ATOM: early only up to 4 rows)"
    if k.pre == "args" and (spec.stage != "a1" or spec.kind == "delta_ple"):
        return "pre=args: A1 of a memory-read attention HC only"
    if k.spread:
        n = spread_down_tasks(k.spread, spec.rows) + spec.rows * HC_COUNT
        if n > GRID:
            return f"spread {k.spread} at {spec.rows} rows: {n} tasks > {GRID}"
    if k.scale == "opsel" and k.acc != 1:
        return "scale=opsel with acc=2 is not built"
    return None


# ------------------------------------------------------------------ the emitter
class _Hc:
    """Per-build emitter state: builder, ABI values, layout, knobs, LDS pool, mailbox access."""

    def __init__(self, b: IRBuilder, a, spec: HcSpec, lds: SimpleNamespace):
        self.b, self.a, self.spec, self.k = b, a, spec, spec.knobs
        self.rows = spec.rows
        # wide builds (fix round): kept by the one-row views (``c.L.rows`` / ``hc.wide_inv`` in ATOM)
        self.wide = spec.rows > A.MAX_ROWS
        self.wide_inv = spec.rows >= INV_ROWS
        self.L = lds
        self.tid, self.lane, self.wave = a.tid, a.lane, a.wave
        self.tag = a.tag
        self.pool = b.smem_alloc(I32, [lds.words], name_hint="hc_lds")
        mk = self.k.mailbox
        self.put_knobs = replace(mk, numa_copies=2) if self.k.numa else mk
        self.anch = {}
        if self.k.anchor:
            from rocke.helpers import device_ops as dops

            used = ("red", "hsum", "inv", "n8", "ne", "nbf", "injw") if spec.stage in ("a1", "m1") else (
                "hsum", "inv", "l8", "le", "prod")
            for n in used:
                if 4 * (lds.lds[n] + lds.stride[n] * self.rows) > 0xFFFF:
                    self.anch[n] = dops.fresh_s(b, b.const_i32(lds.lds[n]))

    # ---------------------------------------------------- scalars
    def c(self, v):
        return v if isinstance(v, Value) else self.b.const_i32(int(v))

    def f(self, v: float):
        return self.b.const_f32(float(v))

    def fbits(self, bits: int):
        return self.b.bitcast(self.c(bits), F32)

    def add(self, *xs):
        acc = self.c(xs[0])
        for x in xs[1:]:
            if isinstance(x, int) and x == 0:
                continue
            acc = self.b.add(acc, self.c(x))
        return acc

    def mul(self, a, b_):
        if isinstance(a, int) and isinstance(b_, int):
            return self.c(a * b_)
        return self.b.mul(self.c(a), self.c(b_))

    def lt(self, a, b_):
        return self.b.cmp_lt(self.c(a), self.c(b_))

    def ge(self, a, b_):
        return self.b.cmp_ge(self.c(a), self.c(b_))

    def eq(self, a, b_):
        return self.b.cmp_eq(self.c(a), self.c(b_))

    def imin(self, a, b_):
        return self.b.smin(self.c(a), self.c(b_))

    def imax(self, a, b_):
        return self.b.smax(self.c(a), self.c(b_))

    def sel(self, cond, x, y):
        return self.b.select(cond, self.c(x) if not isinstance(x, Value) else x, self.c(y) if not isinstance(y, Value) else y)

    def udiv(self, a, d: int):
        if isinstance(a, int):
            return self.c(a // d)
        if d & (d - 1) == 0:
            return self.b.lshr(a, self.c(d.bit_length() - 1))
        return self.b.div(a, self.c(d))

    def umod(self, a, d: int):
        if isinstance(a, int):
            return self.c(a % d)
        if d & (d - 1) == 0:
            return self.b.land(a, self.c(d - 1))
        return self.b.mod(a, self.c(d))

    def row_ix(self, i, r, stride):
        """Index ``i`` in request row ``r`` (int or Value) of rows ``stride`` apart (``common.row_ix``)."""
        if isinstance(r, int) and r == 0:
            return self.c(i)
        return self.add(i, self.mul(r, stride))

    def mb_ix(self, i, r):
        return self.row_ix(i, r, A.ROW_PAIRS)

    # ---------------------------------------------------- math (ATOM device / stage helpers)
    def bf16r(self, x):
        b = self.b
        return b.cast_to_f32(b.trunc_f32_to_bf16(x))

    def bf16x8(self, words):
        b = self.b
        out = []
        for w in words:
            out += [b.bitcast(b.shl(w, self.c(16)), F32), b.bitcast(b.land(w, self.c(-65536)), F32)]
        return out

    def sigmoid_acc(self, x):
        b = self.b
        if self.k.sigm == "hw":
            return b.rcp_fast(b.fadd(self.f(1.0), b.exp2_fast(b.fmul(x, self.f(-L2E)))))
        return b.fdiv(self.f(1.0), b.fadd(self.f(1.0), b.exp(b.fsub(self.f(0.0), x))))

    def silu_q(self, low):
        """``s = bf16(bf16(low) * 0.25)``, ``bf16(s / (1 + exp(0 - s)))`` (the down's activation)."""
        b = self.b
        s = self.bf16r(b.fmul(self.bf16r(low), self.f(0.25)))
        if self.k.sigm == "hw":
            return self.bf16r(b.fmul(s, self.sigmoid_acc(s)))
        return self.bf16r(b.fdiv(s, b.fadd(self.f(1.0), b.exp(b.fsub(self.f(0.0), s)))))

    def shfl(self, v, off):
        """``v`` (f32 / i32) of lane ``lane ^ off`` for off < 16, VALU only (ATOM ``xshfl``)."""
        b = self.b
        is_f = v.type == F32
        x = b.bitcast(v, I32) if is_f else v
        if off in (1, 2, 8):
            r = b.dpp_xor(x, off)
        elif off == 4 and self.k.shfl4 == "rowshift":
            # FlyDSL's form: row_shr:4 into banks 1 / 3, row_shl:4 into banks 0 / 2 (the s_nop covers a VALU write
            # of the source right before: LLVM does not see into the asm)
            r = b.inline_asm(
                "s_nop 1\n\tv_mov_b32_dpp $0, $1 row_shr:4 row_mask:0xf bank_mask:0xa"
                "\n\tv_mov_b32_dpp $0, $1 row_shl:4 row_mask:0xf bank_mask:0x5",
                "=&v,v", [x], I32, sideeffect=False, result_name_hint="xs4",
            )
        elif off == 4:
            up = b.mov_dpp(x, row_shl=4)
            dn = b.mov_dpp(x, row_shr=4)
            r = b.select(self.eq(b.land(self.lane, self.c(4)), 0), up, dn)
        else:
            raise ValueError(off)
        return b.bitcast(r, F32) if is_f else r

    def xred(self, v, off, op):
        """ATOM ``xred``: ``op(v, v[lane ^ off])``; 16 / 32 through the permlane swaps (one of the swapped pair is
        the lane's own value, the other its partner: ``op`` is symmetric)."""
        b = self.b
        if off < 16:
            return op(v, self.shfl(v, off))
        is_f = v.type == F32
        x = b.bitcast(v, I32) if is_f else v
        lo, hi = (b.permlane32_swap if off == 32 else b.permlane16_swap)(x, x)
        if is_f:
            return op(b.bitcast(lo, F32), b.bitcast(hi, F32))
        return op(lo, hi)

    def fadd2(self, x, y):
        return self.b.fadd(x, y)

    def fmax2(self, x, y):
        return self.b.fmax(x, y)

    def absmax_start(self, x):
        """|x| in the amax's domain: f32 (``amax=fmax``) or its bits (``umax``: for values >= +0 the unsigned order of
        the bit patterns is the float order, NaN (positive after fabs) above +inf: FlyDSL's ``llvm.maximum``
        result, bit for bit, without canonicalizing maxnum)."""
        a = self.b.fabs(x)
        return self.b.bitcast(a, I32) if self.k.amax == "umax" else a

    def absmax2(self, x, y):
        # fabs cleared the sign bit: signed max of the bit patterns = unsigned max
        return self.b.smax(x, y) if self.k.amax == "umax" else self.b.fmax(x, y)

    def absmax_end(self, m):
        return self.b.bitcast(m, F32) if self.k.amax == "umax" else m

    def natural_sum(self, v):
        for off in (1, 2, 4, 8, 16, 32):
            v = self.xred(v, off, self.fadd2)
        return v

    def lane_gather(self, v, first, n):
        """Lane ``l`` gets ``v`` of lanes ``first(l) .. first(l) + n - 1`` (``ds_bpermute``)."""
        b = self.b
        bits = b.bitcast(v, I32)
        return [b.bitcast(b.ds_bpermute(self.mul(self.add(first, e), 4), bits), F32) for e in range(n)]

    def e8m0(self, amax):
        ex, scale = Q.e8m0_fp8(self.b, amax)
        return ex, scale

    def fp8_word(self, v0, v1, v2, v3, scale):
        b = self.b
        w = b.cvt_scalef32_pk_fp8_f32(self.c(0), v0, v1, scale, hi=False)
        return b.cvt_scalef32_pk_fp8_f32(w, v2, v3, scale, hi=True)

    def bf16_pair(self, lo, hi):
        return M.bf16_pair(self.b, lo, hi)

    # ---------------------------------------------------- LDS (one pool = the mono's arena, word indices)
    def lbase(self, name):
        """Word index of LDS entry ``name``: a constant, or with ``anchor`` an opaque SGPR copy for entries past
        the DS instructions' 16-bit byte offset (the per-access constants then fold into the offset field)."""
        a = self.anch.get(name)
        return a if a is not None else self.c(self.L.lds[name])

    def lix(self, name, i=0, row=0):
        a = self.anch.get(name)
        if a is not None:
            if isinstance(row, int):
                return self.add(a, self.add(i, row * self.L.stride[name]) if isinstance(i, Value) else self.c(i + row * self.L.stride[name]))
            return self.add(a, self.add(self.mul(row, self.L.stride[name]), i))
        base = self.L.lds[name]
        if isinstance(row, int):
            base += row * self.L.stride[name]
            return self.add(base, i) if isinstance(i, Value) else self.c(base + i)
        return self.add(self.mul(row, self.L.stride[name]), base, i)

    def ld(self, idx, dtype=I32):
        v = self.b.smem_load_vN(self.pool, self.c(idx), dtype=dtype, n=1)
        return self.b.vec_extract(v, 0)

    def ldn(self, idx, n, dtype=I32):
        return self.b.smem_load_vN(self.pool, self.c(idx), dtype=dtype, n=n)

    def st(self, idx, v, n=1):
        self.b.smem_store_vN(self.pool, [self.c(idx)], v, n)

    def barrier(self):
        self.b.s_barrier_bare(fence="workgroup")

    # ---------------------------------------------------- weights
    def rsrc(self, arg: str, nbytes: int = 0x7FFFFFF0):
        b = self.b
        ptr = b.int_to_ptr(b.readfirstlane(getattr(self.a, arg)), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(nbytes))

    def load(self, rs, byte_off, dwords, cpol=0):
        b = self.b
        if cpol:
            return b.buffer_load_cp(rs, self.c(byte_off), self.c(0), dwords=dwords, aux=cpol)
        if dwords == 1:
            return b.buffer_load(rs, self.c(byte_off), self.c(0), I32)
        return b.buffer_load_vN(rs, self.c(byte_off), self.c(0), I32, dwords)

    def vec(self, v, n=4):
        return [self.b.vec_extract(v, e) for e in range(n)]

    # ---------------------------------------------------- mailboxes
    def _area(self):
        """ATOM's NUMA area as one mailbox and this CTA's copy selector (the header's page mask with the IOD folded
        in). Emitted at every use: a value cached across ``scf_if`` regions would not dominate its later uses (LLVM
        merges the copies)."""
        b = self.b
        hdr = self.k.header_offset
        a_ = b.add(self.a.scratch, b.const_i64(hdr + 256 + NUMA_PAGE - 1))
        area = b.land(a_, b.const_i64(-NUMA_PAGE))
        mb = M.mailbox_from_i64(b, area, self.c(0x7FFFFFF0), self.k.mailbox)
        hp = b.int_to_ptr(b.add(self.a.scratch, b.const_i64(hdr)), PtrType(I32, "global"))
        mask = b.readfirstlane(b.global_load(hp, self.c(0), I32, align=4))
        sel = b.readfirstlane(b.xor(mask, b.sub(self.c(0), M.iod_id(b))))
        return mb, sel

    def mb(self, region):
        if self.k.numa:
            return self._area()[0]
        b = self.b
        addr = b.add(self.a.scratch, b.const_i64(A.SC[region]))
        return M.mailbox_from_i64(b, addr, self.c(0x7FFFFFF0), self.k.mailbox)

    def _xi(self, region, i):
        b = self.b
        x = b.add(self.c(A.SC[region]), b.mul(self.c(i), self.c(M.PAIR_BYTES)))
        return b.add(x, b.land(x, self.c(-NUMA_CHUNK)))

    def put_ix(self, region, i):
        if not self.k.numa:
            return self.c(i)
        return self.b.lshr(self._xi(region, i), self.c(3))

    def poll_ix(self, region, i):
        if not self.k.numa:
            return self.c(i)
        b = self.b
        _, sel = self._area()
        xi = self._xi(region, i)
        s = b.land(b.lshr(sel, b.lshr(xi, self.c(21))), self.c(1))
        return b.lshr(b.add(xi, b.shl(s, self.c(13))), self.c(3))

    def put(self, region, i, v):
        bits = v if v.type == I32 else self.b.bitcast(v, I32)
        if self.k.mutate == "pack_scale" and region in ("lre", "mlre"):
            bits = self.b.add(bits, self.c(1))
        M.put(self.b, self.mb(region), self.put_ix(region, i), bits, self.tag, self.put_knobs)

    def put2(self, region, i, v0, v1):
        M.put2(self.b, self.mb(region), self.put_ix(region, i), v0, v1, self.tag, self.put_knobs)

    def _pause_text(self):
        if self.k.pause_sleep:
            return f"s_sleep {self.k.pause_sleep}"
        from rocke.helpers import device_ops as dops

        return dops.PAUSES[self.k.mailbox.pause]

    def poll(self, specs):
        """``specs`` = [(region, pair index, npairs)] -> one list of payload words per spec (batches of
        ``POLL_MAX``)."""
        b, mk = self.b, self.k.mailbox
        out = []
        pause = self._pause_text()
        for first in range(0, len(specs), M.POLL_MAX):
            batch = specs[first : first + M.POLL_MAX]
            mspecs = [(self.mb(r), self.poll_ix(r, i), n) for r, i, n in batch]
            if mk.impl in ("ir", "irf"):
                tspecs = []
                for mbx_, off, n in mspecs:
                    tspecs.append((mbx_.rsrc, M.pair_offset(b, off), 2 * n))
                vecs = b.tagged_poll(self.tag, tspecs, aux=mk.cpol, pause=pause,
                                     reload="launder" if mk.impl == "irf" else "")  # fmt: skip
                out += [[b.vec_extract(v, 2 * q) for q in range(n)] for v, (_, _, n) in zip(vecs, mspecs)]
            else:
                out += _poll_batch_asm(b, mspecs, self.tag, mk, pause)
        return out

    def wait_producers(self, hint, n, after_barrier=False):
        """``payload_wait``: a CTA barrier (narrow builds), or with ``wait=prepoll`` wave 0's hint poll first.
        ``after_barrier``: a CTA barrier ends the code right before (no LDS access between): the plain wait's barrier
        is redundant there (FlyDSL's two adjacent ``gpu.barrier``s compile to one; rocKE's fenced barriers do not
        merge), so it is left out."""
        b = self.b
        if self.wide:
            # ``payload_wait`` / ``pre_poll_wide`` past MAX_ROWS rows: wave 0 spins on the hints in batches of
            # PRE_POLL_MAX, a CTA barrier after each (the ``wait`` knob is inert there)
            for first in range(0, n, PRE_POLL_MAX):
                nb = min(PRE_POLL_MAX, n - first)
                with b.scf_if(self.eq(self.wave, 0)):
                    specs = []
                    for r in range((nb + 63) // 64):
                        m = self.imin(self.add(self.lane, 64 * r), nb - 1)
                        reg, i = hint(self.add(m, first) if first else m)
                        specs.append((reg, i, 1))
                    self.poll(specs)
                self.barrier()
            return
        if self.k.wait != "prepoll" and after_barrier:
            return
        if self.k.wait == "prepoll":
            with b.scf_if(self.eq(self.wave, 0)):
                specs = []
                for r in range((n + 63) // 64):
                    m = self.imin(self.add(self.lane, 64 * r), n - 1)
                    reg, i = hint(m)
                    specs.append((reg, i, 1))
                self.poll(specs)
        self.barrier()


def _poll_batch_asm(b, mspecs, tag, knobs, pause):
    """``rocke.helpers.mailbox``'s asm poll with an arbitrary pause statement."""
    from rocke.core.ir import VectorType

    bases, base_of = [], []
    for mb, _, _ in mspecs:
        for j, x in enumerate(bases):
            if x is mb.addr:
                base_of.append(j)
                break
        else:
            base_of.append(len(bases))
            bases.append(mb.addr)
    npairs = [n for _, _, n in mspecs]
    template, cons = M.asm_poll_template(npairs, len(bases), base_of, knobs.cpol_asm, pause)
    voffs = [M.pair_offset(b, i) if mb.copy_off is None else b.add(M.pair_offset(b, i), mb.copy_off)
             for mb, i, _ in mspecs]  # fmt: skip
    outs = b.inline_asm_multi(template, cons, voffs + bases + [b.zext(tag, I64)],
                              result_types=[VectorType(I32, 2)] * sum(npairs) + [I64, I64],
                              result_name_hint="apoll")  # fmt: skip
    res, p = [], 0
    for n in npairs:
        res.append([b.vec_extract(outs[p + q], 0) for q in range(n)])
        p += n
    return res


# ------------------------------------------------------------------ HC descriptor (``attn_hc`` / ``mlp_hc``)
def _hc(spec: HcSpec) -> SimpleNamespace:
    if spec.mlp:
        return SimpleNamespace(down="m1", up="m2", polled=True, h="w", hs="whs", producers=C1_TILES, hint_stride=1,
                               norm_w="m_norm_w", w_down="m_w_down", s_down="m_s_down", w_up="m_w_up", s_up="m_s_up",
                               w_inj="m_w_inj", lr8="mlr8", lre="mlre", inj="minj", x8="mx8", xe="mxe", xbf="mxbf",
                               inv="minv")  # fmt: skip
    if spec.kind == "delta_ple":
        src = dict(polled=True, h="ph", hs="phs", producers=P2_TASKS, hint_stride=2)
    elif spec.knobs.src == "xl":
        src = dict(polled=True, h="ph", hs="phs", producers=M5_TASKS, hint_stride=1)
    else:
        src = dict(polled=False, h="h_in", hs=None, producers=0, hint_stride=0)
    return SimpleNamespace(down="a1", up="a2", **src, norm_w="hc_norm_w", w_down="w_down", s_down="s_down",
                           w_up="w_up", s_up="s_up", w_inj="w_inj", lr8="lr8", lre="lre", inj="inj", x8="x8",
                           xe="xe", xbf="xbf", inv="inv")  # fmt: skip


# ------------------------------------------------------------------ norm pieces
def _norm_load(e: _Hc, hc, row=0, keep=None):
    """This thread's 3 vectors of 8 bf16 (vector ``tid + 512 r``, clamped) of request row ``row`` of the memory
    stream ``hc.h`` (``h_in``) or of ``w'`` (``hc.norm_w``, when ``row`` is None)."""
    out = []
    arg = hc.norm_w if row is None else hc.h
    rs = e.rsrc(arg)
    for r in range(ROUNDS):
        v = e.imin(e.add(e.tid, THREADS * r), VECS - 1)
        i = e.mul(v, 4) if row is None else e.row_ix(e.mul(v, 4), row, HC_WORDS)
        off = e.mul(i, 4)
        if keep is not None:
            off = e.sel(keep, off, OOB_BYTES)
        out.append(e.vec(e.load(rs, off, 4, e.k.nw_cpol)))
    return out


def _norm_w(e: _Hc, hc):
    return _norm_load(e, hc, None)


def _poll_streams(e: _Hc, hc, rows):
    """``poll_streams``: the 3 vectors of each request row of ``rows`` from the ``w`` / ``ph`` mailbox."""
    specs = []
    for row in rows:
        for r in range(ROUNDS):
            v = e.imin(e.add(e.tid, THREADS * r), VECS - 1)
            specs += [(hc.h, e.mb_ix(e.mul(v, 4), row), 2), (hc.h, e.mb_ix(e.add(e.mul(v, 4), 2), row), 2)]
    w = e.poll(specs)
    return [[w[2 * (q * ROUNDS + r)] + w[2 * (q * ROUNDS + r) + 1] for r in range(ROUNDS)] for q in range(len(rows))]


def _chunk_sum(e: _Hc, words):
    b = e.b
    q = [b.fmul(x, x) for x in e.bf16x8(words)]
    s = b.fadd(b.fadd(b.fadd(q[0], q[1]), b.fadd(q[2], q[3])), b.fadd(b.fadd(q[4], q[5]), b.fadd(q[6], q[7])))
    return e.xred(e.xred(s, 1, e.fadd2), 2, e.fadd2)


def _norm_sums(e: _Hc, hs, row=0):
    b = e.b
    sums = [_chunk_sum(e, h) for h in hs]
    with b.scf_if(e.eq(e.umod(e.lane, 4), 0)):
        e.st(e.lix("hsum", e.udiv(e.tid, 4), row), sums[0])
        e.st(e.lix("hsum", e.udiv(e.add(e.tid, THREADS), 4), row), sums[1])
        with b.scf_if(e.lt(e.tid, VECS - 2 * THREADS)):
            e.st(e.lix("hsum", e.udiv(e.add(e.tid, 2 * THREADS), 4), row), sums[2])


def _halves_sums(e: _Hc, hc, rows=None):
    """``halves_sums``: chunk ``i``'s square sum = the producer's 16-column sums ``2 i`` + ``2 i + 1`` (C1 / P2 build
    them with the norm's own tree); ``rows``: the mailbox rows (LDS rows 0, 1, ...), default every row."""
    b = e.b
    rows = list(range(e.rows)) if rows is None else rows
    with b.scf_if(e.lt(e.tid, VECS // 4)):
        vs = e.poll([(hc.hs, e.mb_ix(e.mul(e.tid, 2), r), 2) for r in rows])
        for q in range(len(rows)):
            v = vs[q]
            e.st(e.lix("hsum", e.tid, q), b.fadd(b.bitcast(v[0], F32), b.bitcast(v[1], F32)))


def _inv_of(e: _Hc, bb, row):
    """The inverse RMS of stream ``bb`` from ``hsum`` (every lane gets the same bits)."""
    b = e.b
    v = e.ld(e.lix("hsum", e.add(e.mul(bb, STREAM_CHUNKS), e.lane), row), F32)
    hi = e.ld(e.lix("hsum", e.add(e.mul(bb, STREAM_CHUNKS), e.add(e.imin(e.lane, 15), 64)), row), F32)
    v = b.select(e.lt(e.lane, 16), b.fadd(v, hi), v)
    ms = b.fmul(e.natural_sum(v), e.fbits(INV_HIDDEN))
    stab = b.fadd(ms, e.fbits(EPS + ((20 << 23) if e.k.mutate == "eps" else 0)))
    if e.k.invmath == "rsq32":
        return b.rsqrt(stab)
    return b.cvt_f64_to_f32(b.fdiv(b.const_f64(1.0), b.sqrt(b.cvt_f32_to_f64(stab))))


def _norm_inv(e: _Hc, rows):
    """Inverse RMS of the 4 streams from ``hsum`` into LDS ``inv`` (``rows`` request rows of the view). Barriers at
    both ends; every thread calls it. ``inv=regs`` (one-row views): every wave computes the 4 values itself and keeps
    them in registers (the same operations: the same bits), so the trailing barrier and the LDS round trip go."""
    b = e.b
    e.barrier()
    e.invregs = None
    if rows == 1 and e.k.inv == "regs":
        e.invregs = [_inv_of(e, e.c(q), 0) for q in range(HC_COUNT)]
        return
    for k in range(-(-rows * HC_COUNT // WAVES)):
        if rows == 1:
            bb, row = e.wave, 0
            cond = e.lt(bb, HC_COUNT)
        else:
            q = e.add(e.wave, WAVES * k)
            bb, row = e.umod(q, HC_COUNT), e.udiv(q, HC_COUNT)
            cond = e.lt(e.add(bb, e.mul(row, HC_COUNT)), rows * HC_COUNT)
        with b.scf_if(cond):
            inv = _inv_of(e, bb, row)
            with b.scf_if(e.eq(e.lane, 0)):
                e.st(e.lix("inv", bb, row), inv)
    e.barrier()


def _inv_for(e: _Hc, stream, row):
    regs = getattr(e, "invregs", None)
    if regs is not None and isinstance(row, int) and row == 0:
        b = e.b
        x = regs[0]
        for q in range(1, HC_COUNT):
            x = b.select(e.eq(stream, q), regs[q], x)
        return x
    return e.ld(e.lix("inv", stream, row), F32)


def _norm_values(e: _Hc, h, w, r, row=0, vv=None):
    b = e.b
    v = vv if vv is not None else e.imin(e.add(e.tid, THREADS * r), VECS - 1)
    inv = _inv_for(e, e.udiv(v, STREAM_VECS), row)
    xs, ws = e.bf16x8(h), e.bf16x8(w)
    return [e.bf16r(b.fmul(b.fmul(xs[i], inv), ws[i])) for i in range(8)]


def _pack8(e: _Hc, y):
    """amax of a 32-value block (8 values here, lanes ^1 ^2), its exponent and the two fp8 words."""
    b = e.b
    amax = e.absmax_start(y[0])
    for i in range(1, 8):
        amax = e.absmax2(amax, e.absmax_start(y[i]))
    amax = e.absmax_end(e.xred(e.xred(amax, 1, e.absmax2), 2, e.absmax2))
    ex, scale = e.e8m0(amax)
    words = b.vec_pack([e.fp8_word(*y[0:4], scale), e.fp8_word(*y[4:8], scale)], I32)
    return ex, words


def _norm_pack(e: _Hc, hs_rows, ws):
    b = e.b
    for q in range(len(hs_rows)):
        for r in range(ROUNDS):
            y = _norm_values(e, hs_rows[q][r], ws[r], r, q)
            ex, words = _pack8(e, y)
            v = e.add(e.tid, THREADS * r)
            with b.scf_if(e.lt(v, VECS)):
                e.st(e.lix("n8", e.mul(v, 2), q), words, 2)
                with b.scf_if(e.eq(e.umod(e.lane, 4), 0)):
                    e.st(e.lix("ne", e.udiv(v, 4), q), ex)
    e.barrier()


def _quarter_vec(e: _Hc, part):
    t = e.imin(e.tid, QUARTER_VECS - 1)
    return e.add(e.mul(e.add(part, e.mul(e.udiv(t, 16), 4)), 16), e.umod(t, 16))


def _quarter_pack(e: _Hc, h, w, v):
    b = e.b
    y = _norm_values(e, h, w, 0, 0, vv=v)
    ex, words = _pack8(e, y)
    with b.scf_if(e.lt(e.tid, QUARTER_VECS)):
        e.st(e.lix("n8", e.mul(v, 2)), words, 2)
        with b.scf_if(e.eq(e.umod(e.lane, 4), 0)):
            e.st(e.lix("ne", e.udiv(v, 4)), ex)
    e.barrier()


def _inj_word(e: _Hc, v):
    return e.add(e.mul(e.udiv(v, INJ_VECS), INJ_SEG), e.mul(e.umod(v, INJ_VECS), 4))


def _norm_bf16(e: _Hc, hs, ws):
    b = e.b
    for r in range(ROUNDS):
        y = _norm_values(e, hs[r], ws[r], r)
        words = b.vec_pack([e.bf16_pair(y[2 * i], y[2 * i + 1]) for i in range(4)], I32)
        v = e.add(e.tid, THREADS * r)
        with b.scf_if(e.lt(v, VECS)):
            e.st(e.add(e.lbase("nbf"), _inj_word(e, v)), words, 4)
    e.barrier()


# ------------------------------------------------------------------ the scaled-MFMA chains
def _mfma(e: _Hc, acc, a8, w4, ea, sw, opsel_a, opsel_b):
    return e.b.mfma_scale_mx(a8, w4, acc, ea, sw, shape="16x16x128", fmt_a="fp8", fmt_b="fp4",
                             opsel_a=opsel_a, opsel_b=opsel_b)  # fmt: skip


def _a_frag(e: _Hc, x8_base, kit):
    """Lane's fp8 A operand of k_iter ``kit``: two 16-byte halves 64 bytes apart (``common.a_operand``)."""
    b = e.b
    off = e.add(x8_base, kit * 32, e.mul(e.udiv(e.lane, 16), 4))
    a0 = e.ldn(off, 4)
    a1 = e.ldn(e.add(off, 16), 4)
    return b.vec_pack(e.vec(a0) + e.vec(a1), I32)


def _chain(e: _Hc, steps, frag, scale_a, weight, scale_b, opsel_b):
    """``acc = sum_i MFMA(frag(i), weight(i), scale_a(i), scale_b(i))`` in step order with the read-ahead / interleave
    / scale knobs. ``scale_a(i)`` -> (i32 dword, opsel)."""
    b, k = e.b, e.k
    acc = b.vec_pack([e.f(0.0)] * 4, F32)
    acc2 = b.vec_pack([e.f(0.0)] * 4, F32) if k.acc == 2 else None
    ra = k.ra
    ops = {}

    def fetch(i):
        if i < steps and i not in ops:
            ops[i] = frag(i)

    sa = None
    if k.scale in ("hoist", "opsel"):
        sa = [scale_a(i) for i in range(steps)]
    for i in range(min(ra, steps)):
        fetch(i)
    for i in range(steps):
        fetch(i)
        fetch(i + ra)
        a8 = ops.pop(i)
        ea, osa = sa[i] if sa is not None else scale_a(i)
        tgt = acc2 if (acc2 is not None and i % 2) else acc
        tgt = _mfma(e, tgt, a8, weight(i), ea, scale_b(i), osa, opsel_b(i))
        if acc2 is not None and i % 2:
            acc2 = tgt
        else:
            acc = tgt
    if k.sgb:
        # the chain's pipeline: the first (ra + 1) steps' LDS reads, then per step its MFMA and the reads of step
        # i + ra + 1 (``ra`` = 0: FlyDSL's order, pinned)
        per = 3 if sa is None else 2
        if sa is not None:  # the hoisted / packed scale words first
            b.sched_group_barrier(SGB_DS_READ, steps, 0)
        b.sched_group_barrier(SGB_DS_READ, per * (ra + 1), 0)
        for i in range(steps):
            b.sched_group_barrier(SGB_MFMA, 1, 0)
            if i + ra + 1 < steps:
                b.sched_group_barrier(SGB_DS_READ, per, 0)
    if acc2 is not None:  # tolerance class: K-split partial sums
        acc = b.vec_pack([b.fadd(x, y) for x, y in zip(e.vec(acc), e.vec(acc2))], F32)
    return acc


def _down_scales(e: _Hc, ne_base, part, nr_view=False):
    """``scale_a(i)`` of the down chain over ``kit = part + 4 i`` with exponents at ``ne_base`` (word index):
    per-step dwords (op_sel 0), or for ``scale=opsel`` 5 dwords of 4 packed bytes (op_sel ``i % 4``)."""
    b = e.b
    g = e.udiv(e.lane, 16)

    def word(i):
        return e.ld(e.add(ne_base, e.mul(e.add(part, 4 * i), 4), g))

    if e.k.scale != "opsel":
        return lambda i: (word(i), 0)
    words = [word(i) for i in range(DOWN_CHAIN)]
    packed = []
    for j in range(DOWN_CHAIN // 4):
        p = words[4 * j]
        for q in range(1, 4):
            p = b.lor(p, b.shl(words[4 * j + q], e.c(8 * q)))
        packed.append(p)
    return lambda i: (packed[i // 4], i % 4)


# ------------------------------------------------------------------ A1 / M1
def _a1_units(e: _Hc, hc, down, nb, part):
    """The wave's down chain: 20 weight dwordx4 and the 5 scale dwords (a dwordx4 + a dword), out of range unless
    ``down``."""
    rw, rs = e.rsrc(hc.w_down, RECORDS["w_down"]), e.rsrc(hc.s_down, RECORDS["s_down"])
    cp = e.k.w_cpol
    ws = []
    for i in range(DOWN_CHAIN):
        off = e.mul(e.add(e.mul(e.add(e.mul(nb, DOWN_KI), e.add(part, 4 * i)), 64), e.lane), 16)
        ws.append(e.load(rw, e.sel(down, off, OOB_BYTES), 4, cp))
    sw = e.mul(e.add(e.mul(e.add(e.mul(nb, 4), part), DOWN_SBLOCKS * 64), e.lane), 16)
    s0 = e.vec(e.load(rs, e.sel(down, sw, OOB_BYTES), 4, cp))
    s1 = e.load(rs, e.sel(down, e.add(sw, 64 * 16), OOB_BYTES), 1, cp)
    return ws, s0 + [s1]


def _down_chain(e: _Hc, units, part, x8=None, xe=None):
    n8 = x8 if x8 is not None else e.lbase("n8")
    ne = xe if xe is not None else e.lbase("ne")
    ws, ss = units
    return _chain(
        e, DOWN_CHAIN,
        frag=lambda i: _a_frag(e, n8, part + 4 * i) if isinstance(part, int) else _a_frag_dyn(e, n8, part, i),
        scale_a=_down_scales(e, ne, part),
        weight=lambda i: ws[i],
        scale_b=lambda i: ss[i // 4],
        opsel_b=lambda i: i % 4,
    )  # fmt: skip


def _a_frag_dyn(e: _Hc, x8_base, part, i):
    b = e.b
    kit = e.add(part, 4 * i)
    off = e.add(x8_base, e.mul(kit, 32), e.mul(e.udiv(e.lane, 16), 4))
    a0 = e.ldn(off, 4)
    a1 = e.ldn(e.add(off, 16), 4)
    return b.vec_pack(e.vec(a0) + e.vec(a1), I32)


def _a1_down(e: _Hc, hc, g, hs, ws, units, rr):
    b = e.b
    lane, wave = e.lane, e.wave
    part = e.umod(wave, 4)
    _norm_pack(e, [hs], ws)
    acc = _down_chain(e, units, part)
    with b.scf_if(e.lt(lane, 16)):
        e.st(e.lix("red", e.add(e.mul(wave, 16), lane)), b.vec_extract(acc, 0))
    e.barrier()
    with b.scf_if(e.eq(wave, 0)):
        col = e.add(e.mul(e.udiv(e.umod(lane, 32), 16), 64), e.umod(lane, 16))
        low = e.ld(e.lix("red", col), F32)
        nq = 3 if e.k.mutate == "down_part" else 4
        for q in range(1, nq):
            low = b.fadd(low, e.ld(e.lix("red", e.add(col, 16 * q)), F32))
        act = e.silu_q(low)
        amax = e.absmax_start(act)
        for off in (1, 2, 4, 8, 16):
            amax = e.xred(amax, off, e.absmax2)
        ex, scale = e.e8m0(e.absmax_end(amax))
        v = e.lane_gather(act, e.mul(e.umod(lane, 4), 8), 8)
        lo, hi = e.fp8_word(*v[0:4], scale), e.fp8_word(*v[4:8], scale)
        with b.scf_if(e.lt(lane, 4)):
            e.put2(hc.lr8, e.mb_ix(e.add(e.mul(g, 8), e.mul(lane, 2)), rr), lo, hi)
        with b.scf_if(e.eq(lane, 0)):
            e.put(hc.lre, e.mb_ix(g, rr), ex)


def _a1_inject(e: _Hc, hc, row, hs, ws, rr):
    b = e.b
    tid, lane, wave = e.tid, e.lane, e.wave
    rw = e.rsrc(hc.w_inj)
    wl = []
    for r in range(ROUNDS):
        v = e.imin(e.add(tid, THREADS * r), INJ_ROW_VECS - 1)
        wl.append(e.load(rw, e.mul(e.add(e.mul(row, INJ_ROW_VECS), v), 16), 4, e.k.wi_cpol))
    for r in range(ROUNDS):
        v = e.add(tid, THREADS * r)
        with b.scf_if(e.lt(v, INJ_ROW_VECS)):
            e.st(e.add(e.lbase("injw"), _inj_word(e, v)), wl[r], 4)
    _norm_bf16(e, hs, ws)
    with b.scf_if(e.eq(wave, 0)):
        base = e.mul(e.umod(lane, INJ_LANES), INJ_SEG)
        part = e.f(0.0)
        for ch in range(INJ_VECS // INJ_CHUNK):
            b.sched_barrier(0)
            for i in range(INJ_CHUNK):
                off = e.add(base, 4 * (ch * INJ_CHUNK + i))
                wv = e.vec(e.ldn(e.add(e.lbase("injw"), off), 4))
                yv = e.vec(e.ldn(e.add(e.lbase("nbf"), off), 4))
                wf, yf = e.bf16x8(wv), e.bf16x8(yv)
                for k in range(8):
                    part = b.fma(wf[k], yf[k], part)
        b.sched_barrier(0)
        bits = b.bitcast(part, I32)
        s = b.bitcast(b.readlane(bits, e.c(0)), F32)
        last = INJ_LANES - 1 if e.k.mutate == "inj_lane" else INJ_LANES
        for k in range(1, last):
            s = b.fadd(s, b.bitcast(b.readlane(bits, e.c(k)), F32))
        val = e.bf16r(b.fmul(e.bf16r(s), e.f(0.25)))
        val = e.bf16r(e.sigmoid_acc(val))
        val = e.bf16r(b.fmul(val, e.f(2.0)))
        with b.scf_if(e.eq(lane, 0)):
            e.put(hc.inj, e.mb_ix(row, rr), val)


def _a1_task(e: _Hc, hc, t, rr, live, pre=None):
    """One request's A1 / M1 task ``t`` (< 14) on request row ``rr`` (the CTA has a task where ``live``)."""
    b = e.b
    with b.scf_if(live):
        g = e.imin(t, DOWN_GROUPS - 1)
        nb = e.add(e.mul(g, 2), e.udiv(e.wave, 4))
        part = e.umod(e.wave, 4)
        row = e.b.sub(t, e.c(INJ_TASK))
        down = e.lt(t, INJ_TASK)
        if hc.polled and e.k.order == "streams":
            # P1 ``a1hf``: the streams and w' first, then the CTA barrier, then the down weights
            (hs,) = _poll_streams(e, hc, [rr])
            ws = _norm_w(e, hc)
            b.sched_barrier(0)
            e.barrier()
            units = _a1_units(e, hc, down, nb, part)
        elif hc.polled:
            ws = _norm_w(e, hc)
            units = _a1_units(e, hc, down, nb, part)
            e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.mul(m, hc.hint_stride), rr)), hc.producers)
            (hs,) = _poll_streams(e, hc, [rr])
        else:
            if pre is not None:
                hs, ws = pre
            else:
                hs = _norm_load(e, hc, rr)
                ws = _norm_w(e, hc)
            b.sched_barrier(0)
            if e.k.hbar:
                e.barrier()
            units = _a1_units(e, hc, down, nb, part)
        if hc.polled and e.k.stats == "halves":
            _halves_sums(e, hc, [rr])
        else:
            _norm_sums(e, hs)
        _norm_inv(e, 1)
        if e.wide_inv:
            # ``put_inv``: task 0 of the row hands A2 / M2 the 4 streams' inverse RMS
            with b.scf_if(b.land(e.eq(t, 0), e.lt(e.tid, HC_COUNT))):
                e.put(hc.inv, e.mb_ix(e.tid, rr), _inv_for(e, e.tid, 0))
        with b.scf_if(down):
            _a1_down(e, hc, g, hs, ws, units, rr)
        with b.scf_if(e.ge(t, INJ_TASK)):
            _a1_inject(e, hc, row, hs, ws, rr)


def _pre_values(e: _Hc):
    """``pre=args``: ATOM's ``a1_preload`` registers (24 i32: h vectors 0-2, then w' vectors 0-2)."""
    w = e.a.pre
    return [w[4 * r : 4 * r + 4] for r in range(3)], [w[12 + 4 * r : 16 + 4 * r] for r in range(3)]


def _xhold(e: _Hc, t, tasks):
    """P1 ``_xhold``: wave 0 spins on every M5 producer's last-row sum pair of this layer's ``phs``, then the CTA
    barriers; ``xpre``: only the CTAs without an A1 task (``t >= tasks``), the others poll the streams in A1."""
    b = e.b
    rl = e.rows - 1

    def hold():
        with b.scf_if(e.eq(e.wave, 0)):
            specs = []
            for r in range((M5_TASKS + 63) // 64):
                m = e.imin(e.add(e.lane, 64 * r), M5_TASKS - 1)
                specs.append(("phs", e.mb_ix(m, rl), 1))
            e.poll(specs)
        e.barrier()

    if e.k.xhold == "xpre":
        with b.scf_if(e.ge(t, tasks)):
            hold()
    else:
        hold()


def _a1_tasks(e: _Hc) -> int:
    sv = e.k.spread
    if sv:
        return spread_down_tasks(sv, e.rows) + e.rows * HC_COUNT
    return e.rows * TASKS_HC


def _emit_a1(e: _Hc, hc):
    t = e.a.task
    sv = e.k.spread
    if e.k.xhold != "none":
        _xhold(e, t, _a1_tasks(e))
    pre = _pre_values(e) if e.k.pre == "args" else None
    if sv:
        return _spread_a1(e, hc, t, sv, pre)
    if e.rows == 1:
        return _a1_task(e, hc, t, 0, e.lt(t, TASKS_HC), pre)
    n = e.rows * DOWN_GROUPS
    inj = e.imax(e.b.sub(t, e.c(n)), 0)
    down = e.lt(t, n)
    rr = e.b.select(down, e.udiv(t, DOWN_GROUPS), e.udiv(inj, HC_COUNT))
    lt = e.b.select(down, e.umod(t, DOWN_GROUPS), e.add(INJ_TASK, e.umod(inj, HC_COUNT)))
    _a1_task(_one_row(e), hc, lt, rr, e.lt(t, e.rows * TASKS_HC), pre)


def _one_row(e: _Hc) -> _Hc:
    """A one-request view (``one.rows = 1``): the same LDS entries (row 0), mailboxes and builder."""
    one = object.__new__(_Hc)
    one.__dict__.update(e.__dict__)
    one.rows = 1
    return one


# ------------------------------------------------------------------ the P3 spread (A1 / M1 side)
def _spread_row(e: _Hc, t, sv):
    if e.rows == 1:
        return 0
    units, dedupe = spread_mode(sv)
    n_down = spread_down_tasks(sv, e.rows)
    inj = e.udiv(e.imax(e.b.sub(t, e.c(n_down)), 0), HC_COUNT)
    if dedupe:
        return e.b.select(e.lt(t, n_down), e.c(0), inj)
    per = SPREAD_UNITS if units else SPREAD_NB
    return e.b.select(e.lt(t, n_down), e.udiv(e.imin(t, n_down - 1), per), inj)


def _spread_box_pair(e: _Hc, hc, i):
    return e.add(SPREAD_OFF[hc.down], i)


def _spread_down(e: _Hc, hc, k, rrs, sv, pre):
    """Down task ``k`` (a unit or an n-block) of the request rows ``rrs`` (LDS rows 0.., ``e`` a ``len(rrs)``-row
    view)."""
    b = e.b
    tid, lane, wave = e.tid, e.lane, e.wave
    units, _ = spread_mode(sv)
    nr = len(rrs)
    if units:
        nb, part = e.udiv(k, 4), e.umod(k, 4)
        active = e.eq(wave, 0)
    else:
        nb, part = k, e.umod(wave, 4)
        active = e.lt(wave, 4)
    quarter = sv in (10, 11) and not hc.polled and nr == 1
    if hc.polled:
        ws = _norm_w(e, hc)
        units_w = _a1_units(e, hc, active, nb, part)
        e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.mul(m, hc.hint_stride), 0)), hc.producers)
        hs_rows = _poll_streams(e, hc, rrs)
    else:
        if pre is not None and hc.down == "a1":
            hs_rows = [pre[0]] + [_norm_load(e, hc, r) for r in rrs[1:]]
            ws = pre[1]
        else:
            hs_rows = [_norm_load(e, hc, r) for r in rrs]
            ws = _norm_w(e, hc)
        if quarter:
            qpart = e.umod(e.add(part, 1), 4) if (sv == 11 or e.k.mutate == "quarter") else part
            qv = _quarter_vec(e, qpart)
            qh = e.vec(e.load(e.rsrc(hc.h), e.mul(e.row_ix(e.mul(qv, 4), rrs[0], HC_WORDS), 4), 4, e.k.nw_cpol))
            qw = e.vec(e.load(e.rsrc(hc.norm_w), e.mul(e.mul(qv, 4), 4), 4, e.k.nw_cpol))
        b.sched_barrier(0)
        if e.k.hbar:
            e.barrier()
        units_w = _a1_units(e, hc, active, nb, part)
    if hc.polled and e.k.stats == "halves":
        _halves_sums(e, hc, list(rrs))
    else:
        for q in range(nr):
            _norm_sums(e, hs_rows[q], q)
    _norm_inv(e, nr)
    if quarter:
        _quarter_pack(e, qh, qw, qv)
    else:
        _norm_pack(e, hs_rows, ws)
    if nr == 1:
        n8, ne = e.lbase("n8"), e.lbase("ne")
    else:
        r = e.imin(e.umod(lane, 16), nr - 1)
        n8 = e.add(e.lbase("n8"), e.mul(r, e.L.stride["n8"]))
        ne = e.add(e.lbase("ne"), e.mul(r, e.L.stride["ne"]))
    with b.scf_if(active):
        acc = _down_chain(e, units_w, part, n8, ne)
        with b.scf_if(e.lt(lane, 16)):
            for q in range(nr):
                aq = b.vec_extract(acc, q)
                if units:
                    e.put("isc", e.mb_ix(_spread_box_pair(e, hc, e.add(e.mul(k, 16), lane)), rrs[q]), aq)
                else:
                    e.st(e.lix("red", e.add(e.mul(wave, 16), lane), q), aq)
    if not units:
        e.barrier()
        cond = e.eq(wave, 0) if nr == 1 else e.lt(wave, nr)
        with b.scf_if(cond):
            rq = 0 if nr == 1 else e.imin(wave, nr - 1)
            col = e.umod(lane, 16)

            def red(i):
                return e.ld(e.lix("red", e.add(col, 16 * i), rq), F32)

            if sv == 5:
                low = b.fadd(b.fadd(red(0), red(1)), b.fadd(red(2), red(3)))
            elif sv in (6, 9) or e.k.mutate == "down_part":
                low = b.fadd(b.fadd(red(0), red(1)), red(2))
            else:
                low = red(0)
                for i in range(1, 4):
                    low = b.fadd(low, red(i))
            act = e.silu_q(low)
            with b.scf_if(e.lt(lane, 16)):
                row = rrs[0] if nr == 1 else rq
                e.put("isc", e.mb_ix(_spread_box_pair(e, hc, e.add(e.mul(nb, 16), lane)), row), act)


def _spread_a1(e: _Hc, hc, t, sv, pre):
    b = e.b
    rows = e.rows
    units, dedupe = spread_mode(sv)
    n_down = spread_down_tasks(sv, rows)
    per = SPREAD_UNITS if units else SPREAD_NB
    one = _one_row(e)
    with b.scf_if(e.lt(t, n_down)):
        if dedupe and rows > 1:
            _spread_down(e, hc, t, list(range(rows)), sv, pre)
        elif rows == 1:
            _spread_down(one, hc, t, [0], sv, pre)
        else:
            _spread_down(one, hc, e.umod(t, per), [e.udiv(t, per)], sv, pre)
    with b.scf_if(e.ge(t, n_down)):
        inj = e.imax(b.sub(t, e.c(n_down)), 0)
        _a1_task(one, hc, e.add(INJ_TASK, e.umod(inj, HC_COUNT)), e.udiv(inj, HC_COUNT),
                 e.lt(t, n_down + rows * HC_COUNT), pre)  # fmt: skip


# ------------------------------------------------------------------ A2 / M2
def _spread_pack(e: _Hc, hc, sv):
    b = e.b
    lane, wave, rows = e.lane, e.wave, e.rows
    units, _ = spread_mode(sv)
    items = DOWN_GROUPS * rows
    slots = 2 * WAVES
    passes = -(-items // slots)
    half = e.udiv(lane, 32)
    c32 = e.umod(lane, 32)
    per = 4 if units else 1
    specs, its = [], []
    for ps in range(passes):
        raw = e.add(e.mul(wave, 2), half, ps * slots)
        it = e.imin(raw, items - 1)
        r = 0 if rows == 1 else e.udiv(it, DOWN_GROUPS)
        g = e.umod(it, DOWN_GROUPS)
        col = e.add(e.mul(g, 32), c32)
        if units:
            nb, j = e.udiv(col, 16), e.umod(col, 16)
            specs += [("isc", e.mb_ix(_spread_box_pair(e, hc, e.add(e.mul(e.add(e.mul(nb, 4), q), 16), j)), r), 1)
                      for q in range(4)]  # fmt: skip
        else:
            specs += [("isc", e.mb_ix(_spread_box_pair(e, hc, col), r), 1)]
        its.append((raw, r, g))
    vals = e.poll(specs)
    for ps in range(passes):
        raw, r, g = its[ps]
        v = vals[ps * per : ps * per + per]
        if units:
            low = b.fadd(b.bitcast(v[0][0], F32), b.bitcast(v[1][0], F32))
            low = b.fadd(low, b.bitcast(v[2][0], F32))
            if sv not in (7, 8):
                low = b.fadd(low, b.bitcast(v[3][0], F32))
            act = e.silu_q(low)
        else:
            act = b.bitcast(v[0][0], F32)
        amax = e.absmax_start(act)
        for off in (1, 2, 4, 8, 16):
            amax = e.xred(amax, off, e.absmax2)
        ex, scale = e.e8m0(e.absmax_end(amax))
        w8 = e.lane_gather(act, e.add(e.mul(half, 32), e.mul(e.umod(lane, 4), 8)), 8)
        lo, hi = e.fp8_word(*w8[0:4], scale), e.fp8_word(*w8[4:8], scale)
        ok = e.lt(raw, items)
        with b.scf_if(b.land(e.lt(c32, 4), ok)):
            e.st(e.lix("l8", e.add(e.mul(g, 8), e.mul(c32, 2)), r), b.vec_pack([lo, hi], I32), 2)
        with b.scf_if(b.land(e.eq(c32, 0), ok)):
            e.st(e.lix("le", g, r), ex)


def _emit_a2(e: _Hc, hc):
    if e.wide:
        return _emit_a2_wide(e, hc)
    b = e.b
    tid, lane, wave, rows = e.tid, e.lane, e.wave, e.rows
    t = e.a.task
    stream = e.udiv(wave, 2)
    atom = e.umod(wave, 2)
    nb = e.add(e.mul(t, 2), e.mul(stream, UP_NB), atom)
    j = e.umod(lane, 16)
    idx = e.add(e.mul(stream, HIDDEN), e.mul(t, 32), e.mul(atom, 16), j)
    with b.scf_if(e.lt(t, TASKS_UP)):
        rwu, rsu, rnw = e.rsrc(hc.w_up), e.rsrc(hc.s_up), e.rsrc(hc.norm_w)
        cp = e.k.w_cpol

        def weights():
            wu = [e.load(rwu, e.mul(e.add(e.mul(e.add(e.mul(nb, 4), kit), 64), lane), 16), 4, cp) for kit in range(UP_KI)]
            su = e.load(rsu, e.mul(e.add(e.mul(nb, 64), lane), 4), 1, cp)
            ww = e.load(rnw, e.mul(e.udiv(idx, 2), 4), 1, e.k.nw_cpol)
            return wu, su, ww

        h_first = e.k.a2order == "h_first" and not hc.polled
        if not h_first:
            wu, su, ww = weights()
        if hc.polled:
            e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.mul(e.umod(m, hc.producers), hc.hint_stride),
                                                       e.udiv(m, hc.producers))), hc.producers * rows)  # fmt: skip
            hws = [v[0] for v in e.poll([(hc.h, e.mb_ix(e.udiv(idx, 2), r), 1) for r in range(rows)])]
            _halves_sums(e, hc)
        else:
            rh = e.rsrc(hc.h)
            hws = [e.load(rh, e.mul(e.row_ix(e.udiv(idx, 2), r, HC_WORDS), 4), 1, e.k.nw_cpol) for r in range(rows)]
            nl = [_norm_load(e, hc, r) for r in range(rows)] if h_first else None
            if h_first:
                wu, su, ww = weights()
            for r in range(rows):
                _norm_sums(e, nl[r] if h_first else _norm_load(e, hc, r), r)
        _norm_inv(e, rows)
        # the low-rank pack (80 fp8 words then zeros to 96, 10 exponents then 127)
        sv = e.k.spread
        if sv:
            # the spread HC's producers: one unit / n-block task each; its last pair (j = 15) is the hint
            units, dedupe = spread_mode(sv)
            per = SPREAD_UNITS if units else SPREAD_NB
            nrow = 1 if dedupe else rows
            e.wait_producers(lambda m: ("isc", e.mb_ix(_spread_box_pair(e, hc, e.add(e.mul(e.umod(m, per), 16), 15)),
                                                       e.udiv(m, per))), per * nrow, after_barrier=True)  # fmt: skip
        else:
            e.wait_producers(lambda m: (hc.lre, e.mb_ix(e.umod(m, DOWN_GROUPS), e.udiv(m, DOWN_GROUPS))),
                             DOWN_GROUPS * rows, after_barrier=True)  # fmt: skip
        if sv:
            _spread_pack(e, hc, sv)
        else:
            with b.scf_if(e.lt(tid, LOWRANK // 8)):
                vs = e.poll([(hc.lr8, e.mb_ix(e.mul(tid, 2), r), 2) for r in range(rows)])
                for r in range(rows):
                    e.st(e.lix("l8", e.mul(tid, 2), r), b.vec_pack(vs[r], I32), 2)
            with b.scf_if(b.land(e.ge(tid, 64), e.lt(tid, 64 + DOWN_GROUPS))):
                es = e.poll([(hc.lre, e.mb_ix(b.sub(tid, e.c(64)), r), 1) for r in range(rows)])
                for r in range(rows):
                    e.st(e.lix("le", b.sub(tid, e.c(64)), r), es[r][0])
        with b.scf_if(b.land(e.ge(tid, 128), e.lt(tid, 128 + 16))):
            for r in range(rows):
                e.st(e.lix("l8", e.add(b.sub(tid, e.c(128)), LOWRANK // 4), r), e.c(0))
        with b.scf_if(b.land(e.ge(tid, 192), e.lt(tid, 192 + 2))):
            for r in range(rows):
                e.st(e.lix("le", e.add(b.sub(tid, e.c(192)), DOWN_GROUPS), r), e.c(SCALE_ONE))
        e.barrier()
        if rows == 1:
            xl8, xle = e.lbase("l8"), e.lbase("le")
        else:
            mr = e.imin(e.umod(lane, 16), rows - 1)
            xl8 = e.add(e.lbase("l8"), e.mul(mr, e.L.stride["l8"]))
            xle = e.add(e.lbase("le"), e.mul(mr, e.L.stride["le"]))
        g = e.udiv(lane, 16)
        kit_of = (lambda i: 1 if i == 2 else i) if e.k.mutate == "up_kit" else (lambda i: i)
        sa = (lambda i: (e.ld(e.add(xle, e.mul(kit_of(i), 4), g)), 0))
        if e.k.scale == "opsel":
            w3 = [e.ld(e.add(xle, 4 * i, g)) for i in range(UP_KI)]
            p = b.lor(b.lor(w3[0], b.shl(w3[1], e.c(8))), b.shl(w3[kit_of(2)], e.c(16)))
            sa = lambda i: (p, i)  # noqa: E731
        elif e.k.scale == "hoist":
            hs_ = [e.ld(e.add(xle, e.mul(kit_of(i), 4), g)) for i in range(UP_KI)]
            sa = lambda i: (hs_[i], 0)  # noqa: E731
        acc = _chain(e, UP_KI, frag=lambda i: _a_frag(e, xl8, kit_of(i)), scale_a=sa, weight=lambda i: wu[i],
                     scale_b=lambda i: su, opsel_b=lambda i: i)  # fmt: skip
        for r in range(rows):
            hw = hws[r]
            rw = e.bf16r(e.sigmoid_acc(e.bf16r(b.vec_extract(acc, r))))
            even = e.eq(e.umod(j, 2), 0)
            x = b.bitcast(b.select(even, b.shl(hw, e.c(16)), b.land(hw, e.c(-65536))), F32)
            w = b.bitcast(b.select(even, b.shl(ww, e.c(16)), b.land(ww, e.c(-65536))), F32)
            inv = _inv_for(e, stream, r)
            y = e.bf16r(b.fmul(b.fmul(x, inv), w))
            with b.scf_if(e.lt(lane, 16)):
                e.st(e.lix("prod", e.add(e.mul(stream, 32), e.mul(atom, 16), j), r), e.bf16r(b.fmul(rw, y)))
        e.barrier()
        cond = e.eq(wave, 0) if rows == 1 else e.lt(wave, rows)
        with b.scf_if(cond):
            rq = 0 if rows == 1 else e.imin(wave, rows - 1)
            col = e.umod(lane, 32)
            s = e.f(0.0)
            for bb in range(HC_COUNT):
                s = b.fadd(s, e.ld(e.lix("prod", e.add(bb * 32, col), rq), F32))
            mixed = e.bf16r(b.fmul(s, e.f(0.5 if e.k.mutate == "mix_avg" else 0.25)))
            amax = e.absmax_start(mixed)
            for off in (1, 2, 4, 8, 16):
                amax = e.xred(amax, off, e.absmax2)
            ex, scale = e.e8m0(e.absmax_end(amax))
            v8 = e.lane_gather(mixed, e.mul(e.umod(lane, 4), 8), 8)
            lo, hi = e.fp8_word(*v8[0:4], scale), e.fp8_word(*v8[4:8], scale)
            v4 = e.lane_gather(mixed, e.mul(e.umod(lane, 8), 4), 4)
            with b.scf_if(e.lt(lane, 4)):
                e.put2(hc.x8, e.mb_ix(e.add(e.mul(t, 8), e.mul(lane, 2)), rq), lo, hi)
            with b.scf_if(e.lt(lane, 8)):
                e.put2(hc.xbf, e.mb_ix(e.add(e.mul(t, 16), e.mul(lane, 2)), rq),
                       e.bf16_pair(v4[0], v4[1]), e.bf16_pair(v4[2], v4[3]))  # fmt: skip
            with b.scf_if(e.eq(lane, 0)):
                e.put(hc.xe, e.mb_ix(t, rq), ex)


def _a2_mix(e: _Hc, hc, t, q, cond):
    """A2's epilogue for request row ``q`` (where ``cond``): the stream average, its MXFP8 pack and bf16 words
    (``a2``'s wave ``r`` part; ``hc_wide._a2_out`` past ``WAVES`` rows)."""
    b = e.b
    lane = e.lane
    with b.scf_if(cond):
        rq = e.imin(q, e.rows - 1)
        col = e.umod(lane, 32)
        s = e.f(0.0)
        for bb in range(HC_COUNT):
            s = b.fadd(s, e.ld(e.lix("prod", e.add(bb * 32, col), rq), F32))
        mixed = e.bf16r(b.fmul(s, e.f(0.5 if e.k.mutate == "mix_avg" else 0.25)))
        amax = e.absmax_start(mixed)
        for off in (1, 2, 4, 8, 16):
            amax = e.xred(amax, off, e.absmax2)
        ex, scale = e.e8m0(e.absmax_end(amax))
        v8 = e.lane_gather(mixed, e.mul(e.umod(lane, 4), 8), 8)
        lo, hi = e.fp8_word(*v8[0:4], scale), e.fp8_word(*v8[4:8], scale)
        v4 = e.lane_gather(mixed, e.mul(e.umod(lane, 8), 4), 4)
        with b.scf_if(e.lt(lane, 4)):
            e.put2(hc.x8, e.mb_ix(e.add(e.mul(t, 8), e.mul(lane, 2)), rq), lo, hi)
        with b.scf_if(e.lt(lane, 8)):
            e.put2(hc.xbf, e.mb_ix(e.add(e.mul(t, 16), e.mul(lane, 2)), rq),
                   e.bf16_pair(v4[0], v4[1]), e.bf16_pair(v4[2], v4[3]))  # fmt: skip
        with b.scf_if(e.eq(lane, 0)):
            e.put(hc.xe, e.mb_ix(t, rq), ex)


def _emit_a2_wide(e: _Hc, hc):
    """A2 / M2 of a wide build (``hc_wide.a2``, more than ``MAX_ROWS`` rows): ``a2`` with the pre-polls batched
    (``pre_poll_wide``), request row ``r``'s up result from its lanes of the MFMA tile (``acc[r % 4]`` of lanes
    ``16 (r // 4) ..``), from ``INV_ROWS`` rows the inverse RMS polled from A1 / M1 (``poll_inv``) instead of rebuilt,
    and the epilogue in passes of ``WAVES`` rows. Knobs as ``a2`` (no spread: ATOM builds none for grouped layers)."""
    b = e.b
    tid, lane, wave, rows = e.tid, e.lane, e.wave, e.rows
    t = e.a.task
    stream = e.udiv(wave, 2)
    atom = e.umod(wave, 2)
    nb = e.add(e.mul(t, 2), e.mul(stream, UP_NB), atom)
    j = e.umod(lane, 16)
    idx = e.add(e.mul(stream, HIDDEN), e.mul(t, 32), e.mul(atom, 16), j)
    with b.scf_if(e.lt(t, TASKS_UP)):
        rwu, rsu, rnw = e.rsrc(hc.w_up), e.rsrc(hc.s_up), e.rsrc(hc.norm_w)
        cp = e.k.w_cpol

        def weights():
            wu = [e.load(rwu, e.mul(e.add(e.mul(e.add(e.mul(nb, 4), kit), 64), lane), 16), 4, cp) for kit in range(UP_KI)]
            su = e.load(rsu, e.mul(e.add(e.mul(nb, 64), lane), 4), 1, cp)
            ww = e.load(rnw, e.mul(e.udiv(idx, 2), 4), 1, e.k.nw_cpol)
            return wu, su, ww

        def poll_h():
            return [v[0] for v in e.poll([(hc.h, e.mb_ix(e.udiv(idx, 2), r), 1) for r in range(rows)])]

        h_first = e.k.a2order == "h_first" and not hc.polled
        if not h_first:
            wu, su, ww = weights()
        hws = None
        if hc.polled:
            if not e.wide_inv:
                e.wait_producers(lambda m: (hc.hs, e.mb_ix(e.mul(e.umod(m, hc.producers), hc.hint_stride),
                                                           e.udiv(m, hc.producers))), hc.producers * rows)  # fmt: skip
                hws = poll_h()
                _halves_sums(e, hc)
        else:
            rh = e.rsrc(hc.h)
            hws = [e.load(rh, e.mul(e.row_ix(e.udiv(idx, 2), r, HC_WORDS), 4), 1, e.k.nw_cpol) for r in range(rows)]
            nl = None
            if not e.wide_inv and h_first:
                nl = [_norm_load(e, hc, r) for r in range(rows)]
            if h_first:
                wu, su, ww = weights()
            if not e.wide_inv:
                for r in range(rows):
                    _norm_sums(e, nl[r] if h_first else _norm_load(e, hc, r), r)
        if not e.wide_inv:
            _norm_inv(e, rows)
        # the low-rank pack (80 fp8 words then zeros to 96, 10 exponents then 127); from INV_ROWS rows the
        # streams' inverse RMS (A1's task 0 of each row)
        e.wait_producers(lambda m: (hc.lre, e.mb_ix(e.umod(m, DOWN_GROUPS), e.udiv(m, DOWN_GROUPS))),
                         DOWN_GROUPS * rows)  # fmt: skip
        if hc.polled and e.wide_inv:
            hws = poll_h()  # A1's tasks of a row polled all its streams before publishing
        with b.scf_if(e.lt(tid, LOWRANK // 8)):
            vs = e.poll([(hc.lr8, e.mb_ix(e.mul(tid, 2), r), 2) for r in range(rows)])
            for r in range(rows):
                e.st(e.lix("l8", e.mul(tid, 2), r), b.vec_pack(vs[r], I32), 2)
        with b.scf_if(b.land(e.ge(tid, 64), e.lt(tid, 64 + DOWN_GROUPS))):
            es = e.poll([(hc.lre, e.mb_ix(b.sub(tid, e.c(64)), r), 1) for r in range(rows)])
            for r in range(rows):
                e.st(e.lix("le", b.sub(tid, e.c(64)), r), es[r][0])
        with b.scf_if(b.land(e.ge(tid, 128), e.lt(tid, 128 + 16))):
            for r in range(rows):
                e.st(e.lix("l8", e.add(b.sub(tid, e.c(128)), LOWRANK // 4), r), e.c(0))
        with b.scf_if(b.land(e.ge(tid, 192), e.lt(tid, 192 + 2))):
            for r in range(rows):
                e.st(e.lix("le", e.add(b.sub(tid, e.c(192)), DOWN_GROUPS), r), e.c(SCALE_ONE))
        if e.wide_inv:
            # ``poll_inv``: every row's inverse RMS, one value per thread from INV_TID
            with b.scf_if(b.land(e.ge(tid, INV_TID), e.lt(tid, INV_TID + HC_COUNT * rows))):
                q = b.sub(tid, e.c(INV_TID))
                bb, r = e.umod(q, HC_COUNT), e.udiv(q, HC_COUNT)
                (v,) = e.poll([(hc.inv, e.mb_ix(bb, r), 1)])
                e.st(e.lix("inv", bb, r), v[0])
        e.barrier()
        mr = e.imin(e.umod(lane, 16), rows - 1)
        xl8 = e.add(e.lbase("l8"), e.mul(mr, e.L.stride["l8"]))
        xle = e.add(e.lbase("le"), e.mul(mr, e.L.stride["le"]))
        g = e.udiv(lane, 16)
        kit_of = (lambda i: 1 if i == 2 else i) if e.k.mutate == "up_kit" else (lambda i: i)
        sa = (lambda i: (e.ld(e.add(xle, e.mul(kit_of(i), 4), g)), 0))
        if e.k.scale == "opsel":
            w3 = [e.ld(e.add(xle, 4 * i, g)) for i in range(UP_KI)]
            p = b.lor(b.lor(w3[0], b.shl(w3[1], e.c(8))), b.shl(w3[kit_of(2)], e.c(16)))
            sa = lambda i: (p, i)  # noqa: E731
        elif e.k.scale == "hoist":
            hs_ = [e.ld(e.add(xle, e.mul(kit_of(i), 4), g)) for i in range(UP_KI)]
            sa = lambda i: (hs_[i], 0)  # noqa: E731
        acc = _chain(e, UP_KI, frag=lambda i: _a_frag(e, xl8, kit_of(i)), scale_a=sa, weight=lambda i: wu[i],
                     scale_b=lambda i: su, opsel_b=lambda i: i)  # fmt: skip
        quad = e.udiv(lane, 16)
        rws = {}  # row r's up result is acc[r % 4] (its lanes): one sigmoid per tile row, as FlyDSL's trace CSEs it
        for r in range(rows):
            hw = hws[r]
            if r % 4 not in rws:
                rws[r % 4] = e.bf16r(e.sigmoid_acc(e.bf16r(b.vec_extract(acc, r % 4))))
            rw = rws[r % 4]
            even = e.eq(e.umod(j, 2), 0)
            x = b.bitcast(b.select(even, b.shl(hw, e.c(16)), b.land(hw, e.c(-65536))), F32)
            w = b.bitcast(b.select(even, b.shl(ww, e.c(16)), b.land(ww, e.c(-65536))), F32)
            inv = _inv_for(e, stream, r)
            y = e.bf16r(b.fmul(b.fmul(x, inv), w))
            with b.scf_if(e.eq(quad, r // 4)):  # ``row_lanes``
                e.st(e.lix("prod", e.add(e.mul(stream, 32), e.mul(atom, 16), j), r), e.bf16r(b.fmul(rw, y)))
        e.barrier()
        # lane L (and L + 32): mixed[32 t + L]; wave w mixes request row w + WAVES k in pass k
        for ek in range(-(-rows // WAVES)):
            q = e.add(wave, WAVES * ek) if ek else wave
            _a2_mix(e, hc, t, q, e.lt(q, rows))


# ------------------------------------------------------------------ stage bodies, ABI, writer
PRE_WORDS = 24


def abi_params(spec: HcSpec):
    """The function's parameters: the stage-function ABI (``stage_fn.ABI_PARAMS``), then for ``pre=args`` the 24
    preload words (``pre0..pre23``); ``lds_base`` is appended by the device-function lowering."""
    from ..stage_fn import ABI_PARAMS

    out = list(ABI_PARAMS)
    if spec.knobs.pre == "args":
        out += [(f"pre{i}", "i32") for i in range(PRE_WORDS)]
    return out


def build_hc_kernel(spec: HcSpec):
    """The stage function's ``KernelDef``."""
    why = supported(spec)
    if why:
        raise ValueError(f"{spec.stage} {spec.kind} {spec.variant}: {why}")
    b = IRBuilder(spec.symbol)
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in abi_params(spec):
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(**vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
                        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))))  # fmt: skip
    if spec.knobs.pre == "args":
        a.pre = [vals[f"pre{i}"] for i in range(PRE_WORDS)]
    emit_hc(b, a, spec)
    b.ret()
    return b.kernel


def emit_hc(b: IRBuilder, a, spec: HcSpec, lds: SimpleNamespace | None = None) -> None:
    """Emit the stage body into ``b`` (ABI namespace ``a``: tag, task, the 41 arguments as i64, tid, lane, wave)."""
    e = _Hc(b, a, spec, lds or spec_lds(spec))
    hc = _hc(spec)
    if spec.stage in ("a1", "m1"):
        _emit_a1(e, hc)
    else:
        _emit_a2(e, hc)


def lower_hc_function(spec: HcSpec, *, arch="gfx950", flavor=None):
    return lower_device_function(build_hc_kernel(spec), arch=arch, symbol=spec.symbol, llvm_flavor=flavor,
                                 knobs=spec.knobs.fn)  # fmt: skip


def sidecar(spec: HcSpec) -> dict:
    """Extra sidecar keys: ``stage_knobs``, the NUMA layout, the spread value and the preload ABI."""
    k = spec.knobs
    meta = {"stage_knobs": knobs_spec(k), "hc": {"stage": spec.stage, "kind": spec.kind, "spread": k.spread,
            "pre": k.pre, "tolerance": k.tolerance}}  # fmt: skip
    if k.numa:
        meta.update({"mailboxes": "numa1", "numa_hdr": k.header_offset})
    if k.pre == "args":
        meta["pre_args"] = "a1_preload"
    return meta


def write_hc_functions(out_dir: str, specs, flavor=None) -> list:
    """Write ``<out_dir>/<kind>/<stage>_<variant>.{ll,json}`` per spec (ATOM:
    ``ATOM_Q38_EXTERN_STAGES=a1=<out_dir>/{kind},a2=<out_dir>/{kind},m1=...,m2=...``). Returns the paths."""
    written = []
    for spec in specs:
        fn = lower_hc_function(spec, flavor=flavor)
        d = os.path.join(out_dir, spec.kind)
        paths = write_device_function(fn, d, stem=f"{spec.stage}_{spec.variant}", bitcode=False)
        with open(paths["json"]) as f:
            meta = json.load(f)
        meta.update(sidecar(spec))
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        written.append(paths["ll"])
    return written


def framework_factory(stage: str, knobs: HcKnobs | None = None):
    """A ``framework`` stage factory (rocKE layer kernels, Path B ``layers=1`` / WP3): the HC body of ``stage`` per CTA
    with its task (``per_cta``), the layer kind from the build (the PLE layer's attention HC is polled), ``knobs``
    (default: the FlyDSL form; ``pre="args"`` is a hybrid-path ABI and is refused here: a framework kernel issues A1's
    loads in the stage, or through the framework's entry hook in WP3). The body allocates the mono's arena as its own
    LDS pool (``stage_adapt``'s rule for adapted bodies)."""
    from .. import stage_adapt as SA
    from ..framework import StageSpec as FrameworkStage

    k = knobs or HcKnobs()
    if k.pre == "args":
        raise ValueError("pre=args is the hybrid path's extern ABI; not in a framework kernel")

    def factory(kind, shape, st):
        spec = HcSpec(stage, kind, shape.rows, shape.seq, k, group=shape.group)
        why = supported(spec)
        if why:
            raise ValueError(f"{stage} {kind} {spec.variant}: {why}")

        def emit(b, ctx, p, t):
            a = SA._abi(b, ctx, p, t, [b.const_i32(0)] * 4)
            old = b.fence_barriers
            b.fence_barriers = "compiler"
            sp = spec
            if stage == "a1" and k.src == "xl" and k.xhold == "none" and getattr(ctx, "persist_knob", None):
                # the persistent kernel's hold policy (``PersistKnobs.xhold``: xpre | layer) at this layer's entry
                sp = replace(spec, knobs=replace(k, xhold=ctx.persist_knob("xhold")))
            try:
                emit_hc(b, a, sp)
            finally:
                b.fence_barriers = old

        return FrameworkStage(stage=st, emit=emit, per_cta=True)

    return factory


def _register():
    """``stage_fn`` registry entries (framework path b via ``stage_adapt``): default knobs, memory-read attention HC."""
    from ..stage_fn import register_stage

    def body_for(stage):
        def body(b, a, layout, sspec):
            spec = HcSpec(stage, getattr(sspec, "kind", "delta"), sspec.rows, sspec.seq,
                          getattr(sspec, "hc_knobs", None) or HcKnobs())  # fmt: skip
            emit_hc(b, a, spec)

        return body

    def ok(sspec, layout):
        return sspec.group == 0 and 1 <= sspec.rows <= A.MAX_ROWS and (not sspec.seq or sspec.rows > 1)

    for st in HC_STAGES:
        register_stage(st, body_for(st), ok)


_register()


def parse_variant(v: str):
    """``r<rows>[s][g<S>]`` -> (rows, seq, group)."""
    import re

    m = re.fullmatch(r"r(\d+)(s?)(?:g(\d+))?", v)
    if not m:
        raise ValueError(f"variant {v!r}: r<rows>[s][g<S>]")
    return int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="emit the rocKE HC stage functions (WP2a)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="a1,a2,m1,m2")
    ap.add_argument("--kinds", default="delta,delta_ple,qsa")
    ap.add_argument("--variants", default="r1,r2,r3,r4,r2s,r3s,r4s")
    ap.add_argument("--knobs", default="", help="HcKnobs spec (knobs_spec form) for every function")
    ap.add_argument("--spread", default="", help="per-stage spread overrides: a1=10,m1=2 (else the knobs' value)")
    ap.add_argument("--flavor", default=None)
    a = ap.parse_args(argv)
    base = knobs_from_spec(a.knobs)
    sp = dict(kv.split("=") for kv in a.spread.split(",") if kv)
    specs = []
    for kind in a.kinds.split(","):
        for v in a.variants.split(","):
            rows, seq, group = parse_variant(v)
            for st in a.stages.split(","):
                down = "m1" if st in ("m1", "m2") else "a1"
                k = replace(base, spread=int(sp[down])) if down in sp else base
                if k.pre == "args" and (st != "a1" or kind == "delta_ple" or rows > A.MAX_ROWS):
                    k = replace(k, pre="none")
                if rows > A.MAX_ROWS and k.numa:
                    k = replace(k, numa=0)  # ATOM drops the numa lever above 4 rows
                spec = HcSpec(st, kind, rows, seq, k, group=group)
                if supported(spec) is None:
                    specs.append(spec)
    print(json.dumps(write_hc_functions(a.out, specs, flavor=a.flavor), indent=1))
    return 0


if __name__ == "__main__":
    import importlib
    import sys

    sys.exit(importlib.import_module(__spec__.name).main())
