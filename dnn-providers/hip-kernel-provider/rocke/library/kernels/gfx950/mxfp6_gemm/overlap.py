# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Overlap the per-call A pack (canonical MXFP6 -> c0c1 + tile scales) with the gfx950 MXFP6 GEMM.

Live activations arrive in the canonical layout (dense 6-bit codes, raw ``[rows][K/32]`` E8M0 scales); the fastest
GEMM reads the pre-tiled c0c1 image. Run serially, the pack (memory-bound) and the GEMM (power-bound) add up. This
module holds the host-side plans and the device kernels that let them run at the same time:

* **M-chunk pipelines** (:func:`chunk_plan`, :func:`schedule_chunked`): the A rows split into ``chunks`` row panels.
  Chunk ``c``'s pack runs on a pack stream; chunk ``c``'s GEMM (an ``m = rows / chunks`` build of the same spec) waits
  for it on the GEMM stream while the pack of chunk ``c + 1`` runs. The chunk GEMMs keep the full GEMM's per-XCD
  operand panel count where possible (:data:`CHUNK_XCD`: each XCD keeps the same B panels in every chunk).
* **Next-call overlap** (:func:`schedule_next_call`, a serving / batched scenario, NOT the single-call latency):
  the pack of call ``i + 1``'s A runs during call ``i``'s GEMM (double buffer).
* :func:`build_pack_lite`: a pack kernel that fits next to a resident GEMM CTA (no LDS or a 6.4 KB LDS stage, a
  capped register budget, a fixed grid of persistent work-groups) so it shares CUs with the GEMM instead of blocking
  its CTAs.
* **Fused K-flag overlap** (:func:`build_gemm_kflag`, :func:`build_pack_lite` ``flag_tiles``, :func:`schedule_kflag`):
  the pack and the GEMM of one call run at the same time on two streams; the pack publishes per-K-chunk ready flags
  (write-through stores, a system-scope arrival counter), the GEMM polls them once per K-loop body (uncached scalar
  loads prefetched one body ahead) before refilling A, so it starts as soon as the first chunk is packed.
  ``dep="spin"``: the next call's pack waits in-kernel for the GEMM's done word -- no stream event between calls.

Schedules are lists of :class:`StreamOp` (launch / record / wait on named streams and events, the GEMM's done signal
and the pack's in-kernel wait, with the buffers each launch reads and writes); :func:`check_schedule` proves on the
host that every GEMM read of a packed buffer happens after the pack that wrote it (or under in-kernel flags) and that
no pack overwrites a buffer an earlier call's GEMM may still read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rocke.core.ir import I8, I32, IRBuilder, KernelDef, PtrType

from .pack import _hint, _transpose4x4, _udiv, _umod

__all__ = [
    "CHUNK_XCD",
    "CHUNK_XCD_ALT",
    "FLAG_POLL_LIMIT",
    "KFLAG_MUTANTS",
    "LITE_ORDERS",
    "LITE_MUTANTS",
    "Schedule",
    "StreamOp",
    "build_gemm_kflag",
    "build_pack_lite",
    "check_schedule",
    "chunk_plan",
    "chunk_spec_overrides",
    "gemm_kflag_args",
    "kflag_parity",
    "pack_lite_block",
    "pack_lite_flag_args",
    "pack_lite_grid",
    "schedule_chunked",
    "schedule_kflag",
    "schedule_next_call",
    "schedule_serial",
    "xcd_panels",
]

THREADS = 256
NUM_XCDS = 8

#: Default CTA block per XCD of a chunk GEMM (``chunks`` -> xcd_block) for a 4096-column C of 256x256 tiles: per XCD
#: ``chunks * bm`` A panels + ``bn`` B panels over all chunks (12 = the full GEMM's 4x8 for 1 / 2 / 4 / 8 chunks; 16
#: chunks: 18). :data:`CHUNK_XCD_ALT`: the other 12-panel split (more B, fewer A panels per XCD).
CHUNK_XCD: Dict[int, str] = {1: "4x8", 2: "2x8", 4: "1x8", 8: "1x4", 16: "1x2"}
CHUNK_XCD_ALT: Dict[int, str] = {1: "8x4", 2: "4x4", 4: "2x4"}


# ------------------------------------------------------------------------------------------------- chunk plans


@dataclass(frozen=True)
class Chunk:
    index: int
    row0: int  # first A / C row
    rows: int  # A / C rows of the chunk
    xcd_block: str  # the chunk GEMM's per-XCD CTA block


def chunk_plan(m: int, chunks: int, *, tile_m: int = 256, xcd_block: Optional[str] = None) -> List[Chunk]:
    """Contiguous row panels of A / C, one per chunk. ``xcd_block`` defaults to :data:`CHUNK_XCD`."""
    if chunks < 1 or m % chunks or (m // chunks) % tile_m:
        raise ValueError(f"m={m} must split into {chunks} chunks of whole {tile_m}-row tiles")
    rows = m // chunks
    blk = xcd_block or CHUNK_XCD.get(chunks)
    if blk is None:
        raise ValueError(f"no default xcd_block for {chunks} chunks; pass one")
    return [Chunk(c, c * rows, rows, blk) for c in range(chunks)]


def chunk_spec_overrides(base: Dict[str, object], chunk: Chunk) -> Dict[str, object]:
    """``Mxfp6GemmSpec`` overrides of a chunk's GEMM: ``base`` with ``m`` = the chunk rows and its XCD block."""
    return {**base, "m": chunk.rows, "xcd_block": chunk.xcd_block}


def xcd_panels(m: int, n: int, chunks: int, xcd_block: Optional[str] = None, tile: int = 256
               ) -> Tuple[List[set], List[set], List[Tuple[int, int, int]]]:
    """Host model of the chunk GEMMs' CTA placement: ``(A panels per XCD, B panels per XCD, [(chunk, tile row, tile
    col)] of every CTA)``. A CTA ``wg`` of a launch lands on XCD ``wg % 8`` (round-robin dispatch per launch)."""
    from .gemm import Mxfp6GemmSpec, tile_of_wg

    a = [set() for _ in range(NUM_XCDS)]
    bp = [set() for _ in range(NUM_XCDS)]
    ctas = []
    for ch in chunk_plan(m, chunks, tile_m=tile, xcd_block=xcd_block):
        spec = Mxfp6GemmSpec(m=ch.rows, n=n, tile_m=tile, tile_n=tile, xcd_block=ch.xcd_block)
        for wg in range((ch.rows // tile) * (n // tile)):
            tm, tn = tile_of_wg(spec, wg)
            gm = ch.row0 // tile + tm
            a[wg % NUM_XCDS].add(gm)
            bp[wg % NUM_XCDS].add(tn)
            ctas.append((ch.index, gm, tn))
    return a, bp, ctas


# ------------------------------------------------------------------------------------------------- schedules


@dataclass(frozen=True)
class StreamOp:
    """One host-side operation of a multi-stream schedule.

    ``kind``: "launch" (``what`` = kernel key, ``arg`` = its argument key), "record" / "wait" (``what`` = event name).
    ``reads`` / ``writes``: the buffers a launch touches (for :func:`check_schedule`)."""

    kind: str
    stream: str
    what: str
    arg: Tuple = ()
    reads: Tuple = ()
    writes: Tuple = ()


def _launch(stream, what, arg=(), reads=(), writes=()):
    return StreamOp("launch", stream, what, tuple(arg), tuple(reads), tuple(writes))


def _rec(stream, ev):
    return StreamOp("record", stream, ev)


def _wait(stream, ev):
    return StreamOp("wait", stream, ev)


@dataclass
class Schedule:
    """``setup``: once before the first call (event pre-records, prologue packs); ``call``: one GEMM call, repeated;
    ``streams``: the stream names used; ``label`` / ``scenario``: "single-call" or "next-call"."""

    setup: List[StreamOp]
    call: List[StreamOp]
    streams: Tuple[str, ...]
    label: str
    scenario: str = "single-call"
    meta: Dict[str, object] = field(default_factory=dict)


def schedule_serial(stream: str = "G") -> Schedule:
    """pack(all rows) then GEMM(all rows), one stream (the packA + GEMM reference)."""
    call = [_launch(stream, "pack", ("all",), reads=("A_can",), writes=("img", "sc")),
            _launch(stream, "gemm", ("all",), reads=("img", "sc"), writes=("C",))]
    return Schedule([], call, (stream,), "serial")


def schedule_chunked(chunks: int, *, mode: str = "strict", gemm_stream: str = "G", pack_stream: str = "P",
                     per_chunk_streams: bool = True, gemm_only: bool = False, mutant: str = "none") -> Schedule:
    """M-chunk pipeline. Chunk ``c``'s GEMM runs on stream ``G{c}`` (``per_chunk_streams``; kernels of ONE stream run
    one after the other, so a single GEMM stream would serialize the chunk GEMMs) after the pack of chunk ``c`` on
    the pack stream. ``mode``: "strict" (a call's packs start after every GEMM of the previous call: single-call
    semantics) | "pipelined" (pack ``c`` of call ``i + 1`` waits only for GEMM ``c`` of call ``i``: the next call's
    packs overlap this call's tail -- a serving scenario). ``gemm_only``: no packs (the chunk GEMMs on pre-packed A:
    the cost of chunking alone). ``mutant`` (tests only): "nowait" drops the GEMM's wait for its chunk's pack,
    "nowar" drops the pack's wait for the previous call's GEMMs."""
    if mode not in ("strict", "pipelined") or mutant not in ("none", "nowait", "nowar"):
        raise ValueError("mode strict / pipelined, mutant none / nowait / nowar")
    P = pack_stream
    gs = [f"{gemm_stream}{c}" if per_chunk_streams else gemm_stream for c in range(chunks)]
    setup = [_rec(gs[c], f"gm{c}") for c in range(chunks)]
    call = []
    if not gemm_only:
        if mode == "strict" and mutant != "nowar":
            call += [_wait(P, f"gm{c}") for c in range(chunks)]
        for c in range(chunks):
            if mode == "pipelined" and mutant != "nowar":
                call.append(_wait(P, f"gm{c}"))
            call.append(_launch(P, "pack", (c,), reads=("A_can",), writes=(("img", c), ("sc", c))))
            call.append(_rec(P, f"pk{c}"))
    for c in range(chunks):
        if not gemm_only and mutant != "nowait":
            call.append(_wait(gs[c], f"pk{c}"))
        reads = ("img_pre",) if gemm_only else (("img", c), ("sc", c))
        call.append(_launch(gs[c], "gemm", (c,), reads=reads, writes=(("C", c),)))
        call.append(_rec(gs[c], f"gm{c}"))
    if gemm_only and mode == "strict":  # every chunk GEMM of a call after every one of the previous call
        call += [_wait(gs[c], f"gm{d}") for c in range(chunks) for d in range(chunks) if d != c]
    streams = tuple(dict.fromkeys(gs + ([] if gemm_only else [P])))
    return Schedule(setup, call, streams, f"chunk{chunks}-{mode}" + ("-gemmonly" if gemm_only else ""),
                    "single-call" if mode == "strict" else "next-call", {"chunks": chunks, "mode": mode})


def schedule_next_call(*, gemm_stream: str = "G", pack_stream: str = "P", gemm_first: bool = True,
                       mutant: str = "none") -> Schedule:
    """Next-call overlap (serving scenario): call ``i``'s GEMM reads buffer ``i % 2`` while the pack of call ``i + 1``
    writes buffer ``(i + 1) % 2``. The setup packs call 0's A. ``gemm_first``: enqueue the GEMM before the pack
    (its CTAs reach the CUs first). The call op list covers two calls (both buffer parities)."""
    if mutant not in ("none", "nowait", "nowar"):
        raise ValueError("mutant none / nowait / nowar")
    G, P = gemm_stream, pack_stream
    # launch args: (buffer slot, logical call offset within the two-call list; the setup pack is call 0's)
    setup = [_rec(G, "gm1"), _launch(P, "pack", (0, 0), reads=("A_can",), writes=(("img", 0), ("sc", 0))),
             _rec(P, "pk0")]
    call = []
    for bb in (0, 1):
        g = ([_wait(G, f"pk{bb}")] if mutant != "nowait" else []) + [
            _launch(G, "gemm", (bb, bb), reads=(("img", bb), ("sc", bb)), writes=("C",)), _rec(G, f"gm{bb}")]
        p = ([_wait(P, f"gm{1 - bb}")] if mutant != "nowar" else []) + [
            _launch(P, "pack", (1 - bb, bb + 1), reads=("A_can",), writes=(("img", 1 - bb), ("sc", 1 - bb))),
            _rec(P, f"pk{1 - bb}")]
        call += g + p if gemm_first else p + g
    return Schedule(setup, call, (G, P), "nextcall" + ("" if gemm_first else "-packfirst"), "next-call",
                    {"calls_per_list": 2})


def check_schedule(s: Schedule, calls: int = 3) -> List[str]:
    """Host proof of a schedule's buffer hazards over ``calls`` repetitions of ``s.call`` after ``s.setup``.

    Semantics: ops of one stream run in issue order; ``record`` marks everything issued on its stream before it;
    ``wait`` makes later ops of its stream wait for the last ``record`` of that event issued before the wait.
    Returns the violations (empty = safe): RAW (a launch reading buffer X not ordered after the last launch that
    wrote X in program order) and WAR / WAW (a write not ordered after every earlier read / write of X). Buffers in
    ``s.meta["flag_guarded"]`` are read under in-kernel ready flags (the reader polls the writer's flags), so their
    RAW needs no event -- their WAR / WAW still do."""
    guarded = set(s.meta.get("flag_guarded", ()))
    ops = list(s.setup) + [op for _ in range(calls) for op in s.call]
    node_preds: List[set] = []
    last_in_stream: Dict[str, int] = {}
    pending_wait: Dict[str, set] = {}
    rec_of: Dict[str, int] = {}
    launches: List[Tuple[int, StreamOp]] = []
    for op in ops:
        if op.kind == "signal":  # written by the preceding launch of the stream at its end: a record
            op = StreamOp("record", op.stream, op.what)
        if op.kind in ("waitvalue", "kwait"):  # a CP wait / the kernel's own wait at its start
            op = StreamOp("wait", op.stream, op.what)
        if op.kind == "wait":
            if op.what in rec_of:
                pending_wait.setdefault(op.stream, set()).add(rec_of[op.what])
            continue
        i = len(node_preds)
        preds = set()
        if op.stream in last_in_stream:
            preds.add(last_in_stream[op.stream])
        preds |= pending_wait.pop(op.stream, set())
        node_preds.append(preds)
        last_in_stream[op.stream] = i
        if op.kind == "record":
            rec_of[op.what] = i
        elif op.kind == "launch":
            launches.append((i, op))
        else:
            raise ValueError(op.kind)
    anc: List[set] = []
    for preds in node_preds:  # nodes are numbered in issue order: predecessors come first
        a = set(preds)
        for p in preds:
            a |= anc[p]
        anc.append(a)
    bad = []
    last_w: Dict[object, int] = {}
    reads_since: Dict[object, List[int]] = {}
    for i, op in launches:
        for x in op.reads:
            w = last_w.get(x)
            if w is not None and w not in anc[i] and x not in guarded:
                bad.append(f"RAW {x}: launch {i} ({op.what}{op.arg}) not after its writer {w}")
            reads_since.setdefault(x, []).append(i)
        for x in op.writes:
            for r in reads_since.get(x, []):
                if r != i and r not in anc[i]:
                    bad.append(f"WAR {x}: launch {i} ({op.what}{op.arg}) not after reader {r}")
            w = last_w.get(x)
            if w is not None and w not in anc[i]:
                bad.append(f"WAW {x}: launch {i} ({op.what}{op.arg}) not after writer {w}")
            last_w[x] = i
            reads_since[x] = []
    return bad


# ------------------------------------------------------------------------------------- co-resident ("lite") pack

LITE_ORDERS = ("k", "row")
LITE_MUTANTS = ("none", "kg", "scale")


def pack_lite_grid(n_wg: int = 256) -> int:
    return n_wg


def pack_lite_block(stage: str = "none") -> int:
    """Threads per work-group of :func:`build_pack_lite` (launch with exactly this block size)."""
    return 128 if stage == "lds16" else THREADS


def build_pack_lite(rows: int, k: int, *, n_wg: int = 256, order: str = "k", nt_store: bool = False,
                    block_rows: int = 256, scales: bool = True, waves_per_eu: int = 8, flag_tiles: int = 0,
                    release: str = "fence", wait_prev: bool = False, sc1_store: bool = False,
                    stage: str = "none", nt_load: bool = False, mutant: str = "none") -> KernelDef:
    """Canonical codes + raw scales -> the c0c1 image + tile scales (byte-identical to ``pack.build_pack_operand``),
    built to share CUs with a resident GEMM CTA: no LDS, a register budget of ``waves_per_eu`` waves per SIMD
    (8 -> at most 64 VGPRs; the built kernel's count is checked against what the GEMM leaves free: a 460-register
    GEMM wave leaves 48 per lane), and ``n_wg`` persistent work-groups of 256 threads that walk the work
    grid-stride (``n_wg = 256``: one wave per SIMD on every CU).

    Codes: one c0c1 slot (24 bytes of one row and 32-K group) per thread and step: three 8-byte loads, a 16-byte c0
    and an 8-byte c1 store; 64 consecutive threads write one contiguous 1 KiB c0 run. ``order``: "k" (every row of
    K tile 0, then tile 1, ...) or "row" (256-row block outer). Scales: one (K tile, 128-row group, row r16) unit per
    thread and step: 8 raw dwords (4 K blocks of 8 rows), two 4x4 byte transposes, four 8-byte stores.
    Signature ``(codes_src, scales_src, codes_dst, scales_dst)``; grid :func:`pack_lite_grid`.

    ``flag_tiles = G > 0`` (the producer of the fused K-flag path, :func:`build_gemm_kflag`): the K tiles are
    packed in chunks of G (order "k"); after its share of chunk q every work-group publishes its stores
    (``release``: "fence" = an agent-scope release fence, i.e. L2 write-back + vmcnt(0); "sc1" = every store
    written through at system scope, then vmcnt(0)) and counts itself in ``counters[q]`` (system-scope atomic); the
    last arrival resets the counter and writes ``flags[q] = parity``. Extra parameters ``(counters, flags, gdone,
    timeouts, parity)``. ``wait_prev``: every work-group first waits (uncached scalar polls with s_sleep) until
    ``gdone[0] == 1 - parity`` -- the previous call's GEMM finished (``build_gemm_kflag(done_signal=True)``) -- so
    no stream event is needed between the calls; a wait that hits the poll limit proceeds and counts itself in
    ``timeouts[0]`` (evidence that the guard never fired). Without ``wait_prev`` the producer never waits.
    ``sc1_store`` (outside flag mode): the same write-through stores, for measuring their cost alone.

    ``nt_load``: streaming (nt) hint on the code loads (the canonical source is read once: keep it out of the
    caches the GEMM shares).

    ``stage="lds16"``: codes through LDS instead of per-slot 8-byte loads: a 128-thread work-group takes units of
    16 rows x 4 K tiles (384 B = three whole 128-byte lines per row), loads them with coalesced 16-byte loads into
    6.4 KB of LDS (pitch 400 B, as pack.build_pack_operand), then writes the 256 c0c1 slots (2 per thread; a wave =
    one contiguous 1 KiB c0 / 512 B c1 run). 6.4 KB + the GEMM's 150 KiB fit one CU (one such work-group per CU next
    to a GEMM CTA). Scales as above.

    ``mutant`` (tests only): "kg" swaps two 32-K groups, "scale" swaps two transposed scale words, "early" (flag
    mode) publishes each chunk's flag before its stores are made visible."""
    R = block_rows
    if order not in LITE_ORDERS or mutant not in LITE_MUTANTS + ("early",):
        raise ValueError(f"order one of {LITE_ORDERS}, mutant one of {LITE_MUTANTS + ('early',)}")
    if R % 16 or rows % R or rows % 128 or k % 128 or n_wg < 1:
        raise ValueError("rows: a multiple of block_rows (a multiple of 16) and of 128; K: a multiple of 128")
    if release not in ("fence", "sc1"):
        raise ValueError("release must be fence or sc1")
    if stage not in ("none", "lds16"):
        raise ValueError("stage must be none or lds16")
    if stage == "lds16" and (order != "k" or k % 512 or (flag_tiles and flag_tiles % 4)):
        raise ValueError("stage lds16 needs order k, K a multiple of 4 tiles and flag_tiles a multiple of 4")
    nk, nq2 = k // 128, rows // 128
    if flag_tiles and (order != "k" or not scales or nk % flag_tiles):
        raise ValueError("flag mode needs order k, scales, and K tiles a multiple of flag_tiles")
    if mutant == "early" and not flag_tiles:
        raise ValueError("mutant early needs flag mode")
    sc1 = (flag_tiles > 0 and release == "sc1") or sc1_store
    if wait_prev and not flag_tiles:
        raise ValueError("wait_prev needs flag mode")
    row_bytes, nbd = k * 3 // 4, k // 32
    WT = pack_lite_block(stage)  # threads per work-group
    if stage == "lds16" and ((16 * 24) % WT or 256 % WT):  # whole 16-byte chunks / slots per thread
        raise ValueError("lds16: the work-group must split 384 loads and 256 slots evenly")
    T = n_wg * WT
    n_slots = rows * nk * 4
    n_units = nk * nq2 * 16
    sfx = (f"_g{n_wg}_{order}" + ("_nts" if nt_store else "") + ("" if scales else "_nosc")
           + (f"_w{waves_per_eu}" if waves_per_eu else "") + (f"_kf{flag_tiles}{release}" if flag_tiles else "")
           + ("_wp" if wait_prev else "") + ("_sc1" if sc1_store else "") + ("_lds16" if stage == "lds16" else "")
           + ("_ntl" if nt_load else "")
           + ("" if mutant == "none" else f"_mut{mutant}"))
    b = IRBuilder(f"rocke_mxfp6_pack_lite_{rows}x{k}_r{R}{sfx}")
    b.kernel.attrs["max_workgroup_size"] = WT
    if waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (waves_per_eu, waves_per_eu)
    csrc = b.param("codes_src", PtrType(I8, "global"))
    ssrc = b.param("scales_src", PtrType(I8, "global"))
    cdst = b.param("codes_dst", PtrType(I8, "global"))
    sdst = b.param("scales_dst", PtrType(I8, "global"))
    if flag_tiles:
        counters = b.param("counters", PtrType(I32, "global"))
        flags = b.param("flags", PtrType(I32, "global"))
        gdone = b.param("gdone", PtrType(I32, "global"))
        timeouts = b.param("timeouts", PtrType(I32, "global"))
        parity = b.param("parity", I32)
    c = b.const_i32
    gtid = b.add(b.mul(b.block_id_x(), c(WT)), b.thread_id_x())
    rs = b.buffer_rsrc(csrc, c(rows * row_bytes))
    hint = _hint(nt_store)
    if sc1:  # written through to the system coherence point (cpol sc0 sc1 [nt]); offsets in dwords as global_store_vN
        pol = "sc0 sc1" + (" nt" if nt_store else "")

        def store(ptr, dw, vec, n):
            addr = b.global_ptr_add(ptr, b.mul(dw, c(4)))
            op = {4: "global_store_dwordx4", 2: "global_store_dwordx2"}[n]
            b.inline_asm(f"{op} $0, $1, off {pol}", "v,v", [addr, vec], None)
    else:
        def store(ptr, dw, vec, n):
            b.global_store_vN(ptr, dw, vec, n, temporal_hint=hint)

    def code_slot(j):
        if order == "k":  # j = (s * (rows / R) + t) * 4R + o
            o, ts = _umod(b, j, 4 * R), _udiv(b, j, 4 * R)
            t, s = _umod(b, ts, rows // R), _udiv(b, ts, rows // R)
        else:  # j = (t * nk + s) * 4R + o
            o, ts = _umod(b, j, 4 * R), _udiv(b, j, 4 * R)
            s, t = _umod(b, ts, nk), _udiv(b, ts, nk)
        rb, ln = _udiv(b, o, 64), _umod(b, o, 64)
        kg, r16 = _udiv(b, ln, 16), _umod(b, ln, 16)
        if mutant == "kg":
            kg = b.xor(kg, c(1))
        row = b.add(b.add(b.mul(t, c(R)), b.mul(rb, c(16))), r16)
        src_off = b.add(b.mul(row, c(row_bytes)), b.add(b.mul(s, c(96)), b.mul(kg, c(24))))
        if nt_load:  # dword index of the 8-byte loads (src_off is a multiple of 8)
            q = [b.global_load_vN(csrc, b.add(_udiv(b, src_off, 4), c(2 * e)), I32, 2,
                                  temporal_hint=_hint(True)) for e in range(3)]
        else:
            q = [b.buffer_load_vN(rs, b.add(src_off, c(8 * e)), c(0), I32, 2) for e in range(3)]
        base = b.mul(b.add(b.mul(t, c(nk)), s), c(R * 24))  # dwords
        c0 = b.vec_pack([b.vec_extract(q[0], 0), b.vec_extract(q[0], 1), b.vec_extract(q[1], 0),
                         b.vec_extract(q[1], 1)], I32)
        store(cdst, b.add(base, b.mul(o, c(4))), c0, 4)
        store(cdst, b.add(base, b.add(c(16 * R), b.mul(o, c(2)))), q[2], 2)

    RG = rows // 16  # 16-row groups
    PWD = (4 * 96 + 16) // 4  # LDS row pitch (dwords) of a 16-row x 4-tile unit
    row_dw = k * 3 // 16
    smem = b.smem_alloc(I32, [16 * PWD], name_hint="pl16") if stage == "lds16" else None
    tid = b.thread_id_x()

    def code_unit(u):  # lds16: u = s4 * RG + rg; every thread of the work-group takes part (uniform u)
        rg, s4 = _umod(b, u, RG), _udiv(b, u, RG)
        row0, s0 = b.mul(rg, c(16)), b.mul(s4, c(4))
        for e in range(16 * 24 // WT):  # 384 16-byte chunks: row r, chunk cc of its 384 bytes
            ch = b.add(c(e * WT), tid)
            r, cc = _udiv(b, ch, 24), _umod(b, ch, 24)
            gi = b.add(b.add(b.mul(b.add(row0, r), c(row_dw)), b.mul(s0, c(24))), b.mul(cc, c(4)))
            v = b.global_load_vN(csrc, gi, I32, 4, temporal_hint=_hint(nt_load))
            b.smem_store_vN(smem, [b.add(b.mul(r, c(PWD)), b.mul(cc, c(4)))], v, 4)
        b.sync()
        for e in range(256 // WT):  # 256 slots (tile sl, kg, r16)
            sid = b.add(c(e * WT), tid)
            sl, ln = _udiv(b, sid, 64), _umod(b, sid, 64)
            kg, r16 = _udiv(b, ln, 16), _umod(b, ln, 16)
            kg_src = b.xor(kg, c(1)) if mutant == "kg" else kg
            li = b.add(b.add(b.mul(r16, c(PWD)), b.mul(sl, c(24))), b.mul(kg_src, c(6)))
            q = [b.smem_load_vN(smem, b.add(li, c(2 * e2)), dtype=I32, n=2) for e2 in range(3)]
            row = b.add(row0, r16)
            t_blk, rb = _udiv(b, row, R), _udiv(b, _umod(b, row, R), 16)
            o = b.add(b.add(b.mul(rb, c(64)), b.mul(kg, c(16))), r16)
            base = b.mul(b.add(b.mul(t_blk, c(nk)), b.add(s0, sl)), c(R * 24))  # dwords
            c0 = b.vec_pack([b.vec_extract(q[0], 0), b.vec_extract(q[0], 1), b.vec_extract(q[1], 0),
                             b.vec_extract(q[1], 1)], I32)
            store(cdst, b.add(base, b.mul(o, c(4))), c0, 4)
            store(cdst, b.add(base, b.add(c(16 * R), b.mul(o, c(2)))), q[2], 2)
        b.sync()  # the next unit's loads overwrite the stage

    def wg_stride(total, body, name, base=None):
        """``body(u)`` for u in ``base + range(total)``, one unit per work-group and step (uniform per group)."""
        steps, rem = divmod(total, n_wg)
        wg = b.block_id_x()
        off = wg if base is None else b.add(base, wg)
        if steps:
            with b.scf_for(c(0), c(steps), c(1), iv_name=name) as it:
                body(b.add(b.mul(it, c(n_wg)), off))
        if rem:
            with b.scf_if(b.cmp_lt(wg, c(rem))):
                body(b.add(c(steps * n_wg), off))

    def scale_unit(u):  # u = (s * nq2 + q2) * 16 + r16; tile dword ((s nq2 + q2) 64 + l) 2 + d
        r16, sq = _umod(b, u, 16), _udiv(b, u, 16)
        q2, s = _umod(b, sq, nq2), _udiv(b, sq, nq2)
        row0 = b.add(b.mul(q2, c(128)), r16)
        words = {}
        for d in range(2):  # raw dword (row, s) = blocks 4s..4s+3 = kg 0..3 of tile s
            raw = [b.global_load_i32(ssrc, b.add(b.mul(b.add(row0, c(64 * d + 16 * i)), c(nbd // 4)), s))
                   for i in range(4)]
            words[d] = _transpose4x4(b, raw)  # words[d][kg] byte i = scale[row 64d + 16i + r16, block 4s + kg]
        if mutant == "scale":
            words[0][1], words[0][2] = words[0][2], words[0][1]
        for kg in range(4):
            dw = b.mul(b.add(b.mul(sq, c(64)), b.add(c(16 * kg), r16)), c(2))
            store(sdst, dw, b.vec_pack([words[0][kg], words[1][kg]], I32), 2)

    def grid_stride(total, body, name, base=None):
        """``body(j)`` for j in ``base + range(total)`` (base: a runtime offset or 0) spread over the grid."""
        steps, rem = divmod(total, T)
        off = gtid if base is None else b.add(base, gtid)
        if steps:
            with b.scf_for(c(0), c(steps), c(1), iv_name=name) as it:
                body(b.add(b.mul(it, c(T)), off))
        if rem:
            j_loc = b.add(c(steps * T), gtid)
            with b.scf_if(b.cmp_lt(j_loc, c(total))):
                body(b.add(c(steps * T), off))

    def codes(total_slots, name, base_slots=None):
        if stage == "lds16":  # units of 16 rows x 4 tiles = 256 slots
            wg_stride(total_slots // 256, code_unit, name, None if base_slots is None else _udiv(b, base_slots, 256))
        else:
            grid_stride(total_slots, code_slot, name, base_slots)

    if not flag_tiles:
        codes(n_slots, "pl_it")
        if scales:
            grid_stride(n_units, scale_unit, "pl_su")
        return b.kernel
    G = flag_tiles
    per_codes, per_units = G * rows * 4, G * nq2 * 16
    tid0 = b.cmp_eq(b.thread_id_x(), c(0))
    if wait_prev:  # the previous call's GEMM published 1 - parity when its last CTA finished
        _, n_polls = _spin_until_asm(b, gdone, b.xor(parity, c(1)), c(FLAG_POLL_LIMIT))
        with b.scf_if(tid0):
            with b.scf_if(b.cmp_ge(n_polls, c(FLAG_POLL_LIMIT))):
                b.global_atomic_add(timeouts, c(0), c(1))
    with b.scf_for(c(0), c(nk // G), c(1), iv_name="pl_q") as qv:
        arrive_order = "first" if mutant == "early" else "last"

        def arrive():
            if mutant != "early":
                if release == "fence":
                    b.fence(scope="agent", ordering="release")  # L2 write-back + vmcnt(0)
                else:
                    b.s_waitcnt(vmcnt=0)  # write-through stores: complete = visible
            b.sync()
            with b.scf_if(tid0):
                old = b.global_atomic_add(counters, qv, c(1))  # system scope: one counter for every XCD
                with b.scf_if(b.cmp_eq(old, c(n_wg - 1))):
                    b.global_flag_store(counters, qv, c(0), scope="agent", ordering="monotonic")
                    b.global_flag_store(flags, qv, parity, scope="agent", ordering="monotonic")

        if arrive_order == "first":
            arrive()
        codes(per_codes, "pl_qc", b.mul(qv, c(per_codes)))
        grid_stride(per_units, scale_unit, "pl_qs", base=b.mul(qv, c(per_units)))
        if arrive_order == "last":
            arrive()
    return b.kernel


def pack_lite_flag_args(codes_src: int, scales_src: int, codes_dst: int, scales_dst: int, counters: int, flags: int,
                        parity: int, gdone: int = 0, timeouts: int = 0) -> bytes:
    """Kernel arguments of a flag-mode :func:`build_pack_lite` (eight pointers, the i32 parity)."""
    import struct

    return struct.pack("<QQQQQQQQi", codes_src, scales_src, codes_dst, scales_dst, counters, flags, gdone, timeouts,
                       parity)


def _spin_until_asm(b, addr, want, limit):
    """Poll the i32 at ``addr`` (uncached memory, ``s_load_dword ... glc``) until it equals ``want`` or ``limit``
    polls passed; -> (last value, polls). Fences memory: nothing after it is issued before the wait ended."""
    tpl = ("s_mov_b32 $1, 0\n"
           "${:private}spw${:uid}_top:\n"
           "s_load_dword $0, $2, 0 glc\n"
           "s_waitcnt lgkmcnt(0)\n"
           "s_cmp_eq_u32 $0, $3\n"
           "s_cbranch_scc1 ${:private}spw${:uid}_done\n"
           "s_sleep 2\n"
           "s_add_u32 $1, $1, 1\n"
           "s_cmp_lt_u32 $1, $4\n"
           "s_cbranch_scc1 ${:private}spw${:uid}_top\n"
           "${:private}spw${:uid}_done:")
    return b.inline_asm_multi(tpl, "=&s,=&s,s,s,s,~{scc},~{memory}", [addr, want, limit], result_types=[I32, I32],
                              sideeffect=True)


# ------------------------------------------------------------------------------------- fused K-flag GEMM

#: polls of an unready flag before a waiting wave gives up and reads anyway (a hang guard: a missing producer then
#: shows as a wrong result, never as a hung GPU); each poll is an s_sleep 1 plus an uncached scalar load (~1-2 us
#: under load: ~0.2-0.4 s per wait site, far above any legitimate wait for a running producer)
FLAG_POLL_LIMIT = 200_000
KFLAG_MUTANTS = ("none", "noflag", "parity")


def _flag_poll_asm(b, prefetched, addr, want, limit):
    """Spin until the i32 at ``addr`` (uniform global pointer; flags live in uncached memory, read with
    ``s_load_dword ... glc``: fresh across XCDs) equals ``want``. ``prefetched`` is a value of that word loaded
    earlier: if it already matches, the check costs a wait for it (``lgkmcnt(0)``: the LLVM path cannot see the
    load inside the asm; on the direct-compiler v6 build the tile ends with ``lgkmcnt(0)`` anyway), one compare and a
    branch. Fences memory (``~{memory}``): no load after it is issued before the flag was seen."""
    from rocke.core.ir import I32 as _I32

    tpl = ("s_waitcnt lgkmcnt(0)\n"
           "s_mov_b32 $1, 0\n"
           "s_cmp_eq_u32 $2, $4\n"
           "s_cbranch_scc1 ${:private}kfw${:uid}_done\n"
           "${:private}kfw${:uid}_top:\n"
           "s_sleep 1\n"
           "s_load_dword $0, $3, 0 glc\n"
           "s_waitcnt lgkmcnt(0)\n"
           "s_add_u32 $1, $1, 1\n"
           "s_cmp_eq_u32 $0, $4\n"
           "s_cbranch_scc1 ${:private}kfw${:uid}_done\n"
           "s_cmp_lt_u32 $1, $5\n"
           "s_cbranch_scc1 ${:private}kfw${:uid}_top\n"
           "${:private}kfw${:uid}_done:")
    return b.inline_asm_multi(tpl, "=&s,=&s,s,s,s,s,~{scc},~{memory}", [prefetched, addr, want, limit],
                              result_types=[_I32, _I32], sideeffect=True)


def _flag_load_asm(b, addr):
    """Issue an uncached scalar load of the flag word at ``addr`` (no wait: the value is used later)."""
    from rocke.core.ir import I32 as _I32

    return b.inline_asm("s_load_dword $0, $1, 0 glc", "=s,s", [addr], _I32, sideeffect=True)


def _kflag_builder_class():
    from . import gemm as GM

    class KFlagBuilder(GM._Builder):
        """v6's builder + per-K-chunk ready flags: before a K-loop body (U tiles) issues refills of K tiles up to
        ``i0 + U - 1 + ahead``, the flag of their chunk must equal ``parity``. The flag of the NEXT body's chunk is
        loaded one body ahead (issued right after the first tile's acquire, so the tile's LDS waits never wait
        for it) and checked at the next body's top; the prologue and the static tail wait for their chunks."""

        def __init__(self, spec, flag_tiles: int, mutant: str = "none", done_signal: bool = False):
            super().__init__(spec)
            self.kf_done = done_signal
            if spec.tail_mode == "clamp" or spec.peel_first or spec.probe != "none" or spec.carry:
                raise ValueError("the K-flag GEMM needs tail_mode static, no peel_first / carry / probe")
            if spec.skew_dtl or spec.skew_read or spec.reg_stages or spec.b_direct or self.line_ops:
                raise ValueError("the K-flag GEMM supports the stage ring with DTL refills only")
            if mutant not in KFLAG_MUTANTS:
                raise ValueError(f"mutant one of {KFLAG_MUTANTS}")
            self.kf_G, self.kf_mut = flag_tiles, mutant
            self.kf_n = -(-self.n_tiles // flag_tiles)
            self.kf_U = self.S * self.P // GM._gcd(self.S, self.P) * spec.unroll
            self.b = GM.IRBuilder(GM.kernel_name(spec) + f"_kf{flag_tiles}" + ("_sig" if done_signal else "")
                                  + ("" if mutant == "none" else f"_mut{mutant}"))
            self._kf_issue = None
            self._kf_new = None

        def build(self):
            b, spec = self.b, self.spec
            k = b.kernel
            k.attrs["max_workgroup_size"] = 64 * self.nw
            if spec.lds_alias_scopes:
                k.attrs["lds_alias_scopes"] = True
            if spec.acc_agpr:
                k.attrs["agpr_alloc"] = (GM.acc_regs(spec), GM.acc_regs(spec))
                k.attrs["mfma_agpr_form"] = True
            if self.nw > 4:
                k.attrs["waves_per_eu"] = self.nw // 4
            self.pa = b.param("a", PtrType(I8, "global"))
            self.pb = b.param("b", PtrType(I8, "global"))
            self.psa = b.param("sa", PtrType(I32, "global"))
            self.psb = b.param("sb", PtrType(I32, "global"))
            self.pc = b.param("c", PtrType(GM.BF16, "global"))
            self.pflags = b.param("flags", PtrType(I32, "global"))
            self.pdone_cnt = b.param("done_cnt", PtrType(I32, "global"))
            self.pdone_sig = b.param("done_sig", PtrType(I32, "global"))
            self.ptmo = b.param("timeouts", PtrType(I32, "global"))
            self.kf_parity = b.param("parity", I32)
            self.kf_want = (b.xor(self.kf_parity, self.c32(1)) if self.kf_mut == "parity" else self.kf_parity)
            self.kf_limit = self.c32(FLAG_POLL_LIMIT)
            self._setup()
            self._mainloop()
            self._epilogue()
            if self.kf_done:  # after every C store of the CTA: count it; the last CTA publishes the call parity
                b.s_waitcnt(vmcnt=0)
                b.s_barrier_bare()
                with b.scf_if(b.cmp_eq(b.thread_id_x(), self.c32(0))):
                    old = b.global_atomic_add(self.pdone_cnt, self.c32(0), self.c32(1))
                    nwg = (spec.m // spec.tile_m) * (spec.n // spec.tile_n)
                    with b.scf_if(b.cmp_eq(old, self.c32(nwg - 1))):
                        b.global_flag_store(self.pdone_cnt, self.c32(0), self.c32(0), scope="agent",
                                            ordering="monotonic")
                        # system-scope write-through store (sc0 sc1): the CP polls this HSA signal word
                        b.inline_asm("global_store_dword $0, $1, off sc0 sc1", "v,v", [self.pdone_sig,
                                                                                     self.kf_parity], None)
            b.ret()
            return k

        # chunk of K tile t (t a Value), clamped to the last chunk
        def _kf_q(self, t):
            return self.b.smin(self.b.div(t, self.c32(self.kf_G)), self.c32(self.kf_n - 1))

        def _kf_addr(self, q):
            return self.b.global_ptr_add(self.pflags, self.b.mul(q, self.c32(4)))

        def _kf_wait(self, q):  # blocking: load, then poll
            if self.kf_mut == "noflag":
                return
            addr = self._kf_addr(q)
            self._kf_check(_flag_load_asm(self.b, addr), addr)

        def _kf_check(self, value, addr):
            if self.kf_mut == "noflag":
                return
            _, polls = _flag_poll_asm(self.b, value, addr, self.kf_want, self.kf_limit)
            with self.b.scf_if(self.b.cmp_ge(polls, self.kf_limit)):  # the hang guard fired: count it
                self.b.global_atomic_add(self.ptmo, self.c32(0), self.c32(1))

        def _prologue(self):
            self._kf_wait(self.c32(min(self.kf_n - 1, (self.ahead - 1) // self.kf_G)))
            out = super()._prologue()
            if self.kf_mut != "noflag":  # the first loop body's chunk, checked at its top
                self._kf_pf = _flag_load_asm(self.b, self._kf_addr(self._kf_q(self.c32(self.kf_U - 1 + self.ahead))))
            else:
                self._kf_pf = self.c32(0)
            return out

        def _sc_lds_read(self, stage, X):
            if self._kf_issue is not None:  # right after the acquire of the body's first tile
                self._kf_new = _flag_load_asm(self.b, self._kf_issue)
                self._kf_issue = None
            return super()._sc_lds_read(stage, X)

        def _emit_loop(self, tag, trips, U, H, P, A, acc, fr, sc_cur, frag_keys, sc_keys, body_wait):
            b = self.b
            if H != 0 or U != self.kf_U:
                raise AssertionError("K-flag loop: static head / body length mismatch")
            iter_args = [(f"acc{tag}{i}", v) for i, v in enumerate(acc)]
            iter_args += [("fr" + tag + "_" + "_".join(str(x) for x in key), fr[key]) for key in frag_keys]
            iter_args += [(f"sc{tag}_{X}{q}", sc_cur[(X, q)]) for (X, q) in sc_keys]
            iter_args += [(f"kfv{tag}", self._kf_pf)]
            na, nf, ns = len(acc), len(frag_keys), len(sc_keys)
            loop = b.scf_for_iter(self.c32(0), trips, self.c32(1), iter_args, iv_name="kt" + tag,
                                  elide_trailing_barrier=False)
            with loop as (iv, vs):
                acc_l = list(vs[:na])
                fr_l = {key: vs[na + i] for i, key in enumerate(frag_keys)}
                sc_l = {key: vs[na + nf + i] for i, key in enumerate(sc_keys)}
                kfv = vs[na + nf + ns]
                i0 = b.add(b.mul(iv, self.c32(U)), self.c32(H))
                if self.kf_mut != "noflag":
                    need = self._kf_q(b.add(i0, self.c32(U - 1 + self.ahead)))
                    self._kf_check(kfv, self._kf_addr(need))
                    self._kf_issue = self._kf_addr(self._kf_q(b.add(i0, self.c32(2 * U - 1 + self.ahead))))
                    self._kf_new = None
                pend = None
                for p in range(U):
                    i_dyn = b.add(i0, self.c32(p)) if p else i0
                    sc_next = {}
                    acc_l, fr_l = self._tile(i_dyn, None, (H + p) % U, acc_l, fr_l, sc_l, sc_next, body_wait[p])
                    if self._kf_issue is not None:  # no scale read in the first tile: issue after it
                        self._kf_new = _flag_load_asm(b, self._kf_issue)
                        self._kf_issue = None
                    if sc_next:
                        pend = sc_next
                    if (H + p + 1) % P == 0:
                        if pend is None and not self.sc_lds:
                            raise AssertionError("scale period without a prefetch")
                        if pend is not None:
                            sc_l, pend = pend, None
                kf_next = self._kf_new if self.kf_mut != "noflag" else kfv
                b.scf_yield(*(acc_l + [fr_l[key] for key in frag_keys] + [sc_l[key] for key in sc_keys]
                              + [kf_next]))
            res = loop.results
            acc = list(res[:na])
            fr = {key: res[na + i] for i, key in enumerate(frag_keys)}
            sc_cur = {key: res[na + nf + i] for i, key in enumerate(sc_keys)}
            self._kf_wait(self.c32(self.kf_n - 1))  # the static tail refills up to the last tile
            return acc, fr, sc_cur

    return KFlagBuilder


def build_gemm_kflag(spec, flag_tiles: int = 16, *, mutant: str = "none", done_signal: bool = False) -> KernelDef:
    """The fused-overlap GEMM: ``spec``'s kernel with more parameters ``(flags, done_cnt, done_sig, timeouts: i32*,
    parity: i32)``; K-chunk ``q`` (``flag_tiles`` K tiles) of A's image / tile scales is read only after
    ``flags[q] == parity`` (a wave whose wait hits the poll limit proceeds and counts itself in ``timeouts[0]``). ``done_signal``: after its C stores every CTA counts itself in ``done_cnt[0]``; the last
    one resets it and writes ``done_sig[0] = parity`` (system scope: a CP ``hipStreamWaitValue32`` on another
    stream can start the next call's pack without an event). ``mutant`` (tests only): "noflag" never waits,
    "parity" waits for the previous call's parity."""
    from .gemm import validate_spec

    ok, why = validate_spec(spec)
    if not ok:
        raise ValueError(f"invalid Mxfp6GemmSpec: {why}")
    if flag_tiles < 1 or (spec.k // 128) % flag_tiles:
        raise ValueError("flag_tiles must divide the K tiles")
    return _kflag_builder_class()(spec, flag_tiles, mutant, done_signal).build()


def gemm_kflag_args(a: int, b_: int, sa: int, sb: int, c: int, flags: int, parity: int, done_cnt: int = 0,
                    done_sig: int = 0, timeouts: int = 0) -> bytes:
    """Kernel arguments of :func:`build_gemm_kflag` (done_cnt / done_sig are used by a ``done_signal`` build;
    ``timeouts[0]`` counts waves whose flag wait hit the poll limit)."""
    import struct

    return struct.pack("<QQQQQQQQQi", a, b_, sa, sb, c, flags, done_cnt, done_sig, timeouts, parity)


def schedule_kflag(*, mode: str = "strict", gemm_stream: str = "G", pack_stream: str = "P",
                   dep: str = "event", mutant: str = "none") -> Schedule:
    """Fused K-flag overlap. The GEMM's reads of the image are ordered by the in-kernel ready flags
    (``flag_guarded``), not by events, so the GEMM stream never waits on an event.

    ``mode`` "strict" (single-call semantics): per call the flag-mode pack (pack stream, after the previous call's
    GEMM: the call's input is ready only then, and it overwrites the image that GEMM read) and the K-flag GEMM
    (GEMM stream, right after the previous GEMM) start together; one image, one flag array. "next" (serving
    scenario): call ``i``'s GEMM reads slot ``i % 2`` while the pack of call ``i + 1`` fills the other slot (after
    the GEMM that last read it); one flag array per slot. Launch args: (slot, logical call offset).
    ``dep`` (strict): how the pack waits for the previous GEMM: "event" (record after the GEMM / stream wait) or
    "value" (the GEMM's done signal, ``build_gemm_kflag(done_signal=True)``, and a CP wait on its value: no packet
    between consecutive GEMMs of the GEMM stream). ``mutant`` "nowar": the pack does not wait for the GEMM that
    read its slot."""
    G, P = gemm_stream, pack_stream
    if (mode not in ("strict", "next", "gemmonly") or mutant not in ("none", "nowar")
            or dep not in ("event", "value", "spin")):
        raise ValueError("mode strict / next / gemmonly, dep event / value / spin, mutant none / nowar")
    if mode == "gemmonly":  # probe: the K-flag GEMM on the pre-packed image with every flag already set
        call = [_launch(G, "gemm", (0, 0), reads=("img_pre",), writes=("C",))]
        return Schedule([], call, (G,), "kflag-gemmonly", "single-call", {"slots": 1, "flags_preset": True})

    def slot_res(bb):
        return (("img", bb), ("sc", bb))

    if mode == "strict":
        rec, wait = {"event": (_rec, _wait),
                     "value": (lambda st, ev: StreamOp("signal", st, ev), lambda st, ev: StreamOp("waitvalue", st, ev)),
                     "spin": (lambda st, ev: StreamOp("signal", st, ev), lambda st, ev: StreamOp("kwait", st, ev)),
                     }[dep]
        setup = [rec(G, "gm")]
        call = ([wait(P, "gm")] if mutant != "nowar" else []) + [
            _launch(P, "pack", (0, 0), reads=("A_can",), writes=slot_res(0)),
            _launch(G, "gemm", (0, 0), reads=slot_res(0), writes=("C",)),
            rec(G, "gm")]
        return Schedule(setup, call, (G, P), "kflag" + ("" if dep == "event" else f"-{dep}"), "single-call",
                        {"flag_guarded": slot_res(0), "slots": 1, "dep": dep})
    # four calls per list: a slot's flag parity flips on every use, so the list repeats after two uses per slot
    setup = [_rec(G, "gm1"), _launch(P, "pack", (0, 0), reads=("A_can",), writes=slot_res(0))]
    call = []
    for u in range(4):
        bb = u % 2
        call += [_launch(G, "gemm", (bb, u), reads=slot_res(bb), writes=("C",)), _rec(G, f"gm{bb}")]
        call += ([_wait(P, f"gm{1 - bb}")] if mutant != "nowar" else []) + [
            _launch(P, "pack", (1 - bb, u + 1), reads=("A_can",), writes=slot_res(1 - bb))]
    return Schedule(setup, call, (G, P), "kflag-next", "next-call",
                    {"flag_guarded": slot_res(0) + slot_res(1), "slots": 2, "calls_per_list": 4})


def kflag_parity(call: int, slots: int) -> int:
    """Flag value a K-flag call waits for / its pack writes: flips on every use of a slot (flags start at 0)."""
    return 1 - (call // slots) % 2
