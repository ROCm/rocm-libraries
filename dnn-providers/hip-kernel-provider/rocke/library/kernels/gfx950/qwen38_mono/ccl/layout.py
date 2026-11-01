# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knobs, specs, error codes and the arena map of the in-kernel CCL.

Everything here is host data (no IR): the kernel builders in
:mod:`.collectives` bake the offsets in as constants, and the host side
(``rocke.runtime.peer``) owns the control words at the start of every arena.

Arena of one rank (every rank has the same map; peers write into it):

* ``[0, 256)`` control words (``rocke.runtime.peer.OFF_*``): epoch, abort,
  exit counter, error claim, launch count, slot base, grid-sync counter.
* ``[HDR_OFF, HDR_OFF + 2 * 8 * 32)`` launch-header pairs: slot (launch
  parity) x src rank x 32 bytes (4 tagged pairs).
* exchanges, each 16 KiB aligned: ``payload`` = slots x ranks x units x 1 KiB
  (a unit is one wave's 64 x 16-byte stores: 256 payload dwords in the
  ``line`` form, 128 tagged pairs in the ``pair`` form), then for the ``line``
  form the flags: one u32 per (slot, src, unit), with ``homing="dual"`` two
  copies 8 KiB apart inside 16 KiB blocks (the NUMA model of
  ``rocke.helpers.mailbox``: an 8 KiB block is homed at one IOD, the copy at
  +8 KiB at the other). A flag region never crosses a 2 MiB page, so one
  calibration bit per page names the IOD-local copy.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ARENA_PAGE",
    "AllGatherSpec",
    "AllReduceSpec",
    "ArgmaxSpec",
    "CclKnobs",
    "ERR",
    "Exchange",
    "HDR_OFF",
    "HDR_SRC_BYTES",
    "Layout",
    "MUTANTS",
    "OFF",
    "STAGES",
    "UNIT_BYTES",
    "make_layout",
]

UNIT_BYTES = 1024
LANES = 64
ARENA_PAGE = 2 << 20
NUMA_COPY_BYTES = 8192
NUMA_BLOCK = 16384
FLAGS_PER_BLOCK = NUMA_COPY_BYTES // 4  # 2048
HDR_OFF = 1024
HDR_SRC_BYTES = 32
HDR_SLOTS = 2
EXCH_ALIGN = 16384

# control words (mirror rocke.runtime.peer)
OFF = {"epoch": 0, "abort": 4, "exit": 8, "claim": 12, "launches": 16, "slotbase": 20, "gsync": 24}
# error codes (mirror rocke.runtime.peer.ERROR_NAMES)
ERR = {
    "RESIDENCY_TIMEOUT": -1,
    "GRID_SYNC_TIMEOUT": -2,
    "PEER_TIMEOUT": -3,
    "QSA_TIMEOUT": -4,
    "CCL_MISMATCH": -5,
    "PEER_ABORT": -6,
    "TAG_AHEAD": -7,
}
# ``stage`` word of an error record
STAGES = {"header": 9, "ar": 1, "rs": 2, "ag": 3, "allgather": 4, "argmax": 5, "gsync": 6, "hook": 99}

FORMS = ("pair", "line", "lamport", "pull")
ARS = ("oneshot", "twoshot")
HOMINGS = ("single", "dual")
PUSH_CPOLS = ("sc0sc1", "sc0sc1nt")
POLL_CPOLS = ("sc0sc1", "sc1", "none")
PAUSES = ("s_nop", "s_sleep1", "none", "s_sleep4", "s_sleep16")  # poll-spin pause (s_sleep n = 64 n clocks); the last two are CCL-LM levers
MUTANTS = ("none", "ge", "any", "slot", "revsum", "deadpush", "noreset", "nocanon", "gsearly")
SYNCS = ("none", "grid")
DTYPES = {"bf16": 2, "f32": 4}
# CCL-LM (large-message) knobs; their defaults reproduce the earlier builds byte for byte
UNIT_KIBS = (1, 2, 4)
AR_DTYPES = ("wire", "bf16")
PUBLISHES = ("unit_end", "counted", "comm_wave")
GSYNCS = ("word", "stripe")
PLACES = ("rr", "table")
GSYNC_STRIPES = 64  # 8 stripes per XCD (CTA c -> stripe c mod 64; xcc = c mod 8)
GSB_WORDS = GSYNC_STRIPES + 256  # stripes + the per-CTA release-check words of hooks builds
DIAG_FROM = 8  # diag stamps: collectives DIAG_FROM .. DIAG_FROM + DIAG_N - 1 of a launch
DIAG_N = 4
DIAG_PH = 8  # phases per collective (collectives.DIAG_PHASES)


@dataclass(frozen=True)
class CclKnobs:
    """Trace-time knobs of a CCL build; the kernel name names non-defaults.

    ``form``: ``line`` (1 KiB lines + flag, 0.996 payload on the link) or
    ``pair`` (tagged 8-byte pairs, the data is its own flag; 0.5 payload).
    ``ar``: one-shot (push to all, consumers sum) or two-shot (reduce-scatter
    to the block owner, owner sums and all-gathers). ``slots``: ring depth S
    (>= 2; the WAR argument in :mod:`.war`). ``homing``: ``dual`` stores each
    line flag twice on the receiver (both IODs) and polls the local copy.
    ``push_cpol``: ``sc0 sc1`` (system scope) or ``+ nt``. ``pause``: poll
    spin pause. ``header``: launch-header exchange (rows / cols / uses / hash)
    that turns a mismatched launch into ``CCL_MISMATCH``; on by default
    (DESIGN-CCL 4.10), off only in microbenchmarks (the name then carries
    ``nohdr``). ``hooks``: test build (drop a use, abort at a use, delay
    injection). ``mutant``: a must-fail mutant (tests only): ``ge`` accepts
    any tag >= the expected one (no tag-ahead check), ``any`` accepts any
    non-zero tag, ``slot`` reads the wrong payload slot, ``revsum`` sums in
    reverse rank order on rank 0, ``deadpush`` lets a wave that recorded an
    error keep publishing. ``sync``: ``grid`` adds a grid-wide barrier after
    every collective of the chain (a consumer that needs the whole vector,
    as a model layer does; implies the ``xbuf`` carry). ``carry``: how the all-reduce
    chain carries its next input (``auto``: registers when every wave owns
    at most one unit, ``mem``: the ``xbuf`` scratch). ``poll_cpol``: cache
    bits of the mailbox / payload loads (``sc0sc1`` system scope; ``sc1`` and
    ``none`` are levers and the coherence negative control: plain loads of a
    coarse-grained arena may read stale lines).

    CCL-LM levers (large messages; every default reproduces the earlier
    builds byte for byte):

    * ``form="lamport"``: payload-only lines, no flag and no drain before a
      flag: the receiver polls the payload itself. Empty = zero; the producer
      stores the bit complement of its (canonicalised) wire data, so the only
      wire pattern that would read as empty (bf16 ``0xFFFF`` / f32
      ``0xFFFFFFFF``, NaN patterns) is first mapped to the quiet NaN
      ``0x7FC0`` / ``0x7FC00000``. The consumer stores zeros over the lines it
      read and every wave drains its stores before its next push, so the slot
      is empty again before any peer can push into it (S >= 2 suffices;
      :mod:`.war` checks the reset edges). Must-fail mutants: ``noreset``
      (stale lines accepted), ``nocanon`` (a ``0xFFFF`` input never arrives).
    * ``unit_kib`` (1 / 2 / 4): a unit (one wave's push) is ``unit_kib`` KiB:
      each lane moves ``unit_kib`` 16-byte vectors per push (wider lines,
      fewer flags and units per wave); line and lamport forms.
    * ``ar_dtype="bf16"``: f32 data with bf16 partials on the wire (rounded
      to nearest even before the push, summed in f32; tolerance class, half
      the bytes).
    * ``publish``: ``unit_end`` (drain, then flag), ``counted`` (the next
      unit's input load is in flight while the stores drain: ``vmcnt(n)``
      instead of ``vmcnt(0)``), ``comm_wave`` (the waves stage their units in
      LDS, wave 0 of the CTA pushes them all, drains its own stores and
      publishes); line form, one-shot.
    * ``self_local``: lamport one-shot keeps its own partial in registers
      (pushes to the N - 1 peers only, aiter's LL arrangement).
    * ``gsync`` (``sync="grid"``): ``word`` (one system-scope arena counter)
      or ``stripe`` (64 agent-scope stripes in device memory, kernel argument
      ``gsb``; wave 0 polls all stripes lane-parallel, P4's ``s8``).
      Mutant ``gsearly`` releases one arrival early (hooks builds check the
      release and must catch it).
    * ``form="pull"`` (one-shot): every rank stores its partial into its OWN
      arena slot (system scope), drains (local), publishes the unit's flag to
      every rank; a consumer polls its local flags, then reads the N partials
      with system-scope loads through the peer bases (remote reads, aiter's
      large-message 1-stage scheme) and sums in rank order. A rank rewrites
      its slot only after every peer has published the next use, which
      follows that peer's reads (S = 2). Mutant ``slot``: the consumer reads
      the other slot (must fail). ``publish="counted"`` stages and publishes
      unit by unit (each unit's flag goes out as soon as ITS local stores are
      performed, the next unit's input load in flight), so peers start
      reading before the whole partial is staged.
    * ``diag``: per-CTA phase stamps of collectives ``DIAG_FROM ..`` (the
      stamps buffer grows to :func:`stamp_cols` columns); diagnostics only.
    * ``place``: unit -> wave placement. ``rr`` (round robin: unit ``u`` on
      global wave ``u mod (G W)``) or ``table`` (kernel argument ``utab``,
      i32[G W + 1]: wave ``w`` owns units ``utab[w] .. utab[w + 1] - 1``; the
      host weights the units per XCD so that the XCDs whose pushes reach the
      peer link last get fewer, :func:`xcd_table`). Memory-carried chains
      only; line / lamport, ``publish="unit_end"``.
    * ``ctas``: cap of the build's grid (``AllReduceSpec.grid``; 0 = the
      default cap of 256): fewer, longer-running pushers / pullers (every wave
      loops over more units). A launch-shape knob: the IR is the same, the
      name carries ``g<n>``."""

    form: str = "line"
    ar: str = "oneshot"
    slots: int = 2
    homing: str = "single"
    push_cpol: str = "sc0sc1"
    pause: str = "s_nop"
    header: bool = True
    hooks: bool = False
    mutant: str = "none"
    waves: int = 8
    carry: str = "auto"
    poll_cpol: str = "sc0sc1"
    sync: str = "none"
    unit_kib: int = 1
    ar_dtype: str = "wire"
    publish: str = "unit_end"
    self_local: bool = False
    gsync: str = "word"
    diag: bool = False
    place: str = "rr"
    ctas: int = 0

    @property
    def lm_path(self) -> bool:
        """The build uses the CCL-LM all-reduce emitters (any new data-path knob)."""
        return (self.form in ("lamport", "pull") or self.unit_kib != 1 or self.ar_dtype != "wire" or self.publish != "unit_end" or self.self_local
                or self.place != "rr")

    def __post_init__(self) -> None:
        if self.unit_kib not in UNIT_KIBS:
            raise ValueError(f"unit_kib must be one of {UNIT_KIBS}")
        if self.ar_dtype not in AR_DTYPES:
            raise ValueError(f"ar_dtype must be one of {AR_DTYPES}")
        if self.publish not in PUBLISHES:
            raise ValueError(f"publish must be one of {PUBLISHES}")
        if self.gsync not in GSYNCS:
            raise ValueError(f"gsync must be one of {GSYNCS}")
        if self.place not in PLACES:
            raise ValueError(f"place must be one of {PLACES}")
        if not 0 <= self.ctas <= 256:
            raise ValueError("ctas must be 0 (default cap) .. 256")
        if self.place != "rr" and (self.form == "pair" or self.publish != "unit_end"):
            raise ValueError("place='table': line / lamport / pull forms with publish='unit_end'")
        if self.gsync != "word" and self.sync != "grid":
            raise ValueError("gsync applies to sync='grid'")
        if self.form == "pair" and (self.unit_kib != 1 or self.ar_dtype != "wire"):
            raise ValueError("unit_kib / ar_dtype apply to the line and lamport forms")
        if self.publish != "unit_end" and (self.form not in ("line", "pull") or self.ar != "oneshot" or self.homing != "single"
                                           or (self.form == "pull" and self.publish != "counted")):
            raise ValueError("publish='counted' / 'comm_wave' are line one-shot (single homing) variants; pull takes 'counted'")
        if self.self_local and (self.form != "lamport" or self.ar != "oneshot"):
            raise ValueError("self_local applies to the lamport one-shot")
        if self.form == "pull" and (self.ar != "oneshot" or self.homing != "single" or self.self_local
                                    or self.mutant not in ("none", "slot", "revsum")):
            raise ValueError("pull: one-shot, single homing, publish 'unit_end' / 'counted'; mutants slot / revsum")
        if self.form == "lamport":
            if self.homing != "single" or self.poll_cpol != "sc0sc1":
                raise ValueError("lamport: homing='single', poll_cpol='sc0sc1' (no flags; payload polls need system scope)")
            if self.mutant in ("ge", "any", "slot", "deadpush"):
                raise ValueError(f"mutant {self.mutant!r} is a tagged-form mutant")
        elif self.mutant in ("noreset", "nocanon"):
            raise ValueError(f"mutant {self.mutant!r} is a lamport mutant")
        if self.mutant == "gsearly" and self.gsync != "stripe":
            raise ValueError("mutant 'gsearly' is a gsync='stripe' mutant")
        if self.lm_path and self.mutant in ("ge", "any", "slot", "deadpush") and self.publish == "unit_end" and self.form == "line":
            raise ValueError("tagged-form mutants run on the line form's default path (unit_kib=1)")
        if self.form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}")
        if self.ar not in ARS:
            raise ValueError(f"ar must be one of {ARS}")
        if not 2 <= self.slots <= 8:
            raise ValueError("slots must be 2..8")
        if self.homing not in HOMINGS:
            raise ValueError(f"homing must be one of {HOMINGS}")
        if self.homing == "dual" and self.form != "line":
            raise ValueError("homing='dual' applies to line flags (form='line')")
        if self.push_cpol not in PUSH_CPOLS:
            raise ValueError(f"push_cpol must be one of {PUSH_CPOLS}")
        if self.pause not in PAUSES:
            raise ValueError(f"pause must be one of {PAUSES}")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant must be one of {MUTANTS}")
        if self.waves not in (1, 2, 4, 8, 16):
            raise ValueError("waves must be 1, 2, 4, 8 or 16")
        if self.poll_cpol not in POLL_CPOLS:
            raise ValueError(f"poll_cpol must be one of {POLL_CPOLS}")
        if self.carry not in ("auto", "mem"):
            raise ValueError("carry must be 'auto' or 'mem'")
        if self.sync not in SYNCS:
            raise ValueError(f"sync must be one of {SYNCS}")

    def suffix(self) -> str:
        d = CclKnobs()
        parts = []
        if self.form != d.form:
            parts.append(self.form)
        if self.ar != d.ar:
            parts.append("ts")
        if self.slots != d.slots:
            parts.append(f"s{self.slots}")
        if self.homing != d.homing:
            parts.append("hd")
        if self.push_cpol != d.push_cpol:
            parts.append("nt")
        if self.pause != d.pause:
            parts.append(f"p{self.pause}")
        if self.header != d.header:
            parts.append("hdr" if self.header else "nohdr")
        if self.hooks:
            parts.append("hk")
        if self.mutant != "none":
            parts.append(f"mut{self.mutant}")
        if self.waves != d.waves:
            parts.append(f"w{self.waves}")
        if self.carry != d.carry:
            parts.append(f"c{self.carry}")
        if self.poll_cpol != d.poll_cpol:
            parts.append(f"pc{self.poll_cpol}")
        if self.sync != d.sync:
            parts.append(f"sy{self.sync}")
        if self.unit_kib != d.unit_kib:
            parts.append(f"u{self.unit_kib}")
        if self.ar_dtype != d.ar_dtype:
            parts.append(f"w{self.ar_dtype}")
        if self.publish != d.publish:
            parts.append({"counted": "pubcnt", "comm_wave": "pubcw"}[self.publish])
        if self.self_local:
            parts.append("selfl")
        if self.gsync != d.gsync:
            parts.append(f"gs{self.gsync}")
        if self.diag:
            parts.append("diag")
        if self.place != d.place:
            parts.append(f"p{self.place}")
        if self.ctas != d.ctas:
            parts.append(f"g{self.ctas}")
        return "_".join(parts)


def stamp_cols(knobs: "CclKnobs") -> int:
    """i64 columns per CTA of a build's ``stamps`` buffer."""
    return 3 + (DIAG_N * DIAG_PH if knobs.diag else 0)


def _check_shape(ranks: int, rows: int, cols: int, dtype: str) -> None:
    if not 1 <= ranks <= 8:
        raise ValueError("ranks must be 1..8")
    if rows < 1 or cols < 1:
        raise ValueError("rows, cols >= 1")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {tuple(DTYPES)}")


@dataclass(frozen=True)
class AllReduceSpec:
    """All-reduce (sum) of ``rows x cols`` ``dtype`` partials over ``ranks``
    (the wire dtype; sums are f32 in rank order). The kernel runs ``iters``
    chained collectives per launch (runtime): the input of collective i + 1
    on rank r is ``wire(out_i * scale_r)``."""

    ranks: int
    rows: int
    cols: int
    dtype: str = "bf16"
    knobs: CclKnobs = field(default_factory=CclKnobs)

    def __post_init__(self) -> None:
        _check_shape(self.ranks, self.rows, self.cols, self.dtype)
        if self.knobs.ar_dtype == "bf16" and self.dtype != "f32":
            raise ValueError("ar_dtype='bf16' narrows f32 data (dtype='f32')")
        if self.payload_dwords % self.unit_dwords:
            raise ValueError(
                f"rows x cols x {self.wire_dtype} = {self.payload_dwords} dwords, not a multiple of the "
                f"{self.unit_dwords}-dword unit of form {self.knobs.form!r}"
            )
        if self.knobs.publish == "comm_wave" and (self.units > self.grid() * self.knobs.waves or self.ranks * self.knobs.waves > 64):
            raise ValueError("publish='comm_wave': one unit per wave at grid() CTAs and ranks x waves <= 64 flag lanes")

    @property
    def esize(self) -> int:
        return DTYPES[self.dtype]

    @property
    def wire_dtype(self) -> str:
        """Dtype of the partials on the wire (``ar_dtype``; ``wire`` = the data dtype)."""
        return self.dtype if self.knobs.ar_dtype == "wire" else self.knobs.ar_dtype

    @property
    def payload_dwords(self) -> int:
        ws = DTYPES[self.wire_dtype]
        return self.rows * self.cols * ws // 4 if (self.rows * self.cols * ws) % 4 == 0 else -1

    @property
    def unit_dwords(self) -> int:
        return 256 * self.knobs.unit_kib if self.knobs.form != "pair" else 128

    @property
    def in_dwords(self) -> int:
        """Input / ``xbuf`` dwords of the chain (data dtype)."""
        return self.rows * self.cols * self.esize // 4

    @property
    def units(self) -> int:
        return self.payload_dwords // self.unit_dwords

    @property
    def lane_dwords(self) -> int:
        return self.unit_dwords // LANES

    def grid(self, max_ctas: int = 256) -> int:
        cap = min(max_ctas, self.knobs.ctas) if self.knobs.ctas else max_ctas
        return max(1, min(cap, -(-self.units // self.knobs.waves)))

    @property
    def threads(self) -> int:
        return 64 * self.knobs.waves

    @property
    def reg_carry(self) -> bool:
        """The chain carries the next input in registers: every wave owns at
        most one unit at ``grid()`` CTAs (``carry="auto"``, no grid sync)."""
        return (self.knobs.carry == "auto" and self.knobs.sync == "none" and self.units <= self.grid() * self.knobs.waves and self.knobs.publish != "comm_wave"
                and self.knobs.place == "rr")

    def name(self) -> str:
        parts = ["q38ccl_ar", f"n{self.ranks}", f"r{self.rows}", f"c{self.cols}", self.dtype]
        s = self.knobs.suffix()
        if s:
            parts.append(s)
        return "_".join(parts)

    def header_hash(self) -> int:
        """Schedule hash carried by the launch header (rank-independent)."""
        return zlib.crc32(self.name().encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class AllGatherSpec:
    """All-gather of each rank's ``rows x cols`` slice into ``rows x (ranks
    cols)`` (wire dtype, a copy). Chain: the input of collective i + 1 on rank
    r is the gathered slice of rank ``(r + 1) mod ranks``."""

    ranks: int
    rows: int
    cols: int
    dtype: str = "bf16"
    knobs: CclKnobs = field(default_factory=CclKnobs)

    def __post_init__(self) -> None:
        _check_shape(self.ranks, self.rows, self.cols, self.dtype)
        if self.knobs.ar != "oneshot":
            raise ValueError("all-gather is one-shot")
        if self.knobs.sync != "none":
            raise ValueError("sync='grid' is an all-reduce chain knob")
        if self.knobs.lm_path or self.knobs.diag or self.knobs.ctas:
            raise ValueError("the CCL-LM knobs (lamport, unit_kib, ar_dtype, publish, self_local, diag, ctas) are all-reduce knobs")
        if (self.cols * self.esize) % (4 * self.lane_dwords):
            raise ValueError("a row must hold whole lane chunks (cols x esize multiple of 16 / 8 bytes)")
        if self.payload_dwords % self.unit_dwords:
            raise ValueError("rows x cols not a whole number of units")

    esize = AllReduceSpec.esize
    unit_dwords = AllReduceSpec.unit_dwords
    lane_dwords = AllReduceSpec.lane_dwords
    grid = AllReduceSpec.grid
    threads = AllReduceSpec.threads

    @property
    def payload_dwords(self) -> int:
        return self.rows * self.cols * self.esize // 4

    @property
    def units(self) -> int:
        return self.payload_dwords // self.unit_dwords

    @property
    def row_dwords(self) -> int:
        return self.cols * self.esize // 4

    def name(self) -> str:
        parts = ["q38ccl_ag", f"n{self.ranks}", f"r{self.rows}", f"c{self.cols}", self.dtype]
        s = self.knobs.suffix()
        if s:
            parts.append(s)
        return "_".join(parts)

    header_hash = AllReduceSpec.header_hash


@dataclass(frozen=True)
class ArgmaxSpec:
    """Vocab-parallel argmax merge: every rank holds per-row ``(max value,
    global id)``; the all-gather of tagged pairs merges in rank order (larger
    value wins, equal values take the lower global id). ``rows <= 64 * waves``."""

    ranks: int
    rows: int
    knobs: CclKnobs = field(default_factory=lambda: CclKnobs(form="pair", waves=2))

    def __post_init__(self) -> None:
        _check_shape(self.ranks, self.rows, 1, "f32")
        if self.knobs.form != "pair" or self.knobs.ar != "oneshot":
            raise ValueError("argmax merge uses the one-shot pair form")
        if self.knobs.sync != "none":
            raise ValueError("sync='grid' is an all-reduce chain knob")
        if self.knobs.lm_path or self.knobs.diag or self.knobs.ctas:
            raise ValueError("the CCL-LM knobs are all-reduce knobs")
        if self.rows > 64 * self.knobs.waves:
            raise ValueError("rows > 64 x waves")

    @property
    def units(self) -> int:  # one unit = 64 rows (lane = row), 1 KiB per (slot, src)
        return -(-self.rows // 64)

    @property
    def threads(self) -> int:
        return 64 * self.knobs.waves

    def name(self) -> str:
        parts = ["q38ccl_amax", f"n{self.ranks}", f"r{self.rows}"]
        s = self.knobs.suffix()
        if s and s != "pair_w2":
            parts.append(s)
        return "_".join(parts)

    header_hash = AllReduceSpec.header_hash


# -------------------------------------------------------------- arena map


@dataclass(frozen=True)
class Exchange:
    """One exchange region: ``srcs`` sources (ranks, or 1 for the two-shot
    all-gather whose single source is the block owner) x ``slots`` x
    ``units`` lines/pair-units of 1 KiB, and (``flags``) one u32 flag per
    (slot, src, unit), ``copies`` 1 or 2."""

    name: str
    payload: int
    slots: int
    srcs: int
    units: int
    flags: int = -1  # byte offset of the flag region, -1 = none
    copies: int = 1
    unit_bytes: int = UNIT_BYTES  # CCL-LM ``unit_kib`` x 1 KiB

    def payload_off(self, slot: int, src: int, unit: int) -> int:
        return self.payload + ((slot * self.srcs + src) * self.units + unit) * self.unit_bytes

    def flag_index(self, slot: int, src: int, unit: int) -> int:
        return (slot * self.srcs + src) * self.units + unit

    def flag_off(self, slot: int, src: int, unit: int, copy: int = 0) -> int:
        if self.flags < 0:
            raise ValueError(f"exchange {self.name} has no flags")
        i = self.flag_index(slot, src, unit)
        if self.copies == 1:
            return self.flags + 4 * i
        return self.flags + (i // FLAGS_PER_BLOCK) * NUMA_BLOCK + (i % FLAGS_PER_BLOCK) * 4 + copy * NUMA_COPY_BYTES

    @property
    def payload_bytes(self) -> int:
        return self.slots * self.srcs * self.units * self.unit_bytes

    @property
    def flag_count(self) -> int:
        return self.slots * self.srcs * self.units if self.flags >= 0 else 0

    @property
    def flag_bytes(self) -> int:
        n = self.flag_count
        if n == 0:
            return 0
        if self.copies == 1:
            return 4 * n
        return -(-n // FLAGS_PER_BLOCK) * NUMA_BLOCK

    @property
    def end(self) -> int:
        return max(self.payload + self.payload_bytes, self.flags + self.flag_bytes if self.flags >= 0 else 0)


@dataclass(frozen=True)
class Layout:
    exchanges: Tuple[Exchange, ...]
    nbytes: int

    def ex(self, name: str) -> Exchange:
        for e in self.exchanges:
            if e.name == name:
                return e
        raise KeyError(name)

    def flag_page(self, name: str) -> int:
        e = self.ex(name)
        return e.flags // ARENA_PAGE

    def validate(self) -> None:
        # [2048, 10240): the counted publish's dummy flag words (collectives.JUNK_BASE, 32 wave slots x 64 lanes)
        spans: List[Tuple[int, int, str]] = [(0, 256, "ctrl"), (HDR_OFF, HDR_OFF + HDR_SLOTS * 8 * HDR_SRC_BYTES, "hdr"), (2048, 2048 + 32 * 64 * 4, "junk")]
        for e in self.exchanges:
            if e.payload % EXCH_ALIGN:
                raise ValueError(f"{e.name}: payload not {EXCH_ALIGN} aligned")
            spans.append((e.payload, e.payload + e.payload_bytes, e.name + ".payload"))
            if e.flags >= 0:
                if e.flags % (NUMA_BLOCK if e.copies == 2 else 256):
                    raise ValueError(f"{e.name}: flags misaligned")
                lo, hi = e.flags, e.flags + e.flag_bytes
                if lo // ARENA_PAGE != (hi - 1) // ARENA_PAGE:
                    raise ValueError(f"{e.name}: flag region crosses a 2 MiB page")
                spans.append((lo, hi, e.name + ".flags"))
        spans.sort()
        for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
            if a1 > b0:
                raise ValueError(f"arena regions overlap: {an} [{a0},{a1}) and {bn} [{b0},{b1})")
        if spans[-1][1] > self.nbytes:
            raise ValueError("layout exceeds the arena")
        if self.nbytes >= 1 << 31:
            raise ValueError("arena offsets must fit in i32")


def _align(x: int, a: int) -> int:
    return -(-x // a) * a


def _place_flags(cur: int, nbytes_flags: int, copies: int) -> int:
    """First offset >= cur, aligned, whose flag span stays within one page."""
    a = NUMA_BLOCK if copies == 2 else 256
    off = _align(cur, a)
    if nbytes_flags > ARENA_PAGE:
        raise ValueError("flag region larger than a 2 MiB page")
    if off // ARENA_PAGE != (off + nbytes_flags - 1) // ARENA_PAGE:
        off = _align(off, ARENA_PAGE)
    return off


def make_layout(specs, *, min_bytes: int = 0) -> Layout:
    """Arena map holding the exchanges of every spec in ``specs`` (one
    exchange set per spec, packed one after another; flags first so that
    their pages stay small). Names: ``<i>.ar`` / ``<i>.rs`` + ``<i>.ag`` /
    ``<i>.gather`` / ``<i>.amax`` for spec index ``i``."""
    raw: List[Tuple[str, int, int, int, bool, int]] = []  # name, slots, srcs, units, flags, copies
    ubytes: Dict[str, int] = {}
    for i, sp in enumerate(specs):
        kn = sp.knobs
        copies = 2 if kn.homing == "dual" else 1
        line = kn.form in ("line", "pull")
        if isinstance(sp, AllReduceSpec):
            if kn.ar == "oneshot":
                raw.append((f"{i}.ar", kn.slots, sp.ranks, sp.units, line, copies))
            else:
                raw.append((f"{i}.rs", kn.slots, sp.ranks, sp.units, line, copies))
                raw.append((f"{i}.ag", kn.slots, 1, sp.units, line, copies))
            if kn.unit_kib != 1:
                for nm in ((f"{i}.ar",) if kn.ar == "oneshot" else (f"{i}.rs", f"{i}.ag")):
                    ubytes[nm] = UNIT_BYTES * kn.unit_kib
        elif isinstance(sp, AllGatherSpec):
            raw.append((f"{i}.gather", kn.slots, sp.ranks, sp.units, line, copies))
        elif isinstance(sp, ArgmaxSpec):
            raw.append((f"{i}.amax", kn.slots, sp.ranks, sp.units, False, 1))
        else:
            raise TypeError(f"unknown spec {type(sp).__name__}")
    cur = _align(HDR_OFF + HDR_SLOTS * 8 * HDR_SRC_BYTES, EXCH_ALIGN)
    flag_off: Dict[str, int] = {}
    for name, slots, srcs, units, line, copies in raw:
        if not line:
            continue
        n = slots * srcs * units
        nb = 4 * n if copies == 1 else -(-n // FLAGS_PER_BLOCK) * NUMA_BLOCK
        off = _place_flags(cur, nb, copies)
        flag_off[name] = off
        cur = off + nb
    cur = _align(cur, EXCH_ALIGN)
    exs = []
    for name, slots, srcs, units, line, copies in raw:
        e = Exchange(name, cur, slots, srcs, units, flag_off.get(name, -1), copies, ubytes.get(name, UNIT_BYTES))
        exs.append(e)
        cur = _align(cur + e.payload_bytes, EXCH_ALIGN)
    nbytes = max(_align(cur, ARENA_PAGE), _align(min_bytes, ARENA_PAGE))
    lay = Layout(tuple(exs), nbytes)
    lay.validate()
    return lay


def with_knobs(spec, **kw):
    return replace(spec, knobs=replace(spec.knobs, **kw))


# ------------------------------------------------------- measured defaults

# ``form`` / ``ar`` / ``homing`` choice of the all-reduce from the measured
# crossover (gfx950, uncached arena, TP2 / TP3 / TP4, per-rank payload;
# ``platform/python/rocke/examples/gfx950/ccl/ccl_case_study.md``). Two shapes:
#
# ``shape="layer"`` (default): the chain with a grid-wide barrier after every
# collective (``sync="grid"`` arms: a consumer that needs the whole vector, as
# a model layer does):
#   TP2: pair one-shot up to 80 KiB, line one-shot up to 320 KiB, line
#        two-shot above;
#   TP3: pair one-shot up to 80 KiB, pair two-shot up to 160 KiB, line two-shot above;
#   TP4 (and the unmeasured TP5-8): pair one-shot up to 20 KiB, pair two-shot
#        up to 160 KiB, line two-shot above.
# ``shape="chain"``: the persistent chain in which a wave waits only for its
# own unit (the best case; the earlier crossover):
#   TP2: pair one-shot up to 80 KiB, line one-shot above (two-shot never wins);
#   TP3: pair one-shot up to 80 KiB, line one-shot up to 160 KiB, line two-shot above;
#   TP4+: pair two-shot up to 160 KiB, line two-shot above.
# Dual flag homing (``homing="dual"``, needs the per-page IOD calibration of
# the receive arena) won for line forms up to 160 KiB in the chain and lost
# above; it was not measured under the grid sync.
AUTO_PAIR_MAX_BYTES = 80 << 10
AUTO_TP3_LINE_OS_MAX_BYTES = 160 << 10
AUTO_TS_PAIR_MAX_BYTES = 160 << 10
AUTO_DUAL_MAX_BYTES = 160 << 10
LAYER_TP2_LINE_OS_MAX_BYTES = 320 << 10
LAYER_TP4_PAIR_OS_MAX_BYTES = 20 << 10
SHAPES = ("layer", "chain")


def _select_chain(ranks: int, payload_bytes: int):
    if ranks <= 2 or (ranks == 3 and payload_bytes <= AUTO_TP3_LINE_OS_MAX_BYTES):
        return ("pair" if payload_bytes <= AUTO_PAIR_MAX_BYTES else "line"), "oneshot"
    return ("pair" if ranks > 3 and payload_bytes <= AUTO_TS_PAIR_MAX_BYTES else "line"), "twoshot"


def _select_layer(ranks: int, payload_bytes: int):
    if ranks <= 2:
        if payload_bytes <= AUTO_PAIR_MAX_BYTES:
            return "pair", "oneshot"
        return "line", ("oneshot" if payload_bytes <= LAYER_TP2_LINE_OS_MAX_BYTES else "twoshot")
    pair_os_max = AUTO_PAIR_MAX_BYTES if ranks == 3 else LAYER_TP4_PAIR_OS_MAX_BYTES
    if payload_bytes <= pair_os_max:
        return "pair", "oneshot"
    if payload_bytes <= AUTO_TS_PAIR_MAX_BYTES:
        return "pair", "twoshot"
    return "line", "twoshot"


def select_knobs(ranks: int, payload_bytes: int, *, calibrated: bool = False, shape: str = "layer", **kw) -> CclKnobs:
    """The measured best ``form`` / ``ar`` (and ``homing`` when the arena's
    IOD homes are calibrated) for an all-reduce of ``payload_bytes`` per
    rank over ``ranks``, for a consumer of the ``shape`` above; ``kw`` sets
    other knobs."""
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    form, ar = (_select_layer if shape == "layer" else _select_chain)(ranks, payload_bytes)
    homing = "dual" if calibrated and form == "line" and payload_bytes <= AUTO_DUAL_MAX_BYTES else "single"
    kw.setdefault("homing", homing)
    return CclKnobs(form=form, ar=ar, **kw)


def auto_allreduce_spec(ranks: int, rows: int, cols: int, dtype: str = "bf16", *, calibrated: bool = False, shape: str = "layer", lm: bool = False,
                        **kw) -> AllReduceSpec:
    """The measured best all-reduce build; ``lm=True`` uses the CCL-LM
    selection (:func:`lm_select`) where it was measured, ``select_knobs``
    elsewhere."""
    nbytes = rows * cols * DTYPES[dtype]
    kn = lm_select(ranks, nbytes, dtype, **kw) if lm else select_knobs(ranks, nbytes, calibrated=calibrated, shape=shape, **kw)
    return AllReduceSpec(ranks, rows, cols, dtype, kn)


# CCL-LM selection: the layer-shaped (grid-synced) all-reduce at TP2, measured against the aiter custom all-reduce and
# W3's forms (``experiments/qwen38-true-mono/large-batch/CCL-LM.md``; per-rank payload bytes, data dtype):
#   80 KiB: lamport one-shot with 4 waves per CTA (fix round; 1 KiB units);
#   .. 320 KiB: lamport one-shot (bf16: 1 KiB units, f32: 2 KiB units);
#   .. 640 KiB: bf16 pull with 2 KiB units on 32 CTAs, f32 lamport with 2 KiB units;
#   .. 1.25 MiB: bf16 pull with 2 KiB units on 64 CTAs, f32 pull with 4 KiB units on 32 CTAs and the counted publish
#   (fix round);
#   above (measured up to 5 MiB): pull with 4 KiB units on 64 CTAs and the one-word grid sync (fix round; was 48 CTAs
#   with the striped sync); every other band of the layer shape uses the striped grid sync.
# Smaller payloads and other rank counts keep ``select_knobs`` (not re-measured by CCL-LM).
LM_MIN_BYTES = 80 << 10


def lm_select(ranks: int, payload_bytes: int, dtype: str = "bf16", *, layer: bool = False, **kw) -> CclKnobs:
    """CCL-LM ``form`` / ``unit_kib`` / ``ctas`` / ``waves`` for an all-reduce
    of ``payload_bytes`` per rank (``kw`` sets other knobs). ``layer=True``:
    the layer shape (``sync="grid"``) with the measured grid sync of the band
    (``gsync``)."""
    if ranks != 2 or payload_bytes < LM_MIN_BYTES:
        if layer:
            kw = {"sync": "grid", "gsync": "stripe", **kw}
        return select_knobs(ranks, payload_bytes, shape="layer", **kw)
    f32 = dtype == "f32"
    gsync = "stripe"
    if payload_bytes <= LM_MIN_BYTES:
        sel = dict(form="lamport", unit_kib=2 if f32 else 1, waves=4)
    elif payload_bytes <= 320 << 10:
        sel = dict(form="lamport", unit_kib=2 if f32 else 1)
    elif payload_bytes <= 640 << 10:
        sel = dict(form="lamport", unit_kib=2) if f32 else dict(form="pull", unit_kib=2, ctas=32)
    elif payload_bytes <= 1280 << 10:
        sel = dict(form="pull", unit_kib=4, ctas=32, publish="counted") if f32 else dict(form="pull", unit_kib=2, ctas=64)
    else:
        sel = dict(form="pull", unit_kib=4, ctas=64)
        gsync = "word"
    if layer:
        sel.update(sync="grid", gsync=gsync)
    sel.update(kw)
    return CclKnobs(**sel)


def xcd_table(units: int, nctas: int, waves: int, shares=None, xcds: int = 8) -> List[int]:
    """``place="table"`` placement: ``utab`` (i32[nctas x waves + 1], wave
    ``w`` owns units ``utab[w] .. utab[w + 1] - 1``). CTA ``c`` runs on XCD
    ``c mod xcds`` (one CTA per CU, round-robin dispatch). XCD ``x`` gets
    ``units x shares[x]`` units (rounded, the sum exact), spread evenly over
    its waves in wave order; ``shares=None`` = equal shares (per XCD, so
    with ``nctas`` not a multiple of ``xcds`` an XCD with fewer CTAs gives
    each wave more units). XCDs without a CTA (``nctas < xcds``) get none;
    the shares of the others are renormalised. Every unit is owned by
    exactly one wave."""
    if nctas < 1 or waves < 1:
        raise ValueError("nctas, waves >= 1")
    sh = np.full(xcds, 1.0 / xcds) if shares is None else np.asarray(shares, dtype=np.float64)
    if sh.shape != (xcds,) or (sh < 0).any() or sh.sum() <= 0:
        raise ValueError("shares: one non-negative weight per XCD")
    present = np.array([x < nctas for x in range(xcds)])
    sh = np.where(present, sh, 0.0)
    if sh.sum() <= 0:
        sh = present.astype(np.float64)
    sh = sh / sh.sum()
    raw = units * sh
    cnt = np.floor(raw).astype(np.int64)
    for x in np.argsort(-(raw - cnt))[: units - int(cnt.sum())]:
        cnt[x] += 1
    per_wave = np.zeros(nctas * waves, dtype=np.int64)
    for x in range(xcds):
        ws = [c * waves + w for c in range(x, nctas, xcds) for w in range(waves)]
        if not ws:
            continue  # no CTA on this XCD: its share is 0
        q, r = divmod(int(cnt[x]), len(ws))
        for i, g in enumerate(ws):
            per_wave[g] = q + (1 if i < r else 0)
    tab = [0]
    for n in per_wave:
        tab.append(tab[-1] + int(n))
    assert tab[-1] == units
    return tab
