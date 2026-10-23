# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LDS planning for the attention backward inner body (host side only).

Nothing in this module emits IR. It answers three questions for the tile step:

1. **Where does every LDS buffer live?** :func:`plan_bwd_lds` places the
   buffers of one CTA inside a single ``smem_alloc``. Buffers that live for the
   whole kernel (LDS-resident K/V, K^T, LDS accumulators) sit at the bottom.
   The one-time stage-0 staging buffers (K, K^T, then V) and the per-q-step
   ring (Q, dO, Q^T, dO^T, and lse2 / Dsum under a ring, times
   ``ring_depth``) plus dS / P^T,
   and the epilogue's dK / dV transpose buffer, are *phase aliased*: they are
   never live at the same time, so they share the region above the resident
   buffers. The footprint is ``resident + max(stage-0 K phase, stage-0 V
   phase, loop phase, epilogue phase)``, which is
   what lets the d128 / kN0 128 gfx942 start point use the whole 64 KiB for
   K + K^T and still run a ring. The plan is checked against the per-arch LDS
   capacity of the core catalog (``ArchTarget.lds_capacity_bytes``).

2. **How is a transposed operand laid out?** :class:`XtLayout` is the "XT"
   layout of the gfx942 body: the transposed image of Q, dO (per q step) or K
   (once per CTA), ``rows = D`` and ``cols = sequence``, so that the 16x16x16
   B-operand read of one lane (four consecutive sequence positions at one
   head-dim column) is one 8-byte ``ds_read_b64``. The layout is a 128-byte
   line XOR swizzle on 8-byte chunks (``swizzle = "xor"``), a row pad
   (``"padN"``, N bytes per row) or plain (``"none"``). :class:`XtWriter` is the
   writer's register distribution after ``shuffle_tile`` (each thread holds
   ``k_per_thread`` sequence rows times ``vec_d`` head-dim columns, loaded from
   global with ``vec_d``-wide vectors, written to LDS as ``vec_d`` stores of
   ``k_per_thread`` elements). Both sides use the same :meth:`XtLayout.offset`.
   A row-major buffer with ``lds_swizzle = "xor"`` uses the same line XOR on
   16-byte chunks (:class:`RowXorLayout`). The body realises both through
   ``_attention_bwd_frag.XorLdsView``.

3. **How is a DMA-filled buffer laid out?** :class:`DmaSlabLayout` is the
   gfx950 slab layout: 128-byte slabs, 16-byte chunk index XOR ``row % 8``. A
   DMA lane writes 16 contiguous LDS bytes, so the swizzle is applied on the
   *source* side: :meth:`DmaSlabLayout.dma_source` names the logical element a
   lane must fetch so that the bytes land swizzled; straight and transposed
   reads use :meth:`DmaSlabLayout.offset`. ``LdsLayout.validate_for_async`` is
   not used or changed.

Conflict predictions (:func:`xt_conflicts`, :func:`dma_slab_read_conflicts`)
go through the production predictor ``rocke.analysis.lds``; they are model
results, not hardware facts (a device probe confirms them separately).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from typing import Any

from rocke.core.arch import ArchTarget

__all__ = [
    "BUFFER_ALIGN_BYTES",
    "DKV_EPILOGUES",
    "DMA_FILLED",
    "LDS_LINE_BYTES",
    "LDS_ORDERS",
    "PHASES",
    "PHASE_EPILOGUE",
    "PHASE_LOOP",
    "PHASE_STAGE0_K",
    "PHASE_STAGE0_V",
    "BwdLdsRequest",
    "DmaSlabLayout",
    "LdsBuffer",
    "LdsPlan",
    "RowXorLayout",
    "XtLayout",
    "XtWriter",
    "choose_xt_swizzle",
    "default_xt_writer",
    "dma_slab_read_accesses",
    "dma_slab_read_conflicts",
    "lds_capacity_bytes",
    "plan_bwd_lds",
    "planner_xt_swizzle",
    "xt_conflicts",
    "xt_read_accesses",
    "xt_write_accesses",
]

LDS_LINE_BYTES = 128
# Every buffer starts on a 128-byte line so the per-buffer swizzles (which are
# defined modulo a line) see the same address classes at any offset.
BUFFER_ALIGN_BYTES = 128

PHASE_STAGE0_K = "stage0_k"
PHASE_STAGE0_V = "stage0_v"
PHASE_LOOP = "loop"
PHASE_EPILOGUE = "epilogue"
PHASES = (PHASE_STAGE0_K, PHASE_STAGE0_V, PHASE_LOOP, PHASE_EPILOGUE)

_ELEM_BYTES = {"fp16": 2, "f16": 2, "bf16": 2}
_SWIZZLES = ("xor", "none", "pad8", "pad16", "pad32")
_TRANSPOSE_SOURCES = ("xt_lds", "tr_read", "lds_plain", "wmma_lds")
# How the register dK / dV accumulators leave the CTA in direct mode:
# "lds_staged" transposes them through one epilogue buffer (dK, then dV),
# "lds_staged_pair" through two (dK and dV staged together, one barrier),
# "direct" stores them from the accumulators (no epilogue buffer).
DKV_EPILOGUES = ("lds_staged", "lds_staged_pair", "direct")
# Placement of the resident region relative to the phase-aliased region:
# "resident_low" puts the resident buffers at offset 0 and the phases above
# them; "phase_low" puts the phases at offset 0 and the residents above the
# largest phase. The footprint is the same; only the offsets move.
LDS_ORDERS = ("resident_low", "phase_low")
# Buffers the q-step DMA fills under ``global_path = "dma"`` (Q and dO of every
# ring slot); they take the slab-XOR layout. The one-time K / V staging stays
# on the register path, and dS / P^T / the epilogue buffer are written from
# registers, so every other buffer keeps its row-major layout.
DMA_FILLED = ("q", "do")


def lds_capacity_bytes(arch: str) -> int:
    """Per-arch LDS capacity of one workgroup, from the core catalog."""
    return int(ArchTarget.from_gfx(arch).lds_capacity_bytes)


def _align(n: int, a: int = BUFFER_ALIGN_BYTES) -> int:
    return (int(n) + a - 1) // a * a


def _pad_bytes(swizzle: str) -> int:
    if swizzle.startswith("pad"):
        return int(swizzle[3:])
    return 0


def _check_swizzle(swizzle: str) -> None:
    if swizzle not in _SWIZZLES:
        raise ValueError(
            f"unknown LDS swizzle {swizzle!r}; expected one of {_SWIZZLES}"
        )


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# XT transposed layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XtLayout:
    """Transposed LDS image ``[rows][cols]`` with ``cols`` contiguous.

    ``rows`` is the head dim D (the B operand's N extent) and ``cols`` the
    sequence extent (the MMA K extent: ``kM0`` for Q^T / dO^T, ``kN0`` for K^T).

    ``swizzle = "xor"`` keeps the packed size and permutes ``chunk_bytes``
    chunks inside each 128-byte line: with ``P = cols * elem_bytes`` bytes per
    row, ``rpl = max(1, 128 // P)`` rows per line and ``g = row // rpl``, the
    chunk index inside the line is XORed with
    ``h = (g ^ (g >> log2(nch))) % nch`` (``nch = 128 // chunk_bytes``). The
    low term makes 16 consecutive rows (one B read) hit distinct chunks; the
    high term separates rows that are ``vec_d`` apart (one writer store),
    so both directions are conflict-free in the predictor for the start-point
    shapes. ``"padN"`` adds N bytes to every row; ``"none"`` is packed.
    """

    rows: int
    cols: int
    elem_bytes: int = 2
    swizzle: str = "xor"
    chunk_bytes: int = 8

    def __post_init__(self) -> None:
        _check_swizzle(self.swizzle)
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("XtLayout rows and cols must be positive")
        if self.chunk_bytes not in (4, 8, 16):
            raise ValueError("XtLayout chunk_bytes must be 4, 8 or 16")
        p = self.cols * self.elem_bytes
        if p % self.chunk_bytes:
            raise ValueError(
                f"row of {p} bytes is not a multiple of the {self.chunk_bytes}-byte chunk"
            )
        if self.swizzle == "xor" and not (
            p % LDS_LINE_BYTES == 0 or LDS_LINE_BYTES % p == 0
        ):
            raise ValueError(
                f"xor swizzle needs the row ({p} bytes) to divide or be a multiple "
                f"of the {LDS_LINE_BYTES}-byte line"
            )
        pad = _pad_bytes(self.swizzle)
        if pad % self.elem_bytes:
            raise ValueError("row pad must be a whole number of elements")

    @property
    def row_pitch_bytes(self) -> int:
        return self.cols * self.elem_bytes + _pad_bytes(self.swizzle)

    @property
    def size_bytes(self) -> int:
        return self.rows * self.row_pitch_bytes

    def _hash(self, row: int) -> int:
        nch = LDS_LINE_BYTES // self.chunk_bytes
        p = self.cols * self.elem_bytes
        rpl = max(1, LDS_LINE_BYTES // p)
        g = row // rpl
        shift = nch.bit_length() - 1
        return (g ^ (g >> shift)) % nch

    def offset(self, row: int, col: int) -> int:
        """Byte offset of logical element ``(row, col)`` (row = d, col = seq)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"({row}, {col}) outside the {self.rows}x{self.cols} image"
            )
        if self.swizzle != "xor":
            return row * self.row_pitch_bytes + col * self.elem_bytes
        lin = row * self.cols * self.elem_bytes + col * self.elem_bytes
        line, within = divmod(lin, LDS_LINE_BYTES)
        chunk, inner = divmod(within, self.chunk_bytes)
        return (
            line * LDS_LINE_BYTES + (chunk ^ self._hash(row)) * self.chunk_bytes + inner
        )

    def check_vector(self, col: int, n: int) -> None:
        """A vector of ``n`` elements at ``col`` must stay inside one chunk."""
        nbytes = n * self.elem_bytes
        if self.swizzle == "xor":
            if nbytes > self.chunk_bytes or (col * self.elem_bytes) % nbytes:
                raise ValueError(
                    f"{nbytes}-byte vector at col {col} crosses a {self.chunk_bytes}-byte "
                    "xor chunk"
                )
        elif (col * self.elem_bytes) % nbytes:
            raise ValueError(f"{nbytes}-byte vector at col {col} is misaligned")


@dataclass(frozen=True)
class RowXorLayout:
    """Row-major ``[rows][cols]`` buffer (``lds_swizzle = "xor"``) with the XT
    line XOR on ``chunk_bytes`` chunks.

    The offset is :meth:`XtLayout.offset` of the same shape with ``swizzle =
    "xor"`` (the packed size, chunks permuted inside each 128-byte line by
    ``h(row)``); the default 16-byte chunk keeps the 16-byte staging stores and
    the K-inner vector reads of a row-major buffer whole. A row of ``cols *
    elem_bytes`` bytes must divide the line or be a multiple of it.
    """

    rows: int
    cols: int
    elem_bytes: int = 2
    chunk_bytes: int = 16

    def __post_init__(self) -> None:
        _ = self.image  # validates the shape

    @property
    def image(self) -> XtLayout:
        return XtLayout(self.rows, self.cols, self.elem_bytes, "xor", self.chunk_bytes)

    @property
    def swizzle(self) -> str:
        return "xor"

    @property
    def row_pitch_bytes(self) -> int:
        return self.cols * self.elem_bytes

    @property
    def size_bytes(self) -> int:
        return self.rows * self.row_pitch_bytes

    def offset(self, row: int, col: int) -> int:
        """Byte offset of logical element ``(row, col)``."""
        return self.image.offset(row, col)

    def check_vector(self, col: int, n: int) -> None:
        self.image.check_vector(col, n)


@dataclass(frozen=True)
class XtWriter:
    """Register distribution of the XT writer (after ``shuffle_tile``).

    The source tile is ``seq x depth`` (sequence rows, head-dim columns,
    head-dim contiguous in global memory). A work unit is ``k_per_thread``
    sequence rows times ``vec_d`` head-dim columns: loaded with
    ``k_per_thread`` global vector loads of ``vec_d`` elements, then written to
    the XT image as ``vec_d`` stores of ``k_per_thread`` elements (row
    ``d0 + j``, cols ``s0 .. s0 + k_per_thread``).

    Unit ``u`` (``u = iteration * threads + tid``) decodes head-dim-group
    fastest within ``d_lanes`` lanes, then the sequence group, then the
    remaining head-dim groups: ``dg = dg_hi * d_lanes + u % d_lanes``,
    ``sg = (u // d_lanes) % n_sg``, ``dg_hi = u // (d_lanes * n_sg)``.
    ``d_lanes * vec_d * elem_bytes`` contiguous global bytes per sequence row
    per load instruction keeps the global side coalesced.
    """

    seq: int
    depth: int
    threads: int
    vec_d: int = 4
    k_per_thread: int = 2
    d_lanes: int = 16
    elem_bytes: int = 2

    def __post_init__(self) -> None:
        if self.depth % self.vec_d or self.seq % self.k_per_thread:
            raise ValueError("XtWriter: depth % vec_d and seq % k_per_thread must be 0")
        if self.vec_d not in (1, 2, 4, 8) or self.k_per_thread not in (1, 2, 4, 8):
            raise ValueError("XtWriter: vec_d and k_per_thread must be in {1, 2, 4, 8}")
        if self.n_dg % self.d_lanes_eff:
            raise ValueError("XtWriter: head-dim groups must be a multiple of d_lanes")
        if self.threads <= 0:
            raise ValueError("XtWriter: threads must be positive")

    @property
    def n_dg(self) -> int:
        return self.depth // self.vec_d

    @property
    def n_sg(self) -> int:
        return self.seq // self.k_per_thread

    @property
    def d_lanes_eff(self) -> int:
        return min(self.d_lanes, self.depth // self.vec_d)

    @property
    def units(self) -> int:
        return self.n_dg * self.n_sg

    @property
    def iterations(self) -> int:
        return -(-self.units // self.threads)

    @property
    def write_bytes(self) -> int:
        return self.k_per_thread * self.elem_bytes

    @property
    def stores_per_thread(self) -> int:
        """LDS store instructions per thread for one image (all iterations)."""
        return self.iterations * self.vec_d

    @property
    def loads_per_thread(self) -> int:
        """Global vector loads per thread for one image (all iterations)."""
        return self.iterations * self.k_per_thread

    def unit_coords(self, tid: int, iteration: int) -> tuple[int, int] | None:
        """``(s0, d0)`` of the unit of ``tid`` in ``iteration``; None if idle."""
        u = iteration * self.threads + tid
        if u >= self.units:
            return None
        dl = self.d_lanes_eff
        dg_lo = u % dl
        rest = u // dl
        sg = rest % self.n_sg
        dg_hi = rest // self.n_sg
        dg = dg_hi * dl + dg_lo
        return sg * self.k_per_thread, dg * self.vec_d


def default_xt_writer(
    seq: int, depth: int, threads: int, *, elem_bytes: int = 2, max_vec_bytes: int = 16
) -> XtWriter:
    """The writer the planner assumes for a ``seq x depth`` tile.

    Picks the widest global vector (``vec_d``) that still leaves each unit at
    least two sequence rows (so the transposed store is at least 4 bytes), and
    up to four sequence rows per unit (8-byte stores) when the tile is large
    enough to keep every thread busy.
    """
    max_vec = max(1, max_vec_bytes // elem_bytes)
    per_thread = max(1, (seq * depth) // threads)
    vec_d = min(max_vec, depth)
    while vec_d > 1 and (per_thread // vec_d < 2 or depth % vec_d):
        vec_d //= 2
    kpt = 2
    if per_thread // vec_d >= 4 and seq % 4 == 0:
        kpt = 4
    kpt = min(kpt, seq)
    d_lanes = min(16, depth // vec_d)
    return XtWriter(
        seq=seq,
        depth=depth,
        threads=threads,
        vec_d=vec_d,
        k_per_thread=kpt,
        d_lanes=d_lanes,
        elem_bytes=elem_bytes,
    )


def xt_read_accesses(
    layout: XtLayout, *, n0: int, k0: int, atom_k: int = 16, wave_size: int = 64
) -> list[tuple[int, int, int]]:
    """``(lane, byte, width)`` of one B-operand fragment read from an XT image.

    16x16xK MFMA B layout: lane ``l`` holds column ``n0 + l % 16`` and the
    ``atom_k / 4`` consecutive K values starting at ``k0 + (l // 16) * atom_k / 4``
    (``atom_k = 16``: one ``ds_read_b64``; ``atom_k = 32``: one ``ds_read_b128``).
    """
    if wave_size != 64:
        raise ValueError("xt_read_accesses models the wave64 MFMA B layout")
    per_lane = atom_k // 4
    width = per_lane * layout.elem_bytes
    out = []
    for lane in range(wave_size):
        row = n0 + lane % 16
        col = k0 + (lane // 16) * per_lane
        layout.check_vector(col, per_lane)
        out.append((lane, layout.offset(row, col), width))
    return out


def xt_write_accesses(
    layout: XtLayout,
    writer: XtWriter,
    *,
    wave: int,
    iteration: int,
    j: int,
    wave_size: int = 64,
) -> list[tuple[int, int, int]]:
    """``(lane, byte, width)`` of store ``j`` of one wave in one writer iteration."""
    if layout.rows != writer.depth or layout.cols != writer.seq:
        raise ValueError("XtWriter tile does not match the XtLayout image")
    out = []
    for lane in range(wave_size):
        tid = wave * wave_size + lane
        if tid >= writer.threads:
            continue
        coords = writer.unit_coords(tid, iteration)
        if coords is None:
            continue
        s0, d0 = coords
        layout.check_vector(s0, writer.k_per_thread)
        out.append((lane, layout.offset(d0 + j, s0), writer.write_bytes))
    return out


_READ_OPCODE = {4: "ds_read_b32", 8: "ds_read_b64", 16: "ds_read_b128"}
_WRITE_OPCODE = {4: "ds_write_b32", 8: "ds_write_b64", 16: "ds_write_b128"}


def _predict_max(
    arch: str, opcode: str, accesses: Sequence[tuple[int, int, int]]
) -> int:
    """Largest conflict multiplicity of one instruction (1 = conflict-free)."""
    from rocke.analysis.lds import LdsAccess, predict_lds_conflicts

    recs = tuple(
        LdsAccess(access_id=i, lane=lane, lds_byte_address=addr, access_width_bytes=w)
        for i, (lane, addr, w) in enumerate(accesses)
    )
    res = predict_lds_conflicts(target=arch, opcode=opcode, wave_size=64, accesses=recs)
    if res.summary.conflict_group_count == 0:
        return 1
    return int(res.summary.maximum_multiplicity)


def _predictor_targets() -> tuple[str, ...]:
    """Targets with a wave64 conflict profile: the XT accesses are modelled per
    64-lane wave, so a wave32-only profile (gfx1201) is not one of them and
    keeps the RDNA default."""
    from rocke.analysis.lds import registered_targets
    from rocke.analysis.lds.registry import resolve_profile

    return tuple(
        t for t in registered_targets() if 64 in resolve_profile(t).supported_wave_sizes
    )


def xt_conflicts(
    layout: XtLayout, writer: XtWriter | None, arch: str, *, atom_k: int = 16
) -> dict[str, int] | None:
    """Predicted worst conflict multiplicity of the XT reads and writes on ``arch``.

    Returns ``{"read": m_r, "write": m_w}`` (1 = conflict-free) over every B
    fragment read of the image and every writer store of every wave, or None
    when the predictor has no wave64 profile for ``arch`` (RDNA).
    """
    if arch not in _predictor_targets():
        return None
    read_w = (atom_k // 4) * layout.elem_bytes
    worst_r = 1
    for n0 in range(0, layout.rows, 16):
        for k0 in range(0, layout.cols, atom_k):
            acc = xt_read_accesses(layout, n0=n0, k0=k0, atom_k=atom_k)
            worst_r = max(worst_r, _predict_max(arch, _READ_OPCODE[read_w], acc))
    worst_w = 1
    if writer is not None:
        if writer.write_bytes not in _WRITE_OPCODE:
            raise ValueError(f"no ds_write opcode for {writer.write_bytes}-byte stores")
        waves = -(-writer.threads // 64)
        for it in range(writer.iterations):
            for wave in range(waves):
                for j in range(writer.vec_d):
                    acc = xt_write_accesses(
                        layout, writer, wave=wave, iteration=it, j=j
                    )
                    if acc:
                        worst_w = max(
                            worst_w,
                            _predict_max(arch, _WRITE_OPCODE[writer.write_bytes], acc),
                        )
    return {"read": worst_r, "write": worst_w}


def choose_xt_swizzle(
    rows: int,
    cols: int,
    writer: XtWriter | None,
    arch: str,
    *,
    candidates: Sequence[str] = ("xor", "pad8", "pad16", "pad32", "none"),
    elem_bytes: int = 2,
    atom_k: int = 16,
) -> tuple[str, dict[str, Any]]:
    """Pick the per-buffer swizzle for an XT image with the conflict predictor.

    Ranking: worst read multiplicity, then worst write multiplicity, then
    bytes. Ties keep the candidate order. Returns ``(swizzle, table)`` where
    ``table`` maps every legal candidate to its prediction and size (the
    ``lds_swizzle`` knob of a buffer enters the sweep only when two candidates
    tie on the first two keys). RDNA (no predictor profile) gets ``"pad8"``
    unless it is not in the candidates.
    """
    chunk = (atom_k // 4) * elem_bytes
    table: dict[str, Any] = {}
    for sw in candidates:
        try:
            lay = XtLayout(rows, cols, elem_bytes, sw, chunk_bytes=chunk)
            pred = xt_conflicts(lay, writer, arch, atom_k=atom_k)
        except ValueError as exc:
            table[sw] = {"illegal": str(exc)}
            continue
        table[sw] = {"prediction": pred, "bytes": lay.size_bytes}
    legal = [sw for sw in candidates if "illegal" not in table[sw]]
    if not legal:
        raise ValueError(f"no legal XT swizzle among {tuple(candidates)}")
    if table[legal[0]]["prediction"] is None:
        return ("pad8" if "pad8" in legal else legal[0]), table

    def key(sw: str) -> tuple[int, int, int, int]:
        p = table[sw]["prediction"]
        return (p["read"], p["write"], table[sw]["bytes"], list(candidates).index(sw))

    return min(legal, key=key), table


@cache
def planner_xt_swizzle(
    rows: int, cols: int, threads: int, arch: str, elem_bytes: int = 2, atom_k: int = 16
) -> str:
    """The swizzle the planner gives an XT image no ``lds_swizzle`` entry
    names: :func:`choose_xt_swizzle` over the image's default writer and the
    reading atom (``"xor"`` on the gfx942 start points; RDNA, without a
    predictor profile, ``"pad8"``)."""
    writer = default_xt_writer(cols, rows, threads, elem_bytes=elem_bytes)
    best, _table = choose_xt_swizzle(
        rows, cols, writer, arch, elem_bytes=elem_bytes, atom_k=atom_k
    )
    return best


# ---------------------------------------------------------------------------
# DMA slab layout (gfx950)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DmaSlabLayout:
    """128-byte slab layout with a 16-byte chunk XOR, for LDS-DMA filled tiles.

    Logical tile ``[rows][cols]`` (``cols`` contiguous in global memory).
    Rows of ``R = cols * elem_bytes >= 128`` bytes are split into ``R / 128``
    column slabs; slab ``s`` stores 128 bytes of every row, row stride 128
    bytes, slabs stacked. Rows shorter than a line are folded ``128 / R`` to a
    line. Inside a line the 16-byte chunk index is XORed with ``line_row % 8``
    (``line_row`` = row inside the slab, or the folded line index).

    The DMA writes lane-contiguous 16-byte pieces, so the producer cannot
    apply the XOR; :meth:`dma_source` gives, for each destination 16-byte
    slot, the logical element whose 16-byte vector must be fetched.
    """

    rows: int
    cols: int
    elem_bytes: int = 2

    CHUNK = 16

    def __post_init__(self) -> None:
        r = self.row_bytes
        if self.rows <= 0 or r <= 0:
            raise ValueError("DmaSlabLayout rows and cols must be positive")
        if r % self.CHUNK:
            raise ValueError("DMA slab rows must be whole 16-byte chunks")
        if r >= LDS_LINE_BYTES and r % LDS_LINE_BYTES:
            raise ValueError("DMA slab rows of 128 bytes or more must be whole lines")
        if r < LDS_LINE_BYTES and (
            LDS_LINE_BYTES % r or self.rows % (LDS_LINE_BYTES // r)
        ):
            raise ValueError("folded DMA slab rows must tile whole lines")

    @property
    def row_bytes(self) -> int:
        return self.cols * self.elem_bytes

    @property
    def fold(self) -> int:
        return max(1, LDS_LINE_BYTES // self.row_bytes)

    @property
    def slabs(self) -> int:
        return max(1, self.row_bytes // LDS_LINE_BYTES)

    @property
    def size_bytes(self) -> int:
        return self.rows * self.row_bytes

    def offset(self, row: int, col: int) -> int:
        """Byte offset of logical ``(row, col)`` (read side, straight or tr)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"({row}, {col}) outside the {self.rows}x{self.cols} tile")
        b = col * self.elem_bytes
        if self.row_bytes >= LDS_LINE_BYTES:
            s, within = divmod(b, LDS_LINE_BYTES)
            line_row = row
            base = s * self.rows * LDS_LINE_BYTES + row * LDS_LINE_BYTES
        else:
            line_row, sub = divmod(row, self.fold)
            within = sub * self.row_bytes + b
            base = line_row * LDS_LINE_BYTES
        chunk, inner = divmod(within, self.CHUNK)
        return base + (chunk ^ (line_row % 8)) * self.CHUNK + inner

    def dma_source(self, slot: int) -> tuple[int, int]:
        """Logical ``(row, col)`` whose 16-byte vector lands in LDS slot ``slot``.

        ``slot`` counts 16-byte destination slots from the buffer base (DMA
        lane ``i`` of an instruction whose destination base is slot ``s0``
        writes slot ``s0 + i``).
        """
        p = slot * self.CHUNK
        if not 0 <= p < self.size_bytes:
            raise ValueError(f"slot {slot} outside the buffer")
        if self.row_bytes >= LDS_LINE_BYTES:
            per_slab = self.rows * LDS_LINE_BYTES
            s, rem = divmod(p, per_slab)
            row, w = divmod(rem, LDS_LINE_BYTES)
            chunk = (w // self.CHUNK) ^ (row % 8)
            col = (s * LDS_LINE_BYTES + chunk * self.CHUNK) // self.elem_bytes
            return row, col
        line_row, w = divmod(p, LDS_LINE_BYTES)
        within = ((w // self.CHUNK) ^ (line_row % 8)) * self.CHUNK
        sub, b = divmod(within, self.row_bytes)
        return line_row * self.fold + sub, b // self.elem_bytes


def dma_slab_read_accesses(
    layout: DmaSlabLayout, *, m0: int, k0: int, atom_k: int = 32, wave_size: int = 64
) -> list[tuple[int, int, int]]:
    """``(lane, byte, width)`` of one straight A-operand read of a slab tile.

    16x16xK MFMA A layout: lane ``l`` holds row ``m0 + l % 16`` and the
    ``atom_k / 4`` K values from ``k0 + (l // 16) * atom_k / 4``.
    """
    if wave_size != 64:
        raise ValueError("dma_slab_read_accesses models the wave64 MFMA A layout")
    per_lane = atom_k // 4
    return [
        (
            lane,
            layout.offset(m0 + lane % 16, k0 + (lane // 16) * per_lane),
            per_lane * layout.elem_bytes,
        )
        for lane in range(wave_size)
    ]


def dma_slab_read_conflicts(
    layout: DmaSlabLayout, arch: str, *, atom_k: int = 32
) -> int | None:
    """Worst predicted multiplicity of the straight A reads of a slab tile."""
    if arch not in _predictor_targets():
        return None
    width = (atom_k // 4) * layout.elem_bytes
    worst = 1
    for m0 in range(0, layout.rows - layout.rows % 16, 16):
        for k0 in range(0, layout.cols, atom_k):
            acc = dma_slab_read_accesses(layout, m0=m0, k0=k0, atom_k=atom_k)
            worst = max(worst, _predict_max(arch, _READ_OPCODE[width], acc))
    return worst


# ---------------------------------------------------------------------------
# Phase planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwdLdsRequest:
    """Every input of the LDS phase planner (resolved tile values, no Nones).

    The field names and meanings follow ``BwdTileGeometry``;
    :meth:`from_geometry` reads them off any object with those attributes.
    ``lds_swizzle`` maps a buffer name to its swizzle (missing names use the
    predictor's choice :func:`planner_xt_swizzle` for XT images and
    ``"none"`` for the rest; ``"xor"`` on a row-major buffer is
    :class:`RowXorLayout`; DMA buffers are always slab-XOR; the fp32
    accumulator images take no swizzle). ``dkv_epilogue`` (:data:`DKV_EPILOGUES`) decides the
    epilogue buffers and ``lds_order`` (:data:`LDS_ORDERS`) the placement of
    the resident region. ``ds_slots`` is the number of dS copies (2 under
    ``barrier_merge = "on"``, whose steps alternate between them).
    """

    head_size: int
    k_m0: int
    k_n0: int
    waves: int
    wave_size: int = 64
    dtype: str = "fp16"
    kv_residency: str = "reg"
    kt_source: str = "reg"
    transpose_source: str = "xt_lds"
    ring_depth: int = 1
    global_path: str = "vgpr"
    pt_route: str = "relabel"
    acc_in_lds: bool = False
    dq_mode: str = "atomic"
    lds_swizzle: tuple[tuple[str, str], ...] = ()
    dkv_epilogue: str = "lds_staged"
    lds_order: str = "resident_low"
    ds_slots: int = 1
    # K extent of the atoms that read the XT images (G1 / G3 read Q^T and
    # dO^T, G4 reads K^T): a lane reads atom_k / 4 consecutive elements, the
    # XOR chunk of that image (8 bytes at K = 16, 16 bytes at K = 32).
    atom_k_g13: int = 16
    atom_k_g4: int = 16

    def __post_init__(self) -> None:
        if self.dtype not in _ELEM_BYTES:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.kv_residency not in ("reg", "lds"):
            raise ValueError("kv_residency must be 'reg' or 'lds'")
        if self.kt_source not in ("reg", "lds"):
            raise ValueError("kt_source must be 'reg' or 'lds'")
        if self.kt_source == "reg" and self.kv_residency != "reg":
            raise ValueError("kt_source = 'reg' requires kv_residency = 'reg'")
        if self.transpose_source not in _TRANSPOSE_SOURCES:
            raise ValueError(f"unknown transpose_source {self.transpose_source!r}")
        if self.ring_depth not in (1, 2, 3):
            raise ValueError("ring_depth must be 1, 2 or 3")
        if self.global_path not in ("vgpr", "dma"):
            raise ValueError("global_path must be 'vgpr' or 'dma'")
        if self.global_path == "dma" and self.transpose_source == "xt_lds":
            raise ValueError("global_path = 'dma' excludes transpose_source = 'xt_lds'")
        if self.pt_route not in ("relabel", "lds"):
            raise ValueError("pt_route must be 'relabel' or 'lds'")
        if self.dq_mode not in ("atomic", "split"):
            raise ValueError("dq_mode must be 'atomic' or 'split'")
        if self.dkv_epilogue not in DKV_EPILOGUES:
            raise ValueError(f"dkv_epilogue must be one of {DKV_EPILOGUES}")
        if self.lds_order not in LDS_ORDERS:
            raise ValueError(f"lds_order must be one of {LDS_ORDERS}")
        if self.ds_slots not in (1, 2):
            raise ValueError("ds_slots must be 1 or 2")
        for name, sw in self.lds_swizzle:
            _check_swizzle(sw)
        if self.atom_k_g13 not in (16, 32) or self.atom_k_g4 not in (16, 32):
            raise ValueError("atom_k_g13 / atom_k_g4 must be 16 or 32")
        for v in (self.head_size, self.k_m0, self.k_n0, self.waves):
            if int(v) <= 0:
                raise ValueError("tile extents and waves must be positive")

    @property
    def elem_bytes(self) -> int:
        return _ELEM_BYTES[self.dtype]

    @property
    def threads(self) -> int:
        return self.waves * self.wave_size

    def swizzle_of(self, name: str, default: str) -> str:
        return dict(self.lds_swizzle).get(name, default)

    @staticmethod
    def canonical_swizzle(value: Any) -> tuple[tuple[str, str], ...]:
        """Normalise a swizzle mapping or ``"k=xor,q=pad8"`` string."""
        if value is None:
            return ()
        if isinstance(value, str):
            items = []
            for part in value.split(","):
                part = part.strip()
                if not part:
                    continue
                name, _, sw = part.partition("=")
                items.append((name.strip(), sw.strip()))
            return tuple(sorted(items))
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), str(v)) for k, v in value.items()))
        return tuple(sorted((str(k), str(v)) for k, v in value))

    @classmethod
    def from_geometry(
        cls,
        geom: Any,
        *,
        dtype: str = "fp16",
        acc_in_lds: bool = False,
        dq_mode: str = "atomic",
    ) -> BwdLdsRequest:
        """Build a request from a resolved tile geometry (duck-typed)."""
        wave_size = int(getattr(geom, "wave_size", 64))
        atoms = getattr(geom, "atoms", None)
        return cls(
            head_size=int(geom.head_size),
            k_m0=int(geom.k_m0),
            k_n0=int(geom.k_n0),
            waves=int(geom.waves),
            wave_size=wave_size,
            dtype=dtype,
            kv_residency=str(geom.kv_residency),
            kt_source=str(geom.kt_source),
            transpose_source=str(geom.transpose_source),
            ring_depth=int(geom.ring_depth),
            global_path=str(geom.global_path),
            pt_route=str(geom.pt_route),
            acc_in_lds=bool(getattr(geom, "acc_in_lds", acc_in_lds)),
            dq_mode=str(getattr(geom, "dq_mode", dq_mode)),
            lds_swizzle=cls.canonical_swizzle(getattr(geom, "lds_swizzle", None)),
            dkv_epilogue=str(getattr(geom, "dkv_epilogue", "lds_staged")),
            lds_order=str(getattr(geom, "lds_order", "resident_low")),
            ds_slots=2 if getattr(geom, "barrier_merge", "off") == "on" else 1,
            atom_k_g13=int(getattr(atoms, "g13", 16)),
            atom_k_g4=int(getattr(atoms, "g4", 16)),
        )


@dataclass(frozen=True)
class LdsBuffer:
    """One placed buffer: ``[offset, offset + nbytes)`` live in ``phases``."""

    name: str
    slot: int
    offset: int
    nbytes: int
    phases: tuple[str, ...]
    layout: str
    resident: bool = False

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class LdsPlan:
    """Result of :func:`plan_bwd_lds`: offsets inside one ``smem_alloc``."""

    arch: str
    request: BwdLdsRequest
    buffers: tuple[LdsBuffer, ...]
    resident_bytes: int
    total_bytes: int
    capacity_bytes: int
    phase_bytes: tuple[tuple[str, int], ...]
    layouts: tuple[tuple[str, Any], ...] = field(default=(), compare=False)

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.capacity_bytes

    @property
    def phase_offset(self) -> int:
        """Byte offset of the phase-aliased region (see :data:`LDS_ORDERS`)."""
        if self.request.lds_order == "phase_low":
            return 0
        return self.resident_bytes

    @property
    def unaliased_bytes(self) -> int:
        """Footprint if no buffer were phase aliased (sum of all buffers)."""
        return sum(_align(b.nbytes) for b in self.buffers)

    def buffer(self, name: str, slot: int = 0) -> LdsBuffer:
        for b in self.buffers:
            if b.name == name and b.slot == slot:
                return b
        raise KeyError(f"no LDS buffer {name!r} slot {slot}")

    def has(self, name: str) -> bool:
        return any(b.name == name for b in self.buffers)

    def offset(self, name: str, slot: int = 0) -> int:
        return self.buffer(name, slot).offset

    def ring_offsets(self, name: str) -> tuple[int, ...]:
        return tuple(
            b.offset
            for b in sorted(self.buffers, key=lambda x: x.slot)
            if b.name == name
        )

    def ring_stride(self, name: str) -> int:
        """Byte distance between consecutive ring slots of ``name`` (0 for one
        slot); raises unless the slots form an arithmetic progression, which a
        run-time slot base ``slot * stride`` relies on."""
        offs = self.ring_offsets(name)
        if not offs:
            raise KeyError(f"no LDS buffer {name!r}")
        if len(offs) == 1:
            return 0
        stride = offs[1] - offs[0]
        if stride <= 0 or any(b - a != stride for a, b in pairwise(offs)):
            raise ValueError(f"ring slots of {name!r} are not evenly spaced: {offs}")
        return stride

    def layout(self, name: str) -> Any:
        return dict(self.layouts)[name]

    def live(self, phase: str) -> tuple[LdsBuffer, ...]:
        return tuple(b for b in self.buffers if phase in b.phases)

    def check(self) -> None:
        """Raise if two buffers live in one phase overlap, or a buffer escapes."""
        for b in self.buffers:
            if (
                b.offset % BUFFER_ALIGN_BYTES
                or b.offset < 0
                or b.end > self.total_bytes
            ):
                raise ValueError(f"LDS buffer {b.name}[{b.slot}] misplaced")
        for phase in PHASES:
            live = sorted(self.live(phase), key=lambda x: x.offset)
            for a, c in pairwise(live):
                if a.end > c.offset:
                    raise ValueError(
                        f"LDS buffers {a.name}[{a.slot}] and {c.name}[{c.slot}] overlap "
                        f"in phase {phase}"
                    )

    def check_capacity(self) -> None:
        if not self.fits:
            raise ValueError(
                f"LDS footprint {self.total_bytes} bytes exceeds the {self.arch} "
                f"capacity of {self.capacity_bytes} bytes"
            )


def _normal_layout(
    rows: int, cols: int, eb: int, swizzle: str, dma: bool
) -> tuple[int, str, Any]:
    if dma:
        lay = DmaSlabLayout(rows, cols, eb)
        return lay.size_bytes, "dma_slab_xor", lay
    if swizzle == "xor":
        lay = RowXorLayout(rows, cols, eb)
        return lay.size_bytes, "rowmajor:xor", lay
    pitch = cols * eb + _pad_bytes(swizzle)
    return rows * pitch, f"rowmajor:{swizzle}", None


def _xt_layout(
    rows: int, cols: int, eb: int, swizzle: str, atom_k: int = 16
) -> tuple[int, str, Any]:
    lay = XtLayout(rows, cols, eb, swizzle, chunk_bytes=(atom_k // 4) * eb)
    return lay.size_bytes, f"xt:{swizzle}", lay


def plan_bwd_lds(req: BwdLdsRequest, *, arch: str, strict: bool = True) -> LdsPlan:
    """Place every LDS buffer of one backward CTA; phase-alias stage 0 and the loop.

    Buffers (names are stable; ``slot`` indexes ring copies):

    * resident (all phases): ``k_res``/``v_res`` when ``kv_residency = "lds"``;
      ``kt_res`` (XT image) when ``kt_source = "lds"`` and the transpose source
      is ``xt_lds``; ``k_res`` also when K^T is read from a plain or DMA copy of
      K; ``dk_acc``/``dv_acc`` (fp32) when ``acc_in_lds``. K^T is read by G4
      only, so ``dq_mode = "split"`` (no G4) plans no K^T buffer.
    * stage 0, K phase: ``k_stage`` (K staged to registers; omitted when
      ``k_res`` is resident) and ``kt_stage`` (XT image for register K^T,
      ``dq_mode = "atomic"`` only) when ``kv_residency = "reg"``.
    * stage 0, V phase: ``v_stage`` when ``kv_residency = "reg"``.
    * loop (per ring slot): ``q``, ``do``; ``qt``, ``dot`` (XT images) when
      ``transpose_source = "xt_lds"``. The per-row ``lse2`` / ``Dsum`` stats
      have no LDS buffer under the synchronous staging (every lane loads its
      rows from the workspace); a ring (``ring_depth > 1``, the DMA path, the
      barrier merge) stages them per slot (``max(kM0, wave_size) * 4`` bytes
      each).
    * loop (single): ``ds`` (dS, for G4 and LDS-routed dS^T) unless
      ``dq_mode = "split"`` with a relabel route (``ds_slots`` copies: two
      under the barrier merge); ``pt`` (P) when ``pt_route = "lds"``.
    * epilogue: ``dkv_out`` (``kN0 x D`` in the I/O dtype, row pad from
      ``lds_swizzle``), through which the register dK / dV accumulators are
      transposed to row-contiguous vector stores; two slots under
      ``dkv_epilogue = "lds_staged_pair"`` (dK in slot 0, dV in slot 1);
      absent with ``acc_in_lds`` (the fp32 accumulator images are read
      directly) and under ``dkv_epilogue = "direct"`` (stores straight from
      the accumulators).

    ``lds_order = "resident_low"`` places the resident buffers from offset 0
    and every phase above them; ``"phase_low"`` places every phase from
    offset 0 and the resident buffers above the largest phase. The order of
    the buffers inside a phase is fixed (the order listed above).

    The plan always passes :meth:`LdsPlan.check`; with ``strict`` it also
    raises ``ValueError`` when the footprint exceeds the arch capacity.
    """
    eb = req.elem_bytes
    d, m0, n0 = req.head_size, req.k_m0, req.k_n0
    dma = req.global_path == "dma"
    xt = req.transpose_source == "xt_lds"
    resident: list[tuple[str, int, str, Any]] = []
    stage_k: list[tuple[str, int, str, Any]] = []
    stage_v: list[tuple[str, int, str, Any]] = []
    loop: list[tuple[str, int, int, str, Any]] = []  # name, slot, bytes, layout, obj
    epilogue: list[tuple[str, int, str, Any]] = []

    def normal(name: str, rows: int, cols: int) -> tuple[str, int, str, Any]:
        nbytes, desc, obj = _normal_layout(
            rows, cols, eb, req.swizzle_of(name, "none"), dma and name in DMA_FILLED
        )
        return (name, nbytes, desc, obj)

    def xt_img(
        name: str, rows: int, cols: int, atom_k: int
    ) -> tuple[str, int, str, Any]:
        sw = req.swizzle_of(name, "")
        if not sw:
            sw = planner_xt_swizzle(rows, cols, req.threads, arch, eb, atom_k)
        nbytes, desc, obj = _xt_layout(rows, cols, eb, sw, atom_k)
        return (name, nbytes, desc, obj)

    # K^T feeds G4 only; the split-dQ main kernel has no G4, so it plans no
    # K^T image or copy whatever kt_source says.
    kt_live = req.dq_mode == "atomic"
    if req.kv_residency == "lds":
        resident.append(normal("k_res", n0, d))
        resident.append(normal("v_res", n0, d))
    if req.kt_source == "lds" and kt_live:
        if xt:
            resident.append(xt_img("kt_res", d, n0, req.atom_k_g4))
        elif req.kv_residency == "reg":
            resident.append(normal("k_res", n0, d))
    if req.acc_in_lds:
        resident.append(("dk_acc", n0 * d * 4, "rowmajor:f32", None))
        resident.append(("dv_acc", n0 * d * 4, "rowmajor:f32", None))
    if req.kv_residency == "reg":
        # A resident copy of K (kept for K^T reads) also feeds the register
        # staging of K, so no second staging copy is needed then.
        if not any(name == "k_res" for name, *_ in resident):
            stage_k.append(normal("k_stage", n0, d))
        if req.kt_source == "reg" and xt and kt_live:
            stage_k.append(xt_img("kt_stage", d, n0, req.atom_k_g4))
        stage_v.append(normal("v_stage", n0, d))

    # lse2 / Dsum: under the synchronous staging the body reads the stats
    # straight from the workspace (StatsView), so an LDS copy would be a dead
    # reservation; a ring (ring_depth > 1, the DMA path, the barrier merge)
    # stages each slot's stats rows next to its Q / dO.
    stats_bytes = max(m0, req.wave_size) * 4
    stats_in_lds = req.ring_depth > 1 or req.global_path == "dma"
    for slot in range(req.ring_depth):
        for name, rows, cols in (("q", m0, d), ("do", m0, d)):
            nm, nb, desc, obj = normal(name, rows, cols)
            loop.append((nm, slot, nb, desc, obj))
        if xt:
            for name in ("qt", "dot"):
                nm, nb, desc, obj = xt_img(name, d, m0, req.atom_k_g13)
                loop.append((nm, slot, nb, desc, obj))
        if stats_in_lds:
            loop.append(("lse2", slot, stats_bytes, "vector:f32", None))
            loop.append(("dsum", slot, stats_bytes, "vector:f32", None))
    if req.dq_mode == "atomic" or req.pt_route == "lds":
        nm, nb, desc, obj = normal("ds", m0, n0)
        for slot in range(req.ds_slots):
            loop.append((nm, slot, nb, desc, obj))
    if req.pt_route == "lds":
        nm, nb, desc, obj = normal("pt", m0, n0)
        loop.append((nm, 0, nb, desc, obj))
    if not req.acc_in_lds and req.dkv_epilogue != "direct":
        # Not DMA-filled: a plain row-major image whatever the global path.
        sw = req.swizzle_of("dkv_out", "none")
        out_bytes = n0 * (d * eb + _pad_bytes(sw))
        out_obj = RowXorLayout(n0, d, eb) if sw == "xor" else None
        for slot in range(2 if req.dkv_epilogue == "lds_staged_pair" else 1):
            epilogue.append(("dkv_out", slot, out_bytes, f"rowmajor:{sw}", out_obj))

    placed: list[LdsBuffer] = []
    layouts: dict[str, Any] = {}

    def place_phase(items, phase, base):
        cur = base
        for item in items:
            if len(item) == 4:
                name, nbytes, desc, obj = item
                slot = 0
            else:
                name, slot, nbytes, desc, obj = item
            placed.append(LdsBuffer(name, slot, cur, nbytes, (phase,), desc))
            layouts.setdefault(name, obj if obj is not None else desc)
            cur = _align(cur + nbytes)
        return cur - base

    def place_resident(base):
        off = base
        for name, nbytes, desc, obj in resident:
            placed.append(LdsBuffer(name, 0, off, nbytes, PHASES, desc, resident=True))
            layouts[name] = obj if obj is not None else desc
            off = _align(off + nbytes)
        return off - base

    def place_phases(base):
        return (
            (PHASE_STAGE0_K, place_phase(stage_k, PHASE_STAGE0_K, base)),
            (PHASE_STAGE0_V, place_phase(stage_v, PHASE_STAGE0_V, base)),
            (PHASE_LOOP, place_phase(loop, PHASE_LOOP, base)),
            (PHASE_EPILOGUE, place_phase(epilogue, PHASE_EPILOGUE, base)),
        )

    if req.lds_order == "phase_low":
        phase_sizes = place_phases(0)
        resident_bytes = place_resident(max(s for _, s in phase_sizes))
    else:
        resident_bytes = place_resident(0)
        phase_sizes = place_phases(resident_bytes)
    total = resident_bytes + max(s for _, s in phase_sizes)
    plan = LdsPlan(
        arch=arch,
        request=req,
        buffers=tuple(placed),
        resident_bytes=resident_bytes,
        total_bytes=total,
        capacity_bytes=lds_capacity_bytes(arch),
        phase_bytes=phase_sizes,
        layouts=tuple(sorted(layouts.items())),
    )
    plan.check()
    if strict:
        plan.check_capacity()
    return plan
