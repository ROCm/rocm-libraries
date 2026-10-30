# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 / complex-f64 GETT: a tensor contraction computed as one GEMM launch.

``D[m, n] = alpha * sum_k A[m, k] * B[n, k] + beta * C[m, n]`` where every
operand is read in place through its own strides: the GEMM row ``m`` of ``A``
sits at element offset ``rowoff_A(m) + koff_A(k)``, each of ``rowoff`` /
``koff`` a mixed-radix function of the index (``digits``: ``(extent,
stride)`` pairs, fastest first). A plain strided GEMM is the one-digit case;
a hipTensor contraction with arbitrary mode orders maps onto it through
:mod:`.gett_f64_plan` (M modes, N modes, K modes and batch modes become
digits, so no permute pass runs before or after the kernel).

Per CTA (``warps_m x warps_n`` waves, one ``tile_m x tile_n`` output tile):

* **Loads.** Each operand tile (``tile_k`` K rows by ``tile_m`` / ``tile_n``)
  is read with 16-byte buffer loads where the operand is contiguous along M/N
  or along K (8-byte loads otherwise), into registers one K tile ahead of the
  MFMAs, then written to a double-buffered LDS stage: one barrier per K tile.
  Out-of-range rows, columns and K rows get an out-of-bounds buffer offset,
  so the hardware returns zero and no lane branches. The last K tile issues
  no LDS store and no barrier.
* **LDS layout.** ``[k][mn]`` (M/N fastest) with ``lds_pad`` elements of row
  padding. The fragment each lane reads, and the output element each
  accumulator slot holds, come from the catalog lane maps of the f64 atom
  (``a_layout`` / ``b_layout`` / ``c_layout``): one ``ds_read_b64`` per
  operand fragment (complex: ``ds_read_b128`` of the interleaved pair).
* **Math.** f64 MFMA 16x16x4 with an f64 accumulator. Complex data stay
  interleaved; each complex atom is four real MFMAs into a real and an
  imaginary accumulator (fused 4M). Conjugating an operand only moves the
  sign: the negated fragment is one ``v_xor`` of its sign bit, reused by every
  atom of the k-step. ``k_block`` sums K in blocks of K tiles (a second set of
  accumulators adds the block sums), which shortens the chain of roundings
  an output goes through at long K.
* **Epilogue.** ``d = fma(alpha, acc, beta * c)`` in f64: two roundings
  (``beta * c``, then the fma); complex: the complex products in f64. D is
  written straight to its strides; C is read through its own strides and
  only when ``beta != 0``. BLAS rules: with ``beta == 0`` C is not read (a
  NaN in C is ignored), and with ``alpha == 0`` the accumulator is not used
  (``d = beta * c``; a NaN or Inf in A or B does not reach D). A complex
  ``alpha`` (``beta``) with a zero imaginary part scales both components by
  its real part, so an Inf in one accumulator component does not turn the
  other component into NaN through ``0 * Inf``.

Sizes are compile-time constants by default (one kernel per index map). With
``runtime_sizes`` the kernel takes ``M, N, K, L`` and every stride as kernel
arguments: one binary then serves every strided batched GEMM of the same
layout class (which strides are 1, and the parity of the sizes the vector
loads depend on), see :func:`runtime_template`.

Addressing: the batch offset and (when the K digits split at the K tile, see
:func:`k_split`) the K-tile offset move a 64-bit base pointer, and each K tile
gets its own buffer resource; only the per-lane part -- rows plus the K rows of
one tile for A / B, rows plus columns for C / D -- is a 32-bit buffer offset and
must stay below 2^31 bytes (:meth:`GettF64Problem.addr_span_bytes`). The planner
splits a contraction whose output tile space is larger into several launches.

f32 / complex-f32 data (``GettF64Problem.dtype = "f32"``): the same kernel
with f32 words: the f32 MFMA 16x16x4 atom of the catalog with an f32
accumulator (IEEE f32 products, native f32 accumulation), f32 LDS tiles and
fragments (one ``ds_read_b32``; complex ``ds_read_b64``), f32 alpha / beta and
the epilogue in f32. Real operands contiguous along a run whose length is a
multiple of 4 load 16 bytes per lane (4 f32), even runs 8 bytes; complex
operands load one element (8 bytes) per lane. The f32 kernel takes the knobs
both engines build (no ``split_k`` / ``split_serial`` and none of the Python-only
schedule levers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.arch import ArchTarget, known_arches
from ...core.ir import (
    F32,
    F64,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from ...helpers.spec import SignatureBuilder, kernel_name_join

#: (extent, stride) pairs, fastest digit first; strides are in elements
#: (complex elements for complex problems).
Digits = Tuple[Tuple[int, int], ...]


def f64_atom(arch: str):
    """The catalog's ``16x16x4`` f64 MMA atom of ``arch`` (None if it has none)."""
    try:
        target = ArchTarget.from_gfx(arch)
    except (KeyError, ValueError):
        return None
    return target.mma.op_for_shape(
        family="mma", a_dtype="fp64", b_dtype="fp64", c_dtype="fp64", m=16, n=16, k=4
    )


def f32_atom(arch: str):
    """The catalog's ``16x16x4`` f32 MMA atom of ``arch`` (None if it has none)."""
    try:
        target = ArchTarget.from_gfx(arch)
    except (KeyError, ValueError):
        return None
    return target.mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )


def gett_atom(arch: str, dtype: str = "f64"):
    """The ``16x16x4`` MMA atom of ``dtype`` ("f64" or "f32") on ``arch``."""
    return f32_atom(arch) if dtype == "f32" else f64_atom(arch)


def f64_gett_archs() -> Tuple[str, ...]:
    """Every catalog target with the f64 atom the kernel is built on."""
    return tuple(a for a in known_arches() if f64_atom(a) is not None)


def f32_gett_archs() -> Tuple[str, ...]:
    """Every catalog target with the f32 atom the f32 kernel is built on."""
    return tuple(a for a in known_arches() if f32_atom(a) is not None)


#: Targets whose MMA catalog has the f64 atom (read from the catalog).
F64_GETT_ARCHS = f64_gett_archs()
#: Targets whose MMA catalog has the f32 atom (read from the catalog).
F32_GETT_ARCHS = f32_gett_archs()
#: Data types of the kernel's words ("f64": f64 / complex f64, "f32": f32 /
#: complex f32).
GETT_DTYPES = ("f64", "f32")

_MASK_OFFSET = -(1 << 31)  # 0x80000000 as a buffer byte offset: always out of range
_MAX_SPAN = (1 << 31) - 16
#: Largest grid y (the batch dimension of a launch).
MAX_GRID_Y = 65535


def digits_span(digits: Digits) -> int:
    """Largest offset of the index space plus one (0 for an empty space)."""
    return sum((e - 1) * s for e, s in digits) + 1


def digits_extent(digits: Digits) -> int:
    return math.prod(e for e, _ in digits) if digits else 1


def mixed_offset(idx: int, digits: Digits) -> int:
    """Host reference of the device ``rowoff`` / ``koff`` functions."""
    off = 0
    for e, s in digits:
        off += (idx % e) * s
        idx //= e
    return off


def _norm(digits: Sequence[Sequence[int]]) -> Digits:
    out = tuple((int(e), int(s)) for e, s in digits if int(e) != 1)
    return out or ((1, 0),)


@dataclass(frozen=True)
class GettF64Problem:
    """The index maps of one launch.

    ``a_m`` / ``a_k`` / ``a_l``: digits of A's GEMM row (M), K and batch
    indices; likewise ``b_n`` / ``b_k`` / ``b_l`` for B, ``d_m`` / ``d_n`` /
    ``d_l`` for D and ``c_*`` for C (``None``: C has D's layout). ``M``, ``N``,
    ``K`` and ``L`` are the extents of those index spaces. A kernel bakes them
    in as constants, unless it is built with ``runtime_sizes`` (then this
    problem is only the layout template, see :func:`runtime_template`).
    ``dtype``: the word type of every operand, ``"f64"`` (f64 / complex f64)
    or ``"f32"`` (f32 / complex f32).
    """

    M: int
    N: int
    K: int
    a_m: Digits
    a_k: Digits
    b_n: Digits
    b_k: Digits
    d_m: Digits
    d_n: Digits
    L: int = 1
    a_l: Digits = ((1, 0),)
    b_l: Digits = ((1, 0),)
    d_l: Digits = ((1, 0),)
    c_m: Optional[Digits] = None
    c_n: Optional[Digits] = None
    c_l: Optional[Digits] = None
    complex_: bool = False
    conj_a: bool = False
    conj_b: bool = False
    dtype: str = "f64"

    def __post_init__(self):
        for name in ("a_m", "a_k", "b_n", "b_k", "d_m", "d_n", "a_l", "b_l", "d_l"):
            object.__setattr__(self, name, _norm(getattr(self, name)))
        for name in ("c_m", "c_n", "c_l"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _norm(getattr(self, name)))

    @property
    def word_bytes(self) -> int:
        """Bytes of one real word (an f64 or an f32)."""
        return 4 if self.dtype == "f32" else 8

    @property
    def elem_bytes(self) -> int:
        return self.word_bytes * (2 if self.complex_ else 1)

    def c_digits(self) -> Tuple[Digits, Digits, Digits]:
        return (
            self.c_m if self.c_m is not None else self.d_m,
            self.c_n if self.c_n is not None else self.d_n,
            self.c_l if self.c_l is not None else self.d_l,
        )

    def span_bytes(self, which: str) -> int:
        """Bytes from the first to past the last element ``which`` touches."""
        if which == "A":
            ds = (self.a_m, self.a_k, self.a_l)
        elif which == "B":
            ds = (self.b_n, self.b_k, self.b_l)
        elif which == "C":
            ds = self.c_digits()
        else:
            ds = (self.d_m, self.d_n, self.d_l)
        return (sum(digits_span(d) - 1 for d in ds) + 1) * self.elem_bytes

    def addr_span_bytes(self, which: str, tile_k: int) -> int:
        """Bytes one buffer resource of the kernel addresses for ``which``.

        Batch and (for K digits that split at ``tile_k``) K-tile offsets move
        the 64-bit base pointer instead, so they do not count.
        """
        if which in ("A", "B"):
            side, kd = (self.a_m, self.a_k) if which == "A" else (self.b_n, self.b_k)
            split = k_split(kd, tile_k)
            k_span = digits_span(split[0]) if split is not None else digits_span(kd)
            return (digits_span(side) - 1 + k_span - 1 + 1) * self.elem_bytes
        m, n, _ = self.c_digits() if which == "C" else (self.d_m, self.d_n, self.d_l)
        return (digits_span(m) - 1 + digits_span(n) - 1 + 1) * self.elem_bytes

    def check(self) -> Tuple[bool, str]:
        pairs = (
            ("M", self.M, (self.a_m, self.d_m)),
            ("N", self.N, (self.b_n, self.d_n)),
            ("K", self.K, (self.a_k, self.b_k)),
            ("L", self.L, (self.a_l, self.b_l, self.d_l)),
        )
        for label, ext, dl in pairs:
            if ext <= 0:
                return False, f"{label} must be positive, got {ext}"
            for d in dl:
                if digits_extent(d) != ext:
                    return False, f"{label} digits {d} do not cover extent {ext}"
        cm, cn, cl = self.c_digits()
        if (digits_extent(cm), digits_extent(cn), digits_extent(cl)) != (
            self.M,
            self.N,
            self.L,
        ):
            return False, "C digits do not cover (M, N, L)"
        if (self.conj_a or self.conj_b) and not self.complex_:
            return False, "conjugation needs complex data"
        if self.dtype not in GETT_DTYPES:
            return False, f"dtype must be one of {GETT_DTYPES}, got {self.dtype!r}"
        return True, "ok"


def gemm_problem(
    M: int,
    N: int,
    K: int,
    *,
    a_strides: Tuple[int, int],
    b_strides: Tuple[int, int],
    d_strides: Tuple[int, int],
    c_strides: Optional[Tuple[int, int]] = None,
    batch: int = 1,
    batch_strides: Tuple[int, int, int] = (0, 0, 0),
    c_batch_stride: Optional[int] = None,
    complex_: bool = False,
    conj_a: bool = False,
    conj_b: bool = False,
    dtype: str = "f64",
) -> GettF64Problem:
    """A (batched) strided GEMM ``D = alpha * A @ B^T + beta * C``.

    ``a_strides = (stride_m, stride_k)`` of ``A[M, K]``; ``b_strides =
    (stride_n, stride_k)`` of ``B[N, K]``; ``d_strides = (stride_m,
    stride_n)``. Any layout (row / column major, transposed) is a choice of
    strides. ``dtype``: ``"f64"`` or ``"f32"`` words.
    """
    c_strides = c_strides or d_strides
    cbs = batch_strides[2] if c_batch_stride is None else c_batch_stride
    return GettF64Problem(
        M=M,
        N=N,
        K=K,
        L=batch,
        a_m=((M, a_strides[0]),),
        a_k=((K, a_strides[1]),),
        b_n=((N, b_strides[0]),),
        b_k=((K, b_strides[1]),),
        d_m=((M, d_strides[0]),),
        d_n=((N, d_strides[1]),),
        a_l=((batch, batch_strides[0]),),
        b_l=((batch, batch_strides[1]),),
        d_l=((batch, batch_strides[2]),),
        c_m=((M, c_strides[0]),),
        c_n=((N, c_strides[1]),),
        c_l=((batch, cbs),),
        complex_=complex_,
        conj_a=conj_a,
        conj_b=conj_b,
        dtype=dtype,
    )


@dataclass(frozen=True)
class GettF64Spec:
    """One f64 / cf64 GETT kernel (f32 / cf32 with ``problem.dtype = "f32"``).

    Tiling knobs: ``tile_m`` x ``tile_n`` per CTA, ``warps_m`` x
    ``warps_n`` waves (each owns a ``tile_m / warps_m`` x ``tile_n / warps_n``
    block of 16x16 MFMA atoms), ``tile_k`` K rows per LDS stage. Other knobs:

    * ``lds_swizzle``: real data only -- K row ``k`` of an LDS stage stores
      column ``c`` at ``c ^ swz(k)``, ``swz(k) = 16*(k & 1) ^ 8*((k >> 2) & 7)``
      (masked to the 64- or 32-column block): the two K rows a half-wave of
      ``ds_read_b64`` reads land on disjoint banks without padding, and the
      K-contiguous loader's column writes (8 K rows of one column per
      instruction) spread over the banks too. Both CTA tile sides must be
      multiples of 32. Off: padding instead.
    * ``lds_swizzle_mode``: the swizzle function (with ``lds_swizzle``).
      0 is ``swz`` above. 1 is ``16*(k & 1) ^ 4*((k >> 1) & 3)`` and 2 is
      ``4*((k >> 1) & 7)``: the K-contiguous loader stores a K pair per lane,
      so the K rows ``2c`` one store instruction writes take distinct 32-byte
      bank groups (mode 2: all eight; mode 1: four, keeping the odd-row flip
      of mode 0 for the fragment reads).
    * ``lds_pad``: elements of padding per LDS row (``None``: 0 with the
      swizzle; otherwise chosen so the two K rows of a half-wave land on
      disjoint banks).
    * ``group_m``: CTA raster -- consecutive CTAs walk ``group_m`` M tiles
      before moving to the next N tile (L2 reuse of the B panel); 1 = row
      major.
    * ``xcds``: remap the CTA id so the CTAs one chiplet runs are consecutive
      tiles (0 = off; 8 for the 8-XCD parts).
    * ``waves_per_eu``: occupancy hint (``None`` = compiler default).
    * ``bilinear``: read C (``beta * C``); otherwise D = alpha * acc.
    * ``prefetch``: issue the next K tile's global loads before the MFMAs of
      the current one (False: load after, the single-buffer reference
      schedule kept as a knob).
    * ``sched_fence``: scheduling barriers around the MFMA block, so the
      prefetch loads stay above it and the LDS stores (which wait for those
      loads) stay below it. Without them the backend hoists the stores into
      the MFMA block and the load latency is exposed every K tile.
    * ``prefetch_depth``: K tiles the register prefetch runs ahead of the
      MFMAs (with ``prefetch``). 1 loads tile ``kt + 1`` during tile ``kt``;
      2 loads tile ``kt + 2`` and carries tile ``kt + 1`` in registers through
      the loop, so a global load has two MFMA blocks to land (the 64 KiB-LDS
      parts, whose K tiles are short).
    * ``k_stagger``: 0 (default) walks the K tiles in order. ``n > 0``
      starts CTA ``bid`` at K tile ``((bid % 8) * n) % nk`` and wraps around
      (Tensile's StaggerU): concurrently running CTAs then read different
      K offsets, so operands with power-of-two strides do not pile onto the
      same memory channel. It reorders the K summation per CTA, so results
      are no longer bit-identical across tile configurations (still the f64
      error bound).
    * ``unroll2``: unroll the K loop by two K tiles (two LDS stages,
      prefetch depth 1), so each half reads a compile-time LDS stage: the
      fragment addresses become loop-invariant registers plus immediate
      offsets (no per-tile address arithmetic) at one barrier per K tile.
    * ``complex_accs``: complex only. 2 (default) keeps a real and an
      imaginary accumulator per atom and negates the fragment a product
      subtracts (one ``v_xor`` per k-step and atom). 4 keeps the four real
      products ``ar*br``, ``ai*bi``, ``ar*bi``, ``ai*br`` in their own
      accumulators and applies every sign (subtraction, CONJ) once in the
      epilogue: no negation in the K loop, twice the accumulator registers.
    * ``lds_stages``: 2 (default) double-buffers LDS, one barrier per K
      tile; 1 keeps a single LDS stage (the register prefetch still hides
      the global loads) at the cost of a second barrier per K tile, which
      halves the LDS of a CTA: a ``tile_k`` twice as deep at the same
      occupancy (K-contiguous rows of 128 bytes on the 64 KiB-LDS parts).
    * ``k_block``: 0 (default) accumulates the whole K range in one chain of
      MFMAs. ``n > 0`` restarts the accumulators every ``n`` K tiles and adds
      each block's sum into a second set of f64 accumulators, so an output
      goes through about ``n * tile_k + K / (n * tile_k)`` roundings instead of
      ``K`` (the error at long K drops accordingly; the bits change). It holds
      twice the accumulator registers. With ``unroll2``, ``n`` must be even.
    * ``runtime_sizes``: take ``M, N, K, L`` and every stride as kernel
      arguments instead of baking them in (one digit per index space: a
      strided batched GEMM). ``problem`` is then the layout template
      (:func:`runtime_template`); launch with :func:`gett_f64_runtime_args`
      and :func:`gett_f64_grid` of the actual problem. With ``unroll2`` and an
      odd K-tile count the last pair's second half multiplies zero tiles,
      which can turn an exact ``-0.0`` result into ``+0.0``.
    * ``split_k``: 1 (default) computes each output tile in one CTA. ``n > 1``
      splits the K tiles into ``n`` near-equal contiguous ranges, one CTA per
      (output tile, range) on grid z: the kernel then writes its partial sums
      to a workspace ``W`` (f64, ``[n][L][M][N]``, :func:`gett_f64_workspace_bytes`)
      and takes no C, alpha or beta; :func:`build_gett_f64_reduce` adds the
      ``n`` partials in order and applies the epilogue. Deterministic; an output
      goes through about ``K/n + n`` roundings instead of ``K``. For problems
      with few output tiles per CU and a long K.
    * ``acc_agpr``: keep the MFMA accumulators in AGPRs (Python engine only).
      Reserves exactly the accumulator registers as AGPRs
      (``amdgpu-agpr-alloc``), forces the AGPR form of the MFMA, and seeds
      every accumulator from an opaque AGPR zero so the loop-carried
      accumulators stay whole vectors. Lets a wave hold up to 256
      accumulator registers (a 128x64 wave tile) beside its fragments and
      prefetch registers without spilling.
    * ``split_serial``: with ``split_k``, reduce the partial sums inside the
      same launch instead of a second (reduce) kernel (Python engine only).
      The kernel then takes ``C``, ``D``, alpha and beta as without split_k,
      plus ``W`` (f64 partial slots, :func:`gett_f64_workspace_bytes`) and
      ``Cnt`` (i32 counters, :func:`gett_f64_counter_count`, zeroed once when
      allocated). Every wave stores its partial accumulators into its slot
      (lane-coalesced 16-byte stores), releases them (agent-scope fence) and
      bumps its counter; the wave that arrives last acquires, sums the
      ``split_k`` partials in the fixed order z = 0 .. split_k - 1 (the bits
      do not depend on the arrival order and equal the reduce kernel's),
      resets the counter to 0 and runs the epilogue. 1: agent-scope release
      / acquire fences around the counter (each writes back / invalidates the
      L2); 2 (gfx942 / gfx950): write-through partial stores and loads
      (``sc1``, they bypass the per-XCD L2; completed by ``s_waitcnt
      vmcnt(0)``) and no L2 fences.
    * ``frag_pipe``: software-pipeline the LDS fragments (Python engine
      only). The fragments of k-step ``ks + 1`` are read into a second set of
      registers before the MFMAs of k-step ``ks`` issue, and the LDS store of
      the next K tile plus its barrier move in front of the last k-step, whose
      MFMAs then overlap the reads of the next tile's first k-step: no MFMA
      waits on an LDS read inside the loop. Needs two LDS stages, prefetch
      depth 1 and no ``k_block``; ``unroll2`` is supported (compile-time
      stages).
    * ``k_inner``: real data only (Python engine only). 1 (default) issues a
      K tile's MFMAs k-step by k-step (every accumulator once per k-step).
      ``n > 1`` reads the fragments of ``n`` k-steps first and then issues
      the ``n`` MFMAs of each accumulator back to back (the K order of every
      output is unchanged, so the bits are too): a chain of MFMAs on one
      accumulator keeps the matrix pipe fed better than a round robin over
      many accumulators. ``n`` divides ``tile_k / 4``; with ``frag_pipe`` the
      groups of ``n`` k-steps are what the pipeline prefetches.
    * ``mfma_fence``: a scheduling barrier after every MFMA of the K loop, so
      the backend keeps the emitted MFMA order (Python engine only).
    * ``ilv``: (Python engine only) 0 (default) off. ``n > 0``
      emits the K loop as one instruction-level schedule per K tile: the
      tile's MFMAs in order with, between them, the fragment reads of the
      next k-step group (spread over the current group), the LDS stores of
      the next K tile (it rides in registers, ``prefetch_depth`` 2) each
      followed by the global load of the tile after it into the freed
      registers (spread evenly up to ``ilv_wgap`` MFMAs before the barrier),
      and the barrier ``n`` MFMAs before the end of the tile; the reads of
      the next tile's first k-step group follow the barrier, between the last
      ``n`` MFMAs. A scheduling barrier after every MFMA keeps the order, so
      no MFMA waits for an LDS read or a global load and no burst of loads or
      stores drains the matrix pipe (a hand-scheduled loop). Needs two LDS
      stages, prefetch depth 2, no ``k_block`` / ``k_stagger``; complex
      data schedule each fused 4M atom (its four MFMAs) as one slot; with
      ``unroll2`` (compile-time sizes) two K tiles per iteration read
      compile-time LDS stages.
    * ``ilv_wgap``: MFMAs between the last LDS store of a tile and its
      barrier (``ilv``).
    * ``kmajor_lds``: real data only (Python engine only). An operand the
      loader reads along K (K-contiguous in memory) gets a K-major LDS tile,
      ``[stage * side + mn][k]`` with ``kmajor_pad`` elements of padding per
      row, instead of the ``[k][mn]`` tile: every 16-byte load is one 16-byte
      LDS store (no scatter into two K rows), and with the padded pitch the
      fragment reads of a half-wave (16 rows, one K) and of its partner
      half-wave (the next K) land on disjoint banks. The swizzle does not
      apply to such an operand. The K order of every output is unchanged.
    * ``kmajor_pad``: elements of padding per K-major LDS row (even, so rows
      stay 16-byte aligned).
    """

    problem: GettF64Problem
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 16
    warps_m: int = 2
    warps_n: int = 2
    lds_pad: Optional[int] = None
    lds_swizzle: bool = True
    lds_swizzle_mode: int = 0
    group_m: int = 8
    xcds: int = 0
    waves_per_eu: Optional[int] = None
    bilinear: bool = True
    prefetch: bool = True
    sched_fence: bool = True
    lds_stages: int = 2
    prefetch_depth: int = 1
    complex_accs: int = 2
    unroll2: bool = False
    k_stagger: int = 0
    name: str = "rocke_gett_f64"
    k_block: int = 0
    runtime_sizes: bool = False
    split_k: int = 1
    acc_agpr: bool = False
    split_serial: int = 0
    frag_pipe: bool = False
    k_inner: int = 1
    mfma_fence: bool = False
    ilv: int = 0
    ilv_wgap: int = 4
    kmajor_lds: bool = False
    kmajor_pad: int = 2

    @property
    def threads(self) -> int:
        return 64 * self.warps_m * self.warps_n

    @property
    def wave_m(self) -> int:
        return self.tile_m // self.warps_m

    @property
    def wave_n(self) -> int:
        return self.tile_n // self.warps_n

    def uses_swizzle(self) -> bool:
        return (
            self.lds_swizzle
            and not self.problem.complex_
            and self.tile_m % 32 == 0
            and self.tile_n % 32 == 0
        )

    def swizzle_mask(self, side: int) -> int:
        """Column bits the swizzle may flip for an LDS row of ``side`` elements."""
        return 63 if side % 64 == 0 else 31

    def resolved_pad(self) -> int:
        """LDS row padding in words (complex rows hold 2 words per element)."""
        if self.lds_pad is not None:
            return self.lds_pad
        if self.problem.complex_ or self.uses_swizzle():
            # complex: ds_read_b128, a quarter-wave touches one 256-byte row
            return 0
        if self.problem.dtype == "f32":
            # A wave of ds_read_b32 reads 16 words of four consecutive K rows:
            # make both row pitches 64 bytes off a multiple of 128 bytes, so
            # the four 64-byte row pieces take distinct bank groups.
            for pad in (0, 16, 8, 24):
                if all(((t + pad) * 4) % 128 == 64 for t in (self.tile_m, self.tile_n)):
                    return pad
            return 16
        # A half-wave of ds_read_b64 reads 16 words of two consecutive K rows:
        # make both row pitches odd multiples of 128 bytes.
        for pad in (0, 16, 8, 24):
            if all(((t + pad) * 8) % 256 == 128 for t in (self.tile_m, self.tile_n)):
                return pad
        return 16

    def kernel_name(self) -> str:
        p = self.problem
        if self.runtime_sizes:
            # The template's layout class names the kernel; sizes are arguments.
            tp = runtime_template(p)
            sizes, h = "rt", _problem_hash(tp) if tp is not None else "invalid"
        else:
            sizes, h = f"M{p.M}N{p.N}K{p.K}L{p.L}", _problem_hash(p)
        if p.dtype == "f32":
            dt = "c64" if p.complex_ else "f32"
        else:
            dt = "c128" if p.complex_ else "f64"
        return kernel_name_join(
            self.name,
            dt,
            sizes,
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warps_m}x{self.warps_n}",
            f"g{self.group_m}",
            f"x{self.xcds}" if self.xcds else "",
            f"p{self.lds_pad}" if self.lds_pad is not None else "",
            "" if self.uses_swizzle() or p.complex_ else "nsw",
            (
                f"z{self.lds_swizzle_mode}"
                if self.uses_swizzle() and self.lds_swizzle_mode
                else ""
            ),
            f"e{self.waves_per_eu}" if self.waves_per_eu else "",
            f"s{self.lds_stages}" if self.lds_stages != 2 else "",
            f"d{self.prefetch_depth}" if self.prefetch_depth != 1 else "",
            f"a{self.complex_accs}" if p.complex_ and self.complex_accs != 2 else "",
            f"k{self.k_stagger}" if self.k_stagger else "",
            f"b{self.k_block}" if self.k_block else "",
            f"sk{self.split_k}" if self.split_k > 1 else "",
            f"h{h}",
            flags={
                "bil": self.bilinear,
                "nopf": not self.prefetch,
                "nofence": not self.sched_fence,
                "u2": self.unroll2,
                "agpr": self.acc_agpr,
                f"sks{self.split_serial}": bool(self.split_serial),
                "fp": self.frag_pipe,
                f"ki{self.k_inner}": self.k_inner != 1,
                "mf": self.mfma_fence,
                f"il{self.ilv}": self.ilv > 0,
                f"iw{self.ilv_wgap}": self.ilv > 0 and self.ilv_wgap != 4,
                "km": self.kmajor_lds,
                f"kp{self.kmajor_pad}": self.kmajor_lds and self.kmajor_pad != 2,
                "cja": p.conj_a,
                "cjb": p.conj_b,
            },
        )


def _digits_key(digits: Digits) -> str:
    return ",".join(f"{e}:{s}" for e, s in digits)


def _problem_hash(p: GettF64Problem) -> str:
    """10 hex digits of FNV-1a 64 over the index maps (the C++ mirror hashes
    the same text)."""
    cm, cn, cl = p.c_digits()
    key = "|".join(
        _digits_key(d)
        for d in (
            p.a_m,
            p.a_k,
            p.a_l,
            p.b_n,
            p.b_k,
            p.b_l,
            p.d_m,
            p.d_n,
            p.d_l,
            cm,
            cn,
            cl,
        )
    )
    h = 0xCBF29CE484222325
    for byte in key.encode():
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"[:10]


# ------------------------------------------------------- runtime sizes


def runtime_template(problem: GettF64Problem) -> Optional[GettF64Problem]:
    """The layout class of a strided batched GEMM, as a canonical problem.

    A ``runtime_sizes`` kernel depends on the problem only through: which
    index strides are 1 (those offsets need no multiply, and the vector loads
    run along them), the parity of an extent an operand is contiguous along (a
    two-element vector load needs an even run), complex data, conjugation and
    the word type. The template keeps exactly that: extent 3 for an odd run an
    operand loads along, 2 otherwise, stride 1 or 2, ``L = 2`` with stride 2,
    and C's digits spelled out. Real f32 data also keep whether such a run is
    a multiple of 4 (a four-element load): extent 4. Problems with the same
    template share one kernel. None when an index space has more than one
    digit.
    """
    p = problem
    cm, cn, cl = p.c_digits()
    spaces = (p.a_m, p.a_k, p.a_l, p.b_n, p.b_k, p.b_l, p.d_m, p.d_n, p.d_l, cm, cn, cl)
    if any(len(d) != 1 for d in spaces):
        return None

    def unit(d: Digits) -> bool:
        return d[0][1] == 1 and d[0][0] > 1

    quad = p.dtype == "f32" and not p.complex_

    def parity(extent: int, *runs: Digits) -> int:
        loaded = any(unit(d) for d in runs)
        if quad and loaded and extent % 4 == 0:
            return 4
        return 3 if loaded and extent % 2 else 2

    ext = {
        "M": parity(p.M, p.a_m),
        "N": parity(p.N, p.b_n),
        "K": parity(p.K, p.a_k, p.b_k),
    }

    def dig(d: Digits, space: str) -> Digits:
        return ((ext[space], 1 if unit(d) else 2),)

    batch = ((2, 2),)
    return GettF64Problem(
        M=ext["M"],
        N=ext["N"],
        K=ext["K"],
        L=2,
        a_m=dig(p.a_m, "M"),
        a_k=dig(p.a_k, "K"),
        b_n=dig(p.b_n, "N"),
        b_k=dig(p.b_k, "K"),
        d_m=dig(p.d_m, "M"),
        d_n=dig(p.d_n, "N"),
        a_l=batch,
        b_l=batch,
        d_l=batch,
        c_m=dig(cm, "M"),
        c_n=dig(cn, "N"),
        c_l=batch,
        complex_=p.complex_,
        conj_a=p.conj_a,
        conj_b=p.conj_b,
        dtype=p.dtype,
    )


def _stride(d: Digits) -> int:
    return int(d[0][1])


def runtime_compatible(
    spec: "GettF64Spec", problem: GettF64Problem
) -> Tuple[bool, str]:
    """Whether the ``runtime_sizes`` kernel of ``spec`` computes ``problem``."""
    if not spec.runtime_sizes:
        return False, "the spec bakes its sizes in (runtime_sizes is off)"
    ok, why = problem.check()
    if not ok:
        return False, why
    tp = runtime_template(problem)
    if tp is None:
        return False, "runtime_sizes needs one digit per index space (a strided GEMM)"
    if tp != runtime_template(spec.problem):
        return False, "the problem's layout class differs from the kernel's template"
    cm, cn, cl = problem.c_digits()
    for d in (problem.a_m, problem.a_k, problem.a_l, problem.b_n, problem.b_k):
        if _stride(d) < 0:
            return False, "runtime_sizes needs non-negative strides"
    for d in (problem.b_l, problem.d_m, problem.d_n, problem.d_l, cm, cn, cl):
        if _stride(d) < 0:
            return False, "runtime_sizes needs non-negative strides"
    for which in ("A", "B", "C", "D"):
        span = problem.addr_span_bytes(which, spec.tile_k)
        if span > _MAX_SPAN:
            return False, f"{which} addresses {span} bytes per buffer; split the launch"
    if problem.L > MAX_GRID_Y:
        return (
            False,
            f"L = {problem.L} exceeds the grid limit {MAX_GRID_Y}; split the batch",
        )
    if spec.split_k > 1 and -(-problem.K // spec.tile_k) < spec.split_k:
        return False, "split_k needs at least one K tile per split"
    return True, "ok"


#: Kernel arguments a ``runtime_sizes`` kernel takes after alpha / beta:
#: (name, dtype) in order. The C strides follow only when the kernel is
#: bilinear (``c_sm``, ``c_sn`` after ``d_sn``; ``c_sl`` last).
RUNTIME_SIZE_ARGS = (
    ("M", "i32"),
    ("N", "i32"),
    ("K", "i32"),
    ("L", "i32"),
    ("a_sm", "i32"),
    ("a_sk", "i32"),
    ("b_sn", "i32"),
    ("b_sk", "i32"),
    ("d_sm", "i32"),
    ("d_sn", "i32"),
    ("c_sm", "i32"),
    ("c_sn", "i32"),
    ("a_sl", "i64"),
    ("b_sl", "i64"),
    ("d_sl", "i64"),
    ("c_sl", "i64"),
)


def _runtime_arg_names(spec: "GettF64Spec") -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (n, t)
        for n, t in RUNTIME_SIZE_ARGS
        if spec.bilinear or n not in ("c_sm", "c_sn", "c_sl")
    )


def gett_f64_runtime_args(spec: "GettF64Spec", problem: GettF64Problem) -> dict:
    """The size and stride kernel arguments of ``problem`` for a
    ``runtime_sizes`` kernel (raises when it cannot compute it)."""
    ok, why = runtime_compatible(spec, problem)
    if not ok:
        raise ValueError(f"gett_f64 runtime kernel cannot run this problem: {why}")
    p = problem
    cm, cn, cl = p.c_digits()
    vals = {
        "M": p.M,
        "N": p.N,
        "K": p.K,
        "L": p.L,
        "a_sm": _stride(p.a_m),
        "a_sk": _stride(p.a_k),
        "b_sn": _stride(p.b_n),
        "b_sk": _stride(p.b_k),
        "d_sm": _stride(p.d_m),
        "d_sn": _stride(p.d_n),
        "c_sm": _stride(cm),
        "c_sn": _stride(cn),
        "a_sl": _stride(p.a_l),
        "b_sl": _stride(p.b_l),
        "d_sl": _stride(p.d_l),
        "c_sl": _stride(cl),
    }
    return {n: vals[n] for n, _ in _runtime_arg_names(spec)}


# --------------------------------------------------------------- validity


def k_split(digits: Digits, tile: int):
    """Split the K index space so ``koff(kt*T + kl) = ktile(kt) + klocal(kl)``.

    Returns ``(local_digits, tile_digits)`` (each over its own index) or None
    when the tile does not divide the digit structure. A single digit always
    splits (a K tail is masked); several digits split when the inner extents
    multiply into ``tile`` with the last one divided evenly.
    """
    if len(digits) == 1:
        e, s = digits[0]
        return ((tile, s),), ((max(1, -(-e // tile)), tile * s),)
    rem = tile
    local: List[Tuple[int, int]] = []
    tiled: List[Tuple[int, int]] = []
    for e, s in digits:
        if rem == 1:
            tiled.append((e, s))
        elif e <= rem:
            if rem % e:
                return None
            local.append((e, s))
            rem //= e
        else:
            if e % rem:
                return None
            local.append((rem, s))
            tiled.append((e // rem, rem * s))
            rem = 1
    if rem != 1:
        return None
    return tuple(local), tuple(tiled) or ((1, 0),)


def _contiguous(digits: Digits) -> int:
    """Run length along the index with element stride 1 (0 if not unit stride)."""
    e, s = digits[0]
    return e if s == 1 else 0


@dataclass(frozen=True)
class _LoadPlan:
    along_k: bool  # vector direction (True: along K, False: along M/N)
    vec: int  # elements per load
    cpl: int  # chunks per line (threads covering one line)
    lpp: int  # lines per pass
    passes: int


def _load_plan(
    spec: GettF64Spec, side_digits: Digits, k_digits: Digits, side_tile: int
):
    p = spec.problem
    nt = spec.threads
    if p.complex_:
        cands = [(False, 1)] if _contiguous(side_digits) else []
        cands += [(True, 1)] if _contiguous(k_digits) else []
        cands += [(False, 1)]
    else:
        cands = []
        run_s, run_k = _contiguous(side_digits), _contiguous(k_digits)
        if p.dtype == "f32":
            # 16-byte loads: four f32
            if run_s and run_s % 4 == 0:
                cands.append((False, 4))
            if run_k and run_k % 4 == 0 and spec.tile_k % 4 == 0:
                cands.append((True, 4))
        if run_s and run_s % 2 == 0:
            cands.append((False, 2))
        if run_k and run_k % 2 == 0 and spec.tile_k % 2 == 0:
            cands.append((True, 2))
        if run_s:
            cands.append((False, 1))
        if run_k:
            cands.append((True, 1))
        cands.append((False, 1))
    for along_k, vec in cands:
        line = spec.tile_k if along_k else side_tile
        other = side_tile if along_k else spec.tile_k
        if line % vec:
            continue
        cpl = line // vec
        if cpl > nt or nt % cpl:
            continue
        lpp = nt // cpl
        if other % lpp:
            continue
        return _LoadPlan(along_k, vec, cpl, lpp, other // lpp)
    return None


def _kmajor(
    spec: GettF64Spec, side_digits: Digits, k_digits: Digits, side: int
) -> bool:
    """``kmajor_lds`` applies to this operand (its loader reads along K)."""
    if not spec.kmajor_lds:
        return False
    pl = _load_plan(spec, side_digits, k_digits, side)
    return pl is not None and pl.along_k


def _lds_shape(spec: GettF64Spec, side_digits: Digits, k_digits: Digits, side: int):
    """``[rows, cols]`` of an operand's LDS tile (words)."""
    words = 2 if spec.problem.complex_ else 1
    if _kmajor(spec, side_digits, k_digits, side):
        return [spec.lds_stages * side, spec.tile_k + spec.kmajor_pad]
    return [spec.lds_stages * spec.tile_k, side * words + spec.resolved_pad()]


def is_valid_spec(spec: GettF64Spec, arch: str = "gfx950") -> Tuple[bool, str]:
    p = spec.problem
    ok, why = p.check()
    if not ok:
        return False, why
    if spec.runtime_sizes:
        tp = runtime_template(p)
        if tp is None:
            return (
                False,
                "runtime_sizes needs one digit per index space (a strided GEMM)",
            )
        # The kernel depends on the problem only through its layout class.
        spec = replace(spec, problem=tp)
        p = tp
    atom = gett_atom(arch, p.dtype)
    if atom is None:
        return False, f"{arch!r} has no 16x16x4 {p.dtype} MMA atom in its catalog"
    if p.dtype == "f32":
        for lever, on in (
            ("split_k", spec.split_k > 1),
            ("split_serial", bool(spec.split_serial)),
            ("acc_agpr", spec.acc_agpr),
            ("frag_pipe", spec.frag_pipe),
            ("k_inner", spec.k_inner != 1),
            ("mfma_fence", spec.mfma_fence),
            ("ilv", spec.ilv > 0),
            ("kmajor_lds", spec.kmajor_lds),
        ):
            if on:
                return False, f"{lever} is not built for f32 data"
    if not spec.runtime_sizes and p.L > MAX_GRID_Y:
        return False, f"L = {p.L} exceeds the grid limit {MAX_GRID_Y}; split the batch"
    if spec.warps_m * spec.warps_n not in (1, 2, 4, 8):
        return False, "warps_m * warps_n must be 1, 2, 4 or 8"
    if spec.tile_m % (atom.m * spec.warps_m) or spec.tile_n % (atom.n * spec.warps_n):
        return False, "each wave tile must be a whole number of 16x16 atoms"
    if spec.tile_k % atom.k or spec.tile_k <= 0:
        return False, "tile_k must be a positive multiple of 4"
    if spec.k_block < 0:
        return False, "k_block must be >= 0"
    if spec.split_k < 1 or spec.split_k > MAX_GRID_Y:
        return False, f"split_k must be in [1, {MAX_GRID_Y}]"
    if spec.kmajor_lds and (
        p.complex_ or spec.kmajor_pad < 0 or (spec.tile_k + spec.kmajor_pad) % 2
    ):
        return False, "kmajor_lds needs real data and an even tile_k + kmajor_pad"
    if spec.ilv < 0 or spec.ilv_wgap < 0:
        return False, "ilv and ilv_wgap must be >= 0"
    if spec.ilv and (
        spec.lds_stages != 2
        or spec.prefetch_depth != 2
        or not spec.prefetch
        or spec.k_block
        or spec.k_stagger
        or spec.frag_pipe
        or (p.complex_ and spec.k_inner != 1)
    ):
        return (
            False,
            "ilv needs two LDS stages, prefetch depth 2, no k_block / k_stagger / "
            "frag_pipe (complex: k_inner 1)",
        )
    if spec.k_inner < 1 or (spec.tile_k // 4) % spec.k_inner:
        return False, "k_inner must divide the k-steps of a K tile (tile_k / 4)"
    if spec.k_inner > 1 and p.complex_:
        return False, "k_inner needs real data"
    if spec.frag_pipe and (
        spec.lds_stages != 2
        or spec.prefetch_depth != 1
        or not spec.prefetch
        or spec.k_block
        or spec.k_stagger
    ):
        return (
            False,
            "frag_pipe needs two LDS stages, prefetch depth 1, no k_block / k_stagger",
        )
    if spec.split_serial not in (0, 1, 2):
        return False, "split_serial must be 0, 1 or 2"
    if spec.split_serial and (spec.split_k < 2 or spec.runtime_sizes):
        return False, "split_serial needs split_k > 1 and compile-time sizes"
    if spec.split_serial == 2 and arch not in ("gfx942", "gfx950"):
        return False, "split_serial 2 (write-through partials) needs gfx942 / gfx950"
    if spec.split_k > 1 and (spec.unroll2 or spec.k_stagger):
        return False, "split_k does not combine with unroll2 or k_stagger"
    if (
        spec.split_k > 1
        and not spec.runtime_sizes
        and -(-p.K // spec.tile_k) < spec.split_k
    ):
        return False, "split_k needs at least one K tile per split"
    if spec.k_block and spec.unroll2 and spec.k_block % 2:
        return False, "k_block must be even with unroll2"
    if spec.group_m < 1:
        return False, "group_m must be >= 1"
    if spec.xcds < 0:
        return False, "xcds must be >= 0"
    if spec.lds_swizzle_mode not in (0, 1, 2):
        return False, "lds_swizzle_mode must be 0, 1 or 2"
    if spec.lds_stages not in (1, 2):
        return False, "lds_stages must be 1 or 2"
    if spec.prefetch_depth not in (1, 2):
        return False, "prefetch_depth must be 1 or 2"
    if spec.prefetch_depth == 2 and not spec.prefetch:
        return False, "prefetch_depth 2 needs prefetch"
    if spec.complex_accs not in (2, 4):
        return False, "complex_accs must be 2 or 4"
    if spec.k_stagger < 0:
        return False, "k_stagger must be >= 0"
    if spec.unroll2 and (
        spec.lds_stages != 2
        or spec.prefetch_depth != (2 if spec.ilv else 1)
        or not spec.prefetch
    ):
        return False, "unroll2 needs two LDS stages and prefetch depth 1 (ilv: 2)"
    if spec.unroll2 and spec.ilv and spec.runtime_sizes:
        return False, "ilv with unroll2 needs compile-time sizes"
    acc_words = (
        (spec.wave_m // atom.m)
        * (spec.wave_n // atom.n)
        * atom.c_frag_len
        * (spec.complex_accs if p.complex_ else 1)
        * (2 if spec.k_block else 1)
    )
    dw = p.word_bytes // 4  # registers per accumulator word
    if acc_words * dw > 384:
        return False, f"accumulator needs {acc_words * dw} registers per lane (max 384)"
    if spec.acc_agpr and acc_words * 2 > 256:
        return (
            False,
            f"acc_agpr: accumulator needs {acc_words * 2} AGPRs per lane (max 256)",
        )
    # A CTA of more than four waves puts several waves on one SIMD: each gets
    # a share of the 512 registers, and the AGPR reservation must leave room
    # for the fragments and addresses (at least 64 VGPRs).
    per_simd = -(-(spec.warps_m * spec.warps_n) // 4)
    if spec.acc_agpr and acc_words * 2 > 512 // per_simd - 64:
        return (
            False,
            f"acc_agpr: {acc_words * 2} AGPRs leave too few VGPRs with "
            f"{per_simd} waves per SIMD",
        )
    for which in ("A", "B", "C", "D"):
        span = p.addr_span_bytes(which, spec.tile_k)
        if span > _MAX_SPAN:
            return False, f"{which} addresses {span} bytes per buffer; split the launch"
    if _load_plan(spec, p.a_m, p.a_k, spec.tile_m) is None:
        return False, "no thread mapping for the A tile"
    if _load_plan(spec, p.b_n, p.b_k, spec.tile_n) is None:
        return False, "no thread mapping for the B tile"
    words = 2 if p.complex_ else 1
    lds = (
        spec.lds_stages
        * spec.tile_k
        * ((spec.tile_m + spec.tile_n) * words + 2 * spec.resolved_pad())
        * p.word_bytes
    )
    if spec.kmajor_lds:
        lds = sum(
            _lds_shape(spec, side_d, k_d, side)[0]
            * _lds_shape(spec, side_d, k_d, side)[1]
            * 8
            for side_d, k_d, side in (
                (p.a_m, p.a_k, spec.tile_m),
                (p.b_n, p.b_k, spec.tile_n),
            )
        )
    cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    if lds > cap:
        return False, f"LDS {lds} bytes exceeds {cap} on {arch}"
    return True, "ok"


# ------------------------------------------------------------- the kernel


def _ci(b: IRBuilder, x) -> Value:
    """An i32 operand: a compile-time int becomes a constant, a Value stays."""
    return b.const_i32(x) if isinstance(x, int) else x


def _wty(p: GettF64Problem):
    """The IR type of one real word of the problem's data."""
    return F32 if p.dtype == "f32" else F64


def _wconst(b: IRBuilder, p: GettF64Problem, x: float) -> Value:
    """A constant word of the problem's data type."""
    return b.const_f32(x) if p.dtype == "f32" else b.const_f64(x)


def _mixed(b: IRBuilder, idx: Value, digits) -> Value:
    """Element offset of index ``idx`` (i32) under ``digits`` (device side).

    A ``runtime_sizes`` index space is one digit whose stride is an i32 kernel
    argument (or the int 1 for a unit stride); its extent is not used.
    """
    off = None
    rest = idx
    last = len(digits) - 1
    for i, (e, s) in enumerate(digits):
        if isinstance(s, int) and (s == 0 or e == 1):
            if i < last:
                rest = b.div(rest, b.const_i32(e))
            continue
        d = rest if i == last else b.mod(rest, b.const_i32(e))
        term = d if (isinstance(s, int) and s == 1) else b.mul(d, _ci(b, s))
        off = term if off is None else b.add(off, term)
        if i < last:
            rest = b.div(rest, b.const_i32(e))
    return off if off is not None else b.const_i32(0)


def _mixed64(b: IRBuilder, idx: Value, digits: Digits, scale: int) -> Value:
    """``scale * offset(idx)`` under ``digits`` as an i64 (64-bit base offsets)."""
    off = None
    rest = idx
    last = len(digits) - 1
    for i, (e, s) in enumerate(digits):
        if s == 0 or e == 1:
            if i < last:
                rest = b.div(rest, b.const_i32(e))
            continue
        d = rest if i == last else b.mod(rest, b.const_i32(e))
        term = b.mul(b.zext(d, I64), b.const_i64(s * scale))
        off = term if off is None else b.add(off, term)
        if i < last:
            rest = b.div(rest, b.const_i32(e))
    return off if off is not None else b.const_i64(0)


def _rt_span(b: IRBuilder, terms, es: int) -> Value:
    """``(sum (extent - 1) * stride + 1) * es`` as an i32 (runtime sizes).

    ``terms``: ``(extent_minus_one, stride)`` pairs, each an int or a Value; a
    stride of the int 1 needs no multiply.
    """
    acc = None
    for em1, s in terms:
        if isinstance(s, int) and s == 1:
            t = _ci(b, em1)
        else:
            t = b.mul(_ci(b, em1), _ci(b, s))
        acc = t if acc is None else b.add(acc, t)
    acc = b.add(acc, b.const_i32(1))
    return b.mul(acc, b.const_i32(es))


class _Geo:
    """Sizes and index maps the kernel addresses with.

    Compile-time: the problem's ints and digits. ``runtime_sizes``: the
    ``M, N, K, L`` kernel arguments, and one-digit index maps whose strides
    are kernel arguments (the int 1 where the template's stride is 1).
    """

    def __init__(self, b: IRBuilder, spec: GettF64Spec, rt_args: Optional[dict]):
        p = spec.problem
        self.rt = rt_args is not None
        if not self.rt:
            self.M, self.N, self.K = p.M, p.N, p.K
            self.a_m, self.a_k, self.b_n, self.b_k = p.a_m, p.a_k, p.b_n, p.b_k
            self.d_m, self.d_n = p.d_m, p.d_n
            self.c_m, self.c_n, _ = p.c_digits()
            return
        a = rt_args
        self.M, self.N, self.K, self.L = a["M"], a["N"], a["K"], a["L"]
        cm, cn, _ = p.c_digits()

        def one(d: Digits, name: str):
            return ((None, 1 if d[0][1] == 1 else a.get(name)),)

        self.a_m, self.a_k = one(p.a_m, "a_sm"), one(p.a_k, "a_sk")
        self.b_n, self.b_k = one(p.b_n, "b_sn"), one(p.b_k, "b_sk")
        self.d_m, self.d_n = one(p.d_m, "d_sm"), one(p.d_n, "d_sn")
        self.c_m, self.c_n = one(cm, "c_sm"), one(cn, "c_sn")
        self.batch = {k: a.get(k) for k in ("a_sl", "b_sl", "d_sl", "c_sl")}

    def stride(self, digits) -> object:
        """The stride of a one-digit runtime index map."""
        return digits[0][1]


class _Operand:
    """Device-side loader of one operand tile into registers and LDS."""

    def __init__(
        self,
        b,
        spec,
        name,
        base_ptr,
        side_digits,
        k_digits,
        side_tile,
        side_base,
        side_extent,
        lds,
        geo: Optional[_Geo] = None,
    ):
        self.b = b
        self.spec = spec
        p = spec.problem
        self.cx = p.complex_
        self.es = p.elem_bytes
        self.wty = _wty(p)
        self.dw = p.word_bytes // 4  # dwords per word
        self.base_ptr = base_ptr  # operand pointer moved by the batch offset
        self.k_digits = k_digits
        self.side_tile = side_tile
        self.lds = lds
        self.kmap = None  # K-tile index remap (k_stagger), set by the builder
        self.nk = None  # K-tile count (int or Value), set with kmap
        self.K = geo.K if geo is not None else p.K
        self.rt = geo is not None and geo.rt
        tmpl_side = p.a_m if name == "A" else p.b_n
        tmpl_k = p.a_k if name == "A" else p.b_k
        self.plan = _load_plan(spec, tmpl_side, tmpl_k, side_tile)
        self.kmajor = _kmajor(spec, tmpl_side, tmpl_k, side_tile)
        if self.rt:
            # One digit per space: the K tile always splits; the per-buffer
            # span and the bytes per K tile come from the size arguments.
            s_side, s_k = geo.stride(side_digits), geo.stride(k_digits)
            self.split = (((spec.tile_k, s_k),), None)
            self.span = _rt_span(
                b,
                ((b.sub(side_extent, b.const_i32(1)), s_side), (spec.tile_k - 1, s_k)),
                self.es,
            )
            if isinstance(s_k, int):
                self.kstep = b.const_i64(spec.tile_k * self.es)
            else:
                self.kstep = b.mul(b.zext(s_k, I64), b.const_i64(spec.tile_k * self.es))
            self.rsrc = None
        else:
            self.span = p.addr_span_bytes(name, spec.tile_k)
            self.split = k_split(k_digits, spec.tile_k)
            # Without a K split the K offset stays per lane: one resource for all.
            self.rsrc = (
                None
                if self.split is not None
                else b.buffer_rsrc(base_ptr, b.const_i32(self.span))
            )
        tid = b.thread_id_x()
        pl = self.plan
        c = b.mod(tid, b.const_i32(pl.cpl))
        line = b.div(tid, b.const_i32(pl.cpl))
        self.coords = (
            []
        )  # per pass: (side_local, k_local, base voffset bytes, side_valid)
        for ps in range(pl.passes):
            ln = line if ps == 0 else b.add(line, b.const_i32(ps * pl.lpp))
            v0 = b.mul(c, b.const_i32(pl.vec)) if pl.vec > 1 else c
            s_loc, k_loc = (ln, v0) if pl.along_k else (v0, ln)
            s_g = b.add(side_base, s_loc)
            s_ok = b.cmp_lt(s_g, _ci(b, side_extent))
            s_g_safe = b.select(s_ok, s_g, b.const_i32(0))
            off = _mixed(b, s_g_safe, side_digits)
            if self.split is not None:
                off = b.add(off, _mixed(b, k_loc, self.split[0]))
            off = b.mul(off, b.const_i32(self.es))
            self.coords.append((s_loc, k_loc, off, s_ok))
        # LDS store columns, loop-invariant: one per K row an element lands in.
        wpe = 2 if self.cx else 1
        swz = spec.uses_swizzle()
        self.store_cols = []
        for s_loc, k_loc, _, _ in self.coords:
            rows = range(pl.vec) if pl.along_k else range(1)
            cols = []
            for j in rows:
                col = b.mul(s_loc, b.const_i32(wpe)) if wpe > 1 else s_loc
                if swz:
                    kr = b.add(k_loc, b.const_i32(j)) if j else k_loc
                    col = b.xor(
                        col,
                        _swz(
                            b,
                            kr,
                            spec.swizzle_mask(side_tile),
                            spec.lds_swizzle_mode,
                        ),
                    )
                cols.append(col)
            self.store_cols.append(cols)

    def load(self, kt: Value):
        """Global loads of K tile ``kt`` into registers (one value per pass).

        A tile index at or past the K-tile count loads zeros (every lane gets
        the out-of-range offset), also when ``k_stagger`` would wrap it onto a
        real tile.
        """
        st = self.load_begin(kt)
        return [self.load_piece(st, i) for i in range(len(self.coords))]

    def load_begin(self, kt: Value):
        """The per-K-tile part of :meth:`load` (state for :meth:`load_piece`)."""
        b, spec = self.b, self.spec
        live = None
        if self.kmap is not None:
            live = b.cmp_lt(kt, _ci(b, self.nk))
            kt = self.kmap(kt)
        k0 = b.mul(kt, b.const_i32(spec.tile_k))
        rsrc = self.rsrc
        if self.split is not None:
            # The K-tile offset moves the 64-bit base; the lanes keep their
            # loop-invariant offsets.
            if self.rt:
                koff = b.mul(b.zext(kt, I64), self.kstep)
            else:
                koff = _mixed64(b, kt, self.split[1], self.es)
            tile_ptr = b.global_ptr_add(self.base_ptr, koff)
            rsrc = b.buffer_rsrc(tile_ptr, _ci(b, self.span))
        return (k0, rsrc, live)

    def load_piece(self, st, i: int) -> Value:
        """The global load of pass ``i`` of the K tile :meth:`load_begin` set up."""
        b, pl = self.b, self.plan
        k0, rsrc, live = st
        if True:
            s_loc, k_loc, base, s_ok = self.coords[i]
            kg = b.add(k0, k_loc)
            # The vector covers vec consecutive K rows when along K; every K
            # tail is a whole vector because K-contiguous runs are even.
            ok = b.land(s_ok, b.cmp_lt(kg, _ci(b, self.K)))
            if live is not None:
                ok = b.land(ok, live)
            if self.split is not None:
                voff = base
            else:
                kg_safe = b.select(ok, kg, b.const_i32(0))
                voff = b.add(
                    base, b.mul(_mixed(b, kg_safe, self.k_digits), b.const_i32(self.es))
                )
            voff = b.select(ok, voff, b.const_i32(_MASK_OFFSET))
            words = pl.vec * (2 if self.cx else 1)
            raw = b.buffer_load_vN(rsrc, voff, b.const_i32(0), I32, self.dw * words)
            return b.vec_bitcast(raw, VectorType(self.wty, words))

    def store_lds(self, vals, stage: Value):
        """Write one K tile (from :meth:`load`) into LDS stage ``stage``."""
        row0 = self.b.mul(stage, self.b.const_i32(self.spec.tile_k))
        for i, v in enumerate(vals):
            self.store_piece(v, i, row0)

    def stage_rows(self, row0: Value) -> Value:
        """K-major tile: the first LDS row of the stage whose [k][mn] row
        offset is ``row0`` (``stage * tile_k``)."""
        b, spec = self.b, self.spec
        if self.side_tile % spec.tile_k == 0:
            return b.mul(row0, b.const_i32(self.side_tile // spec.tile_k))
        return b.div(b.mul(row0, b.const_i32(self.side_tile)), b.const_i32(spec.tile_k))

    def frag(self, row0: Value, row: Value, col: Value):
        """The fragment of LDS K row ``row`` (stage rows from ``row0``) at
        column ``col``."""
        if self.kmajor:
            b = self.b
            return _frag(
                b, self.lds, b.add(self.stage_rows(row0), col), b.sub(row, row0), False
            )
        return _frag(self.b, self.lds, row, col, self.cx, self.wty)

    def store_piece(self, v, i: int, row0: Value):
        """The LDS store of pass ``i`` (value ``v``) into the stage at ``row0``."""
        b, pl = self.b, self.plan
        wpe = 2 if self.cx else 1
        s_loc, k_loc, _, _ = self.coords[i]
        cols = self.store_cols[i]
        if self.kmajor:
            # [stage * side + mn][k]: the vector's K values are adjacent
            r = b.add(self.stage_rows(row0), s_loc)
            if pl.vec == 1:
                b.smem_store_vN(self.lds, [r, k_loc], b.vec_extract(v, 0), 1)
            else:
                b.smem_store_vN(self.lds, [r, k_loc], v, pl.vec)
            return
        if True:
            if pl.along_k:
                # vec K rows of one column (one element each).
                for j in range(pl.vec):
                    row = b.add(row0, b.add(k_loc, b.const_i32(j)) if j else k_loc)
                    col = cols[j]
                    if self.cx:
                        b.smem_store_vN(self.lds, [row, col], v, 2)
                    else:
                        b.smem_store_vN(self.lds, [row, col], b.vec_extract(v, j), 1)
            else:
                row = b.add(row0, k_loc)
                col = cols[0]
                n = pl.vec * wpe
                if n == 1:
                    b.smem_store_vN(self.lds, [row, col], b.vec_extract(v, 0), 1)
                else:
                    b.smem_store_vN(self.lds, [row, col], v, n)


def _raster(
    b: IRBuilder, spec: GettF64Spec, bid: Value, geo: _Geo
) -> Tuple[Value, Value]:
    if geo.rt:
        return _raster_rt(b, spec, bid, geo)
    p = spec.problem
    ntm = -(-p.M // spec.tile_m)
    ntn = -(-p.N // spec.tile_n)
    total = ntm * ntn
    if spec.xcds and total % spec.xcds == 0:
        per = total // spec.xcds
        bid = b.add(
            b.mul(b.mod(bid, b.const_i32(spec.xcds)), b.const_i32(per)),
            b.div(bid, b.const_i32(spec.xcds)),
        )
    g = min(spec.group_m, ntm)
    if g <= 1:
        return b.div(bid, b.const_i32(ntn)), b.mod(bid, b.const_i32(ntn))
    in_group = g * ntn
    group = b.div(bid, b.const_i32(in_group))
    first = b.mul(group, b.const_i32(g))
    size = b.smin(b.sub(b.const_i32(ntm), first), b.const_i32(g))
    r = b.mod(bid, b.const_i32(in_group))
    tm = b.add(first, b.mod(r, size))
    tn = b.div(r, size)
    return tm, tn


def _raster_rt(b: IRBuilder, spec: GettF64Spec, bid: Value, geo: _Geo):
    """:func:`_raster` with the tile counts computed from the size arguments."""
    ntm = b.div(b.add(geo.M, b.const_i32(spec.tile_m - 1)), b.const_i32(spec.tile_m))
    ntn = b.div(b.add(geo.N, b.const_i32(spec.tile_n - 1)), b.const_i32(spec.tile_n))
    if spec.xcds:
        total = b.mul(ntm, ntn)
        per = b.div(total, b.const_i32(spec.xcds))
        remap = b.add(
            b.mul(b.mod(bid, b.const_i32(spec.xcds)), per),
            b.div(bid, b.const_i32(spec.xcds)),
        )
        even = b.cmp_eq(b.mod(total, b.const_i32(spec.xcds)), b.const_i32(0))
        bid = b.select(even, remap, bid)
    if spec.group_m <= 1:
        return b.div(bid, ntn), b.mod(bid, ntn)
    g = b.smin(ntm, b.const_i32(spec.group_m))
    in_group = b.mul(g, ntn)
    group = b.div(bid, in_group)
    first = b.mul(group, g)
    size = b.smin(b.sub(ntm, first), g)
    r = b.mod(bid, in_group)
    tm = b.add(first, b.mod(r, size))
    tn = b.div(r, size)
    return tm, tn


class _Frag:
    """Fragment read coordinates of one operand: the K row of the lane within
    a k-step and the LDS column of each atom at each k-step (loop-invariant)."""

    def __init__(self, k_row, cols):
        self.k_row = k_row
        self.cols = cols


def build_gett_f64(spec: GettF64Spec, *, arch: str = "gfx950") -> KernelDef:
    """Build the f64 / cf64 GETT kernel for ``spec`` on ``arch``.

    Kernel arguments: ``A, B, [C,] D`` (``ptr<f64>``, ``ptr<f32>`` for f32
    data; complex data is interleaved ``(re, im)``), then ``alpha`` (complex:
    ``alpha_re, alpha_im``) and, when bilinear, ``beta`` (``beta_re,
    beta_im``), all of the data's word type; with ``runtime_sizes`` then the
    sizes and strides of :data:`RUNTIME_SIZE_ARGS`. Grid:
    :func:`gett_f64_grid`; block: ``spec.threads``.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gett_f64 spec for {arch}: {why}")
    if spec.runtime_sizes:
        spec = replace(spec, problem=runtime_template(spec.problem))
    p = spec.problem
    cx = p.complex_
    es = p.elem_bytes
    wpe = 2 if cx else 1
    wty = _wty(p)
    atom = gett_atom(arch, p.dtype)
    nt = spec.threads
    am_n, an_n = spec.wave_m // atom.m, spec.wave_n // atom.n
    pad = spec.resolved_pad()

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = nt
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    gptr = PtrType(wty, "global")
    # A and B never overlap D. C may be D itself (in-place update), so neither
    # carries noalias. 8-byte alignment: an f64 subtensor view need not be
    # 16-byte aligned (the buffer loads only need 4). f32 data: 4-byte
    # alignment.
    al = p.word_bytes
    split = spec.split_k > 1
    # split_serial: the partials are reduced in this launch (C / D / alpha /
    # beta as without split_k, plus W and Cnt)
    partial_only = split and not spec.split_serial
    A = b.param("A", gptr, noalias=True, readonly=True, align=al)
    Bp = b.param("B", gptr, noalias=True, readonly=True, align=al)
    C = D = W = Cnt = None
    alpha, beta = [], []
    if partial_only:
        # Partial sums only: the reduce kernel reads C and applies alpha / beta.
        W = b.param("W", gptr, noalias=True, align=al)
    else:
        C = b.param("C", gptr, readonly=True, align=al) if spec.bilinear else None
        D = b.param("D", gptr, align=al)
        alpha = [b.param("alpha_re" if cx else "alpha", wty)]
        if cx:
            alpha.append(b.param("alpha_im", wty))
        if spec.bilinear:
            beta = [b.param("beta_re" if cx else "beta", wty)]
            if cx:
                beta.append(b.param("beta_im", wty))
        if split:
            W = b.param("W", gptr, noalias=True, align=al)
            Cnt = b.param("Cnt", PtrType(I32, "global"), align=4)
    rt_args = None
    if spec.runtime_sizes:
        rt_args = {
            n: b.param(n, I32 if t == "i32" else I64)
            for n, t in _runtime_arg_names(spec)
        }
    geo = _Geo(b, spec, rt_args)

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))
    wave = b.readfirstlane(b.div(tid, b.const_i32(64)))
    wm = b.mod(wave, b.const_i32(spec.warps_m))
    wn = b.div(wave, b.const_i32(spec.warps_m))
    tm, tn = _raster(b, spec, b.block_id_x(), geo)
    m0 = b.mul(tm, b.const_i32(spec.tile_m))
    n0 = b.mul(tn, b.const_i32(spec.tile_n))
    lb = b.block_id_y()

    cm, cn, cl = p.c_digits()

    def at_batch(ptr, digits, rt_name):
        """``ptr`` moved to batch ``lb`` (64-bit)."""
        if geo.rt:
            off = b.mul(b.zext(lb, I64), geo.batch[rt_name])
            return b.global_ptr_add(ptr, b.mul(off, b.const_i64(es)))
        if digits_extent(digits) == 1:
            return ptr
        return b.global_ptr_add(ptr, _mixed64(b, lb, digits, es))

    def out_span(m_dig, n_dig, which):
        if geo.rt:
            return _rt_span(
                b,
                (
                    (b.sub(geo.M, b.const_i32(1)), geo.stride(m_dig)),
                    (b.sub(geo.N, b.const_i32(1)), geo.stride(n_dig)),
                ),
                es,
            )
        return b.const_i32(p.addr_span_bytes(which, spec.tile_k))

    rd = rc = rw = kz = None
    if partial_only:
        kz = b.block_id_z()
        rw = _workspace_rsrc(b, spec, geo, W, kz, lb, es)
    else:
        if split:
            kz = b.block_id_z()
        dptr = at_batch(D, p.d_l, "d_sl")
        rd = b.buffer_rsrc(dptr, out_span(geo.d_m, geo.d_n, "D"))
        if C is not None:
            cptr = at_batch(C, cl, "c_sl")
            rc = b.buffer_rsrc(cptr, out_span(geo.c_m, geo.c_n, "C"))

    ns = spec.lds_stages
    if spec.kmajor_lds:
        a_lds = b.smem_alloc(
            wty, _lds_shape(spec, p.a_m, p.a_k, spec.tile_m), name_hint="a_lds"
        )
        b_lds = b.smem_alloc(
            wty, _lds_shape(spec, p.b_n, p.b_k, spec.tile_n), name_hint="b_lds"
        )
    else:
        a_lds = b.smem_alloc(
            wty, [ns * spec.tile_k, spec.tile_m * wpe + pad], name_hint="a_lds"
        )
        b_lds = b.smem_alloc(
            wty, [ns * spec.tile_k, spec.tile_n * wpe + pad], name_hint="b_lds"
        )
    aptr = at_batch(A, p.a_l, "a_sl")
    opa = _Operand(
        b, spec, "A", aptr, geo.a_m, geo.a_k, spec.tile_m, m0, geo.M, a_lds, geo
    )
    bptr = at_batch(Bp, p.b_l, "b_sl")
    opb = _Operand(
        b, spec, "B", bptr, geo.b_n, geo.b_k, spec.tile_n, n0, geo.N, b_lds, geo
    )

    # Fragment read coordinates from the atom's catalog lane maps: A lane
    # ``l`` holds (row, k) = a_layout(l), B lane ``l`` holds (k, col) =
    # b_layout(l). The swizzle of K row ``4*ks + k`` within a stage depends
    # only on the lane and the k-step, so every column is loop-invariant.
    a_mn, a_k = atom.a_layout().coord(b, lane, 0)
    b_k, b_mn = atom.b_layout().coord(b, lane, 0)
    a_col0 = b.add(b.mul(wm, b.const_i32(spec.wave_m)), a_mn)
    b_col0 = b.add(b.mul(wn, b.const_i32(spec.wave_n)), b_mn)
    swz_on = spec.uses_swizzle()

    def frag_cols(col0, n_atoms, side, k_row, kmaj=False):
        """cols[ks][i]: column of atom i at k-step ks."""
        if kmaj:
            # K-major tile: the column is the M / N index, no swizzle
            cols = [
                b.add(col0, b.const_i32(i * atom.m)) if i else col0
                for i in range(n_atoms)
            ]
            return _Frag(k_row, [cols] * (spec.tile_k // atom.k))
        mask = spec.swizzle_mask(side)
        odd = None
        if swz_on and not spec.lds_swizzle_mode:
            odd = b.mul(b.land(k_row, b.const_i32(1)), b.const_i32(16))
        out = []
        for ks in range(spec.tile_k // atom.k):
            cols = []
            for i in range(n_atoms):
                col = b.add(col0, b.const_i32(i * atom.m)) if i else col0
                if swz_on and spec.lds_swizzle_mode:
                    kr = b.add(k_row, b.const_i32(atom.k * ks)) if ks else k_row
                    col = b.xor(col, _swz(b, kr, mask, spec.lds_swizzle_mode))
                elif swz_on:
                    # swz(4*ks + k) = 16*(k & 1) ^ 8*(ks & 7), masked.
                    flip = b.xor(odd, b.const_i32((8 * (ks & 7)) & mask))
                    col = b.xor(col, b.land(flip, b.const_i32(mask)))
                cols.append(col)
            out.append(cols)
        return _Frag(k_row, out)

    fa = frag_cols(a_col0, am_n, spec.tile_m, a_k, opa.kmajor)
    fb = frag_cols(b_col0, an_n, spec.tile_n, b_k, opb.kmajor)

    if geo.rt:
        nk = b.div(b.add(geo.K, b.const_i32(spec.tile_k - 1)), b.const_i32(spec.tile_k))
    else:
        nk = -(-p.K // spec.tile_k)
    if spec.k_stagger and (geo.rt or nk > 1):
        # start tile ((bid % 8) * n) % nk, then wrap: kt -> kt + start (- nk)
        sid = b.mod(b.block_id_x(), b.const_i32(8))
        start = b.mod(b.mul(sid, b.const_i32(spec.k_stagger)), _ci(b, nk))

        def kmap(kt):
            t = b.add(kt, start)
            wrap = b.cmp_lt(t, _ci(b, nk))
            return b.select(wrap, t, b.sub(t, _ci(b, nk)))

        for op in (opa, opb):
            op.kmap = kmap
            op.nk = nk
    n_acc = am_n * an_n * (spec.complex_accs if cx else 1)
    zero = b.vec_pack([_wconst(b, p, 0.0)] * atom.c_frag_len, wty)
    if spec.acc_agpr:
        # AGPR-resident accumulators: reserve exactly their registers (the
        # attr's minimum also selects the AGPR form of every MFMA) and seed
        # them from an opaque AGPR zero, so the loop-carried accumulator phis
        # stay whole vectors instead of per-element copies.
        n_regs = n_acc * atom.c_frag_len * 2 * (2 if spec.k_block else 1)
        b.kernel.attrs["agpr_alloc"] = (n_regs, n_regs)
        b.kernel.attrs["mfma_agpr_form"] = True
        zero = b.inline_asm("", "=a,0", [zero], zero.type, sideeffect=False)

    # split_k: this CTA's K tiles are [lo, hi) with lo = z*nk/S, hi = (z+1)*nk/S.
    k_lo, count = None, nk
    if split:
        k_lo = b.div(b.mul(kz, _ci(b, nk)), b.const_i32(spec.split_k))
        k_hi = b.div(
            b.mul(b.add(kz, b.const_i32(1)), _ci(b, nk)), b.const_i32(spec.split_k)
        )
        count = b.sub(k_hi, k_lo)

    # Prologue: the first K tile into stage 0.
    zero_i = b.const_i32(0)
    first = zero_i if k_lo is None else k_lo
    regs_a = opa.load(first)
    regs_b = opb.load(first)
    nx = None
    if spec.prefetch_depth == 2:
        # the second tile rides in registers into the loop (carried values below)
        one_i = b.const_i32(1) if k_lo is None else b.add(k_lo, b.const_i32(1))
        nx = (opa.load(one_i), opb.load(one_i))
    opa.store_lds(regs_a, zero_i)
    opb.store_lds(regs_b, zero_i)
    b.sync()

    ctx = _LoopCtx(
        b, spec, atom, opa, opb, a_lds, b_lds, fa, fb, count, zero, n_acc, zero_i
    )
    ctx.na = len(nx[0]) if nx is not None else 0
    ctx.k_lo = k_lo
    if spec.ilv:
        accs = _k_loop_ilv(ctx, nx)
    elif spec.frag_pipe:
        accs = _k_loop_pipe(ctx)
    elif spec.unroll2:
        accs = _k_loop_unroll2(ctx)
    else:
        accs = _k_loop(ctx, nx)
    if partial_only:
        _epilogue_partial(b, spec, atom, accs, lane, wm, wn, m0, n0, rw, geo)
    elif split:
        tile = b.add(b.mul(tm, b.const_i32(-(-p.N // spec.tile_n))), tn)
        red = _SerialReduce(b, spec, atom, accs, W, Cnt, kz, lb, tile, wave, lane)
        with red as summed:
            _epilogue(
                b, spec, atom, summed, lane, wm, wn, m0, n0, rd, rc, alpha, beta, geo
            )
    else:
        _epilogue(b, spec, atom, accs, lane, wm, wn, m0, n0, rd, rc, alpha, beta, geo)
    b.ret()
    return b.kernel


class _LoopCtx:
    """Everything the K-loop emitters share."""

    def __init__(
        self, b, spec, atom, opa, opb, a_lds, b_lds, fa, fb, nk, zero, n_acc, zero_i
    ):
        self.b, self.spec, self.atom = b, spec, atom
        self.opa, self.opb = opa, opb
        self.a_lds, self.b_lds = a_lds, b_lds
        self.fa, self.fb = fa, fb
        self.nk, self.zero, self.n_acc, self.zero_i = nk, zero, n_acc, zero_i
        self.k_lo = None  # split_k: the first K tile of this CTA
        self.cx = spec.problem.complex_

    def read_group(self, row0, g):
        """Fragments of k-step group ``g`` (``k_inner`` k-steps): per k-step,
        the A fragments of the wave's atom rows and the B fragments of its
        atom columns."""
        b, spec, atom = self.b, self.spec, self.atom
        out = []
        for ks in range(g * spec.k_inner, (g + 1) * spec.k_inner):
            ka = self.fa.k_row
            row_a = b.add(row0, b.add(ka, b.const_i32(ks * atom.k)) if ks else ka)
            fa = [self.opa.frag(row0, row_a, col) for col in self.fa.cols[ks]]
            kb = self.fb.k_row
            row_b = b.add(row0, b.add(kb, b.const_i32(ks * atom.k)) if ks else kb)
            fb = [self.opb.frag(row0, row_b, col) for col in self.fb.cols[ks]]
            out.append((fa, fb))
        return out

    def mma_group(self, accs, frags):
        """The MFMAs of one k-step group (from :meth:`read_group`)."""
        b, spec, atom = self.b, self.spec, self.atom
        if len(frags) == 1 and not spec.mfma_fence:
            fa, fb = frags[0]
            return _mma_step(b, atom, fa, fb, accs, spec.problem, spec.complex_accs)
        if spec.problem.complex_:
            for fa, fb in frags:
                accs = _mma_step(b, atom, fa, fb, accs, spec.problem, spec.complex_accs)
                if spec.mfma_fence:
                    b.sched_barrier(0)
            return accs
        accs = list(accs)
        am_n, an_n = len(frags[0][0]), len(frags[0][1])
        for am in range(am_n):
            for an in range(an_n):
                i = am * an_n + an
                for fa, fb in frags:
                    accs[i] = b.mma(atom, fa[am], fb[an], accs[i])
                    if spec.mfma_fence:
                        b.sched_barrier(0)
        return accs

    def mfma_block(self, accs, row0):
        """The MFMAs of one K tile read from the LDS rows at ``row0``."""
        b, spec, atom = self.b, self.spec, self.atom
        accs = list(accs)
        if spec.k_inner > 1 or spec.mfma_fence:
            for g in range(spec.tile_k // atom.k // spec.k_inner):
                accs = self.mma_group(accs, self.read_group(row0, g))
            return accs
        for ks in range(spec.tile_k // atom.k):
            ka = self.fa.k_row
            row_a = b.add(row0, b.add(ka, b.const_i32(ks * atom.k)) if ks else ka)
            fa = [self.opa.frag(row0, row_a, col) for col in self.fa.cols[ks]]
            kb = self.fb.k_row
            row_b = b.add(row0, b.add(kb, b.const_i32(ks * atom.k)) if ks else kb)
            fb = [self.opb.frag(row0, row_b, col) for col in self.fb.cols[ks]]
            accs = _mma_step(b, atom, fa, fb, accs, spec.problem, spec.complex_accs)
        return accs


def _blocked(ctx: _LoopCtx, n_units, per_block: int, run, carry):
    """``k_block``: run the K range in blocks of ``per_block`` loop units, each
    from zero accumulators, and add every block's sum into ``sums``.

    ``run(lo, hi, accs, carry) -> (accs, carry)`` emits the loop over units
    ``[lo, hi)``. Returns ``(sums, carry)``.
    """
    b = ctx.b
    if isinstance(n_units, int):
        nblk = b.const_i32(-(-n_units // per_block))
    else:
        nblk = b.div(b.add(n_units, b.const_i32(per_block - 1)), b.const_i32(per_block))
    iter_args = [(f"sum{i}", ctx.zero) for i in range(ctx.n_acc)]
    iter_args += [(f"cr{i}", v) for i, v in enumerate(carry)]
    outer = b.scf_for_iter(
        b.const_i32(0), nblk, b.const_i32(1), iter_args, iv_name="kb"
    )
    with outer as (ob, carried):
        carried = list(carried)
        sums = carried[: ctx.n_acc]
        lo = b.mul(ob, b.const_i32(per_block))
        hi = b.smin(b.add(lo, b.const_i32(per_block)), _ci(b, n_units))
        accs, rest = run(lo, hi, [ctx.zero] * ctx.n_acc, carried[ctx.n_acc :])
        sums = [b.fadd(s, a) for s, a in zip(sums, accs)]
        b.scf_yield(*sums, *rest)
    res = list(outer.results)
    return res[: ctx.n_acc], res[ctx.n_acc :]


def _k_loop(ctx: _LoopCtx, nx) -> List[Value]:
    """The K loop over every K tile but the last: prefetch the next tile into
    registers, the MFMAs of this tile, the LDS store of the next one. The last
    tile runs after the loop with its MFMAs only (no loads, stores or barrier),
    so the loop body has no branch."""
    b, spec = ctx.b, ctx.spec
    ns, pd = spec.lds_stages, spec.prefetch_depth
    opa, opb, zero_i = ctx.opa, ctx.opb, ctx.zero_i

    def run(lo, hi, accs0, carry):
        n_acc = ctx.n_acc
        iter_args = [(f"acc{i}", v) for i, v in enumerate(accs0)]
        if pd == 2:
            na = ctx.na
            iter_args += [(f"ra{i}", v) for i, v in enumerate(carry[:na])]
            iter_args += [(f"rb{i}", v) for i, v in enumerate(carry[na:])]
        loop = b.scf_for_iter(lo, hi, b.const_i32(1), iter_args, iv_name="kt")
        with loop as (kt, carried):
            carried = list(carried)
            accs = carried[:n_acc]
            if pd == 2:
                na = ctx.na
                cur_a = carried[n_acc : n_acc + na]
                cur_b = carried[n_acc + na :]
            # One LDS stage: always stage 0, and every wave must be done
            # reading it (second barrier below) before the next tile
            # overwrites it.
            stage = b.land(kt, b.const_i32(1)) if ns == 2 else zero_i
            # split_k: kt counts this CTA's tiles; the loads take the global index.
            ktg = kt if ctx.k_lo is None else b.add(kt, ctx.k_lo)
            nxt = b.add(ktg, b.const_i32(pd))
            if spec.prefetch:
                pre_a = opa.load(nxt)
                pre_b = opb.load(nxt)
            if spec.sched_fence:
                b.sched_barrier(0)
            row0 = b.mul(stage, b.const_i32(spec.tile_k))
            accs = ctx.mfma_block(accs, row0)
            if spec.sched_fence:
                b.sched_barrier(0)
            if not spec.prefetch:
                pre_a = opa.load(nxt)
                pre_b = opb.load(nxt)
            if ns == 2:
                nstage = b.xor(stage, b.const_i32(1))
            else:
                nstage = zero_i
                b.sync()
            opa.store_lds(cur_a if pd == 2 else pre_a, nstage)
            opb.store_lds(cur_b if pd == 2 else pre_b, nstage)
            b.sync()
            if pd == 2:
                b.scf_yield(*accs, *pre_a, *pre_b)
            else:
                b.scf_yield(*accs)
        res = list(loop.results)
        return res[:n_acc], res[n_acc:]

    carry = [*nx[0], *nx[1]] if pd == 2 else []
    nk = ctx.nk
    last = nk - 1 if isinstance(nk, int) else b.sub(nk, b.const_i32(1))
    if spec.k_block:
        accs, _ = _blocked(ctx, last, spec.k_block, run, carry)
    else:
        lo = b.const_i32(0)
        hi = _ci(b, last)
        accs, _ = run(lo, hi, [ctx.zero] * ctx.n_acc, carry)
    # The last K tile: its MFMAs only, from the stage the loop left it in.
    kt = _ci(b, last)
    stage = b.land(kt, b.const_i32(1)) if ns == 2 else zero_i
    if spec.sched_fence:
        b.sched_barrier(0)
    accs = ctx.mfma_block(accs, b.mul(stage, b.const_i32(spec.tile_k)))
    if spec.sched_fence:
        b.sched_barrier(0)
    return accs


def _k_loop_unroll2(ctx: _LoopCtx) -> List[Value]:
    """The K loop unrolled by two tiles (``unroll2``): the first half reads
    LDS stage 0 and fills stage 1, the second the reverse. The last pair runs
    after the loop, its second half without a store; an odd last tile runs
    after the loop from stage 0 (``runtime_sizes``: the last pair's second half
    reads the zeros the first half stored). Returns the accumulators."""
    b, spec = ctx.b, ctx.spec
    opa, opb = ctx.opa, ctx.opb

    def half(accs, kt, stg, do_next):
        if do_next:
            nxt = b.add(kt, b.const_i32(1))
            pre_a = opa.load(nxt)
            pre_b = opb.load(nxt)
        if spec.sched_fence:
            b.sched_barrier(0)
        row0 = b.const_i32(stg * spec.tile_k)
        accs = ctx.mfma_block(accs, row0)
        if spec.sched_fence:
            b.sched_barrier(0)
        if do_next:
            other = b.const_i32(1 - stg)
            opa.store_lds(pre_a, other)
            opb.store_lds(pre_b, other)
            b.sync()
        return accs

    def run(lo, hi, accs0, carry):
        iter_args = [(f"acc{i}", v) for i, v in enumerate(accs0)]
        loop = b.scf_for_iter(lo, hi, b.const_i32(1), iter_args, iv_name="kt2")
        with loop as (kt2, accs):
            kt = b.mul(kt2, b.const_i32(2))
            accs = half(accs, kt, 0, True)
            accs = half(accs, b.add(kt, b.const_i32(1)), 1, True)
            b.scf_yield(*accs)
        return list(loop.results), carry

    nk = ctx.nk
    if isinstance(nk, int):
        # odd: every pair's stores are live, the last tile follows alone;
        # even: the last pair follows the loop.
        looped = nk // 2 if nk % 2 else nk // 2 - 1
        tail_pair = nk % 2 == 0
    else:
        pairs = b.div(b.add(nk, b.const_i32(1)), b.const_i32(2))
        looped = b.sub(pairs, b.const_i32(1))
        tail_pair = True
    if spec.k_block:
        accs, _ = _blocked(ctx, looped, spec.k_block // 2, run, [])
    else:
        lo = b.const_i32(0)
        hi = _ci(b, looped)
        accs, _ = run(lo, hi, [ctx.zero] * ctx.n_acc, [])
    if tail_pair:
        kt = b.mul(_ci(b, looped), b.const_i32(2))
        accs = half(accs, kt, 0, True)
        accs = half(accs, b.add(kt, b.const_i32(1)), 1, False)
    else:
        accs = half(accs, b.const_i32(nk - 1), 0, False)
    return accs


def _k_loop_pipe(ctx: _LoopCtx) -> List[Value]:
    """``frag_pipe``: the K loop with register double-buffered fragments.

    Per K tile ``t`` (LDS stage ``s``, next stage ``s'``): the global loads of
    tile ``t + 1``; then for every k-step but the last, the fragment reads of
    the next k-step followed by the MFMAs of this one; then the LDS store of
    tile ``t + 1`` into ``s'`` and the barrier; then the reads of k-step 0 of
    tile ``t + 1`` (from ``s'``) and the MFMAs of the last k-step of ``t``.
    The fragments of the next k-step ride into the loop iteration as carried
    values. Every k-step group is fenced (``sched_barrier``) so the backend
    keeps the reads ahead of the MFMAs that do not need them.

    Stage safety: every read of ``s'`` (tile ``t - 1``) was issued before the
    barrier of tile ``t - 1``, so the store of tile ``t + 1`` into ``s'`` after
    that barrier overwrites nothing still to be read; the reads of ``s'``
    after the barrier of tile ``t`` see that store.
    """
    b, spec, atom = ctx.b, ctx.spec, ctx.atom
    opa, opb = ctx.opa, ctx.opb
    cx = ctx.cx
    ki = spec.k_inner
    nks = spec.tile_k // atom.k // ki  # k-step groups per K tile

    def read(row0, g):
        return ctx.read_group(row0, g)

    def flat(f):
        out = []
        for fa, fb in f:
            for x in list(fa) + list(fb):
                out += list(x) if cx else [x]
        return out

    na = len(ctx.fa.cols[0])
    nb = len(ctx.fb.cols[0])
    wf = 2 if cx else 1

    def unflat(vals):
        per = (na + nb) * wf
        out = []
        for j in range(ki):
            v = vals[j * per : (j + 1) * per]
            fa = v[: na * wf]
            fb = v[na * wf :]
            if cx:
                fa = [(fa[2 * i], fa[2 * i + 1]) for i in range(na)]
                fb = [(fb[2 * i], fb[2 * i + 1]) for i in range(nb)]
            out.append((fa, fb))
        return out

    def mma(accs, f):
        return ctx.mma_group(accs, f)

    def row_of(stg):
        if isinstance(stg, int):
            return b.const_i32(stg * spec.tile_k)
        return b.mul(stg, b.const_i32(spec.tile_k))

    def tile(accs, cur, kt, stg, nstg, has_next):
        """One K tile from stage ``stg``; returns (accs, fragments of the next
        tile's k-step 0, or None for the last tile)."""
        row0 = row_of(stg)
        if has_next:
            ktg = kt if ctx.k_lo is None else b.add(kt, ctx.k_lo)
            nxt = b.add(ktg, b.const_i32(1))
            pre_a = opa.load(nxt)
            pre_b = opb.load(nxt)
        b.sched_barrier(0)
        for ks in range(nks - 1):
            f_next = read(row0, ks + 1)
            b.sched_barrier(0)
            accs = mma(accs, cur)
            cur = f_next
            b.sched_barrier(0)
        nf = None
        if has_next:
            other = b.const_i32(nstg) if isinstance(nstg, int) else nstg
            opa.store_lds(pre_a, other)
            opb.store_lds(pre_b, other)
            b.sync()
            nf = read(row_of(nstg), 0)
            b.sched_barrier(0)
        accs = mma(accs, cur)
        b.sched_barrier(0)
        return accs, nf

    nk = ctx.nk
    last = nk - 1 if isinstance(nk, int) else b.sub(nk, b.const_i32(1))
    f0 = read(b.const_i32(0), 0)
    accs0 = [ctx.zero] * ctx.n_acc
    if spec.unroll2:
        # two tiles per iteration with compile-time stages 0 / 1
        if isinstance(nk, int):
            looped = nk // 2 if nk % 2 else nk // 2 - 1
            tail_pair = nk % 2 == 0
        else:
            pairs = b.div(b.add(nk, b.const_i32(1)), b.const_i32(2))
            looped = b.sub(pairs, b.const_i32(1))
            tail_pair = True
        iter_args = [(f"acc{i}", v) for i, v in enumerate(accs0)]
        iter_args += [(f"fr{i}", v) for i, v in enumerate(flat(f0))]
        loop = b.scf_for_iter(
            b.const_i32(0), _ci(b, looped), b.const_i32(1), iter_args, iv_name="kp2"
        )
        with loop as (kt2, carried):
            carried = list(carried)
            accs = carried[: ctx.n_acc]
            cur = unflat(carried[ctx.n_acc :])
            kt = b.mul(kt2, b.const_i32(2))
            accs, cur = tile(accs, cur, kt, 0, 1, True)
            accs, cur = tile(accs, cur, b.add(kt, b.const_i32(1)), 1, 0, True)
            b.scf_yield(*accs, *flat(cur))
        res = list(loop.results)
        accs = res[: ctx.n_acc]
        cur = unflat(res[ctx.n_acc :])
        if tail_pair:
            kt = b.mul(_ci(b, looped), b.const_i32(2))
            accs, cur = tile(accs, cur, kt, 0, 1, True)
            accs, _ = tile(accs, cur, b.add(kt, b.const_i32(1)), 1, 0, False)
        else:
            accs, _ = tile(accs, cur, b.const_i32(nk - 1), 0, 1, False)
        return accs
    iter_args = [(f"acc{i}", v) for i, v in enumerate(accs0)]
    iter_args += [(f"fr{i}", v) for i, v in enumerate(flat(f0))]
    loop = b.scf_for_iter(
        b.const_i32(0), _ci(b, last), b.const_i32(1), iter_args, iv_name="kp"
    )
    with loop as (kt, carried):
        carried = list(carried)
        accs = carried[: ctx.n_acc]
        cur = unflat(carried[ctx.n_acc :])
        stg = b.land(kt, b.const_i32(1))
        nstg = b.xor(stg, b.const_i32(1))
        accs, cur = tile(accs, cur, kt, stg, nstg, True)
        b.scf_yield(*accs, *flat(cur))
    res = list(loop.results)
    accs = res[: ctx.n_acc]
    cur = unflat(res[ctx.n_acc :])
    kt = _ci(b, last)
    stg = b.land(kt, b.const_i32(1))
    accs, _ = tile(accs, cur, kt, stg, None, False)
    return accs


def _spread(n: int, lo: int, hi: int) -> List[int]:
    """``n`` slot indices spread evenly over ``[lo, hi)`` (clamped to ``lo``
    when the range is empty)."""
    if n <= 0:
        return []
    width = max(hi - lo, 0)
    return [lo + (j * width) // n for j in range(n)]


def _k_loop_ilv(ctx: _LoopCtx, nx) -> List[Value]:
    """``ilv``: the K loop as one instruction-level schedule per K tile.

    Tile ``t`` (stage ``s``, next stage ``s'``; tile ``t + 1`` in registers):
    the MFMAs of every k-step group in order; the fragment reads of group
    ``g + 1`` spread over the MFMAs of group ``g``; the LDS stores of tile
    ``t + 1`` into ``s'`` spread over the MFMAs up to ``ilv_wgap`` before the
    barrier, each followed by the global load of the same pass of tile
    ``t + 2``; the barrier ``ilv`` MFMAs before the end; then the reads of
    tile ``t + 1``'s first group (from ``s'``) spread over the last MFMAs. A
    scheduling barrier after every MFMA pins the order.

    Stage safety as in :func:`_k_loop_pipe`: every read of ``s'`` (tile
    ``t - 1``) completed before the barrier of tile ``t - 1``.
    """
    b, spec, atom = ctx.b, ctx.spec, ctx.atom
    opa, opb = ctx.opa, ctx.opb
    ki = spec.k_inner
    ng = spec.tile_k // atom.k // ki
    am_n = len(ctx.fa.cols[0])
    an_n = len(ctx.fb.cols[0])
    n_a, n_b = len(opa.coords), len(opb.coords)
    per_group = am_n * an_n * ki
    total = ng * per_group
    tail = min(spec.ilv, per_group)
    bpos = total - tail

    def reads_of(row0, g):
        """Thunks of the fragment reads of group ``g`` -> (kind, kl, idx)."""
        out = []
        for kl in range(ki):
            ks = g * ki + kl
            ka = ctx.fa.k_row
            kb = ctx.fb.k_row
            for i, col in enumerate(ctx.fa.cols[ks]):
                out.append(("a", kl, i, ka, ks, col))
            for i, col in enumerate(ctx.fb.cols[ks]):
                out.append(("b", kl, i, kb, ks, col))
        return out

    def do_read(row0, r):
        kind, kl, i, kr, ks, col = r
        row = b.add(row0, b.add(kr, b.const_i32(ks * atom.k)) if ks else kr)
        op = ctx.opa if kind == "a" else ctx.opb
        return op.frag(row0, row, col)

    def empty_frags():
        return [([None] * am_n, [None] * an_n) for _ in range(ki)]

    def put(fr, r, v):
        kind, kl, i = r[0], r[1], r[2]
        fr[kl][0 if kind == "a" else 1][i] = v

    cx = ctx.cx
    wf = 2 if cx else 1

    def flat(fr):
        out = []
        for fa, fb in fr:
            for x in list(fa) + list(fb):
                out += list(x) if cx else [x]
        return out

    def unflat(vals):
        out = []
        per = (am_n + an_n) * wf
        for kl in range(ki):
            v = vals[kl * per : (kl + 1) * per]
            if cx:
                v = [(v[2 * j], v[2 * j + 1]) for j in range(am_n + an_n)]
            out.append((list(v[:am_n]), list(v[am_n:])))
        return out

    p = spec.problem
    sa = -1 if p.conj_a else 1
    sb = -1 if p.conj_b else 1

    def atom_mma(accs, fa, fb, am, an, negs):
        """The MFMAs of atom (am, an) (real: one; complex: the fused 4M
        products, as :func:`_mma_step`)."""
        i = am * an_n + an
        if not cx:
            accs[i] = b.mma(atom, fa[am], fb[an], accs[i])
            return
        ar, ai = fa[am]
        br, bi = fb[an]
        if spec.complex_accs == 4:
            j = 4 * i
            accs[j] = b.mma(atom, ar, br, accs[j])
            accs[j + 1] = b.mma(atom, ai, bi, accs[j + 1])
            accs[j + 2] = b.mma(atom, ar, bi, accs[j + 2])
            accs[j + 3] = b.mma(atom, ai, br, accs[j + 3])
            return

        def neg_a():
            if ("a", am) not in negs:
                negs[("a", am)] = b.fneg(ai)
            return negs[("a", am)]

        def neg_b():
            if ("b", an) not in negs:
                negs[("b", an)] = b.fneg(bi)
            return negs[("b", an)]

        j = 2 * i
        acc_r, acc_i = accs[j], accs[j + 1]
        acc_r = b.mma(atom, ar, br, acc_r)
        acc_r = b.mma(atom, neg_a() if sa * sb == 1 else ai, bi, acc_r)
        acc_i = b.mma(atom, ar, neg_b() if sb == -1 else bi, acc_i)
        acc_i = b.mma(atom, neg_a() if sa == -1 else ai, br, acc_i)
        accs[j], accs[j + 1] = acc_r, acc_i

    def row_of(stg):
        if isinstance(stg, int):
            return b.const_i32(stg * spec.tile_k)
        return b.mul(stg, b.const_i32(spec.tile_k))

    def tile(accs, cur, regs, kt, stg, nstg, has_next):
        """One K tile; returns (accs, next tile's group-0 fragments, regs of
        tile t + 2) (the last two None for the last tile)."""
        accs = list(accs)
        row0 = row_of(stg)
        events = {}  # MFMA index -> list of thunks emitted before it

        def at(idx, fn):
            events.setdefault(idx, []).append(fn)

        frags = [cur] + [empty_frags() for _ in range(ng - 1)]
        for g in range(ng - 1):
            rl = reads_of(row0, g + 1)
            for pos, r in zip(_spread(len(rl), g * per_group, (g + 1) * per_group), rl):
                at(pos, lambda r=r, g=g: put(frags[g + 1], r, do_read(row0, r)))
        nxt_frags = None
        new_regs = None
        if has_next:
            nrow0 = row_of(nstg)
            ra, rb = regs
            new_a, new_b = [None] * n_a, [None] * n_b
            ktg = kt if ctx.k_lo is None else b.add(kt, ctx.k_lo)
            st = {}

            def store_load(which, i):
                op = opa if which == "a" else opb
                src = ra if which == "a" else rb
                dst = new_a if which == "a" else new_b
                op.store_piece(src[i], i, nrow0)
                if which not in st:
                    st[which] = op.load_begin(b.add(ktg, b.const_i32(2)))
                dst[i] = op.load_piece(st[which], i)

            pieces = [("a", i) for i in range(n_a)] + [("b", i) for i in range(n_b)]
            hi = max(bpos - spec.ilv_wgap, 1)
            for pos, pc in zip(_spread(len(pieces), 0, hi), pieces):
                at(pos, lambda pc=pc: store_load(*pc))
            at(bpos, lambda: b.sync_lds_only())
            nxt_frags = empty_frags()
            rl = reads_of(nrow0, 0)
            for pos, r in zip(_spread(len(rl), bpos, total), rl):
                at(pos, lambda r=r: put(nxt_frags, r, do_read(nrow0, r)))
            new_regs = (new_a, new_b)
        idx = 0
        for g in range(ng):
            negs = [{} for _ in range(ki)]
            for am in range(am_n):
                for an in range(an_n):
                    for kl in range(ki):
                        for fn in events.get(idx, ()):
                            fn()
                        fa, fb = frags[g][kl]
                        atom_mma(accs, fa, fb, am, an, negs[kl])
                        b.sched_barrier(0)
                        idx += 1
        for fn in events.get(idx, ()):  # anything placed past the last MFMA
            fn()
        return accs, nxt_frags, new_regs

    nk = ctx.nk
    last = nk - 1 if isinstance(nk, int) else b.sub(nk, b.const_i32(1))
    cur = empty_frags()
    for r in reads_of(b.const_i32(0), 0):
        put(cur, r, do_read(b.const_i32(0), r))
    ra, rb = nx
    iter_args = [(f"acc{i}", v) for i, v in enumerate([ctx.zero] * ctx.n_acc)]
    iter_args += [(f"fr{i}", v) for i, v in enumerate(flat(cur))]
    iter_args += [(f"ra{i}", v) for i, v in enumerate(ra)]
    iter_args += [(f"rb{i}", v) for i, v in enumerate(rb)]
    nf = len(flat(cur))
    if spec.unroll2 and isinstance(nk, int):
        # two tiles per iteration with compile-time stages 0 / 1; the tiles
        # 0 .. nk - 2 have a successor, tile nk - 1 is the last
        pairs = (nk - 1) // 2
        loop = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(pairs),
            b.const_i32(1),
            iter_args,
            iv_name="kil2",
        )
        with loop as (kt2, carried):
            carried = list(carried)
            accs = carried[: ctx.n_acc]
            o = ctx.n_acc
            cur = unflat(carried[o : o + nf])
            o += nf
            regs = (carried[o : o + n_a], carried[o + n_a : o + n_a + n_b])
            kt = b.mul(kt2, b.const_i32(2))
            accs, cur, regs = tile(accs, cur, regs, kt, 0, 1, True)
            accs, cur, regs = tile(
                accs, cur, regs, b.add(kt, b.const_i32(1)), 1, 0, True
            )
            b.scf_yield(*accs, *flat(cur), *regs[0], *regs[1])
        res = list(loop.results)
        accs = res[: ctx.n_acc]
        o = ctx.n_acc
        cur = unflat(res[o : o + nf])
        o += nf
        regs = (res[o : o + n_a], res[o + n_a : o + n_a + n_b])
        if (nk - 1) % 2:
            accs, cur, regs = tile(accs, cur, regs, b.const_i32(nk - 2), 0, 1, True)
        stg = (nk - 1) % 2
        accs, _, _ = tile(accs, cur, None, b.const_i32(nk - 1), stg, None, False)
        return accs
    loop = b.scf_for_iter(
        b.const_i32(0), _ci(b, last), b.const_i32(1), iter_args, iv_name="kil"
    )
    with loop as (kt, carried):
        carried = list(carried)
        accs = carried[: ctx.n_acc]
        o = ctx.n_acc
        cur = unflat(carried[o : o + nf])
        o += nf
        regs = (carried[o : o + n_a], carried[o + n_a : o + n_a + n_b])
        stg = b.land(kt, b.const_i32(1))
        nstg = b.xor(stg, b.const_i32(1))
        accs, cur, regs = tile(accs, cur, regs, kt, stg, nstg, True)
        b.scf_yield(*accs, *flat(cur), *regs[0], *regs[1])
    res = list(loop.results)
    accs = res[: ctx.n_acc]
    cur = unflat(res[ctx.n_acc : ctx.n_acc + nf])
    kt = _ci(b, last)
    stg = b.land(kt, b.const_i32(1))
    accs, _, _ = tile(accs, cur, None, kt, stg, None, False)
    return accs


def _swz_const(k: int, mask: int, mode: int = 0) -> int:
    """``swz(k)`` for a compile-time K row (see ``GettF64Spec.lds_swizzle``)."""
    if mode == 1:
        return (16 * (k & 1) ^ 4 * ((k >> 1) & 3)) & mask
    if mode == 2:
        return (4 * ((k >> 1) & 7)) & mask
    return (16 * (k & 1) ^ 8 * ((k >> 2) & 7)) & mask


def _swz(b: IRBuilder, k: Value, mask: int, mode: int = 0) -> Value:
    """``swz(k)`` for a per-lane K row index."""
    if mode == 2:
        pair = b.land(b.lshr(k, b.const_i32(1)), b.const_i32(7))
        return b.land(b.mul(pair, b.const_i32(4)), b.const_i32(mask))
    odd = b.mul(b.land(k, b.const_i32(1)), b.const_i32(16))
    if mode == 1:
        pair = b.mul(b.land(b.lshr(k, b.const_i32(1)), b.const_i32(3)), b.const_i32(4))
        return b.land(b.xor(odd, pair), b.const_i32(mask))
    blk = b.mul(b.land(b.lshr(k, b.const_i32(2)), b.const_i32(7)), b.const_i32(8))
    return b.land(b.xor(odd, blk), b.const_i32(mask))


def _frag(b, lds, row, col, cx, wty=F64):
    """One operand fragment: a word, or the (re, im) pair of a complex element."""
    if cx:
        v = b.smem_load_vN(lds, row, b.mul(col, b.const_i32(2)), dtype=wty, n=2)
        return (b.vec_extract(v, 0), b.vec_extract(v, 1))
    return b.vec_extract(b.smem_load_vN(lds, row, col, dtype=wty, n=1), 0)


def _mma_step(b, atom, fa, fb, accs, p, complex_accs=2):
    """One k-step (K = 4) of MFMAs over the wave's atoms."""
    am_n, an_n = len(fa), len(fb)
    out = list(accs)
    if not p.complex_:
        for am in range(am_n):
            for an in range(an_n):
                i = am * an_n + an
                out[i] = b.mma(atom, fa[am], fb[an], out[i])
        return out
    if complex_accs == 4:
        # rr, ii, ri, ir: plain products, every sign is applied in the epilogue
        for am in range(am_n):
            ar, ai = fa[am]
            for an in range(an_n):
                br, bi = fb[an]
                i = 4 * (am * an_n + an)
                out[i] = b.mma(atom, ar, br, out[i])
                out[i + 1] = b.mma(atom, ai, bi, out[i + 1])
                out[i + 2] = b.mma(atom, ar, bi, out[i + 2])
                out[i + 3] = b.mma(atom, ai, br, out[i + 3])
        return out
    # Complex: Cr += ar*br - sa*sb*ai*bi ; Ci += sb*ar*bi + sa*ai*br, with
    # sa / sb = -1 for a conjugated operand. Negated fragments are built once.
    sa = -1 if p.conj_a else 1
    sb = -1 if p.conj_b else 1
    neg_ai = [b.fneg(a[1]) for a in fa] if (sa * sb == 1 or sa == -1) else None
    neg_bi = [b.fneg(x[1]) for x in fb] if sb == -1 else None
    for am in range(am_n):
        ar, ai = fa[am]
        for an in range(an_n):
            br, bi = fb[an]
            i = 2 * (am * an_n + an)
            acc_r, acc_i = out[i], out[i + 1]
            acc_r = b.mma(atom, ar, br, acc_r)
            acc_r = b.mma(atom, neg_ai[am] if sa * sb == 1 else ai, bi, acc_r)
            acc_i = b.mma(atom, ar, neg_bi[an] if sb == -1 else bi, acc_i)
            acc_i = b.mma(atom, neg_ai[am] if sa == -1 else ai, br, acc_i)
            out[i], out[i + 1] = acc_r, acc_i
    return out


class _Scalars:
    """alpha / beta and the uniform tests the epilogue selects on."""

    def __init__(self, b, alpha, beta, cx, p=None):
        self.alpha, self.beta, self.cx = alpha, beta, cx
        self.zero = b.const_f64(0.0) if p is None else _wconst(b, p, 0.0)
        if cx:
            ar_z = b.fcmp("oeq", alpha[0], self.zero)
            self.ai_zero = b.fcmp("oeq", alpha[1], self.zero)
            self.a_zero = b.land(ar_z, self.ai_zero)
            self.bi_zero = b.fcmp("oeq", beta[1], self.zero) if beta else None
        else:
            self.a_zero = b.fcmp("oeq", alpha[0], self.zero)


def _epilogue(b, spec, atom, accs, lane, wm, wn, m0, n0, rd, rc, alpha, beta, geo):
    p = spec.problem
    cx = p.complex_
    es = p.elem_bytes
    wty = _wty(p)
    dw = p.word_bytes // 4
    am_n, an_n = spec.wave_m // atom.m, spec.wave_n // atom.n
    # Output coordinates of each accumulator slot from the atom's catalog
    # lane map: slot ``s`` of lane ``l`` holds (row, col) = c_layout(l, s).
    c_rows = []
    col_l = None
    for slot in range(atom.c_frag_len):
        r, c = atom.c_layout().coord(b, lane, slot)
        c_rows.append(r)
        if col_l is None:
            col_l = c
    mw = b.add(m0, b.mul(wm, b.const_i32(spec.wave_m)))
    nw = b.add(n0, b.mul(wn, b.const_i32(spec.wave_n)))

    cols = []
    for an in range(an_n):
        n = b.add(nw, b.add(col_l, b.const_i32(an * atom.n)) if an else col_l)
        ok = b.cmp_lt(n, _ci(b, geo.N))
        n_safe = b.select(ok, n, b.const_i32(0))
        dn = _mixed(b, n_safe, geo.d_n)
        c_n = _mixed(b, n_safe, geo.c_n) if rc is not None else None
        cols.append((ok, dn, c_n))
    use_c = None
    if rc is not None:
        use_c = b.lnot(b.fcmp("oeq", beta[0], _wconst(b, p, 0.0)))
        if cx:
            use_c = b.lor(use_c, b.lnot(b.fcmp("oeq", beta[1], _wconst(b, p, 0.0))))
    sc = _Scalars(b, alpha, beta, cx, p)
    words = 2 if cx else 1
    # Per row group of atoms: every C load first, then the combines and the D
    # stores. C may be D itself (in place), so the compiler may not move a C
    # load above an earlier D store; issuing the loads of a group first keeps
    # them in flight together, and reading every C element of the group before
    # writing it keeps the in-place update correct.
    for am in range(am_n):
        elems = []
        for slot in range(atom.c_frag_len):
            r = c_rows[slot]
            m = b.add(mw, b.add(r, b.const_i32(am * atom.m)) if am else r)
            mok = b.cmp_lt(m, _ci(b, geo.M))
            m_safe = b.select(mok, m, b.const_i32(0))
            dm = _mixed(b, m_safe, geo.d_m)
            c_m = _mixed(b, m_safe, geo.c_m) if rc is not None else None
            for an in range(an_n):
                nok, dn, c_n = cols[an]
                ok = b.land(mok, nok)
                c_val = None
                if rc is not None:
                    coff = b.mul(b.add(c_m, c_n), b.const_i32(es))
                    coff = b.select(b.land(ok, use_c), coff, b.const_i32(_MASK_OFFSET))
                    raw = b.buffer_load_vN(rc, coff, b.const_i32(0), I32, dw * words)
                    c_val = b.vec_bitcast(raw, VectorType(wty, words))
                elems.append((slot, an, ok, dm, dn, c_val))
        for slot, an, ok, dm, dn, cv in elems:
            i = am * an_n + an
            if cx and spec.complex_accs == 4:
                # Cr = rr - sa*sb*ii ; Ci = sb*ri + sa*ir (sa / sb = -1: CONJ)
                rr, ii, ri, ir = (
                    b.vec_extract(accs[4 * i + j], slot) for j in range(4)
                )
                sa = -1 if p.conj_a else 1
                sb = -1 if p.conj_b else 1
                xr = b.fsub(rr, ii) if sa * sb == 1 else b.fadd(rr, ii)
                if sa == sb:
                    xi = b.fadd(ri, ir) if sa == 1 else b.fneg(b.fadd(ri, ir))
                else:
                    xi = b.fsub(ir, ri) if sa == 1 else b.fsub(ri, ir)
                acc = (xr, xi)
            elif cx:
                acc = (
                    b.vec_extract(accs[2 * i], slot),
                    b.vec_extract(accs[2 * i + 1], slot),
                )
            else:
                acc = (b.vec_extract(accs[i], slot),)
            c_val = None
            if cv is not None:
                c_val = tuple(b.vec_extract(cv, j) for j in range(words))
            out = _combine(b, acc, c_val, sc)
            doff = b.mul(b.add(dm, dn), b.const_i32(es))
            doff = b.select(ok, doff, b.const_i32(_MASK_OFFSET))
            vec = b.vec_pack(list(out), wty)
            if wty is not F32:
                vec = b.vec_bitcast(vec, VectorType(F32, dw * words))
            b.buffer_store_vN_f32(rd, doff, b.const_i32(0), vec, dw * words)


def _combine(b, acc, c_val, sc: _Scalars):
    """``alpha * acc + beta * c`` in f64 (C was read as 0 when ``beta == 0``).

    ``alpha == 0`` gives ``beta * c`` (the accumulator is not used); a complex
    alpha / beta with a zero imaginary part scales by its real part only.
    """
    if not sc.cx:
        (x,) = acc
        (a,) = sc.alpha
        if c_val is None:
            t = b.fmul(a, x)
            return (b.select(sc.a_zero, sc.zero, t),)
        bc = b.fmul(sc.beta[0], c_val[0])
        t = b.fma(a, x, bc)
        return (b.select(sc.a_zero, bc, t),)
    xr, xi = acc
    ar, ai = sc.alpha
    if c_val is None:
        # alpha * x: re = ar*xr - ai*xi, im = ar*xi + ai*xr (one fma each).
        nt = b.fneg(b.fmul(ai, xi))
        re_f = b.fma(ar, xr, nt)
        im_f = b.fma(ar, xi, b.fmul(ai, xr))
        re_r = b.fmul(ar, xr)
        im_r = b.fmul(ar, xi)
        re = b.select(sc.a_zero, sc.zero, b.select(sc.ai_zero, re_r, re_f))
        im = b.select(sc.a_zero, sc.zero, b.select(sc.ai_zero, im_r, im_f))
        return (re, im)
    br, bi = sc.beta
    cr, ci = c_val
    # beta * c first, then fold alpha * x in with fmas.
    tr_f = b.fma(br, cr, b.fneg(b.fmul(bi, ci)))
    ti_f = b.fma(br, ci, b.fmul(bi, cr))
    tr = b.select(sc.bi_zero, b.fmul(br, cr), tr_f)
    ti = b.select(sc.bi_zero, b.fmul(br, ci), ti_f)
    re_f = b.fma(ar, xr, b.fma(b.fneg(ai), xi, tr))
    im_f = b.fma(ar, xi, b.fma(ai, xr, ti))
    re_r = b.fma(ar, xr, tr)
    im_r = b.fma(ar, xi, ti)
    re = b.select(sc.a_zero, tr, b.select(sc.ai_zero, re_r, re_f))
    im = b.select(sc.a_zero, ti, b.select(sc.ai_zero, im_r, im_f))
    return (re, im)


def _workspace_rsrc(b, spec, geo, W, kz, lb, es):
    """Buffer resource of the partial-sum slice ``W[z][l]`` (``M*N`` elements,
    N fastest) this CTA writes (``split_k``)."""
    if geo.rt:
        zl = b.add(b.mul(kz, geo.L), lb)
        mn = b.mul(geo.M, geo.N)
        off = b.mul(b.mul(b.zext(zl, I64), b.zext(mn, I64)), b.const_i64(es))
        span = b.mul(mn, b.const_i32(es))
    else:
        p = spec.problem
        zl = b.add(b.mul(kz, b.const_i32(p.L)), lb)
        off = b.mul(b.zext(zl, I64), b.const_i64(p.M * p.N * es))
        span = b.const_i32(p.M * p.N * es)
    return b.buffer_rsrc(b.global_ptr_add(W, off), span)


def _acc_value(b, spec, accs, i, slot):
    """The (re[, im]) accumulator value of atom ``i`` at ``slot``."""
    p = spec.problem
    if p.complex_ and spec.complex_accs == 4:
        # Cr = rr - sa*sb*ii ; Ci = sb*ri + sa*ir (sa / sb = -1: CONJ)
        rr, ii, ri, ir = (b.vec_extract(accs[4 * i + j], slot) for j in range(4))
        sa = -1 if p.conj_a else 1
        sb = -1 if p.conj_b else 1
        xr = b.fsub(rr, ii) if sa * sb == 1 else b.fadd(rr, ii)
        if sa == sb:
            xi = b.fadd(ri, ir) if sa == 1 else b.fneg(b.fadd(ri, ir))
        else:
            xi = b.fsub(ir, ri) if sa == 1 else b.fsub(ri, ir)
        return (xr, xi)
    if p.complex_:
        return (
            b.vec_extract(accs[2 * i], slot),
            b.vec_extract(accs[2 * i + 1], slot),
        )
    return (b.vec_extract(accs[i], slot),)


def _epilogue_partial(b, spec, atom, accs, lane, wm, wn, m0, n0, rw, geo):
    """``split_k``: the accumulators to the workspace slice, row ``m`` at
    ``m*N`` (no C, no alpha / beta)."""
    p = spec.problem
    words = 2 if p.complex_ else 1
    es = p.elem_bytes
    am_n, an_n = spec.wave_m // atom.m, spec.wave_n // atom.n
    c_rows = []
    col_l = None
    for slot in range(atom.c_frag_len):
        r, c = atom.c_layout().coord(b, lane, slot)
        c_rows.append(r)
        if col_l is None:
            col_l = c
    mw = b.add(m0, b.mul(wm, b.const_i32(spec.wave_m)))
    nw = b.add(n0, b.mul(wn, b.const_i32(spec.wave_n)))
    cols = []
    for an in range(an_n):
        n = b.add(nw, b.add(col_l, b.const_i32(an * atom.n)) if an else col_l)
        cols.append((b.cmp_lt(n, _ci(b, geo.N)), n))
    for am in range(am_n):
        for slot in range(atom.c_frag_len):
            r = c_rows[slot]
            m = b.add(mw, b.add(r, b.const_i32(am * atom.m)) if am else r)
            mok = b.cmp_lt(m, _ci(b, geo.M))
            row = b.mul(m, _ci(b, geo.N))
            for an in range(an_n):
                nok, n = cols[an]
                ok = b.land(mok, nok)
                acc = _acc_value(b, spec, accs, am * an_n + an, slot)
                off = b.mul(b.add(row, n), b.const_i32(es))
                off = b.select(ok, off, b.const_i32(_MASK_OFFSET))
                vec = b.vec_bitcast(
                    b.vec_pack(list(acc), F64), VectorType(F32, 2 * words)
                )
                b.buffer_store_vN_f32(rw, off, b.const_i32(0), vec, 2 * words)


def _serial_slot_bytes(spec: GettF64Spec) -> int:
    """Bytes of one wave's partial-sum slot (``split_serial``): the wave's
    accumulator registers, f64 16x16 atoms (4 values per lane)."""
    p = spec.problem
    n_acc = (
        (spec.wave_m // 16)
        * (spec.wave_n // 16)
        * (spec.complex_accs if p.complex_ else 1)
    )
    return n_acc * 4 * 64 * 8


def _tiles(spec: GettF64Spec) -> int:
    p = spec.problem
    return -(-p.M // spec.tile_m) * -(-p.N // spec.tile_n)


class _SerialReduce:
    """``split_serial``: store this wave's partial accumulators, count the
    arrivals of its (batch, tile, wave) and, in the wave that arrives last,
    yield the sums of the ``split_k`` partials taken in the order z = 0 ..
    split_k - 1 (the body of the ``with`` runs only in that wave)."""

    def __init__(self, b, spec, atom, accs, W, Cnt, kz, lb, tile, wave, lane):
        self.b, self.spec, self.atom = b, spec, atom
        self.accs, self.W, self.Cnt = list(accs), W, Cnt
        self.kz, self.lb, self.tile, self.wave, self.lane = kz, lb, tile, wave, lane
        self._if = None

    def _rsrc(self, z):
        b, spec = self.b, self.spec
        p = spec.problem
        waves = spec.warps_m * spec.warps_n
        sb = _serial_slot_bytes(spec)
        zl = b.add(b.mul(z, b.const_i32(p.L)), self.lb)
        sl = b.add(
            b.mul(
                b.add(b.mul(zl, b.const_i32(_tiles(spec))), self.tile),
                b.const_i32(waves),
            ),
            self.wave,
        )
        off = b.mul(b.zext(sl, I64), b.const_i64(sb))
        return b.buffer_rsrc(b.global_ptr_add(self.W, off), b.const_i32(sb))

    def _addr(self, z, i, h):
        """64-bit address of half ``h`` of accumulator ``i`` of this lane in
        the slot of split ``z`` (write-through mode)."""
        b, spec = self.b, self.spec
        p = spec.problem
        waves = spec.warps_m * spec.warps_n
        zl = b.add(b.mul(z, b.const_i32(p.L)), self.lb)
        sl = b.add(
            b.mul(
                b.add(b.mul(zl, b.const_i32(_tiles(spec))), self.tile),
                b.const_i32(waves),
            ),
            self.wave,
        )
        off = b.add(
            b.mul(b.zext(sl, I64), b.const_i64(_serial_slot_bytes(spec))),
            b.const_i64((i * 2 + h) * 64 * 16),
        )
        lane_b = b.zext(b.mul(self.lane, b.const_i32(16)), I64)
        return b.add(b.global_ptr_to_i64(self.W), b.add(off, lane_b))

    def __enter__(self):
        b, spec, atom = self.b, self.spec, self.atom
        if atom.c_frag_len != 4:  # pragma: no cover - the f64 16x16x4 atom
            raise ValueError("split_serial expects 4 accumulator values per lane")
        voff = b.mul(self.lane, b.const_i32(16))

        def soff(i, h):
            return b.const_i32((i * 2 + h) * 64 * 16)

        wt = spec.split_serial == 2
        halves = [
            (
                i,
                h,
                b.vec_bitcast(
                    b.vec_pack([b.vec_extract(acc, 2 * h + j) for j in range(2)], F64),
                    VectorType(F32, 4),
                ),
            )
            for i, acc in enumerate(self.accs)
            for h in range(2)
        ]
        if wt:
            # write-through: sc1 stores in asm groups that end with
            # vmcnt(0), so the partial is in memory before the counter bump
            for g0 in range(0, len(halves), 8):
                grp = halves[g0 : g0 + 8]
                tmpl, ops = [], []
                for j, (i, h, v4) in enumerate(grp):
                    tmpl.append(f"global_store_dwordx4 ${2 * j}, ${2 * j + 1}, off sc1")
                    ops += [self._addr(self.kz, i, h), v4]
                tmpl.append("s_waitcnt vmcnt(0)")
                b.inline_asm("\n".join(tmpl), ",".join(["v"] * len(ops)), ops)
        else:
            rs = self._rsrc(self.kz)
            for i, h, v4 in halves:
                b.buffer_store_vN_f32(rs, voff, soff(i, h), v4, 4)
            b.fence(scope="agent", ordering="release")
        waves = spec.warps_m * spec.warps_n
        idx = b.add(
            b.mul(
                b.add(b.mul(self.lb, b.const_i32(_tiles(spec))), self.tile),
                b.const_i32(waves),
            ),
            self.wave,
        )
        one = b.select(
            b.cmp_eq(self.lane, b.const_i32(0)), b.const_i32(1), b.const_i32(0)
        )
        old = b.global_atomic_add(
            self.Cnt, idx, one, ordering="monotonic" if wt else "acq_rel"
        )
        ticket = b.readfirstlane(old)
        self._if = b.scf_if(b.cmp_eq(ticket, b.const_i32(spec.split_k - 1)))
        self._if.__enter__()
        sums = [None] * len(self.accs)
        if wt:
            # write-through loads, up to 16 in flight per asm group, summed
            # in z order
            lq = [
                (z, i, h)
                for z in range(spec.split_k)
                for i in range(len(self.accs))
                for h in range(2)
            ]
            got = {}
            for g0 in range(0, len(lq), 16):
                grp = lq[g0 : g0 + 16]
                n = len(grp)
                tmpl = [
                    f"global_load_dwordx4 ${j}, ${n + j}, off sc1" for j in range(n)
                ]
                tmpl.append("s_waitcnt vmcnt(0)")
                res = b.inline_asm_multi(
                    "\n".join(tmpl),
                    ",".join(["=&v"] * n + ["v"] * n),
                    [self._addr(b.const_i32(z), i, h) for z, i, h in grp],
                    result_types=[VectorType(F32, 4)] * n,
                )
                for (z, i, h), r in zip(grp, res):
                    got[(z, i, h)] = b.vec_bitcast(r, VectorType(F64, 2))
                for z, i in sorted({(z, i) for z, i, _ in grp}):
                    if (z, i, 0) in got and (z, i, 1) in got:
                        lo, hi = got.pop((z, i, 0)), got.pop((z, i, 1))
                        v = b.vec_pack(
                            [
                                b.vec_extract(lo, 0),
                                b.vec_extract(lo, 1),
                                b.vec_extract(hi, 0),
                                b.vec_extract(hi, 1),
                            ],
                            F64,
                        )
                        sums[i] = v if sums[i] is None else b.fadd(sums[i], v)
            b.global_store(self.Cnt, idx, b.const_i32(0), align=4)
            return sums
        b.fence(scope="agent", ordering="acquire")
        for z in range(spec.split_k):
            rz = self._rsrc(b.const_i32(z))
            for i in range(len(self.accs)):
                parts = []
                for h in range(2):
                    raw = b.buffer_load_vN(rz, voff, soff(i, h), I32, 4)
                    two = b.vec_bitcast(raw, VectorType(F64, 2))
                    parts += [b.vec_extract(two, j) for j in range(2)]
                v = b.vec_pack(parts, F64)
                sums[i] = v if sums[i] is None else b.fadd(sums[i], v)
        b.global_store(self.Cnt, idx, b.const_i32(0), align=4)
        return sums

    def __exit__(self, *exc):
        return self._if.__exit__(*exc)


#: Threads per block of :func:`build_gett_f64_reduce` (one output each).
REDUCE_THREADS = 256


def gett_f64_workspace_bytes(
    spec: GettF64Spec, problem: Optional[GettF64Problem] = None
) -> int:
    """Bytes of the ``split_k`` partial-sum workspace (0 without split_k)."""
    if spec.split_k <= 1:
        return 0
    if problem is None:
        if spec.runtime_sizes:
            raise ValueError(
                "a runtime_sizes kernel needs the problem for its workspace"
            )
        problem = spec.problem
    p = problem
    if spec.split_serial:
        return (
            spec.split_k
            * p.L
            * -(-p.M // spec.tile_m)
            * -(-p.N // spec.tile_n)
            * spec.warps_m
            * spec.warps_n
            * _serial_slot_bytes(spec)
        )
    return spec.split_k * p.L * p.M * p.N * p.elem_bytes


def gett_f64_counter_count(
    spec: GettF64Spec, problem: Optional[GettF64Problem] = None
) -> int:
    """i32 counters of a ``split_serial`` launch (0 otherwise): one per batch,
    output tile and wave. Zero them once when they are allocated; every launch
    leaves them at zero."""
    if not spec.split_serial:
        return 0
    p = problem or spec.problem
    return (
        p.L
        * -(-p.M // spec.tile_m)
        * -(-p.N // spec.tile_n)
        * spec.warps_m
        * spec.warps_n
    )


def gett_f64_reduce_name(spec: GettF64Spec) -> str:
    """Kernel name of :func:`build_gett_f64_reduce` for ``spec``."""
    p = spec.problem
    if spec.runtime_sizes:
        tp = runtime_template(p)
        sizes, h = "rt", _problem_hash(tp) if tp is not None else "invalid"
    else:
        sizes, h = f"M{p.M}N{p.N}K{p.K}L{p.L}", _problem_hash(p)
    return kernel_name_join(
        spec.name + "_reduce",
        "c128" if p.complex_ else "f64",
        sizes,
        f"sk{spec.split_k}",
        f"h{h}",
        flags={"bil": spec.bilinear},
    )


def build_gett_f64_reduce(spec: GettF64Spec, *, arch: str = "gfx950") -> KernelDef:
    """Second kernel of ``split_k``: ``D = alpha * sum_z W[z] + beta * C``.

    The ``split_k`` partials of each output are added in order ``z = 0, 1,
    ...`` (deterministic), then the GETT epilogue applies (f64 alpha / beta, C
    through its own strides and only when ``beta != 0``, ``alpha == 0`` gives
    ``beta * c``, D through its own strides). One thread per output: grid
    :func:`gett_f64_reduce_grid`, block :data:`REDUCE_THREADS`. Kernel
    arguments: ``W, [C,] D``, alpha, [beta], then the size arguments with
    ``runtime_sizes`` (:func:`gett_f64_reduce_signature`).
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gett_f64 spec for {arch}: {why}")
    if spec.split_k <= 1:
        raise ValueError("build_gett_f64_reduce needs split_k > 1")
    if spec.runtime_sizes:
        spec = replace(spec, problem=runtime_template(spec.problem))
    p = spec.problem
    cx = p.complex_
    es = p.elem_bytes
    words = 2 if cx else 1
    b = IRBuilder(gett_f64_reduce_name(spec))
    b.kernel.attrs["max_workgroup_size"] = REDUCE_THREADS
    gptr = PtrType(F64, "global")
    W = b.param("W", gptr, noalias=True, readonly=True, align=8)
    C = b.param("C", gptr, readonly=True, align=8) if spec.bilinear else None
    D = b.param("D", gptr, align=8)
    alpha = [b.param("alpha_re" if cx else "alpha", F64)]
    if cx:
        alpha.append(b.param("alpha_im", F64))
    beta = []
    if spec.bilinear:
        beta = [b.param("beta_re" if cx else "beta", F64)]
        if cx:
            beta.append(b.param("beta_im", F64))
    rt_args = None
    if spec.runtime_sizes:
        rt_args = {
            n: b.param(n, I32 if t == "i32" else I64)
            for n, t in _runtime_arg_names(spec)
        }
    geo = _Geo(b, spec, rt_args)
    cm, cn, cl = p.c_digits()

    tid = b.thread_id_x()
    idx = b.add(b.mul(b.block_id_x(), b.const_i32(REDUCE_THREADS)), tid)
    lb = b.block_id_y()
    if geo.rt:
        total = b.mul(geo.M, geo.N)
        slice64 = b.mul(b.zext(total, I64), b.const_i64(es))
        span = b.mul(total, b.const_i32(es))
    else:
        total = p.M * p.N
        slice64 = None
        span = b.const_i32(total * es)
    ok = b.cmp_lt(idx, _ci(b, total))
    idx_safe = b.select(ok, idx, b.const_i32(0))
    n = b.mod(idx_safe, _ci(b, geo.N))
    m = b.div(idx_safe, _ci(b, geo.N))
    woff = b.select(ok, b.mul(idx, b.const_i32(es)), b.const_i32(_MASK_OFFSET))
    parts = []
    for z in range(spec.split_k):
        if geo.rt:
            zl = b.add(b.mul(b.const_i32(z), geo.L), lb)
            base = b.global_ptr_add(W, b.mul(b.zext(zl, I64), slice64))
        else:
            zl = b.add(b.const_i32(z * p.L), lb)
            base = b.global_ptr_add(W, b.mul(b.zext(zl, I64), b.const_i64(total * es)))
        raw = b.buffer_load_vN(
            b.buffer_rsrc(base, span), woff, b.const_i32(0), I32, 2 * words
        )
        parts.append(b.vec_bitcast(raw, VectorType(F64, words)))
    acc = parts[0]
    for v in parts[1:]:
        acc = b.fadd(acc, v)

    def at_batch(ptr, digits, rt_name):
        if geo.rt:
            off = b.mul(b.zext(lb, I64), geo.batch[rt_name])
            return b.global_ptr_add(ptr, b.mul(off, b.const_i64(es)))
        if digits_extent(digits) == 1:
            return ptr
        return b.global_ptr_add(ptr, _mixed64(b, lb, digits, es))

    def out_span(m_dig, n_dig, which):
        if geo.rt:
            return _rt_span(
                b,
                (
                    (b.sub(geo.M, b.const_i32(1)), geo.stride(m_dig)),
                    (b.sub(geo.N, b.const_i32(1)), geo.stride(n_dig)),
                ),
                es,
            )
        return b.const_i32(p.addr_span_bytes(which, spec.tile_k))

    dptr = at_batch(D, p.d_l, "d_sl")
    rd = b.buffer_rsrc(dptr, out_span(geo.d_m, geo.d_n, "D"))
    c_val = None
    sc_beta = beta
    if C is not None:
        cptr = at_batch(C, cl, "c_sl")
        rc = b.buffer_rsrc(cptr, out_span(geo.c_m, geo.c_n, "C"))
        use_c = b.lnot(b.fcmp("oeq", beta[0], b.const_f64(0.0)))
        if cx:
            use_c = b.lor(use_c, b.lnot(b.fcmp("oeq", beta[1], b.const_f64(0.0))))
        coff = b.mul(
            b.add(_mixed(b, m, geo.c_m), _mixed(b, n, geo.c_n)), b.const_i32(es)
        )
        coff = b.select(b.land(ok, use_c), coff, b.const_i32(_MASK_OFFSET))
        craw = b.buffer_load_vN(rc, coff, b.const_i32(0), I32, 2 * words)
        cv = b.vec_bitcast(craw, VectorType(F64, words))
        c_val = tuple(b.vec_extract(cv, j) for j in range(words))
    sc = _Scalars(b, alpha, sc_beta, cx)
    accv = tuple(b.vec_extract(acc, j) for j in range(words))
    out = _combine(b, accv, c_val, sc)
    doff = b.mul(b.add(_mixed(b, m, geo.d_m), _mixed(b, n, geo.d_n)), b.const_i32(es))
    doff = b.select(ok, doff, b.const_i32(_MASK_OFFSET))
    vec = b.vec_bitcast(b.vec_pack(list(out), F64), VectorType(F32, 2 * words))
    b.buffer_store_vN_f32(rd, doff, b.const_i32(0), vec, 2 * words)
    b.ret()
    return b.kernel


def gett_f64_reduce_grid(
    spec: GettF64Spec, problem: Optional[GettF64Problem] = None
) -> Tuple[int, int, int]:
    """Grid of :func:`build_gett_f64_reduce`: one thread per output, batch on y."""
    if problem is None:
        if spec.runtime_sizes:
            raise ValueError("a runtime_sizes kernel needs the problem for its grid")
        problem = spec.problem
    p = problem
    if p.L > MAX_GRID_Y:
        raise ValueError(f"L = {p.L} exceeds the grid limit {MAX_GRID_Y}")
    return (-(-(p.M * p.N) // REDUCE_THREADS), p.L, 1)


def gett_f64_reduce_signature(spec: GettF64Spec):
    """Kernel arguments of :func:`build_gett_f64_reduce`."""
    sb = SignatureBuilder().ptr("W", "f64")
    if spec.bilinear:
        sb = sb.ptr("C", "f64")
    sb = sb.ptr("D", "f64")
    if spec.problem.complex_:
        sb = sb.scalar("alpha_re", "f64").scalar("alpha_im", "f64")
        if spec.bilinear:
            sb = sb.scalar("beta_re", "f64").scalar("beta_im", "f64")
    else:
        sb = sb.scalar("alpha", "f64")
        if spec.bilinear:
            sb = sb.scalar("beta", "f64")
    if spec.runtime_sizes:
        for name, dtype in _runtime_arg_names(spec):
            sb = sb.scalar(name, dtype)
    return sb.build()


#: K tiles :func:`pick_tile_k` tries after the per-arch default.
TILE_K_CANDIDATES = (8, 16, 12, 24, 20, 4)


def pick_tile_k(problem: GettF64Problem, preferred: int, max_tile_k: int = 32) -> int:
    """``preferred`` if both operands' K digits split at it within the address
    budget, else the first such K tile of :data:`TILE_K_CANDIDATES`; falls back
    to ``preferred`` (per-lane K offsets) when none splits."""

    def fits(t: int) -> bool:
        if t > max_tile_k:
            return False
        for which, kd in (("A", problem.a_k), ("B", problem.b_k)):
            if k_split(kd, t) is None or problem.addr_span_bytes(which, t) > _MAX_SPAN:
                return False
        return True

    for t in (preferred, *TILE_K_CANDIDATES):
        if fits(t):
            return t
    return preferred


def arch_spec(problem: GettF64Problem, arch: str, **knobs) -> GettF64Spec:
    """A spec with the per-arch defaults, overridden by ``knobs``.

    gfx90a / gfx942 have 64 KiB of LDS, so the two K stages there hold 8 rows
    (16 on gfx950): two CTAs fit per CU on every target. Complex problems
    default to 64x64 CTA tiles (each complex atom keeps two accumulators).
    ``waves_per_eu=2`` lets the backend keep a k-step of fragments in flight.
    """
    base = {"waves_per_eu": 2}
    if problem.complex_:
        base.update(tile_m=64, tile_n=64)
    if "tile_k" not in knobs:
        # The K tile is also where the K digits must split (the K-tile offset
        # then moves the base pointer); keep the LDS of two stages per CTA
        # within the per-arch default.
        preferred = 16 if arch == "gfx950" else 8
        base["tile_k"] = pick_tile_k(problem, preferred, max_tile_k=2 * preferred)
    base.update(knobs)
    return GettF64Spec(problem, **base)


def resolve_num_cus(arch: str, num_cus: Optional[int] = None) -> int:
    """The CU count the shape classes use.

    ``num_cus`` when given; otherwise the visible device's count when that
    device is ``arch``. Parts that share a gfx id differ in CU count, so there
    is no per-arch default: without a device to ask, the caller must say.
    """
    if num_cus is not None:
        if int(num_cus) <= 0:
            raise ValueError(f"num_cus must be positive, got {num_cus}")
        return int(num_cus)
    from ...runtime.hip_module import get_device_arch, get_device_num_cus

    if (get_device_arch() or "") == arch:
        n = get_device_num_cus()
        if n:
            return int(n)
    raise ValueError(
        f"num_cus is required for {arch}: no {arch} device to query, and parts "
        "with one gfx id differ in CU count"
    )


_SWZ0 = dict(lds_pad=None, lds_swizzle=True, lds_swizzle_mode=0)
_SWZ1 = dict(lds_pad=None, lds_swizzle=True, lds_swizzle_mode=1)
_SWZ2 = dict(lds_pad=None, lds_swizzle=True, lds_swizzle_mode=2)

#: Tuned configurations per (arch, real / complex, shape class), selected by
#: :func:`tuned_spec` through :func:`gett_f64_shape_class`. See
#: dsl_docs/instances/gett_f64.md, "Tuned configurations".
TUNED_CONFIGS = {
    ("gfx950", "real", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=8,
        xcds=8,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=True,
        **_SWZ2,
    ),
    ("gfx950", "real", "medium"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=8,
        warps_m=2,
        warps_n=2,
        group_m=32,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=True,
        **_SWZ2,
    ),
    ("gfx950", "real", "large"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=32,
        waves_per_eu=2,
        lds_stages=2,
        unroll2=True,
        sched_fence=False,
        **_SWZ0,
    ),
    ("gfx950", "complex", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=24,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=8,
        waves_per_eu=3,
        lds_stages=2,
        unroll2=True,
        sched_fence=False,
        **_SWZ0,
    ),
    ("gfx950", "complex", "medium"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=8,
        warps_m=1,
        warps_n=4,
        group_m=32,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=True,
        **_SWZ0,
    ),
    ("gfx950", "complex", "large"): dict(
        tile_m=128,
        tile_n=64,
        tile_k=8,
        warps_m=1,
        warps_n=4,
        group_m=16,
        waves_per_eu=2,
        lds_stages=1,
        sched_fence=False,
        lds_swizzle=False,
        lds_pad=4,
    ),
    ("gfx90a", "real", "small"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=12,
        warps_m=2,
        warps_n=2,
        group_m=4,
        waves_per_eu=3,
        lds_stages=1,
        sched_fence=True,
        lds_swizzle=False,
        lds_pad=4,
    ),
    ("gfx90a", "real", "medium"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=4,
        warps_n=2,
        group_m=4,
        waves_per_eu=1,
        lds_stages=1,
        sched_fence=True,
        **_SWZ0,
    ),
    ("gfx90a", "real", "kmixed"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=4,
        warps_n=2,
        group_m=8,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=True,
        **_SWZ0,
    ),
    ("gfx90a", "real", "kk"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=4,
        warps_n=2,
        group_m=32,
        waves_per_eu=1,
        lds_stages=1,
        sched_fence=True,
        lds_swizzle=False,
        lds_pad=1,
    ),
    ("gfx90a", "real", "wide"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=4,
        waves_per_eu=2,
        lds_stages=1,
        sched_fence=False,
        **_SWZ2,
    ),
    ("gfx90a", "real", "pow2"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=8,
        warps_m=2,
        warps_n=2,
        group_m=1,
        waves_per_eu=2,
        lds_stages=1,
        sched_fence=True,
        **_SWZ2,
    ),
    ("gfx90a", "complex", "medium"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=8,
        warps_m=2,
        warps_n=2,
        group_m=8,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=True,
        **_SWZ1,
    ),
    ("gfx90a", "complex", "wide"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=4,
        warps_m=2,
        warps_n=2,
        group_m=8,
        waves_per_eu=2,
        lds_stages=2,
        unroll2=True,
        sched_fence=False,
        **_SWZ1,
    ),
    ("gfx90a", "complex", "large"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=8,
        warps_m=4,
        warps_n=2,
        group_m=1,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=False,
        prefetch_depth=2,
        **_SWZ1,
    ),
    ("gfx942", "real", "small"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=32,
        warps_m=2,
        warps_n=4,
        group_m=4,
        waves_per_eu=3,
        lds_stages=1,
        sched_fence=True,
        prefetch_depth=2,
        **_SWZ2,
    ),
    ("gfx942", "real", "medium"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=4,
        xcds=8,
        waves_per_eu=2,
        lds_stages=1,
        sched_fence=True,
        **_SWZ1,
    ),
    ("gfx942", "complex", "small"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=8,
        warps_m=2,
        warps_n=2,
        group_m=8,
        waves_per_eu=2,
        lds_stages=2,
        sched_fence=True,
    ),
    ("gfx942", "complex", "medium"): dict(
        tile_m=128,
        tile_n=64,
        tile_k=8,
        warps_m=4,
        warps_n=2,
        group_m=1,
        xcds=8,
        waves_per_eu=4,
        lds_stages=1,
        sched_fence=False,
        prefetch_depth=2,
        lds_swizzle=False,
        lds_pad=4,
    ),
}


#: Tuned configurations of the f32 / complex-f32 GETT (``problem.dtype ==
#: "f32"``: f32 words, the f32 16x16x4 atom) per (arch, real / complex, shape
#: class), selected by :func:`tuned_spec` through :func:`gett_f32_shape_class`.
#: Found with the search engine (rocke.dispatch.tuning.search) on the packed,
#: K-contiguous layouts hipTensor's native f32 route launches (runtime sizes),
#: then chosen from the cross-evaluation of every problem's winner on every
#: problem; every knob is explicit. Targets without entries use
#: :data:`TUNED_CONFIGS` (:func:`tuned_configs`). See
#: dsl_docs/instances/gett_f64.md, "Tuned f32 configurations".
TUNED_CONFIGS_F32 = {
    ("gfx942", "real", "compact"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=16,
        xcds=0,
        waves_per_eu=3,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "real", "huge"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=2,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "real", "large"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=4,
        xcds=8,
        waves_per_eu=3,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "real", "medium"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=3,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "real", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warps_m=1,
        warps_n=2,
        group_m=8,
        xcds=0,
        waves_per_eu=3,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "real", "tiny"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=64,
        warps_m=2,
        warps_n=2,
        group_m=4,
        xcds=0,
        waves_per_eu=2,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "real", "xlarge"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=4,
        xcds=8,
        waves_per_eu=3,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "complex", "compact"): dict(
        tile_m=64,
        tile_n=32,
        tile_k=24,
        warps_m=4,
        warps_n=1,
        group_m=1,
        xcds=0,
        waves_per_eu=3,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=4,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "complex", "large"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=4,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=2,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "complex", "medium"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=8,
        warps_m=4,
        warps_n=2,
        group_m=2,
        xcds=0,
        waves_per_eu=1,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "complex", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=24,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=3,
        lds_stages=1,
        prefetch_depth=2,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx942", "complex", "wide"): dict(
        tile_m=128,
        tile_n=64,
        tile_k=4,
        warps_m=2,
        warps_n=2,
        group_m=2,
        xcds=0,
        waves_per_eu=3,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "real", "huge"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=3,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=True,
        sched_fence=False,
        k_stagger=1,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=1,
    ),
    ("gfx950", "real", "large"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=8,
        waves_per_eu=4,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "real", "medium"): dict(
        tile_m=64,
        tile_n=32,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=8,
        waves_per_eu=4,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=1,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "real", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=64,
        warps_m=2,
        warps_n=2,
        group_m=4,
        xcds=8,
        waves_per_eu=4,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=1,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "real", "tiny"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=64,
        warps_m=2,
        warps_n=2,
        group_m=8,
        xcds=8,
        waves_per_eu=4,
        lds_stages=2,
        prefetch_depth=2,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=1,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "real", "xlarge"): dict(
        tile_m=128,
        tile_n=64,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=16,
        xcds=0,
        waves_per_eu=4,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=True,
        sched_fence=False,
        k_stagger=0,
        lds_pad=None,
        lds_swizzle=True,
        lds_swizzle_mode=1,
    ),
    ("gfx950", "complex", "large"): dict(
        tile_m=64,
        tile_n=128,
        tile_k=8,
        warps_m=1,
        warps_n=4,
        group_m=32,
        xcds=0,
        waves_per_eu=4,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=4,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "complex", "longk"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=4,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=True,
        sched_fence=True,
        k_stagger=0,
        lds_pad=4,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "complex", "medium"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=8,
        warps_m=2,
        warps_n=2,
        group_m=32,
        xcds=0,
        waves_per_eu=4,
        lds_stages=1,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=4,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
    ("gfx950", "complex", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=4,
        xcds=8,
        waves_per_eu=4,
        lds_stages=2,
        prefetch_depth=1,
        unroll2=False,
        sched_fence=True,
        k_stagger=0,
        lds_pad=4,
        lds_swizzle=False,
        lds_swizzle_mode=0,
    ),
}


def _along_k_count(problem: GettF64Problem, tile_k: int = 16) -> int:
    """Operands the loader reads along K (K-contiguous in memory)."""
    probe = GettF64Spec(problem, tile_k=tile_k)
    n = 0
    for side, kd, tile in (
        (problem.a_m, problem.a_k, 128),
        (problem.b_n, problem.b_k, 128),
    ):
        lp = _load_plan(probe, side, kd, tile)
        n += int(lp is not None and lp.along_k)
    return n


def _pow2_k_rows(problem: GettF64Problem, min_bytes: int = 64 << 10) -> bool:
    """Some operand's consecutive K rows (its first K digit) sit a multiple of
    ``min_bytes`` apart: every K row of a tile maps to the same memory
    channels (power-of-two leading dimensions)."""
    es = problem.elem_bytes
    for kd in (problem.a_k, problem.b_k):
        stride = kd[0][1] * es
        if stride >= min_bytes and stride % min_bytes == 0:
            return True
    return False


def gett_f64_shape_class(
    problem: GettF64Problem, arch: str, num_cus: Optional[int] = None
) -> str:
    """The :data:`TUNED_CONFIGS` class of ``problem`` on ``arch``.

    Real: ``small`` when the 128x128 tile grid is under one tile per CU (two
    on gfx90a); on gfx950 ``large`` when K is long (>= 4096) or the grid is
    >= 8 tiles per CU; otherwise ``medium``. gfx90a
    classes the medium / large real problems by the operands read along K
    instead (``kk`` both, ``kmixed`` one); with none, ``pow2`` when an
    operand's K rows are a multiple of 64 KiB apart, ``wide`` for >= 64
    tiles per CU with K <= 1024, else ``medium``. Complex: ``small`` when
    the 64x64 grid is under four tiles per CU (two on gfx942), ``large``
    when K >= 2048; on gfx90a ``wide`` for >= 64 64x64 tiles per CU.

    ``num_cus``: see :func:`resolve_num_cus` (required without an ``arch``
    device to query).
    """
    cus = resolve_num_cus(arch, num_cus)
    p = problem
    if p.complex_:
        tiles = -(-p.M // 64) * -(-p.N // 64) * p.L
        per_cu = 2 if arch == "gfx942" else 4
        if tiles < per_cu * cus and (arch, "complex", "small") in TUNED_CONFIGS:
            return "small"
        if p.K >= 2048 and (arch, "complex", "large") in TUNED_CONFIGS:
            return "large"
        if tiles >= 64 * cus and (arch, "complex", "wide") in TUNED_CONFIGS:
            return "wide"
        return "medium"
    tiles = -(-p.M // 128) * -(-p.N // 128) * p.L
    if tiles < (2 if arch == "gfx90a" else 1) * cus:
        return "small"
    if arch == "gfx90a":
        nk = _along_k_count(p)
        if nk:
            return "kk" if nk == 2 else "kmixed"
        if _pow2_k_rows(p):
            return "pow2"
        if tiles >= 64 * cus and p.K <= 1024:
            return "wide"
        return "medium"
    if arch == "gfx950" and (p.K >= 4096 or tiles >= 8 * cus):
        return "large"
    return "medium"


def _tiles_le(tiles: int, cus: int, octave: int) -> bool:
    """``tiles <= cus * 2^octave`` in integers."""
    if octave >= 0:
        return tiles <= cus << octave
    return tiles << -octave <= cus


def tuned_configs(arch: str, kind: str, dtype: str = "f64") -> Dict[str, dict]:
    """``{class: knobs}`` of the tuned configurations of ``(arch, kind)`` for
    ``dtype``: f32 problems take :data:`TUNED_CONFIGS_F32` where it has
    classes for ``(arch, kind)``, else (and f64 problems) :data:`TUNED_CONFIGS`."""
    if dtype == "f32":
        own = {
            c: cfg
            for (a, k, c), cfg in TUNED_CONFIGS_F32.items()
            if a == arch and k == kind
        }
        if own:
            return own
    return {
        c: cfg for (a, k, c), cfg in TUNED_CONFIGS.items() if a == arch and k == kind
    }


def gett_f32_shape_class(
    problem: GettF64Problem, arch: str, num_cus: Optional[int] = None
) -> str:
    """The :data:`TUNED_CONFIGS_F32` class of an f32 problem on ``arch``
    (:func:`gett_f64_shape_class` on a target without f32 classes).

    gfx942 real, by the 128x128 tiles per CU (``t``): ``tiny`` for t <= 1/8,
    ``small`` <= 1/4; up to 8 per CU ``medium`` when min(M, N) <= 4096, else
    ``large``; above, ``xlarge`` for K <= 1024, else ``huge`` (``compact``
    serves the selection models only). gfx942 complex, by the 64x64 tiles per
    CU: ``small`` <= 1/2, ``compact`` <= 1, ``medium`` <= 64, else ``large``
    (``wide``: the selection models only).
    gfx950 real, by the 128x128 tiles of the grid per CU (``t``): ``tiny``
    for t <= 1/8, ``small`` <= 1/2, ``medium`` <= 1, ``large`` <= 4,
    ``xlarge`` <= 8, else ``huge``. gfx950 complex, by the 64x64 tiles per
    CU: up to 4 per CU, ``small`` (K <= 256 and min(M, N) <= 512),
    ``medium`` (K <= 256) or ``longk``; up to 32 per CU ``medium``, else
    ``large``.
    hipTensor's planner applies the same rule to its packed operands
    (``rocke_plan.cpp`` gett32Class).

    ``num_cus``: see :func:`resolve_num_cus`.
    """
    kind = "complex" if problem.complex_ else "real"
    if not any(a == arch and k == kind for (a, k, _c) in TUNED_CONFIGS_F32):
        return gett_f64_shape_class(problem, arch, num_cus)
    cus = resolve_num_cus(arch, num_cus)
    p = problem
    if arch == "gfx942":
        if p.complex_:
            t = -(-p.M // 64) * -(-p.N // 64) * p.L
            if _tiles_le(t, cus, 0):
                return "small" if _tiles_le(t, cus, -1) else "compact"
            return "medium" if _tiles_le(t, cus, 6) else "large"
        t = -(-p.M // 128) * -(-p.N // 128) * p.L
        if _tiles_le(t, cus, -2):
            return "tiny" if _tiles_le(t, cus, -3) else "small"
        if _tiles_le(t, cus, 3):
            return "medium" if min(p.M, p.N) <= 4096 else "large"
        return "xlarge" if p.K <= 1024 else "huge"
    if arch == "gfx950":
        if p.complex_:
            t = -(-p.M // 64) * -(-p.N // 64) * p.L
            if _tiles_le(t, cus, 2):
                if p.K <= 256:
                    return "small" if min(p.M, p.N) <= 512 else "medium"
                return "longk"
            return "medium" if _tiles_le(t, cus, 5) else "large"
        t = -(-p.M // 128) * -(-p.N // 128) * p.L
        if _tiles_le(t, cus, 0):
            if _tiles_le(t, cus, -1):
                return "tiny" if _tiles_le(t, cus, -3) else "small"
            return "medium"
        if _tiles_le(t, cus, 2):
            return "large"
        return "xlarge" if _tiles_le(t, cus, 3) else "huge"
    raise ValueError(f"no f32 class rule for {arch}")


#: CTAs per CU :func:`tuned_spec` aims for when it splits K (``split_k``).
SPLIT_CTAS_PER_CU = 2
#: The largest split count :func:`tuned_spec` picks.
SPLIT_K_MAX = 32
#: The fewest K elements one split covers.
SPLIT_MIN_K = 512
#: K per CTA at and above which :func:`tuned_spec` also sums K in blocks
#: (``k_block``) inside a CTA: the rounding chain of an output is longest
#: relative to the work there, and the blocks cost little.
K_BLOCK_MIN_K = 4096
#: K elements per accumulation block :func:`tuned_spec` uses there.
K_BLOCK_ELEMS = 256


def _with_k_block(spec: GettF64Spec, arch: str) -> GettF64Spec:
    kb = max(1, K_BLOCK_ELEMS // spec.tile_k)
    if spec.unroll2 and kb % 2:
        kb += 1
    cand = replace(spec, k_block=kb)
    return cand if is_valid_spec(cand, arch=arch)[0] else spec


def _long_k(
    spec: GettF64Spec, problem: GettF64Problem, arch: str, cus: int, knobs
) -> GettF64Spec:
    """Fewer than :data:`SPLIT_CTAS_PER_CU` CTAs per CU and a long K: split K
    across CTAs until about
    :data:`SPLIT_CTAS_PER_CU` CTAs per CU run (a power of two, at most
    :data:`SPLIT_K_MAX`, every split at least :data:`SPLIT_MIN_K` long); sum
    in blocks inside a CTA whose K range is still :data:`K_BLOCK_MIN_K` or
    longer. ``knobs`` that set ``split_k`` / ``k_block`` win."""
    p = problem
    ctas = -(-p.M // spec.tile_m) * -(-p.N // spec.tile_n) * p.L
    if ctas >= SPLIT_CTAS_PER_CU * cus:
        return spec  # enough CTAs: one per output tile, one chain per output
    if "split_k" not in knobs:
        want = -(-SPLIT_CTAS_PER_CU * cus // max(ctas, 1))
        s = 1
        while s < want and s * 2 <= SPLIT_K_MAX and p.K // (s * 2) >= SPLIT_MIN_K:
            s *= 2
        if s > 1:
            cand = replace(spec, split_k=s, unroll2=False, k_stagger=0)
            if is_valid_spec(cand, arch=arch)[0] and (
                not cand.runtime_sizes or runtime_compatible(cand, p)[0]
            ):
                spec = cand
    if "k_block" not in knobs and p.K // spec.split_k >= K_BLOCK_MIN_K:
        spec = _with_k_block(spec, arch)
    return spec


def tuned_spec(
    problem: GettF64Problem,
    arch: str,
    *,
    bilinear: bool = True,
    num_cus: Optional[int] = None,
    **knobs,
) -> GettF64Spec:
    """The tuned configuration of ``problem``'s shape class on ``arch``
    (:func:`gett_f64_shape_class`; f32 problems :func:`gett_f32_shape_class`
    and :data:`TUNED_CONFIGS_F32`, see :func:`tuned_configs`), overridden by
    ``knobs``.

    Falls back to the other classes of the same arch and dtype, then to
    :func:`arch_spec`, when the class configuration is not valid for the
    problem (an odd tile split, the address budget). Targets without tuned
    configurations get :func:`arch_spec`.

    With ``runtime_sizes=True`` in ``knobs`` the returned spec carries the
    problem's :func:`runtime_template`, so every problem of one layout class
    and shape class gets the same spec (and kernel).

    The chosen configuration then gets ``split_k`` and ``k_block`` by
    :func:`_long_k` (few CTAs per CU and a long K).
    """
    kind = "complex" if problem.complex_ else "real"
    table = tuned_configs(arch, kind, problem.dtype)
    if problem.dtype == "f32":
        cls = gett_f32_shape_class(problem, arch, num_cus)
    else:
        cls = gett_f64_shape_class(problem, arch, num_cus)
    order = [cls] + [c for c in table if c != cls]

    def usable(spec: GettF64Spec) -> bool:
        if not is_valid_spec(spec, arch=arch)[0]:
            return False
        return not spec.runtime_sizes or runtime_compatible(spec, problem)[0]

    def finish(spec: GettF64Spec) -> GettF64Spec:
        if spec.runtime_sizes:
            return replace(spec, problem=runtime_template(problem))
        return spec

    cus = resolve_num_cus(arch, num_cus)
    for c in order:
        cfg = table.get(c)
        if cfg is None:
            continue
        spec = arch_spec(problem, arch, bilinear=bilinear, **{**cfg, **knobs})
        if usable(spec):
            return finish(_long_k(spec, problem, arch, cus, knobs))
    spec = arch_spec(problem, arch, bilinear=bilinear, **knobs)
    return finish(_long_k(spec, problem, arch, cus, knobs))


_PIPE = dict(
    lds_stages=2,
    unroll2=True,
    prefetch_depth=2,
    sched_fence=False,
    kmajor_lds=True,
    acc_agpr=True,
)

#: Pipelined configurations (Python engine only: ``ilv``, ``kmajor_lds`` and
#: ``acc_agpr`` have no C++ builder mirror yet) per (arch, real / complex,
#: pipe class), selected by :func:`pipe_spec` through
#: :func:`gett_f64_pipe_class` for the ``gett_f64_pipe`` dispatch candidate.
#: Found with the search engine (the tuned problems' winners, then
#: cross-evaluated on held-out problems); see dsl_docs/instances/gett_f64.md.
PIPE_CONFIGS = {
    # large grids with long contiguous runs: 256x128 CTA tiles, 4x2 waves
    # (64x64 per wave, 2 waves per SIMD from one CTA), AGPR accumulators
    ("gfx950", "real", "long"): dict(
        tile_m=256,
        tile_n=128,
        tile_k=16,
        warps_m=4,
        warps_n=2,
        group_m=16,
        waves_per_eu=1,
        ilv=8,
        **_SWZ0,
        **_PIPE,
    ),
    # medium grids and short contiguous runs (interleaved contraction
    # layouts): 128x64 tiles, 4x1 waves, three CTAs per CU
    ("gfx950", "real", "short"): dict(
        tile_m=128,
        tile_n=64,
        tile_k=16,
        warps_m=4,
        warps_n=1,
        group_m=2,
        xcds=8,
        waves_per_eu=3,
        ilv=4,
        lds_stages=2,
        unroll2=True,
        prefetch_depth=2,
        sched_fence=True,
        **_SWZ0,
    ),
    # large grids with a short K (< PIPE_LONG_MIN_K): 128x128x16 tiles, 2x4
    # waves, padded LDS rows
    ("gfx950", "real", "wide"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=16,
        warps_m=2,
        warps_n=4,
        group_m=32,
        waves_per_eu=1,
        ilv=16,
        lds_pad=None,
        lds_swizzle=False,
        lds_swizzle_mode=0,
        **_PIPE,
    ),
    # small grids, at most one 64x64 tile per CU: 64x64x16 tiles, 2x2 waves
    ("gfx950", "real", "tiny"): dict(
        tile_m=64,
        tile_n=64,
        tile_k=16,
        warps_m=2,
        warps_n=2,
        group_m=8,
        xcds=8,
        waves_per_eu=2,
        ilv=4,
        lds_stages=2,
        unroll2=True,
        prefetch_depth=2,
        sched_fence=False,
        acc_agpr=True,
        **_SWZ0,
    ),
    # small grids with more 64x64 tiles: 32x32x32 tiles, 2x2 waves, four CTAs
    # per CU
    ("gfx950", "real", "small"): dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warps_m=2,
        warps_n=2,
        group_m=8,
        xcds=8,
        waves_per_eu=4,
        ilv=1,
        lds_stages=2,
        unroll2=True,
        prefetch_depth=2,
        sched_fence=False,
        acc_agpr=True,
        **_SWZ2,
    ),
    # complex: 128x128x8 tiles, 4x2 waves (32x64 per wave, two per SIMD),
    # AGPR accumulators, the fused 4M atoms as ilv slots
    ("gfx950", "complex", "long"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=8,
        warps_m=4,
        warps_n=2,
        group_m=16,
        waves_per_eu=1,
        ilv=8,
        lds_stages=2,
        unroll2=True,
        prefetch_depth=2,
        sched_fence=False,
        acc_agpr=True,
        **_SWZ0,
    ),
    # gfx90a, at least 8 128x128 tiles per CU and K > 512: 256x128x8 tiles,
    # 2x2 waves (128x64 per wave in AGPRs), one wave per SIMD
    ("gfx90a", "real", "long"): dict(
        tile_m=256,
        tile_n=128,
        tile_k=8,
        warps_m=2,
        warps_n=2,
        group_m=4,
        waves_per_eu=1,
        ilv=16,
        **_SWZ0,
        **_PIPE,
    ),
    # gfx90a complex, at least 8 128x128 tiles per CU: 128x128x8 tiles, 4x2
    # waves, AGPR accumulators
    ("gfx90a", "complex", "long"): dict(
        tile_m=128,
        tile_n=128,
        tile_k=8,
        warps_m=4,
        warps_n=2,
        group_m=8,
        waves_per_eu=1,
        ilv=8,
        lds_stages=2,
        unroll2=True,
        prefetch_depth=2,
        sched_fence=False,
        acc_agpr=True,
        **_SWZ0,
    ),
    # gfx90a, fewer tiles: 128x64x16 tiles, 4x2 waves, padded LDS rows
    ("gfx90a", "real", "short"): dict(
        tile_m=128,
        tile_n=64,
        tile_k=16,
        warps_m=4,
        warps_n=2,
        group_m=1,
        waves_per_eu=1,
        ilv=8,
        lds_pad=4,
        lds_swizzle=False,
        lds_swizzle_mode=0,
        lds_stages=2,
        unroll2=True,
        prefetch_depth=2,
        sched_fence=False,
        acc_agpr=True,
    ),
}

#: gfx90a: 128x128 tiles per CU from which the long configuration applies,
#: and the K it needs (shorter K keeps the tuned configurations).
PIPE_GFX90A_LONG_TILES_PER_CU = 8
PIPE_GFX90A_LONG_MIN_K = 768

#: Contiguous run (elements) from which an operand counts as long for
#: :func:`gett_f64_pipe_class`, and the shortest run the short class takes.
PIPE_LONG_RUN = 64
PIPE_MIN_RUN = 16
#: gfx950 long class: the shortest K, and the smallest busy share of the last
#: wave of 256x128 CTAs (one CTA per CU: a ragged last wave idles CUs).
PIPE_LONG_MIN_K = 1024
PIPE_LONG_MIN_WAVE_EFF = 0.95
#: Complex: the pipelined configuration from one 128x128 tile per CU and this K.
PIPE_COMPLEX_MIN_K = 768
#: gfx950 short class from a large problem: fewer 128x128 tiles per CU than
#: this (bigger interleaved grids keep the tuned configuration).
PIPE_SHORT_MAX_TILES_PER_CU = 16


def _run(digits: Digits) -> int:
    e, s = digits[0]
    return e if s == 1 else 0


def gett_f64_pipe_class(
    problem: GettF64Problem, arch: str, num_cus: Optional[int] = None
) -> Optional[str]:
    """The :data:`PIPE_CONFIGS` class of ``problem`` on ``arch`` (None: the
    pipelined candidate does not apply and dispatch keeps the tuned one).

    Complex data: ``long`` from one 128x128 tile per CU (gfx90a:
    :data:`PIPE_GFX90A_LONG_TILES_PER_CU`) and a K of at least
    :data:`PIPE_COMPLEX_MIN_K`. Real data, gfx950 (:func:`gett_f64_shape_class`
    classes): ``wide`` is a ``large`` problem with a K under
    :data:`PIPE_LONG_MIN_K`; ``long`` a ``large`` problem whose last wave of
    256x128 CTAs is at least :data:`PIPE_LONG_MIN_WAVE_EFF` busy, whose D
    contiguous run is at least half of :data:`PIPE_LONG_RUN`, and whose A and
    B contiguous runs (along M / N or K) are at least :data:`PIPE_LONG_RUN`
    elements, or at least :data:`PIPE_MIN_RUN` with at least
    :data:`PIPE_SHORT_MAX_TILES_PER_CU` 128x128 tiles per CU; ``short`` a
    ``medium`` problem, or a ``large`` one under that tile count whose shorter
    A / B run lies in [:data:`PIPE_MIN_RUN`, :data:`PIPE_LONG_RUN`); ``tiny``
    a ``small`` problem with at most one 64x64 tile per CU, ``small`` the
    other ``small`` problems. The other problems keep the tuned
    configurations. gfx90a:
    ``long`` from :data:`PIPE_GFX90A_LONG_TILES_PER_CU` 128x128 tiles per CU
    with a K of at least :data:`PIPE_GFX90A_LONG_MIN_K`, ``short`` below that
    tile count; a short K with many tiles keeps the tuned configurations.
    """
    p = problem
    if p.complex_:
        cus = resolve_num_cus(arch, num_cus)
        tiles = -(-p.M // 128) * -(-p.N // 128) * p.L
        per_cu = PIPE_GFX90A_LONG_TILES_PER_CU if arch == "gfx90a" else 1
        if tiles >= per_cu * cus and p.K >= PIPE_COMPLEX_MIN_K:
            return "long" if (arch, "complex", "long") in PIPE_CONFIGS else None
        return None
    if arch == "gfx90a":
        cus = resolve_num_cus(arch, num_cus)
        tiles = -(-p.M // 128) * -(-p.N // 128) * p.L
        if tiles >= PIPE_GFX90A_LONG_TILES_PER_CU * cus:
            name = "long" if p.K >= PIPE_GFX90A_LONG_MIN_K else None
        else:
            name = "short"
        return name if name and (arch, "real", name) in PIPE_CONFIGS else None
    cls = gett_f64_shape_class(p, arch, num_cus)
    cus = resolve_num_cus(arch, num_cus)
    run_ab = min(max(_run(p.a_m), _run(p.a_k)), max(_run(p.b_n), _run(p.b_k)))
    run_d = max(_run(p.d_m), _run(p.d_n))
    ctas = -(-p.M // 256) * -(-p.N // 128) * p.L
    wave_eff = ctas / (-(-ctas // cus) * cus)
    tiles = -(-p.M // 128) * -(-p.N // 128) * p.L
    if cls == "small":
        tiles64 = -(-p.M // 64) * -(-p.N // 64) * p.L
        name = "tiny" if tiles64 <= cus else "small"
    elif cls == "large" and p.K < PIPE_LONG_MIN_K:
        name = "wide"
    elif cls == "medium":
        name = "short"
    elif (
        cls == "large"
        and 2 * run_d >= PIPE_LONG_RUN
        and p.K >= PIPE_LONG_MIN_K
        and wave_eff >= PIPE_LONG_MIN_WAVE_EFF
        and (
            run_ab >= PIPE_LONG_RUN
            or (run_ab >= PIPE_MIN_RUN and tiles >= PIPE_SHORT_MAX_TILES_PER_CU * cus)
        )
    ):
        name = "long"
    elif (
        cls == "large"
        and PIPE_MIN_RUN <= run_ab < PIPE_LONG_RUN
        and tiles < PIPE_SHORT_MAX_TILES_PER_CU * cus
    ):
        name = "short"
    else:
        return None
    return name if (arch, "real", name) in PIPE_CONFIGS else None


def pipe_spec(
    problem: GettF64Problem,
    arch: str,
    *,
    bilinear: bool = True,
    num_cus: Optional[int] = None,
    **knobs,
) -> Optional[GettF64Spec]:
    """The pipelined configuration of ``problem`` (:data:`PIPE_CONFIGS`,
    :func:`gett_f64_pipe_class`) on ``arch``, overridden by ``knobs``; None
    when no class applies, the configuration is not valid for the problem, or
    the tuned path sums K in blocks or splits a K of :data:`K_BLOCK_MIN_K` or
    more (few CTAs and a long K: the tuned configuration keeps the shorter
    rounding chains)."""
    cls = gett_f64_pipe_class(problem, arch, num_cus)
    if cls is None or knobs.get("runtime_sizes"):
        return None
    base = tuned_spec(problem, arch, bilinear=bilinear, num_cus=num_cus)
    if base.k_block or (base.split_k > 1 and problem.K >= K_BLOCK_MIN_K):
        return None
    cfg = PIPE_CONFIGS[(arch, "complex" if problem.complex_ else "real", cls)]
    spec = arch_spec(problem, arch, bilinear=bilinear, **{**cfg, **knobs})
    return spec if is_valid_spec(spec, arch=arch)[0] else None


def gett_f64_grid(
    spec: GettF64Spec, problem: Optional[GettF64Problem] = None
) -> Tuple[int, int, int]:
    """Launch grid: the output tiles on x, the batch on y.

    A ``runtime_sizes`` kernel needs the actual ``problem`` (its spec only
    carries the layout template).
    """
    if problem is None:
        if spec.runtime_sizes:
            raise ValueError("a runtime_sizes kernel needs the problem for its grid")
        problem = spec.problem
    p = problem
    if p.L > MAX_GRID_Y:
        raise ValueError(f"L = {p.L} exceeds the grid limit {MAX_GRID_Y}")
    return (-(-p.M // spec.tile_m) * -(-p.N // spec.tile_n), p.L, spec.split_k)


def gett_f64_signature(spec: GettF64Spec):
    """Kernel arguments of :func:`build_gett_f64` (``split_k``: ``A, B, W``
    and the size arguments)."""
    w = spec.problem.dtype
    sb = SignatureBuilder().ptr("A", w).ptr("B", w)
    if spec.split_k > 1 and not spec.split_serial:
        sb = sb.ptr("W", w)
        if spec.runtime_sizes:
            for name, dtype in _runtime_arg_names(spec):
                sb = sb.scalar(name, dtype)
        return sb.build()
    if spec.bilinear:
        sb = sb.ptr("C", w)
    sb = sb.ptr("D", w)
    if spec.problem.complex_:
        sb = sb.scalar("alpha_re", w).scalar("alpha_im", w)
        if spec.bilinear:
            sb = sb.scalar("beta_re", w).scalar("beta_im", w)
    else:
        sb = sb.scalar("alpha", w)
        if spec.bilinear:
            sb = sb.scalar("beta", w)
    if spec.split_serial:
        sb = sb.ptr("W", "f64").ptr("Cnt", "i32")
    if spec.runtime_sizes:
        for name, dtype in _runtime_arg_names(spec):
            sb = sb.scalar(name, dtype)
    return sb.build()


__all__ = [
    "F32_GETT_ARCHS",
    "F64_GETT_ARCHS",
    "GETT_DTYPES",
    "REDUCE_THREADS",
    "K_BLOCK_ELEMS",
    "K_BLOCK_MIN_K",
    "SPLIT_CTAS_PER_CU",
    "SPLIT_K_MAX",
    "SPLIT_MIN_K",
    "MAX_GRID_Y",
    "RUNTIME_SIZE_ARGS",
    "PIPE_COMPLEX_MIN_K",
    "PIPE_CONFIGS",
    "PIPE_GFX90A_LONG_MIN_K",
    "PIPE_GFX90A_LONG_TILES_PER_CU",
    "PIPE_LONG_MIN_K",
    "PIPE_LONG_MIN_WAVE_EFF",
    "PIPE_SHORT_MAX_TILES_PER_CU",
    "PIPE_LONG_RUN",
    "PIPE_MIN_RUN",
    "TUNED_CONFIGS",
    "TUNED_CONFIGS_F32",
    "GettF64Problem",
    "GettF64Spec",
    "arch_spec",
    "build_gett_f64",
    "build_gett_f64_reduce",
    "digits_span",
    "f32_atom",
    "f32_gett_archs",
    "f64_atom",
    "f64_gett_archs",
    "gemm_problem",
    "gett_f64_counter_count",
    "gett_f64_grid",
    "gett_f64_reduce_grid",
    "gett_f64_reduce_name",
    "gett_f64_reduce_signature",
    "gett_atom",
    "gett_f64_workspace_bytes",
    "gett_f64_runtime_args",
    "gett_f64_pipe_class",
    "gett_f64_shape_class",
    "gett_f32_shape_class",
    "gett_f64_signature",
    "is_valid_spec",
    "k_split",
    "mixed_offset",
    "pick_tile_k",
    "pipe_spec",
    "resolve_num_cus",
    "runtime_compatible",
    "runtime_template",
    "tuned_configs",
    "tuned_spec",
]
