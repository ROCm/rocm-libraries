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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

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


def f64_gett_archs() -> Tuple[str, ...]:
    """Every catalog target with the f64 atom the kernel is built on."""
    return tuple(a for a in known_arches() if f64_atom(a) is not None)


#: Targets whose MMA catalog has the f64 atom (read from the catalog).
F64_GETT_ARCHS = f64_gett_archs()

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

    def __post_init__(self):
        for name in ("a_m", "a_k", "b_n", "b_k", "d_m", "d_n", "a_l", "b_l", "d_l"):
            object.__setattr__(self, name, _norm(getattr(self, name)))
        for name in ("c_m", "c_n", "c_l"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _norm(getattr(self, name)))

    @property
    def elem_bytes(self) -> int:
        return 16 if self.complex_ else 8

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
) -> GettF64Problem:
    """A (batched) strided GEMM ``D = alpha * A @ B^T + beta * C``.

    ``a_strides = (stride_m, stride_k)`` of ``A[M, K]``; ``b_strides =
    (stride_n, stride_k)`` of ``B[N, K]``; ``d_strides = (stride_m,
    stride_n)``. Any layout (row / column major, transposed) is a choice of
    strides.
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
    )


@dataclass(frozen=True)
class GettF64Spec:
    """One f64 / cf64 GETT kernel.

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
        """LDS row padding in f64 words (complex rows hold 2 words per element)."""
        if self.lds_pad is not None:
            return self.lds_pad
        if self.problem.complex_ or self.uses_swizzle():
            # complex: ds_read_b128, a quarter-wave touches one 256-byte row
            return 0
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
        return kernel_name_join(
            self.name,
            "c128" if p.complex_ else "f64",
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
    two-element vector load needs an even run), complex data and conjugation.
    The template keeps exactly that: extent 3 for an odd run an operand loads
    along, 2 otherwise, stride 1 or 2, ``L = 2`` with stride 2, and C's digits
    spelled out. Problems with the same template share one kernel. None when
    an index space has more than one digit.
    """
    p = problem
    cm, cn, cl = p.c_digits()
    spaces = (p.a_m, p.a_k, p.a_l, p.b_n, p.b_k, p.b_l, p.d_m, p.d_n, p.d_l, cm, cn, cl)
    if any(len(d) != 1 for d in spaces):
        return None

    def unit(d: Digits) -> bool:
        return d[0][1] == 1 and d[0][0] > 1

    def parity(extent: int, *runs: Digits) -> int:
        loaded = any(unit(d) for d in runs)
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
    atom = f64_atom(arch)
    if atom is None:
        return False, f"{arch!r} has no 16x16x4 f64 MMA atom in its catalog"
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
        spec.lds_stages != 2 or spec.prefetch_depth != 1 or not spec.prefetch
    ):
        return False, "unroll2 needs two LDS stages and prefetch depth 1"
    acc_words = (
        (spec.wave_m // atom.m)
        * (spec.wave_n // atom.n)
        * atom.c_frag_len
        * (spec.complex_accs if p.complex_ else 1)
        * (2 if spec.k_block else 1)
    )
    if acc_words * 2 > 384:
        return False, f"accumulator needs {acc_words * 2} registers per lane (max 384)"
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
        * 8
    )
    cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    if lds > cap:
        return False, f"LDS {lds} bytes exceeds {cap} on {arch}"
    return True, "ok"


# ------------------------------------------------------------- the kernel


def _ci(b: IRBuilder, x) -> Value:
    """An i32 operand: a compile-time int becomes a constant, a Value stays."""
    return b.const_i32(x) if isinstance(x, int) else x


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
        b, spec, pl = self.b, self.spec, self.plan
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
        vals = []
        for s_loc, k_loc, base, s_ok in self.coords:
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
            raw = b.buffer_load_vN(rsrc, voff, b.const_i32(0), I32, 2 * words)
            vals.append(b.vec_bitcast(raw, VectorType(F64, words)))
        return vals

    def store_lds(self, vals, stage: Value):
        """Write one K tile (from :meth:`load`) into LDS stage ``stage``."""
        b, spec, pl = self.b, self.spec, self.plan
        row0 = b.mul(stage, b.const_i32(spec.tile_k))
        wpe = 2 if self.cx else 1
        for (s_loc, k_loc, _, _), cols, v in zip(self.coords, self.store_cols, vals):
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

    Kernel arguments: ``A, B, [C,] D`` (``ptr<f64>``; complex data is
    interleaved ``(re, im)``), then ``alpha`` (complex: ``alpha_re,
    alpha_im``) and, when bilinear, ``beta`` (``beta_re, beta_im``), all f64;
    with ``runtime_sizes`` then the sizes and strides of
    :data:`RUNTIME_SIZE_ARGS`. Grid: :func:`gett_f64_grid`; block:
    ``spec.threads``.
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
    atom = f64_atom(arch)
    nt = spec.threads
    am_n, an_n = spec.wave_m // atom.m, spec.wave_n // atom.n
    pad = spec.resolved_pad()

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = nt
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    gptr = PtrType(F64, "global")
    # A and B never overlap D. C may be D itself (in-place update), so neither
    # carries noalias. 8-byte alignment: an f64 subtensor view need not be
    # 16-byte aligned (the buffer loads only need 4).
    split = spec.split_k > 1
    A = b.param("A", gptr, noalias=True, readonly=True, align=8)
    Bp = b.param("B", gptr, noalias=True, readonly=True, align=8)
    C = D = W = None
    alpha, beta = [], []
    if split:
        # Partial sums only: the reduce kernel reads C and applies alpha / beta.
        W = b.param("W", gptr, noalias=True, align=8)
    else:
        C = b.param("C", gptr, readonly=True, align=8) if spec.bilinear else None
        D = b.param("D", gptr, align=8)
        alpha = [b.param("alpha_re" if cx else "alpha", F64)]
        if cx:
            alpha.append(b.param("alpha_im", F64))
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
    if split:
        kz = b.block_id_z()
        rw = _workspace_rsrc(b, spec, geo, W, kz, lb, es)
    else:
        dptr = at_batch(D, p.d_l, "d_sl")
        rd = b.buffer_rsrc(dptr, out_span(geo.d_m, geo.d_n, "D"))
        if C is not None:
            cptr = at_batch(C, cl, "c_sl")
            rc = b.buffer_rsrc(cptr, out_span(geo.c_m, geo.c_n, "C"))

    ns = spec.lds_stages
    a_lds = b.smem_alloc(
        F64, [ns * spec.tile_k, spec.tile_m * wpe + pad], name_hint="a_lds"
    )
    b_lds = b.smem_alloc(
        F64, [ns * spec.tile_k, spec.tile_n * wpe + pad], name_hint="b_lds"
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

    def frag_cols(col0, n_atoms, side, k_row):
        """cols[ks][i]: column of atom i at k-step ks."""
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

    fa = frag_cols(a_col0, am_n, spec.tile_m, a_k)
    fb = frag_cols(b_col0, an_n, spec.tile_n, b_k)

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
    zero = b.vec_pack([b.const_f64(0.0)] * atom.c_frag_len, F64)

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
    if spec.unroll2:
        accs = _k_loop_unroll2(ctx)
    else:
        accs = _k_loop(ctx, nx)
    if split:
        _epilogue_partial(b, spec, atom, accs, lane, wm, wn, m0, n0, rw, geo)
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

    def mfma_block(self, accs, row0):
        """The MFMAs of one K tile read from the LDS rows at ``row0``."""
        b, spec, atom = self.b, self.spec, self.atom
        accs = list(accs)
        for ks in range(spec.tile_k // atom.k):
            ka = self.fa.k_row
            row_a = b.add(row0, b.add(ka, b.const_i32(ks * atom.k)) if ks else ka)
            fa = [_frag(b, self.a_lds, row_a, col, self.cx) for col in self.fa.cols[ks]]
            kb = self.fb.k_row
            row_b = b.add(row0, b.add(kb, b.const_i32(ks * atom.k)) if ks else kb)
            fb = [_frag(b, self.b_lds, row_b, col, self.cx) for col in self.fb.cols[ks]]
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


def _frag(b, lds, row, col, cx):
    """One operand fragment: an f64, or the (re, im) pair of a complex element."""
    if cx:
        v = b.smem_load_vN(lds, row, b.mul(col, b.const_i32(2)), dtype=F64, n=2)
        return (b.vec_extract(v, 0), b.vec_extract(v, 1))
    return b.vec_extract(b.smem_load_vN(lds, row, col, dtype=F64, n=1), 0)


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

    def __init__(self, b, alpha, beta, cx):
        self.alpha, self.beta, self.cx = alpha, beta, cx
        self.zero = b.const_f64(0.0)
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
        use_c = b.lnot(b.fcmp("oeq", beta[0], b.const_f64(0.0)))
        if cx:
            use_c = b.lor(use_c, b.lnot(b.fcmp("oeq", beta[1], b.const_f64(0.0))))
    sc = _Scalars(b, alpha, beta, cx)
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
                    raw = b.buffer_load_vN(rc, coff, b.const_i32(0), I32, 2 * words)
                    c_val = b.vec_bitcast(raw, VectorType(F64, words))
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
            vec = b.vec_bitcast(b.vec_pack(list(out), F64), VectorType(F32, 2 * words))
            b.buffer_store_vN_f32(rd, doff, b.const_i32(0), vec, 2 * words)


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
    return spec.split_k * p.L * p.M * p.N * p.elem_bytes


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
    (:func:`gett_f64_shape_class`), overridden by ``knobs``.

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
    cls = gett_f64_shape_class(problem, arch, num_cus)
    order = [cls] + [
        c for (a, k, c) in TUNED_CONFIGS if a == arch and k == kind and c != cls
    ]

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
        cfg = TUNED_CONFIGS.get((arch, kind, c))
        if cfg is None:
            continue
        spec = arch_spec(problem, arch, bilinear=bilinear, **{**cfg, **knobs})
        if usable(spec):
            return finish(_long_k(spec, problem, arch, cus, knobs))
    spec = arch_spec(problem, arch, bilinear=bilinear, **knobs)
    return finish(_long_k(spec, problem, arch, cus, knobs))


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
    sb = SignatureBuilder().ptr("A", "f64").ptr("B", "f64")
    if spec.split_k > 1:
        sb = sb.ptr("W", "f64")
        if spec.runtime_sizes:
            for name, dtype in _runtime_arg_names(spec):
                sb = sb.scalar(name, dtype)
        return sb.build()
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


__all__ = [
    "F64_GETT_ARCHS",
    "REDUCE_THREADS",
    "K_BLOCK_ELEMS",
    "K_BLOCK_MIN_K",
    "SPLIT_CTAS_PER_CU",
    "SPLIT_K_MAX",
    "SPLIT_MIN_K",
    "MAX_GRID_Y",
    "RUNTIME_SIZE_ARGS",
    "TUNED_CONFIGS",
    "GettF64Problem",
    "GettF64Spec",
    "arch_spec",
    "build_gett_f64",
    "build_gett_f64_reduce",
    "digits_span",
    "f64_atom",
    "f64_gett_archs",
    "gemm_problem",
    "gett_f64_grid",
    "gett_f64_reduce_grid",
    "gett_f64_reduce_name",
    "gett_f64_reduce_signature",
    "gett_f64_workspace_bytes",
    "gett_f64_runtime_args",
    "gett_f64_shape_class",
    "gett_f64_signature",
    "is_valid_spec",
    "k_split",
    "mixed_offset",
    "pick_tile_k",
    "resolve_num_cus",
    "runtime_compatible",
    "runtime_template",
    "tuned_spec",
]
