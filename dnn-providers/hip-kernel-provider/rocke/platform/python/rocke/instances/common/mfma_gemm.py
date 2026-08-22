# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA-tiled GEMM kernel (production CK Tile pattern).

The **first MFMA-based instance** in the kernel set: a real f16 GEMM
that consumes the ``mfma_f32_16x16x*_f16`` (or 32x32x*) atom directly
rather than emitting scalar FMUL / FADD per output cell.

Why it matters
==============

* CDNA3 (MI300X / gfx950) ``mfma_f32_16x16x32_f16`` emits **one MFMA
  per 4 cycles** for a 16x16x32 matmul; that is 512 MAC ops per atom
  or **128 FLOPS/cycle/lane**. The legacy ``mfma_f32_16x16x16_f16``
  atom yields 256 MACs / 64 FLOPS/cycle/lane; the scalar-inner v1 in
  :mod:`streamk_gemm` emits one FMUL + one FADD per K-element which is
  2 FLOPS/cycle/lane. The K-packed MFMA path is **~64x denser** in
  FLOPS than scalar inner for the same K-loop trip count and **2x
  denser than the legacy 16x16x16 atom for the same lane layout**.
* CK Tile's GEMM hero kernels (``03_gemm`` Preshuffle config,
  ``18_flatmm`` ``FlatmmConfig32``) all bottom out on the K-packed
  16x16x32 / 32x32x16 atoms when ``DataType`` is f16 / bf16;
  Warp-specialized register-tile MMA wrappers (``mfma161632`` /
  ``mfma323216``) wrap the same intrinsics. Matching them is the "beat CK Tile" baseline.

What the kernel does per CTA
============================

1. Grid = ``(N // atom.n, M // atom.m, 1)`` -- one CTA per
   ``atom.m x atom.n`` output tile.
2. Block_size = 64 (one wave64 warp). The MFMA atom is per-wave.
3. Each lane owns:

   * **A**: a ``<a_per_lane x f16>`` per K-iter
     (4 K-elements at lane row on the legacy 16x16x16 atom,
     8 on the K-packed 16x16x32 / 32x32x16 atoms).
   * **B**: a ``<b_per_lane x f16>`` per K-iter.
   * **C**: a ``<c_per_lane x f32>`` accumulator across the whole
     K-loop (4 floats on 16x16; 16 floats on 32x32).
4. K-loop: ``scf.for k_blk in [0, K // atom.k)``: each iter loads the
   ``atom.m x atom.n x atom.k`` A/B slab into per-lane vector regs,
   fires one MFMA, accumulates into C. The K-packed atom halves
   ``K // atom.k`` for the same K, doubling K-loop density.
5. Epilogue: each lane writes ``c_per_lane`` output cells of
   ``C[m_tile*atom.m : m_tile*atom.m + atom.m,
   n_tile*atom.n : n_tile*atom.n + atom.n]`` using the atom's
   ``lane_to_output`` map (see :mod:`rocke.helpers.atoms`).

Levers exposed on the spec
==========================

* ``tile_m`` / ``tile_n`` in ``{16, 32}`` -- pick the 16x16 hero atom
  (smaller per-CTA output, fewer VGPRs per lane) or the canonical
  32x32 hero atom (4x output per atom; ``mfmas_per_warp = 1``).
* ``kpack`` (default ``True``) -- choose the K-packed atom on gfx950+
  (16x16x32 / 32x32x16). Halves the K-loop trip count vs. the legacy
  atom at the cost of a 2x wider per-lane A/B fragment (same
  accumulator footprint). Per the K-packed lane-layout note in
  :doc:`reference/mfma_atom_catalog`, the output lane layout is
  identical to the legacy atoms; no epilogue change required.
* ``dtype`` -- ``"f16"`` is the only shipped option (bf16 / fp8 / bf8
  atoms exist in :class:`rocke.helpers.atoms.MfmaAtom` and the
  dispatch lives in :func:`mfma_atom_for_dtype`; a one-line spec
  extension wires them up).

Limitations of v1
=================

* **One atom per CTA**; larger M/N tiles need warp-level tiling +
  cshuffle epilogue (v2). The 32x32 atom partially addresses this by
  raising the per-CTA output footprint to 1024 cells.
* **No LDS staging**; all A/B loads come from global memory. The
  L2-cached path is enough for parity vs the scalar inner; LDS
  staging + async DMA (``AsyncTileLoader`` from
  :mod:`rocke.helpers.loads`) is the v2 perf hoist.
* **Persistent + StreamK split-K** lives in :mod:`streamk_gemm`; this
  kernel is the dense one-CTA-per-tile baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import BF16, F16, I32, IRBuilder, KernelDef, PtrType
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_atom_for_dtype,
    mfma_k_loop,
    store_acc_to_global,
)
from ...helpers.spec import SignatureBuilder, kernel_name_join


# P53 + P88: bf16 lands via the same helper-side atom dispatch
# (``mfma_atom_for_dtype``) plus a typed-pointer flip in the kernel
# signature. fp8 / bf8 paths require more plumbing (lane-decode +
# byte-wise A / B loads) and are tracked separately.
DType = Literal["f16", "bf16"]


# The two atom shapes shipped by v1. 16x16 keeps the small-CTA layout
# the v1 parity tests were written against; 32x32 is the canonical
# hero shape and matches CK Tile's ``FlatmmConfig32`` K-packed
# atom reference. Both shapes K-pack on gfx950+.
_SUPPORTED_ATOM_MN: Tuple[Tuple[int, int], ...] = ((16, 16), (32, 32))


@dataclass(frozen=True)
class MfmaGemmSpec:
    """One concrete MFMA GEMM kernel configuration.

    ``M`` / ``N`` / ``K`` are compile-time so the partitioner can
    statically derive the grid + K-loop trip count.

    Atom selection
    --------------

    ``tile_m`` / ``tile_n`` pick the MFMA hero shape (one MFMA atom
    per CTA). ``(16, 16)`` is the small-tile baseline; ``(32, 32)`` is
    the canonical hero (4x output per atom, 16-float accumulator per
    lane). Both shapes flip between the legacy and the K-packed
    variant via ``kpack``.

    ``kpack=True`` (default) selects ``f16_16x16x32`` / ``f16_32x32x16``
    on gfx950+: each MFMA processes 2x the K of the legacy atom for
    the same lane / output layout. The K-loop trip count is
    ``K // atom.k`` so this halves it. Set ``kpack=False`` to fall back
    to the legacy atoms (``f16_16x16x16`` / ``f16_32x32x8``) on pre-
    CDNA3 cards.
    """

    M: int
    N: int
    K: int
    dtype: DType = "f16"
    tile_m: int = 16
    tile_n: int = 16
    kpack: bool = True
    name: str = "rocke_mfma_gemm"

    @property
    def atom(self) -> MfmaAtom:
        # Centralised dispatch lives in helpers.mfma_gemm_inner so the
        # same (dtype, m, n, kpack) -> MfmaAtom mapping powers the
        # other GEMM-family instances too.
        return mfma_atom_for_dtype(
            self.dtype,
            self.tile_m,
            self.tile_n,
            prefer_packed_k=self.kpack,
        )

    @property
    def tile_k(self) -> int:
        return self.atom.k

    @property
    def block_size(self) -> int:
        # One wave64 warp per CTA -- the MFMA atom is per-wave.
        return 64

    def kernel_name(self) -> str:
        atom = self.atom
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.dtype,
            f"atom{atom.m}x{atom.n}x{atom.k}",
            flags={"kpack": self.kpack},
        )


_SUPPORTED_DTYPES: Tuple[str, ...] = ("f16", "bf16")


# Map the kernel's dtype string to the MFMA-catalog dtype name used by
# :class:`rocke.core.arch.ArchTarget`. The mfma_gemm spec exposes
# ``"f16"`` / ``"bf16"``; the catalog keys on ``"fp16"`` / ``"bf16"``.
_CATALOG_DTYPE = {"f16": "fp16", "fp16": "fp16", "bf16": "bf16"}


def is_valid_spec(spec: MfmaGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    # P88 (partial) + P53: gate flip — accept bf16 in addition to f16.
    # The atom dispatch in ``mfma_atom_for_dtype`` handles bf16 via
    # the bf16 atom factory landed by P53. fp8 / bf8 require typed
    # pointer plumbing + lane-decode rewiring; tracked separately.
    if spec.dtype not in _SUPPORTED_DTYPES:
        return False, (
            f"mfma_gemm dtype must be one of {_SUPPORTED_DTYPES}, got {spec.dtype!r}"
        )
    if (spec.tile_m, spec.tile_n) not in _SUPPORTED_ATOM_MN:
        return False, (
            f"tile_m x tile_n must be one of {_SUPPORTED_ATOM_MN}; "
            f"got {(spec.tile_m, spec.tile_n)!r}"
        )
    atom = spec.atom
    if spec.M % atom.m or spec.N % atom.n or spec.K % atom.k:
        return False, (
            f"M / N / K must be divisible by the {atom.m}x{atom.n}x{atom.k} "
            f"atom shape; got M={spec.M}, N={spec.N}, K={spec.K}"
        )
    # Arch gating: the K-packed atoms (``kpack=True`` -> 16x16x32 /
    # 32x32x16 on f16/bf16) exist only on gfx950 (CDNA4). Requesting a
    # K-packed atom on gfx942 would emit an MFMA intrinsic the gfx942
    # backend cannot select, hard-crashing comgr ("LLVM ERROR: Cannot
    # select intrinsic"). Source the legal atom set from the target's
    # MFMA catalog and reject the absent atom with a structured reason.
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    cat_dtype = _CATALOG_DTYPE.get(spec.dtype)
    if cat_dtype is None:
        return False, f"no MFMA-catalog dtype mapping for {spec.dtype!r}"
    if not target.mma.has_shape(
        a_dtype=cat_dtype,
        b_dtype=cat_dtype,
        c_dtype="fp32",
        m=atom.m,
        n=atom.n,
        k=atom.k,
    ):
        return False, (
            f"{spec.dtype} MFMA atom {atom.m}x{atom.n}x{atom.k} "
            f"(kpack={spec.kpack}) not available on {arch}; "
            f"set kpack=False for the legacy atom on pre-CDNA4 targets"
        )
    return True, "ok"


def build_mfma_gemm(spec: MfmaGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build a one-atom-per-CTA MFMA GEMM kernel.

    Kernel signature: ``(A: ptr<dtype>, B: ptr<dtype>, C: ptr<dtype>,
    M: i32, N: i32, K: i32)`` where ``dtype`` is ``f16`` or ``bf16``.

    Grid: ``(N // atom.n, M // atom.m, 1)``. Block: 64 threads
    (one wave64 warp).

    ``arch`` selects the target GPU. With ``kpack=True`` (default) the
    K-packed atoms ``16x16x32`` / ``32x32x16`` are selected, which exist
    only on gfx950 (CDNA4); requesting ``arch="gfx942"`` for a K-packed
    spec is rejected with a structured Python error before comgr. Set
    ``kpack=False`` for the legacy gfx942-capable atoms.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid mfma_gemm spec for {arch}: {why}")

    atom = spec.atom
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    elem_ir = BF16 if spec.dtype == "bf16" else F16
    A = b.param("A", PtrType(elem_ir, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(elem_ir, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(elem_ir, "global"), noalias=True, writeonly=True, align=16)
    _M = b.param("M", I32)  # noqa: F841 - ABI
    _N = b.param("N", I32)  # noqa: F841 - ABI
    _K = b.param("K", I32)  # noqa: F841 - ABI

    lane = b.thread_id_x()
    bid_n = b.block_id_x()
    bid_m = b.block_id_y()
    m_tile_base = b.mul(bid_m, b.const_i32(atom.m))
    n_tile_base = b.mul(bid_n, b.const_i32(atom.n))

    # Lane decode (lane -> (m_in_atom, n_in_atom, k_blk)) is shared
    # across the load helpers and the epilogue; building it once keeps
    # the lowered IR's scalar arithmetic on the SGPR path.
    lane_decode = decode_mfma_lanes(b, atom, lane)
    c_atom_k = b.const_i32(atom.k)

    def _load_a(b, kt):
        return load_a_row_major_contiguous(
            b,
            A=A,
            atom=atom,
            lane_decode=lane_decode,
            m_tile_base=m_tile_base,
            k_tile_base=b.mul(kt, c_atom_k),
            K=spec.K,
        )

    def _load_b(b, kt):
        return load_b_col_strided_scalars(
            b,
            B=Bp,
            atom=atom,
            lane_decode=lane_decode,
            n_tile_base=n_tile_base,
            k_tile_base=b.mul(kt, c_atom_k),
            N=spec.N,
        )

    acc_final = mfma_k_loop(
        b,
        K=spec.K,
        atom=atom,
        load_a=_load_a,
        load_b=_load_b,
    )

    store_acc_to_global(
        b,
        C=C,
        atom=atom,
        lane_decode=lane_decode,
        m_tile_base=m_tile_base,
        n_tile_base=n_tile_base,
        acc=acc_final,
        N=spec.N,
        out_dtype=spec.dtype,
    )

    b.ret()
    return b.kernel


def mfma_gemm_grid(spec: MfmaGemmSpec) -> Tuple[int, int, int]:
    atom = spec.atom
    n_tiles = spec.N // atom.n
    m_tiles = spec.M // atom.m
    return (n_tiles, m_tiles, 1)


def mfma_gemm_signature(spec: MfmaGemmSpec):
    return (
        SignatureBuilder()
        .ptr("A", spec.dtype)
        .ptr("B", spec.dtype)
        .ptr("C", spec.dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .build()
    )


__all__ = [
    "MfmaGemmSpec",
    "build_mfma_gemm",
    "is_valid_spec",
    "mfma_gemm_grid",
    "mfma_gemm_signature",
]
