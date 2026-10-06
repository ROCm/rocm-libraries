# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct non-grouped (``groups == 1``) convolution, NHWC layout.

Why a second direct-conv family
-------------------------------
:mod:`kernels.common.conv_direct_grouped` draws all of its parallelism from the
``groups`` axis: one wave owns one convolution group, and the accumulator tile is
``ceil(kpg/16)`` M-tiles deep.

This module is the ``groups == 1`` counterpart.  It keeps the property that
makes a direct conv beat implicit GEMM on 3x3 stride<=2 shapes: the input tile
is staged in LDS **once per channel chunk, with its halo**, and all ``KH*KW``
filter taps read shifted sub-tiles out of that one staged copy.  An
implicit-GEMM formulation re-reads the activations once per tap, so its A-side
traffic is ~``KH*KW`` times larger, and the L2 has to absorb that redundancy.

Shape of the computation
------------------------
Per workgroup, with M = output channels and N = output pixels::

    tile_k          output channels   (M, multiple of 32)
    tile_h x tile_w output pixels     (N, tile_w a multiple of 32)
    ck              input channels reduced per LDS stage

    for c0 in range(0, C, ck):          # runtime scf.for, accumulators carried
        stage X[(tile_h-1)*s+KH, (tile_w-1)*s+KW, ck] -> LDS   (halo included)
        stage W[tile_k, KH, KW, ck]                   -> LDS   (fragment order)
        for (r, s) in taps:             # Python-unrolled
            for m_tile, n_tile, k_atom:
                acc = mfma_f32_32x32xK(W_lds, X_lds, acc)

LDS layouts
-----------
``X`` is stored ``[pos][c]`` with ``pos = ih_lds * LDS_IN_W + iw_lds`` and a
channel stride of ``ck + lds_pad``.  The pad is what keeps the 32 lanes of a
``ds_read_b128`` on distinct bank quads: with ``ck=16, pad=8`` the stride is
24 halves = 12 dwords, so lanes 0..7 land on bank quads 0,12,24,4,16,28,8,20 --
eight distinct quads covering the full 128-byte LDS service width.

``W`` is stored **pre-swizzled into MFMA fragment order**: fragment slot
``((tap * M_TILES + m_tile) * KATOMS + k_atom) * 64 + lane`` holds exactly the
``FRAG`` halves that ``lane`` feeds to the MFMA.  Both the staging store and the
consuming read are then linear in ``lane``, so no padding is needed and neither
side can bank-conflict.  The staging thread's global read is ``FRAG`` contiguous
channels of one ``k_out`` row, which is what the KRSC layout makes contiguous.

AOT kernel arguments
--------------------
The kernel takes the direct-conv kernarg block of
:func:`kernels.common.conv_abi.conv_direct_arg_names` (``direction="fwd"``):
batch, input and output extents, ``C`` (``p_total_c``), ``K`` (``p_total_k``)
and the NHWC / NHWK strides are runtime values, so one binary serves any image
size and any channel counts its tile divides. The filter (``KH``/``KW``),
``stride`` and ``PAD`` shape the staged halo and the unrolled tap loop, so they
stay build-time capabilities, like the tile geometry. Unlike the grouped
kernels this family does not bake ``C``/``K`` either: they only set the channel
loop's trip count and the store / staging masks.

Software pipeline
-----------------
The global loads for chunk ``i+1`` are issued immediately after the barrier that
publishes chunk ``i``, and travel to the next iteration as loop-carried registers.
The MFMAs of chunk ``i`` therefore cover the DRAM latency of chunk ``i+1``.  The
tail iteration loads one chunk past ``C``; the result is discarded, and buffer
loads clamp rather than fault, so no masking is needed on that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

from rocke.core.ir import (
    F32,
    IRBuilder,
    KernelDef,
    Value,
)

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _buf_load_vN,
    _buf_store_vN,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
)

# Global load width for the activation staging path, in halves (dwordx4).
_X_LOAD_VEC = 8

# Byte offset every masked-out buffer access is redirected to (see
# build_direct_conv_nongrouped): the staging loads that fall outside the image
# or past K, and the epilogue stores that fall outside the output. A buffer
# access whose offset is at or past num_records is dropped (a load returns
# zero), so every operand tensor must end at or below this offset, and
# ``_OOB_BASE + chunk offset`` must still fit in i32 --
# DirectNongroupedConvSpec.validate() enforces both. The tensor sizes are
# runtime values in the AOT kernel, so this is a launch-time contract: the
# spec has to be validated against the problem it is launched on.
_OOB_BASE = 0x7F000000
_I32_MAX = (1 << 31) - 1

# Atom name -> (tile, K, frag). All supported atoms are square (M == N == tile),
# which is what lets one code path serve them: a lane's operand row/column is
# ``lane % tile`` and its K slice is ``(lane // tile) * frag``. ``frag`` is the
# per-lane operand width in halves, and ``tile * tile / 64`` accumulator floats
# per lane follow from the shape.
#
# The 16-wide atoms exist for output widths that are not a multiple of 32: with
# a 32-wide pixel tile the last tile in each row computes columns past ``Wo``,
# and that wasted MFMA work is spread over every block of the row.
_ATOMS = {
    "32x32x16": (32, 16, 8),
    "32x32x8": (32, 8, 4),
    "16x16x32": (16, 32, 8),
    "16x16x16": (16, 16, 4),
}


@dataclass(frozen=True)
class DirectNongroupedConvSpec:
    """One concrete non-grouped direct-conv kernel configuration.

    ``problem.groups`` must be 1; the grouped families in
    :mod:`kernels.common.conv_direct_grouped` cover ``groups > 1``.
    """

    problem: DirectConvProblem
    name: str = "direct_conv_nongrouped"

    tile_h: int = 16  # output rows per workgroup
    tile_w: int = 32  # output cols per workgroup (multiple of 32)
    tile_k: int = 128  # output channels per workgroup (multiple of 32)
    ck: int = 16  # input channels staged per LDS chunk

    waves_m: int = 2  # waves splitting tile_k
    waves_n: int = 4  # waves splitting tile_h (each keeps all tile_w columns)

    atom: str = "32x32x16"
    wave_size: int = 64
    lds_pad: int = 8  # extra halves on the LDS activation channel stride

    # Grid swizzle. The launch grid is flattened to 1-D and remapped so that the
    # workgroups a single XCD receives share operands. ``swizzle_wgm`` is the
    # number of *channel* tiles grouped per spatial cell: wgm = n_k_tiles walks
    # every output-channel tile of one spatial cell back to back (activation tile
    # loaded into that XCD's L2 once), wgm = 1 walks every spatial cell of one
    # channel tile (weight slab loaded once). Intermediate values trade between.
    chiplet_swizzle: bool = True
    swizzle_wgm: int = 8
    chiplet_chunk: int = 64
    num_xcds: int = 8

    # Ping-pong the staged tiles across two LDS buffers. Single-buffered, the
    # channel loop needs two barriers per chunk with nothing but ds_writes
    # between them; ping-ponged, chunk i+1 is written while chunk i is still
    # feeding MFMAs and one barrier per chunk suffices. Costs 2x LDS.
    double_buffer: bool = False
    # ``iglp_opt`` level for the channel loop, or None to leave the backend
    # scheduler alone. 0 is the canned GEMM MFMA/memory interleave.
    iglp: "int | None" = None
    # ``amdgpu-waves-per-eu`` occupancy hint. The prefetch registers carried
    # across the channel loop can push register use high enough to leave one
    # wave per SIMD, which leaves nothing to cover the barriers per chunk.
    waves_per_eu: "int | None" = None

    # ---- derived geometry -------------------------------------------------

    @property
    def atom_tile(self) -> int:
        """MFMA M == N (channels per M-tile, pixels per N-tile)."""
        return _ATOMS[self.atom][0]

    @property
    def atom_k(self) -> int:
        return _ATOMS[self.atom][1]

    @property
    def frag(self) -> int:
        """Halves per lane in an A/B operand fragment."""
        return _ATOMS[self.atom][2]

    @property
    def acc_per_lane(self) -> int:
        t = self.atom_tile
        return t * t // self.wave_size

    @property
    def n_col_blocks(self) -> int:
        return self.tile_w // self.atom_tile

    @property
    def m_tiles_total(self) -> int:
        return self.tile_k // self.atom_tile

    @property
    def k_atoms(self) -> int:
        return self.ck // self.atom_k

    @property
    def rows_per_wave(self) -> int:
        return self.tile_h // self.waves_n

    @property
    def m_tiles_per_wave(self) -> int:
        return self.m_tiles_total // self.waves_m

    @property
    def n_tiles_per_wave(self) -> int:
        return self.rows_per_wave * self.n_col_blocks

    @property
    def threads_per_block(self) -> int:
        return self.waves_m * self.waves_n * self.wave_size

    @property
    def lds_in_h(self) -> int:
        p = self.problem
        return (self.tile_h - 1) * p.stride + p.KH

    @property
    def lds_in_w(self) -> int:
        p = self.problem
        return (self.tile_w - 1) * p.stride + p.KW

    @property
    def c_stride(self) -> int:
        """LDS channel stride of the staged activation tile, in halves."""
        return self.ck + self.lds_pad

    @property
    def lds_x_halves(self) -> int:
        return self.lds_in_h * self.lds_in_w * self.c_stride

    @property
    def lds_w_halves(self) -> int:
        p = self.problem
        return self.tile_k * p.KH * p.KW * self.ck

    @property
    def lds_bytes(self) -> int:
        # + one scratch fragment per array (see build_direct_conv_nongrouped).
        one = self.lds_x_halves + _X_LOAD_VEC + self.lds_w_halves + self.frag
        return 2 * one * (2 if self.double_buffer else 1)

    @property
    def acc_vgprs(self) -> int:
        """f32 accumulator registers per lane."""
        return self.m_tiles_per_wave * self.n_tiles_per_wave * self.acc_per_lane

    @property
    def tile_counts(self) -> Tuple[int, int, int]:
        """``(n_w_tiles, n_h_tiles, n_k_tiles)``."""
        p = self.problem
        return (
            (p.Wo + self.tile_w - 1) // self.tile_w,
            (p.Ho + self.tile_h - 1) // self.tile_h,
            (p.kpg + self.tile_k - 1) // self.tile_k,
        )

    def grid(self) -> Tuple[int, int, int]:
        n_wt, n_ht, n_kt = self.tile_counts
        return (n_wt * n_ht * self.problem.N * n_kt, 1, 1)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_h}x{self.tile_w}x{self.tile_k}",
            f"ck{self.ck}",
            f"w{self.waves_m}x{self.waves_n}",
            f"a{self.atom}",
            f"g{self.swizzle_wgm}" if self.chiplet_swizzle else "gnone",
            "db" if self.double_buffer else "",
            f"iglp{self.iglp}" if self.iglp is not None else "",
            f"we{self.waves_per_eu}" if self.waves_per_eu is not None else "",
            "bf16" if p.dtype == "bf16" else "",
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(f"DirectNongroupedConvSpec: unsupported dtype {p.dtype!r}")
        if p.groups != 1:
            raise ValueError(
                f"DirectNongroupedConvSpec is the groups==1 family (got groups={p.groups}); "
                f"use conv_direct_grouped for grouped shapes"
            )
        if self.atom not in _ATOMS:
            raise ValueError(
                f"unknown atom {self.atom!r}; expected one of {list(_ATOMS)}"
            )
        if (
            self.tile_h <= 0
            or self.tile_w <= 0
            or self.tile_k <= 0
            or self.ck <= 0
            or self.waves_m <= 0
            or self.waves_n <= 0
            or self.wave_size <= 0
            or p.stride <= 0
            or p.cpg <= 0
            or p.kpg <= 0
        ):
            # Checked before any derived geometry: those divide by these. A
            # positive C is also what lets the kernel assume its channel loop
            # runs at least once.
            raise ValueError(
                "DirectNongroupedConvSpec: tile, wave, stride and channel "
                "parameters must be positive"
            )
        t = self.atom_tile
        if self.tile_w % t != 0:
            raise ValueError(f"tile_w must be a multiple of {t} (got {self.tile_w})")
        if self.tile_k % t != 0:
            raise ValueError(f"tile_k must be a multiple of {t} (got {self.tile_k})")
        if self.tile_h % self.waves_n != 0:
            raise ValueError(
                f"tile_h {self.tile_h} not divisible by waves_n {self.waves_n}"
            )
        if self.m_tiles_total % self.waves_m != 0:
            raise ValueError(
                f"tile_k/{self.atom_tile} = {self.m_tiles_total} not divisible by "
                f"waves_m {self.waves_m}"
            )
        if self.ck % self.atom_k != 0:
            raise ValueError(f"ck {self.ck} not divisible by atom K {self.atom_k}")
        if self.ck % _X_LOAD_VEC != 0:
            raise ValueError(f"ck {self.ck} not divisible by {_X_LOAD_VEC}")
        if p.cpg % self.ck != 0:
            raise ValueError(f"C {p.cpg} not divisible by ck {self.ck}")
        if p.kpg % self.atom_tile != 0:
            raise ValueError(
                f"K {p.kpg} must be a multiple of {self.atom_tile} (got {p.kpg})"
            )
        if p.cpg % _X_LOAD_VEC != 0:
            raise ValueError(f"C {p.cpg} must be a multiple of {_X_LOAD_VEC}")
        if self.threads_per_block > 1024:
            raise ValueError(f"threads_per_block {self.threads_per_block} > 1024")
        if self.lds_pad < 0 or self.lds_pad % 8 != 0:
            # A negative pad shrinks the pixel stride below the ck halves each
            # pixel stages, so neighbouring pixels overlap in LDS. The vec8 LDS
            # store/load is emitted with align 16, so the pixel stride
            # (ck + lds_pad halves) must be a multiple of 8 halves.
            raise ValueError(
                f"lds_pad must be a non-negative multiple of 8 to keep "
                f"ds_read_b128 aligned (got {self.lds_pad})"
            )
        if self.swizzle_wgm < 1:
            raise ValueError(f"swizzle_wgm must be >= 1 (got {self.swizzle_wgm})")
        if self.iglp is not None and self.iglp < 0:
            raise ValueError(f"iglp must be None or >= 0 (got {self.iglp})")
        if self.waves_per_eu is not None and self.waves_per_eu < 1:
            raise ValueError(
                f"waves_per_eu must be None or >= 1 (got {self.waves_per_eu})"
            )
        if self.grid()[0] > _I32_MAX:
            # The kernel decodes its tile from a flat 1-D block id and computes
            # the workgroup count in i32; past gridDim.x's 2**31 - 1 limit both
            # the launch and that product would overflow. (The D-size bound
            # below implies this one; it is checked on its own so the reason
            # names the real limit.)
            raise ValueError(
                f"flattened grid of {self.grid()[0]} workgroups exceeds "
                f"{_I32_MAX}: grow the tile or split the batch"
            )
        a_bytes = 2 * p.N * p.H * p.W * p.cpg
        b_bytes = 2 * p.kpg * p.KH * p.KW * p.cpg
        d_bytes = 2 * p.N * p.Ho * p.Wo * p.kpg
        if max(a_bytes, b_bytes, d_bytes) > _OOB_BASE:
            # Masked staging loads and epilogue stores go to _OOB_BASE and rely
            # on it being past num_records; a larger tensor would hand the
            # loads real data and take the stores.
            raise ValueError(
                f"tensors must fit below the {_OOB_BASE:#x}-byte masked-access "
                f"offset (A {a_bytes} B, B {b_bytes} B, D {d_bytes} B)"
            )
        if _OOB_BASE + 2 * (p.cpg + self.ck) > _I32_MAX:
            # The last prefetch runs one chunk (two when double-buffered) past C.
            raise ValueError(
                f"C {p.cpg} too large: the masked-load offset plus the channel "
                f"offset would overflow i32"
            )
        if self.acc_vgprs > 256:
            # A lane cannot hold more than the 256-entry accumulator file; past
            # that the config is not merely slow, it makes the backend
            # scheduler blow up (a 2048-accumulator tile hung the compiler).
            raise ValueError(
                f"accumulator tile needs {self.acc_vgprs} registers per lane "
                f"(max 256): shrink tile_k/tile_h/tile_w or add waves"
            )


def is_valid_nongrouped_spec(
    spec: DirectNongroupedConvSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch`` without raising."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:  # pragma: no cover - arch table lookup
        return False, str(e)

    # Everything past the arch lookup stays inside the try: the arch-table
    # queries are part of the no-raise contract too, not just validate().
    try:
        spec.validate()
        p = spec.problem
        if p.stride not in (1, 2):
            return False, f"stride {p.stride} is not supported (expected 1 or 2)"
        ab = "bf16" if p.dtype == "bf16" else "f16"
        t, k = spec.atom_tile, spec.atom_k
        if not target.mma.has_shape(
            a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=t, n=t, k=k
        ):
            return False, f"missing mfma_f32_{spec.atom}_{ab} on {arch}"
        if spec.wave_size != target.wave_size:
            return (
                False,
                f"wave_size {spec.wave_size} != {arch} wave {target.wave_size}",
            )
        if spec.threads_per_block > target.max_threads_per_block:
            return (
                False,
                f"threads_per_block {spec.threads_per_block} exceeds arch limit",
            )
        if not target.fits_lds(spec.lds_bytes):
            return (
                False,
                f"LDS {spec.lds_bytes} B exceeds {target.lds_capacity_bytes} B",
            )
    except (ValueError, KeyError) as e:
        return False, str(e)
    return True, "ok"


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

# Swept geometry. ``tile_w`` is not in this table because the useful values
# depend on ``Wo``: when ``Wo`` is not a multiple of ``tile_w``, the last tile in
# each row spends MFMA work on pixels past the image edge.
# ``tile_h`` stops at 8 because a shorter tile re-reads a larger share of halo
# rows per output row (``KH - 1`` extra rows amortised over ``tile_h``).
_SWEEP_TILE_H = (8, 16)
_SWEEP_TILE_K = (32, 64, 128, 256)
_SWEEP_CK = (16, 32, 64)
_SWEEP_WAVES = ((2, 2), (1, 4), (2, 4), (4, 2), (1, 8))
# Per MFMA tile size, the atoms to sweep in order of preference: the widest-K
# atom the target has wins (fewer MFMAs per chunk), so gfx950 sweeps
# 32x32x16 / 16x16x32 and gfx942 -- which has neither -- 32x32x8 / 16x16x16.
_SWEEP_ATOMS = (("32x32x16", "32x32x8"), ("16x16x32", "16x16x16"))


def _sweep_atoms(arch: str, dtype: str) -> "list[str]":
    """The preferred supported atom of each tile size on ``arch``."""
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    ab = "bf16" if dtype == "bf16" else "f16"
    out = []
    for prefs in _SWEEP_ATOMS:
        for atom in prefs:
            t, k, _ = _ATOMS[atom]
            if target.mma.has_shape(
                a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=t, n=t, k=k
            ):
                out.append(atom)
                break
    return out


def tile_w_candidates(Wo: int, atom_tile: int, max_mult: int = 6) -> "list[int]":
    """Widths worth trying for ``Wo`` with an ``atom_tile``-wide MFMA N dimension.

    Prefers widths that tile ``Wo`` with no wasted columns; if none do, falls
    back to the single width with the least waste so the shape is still covered.
    """
    cands = [atom_tile * m for m in range(1, max_mult + 1)]
    exact = [w for w in cands if Wo % w == 0]
    if exact:
        return exact
    return [min(cands, key=lambda w: -(-Wo // w) * w)]


def nongrouped_knobs(
    arch: str,
    dtype: str,
    *,
    Wo: "int | None" = None,
    iglp: "tuple[int | None, ...]" = (0,),
    waves_per_eu: "tuple[int | None, ...]" = (None, 3),
    swizzle_wgm: "tuple[int, ...]" = (8,),
) -> "Iterator[dict]":
    """The swept :class:`DirectNongroupedConvSpec` geometry, as spec kwargs.

    With ``Wo`` the widths are the :func:`tile_w_candidates` of that output
    width -- what a sweep for one shape needs. Without it every width up to
    six atoms is produced: the shape-independent grid an AOT cache builds, since
    a cached binary serves any ``Wo`` and the run side picks the widths that fit
    the shape. Unvalidated: :func:`is_valid_nongrouped_spec` drops what a given
    problem or arch cannot run.
    """
    import itertools

    for atom in _sweep_atoms(arch, dtype):
        at = _ATOMS[atom][0]
        widths = (
            tile_w_candidates(Wo, at)
            if Wo is not None
            else [at * m for m in range(1, 7)]
        )
        for tw in widths:
            for th, tk, ck, (wm, wn), wgm, ig, we in itertools.product(
                _SWEEP_TILE_H,
                _SWEEP_TILE_K,
                _SWEEP_CK,
                _SWEEP_WAVES,
                swizzle_wgm,
                iglp,
                waves_per_eu,
            ):
                yield dict(
                    tile_h=th,
                    tile_w=tw,
                    tile_k=tk,
                    ck=ck,
                    waves_m=wm,
                    waves_n=wn,
                    atom=atom,
                    swizzle_wgm=wgm,
                    iglp=ig,
                    waves_per_eu=we,
                )


def nongrouped_specs(
    problem: DirectConvProblem,
    *,
    arch: str = "gfx950",
    name: str = "direct_conv_nongrouped",
    iglp: "tuple[int | None, ...]" = (0,),
    waves_per_eu: "tuple[int | None, ...]" = (None, 3),
    swizzle_wgm: "tuple[int, ...]" = (8,),
) -> "list[DirectNongroupedConvSpec]":
    """Every valid :class:`DirectNongroupedConvSpec` worth benchmarking for ``problem``.

    Deduplicated by kernel name, so callers can compile the list directly.
    """
    out: "list[DirectNongroupedConvSpec]" = []
    seen = set()
    for knobs in nongrouped_knobs(
        arch,
        problem.dtype,
        Wo=problem.Wo,
        iglp=iglp,
        waves_per_eu=waves_per_eu,
        swizzle_wgm=swizzle_wgm,
    ):
        spec = DirectNongroupedConvSpec(problem=problem, name=name, **knobs)
        ok, _ = is_valid_nongrouped_spec(spec, arch=arch)
        if not ok:
            continue
        key = spec.kernel_name()
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def build_direct_conv_nongrouped(
    spec: DirectNongroupedConvSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for one non-grouped NHWC direct convolution kernel.

    AOT: the binary bakes the filter, ``stride``, ``PAD``, dtype and the tile
    geometry of ``spec``; batch, extents and channel counts are kernargs (see
    the module docstring). ``spec.problem`` only has to pass the validator --
    the emitted IR does not depend on its ``N``/``H``/``W``/``C``/``K``.
    """
    ok, why = is_valid_nongrouped_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectNongroupedConvSpec for {arch}: {why}")

    p = spec.problem
    io_type = _io_type(p.dtype)
    dtype = p.dtype

    KH, KW, S, PAD = p.KH, p.KW, p.stride, p.PAD
    N_TAPS = KH * KW

    TH, TW, TK, CK = spec.tile_h, spec.tile_w, spec.tile_k, spec.ck
    THREADS = spec.threads_per_block
    WAVE = spec.wave_size
    FRAG = spec.frag
    AK = spec.atom_k
    AT = spec.atom_tile  # MFMA M == N
    ACC = spec.acc_per_lane
    QUADS = ACC // 4  # accumulator slots come in quads of consecutive channels
    NCB = spec.n_col_blocks
    M_TILES = spec.m_tiles_total
    KATOMS = spec.k_atoms
    ROWS_W = spec.rows_per_wave
    MT_W = spec.m_tiles_per_wave
    NT_W = spec.n_tiles_per_wave
    CSTRIDE = spec.c_stride
    LDS_IN_H, LDS_IN_W = spec.lds_in_h, spec.lds_in_w

    # Staging pass counts.
    X_CV = CK // _X_LOAD_VEC  # channel vectors per staged pixel
    X_VECS = LDS_IN_H * LDS_IN_W * X_CV
    X_PASSES = (X_VECS + THREADS - 1) // THREADS
    W_SLOTS = N_TAPS * M_TILES * KATOMS * WAVE
    W_PASSES = (W_SLOTS + THREADS - 1) // THREADS

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    params = emit_direct_params(b, io_type=io_type)
    A = params["A"]
    Bp = params["B"]
    D = params["D"]
    A_bytes = params["A_bytes"]
    B_bytes = params["B_bytes"]
    D_bytes = params["D_bytes"]
    p_N = params["p_N"]
    p_Hi = params["p_Hi"]
    p_Wi = params["p_Wi"]
    p_Ho = params["p_Ho"]
    p_Wo = params["p_Wo"]
    # groups == 1, so the totals are the full channel counts.
    C = params["p_total_c"]
    K = params["p_total_k"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_half = b.const_i32(2)
    # Predication for the staging loads is loop-invariant, so it is folded into
    # the *base* offset once, before the channel loop, rather than re-selected
    # every iteration. validate() guarantees ``_OOB_BASE`` is at or past the
    # end of every operand tensor and leaves headroom for ``+ c_off`` without
    # wrapping i32, so ``base + c_off`` stays out of range for the whole loop.
    # A buffer load whose voffset exceeds num_records returns zero, which is
    # exactly the value a padded pixel or an out-of-range filter row needs --
    # so the masked *value* select disappears from the loop as well. The
    # epilogue redirects its masked stores to the same offset.
    oob_base = b.const_i32(_OOB_BASE)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # The staging loops are sized in whole thread-passes, so the last pass can
    # own slots past the end of the tile. Those slots write into a scratch tail
    # instead of running off the array (and over the neighbouring allocation).
    DB = spec.double_buffer
    x_stage = spec.lds_x_halves + _X_LOAD_VEC
    w_stage = spec.lds_w_halves + FRAG
    x_dump = spec.lds_x_halves
    w_dump = spec.lds_w_halves
    nbuf = 2 if DB else 1
    X_smem = b.smem_alloc(io_type, [1, x_stage * nbuf], name_hint="lds_x")
    W_smem = b.smem_alloc(io_type, [1, w_stage * nbuf], name_hint="lds_w")

    # ---- thread / wave decomposition -------------------------------------
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE))
    wave_id = b.div(tid, b.const_i32(WAVE))
    wave_m = b.div(wave_id, b.const_i32(spec.waves_n))
    wave_n = b.mod(wave_id, b.const_i32(spec.waves_n))
    lane_lo = b.mod(lane, b.const_i32(AT))  # M row / N column inside the atom
    lane_hi = b.div(lane, b.const_i32(AT))  # K slice inside the atom

    # ---- grid decode ------------------------------------------------------
    # Flat 1-D grid of (spatial cell x channel tile). The tile *sizes* are
    # build-time, but how many tiles the problem needs follows the runtime
    # extents, so the counts are a handful of scalar divides once per
    # workgroup. The swizzle decides which of the two axes a single XCD walks
    # contiguously; see DirectNongroupedConvSpec.
    n_wt = b.div(b.add(p_Wo, b.const_i32(TW - 1)), b.const_i32(TW))
    n_ht = b.div(b.add(p_Ho, b.const_i32(TH - 1)), b.const_i32(TH))
    n_kt = b.div(b.add(K, b.const_i32(TK - 1)), b.const_i32(TK))
    n_hw = b.mul(n_wt, n_ht)
    n_cells = b.mul(n_hw, p_N)

    wgid = b.block_id_x()
    if spec.chiplet_swizzle:
        from rocke.helpers.grid import chiplet_aware_super_tile_dynamic

        # A swizzle_wgm above the runtime channel-tile count needs no clamp:
        # the swizzle sizes its last group as min(wgm, tiles left).
        sw = chiplet_aware_super_tile_dynamic(
            b,
            wgid,
            num_pid_m=n_kt,
            num_pid_n=n_cells,
            wgm=spec.swizzle_wgm,
            num_xcds=spec.num_xcds,
            chunk_size=spec.chiplet_chunk,
        )
        k_tile, cell = sw.row, sw.col
    else:
        k_tile = b.mod(wgid, n_kt)
        cell = b.div(wgid, n_kt)

    n_img = b.div(cell, n_hw)
    st = b.mod(cell, n_hw)
    w_tile = b.mod(st, n_wt)
    h_tile = b.div(st, n_wt)

    # Output-tile origins.
    out_h0 = b.mul(h_tile, b.const_i32(TH))
    out_w0 = b.mul(w_tile, b.const_i32(TW))
    k_base = b.mul(k_tile, b.const_i32(TK))

    # Input origin of the staged (halo-inclusive) activation tile.
    in_h0 = b.sub(b.mul(out_h0, b.const_i32(S)), b.const_i32(PAD))
    in_w0 = b.sub(b.mul(out_w0, b.const_i32(S)), b.const_i32(PAD))

    # ---- staging metadata (channel-independent; hoisted out of the C loop) --

    def _x_pass_meta():
        """Per-pass (global byte offset base, LDS half index) for X."""
        meta = []
        img_base = b.mul(n_img, params["p_A_stride_n"])
        for j in range(X_PASSES):
            v = b.add(tid, b.const_i32(j * THREADS))
            cv = b.mod(v, b.const_i32(X_CV))
            pos = b.div(v, b.const_i32(X_CV))
            ih_l = b.div(pos, b.const_i32(LDS_IN_W))
            iw_l = b.mod(pos, b.const_i32(LDS_IN_W))
            ih = b.add(in_h0, ih_l)
            iw = b.add(in_w0, iw_l)
            ok_h = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, p_Hi))
            ok_w = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, p_Wi))
            valid = b.land(ok_h, ok_w)
            if X_VECS % THREADS != 0:
                in_tile = b.cmp_lt(v, b.const_i32(X_VECS))
                valid = b.land(valid, in_tile)
            else:
                in_tile = None
            # base = n*stride_n + ih*stride_hi + iw*stride_wi + cv*VEC; the
            # chunk offset is added per iteration.
            pix_off = b.add(
                b.mul(ih, params["p_A_stride_hi"]), b.mul(iw, params["p_A_stride_wi"])
            )
            elems = b.add(b.add(img_base, pix_off), b.mul(cv, b.const_i32(_X_LOAD_VEC)))
            lds_idx = b.add(
                b.mul(pos, b.const_i32(CSTRIDE)), b.mul(cv, b.const_i32(_X_LOAD_VEC))
            )
            if in_tile is not None:
                lds_idx = b.select(in_tile, lds_idx, b.const_i32(x_dump))
            base = b.select(valid, b.mul(elems, c_half), oob_base)
            meta.append((base, lds_idx))
        return meta

    def _w_pass_meta():
        """Per-pass (global byte offset base, LDS half index) for W.

        Slot ``g`` decodes to the MFMA fragment that lane ``g % 64`` consumes for
        ``(tap, m_tile, k_atom)``, so the LDS index is simply ``g * FRAG``.
        """
        meta = []
        for j in range(W_PASSES):
            g = b.add(tid, b.const_i32(j * THREADS))
            g_lane = b.mod(g, b.const_i32(WAVE))
            rest = b.div(g, b.const_i32(WAVE))
            katom = b.mod(rest, b.const_i32(KATOMS))
            rest2 = b.div(rest, b.const_i32(KATOMS))
            m_tile = b.mod(rest2, b.const_i32(M_TILES))
            tap = b.div(rest2, b.const_i32(M_TILES))
            r = b.div(tap, b.const_i32(KW))
            s = b.mod(tap, b.const_i32(KW))

            k_out = b.add(
                k_base,
                b.add(
                    b.mul(m_tile, b.const_i32(AT)),
                    b.mod(g_lane, b.const_i32(AT)),
                ),
            )
            c_in_chunk = b.add(
                b.mul(katom, b.const_i32(AK)),
                b.mul(b.div(g_lane, b.const_i32(AT)), b.const_i32(FRAG)),
            )
            valid = b.cmp_lt(k_out, K)
            if W_SLOTS % THREADS != 0:
                in_tile = b.cmp_lt(g, b.const_i32(W_SLOTS))
                valid = b.land(valid, in_tile)
            else:
                in_tile = None
            # base = ((k_out*KH + r)*KW + s)*C + c_in_chunk (KRSC)
            krs_h = b.mul(k_out, b.const_i32(KH))
            krs_hr = b.add(krs_h, r)
            krs_w = b.mul(krs_hr, b.const_i32(KW))
            krs = b.add(krs_w, s)
            elems = b.add(b.mul(krs, C), c_in_chunk)
            lds_idx = b.mul(g, b.const_i32(FRAG))
            if in_tile is not None:
                lds_idx = b.select(in_tile, lds_idx, b.const_i32(w_dump))
            base = b.select(valid, b.mul(elems, c_half), oob_base)
            meta.append((base, lds_idx))
        return meta

    x_meta = _x_pass_meta()
    w_meta = _w_pass_meta()

    x_dwords = _X_LOAD_VEC // 2
    w_dwords = FRAG // 2

    def issue_stage_loads(c_off: Value):
        """Issue the global loads for one channel chunk; returns raw registers."""
        xs = [
            _buf_load_vN(b, dtype, a_rsrc, b.add(base, c_off), c0, x_dwords)
            for base, _ in x_meta
        ]
        ws = [
            _buf_load_vN(b, dtype, b_rsrc, b.add(base, c_off), c0, w_dwords)
            for base, _ in w_meta
        ]
        return xs, ws

    def commit_stage(xs, ws, x_buf=None, w_buf=None):
        for (_, lds_idx), val in zip(x_meta, xs):
            idx = lds_idx if x_buf is None else b.add(lds_idx, x_buf)
            b.smem_store_vN(X_smem, [c0, idx], val, _X_LOAD_VEC)
        for (_, lds_idx), val in zip(w_meta, ws):
            idx = lds_idx if w_buf is None else b.add(lds_idx, w_buf)
            b.smem_store_vN(W_smem, [c0, idx], val, FRAG)

    # ---- per-lane LDS read bases -----------------------------------------
    # X: idx = wave row term + lane term + compile-time term.
    x_wave_term = b.mul(
        b.mul(wave_n, b.const_i32(ROWS_W * S)), b.const_i32(LDS_IN_W * CSTRIDE)
    )
    x_lane_term = b.add(
        b.mul(lane_lo, b.const_i32(S * CSTRIDE)), b.mul(lane_hi, b.const_i32(FRAG))
    )
    x_read_base = b.add(x_wave_term, x_lane_term)
    # W: idx = wave_m block + lane*FRAG + compile-time term.
    w_read_base = b.add(
        b.mul(wave_m, b.const_i32(MT_W * KATOMS * WAVE * FRAG)),
        b.mul(lane, b.const_i32(FRAG)),
    )

    def read_a_frag(base: Value, tap: int, mt: int, katom: int) -> Value:
        off = ((tap * M_TILES + mt) * KATOMS + katom) * WAVE * FRAG
        idx = b.add(base, b.const_i32(off))
        return b.smem_load_vN(W_smem, c0, idx, dtype=io_type, n=FRAG)

    def read_b_frag(base: Value, in_row: int, cb: int, s: int, katom: int) -> Value:
        """One activation fragment, addressed by *input* row inside the wave window.

        Taps share activations: output row ``row`` at tap row ``r`` reads input
        row ``row * stride + r``, so the ``KH`` taps of a column only need
        ``(ROWS_W - 1) * stride + KH`` distinct fragments instead of
        ``ROWS_W * KH``.  Indexing by input row is what makes that reuse
        expressible.
        """
        pos = in_row * LDS_IN_W + cb * AT * S + s
        off = pos * CSTRIDE + katom * AK
        idx = b.add(base, b.const_i32(off))
        return b.smem_load_vN(X_smem, c0, idx, dtype=io_type, n=FRAG)

    # ---- main channel loop ------------------------------------------------
    zero_acc = b.zero_vec_f32(ACC)
    n_acc = MT_W * NT_W
    n_in_rows = (ROWS_W - 1) * S + KH

    def emit_mfmas(accs, x_base, w_base):
        """One channel chunk of MFMAs against the staged tiles.

        Filter-column (``s``) outermost: every activation fragment loaded for
        one ``(s, k_atom)`` is consumed by all ``KH`` tap rows and all ``MT_W``
        channel tiles.
        """
        for katom in range(KATOMS):
            for s_c in range(KW):
                b_frags = [
                    [read_b_frag(x_base, ir, cb, s_c, katom) for cb in range(NCB)]
                    for ir in range(n_in_rows)
                ]
                for r_c in range(KH):
                    tap = r_c * KW + s_c
                    a_frags = [
                        read_a_frag(w_base, tap, mt, katom) for mt in range(MT_W)
                    ]
                    for row in range(ROWS_W):
                        bf_row = b_frags[row * S + r_c]
                        for cb in range(NCB):
                            nt = row * NCB + cb
                            for mt in range(MT_W):
                                idx = mt * NT_W + nt
                                accs[idx] = _mfma(
                                    b,
                                    dtype,
                                    spec.atom,
                                    a_frags[mt],
                                    bf_row[cb],
                                    accs[idx],
                                )

    c_ck_bytes = b.const_i32(CK * 2)
    xs0, ws0 = issue_stage_loads(c0)

    if DB:
        # Prologue: publish chunk 0 into buffer 0 and have chunk 1 in flight.
        commit_stage(xs0, ws0)
        xs0, ws0 = issue_stage_loads(c_ck_bytes)

    iter_args: List[Tuple[str, Value]] = [(f"acc{i}", zero_acc) for i in range(n_acc)]
    iter_args += [(f"xs{i}", v) for i, v in enumerate(xs0)]
    iter_args += [(f"ws{i}", v) for i, v in enumerate(ws0)]

    # Runtime trip count. validate() guarantees C is a positive whole number of
    # chunks, so the clamp to one chunk never changes it. It is there for LLVM:
    # unable to prove the loop runs at all, it keeps a zero-trip guard and
    # schedules the prefetch loads late in the body.
    n_chunks = b.smax(b.div(C, b.const_i32(CK)), c1)
    loop = b.scf_for_iter(
        c0,
        n_chunks,
        c1,
        iter_args,
        iv_name="c_iter",
        elide_trailing_barrier=False,
    )
    with loop as (c_iv, carried):
        accs = list(carried[:n_acc])
        xs = list(carried[n_acc : n_acc + X_PASSES])
        ws = list(carried[n_acc + X_PASSES :])

        if spec.iglp is not None:
            b.iglp_opt(spec.iglp)

        if DB:
            # Parity of the induction variable picks the live buffer; every
            # fragment offset stays a compile-time constant off these bases.
            par = b.mod(c_iv, b.const_i32(2))
            cur_x = b.mul(par, b.const_i32(x_stage))
            cur_w = b.mul(par, b.const_i32(w_stage))
            nxt_x = b.sub(b.const_i32(x_stage), cur_x)
            nxt_w = b.sub(b.const_i32(w_stage), cur_w)

            b.sync()
            # Stage chunk i+1 into the idle buffer while chunk i still feeds MFMAs.
            commit_stage(xs, ws, nxt_x, nxt_w)
            c_off = b.mul(b.add(c_iv, b.const_i32(2)), c_ck_bytes)
            xs_n, ws_n = issue_stage_loads(c_off)
            emit_mfmas(accs, b.add(x_read_base, cur_x), b.add(w_read_base, cur_w))
        else:
            # Publish the chunk prefetched during the previous iteration.
            b.sync()
            commit_stage(xs, ws)
            b.sync()
            c_off = b.mul(b.add(c_iv, c1), c_ck_bytes)
            xs_n, ws_n = issue_stage_loads(c_off)
            emit_mfmas(accs, x_read_base, w_read_base)

        b.scf_yield(*accs, *xs_n, *ws_n)

    accs_out = list(loop.results[:n_acc])

    # ---- epilogue ---------------------------------------------------------
    # Square-atom accumulator: slot i -> row = (i//4)*(AT//4) + lane_hi*4 + (i%4),
    # col = lane_lo.  Slots 4q..4q+3 are four consecutive output channels, so a
    # quad packs into one dwordx2 store.  (AT=32 -> 4 quads spaced 8 channels
    # apart; AT=16 -> a single quad.)
    k_lane_base = b.add(
        b.add(k_base, b.mul(wave_m, b.const_i32(MT_W * AT))),
        b.mul(lane_hi, b.const_i32(4)),
    )
    row_base = b.add(out_h0, b.mul(wave_n, b.const_i32(ROWS_W)))
    img_out_base = b.mul(n_img, params["p_D_stride_n"])

    for mt in range(MT_W):
        k_mt = b.add(k_lane_base, b.const_i32(mt * AT))
        for row in range(ROWS_W):
            out_h = b.add(row_base, b.const_i32(row))
            h_ok = b.cmp_lt(out_h, p_Ho)
            row_off = b.add(img_out_base, b.mul(out_h, params["p_D_stride_ho"]))
            for cb in range(NCB):
                nt = row * NCB + cb
                acc = accs_out[mt * NT_W + nt]
                out_w = b.add(b.add(out_w0, b.const_i32(cb * AT)), lane_lo)
                hw_ok = b.land(h_ok, b.cmp_lt(out_w, p_Wo))
                base_off = b.add(row_off, b.mul(out_w, params["p_D_stride_wo"]))
                for q in range(QUADS):
                    k_out = b.add(k_mt, b.const_i32(q * (AT // 4)))
                    valid = b.land(hw_ok, b.cmp_lt(k_out, K))
                    d_off = b.mul(b.add(base_off, k_out), c_half)
                    # Same sentinel as the staging loads: validate() keeps D
                    # at or below _OOB_BASE, so the redirected store lies
                    # past num_records and the hardware drops it.
                    safe = b.select(valid, d_off, oob_base)
                    quad = b.vec_pack(
                        [b.vec_extract(acc, 4 * q + j) for j in range(4)], F32
                    )
                    _buf_store_vN(
                        b, dtype, d_rsrc, safe, c0, _trunc_f32(b, dtype, quad), 2
                    )

    return b.kernel
