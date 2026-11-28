# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native-NCHW direct-style backward-weight (wgrad) convolution (gfx950).

A productionized AOT-buildable kernel family: a C++ mirror lives at
``library/cpp/instances/common/conv_direct_nchw_wgrad.cpp`` and is gated by
``tools/check_byte_identity.py`` like every other family, with a ``.py``/``.c``
parity pair under ``library/tests/parity/``. Runtime dispatch selection is
**deliberately not yet wired**: this family is not registered in
``library/dispatch/grouped_convolution.py`` or its instance JSONs, so
``algorithm="auto"`` never reaches it. Callers build and launch it explicitly
(directly via :func:`build_direct_conv_nchw_wgrad`, or through the AOT sweep
in ``kernels/conv/direct_variants.py`` / ``kernels/conv/direct_aot_jobs.py``
with variant ``"direct_nchw_wgrad"``), not via the request-level dispatcher.
Sibling of :mod:`kernels.common.conv_wgrad_nchw_proto` (the implicit-GEMM-style
NCHW wgrad spike, still a research spike). Every problem extent is compiled
in, ``groups == 1``, no dilation, fp16 / bf16, ``gfx950``.

Algorithm
---------
::

    dW[k, c, y, x] = sum_{n, ho, wo} dY[n, k, ho, wo] * X[n, c, ho*S-P+y, wo*S-P+x]

A workgroup owns a ``tile_m x tile_c`` block of ``(k, c)`` for all the filter
taps of ``KH / y_groups`` filter rows, and walks a list of *spatial tiles*
(``tile_h x tile_w`` output positions of one image) -- the reduction. Per
spatial tile it stages, **once**:

* the ``dY`` tile ``[k][ho][wo]`` (reduction innermost): aligned 8-wide ``wo``
  runs, one ``buffer_load_dwordx4`` + ``ds_write_b128`` each, no transpose;
* the ``X`` *window* the tile's taps read: ``(tile_h-1)*S + rows`` input rows
  of ``S*tile_w`` aligned body columns (whole 8-element runs, one
  ``dwordx4`` each, entirely inside or outside the image) plus the left /
  right halo columns (one narrow aligned load per row and channel), stored
  ``[row][col][c]`` with ``c`` innermost (padded by ``lds_pad``).

Every tap then reads the same LDS window: the tap ``(y, x)`` is the build-time
LDS offset ``(y*LDS_W + x) * c_stride``, the stride ``S`` only scales the
per-lane position offsets. No per-tap global reload, no stride-2
over-fetch, no even-element picking, no runtime shuffles.

MFMA feed (``v_mfma_f32_16x16x32_{f16,bf16}``, ``dY`` = A, ``X`` = B):

* A: lane ``l`` needs ``k = l % 16`` and 8 consecutive reduction positions;
  ``dY`` is position-contiguous in LDS, so that is one ``ds_read_b128``,
  shared by every tap.
* B: lane ``l`` needs ``c = l % 16`` and the same 8 positions shifted by the
  tap. In the ``[row][col][c]`` window a position is an LDS row, so the
  fragment is two ``ds_read_b64_tr_b16`` transpose reads (the
  ``conv_direct_wgrad_tiled`` access pattern) whose per-lane row is
  ``position * S + tap`` -- 8-byte aligned at any tap and any stride.

Moving ``c`` innermost is a register transpose of the NCHW staging loads:
each staging thread loads ``stage_group`` (``G``) channels' runs and writes
``G``-channel vectors per position (``G = 4``: ``ds_write_b64``; ``G = 8``:
``ds_write_b128``), the ``stage_group`` lever of the NCHW dgrad spike.

Parallelism: ``y_groups`` spreads filter rows over workgroups, ``split`` (a
kernarg) splits the spatial-tile list, and the partials meet in a
caller-zeroed KCYX ``dW`` through packed 16-bit atomics, the numerics of the
implicit-GEMM NCHW spike. ``waves_t`` splits a workgroup's taps over waves
that share the staged window. With ``xcd_remap`` (default) the grid is 1-D
and renumbered so the workgroups of one spatial slice share an XCD's L2
(:meth:`DirectNchwWgradSpec.launch_grid`).

Levers (see the spec fields): pinning the register
prefetch ahead of the MFMAs (the scheduler otherwise sinks it), an LDS-staged
epilogue whose atomics cover contiguous KCYX runs (the per-lane one strides
18 bytes), the "2row" k-step map with model-picked pads that makes the X
transpose reads bank-conflict free, ``waves_per_eu`` / ``prefetch`` for two
waves per SIMD, double-buffered LDS and a plain-store split-1 epilogue.

Preconditions (checked by :func:`is_valid_direct_nchw_wgrad_spec`): ``W`` and
``Wo`` multiples of 8, ``tile_w`` a power of two >= 8, ``tile_h * tile_w`` a
multiple of 32, halo widths <= 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.ir import BF16, F16, I32, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join

from kernels.common.conv_direct_grouped import DirectConvProblem

_DTYPES = ("fp16", "bf16")
# Masked loads are redirected past every operand's num_records (reads zero).
_OOB_BASE = 0x7F000000
_ATOM = 16  # mfma 16x16x32
_AK = 32  # reduction depth of one MFMA
_RUN = 8  # 16-bit elements in one dwordx4
_WAVE = 64
_XCDS = 8  # gfx950 XCDs per device (block ids are dealt round-robin over them)
_DY_PAD = 8  # dY row stride = tile_h*tile_w + 8: conflict-free ds_read_b128

# Test / profiling-only fault injection: a set of names among
# ``"halo_unmasked"`` (halo loads skip the image-column check, so padding
# columns read the neighbouring row) and ``"rows_unmasked"`` (window rows skip
# the image-row check) -- the mutation check of the test module -- and the
# profiling cuts ``"no_epilogue"`` (no dW writes), ``"no_mfma"``,
# ``"no_loads"`` (X window loads read zero) and ``"no_b_reads"`` (a register
# stand-in for every X fragment read). Empty in every real build.
_MUTATIONS: frozenset = frozenset()


@dataclass(frozen=True)
class DirectNchwWgradSpec:
    """One direct-style native-NCHW wgrad configuration.

    ``tile_m`` output channels x ``tile_c`` input channels x (``KH/y_groups *
    KW`` taps) per workgroup; ``tile_h x tile_w`` output positions per
    reduction step. Waves: ``waves_m`` split ``tile_m``, ``waves_c`` split
    ``tile_c``, ``waves_t`` split the workgroup's taps.
    """

    problem: DirectConvProblem
    name: str = "conv_direct_nchw_wgrad"
    tile_m: int = 64
    tile_c: int = 64
    tile_h: int = 4
    tile_w: int = 32
    waves_m: int = 2
    waves_c: int = 2
    waves_t: int = 1
    y_groups: int = 1
    stage_group: int = 4
    lds_pad: int = 8
    waves_per_eu: int = 0
    # MFMA issue order: "row" groups by X window row (each X fragment read
    # once, shared by every k step / filter row on it); "ks" walks k steps.
    mfma_order: str = "row"
    # Register-prefetch the next spatial tile's global loads (else load,
    # wait and stage at the top of every tile).
    prefetch: bool = True
    # Flatten the grid and renumber block ids so the workgroups of one
    # spatial slice share an XCD (and its L2).
    xcd_remap: bool = True
    # dW epilogue: "lds" stages each (m atom, c atom) slab of the
    # accumulators in LDS as [k][c][tap] and issues packed atomics over
    # contiguous KCYX runs (64 lanes = 256 contiguous bytes); "direct" has
    # every lane add its own (k, c, tap pair) values straight from the
    # accumulators (lanes 18 bytes apart). "lds" needs y_groups == 1 (a
    # workgroup's taps of one (k, c) are then a contiguous run).
    epilogue: str = "lds"
    # k-step -> tile positions: "row" = 32 consecutive positions (4 groups of
    # 8 in one row when tile_w >= 32); "2row" = 16 columns of rows r and
    # r + tile_h/2, lane groups g alternating between them, so the two groups
    # of a half-wave sit tile_h/2 window rows apart -- with the matching pads
    # (``lds_pad``, ``lds_wpad``) the X transpose reads are bank-conflict
    # free (see :func:`x_read_conflict_ways`), and k steps one row apart
    # still share X fragments across filter rows.
    kmap: str = "row"
    # Extra positions per X window row (the row stride is lds_w + lds_wpad).
    lds_wpad: int = 0
    # 2: double-buffered LDS (needs prefetch): tile t+1 is staged into the
    # other buffer under tile t's MFMAs, the loads of t+2 are in flight, one
    # barrier per tile. 1: stage, barrier, MFMA, barrier.
    lds_buffers: int = 1
    # False: plain dword stores instead of packed atomics ("lds" epilogue
    # only). Each (k, c, tap) then has one writer, so the host must launch
    # with split == 1 -- and need not zero dW (no memset launch).
    atomic: bool = True

    # ---- derived ----------------------------------------------------------

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def rows_per_group(self) -> int:
        return self.problem.KH // self.y_groups

    @property
    def wg_taps(self) -> int:
        return self.rows_per_group * self.problem.KW

    @property
    def taps_per_wave(self) -> int:
        return self.wg_taps // self.waves_t

    @property
    def mt_w(self) -> int:
        return self.tile_m // (self.waves_m * _ATOM)

    @property
    def ct_w(self) -> int:
        return self.tile_c // (self.waves_c * _ATOM)

    @property
    def threads(self) -> int:
        return self.waves_m * self.waves_c * self.waves_t * _WAVE

    @property
    def halo_l(self) -> int:
        return self.problem.PAD

    @property
    def halo_r(self) -> int:
        p = self.problem
        return max(0, p.KW - p.PAD - self.S)

    @staticmethod
    def halo_vec(h: int) -> int:
        v = 2
        while v < h:
            v *= 2
        return v

    @property
    def lds_h(self) -> int:
        return (self.tile_h - 1) * self.S + self.rows_per_group

    @property
    def lds_w(self) -> int:
        return self.halo_l + self.S * self.tile_w + self.halo_r

    @property
    def lds_w_stride(self) -> int:
        return self.lds_w + self.lds_wpad

    @property
    def c_stride(self) -> int:
        return self.tile_c + self.lds_pad

    @property
    def dy_stride(self) -> int:
        return self.tile_h * self.tile_w + _DY_PAD

    @property
    def lds_x_elems(self) -> int:
        return self.lds_h * self.lds_w_stride * self.c_stride

    @property
    def lds_dy_elems(self) -> int:
        return self.tile_m * self.dy_stride

    @property
    def epi_elems(self) -> int:
        """LDS elements of one "lds" epilogue slab (it reuses the X buffer)."""
        if self.epilogue != "lds":
            return 0
        return self.waves_m * _ATOM * self.waves_c * _ATOM * self.wg_taps

    @property
    def x_buf_elems(self) -> int:
        # + one dump slot for a partial last staging pass.
        return max(self.lds_x_elems + _RUN, self.epi_elems)

    @property
    def x_buf_stride(self) -> int:
        return ceil_div(self.lds_x_elems + _RUN, _RUN) * _RUN

    @property
    def dy_buf_stride(self) -> int:
        return self.lds_dy_elems + _RUN

    @property
    def lds_bytes(self) -> int:
        nb = self.lds_buffers
        return 2 * (
            max(nb * self.x_buf_stride, self.epi_elems) + nb * self.dy_buf_stride
        )

    @property
    def k_steps(self) -> int:
        return self.tile_h * self.tile_w // _AK

    @property
    def acc_vgprs(self) -> int:
        return self.mt_w * self.ct_w * self.taps_per_wave * 4

    def n_tiles_hw(self) -> Tuple[int, int]:
        p = self.problem
        return ceil_div(p.Ho, self.tile_h), ceil_div(p.Wo, self.tile_w)

    @property
    def n_tiles(self) -> int:
        nh, nw = self.n_tiles_hw()
        return self.problem.N * nh * nw

    def grid(self, split: int) -> Tuple[int, int, int]:
        """The 3-D (c x y-group, k, split) tile grid; see :meth:`launch_grid`."""
        p = self.problem
        return (
            p.total_c // self.tile_c * self.y_groups,
            p.total_k // self.tile_m,
            split,
        )

    def launch_grid(self, split: int) -> Tuple[int, int, int]:
        """The grid to launch: :meth:`grid`, flattened to 1-D with ``xcd_remap``."""
        gx, gy, gz = self.grid(split)
        return (gx * gy * gz, 1, 1) if self.xcd_remap else (gx, gy, gz)

    def kernel_name(self) -> str:
        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_c}",
            f"s{self.tile_h}x{self.tile_w}",
            f"w{self.waves_m}x{self.waves_c}x{self.waves_t}",
            f"g{self.stage_group}",
            f"p{self.lds_pad}",
            f"yg{self.y_groups}" if self.y_groups > 1 else "",
            f"wpe{self.waves_per_eu}" if self.waves_per_eu else "",
            "oks" if self.mfma_order == "ks" else "",
            "" if self.prefetch else "nopf",
            "" if self.xcd_remap else "noxcd",
            "" if self.epilogue == "lds" else "edir",
            "k2r" if self.kmap == "2row" else "",
            f"wp{self.lds_wpad}" if self.lds_wpad else "",
            "db" if self.lds_buffers == 2 else "",
            "" if self.atomic else "st",
            "bf16" if p.dtype == "bf16" else "",
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in _DTYPES:
            raise ValueError(
                f"unsupported dtype {p.dtype!r}; expected one of {_DTYPES}"
            )
        if p.groups != 1:
            raise ValueError(f"groups must be 1 (got {p.groups})")
        if p.stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2 (got {p.stride})")
        if p.is_dilated:
            raise ValueError("dilation is not supported")
        if p.PAD_W != p.PAD:
            raise ValueError("per-axis padding is not supported")
        if p.W % _RUN or p.Wo % _RUN:
            raise ValueError(f"W {p.W} and Wo {p.Wo} must be multiples of {_RUN}")
        if self.halo_l > _RUN or self.halo_r > _RUN:
            raise ValueError("halo wider than 8 columns")
        tw = self.tile_w
        if tw < _RUN or tw & (tw - 1):
            raise ValueError(f"tile_w {tw} must be a power of two >= 8")
        if self.tile_h < 1 or (self.tile_h * tw) % _AK:
            raise ValueError("tile_h * tile_w must be a positive multiple of 32")
        if self.y_groups < 1 or p.KH % self.y_groups:
            raise ValueError(f"y_groups {self.y_groups} must divide KH = {p.KH}")
        if self.waves_t < 1 or self.wg_taps % self.waves_t:
            raise ValueError(
                f"waves_t {self.waves_t} must divide the {self.wg_taps} taps"
            )
        tpw, kw = self.taps_per_wave, p.KW
        if tpw % kw and kw % tpw:
            # Keeps every tap offset a build-time constant past a per-wave base.
            raise ValueError("taps per wave must be a multiple or a divisor of KW")
        if self.tile_m % (self.waves_m * _ATOM) or p.total_k % self.tile_m:
            raise ValueError("tile_m must divide K and be a multiple of waves_m*16")
        if self.tile_c % (self.waves_c * _ATOM) or p.total_c % self.tile_c:
            raise ValueError("tile_c must divide C and be a multiple of waves_c*16")
        g = self.stage_group
        if g not in (4, 8) or self.tile_c % g:
            raise ValueError("stage_group must be 4 or 8 and divide tile_c")
        if self.lds_pad < 0 or self.lds_pad % g:
            raise ValueError("lds_pad must be a non-negative multiple of stage_group")
        if p.total_c % 2:
            raise ValueError("C must be even (packed 16-bit dW atomics)")
        if self.acc_vgprs > 256:
            raise ValueError(
                f"{self.acc_vgprs} accumulator registers per lane (max 256)"
            )
        if self.threads > 1024:
            raise ValueError("too many threads")
        if self.waves_per_eu < 0:
            raise ValueError("waves_per_eu must be >= 0")
        if self.epilogue not in ("lds", "direct"):
            raise ValueError(f"epilogue {self.epilogue!r} must be 'lds' or 'direct'")
        if self.epilogue == "lds" and self.y_groups != 1:
            raise ValueError("the lds epilogue needs y_groups == 1")
        if self.kmap not in ("row", "2row"):
            raise ValueError(f"kmap {self.kmap!r} must be 'row' or '2row'")
        if self.kmap == "2row" and (self.tile_w % 16 or self.tile_h % 2):
            raise ValueError("kmap '2row' needs tile_w % 16 == 0 and an even tile_h")
        if self.lds_buffers not in (1, 2) or (
            self.lds_buffers == 2 and not self.prefetch
        ):
            raise ValueError("lds_buffers must be 1, or 2 with prefetch")
        if not self.atomic and self.epilogue != "lds":
            raise ValueError("atomic=False needs the lds epilogue")
        if self.lds_wpad < 0:
            raise ValueError("lds_wpad must be >= 0")
        if self.mfma_order not in ("row", "ks"):
            raise ValueError(f"mfma_order {self.mfma_order!r} must be 'row' or 'ks'")
        eb = 2
        big = max(
            eb * p.N * p.total_c * p.H * p.W,
            eb * p.N * p.total_k * p.Ho * p.Wo,
            eb * p.total_k * p.total_c * p.KH * p.KW,
        )
        # Masked loads add a per-channel soffset to the OOB base.
        if big + eb * self.stage_group * p.H * p.W + 64 > _OOB_BASE:
            raise ValueError("tensors must fit below the masked-load offset")


def kstep_origin(spec: DirectNchwWgradSpec, ks: int) -> Tuple[int, int]:
    """Build-time (row, col) of k step ``ks`` (lane group 0's first position)."""
    tw = spec.tile_w
    if spec.kmap == "2row":
        halves = tw // 16
        return ks // halves, 16 * (ks % halves)
    if tw >= _AK:
        return (ks * _AK) // tw, (ks * _AK) % tw
    return ks * (_AK // tw), 0


def kstep_group_pos(spec: DirectNchwWgradSpec, ks: int, g: int) -> Tuple[int, int]:
    """Tile (row, col) of the first of lane group ``g``'s 8 positions."""
    rk, ck = kstep_origin(spec, ks)
    if spec.kmap == "2row":
        return rk + (g % 2) * (spec.tile_h // 2), ck + 8 * (g // 2)
    if spec.tile_w >= _AK:
        return rk, ck + 8 * g
    return rk + (8 * g) // spec.tile_w, (8 * g) % spec.tile_w


def x_read_conflict_ways(spec: DirectNchwWgradSpec, banks: int = 64) -> int:
    """Worst LDS bank-conflict degree of the X transpose reads (model).

    ``ds_read_b64_tr_b16`` is serviced a 32-lane half-wave per pass over
    ``banks`` 4-byte banks; returns the largest number of distinct dwords
    one bank serves in one pass, over every read the kernel issues (any
    k step, tap, ``rd`` half and c atom; wave-uniform offsets excluded).
    """
    p = spec.problem
    S, CS, LW = spec.S, spec.c_stride, spec.lds_w_stride
    worst = 1
    for ks in range(spec.k_steps):
        for y in range(spec.rows_per_group):
            for x in range(p.KW):
                for rd in range(2):
                    for half in range(2):
                        hit: dict = {}
                        for lane in range(32 * half, 32 * half + 32):
                            g, q, c4 = lane // 16, (lane // 4) % 4, lane % 4
                            r, c = kstep_group_pos(spec, ks, g)
                            pos = (r * S + y) * LW + (c + rd * 4 + q) * S + x
                            d = (pos * CS + c4 * 4) // 2
                            for w in range(2):
                                hit.setdefault((d + w) % banks, set()).add(d + w)
                        worst = max(worst, max(len(v) for v in hit.values()))
    return worst


def conflict_free_pads(
    spec: DirectNchwWgradSpec, pads=range(0, 65, 4), wpads=range(0, 16)
) -> Tuple[int, int, int]:
    """``(ways, lds_pad, lds_wpad)`` minimising the modelled X-read conflicts,
    then the X window's LDS footprint."""
    from dataclasses import replace

    best = None
    for pad in pads:
        if pad % spec.stage_group:
            continue
        for wp in wpads:
            s2 = replace(spec, lds_pad=pad, lds_wpad=wp)
            key = (x_read_conflict_ways(s2), s2.lds_x_elems, pad, wp)
            if best is None or key < best:
                best = key
            if best[0] == 1 and key[0] == 1:
                break
    return best[0], best[2], best[3]


def is_valid_direct_nchw_wgrad_spec(
    spec: DirectNchwWgradSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """``(ok, reason)`` for ``spec`` on ``arch`` without raising."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
        spec.validate()
        ab = "bf16" if spec.problem.dtype == "bf16" else "f16"
        if not target.mma.has_shape(
            a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
        ):
            return False, f"missing mfma_f32_16x16x32_{ab} on {arch}"
        if target.wave_size != _WAVE or not target.memory.has_ds_read_tr:
            return False, f"needs wave64 + ds_read_b64_tr_b16 (absent on {arch})"
        if spec.threads > target.max_threads_per_block:
            return False, f"{spec.threads} threads exceed the arch limit"
        if not target.fits_lds(spec.lds_bytes):
            return (
                False,
                f"LDS {spec.lds_bytes} B exceeds {target.lds_capacity_bytes} B",
            )
    except (ValueError, KeyError) as e:
        return False, str(e)
    return True, "ok"


def direct_nchw_wgrad_signature(spec: DirectNchwWgradSpec) -> list:
    dt = "bf16" if spec.problem.dtype == "bf16" else "f16"
    return (
        SignatureBuilder()
        .ptr("dY", dt)
        .ptr("X", dt)
        .ptr("dW", dt)
        .scalar("dY_bytes", "i32")
        .scalar("X_bytes", "i32")
        .scalar("split", "i32")
        .build()
    )


def direct_nchw_wgrad_launch_values(
    dY_ptr: int, X_ptr: int, dW_ptr: int, dY_bytes: int, X_bytes: int, split: int
) -> dict:
    return {
        "dY": int(dY_ptr),
        "X": int(X_ptr),
        "dW": int(dW_ptr),
        "dY_bytes": int(dY_bytes),
        "X_bytes": int(X_bytes),
        "split": int(split),
    }


def build_direct_conv_nchw_wgrad(
    spec: DirectNchwWgradSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR of one direct-style native-NCHW wgrad prototype kernel."""
    ok, why = is_valid_direct_nchw_wgrad_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectNchwWgradSpec for {arch}: {why}")

    p = spec.problem
    dtype = p.dtype
    io = BF16 if dtype == "bf16" else F16
    EB = 2
    C, K, H, W = p.total_c, p.total_k, p.H, p.W
    Ho, Wo, S, P = p.Ho, p.Wo, p.stride, p.PAD
    KW = p.KW
    T = p.KH * KW
    HW, HoWo = H * W, Ho * Wo
    YG = spec.rows_per_group
    TM, TC, TH, TW = spec.tile_m, spec.tile_c, spec.tile_h, spec.tile_w
    G = spec.stage_group
    THREADS = spec.threads
    MT_W, CT_W, TPW = spec.mt_w, spec.ct_w, spec.taps_per_wave
    # LDS_W is the window's row stride in positions (lds_w + lds_wpad).
    LDS_H, LDS_W, CS = spec.lds_h, spec.lds_w_stride, spec.c_stride
    DYS = spec.dy_stride
    HALO_L, HALO_R = spec.halo_l, spec.halo_r
    KSTEPS = spec.k_steps
    NHT, NWT = spec.n_tiles_hw()
    N_TILES = spec.n_tiles

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, 8)

    dY = b.param("dY", PtrType(io, "global"), noalias=True, readonly=True, align=16)
    X = b.param("X", PtrType(io, "global"), noalias=True, readonly=True, align=16)
    dW = b.param("dW", PtrType(io, "global"), noalias=True, align=4)
    dY_bytes = b.param("dY_bytes", I32)
    X_bytes = b.param("X_bytes", I32)
    split = b.param("split", I32)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1 = ci(0), ci(1)
    oob = ci(_OOB_BASE)
    dy_rsrc = b.buffer_rsrc(dY, dY_bytes)
    x_rsrc = b.buffer_rsrc(X, X_bytes)

    NBUF = spec.lds_buffers
    XB, DB = spec.x_buf_stride, spec.dy_buf_stride
    x_lds = b.smem_alloc(io, [1, max(NBUF * XB, spec.epi_elems)], name_hint="lds_x")
    dy_lds = b.smem_alloc(io, [1, NBUF * DB], name_hint="lds_dy")
    x_dump = ci(spec.lds_x_elems)
    dy_dump = ci(spec.lds_dy_elems)

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave_id = b.readfirstlane(b.div(tid, ci(_WAVE)))
    wave_m = b.mod(wave_id, ci(spec.waves_m))
    rest_w = b.div(wave_id, ci(spec.waves_m))
    wave_c = b.mod(rest_w, ci(spec.waves_c))
    wave_t = b.div(rest_w, ci(spec.waves_c))

    if spec.xcd_remap:
        # 1-D grid. Dispatch deals block ids round-robin over the XCDs;
        # renumber so each XCD gets a contiguous run of logical ids, i.e. the
        # (c, y-group, k) tiles of the same spatial slice, which then share
        # that XCD's L2 for the X window and the dY tile.
        n_xy = spec.grid(1)[0] * spec.grid(1)[1]
        lin = b.block_id_x()
        n_blocks = b.mul(split, ci(n_xy))
        xcd = b.mod(lin, ci(_XCDS))
        pos = b.div(lin, ci(_XCDS))
        per_x = b.div(n_blocks, ci(_XCDS))
        rem = b.mod(n_blocks, ci(_XCDS))
        lin = b.add(b.add(b.mul(xcd, per_x), b.smin(xcd, rem)), pos)
        bx = b.mod(lin, ci(spec.grid(1)[0]))
        rest = b.div(lin, ci(spec.grid(1)[0]))
        by = b.mod(rest, ci(spec.grid(1)[1]))
        z = b.div(rest, ci(spec.grid(1)[1]))
    else:
        bx, by, z = b.block_id_x(), b.block_id_y(), b.block_id_z()
    if spec.y_groups > 1:
        y_grp = b.mod(bx, ci(spec.y_groups))
        c_tile = b.div(bx, ci(spec.y_groups))
        y0 = b.mul(y_grp, ci(YG))
    else:
        c_tile = bx
        y0 = c0
    m_base = b.mul(by, ci(TM))
    c_base = b.mul(c_tile, ci(TC))

    # ---- spatial-tile slice of this split ----------------------------------
    per = b.div(b.add(ci(N_TILES), b.sub(split, c1)), split)
    t_lo = b.mul(z, per)
    t_hi = b.smin(b.add(t_lo, per), ci(N_TILES))

    # ---- staging units ------------------------------------------------------
    def passes(n_units: int):
        out = []
        for j in range(ceil_div(n_units, THREADS)):
            u = b.add(tid, ci(j * THREADS)) if j else tid
            in_tile = b.cmp_lt(u, ci(n_units)) if n_units % THREADS else None
            out.append((u, in_tile))
        return out

    # dY units (k, row, v), v fastest: one dwordx4 + one ds_write_b128.
    NVY = TW // _RUN
    dy_units = []
    for u, in_tile in passes(TM * TH * NVY):
        v = b.mod(u, ci(NVY))
        r2 = b.div(u, ci(NVY))
        row = b.mod(r2, ci(TH))
        k = b.div(r2, ci(TH))
        src = b.add(b.mul(b.add(m_base, k), ci(HoWo)), b.mul(v, ci(_RUN)))
        dst = b.add(b.mul(k, ci(DYS)), b.add(b.mul(row, ci(TW)), b.mul(v, ci(_RUN))))
        if in_tile is not None:
            dst = b.select(in_tile, dst, dy_dump)
        dy_units.append((src, row, v, dst, in_tile))

    # X body units (cg, wr, v), v fastest: G dwordx4 (one per channel), then
    # 8 G-wide LDS vectors at [wr][HALO_L + 8v + e][cg*G].
    NVX = S * TW // _RUN
    NCG = TC // G
    x_body = []
    for u, in_tile in passes(NCG * LDS_H * NVX):
        v = b.mod(u, ci(NVX))
        r2 = b.div(u, ci(NVX))
        cg = b.mod(r2, ci(NCG))
        wr = b.div(r2, ci(NCG))
        src = b.add(b.mul(b.add(c_base, b.mul(cg, ci(G))), ci(HW)), b.mul(v, ci(_RUN)))
        dst0 = b.add(
            b.mul(
                b.add(b.mul(wr, ci(LDS_W)), b.add(b.mul(v, ci(_RUN)), ci(HALO_L))),
                ci(CS),
            ),
            b.mul(cg, ci(G)),
        )
        if in_tile is not None:
            dst0 = b.select(in_tile, dst0, x_dump)
            step = b.select(in_tile, ci(CS), c0)
        else:
            step = None
        dsts = []
        for e in range(_RUN):
            if e == 0:
                dsts.append(dst0)
            elif step is None:
                dsts.append(b.add(dst0, ci(e * CS)))
            else:
                dsts.append(b.add(dst0, b.mul(step, ci(e))))
        x_body.append((src, wr, v, list(range(_RUN)), dsts, _RUN, in_tile))

    # X halo units (cg, wr) per side: G aligned hv-wide loads, the h columns
    # next to the body are written.
    x_halo = []
    for side, h in (("l", HALO_L), ("r", HALO_R)):
        if not h:
            continue
        hv = spec.halo_vec(h)
        for u, in_tile in passes(NCG * LDS_H):
            cg = b.mod(u, ci(NCG))
            wr = b.div(u, ci(NCG))
            src = b.mul(b.add(c_base, b.mul(cg, ci(G))), ci(HW))
            if side == "l":
                elems = list(range(hv - h, hv))
                col0 = 0  # LDS columns [0, h)
                wofs = -hv
            else:
                elems = list(range(h))
                col0 = HALO_L + S * TW
                wofs = S * TW
            base = b.add(
                b.mul(b.add(b.mul(wr, ci(LDS_W)), ci(col0)), ci(CS)), b.mul(cg, ci(G))
            )
            dsts = []
            for i, _e in enumerate(elems):
                d = b.add(base, ci(i * CS)) if i else base
                if in_tile is not None:
                    d = b.select(in_tile, d, x_dump)
                dsts.append(d)
            x_halo.append((src, wr, (side, wofs), elems, dsts, hv, in_tile))

    def tile_origin(t: Value):
        tw_i = b.mod(t, ci(NWT))
        r2 = b.div(t, ci(NWT))
        th_i = b.mod(r2, ci(NHT))
        n = b.div(r2, ci(NHT))
        return n, b.mul(th_i, ci(TH)), b.mul(tw_i, ci(TW))

    def issue_loads(t: Value):
        n, ho0, wo0 = tile_origin(t)
        live = b.cmp_lt(t, t_hi)
        # dY
        dy_img = b.add(b.mul(n, ci(K * HoWo)), b.add(b.mul(ho0, ci(Wo)), wo0))
        dys = []
        for src, row, v, _dst, in_tile in dy_units:
            ho = b.add(ho0, row)
            wo = b.add(wo0, b.mul(v, ci(_RUN)))
            ok = b.land(live, b.land(b.cmp_lt(ho, ci(Ho)), b.cmp_lt(wo, ci(Wo))))
            if in_tile is not None:
                ok = b.land(ok, in_tile)
            e = b.add(dy_img, b.add(src, b.mul(row, ci(Wo))))
            off = b.select(ok, b.mul(e, ci(EB)), oob)
            dys.append(b.buffer_load_vN(dy_rsrc, off, c0, io, _RUN))
        # X window: rows h = S*ho0 - P + y0 + wr, body columns from S*wo0.
        h0 = b.add(b.sub(b.mul(ho0, ci(S)), ci(P)), y0)
        w0 = b.mul(wo0, ci(S))
        x_img = b.add(b.mul(n, ci(C * HW)), w0)
        xs = []
        for src, wr, kind, _elems, _dsts, width, in_tile in x_body + x_halo:
            h = b.add(h0, wr)
            ok = live
            if "rows_unmasked" not in _MUTATIONS:
                ok = b.land(ok, b.land(b.cmp_ge(h, c0), b.cmp_lt(h, ci(H))))
            if not isinstance(kind, tuple):  # body unit: kind is its run index v
                col = b.mul(kind, ci(_RUN))
                w = b.add(w0, col)
                ok = b.land(ok, b.cmp_lt(w, ci(W)))
                e = b.add(b.add(x_img, src), b.mul(h, ci(W)))
            else:
                side, wofs = kind
                w = b.add(w0, ci(wofs))
                if "halo_unmasked" not in _MUTATIONS:
                    wok = b.cmp_ge(w, c0) if side == "l" else b.cmp_lt(w, ci(W))
                    ok = b.land(ok, wok)
                e = b.add(b.add(b.add(x_img, src), b.mul(h, ci(W))), ci(wofs))
            if in_tile is not None:
                ok = b.land(ok, in_tile)
            if "no_loads" in _MUTATIONS:  # profiling only: every load reads zero
                ok = b.cmp_lt(t_hi, c0)
            off = b.select(ok, b.mul(e, ci(EB)), oob)
            for g in range(G):
                xs.append(b.buffer_load_vN(x_rsrc, off, ci(g * HW * EB), io, width))
        return dys, xs

    def commit(dys, xs, yo=None, xo=None) -> None:
        """Stage one tile's loads; ``yo`` / ``xo``: runtime buffer offsets."""
        for (_s, _r, _v, dst, _it), val in zip(dy_units, dys):
            d = dst if yo is None else b.add(dst, yo)
            b.smem_store_vN(dy_lds, [c0, d], val, _RUN)
        i = 0
        for _s, _wr, _k, elems, dsts, _w, _it in x_body + x_halo:
            runs = xs[i : i + G]
            i += G
            for e, d in zip(elems, dsts):
                vec = b.vec_pack([b.vec_extract(r, e) for r in runs], io)
                d = d if xo is None else b.add(d, xo)
                b.smem_store_vN(x_lds, [c0, d], vec, G)

    # ---- MFMA fragment addresses ---------------------------------------------
    lane_lo = b.mod(lane, ci(_ATOM))
    lane_g = b.div(lane, ci(_ATOM))  # 8-position group of the fragment
    lane_q = b.mod(b.div(lane, ci(4)), ci(4))
    lane_c4 = b.mod(lane, ci(4))
    # Lane group g = lane / 16 holds the 8 positions starting at tile
    # (row, col) = group_pos(ks, g): a build-time part from the k step plus
    # a lane part from g (see :func:`kstep_group_pos`).
    if spec.kmap == "2row":
        rowg = b.mul(b.mod(lane_g, ci(2)), ci(TH // 2))
        colg = b.mul(b.div(lane_g, ci(2)), ci(_RUN))
    elif TW >= _AK:
        rowg, colg = c0, b.mul(lane_g, ci(_RUN))
    else:
        rowg = b.div(b.mul(lane_g, ci(_RUN)), ci(TW))
        colg = b.mod(b.mul(lane_g, ci(_RUN)), ci(TW))
    # A (dY): [k][row*TW + col], k = lane % 16, the group's 8 positions.
    a_base = b.add(
        b.mul(b.add(b.mul(wave_m, ci(MT_W * _ATOM)), lane_lo), ci(DYS)),
        b.add(b.mul(rowg, ci(TW)), colg),
    )
    # B (X): lane reads position (group start + rd*4 + q), channels c4*4 .. +3.
    tap0 = b.mul(wave_t, ci(TPW))
    y_w = b.div(tap0, ci(KW))
    x_w = b.mod(tap0, ci(KW))
    b_base = b.add(
        b.add(
            b.mul(
                b.add(
                    b.mul(b.add(b.mul(rowg, ci(S)), y_w), ci(LDS_W)),
                    b.add(b.mul(b.add(colg, lane_q), ci(S)), x_w),
                ),
                ci(CS),
            ),
            b.mul(lane_c4, ci(4)),
        ),
        b.mul(wave_c, ci(CT_W * _ATOM)),
    )

    def ks_pos(ks: int) -> Tuple[int, int]:
        return kstep_origin(spec, ks)

    mfma = getattr(b, f"mfma_f32_16x16x32_{'bf16' if dtype == 'bf16' else 'f16'}")

    def acc_i(mt: int, t: int, ct: int) -> int:
        return (mt * TPW + t) * CT_W + ct

    def _trunc(v: Value) -> Value:
        return b.trunc_f32_to_bf16(v) if dtype == "bf16" else b.trunc_f32_to_f16(v)

    if "no_b_reads" in _MUTATIONS:
        # A lane-varying stand-in operand so the MFMAs cannot be folded.
        lane_h = _trunc(b.sitofp_f32(lane))
        lane_vec = b.vec_pack([lane_h] * _RUN, io)

    # Fragment-read bases of the buffer the MFMAs of this tile read.
    cur = {"a": a_base, "b": None}

    def read_a(ks: int, mt: int) -> Value:
        rk, ck = ks_pos(ks)
        off = mt * _ATOM * DYS + rk * TW + ck
        return b.smem_load_vN(dy_lds, c0, b.add(cur["a"], ci(off)), dtype=io, n=_RUN)

    def read_b(wrow: int, ck: int, xx: int, ct: int) -> Value:
        if "no_b_reads" in _MUTATIONS:  # profiling only
            return lane_vec
        frs = []
        for rd in range(2):
            off = (wrow * LDS_W + (ck + rd * 4) * S + xx) * CS + ct * _ATOM
            frs.append(
                b.ds_read_tr16_b64(x_lds, c0, b.add(cur["b"], ci(off)), dtype=io)
            )
        return b.vec_concat(frs[0], frs[1])

    def emit_mfmas(accs: List[Value]) -> None:
        if "no_mfma" in _MUTATIONS:  # profiling only
            return
        if spec.mfma_order == "ks":
            for ks in range(KSTEPS):
                a_fr = [read_a(ks, mt) for mt in range(MT_W)]
                rk, ck = ks_pos(ks)
                for t in range(TPW):
                    for ct in range(CT_W):
                        b_fr = read_b(rk * S + t // KW, ck, t % KW, ct)
                        for mt in range(MT_W):
                            i = acc_i(mt, t, ct)
                            accs[i] = mfma(a_fr[mt], b_fr, accs[i])
            return
        # "row": one X fragment per (window row, column phase, tap column,
        # c atom), shared by every (k step, filter row) pair that lands on it
        # -- at stride 1 a window row serves up to KH k steps.
        a_fr = {}
        uses = {}
        for ks in range(KSTEPS):
            rk, ck = ks_pos(ks)
            for t in range(TPW):
                uses.setdefault((rk * S + t // KW, ck, t % KW), []).append((ks, t))
        for key in sorted(uses):
            wrow, ck, xx = key
            for ct in range(CT_W):
                b_fr = read_b(wrow, ck, xx, ct)
                for ks, t in uses[key]:
                    for mt in range(MT_W):
                        if (ks, mt) not in a_fr:
                            a_fr[(ks, mt)] = read_a(ks, mt)
                        i = acc_i(mt, t, ct)
                        accs[i] = mfma(a_fr[(ks, mt)], b_fr, accs[i])

    n_acc = MT_W * TPW * CT_W
    iter_args = [(f"acc{i}", b.zero_vec_f32(4)) for i in range(n_acc)]
    if spec.prefetch:
        # Register prefetch: the next tile's global loads are in flight
        # under this tile's MFMAs (two barriers per tile).
        dy0, xs0 = issue_loads(t_lo)
        n_dy = len(dy0)
        iter_args += [(f"y{i}", v) for i, v in enumerate(dy0)]
        iter_args += [(f"x{i}", v) for i, v in enumerate(xs0)]
    cur["b"] = b_base
    if NBUF == 2:
        # Prologue: stage tile t_lo into buffer 0, loads of t_lo + 1 in flight.
        commit(dy0, xs0)
        dy1, xs1 = issue_loads(b.add(t_lo, c1))
        iter_args = iter_args[:n_acc]
        iter_args += [(f"y{i}", v) for i, v in enumerate(dy1)]
        iter_args += [(f"x{i}", v) for i, v in enumerate(xs1)]
        b.sync(fence="workgroup")
    loop = b.scf_for_iter(
        t_lo, t_hi, c1, iter_args, iv_name="tile", elide_trailing_barrier=False
    )
    with loop as (tv, carried):
        accs = list(carried[:n_acc])
        if NBUF == 2:
            dys = list(carried[n_acc : n_acc + n_dy])
            xs = list(carried[n_acc + n_dy :])
            par = b.land(b.sub(tv, t_lo), c1)
            nxt = b.sub(c1, par)
            # Tile tv + 1 into the buffer the previous tile's MFMAs read
            # (released by the barrier that ended that tile).
            commit(dys, xs, yo=b.mul(nxt, ci(DB)), xo=b.mul(nxt, ci(XB)))
            dyn, xsn = issue_loads(b.add(tv, ci(2)))
            b.sched_barrier(0)
            cur["a"] = b.add(a_base, b.mul(par, ci(DB)))
            cur["b"] = b.add(b_base, b.mul(par, ci(XB)))
            emit_mfmas(accs)
            # LDS-only barrier: a workgroup fence would also wait for the
            # tile tv + 2 loads (vmcnt(0)) and serialise the prefetch.
            b.s_waitcnt(lgkmcnt=0)
            b.s_barrier_bare()
            b.scf_yield(*accs, *dyn, *xsn)
        else:
            if spec.prefetch:
                dys = list(carried[n_acc : n_acc + n_dy])
                xs = list(carried[n_acc + n_dy :])
            else:
                # No prefetch registers: the loads of this tile are waited
                # for here; other workgroups on the CU cover the latency.
                dys, xs = issue_loads(tv)
            b.sync(fence="workgroup")
            commit(dys, xs)
            b.sync(fence="workgroup")
            if spec.prefetch:
                dyn, xsn = issue_loads(b.add(tv, c1))
                # Pin the loads ahead of the MFMAs: left alone the scheduler
                # sinks them past most of the MFMA stream and the prefetch
                # hides nothing.
                b.sched_barrier(0)
                emit_mfmas(accs)
                b.scf_yield(*accs, *dyn, *xsn)
            else:
                emit_mfmas(accs)
                b.scf_yield(*accs)
    accs_out = list(loop.results[:n_acc])

    # ---- epilogue: packed 16-bit atomics into KCYX dW -------------------------
    # Lane holds, per (mt, ct) and slot i, dW[k, c, tap] for the wave's taps
    # [t0, t0 + TPW): k = lane_g*4 + i (+ tile), c = lane % 16 (+ tile). The
    # KCYX flat index k*C*T + c*T + tap; the pair alignment is the parity of
    # c*T + t0, picked per lane between two build-time pairings.
    t0 = b.add(b.mul(y0, ci(KW)), tap0)

    def _pairs(par: int) -> List[int]:
        return [s for s in range(-1, TPW) if (par + s) % 2 == 0]

    p0, p1 = _pairs(0), _pairs(1)
    n_pairs = max(len(p0), len(p1))
    pad0 = [False] * len(p0) + [True] * (n_pairs - len(p0))
    pad1 = [False] * len(p1) + [True] * (n_pairs - len(p1))
    p0 = p0 + [p0[0]] * (n_pairs - len(p0))
    p1 = p1 + [p1[0]] * (n_pairs - len(p1))

    zero16 = b.vec_extract(b.zero_vec(io, 2), 0)
    has_work = b.cmp_lt(t_lo, t_hi)
    if "no_epilogue" in _MUTATIONS:  # profiling only: drop the dW writes
        has_work = b.land(has_work, b.cmp_lt(split, c0))
    if not spec.atomic:
        dw_rsrc = b.buffer_rsrc(dW, ci(EB * K * C * T))

    def _atomic(idx: Value, pk: Value) -> None:
        if not spec.atomic:
            b.buffer_store_vN(dw_rsrc, b.mul(idx, ci(EB)), c0, pk)
            return
        if dtype == "bf16":
            b.global_atomic_add_pk_bf16(dW, idx, pk)
        else:
            b.global_atomic_add_pk_f16(dW, idx, pk)

    with b.scf_if(has_work):
        m_wave = b.add(m_base, b.mul(wave_m, ci(MT_W * _ATOM)))
        if spec.epilogue == "lds":
            # Per (mt, ct) slab: rows wave_m*16 + r (k), columns wave_c*16 + l
            # (c), all WG_T taps; then one packed atomic per contiguous dword
            # of each (row, wave_c) run of 16 c x WG_T taps in KCYX.
            WG_T = spec.wg_taps
            WC16 = spec.waves_c * _ATOM
            RUN_D = _ATOM * WG_T // 2  # dwords per (row, wave_c) run
            n_dw = spec.waves_m * _ATOM * spec.waves_c * RUN_D
            row0 = b.add(b.mul(wave_m, ci(_ATOM)), b.mul(lane_g, ci(4)))
            col = b.add(b.mul(wave_c, ci(_ATOM)), lane_lo)
            for mt in range(MT_W):
                for ct in range(CT_W):
                    b.sync(fence="workgroup")
                    for i in range(4):
                        base_i = b.add(
                            b.mul(
                                b.add(b.mul(b.add(row0, ci(i)), ci(WC16)), col),
                                ci(WG_T),
                            ),
                            tap0,
                        )
                        for t in range(TPW):
                            v = _trunc(b.vec_extract(accs_out[acc_i(mt, t, ct)], i))
                            b.smem_store_vN(x_lds, [c0, b.add(base_i, ci(t))], v, 1)
                    b.sync(fence="workgroup")
                    for j in range(ceil_div(n_dw, THREADS)):
                        d = b.add(tid, ci(j * THREADS)) if j else tid
                        seg = b.div(d, ci(RUN_D))
                        off = b.mod(d, ci(RUN_D))
                        row = b.div(seg, ci(spec.waves_c))
                        wcb = b.mod(seg, ci(spec.waves_c))
                        lds_i = b.add(
                            b.mul(
                                b.add(b.mul(row, ci(WC16)), b.mul(wcb, ci(_ATOM))),
                                ci(WG_T),
                            ),
                            b.mul(off, ci(2)),
                        )
                        k = b.add(
                            b.add(m_base, ci(mt * _ATOM)),
                            b.add(
                                b.mul(b.div(row, ci(_ATOM)), ci(MT_W * _ATOM)),
                                b.mod(row, ci(_ATOM)),
                            ),
                        )
                        cg = b.add(
                            b.add(c_base, ci(ct * _ATOM)), b.mul(wcb, ci(CT_W * _ATOM))
                        )
                        idx = b.add(
                            b.add(b.mul(k, ci(C * T)), b.mul(cg, ci(T))),
                            b.mul(off, ci(2)),
                        )
                        if n_dw % THREADS:
                            with b.scf_if(b.cmp_lt(d, ci(n_dw))):
                                _atomic(
                                    idx, b.smem_load_vN(x_lds, c0, lds_i, dtype=io, n=2)
                                )
                        else:
                            _atomic(
                                idx, b.smem_load_vN(x_lds, c0, lds_i, dtype=io, n=2)
                            )
        else:
            for ct in range(CT_W):
                c = b.add(
                    b.add(c_base, b.mul(wave_c, ci(CT_W * _ATOM))),
                    b.add(ci(ct * _ATOM), lane_lo),
                )
                ctap = b.add(b.mul(c, ci(T)), t0)
                odd = b.cmp_ne(b.mod(ctap, ci(2)), c0)
                for mt in range(MT_W):
                    for i in range(4):
                        k = b.add(
                            b.add(m_wave, ci(mt * _ATOM + i)), b.mul(lane_g, ci(4))
                        )
                        flat0 = b.add(b.mul(k, ci(C * T)), ctap)
                        vals = [
                            _trunc(b.vec_extract(accs_out[acc_i(mt, t, ct)], i))
                            for t in range(TPW)
                        ]

                        def _v(t: int) -> Value:
                            return vals[t] if 0 <= t < TPW else zero16

                        for q in range(n_pairs):
                            s0, s1 = p0[q], p1[q]
                            lo = b.select(
                                odd,
                                zero16 if pad1[q] else _v(s1),
                                zero16 if pad0[q] else _v(s0),
                            )
                            hi = b.select(
                                odd,
                                zero16 if pad1[q] else _v(s1 + 1),
                                zero16 if pad0[q] else _v(s0 + 1),
                            )
                            idx = b.select(
                                odd, b.add(flat0, ci(s1)), b.add(flat0, ci(s0))
                            )
                            _atomic(idx, b.vec_pack([lo, hi], io))
    return b.kernel


__all__ = [
    "DirectNchwWgradSpec",
    "conflict_free_pads",
    "kstep_group_pos",
    "kstep_origin",
    "x_read_conflict_ways",
    "build_direct_conv_nchw_wgrad",
    "is_valid_direct_nchw_wgrad_spec",
    "direct_nchw_wgrad_launch_values",
    "direct_nchw_wgrad_signature",
]
