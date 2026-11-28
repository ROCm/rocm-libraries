# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native-NCHW backward-data (dgrad) direct convolution (gfx950, stride 1/2).

A productionized AOT-buildable kernel family: a C++ mirror lives at
``library/cpp/instances/common/conv_direct_nchw_dgrad.cpp`` and is gated by
``tools/check_byte_identity.py`` like every other family, with a ``.py``/``.c``
parity pair under ``library/tests/parity/``. Runtime dispatch selection is
**deliberately not yet wired**: this family is not registered in
``library/dispatch/grouped_convolution.py`` or its instance JSONs, so
``algorithm="auto"`` never reaches it. Callers build and launch it explicitly
(directly via :func:`build_direct_conv_nchw_dgrad`, or through the AOT sweep
in ``kernels/conv/direct_variants.py`` / ``kernels/conv/direct_aot_jobs.py``
with variant ``"direct_nchw_dgrad"``), not via the request-level dispatcher.

It consumes the output gradient ``dY`` in NKHW and produces the input
gradient ``dX`` in NCHW directly -- no host transpose of the activations
(transpose ``dY`` NKHW -> NHWK, production NHWC implicit-GEMM dgrad,
transpose ``dX`` NHWC -> NCHW is the pipeline an NCHW caller has without this
family). ``groups == 1``, fp16 / bf16 storage, dilation 1, stride 1 or 2.

Algorithm: stride-phase direct dgrad
------------------------------------
``dX[n, c, h, w] = sum_{k, r, s} dY[n, k, ho, wo] * W[k, c, r, s]`` with
``h = ho*S - PAD + r``. Split the ``dX`` grid into its ``S x S`` stride phases
``h = S*i + ph``, ``w = S*j + pw``. For one phase only the taps with
``(ph + PAD - r) % S == 0`` contribute, and they read
``ho = i + d_r`` with the build-time constant ``d_r = (ph + PAD - r) // S``
(same for ``w``). Every phase is therefore a small dense stride-1 correlation
of ``dY`` over the phase grid ``(i, j)`` -- no zero-insertion, no wasted MACs.
At stride 1 there is one phase and this *is* the forward conv of ``dY`` with
the transposed weights (pad ``KH - 1 - PAD``).

One workgroup owns a ``tile_i x tile_j`` block of the phase grid, *all*
``S*S`` phases of it (a ``S*tile_i x S*tile_j`` block of ``dX``) and
``tile_c`` input channels. The ``dY`` window all phases read is staged once
per ``ck``-deep chunk of the reduction channels ``k``:

* **dY staging** (NKHW source): the window is ``tile_i + span_h`` rows of
  ``tile_j + span_w`` columns. ``tile_j`` is a multiple of 8 and the
  workgroup's first column ``j0`` is too, so the window body is whole aligned
  8-element runs along ``wo`` (one ``buffer_load_dwordx4`` each, always
  all-in or all-out of the image -- no partial vector, no scalar fallback),
  and the ``|d|``-wide halo columns on either side come from one narrow
  aligned load per row (a dword at stride 1/2 with a 3x3 filter). The LDS
  tile is ``[row][col][k]`` (``k`` contiguous, padded), which the MFMA reads
  with plain ``ds_read_b128`` at any tap offset -- no ``ds_read_tr`` alignment
  copies. Moving ``k`` innermost is a transpose; ``stage_group`` (``G``)
  picks how: each staging thread loads ``G`` channels' runs and writes
  ``G``-channel vectors per position (``G=8``: an 8x8 register transpose and
  ``ds_write_b128``; ``G=1``: the per-element ``ds_write_b16`` scatter the
  forward prototype's ``pos_c`` layout uses).
* **W staging**: weights repacked once at plan time to ``[C][KH][KW][K]``
  (``k`` contiguous). Each MFMA fragment is one contiguous vector load into
  the forward kernel's fragment-slot LDS layout.
* **MFMA**: operands swapped (``dY`` fragment is the A operand), so the
  accumulator rows are phase-grid positions and each lane holds 4
  consecutive ``j`` of one ``c``.
* **Epilogue** (NCHW destination, no LDS): stride 1 stores the 4 values as
  one ``dwordx2``. Stride 2 holds the ``pw = 0`` and ``pw = 1`` accumulators
  of the same 4 ``j`` in the same lane, so it interleaves them in registers
  into 8 consecutive ``w`` and stores one ``dwordx4`` -- the stride-2 scatter
  along ``w`` (NCHW's vector axis) never reaches memory. Masked per-element
  stores only on tiles that cross the image edge.

Runtime precondition (checked by :func:`is_valid_direct_nchw_dgrad_spec` on the
spec's problem, *not* by the binary): ``Wo % 8 == 0`` so every 8-wide ``dY``
run is entirely inside or outside a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rocke.core.ir import IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _buf_load_vN,
    _buf_store_vN,
    _io_type,
    _trunc_f32,
    emit_direct_params,
)

_DTYPES = ("fp16", "bf16")
_OOB_BASE = 0x7F000000
_I32_MAX = (1 << 31) - 1
_ALIGN = 8  # elements in 16 bytes of a 16-bit dtype

# Atom name -> (tile, K, frag) -- same table as the forward prototype.
_ATOMS = {
    "32x32x16": (32, 16, 8),
    "32x32x8": (32, 8, 4),
    "16x16x32": (16, 32, 8),
    "16x16x16": (16, 16, 4),
}


def phase_taps(n_taps: int, pad: int, stride: int) -> List[List[Tuple[int, int]]]:
    """Per stride phase ``p``, the ``(tap, d)`` pairs that reach it.

    Output coordinate ``S*i + p`` receives tap ``t`` from input coordinate
    ``i + d`` with ``d = (p + pad - t) / S`` whenever that division is exact.
    """
    out = []
    for ph in range(stride):
        out.append(
            [
                (t, (ph + pad - t) // stride)
                for t in range(n_taps)
                if (ph + pad - t) % stride == 0
            ]
        )
    return out


def repack_weights_cyxk(W_kcyx):
    """Plan-time repack of torch KCYX weights into this kernel's ``[C][KH][KW][K]``."""
    return W_kcyx.permute(1, 2, 3, 0).contiguous()


@dataclass(frozen=True)
class DirectNchwDgradSpec:
    """One native-NCHW dgrad prototype configuration.

    ``problem`` is the *forward* problem (``H``/``W`` are ``dX``'s extents,
    ``Ho``/``Wo`` are ``dY``'s; ``cpg`` = C, ``kpg`` = K). Tile sizes are in
    phase-grid units: a workgroup writes ``stride*tile_i`` rows and
    ``stride*tile_j`` columns of ``dX``.
    """

    problem: DirectConvProblem
    name: str = "conv_direct_nchw_dgrad"

    tile_i: int = 8
    tile_j: int = 32
    tile_c: int = 64
    ck: int = 32

    waves_m: int = 2  # split tile_c
    waves_n: int = 2  # split tile_i

    atom: str = "32x32x16"
    wave_size: int = 64
    lds_pad: int = 8
    # Channels per dY staging unit: G runs are loaded and written as G-wide
    # LDS vectors per position (8: register transpose + ds_write_b128;
    # 1: per-element ds_write_b16 scatter).
    stage_group: int = 8

    # ---- derived ----------------------------------------------------------

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def taps_h(self):
        return phase_taps(self.problem.KH, self.problem.PAD, self.S)

    @property
    def taps_w(self):
        return phase_taps(self.problem.KW, self.problem.PAD_W, self.S)

    def _span(self, taps) -> Tuple[int, int]:
        ds = [d for ph in taps for (_, d) in ph]
        if not ds:
            return 0, 0
        return min(ds), max(ds)

    @property
    def d_h(self) -> Tuple[int, int]:
        return self._span(self.taps_h)

    @property
    def d_w(self) -> Tuple[int, int]:
        return self._span(self.taps_w)

    @property
    def atom_tile(self) -> int:
        return _ATOMS[self.atom][0]

    @property
    def atom_k(self) -> int:
        return _ATOMS[self.atom][1]

    @property
    def frag(self) -> int:
        return _ATOMS[self.atom][2]

    @property
    def acc_per_lane(self) -> int:
        return self.atom_tile * self.atom_tile // self.wave_size

    @property
    def n_col_blocks(self) -> int:
        return self.tile_j // self.atom_tile

    @property
    def m_tiles_total(self) -> int:
        return self.tile_c // self.atom_tile

    @property
    def m_tiles_per_wave(self) -> int:
        return self.m_tiles_total // self.waves_m

    @property
    def rows_per_wave(self) -> int:
        return self.tile_i // self.waves_n

    @property
    def k_atoms(self) -> int:
        return self.ck // self.atom_k

    @property
    def threads_per_block(self) -> int:
        return self.waves_m * self.waves_n * self.wave_size

    @property
    def lds_h(self) -> int:
        lo, hi = self.d_h
        return self.tile_i + hi - lo

    @property
    def lds_w(self) -> int:
        lo, hi = self.d_w
        return self.tile_j + hi - lo

    @property
    def halo_l(self) -> int:
        return max(0, -self.d_w[0])

    @property
    def halo_r(self) -> int:
        return max(0, self.d_w[1])

    @staticmethod
    def _halo_vec(h: int) -> int:
        v = 2  # at least one dword
        while v < h:
            v *= 2
        return v

    @property
    def c_stride(self) -> int:
        return self.ck + self.lds_pad

    @property
    def lds_x_elems(self) -> int:
        return self.lds_h * self.lds_w * self.c_stride

    @property
    def lds_w_elems(self) -> int:
        p = self.problem
        return self.tile_c * p.KH * p.KW * self.ck

    @property
    def lds_bytes(self) -> int:
        # + one dump slot per tile for a partial last staging pass.
        return 2 * (self.lds_x_elems + _ALIGN + self.lds_w_elems + self.frag)

    @property
    def n_phases(self) -> int:
        return self.S * self.S

    @property
    def acc_vgprs(self) -> int:
        return (
            self.n_phases
            * self.rows_per_wave
            * self.n_col_blocks
            * self.m_tiles_per_wave
            * self.acc_per_lane
        )

    def phase_grid(self) -> Tuple[int, int]:
        p = self.problem
        return ceil_div(p.H, self.S), ceil_div(p.W, self.S)

    def grid(self) -> Tuple[int, int, int]:
        hp, wp = self.phase_grid()
        p = self.problem
        n = (
            ceil_div(wp, self.tile_j)
            * ceil_div(hp, self.tile_i)
            * p.N
            * ceil_div(p.total_c, self.tile_c)
        )
        return (n, 1, 1)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}s{p.stride}p{p.PAD}",
            f"t{self.tile_i}x{self.tile_j}x{self.tile_c}",
            f"ck{self.ck}",
            f"w{self.waves_m}x{self.waves_n}",
            f"a{self.atom}",
            f"g{self.stage_group}",
            f"lp{self.lds_pad}",
            "bf16" if p.dtype == "bf16" else "",
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype {p.dtype!r} (16-bit storage only)")
        if p.groups != 1:
            raise ValueError("groups==1 only")
        if p.is_dilated:
            raise ValueError("dilation is not supported")
        if p.stride not in (1, 2):
            raise ValueError(f"stride {p.stride} not supported (1 or 2)")
        if self.atom not in _ATOMS:
            raise ValueError(f"unknown atom {self.atom!r}")
        t = self.atom_tile
        if self.tile_j % t or self.tile_j % _ALIGN:
            raise ValueError(f"tile_j must be a multiple of {t} and {_ALIGN}")
        if self.tile_c % t:
            raise ValueError(f"tile_c must be a multiple of {t}")
        if self.m_tiles_total % self.waves_m:
            raise ValueError("tile_c / atom not divisible by waves_m")
        if self.tile_i % self.waves_n:
            raise ValueError("tile_i not divisible by waves_n")
        if self.ck % self.atom_k or self.ck % _ALIGN:
            raise ValueError(f"ck must be a multiple of {self.atom_k} and {_ALIGN}")
        if self.stage_group not in (1, 2, 4, 8) or self.ck % self.stage_group:
            raise ValueError("stage_group must be 1/2/4/8 and divide ck")
        if self.lds_pad < 0 or self.lds_pad % _ALIGN:
            raise ValueError(f"lds_pad must be a non-negative multiple of {_ALIGN}")
        if p.kpg % self.ck:
            raise ValueError(f"K {p.kpg} not divisible by ck {self.ck}")
        if p.Wo % _ALIGN:
            raise ValueError(
                f"Wo {p.Wo} must be a multiple of {_ALIGN} (whole dY runs)"
            )
        if not (self.d_w[0] <= 0 <= self.d_w[1]):
            raise ValueError(f"column tap offsets {self.d_w} must straddle 0")
        if max(self.halo_l, self.halo_r) > _ALIGN:
            raise ValueError("halo wider than 8 columns")
        if self.threads_per_block > 1024:
            raise ValueError("too many threads")
        if self.acc_vgprs > 256:
            raise ValueError(f"accumulators need {self.acc_vgprs} registers per lane")
        eb = 2
        big = max(
            eb * p.N * p.total_k * p.Ho * p.Wo,
            eb * p.total_c * p.total_k * p.KH * p.KW,
            eb * p.N * p.total_c * p.H * p.W,
        )
        if big + eb * self.ck * p.Ho * p.Wo > _OOB_BASE:
            raise ValueError("tensors must fit below the masked-access offset")


def is_valid_direct_nchw_dgrad_spec(spec: DirectNchwDgradSpec, arch: str = "gfx950"):
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
        spec.validate()
        p = spec.problem
        t, k = spec.atom_tile, spec.atom_k
        ab = "bf16" if p.dtype == "bf16" else "f16"
        if not target.mma.has_shape(
            a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=t, n=t, k=k
        ):
            return False, f"missing mfma {spec.atom} {ab} on {arch}"
        if spec.wave_size != target.wave_size:
            return False, "wave size mismatch"
        if spec.threads_per_block > target.max_threads_per_block:
            return False, "too many threads for arch"
        if not target.fits_lds(spec.lds_bytes):
            return (
                False,
                f"LDS {spec.lds_bytes} B exceeds {target.lds_capacity_bytes} B",
            )
    except (ValueError, KeyError) as e:
        return False, str(e)
    return True, "ok"


def build_direct_conv_nchw_dgrad(
    spec: DirectNchwDgradSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR of one native-NCHW dgrad prototype kernel.

    Kernargs: the direct-conv forward ABI (:func:`emit_direct_params`) with
    ``A`` = dY (NKHW), ``B`` = W repacked to ``[C][KH][KW][K]``, ``D`` = dX
    (NCHW); launch values from ``ConvArgs.from_problem(forward_problem)``.
    """
    ok, why = is_valid_direct_nchw_dgrad_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DirectNchwDgradSpec for {arch}: {why}")

    p = spec.problem
    dtype = p.dtype
    io = _io_type(dtype)
    EB = 2
    KH, KW, S = p.KH, p.KW, spec.S
    TI, TJ, TC, CK = spec.tile_i, spec.tile_j, spec.tile_c, spec.ck
    THREADS = spec.threads_per_block
    WAVE = spec.wave_size
    AT, AK, FRAG = spec.atom_tile, spec.atom_k, spec.frag
    ACC = spec.acc_per_lane
    QUADS = ACC // 4
    NCB = spec.n_col_blocks
    M_TILES = spec.m_tiles_total
    MT_W = spec.m_tiles_per_wave
    ROWS_W = spec.rows_per_wave
    KATOMS = spec.k_atoms
    CSTRIDE = spec.c_stride
    LDS_H, LDS_W = spec.lds_h, spec.lds_w
    DH_LO, _ = spec.d_h
    DW_LO, _ = spec.d_w
    G = spec.stage_group
    HALO_L, HALO_R = spec.halo_l, spec.halo_r
    HV_L = spec._halo_vec(HALO_L) if HALO_L else 0
    HV_R = spec._halo_vec(HALO_R) if HALO_R else 0
    TAPS_H, TAPS_W = spec.taps_h, spec.taps_w
    N_TAPS = KH * KW

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    params = emit_direct_params(b, io_type=io)
    dY, Wt, dX = params["A"], params["B"], params["D"]
    p_Hi, p_Wi = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C = params["p_total_c"]
    K = params["p_total_k"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_eb = b.const_i32(EB)
    oob = b.const_i32(_OOB_BASE)
    dy_rsrc = b.buffer_rsrc(dY, params["A_bytes"])
    w_rsrc = b.buffer_rsrc(Wt, params["B_bytes"])
    dx_rsrc = b.buffer_rsrc(dX, params["D_bytes"])

    # NKHW / NCHW strides from runtime extents.
    dy_k_stride = b.mul(p_Ho, p_Wo)
    dy_n_stride = b.mul(K, dy_k_stride)
    dx_c_stride = b.mul(p_Hi, p_Wi)
    dx_n_stride = b.mul(C, dx_c_stride)

    x_stage = spec.lds_x_elems + _ALIGN
    w_stage = spec.lds_w_elems + FRAG
    X_smem = b.smem_alloc(io, [1, x_stage], name_hint="lds_dy")
    W_smem = b.smem_alloc(io, [1, w_stage], name_hint="lds_w")
    x_dump = b.const_i32(spec.lds_x_elems)
    w_dump = spec.lds_w_elems

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE))
    wave_id = b.div(tid, b.const_i32(WAVE))
    wave_m = b.div(wave_id, b.const_i32(spec.waves_n))
    wave_n = b.mod(wave_id, b.const_i32(spec.waves_n))
    lane_lo = b.mod(lane, b.const_i32(AT))
    lane_hi = b.div(lane, b.const_i32(AT))

    # ---- grid decode: c tile fastest (the dY window is shared by c tiles) --
    hp = b.div(b.add(p_Hi, b.const_i32(S - 1)), b.const_i32(S))
    wp = b.div(b.add(p_Wi, b.const_i32(S - 1)), b.const_i32(S))
    n_jt = b.div(b.add(wp, b.const_i32(TJ - 1)), b.const_i32(TJ))
    n_it = b.div(b.add(hp, b.const_i32(TI - 1)), b.const_i32(TI))
    n_ct = b.div(b.add(C, b.const_i32(TC - 1)), b.const_i32(TC))
    wgid = b.block_id_x()
    c_tile = b.mod(wgid, n_ct)
    cell = b.div(wgid, n_ct)
    n_ij = b.mul(n_jt, n_it)
    n_img = b.div(cell, n_ij)
    st = b.mod(cell, n_ij)
    j_tile = b.mod(st, n_jt)
    i_tile = b.div(st, n_jt)
    i0 = b.mul(i_tile, b.const_i32(TI))
    j0 = b.mul(j_tile, b.const_i32(TJ))
    c_base = b.mul(c_tile, b.const_i32(TC))

    dy_img = b.mul(n_img, dy_n_stride)
    row0 = b.add(i0, b.const_i32(DH_LO))  # first staged dY row

    # ---- dY staging metadata ----------------------------------------------
    # A unit is (channel group kg, window row, 8-run v): G loads of 8 wo each,
    # written as up to 8 G-wide LDS vectors at [row][col][kg*G].
    n_kg = CK // G
    NV = TJ // _ALIGN

    def _unit_passes(n_units: int):
        out = []
        for j in range(ceil_div(n_units, THREADS)):
            u = b.add(tid, b.const_i32(j * THREADS)) if j else tid
            in_tile = b.cmp_lt(u, b.const_i32(n_units)) if n_units % THREADS else None
            out.append((u, in_tile))
        return out

    def _row_meta(u, n_inner):
        """Split ``u`` = (kg, row, inner) with inner fastest."""
        inner = b.mod(u, b.const_i32(n_inner)) if n_inner > 1 else c0
        rest = b.div(u, b.const_i32(n_inner)) if n_inner > 1 else u
        row = b.mod(rest, b.const_i32(LDS_H))
        kg = b.div(rest, b.const_i32(LDS_H))
        ho = b.add(row0, row)
        ok_h = b.land(b.cmp_ge(ho, c0), b.cmp_lt(ho, p_Ho))
        k0 = b.mul(kg, b.const_i32(G))
        src = b.add(b.add(dy_img, b.mul(k0, dy_k_stride)), b.mul(ho, p_Wo))
        return inner, row, kg, ok_h, src

    # Body: columns [j0, j0+TJ) of the phase grid == dY columns (offset 0).
    body = []
    for u, in_tile in _unit_passes(n_kg * LDS_H * NV):
        v, row, kg, ok_h, src = _row_meta(u, NV)
        wo0 = b.add(j0, b.mul(v, b.const_i32(_ALIGN)))
        valid = b.land(ok_h, b.cmp_lt(wo0, p_Wo))
        if in_tile is not None:
            valid = b.land(valid, in_tile)
        off = b.select(valid, b.mul(b.add(src, wo0), c_eb), oob)
        col0 = b.add(b.mul(v, b.const_i32(_ALIGN)), b.const_i32(-DW_LO))
        dst0 = b.add(
            b.mul(b.add(b.mul(row, b.const_i32(LDS_W)), col0), b.const_i32(CSTRIDE)),
            b.mul(kg, b.const_i32(G)),
        )
        if in_tile is not None:
            dst0 = b.select(in_tile, dst0, x_dump)
            step = b.select(in_tile, b.const_i32(CSTRIDE), c0)
        else:
            step = None
        dsts = []
        for e in range(_ALIGN):
            if e == 0:
                dsts.append(dst0)
            elif step is None:
                dsts.append(b.add(dst0, b.const_i32(e * CSTRIDE)))
            else:
                dsts.append(b.add(dst0, b.mul(step, b.const_i32(e))))
        body.append((off, list(range(_ALIGN)), dsts, _ALIGN))

    # Halo columns: one aligned HV-wide load per (kg, row), only the |d|
    # columns adjacent to the body are written.
    halos = []
    for side, h, hv in (("l", HALO_L, HV_L), ("r", HALO_R, HV_R)):
        if not h:
            continue
        for u, in_tile in _unit_passes(n_kg * LDS_H):
            _, row, kg, ok_h, src = _row_meta(u, 1)
            if side == "l":
                wo0 = b.sub(j0, b.const_i32(hv))
                valid = b.land(ok_h, b.cmp_ge(wo0, c0))
                elems = list(range(hv - h, hv))
                col0 = b.const_i32(-DW_LO - hv)
            else:
                wo0 = b.add(j0, b.const_i32(TJ))
                valid = b.land(ok_h, b.cmp_lt(wo0, p_Wo))
                elems = list(range(h))
                col0 = b.const_i32(-DW_LO + TJ)
            if in_tile is not None:
                valid = b.land(valid, in_tile)
            off = b.select(valid, b.mul(b.add(src, wo0), c_eb), oob)
            base = b.add(
                b.mul(
                    b.add(b.mul(row, b.const_i32(LDS_W)), col0), b.const_i32(CSTRIDE)
                ),
                b.mul(kg, b.const_i32(G)),
            )
            dsts = []
            for e in elems:
                d = b.add(base, b.const_i32(e * CSTRIDE))
                if in_tile is not None:
                    d = b.select(in_tile, d, x_dump)
                dsts.append(d)
            halos.append((off, elems, dsts, hv))

    x_units = body + halos

    # ---- W staging metadata: fragment-slot layout, [C][KH][KW][K] source --
    W_SLOTS = N_TAPS * M_TILES * KATOMS * WAVE
    w_meta = []
    for j in range(ceil_div(W_SLOTS, THREADS)):
        g = b.add(tid, b.const_i32(j * THREADS))
        g_lane = b.mod(g, b.const_i32(WAVE))
        rest = b.div(g, b.const_i32(WAVE))
        katom = b.mod(rest, b.const_i32(KATOMS))
        rest2 = b.div(rest, b.const_i32(KATOMS))
        m_tile = b.mod(rest2, b.const_i32(M_TILES))
        tap = b.div(rest2, b.const_i32(M_TILES))
        c_out = b.add(
            c_base,
            b.add(b.mul(m_tile, b.const_i32(AT)), b.mod(g_lane, b.const_i32(AT))),
        )
        k_in = b.add(
            b.mul(katom, b.const_i32(AK)),
            b.mul(b.div(g_lane, b.const_i32(AT)), b.const_i32(FRAG)),
        )
        valid = b.cmp_lt(c_out, C)
        lds_idx = b.mul(g, b.const_i32(FRAG))
        if W_SLOTS % THREADS:
            in_tile = b.cmp_lt(g, b.const_i32(W_SLOTS))
            valid = b.land(valid, in_tile)
            lds_idx = b.select(in_tile, lds_idx, b.const_i32(w_dump))
        elems = b.add(b.mul(b.add(b.mul(c_out, b.const_i32(N_TAPS)), tap), K), k_in)
        w_meta.append((b.select(valid, b.mul(elems, c_eb), oob), lds_idx))

    x_chunk_bytes = b.mul(b.mul(b.const_i32(CK), dy_k_stride), c_eb)
    w_chunk_bytes = b.const_i32(CK * EB)

    def issue_loads(cx: Value, cw: Value):
        xs = []
        for off, _elems, _dsts, width in x_units:
            for e in range(G):
                a = (
                    b.add(off, b.add(cx, b.mul(dy_k_stride, b.const_i32(e * EB))))
                    if e
                    else b.add(off, cx)
                )
                if width == 1:
                    v = (
                        b.buffer_load_bf16(dy_rsrc, a, c0)
                        if dtype == "bf16"
                        else b.buffer_load_f16(dy_rsrc, a, c0)
                    )
                    v = b.vec_pack([v], io)
                else:
                    v = _buf_load_vN(b, dtype, dy_rsrc, a, c0, width // 2)
                xs.append(v)
        ws = [
            _buf_load_vN(b, dtype, w_rsrc, b.add(off, cw), c0, FRAG // 2)
            for off, _ in w_meta
        ]
        return xs, ws

    def commit(xs, ws):
        i = 0
        for _off, elems, dsts, _width in x_units:
            runs = xs[i : i + G]
            i += G
            for e, d in zip(elems, dsts):
                vals = [b.vec_extract(r, e) for r in runs]
                if G == 1:
                    b.smem_store_vN(X_smem, [c0, d], vals[0], 1)
                else:
                    b.smem_store_vN(X_smem, [c0, d], b.vec_pack(vals, io), G)
        for (_, lds_idx), v in zip(w_meta, ws):
            b.smem_store_vN(W_smem, [c0, lds_idx], v, FRAG)

    # ---- LDS read bases ---------------------------------------------------
    x_base = b.add(
        b.mul(
            b.add(b.mul(wave_n, b.const_i32(ROWS_W * LDS_W)), lane_lo),
            b.const_i32(CSTRIDE),
        ),
        b.mul(lane_hi, b.const_i32(FRAG)),
    )
    w_base = b.add(
        b.mul(wave_m, b.const_i32(MT_W * KATOMS * WAVE * FRAG)),
        b.mul(lane, b.const_i32(FRAG)),
    )

    def read_w(tap: int, mt: int, katom: int) -> Value:
        off = ((tap * M_TILES + mt) * KATOMS + katom) * WAVE * FRAG
        return b.smem_load_vN(
            W_smem, c0, b.add(w_base, b.const_i32(off)), dtype=io, n=FRAG
        )

    def read_x(lrow: int, lcol: int, katom: int) -> Value:
        off = (lrow * LDS_W + lcol) * CSTRIDE + katom * AK
        return b.smem_load_vN(
            X_smem, c0, b.add(x_base, b.const_i32(off)), dtype=io, n=FRAG
        )

    def _mfma(a: Value, bv: Value, acc: Value) -> Value:
        fn = getattr(b, f"mfma_f32_{spec.atom}_{'bf16' if dtype == 'bf16' else 'f16'}")
        return fn(a, bv, acc)

    # Accumulator index: (ph, pw, row, cb, mt).
    acc_idx: Dict[Tuple[int, int, int, int, int], int] = {}
    for ph in range(S):
        for pw in range(S):
            for r in range(ROWS_W):
                for cb in range(NCB):
                    for mt in range(MT_W):
                        acc_idx[(ph, pw, r, cb, mt)] = len(acc_idx)
    n_acc = len(acc_idx)

    # Group the (pw, s, d_s) users by column offset d_s so each dY fragment
    # column is read once per k atom.
    by_dw: Dict[int, List[Tuple[int, int]]] = {}
    for pw in range(S):
        for s, dw in TAPS_W[pw]:
            by_dw.setdefault(dw, []).append((pw, s))

    def emit_mfmas(accs):
        for katom in range(KATOMS):
            wf: Dict[Tuple[int, int], Value] = {}
            for dw in sorted(by_dw):
                xf: Dict[Tuple[int, int], Value] = {}
                for pw, s in by_dw[dw]:
                    for ph in range(S):
                        for r, dh in TAPS_H[ph]:
                            tap = r * KW + s
                            for mt in range(MT_W):
                                if (tap, mt) not in wf:
                                    wf[(tap, mt)] = read_w(tap, mt, katom)
                            for row in range(ROWS_W):
                                lrow = row + dh - DH_LO
                                for cb in range(NCB):
                                    key = (lrow, cb)
                                    if key not in xf:
                                        xf[key] = read_x(
                                            lrow, cb * AT + dw - DW_LO, katom
                                        )
                                    for mt in range(MT_W):
                                        ix = acc_idx[(ph, pw, row, cb, mt)]
                                        accs[ix] = _mfma(
                                            xf[key], wf[(tap, mt)], accs[ix]
                                        )

    xs0, ws0 = issue_loads(c0, c0)
    n_x = len(xs0)
    zero = b.zero_vec_f32(ACC)
    iter_args = [(f"acc{i}", zero) for i in range(n_acc)]
    iter_args += [(f"xs{i}", v) for i, v in enumerate(xs0)]
    iter_args += [(f"ws{i}", v) for i, v in enumerate(ws0)]
    n_chunks = b.smax(b.div(K, b.const_i32(CK)), c1)
    loop = b.scf_for_iter(
        c0, n_chunks, c1, iter_args, iv_name="k_iter", elide_trailing_barrier=False
    )
    with loop as (k_iv, carried):
        accs = list(carried[:n_acc])
        xs = list(carried[n_acc : n_acc + n_x])
        ws = list(carried[n_acc + n_x :])
        b.sync(fence="workgroup")
        commit(xs, ws)
        b.sync(fence="workgroup")
        k_next = b.add(k_iv, c1)
        xs_n, ws_n = issue_loads(
            b.mul(k_next, x_chunk_bytes), b.mul(k_next, w_chunk_bytes)
        )
        emit_mfmas(accs)
        b.scf_yield(*accs, *xs_n, *ws_n)
    accs_out = list(loop.results[:n_acc])

    # ---- epilogue: NCHW, w-contiguous vector stores -----------------------
    c_wave = b.add(c_base, b.mul(wave_m, b.const_i32(MT_W * AT)))
    i_wave = b.add(i0, b.mul(wave_n, b.const_i32(ROWS_W)))
    dx_img = b.mul(n_img, dx_n_stride)
    lane_j = b.mul(lane_hi, b.const_i32(4))
    VW = 4 * S  # elements per lane per quad
    for mt in range(MT_W):
        c_out = b.add(b.add(c_wave, b.const_i32(mt * AT)), lane_lo)
        c_ok = b.cmp_lt(c_out, C)
        c_off = b.add(dx_img, b.mul(c_out, dx_c_stride))
        for ph in range(S):
            for row in range(ROWS_W):
                h = b.add(
                    b.mul(b.add(i_wave, b.const_i32(row)), b.const_i32(S)),
                    b.const_i32(ph),
                )
                hc_ok = b.land(c_ok, b.cmp_lt(h, p_Hi))
                row_off = b.add(c_off, b.mul(h, p_Wi))
                for cb in range(NCB):
                    vals_pw = [
                        _trunc_f32(b, dtype, accs_out[acc_idx[(ph, pw, row, cb, mt)]])
                        for pw in range(S)
                    ]
                    for q in range(QUADS):
                        j = b.add(
                            j0, b.add(b.const_i32(cb * AT + q * (AT // 4)), lane_j)
                        )
                        w0 = b.mul(j, b.const_i32(S)) if S > 1 else j
                        comps = [
                            b.vec_extract(vals_pw[e % S], 4 * q + e // S)
                            for e in range(VW)
                        ]
                        vec = b.vec_pack(comps, io)
                        d_el = b.add(row_off, w0)
                        full = b.land(
                            hc_ok, b.cmp_lt(b.add(w0, b.const_i32(VW - 1)), p_Wi)
                        )
                        br = b.scf_if_else(full)
                        with br as (then_ctx, else_ctx):
                            with then_ctx:
                                _buf_store_vN(
                                    b,
                                    dtype,
                                    dx_rsrc,
                                    b.mul(d_el, c_eb),
                                    c0,
                                    vec,
                                    VW // 2,
                                )
                            with else_ctx:
                                for e in range(VW):
                                    we = b.add(w0, b.const_i32(e)) if e else w0
                                    ok_e = b.land(hc_ok, b.cmp_lt(we, p_Wi))
                                    de = b.add(d_el, b.const_i32(e)) if e else d_el
                                    o = b.select(ok_e, b.mul(de, c_eb), oob)
                                    if dtype == "bf16":
                                        b.buffer_store_bf16(dx_rsrc, o, c0, comps[e])
                                    else:
                                        b.buffer_store_f16(dx_rsrc, o, c0, comps[e])
    return b.kernel
