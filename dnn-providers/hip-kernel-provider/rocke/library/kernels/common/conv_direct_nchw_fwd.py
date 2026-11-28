# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native-NCHW forward direct convolution (gfx950, stride 1/2).

A productionized AOT-buildable kernel family: a C++ mirror lives at
``library/cpp/instances/common/conv_direct_nchw_fwd.cpp`` and is gated by
``tools/check_byte_identity.py`` like every other family, with a ``.py``/``.c``
parity pair under ``library/tests/parity/``. Runtime dispatch selection is
**deliberately not yet wired**: this family is not registered in
``library/dispatch/grouped_convolution.py`` or its instance JSONs, so
``algorithm="auto"`` never reaches it. Callers build and launch it explicitly
(directly via :func:`build_direct_conv_nchw_fwd`, or through the AOT sweep in
``kernels/conv/direct_variants.py`` / ``kernels/conv/direct_aot_jobs.py`` with
variant ``"direct_nchw_fwd"``), not via the request-level dispatcher.

``groups == 1``, fp16 / bf16 storage, dilation 1, stride 1 or 2. It consumes
``X`` in NCHW and produces ``Y`` in NKHW, and reuses the design of the native
dgrad family
(:mod:`kernels.common.conv_direct_nchw_dgrad`), whose stride-1 case already is
a forward conv of the transposed problem. What changes for the forward
direction is where the stride sits: on the *input* side. The output-side
stride-phase trick of dgrad does not apply; instead the MFMA reads the
``[row][col][c]`` LDS tile at column ``wo*S + s`` for tap ``s``.

* **X staging** (NCHW source): one workgroup owns a ``tile_i x tile_j`` block
  of output positions and ``tile_k`` output channels. Per ``ck``-deep chunk of
  the input channels it stages the input window: ``(tile_i-1)*S + KH`` rows of
  ``PAD + tile_j*S + max(0, KW-PAD-S)`` columns. ``tile_j*S`` is a multiple of
  8 and the window body starts at input column ``j0*S`` (a multiple of 8), so
  the body is whole aligned 8-element runs along ``w``
  (``buffer_load_dwordx4``, always all-in or all-out of the row given
  ``W % 8 == 0``), and the ``PAD``-wide left / ``KW-PAD-S``-wide right halos
  are one narrow aligned load per row. ``stage_group`` channels are moved to
  the innermost LDS axis per position (register transpose, ``G``-wide LDS
  writes), exactly as in the dgrad prototype.
* **LDS layout**: ``[row][col][c]`` (``c`` contiguous, padded by ``lds_pad``;
  each row padded by ``lds_wpad`` positions). With ``col_split`` (stride 2)
  the columns are stored as two planes, even input columns then odd ones, so
  the 32 lanes of one MFMA fragment read adjacent positions instead of every
  other one -- this keeps the ``ds_read_b128`` lane stride an odd number of
  16-byte units (bank-conflict free, see :func:`x_read_conflict_ways`).
* **W staging**: weights repacked once at plan time to ``[K][KH][KW][C]``
  (KRSC, ``c`` contiguous, :func:`repack_weights_krsc`); fragment-slot LDS
  layout as in the dgrad prototype.
* **MFMA**: operands swapped (the X fragment is the A operand) so the
  accumulator rows are output positions and each lane holds 4 consecutive
  ``wo`` of one ``k``; the epilogue stores them as one ``dwordx2`` along
  NKHW's contiguous axis, masked per element only on edge tiles.
* **Levers** (as in the wgrad direct prototype): ``waves_per_eu``,
  ``pin_prefetch`` (a scheduling barrier keeps the next chunk's register
  prefetch ahead of the MFMA stream) and the LDS pads.

Runtime precondition (checked by :func:`is_valid_direct_nchw_fwd_spec` on the
spec's problem, *not* by the binary): ``W % 8 == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
_ALIGN = 8  # elements in 16 bytes of a 16-bit dtype

# Atom name -> (tile, K, frag).
_ATOMS = {
    "32x32x16": (32, 16, 8),
    "32x32x8": (32, 8, 4),
    "16x16x32": (16, 32, 8),
    "16x16x16": (16, 16, 4),
}


def repack_weights_krsc(W_kcyx):
    """Plan-time repack of torch KCYX weights into this kernel's ``[K][KH][KW][C]``."""
    return W_kcyx.permute(0, 2, 3, 1).contiguous()


@dataclass(frozen=True)
class DirectNchwFwdSpec:
    """One native-NCHW forward prototype configuration.

    Tile sizes are in output units: a workgroup writes ``tile_i`` rows,
    ``tile_j`` columns and ``tile_k`` channels of ``Y``.
    """

    problem: DirectConvProblem
    name: str = "conv_direct_nchw_fwd"

    tile_i: int = 8
    tile_j: int = 32
    tile_k: int = 64
    ck: int = 32

    waves_m: int = 2  # split tile_k
    waves_n: int = 2  # split tile_i

    atom: str = "32x32x16"
    wave_size: int = 64
    lds_pad: int = 8  # elements on the channel stride of one LDS position
    lds_wpad: int = 0  # extra positions per LDS row
    stage_group: int = 4
    # Stride 2: store even / odd input columns as two LDS planes.
    col_split: bool = False
    waves_per_eu: int = 0
    # Scheduling barrier after the next chunk's global loads.
    pin_prefetch: bool = False

    # ---- derived ----------------------------------------------------------

    @property
    def S(self) -> int:
        return self.problem.stride

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
        return self.tile_k // self.atom_tile

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
    def halo_l(self) -> int:
        return self.problem.PAD_W

    @property
    def halo_r(self) -> int:
        return max(0, self.problem.KW - self.problem.PAD_W - self.S)

    @property
    def body_w(self) -> int:
        return self.tile_j * self.S

    @property
    def lds_h(self) -> int:
        return (self.tile_i - 1) * self.S + self.problem.KH

    @property
    def lds_w(self) -> int:
        """Window columns (input columns ``j0*S - PAD ..``)."""
        return self.halo_l + self.body_w + self.halo_r

    @property
    def plane_w(self) -> int:
        """Positions per even/odd column plane (``col_split``)."""
        return ceil_div(self.lds_w, 2)

    @property
    def lds_row(self) -> int:
        """Positions per LDS row (incl. ``lds_wpad``)."""
        w = 2 * self.plane_w if self.col_split else self.lds_w
        return w + self.lds_wpad

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
        return self.lds_h * self.lds_row * self.c_stride

    @property
    def lds_w_elems(self) -> int:
        p = self.problem
        return self.tile_k * p.KH * p.KW * self.ck

    @property
    def lds_bytes(self) -> int:
        # + one dump slot per tile for a partial last staging pass.
        return 2 * (self.lds_x_elems + _ALIGN + self.lds_w_elems + self.frag)

    @property
    def acc_vgprs(self) -> int:
        return (
            self.rows_per_wave
            * self.n_col_blocks
            * self.m_tiles_per_wave
            * self.acc_per_lane
        )

    def lds_col(self, x: int) -> int:
        """LDS position (within a row) of window column ``x``."""
        if self.col_split:
            return (x % 2) * self.plane_w + x // 2
        return x

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        n = (
            ceil_div(p.Wo, self.tile_j)
            * ceil_div(p.Ho, self.tile_i)
            * p.N
            * ceil_div(p.total_k, self.tile_k)
        )
        return (n, 1, 1)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}s{p.stride}p{p.PAD}",
            f"t{self.tile_i}x{self.tile_j}x{self.tile_k}",
            f"ck{self.ck}",
            f"w{self.waves_m}x{self.waves_n}",
            f"a{self.atom}",
            f"g{self.stage_group}",
            f"lp{self.lds_pad}",
            f"wp{self.lds_wpad}" if self.lds_wpad else "",
            "cs" if self.col_split else "",
            f"wpe{self.waves_per_eu}" if self.waves_per_eu else "",
            "pin" if self.pin_prefetch else "",
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
        if self.tile_k % t:
            raise ValueError(f"tile_k must be a multiple of {t}")
        if self.m_tiles_total % self.waves_m:
            raise ValueError("tile_k / atom not divisible by waves_m")
        if self.tile_i % self.waves_n:
            raise ValueError("tile_i not divisible by waves_n")
        if self.ck % self.atom_k or self.ck % _ALIGN:
            raise ValueError(f"ck must be a multiple of {self.atom_k} and {_ALIGN}")
        if self.stage_group not in (1, 2, 4, 8) or self.ck % self.stage_group:
            raise ValueError("stage_group must be 1/2/4/8 and divide ck")
        if self.lds_pad < 0 or self.lds_pad % _ALIGN:
            raise ValueError(f"lds_pad must be a non-negative multiple of {_ALIGN}")
        if self.lds_wpad < 0:
            raise ValueError("lds_wpad must be >= 0")
        if self.col_split and self.S != 2:
            raise ValueError("col_split needs stride 2")
        if self.waves_per_eu < 0:
            raise ValueError("waves_per_eu must be >= 0")
        if p.cpg % self.ck:
            raise ValueError(f"C {p.cpg} not divisible by ck {self.ck}")
        if p.W % _ALIGN:
            raise ValueError(f"W {p.W} must be a multiple of {_ALIGN} (whole X runs)")
        if not (0 <= p.PAD_W <= _ALIGN) or self.halo_r > _ALIGN:
            raise ValueError("halo wider than 8 columns")
        if p.PAD < 0:
            raise ValueError("negative padding")
        if self.threads_per_block > 1024:
            raise ValueError("too many threads")
        if self.acc_vgprs > 256:
            raise ValueError(f"accumulators need {self.acc_vgprs} registers per lane")
        eb = 2
        big = max(
            eb * p.N * p.total_c * p.H * p.W,
            eb * p.total_c * p.total_k * p.KH * p.KW,
            eb * p.N * p.total_k * p.Ho * p.Wo,
        )
        if big + eb * self.ck * p.H * p.W > _OOB_BASE:
            raise ValueError("tensors must fit below the masked-access offset")


def is_valid_direct_nchw_fwd_spec(spec: DirectNchwFwdSpec, arch: str = "gfx950"):
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


# ---------------------------------------------------------------------------
# LDS bank-conflict model of the X fragment reads (gfx950 ds_read_b128)
# ---------------------------------------------------------------------------


def _b128_read_phase(lane: int) -> Tuple[int, int]:
    b = [(lane >> i) & 1 for i in range(6)]
    return b[5], b[4] ^ b[3] ^ b[2]


def x_read_conflict_ways(spec: DirectNchwFwdSpec) -> int:
    """Worst modelled bank-conflict degree of the X ``ds_read_b128`` reads.

    gfx950 groups the 64 lanes of a ``ds_read_b128`` into four 16-lane
    phases; within a phase, distinct addresses with the same byte address
    modulo 256 conflict (the lds-bank-conflict-expert gfx950 profile).
    Returns the largest number of distinct addresses sharing a class over
    every tap column the kernel reads (row offsets are wave-uniform).
    """
    AT, FRAG, CS, S = spec.atom_tile, spec.frag, spec.c_stride, spec.S
    worst = 1
    for s in range(spec.problem.KW):
        for cb in range(spec.n_col_blocks):
            groups: Dict[Tuple[int, int, int], set] = {}
            for lane in range(spec.wave_size):
                j = cb * AT + lane % AT
                col = spec.lds_col(j * S + s)
                addr = 2 * (col * CS + (lane // AT) * FRAG)
                key = (*_b128_read_phase(lane), addr % 256)
                groups.setdefault(key, set()).add(addr)
            worst = max(worst, max(len(v) for v in groups.values()))
    return worst


def with_conflict_free_pad(spec: DirectNchwFwdSpec, pads=(0, 8, 16, 24, 32)):
    """``spec`` with the smallest ``lds_pad`` minimising the read model."""
    best = None
    for pad in pads:
        s2 = replace(spec, lds_pad=pad)
        key = (x_read_conflict_ways(s2), pad)
        if best is None or key < best:
            best = key
    return replace(spec, lds_pad=best[1])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_direct_conv_nchw_fwd(
    spec: DirectNchwFwdSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR of one native-NCHW forward prototype kernel.

    Kernargs: the direct-conv forward ABI (:func:`emit_direct_params`) with
    ``A`` = X (NCHW), ``B`` = W repacked to ``[K][KH][KW][C]``, ``D`` = Y
    (NKHW); launch values from ``ConvArgs.from_problem(problem)``.
    """
    ok, why = is_valid_direct_nchw_fwd_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DirectNchwFwdSpec for {arch}: {why}")

    p = spec.problem
    dtype = p.dtype
    io = _io_type(dtype)
    EB = 2
    KH, KW, S = p.KH, p.KW, spec.S
    TI, TJ, TK, CK = spec.tile_i, spec.tile_j, spec.tile_k, spec.ck
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
    LDS_H, LDS_ROW = spec.lds_h, spec.lds_row
    G = spec.stage_group
    HALO_L, HALO_R = spec.halo_l, spec.halo_r
    HV_L = spec._halo_vec(HALO_L) if HALO_L else 0
    HV_R = spec._halo_vec(HALO_R) if HALO_R else 0
    BODY = spec.body_w
    N_TAPS = KH * KW
    SPLIT = spec.col_split

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, 8)
    params = emit_direct_params(b, io_type=io)
    X, Wt, Y = params["A"], params["B"], params["D"]
    p_Hi, p_Wi = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C = params["p_total_c"]
    K = params["p_total_k"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_eb = b.const_i32(EB)
    oob = b.const_i32(_OOB_BASE)
    x_rsrc = b.buffer_rsrc(X, params["A_bytes"])
    w_rsrc = b.buffer_rsrc(Wt, params["B_bytes"])
    y_rsrc = b.buffer_rsrc(Y, params["D_bytes"])

    x_c_stride = b.mul(p_Hi, p_Wi)
    x_n_stride = b.mul(C, x_c_stride)
    y_k_stride = b.mul(p_Ho, p_Wo)
    y_n_stride = b.mul(K, y_k_stride)

    x_stage = spec.lds_x_elems + _ALIGN
    w_stage = spec.lds_w_elems + FRAG
    X_smem = b.smem_alloc(io, [1, x_stage], name_hint="lds_x")
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

    # ---- grid decode: k tile fastest (the X window is shared by k tiles) --
    n_jt = b.div(b.add(p_Wo, b.const_i32(TJ - 1)), b.const_i32(TJ))
    n_it = b.div(b.add(p_Ho, b.const_i32(TI - 1)), b.const_i32(TI))
    n_kt = b.div(b.add(K, b.const_i32(TK - 1)), b.const_i32(TK))
    wgid = b.block_id_x()
    k_tile = b.mod(wgid, n_kt)
    cell = b.div(wgid, n_kt)
    n_ij = b.mul(n_jt, n_it)
    n_img = b.div(cell, n_ij)
    st = b.mod(cell, n_ij)
    j_tile = b.mod(st, n_jt)
    i_tile = b.div(st, n_jt)
    i0 = b.mul(i_tile, b.const_i32(TI))
    j0 = b.mul(j_tile, b.const_i32(TJ))
    k_base = b.mul(k_tile, b.const_i32(TK))

    x_img = b.mul(n_img, x_n_stride)
    row0 = b.sub(b.mul(i0, b.const_i32(S)), b.const_i32(p.PAD))  # first X row
    xcol0 = b.mul(j0, b.const_i32(S))  # first body column (multiple of 8)

    # ---- X staging metadata ------------------------------------------------
    n_kg = CK // G
    NV = BODY // _ALIGN

    def _unit_passes(n_units: int):
        out = []
        for j in range(ceil_div(n_units, THREADS)):
            u = b.add(tid, b.const_i32(j * THREADS)) if j else tid
            in_tile = b.cmp_lt(u, b.const_i32(n_units)) if n_units % THREADS else None
            out.append((u, in_tile))
        return out

    def _row_meta(u, n_inner):
        """Split ``u`` = (cg, row, inner) with inner fastest."""
        inner = b.mod(u, b.const_i32(n_inner)) if n_inner > 1 else c0
        rest = b.div(u, b.const_i32(n_inner)) if n_inner > 1 else u
        row = b.mod(rest, b.const_i32(LDS_H))
        cg = b.div(rest, b.const_i32(LDS_H))
        h = b.add(row0, row)
        ok_h = b.land(b.cmp_ge(h, c0), b.cmp_lt(h, p_Hi))
        cc0 = b.mul(cg, b.const_i32(G))
        src = b.add(b.add(x_img, b.mul(cc0, x_c_stride)), b.mul(h, p_Wi))
        return inner, row, cg, ok_h, src

    def _pos(row, x_const_or_val, cg):
        """LDS element index of window (row, column x), channel group cg."""
        return b.add(
            b.mul(
                b.add(b.mul(row, b.const_i32(LDS_ROW)), x_const_or_val),
                b.const_i32(CSTRIDE),
            ),
            b.mul(cg, b.const_i32(G)),
        )

    # Body: window columns [HALO_L, HALO_L + BODY) == input [xcol0, xcol0+BODY).
    body = []
    for u, in_tile in _unit_passes(n_kg * LDS_H * NV):
        v, row, cg, ok_h, src = _row_meta(u, NV)
        w0 = b.add(xcol0, b.mul(v, b.const_i32(_ALIGN)))
        valid = b.land(ok_h, b.cmp_lt(w0, p_Wi))
        if in_tile is not None:
            valid = b.land(valid, in_tile)
        off = b.select(valid, b.mul(b.add(src, w0), c_eb), oob)
        # Window column of element e: HALO_L + 8*v + e. With col_split the
        # plane is fixed by the parity of (HALO_L + e) (8*v is even).
        dsts = []
        for e in range(_ALIGN):
            x = HALO_L + e
            if SPLIT:
                colv = b.add(
                    b.mul(v, b.const_i32(_ALIGN // 2)),
                    b.const_i32((x % 2) * spec.plane_w + x // 2),
                )
            else:
                colv = b.add(b.mul(v, b.const_i32(_ALIGN)), b.const_i32(x))
            d = _pos(row, colv, cg)
            if in_tile is not None:
                d = b.select(in_tile, d, x_dump)
            dsts.append(d)
        body.append((off, list(range(_ALIGN)), dsts, _ALIGN))

    # Halo columns: one aligned HV-wide load per (cg, row); only the columns
    # adjacent to the body are written.
    halos = []
    for side, h, hv in (("l", HALO_L, HV_L), ("r", HALO_R, HV_R)):
        if not h:
            continue
        for u, in_tile in _unit_passes(n_kg * LDS_H):
            _, row, cg, ok_h, src = _row_meta(u, 1)
            if side == "l":
                w0 = b.sub(xcol0, b.const_i32(hv))
                valid = b.land(ok_h, b.cmp_ge(w0, c0))
                elems = list(range(hv - h, hv))
                xs_ = [e - (hv - h) for e in elems]  # window columns 0..h-1
            else:
                w0 = b.add(xcol0, b.const_i32(BODY))
                valid = b.land(ok_h, b.cmp_lt(w0, p_Wi))
                elems = list(range(h))
                xs_ = [HALO_L + BODY + e for e in elems]
            if in_tile is not None:
                valid = b.land(valid, in_tile)
            off = b.select(valid, b.mul(b.add(src, w0), c_eb), oob)
            dsts = []
            for x in xs_:
                d = _pos(row, b.const_i32(spec.lds_col(x)), cg)
                if in_tile is not None:
                    d = b.select(in_tile, d, x_dump)
                dsts.append(d)
            halos.append((off, elems, dsts, hv))

    x_units = body + halos

    # ---- W staging metadata: fragment-slot layout, [K][KH][KW][C] source --
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
        k_out = b.add(
            k_base,
            b.add(b.mul(m_tile, b.const_i32(AT)), b.mod(g_lane, b.const_i32(AT))),
        )
        c_in = b.add(
            b.mul(katom, b.const_i32(AK)),
            b.mul(b.div(g_lane, b.const_i32(AT)), b.const_i32(FRAG)),
        )
        valid = b.cmp_lt(k_out, K)
        lds_idx = b.mul(g, b.const_i32(FRAG))
        if W_SLOTS % THREADS:
            in_tile = b.cmp_lt(g, b.const_i32(W_SLOTS))
            valid = b.land(valid, in_tile)
            lds_idx = b.select(in_tile, lds_idx, b.const_i32(w_dump))
        elems = b.add(b.mul(b.add(b.mul(k_out, b.const_i32(N_TAPS)), tap), C), c_in)
        w_meta.append((b.select(valid, b.mul(elems, c_eb), oob), lds_idx))

    x_chunk_bytes = b.mul(b.mul(b.const_i32(CK), x_c_stride), c_eb)
    w_chunk_bytes = b.const_i32(CK * EB)

    def issue_loads(cx: Value, cw: Value):
        xs = []
        for off, _elems, _dsts, width in x_units:
            for e in range(G):
                a = (
                    b.add(off, b.add(cx, b.mul(x_c_stride, b.const_i32(e * EB))))
                    if e
                    else b.add(off, cx)
                )
                xs.append(_buf_load_vN(b, dtype, x_rsrc, a, c0, width // 2))
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
    # Lane (lane_lo, lane_hi) of col block cb reads output column
    # j = cb*AT + lane_lo at tap s: window column j*S + s.
    if SPLIT:
        lane_col = lane_lo  # j*2 + s -> plane s%2, position j + s//2
    else:
        lane_col = b.mul(lane_lo, b.const_i32(S)) if S > 1 else lane_lo
    x_base = b.add(
        b.mul(
            b.add(b.mul(wave_n, b.const_i32(ROWS_W * S * LDS_ROW)), lane_col),
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

    def _col_off(cb: int, s: int) -> int:
        if SPLIT:
            return (s % 2) * spec.plane_w + cb * AT + s // 2
        return cb * AT * S + s

    def read_x(lrow: int, cb: int, s: int, katom: int) -> Value:
        off = (lrow * LDS_ROW + _col_off(cb, s)) * CSTRIDE + katom * AK
        return b.smem_load_vN(
            X_smem, c0, b.add(x_base, b.const_i32(off)), dtype=io, n=FRAG
        )

    def _mfma(a: Value, bv: Value, acc: Value) -> Value:
        fn = getattr(b, f"mfma_f32_{spec.atom}_{'bf16' if dtype == 'bf16' else 'f16'}")
        return fn(a, bv, acc)

    acc_idx: Dict[Tuple[int, int, int], int] = {}
    for row in range(ROWS_W):
        for cb in range(NCB):
            for mt in range(MT_W):
                acc_idx[(row, cb, mt)] = len(acc_idx)
    n_acc = len(acc_idx)

    def emit_mfmas(accs):
        # Tap column s outer: the X fragments of one (row, cb, s) are reused
        # by every r that lands on the same window row.
        for katom in range(KATOMS):
            wf: Dict[Tuple[int, int], Value] = {}
            for s in range(KW):
                xf: Dict[Tuple[int, int], Value] = {}
                for r in range(KH):
                    tap = r * KW + s
                    for mt in range(MT_W):
                        if (tap, mt) not in wf:
                            wf[(tap, mt)] = read_w(tap, mt, katom)
                    for row in range(ROWS_W):
                        lrow = row * S + r
                        for cb in range(NCB):
                            key = (lrow, cb)
                            if key not in xf:
                                xf[key] = read_x(lrow, cb, s, katom)
                            for mt in range(MT_W):
                                ix = acc_idx[(row, cb, mt)]
                                accs[ix] = _mfma(xf[key], wf[(tap, mt)], accs[ix])

    xs0, ws0 = issue_loads(c0, c0)
    n_x = len(xs0)
    zero = b.zero_vec_f32(ACC)
    iter_args = [(f"acc{i}", zero) for i in range(n_acc)]
    iter_args += [(f"xs{i}", v) for i, v in enumerate(xs0)]
    iter_args += [(f"ws{i}", v) for i, v in enumerate(ws0)]
    n_chunks = b.smax(b.div(C, b.const_i32(CK)), c1)
    loop = b.scf_for_iter(
        c0, n_chunks, c1, iter_args, iv_name="c_iter", elide_trailing_barrier=False
    )
    with loop as (c_iv, carried):
        accs = list(carried[:n_acc])
        xs = list(carried[n_acc : n_acc + n_x])
        ws = list(carried[n_acc + n_x :])
        b.sync(fence="workgroup")
        commit(xs, ws)
        b.sync(fence="workgroup")
        c_next = b.add(c_iv, c1)
        xs_n, ws_n = issue_loads(
            b.mul(c_next, x_chunk_bytes), b.mul(c_next, w_chunk_bytes)
        )
        if spec.pin_prefetch:
            b.sched_barrier(0)
        emit_mfmas(accs)
        b.scf_yield(*accs, *xs_n, *ws_n)
    accs_out = list(loop.results[:n_acc])

    # ---- epilogue: NKHW, wo-contiguous dwordx2 stores ----------------------
    k_wave = b.add(k_base, b.mul(wave_m, b.const_i32(MT_W * AT)))
    i_wave = b.add(i0, b.mul(wave_n, b.const_i32(ROWS_W)))
    y_img = b.mul(n_img, y_n_stride)
    lane_j = b.mul(lane_hi, b.const_i32(4))
    for mt in range(MT_W):
        k_out = b.add(b.add(k_wave, b.const_i32(mt * AT)), lane_lo)
        k_ok = b.cmp_lt(k_out, K)
        k_off = b.add(y_img, b.mul(k_out, y_k_stride))
        for row in range(ROWS_W):
            ho = b.add(i_wave, b.const_i32(row))
            hk_ok = b.land(k_ok, b.cmp_lt(ho, p_Ho))
            row_off = b.add(k_off, b.mul(ho, p_Wo))
            for cb in range(NCB):
                vals = _trunc_f32(b, dtype, accs_out[acc_idx[(row, cb, mt)]])
                for q in range(QUADS):
                    wo = b.add(j0, b.add(b.const_i32(cb * AT + q * (AT // 4)), lane_j))
                    comps = [b.vec_extract(vals, 4 * q + e) for e in range(4)]
                    vec = b.vec_pack(comps, io)
                    d_el = b.add(row_off, wo)
                    full = b.land(hk_ok, b.cmp_lt(b.add(wo, b.const_i32(3)), p_Wo))
                    br = b.scf_if_else(full)
                    with br as (then_ctx, else_ctx):
                        with then_ctx:
                            _buf_store_vN(
                                b, dtype, y_rsrc, b.mul(d_el, c_eb), c0, vec, 2
                            )
                        with else_ctx:
                            for e in range(4):
                                we = b.add(wo, b.const_i32(e)) if e else wo
                                ok_e = b.land(hk_ok, b.cmp_lt(we, p_Wo))
                                de = b.add(d_el, b.const_i32(e)) if e else d_el
                                o = b.select(ok_e, b.mul(de, c_eb), oob)
                                if dtype == "bf16":
                                    b.buffer_store_bf16(y_rsrc, o, c0, comps[e])
                                else:
                                    b.buffer_store_f16(y_rsrc, o, c0, comps[e])
    return b.kernel


__all__: List[str] = [
    "DirectNchwFwdSpec",
    "build_direct_conv_nchw_fwd",
    "is_valid_direct_nchw_fwd_spec",
    "repack_weights_krsc",
    "with_conflict_free_pad",
    "x_read_conflict_ways",
]
