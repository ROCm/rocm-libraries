# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Experimental NCHW prototype of the non-grouped direct-conv forward kernel.

This is a **spike**, not a production kernel family: it measures whether a
direct conv that consumes/produces NCHW tensors natively (activation ``X`` in
NCHW, weights ``W`` in KCYX, output ``D`` in NKHW) can beat the
transpose -> NHWC-conv -> transpose pipeline that a framework boundary would
otherwise force. It is deliberately out of scope for the production
byte-identity contract:

* no C++ mirror (``library/cpp/`` has no sibling for this file);
* not byte-identical with anything, and not gated by
  ``tools/check_byte_identity.py``;
* not wired into ``dispatch``, the provider, or the AOT cache;
* not enrolled in pinned-reference coverage.

It reuses :mod:`kernels.common.conv_direct_nongrouped`'s one real invariant:
the MFMA compute loop (``emit_mfmas`` / ``read_a_frag`` / ``read_b_frag``)
only ever touches the two staged LDS tiles in ``[pos][c]`` (activation) /
fragment-slot (weight) order. That addressing is layout-agnostic, so it is
reproduced here unchanged; only three things differ from the NHWC original:

1. **X staging** (NCHW input): for a fixed ``(n, c, h)`` a run of ``w`` is
   contiguous in NCHW (unlike the channel axis the LDS destination iterates
   fastest), so each staging thread issues one ``XV_X``-wide vector load
   (``buffer_load_vN``) whenever the whole run is provably in-bounds
   (``scf_if_else``-guarded), falling back to the original per-element
   masked scalar loads only for runs that straddle a padding / image-edge
   boundary, plus a small scalar tail pass for the columns left over when
   the halo width isn't a multiple of ``XV_X``. The thread index is
   deliberately decomposed with the channel ``c`` fastest and the column
   group slowest (the opposite of every other staging pass in this family):
   boundary-ness depends only on ``(row, column-group)``, not on ``c``, so
   putting ``c`` in the fast slot keeps whole wavefronts agreeing on which
   branch to take as often as possible -- a wavefront that disagrees pays
   for both branches of the ``scf_if_else``. The LDS *destination* index
   (``pos * c_stride + c``) and the LDS store itself are unchanged (still
   one scalar store per element: the destinations of a vector's elements are
   ``c_stride`` apart, not contiguous, so only the *load* side vectorizes).
2. **W staging** (KCYX weights): for one ``(k, tap)`` the ``FRAG`` channels a
   lane needs are ``KH*KW`` elements apart in memory (KCYX puts the taps,
   not the channels, contiguous for fixed ``k``), so they are gathered with
   ``FRAG`` strided scalar loads and packed into the same fragment vector the
   NHWC kernel's vectorised KRSC load produces. The destination fragment-slot
   layout is untouched. Left as-is: see the optimization notes below.
   With ``weight_layout="krsc"`` the caller pre-packs the weights into the
   NHWC kernel's K,KH,KW,C order instead, and each fragment becomes one
   contiguous vector load exactly as in the NHWC original.
3. **Epilogue** (NKHW output): the accumulator-to-global store gets one extra
   LDS round-trip scoped to a single ``(atom_tile x atom_tile)`` output tile:
   each lane scatters its accumulator slice into a small per-wave scratch
   buffer in the K-fastest order the accumulator already has, a barrier
   publishes it, and then every lane reads back a contiguous ``ACC``-wide
   block of scratch (``idx = lane * ACC + sub``, which always shares one
   ``k_local`` because ``ACC`` divides ``atom_tile``) and stores it
   ``VEC_EPI`` elements at a time with ``buffer_store_vN`` into the NKHW
   output -- NKHW is w-fastest, so a block's elements are contiguous in
   ``D`` too. A block that would cross the output's right / bottom / channel
   edge falls back to the original per-element masked scalar stores via
   ``scf_if_else``. The barrier structure (two barriers per ``(mt, row,
   cb)`` step) is unchanged: batching more steps behind one barrier pair
   would need ``cb``-many times more scratch, and this prototype's default
   geometry is already close to the LDS budget at ``stride=2``.

Two opt-in knobs replace pieces 1 and 3. ``x_lds_layout="chw_tr"`` (gfx950)
stages X as ``[c][h][copy][w]`` with wide ``ds_write`` and builds the MFMA
fragments with ``ds_read_b64_tr_b16``. That read needs 8-byte-aligned lane
addresses, so every distinct tap column offset mod 4 gets its own row copy,
staged from a correspondingly shifted global load. At stride 2 every copy is
further split into even and odd column planes, so each tap reads a contiguous
run. ``epilogue="direct"`` swaps the MFMA operands (their per-lane layouts are
symmetric), so the accumulator rows are output positions and each lane stores
4 consecutive ``w`` of one ``k`` straight to NKHW.

To keep the spike small, this file drops the grouped-halo, split-C,
double-buffer and chiplet-swizzle knobs the production NHWC kernel has:
``groups == 1`` only, single-buffered, flat grid decode, fp16 / bf16 storage
only (no fp32 / TF32). Every geometry knob that is *not* about I/O layout
(tile sizes, ``ck``, waves, atom, ``lds_pad``) keeps the same meaning as
:mod:`kernels.common.conv_direct_nongrouped`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import IRBuilder, KernelDef, Value, VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _buf_load_vN,
    _buf_store_vN,
    _io_type,
    _trunc_f32,
    _trunc_f32_elem,
    emit_direct_params,
)

_NCHW_PROTO_DTYPES = ("fp16", "bf16")

# Same trick as the NHWC kernel: a masked load/store is redirected to a byte
# offset past every operand's ``num_records`` so the hardware drops it (a load
# returns zero, a store is not written). See conv_direct_nongrouped.py's
# module docstring for the derivation; the bound is reused verbatim here.
_OOB_BASE = 0x7F000000
_I32_MAX = (1 << 31) - 1

# Atom name -> (tile, K, frag); identical table to conv_direct_nongrouped.
_ATOMS = {
    "32x32x16": (32, 16, 8),
    "32x32x8": (32, 8, 4),
    "16x16x32": (16, 32, 8),
    "16x16x16": (16, 16, 4),
}

# The 16-byte alignment every ds_read_b128 / buffer dwordx4 needs, in
# elements, for a 16-bit element: the conflict-free LDS pad and ``ck``
# granularity the NHWC kernel also uses.
_ALIGN_ELEMS = 8

_WEIGHT_LAYOUTS = ("kcyx", "krsc")
_X_LDS_LAYOUTS = ("pos_c", "chw_tr")
_EPILOGUES = ("lds_transpose", "direct")


@dataclass(frozen=True)
class DirectNongroupedConvNchwProtoSpec:
    """One concrete NCHW-prototype direct-conv kernel configuration.

    ``problem.groups`` must be 1. Only ``problem.dtype in {"fp16", "bf16"}``
    is supported (no fp32 / TF32 storage in this prototype).
    """

    problem: DirectConvProblem
    name: str = "direct_conv_nongrouped_nchw_proto"

    tile_h: int = 16
    tile_w: int = 32
    tile_k: int = 128
    ck: int = 16

    waves_m: int = 2
    waves_n: int = 4

    atom: str = "32x32x16"
    wave_size: int = 64
    lds_pad: int = 8
    # "kcyx": weights as torch stores them (scalar strided gather per
    # fragment). "krsc": weights pre-packed host-side into the NHWC kernel's
    # K,KH,KW,C order, so each fragment is one contiguous vector load.
    weight_layout: str = "kcyx"
    # "pos_c": X staged into LDS as [pos][c] (per-element scatter).
    # "chw_tr": X staged as [c][h][w] with wide ds_writes; MFMA fragments are
    # assembled by ds_read_b64_tr_b16 (gfx950 only). At stride 2 every halo
    # row is de-interleaved into an even-column and an odd-column plane so
    # each tap reads a contiguous run.
    x_lds_layout: str = "pos_c"
    # "lds_transpose": accumulator -> LDS scratch -> w-contiguous stores.
    # "direct": MFMA operands swapped (X is A, W is B) so each lane's
    # accumulator holds 4 consecutive output w for one k: stored straight to
    # NKHW, no LDS round-trip, no epilogue barriers.
    epilogue: str = "lds_transpose"

    # ---- derived geometry -------------------------------------------------

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
        return (self.tile_h - 1) * p.stride + (p.KH - 1) * p.dil_h + 1

    @property
    def lds_in_w(self) -> int:
        p = self.problem
        return (self.tile_w - 1) * p.stride + (p.KW - 1) * p.dil_w + 1

    @property
    def c_stride(self) -> int:
        return self.ck + self.lds_pad

    @property
    def x_tr(self) -> bool:
        return self.x_lds_layout == "chw_tr"

    @property
    def x_n_vec(self) -> int:
        """chw_tr: 8-wide global load runs per halo row (per copy)."""
        return ceil_div(self.lds_in_w, _ALIGN_ELEMS)

    @property
    def x_tr_shifts(self) -> Tuple[int, ...]:
        """chw_tr: tap column offsets mod 4 (in plane units at stride 2).

        ds_read_b64_tr_b16 needs 8-byte-aligned lane addresses, so a tap whose
        column offset is not a multiple of 4 elements cannot read the shared
        row. Each distinct residue m gets its own copy of the row, staged from
        a global load shifted by m columns (m*stride at stride 2), so every
        LDS write and every tr read stays aligned."""
        p = self.problem
        offs = [s * p.dil_w // (2 if p.stride == 2 else 1) for s in range(p.KW)]
        return tuple(sorted({o % 4 for o in offs}))

    @property
    def x_copy_w(self) -> int:
        """chw_tr: width of one row copy (one parity plane at stride 2)."""
        w = self.x_n_vec * _ALIGN_ELEMS
        return w // 2 if self.problem.stride == 2 else w

    @property
    def x_row_stride(self) -> int:
        n = len(self.x_tr_shifts) * (2 if self.problem.stride == 2 else 1)
        return n * self.x_copy_w

    @property
    def x_ch_stride(self) -> int:
        """chw_tr: per-channel LDS stride, an odd multiple of 32 elements so
        the four channel rows one 16-lane tr16 group reads land on disjoint
        bank ranges."""
        s = ceil_div(self.lds_in_h * self.x_row_stride, 32) * 32
        return s + 32 if s % 64 == 0 else s

    @property
    def lds_x_elems(self) -> int:
        if self.x_tr:
            return self.ck * self.x_ch_stride
        return self.lds_in_h * self.lds_in_w * self.c_stride

    @property
    def lds_x_dump(self) -> int:
        return _ALIGN_ELEMS if self.x_tr else 1

    @property
    def lds_w_elems(self) -> int:
        p = self.problem
        return self.tile_k * p.KH * p.KW * self.ck

    @property
    def epilogue_scratch_elems(self) -> int:
        """Elements of the small per-wave accumulator-to-NKHW scratch tile."""
        if self.epilogue == "direct":
            return 0
        return self.waves_m * self.waves_n * self.atom_tile * self.atom_tile

    @property
    def lds_bytes(self) -> int:
        # +dump slot for X's last partial pass, +frag for W's.
        one = (
            self.lds_x_elems
            + self.lds_x_dump
            + self.lds_w_elems
            + self.frag
            + self.epilogue_scratch_elems
        )
        return 2 * one  # fp16 / bf16: 2 bytes per element

    @property
    def acc_vgprs(self) -> int:
        return self.m_tiles_per_wave * self.n_tiles_per_wave * self.acc_per_lane

    @property
    def tile_counts(self) -> Tuple[int, int, int]:
        p = self.problem
        return (
            ceil_div(p.Wo, self.tile_w),
            ceil_div(p.Ho, self.tile_h),
            ceil_div(p.total_k, self.tile_k),
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
            f"d{p.dil_h}x{p.dil_w}" if p.is_dilated else "",
            f"t{self.tile_h}x{self.tile_w}x{self.tile_k}",
            f"ck{self.ck}",
            f"w{self.waves_m}x{self.waves_n}",
            f"a{self.atom}",
            "wkrsc" if self.weight_layout == "krsc" else "",
            "xtr" if self.x_tr else "",
            "edir" if self.epilogue == "direct" else "",
            "bf16" if p.dtype == "bf16" else "",
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in _NCHW_PROTO_DTYPES:
            raise ValueError(
                f"DirectNongroupedConvNchwProtoSpec: unsupported dtype {p.dtype!r} "
                f"(expected one of {list(_NCHW_PROTO_DTYPES)}; this prototype is "
                f"16-bit storage only)"
            )
        if p.groups != 1:
            raise ValueError(
                f"DirectNongroupedConvNchwProtoSpec is the groups==1 family "
                f"(got groups={p.groups})"
            )
        if self.weight_layout not in _WEIGHT_LAYOUTS:
            raise ValueError(
                f"unknown weight_layout {self.weight_layout!r}; expected one of "
                f"{list(_WEIGHT_LAYOUTS)}"
            )
        if self.x_lds_layout not in _X_LDS_LAYOUTS:
            raise ValueError(
                f"unknown x_lds_layout {self.x_lds_layout!r}; expected one of "
                f"{list(_X_LDS_LAYOUTS)}"
            )
        if self.epilogue not in _EPILOGUES:
            raise ValueError(
                f"unknown epilogue {self.epilogue!r}; expected one of {list(_EPILOGUES)}"
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
            raise ValueError(
                "DirectNongroupedConvNchwProtoSpec: tile, wave, stride and "
                "channel parameters must be positive"
            )
        if p.dil_h < 1 or p.dil_w < 1:
            raise ValueError(
                f"DirectNongroupedConvNchwProtoSpec: dilation must be positive "
                f"(got {p.dil_h}x{p.dil_w})"
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
                f"tile_k/{t} = {self.m_tiles_total} not divisible by "
                f"waves_m {self.waves_m}"
            )
        if self.ck % self.atom_k != 0:
            raise ValueError(f"ck {self.ck} not divisible by atom K {self.atom_k}")
        if self.ck % _ALIGN_ELEMS != 0:
            raise ValueError(f"ck {self.ck} must be a multiple of {_ALIGN_ELEMS}")
        if p.cpg % self.ck != 0:
            raise ValueError(f"C {p.cpg} not divisible by ck {self.ck}")
        if p.total_k % t != 0:
            raise ValueError(f"K {p.total_k} must be a multiple of {t}")
        if self.threads_per_block > 1024:
            raise ValueError(f"threads_per_block {self.threads_per_block} > 1024")
        if self.lds_pad < 0 or self.lds_pad % _ALIGN_ELEMS != 0:
            raise ValueError(
                f"lds_pad must be a non-negative multiple of {_ALIGN_ELEMS} "
                f"(got {self.lds_pad})"
            )
        if self.grid()[0] > _I32_MAX:
            raise ValueError(
                f"flattened grid of {self.grid()[0]} workgroups exceeds {_I32_MAX}"
            )
        eb = 2
        a_bytes = eb * p.N * p.H * p.W * p.total_c
        b_bytes = eb * p.total_k * p.KH * p.KW * p.cpg
        d_bytes = eb * p.N * p.Ho * p.Wo * p.total_k
        if max(a_bytes, b_bytes, d_bytes) > _OOB_BASE:
            raise ValueError(
                f"tensors must fit below the {_OOB_BASE:#x}-byte masked-access "
                f"offset (A {a_bytes} B, B {b_bytes} B, D {d_bytes} B)"
            )
        # The X staging chunk advance is ck * stride_c elements, with
        # stride_c == H * W (the NCHW channel stride, concrete here since
        # the spec's problem fixes H/W). The W staging chunk advance is
        # ck * KH * KW elements (KCYX's channel stride).
        x_tail = eb * self.ck * p.H * p.W
        w_tail = eb * self.ck * (1 if self.weight_layout == "krsc" else p.KH * p.KW)
        if _OOB_BASE + max(x_tail, w_tail) > _I32_MAX:
            raise ValueError(
                f"ck {self.ck} too large: the masked-load offset plus the "
                f"channel-chunk advance would overflow i32"
            )
        if self.acc_vgprs > 256:
            raise ValueError(
                f"accumulator tile needs {self.acc_vgprs} registers per lane "
                f"(max 256): shrink tile_k/tile_h/tile_w or add waves"
            )


def is_valid_nchw_proto_spec(
    spec: DirectNongroupedConvNchwProtoSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch`` without raising."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:  # pragma: no cover - arch table lookup
        return False, str(e)
    try:
        spec.validate()
        p = spec.problem
        if p.stride not in (1, 2):
            return False, f"stride {p.stride} is not supported (expected 1 or 2)"
        t, k = spec.atom_tile, spec.atom_k
        ab = "bf16" if p.dtype == "bf16" else "f16"
        if not target.mma.has_shape(
            a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=t, n=t, k=k
        ):
            return False, f"missing mfma_f32_{spec.atom}_{ab} on {arch}"
        if spec.x_tr and arch != "gfx950":
            return False, "x_lds_layout='chw_tr' needs ds_read_b64_tr_b16 (gfx950 only)"
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


def build_direct_conv_nongrouped_nchw_proto(
    spec: DirectNongroupedConvNchwProtoSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR of one NCHW-prototype non-grouped direct-conv kernel.

    Mirrors :func:`kernels.common.conv_direct_nongrouped.build_direct_conv_nongrouped`'s
    overall shape (stage X and W into LDS, run the MFMA channel loop, epilogue
    to global); see the module docstring for the three places this differs.
    """
    ok, why = is_valid_nchw_proto_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectNongroupedConvNchwProtoSpec for {arch}: {why}")

    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    EB = 2

    KH, KW, S, PAD = p.KH, p.KW, p.stride, p.PAD
    DH, DW = p.dil_h, p.dil_w
    N_TAPS = KH * KW
    W_KRSC = spec.weight_layout == "krsc"

    TH, TW, TK, CK = spec.tile_h, spec.tile_w, spec.tile_k, spec.ck
    THREADS = spec.threads_per_block
    WAVE = spec.wave_size
    FRAG = spec.frag
    AK = spec.atom_k
    AT = spec.atom_tile
    ACC = spec.acc_per_lane
    QUADS = ACC // 4
    NCB = spec.n_col_blocks
    M_TILES = spec.m_tiles_total
    KATOMS = spec.k_atoms
    ROWS_W = spec.rows_per_wave
    MT_W = spec.m_tiles_per_wave
    NT_W = spec.n_tiles_per_wave
    CSTRIDE = spec.c_stride
    LDS_IN_H, LDS_IN_W = spec.lds_in_h, spec.lds_in_w

    # X staging is split into a vectorised bulk pass (contiguous runs of
    # XV_X columns along w, the axis NCHW keeps contiguous for fixed
    # (n, c, h)) and a scalar tail pass for the columns left over when
    # LDS_IN_W isn't a multiple of XV_X. 8 elements is the widest vector this
    # IR builder's buffer_load_vN_f16/bf16 supports (dwords=4).
    X_TR = spec.x_tr
    EPI_DIRECT = spec.epilogue == "direct"
    XV_X = 8
    if X_TR:
        # chw_tr stages whole 8-wide runs up to the padded row width: no tail.
        VEC_W = spec.x_n_vec
        TAIL_W = 0
        ROW_STRIDE = spec.x_row_stride
        COPY_W = spec.x_copy_w
        CH_STRIDE = spec.x_ch_stride
        TR_SHIFTS = spec.x_tr_shifts
    else:
        VEC_W = LDS_IN_W // XV_X
        TAIL_W = LDS_IN_W % XV_X
    X_VEC_SLOTS = CK * LDS_IN_H * VEC_W
    X_TAIL_SLOTS = CK * LDS_IN_H * TAIL_W
    X_VEC_PASSES_1 = ceil_div(X_VEC_SLOTS, THREADS) if X_VEC_SLOTS else 0
    X_VEC_PASSES = X_VEC_PASSES_1 * (len(TR_SHIFTS) if X_TR else 1)
    X_TAIL_PASSES = ceil_div(X_TAIL_SLOTS, THREADS) if X_TAIL_SLOTS else 0
    X_PASSES = X_VEC_PASSES + X_TAIL_PASSES
    W_SLOTS = N_TAPS * M_TILES * KATOMS * WAVE
    W_PASSES = ceil_div(W_SLOTS, THREADS)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    params = emit_direct_params(b, io_type=io_type)
    A = params["A"]
    Bp = params["B"]
    D = params["D"]
    A_bytes = params["A_bytes"]
    B_bytes = params["B_bytes"]
    D_bytes = params["D_bytes"]
    p_Hi = params["p_Hi"]
    p_Wi = params["p_Wi"]
    p_Ho = params["p_Ho"]
    p_Wo = params["p_Wo"]
    # groups == 1: the totals are the full channel counts. The NHWC / NHWK
    # stride kernargs in ``params`` are left unused: NCHW / NKHW strides are
    # computed on-device below from extents that are already runtime values.
    C = params["p_total_c"]
    K = params["p_total_k"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_half = b.const_i32(EB)
    oob_base = b.const_i32(_OOB_BASE)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # ---- NCHW / KCYX / NKHW strides, computed from runtime extents --------
    stride_h_a = p_Wi
    stride_c_a = b.mul(p_Hi, p_Wi)
    stride_n_a = b.mul(C, stride_c_a)

    stride_h_d = p_Wo
    stride_k_d = b.mul(p_Ho, p_Wo)
    stride_n_d = b.mul(K, stride_k_d)

    # KCYX: for one k_out, C * KH * KW contiguous elements.
    w_per_k_elems = b.mul(C, b.const_i32(KH * KW))

    # ---- LDS allocations ---------------------------------------------------
    # X's destination array is strided by CSTRIDE (CK padded by lds_pad), not
    # by CK: the array must hold the padding read_b_frag's fragment reads
    # stride over, even though only CK elements are staged per pixel.
    x_stage = spec.lds_x_elems + spec.lds_x_dump  # dump slot for a partial last pass
    w_stage = spec.lds_w_elems + FRAG  # +1 fragment dump slot
    X_smem = b.smem_alloc(io_type, [1, x_stage], name_hint="lds_x")
    W_smem = b.smem_alloc(io_type, [1, w_stage], name_hint="lds_w")
    EPI_smem = None
    if not EPI_DIRECT:
        EPI_smem = b.smem_alloc(
            io_type, [1, spec.epilogue_scratch_elems], name_hint="lds_epi"
        )
    x_dump = spec.lds_x_elems
    w_dump = spec.lds_w_elems

    # ---- thread / wave decomposition ---------------------------------------
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(WAVE))
    wave_id = b.div(tid, b.const_i32(WAVE))
    wave_m = b.div(wave_id, b.const_i32(spec.waves_n))
    wave_n = b.mod(wave_id, b.const_i32(spec.waves_n))
    lane_lo = b.mod(lane, b.const_i32(AT))
    lane_hi = b.div(lane, b.const_i32(AT))

    # ---- grid decode (flat, no chiplet swizzle) -----------------------------
    n_wt = b.div(b.add(p_Wo, b.const_i32(TW - 1)), b.const_i32(TW))
    n_ht = b.div(b.add(p_Ho, b.const_i32(TH - 1)), b.const_i32(TH))
    n_kt = b.div(b.add(K, b.const_i32(TK - 1)), b.const_i32(TK))
    n_hw = b.mul(n_wt, n_ht)

    wgid = b.block_id_x()
    k_tile = b.mod(wgid, n_kt)
    cell = b.div(wgid, n_kt)
    n_img = b.div(cell, n_hw)
    st = b.mod(cell, n_hw)
    w_tile = b.mod(st, n_wt)
    h_tile = b.div(st, n_wt)

    out_h0 = b.mul(h_tile, b.const_i32(TH))
    out_w0 = b.mul(w_tile, b.const_i32(TW))
    k_base = b.mul(k_tile, b.const_i32(TK))

    in_h0 = b.sub(b.mul(out_h0, b.const_i32(S)), b.const_i32(PAD))
    in_w0 = b.sub(b.mul(out_w0, b.const_i32(S)), b.const_i32(PAD))

    # ---- scalar dtype dispatch ----------------------------------------------
    def _scalar_load(rsrc: Value, voff: Value, soff: Value) -> Value:
        if dtype == "bf16":
            return b.buffer_load_bf16(rsrc, voff, soff)
        return b.buffer_load_f16(rsrc, voff, soff)

    def _scalar_store(rsrc: Value, voff: Value, soff: Value, val: Value) -> None:
        if dtype == "bf16":
            b.buffer_store_bf16(rsrc, voff, soff, val)
        else:
            b.buffer_store_f16(rsrc, voff, soff, val)

    # ---- X staging metadata (change 1: NCHW source addressing) -------------
    # For a fixed (n, c, h), w is contiguous: a run of XV_X columns can be
    # fetched with one vector load whenever the whole run stays in-bounds
    # (no padding / image-edge crossing mid-vector). The destination LDS
    # index (``pos * CSTRIDE + c``) is exactly what the NHWC kernel writes;
    # only the *source* addressing differs, and only here. Columns that
    # don't fill a whole XV_X run (the row's tail) keep the original
    # one-scalar-per-thread-pass path.

    def _x_vec_pass_meta():
        """Per-pass data for one XV_X-wide run of contiguous w columns.

        Returns a list of ``(fully_valid, elems0, ok_h, iw0, dests)``: the
        source address of the run's first element (unscaled, element
        units), a flag that's true iff every one of the XV_X elements is
        in-bounds, the row validity / first column (reused to rebuild
        per-element validity in the scalar fallback), and the XV_X LDS
        destination indices.

        Thread index decomposes with ``c`` fastest, ``colg`` slowest (unlike
        every other staging pass in this family, which puts the destination's
        outermost index first): ``fully_valid`` depends only on
        ``(ih_l, colg)``, not on ``c``, so putting ``c`` (``CK`` values) in
        the fast slot keeps whole wavefronts agreeing on it whenever
        possible -- a wavefront that disagrees pays for *both* branches of
        the ``scf_if_else`` below, which would otherwise cancel out most of
        the vector load's win on the boundary tiles this prototype's small
        benchmark shapes are full of.
        """
        meta = []
        img_base = b.mul(n_img, stride_n_a)
        for j in range(X_VEC_PASSES):
            mi, jj = divmod(j, X_VEC_PASSES_1)
            v = b.add(tid, b.const_i32(jj * THREADS))
            c = b.mod(v, b.const_i32(CK))
            rest = b.div(v, b.const_i32(CK))
            ih_l = b.mod(rest, b.const_i32(LDS_IN_H))
            colg = b.div(rest, b.const_i32(LDS_IN_H))
            iw_l0 = b.mul(colg, b.const_i32(XV_X))
            if X_TR and TR_SHIFTS[mi]:
                iw_l0 = b.add(iw_l0, b.const_i32(TR_SHIFTS[mi] * S))
            ih = b.add(in_h0, ih_l)
            iw0 = b.add(in_w0, iw_l0)
            ok_h = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, p_Hi))
            iw_hi = b.add(iw0, b.const_i32(XV_X - 1))
            fully_valid = b.land(ok_h, b.land(b.cmp_ge(iw0, c0), b.cmp_lt(iw_hi, p_Wi)))
            if X_VEC_SLOTS % THREADS != 0:
                in_tile = b.cmp_lt(v, b.const_i32(X_VEC_SLOTS))
                fully_valid = b.land(fully_valid, in_tile)
            else:
                in_tile = None
            dests = []
            if X_TR:
                # [c][h][copy][w] destination: the run lands contiguously in
                # copy mi (stride 1) or as 4 even + 4 odd columns in copy
                # mi's two parity planes (stride 2).
                n_par = 2 if S == 2 else 1
                row0 = b.add(
                    b.mul(c, b.const_i32(CH_STRIDE)),
                    b.mul(ih_l, b.const_i32(ROW_STRIDE)),
                )
                d0 = b.add(
                    row0,
                    b.add(
                        b.mul(colg, b.const_i32(XV_X // n_par)),
                        b.const_i32(mi * n_par * COPY_W),
                    ),
                )
                dests.append(d0)
                if S == 2:
                    dests.append(b.add(d0, b.const_i32(COPY_W)))
                if in_tile is not None:
                    dests = [b.select(in_tile, d, b.const_i32(x_dump)) for d in dests]
            else:
                pos0 = b.add(b.mul(ih_l, b.const_i32(LDS_IN_W)), iw_l0)
                lds_idx0 = b.add(b.mul(pos0, b.const_i32(CSTRIDE)), c)
                for i in range(XV_X):
                    d = (
                        lds_idx0
                        if i == 0
                        else b.add(lds_idx0, b.const_i32(i * CSTRIDE))
                    )
                    if in_tile is not None:
                        d = b.select(in_tile, d, b.const_i32(x_dump))
                    dests.append(d)
            elems0 = b.add(b.add(img_base, b.mul(ih, stride_h_a)), b.mul(c, stride_c_a))
            elems0 = b.add(elems0, iw0)
            meta.append((fully_valid, elems0, ok_h, iw0, dests))
        return meta

    def _x_tail_pass_meta():
        """Per-pass data for the scalar leftover columns of each row."""
        meta = []
        img_base = b.mul(n_img, stride_n_a)
        row_slots = LDS_IN_H * TAIL_W
        for j in range(X_TAIL_PASSES):
            v = b.add(tid, b.const_i32(j * THREADS))
            c = b.div(v, b.const_i32(row_slots))
            rem = b.mod(v, b.const_i32(row_slots))
            ih_l = b.div(rem, b.const_i32(TAIL_W))
            iw_l = b.add(b.mod(rem, b.const_i32(TAIL_W)), b.const_i32(VEC_W * XV_X))
            ih = b.add(in_h0, ih_l)
            iw = b.add(in_w0, iw_l)
            ok_h = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, p_Hi))
            ok_w = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, p_Wi))
            valid = b.land(ok_h, ok_w)
            if X_TAIL_SLOTS % THREADS != 0:
                in_tile = b.cmp_lt(v, b.const_i32(X_TAIL_SLOTS))
                valid = b.land(valid, in_tile)
            else:
                in_tile = None
            pix_off = b.add(b.mul(ih, stride_h_a), iw)
            elems = b.add(b.add(img_base, pix_off), b.mul(c, stride_c_a))
            pos = b.add(b.mul(ih_l, b.const_i32(LDS_IN_W)), iw_l)
            lds_idx = b.add(b.mul(pos, b.const_i32(CSTRIDE)), c)
            if in_tile is not None:
                lds_idx = b.select(in_tile, lds_idx, b.const_i32(x_dump))
            base = b.select(valid, b.mul(elems, c_half), oob_base)
            meta.append((base, lds_idx))
        return meta

    # ---- W staging metadata (change 2: KCYX source addressing) --------------
    # Destination decode is byte-identical to conv_direct_nongrouped's
    # non-grouped, non-fp32 path: slot g picks exactly the same
    # (tap, m_tile, k_atom, lane) as the NHWC kernel. The source gathers
    # FRAG channels KH*KW elements apart (KCYX keeps the taps, not the
    # channels, contiguous for one k) with FRAG strided scalar loads instead
    # of the NHWC kernel's one contiguous KRSC vector load.
    def _w_pass_meta():
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
                b.add(b.mul(m_tile, b.const_i32(AT)), b.mod(g_lane, b.const_i32(AT))),
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
            lds_idx = b.mul(g, b.const_i32(FRAG))
            if in_tile is not None:
                lds_idx = b.select(in_tile, lds_idx, b.const_i32(w_dump))
            rs = b.add(b.mul(r, b.const_i32(KW)), s)
            if W_KRSC:
                # ((k_out*KH + r)*KW + s)*C + c_in_chunk, as in the NHWC kernel.
                elems = b.add(
                    b.mul(b.add(b.mul(k_out, b.const_i32(N_TAPS)), rs), C), c_in_chunk
                )
                meta.append((valid, elems, None, lds_idx))
                continue
            elems0 = b.add(b.mul(k_out, w_per_k_elems), rs)
            meta.append((valid, elems0, c_in_chunk, lds_idx))
        return meta

    x_vec_meta = _x_vec_pass_meta()
    x_tail_meta = _x_tail_pass_meta()
    w_meta = _w_pass_meta()

    # Per-chunk channel advance, in bytes. X's channel stride is stride_c_a
    # (runtime, NCHW); W's is KH*KW elements (build-time, KCYX) -- the two
    # layouts disagree on how far one channel is, so each tensor gets its own
    # chunk-advance value (the NHWC original could share one because both its
    # tensors keep channel at stride 1).
    x_chunk_bytes = b.mul(b.mul(b.const_i32(CK), stride_c_a), c_half)
    w_chunk_bytes = b.const_i32(CK * EB if W_KRSC else CK * KH * KW * EB)

    def issue_stage_loads(c_off_x: Value, c_off_w: Value):
        xs = []
        for fully_valid, elems0, ok_h, iw0, _dests in x_vec_meta:
            addr = b.add(b.mul(elems0, c_half), c_off_x)
            branch = b.scf_if_else(
                fully_valid, result_types=[VectorType(io_type, XV_X)]
            )
            with branch as (then_ctx, else_ctx):
                with then_ctx:
                    b.scf_yield(_buf_load_vN(b, dtype, a_rsrc, addr, c0, XV_X // 2))
                with else_ctx:
                    parts = []
                    for i in range(XV_X):
                        iw_i = b.add(iw0, b.const_i32(i)) if i else iw0
                        valid_i = b.land(
                            ok_h, b.land(b.cmp_ge(iw_i, c0), b.cmp_lt(iw_i, p_Wi))
                        )
                        e_i = b.add(elems0, b.const_i32(i)) if i else elems0
                        voff = b.add(
                            b.select(valid_i, b.mul(e_i, c_half), oob_base), c_off_x
                        )
                        parts.append(_scalar_load(a_rsrc, voff, c0))
                    b.scf_yield(b.vec_pack(parts, io_type))
            xs.append(branch.results[0])
        for base, _ in x_tail_meta:
            xs.append(_scalar_load(a_rsrc, b.add(base, c_off_x), c0))
        ws = []
        for valid, elems0, c_in_chunk, _ in w_meta:
            if W_KRSC:
                voff = b.add(b.select(valid, b.mul(elems0, c_half), oob_base), c_off_w)
                ws.append(_buf_load_vN(b, dtype, b_rsrc, voff, c0, FRAG // 2))
                continue
            parts = []
            for h in range(FRAG):
                c_h = b.add(c_in_chunk, b.const_i32(h)) if h else c_in_chunk
                e = b.add(elems0, b.mul(c_h, b.const_i32(KH * KW)))
                voff = b.add(b.select(valid, b.mul(e, c_half), oob_base), c_off_w)
                parts.append(_scalar_load(b_rsrc, voff, c0))
            ws.append(b.vec_pack(parts, io_type))
        return xs, ws

    def commit_stage(xs, ws):
        for i, (_, _, _, _, dests) in enumerate(x_vec_meta):
            val = xs[i]
            if X_TR:
                if S == 1:
                    b.smem_store_vN(X_smem, [c0, dests[0]], val, XV_X)
                else:
                    for par in range(2):
                        half = b.vec_pack(
                            [b.vec_extract(val, 2 * e + par) for e in range(XV_X // 2)],
                            io_type,
                        )
                        b.smem_store_vN(X_smem, [c0, dests[par]], half, XV_X // 2)
                continue
            for k in range(XV_X):
                b.smem_store_vN(X_smem, [c0, dests[k]], b.vec_extract(val, k), 1)
        for j, (_, lds_idx) in enumerate(x_tail_meta):
            b.smem_store_vN(X_smem, [c0, lds_idx], xs[X_VEC_PASSES + j], 1)
        for (_, _, _, lds_idx), val in zip(w_meta, ws):
            b.smem_store_vN(W_smem, [c0, lds_idx], val, FRAG)

    # ---- per-lane LDS read bases (unchanged from conv_direct_nongrouped) ---
    x_wave_term = b.mul(
        b.mul(wave_n, b.const_i32(ROWS_W * S)), b.const_i32(LDS_IN_W * CSTRIDE)
    )
    x_lane_term = b.add(
        b.mul(lane_lo, b.const_i32(S * CSTRIDE)), b.mul(lane_hi, b.const_i32(FRAG))
    )
    x_read_base = b.add(x_wave_term, x_lane_term)
    w_read_base = b.add(
        b.mul(wave_m, b.const_i32(MT_W * KATOMS * WAVE * FRAG)),
        b.mul(lane, b.const_i32(FRAG)),
    )

    def read_a_frag(base: Value, tap: int, mt: int, katom: int) -> Value:
        off = ((tap * M_TILES + mt) * KATOMS + katom) * WAVE * FRAG
        idx = b.add(base, b.const_i32(off))
        return b.smem_load_vN(W_smem, c0, idx, dtype=io_type, n=FRAG)

    if X_TR:
        # ds_read_b64_tr_b16: within each 16-lane group, input lane i reads 4
        # consecutive w of channel row (i // 4) at column offset (i % 4) * 4;
        # output lane n then holds element r = column n of channel row r. So
        # lane l ends up with 4 channels (one per r) at position
        # (lane_lo // 16) * 16 + l % 16 -- the MFMA operand layout, with
        # channel = katom*AK + lane_hi*FRAG + read*4 + r, matching W's
        # c_in_chunk + j ordering (j = read*4 + r).
        lane16 = b.mod(lane, b.const_i32(16))
        x_tr_ch = b.add(
            b.mul(lane_hi, b.const_i32(FRAG)), b.div(lane16, b.const_i32(4))
        )
        x_tr_col = b.add(
            b.mul(b.div(lane_lo, b.const_i32(16)), b.const_i32(16)),
            b.mul(b.mod(lane16, b.const_i32(4)), b.const_i32(4)),
        )
        x_read_base = b.add(
            b.add(
                b.mul(x_tr_ch, b.const_i32(CH_STRIDE)),
                b.mul(wave_n, b.const_i32(ROWS_W * S * ROW_STRIDE)),
            ),
            x_tr_col,
        )

    def read_b_frag(base: Value, in_row: int, cb: int, s: int, katom: int) -> Value:
        if X_TR:
            sd = s * DW
            if S == 1:
                m = sd % 4
                col = TR_SHIFTS.index(m) * COPY_W + cb * AT + sd - m
            else:
                q = sd // 2
                m = q % 4
                col = (2 * TR_SHIFTS.index(m) + sd % 2) * COPY_W + cb * AT + q - m
            parts = []
            for rd in range(FRAG // 4):
                off = (katom * AK + rd * 4) * CH_STRIDE + in_row * ROW_STRIDE + col
                idx = b.add(base, b.const_i32(off))
                parts.append(b.ds_read_tr16_b64(X_smem, c0, idx, dtype=io_type))
            return parts[0] if len(parts) == 1 else b.vec_concat(parts[0], parts[1])
        pos = in_row * LDS_IN_W + cb * AT * S + s * DW
        off = pos * CSTRIDE + katom * AK
        idx = b.add(base, b.const_i32(off))
        return b.smem_load_vN(X_smem, c0, idx, dtype=io_type, n=FRAG)

    # ---- main channel loop (reuses the NHWC kernel's MFMA compute loop) ----
    zero_acc = b.zero_vec_f32(ACC)
    n_acc = MT_W * NT_W
    n_in_rows = (ROWS_W - 1) * S + (KH - 1) * DH + 1

    def _mfma(shape: str, a: Value, b_val: Value, acc: Value) -> Value:
        fn = getattr(b, f"mfma_f32_{shape}_{'bf16' if dtype == 'bf16' else 'f16'}")
        return fn(a, b_val, acc)

    def emit_mfmas(accs, x_base, w_base):
        """One channel chunk of MFMAs against the staged tiles.

        This is :func:`conv_direct_nongrouped.build_direct_conv_nongrouped`'s
        ``emit_mfmas`` for the 16-bit path, unchanged: it reads only from the
        two staged LDS tiles ([pos][c] for X, fragment-slot order for W), so
        it never sees the NCHW / KCYX layout change above it.
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
                        bf_row = b_frags[row * S + r_c * DH]
                        for cb in range(NCB):
                            nt = row * NCB + cb
                            for mt in range(MT_W):
                                idx = mt * NT_W + nt
                                # A/B per-lane layouts are symmetric, so the
                                # direct epilogue just swaps them: rows of
                                # the accumulator become output positions.
                                if EPI_DIRECT:
                                    accs[idx] = _mfma(
                                        spec.atom, bf_row[cb], a_frags[mt], accs[idx]
                                    )
                                else:
                                    accs[idx] = _mfma(
                                        spec.atom, a_frags[mt], bf_row[cb], accs[idx]
                                    )

    xs0, ws0 = issue_stage_loads(c0, c0)

    iter_args = [(f"acc{i}", zero_acc) for i in range(n_acc)]
    iter_args += [(f"xs{i}", v) for i, v in enumerate(xs0)]
    iter_args += [(f"ws{i}", v) for i, v in enumerate(ws0)]

    n_chunks = b.smax(b.div(C, b.const_i32(CK)), c1)
    loop = b.scf_for_iter(
        c0, n_chunks, c1, iter_args, iv_name="c_iter", elide_trailing_barrier=False
    )
    with loop as (c_iv, carried):
        accs = list(carried[:n_acc])
        xs = list(carried[n_acc : n_acc + X_PASSES])
        ws = list(carried[n_acc + X_PASSES :])

        b.sync(fence="workgroup")
        commit_stage(xs, ws)
        b.sync(fence="workgroup")
        c_next = b.add(c_iv, c1)
        xs_n, ws_n = issue_stage_loads(
            b.mul(c_next, x_chunk_bytes), b.mul(c_next, w_chunk_bytes)
        )
        emit_mfmas(accs, x_read_base, w_read_base)
        b.scf_yield(*accs, *xs_n, *ws_n)

    accs_out = list(loop.results[:n_acc])

    if EPI_DIRECT:
        # Swapped operands: lane holds k = lane_lo and, for each quad q, the
        # 4 consecutive output columns q*(AT//4) + lane_hi*4 + 0..3 -- one
        # 8-byte store into w-contiguous NKHW, masked scalars on edge quads.
        k_tile_base = b.add(k_base, b.mul(wave_m, b.const_i32(MT_W * AT)))
        row_base = b.add(out_h0, b.mul(wave_n, b.const_i32(ROWS_W)))
        img_out_base = b.mul(n_img, stride_n_d)
        lane_w = b.mul(lane_hi, b.const_i32(4))
        for mt in range(MT_W):
            k_out = b.add(k_tile_base, b.const_i32(mt * AT) if mt else c0)
            k_out = b.add(k_out, lane_lo)
            k_ok = b.cmp_lt(k_out, K)
            k_off = b.add(img_out_base, b.mul(k_out, stride_k_d))
            for row in range(ROWS_W):
                out_h = b.add(row_base, b.const_i32(row))
                hk_ok = b.land(k_ok, b.cmp_lt(out_h, p_Ho))
                row_off = b.add(k_off, b.mul(out_h, stride_h_d))
                for cb in range(NCB):
                    acc = accs_out[mt * NT_W + row * NCB + cb]
                    vals_all = _trunc_f32(b, dtype, acc)
                    for q in range(QUADS):
                        out_w = b.add(
                            out_w0, b.add(b.const_i32(cb * AT + q * (AT // 4)), lane_w)
                        )
                        vals = b.vec_pack(
                            [b.vec_extract(vals_all, 4 * q + j) for j in range(4)],
                            io_type,
                        )
                        d_elems = b.add(row_off, out_w)
                        full = b.land(
                            hk_ok, b.cmp_lt(b.add(out_w, b.const_i32(3)), p_Wo)
                        )
                        branch = b.scf_if_else(full)
                        with branch as (then_ctx, else_ctx):
                            with then_ctx:
                                _buf_store_vN(
                                    b,
                                    dtype,
                                    d_rsrc,
                                    b.mul(d_elems, c_half),
                                    c0,
                                    vals,
                                    2,
                                )
                            with else_ctx:
                                for j in range(4):
                                    ow = b.add(out_w, b.const_i32(j)) if j else out_w
                                    ok = b.land(hk_ok, b.cmp_lt(ow, p_Wo))
                                    de = (
                                        b.add(d_elems, b.const_i32(j)) if j else d_elems
                                    )
                                    off = b.select(ok, b.mul(de, c_half), oob_base)
                                    _scalar_store(
                                        d_rsrc, off, c0, b.vec_extract(vals, j)
                                    )
        return b.kernel

    # ---- epilogue (change 3: NKHW destination, K-fastest -> W-fastest) -----
    # The accumulator keeps the NHWC kernel's layout exactly (K-fastest: slot
    # i -> row = (i//4)*(AT//4) + lane_hi*4 + (i%4), col = lane_lo). Rather
    # than reshape the MFMA / accumulator order, each lane scatters its
    # ACC values into a small per-wave LDS scratch tile (scoped to one
    # atom_tile x atom_tile output patch, not the whole output), a barrier
    # publishes it, and every lane then reads back a *different* (k, w)
    # assignment and stores it into the NKHW output. Two barriers bracket
    # every (mt, row, cb) step since the lane that reads a scratch slot is
    # essentially never the lane that wrote it.
    #
    # The read phase assigns each lane a *contiguous* block of ACC scratch
    # slots (idx = lane*ACC + sub) instead of striding by WAVE: since ACC
    # always divides AT for the atoms this prototype supports, a lane's
    # whole ACC-wide block shares one k_local and walks consecutive w_local
    # values, so it can be read back and stored VEC_EPI elements at a time
    # (one ds_read + one buffer_store_vN) instead of one scalar store per
    # element -- the NKHW output is w-fastest, so those VEC_EPI elements are
    # also contiguous in D. A block that would cross the output's right /
    # bottom / channel edge falls back to the original per-element masked
    # scalar stores.
    VEC_EPI = min(8, ACC)
    epi_wave_base = b.mul(wave_id, b.const_i32(AT * AT))
    k_tile_base = b.add(k_base, b.mul(wave_m, b.const_i32(MT_W * AT)))
    row_base = b.add(out_h0, b.mul(wave_n, b.const_i32(ROWS_W)))
    img_out_base = b.mul(n_img, stride_n_d)

    for mt in range(MT_W):
        k_mt_base = b.add(k_tile_base, b.const_i32(mt * AT))
        for row in range(ROWS_W):
            out_h = b.add(row_base, b.const_i32(row))
            h_ok = b.cmp_lt(out_h, p_Ho)
            row_off_d = b.add(img_out_base, b.mul(out_h, stride_h_d))
            for cb in range(NCB):
                nt = row * NCB + cb
                acc = accs_out[mt * NT_W + nt]
                out_w0_cb = b.add(out_w0, b.const_i32(cb * AT))

                # Write phase: scatter the accumulator (K-fastest) into the
                # scratch tile at [k_local][w_local], w_local = lane_lo fixed.
                for q in range(QUADS):
                    k_local = b.add(
                        b.const_i32(q * (AT // 4)), b.mul(lane_hi, b.const_i32(4))
                    )
                    for j in range(4):
                        kl = b.add(k_local, b.const_i32(j)) if j else k_local
                        val = _trunc_f32_elem(b, dtype, b.vec_extract(acc, 4 * q + j))
                        epi_idx = b.add(
                            epi_wave_base,
                            b.add(b.mul(kl, b.const_i32(AT)), lane_lo),
                        )
                        b.smem_store_vN(EPI_smem, [c0, epi_idx], val, 1)
                b.sync(fence="workgroup")

                # Read phase: lane `lane` owns the contiguous scratch block
                # [lane*ACC, lane*ACC + ACC), split into ACC // VEC_EPI
                # vector-wide groups; every group shares one k_local (ACC
                # divides AT) and walks VEC_EPI consecutive w_local values.
                idx_lane = b.mul(lane, b.const_i32(ACC))
                for g in range(ACC // VEC_EPI):
                    idx0 = b.add(idx_lane, b.const_i32(g * VEC_EPI)) if g else idx_lane
                    k_local = b.div(idx0, b.const_i32(AT))
                    w_local0 = b.mod(idx0, b.const_i32(AT))
                    k_out = b.add(k_mt_base, k_local)
                    out_w0_g = b.add(out_w0_cb, w_local0)
                    epi_idx0 = b.add(
                        epi_wave_base, b.add(b.mul(k_local, b.const_i32(AT)), w_local0)
                    )
                    out_w_hi = b.add(out_w0_g, b.const_i32(VEC_EPI - 1))
                    fully_valid = b.land(
                        h_ok, b.land(b.cmp_lt(out_w_hi, p_Wo), b.cmp_lt(k_out, K))
                    )
                    d_elems0 = b.add(
                        b.add(row_off_d, out_w0_g), b.mul(k_out, stride_k_d)
                    )
                    branch = b.scf_if_else(fully_valid)
                    with branch as (then_ctx, else_ctx):
                        with then_ctx:
                            vals = b.smem_load_vN(
                                EPI_smem, c0, epi_idx0, dtype=io_type, n=VEC_EPI
                            )
                            d_off0 = b.mul(d_elems0, c_half)
                            _buf_store_vN(
                                b, dtype, d_rsrc, d_off0, c0, vals, VEC_EPI // 2
                            )
                        with else_ctx:
                            for e in range(VEC_EPI):
                                epi_idx = (
                                    b.add(epi_idx0, b.const_i32(e)) if e else epi_idx0
                                )
                                val = b.vec_extract(
                                    b.smem_load_vN(
                                        EPI_smem, c0, epi_idx, dtype=io_type, n=1
                                    ),
                                    0,
                                )
                                out_w = (
                                    b.add(out_w0_g, b.const_i32(e)) if e else out_w0_g
                                )
                                valid = b.land(
                                    h_ok,
                                    b.land(b.cmp_lt(out_w, p_Wo), b.cmp_lt(k_out, K)),
                                )
                                d_elems = (
                                    b.add(d_elems0, b.const_i32(e)) if e else d_elems0
                                )
                                d_off = b.select(
                                    valid, b.mul(d_elems, c_half), oob_base
                                )
                                _scalar_store(d_rsrc, d_off, c0, val)
                b.sync(fence="workgroup")

    return b.kernel
