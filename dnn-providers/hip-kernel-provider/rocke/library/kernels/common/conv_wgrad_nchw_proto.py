# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Experimental native-NCHW backward-weight (wgrad) convolution prototype.

This is a **research spike**, not a production kernel family. It measures
whether a wgrad kernel that consumes NCHW ``X`` and NKHW ``dY`` directly and
writes ``dW`` in torch's native KCYX order can beat the pipeline an NCHW caller
has today (transpose both activations to NHWC, run the production NHWC wgrad,
repack dW KYXC -> KCYX). Out of scope, like the forward NCHW spike
(:mod:`kernels.common.conv_direct_nongrouped_nchw_proto`):

* no C++ mirror, not byte-identical with anything, not gated by
  ``tools/check_byte_identity.py``;
* not wired into ``dispatch``, the provider, or the AOT cache;
* every problem extent is compiled in (the production kernel takes them as
  kernargs), ``groups == 1``, no dilation, fp16 / bf16 only.

GEMM view
---------
::

    dW[k, c, y, x] = sum_{n, ho, wo} dY[n, k, ho, wo] * X[n, c, ho*S-P+y, wo*S-P+x]

    M = K (output channels)   N = C * KH * KW   reduction R = N * Ho * Wo

The production NHWC wgrad has both operands contiguous along the GEMM *free*
axis (``k`` for dY, ``c`` for X) and strided along the reduction, so the MFMA
feed needs a transpose (``ds_read_b64_tr_b16`` on gfx950, see
``builders/gfx950/conv_wgrad/wgrad_lds_layout_case_study.md``). In NCHW both
operands are contiguous along the **reduction**:

* ``dY[n, k, :, :]`` is one contiguous run of ``Ho*Wo`` reduction elements, so
  an 8-element run of the reduction is one aligned ``dwordx4`` load;
* ``X[n, c, ih, :]`` for one filter tap ``(y, x)`` is contiguous along ``wo``
  at stride 1 (shifted by ``d = x - P`` elements, so the load is 2-byte
  aligned when ``d`` is odd) and every other element at stride 2.

Both LDS tiles are therefore stored reduction-contiguous (``[row][r]``): one
``ds_write_b128`` per staged run and one ``ds_read_b128`` per MFMA operand
fragment, with no transpose anywhere.

Workgroup tile
--------------
The GEMM-N tile is ``taps x tile_c``: one workgroup owns ``tile_c`` input
channels for all ``KW`` filter columns of ``KH / y_groups`` filter rows. Each
MFMA column block is then a single tap, and the filter column -- with it the
column shift ``d`` -- is a compile-time constant for every X staging pass.
That keeps the padding fix-up a constant shuffle instead of a runtime-amount
shift; the filter row only moves the input row, so ``y_groups > 1`` spreads
the rows over workgroups (more, smaller tiles) at no such cost.

Padding / stride on the vector axis
-----------------------------------
NCHW's vector axis ``w`` is also the axis padding and stride act on (the
known catch from the forward spike). A thread stages ``L = 8*S`` consecutive
input columns starting at ``S*wo0 + d``. When that span leaves the image row
(left padding, right padding, or the tensor end) the thread instead loads the
aligned window ``[S*wo0, S*wo0 + L)`` -- which always lies inside the row when
``S*Wo == W`` -- and picks the needed elements out of it with zeros for the
padding. The two variants differ only by the address and a per-lane
``select`` of two constant shuffles; there is no divergent branch and no
scalar fallback. Rows above / below the image (padding in ``h``) and the
reduction tail are a whole-run predicate onto an out-of-range offset that the
buffer hardware reads as zero.

At stride 2 each staged run needs 16 consecutive elements (two ``dwordx4``)
from which the even ones are kept: the stride costs twice the global load
bytes for X, not LDS.

Split-K / epilogue
------------------
The reduction ``R`` is split into ``split_k`` contiguous slices along
``blockIdx.z``; ``split_k`` is a kernarg, so one binary serves every degree.
Every slice packed-atomic-adds its partial into a caller-zeroed ``dW`` with
``global_atomic_pk_add_{f16,bf16}``, exactly like the production NHWC
single-stage split-K epilogue (same numerics: each slice rounds its partial to
16 bits before the add). For one ``(k, c)`` the ``KH*KW`` taps are contiguous
in KCYX; the pair alignment depends on the parity of ``c*KH*KW``, so each lane
picks between the two compile-time pairings with a ``select``. The caller
must zero ``dW`` before every launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.ir import BF16, F16, I32, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join

from kernels.common.conv_direct_grouped import DirectConvProblem

_DTYPES = ("fp16", "bf16")

# Masked loads are redirected to this byte offset, past every operand's
# num_records, so the hardware returns zero (same trick as the forward
# prototype and conv_direct_nongrouped).
_OOB_BASE = 0x7F000000
_I32_MAX = (1 << 31) - 1

# Atom name -> (tile edge, reduction depth). Both gfx950 16-bit atoms take an
# 8-element operand fragment per lane: lane l holds rows (l % edge) and
# reduction elements (l // edge) * 8 .. + 7.
_ATOMS = {
    "32x32x16": (32, 16),
    "16x16x32": (16, 32),
}
_FRAG = 8
# One staged run: 8 reduction elements = one 16-byte dwordx4 / ds_write_b128.
_RUN = 8


@dataclass(frozen=True)
class WgradNchwProtoSpec:
    """One concrete native-NCHW wgrad prototype configuration.

    ``tile_m`` output channels x (``KH/y_groups*KW`` taps x ``tile_c`` input
    channels) per workgroup, ``tile_r`` reduction elements per K step.
    ``waves_m`` splits ``tile_m``; ``waves_n`` splits the workgroup's taps.
    """

    problem: DirectConvProblem
    name: str = "conv_wgrad_nchw_proto"
    tile_m: int = 64
    tile_c: int = 32
    tile_r: int = 64
    waves_m: int = 2
    waves_n: int = 1
    atom: str = "32x32x16"
    wave_size: int = 64
    # LDS row padding (elements) of both reduction-contiguous tiles.
    lds_pad: int = 8
    # Split the KH filter rows over this many workgroups (grid x). A filter
    # row only moves the input row, so the row offset may be a runtime value;
    # the column shift (filter column) stays compile-time. More, smaller
    # tiles: parallelism without a deeper split-K (and its extra atomics).
    y_groups: int = 1

    # ---- derived ----------------------------------------------------------

    @property
    def taps(self) -> int:
        return self.problem.KH * self.problem.KW

    @property
    def rows_per_group(self) -> int:
        return self.problem.KH // self.y_groups

    @property
    def wg_taps(self) -> int:
        """Filter taps one workgroup owns: ``rows_per_group * KW``, contiguous in KCYX."""
        return self.rows_per_group * self.problem.KW

    @property
    def atom_tile(self) -> int:
        return _ATOMS[self.atom][0]

    @property
    def atom_k(self) -> int:
        return _ATOMS[self.atom][1]

    @property
    def runs(self) -> int:
        return self.tile_r // _RUN

    @property
    def threads(self) -> int:
        return self.waves_m * self.waves_n * self.wave_size

    @property
    def row_stride(self) -> int:
        return self.tile_r + self.lds_pad

    @property
    def m_atoms_per_wave(self) -> int:
        return self.tile_m // (self.waves_m * self.atom_tile)

    @property
    def taps_per_wave(self) -> int:
        return self.wg_taps // self.waves_n

    @property
    def c_blocks(self) -> int:
        return self.tile_c // self.atom_tile

    @property
    def acc_per_lane(self) -> int:
        return self.atom_tile * self.atom_tile // self.wave_size

    @property
    def acc_vgprs(self) -> int:
        return (
            self.m_atoms_per_wave
            * self.taps_per_wave
            * self.c_blocks
            * self.acc_per_lane
        )

    @property
    def a_passes(self) -> int:
        return self.tile_m * self.runs // self.threads

    @property
    def b_passes_per_tap(self) -> int:
        return self.tile_c * self.runs // self.threads

    @property
    def lds_elems(self) -> int:
        return (self.tile_m + self.wg_taps * self.tile_c) * self.row_stride

    @property
    def lds_bytes(self) -> int:
        return 2 * self.lds_elems

    @property
    def reduction(self) -> int:
        p = self.problem
        return p.N * p.Ho * p.Wo

    @property
    def k_steps(self) -> int:
        return ceil_div(self.reduction, self.tile_r)

    def grid(self, split_k: int) -> Tuple[int, int, int]:
        p = self.problem
        return (
            p.total_c // self.tile_c * self.y_groups,
            p.total_k // self.tile_m,
            split_k,
        )

    def kernel_name(self) -> str:
        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_c}x{self.tile_r}",
            f"w{self.waves_m}x{self.waves_n}",
            f"a{self.atom}",
            f"p{self.lds_pad}",
            f"yg{self.y_groups}" if self.y_groups > 1 else "",
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
        if self.atom not in _ATOMS:
            raise ValueError(
                f"unknown atom {self.atom!r}; expected one of {list(_ATOMS)}"
            )
        S = p.stride
        if p.Wo % _RUN != 0:
            raise ValueError(f"Wo {p.Wo} must be a multiple of {_RUN}")
        if S * p.Wo != p.W:
            # The edge fallback loads the aligned window [S*wo0, S*wo0 + 8*S);
            # S*Wo == W keeps that window inside the image row.
            raise ValueError(f"needs stride*Wo == W (got {S}*{p.Wo} != {p.W})")
        if p.PAD >= _RUN or (p.KW - 1 - p.PAD) >= _RUN:
            raise ValueError("|x - pad| must stay below 8 for every tap")
        if p.total_c % 2 != 0:
            raise ValueError("C must be even (packed 16-bit dW atomics)")
        t = self.atom_tile
        if self.tile_r % _RUN or self.tile_r % self.atom_k:
            raise ValueError(
                f"tile_r {self.tile_r} must be a multiple of {_RUN} and of the atom "
                f"depth {self.atom_k}"
            )
        if self.tile_c % t or p.total_c % self.tile_c:
            raise ValueError(
                f"tile_c {self.tile_c} must divide C and be a multiple of {t}"
            )
        if self.tile_m % (self.waves_m * t) or p.total_k % self.tile_m:
            raise ValueError(
                f"tile_m {self.tile_m} must divide K and be a multiple of waves_m*{t}"
            )
        if self.y_groups < 1 or p.KH % self.y_groups:
            raise ValueError(f"y_groups {self.y_groups} must divide KH = {p.KH}")
        if self.wg_taps % self.waves_n:
            raise ValueError(
                f"waves_n {self.waves_n} must divide the workgroup's taps {self.wg_taps}"
            )
        if self.threads % self.runs:
            raise ValueError("threads must be a multiple of tile_r/8")
        if (self.tile_m * self.runs) % self.threads:
            raise ValueError(
                "tile_m * tile_r / 8 must be a multiple of the thread count"
            )
        if (self.tile_c * self.runs) % self.threads:
            # Keeps every X staging pass on one tap (compile-time column shift).
            raise ValueError(
                "tile_c * tile_r / 8 must be a multiple of the thread count"
            )
        if self.lds_pad < 0 or self.lds_pad % _RUN:
            raise ValueError("lds_pad must be a non-negative multiple of 8")
        if self.acc_vgprs > 256:
            raise ValueError(
                f"{self.acc_vgprs} accumulator registers per lane (max 256)"
            )
        eb = 2
        big = max(
            eb * p.N * p.total_c * p.H * p.W,
            eb * p.N * p.total_k * p.Ho * p.Wo,
            eb * p.total_k * p.total_c * self.taps,
        )
        if big + 64 > _OOB_BASE:
            raise ValueError("tensors must fit below the masked-load offset")


def is_valid_wgrad_nchw_proto_spec(
    spec: WgradNchwProtoSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """``(ok, reason)`` for ``spec`` on ``arch`` without raising."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:  # pragma: no cover - arch table lookup
        return False, str(e)
    try:
        spec.validate()
        t, k = spec.atom_tile, spec.atom_k
        ab = "bf16" if spec.problem.dtype == "bf16" else "f16"
        if not target.mma.has_shape(
            a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=t, n=t, k=k
        ):
            return False, f"missing mfma_f32_{spec.atom}_{ab} on {arch}"
        if spec.wave_size != target.wave_size:
            return (
                False,
                f"wave_size {spec.wave_size} != {arch} wave {target.wave_size}",
            )
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


def wgrad_nchw_proto_signature(spec: WgradNchwProtoSpec) -> list:
    dt = "bf16" if spec.problem.dtype == "bf16" else "f16"
    return (
        SignatureBuilder()
        .ptr("dY", dt)
        .ptr("X", dt)
        .ptr("dW", dt)
        .scalar("dY_bytes", "i32")
        .scalar("X_bytes", "i32")
        .scalar("split_k", "i32")
        .build()
    )


def wgrad_nchw_proto_launch_values(
    dY_ptr: int, X_ptr: int, dW_ptr: int, dY_bytes: int, X_bytes: int, split_k: int
) -> dict:
    return {
        "dY": int(dY_ptr),
        "X": int(X_ptr),
        "dW": int(dW_ptr),
        "dY_bytes": int(dY_bytes),
        "X_bytes": int(X_bytes),
        "split_k": int(split_k),
    }


def build_conv_wgrad_nchw_proto(
    spec: WgradNchwProtoSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR of one native-NCHW wgrad prototype kernel."""
    ok, why = is_valid_wgrad_nchw_proto_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid WgradNchwProtoSpec for {arch}: {why}")

    p = spec.problem
    dtype = p.dtype
    io = BF16 if dtype == "bf16" else F16
    EB = 2
    C, K, H, W = p.total_c, p.total_k, p.H, p.W
    Ho, Wo, S, P = p.Ho, p.Wo, p.stride, p.PAD
    KH, KW = p.KH, p.KW
    T = KH * KW
    YG = spec.rows_per_group  # filter rows per workgroup
    HoWo = Ho * Wo
    HW = H * W
    R = spec.reduction
    L = _RUN * S  # staged input columns per run

    TM, TC, TR = spec.tile_m, spec.tile_c, spec.tile_r
    RUNS = spec.runs
    THREADS = spec.threads
    WAVE = spec.wave_size
    AT, AK = spec.atom_tile, spec.atom_k
    RS = spec.row_stride
    MT_W = spec.m_atoms_per_wave
    TPW = spec.taps_per_wave
    CB = spec.c_blocks
    ACC = spec.acc_per_lane
    A_PASSES = spec.a_passes
    B_PPT = spec.b_passes_per_tap
    ROWS_PER_PASS = THREADS // RUNS  # tile rows one pass covers

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    dY = b.param("dY", PtrType(io, "global"), noalias=True, readonly=True, align=16)
    X = b.param("X", PtrType(io, "global"), noalias=True, readonly=True, align=16)
    dW = b.param("dW", PtrType(io, "global"), noalias=True, align=4)
    dY_bytes = b.param("dY_bytes", I32)
    X_bytes = b.param("X_bytes", I32)
    split_k = b.param("split_k", I32)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0 = ci(0)
    c1 = ci(1)
    oob = ci(_OOB_BASE)
    dy_rsrc = b.buffer_rsrc(dY, dY_bytes)
    x_rsrc = b.buffer_rsrc(X, X_bytes)

    lds = b.smem_alloc(io, [1, spec.lds_elems], name_hint="lds_ab")
    B_BASE = TM * RS  # B tile follows the A tile

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(WAVE))
    wave_id = b.div(tid, ci(WAVE))
    wave_m = b.div(wave_id, ci(spec.waves_n))
    wave_n = b.mod(wave_id, ci(spec.waves_n))
    lane_lo = b.mod(lane, ci(AT))
    lane_hi = b.div(lane, ci(AT))

    bx = b.block_id_x()
    if spec.y_groups > 1:
        y_grp = b.mod(bx, ci(spec.y_groups))
        c_tile = b.div(bx, ci(spec.y_groups))
        y0 = b.mul(y_grp, ci(YG))  # first filter row of this workgroup
    else:
        c_tile = bx
        y0 = c0
    y0_mp = b.sub(y0, ci(P))
    m_tile = b.block_id_y()
    z = b.block_id_z()
    c_base = b.mul(c_tile, ci(TC))
    m_base = b.mul(m_tile, ci(TM))

    # ---- reduction slice of this split -------------------------------------
    steps_total = ci(spec.k_steps)
    per = b.div(b.add(steps_total, b.sub(split_k, c1)), split_k)
    k_lo = b.mul(z, per)
    k_hi = b.smin(b.add(k_lo, per), steps_total)

    # ---- per-thread staging invariants --------------------------------------
    run = b.mod(tid, ci(RUNS))
    row_in_pass = b.div(tid, ci(RUNS))
    run_off = b.mul(run, ci(_RUN))
    # dY: (m_base + row) * HoWo, per pass + j * ROWS_PER_PASS * HoWo.
    a_thread = b.mul(b.add(m_base, row_in_pass), ci(HoWo))
    # X: (c_base + row) * HW, per pass + jj * ROWS_PER_PASS * HW.
    b_thread = b.mul(b.add(c_base, row_in_pass), ci(HW))
    a_lds_thread = b.add(b.mul(row_in_pass, ci(RS)), run_off)

    # Column shift of each filter column x, and whether a staged run for it can
    # leave the image row at all (d == 0 at stride 1 never can).
    d_of = [x - P for x in range(KW)]

    def _even_pick(v: Value, idxs: List[int]) -> Value:
        zero = b.vec_extract(b.zero_vec(io, 2), 0)
        parts = [b.vec_extract(v, i) if 0 <= i < L else zero for i in idxs]
        return b.vec_pack(parts, io)

    def issue_loads(step: Value):
        """Global loads of one K step: A runs, then B runs (raw, unshuffled)."""
        r0 = b.add(b.mul(step, ci(TR)), run_off)
        ok_r = b.cmp_lt(r0, ci(R))
        n_img = b.div(r0, ci(HoWo))
        pos = b.mod(r0, ci(HoWo))
        ho = b.div(pos, ci(Wo))
        wo0 = b.mod(pos, ci(Wo))
        a_step = b.add(b.mul(n_img, ci(K * HoWo)), pos)
        a_vals = []
        for j in range(A_PASSES):
            e = b.add(b.add(a_thread, a_step), ci(j * ROWS_PER_PASS * HoWo))
            off = b.select(ok_r, b.mul(e, ci(EB)), oob)
            a_vals.append(b.buffer_load_vN(dy_rsrc, off, c0, io, _RUN))
        # X rows for each filter row y, and the edge flag for each column x.
        img = b.add(b.mul(n_img, ci(C * HW)), b_thread)
        col0 = b.mul(wo0, ci(S))  # aligned window start
        row_ok = []
        row_base = []
        ho_s = b.add(b.mul(ho, ci(S)), y0_mp)
        for y in range(YG):
            ih = b.add(ho_s, ci(y)) if y else ho_s
            okh = b.land(ok_r, b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, ci(H))))
            row_ok.append(okh)
            row_base.append(b.add(img, b.add(b.mul(ih, ci(W)), col0)))
        edge = []
        for x in range(KW):
            d = d_of[x]
            can_left = d < 0
            can_right = d + L > S * _RUN  # span end past the aligned window end
            if not (can_left or can_right):
                edge.append(None)
                continue
            start = b.add(col0, ci(d))
            conds = []
            if can_left:
                conds.append(b.cmp_lt(start, c0))
            if can_right:
                conds.append(b.cmp_gt(b.add(start, ci(L)), ci(W)))
            e_x = conds[0] if len(conds) == 1 else b.lor(conds[0], conds[1])
            edge.append(e_x)
        b_vals = []
        for y in range(YG):
            for x in range(KW):
                d = d_of[x]
                for jj in range(B_PPT):
                    e = b.add(row_base[y], ci(jj * ROWS_PER_PASS * HW))
                    if d != 0:
                        shifted = b.add(e, ci(d))
                        e = (
                            shifted
                            if edge[x] is None
                            else b.select(edge[x], e, shifted)
                        )
                    off = b.select(row_ok[y], b.mul(e, ci(EB)), oob)
                    v = b.buffer_load_vN(x_rsrc, off, c0, io, _RUN)
                    if S == 2:
                        v2 = b.buffer_load_vN(
                            x_rsrc, b.add(off, ci(_RUN * EB)), c0, io, _RUN
                        )
                        v = b.vec_concat(v, v2)
                    b_vals.append(v)
        return a_vals, b_vals, edge

    def commit(a_vals, b_vals, edge) -> None:
        for j, v in enumerate(a_vals):
            idx = b.add(a_lds_thread, ci(j * ROWS_PER_PASS * RS))
            b.smem_store_vN(lds, [c0, idx], v, _RUN)
        i = 0
        for y in range(YG):
            for x in range(KW):
                d = d_of[x]
                tap = y * KW + x
                for jj in range(B_PPT):
                    v = b_vals[i]
                    i += 1
                    if S == 2:
                        main = _even_pick(v, [S * j for j in range(_RUN)])
                    else:
                        main = v
                    if edge[x] is not None:
                        alt = _even_pick(v, [d + S * j for j in range(_RUN)])
                        main = b.select(edge[x], alt, main)
                    row = tap * TC + jj * ROWS_PER_PASS
                    idx = b.add(a_lds_thread, ci(B_BASE + row * RS))
                    b.smem_store_vN(lds, [c0, idx], main, _RUN)

    # ---- MFMA fragment reads ------------------------------------------------
    frag_k = b.mul(lane_hi, ci(_FRAG))
    a_read = b.add(b.mul(b.add(b.mul(wave_m, ci(MT_W * AT)), lane_lo), ci(RS)), frag_k)
    b_read = b.add(
        b.add(b.mul(b.mul(wave_n, ci(TPW * TC)), ci(RS)), b.mul(lane_lo, ci(RS))),
        b.add(frag_k, ci(B_BASE)),
    )
    mfma = getattr(b, f"mfma_f32_{spec.atom}_{'bf16' if dtype == 'bf16' else 'f16'}")

    def emit_mfmas(accs: List[Value]) -> None:
        for ka in range(TR // AK):
            a_fr = [
                b.smem_load_vN(
                    lds,
                    c0,
                    b.add(a_read, ci(mt * AT * RS + ka * AK)),
                    dtype=io,
                    n=_FRAG,
                )
                for mt in range(MT_W)
            ]
            for t in range(TPW):
                for cb in range(CB):
                    off = (t * TC + cb * AT) * RS + ka * AK
                    b_fr = b.smem_load_vN(
                        lds, c0, b.add(b_read, ci(off)), dtype=io, n=_FRAG
                    )
                    for mt in range(MT_W):
                        i = (mt * TPW + t) * CB + cb
                        accs[i] = mfma(a_fr[mt], b_fr, accs[i])

    n_acc = MT_W * TPW * CB
    a0, bv0, edge0 = issue_loads(k_lo)
    n_a, n_b = len(a0), len(bv0)
    iter_args = [(f"acc{i}", b.zero_vec_f32(ACC)) for i in range(n_acc)]
    iter_args += [(f"a{i}", v) for i, v in enumerate(a0)]
    iter_args += [(f"b{i}", v) for i, v in enumerate(bv0)]
    # The edge flags depend on the step, so they ride along as i32 0/1.
    edge_idx = [x for x in range(KW) if edge0[x] is not None]
    iter_args += [(f"e{x}", b.select(edge0[x], c1, c0)) for x in edge_idx]

    loop = b.scf_for_iter(
        k_lo, k_hi, c1, iter_args, iv_name="kstep", elide_trailing_barrier=False
    )
    with loop as (kv, carried):
        accs = list(carried[:n_acc])
        a_cur = list(carried[n_acc : n_acc + n_a])
        b_cur = list(carried[n_acc + n_a : n_acc + n_a + n_b])
        e_cur = list(carried[n_acc + n_a + n_b :])
        edge_cur: list = [None] * KW
        for x, ev in zip(edge_idx, e_cur):
            edge_cur[x] = b.cmp_ne(ev, c0)
        b.sync(fence="workgroup")
        commit(a_cur, b_cur, edge_cur)
        b.sync(fence="workgroup")
        a_n, b_n, edge_n = issue_loads(b.add(kv, c1))
        emit_mfmas(accs)
        b.scf_yield(*accs, *a_n, *b_n, *[b.select(edge_n[x], c1, c0) for x in edge_idx])
    accs_out = list(loop.results[:n_acc])

    # ---- epilogue: packed 16-bit atomics into KCYX dW ------------------------
    # Lane holds, per (mt, cb) and accumulator slot i, dW[k, c, tap] for the
    # wave's taps [t0, t0 + TPW) with k = row(i), c = lane_lo column. The KCYX
    # flat index k*C*T + c*T + tap; C*T is even, so the pair alignment parity
    # is that of c*T. Two compile-time pairings, one per parity, picked per
    # lane; the shorter list is padded with a zero-valued duplicate.
    t0 = b.add(b.mul(y0, ci(KW)), b.mul(wave_n, ci(TPW)))
    tap_lo = 0  # taps are wave-relative below; the absolute tap is t0 + t

    def _pairs(par: int) -> List[int]:
        # Pair starts s (relative tap, may be -1) whose flat parity is even
        # when the lane's (c*T + t0) parity is ``par``.
        out = []
        s = tap_lo - 1
        while s < TPW:
            if (par + s) % 2 == 0 and (s + 1 >= 0) and s < TPW:
                out.append(s)
            s += 1
        return out

    pairs0, pairs1 = _pairs(0), _pairs(1)
    n_pairs = max(len(pairs0), len(pairs1))
    pairs0 += [pairs0[0]] * (n_pairs - len(pairs0))
    pairs1 += [pairs1[0]] * (n_pairs - len(pairs1))
    pad0 = [False] * len(_pairs(0)) + [True] * (n_pairs - len(_pairs(0)))
    pad1 = [False] * len(_pairs(1)) + [True] * (n_pairs - len(_pairs(1)))

    def _trunc(v: Value) -> Value:
        return b.trunc_f32_to_bf16(v) if dtype == "bf16" else b.trunc_f32_to_f16(v)

    zero16 = b.vec_extract(b.zero_vec(io, 2), 0)
    has_work = b.cmp_lt(k_lo, k_hi)
    with b.scf_if(has_work):
        m_wave = b.add(m_base, b.mul(wave_m, ci(MT_W * AT)))
        for cb in range(CB):
            c = b.add(c_base, b.add(ci(cb * AT), lane_lo))
            ct = b.add(b.mul(c, ci(T)), t0)
            par = b.mod(ct, ci(2))
            odd = b.cmp_ne(par, c0)
            for mt in range(MT_W):
                for i in range(ACC):
                    row = (i // 4) * (AT // 4) + (i % 4)
                    k = b.add(b.add(m_wave, ci(mt * AT + row)), b.mul(lane_hi, ci(4)))
                    flat0 = b.add(b.mul(k, ci(C * T)), ct)
                    vals = [
                        _trunc(b.vec_extract(accs_out[(mt * TPW + t) * CB + cb], i))
                        for t in range(TPW)
                    ]

                    def _v(t: int) -> Value:
                        return vals[t] if 0 <= t < TPW else zero16

                    for q in range(n_pairs):
                        s0, s1 = pairs0[q], pairs1[q]
                        lo0 = zero16 if pad0[q] else _v(s0)
                        hi0 = zero16 if pad0[q] else _v(s0 + 1)
                        lo1 = zero16 if pad1[q] else _v(s1)
                        hi1 = zero16 if pad1[q] else _v(s1 + 1)
                        lo = b.select(odd, lo1, lo0)
                        hi = b.select(odd, hi1, hi0)
                        idx = b.select(odd, b.add(flat0, ci(s1)), b.add(flat0, ci(s0)))
                        pk = b.vec_pack([lo, hi], io)
                        if dtype == "bf16":
                            b.global_atomic_add_pk_bf16(dW, idx, pk)
                        else:
                            b.global_atomic_add_pk_f16(dW, idx, pk)
    return b.kernel


__all__ = [
    "WgradNchwProtoSpec",
    "build_conv_wgrad_nchw_proto",
    "is_valid_wgrad_nchw_proto_spec",
    "wgrad_nchw_proto_launch_values",
    "wgrad_nchw_proto_signature",
]
