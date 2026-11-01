# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""3-D Winograd convolution F(2x2x2, 3x3x3), bf16 NDHWC, stride 1 (research prototype).

Operation: ``Y[z,y,x,k] = sum_{c,dz,dy,dx} X[z+dz, y+dy, x+dx, c] * W[k,dz,dy,dx,c]``
(N=1, valid conv: the halo form, ``Do = Di - 2`` ...; every output extent even).

Winograd F(2,3) per axis: ``Y_tile = A^T [ (G g G^T) . (B^T d B) ] A`` on 4x4x4 input
tiles (stride 2) producing 2x2x2 outputs; 64 transform points ``p = (a, b, c)``, each an
independent GEMM ``M_p[tile, k] = sum_c V_p[tile, c] U_p[c, k]`` -- 64 multiplies per 8
outputs instead of 216.

Kernel (one launch, everything fused):

* Workgroup = 4 waves, ``TILES_PER_WG = 32`` output tiles x ``N_T = 32`` output channels.
  Wave ``w`` owns the 16 transform points whose z-index ``a == w`` and keeps one
  ``32x32`` f32 accumulator (``mfma_f32_32x32x8_bf16``) per point: 256 f32 per lane.
* Input transform in registers, straight into the MFMA A layout: lane ``l`` owns tile
  ``l % 32`` and channels ``4*(l//32) .. +3`` of each 8-channel chunk; it loads the two
  z-planes its ``a`` needs (4x4 positions x 4 channels each), combines them (z axis),
  then applies ``B^T`` along y and x in f32 and rounds to bf16 -- 16 A fragments.
  The sign of the z-combination for ``a == 2`` (``d2 - d1``) is folded into the
  transformed weights, so every wave computes ``d[p0] + r * d[p1]`` with ``r = +-1``.
* B fragments (transformed weights ``U[p][k][c]``, c contiguous) are loaded from global
  memory directly in the MFMA B layout (8 bytes per lane).
* Epilogue: each wave applies ``A^T`` along y and x in registers (16 points -> 2x2
  outputs per accumulator slot); waves 1 and 2 (``a = 1, 2``) stage their partials in
  LDS, wave 0 writes the ``oz = 0`` plane (``P0 + P1 + P2``) and wave 3 the ``oz = 1``
  plane (``P1 - P2 - P3``).

The weight transform ``U = s_a * (G (x) G (x) G) g`` is a separate (small) step,
``winograd3d_weight_transform`` (torch reference) -- it is part of the timed call.

Accuracy (bf16 transformed operands, f32 transforms and accumulation): peak-normalised
max error 5-6e-3 on uniform(-1,1) data, against 2-4e-3 for the direct conv (the
benchmark's bf16 tolerance is 5e-2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType, VectorType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.spec import SignatureBuilder, kernel_name_join

# Winograd F(2,3) matrices.
BT = ((1, 0, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, 0, -1))
G = ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.0, 0.0, 1.0))
AT = ((1, 1, 1, 0), (0, 1, -1, -1))
# z-axis combination per wave a: V'_a = d[P0] + R * d[P1]; true V_a = S0 * V'_a.
_ZP0 = (0, 1, 1, 1)
_ZP1 = (2, 2, 2, 3)
_ZR = (-1.0, 1.0, -1.0, -1.0)
ZS0 = (1.0, 1.0, -1.0, 1.0)


@dataclass(frozen=True)
class Winograd3dSpec:
    """One F(2x2x2,3x3x3) forward problem (N=1, NDHWC, valid conv, stride 1)."""

    Di: int
    Hi: int
    Wi: int
    C: int
    K: int
    tiles_per_wg: int = 32
    n_tile: int = 32
    c_chunk: int = 8
    # "ndhwc": X / U read in place (v1; 8-byte lane loads 2*C bytes apart -- poorly
    # coalesced). "c8": X blocked as [C/8][D][H][W][8], U as [64][C/8][K][8] (lanes of a
    # load are contiguous). The relayout of X is a separate pass (winograd3d_x_to_c8).
    layout: str = "ndhwc"
    # v2 main loop (c8 layout only): buffer loads with uniform soffsets, next chunk's raw
    # input and weight fragments prefetched into registers one iteration ahead, bf16
    # rounding round-half-up packed with v_perm (2 VALU per value pair + 1 add each).
    pipe: str = "v1"
    # pf: which operands are prefetched one chunk ahead: "xb" (raw input + weights) or
    # "x" (raw input only; the weight fragments load at the top of their own iteration).
    pf_ops: str = "xb"
    name: str = "conv3d_wino"

    @property
    def Do(self) -> int:
        return self.Di - 2

    @property
    def Ho(self) -> int:
        return self.Hi - 2

    @property
    def Wo(self) -> int:
        return self.Wi - 2

    @property
    def TZ(self) -> int:
        return self.Do // 2

    @property
    def TY(self) -> int:
        return self.Ho // 2

    @property
    def TX(self) -> int:
        return self.Wo // 2

    @property
    def tiles(self) -> int:
        return self.TZ * self.TY * self.TX

    @property
    def block_size(self) -> int:
        return 256

    @property
    def flops(self) -> int:
        """Direct-conv FLOPs (the 'effective TF' numerator)."""
        return 2 * self.C * self.K * 27 * self.Do * self.Ho * self.Wo

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"D{self.Di}H{self.Hi}W{self.Wi}C{self.C}K{self.K}",
            "bf16",
            f"t{self.tiles_per_wg}n{self.n_tile}c{self.c_chunk}",
            *(["xc8"] if self.layout == "c8" else []),
            *([self.pipe] if self.pipe != "v1" else []),
            *([f"pf{self.pf_ops}"] if self.pipe == "pf" and self.pf_ops != "xb" else []),
        )


def is_valid_spec(spec: Winograd3dSpec) -> Tuple[bool, str]:
    if min(spec.Do, spec.Ho, spec.Wo) <= 0:
        return False, "empty output"
    if spec.Do % 2 or spec.Ho % 2 or spec.Wo % 2:
        return False, "F(2,3) needs even output extents"
    if spec.tiles_per_wg != 32 or spec.n_tile != 32 or spec.c_chunk != 8:
        return False, "v1 supports tiles_per_wg=32, n_tile=32, c_chunk=8 only"
    if spec.tiles % spec.tiles_per_wg:
        return False, f"tiles {spec.tiles} not a multiple of {spec.tiles_per_wg}"
    if spec.layout not in ("ndhwc", "c8"):
        return False, f"unknown layout {spec.layout!r}"
    if spec.pipe not in ("v1", "pf"):
        return False, f"unknown pipe {spec.pipe!r}"
    if spec.pipe != "v1" and spec.layout != "c8":
        return False, "pipe 'pf' needs layout 'c8'"
    if spec.K % spec.n_tile or spec.C % spec.c_chunk:
        return False, "K / C not multiples of the tile"
    return True, "ok"


def winograd3d_grid(spec: Winograd3dSpec) -> Tuple[int, int, int]:
    return (spec.tiles // spec.tiles_per_wg, spec.K // spec.n_tile, 1)


def winograd3d_signature(spec: Winograd3dSpec):
    return (
        SignatureBuilder().ptr("X", "bf16").ptr("U", "bf16").ptr("Y", "bf16").build()
    )


def _main_loop_v1(
    b, spec, atom, X, U, off0, off1, r, u_base, vox, chunk_stride, u_point, u_chunk,
    n_chunks,
):
    ci = b.const_i32
    Wi = spec.Wi
    acc0 = [(f"acc{i}", atom.zero_acc(b)) for i in range(16)]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), acc0, iv_name="cc")
    with loop as (cc, accs):
        c0 = b.mul(cc, ci(chunk_stride))
        cu = b.mul(cc, ci(u_chunk))
        # Raw loads: d[pl][y][x] = <4 x bf16> -> 4 f32 each.
        tz_vals = {}
        for y in range(4):
            for x in range(4):
                pos = (y * Wi + x) * vox
                v0 = b.global_load_vN(X, b.add(b.add(off0, ci(pos)), c0), BF16, 4, align=8)
                v1 = b.global_load_vN(X, b.add(b.add(off1, ci(pos)), c0), BF16, 4, align=8)
                comb = []
                for j in range(4):
                    f0 = b.cast_to_f32(b.vec_extract(v0, j))
                    f1 = b.cast_to_f32(b.vec_extract(v1, j))
                    comb.append(b.fma(r, f1, f0))
                tz_vals[(y, x)] = comb
        # y transform
        ty_vals = {}
        for bb in range(4):
            for x in range(4):
                vals = []
                for j in range(4):
                    s = None
                    for y in range(4):
                        co = BT[bb][y]
                        if co == 0:
                            continue
                        term = tz_vals[(y, x)][j]
                        if s is None:
                            s = term if co > 0 else b.fneg(term)
                        else:
                            s = b.fadd(s, term) if co > 0 else b.fsub(s, term)
                    vals.append(s)
                ty_vals[(bb, x)] = vals
        # x transform + pack + MFMA
        new_accs = []
        for bb in range(4):
            for cx in range(4):
                pt = bb * 4 + cx
                vals = []
                for j in range(4):
                    s = None
                    for x in range(4):
                        co = BT[cx][x]
                        if co == 0:
                            continue
                        term = ty_vals[(bb, x)][j]
                        if s is None:
                            s = term if co > 0 else b.fneg(term)
                        else:
                            s = b.fadd(s, term) if co > 0 else b.fsub(s, term)
                    vals.append(b.trunc_f32_to_bf16(s))
                a_frag = b.vec_pack(vals, BF16)
                b_frag = b.global_load_vN(
                    U, b.add(b.add(u_base, ci(pt * u_point)), cu), BF16, 4, align=8
                )
                new_accs.append(atom.emit(b, a_frag, b_frag, accs[pt]))
        b.scf_yield(*new_accs)
    return loop.results


def _bt_axis(b, vals, coef_row):
    """sum_i coef_row[i] * vals[i] with coefficients in {0, +1, -1}."""
    s = None
    for co, term in zip(coef_row, vals):
        if co == 0:
            continue
        if s is None:
            s = term if co > 0 else b.fneg(term)
        else:
            s = b.fadd(s, term) if co > 0 else b.fsub(s, term)
    return s


_PERM_HI16 = 0x07060302  # {src1.hi16, src0.hi16}


def _pack4_bf16_half_up(b, vals):
    """4 f32 -> <4 x bf16>, round half up (ties away from zero in magnitude)."""
    ci = b.const_i32
    ints = [b.add(b.bitcast(v, I32), ci(0x8000)) for v in vals]
    d0 = b.perm_b32(ints[1], ints[0], ci(_PERM_HI16))
    d1 = b.perm_b32(ints[3], ints[2], ci(_PERM_HI16))
    return b.vec_bitcast(b.vec_pack([d0, d1], I32), VectorType(BF16, 4))


def _main_loop_pf(
    b, spec, atom, X, U, corner, p0, p1, r, wave, n_glob, kq, plane, chunk_stride,
    u_point, u_chunk,
):
    ci = b.const_i32
    C, K, Wi = spec.C, spec.K, spec.Wi
    n_chunks = C // spec.c_chunk
    x_bytes = spec.Di * spec.Hi * spec.Wi * C * 2
    u_bytes = 64 * C * K * 2
    rx = b.buffer_rsrc(X, ci(x_bytes))
    ru = b.buffer_rsrc(U, ci(u_bytes))
    vx = b.mul(corner, ci(2))
    vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
    s_p0 = b.mul(p0, ci(plane * 2))
    s_p1 = b.mul(p1, ci(plane * 2))
    s_u = b.mul(wave, ci(16 * u_point * 2))

    def load_b(cc):
        su = b.add(s_u, b.mul(cc, ci(u_chunk * 2)))
        return [
            b.buffer_load_vN(ru, vu, b.add(su, ci(pt * u_point * 2)), BF16, 4)
            for pt in range(16)
        ]

    pfb = spec.pf_ops == "xb"

    def load_chunk(cc):
        sx = b.mul(cc, ci(chunk_stride * 2))
        raw = []
        for sp in (s_p0, s_p1):
            base = b.add(sx, sp)
            for y in range(4):
                for x in range(4):
                    so = b.add(base, ci((y * Wi + x) * 8 * 2))
                    raw.append(b.buffer_load_vN(rx, vx, so, BF16, 4))
        return raw, (load_b(cc) if pfb else [])

    raw0, bfr0 = load_chunk(ci(0))
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(16)]
    iters += [(f"raw{i}", v) for i, v in enumerate(raw0)]
    iters += [(f"bfr{i}", v) for i, v in enumerate(bfr0)]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
    with loop as (cc, its):
        accs, raw = its[:16], its[16:48]
        bfr = its[48:64] if pfb else load_b(cc)
        tz = {}
        for y in range(4):
            for x in range(4):
                v0 = raw[y * 4 + x]
                v1 = raw[16 + y * 4 + x]
                tz[(y, x)] = [
                    b.fma(r, b.cast_to_f32(b.vec_extract(v1, j)),
                          b.cast_to_f32(b.vec_extract(v0, j)))
                    for j in range(4)
                ]
        # Next chunk's loads after the current raw registers are consumed.
        nraw, nbfr = load_chunk(b.smin(b.add(cc, ci(1)), ci(n_chunks - 1)))
        ty = {}
        for bb in range(4):
            for x in range(4):
                ty[(bb, x)] = [
                    _bt_axis(b, [tz[(y, x)][j] for y in range(4)], BT[bb])
                    for j in range(4)
                ]
        new_accs = []
        for bb in range(4):
            for cx in range(4):
                pt = bb * 4 + cx
                vals = [
                    _bt_axis(b, [ty[(bb, x)][j] for x in range(4)], BT[cx])
                    for j in range(4)
                ]
                a_frag = _pack4_bf16_half_up(b, vals)
                new_accs.append(atom.emit(b, a_frag, bfr[pt], accs[pt]))
        b.scf_yield(*new_accs, *nraw, *nbfr)
    return loop.results[:16]


def build_winograd3d_fwd(spec: Winograd3dSpec) -> KernelDef:
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid winograd3d spec: {why}")
    atom = MfmaAtom.bf16_32x32x8()
    C, K = spec.C, spec.K
    Hi, Wi = spec.Hi, spec.Wi
    TY, TX = spec.TY, spec.TX
    Ho, Wo = spec.Ho, spec.Wo

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))  # tile in block (A row) / output col (n) in C layout
    kq = b.div(lane, ci(32))  # channel quad within the 8-channel chunk
    bid_t = b.block_id_x()
    bid_n = b.block_id_y()

    # This lane's tile -> (tz, ty, tx).
    t = b.add(b.mul(bid_t, ci(spec.tiles_per_wg)), m)
    tx = b.mod(t, ci(TX))
    ty = b.mod(b.div(t, ci(TX)), ci(TY))
    tz = b.div(t, ci(TX * TY))
    c8 = spec.layout == "c8"
    # Element strides of X: voxel (x), channel-chunk.
    vox = 8 if c8 else C
    chunk_stride = (spec.Di * Hi * Wi * 8) if c8 else spec.c_chunk
    # Element offset of the tile's input corner (channel 0).
    corner = b.mul(
        b.add(b.mul(b.add(b.mul(b.mul(tz, ci(2)), ci(Hi)), b.mul(ty, ci(2))), ci(Wi)),
              b.mul(tx, ci(2))),
        ci(vox),
    )
    corner = b.add(corner, b.mul(kq, ci(4)))

    # Per-wave z planes and r (wave-uniform).
    is0 = b.cmp_eq(wave, ci(0))
    is3 = b.cmp_eq(wave, ci(3))
    is1 = b.cmp_eq(wave, ci(1))
    p0 = b.select(is0, ci(0), ci(1))
    p1 = b.select(is3, ci(3), ci(2))
    r = b.select(is1, b.const_f32(1.0), b.const_f32(-1.0))
    plane = Hi * Wi * vox
    off0 = b.add(corner, b.mul(p0, ci(plane)))
    off1 = b.add(corner, b.mul(p1, ci(plane)))

    # B (weights) base. ndhwc: U[p][k][c]; c8: U[p][C/8][K][8]. Lane col n = m, quad kq.
    n_glob = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
    if c8:
        u_point = (C // 8) * K * 8
        u_chunk = K * 8
        u_base = b.add(
            b.add(b.mul(b.mul(wave, ci(16)), ci(u_point)), b.mul(n_glob, ci(8))),
            b.mul(kq, ci(4)),
        )
    else:
        u_point = K * C
        u_chunk = spec.c_chunk
        u_base = b.add(
            b.add(b.mul(b.mul(wave, ci(16)), ci(K * C)), b.mul(n_glob, ci(C))),
            b.mul(kq, ci(4)),
        )

    n_chunks = C // spec.c_chunk
    if spec.pipe == "pf":
        acc = _main_loop_pf(
            b, spec, atom, X, U, corner, p0, p1, r, wave, n_glob, kq, plane,
            chunk_stride, u_point, u_chunk,
        )
    else:
        acc = _main_loop_v1(
            b, spec, atom, X, U, off0, off1, r, u_base, vox, chunk_stride, u_point,
            u_chunk, n_chunks,
        )

    # Epilogue: A^T along y and x per accumulator slot i -> P[i][oy][ox] (computed slot
    # by slot inside each branch, so no more than one slot's partials are live).
    def p_slot(i):
        mv = [[b.vec_extract(acc[bb * 4 + cx], i) for cx in range(4)] for bb in range(4)]
        out = {}
        for oy in range(2):
            row = [_bt_axis(b, [mv[bb][cx] for bb in range(4)], AT[oy]) for cx in range(4)]
            for ox in range(2):
                out[(oy, ox)] = _bt_axis(b, row, AT[ox])
        return out

    keys = [(i, oy, ox) for i in range(16) for oy in range(2) for ox in range(2)]
    lds = b.smem_alloc(F32, [2, 64, 64], name_hint="wino_p")
    # waves 1, 2 stage P into LDS slot wave-1
    is12 = b.land(b.cmp_ge(wave, ci(1)), b.cmp_le(wave, ci(2)))
    with b.scf_if(is12):
        slot = b.sub(wave, ci(1))
        for i in range(16):
            pv = p_slot(i)
            for oy in range(2):
                for ox in range(2):
                    j = keys.index((i, oy, ox))
                    b.smem_store_vN_f32(lds, [slot, ci(j), lane], pv[(oy, ox)], 1)
    b.sync()
    is03 = b.lor(is0, is3)
    with b.scf_if(is03):
        # oz = 0 for wave 0, 1 for wave 3
        oz = b.select(is0, ci(0), ci(1))
        sgn = b.select(is0, b.const_f32(1.0), b.const_f32(-1.0))
        n_out = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
        for i in range(16):
            pv = p_slot(i)
            row = (i // 4) * 8 + (i % 4)
            mt = b.add(ci(row), b.mul(kq, ci(4)))  # tile row of this slot
            tt = b.add(b.mul(bid_t, ci(spec.tiles_per_wg)), mt)
            ttx = b.mod(tt, ci(TX))
            tty = b.mod(b.div(tt, ci(TX)), ci(TY))
            ttz = b.div(tt, ci(TX * TY))
            zo = b.add(b.mul(ttz, ci(2)), oz)
            for oy in range(2):
                for ox in range(2):
                    j = keys.index((i, oy, ox))
                    p1v = b.vec_extract(b.smem_load_vN_f32(lds, ci(0), ci(j), lane, n=1), 0)
                    p2v = b.vec_extract(b.smem_load_vN_f32(lds, ci(1), ci(j), lane, n=1), 0)
                    # wave 0: P1 + (P2 + P0); wave 3: P1 - (P2 + P3).
                    out = b.fadd(p1v, b.fmul(sgn, b.fadd(p2v, pv[(oy, ox)])))
                    yo = b.add(b.mul(tty, ci(2)), ci(oy))
                    xo = b.add(b.mul(ttx, ci(2)), ci(ox))
                    yoff = b.add(
                        b.mul(
                            b.add(b.mul(b.add(b.mul(zo, ci(Ho)), yo), ci(Wo)), xo), ci(K)
                        ),
                        n_out,
                    )
                    b.global_store(Y, yoff, b.trunc_f32_to_bf16(out), align=2)
    b.ret()
    return b.kernel


_WT_CONST: dict = {}


def winograd3d_weight_transform(w, *, dtype=None, layout="ndhwc"):
    """Torch reference weight transform. ``w``: [K, 3, 3, 3, C] (KZYXC) ->
    ``U``: [64, K, C] bf16, point ``p = a*16 + b*4 + c``, z-sign ``ZS0[a]`` folded in."""
    import torch

    key = str(w.device)
    if key not in _WT_CONST:
        _WT_CONST[key] = (
            torch.tensor(G, dtype=torch.float32, device=w.device),
            torch.tensor(ZS0, dtype=torch.float32, device=w.device).view(1, 4, 1, 1, 1),
        )
    g, s = _WT_CONST[key]
    wf = w.float()  # [K, z, y, x, C]
    u = torch.einsum("ai,nijkc->najkc", g, wf)
    u = torch.einsum("bj,najkc->nabkc", g, u)
    u = torch.einsum("dk,nabkc->nabdc", g, u)
    u = u * s  # [K, a, b, c, C]
    K, C = w.shape[0], w.shape[-1]
    u = u.permute(1, 2, 3, 0, 4).reshape(64, K, C)
    if layout == "c8":
        u = u.reshape(64, K, C // 8, 8).permute(0, 2, 1, 3)
    return u.to(dtype or torch.bfloat16).contiguous()


def winograd_x_to_r8(x):
    """[D, H, W, C] -> [D, H, C/8, W, 8] (torch reference relayout, "r8")."""
    D, H, W, C = x.shape
    return x.reshape(D, H, W, C // 8, 8).permute(0, 1, 3, 2, 4).contiguous()


def winograd3d_x_to_c8(x):
    """[D, H, W, C] -> [C/8, D, H, W, 8] (torch reference relayout)."""
    D, H, W, C = x.shape
    return x.reshape(D, H, W, C // 8, 8).permute(3, 0, 1, 2, 4).contiguous()


# ---------------------------------------------------------------------------------
# F(2x2, 3x3) on (y, x), direct along z ("yx" variant).
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class WinogradYxSpec:
    """F(2x2,3x3) Winograd on (y, x) + direct 3-tap conv along z; N=1 NDHWC bf16, valid.

    Workgroup = 4 waves; wave ``w`` owns the 4 transform points whose y-index ``b == w``.
    Block = ``tiles_per_wg`` (32) 2-D output tiles (2x2 in y,x) x ``z_tile`` output planes x
    ``n_tile`` (32) output channels. Each input plane's transform (one per 8-channel chunk)
    feeds the 3 z-taps of up to 3 output planes. Accumulators: 4 points x z_tile planes x
    one 32x32 MFMA tile = 64 * z_tile f32 per lane.
    X is read in the c8 layout ``[C/8][D][H][W][8]``; U as ``[3 dz][4 b][4 c][C/8][K][8]``.
    """

    Di: int
    Hi: int
    Wi: int
    C: int
    K: int
    tiles_per_wg: int = 32
    n_tile: int = 32
    z_tile: int = 4
    c_chunk: int = 8
    # "f16": transformed operands rounded to f16 (v_cvt_pkrtz), f16 MFMA -- direct-conv
    # accuracy, but |x| must stay below ~16k (|V| <= 4 max|x|). "bf16": half-up bf16.
    # "bf16t" (lds stage): truncated to bf16 (one v_perm per pair, ~2x the rounding error).
    op_dtype: str = "bf16"
    # software-pipeline lookahead in input planes (0 = load each plane where it is used)
    pf_planes: int = 0
    # sched_barrier(0) after every plane stage (keeps the scheduler from hoisting loads
    # and transforms of later planes, which blows the VGPR budget next to 256 AGPRs)
    sbar: bool = False
    # LLVM occupancy / AGPR-split hints: waves_per_eu (0 = unset), agpr_alloc "min,max"
    wpe: int = 0
    agpr: str = ""
    # "reg": every wave loads its raw rows and transforms them in registers;
    # "lds": the workgroup stages the y-transformed input planes (f32) in LDS once per
    # chunk (cooperative, coalesced 16-byte loads prefetched one chunk ahead), each wave
    # reads its rows conflict-free (even/odd columns split) and only applies B^T along x.
    stage: str = "reg"
    # lds stage: 4 or 8 waves; with 8, waves w and w+4 share the y-point b = w % 4 and
    # split the z_tile output planes (two waves per SIMD at 128 accumulators each).
    waves: int = 4
    # lds16: fill items interleaved across waves (item = lane * waves + wave) so every
    # wave and SIMD gets the same share; fill_first stages chunk c+1 before computing c
    fill_il: bool = False
    fill_first: bool = False
    # lds16 with 8 waves: waves w and w+4 split the output channels (n_tile 64) instead
    # of the z planes -- fewer staged planes per MFMA and no duplicated weight loads
    nsplit: bool = False
    # lds16: number of output-channel groups (n_tile = 32 * nn); the waves // 4 groups of
    # 4 waves split nn ways over channels and the rest over z (nsplit=True means nn=2)
    nn: int = 1
    # lds16 main loop scheduling: iglp_opt level (-1 = none); sgb = manual
    # sched_group_barrier interleave "1 MFMA : sgb VALU" (0 = off)
    iglp: int = -1
    sgb: int = 0
    # lds16: waves 4..7 (second wave on each SIMD) stage the next chunk before their MFMAs,
    # waves 0..3 after, so the two waves sharing a SIMD are out of phase
    phase: bool = False
    # which waves fill first under ``phase``: "hi" = waves >= 4, "odd" = odd waves
    phase_sel: str = "hi"
    # lds16: X layout read by the fill: "c8" [C/8][D][H][W][8] (relayout pass), "r8"
    # [D][H][C/8][W][8] (relayout pass; consecutive chunks of a row are adjacent, so a
    # workgroup stays on the same pages) or "ndhwc" (in place: 16-byte lane loads one voxel
    # apart -- TA-bound, 3.5x slower)
    xin: str = "c8"
    # lds16: weight fragments of two chunks per 16-byte lane load (U layout
    # [3][4][4][C/16][K][2 kq][2 chunk][4]), main loop unrolled by two chunks
    b16: bool = False
    # lds16: U layout [3][4][C/8][K][2 kq][4 c][4] so one 16-byte lane load brings the
    # fragments of two points (c pairs): 6 instead of 12 weight loads per chunk
    bc: bool = False
    # lds16: one staging buffer and two barriers per chunk (fill | compute), so that two
    # 4-wave workgroups fit one CU (32 KiB LDS each) and overlap each other's phases
    single_buf: bool = False
    # lds16 single_buf: the weight fragments are staged in LDS once per workgroup (16-byte
    # coalesced loads by all threads) and shared by the z groups (waves w, w+4 ... use the
    # same y-point b), instead of 8-byte per-wave global loads
    bshare: bool = False
    # lds16: output-channel blocks fastest in a 1-D grid, so the blocks of one tile block
    # run together and read X once (the weights of every n block stay in L2)
    nfast: bool = False
    # lds16 fill: f32 -> f16 pairs with one v_cvt_pkrtz_f16_f32 (inline asm) instead of two
    # converts + a pack (round toward zero, like the x-transform's packed f16 adds)
    pkrtz: bool = False
    # epilogue: the y-combination reads 4 neighbouring channels per LDS read (b128) and
    # every lane stores 4 bf16 (8 bytes) at once -- 4 stores per lane and plane, not 16
    epi_vec: bool = False
    # lds16 main loop: s_setprio(prio) around the MFMA section, 0 around the fill
    prio: int = 0
    # lds16: branch-free fill (padding lanes write a dummy LDS area instead of skipping
    # the stores), so the loop body is one basic block the scheduler can interleave
    bf_fill: bool = False
    # lds16 nfast: remap workgroup ids so consecutive logical blocks land on the same XCD
    # (the hardware deals ids round-robin over xcd XCDs): halo planes / rows shared by
    # neighbouring blocks then hit in that XCD's L2. 0 = off
    xcd: int = 0
    # lds16: persistent workgroups (grid = persist, each loops over blocks in the nfast
    # order); the next block's first two chunks and weights are loaded before the
    # epilogue of the current one. 0 = off
    persist: int = 0
    # lds16: pad the column-parity stride of the staged planes to 8 (mod 16) 8-byte slots,
    # so the fill's even/odd-column lanes hit disjoint LDS banks (2-way conflicts otherwise)
    ppad: bool = False
    # lds16 nfast: fused X relayout ("frel"). The main kernel takes X in NDHWC too and
    # every workgroup relayouts a share of X into the c8 buffer while it computes (one
    # 16-byte item per thread and chunk, loaded at the top of the iteration, stored at the
    # end); it publishes a per-workgroup flag after its main loop. A workgroup first waits
    # for the flags of the producers of its input planes. The first planes (at least
    # ``frel`` z slabs, more if needed so that every producer runs >= frel_gap workgroups
    # ahead of its consumers) are relayouted by the pre-launch (build_yx_frel_aux, which
    # also transforms the weights and zeroes the flags). 0 = off.
    frel: int = 0
    frel_gap: int = 228
    # lds16: half fill items -- a lane stages 4 channels (8-byte row loads) of one column
    # instead of 8, items dealt block-strided: every lane of the workgroup fills in round 0
    # (fill_il leaves half the lanes idle at TX=32), only wave 0 takes the few left over
    fhalf: bool = False
    # fhalf: rounds whose items are all valid are emitted without the per-lane branch, so
    # the scheduler can interleave that fill with the MFMAs (same basic block)
    fh_nb: bool = False
    # lds16 fill: rows converted to f16 first (exact for bf16 in the f16 normal range), the
    # B^T combination along y in packed f16 (v_pk_add_f16): 2 instead of 2.5 VALU per value
    yf16: bool = False
    # nfast: an extra i32 kernel argument BOFF added to the block id, so the grid can be
    # launched in z-slab pieces (pipelined against a piecewise X relayout on another stream)
    boff: bool = False
    # whole call: the aux relayout walks X from the last voxel block to the first, so the
    # first z slabs (read first by the main kernel) are the most recently written (MALL)
    xrev: bool = False
    # lds16 main loop: weight fragments prefetched two chunks ahead (one extra set of 12
    # fragment registers in flight) instead of one
    bpf2: bool = False
    # epilogue (epi_vec): write Y in the next layer's input layout -- c8 with a one-voxel
    # zero halo, [K/8][Do+2][Ho+2][Wo+2][8], interior only (the halo is never written, the
    # caller zeroes it once) -- so a chain of yx convolutions needs no X relayout pass
    yc8h: bool = False
    # yc8h epilogue with 16-byte stores: a lane combines 8 channels (one whole c8 slot) of
    # one voxel, lanes (ox, 8-channel group, tile row) -- 2 stores per lane and plane, not 4;
    # the two b128 LDS reads of a lane are ordered by ox (bank-disjoint halves)
    epi16: bool = False
    # lds16: linear tile mapping -- a block's 32 tiles are the consecutive 2-D tiles
    # t0 .. t0+31 of the plane in row-major order, for any TX (rows wrap inside a block;
    # the last tile of a row reads its right-hand column pair from a "seam" slot). The last
    # block of a plane may be partial (its extra tiles are computed on wrapped rows and
    # never stored). For dgrad as a forward conv of padded dY (TX = 2^n + 1).
    lin: bool = False
    # yc8h halo width (1: the next forward conv's input; 2: the next dgrad's padded dY)
    yc8h_halo: int = 1
    # yc8h: drop the outermost ring of the output (z, y or x at 0 or the last index) -- a
    # chained BWD-data: this dgrad's dX minus its halo-form ring is the next dgrad's dY,
    # and with yc8h_halo 1 the buffer is that dY padded by 2 (ring slots stay zero)
    yc8h_crop: bool = False
    # lds16 (default double-buffered path): split the C chunks over ``ksplit`` workgroups
    # (grid x ksplit, split index slowest); each writes its f32 partial sums to slice
    # ``ks`` of an f32 workspace [ksplit][Do][Ho][Wo][K] (the "Y" argument, f32), and
    # build_ks_reduce sums the slices into the bf16 NDHWC output. For small grids (s7/s8).
    ksplit: int = 1
    # f32 kernel argument OS multiplied into the output (A^T coefficients of the epilogue,
    # free): with a relayout that scales X by 1/OS (a power of two, exact) the f16 operands
    # stay in the f16 normal range for tiny tensors (dgrad's dY) -- per-tensor scaling
    oscale: bool = False
    # debugging / attribution only (wrong results): "nofill" skips the LDS fill,
    # "nomfma" skips the MFMAs, "nowait" uses no barriers in the main loop
    dbg: str = ""
    name: str = "conv3d_winoyx"

    @property
    def Do(self) -> int:
        return self.Di - 2

    @property
    def Ho(self) -> int:
        return self.Hi - 2

    @property
    def Wo(self) -> int:
        return self.Wi - 2

    @property
    def tiles2d(self) -> int:
        return (self.Ho // 2) * (self.Wo // 2)

    @property
    def block_size(self) -> int:
        return 64 * self.waves

    @property
    def flops(self) -> int:
        return 2 * self.C * self.K * 27 * self.Do * self.Ho * self.Wo

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"D{self.Di}H{self.Hi}W{self.Wi}C{self.C}K{self.K}",
            "bf16",
            f"t{self.tiles_per_wg}z{self.z_tile}n{self.n_tile}",
            self.op_dtype,
            *([f"pf{self.pf_planes}"] if self.pf_planes else []),
            *(["sb"] if self.sbar else []),
            *([f"wpe{self.wpe}"] if self.wpe else []),
            *([f"ag{self.agpr.replace(',', '_')}"] if self.agpr else []),
            *([self.stage] if self.stage != "reg" else []),
            *([f"w{self.waves}"] if self.waves != 4 else []),
            *(["fil"] if self.fill_il else []),
            *(["ff"] if self.fill_first else []),
            *(["ns"] if self.nsplit else []),
            *([f"nn{self.nn}"] if self.nn != 1 else []),
            *([f"ig{self.iglp}"] if self.iglp >= 0 else []),
            *([f"sg{self.sgb}"] if self.sgb else []),
            *(["ph" + ("" if self.phase_sel == "hi" else self.phase_sel)] if self.phase else []),
            *(["xn"] if self.xin == "ndhwc" else []),
            *(["xr8"] if self.xin == "r8" else []),
            *(["b16"] if self.b16 else []),
            *(["bc"] if self.bc else []),
            *(["sbuf"] if self.single_buf else []),
            *(["bsh"] if self.bshare else []),
            *(["nf"] if self.nfast else []),
            *(["rtz"] if self.pkrtz else []),
            *(["ev"] if self.epi_vec else []),
            *([f"pr{self.prio}"] if self.prio else []),
            *(["bff"] if self.bf_fill else []),
            *([f"x{self.xcd}"] if self.xcd else []),
            *([f"ps{self.persist}"] if self.persist else []),
            *(["pp"] if self.ppad else []),
            *([f"fr{self.frel}g{self.frel_gap}"] if self.frel else []),
            *(["fh"] if self.fhalf else []),
            *(["nb"] if self.fh_nb else []),
            *(["yh"] if self.yf16 else []),
            *(["bo"] if self.boff else []),
            *(["xrv"] if self.xrev else []),
            *(["bp2"] if self.bpf2 else []),
            *(["yc8h"] if self.yc8h else []),
            *(["e16"] if self.epi16 else []),
            *(["lin"] if self.lin else []),
            *([f"yh{self.yc8h_halo}"] if self.yc8h_halo != 1 else []),
            *(["crop"] if self.yc8h_crop else []),
            *([f"ks{self.ksplit}"] if self.ksplit != 1 else []),
            *(["os"] if self.oscale else []),
            *([f"dbg{self.dbg}"] if self.dbg else []),
        )


def is_valid_yx_spec(spec: WinogradYxSpec) -> Tuple[bool, str]:
    if spec.Ho % 2 or spec.Wo % 2 or spec.Ho <= 0 or spec.Wo <= 0:
        return False, "F(2,3) needs even output extents in y and x"
    if spec.tiles_per_wg != 32 or spec.c_chunk != 8:
        return False, "supports tiles_per_wg=32, c_chunk=8 only"
    nn = 2 if spec.nsplit else spec.nn
    if spec.n_tile != 32 * nn:
        return False, "n_tile must be 32 * nn (nn = 2 with nsplit)"
    if (nn != 1 or spec.waves > 8) and spec.stage not in ("lds16", "wlds"):
        return False, "nn > 1 / more than 8 waves need stage lds16"
    if spec.stage in ("lds16", "wlds"):
        g = spec.waves // 4
        if spec.waves % 4 or g % nn or spec.z_tile % (g // nn):
            return False, "lds16: waves/4 groups must split as nn x (divisor of z_tile)"
    if spec.lin and (spec.stage != "lds16" or spec.fhalf or spec.bf_fill or spec.frel
                     or spec.persist or spec.ppad or spec.b16 or spec.bshare):
        return False, "lin: plain lds16 (no fhalf / bf_fill / frel / persist / ppad / b16)"
    if spec.tiles2d % spec.tiles_per_wg and not spec.lin:
        return False, f"2-D tiles {spec.tiles2d} not a multiple of {spec.tiles_per_wg}"
    _nz = (spec.waves // 4) // (2 if spec.nsplit else spec.nn) if spec.stage in (
        "lds16", "wlds") else (
        spec.waves // 4)
    if spec.z_tile * 64 // max(1, _nz) > 256:
        return False, "z_tile * 64 accumulators must fit 256 AGPRs"
    if spec.K % spec.n_tile or spec.C % spec.c_chunk:
        return False, "K / C not multiples of the tile"
    if spec.op_dtype not in ("bf16", "bf16t", "f16") or (
        spec.op_dtype == "bf16t" and spec.stage != "lds"
    ):
        return False, f"unknown op_dtype {spec.op_dtype!r}"
    if spec.waves not in (4, 8, 12, 16) or (spec.waves > 4 and spec.stage == "reg") or (
        spec.stage == "lds" and spec.waves == 8 and spec.z_tile % 2
    ):
        return False, "waves: 4 | 8 (lds, even z_tile) | 4..16 (lds16)"
    if spec.stage == "lds16" and spec.op_dtype != "f16":
        return False, "stage lds16 stages f16 planes and needs op_dtype f16"
    if spec.xin not in ("c8", "ndhwc", "r8") or (spec.xin != "c8" and (
            spec.stage != "lds16" or spec.Do % spec.z_tile)):
        return False, "xin ndhwc: stage lds16 with z_tile dividing Do"
    if spec.persist and (spec.stage != "lds16" or spec.xin != "c8" or spec.single_buf
                         or spec.b16 or spec.bc or spec.xcd or spec.phase or spec.fill_first):
        return False, "persist: plain double-buffered lds16 (c8 input) only"
    if spec.frel and (spec.stage != "lds16" or not spec.nfast or spec.xin != "c8"
                      or spec.persist or spec.single_buf or spec.b16 or spec.bc or spec.xcd
                      or spec.phase or spec.fill_first or spec.Do % spec.z_tile):
        return False, "frel: plain double-buffered lds16 nfast (c8, no xcd remap)"
    if spec.boff and (not spec.nfast or spec.xcd or spec.persist or spec.frel):
        return False, "boff: nfast without xcd / persist / frel"
    if spec.yc8h and (spec.stage != "lds16" or not spec.epi_vec or spec.frel
                      or spec.Do % spec.z_tile or spec.K % 8):
        return False, "yc8h: lds16 epi_vec (no frel), z_tile dividing Do"
    if spec.ksplit != 1 and (spec.stage != "lds16" or not spec.epi_vec or spec.yc8h
                             or spec.Do % spec.z_tile or (spec.C // 8) % spec.ksplit
                             or not spec.nfast or spec.persist or spec.frel
                             or spec.single_buf or spec.b16 or spec.boff or spec.ksplit < 1):
        return False, "ksplit: lds16 nfast epi_vec NDHWC, z_tile | Do, ksplit | C/8"
    if spec.yc8h_crop and not spec.yc8h:
        return False, "yc8h_crop needs yc8h"
    if spec.epi16 and not spec.yc8h:
        return False, "epi16 needs yc8h"
    if spec.yf16 and (spec.stage != "lds16" or not spec.pkrtz):
        return False, "yf16: lds16 with pkrtz"
    if spec.fhalf and (spec.stage != "lds16" or spec.bf_fill or spec.persist
                       or spec.single_buf or spec.b16):
        return False, "fhalf: double-buffered lds16 only"
    if spec.frel and _frel_plan(spec) is None:
        return False, "frel: no producer schedule keeps producers ahead of consumers"
    if spec.xcd and not spec.nfast:
        return False, "xcd remap needs nfast (1-D grid)"
    if spec.nfast and spec.stage != "lds16":
        return False, "nfast needs stage lds16"
    if spec.bshare and not spec.single_buf:
        return False, "bshare needs single_buf"
    if spec.bc and (spec.stage != "lds16" or spec.b16):
        return False, "bc needs stage lds16 without b16"
    if spec.b16 and (spec.stage != "lds16" or spec.C % 16):
        return False, "b16 needs stage lds16 and C % 16 == 0"
    if spec.stage == "wlds" and (spec.op_dtype != "f16" or spec.Do % spec.z_tile):
        return False, "stage wlds needs op_dtype f16 and z_tile dividing Do"
    if spec.stage not in ("reg", "lds", "lds16", "wlds"):
        return False, f"unknown stage {spec.stage!r}"
    if spec.stage in ("lds", "lds16", "wlds"):
        tx = spec.Wo // 2
        if not (tx % 32 == 0 or 32 % tx == 0 or spec.lin):
            return False, "lds stage needs TX a multiple or divisor of 32"
        fl = {"lds16": _yx_lds16_floats, "wlds": _yx_wlds_floats}.get(
            spec.stage, _yx_lds_floats)(spec)
        if spec.bf_fill:
            fl = max(fl, 2 * _yx_lds16_buf(spec) + 8 * 2 * (spec.Wo // 2 + 1) * 2 + 2)
        epi = min(spec.waves, 8) if spec.stage in ("lds16", "wlds") else spec.waves
        if max(fl, epi * 32 * 64) * 4 > 65536:
            return False, "lds stage: staged planes exceed 64 KiB"
    if not 0 <= spec.pf_planes <= spec.z_tile + 1:
        return False, "pf_planes out of range"
    return True, "ok"


def _frel_plan(spec):
    """Fused-relayout schedule: (A slabs pre-relayouted, base voxel, VPW voxels per
    producer workgroup, nwg). Producer p relayouts voxels [base + p*VPW, base + (p+1)*VPW);
    the workgroups of z slab zb (pids [zb*WPS, (zb+1)*WPS)) need planes
    [zb*ZT, zb*ZT + ZT + 2). Every producer they need must have pid <= zb*WPS - gap
    (and strictly below zb*WPS, which makes the wait deadlock-free)."""
    ZT = spec.z_tile
    PV = spec.Hi * spec.Wi
    V = spec.Di * PV
    nb = spec.K // spec.n_tile
    n_tb = spec.tiles2d // spec.tiles_per_wg
    WPS = n_tb * nb
    nzb = spec.Do // ZT
    nwg = WPS * nzb
    bs = 64 * spec.waves
    gap = max(1, spec.frel_gap)

    def fits(base, vpw):
        for zb in range(nzb):
            hi = min(V, (zb * ZT + ZT + 2) * PV)
            if hi <= base:
                continue
            if (hi - 1 - base) // vpw > zb * WPS - gap:
                return False
        return True

    # minimal pre-relayouted slabs A, then the smallest share per producer (the
    # producers are the first ceil((V - base) / vpw) workgroups: front-loaded, since
    # the consumers walk through X at 2 planes per WPS workgroups)
    for A in range(max(1, spec.frel), nzb + 2):
        base = min(V, A * ZT * PV)
        if base >= V:
            return A, V, 1, nwg
        lo = -(-(V - base) // nwg)
        if lo > bs or not fits(base, bs):
            continue
        hi_ = bs
        while lo < hi_:
            mid = (lo + hi_) // 2
            if fits(base, mid):
                hi_ = mid
            else:
                lo = mid + 1
        return A, base, lo, nwg
    return None


def _yx_ntb(spec) -> int:
    """2-D tile blocks per plane (the last one partial under ``lin``)."""
    return -(-spec.tiles2d // spec.tiles_per_wg)


def _yx_lin_seams(spec) -> int:
    """lin: most row ends (other than at the block's last tile) inside one block."""
    if not getattr(spec, "lin", False):
        return 0
    TX = spec.Wo // 2
    return -(-(spec.tiles_per_wg - 1) // TX)


def _yx_nblocks(spec) -> int:
    return (_yx_ntb(spec) * (-(-spec.Do // spec.z_tile))
            * (spec.K // spec.n_tile))


def winograd_yx_grid(spec: WinogradYxSpec) -> Tuple[int, int, int]:
    if spec.persist:
        return (min(spec.persist, _yx_nblocks(spec)), 1, 1)
    if spec.nfast:
        return (
            _yx_ntb(spec) * (-(-spec.Do // spec.z_tile))
            * (spec.K // spec.n_tile) * spec.ksplit,
            1,
            1,
        )
    return (
        _yx_ntb(spec) * (-(-spec.Do // spec.z_tile)),
        spec.K // spec.n_tile,
        1,
    )


def _pack4_f16_rtz(b, vals):
    from rocke.core.ir import F16

    v = b.vec_pack(list(vals), F32)
    return b.vec_cast_f32_to(v, F16)


def _yx_epilogue(
    b, spec, acc, wave, lane, kq, tb, z0, n_glob, Y, lds_flat=None, zt_w=None, wb=None,
    zh=None, gpr=None, ks_i=None, oscale=None,
):
    """``zt_w``: output planes of this wave (default z_tile); ``wb``: the wave's y-point
    index b (default ``wave``); ``zh``: its z group (waves zh*4 .. zh*4+3 combine)."""
    ci = b.const_i32
    ZT, K, Ho, Wo = spec.z_tile, spec.K, spec.Ho, spec.Wo
    if zt_w is not None:
        ZT = zt_w
    if wb is None:
        wb = wave
    wbase = ci(0) if zh is None else b.mul(zh, ci(4))
    TX = Wo // 2
    if getattr(spec, "yc8h", False):
        h2 = 2 * getattr(spec, "yc8h_halo", 1)
        ry = b.buffer_rsrc(Y, ci((spec.Do + h2) * (Ho + h2) * (Wo + h2) * K * 2))
    elif getattr(spec, "ksplit", 1) > 1:
        ry = b.buffer_rsrc(Y, ci(spec.ksplit * spec.Do * Ho * Wo * K * 4))
    else:
        ry = b.buffer_rsrc(Y, ci(spec.Do * Ho * Wo * K * 2))
    # Epilogue (branch-free, one output plane zo per round). Every wave applies A^T along
    # x in-lane (Px[ox] = sum_c AT[ox][c] m[zo][c], 32 values per lane) and stages them
    # in LDS; then wave w combines the 4 waves' partials along y for output row
    # oy = w // 2 and accumulator slots i in [8 (w % 2), 8 (w % 2) + 8).
    if lds_flat is None:
        lds = b.smem_alloc(F32, [4, 32, 64], name_hint="winoyx_p")

        def st(w, j, v):
            b.smem_store_vN_f32(lds, [w, j, lane], v, 1)

        def ld(w, j):
            return b.vec_extract(b.smem_load_vN_f32(lds, w, j, lane, n=1), 0)
    else:
        def _ix(w, j):
            return b.add(b.mul(b.add(b.mul(w, ci(32)), j), ci(64)), lane)

        def st(w, j, v):
            b.smem_store_vN_f32(lds_flat, [_ix(w, j)], v, 1)

        def ld(w, j):
            return b.vec_extract(b.smem_load_vN_f32(lds_flat, _ix(w, j), n=1), 0)

        def ld4(w, j, lane_off):
            return b.smem_load_vN_f32(
                lds_flat, b.add(b.mul(b.add(b.mul(w, ci(32)), j), ci(64)), lane_off), n=4
            )

        ld.ld4 = ld4
        ld.n_base = b.sub(n_glob, b.mod(lane, ci(32)))
        ld.lane = lane
        ld.ks_i = ks_i
    oy = b.div(wb, ci(2))
    ih = b.mod(wb, ci(2))
    # AT[oy][bb] as wave-uniform f32 coefficients
    coef = []
    for bb in range(4):
        coef.append(
            b.select(b.cmp_eq(oy, ci(0)), b.const_f32(float(AT[0][bb])),
                     b.const_f32(float(AT[1][bb])))
        )
    if oscale is not None:
        coef = [b.fmul(cf, oscale) for cf in coef]
    # A^T along x for every slot right away (256 accumulators -> 128 VGPR values), so
    # no accumulator is live across the epilogue's barriers.
    pxs = {}
    for zo in range(ZT):
        for i in range(16):
            mv = [b.vec_extract(acc[zo * 4 + c], i) for c in range(4)]
            for ox in range(2):
                pxs[(zo, i, ox)] = _bt_axis(b, mv, AT[ox])
    groups = spec.waves // 4
    if gpr is None or gpr >= groups:
        _yx_epi_rounds(b, spec, ZT, pxs, st, ld, wave, wbase, ih, oy, coef, kq, tb, z0,
                       n_glob, ry, None)
        return
    # Groups (4 waves each) take turns in LDS, gpr groups per round.
    grp = b.div(wave, ci(4))
    for r in range(-(-groups // gpr)):
        lo, hi = r * gpr, min(groups, (r + 1) * gpr)
        active = b.land(b.cmp_ge(grp, ci(lo)), b.cmp_lt(grp, ci(hi)))
        w_loc = b.sub(wave, ci(lo * 4))
        wb_loc = b.mul(b.sub(grp, ci(lo)), ci(4))
        if r:
            b.sync()
        _yx_epi_rounds(b, spec, ZT, pxs, st, ld, w_loc, wb_loc, ih, oy, coef, kq, tb, z0,
                       n_glob, ry, active)


def _yx_epi_rounds(b, spec, ZT, pxs, st, ld, wave, wbase, ih, oy, coef, kq, tb, z0, n_glob,
                   ry, active):
    ci = b.const_i32
    K, Ho, Wo = spec.K, spec.Ho, spec.Wo
    TX = Wo // 2

    def guarded(fn):
        if active is None:
            fn()
        else:
            with b.scf_if(active):
                fn()

    for zo in range(ZT):
        if zo:
            b.sync()

        def wr(zo=zo):
            for i in range(16):
                for ox in range(2):
                    st(wave, ci(i * 2 + ox), pxs[(zo, i, ox)])

        guarded(wr)
        b.sync()
        guarded(lambda zo=zo: _yx_epi_store(b, spec, zo, ld, wbase, ih, oy, coef, kq, tb, z0,
                                            n_glob, ry))


def _yx_epi_store(b, spec, zo, ld, wbase, ih, oy, coef, kq, tb, z0, n_glob, ry):
    ci = b.const_i32
    K, Ho, Wo = spec.K, spec.Ho, spec.Wo
    TX = Wo // 2
    if getattr(spec, "epi16", False) and hasattr(ld, "ld4"):
        _yx_epi_store16(b, spec, zo, ld, wbase, ih, oy, coef, tb, z0, ry)
        return
    if getattr(spec, "epi_vec", False) and hasattr(ld, "ld4"):
        lane = ld.lane
        nq = b.mod(lane, ci(8))
        i_dyn = b.add(b.mul(ih, ci(8)), b.div(lane, ci(8)))
        zz = b.add(z0, ci(zo))
        cvec = [b.vector_splat(cf, 4) for cf in coef]
        for kq2 in range(2):
            row = b.add(b.add(b.mul(b.div(i_dyn, ci(4)), ci(8)), b.mod(i_dyn, ci(4))),
                        ci(4 * kq2))
            tt = b.add(b.mul(tb, ci(spec.tiles_per_wg)), row)
            ttx = b.mod(tt, ci(TX))
            tty = b.div(tt, ci(TX))
            yo = b.add(b.mul(tty, ci(2)), oy)
            for ox in range(2):
                jj = b.add(b.mul(i_dyn, ci(2)), ci(ox))
                src = b.add(ci(kq2 * 32), b.mul(nq, ci(4)))
                out = None
                for bb in range(4):
                    pv = ld.ld4(b.add(wbase, ci(bb)), jj, src)
                    out = (b.vector_mul(cvec[bb], pv) if out is None
                           else b.vector_fma(cvec[bb], pv, out))
                xo = b.add(b.mul(ttx, ci(2)), ci(ox))
                if getattr(spec, "yc8h", False):
                    # [K/8][Do+2][Ho+2][Wo+2][8], interior: channel kk = n_base + 4 nq
                    kb = b.add(b.div(ld.n_base, ci(8)), b.div(nq, ci(2)))
                    hh = getattr(spec, "yc8h_halo", 1)
                    vox = b.add(b.mul(b.add(b.mul(b.add(b.mul(
                        kb, ci(spec.Do + 2 * hh)), b.add(zz, ci(hh))), ci(Ho + 2 * hh)),
                        b.add(yo, ci(hh))), ci(Wo + 2 * hh)), b.add(xo, ci(hh)))
                    yoff = b.add(b.mul(vox, ci(8)), b.mul(b.mod(nq, ci(2)), ci(4)))
                    yoff = _yx_crop_mask(b, spec, zz, yo, xo, yoff, 1)
                else:
                    yoff = b.add(
                        b.mul(b.add(b.mul(b.add(b.mul(zz, ci(Ho)), yo), ci(Wo)), xo), ci(K)),
                        b.add(ld.n_base, b.mul(nq, ci(4))),
                    )
                if getattr(spec, "ksplit", 1) > 1:
                    nel = spec.Do * Ho * Wo * K
                    yoff = b.add(yoff, b.mul(ld.ks_i, ci(nel)))
                    yoff = _yx_lin_mask(b, spec, tt, yoff, 1, total=spec.ksplit * nel)
                    b.buffer_store_vN_f32(ry, b.mul(yoff, ci(4)), ci(0), out, 4)
                    continue
                yoff = _yx_lin_mask(b, spec, tt, yoff, 1)
                b.buffer_store_vN_bf16(
                    ry, b.mul(yoff, ci(2)), ci(0), b.vec_trunc_f32_to_bf16(out), 2
                )
        return
    if True:
        zz = b.add(z0, ci(zo))
        for il in range(8):
            i_dyn = b.add(b.mul(ih, ci(8)), ci(il))
            # slot i -> row (i // 4) * 8 + (i % 4) + 4 kq
            row = b.add(
                b.add(b.mul(b.div(i_dyn, ci(4)), ci(8)), b.mod(i_dyn, ci(4))),
                b.mul(kq, ci(4)),
            )
            tt = b.add(b.mul(tb, ci(spec.tiles_per_wg)), row)
            ttx = b.mod(tt, ci(TX))
            tty = b.div(tt, ci(TX))
            yo = b.add(b.mul(tty, ci(2)), oy)
            for ox in range(2):
                jj = b.add(b.mul(i_dyn, ci(2)), ci(ox))
                out = None
                for bb in range(4):
                    pv = ld(b.add(wbase, ci(bb)), jj)
                    out = b.fmul(coef[bb], pv) if out is None else b.fma(coef[bb], pv, out)
                xo = b.add(b.mul(ttx, ci(2)), ci(ox))
                yoff = b.add(
                    b.mul(b.add(b.mul(b.add(b.mul(zz, ci(Ho)), yo), ci(Wo)), xo), ci(K)),
                    n_glob,
                )
                # out-of-range z (last block when z_tile does not divide Do) is dropped
                yoff = _yx_lin_mask(b, spec, tt, yoff, 1)
                b.buffer_store_bf16(ry, b.mul(yoff, ci(2)), ci(0), b.trunc_f32_to_bf16(out))


def _yx_crop_mask(b, spec, zz, yo, xo, off, unit):
    """yc8h_crop: outputs on the outer ring store out of range."""
    if not getattr(spec, "yc8h_crop", False):
        return off
    ci = b.const_i32
    h2 = 2 * getattr(spec, "yc8h_halo", 1)
    nel = (spec.Do + h2) * (spec.Ho + h2) * (spec.Wo + h2) * spec.K
    ok = b.land(b.land(b.land(b.cmp_ge(zz, ci(1)), b.cmp_lt(zz, ci(spec.Do - 1))),
                       b.land(b.cmp_ge(yo, ci(1)), b.cmp_lt(yo, ci(spec.Ho - 1)))),
                b.land(b.cmp_ge(xo, ci(1)), b.cmp_lt(xo, ci(spec.Wo - 1))))
    return b.select(ok, off, ci(-(-nel // unit)))


def _yx_lin_mask(b, spec, tt, off, unit, total=None):
    """lin: tiles past the plane's last one (partial last block) store out of range."""
    if not getattr(spec, "lin", False) or spec.tiles2d % spec.tiles_per_wg == 0:
        return off
    if total is not None:
        nel = total
    elif getattr(spec, "yc8h", False):
        h2 = 2 * getattr(spec, "yc8h_halo", 1)
        nel = (spec.Do + h2) * (spec.Ho + h2) * (spec.Wo + h2) * spec.K
    else:
        nel = spec.Do * spec.Ho * spec.Wo * spec.K
    return b.select(b.cmp_lt(tt, b.const_i32(spec.tiles2d)), off,
                    b.const_i32(-(-nel // unit)))


def _yx_epi_store16(b, spec, zo, ld, wbase, ih, oy, coef, tb, z0, ry):
    """yc8h epilogue, one 16-byte store per (lane, kq2): lane = (r, ox, g) with g = lane % 4
    the 8-channel group, ox = (lane // 4) % 2 and r = lane // 8 the tile row slot."""
    ci = b.const_i32
    Ho, Wo = spec.Ho, spec.Wo
    TX = Wo // 2
    lane = ld.lane
    g = b.mod(lane, ci(4))
    oxl = b.mod(b.div(lane, ci(4)), ci(2))
    is1 = b.cmp_eq(oxl, ci(1))
    i_dyn = b.add(b.mul(ih, ci(8)), b.div(lane, ci(8)))
    zz = b.add(z0, ci(zo))
    cvec = [b.vector_splat(cf, 4) for cf in coef]
    kb = b.add(b.div(ld.n_base, ci(8)), g)
    jj = b.add(b.mul(i_dyn, ci(2)), oxl)
    # ox = 1 lanes read their upper half first: the two lane sets hit disjoint banks
    h0 = b.mul(oxl, ci(4))
    h1 = b.sub(ci(4), h0)
    for kq2 in range(2):
        row = b.add(b.add(b.mul(b.div(i_dyn, ci(4)), ci(8)), b.mod(i_dyn, ci(4))),
                    ci(4 * kq2))
        tt = b.add(b.mul(tb, ci(spec.tiles_per_wg)), row)
        ttx = b.mod(tt, ci(TX))
        tty = b.div(tt, ci(TX))
        yo = b.add(b.mul(tty, ci(2)), oy)
        xo = b.add(b.mul(ttx, ci(2)), oxl)
        src = b.add(ci(kq2 * 32), b.mul(g, ci(8)))
        oa = ob = None
        for bb in range(4):
            pa = ld.ld4(b.add(wbase, ci(bb)), jj, b.add(src, h0))
            pb = ld.ld4(b.add(wbase, ci(bb)), jj, b.add(src, h1))
            oa = b.vector_mul(cvec[bb], pa) if oa is None else b.vector_fma(cvec[bb], pa, oa)
            ob = b.vector_mul(cvec[bb], pb) if ob is None else b.vector_fma(cvec[bb], pb, ob)
        lo = b.select(is1, ob, oa)
        hi = b.select(is1, oa, ob)
        hh = getattr(spec, "yc8h_halo", 1)
        vox = b.add(b.mul(b.add(b.mul(b.add(b.mul(
            kb, ci(spec.Do + 2 * hh)), b.add(zz, ci(hh))), ci(Ho + 2 * hh)),
            b.add(yo, ci(hh))), ci(Wo + 2 * hh)), b.add(xo, ci(hh)))
        vox = _yx_lin_mask(b, spec, tt, vox, 8)
        vox = _yx_crop_mask(b, spec, zz, yo, xo, vox, 8)
        b.buffer_store_vN_bf16(
            ry, b.mul(vox, ci(16)), ci(0), b.vec_trunc_f32_to_bf16(b.vec_concat(lo, hi)), 4
        )


def build_winograd_yx(spec: WinogradYxSpec) -> KernelDef:
    """Dispatch on ``spec.stage``."""
    if spec.stage == "lds":
        return build_winograd_yx_lds_fwd(spec)
    if spec.stage == "lds16" and spec.persist:
        return build_winograd_yx_lds16p_fwd(spec)
    if spec.stage == "lds16":
        return build_winograd_yx_lds16_fwd(spec)
    if spec.stage == "wlds":
        return build_winograd_yx_wlds_fwd(spec)
    return build_winograd_yx_fwd(spec)


def build_winograd_yx_fwd(spec: WinogradYxSpec) -> KernelDef:
    ok, why = is_valid_yx_spec(spec)
    if not ok:
        raise ValueError(f"invalid winograd yx spec: {why}")
    f16 = spec.op_dtype == "f16"
    if f16:
        from rocke.core.ir import F16

        atom = MfmaAtom.f16_32x32x8()
        op_t = F16
    else:
        atom = MfmaAtom.bf16_32x32x8()
        op_t = BF16
    C, K = spec.C, spec.K
    Di, Hi, Wi = spec.Di, spec.Hi, spec.Wi
    Ho, Wo = spec.Ho, spec.Wo
    TX = Wo // 2
    ZT = spec.z_tile
    NPL = ZT + 2
    n_tb = spec.tiles2d // spec.tiles_per_wg

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.wpe:
        b.kernel.attrs["waves_per_eu"] = (spec.wpe, spec.wpe)
    if spec.agpr:
        b.kernel.attrs["agpr_alloc"] = spec.agpr
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(op_t, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))
    kq = b.div(lane, ci(32))
    bid = b.block_id_x()
    bid_n = b.block_id_y()
    tb = b.mod(bid, ci(n_tb))
    zb = b.div(bid, ci(n_tb))
    z0 = b.mul(zb, ci(ZT))

    t = b.add(b.mul(tb, ci(spec.tiles_per_wg)), m)
    tx = b.mod(t, ci(TX))
    ty = b.div(t, ci(TX))

    # wave-uniform y rows / combination sign (true V_b = ZS0[b] * (d[p0] + r d[p1]))
    is0 = b.cmp_eq(wave, ci(0))
    is3 = b.cmp_eq(wave, ci(3))
    is1 = b.cmp_eq(wave, ci(1))
    p0 = b.select(is0, ci(0), ci(1))
    p1 = b.select(is3, ci(3), ci(2))
    r = b.select(is1, b.const_f32(1.0), b.const_f32(-1.0))

    plane_e = Hi * Wi * 8
    chunk_e = Di * plane_e
    x_bytes = Di * Hi * Wi * C * 2
    u_bytes = 3 * 16 * C * K * 2
    rx = b.buffer_rsrc(X, ci(x_bytes))
    ru = b.buffer_rsrc(U, ci(u_bytes))
    vx = b.mul(
        b.add(b.mul(b.add(b.mul(b.mul(ty, ci(2)), ci(Wi)), b.mul(tx, ci(2))), ci(8)),
              b.mul(kq, ci(4))),
        ci(2),
    )
    s_rows = (b.mul(p0, ci(Wi * 8 * 2)), b.mul(p1, ci(Wi * 8 * 2)))
    s_z0 = b.mul(z0, ci(plane_e * 2))
    n_glob = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
    vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
    u_pt = (C // 8) * K * 8  # elements per (dz, b, c) point
    s_ub = b.mul(wave, ci(4 * u_pt * 2))

    def load_plane(cc, zi):
        base = b.add(b.add(b.mul(cc, ci(chunk_e * 2)), s_z0), ci(zi * plane_e * 2))
        raw = []
        for sr in s_rows:
            bs = b.add(base, sr)
            for x in range(4):
                raw.append(b.buffer_load_vN(rx, vx, b.add(bs, ci(x * 16)), I32, 2))
        return raw

    def load_b(cc):
        su = b.add(s_ub, b.mul(cc, ci(K * 8 * 2)))
        return {
            (dz, c): b.buffer_load_vN(
                ru, vu, b.add(su, ci(((dz * 4) * 4 + c) * u_pt * 2)), I32, 2
            )
            for dz in range(3)
            for c in range(4)
        }

    load_b.keys_ = [(dz, c) for dz in range(3) for c in range(4)]

    def transform(raw):
        """8 raw <2 x i32> (= 4 bf16; rows p0, p1 x 4 x) -> 4 A fragments (points c)."""

        def bf16x2_to_f32(d):
            lo = b.bitcast(b.shl(d, ci(16)), F32)
            hi = b.bitcast(b.land(d, ci(-65536)), F32)
            return [lo, hi]

        ycomb = []
        for x in range(4):
            v0, v1 = raw[x], raw[4 + x]
            f0 = bf16x2_to_f32(b.vec_extract(v0, 0)) + bf16x2_to_f32(b.vec_extract(v0, 1))
            f1 = bf16x2_to_f32(b.vec_extract(v1, 0)) + bf16x2_to_f32(b.vec_extract(v1, 1))
            ycomb.append([b.fma(r, f1[j], f0[j]) for j in range(4)])
        frags = []
        for c in range(4):
            vals = [_bt_axis(b, [ycomb[x][j] for x in range(4)], BT[c]) for j in range(4)]
            frags.append(_pack4_f16_rtz(b, vals) if f16 else _pack4_bf16_half_up(b, vals))
        return frags

    n_chunks = C // spec.c_chunk
    # Software pipeline over the flattened (chunk, plane) sequence: plane s+LA is loaded
    # while plane s is transformed; the first LA planes of the next chunk and the next
    # chunk's weight fragments are carried through the loop.
    LA = spec.pf_planes
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZT)]
    if LA:
        for zi in range(LA):
            iters += [(f"pf{zi}_{k}", v) for k, v in enumerate(load_plane(ci(0), zi))]
        b0 = load_b(ci(0))
        iters += [(f"bf{i}", b0[k]) for i, k in enumerate(sorted(b0))]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
    with loop as (cc, its):
        accs = list(its[: 4 * ZT])
        # clamped: the buffer range check ignores soffset (see the lds16 stage)
        cn = b.smin(b.add(cc, ci(1)), ci(n_chunks - 1))
        if LA:
            pre = its[4 * ZT: 4 * ZT + 8 * LA]
            bvals = its[4 * ZT + 8 * LA:]
            bf = dict(zip(sorted(load_b.keys_), bvals))
            planes = {zi: pre[zi * 8:(zi + 1) * 8] for zi in range(LA)}
            nb = load_b(cn)
        else:
            bf = load_b(cc)
            planes = {}
        nxt = []
        for zi in range(NPL):
            if LA:
                tgt = zi + LA
                if tgt < NPL:
                    planes[tgt] = load_plane(cc, tgt)
                else:
                    nxt += load_plane(cn, tgt - NPL)
                raw = planes.pop(zi)
            else:
                raw = load_plane(cc, zi)
            frags = transform(raw)
            for dz in range(3):
                zo = zi - dz
                if 0 <= zo < ZT:
                    for c in range(4):
                        k = zo * 4 + c
                        accs[k] = atom.emit(
                            b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                            accs[k],
                        )
            if spec.sbar:
                b.sched_barrier(0)
        if LA:
            b.scf_yield(*accs, *nxt, *[nb[k] for k in sorted(nb)])
        else:
            b.scf_yield(*accs)
    acc = loop.results[: 4 * ZT]

    _yx_epilogue(b, spec, acc, wave, lane, kq, tb, z0, n_glob, Y)
    b.ret()
    return b.kernel


def _yx_lds_geom(spec):
    TX = spec.Wo // 2
    if getattr(spec, "lin", False):
        return TX, 32, 1
    TCx = min(TX, 32)
    TR = 32 // TCx
    return TX, TCx, TR


def _yx_lds_floats(spec) -> int:
    _, TCx, TR = _yx_lds_geom(spec)
    return (spec.z_tile + 2) * TR * 4 * 2 * (TCx + 1) * 8


def build_winograd_yx_lds_fwd(spec: WinogradYxSpec) -> KernelDef:
    """F(2x2,3x3) on (y,x) + direct z, input planes staged in LDS (see ``stage``)."""
    ok, why = is_valid_yx_spec(spec)
    if not ok or spec.stage != "lds":
        raise ValueError(f"invalid winograd yx lds spec: {why}")
    f16 = spec.op_dtype == "f16"
    if f16:
        from rocke.core.ir import F16

        atom = MfmaAtom.f16_32x32x8()
        op_t = F16
    else:
        atom = MfmaAtom.bf16_32x32x8()
        op_t = BF16
    C, K = spec.C, spec.K
    Di, Hi, Wi = spec.Di, spec.Hi, spec.Wi
    ZT = spec.z_tile
    NPL = ZT + 2
    TX, TCx, TR = _yx_lds_geom(spec)
    NC = 2 * TCx + 2  # staged input columns
    HC = TCx + 1  # columns per parity
    n_tb = spec.tiles2d // spec.tiles_per_wg
    items = NPL * TR * NC
    rounds = (items + spec.block_size - 1) // spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.wpe:
        b.kernel.attrs["waves_per_eu"] = (spec.wpe, spec.wpe)
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(op_t, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    wb = b.mod(wave, ci(4))  # y-point b of this wave
    zh = b.div(wave, ci(4))  # z group
    NG = spec.waves // 4
    ZTW = ZT // NG  # output planes per wave
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))
    kq = b.div(lane, ci(32))
    bid = b.block_id_x()
    bid_n = b.block_id_y()
    tb = b.mod(bid, ci(n_tb))
    zb = b.div(bid, ci(n_tb))
    z0 = b.mul(zb, ci(ZT))
    t0 = b.mul(tb, ci(spec.tiles_per_wg))
    ty0 = b.div(t0, ci(TX))
    tx0 = b.mod(t0, ci(TX))

    lds = b.smem_alloc(
        F32, [max(_yx_lds_floats(spec), spec.waves * 32 * 64)], name_hint="winoyx_planes"
    )

    plane_b = Hi * Wi * 16
    chunk_b = Di * plane_b
    rx = b.buffer_rsrc(X, ci(Di * Hi * Wi * C * 2))
    ru = b.buffer_rsrc(U, ci(3 * 16 * C * K * 2))
    n_glob = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
    vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
    u_pt = (C // 8) * K * 8
    s_ub = b.mul(wb, ci(4 * u_pt * 2))

    # Fill items: (plane zi, tile row tr, column col) -> 4 input rows of 8 channels.
    fill = []
    for rr in range(rounds):
        it = b.add(tid, ci(rr * spec.block_size))
        valid = b.cmp_lt(it, ci(items))
        col = b.mod(it, ci(NC))
        tmp = b.div(it, ci(NC))
        tr = b.mod(tmp, ci(TR))
        zi = b.div(tmp, ci(TR))
        voff = b.add(
            b.add(
                b.mul(b.add(z0, zi), ci(plane_b)),
                b.mul(b.add(b.mul(ty0, ci(2)), b.mul(tr, ci(2))), ci(Wi * 16)),
            ),
            b.mul(b.add(b.mul(tx0, ci(2)), col), ci(16)),
        )
        # LDS (f32 units) of b = 0, this item's parity / half-column
        par = b.mod(col, ci(2))
        half = b.div(col, ci(2))
        lbase = b.add(
            b.mul(b.add(b.mul(b.mul(b.add(b.mul(zi, ci(TR)), tr), ci(4)), ci(2)), par), ci(HC * 8)),
            b.mul(half, ci(8)),
        )
        fill.append((valid, voff, lbase))

    def load_fill(cc):
        sx = b.mul(cc, ci(chunk_b))
        out = []
        for valid, voff, _ in fill:
            for j in range(4):
                out.append(b.buffer_load_vN(rx, voff, b.add(sx, ci(j * Wi * 16)), I32, 4))
        return out

    def bf16x2_to_f32(d):
        lo = b.bitcast(b.shl(d, ci(16)), F32)
        hi = b.bitcast(b.land(d, ci(-65536)), F32)
        return [lo, hi]

    def store_fill(raw):
        for rr, (valid, _, lbase) in enumerate(fill):
            rows = []
            for j in range(4):
                v = raw[rr * 4 + j]
                f = []
                for q in range(4):
                    f += bf16x2_to_f32(b.vec_extract(v, q))
                rows.append(f)  # 8 channels
            with b.scf_if(valid):
                for bb in range(4):
                    vals = [_bt_axis(b, [rows[j][ch] for j in range(4)], BT[bb]) for ch in range(8)]
                    off = b.add(lbase, ci(bb * 2 * HC * 8))
                    b.smem_store_vN_f32(lds, [off], b.vec_pack(vals[:4], F32), 4)
                    b.smem_store_vN_f32(lds, [b.add(off, ci(4))], b.vec_pack(vals[4:], F32), 4)

    def load_b(cc):
        su = b.add(s_ub, b.mul(cc, ci(K * 8 * 2)))
        return {
            (dz, c): b.buffer_load_vN(
                ru, vu, b.add(su, ci(((dz * 4) * 4 + c) * u_pt * 2)), I32, 2
            )
            for dz in range(3)
            for c in range(4)
        }

    # Per-lane read base: tile row / half-column of this lane, its wave's b, channel quad.
    l_tr = b.div(m, ci(TCx))
    l_hx = b.mod(m, ci(TCx))
    rbase = b.add(b.mul(l_hx, ci(8)), b.mul(kq, ci(4)))

    def read_plane(zl):
        zi = b.add(b.mul(zh, ci(ZTW)), ci(zl))  # this wave's plane zl -> block plane
        vs = []
        for x in range(4):
            par = x % 2
            # (((zi*TR + tr)*4 + w)*2 + par)*HC*8 + (hx + x//2)*8 + 4kq
            off = b.add(
                b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(zi, ci(TR)), l_tr), ci(4)), wb),
                                  ci(2)), ci(par)), ci(HC * 8)),
                b.add(rbase, ci((x // 2) * 8)),
            )
            vs.append(b.smem_load_vN_f32(lds, off, n=4))
        return vs

    def x_transform_pack(vs):
        """B^T along x on <4 x f32> (4 channels), rounded/packed to A fragments."""
        frags = []
        for c in range(4):
            v = None
            for x in range(4):
                co = BT[c][x]
                if co == 0:
                    continue
                if v is None:
                    v = vs[x] if co > 0 else b.vector_sub(b.vector_splat(b.const_f32(0.0), 4), vs[x])
                else:
                    v = b.vector_add(v, vs[x]) if co > 0 else b.vector_sub(v, vs[x])
            if f16:
                from rocke.core.ir import F16

                frags.append(b.vec_cast_f32_to(v, F16))
                continue
            vi = b.vec_bitcast(v, VectorType(I32, 4))
            if spec.op_dtype == "bf16":  # round half up; "bf16t" truncates
                vi = b.vector_add(vi, b.vector_splat(ci(0x8000), 4))
            e = [b.vec_extract(vi, j) for j in range(4)]
            d0 = b.perm_b32(e[1], e[0], ci(_PERM_HI16))
            d1 = b.perm_b32(e[3], e[2], ci(_PERM_HI16))
            frags.append(b.vec_bitcast(b.vec_pack([d0, d1], I32), VectorType(BF16, 4)))
        return frags

    n_chunks = C // spec.c_chunk
    raw0 = load_fill(ci(0))
    bkeys = [(dz, c) for dz in range(3) for c in range(4)]
    b0 = load_b(ci(0))
    nfill = len(raw0)
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
    iters += [(f"raw{i}", v) for i, v in enumerate(raw0)]
    iters += [(f"bfr{i}", b0[k]) for i, k in enumerate(bkeys)]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
    with loop as (cc, its):
        accs = list(its[: 4 * ZTW])
        raw = its[4 * ZTW: 4 * ZTW + nfill]
        bf = dict(zip(bkeys, its[4 * ZTW + nfill:]))
        # clamped: the buffer range check ignores soffset (see the lds16 stage)
        cn = b.smin(b.add(cc, ci(1)), ci(n_chunks - 1))
        nraw = load_fill(cn)
        nb = load_b(cn)
        if "nofill" not in spec.dbg:
            store_fill(raw)
        # LDS-only barriers: the next chunk's loads stay in flight across them.
        if "nowait" not in spec.dbg:
            b.sync_lds_only()
        for zi in range(ZTW + 2):
            frags = x_transform_pack(read_plane(zi))
            for dz in range(3):
                zo = zi - dz
                if 0 <= zo < ZTW and "nomfma" not in spec.dbg:
                    for c in range(4):
                        k = zo * 4 + c
                        accs[k] = atom.emit(
                            b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                            accs[k],
                        )
            if spec.sbar:
                b.sched_barrier(0)
        if "nowait" not in spec.dbg:
            b.sync_lds_only()
        b.scf_yield(*accs, *nraw, *[nb[k] for k in bkeys])
    acc = loop.results[: 4 * ZTW]
    _yx_epilogue(
        b, spec, acc, wave, lane, kq, tb, b.add(z0, b.mul(zh, ci(ZTW))), n_glob, Y,
        lds_flat=lds, zt_w=ZTW, wb=wb, zh=zh,
    )
    b.ret()
    return b.kernel


def _lds16_loop_b16(
    b, spec, atom, op_t, ZTW, BUF, n_chunks, bkeys, load_fill, store_fill, read_plane,
    x_transform, ru, vu_unused, s_ub_unused, wave, lane, kq, tb, z0, zh, wb, grp, n_glob, Y,
    lds,
):
    """lds16 main loop unrolled by two chunks with 16-byte weight-fragment loads."""
    ci = b.const_i32
    C, K = spec.C, spec.K
    u_pt = (C // 16) * K * 16  # elements per (dz, b, c) point in the u16 layout
    vu = b.mul(b.add(b.mul(n_glob, ci(16)), b.mul(kq, ci(8))), ci(2))
    s_ub = b.mul(wb, ci(4 * u_pt * 2))
    n_pairs = n_chunks // 2

    def load_b16(cp):
        su = b.add(s_ub, b.mul(cp, ci(K * 16 * 2)))
        return [
            b.buffer_load_vN(ru, vu, b.add(su, ci(((dz * 4) * 4 + c) * u_pt * 2)), I32, 4)
            for dz, c in bkeys
        ]

    store_fill(load_fill(ci(0)), ci(0))
    raw1 = load_fill(ci(min(1, n_chunks - 1)))
    b0 = load_b16(ci(0))
    b.sync_lds_only()
    nfill = len(raw1)
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
    iters += [(f"raw{i}", v) for i, v in enumerate(raw1)]
    iters += [(f"bfr{i}", v) for i, v in enumerate(b0)]
    loop = b.scf_for_iter(ci(0), ci(n_pairs), ci(1), iters, iv_name="cp")
    with loop as (cp, its):
        accs = list(its[: 4 * ZTW])
        raw = its[4 * ZTW: 4 * ZTW + nfill]
        bpair = its[4 * ZTW + nfill:]
        nb = load_b16(b.smin(b.add(cp, ci(1)), ci(n_pairs - 1)))
        for sub in range(2):
            c_cur = b.add(b.mul(cp, ci(2)), ci(sub))
            cur = ci(sub * BUF)
            nxt_off = ci((1 - sub) * BUF)
            nraw = load_fill(b.smin(b.add(c_cur, ci(2)), ci(n_chunks - 1)))
            bf = {}
            for k_, key in enumerate(bkeys):
                d = bpair[k_]
                bf[key] = b.vec_pack([b.vec_extract(d, 2 * sub), b.vec_extract(d, 2 * sub + 1)],
                                     I32)
            for zi in range(ZTW + 2):
                frags = x_transform(read_plane(zi, cur))
                for dz in range(3):
                    zo = zi - dz
                    if 0 <= zo < ZTW:
                        for c in range(4):
                            k = zo * 4 + c
                            accs[k] = atom.emit(
                                b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                                accs[k],
                            )
            store_fill(raw, nxt_off)
            b.sync_lds_only()
            raw = nraw
        b.scf_yield(*accs, *raw, *nb)
    acc = loop.results[: 4 * ZTW]
    _yx_epilogue(
        b, spec, acc, wave, lane, kq, tb, b.add(z0, b.mul(zh, ci(ZTW))), n_glob, Y,
        lds_flat=lds, zt_w=ZTW, wb=wb, zh=grp, gpr=2,
    )


def build_winograd_yx_lds16p_fwd(spec: WinogradYxSpec) -> KernelDef:
    """Persistent variant of the lds16 stage: ``spec.persist`` workgroups loop over the
    blocks (n blocks fastest, as nfast); the loads of the next block's first two chunks and
    weights are issued before the current block's epilogue, so their latency is hidden."""
    from rocke.core.ir import F16

    ok, why = is_valid_yx_spec(spec)
    if not ok or not spec.persist:
        raise ValueError(f"invalid winograd yx lds16p spec: {why}")
    atom = MfmaAtom.f16_32x32x8()
    op_t = F16
    C, K = spec.C, spec.K
    Di, Hi, Wi = spec.Di, spec.Hi, spec.Wi
    ZT = spec.z_tile
    NPL = ZT + 2
    TX, TCx, TR = _yx_lds_geom(spec)
    NC = 2 * TCx + 2
    HC = TCx + 1
    BUF = _yx_lds16_buf(spec)
    n_tb = spec.tiles2d // spec.tiles_per_wg
    NB = K // spec.n_tile
    NBLK = _yx_nblocks(spec)
    G = min(spec.persist, NBLK)
    NK = -(-NBLK // G)
    items = NPL * TR * NC

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.wpe:
        b.kernel.attrs["waves_per_eu"] = (spec.wpe, spec.wpe)
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(op_t, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    wb = b.mod(wave, ci(4))
    NN = 2 if spec.nsplit else spec.nn
    NZ = (spec.waves // 4) // NN
    grp = b.div(wave, ci(4))
    nh = b.div(grp, ci(NZ))
    zh = b.mod(grp, ci(NZ))
    ZTW = ZT // NZ
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))
    kq = b.div(lane, ci(32))
    wg = b.block_id_x()

    lds = b.smem_alloc(
        F32, [max(_yx_lds16_floats(spec), min(spec.waves, 8) * 32 * 64)],
        name_hint="winoyx16p_planes",
    )
    plane_b = Hi * Wi * 16
    chunk_b = Di * plane_b
    rx = b.buffer_rsrc(X, ci(Di * Hi * Wi * C * 2))
    ru = b.buffer_rsrc(U, ci(3 * 16 * C * K * 2))
    u_pt = (C // 8) * K * 8
    s_ub = b.mul(wb, ci(4 * u_pt * 2))

    # Block-independent part of the fill items (lane -> item), as fill_il / tid order.
    per_wave = -(-items // spec.waves)
    if spec.fill_il:
        rounds = -(-per_wave // 64)
    else:
        rounds = (items + spec.block_size - 1) // spec.block_size
    items_l = []
    for rr in range(rounds):
        if spec.fill_il:
            loc = b.add(lane, ci(rr * 64))
            it = b.add(b.mul(wave, ci(per_wave)), loc)
            valid = b.land(b.cmp_lt(loc, ci(per_wave)), b.cmp_lt(it, ci(items)))
        else:
            it = b.add(tid, ci(rr * spec.block_size))
            valid = b.cmp_lt(it, ci(items))
        col = b.mod(it, ci(NC))
        tmp = b.div(it, ci(NC))
        tr = b.mod(tmp, ci(TR))
        zi = b.div(tmp, ci(TR))
        # block-independent part of the global offset and the LDS base
        voff0 = b.add(
            b.add(b.mul(zi, ci(plane_b)), b.mul(b.mul(tr, ci(2)), ci(Wi * 16))),
            b.mul(col, ci(16)),
        )
        par = b.mod(col, ci(2))
        half = b.div(col, ci(2))
        lbase = b.add(
            b.mul(b.add(b.mul(b.mul(b.mul(b.add(b.mul(zi, ci(TR)), tr), ci(4)), ci(2)), ci(2)),
                        par), ci(HC * 2)),
            b.mul(half, ci(2)),
        )
        items_l.append((valid, voff0, lbase))

    def setup(blk):
        """Block coordinates and the lane offsets that depend on them."""
        blk = b.smin(blk, ci(NBLK - 1))  # the tail repeats the last block (same values)
        bid_n = b.mod(blk, ci(NB))
        bid = b.div(blk, ci(NB))
        tb = b.mod(bid, ci(n_tb))
        zb = b.div(bid, ci(n_tb))
        z0 = b.mul(zb, ci(ZT))
        t0 = b.mul(tb, ci(spec.tiles_per_wg))
        ty0 = b.div(t0, ci(TX))
        tx0 = b.mod(t0, ci(TX))
        boff = b.add(b.add(b.mul(z0, ci(plane_b)), b.mul(b.mul(ty0, ci(2)), ci(Wi * 16))),
                     b.mul(b.mul(tx0, ci(2)), ci(16)))
        n_glob = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
        if NN > 1:
            n_glob = b.add(n_glob, b.mul(nh, ci(32)))
        vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
        return dict(tb=tb, z0=z0, boff=boff, n_glob=n_glob, vu=vu)

    def load_fill(S, cc):
        sx = b.add(b.mul(cc, ci(chunk_b)), S["boff"])
        out = []
        for valid, voff0, _ in items_l:
            for j in range(4):
                out.append(b.buffer_load_vN(rx, voff0, b.add(sx, ci(j * Wi * 16)), I32, 4))
        return out

    def bf16x2_to_f32(d):
        lo = b.bitcast(b.shl(d, ci(16)), F32)
        hi = b.bitcast(b.land(d, ci(-65536)), F32)
        return [lo, hi]

    def store_fill(raw, bufoff):
        for rr, (valid, _, lbase) in enumerate(items_l):
            rows = []
            for j in range(4):
                v = raw[rr * 4 + j]
                f = []
                for q in range(4):
                    f += bf16x2_to_f32(b.vec_extract(v, q))
                rows.append(f)
            with b.scf_if(valid):
                base = b.add(lbase, bufoff)
                for bb in range(4):
                    vals = [_bt_axis(b, [rows[j][ch] for j in range(4)], BT[bb])
                            for ch in range(8)]
                    for q in range(2):
                        if spec.pkrtz:
                            pk = [
                                b.inline_asm(
                                    "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v",
                                    [vals[4 * q + 2 * h2], vals[4 * q + 2 * h2 + 1]], I32,
                                    sideeffect=False,
                                )
                                for h2 in range(2)
                            ]
                            w2 = b.vec_bitcast(b.vec_pack(pk, I32), VectorType(F32, 2))
                        else:
                            h = b.vec_cast_f32_to(b.vec_pack(vals[4 * q:4 * q + 4], F32), F16)
                            w2 = b.vec_bitcast(h, VectorType(F32, 2))
                        off = b.add(base, ci((bb * 2 + q) * 2 * HC * 2))
                        b.smem_store_vN_f32(lds, [off], w2, 2)

    bkeys = [(dz, c) for dz in range(3) for c in range(4)]

    def load_b(S, cc):
        su = b.add(s_ub, b.mul(cc, ci(K * 8 * 2)))
        return [
            b.buffer_load_vN(ru, S["vu"], b.add(su, ci(((dz * 4) * 4 + c) * u_pt * 2)), I32, 2)
            for dz, c in bkeys
        ]

    l_tr = b.div(m, ci(TCx))
    l_hx = b.mod(m, ci(TCx))

    def read_plane(zl, bufoff):
        zi = b.add(b.mul(zh, ci(ZTW)), ci(zl))
        rowbase = b.mul(
            b.add(b.mul(b.add(b.mul(b.add(b.mul(zi, ci(TR)), l_tr), ci(4)), wb), ci(2)), kq),
            ci(2),
        )
        vs = []
        for x in range(4):
            off = b.add(
                b.mul(b.add(b.mul(b.add(rowbase, ci(x % 2)), ci(HC)), b.add(l_hx, ci(x // 2))),
                      ci(2)),
                bufoff,
            )
            w2 = b.smem_load_vN_f32(lds, off, n=2)
            vs.append(b.vec_bitcast(w2, VectorType(F16, 4)))
        return vs

    def x_transform(vs):
        return [
            b.vector_sub(vs[0], vs[2]),
            b.vector_add(vs[1], vs[2]),
            b.vector_sub(vs[2], vs[1]),
            b.vector_sub(vs[1], vs[3]),
        ]

    n_chunks = C // spec.c_chunk
    # first block: chunk 0 staged, chunk 1 and chunk 0's weights in registers
    S0 = setup(wg)
    store_fill(load_fill(S0, ci(0)), ci(0))
    raw1 = load_fill(S0, ci(min(1, n_chunks - 1)))
    b0 = load_b(S0, ci(0))
    b.sync_lds_only()
    nfill = len(raw1)
    outer = b.scf_for_iter(
        ci(0), ci(NK), ci(1),
        [(f"oraw{i}", v) for i, v in enumerate(raw1)] + [(f"ob{i}", v) for i, v in enumerate(b0)],
        iv_name="kb",
    )
    with outer as (kb, oits):
        S = setup(b.add(wg, b.mul(kb, ci(G))))
        iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
        iters += [(f"raw{i}", v) for i, v in enumerate(oits[:nfill])]
        iters += [(f"bfr{i}", v) for i, v in enumerate(oits[nfill:])]
        loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
        with loop as (cc, its):
            accs = list(its[: 4 * ZTW])
            raw = its[4 * ZTW: 4 * ZTW + nfill]
            bf = dict(zip(bkeys, its[4 * ZTW + nfill:]))
            par_c = b.mod(cc, ci(2))
            cur = b.mul(par_c, ci(BUF))
            nxt_off = b.mul(b.sub(ci(1), par_c), ci(BUF))
            nraw = load_fill(S, b.smin(b.add(cc, ci(2)), ci(n_chunks - 1)))
            nb = load_b(S, b.smin(b.add(cc, ci(1)), ci(n_chunks - 1)))
            for zi in range(ZTW + 2):
                frags = x_transform(read_plane(zi, cur))
                for dz in range(3):
                    zo = zi - dz
                    if 0 <= zo < ZTW:
                        for c in range(4):
                            k = zo * 4 + c
                            accs[k] = atom.emit(
                                b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                                accs[k],
                            )
            store_fill(raw, nxt_off)
            b.sync_lds_only()
            b.scf_yield(*accs, *nraw, *nb)
        acc = loop.results[: 4 * ZTW]
        # the next block's first loads, in flight during this block's epilogue
        Sn = setup(b.add(wg, b.mul(b.add(kb, ci(1)), ci(G))))
        nraw0 = load_fill(Sn, ci(0))
        nraw1 = load_fill(Sn, ci(min(1, n_chunks - 1)))
        nb0 = load_b(Sn, ci(0))
        _yx_epilogue(
            b, spec, acc, wave, lane, kq, S["tb"], b.add(S["z0"], b.mul(zh, ci(ZTW))),
            S["n_glob"], Y, lds_flat=lds, zt_w=ZTW, wb=wb, zh=grp, gpr=2,
        )
        b.sync_lds_only()
        store_fill(nraw0, ci(0))
        b.sync_lds_only()
        b.scf_yield(*nraw1, *nb0)
    b.ret()
    return b.kernel


def _yx_lds16_ps(spec) -> int:
    """Column-parity stride (8-byte slots) of the lds16 staging buffers."""
    _, TCx, _ = _yx_lds_geom(spec)
    ps = TCx + 1 + _yx_lin_seams(spec)
    if getattr(spec, "ppad", False):
        while ps % 16 != 8:
            ps += 1
    return ps


def _yx_lds16_buf(spec) -> int:
    """f32 words of one staged-chunk buffer of the lds16 stage."""
    _, TCx, TR = _yx_lds_geom(spec)
    return (spec.z_tile + 2) * TR * 4 * 2 * 2 * _yx_lds16_ps(spec) * 2


def _yx_lds16_floats(spec) -> int:
    f = (1 if spec.single_buf else 2) * _yx_lds16_buf(spec)
    if spec.bshare:
        nn = 2 if spec.nsplit else spec.nn
        f += 48 * 32 * nn * 8 // 2  # f16 weight fragments of one chunk, f32 words
    return f


def build_winograd_yx_lds16_fwd(spec: WinogradYxSpec) -> KernelDef:
    """F(2x2,3x3) on (y,x) + direct z with f16 staging (``stage="lds16"``).

    The workgroup converts each chunk's input planes once: B^T along y in f32, stored as
    f16 in a double-buffered LDS ring ([plane][tile row][b][channel quad][column parity]
    [half column][4 ch]); waves apply B^T along x in packed f16 (v_pk_add_f16), which is
    already the f16 MFMA A operand. One barrier per chunk: the fill of chunk c+1 (from
    registers loaded during chunk c-1) overlaps other waves' MFMAs on chunk c.
    Range: |x| below ~16k (f16 transformed operands).
    """
    from rocke.core.ir import F16

    ok, why = is_valid_yx_spec(spec)
    if not ok or spec.stage != "lds16":
        raise ValueError(f"invalid winograd yx lds16 spec: {why}")
    atom = MfmaAtom.f16_32x32x8()
    op_t = F16
    C, K = spec.C, spec.K
    Di, Hi, Wi = spec.Di, spec.Hi, spec.Wi
    ZT = spec.z_tile
    NPL = ZT + 2
    TX, TCx, TR = _yx_lds_geom(spec)
    NC = 2 * TCx + 2
    HC = _yx_lds16_ps(spec)  # parity stride (slots); TCx + 1 unless ppad
    BUF = _yx_lds16_buf(spec)
    n_tb = _yx_ntb(spec)
    LIN = spec.lin
    NSEAM = _yx_lin_seams(spec)
    if LIN:
        NC = 2 * TCx + 2 + 2 * NSEAM  # 66 virtual columns + the seam column pairs
    items = NPL * TR * NC
    rounds = (items + spec.block_size - 1) // spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.wpe:
        b.kernel.attrs["waves_per_eu"] = (spec.wpe, spec.wpe)
    if spec.frel:
        # X (c8) is written by the producers of this very launch
        X = b.param("X", PtrType(BF16, "global"), noalias=True, align=16)
    else:
        X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(op_t, "global"), noalias=True, readonly=True, align=16)
    KS = spec.ksplit
    Y = b.param("Y", PtrType(F32 if KS > 1 else BF16, "global"), noalias=True, writeonly=True,
                align=16)
    if spec.frel:
        Xn = b.param("Xn", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
        Fl = b.param("F", PtrType(I32, "global"), noalias=True, align=4)
    if spec.boff:
        BOFF = b.param("BOFF", I32)
    OS = b.param("OS", F32) if spec.oscale else None

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    wb = b.mod(wave, ci(4))
    NN = 2 if spec.nsplit else spec.nn
    NZ = (spec.waves // 4) // NN
    grp = b.div(wave, ci(4))
    nh = b.div(grp, ci(NZ))  # output-channel group
    zh = b.mod(grp, ci(NZ))  # z group
    ZTW = ZT // NZ
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))
    kq = b.div(lane, ci(32))
    if spec.nfast:
        nb_ = spec.K // spec.n_tile
        pid = b.block_id_x()
        ks_i = None
        if KS > 1 and spec.xcd:
            # ksplit + xcd: remap the whole (split-major) grid first, then split it
            nwg = winograd_yx_grid(spec)[0]
            q_, r_ = nwg // spec.xcd, nwg % spec.xcd
            xx = b.mod(pid, ci(spec.xcd))
            kk = b.div(pid, ci(spec.xcd))
            pid = b.add(b.add(b.mul(xx, ci(q_)), b.smin(xx, ci(r_))), kk)
        if KS > 1:
            nwg0 = winograd_yx_grid(spec)[0] // KS
            ks_i = b.div(pid, ci(nwg0))
            pid = b.mod(pid, ci(nwg0))
        if spec.boff:
            pid = b.add(pid, BOFF)
        if spec.xcd and KS == 1:
            nwg = winograd_yx_grid(spec)[0]
            q_, r_ = nwg // spec.xcd, nwg % spec.xcd
            xx = b.mod(pid, ci(spec.xcd))
            kk = b.div(pid, ci(spec.xcd))
            pid = b.add(b.add(b.mul(xx, ci(q_)), b.smin(xx, ci(r_))), kk)
        bid_n = b.mod(pid, ci(nb_))
        bid = b.div(pid, ci(nb_))
    else:
        bid = b.block_id_x()
        bid_n = b.block_id_y()
    tb = b.mod(bid, ci(n_tb))
    zb = b.div(bid, ci(n_tb))
    z0 = b.mul(zb, ci(ZT))
    t0 = b.mul(tb, ci(spec.tiles_per_wg))
    ty0 = b.div(t0, ci(TX))
    tx0 = b.mod(t0, ci(TX))

    lds_words = max(_yx_lds16_floats(spec), min(spec.waves, 8) * 32 * 64)
    if spec.bf_fill:
        lds_words = max(lds_words, 2 * BUF + 8 * 2 * HC * 2)
    lds = b.smem_alloc(F32, [lds_words], name_hint="winoyx16_planes")

    xn = spec.xin == "ndhwc"
    r8 = spec.xin == "r8"
    vox_b = C * 2 if xn else 16  # bytes between neighbouring voxels
    row_b = Wi * C * 2 if (xn or r8) else Wi * 16  # bytes between neighbouring rows
    plane_b = Hi * row_b
    chunk_b = 16 if xn else (Wi * 16 if r8 else Di * plane_b)
    rx = b.buffer_rsrc(X, ci(Di * Hi * Wi * C * 2))
    ru = b.buffer_rsrc(U, ci(3 * 16 * C * K * 2))
    n_glob = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
    if NN > 1:
        n_glob = b.add(n_glob, b.mul(nh, ci(32)))
    vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
    u_pt = (C // 8) * K * 8
    s_ub = b.mul(wb, ci(4 * u_pt * 2))

    fill = []
    per_wave = -(-items // spec.waves)
    if spec.fill_il:
        rounds = -(-per_wave // 64)
    FH = spec.fhalf
    if FH:
        rounds = -(-(2 * items) // spec.block_size)
    for rr in range(rounds):
        if FH:
            it2 = b.add(tid, ci(rr * spec.block_size))
            valid = b.cmp_lt(it2, ci(2 * items))
            it = b.div(it2, ci(2))
            hq = b.mod(it2, ci(2))
        elif spec.fill_il:
            # wave w stages the contiguous items [w * per_wave, (w + 1) * per_wave)
            loc = b.add(lane, ci(rr * 64))
            it = b.add(b.mul(wave, ci(per_wave)), loc)
            valid = b.land(b.cmp_lt(loc, ci(per_wave)), b.cmp_lt(it, ci(items)))
        else:
            it = b.add(tid, ci(rr * spec.block_size))
            valid = b.cmp_lt(it, ci(items))
        col = b.mod(it, ci(NC))
        tmp = b.div(it, ci(NC))
        tr = b.mod(tmp, ci(TR))
        zi = b.div(tmp, ci(TR))
        if LIN:
            # virtual column v < 66: lane mm = min(v // 2, 31), its column e = v - 2 mm of
            # tile t0 + mm; seam pair s (v >= 66): the right-hand columns 2 TX, 2 TX + 1 of
            # the s-th row end inside the block (lane m0 + s TX, m0 = TX - 1 - tx0)
            NV = 2 * TCx + 2
            is_sm = b.cmp_ge(col, ci(NV))
            sv = b.smax(b.sub(col, ci(NV)), ci(0))
            s_ = b.div(sv, ci(2))
            es = b.mod(sv, ci(2))
            m_s = b.add(b.sub(ci(TX - 1), tx0), b.mul(s_, ci(TX)))
            mm = b.smin(b.div(col, ci(2)), ci(TCx - 1))
            e_ = b.sub(col, b.mul(mm, ci(2)))
            lt = b.select(is_sm, m_s, mm)
            tt_ = b.add(t0, lt)
            tyy = b.div(tt_, ci(TX))
            txx = b.mod(tt_, ci(TX))
            xcol = b.select(is_sm, b.add(ci(2 * TX), es), b.add(b.mul(txx, ci(2)), e_))
            voff = b.add(
                b.add(b.mul(b.add(z0, zi), ci(plane_b)), b.mul(b.mul(tyy, ci(2)), ci(row_b))),
                b.mul(xcol, ci(vox_b)),
            )
            par = b.select(is_sm, es, b.mod(col, ci(2)))
            half = b.select(is_sm, b.add(ci(TCx + 1), s_), b.div(col, ci(2)))
            lbase = b.add(
                b.mul(b.add(b.mul(b.mul(b.mul(zi, ci(4)), ci(2)), ci(2)), par), ci(HC * 2)),
                b.mul(half, ci(2)),
            )
            ok_sm = b.lor(b.cmp_lt(col, ci(NV)), b.cmp_lt(m_s, ci(TCx - 1)))
            if spec.fill_il:
                valid = b.land(valid, ok_sm)
            else:
                valid = b.land(valid, ok_sm)
            fill.append((valid, voff, lbase))
            continue
        voff = b.add(
            b.add(
                b.mul(b.add(z0, zi), ci(plane_b)),
                b.mul(b.add(b.mul(ty0, ci(2)), b.mul(tr, ci(2))), ci(row_b)),
            ),
            b.mul(b.add(b.mul(tx0, ci(2)), col), ci(vox_b)),
        )
        par = b.mod(col, ci(2))
        half = b.div(col, ci(2))
        # f32-word offset of (zi, tr, b=0, kq=0, par, half)
        lbase = b.add(
            b.mul(b.add(b.mul(b.mul(b.mul(b.add(b.mul(zi, ci(TR)), tr), ci(4)), ci(2)), ci(2)),
                        par), ci(HC * 2)),
            b.mul(half, ci(2)),
        )
        if FH:
            voff = b.add(voff, b.mul(hq, ci(8)))
            lbase = b.add(lbase, b.mul(hq, ci(2 * HC * 2)))
        if FH and spec.fh_nb and (rr + 1) * spec.block_size <= 2 * items:
            valid = None  # every lane of this round has an item
        fill.append((valid, voff, lbase))

    def load_fill(cc):
        sx = b.mul(cc, ci(chunk_b))
        out = []
        if "nox" in spec.dbg:  # attribution only: input never loaded
            for _ in fill:
                for j in range(4):
                    out.append(b.vector_splat(b.add(cc, ci(j)), 4))
            return out
        for valid, voff, _ in fill:
            for j in range(4):
                out.append(b.buffer_load_vN(rx, voff, b.add(sx, ci(j * row_b)), I32,
                                            2 if FH else 4))
        return out

    def bf16x2_to_f32(d):
        lo = b.bitcast(b.shl(d, ci(16)), F32)
        hi = b.bitcast(b.land(d, ci(-65536)), F32)
        return [lo, hi]

    def store_fill(raw, bufoff, pred=None):
        for rr, (valid, _, lbase) in enumerate(fill):
            if valid is None and pred is None:
                rows = []
                for j in range(4):
                    v = raw[rr * 4 + j]
                    fr = []
                    for q in range(2 if FH else 4):
                        fr += bf16x2_to_f32(b.vec_extract(v, q))
                    rows.append(fr)
                _fill_writes(rows, b.add(lbase, bufoff))
                continue
            if valid is None:
                valid = pred
            elif pred is not None:
                valid = b.land(valid, pred)
            rows = []
            for j in range(4):
                v = raw[rr * 4 + j]
                f = []
                for q in range(2 if FH else 4):
                    f += bf16x2_to_f32(b.vec_extract(v, q))
                rows.append(f)
            if spec.bf_fill:
                # invalid lanes write the dummy area past the ring (inside the epilogue's
                # 64 KiB allocation)
                base = b.select(valid, b.add(lbase, bufoff), ci(2 * BUF))
                _fill_writes(rows, base)
                continue
            with b.scf_if(valid):
                base = b.add(lbase, bufoff)
                _fill_writes(rows, base)

    def _pk_quad(v4):
        pk = [
            b.inline_asm("v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v",
                         [v4[2 * h2], v4[2 * h2 + 1]], I32, sideeffect=False)
            for h2 in range(2)
        ]
        return b.vec_bitcast(b.vec_pack(pk, I32), VectorType(F16, 4))

    def _fill_writes(rows, base):
        if spec.yf16:
            nq = len(rows[0]) // 4
            rq = [[_pk_quad(rows[j][4 * q:4 * q + 4]) for q in range(nq)] for j in range(4)]
            for bb in range(4):
                for q in range(nq):
                    r0, r1, r2, r3 = (rq[j][q] for j in range(4))
                    y = (b.vector_sub(r0, r2), b.vector_add(r1, r2), b.vector_sub(r2, r1),
                         b.vector_sub(r1, r3))[bb]
                    off = b.add(base, ci((bb * 2 + q) * 2 * HC * 2))
                    b.smem_store_vN_f32(lds, [off], b.vec_bitcast(y, VectorType(F32, 2)), 2)
            return
        if True:
            if True:
                for bb in range(4):
                    vals = [_bt_axis(b, [rows[j][ch] for j in range(4)], BT[bb])
                            for ch in range(4 if FH else 8)]
                    for q in range(1 if FH else 2):
                        if spec.pkrtz:
                            pk = [
                                b.inline_asm(
                                    "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v",
                                    [vals[4 * q + 2 * h2], vals[4 * q + 2 * h2 + 1]], I32,
                                    sideeffect=False,
                                )
                                for h2 in range(2)
                            ]
                            w2 = b.vec_bitcast(b.vec_pack(pk, I32), VectorType(F32, 2))
                        else:
                            h = b.vec_cast_f32_to(b.vec_pack(vals[4 * q:4 * q + 4], F32), F16)
                            w2 = b.vec_bitcast(h, VectorType(F32, 2))
                        off = b.add(base, ci((bb * 2 + q) * 2 * HC * 2))
                        b.smem_store_vN_f32(lds, [off], w2, 2)

    if spec.bc:
        vu = b.mul(b.add(b.mul(n_glob, ci(32)), b.mul(kq, ci(16))), ci(2))
        s_ub = b.mul(wb, ci((C // 8) * K * 32 * 2))

    def load_b(cc):
        if spec.bc:
            out = {}
            for dz in range(3):
                su = b.add(s_ub, b.mul(b.add(ci(dz * 4 * (C // 8)), cc), ci(K * 32 * 2)))
                for cp in range(2):
                    d = b.buffer_load_vN(ru, vu, b.add(su, ci(cp * 16)), I32, 4)
                    for h in range(2):
                        out[(dz, 2 * cp + h)] = b.vec_pack(
                            [b.vec_extract(d, 2 * h), b.vec_extract(d, 2 * h + 1)], I32)
            return out
        if "nob" in spec.dbg:  # attribution only: weight fragments never loaded
            return {(dz, c): b.vector_splat(b.add(cc, ci(dz * 4 + c)), 2)
                    for dz in range(3) for c in range(4)}
        su = b.add(s_ub, b.mul(cc, ci(K * 8 * 2)))
        return {
            (dz, c): b.buffer_load_vN(
                ru, vu, b.add(su, ci(((dz * 4) * 4 + c) * u_pt * 2)), I32, 2
            )
            for dz in range(3)
            for c in range(4)
        }

    l_tr = b.div(m, ci(TCx))
    l_hx = b.mod(m, ci(TCx))
    l_hx1 = b.add(l_hx, ci(1))
    if LIN:
        # the last tile of a row (not the block's last lane) reads its right-hand column
        # pair from the seam slot of that row end
        dm = b.sub(m, b.sub(ci(TX - 1), tx0))
        seam_l = b.land(b.land(b.cmp_ge(dm, ci(0)), b.cmp_eq(b.mod(b.smax(dm, ci(0)), ci(TX)),
                                                              ci(0))),
                        b.cmp_lt(m, ci(TCx - 1)))
        l_hx1 = b.select(seam_l, b.add(ci(TCx + 1), b.div(b.smax(dm, ci(0)), ci(TX))), l_hx1)

    def read_plane(zl, bufoff):
        zi = b.add(b.mul(zh, ci(ZTW)), ci(zl))
        rowbase = b.mul(
            b.add(b.mul(b.add(b.mul(b.add(b.mul(zi, ci(TR)), l_tr), ci(4)), wb), ci(2)), kq),
            ci(2),
        )
        vs = []
        for x in range(4):
            off = b.add(
                b.mul(b.add(b.mul(b.add(rowbase, ci(x % 2)), ci(HC)),
                            l_hx if x < 2 else l_hx1),
                      ci(2)),
                bufoff,
            )
            w2 = b.smem_load_vN_f32(lds, off, n=2)
            vs.append(b.vec_bitcast(w2, VectorType(F16, 4)))
        return vs

    def x_transform(vs):
        # B^T rows: c0 = x0 - x2, c1 = x1 + x2, c2 = x2 - x1, c3 = x1 - x3
        return [
            b.vector_sub(vs[0], vs[2]),
            b.vector_add(vs[1], vs[2]),
            b.vector_sub(vs[2], vs[1]),
            b.vector_sub(vs[1], vs[3]),
        ]

    n_chunks = C // spec.c_chunk
    if KS > 1:
        n_chunks //= KS
        cbase = b.mul(ks_i, ci(n_chunks))
        _lf, _lb = load_fill, load_b

        def load_fill(cc):
            return _lf(b.add(cbase, cc))

        def load_b(cc):
            return _lb(b.add(cbase, cc))
    bkeys = [(dz, c) for dz in range(3) for c in range(4)]
    if spec.single_buf and spec.bshare:
        # weight fragments of one chunk in LDS after the plane buffer:
        # [point (dz, b, c)][n (32 * NN)][8 ch] f16; one 16-byte item per (point, n)
        NNt = 32 * NN
        BOFF = BUF
        bitems = 48 * NNt
        brounds = -(-bitems // spec.block_size)
        n0 = b.mul(bid_n, ci(spec.n_tile))
        u_pt8 = (C // 8) * K * 8
        bfill = []
        for rr in range(brounds):
            it = b.add(tid, ci(rr * spec.block_size))
            ok = b.cmp_lt(it, ci(bitems))
            nl = b.mod(it, ci(NNt))
            pt = b.smin(b.div(it, ci(NNt)), ci(47))
            gof = b.mul(b.add(b.mul(pt, ci(u_pt8)), b.mul(b.add(n0, nl), ci(8))), ci(2))
            lof = b.add(ci(BOFF), b.mul(it, ci(4)))
            bfill.append((ok, gof, lof))

        def load_bfill(cc):
            sb = b.mul(cc, ci(K * 8 * 2))
            return [b.buffer_load_vN(ru, gof, sb, I32, 4) for _, gof, _ in bfill]

        def store_bfill(vals):
            for (ok, _, lof), v in zip(bfill, vals):
                with b.scf_if(ok):
                    b.smem_store_vN_f32(lds, [lof], b.vec_bitcast(v, VectorType(F32, 4)), 4)

        # this lane's fragment (dz, b=wb, c): point (dz*4 + wb)*4 + c, n = nh*32 + m,
        # channels 4kq..4kq+3 -> f32 word offset
        nloc = b.add(b.mul(nh, ci(32)), m)

        def read_b():
            out = {}
            for dz, c in bkeys:
                pt = b.add(b.mul(b.add(ci(dz * 4), wb), ci(4)), ci(c))
                off = b.add(ci(BOFF), b.add(b.mul(b.add(b.mul(pt, ci(NNt)), nloc), ci(4)),
                                            b.mul(kq, ci(2))))
                out[(dz, c)] = b.vec_bitcast(b.smem_load_vN_f32(lds, off, n=2),
                                             VectorType(I32, 2))
            return out

        raw0 = load_fill(ci(0))
        braw0 = load_bfill(ci(0))
        nfill = len(raw0)
        iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
        iters += [(f"raw{i}", v) for i, v in enumerate(raw0)]
        iters += [(f"braw{i}", v) for i, v in enumerate(braw0)]
        loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
        with loop as (cc, its):
            accs = list(its[: 4 * ZTW])
            raw = its[4 * ZTW: 4 * ZTW + nfill]
            braw = its[4 * ZTW + nfill:]
            cn = b.smin(b.add(cc, ci(1)), ci(n_chunks - 1))
            nraw = load_fill(cn)
            nbraw = load_bfill(cn)
            store_fill(raw, ci(0))
            store_bfill(braw)
            b.sync_lds_only()
            bf = read_b()
            for zi in range(ZTW + 2):
                frags = x_transform(read_plane(zi, ci(0)))
                for dz in range(3):
                    zo = zi - dz
                    if 0 <= zo < ZTW:
                        for c in range(4):
                            k = zo * 4 + c
                            accs[k] = atom.emit(
                                b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                                accs[k],
                            )
            b.sync_lds_only()
            b.scf_yield(*accs, *nraw, *nbraw)
        acc = loop.results[: 4 * ZTW]
        _yx_epilogue(
            b, spec, acc, wave, lane, kq, tb, b.add(z0, b.mul(zh, ci(ZTW))), n_glob, Y,
            lds_flat=lds, zt_w=ZTW, wb=wb, zh=grp, gpr=2,
        )
        b.ret()
        return b.kernel
    if spec.single_buf:
        raw0 = load_fill(ci(0))
        b0 = load_b(ci(0))
        nfill = len(raw0)
        iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
        iters += [(f"raw{i}", v) for i, v in enumerate(raw0)]
        iters += [(f"bfr{i}", b0[k]) for i, k in enumerate(bkeys)]
        loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
        with loop as (cc, its):
            accs = list(its[: 4 * ZTW])
            raw = its[4 * ZTW: 4 * ZTW + nfill]
            bf = dict(zip(bkeys, its[4 * ZTW + nfill:]))
            cn = b.smin(b.add(cc, ci(1)), ci(n_chunks - 1))
            nraw = load_fill(cn)
            nb = load_b(cn)
            store_fill(raw, ci(0))
            b.sync_lds_only()
            for zi in range(ZTW + 2):
                frags = x_transform(read_plane(zi, ci(0)))
                for dz in range(3):
                    zo = zi - dz
                    if 0 <= zo < ZTW:
                        for c in range(4):
                            k = zo * 4 + c
                            accs[k] = atom.emit(
                                b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                                accs[k],
                            )
            b.sync_lds_only()
            b.scf_yield(*accs, *nraw, *[nb[k] for k in bkeys])
        acc = loop.results[: 4 * ZTW]
        _yx_epilogue(
            b, spec, acc, wave, lane, kq, tb, b.add(z0, b.mul(zh, ci(ZTW))), n_glob, Y,
            lds_flat=lds, zt_w=ZTW, wb=wb, zh=grp, gpr=2,
        )
        b.ret()
        return b.kernel
    if spec.b16:
        _lds16_loop_b16(
            b, spec, atom, op_t, ZTW, BUF, n_chunks, bkeys, load_fill, store_fill,
            read_plane, x_transform, ru, vu, s_ub, wave, lane, kq, tb, z0, zh, wb, grp,
            n_glob, Y, lds,
        )
        b.ret()
        return b.kernel
    if spec.frel:
        fr_A, fr_base, fr_vpw, fr_nwg = _frel_plan(spec)
        PVv = Hi * Wi
        Vv = Di * PVv
        NCB = C // 8
        WPS = n_tb * (spec.K // spec.n_tile)
        nzb = spec.Do // ZT
        # consumer: wait for the producers of planes [z0, z0 + ZT + 2)
        maxcnt = 0
        for zb_ in range(nzb):
            hi_ = min(Vv, (zb_ * ZT + ZT + 2) * PVv)
            if hi_ <= fr_base:
                continue
            lo_ = max(fr_base, zb_ * ZT * PVv)
            maxcnt = max(maxcnt, (hi_ - 1 - fr_base) // fr_vpw - (lo_ - fr_base) // fr_vpw + 1)
        lo_v = b.smax(b.mul(z0, ci(PVv)), ci(fr_base))
        hi_v = b.smin(b.mul(b.add(z0, ci(ZT + 2)), ci(PVv)), ci(Vv))
        need = b.cmp_gt(hi_v, ci(fr_base))
        p_first = b.div(b.sub(lo_v, ci(fr_base)), ci(fr_vpw))
        p_last = b.div(b.sub(b.sub(hi_v, ci(1)), ci(fr_base)), ci(fr_vpw))
        with b.scf_if(need):
            for r_ in range(-(-maxcnt // 64)):
                pp_ = b.smin(b.add(p_first, b.add(lane, ci(r_ * 64))), p_last)
                b.global_flag_wait_eq(Fl, pp_, ci(1), ordering="monotonic")
            if "noacq" not in spec.dbg:
                b.fence(scope="agent", ordering="acquire")
        # producer: voxels [base + pid * VPW, ...) -- one 16-byte item per thread and chunk
        rxn = b.buffer_rsrc(Xn, ci(Vv * C * 2))
        rx8 = b.buffer_rsrc(X, ci(Vv * C * 2))
        pidr = b.block_id_x()
        fr_v0 = b.add(ci(fr_base), b.mul(pidr, ci(fr_vpw)))
        fr_tok = b.cmp_lt(tid, ci(fr_vpw))

        def fr_offsets(cc):
            e = b.add(b.mul(cc, ci(fr_vpw)), tid)
            vl = b.div(e, ci(NCB))
            g = b.mod(e, ci(NCB))
            vox = b.add(fr_v0, vl)
            ok = b.land(fr_tok, b.cmp_lt(vox, ci(Vv)))
            oob = ci(Vv * C * 2)
            src = b.select(ok, b.add(b.mul(vox, ci(C * 2)), b.mul(g, ci(16))), oob)
            dst = b.select(ok, b.add(b.mul(g, ci(Vv * 16)), b.mul(vox, ci(16))), oob)
            return src, dst

        def fr_load(cc):
            src, dst = fr_offsets(cc)
            return b.buffer_load_vN(rxn, src, ci(0), I32, 4), dst

        def fr_store(val, dst):
            b._op(
                "tile.buffer_store_vN_bf16",
                [rx8, dst, ci(0), b.vec_bitcast(val, VectorType(BF16, 8))],
                attrs={"dwords": 4},
            )

    # prologue: chunk 0 staged in buffer 0, chunk 1 in registers
    store_fill(load_fill(ci(0)), ci(0))
    raw1 = load_fill(ci(min(1, n_chunks - 1)))
    b0 = load_b(ci(0))
    if spec.bpf2:
        b1 = load_b(ci(min(1, n_chunks - 1)))
    b.sync_lds_only()
    nfill = len(raw1)
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
    iters += [(f"raw{i}", v) for i, v in enumerate(raw1)]
    iters += [(f"bfr{i}", b0[k]) for i, k in enumerate(bkeys)]
    if spec.bpf2:
        iters += [(f"bfs{i}", b1[k]) for i, k in enumerate(bkeys)]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
    with loop as (cc, its):
        accs = list(its[: 4 * ZTW])
        raw = its[4 * ZTW: 4 * ZTW + nfill]
        bf = dict(zip(bkeys, its[4 * ZTW + nfill: 4 * ZTW + nfill + len(bkeys)]))
        if spec.bpf2:
            bf_next = [its[4 * ZTW + nfill + len(bkeys) + i] for i in range(len(bkeys))]
        if spec.iglp >= 0:
            b.iglp_opt(spec.iglp)
        par_c = b.mod(cc, ci(2))
        cur = b.mul(par_c, ci(BUF))
        nxt_off = b.mul(b.sub(ci(1), par_c), ci(BUF))
        # prefetch indices clamped to the last chunk: the buffer range check ignores
        # soffset, so a chunk past the end would read outside the tensor
        if spec.frel:
            fr_val, fr_dst = fr_load(cc)
        nraw = load_fill(b.smin(b.add(cc, ci(2)), ci(n_chunks - 1)))
        nb = load_b(b.smin(b.add(cc, ci(2 if spec.bpf2 else 1)), ci(n_chunks - 1)))
        if spec.phase_sel == "odd":
            hi_wave = b.cmp_eq(b.mod(wave, ci(2)), ci(1))
            lo_wave = b.cmp_eq(b.mod(wave, ci(2)), ci(0))
        else:
            hi_wave = b.cmp_ge(wave, ci(4))
            lo_wave = b.cmp_lt(wave, ci(4))
        if spec.fill_first:
            store_fill(raw, nxt_off)
        elif spec.phase:
            store_fill(raw, nxt_off, hi_wave)
        if spec.prio:
            b.s_setprio(spec.prio)
        for zi in range(ZTW + 2):
            frags = x_transform(read_plane(zi, cur))
            for dz in range(3):
                zo = zi - dz
                if 0 <= zo < ZTW:
                    for c in range(4):
                        k = zo * 4 + c
                        accs[k] = atom.emit(
                            b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                            accs[k],
                        )
            if spec.sbar:
                b.sched_barrier(0)
        if spec.prio:
            b.s_setprio(0)
        if spec.phase:
            store_fill(raw, nxt_off, lo_wave)
        elif not spec.fill_first:
            store_fill(raw, nxt_off)
        if spec.frel and "nocopy" not in spec.dbg:
            fr_store(fr_val, fr_dst)
        if spec.sgb:
            nm = 4 * 3 * ZTW
            for _ in range(nm):
                b.sched_group_barrier(0x008, 1, 0)  # 1 MFMA
                b.sched_group_barrier(0x002, spec.sgb, 0)  # VALU
        b.sync_lds_only()
        if spec.bpf2:
            b.scf_yield(*accs, *nraw, *bf_next, *[nb[k] for k in bkeys])
        else:
            b.scf_yield(*accs, *nraw, *[nb[k] for k in bkeys])
    acc = loop.results[: 4 * ZTW]
    if spec.frel:
        # publish this workgroup's relayout share
        if "norel" not in spec.dbg:
            b.fence(scope="agent", ordering="release")
        b.sync()
        with b.scf_if(b.cmp_eq(wave, ci(0))):
            b.global_flag_store(Fl, pidr, ci(1))
    _yx_epilogue(
        b, spec, acc, wave, lane, kq, tb, b.add(z0, b.mul(zh, ci(ZTW))), n_glob, Y,
        lds_flat=lds, zt_w=ZTW, wb=wb, zh=grp, gpr=2, ks_i=ks_i if KS > 1 else None,
        oscale=OS,
    )
    b.ret()
    return b.kernel


def _yx_wlds_wave_words(spec) -> int:
    """f32 words of one wave's private staging buffer (wlds stage)."""
    _, TCx, TR = _yx_lds_geom(spec)
    nn = 2 if spec.nsplit else spec.nn
    nz = (spec.waves // 4) // nn
    nplw = spec.z_tile // nz + 2
    return nplw * TR * 2 * (TCx + 1) * 2 * 2


def _yx_wlds_floats(spec) -> int:
    return spec.waves * _yx_wlds_wave_words(spec)


def build_winograd_yx_wlds_fwd(spec: WinogradYxSpec) -> KernelDef:
    """F(2x2,3x3) on (y,x) + direct z with WAVE-PRIVATE staging (``stage="wlds"``).

    Each wave stages only the two input rows its y-point ``b`` needs (coalesced 8-byte
    lane loads, prefetched one chunk ahead), combines them along y in f32 and stores f16
    into its own LDS buffer, then reads its A operands back and applies B^T along x in
    packed f16. No workgroup barrier in the main loop: waves run decoupled. The y-sign
    of b=2 is folded into the weights (``fold=True`` transform).
    """
    from rocke.core.ir import F16

    ok, why = is_valid_yx_spec(spec)
    if not ok or spec.stage != "wlds":
        raise ValueError(f"invalid winograd yx wlds spec: {why}")
    atom = MfmaAtom.f16_32x32x8()
    op_t = F16
    C, K = spec.C, spec.K
    Di, Hi, Wi = spec.Di, spec.Hi, spec.Wi
    ZT = spec.z_tile
    TX, TCx, TR = _yx_lds_geom(spec)
    NC = 2 * TCx + 2
    HC = TCx + 1
    n_tb = spec.tiles2d // spec.tiles_per_wg
    NN = 2 if spec.nsplit else spec.nn
    NZ = (spec.waves // 4) // NN
    ZTW = ZT // NZ
    NPLW = ZTW + 2
    WBUF = _yx_wlds_wave_words(spec)
    items_w = NPLW * TR * NC * 2  # (plane, tile row, column, channel quad)
    rounds = -(-items_w // 64)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.wpe:
        b.kernel.attrs["waves_per_eu"] = (spec.wpe, spec.wpe)
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(op_t, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    wb = b.mod(wave, ci(4))
    grp = b.div(wave, ci(4))
    nh = b.div(grp, ci(NZ))
    zh = b.mod(grp, ci(NZ))
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))
    kq = b.div(lane, ci(32))
    bid = b.block_id_x()
    bid_n = b.block_id_y()
    tb = b.mod(bid, ci(n_tb))
    zb = b.div(bid, ci(n_tb))
    z0 = b.mul(zb, ci(ZT))
    zw0 = b.add(z0, b.mul(zh, ci(ZTW)))  # this wave's first output plane
    t0 = b.mul(tb, ci(spec.tiles_per_wg))
    ty0 = b.div(t0, ci(TX))
    tx0 = b.mod(t0, ci(TX))

    lds = b.smem_alloc(
        F32, [max(spec.waves * WBUF, min(spec.waves, 8) * 32 * 64)], name_hint="winoyx_w"
    )
    wbase = b.mul(wave, ci(WBUF))

    # wave-uniform rows / sign of this y-point (true V_b = ZS0[b] * (d[p0] + r d[p1]))
    is0 = b.cmp_eq(wb, ci(0))
    is3 = b.cmp_eq(wb, ci(3))
    is1 = b.cmp_eq(wb, ci(1))
    p0 = b.select(is0, ci(0), ci(1))
    p1 = b.select(is3, ci(3), ci(2))
    r = b.select(is1, b.const_f32(1.0), b.const_f32(-1.0))
    rv = b.vector_splat(r, 2)

    plane_b = Hi * Wi * 16
    chunk_b = Di * plane_b
    x_bytes = Di * Hi * Wi * C * 2
    rx = b.buffer_rsrc(X, ci(x_bytes))
    ru = b.buffer_rsrc(U, ci(3 * 16 * C * K * 2))
    n_glob = b.add(b.mul(bid_n, ci(spec.n_tile)), m)
    if NN > 1:
        n_glob = b.add(n_glob, b.mul(nh, ci(32)))
    vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
    u_pt = (C // 8) * K * 8
    s_ub = b.mul(wb, ci(4 * u_pt * 2))
    row0 = b.mul(p0, ci(Wi * 16))
    row1 = b.mul(p1, ci(Wi * 16))

    fill = []
    for rr in range(rounds):
        it = b.add(lane, ci(rr * 64))
        valid = b.cmp_lt(it, ci(items_w))
        q = b.mod(it, ci(2))
        col = b.mod(b.div(it, ci(2)), ci(NC))
        t2 = b.div(it, ci(2 * NC))
        tr = b.mod(t2, ci(TR))
        zl = b.div(t2, ci(TR))
        zl = b.smin(zl, ci(NPLW - 1))  # padding lanes stay inside the tensor
        goff = b.add(
            b.add(
                b.mul(b.add(zw0, zl), ci(plane_b)),
                b.mul(b.add(b.mul(ty0, ci(2)), b.mul(tr, ci(2))), ci(Wi * 16)),
            ),
            b.add(b.mul(b.add(b.mul(tx0, ci(2)), col), ci(16)), b.mul(q, ci(8))),
        )
        loff = b.add(
            b.mul(
                b.add(b.mul(b.add(b.mul(b.add(b.mul(zl, ci(TR)), tr), ci(2)), b.mod(col, ci(2))),
                            ci(HC)), b.div(col, ci(2))),
                ci(4),
            ),
            b.mul(q, ci(2)),
        )
        fill.append((valid, goff, b.add(loff, wbase)))

    def load_fill(cc):
        cb = b.mul(cc, ci(chunk_b))
        out = []
        for valid, goff, _ in fill:
            vo = b.add(goff, cb)
            out.append(b.buffer_load_vN(rx, b.add(vo, row0), ci(0), I32, 2))
            out.append(b.buffer_load_vN(rx, b.add(vo, row1), ci(0), I32, 2))
        return out

    def bf16x2_to_f32(d):
        lo = b.bitcast(b.shl(d, ci(16)), F32)
        hi = b.bitcast(b.land(d, ci(-65536)), F32)
        return b.vec_pack([lo, hi], F32)

    def store_fill(raw):
        for rr, (valid, _, loff) in enumerate(fill):
            d0, d1 = raw[2 * rr], raw[2 * rr + 1]
            halves = []
            for e in range(2):
                a0 = bf16x2_to_f32(b.vec_extract(d0, e))
                a1 = bf16x2_to_f32(b.vec_extract(d1, e))
                halves.append(b.vector_fma(rv, a1, a0))  # d[p0] + r d[p1]
            v4 = b.vec_concat(halves[0], halves[1])
            w2 = b.vec_bitcast(b.vec_cast_f32_to(v4, F16), VectorType(F32, 2))
            with b.scf_if(valid):
                b.smem_store_vN_f32(lds, [loff], w2, 2)

    def load_b(cc):
        su = b.add(s_ub, b.mul(cc, ci(K * 8 * 2)))
        return {
            (dz, c): b.buffer_load_vN(
                ru, vu, b.add(su, ci(((dz * 4) * 4 + c) * u_pt * 2)), I32, 2
            )
            for dz in range(3)
            for c in range(4)
        }

    l_tr = b.div(m, ci(TCx))
    l_hx = b.mod(m, ci(TCx))

    def read_plane(zl):
        vs = []
        for x in range(4):
            off = b.add(
                b.mul(
                    b.add(
                        b.mul(b.add(b.mul(b.add(ci(zl * TR), l_tr), ci(2)), ci(x % 2)), ci(HC)),
                        b.add(l_hx, ci(x // 2)),
                    ),
                    ci(4),
                ),
                b.add(b.mul(kq, ci(2)), wbase),
            )
            w2 = b.smem_load_vN_f32(lds, off, n=2)
            vs.append(b.vec_bitcast(w2, VectorType(F16, 4)))
        return vs

    def x_transform(vs):
        return [
            b.vector_sub(vs[0], vs[2]),
            b.vector_add(vs[1], vs[2]),
            b.vector_sub(vs[2], vs[1]),
            b.vector_sub(vs[1], vs[3]),
        ]

    n_chunks = C // spec.c_chunk
    raw0 = load_fill(ci(0))
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(4 * ZTW)]
    iters += [(f"raw{i}", v) for i, v in enumerate(raw0)]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
    with loop as (cc, its):
        accs = list(its[: 4 * ZTW])
        raw = its[4 * ZTW:]
        bf = load_b(cc)
        nraw = load_fill(b.smin(b.add(cc, ci(1)), ci(n_chunks - 1)))
        store_fill(raw)
        for zl in range(NPLW):
            frags = x_transform(read_plane(zl))
            for dz in range(3):
                zo = zl - dz
                if 0 <= zo < ZTW:
                    for c in range(4):
                        k = zo * 4 + c
                        accs[k] = atom.emit(
                            b, frags[c], b.vec_bitcast(bf[(dz, c)], VectorType(op_t, 4)),
                            accs[k],
                        )
        b.scf_yield(*accs, *nraw)
    acc = loop.results[: 4 * ZTW]
    b.sync()
    _yx_epilogue(
        b, spec, acc, wave, lane, kq, tb, zw0, n_glob, Y,
        lds_flat=lds, zt_w=ZTW, wb=wb, zh=grp, gpr=2,
    )
    b.ret()
    return b.kernel


def winograd_yx_weight_transform(w, *, dtype=None, fold=True, u16=False, uc=False):
    """``w``: [K, 3, 3, 3, C] (KZYXC) -> U [3 dz][4 b][4 c][C/8][K][8] (y-sign folded)."""
    import torch

    key = "yx" + str(w.device)
    if key not in _WT_CONST:
        _WT_CONST[key] = (
            torch.tensor(G, dtype=torch.float32, device=w.device),
            torch.tensor(ZS0, dtype=torch.float32, device=w.device).view(1, 1, 4, 1, 1),
        )
    g, s = _WT_CONST[key]
    K, C = w.shape[0], w.shape[-1]
    wf = w.float()  # [K, z, y, x, C]
    u = torch.einsum("bj,nzjkc->nzbkc", g, wf)
    u = torch.einsum("dk,nzbkc->nzbdc", g, u)  # [K, dz, b, c, C]
    if fold:
        u = u * s
    if uc:  # [3 dz][4 b][C/8][K][2 kq][4 c][4]
        u = u.permute(1, 2, 3, 0, 4).reshape(3, 4, 4, K, C // 8, 2, 4)
        u = u.permute(0, 1, 4, 3, 5, 2, 6)
        return u.to(dtype or torch.bfloat16).contiguous()
    if u16:  # [3][4][4][C/16][K][2 kq][2 chunk][4]
        u = u.permute(1, 2, 3, 0, 4).reshape(3, 4, 4, K, C // 16, 2, 2, 4)
        u = u.permute(0, 1, 2, 4, 3, 6, 5, 7)
        return u.to(dtype or torch.bfloat16).contiguous()
    u = u.permute(1, 2, 3, 0, 4).reshape(3, 4, 4, K, C // 8, 8).permute(0, 1, 2, 4, 3, 5)
    return u.to(dtype or torch.bfloat16).contiguous()


# ---------------------------------------------------------------------------------
# Auxiliary kernels of the yx variant: X relayout to c8 and the weight transform.
# ---------------------------------------------------------------------------------


def build_x_to_c8(D: int, H: int, W: int, C: int, rows: bool = False) -> KernelDef:
    """X [D][H][W][C] bf16 -> X8 [C/8][D][H][W][8]. Grid (ceil(V/64), 1, 1), block 256:
    thread (v = lane, q = wave) copies 16 B chunks cb = q, q+4, ... of voxel v."""
    V = D * H * W
    b = IRBuilder(f"wino_x_to_c8{'r' if rows else ''}_V{V}C{C}")
    b.kernel.attrs["max_workgroup_size"] = 256
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    X8 = b.param("X8", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    ci = b.const_i32
    tid = b.thread_id_x()
    lane = b.mod(tid, ci(64))
    q = b.div(tid, ci(64))
    v = b.add(b.mul(b.block_id_x(), ci(64)), lane)
    rx = b.buffer_rsrc(X, ci(V * C * 2))
    ry = b.buffer_rsrc(X8, ci(V * C * 2))
    # Out-of-range voxels: loads return 0 and stores are dropped -- the whole offset is in
    # voffset (the buffer range check ignores soffset), forced past the end for v >= V.
    vin = b.mul(v, ci(C * 2))
    vok = b.cmp_lt(v, ci(V))
    for cb in range(0, C // 8, 4):
        cbq = b.add(q, ci(cb))
        val = b.buffer_load_vN(rx, b.add(vin, b.mul(cbq, ci(16))), ci(0), I32, 4)
        ok = b.cmp_lt(cbq, ci(C // 8))
        if rows:  # [D][H][C/8][W][8]: (row, cb) blocks of W * 16 bytes
            row = b.div(v, ci(W))
            xx = b.mod(v, ci(W))
            o = b.add(b.mul(b.add(b.mul(row, ci(C // 8)), cbq), ci(W * 16)), b.mul(xx, ci(16)))
        else:
            o = b.add(b.mul(v, ci(16)), b.mul(cbq, ci(V * 16)))
        vout = b.select(vok, o, ci(V * C * 2))
        with b.scf_if(ok):
            b._op(
                "tile.buffer_store_vN_bf16",
                [ry, vout, ci(0), b.vec_bitcast(val, VectorType(BF16, 8))],
                attrs={"dwords": 4},
            )
    b.ret()
    return b.kernel


def build_x_to_c8_range(D: int, H: int, W: int, C: int) -> KernelDef:
    """X [D][H][W][C] -> X8 [C/8][D][H][W][8] for voxels [V0, V1) (kernel arguments), no LDS
    (so it can share CUs with an LDS-full kernel on another stream). Grid
    (ceil((V1 - V0) / 64)), block 256: thread (lane, wave q) copies chunks q, q+4, ..."""
    V = D * H * W
    b = IRBuilder(f"wino_x_to_c8rng_V{V}C{C}")
    b.kernel.attrs["max_workgroup_size"] = 256
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    X8 = b.param("X8", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    V0 = b.param("V0", I32)
    V1 = b.param("V1", I32)
    ci = b.const_i32
    tid = b.thread_id_x()
    lane = b.mod(tid, ci(64))
    q = b.div(tid, ci(64))
    v = b.add(V0, b.add(b.mul(b.block_id_x(), ci(64)), lane))
    rx = b.buffer_rsrc(X, ci(V * C * 2))
    ry = b.buffer_rsrc(X8, ci(V * C * 2))
    vok = b.cmp_lt(v, V1)
    vin = b.select(vok, b.mul(v, ci(C * 2)), ci(V * C * 2))
    for cb in range(0, C // 8, 4):
        cbq = b.add(q, ci(cb))
        val = b.buffer_load_vN(rx, b.add(vin, b.mul(cbq, ci(16))), ci(0), I32, 4)
        ok = b.cmp_lt(cbq, ci(C // 8))
        vout = b.select(vok, b.add(b.mul(v, ci(16)), b.mul(cbq, ci(V * 16))), ci(V * C * 2))
        with b.scf_if(ok):
            b._op(
                "tile.buffer_store_vN_bf16",
                [ry, vout, ci(0), b.vec_bitcast(val, VectorType(BF16, 8))],
                attrs={"dwords": 4},
            )
    b.ret()
    return b.kernel


def x_to_c8_grid(D, H, W, C):
    return (-(-(D * H * W) // 64), 1, 1)


def build_x_to_c8_lds(D: int, H: int, W: int, C: int, block_bytes=None,
                      halo: int = 0, scale: bool = False) -> KernelDef:
    """LDS-transposed X [D][H][W][C] -> X8 [C/8][D][H][W][8]: a block of VB voxels is read
    fully coalesced (16 B per lane, consecutive), transposed through LDS and written as
    C/8 runs of VB * 16 contiguous bytes. Block 256, grid ceil(V / VB)."""
    V = D * H * W
    b = IRBuilder(f"wino_x_to_c8l_V{V}C{C}" + (f"_bb{block_bytes}" if block_bytes else "")
                  + (f"_h{halo}" if halo else "") + ("_s" if scale else ""))
    b.kernel.attrs["max_workgroup_size"] = 256
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    X8 = b.param("X8", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    S = b.param("S", F32) if scale else None
    lds = b.smem_alloc(F32, [_x_to_c8_lds_words(C, block_bytes)], name_hint="x8_t")
    _emit_x_to_c8_lds(b, X, X8, lds, D, H, W, C, b.block_id_x(), block_bytes, halo=halo,
                      scale=S)
    b.ret()
    return b.kernel


X8_BLOCK_BYTES = 16384  # input bytes per relayout block (LDS-transposed relayout)


def _x8_vb(C, block_bytes=None):
    return max(16, (block_bytes or X8_BLOCK_BYTES) // (C * 2))


def _x_to_c8_lds_words(C, block_bytes=None):
    return _x8_vb(C, block_bytes) * (C // 8) * 4


def _emit_x_to_c8_lds(b, X, X8, lds, D, H, W, C, blk, block_bytes=None, vlim=None,
                      nthr=256, halo=0, scale=None):
    """``halo``: X8 is [C/8][D+2h][H+2h][W+2h][8] and only its interior is written (the
    caller zeroes the halo once) -- dgrad's dY padded by 2 for the flipped-weight conv."""
    V = D * H * W
    hp = halo
    Vp = (D + 2 * hp) * (H + 2 * hp) * (W + 2 * hp)
    VL = V if vlim is None else vlim  # voxels [0, VL) are relayouted (strides use V)
    NCB = C // 8
    VB = _x8_vb(C, block_bytes)  # voxels per block
    n16 = VB * NCB  # 16-byte elements per block
    ci = b.const_i32
    tid = b.thread_id_x()
    v0 = b.mul(blk, ci(VB))
    rx = b.buffer_rsrc(X, ci(V * C * 2))
    ry = b.buffer_rsrc(X8, ci(Vp * C * 2))
    nb = Vp * C * 2
    for r in range(-(-n16 // nthr)):
        e = b.add(tid, ci(r * nthr))  # element in input order: (v_local, cb)
        ok = b.cmp_lt(e, ci(n16))
        vl = b.div(e, ci(NCB))
        cb = b.mod(e, ci(NCB))
        goff = b.add(b.mul(b.add(v0, vl), ci(C * 2)), b.mul(cb, ci(16)))
        val = b.buffer_load_vN(rx, goff, ci(0), I32, 4)
        with b.scf_if(ok):
            b.smem_store_vN_f32(lds, [b.mul(b.add(b.mul(cb, ci(VB)), vl), ci(4))],
                                b.vec_bitcast(val, VectorType(F32, 4)), 4)
    b.sync()
    for r in range(-(-n16 // nthr)):
        e = b.add(tid, ci(r * nthr))  # element in output order: (cb, v_local)
        ok = b.cmp_lt(e, ci(n16))
        cb = b.div(e, ci(VB))
        vl = b.mod(e, ci(VB))
        v = b.add(v0, vl)
        with b.scf_if(ok):
            val = b.smem_load_vN_f32(lds, b.mul(e, ci(4)), n=4)
            if scale is not None:  # x * S (S a power of two: exact in bf16)
                vi = b.vec_bitcast(val, VectorType(I32, 4))
                ws_ = []
                for q in range(4):
                    d = b.vec_extract(vi, q)
                    lo = b.bitcast(b.fmul(b.bitcast(b.shl(d, ci(16)), F32), scale), I32)
                    hi = b.bitcast(b.fmul(b.bitcast(b.land(d, ci(-65536)), F32), scale), I32)
                    ws_.append(b.lor(b.land(hi, ci(-65536)), b.lshr(lo, ci(16))))
                val = b.vec_bitcast(b.vec_pack(ws_, I32), VectorType(F32, 4))
            if hp:
                xx_ = b.mod(v, ci(W))
                yy_ = b.mod(b.div(v, ci(W)), ci(H))
                zz_ = b.div(v, ci(H * W))
                vp = b.add(b.mul(b.add(b.mul(b.add(zz_, ci(hp)), ci(H + 2 * hp)),
                                       b.add(yy_, ci(hp))), ci(W + 2 * hp)), b.add(xx_, ci(hp)))
            else:
                vp = v
            ooff = b.select(b.cmp_lt(v, ci(VL)), b.add(b.mul(cb, ci(Vp * 16)), b.mul(vp, ci(16))),
                            ci(nb))
            b._op(
                "tile.buffer_store_vN_bf16",
                [ry, ooff, ci(0), b.vec_bitcast(val, VectorType(BF16, 8))],
                attrs={"dwords": 4},
            )




def x_to_c8_lds_grid(D, H, W, C, block_bytes=None):
    VB = _x8_vb(C, block_bytes)
    return (-(-(D * H * W) // VB), 1, 1)


def build_yx_weight_transform(K: int, C: int, *, out_dtype: str = "f16",
                              fold: bool = False, u16: bool = False,
                              uc: bool = False, flip: bool = False) -> KernelDef:
    """W [K][3][3][3][C] bf16 -> U [3 dz][4 b][4 c][C/8][K][8] (``out_dtype``).
    One thread per (k, c): 27 loads, G (x) G on (y, x), 48 stores. Grid (C*K/256)."""
    from rocke.core.ir import F16

    odt = F16 if out_dtype == "f16" else BF16
    b = IRBuilder(f"wino_yx_wt_K{K}C{C}_{out_dtype}{'_fold' if fold else ''}{'_u16' if u16 else ''}{'_uc' if uc else ''}{'_flip' if flip else ''}")
    b.kernel.attrs["max_workgroup_size"] = 256
    W = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Uo = b.param("U", PtrType(odt, "global"), noalias=True, writeonly=True, align=16)
    _emit_yx_wt(b, W, Uo, K, C, out_dtype, fold, u16, uc, b.block_id_x(), flip=flip)
    b.ret()
    return b.kernel


def _emit_yx_wt(b, W, Uo, K, C, out_dtype, fold, u16, uc, blk, flip=False):
    """``flip``: dgrad weights -- W is the forward filter [C][3][3][3][K] (forward K = this
    transform's C), transformed as Wd[k][dz][dy][dx][c] = W[c][2-dz][2-dy][2-dx][k]."""
    ci = b.const_i32
    t = b.add(b.mul(blk, ci(256)), b.thread_id_x())
    if flip:  # consecutive threads read consecutive k (contiguous in the forward filter)
        k = b.mod(t, ci(K))
        c = b.div(t, ci(K))
    else:
        c = b.mod(t, ci(C))
        k = b.div(t, ci(C))
    w = {}
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if flip:
                    tap = ((2 - z) * 3 + (2 - y)) * 3 + (2 - x)
                    idx = b.add(b.mul(b.add(b.mul(c, ci(27)), ci(tap)), ci(K)), k)
                else:
                    idx = b.add(b.mul(b.add(b.mul(k, ci(27)), ci((z * 3 + y) * 3 + x)), ci(C)), c)
                w[(z, y, x)] = b.cast_to_f32(b.global_load(W, idx, BF16, align=2))
    sgn = ZS0 if fold else (1.0, 1.0, 1.0, 1.0)
    if uc:  # [dz][b][C/8][K][2 kq][4 c][4]: point stride differs, handled below
        cb = b.div(c, ci(8))
        kqq = b.mod(b.div(c, ci(4)), ci(2))
        base = b.add(b.mul(b.add(b.mul(b.add(b.mul(cb, ci(K)), k), ci(2)), kqq), ci(16)),
                     b.mod(c, ci(4)))
    elif u16:  # [.][C/16][K][2 kq][2 chunk][4]
        c16 = b.div(c, ci(16))
        sub = b.mod(b.div(c, ci(8)), ci(2))
        kqq = b.mod(b.div(c, ci(4)), ci(2))
        base = b.add(
            b.mul(b.add(b.mul(c16, ci(K)), k), ci(16)),
            b.add(b.add(b.mul(kqq, ci(8)), b.mul(sub, ci(4))), b.mod(c, ci(4))),
        )
    else:
        cb = b.div(c, ci(8))
        cl = b.mod(c, ci(8))
        base = b.add(b.mul(b.add(b.mul(cb, ci(K)), k), ci(8)), cl)
    for dz in range(3):
        # G along x first: gx[y][cc]
        gx = {}
        for y in range(3):
            for cc in range(4):
                acc = None
                for x in range(3):
                    g = G[cc][x]
                    if g == 0.0:
                        continue
                    term = b.fmul(b.const_f32(g), w[(dz, y, x)])
                    acc = term if acc is None else b.fadd(acc, term)
                gx[(y, cc)] = acc
        for bb in range(4):
            for cc in range(4):
                acc = None
                for y in range(3):
                    g = G[bb][y] * sgn[bb]
                    if g == 0.0:
                        continue
                    term = b.fmul(b.const_f32(g), gx[(y, cc)])
                    acc = term if acc is None else b.fadd(acc, term)
                pt = (dz * 4 + bb) * 4 + cc
                if uc:
                    off = b.add(base, ci((dz * 4 + bb) * 4 * C * K + cc * 4))
                else:
                    off = b.add(base, ci(pt * C * K))
                val = acc if out_dtype != "f16" else acc
                cvt = b.trunc_f32_to_f16(val) if out_dtype == "f16" else b.trunc_f32_to_bf16(val)
                b.global_store(Uo, off, cvt, align=2)




def build_yx_weight_transform_v3(K: int, C: int, *, flip: bool = False,
                                 swz: bool = False) -> KernelDef:
    """W -> U [3][4][4][C/8][K][8] f16 (lds16 layout, no fold), one thread per (k, 8-channel
    group), consecutive threads = consecutive k, so every U store is a 16-byte lane store and
    a wave writes 1 KiB contiguous. Loads: 16 bytes per tap (forward filter, ``flip=False``)
    or, for dgrad (``flip``: W is the forward filter [C][3][3][3][K]), 8 coalesced 2-byte
    loads per tap. Same arithmetic order as _emit_yx_wt. Grid (ceil(K*C/8/256))."""
    from rocke.core.ir import F16

    b = IRBuilder(f"wino_yx_wt3_K{K}C{C}{'_flip' if flip else ''}{'_swz' if swz else ''}")
    b.kernel.attrs["max_workgroup_size"] = 256
    W = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Uo = b.param("U", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    _emit_yx_wt3(b, W, Uo, K, C, flip, b.block_id_x(), swz=swz)
    b.ret()
    return b.kernel


def yx_weight_transform_v3_grid(K, C):
    # one thread per (dz, k, 8-channel group)
    return (-(-(3 * K * C // 8) // 256), 1, 1)


def yx_wt3_swz_ok(K, C, flip=False):
    return not flip and K % 8 == 0 and (C // 8) % 8 == 0


def _emit_yx_wt3(b, W, Uo, K, C, flip, blk, swz=False):
    from rocke.core.ir import F16

    ci = b.const_i32
    NCB = C // 8
    t = b.add(b.mul(blk, ci(256)), b.thread_id_x())
    ok = b.cmp_lt(t, ci(3 * K * NCB))
    tt = b.select(ok, t, ci(0))
    if swz and yx_wt3_swz_ok(K, C, flip):
        # 8x8 lane swizzle (forward filter): each 64-lane group takes 8 k x 8 channel groups,
        # so 8 lanes read 128 contiguous bytes of W[k][tap][:] and 8 lanes write 128
        # contiguous bytes of U[pt][cb][:][8] (wt3 alone: coalesced writes, 16 B per line reads)
        ln = b.mod(tt, ci(64))
        g = b.div(tt, ci(64))
        k = b.add(b.mul(b.mod(g, ci(K // 8)), ci(8)), b.div(ln, ci(8)))
        rest = b.div(g, ci(K // 8))
        cb = b.add(b.mul(b.mod(rest, ci(NCB // 8)), ci(8)), b.mod(ln, ci(8)))
        dzv = b.div(rest, ci(NCB // 8))
    else:
        k = b.mod(tt, ci(K))
        cb = b.mod(b.div(tt, ci(K)), ci(NCB))
        dzv = b.div(tt, ci(K * NCB))  # this thread's z tap (3x the threads: latency-bound otherwise)
    rw = b.buffer_rsrc(W, ci(K * 27 * C * 2))
    ru = b.buffer_rsrc(Uo, ci(48 * C * K * 2))
    w = {}
    for z in range(1):
        for y in range(3):
            for x in range(3):
                # tap (dzv, y, x): runtime z
                tap = b.add(b.mul(dzv, ci(9)), ci(y * 3 + x))
                if flip:
                    ftap = b.sub(ci(26), tap)  # (2-z, 2-y, 2-x)
                    vals = []
                    for j in range(8):
                        idx = b.add(b.mul(b.add(b.mul(b.add(b.mul(cb, ci(8)), ci(j)), ci(27)),
                                                ftap), ci(K)), k)
                        vals.append(b.cast_to_f32(b.global_load(W, idx, BF16, align=2)))
                    w[(z, y, x)] = vals
                else:
                    off = b.mul(b.add(b.mul(b.add(b.mul(k, ci(27)), tap), ci(C)),
                                      b.mul(cb, ci(8))), ci(2))
                    d = b.buffer_load_vN(rw, off, ci(0), I32, 4)
                    vals = []
                    for q in range(4):
                        dq = b.vec_extract(d, q)
                        vals.append(b.bitcast(b.shl(dq, ci(16)), F32))
                        vals.append(b.bitcast(b.land(dq, ci(-65536)), F32))
                    w[(z, y, x)] = vals
    so = b.select(ok, b.add(b.mul(b.add(b.mul(cb, ci(K)), k), ci(16)),
                            b.mul(dzv, ci(16 * C * K * 2))), ci(48 * C * K * 2))
    for dz in range(1):
        gx = {}
        for y in range(3):
            for cc in range(4):
                for j in range(8):
                    acc = None
                    for x in range(3):
                        g = G[cc][x]
                        if g == 0.0:
                            continue
                        term = b.fmul(b.const_f32(g), w[(dz, y, x)][j])
                        acc = term if acc is None else b.fadd(acc, term)
                    gx[(y, cc, j)] = acc
        for bb in range(4):
            for cc in range(4):
                outs = []
                for j in range(8):
                    acc = None
                    for y in range(3):
                        g = G[bb][y]
                        if g == 0.0:
                            continue
                        term = b.fmul(b.const_f32(g), gx[(y, cc, j)])
                        acc = term if acc is None else b.fadd(acc, term)
                    outs.append(acc)
                pt = (dz * 4 + bb) * 4 + cc
                for hh in range(2):
                    h = b.vec_cast_f32_to(b.vec_pack(outs[4 * hh:4 * hh + 4], F32), F16)
                    b.buffer_store_vN_f16(ru, so, ci(pt * C * K * 2 + 8 * hh), h, 2)


def build_yx_weight_transform_v2(K: int, C: int, *, out_dtype: str = "f16",
                                 fold: bool = False) -> KernelDef:
    """Vectorised weight transform into the default U layout [3][4][4][C/8][K][8]:
    one thread per (4-channel quad, k) -- consecutive threads take consecutive k --
    8-byte loads of W and 8-byte stores of U, one dz plane at a time. Grid (C*K/4/256)."""
    from rocke.core.ir import F16

    odt = F16 if out_dtype == "f16" else BF16
    b = IRBuilder(f"wino_yx_wt2_K{K}C{C}_{out_dtype}{'_fold' if fold else ''}")
    b.kernel.attrs["max_workgroup_size"] = 256
    W = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Uo = b.param("U", PtrType(odt, "global"), noalias=True, writeonly=True, align=16)
    ci = b.const_i32
    t = b.add(b.mul(b.block_id_x(), ci(256)), b.thread_id_x())
    k = b.mod(t, ci(K))
    cq = b.div(t, ci(K))
    ok = b.cmp_lt(cq, ci(C // 4))
    cq = b.smin(cq, ci(C // 4 - 1))
    rw = b.buffer_rsrc(W, ci(K * 27 * C * 2))
    ru = b.buffer_rsrc(Uo, ci(48 * C * K * 2))
    sgn = ZS0 if fold else (1.0, 1.0, 1.0, 1.0)
    wbase = b.mul(b.add(b.mul(k, ci(27 * C)), b.mul(cq, ci(4))), ci(2))
    cb = b.div(cq, ci(2))
    ubase = b.mul(
        b.add(b.mul(b.add(b.mul(cb, ci(K)), k), ci(8)), b.mul(b.mod(cq, ci(2)), ci(4))), ci(2)
    )
    for dz in range(3):
        w = {}
        for y in range(3):
            for x in range(3):
                v = b.buffer_load_vN(rw, wbase, ci(((dz * 3 + y) * 3 + x) * C * 2), BF16, 4)
                w[(y, x)] = [b.cast_to_f32(b.vec_extract(v, j)) for j in range(4)]
        gx = {}
        for y in range(3):
            for cc in range(4):
                gx[(y, cc)] = [None] * 4
                for j in range(4):
                    acc = None
                    for x in range(3):
                        g = G[cc][x]
                        if g == 0.0:
                            continue
                        term = b.fmul(b.const_f32(g), w[(y, x)][j])
                        acc = term if acc is None else b.fadd(acc, term)
                    gx[(y, cc)][j] = acc
        for bb in range(4):
            for cc in range(4):
                vals = []
                for j in range(4):
                    acc = None
                    for y in range(3):
                        g = G[bb][y] * sgn[bb]
                        if g == 0.0:
                            continue
                        term = b.fmul(b.const_f32(g), gx[(y, cc)][j])
                        acc = term if acc is None else b.fadd(acc, term)
                    vals.append(acc)
                pt = (dz * 4 + bb) * 4 + cc
                vec = b.vec_cast_f32_to(b.vec_pack(vals, F32), odt)
                off = b.select(ok, ubase, ci(48 * C * K * 2))  # padding threads: dropped
                b._op(
                    "tile.buffer_store_vN_bf16" if out_dtype != "f16" else "tile.buffer_store_vN_f16",
                    [ru, off, ci(pt * C * K * 2), vec],
                    attrs={"dwords": 2},
                )
    b.ret()
    return b.kernel


def yx_weight_transform_v2_grid(K, C):
    return (-(-(K * C // 4) // 256), 1, 1)


def build_yx_aux(D: int, H: int, W: int, C: int, K: int, *, out_dtype: str = "f16",
                 fold: bool = False, rev: bool = False, halo: int = 0,
                 flip: bool = False, wtv: int = 1, scale: bool = False) -> KernelDef:
    """One launch for both auxiliary passes of the yx kernel: blocks [0, NX) relayout X to c8
    (LDS-transposed), blocks [NX, NX + NW) transform the weights -- the two run side by side
    instead of back to back. Signature (X, X8, W, U); grid yx_aux_grid()."""
    from rocke.core.ir import F16

    odt = F16 if out_dtype == "f16" else BF16
    nx = x_to_c8_lds_grid(D, H, W, C)[0]
    b = IRBuilder(f"wino_yx_aux_V{D * H * W}C{C}K{K}_{out_dtype}{'_fold' if fold else ''}"
                  f"{'_rev' if rev else ''}{f'_h{halo}' if halo else ''}{'_flip' if flip else ''}"
                  f"{'_wt3' if wtv == 3 else ''}{'_wt4' if wtv == 4 else ''}{'_s' if scale else ''}")
    b.kernel.attrs["max_workgroup_size"] = 256
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    X8 = b.param("X8", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    Wp = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Uo = b.param("U", PtrType(odt, "global"), noalias=True, writeonly=True, align=16)
    S = b.param("S", F32) if scale else None
    lds = b.smem_alloc(F32, [_x_to_c8_lds_words(C)], name_hint="x8_t")
    ci = b.const_i32
    bid = b.block_id_x()
    with b.scf_if_else(b.cmp_lt(bid, ci(nx))) as (then_ctx, else_ctx):
        with then_ctx:
            _emit_x_to_c8_lds(b, X, X8, lds, D, H, W, C,
                              b.sub(ci(nx - 1), bid) if rev else bid, halo=halo, scale=S)
        with else_ctx:
            if wtv in (3, 4):
                _emit_yx_wt3(b, Wp, Uo, K, C, flip, b.sub(bid, ci(nx)), swz=wtv == 4)
            else:
                _emit_yx_wt(b, Wp, Uo, K, C, out_dtype, fold, False, False, b.sub(bid, ci(nx)),
                            flip=flip)
    b.ret()
    return b.kernel


def build_yx_frel_aux(spec: WinogradYxSpec) -> KernelDef:
    """Pre-launch of the fused-relayout (``frel``) main kernel: blocks [0, NXA) relayout the
    first ``base`` voxels of X to c8 (LDS-transposed), blocks [NXA, NXA + NW) transform the
    weights, the last block zeroes the producer flags. Signature (X, X8, W, U, F)."""
    from rocke.core.ir import F16

    D, H, W_, C, K = spec.Di, spec.Hi, spec.Wi, spec.C, spec.K
    A, base, vpw, nwg = _frel_plan(spec)
    VB = _x8_vb(C)
    nxa = -(-base // VB)
    nw = yx_weight_transform_grid(K, C)[0]
    b = IRBuilder(f"wino_yx_frel_aux_V{D * H * W_}C{C}K{K}_b{base}_n{nwg}")
    b.kernel.attrs["max_workgroup_size"] = 256
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    X8 = b.param("X8", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    Wp = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Uo = b.param("U", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    Fl = b.param("F", PtrType(I32, "global"), noalias=True, writeonly=True, align=4)
    lds = b.smem_alloc(F32, [_x_to_c8_lds_words(C)], name_hint="x8_t")
    ci = b.const_i32
    bid = b.block_id_x()
    with b.scf_if_else(b.cmp_lt(bid, ci(nxa))) as (then_ctx, else_ctx):
        with then_ctx:
            _emit_x_to_c8_lds(b, X, X8, lds, D, H, W_, C, bid, vlim=base)
        with else_ctx:
            with b.scf_if_else(b.cmp_lt(bid, ci(nxa + nw))) as (t2, e2):
                with t2:
                    _emit_yx_wt(b, Wp, Uo, K, C, "f16", False, False, False,
                                b.sub(bid, ci(nxa)))
                with e2:
                    tid = b.thread_id_x()
                    for r in range(-(-nwg // 256)):
                        idx = b.add(tid, ci(r * 256))
                        with b.scf_if(b.cmp_lt(idx, ci(nwg))):
                            b.global_store(Fl, idx, ci(0), align=4)
    b.ret()
    return b.kernel


def build_ks_reduce(nel: int, ks: int) -> KernelDef:
    """Split-K reduction: P [ks][nel] f32 -> Y [nel] bf16 (8 elements per thread)."""
    b = IRBuilder(f"wino_ks_reduce_n{nel}_k{ks}")
    b.kernel.attrs["max_workgroup_size"] = 256
    P = b.param("P", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    Yo = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    ci = b.const_i32
    e = b.mul(b.add(b.mul(b.block_id_x(), ci(256)), b.thread_id_x()), ci(8))
    rp = b.buffer_rsrc(P, ci(ks * nel * 4))
    ry = b.buffer_rsrc(Yo, ci(nel * 2))
    ok = b.cmp_lt(e, ci(nel))
    eo = b.select(ok, e, ci(nel))
    acc = None
    for k in range(ks):
        o = b.select(ok, b.mul(b.add(e, ci(k * nel)), ci(4)), ci(ks * nel * 4))
        v0 = b.buffer_load_vN(rp, o, ci(0), F32, 4)
        v1 = b.buffer_load_vN(rp, o, ci(16), F32, 4)
        acc = (v0, v1) if acc is None else (b.vector_add(acc[0], v0), b.vector_add(acc[1], v1))
    b.buffer_store_vN_bf16(ry, b.mul(eo, ci(2)), ci(0),
                           b.vec_trunc_f32_to_bf16(b.vec_concat(acc[0], acc[1])), 4)
    b.ret()
    return b.kernel


def ks_reduce_grid(nel: int):
    return (-(-nel // 2048), 1, 1)


def yx_frel_aux_grid(spec: WinogradYxSpec):
    A, base, vpw, nwg = _frel_plan(spec)
    return (-(-base // _x8_vb(spec.C)) + yx_weight_transform_grid(spec.K, spec.C)[0] + 1, 1, 1)


def yx_aux_grid(D, H, W, C, K, wtv=1):
    nw = (yx_weight_transform_v3_grid if wtv in (3, 4) else yx_weight_transform_grid)(K, C)[0]
    return (x_to_c8_lds_grid(D, H, W, C)[0] + nw, 1, 1)


def yx_weight_transform_grid(K, C):
    return (-(-(K * C) // 256), 1, 1)


# ---------------------------------------------------------------------------------
# F(2,3) on x only, direct along y and z ("x" variant).
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class WinogradXSpec:
    """1-D Winograd F(2,3) along x + direct 3x3 taps along (z, y); N=1 NDHWC bf16, valid.

    Workgroup = 8 waves: wave w owns the x-point c = w % 4 and the output-channel half
    nh = w // 4 (n_tile = 64). A block is 32 output x-pairs (MR rows of PXB pairs) x
    ``y_groups`` row groups x ``z_tile`` planes x 64 channels; every wave keeps
    y_groups * z_tile 32x32 f32 accumulators. The input chunk (8 channels, c8 layout
    [C/8][D][H][W][8]) is staged raw (bf16) in a double-buffered LDS ring, each wave
    combines its two x columns (B^T row c) in f32 and packs f16 MFMA operands. Weights:
    U [3 dz][3 dy][4 c][C/8][K][8] f16 (G along x only).
    """

    Di: int
    Hi: int
    Wi: int
    C: int
    K: int
    z_tile: int = 2
    y_groups: int = 4
    n_tile: int = 64
    c_chunk: int = 8
    bpf: bool = True  # prefetch the weight fragments one chunk ahead
    pkrtz: bool = True
    dbg: str = ""
    name: str = "conv3d_winox"

    @property
    def Do(self) -> int:
        return self.Di - 2

    @property
    def Ho(self) -> int:
        return self.Hi - 2

    @property
    def Wo(self) -> int:
        return self.Wi - 2

    @property
    def waves(self) -> int:
        return 8

    @property
    def block_size(self) -> int:
        return 512

    @property
    def op_dtype(self) -> str:
        return "f16"

    @property
    def flops(self) -> int:
        return 2 * self.C * self.K * 27 * self.Do * self.Ho * self.Wo

    def geom(self):
        PX = self.Wo // 2
        PXB = min(PX, 32)
        MR = 32 // PXB
        return PX, PXB, MR

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"D{self.Di}H{self.Hi}W{self.Wi}C{self.C}K{self.K}",
            "bf16",
            f"z{self.z_tile}y{self.y_groups}n{self.n_tile}",
            *(["nobpf"] if not self.bpf else []),
            *(["cvt"] if not self.pkrtz else []),
            *([f"dbg{self.dbg}"] if self.dbg else []),
        )


def _x_buf_words(spec) -> int:
    PX, PXB, MR = spec.geom()
    rows = MR * spec.y_groups + 2
    return (spec.z_tile + 2) * rows * 2 * (PXB + 1) * 4  # 16 B per (plane,row,par,half)


def is_valid_x_spec(spec: WinogradXSpec) -> Tuple[bool, str]:
    if spec.Wo % 2 or spec.Wo <= 0 or spec.Ho <= 0 or spec.Do <= 0:
        return False, "F(2,3) along x needs an even output width"
    PX, PXB, MR = spec.geom()
    if not (PX % 32 == 0 or 32 % PX == 0):
        return False, "x pairs per row must be a multiple or divisor of 32"
    if spec.Ho % (MR * spec.y_groups):
        return False, f"Ho={spec.Ho} not a multiple of {MR * spec.y_groups} rows"
    if spec.Do % spec.z_tile:
        return False, "z_tile must divide Do"
    if spec.n_tile != 64 or spec.K % 64 or spec.C % 8 or spec.c_chunk != 8:
        return False, "n_tile 64 (K % 64 == 0), C % 8 == 0, c_chunk 8"
    if spec.z_tile * spec.y_groups * 16 > 160:
        return False, "too many accumulators per wave"
    if max(2 * _x_buf_words(spec), 2 * 4 * 16 * 64) * 4 > 65536:
        return False, "LDS staging exceeds 64 KiB"
    return True, "ok"


def winograd_x_grid(spec: WinogradXSpec) -> Tuple[int, int, int]:
    PX, PXB, MR = spec.geom()
    nblk = (PX // PXB) * (spec.Ho // (MR * spec.y_groups)) * (spec.Do // spec.z_tile) * (
        spec.K // 64)
    return (nblk, 1, 1)


def build_winograd_x_fwd(spec: WinogradXSpec) -> KernelDef:
    from rocke.core.ir import F16

    ok, why = is_valid_x_spec(spec)
    if not ok:
        raise ValueError(f"invalid winograd x spec: {why}")
    atom = MfmaAtom.f16_32x32x8()
    C, K = spec.C, spec.K
    Di, Hi, Wi = spec.Di, spec.Hi, spec.Wi
    Ho, Wo = spec.Ho, spec.Wo
    ZT, YT = spec.z_tile, spec.y_groups
    PX, PXB, MR = spec.geom()
    ROWS = MR * YT + 2
    NPL = ZT + 2
    NCOL = 2 * PXB + 2
    HC = PXB + 1
    BUF = _x_buf_words(spec)
    NXB = PX // PXB
    NYB = Ho // (MR * YT)
    NZB = Do_ = spec.Do // ZT
    NB = K // 64
    items = NPL * ROWS * NCOL
    rounds = -(-items // 512)
    NACC = YT * ZT

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 512
    # LDS reads must not be sunk below / hoisted above the ring barriers
    b.fence_barriers = "compiler"
    X = b.param("X", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)

    ci = b.const_i32
    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(64)))
    cpt = b.mod(wave, ci(4))  # x-point of this wave
    nh = b.div(wave, ci(4))
    lane = b.mod(tid, ci(64))
    m = b.mod(lane, ci(32))
    kq = b.div(lane, ci(32))
    pid = b.block_id_x()
    bid_n = b.mod(pid, ci(NB))
    r_ = b.div(pid, ci(NB))
    xb = b.mod(r_, ci(NXB))
    r_ = b.div(r_, ci(NXB))
    yb = b.mod(r_, ci(NYB))
    zb = b.div(r_, ci(NYB))
    px0 = b.mul(xb, ci(PXB))
    y0 = b.mul(yb, ci(MR * YT))
    z0 = b.mul(zb, ci(ZT))

    lds = b.smem_alloc(F32, [max(2 * BUF, 2 * 4 * 16 * 64)], name_hint="winox_raw")

    plane_b = Hi * Wi * 16
    chunk_b = Di * plane_b
    rx = b.buffer_rsrc(X, ci(Di * Hi * Wi * C * 2))
    ru = b.buffer_rsrc(U, ci(36 * C * K * 2))
    n_glob = b.add(b.add(b.mul(bid_n, ci(64)), b.mul(nh, ci(32))), m)
    vu = b.mul(b.add(b.mul(n_glob, ci(8)), b.mul(kq, ci(4))), ci(2))
    u_pt = (C // 8) * K * 8

    # fill items (plane, row, col): 16 B (8 channels) each
    fill = []
    for rr in range(rounds):
        it = b.add(tid, ci(rr * 512))
        valid = b.cmp_lt(it, ci(items))
        it = b.smin(it, ci(items - 1))
        col = b.mod(it, ci(NCOL))
        t2 = b.div(it, ci(NCOL))
        row = b.mod(t2, ci(ROWS))
        pl = b.div(t2, ci(ROWS))
        goff = b.add(
            b.add(b.mul(b.add(z0, pl), ci(plane_b)), b.mul(b.add(y0, row), ci(Wi * 16))),
            b.mul(b.add(b.mul(px0, ci(2)), col), ci(16)),
        )
        loff = b.mul(
            b.add(b.mul(b.add(b.mul(b.add(b.mul(pl, ci(ROWS)), row), ci(2)), b.mod(col, ci(2))),
                        ci(HC)), b.div(col, ci(2))),
            ci(4),
        )
        fill.append((valid, goff, loff))

    def load_fill(cc):
        sx = b.mul(cc, ci(chunk_b))
        return [b.buffer_load_vN(rx, goff, sx, I32, 4) for _, goff, _ in fill]

    def store_fill(raw, bufoff):
        for (valid, _, loff), v in zip(fill, raw):
            with b.scf_if(valid):
                b.smem_store_vN_f32(lds, [b.add(loff, bufoff)], b.vec_bitcast(v, VectorType(F32, 4)), 4)

    bkeys = [(dz, dy) for dz in range(3) for dy in range(3)]
    s_uc = b.mul(cpt, ci(u_pt * 2))

    def load_b(cc):
        su = b.add(s_uc, b.mul(cc, ci(K * 8 * 2)))
        return [
            b.buffer_load_vN(ru, vu, b.add(su, ci((dz * 3 + dy) * 4 * u_pt * 2)), I32, 2)
            for dz, dy in bkeys
        ]

    # this wave's two x columns and sign: value = x[ja] + r * x[jb]
    ja = b.select(b.cmp_eq(cpt, ci(0)), ci(0), b.select(b.cmp_eq(cpt, ci(2)), ci(2), ci(1)))
    jb = b.select(b.cmp_eq(cpt, ci(0)), ci(2), b.select(b.cmp_eq(cpt, ci(1)), ci(2),
                                                        b.select(b.cmp_eq(cpt, ci(2)), ci(1), ci(3))))
    rr_ = b.select(b.cmp_eq(cpt, ci(1)), b.const_f32(1.0), b.const_f32(-1.0))
    rv = b.vector_splat(rr_, 2)
    l_ry = b.div(m, ci(PXB))
    l_px = b.mod(m, ci(PXB))

    def col_slot(j):
        # word offset of column 2*px + j within a (plane,row): parity j % 2, half px + j // 2
        par = b.mod(j, ci(2))
        half = b.add(l_px, b.div(j, ci(2)))
        return b.mul(b.add(b.mul(par, ci(HC)), half), ci(4))

    cs_a = b.add(col_slot(ja), b.mul(kq, ci(2)))
    cs_b = b.add(col_slot(jb), b.mul(kq, ci(2)))

    def bf16x2_to_f32v(d):
        lo = b.bitcast(b.shl(d, ci(16)), F32)
        hi = b.bitcast(b.land(d, ci(-65536)), F32)
        return b.vec_pack([lo, hi], F32)

    def frag(pl, rofs, bufoff):
        """A operand for input plane pl and row offset rofs (rows rofs + ry of the block)."""
        rowb = b.mul(b.add(b.mul(ci(pl), ci(ROWS)), b.add(ci(rofs), l_ry)), ci(2 * HC * 4))
        base = b.add(rowb, bufoff)
        va = b.vec_bitcast(b.smem_load_vN_f32(lds, b.add(base, cs_a), n=2), VectorType(I32, 2))
        vb = b.vec_bitcast(b.smem_load_vN_f32(lds, b.add(base, cs_b), n=2), VectorType(I32, 2))
        outs = []
        for e in range(2):
            fa = bf16x2_to_f32v(b.vec_extract(va, e))
            fb = bf16x2_to_f32v(b.vec_extract(vb, e))
            outs.append(b.vector_fma(rv, fb, fa))
        if spec.pkrtz:
            # The result feeds an MFMA directly: LLVM's hazard recognizer does not see into
            # inline asm, so the VALU-write -> MFMA-read wait states are spelled out here
            # (without the s_nop the MFMA reads stale operands -- wrong results).
            pk = [
                b.inline_asm(
                    "v_cvt_pkrtz_f16_f32 $0, $1, $2" + ("\n\ts_nop 1" if e == 1 else ""),
                    "=v,v,v", [b.vec_extract(o, 0), b.vec_extract(o, 1)], I32,
                    sideeffect=False,
                )
                for e, o in enumerate(outs)
            ]
            return b.vec_bitcast(b.vec_pack(pk, I32), VectorType(F16, 4))
        return b.vec_cast_f32_to(b.vec_concat(outs[0], outs[1]), F16)

    n_chunks = C // spec.c_chunk
    store_fill(load_fill(ci(0)), ci(0))
    raw1 = load_fill(ci(min(1, n_chunks - 1)))
    b0 = load_b(ci(0)) if spec.bpf else []
    b.sync_lds_only()
    nfill = len(raw1)
    iters = [(f"acc{i}", atom.zero_acc(b)) for i in range(NACC)]
    iters += [(f"raw{i}", v) for i, v in enumerate(raw1)]
    iters += [(f"bfr{i}", v) for i, v in enumerate(b0)]
    loop = b.scf_for_iter(ci(0), ci(n_chunks), ci(1), iters, iv_name="cc")
    with loop as (cc, its):
        accs = list(its[:NACC])
        raw = its[NACC: NACC + nfill]
        if spec.bpf:
            bf = list(its[NACC + nfill:])
            nb = load_b(b.smin(b.add(cc, ci(1)), ci(n_chunks - 1)))
        else:
            bf = load_b(cc)
        par_c = b.mod(cc, ci(2))
        cur = b.mul(par_c, ci(BUF))
        nxt = b.mul(b.sub(ci(1), par_c), ci(BUF))
        nraw = load_fill(b.smin(b.add(cc, ci(2)), ci(n_chunks - 1)))
        bfh = [b.vec_bitcast(v, VectorType(F16, 4)) for v in bf]
        for pl in range(NPL):
            # rows used by this plane: rofs = g * MR + dy
            frags = {}
            for zo in range(ZT):
                dz = pl - zo
                if not 0 <= dz < 3:
                    continue
                for g in range(YT):
                    for dy in range(3):
                        ro = g * MR + dy
                        if ro not in frags:
                            frags[ro] = frag(pl, ro, cur)
                        k = g * ZT + zo
                        accs[k] = atom.emit(b, frags[ro], bfh[dz * 3 + dy], accs[k])
        if "xbar" in spec.dbg:
            b.sync_lds_only()
        store_fill(raw, nxt)
        b.sync_lds_only()
        if spec.bpf:
            b.scf_yield(*accs, *nraw, *nb)
        else:
            b.scf_yield(*accs, *nraw)
    acc = loop.results[:NACC]

    # Epilogue: out[ox] = sum_c AT[ox][c] M_c, M_c held by wave (c, nh); one (g, zo) per round.
    ry_out = b.buffer_rsrc(Y, ci(spec.Do * Ho * Wo * K * 2))
    nq = b.mod(lane, ci(8))
    gq = b.div(lane, ci(8))  # 0..7: slot sub-index (i = 4 cpt + gq % 4) and kq2 = gq // 4
    i_dyn = b.add(b.mul(cpt, ci(4)), b.mod(gq, ci(4)))
    kq2 = b.div(gq, ci(4))
    mrow = b.add(b.add(b.mul(b.div(i_dyn, ci(4)), ci(8)), b.mul(kq2, ci(4))), b.mod(i_dyn, ci(4)))
    o_ry = b.div(mrow, ci(PXB))
    o_px = b.mod(mrow, ci(PXB))
    n_base = b.add(b.add(b.mul(bid_n, ci(64)), b.mul(nh, ci(32))), b.mul(nq, ci(4)))
    src_lane = b.add(b.mul(kq2, ci(32)), b.mul(nq, ci(4)))
    for g in range(YT):
        for zo in range(ZT):
            k = g * ZT + zo
            b.sync_lds_only()
            for i in range(16):
                off = b.add(b.mul(b.add(b.mul(b.add(b.mul(nh, ci(4)), cpt), ci(16)), ci(i)), ci(64)),
                            lane)
                b.smem_store_vN_f32(lds, [off], b.vec_extract(acc[k], i), 1)
            b.sync_lds_only()
            parts = []
            for c in range(4):
                off = b.add(b.mul(b.add(b.mul(b.add(b.mul(nh, ci(4)), ci(c)), ci(16)), i_dyn),
                                  ci(64)), src_lane)
                parts.append(b.smem_load_vN_f32(lds, off, n=4))
            yy = b.add(b.add(y0, ci(g * MR)), o_ry)
            zz = b.add(z0, ci(zo))
            for ox in range(2):
                out = None
                for c in range(4):
                    co = AT[ox][c]
                    if co == 0:
                        continue
                    out = parts[c] if out is None and co > 0 else (
                        b.vector_sub(b.vector_splat(b.const_f32(0.0), 4), parts[c]) if out is None
                        else (b.vector_add(out, parts[c]) if co > 0 else b.vector_sub(out, parts[c])))
                xx = b.add(b.mul(b.add(px0, o_px), ci(2)), ci(ox))
                yoff = b.add(b.mul(b.add(b.mul(b.add(b.mul(zz, ci(Ho)), yy), ci(Wo)), xx), ci(K)),
                             n_base)
                b.buffer_store_vN_bf16(ry_out, b.mul(yoff, ci(2)), ci(0),
                                       b.vec_trunc_f32_to_bf16(out), 2)
    b.ret()
    return b.kernel


def winograd_x_weight_transform(w, *, dtype=None):
    """``w``: [K, 3, 3, 3, C] -> U [3 dz][3 dy][4 c][C/8][K][8] (G along x only)."""
    import torch

    key = "x" + str(w.device)
    if key not in _WT_CONST:
        _WT_CONST[key] = (torch.tensor(G, dtype=torch.float32, device=w.device), None)
    g, _ = _WT_CONST[key]
    K, C = w.shape[0], w.shape[-1]
    u = torch.einsum("ck,nzyks->nzycs", g, w.float())  # [K, 3, 3, 4, C]
    u = u.permute(1, 2, 3, 0, 4).reshape(3, 3, 4, K, C // 8, 8).permute(0, 1, 2, 4, 3, 5)
    return u.to(dtype or torch.float16).contiguous()


def build_x_weight_transform(K: int, C: int) -> KernelDef:
    """W [K][3][3][3][C] bf16 -> U [3 dz][3 dy][4 c][C/8][K][8] f16 (G along x only).
    One thread per (k, c), c fastest: 27 coalesced 2-byte loads, 36 stores. Grid (C*K/256)."""
    from rocke.core.ir import F16

    b = IRBuilder(f"wino_x_wt_K{K}C{C}")
    b.kernel.attrs["max_workgroup_size"] = 256
    W = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Uo = b.param("U", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    ci = b.const_i32
    t = b.add(b.mul(b.block_id_x(), ci(256)), b.thread_id_x())
    c = b.mod(t, ci(C))
    k = b.div(t, ci(C))
    base = b.add(b.mul(b.add(b.mul(b.div(c, ci(8)), ci(K)), k), ci(8)), b.mod(c, ci(8)))
    for dz in range(3):
        for dy in range(3):
            w = [
                b.cast_to_f32(b.global_load(
                    W, b.add(b.mul(b.add(b.mul(k, ci(27)), ci((dz * 3 + dy) * 3 + x)), ci(C)), c),
                    BF16, align=2))
                for x in range(3)
            ]
            for cc in range(4):
                acc = None
                for x in range(3):
                    gv = G[cc][x]
                    if gv == 0.0:
                        continue
                    term = b.fmul(b.const_f32(gv), w[x])
                    acc = term if acc is None else b.fadd(acc, term)
                pt = (dz * 3 + dy) * 4 + cc
                b.global_store(Uo, b.add(base, ci(pt * C * K)), b.trunc_f32_to_f16(acc), align=2)
    b.ret()
    return b.kernel


def x_weight_transform_grid(K, C):
    return (-(-(K * C) // 256), 1, 1)


__all__ = [
    "build_x_weight_transform",
    "x_weight_transform_grid",
    "WinogradXSpec",
    "build_winograd_x_fwd",
    "is_valid_x_spec",
    "winograd_x_grid",
    "winograd_x_weight_transform",
    "build_x_to_c8",
    "build_x_to_c8_lds",
    "x_to_c8_lds_grid",
    "x_to_c8_grid",
    "build_yx_aux",
    "yx_aux_grid",
    "build_yx_weight_transform",
    "build_yx_weight_transform_v2",
    "yx_weight_transform_v2_grid",
    "yx_weight_transform_grid",
    "WinogradYxSpec",
    "build_winograd_yx",
    "build_winograd_yx_fwd",
    "build_winograd_yx_lds_fwd",
    "build_winograd_yx_lds16_fwd",
    "build_winograd_yx_wlds_fwd",
    "is_valid_yx_spec",
    "winograd_yx_grid",
    "winograd_yx_weight_transform",
    "winograd3d_x_to_c8",
    "Winograd3dSpec",
    "build_winograd3d_fwd",
    "is_valid_spec",
    "winograd3d_grid",
    "winograd3d_signature",
    "winograd3d_weight_transform",
]
