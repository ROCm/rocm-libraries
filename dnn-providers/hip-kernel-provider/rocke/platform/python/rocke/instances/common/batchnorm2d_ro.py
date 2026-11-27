# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Read-once (single pass over global memory) BatchNorm kernels.

A persistent grid keeps each CTA's tile in registers across a cross-CTA
exchange of per-channel partial statistics, so the tensor is read from global
memory once. The exchange is an arrival counter per channel plus a bounded
spin; the grid must never exceed what is co-resident (the plan refuses
otherwise) and the host zeroes the counters before every launch. A spin that
runs out of its bound raises a flag in the error word instead of hanging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from ...helpers.reduction import (
    block_lds_reduce_pair,
    block_lds_reduce_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import kernel_name_join
from ...helpers.tensor_view import make_lds_view
from .batchnorm2d import _wave_size_error


# Ints per channel record of the sync buffer: [0] arrival counter, [1:3] and [3:5]
# the two accumulator sets (selected by the launch epoch parity), the rest padding
# so records of different channels never share a cache line.
RO_STRIDE = 32
# Largest group of CTAs that share one channel (one arrival counter).
RO_MAX_SPLITS = 32


# ===================================================================== #
# Read-once NCHW training forward / backward
# ===================================================================== #
#
# One persistent grid of ``Grid`` CTAs. A work item is (channel, batch split,
# spatial split); ``S = SplitN * SplitHw`` adjacent items belong to one channel
# and a round of the persistent loop covers ``Grid / S`` channels. Each CTA
# keeps its tile (``R`` vectors per thread) in registers:
#
#   load tile -> per-thread partial sums -> block reduction -> add the block
#   total into the channel's accumulator pair -> arrive on the channel's counter
#   -> bounded spin -> read the channel totals -> apply from registers.
#
# Both the accumulators and the counter are device-scope atomics: no fence and
# no L2 write-back is needed, and the exchange cost is S-way, not grid-way.
# The sync buffer is host-initialised (zero) once; later launches pass a
# growing ``Epoch``: counters then run up to ``(Epoch + 1) * S`` and the
# accumulator set of the parity not in use is cleared by the channel's first
# item, so a launch needs no memset. A fresh zeroed buffer and ``Epoch = 0``
# is the memset flavour. ``Err[0]`` counts spins that ran out of their bound.


@dataclass(frozen=True)
class BatchNorm2DRoSpec:
    """One read-once kernel (n/c/h/w, split geometry and grid are runtime)."""

    block_size: int = 256
    vec: int = 8
    regs: int = 16  # vectors held per thread
    dtype: str = "f16"
    fast_reduce: bool = False
    save_mean_invstd: bool = True
    wave_size: int = 64
    name: str = "rocke_batchnorm2d_ro"

    @property
    def cap_elems(self) -> int:
        return self.block_size * self.vec * self.regs

    def kernel_name(self, kind: str = "train") -> str:
        return kernel_name_join(
            self.name,
            kind,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            f"r{self.regs}",
            flags={"fr": self.fast_reduce},
        )


@dataclass(frozen=True)
class BatchNorm2DRoPlan:
    n_chunk: int
    hw_chunk: int
    sn: int
    sh: int
    grid: int

    @property
    def splits(self) -> int:
        return self.sn * self.sh


def is_valid_ro_spec(spec: BatchNorm2DRoSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    vecs = (2, 4) if spec.dtype == "f32" else (4, 8)
    if spec.dtype not in ("f16", "bf16", "f32"):
        return False, "unsupported dtype"
    if spec.vec not in vecs:
        return False, f"vec {spec.vec} not in {vecs}"
    if _wave_size_error(spec):
        return False, _wave_size_error(spec)
    if (
        spec.block_size % spec.wave_size
        or spec.block_size > 1024
        or spec.block_size < 64
    ):
        return False, "block_size must be a multiple of the wave size, <= 1024"
    if spec.regs < 1 or spec.regs > 64:
        return False, "regs out of range"
    return True, ""


def batchnorm2d_ro_plan(
    spec: BatchNorm2DRoSpec,
    n: int,
    c: int,
    h: int,
    w: int,
    *,
    sn_req: int,
    sh_req: int,
    max_resident: int,
) -> "BatchNorm2DRoPlan | None":
    """Launch geometry, or ``None`` when the request is not admissible.

    The persistent grid is the largest multiple of the group size ``S`` that is
    co-resident (``max_resident`` CTAs), capped at ``C * S``; a plan is refused
    when no whole group fits or the tile does not fit the register budget.
    """
    hw = int(h) * int(w)
    if n <= 0 or c <= 0 or hw <= 0 or sn_req < 1 or sh_req < 1:
        return None
    if hw % spec.vec or int(n) * int(c) * hw >= (1 << 31):
        return None
    hw_chunk = -(-hw // sh_req)
    hw_chunk = -(-hw_chunk // spec.vec) * spec.vec
    sh = -(-hw // hw_chunk)
    n_chunk = -(-n // sn_req)
    sn = -(-n // n_chunk)
    splits = sn * sh
    if splits > RO_MAX_SPLITS or splits > spec.block_size:
        return None
    if n_chunk * hw_chunk > spec.cap_elems:
        return None
    groups = min(int(max_resident) // splits, int(c))
    if groups < 1:
        return None
    return BatchNorm2DRoPlan(
        n_chunk=n_chunk, hw_chunk=hw_chunk, sn=sn, sh=sh, grid=groups * splits
    )


def batchnorm2d_ro_sync_ints(c: int) -> int:
    return int(c) * RO_STRIDE


def _ro_item(b, p_c, p_n, HW, p_nch, p_hwch, p_sh, p_sn, w_id):
    """Decode a work item: ``(channel, sub, hw_lo, hw_len, n_lo, n_cnt, total)``."""
    zero = b.const_i32(0)
    S = b.mul(p_sh, p_sn)
    ch = b.div(w_id, S)
    sub = b.sub(w_id, b.mul(ch, S))
    sn_id = b.div(sub, p_sh)
    sh_id = b.sub(sub, b.mul(sn_id, p_sh))
    hw_lo = b.mul(sh_id, p_hwch)
    hw_len = b.smax(b.smin(b.sub(HW, hw_lo), p_hwch), b.const_i32(1))
    n_lo = b.mul(sn_id, p_nch)
    n_cnt = b.smax(b.smin(b.sub(p_n, n_lo), p_nch), b.const_i32(1))
    return ch, sub, S, hw_lo, hw_len, n_lo, n_cnt, b.mul(n_cnt, hw_len)


def _ro_tile_index(b, spec, tid, j, p_c, HW, ch, hw_lo, hw_len, n_lo, total):
    """``(valid, flat)`` of vector ``j`` of this thread (flat index clamped)."""
    k = b.mul(b.add(b.const_i32(j * spec.block_size), tid), b.const_i32(spec.vec))
    ok = b.cmp_lt(k, total)
    kk = b.select(ok, k, b.const_i32(0))
    n_rel = b.div(kk, hw_len)
    hw_rel = b.sub(kk, b.mul(n_rel, hw_len))
    row = b.add(b.mul(b.add(n_lo, n_rel), p_c), ch)
    flat = b.add(b.add(b.mul(row, HW), hw_lo), hw_rel)
    return ok, flat


def _ro_block_pair(b, spec, v1, v2, lds, tid):
    lds_a, lds_c = lds
    if spec.fast_reduce:
        return (
            block_lds_reduce_with_wave_prologue(
                b, v1, lds_a, tid, block_size=spec.block_size, wave_size=spec.wave_size
            ),
            block_lds_reduce_with_wave_prologue(
                b, v2, lds_c, tid, block_size=spec.block_size, wave_size=spec.wave_size
            ),
        )
    return block_lds_reduce_pair(
        b, v1, v2, lds_a, lds_c, tid, block_size=spec.block_size
    )


def _ro_exchange(
    b, SyncI, SyncF, Err, tid, lds_bc, ch, sub, S, p_epoch, p_spin, t1, t2
):
    """Add the block totals ``(t1, t2)`` to the channel accumulators, wait for
    the other ``S - 1`` items, return the channel totals on every thread."""
    zero = b.const_i32(0)
    one = b.const_i32(1)
    rec = b.mul(ch, b.const_i32(RO_STRIDE))
    parity = b.land(p_epoch, one)
    acc = b.add(b.add(rec, one), b.mul(parity, b.const_i32(2)))
    other = b.add(b.add(rec, one), b.mul(b.sub(one, parity), b.const_i32(2)))
    target = b.mul(b.add(p_epoch, one), S)
    with b.scf_if(b.cmp_eq(tid, zero)):
        with b.scf_if(b.cmp_eq(sub, zero)):
            b.global_store(SyncF, other, b.const_f32(0.0))
            b.global_store(SyncF, b.add(other, one), b.const_f32(0.0))
        b.global_atomic_add(SyncF, acc, t1, ordering="monotonic", scope="agent")
        b.global_atomic_add(
            SyncF, b.add(acc, one), t2, ordering="monotonic", scope="agent"
        )
        b.s_waitcnt(vmcnt=0)
        b.global_atomic_add(SyncI, rec, one, ordering="monotonic", scope="agent")
        seen = b.global_flag_wait_ge_bounded(
            SyncI, rec, target, p_spin, scope="agent", ordering="monotonic"
        )
        with b.scf_if(b.cmp_lt(seen, target)):
            b.global_atomic_add(Err, zero, one, ordering="monotonic", scope="agent")
        f0 = b.const_f32(0.0)
        a1 = b.global_atomic_add(SyncF, acc, f0, ordering="monotonic", scope="agent")
        a2 = b.global_atomic_add(
            SyncF, b.add(acc, one), f0, ordering="monotonic", scope="agent"
        )
        b.smem_store_vN_f32(lds_bc, [zero], a1, 1)
        b.smem_store_vN_f32(lds_bc, [one], a2, 1)
    b.sync()
    s1 = b.vec_extract(b.smem_load_vN_f32(lds_bc, zero, n=1), 0)
    s2 = b.vec_extract(b.smem_load_vN_f32(lds_bc, one, n=1), 0)
    return s1, s2


def _ro_common_params(b, spec):
    return {
        "p_n": b.param("N", I32),
        "p_c": b.param("C", I32),
        "p_h": b.param("H", I32),
        "p_w": b.param("W", I32),
    }


def _ro_geom_params(b):
    return {
        "p_nch": b.param("NChunk", I32),
        "p_hwch": b.param("HwChunk", I32),
        "p_sh": b.param("SplitHw", I32),
        "p_sn": b.param("SplitN", I32),
        "p_grid": b.param("Grid", I32),
        "p_epoch": b.param("Epoch", I32),
        "p_spin": b.param("MaxSpin", I32),
    }


def build_batchnorm2d_ro_train(spec: BatchNorm2DRoSpec) -> KernelDef:
    """Read-once training forward (statistics, running stats, ``Y``)."""
    ok, why = is_valid_ro_spec(spec)
    if not ok:
        raise ValueError(f"invalid read-once batchnorm spec: {why}")
    io_ty = io_ir_type(spec.dtype)
    BS, VEC, R = spec.block_size, spec.vec, spec.regs

    b = IRBuilder(spec.kernel_name("train"))
    b.kernel.attrs["max_workgroup_size"] = BS
    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Beta = b.param(
        "Beta", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    RunningMean = b.param("RunningMean", PtrType(F32, "global"), noalias=True)
    RunningVar = b.param("RunningVar", PtrType(F32, "global"), noalias=True)
    if spec.save_mean_invstd:
        Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, writeonly=True)
        InvStd = b.param(
            "InvStd", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
    SyncI = b.param("SyncI", PtrType(I32, "global"))
    SyncF = b.param("SyncF", PtrType(F32, "global"))
    Err = b.param("Err", PtrType(I32, "global"), noalias=True)
    P = _ro_common_params(b, spec)
    p_n, p_c, p_h, p_w = P["p_n"], P["p_c"], P["p_h"], P["p_w"]
    eps = b.param("eps", F32)
    momentum = b.param("momentum", F32)
    G = _ro_geom_params(b)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    cta = b.block_id_x()
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_s1", "lds_s2")
    )
    lds_bc = make_lds_view(b, dtype=F32, shape=(2,), name_hint="lds_bc").base
    S = b.mul(G["p_sh"], G["p_sn"])
    total_items = b.mul(p_c, S)
    rounds = b.div(b.sub(b.add(total_items, G["p_grid"]), b.const_i32(1)), G["p_grid"])
    m_f = b.sitofp_f32(b.mul(p_n, HW))

    loop = b.scf_for(b.const_i32(0), rounds, b.const_i32(1), iv_name="ro_round")
    with loop as rnd:
        w_id = b.add(b.mul(rnd, G["p_grid"]), cta)
        live = b.cmp_lt(w_id, total_items)
        branch = b.scf_if_else(live)
        with branch as (then_ctx, else_ctx):
            with then_ctx:
                ch, sub, _S, hw_lo, hw_len, n_lo, _n_cnt, total = _ro_item(
                    b, p_c, p_n, HW, G["p_nch"], G["p_hwch"], G["p_sh"], G["p_sn"], w_id
                )
                shift = load_scalar_as_f32(b, X, b.mul(ch, HW), dtype=spec.dtype)
                raws, oks = [], []
                for j in range(R):
                    ok_j, flat = _ro_tile_index(
                        b, spec, tid, j, p_c, HW, ch, hw_lo, hw_len, n_lo, total
                    )
                    raws.append(b.global_load_vN(X, flat, io_ty, VEC))
                    oks.append((ok_j, flat))
                s1 = b.const_f32(0.0)
                s2 = b.const_f32(0.0)
                zero_f = b.const_f32(0.0)
                for j in range(R):
                    xs = [b.cast_to_f32(b.vec_extract(raws[j], i)) for i in range(VEC)]
                    ds = [b.select(oks[j][0], b.fsub(x, shift), zero_f) for x in xs]
                    s1 = b.fadd(s1, tree_reduce(b, b.fadd, list(ds)))
                    s2 = b.fadd(s2, tree_reduce(b, b.fadd, [b.fmul(d, d) for d in ds]))
                t1, t2 = _ro_block_pair(b, spec, s1, s2, lds, tid)
                a1, a2 = _ro_exchange(
                    b,
                    SyncI,
                    SyncF,
                    Err,
                    tid,
                    lds_bc,
                    ch,
                    sub,
                    S,
                    G["p_epoch"],
                    G["p_spin"],
                    t1,
                    t2,
                )
                mean_d = b.fdiv(a1, m_f)
                mean = b.fadd(shift, mean_d)
                var = b.fmax(
                    b.fsub(b.fdiv(a2, m_f), b.fmul(mean_d, mean_d)), b.const_f32(0.0)
                )
                inv_std = b.rsqrt(b.fadd(var, eps))
                owner = b.land(
                    b.cmp_eq(sub, b.const_i32(0)), b.cmp_eq(tid, b.const_i32(0))
                )
                with b.scf_if(owner):
                    old_mean = b.global_load_f32(RunningMean, ch)
                    old_var = b.global_load_f32(RunningVar, ch)
                    omm = b.fsub(b.const_f32(1.0), momentum)
                    b.global_store(
                        RunningMean,
                        ch,
                        b.fadd(b.fmul(omm, old_mean), b.fmul(momentum, mean)),
                    )
                    unbias = b.fdiv(
                        m_f, b.fmax(b.fsub(m_f, b.const_f32(1.0)), b.const_f32(1.0))
                    )
                    b.global_store(
                        RunningVar,
                        ch,
                        b.fadd(
                            b.fmul(omm, old_var), b.fmul(momentum, b.fmul(var, unbias))
                        ),
                    )
                    if spec.save_mean_invstd:
                        store_scalar_from_f32(b, Mean, ch, mean, dtype=spec.dtype)
                        store_scalar_from_f32(b, InvStd, ch, inv_std, dtype=spec.dtype)
                gamma_v = load_scalar_as_f32(b, Gamma, ch, dtype=spec.dtype)
                beta_v = load_scalar_as_f32(b, Beta, ch, dtype=spec.dtype)
                scale = b.fmul(inv_std, gamma_v)
                for j in range(R):
                    xs = [b.cast_to_f32(b.vec_extract(raws[j], i)) for i in range(VEC)]
                    ys = [b.fadd(b.fmul(b.fsub(x, mean), scale), beta_v) for x in xs]
                    with b.scf_if(oks[j][0]):
                        store_vec(
                            b, Y, oks[j][1], pack_f32_to(b, ys, dtype=spec.dtype), n=VEC
                        )
                b.sync()
            with else_ctx:
                pass
    return b.kernel


def build_batchnorm2d_ro_bwd(spec: BatchNorm2DRoSpec) -> KernelDef:
    """Read-once backward: ``dx`` from registers, ``dScale`` / ``dBias`` by the
    first item of each channel (saved mean / inverse std as inputs)."""
    from .batchnorm2d import _bwd_coefs

    ok, why = is_valid_ro_spec(spec)
    if not ok:
        raise ValueError(f"invalid read-once batchnorm spec: {why}")
    io_ty = io_ir_type(spec.dtype)
    BS, VEC, R = spec.block_size, spec.vec, spec.regs

    b = IRBuilder(spec.kernel_name("bwd"))
    b.kernel.attrs["max_workgroup_size"] = BS
    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    dY = b.param("dY", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Scale = b.param("Scale", PtrType(F32, "global"), noalias=True, readonly=True)
    Mean = b.param("Mean", PtrType(F32, "global"), noalias=True, readonly=True)
    InvStd = b.param("InvStd", PtrType(F32, "global"), noalias=True, readonly=True)
    dX = b.param("dX", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    dScale = b.param("dScale", PtrType(F32, "global"), noalias=True, writeonly=True)
    dBias = b.param("dBias", PtrType(F32, "global"), noalias=True, writeonly=True)
    SyncI = b.param("SyncI", PtrType(I32, "global"))
    SyncF = b.param("SyncF", PtrType(F32, "global"))
    Err = b.param("Err", PtrType(I32, "global"), noalias=True)
    P = _ro_common_params(b, spec)
    p_n, p_c, p_h, p_w = P["p_n"], P["p_c"], P["p_h"], P["p_w"]
    G = _ro_geom_params(b)

    HW = b.mul(p_h, p_w)
    tid = b.thread_id_x()
    cta = b.block_id_x()
    lds = tuple(
        make_lds_view(b, dtype=F32, shape=(BS,), name_hint=nm).base
        for nm in ("lds_s1", "lds_s2")
    )
    lds_bc = make_lds_view(b, dtype=F32, shape=(2,), name_hint="lds_bc").base
    S = b.mul(G["p_sh"], G["p_sn"])
    total_items = b.mul(p_c, S)
    rounds = b.div(b.sub(b.add(total_items, G["p_grid"]), b.const_i32(1)), G["p_grid"])
    m_f = b.sitofp_f32(b.mul(p_n, HW))

    loop = b.scf_for(b.const_i32(0), rounds, b.const_i32(1), iv_name="ro_round")
    with loop as rnd:
        w_id = b.add(b.mul(rnd, G["p_grid"]), cta)
        live = b.cmp_lt(w_id, total_items)
        branch = b.scf_if_else(live)
        with branch as (then_ctx, else_ctx):
            with then_ctx:
                ch, sub, _S, hw_lo, hw_len, n_lo, _n_cnt, total = _ro_item(
                    b, p_c, p_n, HW, G["p_nch"], G["p_hwch"], G["p_sh"], G["p_sn"], w_id
                )
                mean_c = b.global_load_f32(Mean, ch)
                invstd_c = b.global_load_f32(InvStd, ch)
                scale_c = b.global_load_f32(Scale, ch)
                xr, dr, oks = [], [], []
                for j in range(R):
                    ok_j, flat = _ro_tile_index(
                        b, spec, tid, j, p_c, HW, ch, hw_lo, hw_len, n_lo, total
                    )
                    xr.append(b.global_load_vN(X, flat, io_ty, VEC))
                    dr.append(b.global_load_vN(dY, flat, io_ty, VEC))
                    oks.append((ok_j, flat))
                s1 = b.const_f32(0.0)
                s2 = b.const_f32(0.0)
                zero_f = b.const_f32(0.0)
                for j in range(R):
                    xs = [b.cast_to_f32(b.vec_extract(xr[j], i)) for i in range(VEC)]
                    dys = [
                        b.select(
                            oks[j][0], b.cast_to_f32(b.vec_extract(dr[j], i)), zero_f
                        )
                        for i in range(VEC)
                    ]
                    prods = [
                        b.fmul(dys[i], b.fmul(b.fsub(xs[i], mean_c), invstd_c))
                        for i in range(VEC)
                    ]
                    s1 = b.fadd(s1, tree_reduce(b, b.fadd, list(dys)))
                    s2 = b.fadd(s2, tree_reduce(b, b.fadd, prods))
                t1, t2 = _ro_block_pair(b, spec, s1, s2, lds, tid)
                a1, a2 = _ro_exchange(
                    b,
                    SyncI,
                    SyncF,
                    Err,
                    tid,
                    lds_bc,
                    ch,
                    sub,
                    S,
                    G["p_epoch"],
                    G["p_spin"],
                    t1,
                    t2,
                )
                owner = b.land(
                    b.cmp_eq(sub, b.const_i32(0)), b.cmp_eq(tid, b.const_i32(0))
                )
                with b.scf_if(owner):
                    b.global_store(dScale, ch, a2)
                    b.global_store(dBias, ch, a1)
                a, k2, k3 = _bwd_coefs(b, a1, a2, scale_c, mean_c, invstd_c, m_f)
                for j in range(R):
                    xs = [b.cast_to_f32(b.vec_extract(xr[j], i)) for i in range(VEC)]
                    dys = [b.cast_to_f32(b.vec_extract(dr[j], i)) for i in range(VEC)]
                    dxs = [
                        b.fsub(
                            b.fsub(
                                b.fmul(a, dys[i]), b.fmul(k2, b.fsub(xs[i], mean_c))
                            ),
                            k3,
                        )
                        for i in range(VEC)
                    ]
                    with b.scf_if(oks[j][0]):
                        store_vec(
                            b,
                            dX,
                            oks[j][1],
                            pack_f32_to(b, dxs, dtype=spec.dtype),
                            n=VEC,
                        )
                b.sync()
            with else_ctx:
                pass
    return b.kernel
