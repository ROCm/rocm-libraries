# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Probe kernels for the backward block GEMM and the C-to-A relabel.

Shared by the CPU IR tests and the device numeric tests. Two one-CTA kernels:

* ``gemm``: ``C[M x N] = A[M x K] * B[K x N]`` with operands staged to LDS in a
  chosen storage form and fed to the block GEMM either from LDS (per K step)
  or from registers (all fragments loaded first);
* ``chain``: ``S = Q K^T`` on warp grid ``(1, W)``, then ``dV = P^T dO`` on
  ``(W, 1)`` with ``P^T`` produced by the register relabel of ``S``
  (``P = round(S)`` to the operand dtype) and ``dO`` loaded with the plan's K
  permutation when it needs one.

A third, one-wave kernel probes the MFMA result hazard of the inline-asm
route: one AGPR-form MFMA and a dependent consumer in a single asm block, with
a chosen ``s_nop`` between them (:func:`make_hazard_probe`).

Host buffers are given in the LDS storage form, so staging is a plain copy.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
from rocke.core.ir import F32, IRBuilder, PtrType

from kernels.common._attention_bwd_frag import LdsView, ir_type
from kernels.common._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    kperm_b_operand_coords,
    lane_and_wave,
    relabel_acc_to_a,
)

# ---------------------------------------------------------------------------
# kernel construction
# ---------------------------------------------------------------------------


def _stage(b, src, smem, shape, tid, threads, elem):
    """Copy a row-major ``shape`` tile from global ``src`` into ``smem`` (a 2-D
    allocation or an :class:`LdsView` of a flat one)."""
    rows, cols = shape
    total, vec = rows * cols, 8
    if cols % vec:
        raise ValueError("staged tiles need a multiple-of-8 inner dimension")
    step = threads * vec
    for it in range((total + step - 1) // step):
        idx = b.add(b.mul(tid, b.const_i32(vec)), b.const_i32(it * step))

        def body(idx=idx):
            v = b.global_load_vN(src, idx, elem, vec)
            r = b.div(idx, b.const_i32(cols))
            c = b.mod(idx, b.const_i32(cols))
            if isinstance(smem, LdsView):
                smem.store(b, r, c, v, vec)
            else:
                b.smem_store_vN(smem, [r, c], v, vec)

        if (it + 1) * step <= total:
            body()
        else:
            with b.scf_if(b.cmp_lt(idx, b.const_i32(total))):
                body()


def _store_acc(b, g, acc, lane, origin, out, ld):
    for fm in range(g.frags_m):
        for fn in range(g.frags_n):
            for s, (r, c) in enumerate(g.acc_coords(b, lane, origin, fm, fn)):
                idx = b.add(b.mul(r, b.const_i32(ld)), c)
                b.global_store(out, idx, b.vec_extract(acc[fm][fn], s), align=4)


def _storage_shape(storage, mn, k):
    return (mn, k) if storage == "k_inner" else (k, mn)


def gemm_probe_name(
    arch, dtype, m, n, k, grid, atom_k, a_st, b_st, a_src, b_src, asm_nop=None,
    asm_cluster="k_step", asm_zero=False,
):  # fmt: skip
    tail = "" if asm_nop is None else f"_asm{asm_nop}"
    if asm_cluster != "k_step":
        tail += f"_{asm_cluster}"
    if asm_zero:
        tail += "_z"
    return (
        f"bwd_gemm_probe_{arch}_{dtype}_{m}x{n}x{k}_g{grid[0]}x{grid[1]}_k{atom_k}"
        f"_{a_st[2]}{b_st[2]}_{a_src[0]}{b_src[0]}{tail}"
    )


def make_gemm_probe(
    arch,
    dtype,
    *,
    m,
    n,
    k,
    grid,
    atom_k=16,
    a_storage="k_inner",
    b_storage="k_inner",
    a_src="lds",
    b_src="lds",
    asm_nop=None,
    asm_cluster="k_step",
    asm_zero=False,
    k_slice=None,
    slice_buffers=1,
    slice_first=False,
):
    """Return ``(kernel, geometry)`` of the plain block-GEMM probe.

    ``k_slice`` (keys per slice) issues the intrinsic MMAs through
    :meth:`BlockGemm.mma_k_sliced` with ``slice_buffers`` register buffers
    (``slice_first``: slice 0 read by :meth:`BlockGemm.load_k_slice` before
    the call); same products, same K order per accumulator.

    ``asm_nop`` (an int) issues the MFMAs through the block GEMM's
    ``agpr_asm`` route as AGPR-form inline-asm clusters ending in
    ``s_nop asm_nop``, instead of the intrinsics: one cluster per K step
    (``asm_cluster = "k_step"``) or one for the whole GEMM (``"gemm"``; with a
    single fragment per wave its K steps are separated by ``s_nop asm_nop``
    too). ``asm_zero`` passes no accumulator (the G0 / G2 form of the
    ``s_in_agpr`` asm route: srcC ``0`` on the first K step, no tied input)
    instead of zero accumulators tied to the outputs. The accumulation order
    is the same, so the result is bitwise the intrinsic one when the
    clusters' hazards are covered.
    """
    g = BlockGemm(arch, dtype, m, n, k, tuple(grid), atom_k)
    elem = ir_type(dtype)
    if asm_zero and asm_nop is None:
        raise ValueError("asm_zero is a form of the asm route (needs asm_nop)")
    if k_slice is not None and asm_nop is not None:
        raise ValueError("k_slice issues intrinsics (no asm_nop)")
    name = gemm_probe_name(
        arch, dtype, m, n, k, grid, atom_k, a_storage, b_storage, a_src, b_src,
        asm_nop, asm_cluster, asm_zero,
    )  # fmt: skip
    if k_slice is not None:
        name += f"_ks{k_slice}b{slice_buffers}{'f' if slice_first else ''}"
    b = IRBuilder(name)
    pa = b.param("A", PtrType(elem, "global"), readonly=True)
    pb = b.param("B", PtrType(elem, "global"), readonly=True)
    pc = b.param("C", PtrType(F32, "global"))
    b.kernel.attrs["max_workgroup_size"] = max(64, g.threads)
    tid, lane, wave = lane_and_wave(b, g.wave_size)
    sha, shb = _storage_shape(a_storage, m, k), _storage_shape(b_storage, n, k)
    sa = b.smem_alloc(elem, sha, "sa")
    sb = b.smem_alloc(elem, shb, "sb")
    _stage(b, pa, sa, sha, tid, g.threads, elem)
    _stage(b, pb, sb, shb, tid, g.threads, elem)
    b.sync()
    origin = g.wave_origin(b, wave)
    a_op = LdsOperand(sa, a_storage)
    b_op = LdsOperand(sb, b_storage)
    if a_src == "reg":
        a_op = g.load_all_a(b, a_op, lane, origin[0])
    if b_src == "reg":
        b_op = g.load_all_b(b, b_op, lane, origin[1])
    if k_slice is not None:
        first = None
        if slice_first:
            first = g.load_k_slice(b, a_op, b_op, 0, k_slice, lane=lane, origin=origin)
        acc = g.mma_k_sliced(
            b, g.zero_acc(b), a_op, b_op, k_slice=k_slice, buffers=slice_buffers,
            lane=lane, origin=origin, first=first,
        )  # fmt: skip
    elif asm_nop is None:
        acc = g.mma(b, g.zero_acc(b), a_op, b_op, lane=lane, origin=origin)
    else:
        if a_src == "reg" or b_src == "reg":
            raise ValueError("the asm probe reads its operands from LDS")
        acc = g.mma(
            b, None if asm_zero else g.zero_acc(b), a_op, b_op, lane=lane,
            origin=origin, route="agpr_asm", asm_tail_nop=asm_nop,
            asm_cluster=asm_cluster,
        )  # fmt: skip
    _store_acc(b, g, acc, lane, origin, pc, n)
    b.ret()
    return b.kernel, g


def make_chain_probe(
    arch, dtype, *, k_m0, k_n0, d, waves, atom_k0=16, atom_k1=16, do_storage="k_outer",
    do_reader="plain",
):  # fmt: skip
    """Return ``(kernel, g0, g1, plan)`` of the relabel chain probe.

    ``do_reader = "tr16"`` (gfx950, ``do_storage = "k_outer"``) stages ``dO``
    into a flat LDS view and reads its fragments with the 64-bit transpose
    read: the catalog plan, or the stacked plan of the K-permuted map of a
    16x16x32 consumer (the backward's G1 / G3 B path).
    """
    if do_reader not in ("plain", "tr16"):
        raise ValueError(f"do_reader must be 'plain' or 'tr16', got {do_reader!r}")
    if do_reader == "tr16" and do_storage != "k_outer":
        raise ValueError("the transpose read reads a K-outer dO tile")
    g0 = BlockGemm(arch, dtype, k_m0, k_n0, d, (1, waves), atom_k0)
    g1 = BlockGemm(arch, dtype, k_n0, d, k_m0, (waves, 1), atom_k1)
    elem = ir_type(dtype)
    name = (
        f"bwd_chain_probe_{arch}_{dtype}_m{k_m0}_n{k_n0}_d{d}_w{waves}"
        f"_k{atom_k0}k{atom_k1}_{do_storage[2]}"
    )
    if do_reader != "plain":
        name += f"_{do_reader}"
    b = IRBuilder(name)
    pq = b.param("Q", PtrType(elem, "global"), readonly=True)
    pk = b.param("K", PtrType(elem, "global"), readonly=True)
    pdo = b.param("dO", PtrType(elem, "global"), readonly=True)
    ps = b.param("S", PtrType(F32, "global"))
    pdv = b.param("dV", PtrType(F32, "global"))
    b.kernel.attrs["max_workgroup_size"] = max(64, g0.threads)
    tid, lane, wave = lane_and_wave(b, g0.wave_size)
    sq = b.smem_alloc(elem, (k_m0, d), "sq")
    sk = b.smem_alloc(elem, (k_n0, d), "sk")
    sdo_shape = _storage_shape(do_storage, d, k_m0)
    if do_reader == "tr16":
        flat = b.smem_alloc(elem, (sdo_shape[0] * sdo_shape[1],), "sdo")
        sdo = LdsView(flat, 0, sdo_shape[1], sdo_shape[0])
    else:
        sdo = b.smem_alloc(elem, sdo_shape, "sdo")
    _stage(b, pq, sq, (k_m0, d), tid, g0.threads, elem)
    _stage(b, pk, sk, (k_n0, d), tid, g0.threads, elem)
    _stage(b, pdo, sdo, sdo_shape, tid, g0.threads, elem)
    b.sync()
    o0 = g0.wave_origin(b, wave)
    acc0 = g0.mma(
        b, g0.zero_acc(b), LdsOperand(sq), LdsOperand(sk), lane=lane, origin=o0
    )
    _store_acc(b, g0, acc0, lane, o0, ps, k_n0)
    a1, plan = relabel_acc_to_a(b, g0, g1, acc0, lane)
    coords = kperm_b_operand_coords(plan, g0, g1)
    o1 = g1.wave_origin(b, wave)
    acc1 = g1.mma(
        b,
        g1.zero_acc(b),
        a1,
        LdsOperand(sdo, do_storage, coords=coords, reader=do_reader),
        lane=lane,
        origin=o1,
    )
    _store_acc(b, g1, acc1, lane, o1, pdv, d)
    b.ret()
    return b.kernel, g0, g1, plan


# The MFMA result-hazard probe: one wave, one AGPR-form MFMA on a[0:3], then a
# dependent consumer after ``s_nop nop`` inside the same asm block, so neither
# the hazard recognizer nor the waitcnt pass can put anything between them.
HAZARD_CONSUMERS = ("valu", "srcc")


def _hazard_gap(nop):
    if nop == "safe":  # 32 wait states, beyond every MFMA table entry
        return ["s_nop 15", "s_nop 15"]
    if not -1 <= int(nop) <= 15:
        raise ValueError(f"hazard probe nop must be -1..15 or 'safe', got {nop!r}")
    return [] if nop == -1 else [f"s_nop {int(nop)}"]


def hazard_probe_asm(mnem, consumer, nop):
    """Asm text of the probe: ``$0..$3`` the four reads, ``$4..$7`` the srcC
    init values, ``$8`` / ``$9`` the A / B fragments.

    ``consumer``: ``"valu"`` reads the result with ``v_accvgpr_read`` (a VALU
    read of an MFMA result, the epilogue's access); ``"srcc"`` issues a second
    MFMA with the result as its full-overlap srcC, then reads after 32 wait
    states (the accumulation chain). ``nop``: ``-1`` (nothing between them),
    ``0..15`` or ``"safe"``.
    """
    if consumer not in HAZARD_CONSUMERS:
        raise ValueError(f"consumer must be one of {HAZARD_CONSUMERS}")
    mfma = f"{mnem} a[0:3], $8, $9, a[0:3]"
    lines = [f"v_accvgpr_write_b32 a{i}, ${4 + i}" for i in range(4)]
    lines += ["s_nop 7", mfma]  # VALU write -> MFMA read needs 2 wait states
    lines += _hazard_gap(nop)
    if consumer == "srcc":
        lines += [mfma] + _hazard_gap("safe")
    lines += [f"v_accvgpr_read_b32 ${i}, a{i}" for i in range(4)]
    return "\n\t".join(lines)


def hazard_probe_name(arch, dtype, atom_k, consumer, nop):
    tag = "ref" if nop is None else ("none" if nop == -1 else f"n{nop}")
    return f"bwd_mfma_hazard_probe_{arch}_{dtype}_k{atom_k}_{consumer}_{tag}"


def make_hazard_probe(arch, dtype, *, atom_k=16, consumer="valu", nop=None):
    """Return ``(kernel, op)`` of the one-wave MFMA result-hazard probe.

    ``nop=None`` builds the reference: the same MFMA (twice for ``"srcc"``) as
    an intrinsic, so the backend inserts the ISA waits. Otherwise the asm of
    :func:`hazard_probe_asm`. Inputs per lane: the A and B fragments and four
    fp32 srcC values; output per lane: the four result values.
    """
    from rocke.core.ir import I32, VectorType

    from kernels.common._attention_bwd_caps import mfma_asm_mnemonic, select_bwd_atom

    op = select_bwd_atom(arch, dtype, atom_k)
    elem = ir_type(dtype)
    b = IRBuilder(hazard_probe_name(arch, dtype, atom_k, consumer, nop))
    pa = b.param("A", PtrType(elem, "global"), readonly=True)
    pb = b.param("B", PtrType(elem, "global"), readonly=True)
    pc = b.param("C", PtrType(F32, "global"), readonly=True)
    po = b.param("Out", PtrType(F32, "global"))
    b.kernel.attrs["max_workgroup_size"] = 64
    lane = b.thread_id_x()
    fa, fb = op.a_frag_len, op.b_frag_len
    av = b.global_load_vN(pa, b.mul(lane, b.const_i32(fa)), elem, fa)
    bv = b.global_load_vN(pb, b.mul(lane, b.const_i32(fb)), elem, fb)
    cv = b.global_load_vN(pc, b.mul(lane, b.const_i32(4)), F32, 4)
    if nop is None:
        r = b.mma(op, av, bv, cv)
        if consumer == "srcc":
            r = b.mma(op, av, bv, r)
        outs = [b.vec_extract(r, i) for i in range(4)]
    else:
        src = [
            b.vec_bitcast(av, VectorType(I32, fa // 2)),
            b.vec_bitcast(bv, VectorType(I32, fb // 2)),
        ]
        outs = b.inline_asm_multi(
            hazard_probe_asm(mfma_asm_mnemonic(arch, dtype, atom_k), consumer, nop),
            ",".join(["=&v"] * 4 + ["v"] * 6 + [f"~{{a{i}}}" for i in range(4)]),
            [b.vec_extract(cv, i) for i in range(4)] + src,
            result_types=[F32] * 4,
            sideeffect=True,
            convergent=True,
            result_name_hint="hz",
        )
    for i, v in enumerate(outs):
        idx = b.add(b.mul(lane, b.const_i32(4)), b.const_i32(i))
        b.global_store(po, idx, v, align=4)
    b.ret()
    return b.kernel, op


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def enc(x, dtype):
    """float32 -> raw fp16 / bf16 storage (round to nearest even)."""
    x = np.asarray(x, np.float32)
    if dtype == "fp16":
        return x.astype(np.float16)
    u = x.view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def dec(u, dtype):
    if dtype == "fp16":
        return np.asarray(u).astype(np.float32)
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def _rand(rng, shape, dtype):
    raw = enc(rng.standard_normal(shape).astype(np.float32) * 0.5, dtype)
    return raw, dec(raw, dtype).astype(np.float64)


def _launch(arch, kernel, threads, inputs, outputs):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)

    host = [np.ascontiguousarray(a) for a in inputs] + [
        np.zeros(n, np.float32) for n in outputs
    ]
    dev = []
    for a in host:
        p = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(p, u8(a), a.nbytes)
        dev.append(p)
    packed = b"".join(struct.pack("<Q", p) for p in dev)
    rt.launch(fn, (1, 1, 1), (threads, 1, 1), packed)
    rt.sync()
    outs = host[len(inputs) :]
    for a, p in zip(outs, dev[len(inputs) :]):
        rt.memcpy_d2h(u8(a), p, a.nbytes)
    for p in dev:
        rt.free(p)
    mod.unload()
    return outs


def run_gemm_probe(arch, dtype, *, seed=0, **kw):
    """Launch the GEMM probe; returns ``(got, ref)`` as float64 ``[M, N]``."""
    kernel, g = make_gemm_probe(arch, dtype, **kw)
    rng = np.random.default_rng(seed)
    a_raw, a = _rand(rng, (g.m, g.k), dtype)
    b_raw, bm = _rand(rng, (g.k, g.n), dtype)
    a_store = a_raw if kw.get("a_storage", "k_inner") == "k_inner" else a_raw.T
    b_store = b_raw.T if kw.get("b_storage", "k_inner") == "k_inner" else b_raw
    (c,) = _launch(arch, kernel, g.threads, [a_store, b_store], [g.m * g.n])
    return c.reshape(g.m, g.n).astype(np.float64), a @ bm


def hazard_probe_inputs(arch, dtype, *, atom_k=16, seed=0):
    """Host inputs ``[A, B, C]`` of the hazard probe (C away from A * B)."""
    from kernels.common._attention_bwd_caps import select_bwd_atom

    op = select_bwd_atom(arch, dtype, atom_k)
    rng = np.random.default_rng(seed)
    a_raw, _ = _rand(rng, (64 * op.a_frag_len,), dtype)
    b_raw, _ = _rand(rng, (64 * op.b_frag_len,), dtype)
    c = (rng.standard_normal(64 * 4).astype(np.float32) * 4.0) + 8.0
    return [a_raw, b_raw, c]


def run_hazard_probe(arch, dtype, *, atom_k=16, consumer="valu", nop=None, seed=0):
    """Launch the hazard probe; returns the ``[64, 4]`` fp32 lane results."""
    kernel, _ = make_hazard_probe(
        arch, dtype, atom_k=atom_k, consumer=consumer, nop=nop
    )
    ins = hazard_probe_inputs(arch, dtype, atom_k=atom_k, seed=seed)
    (out,) = _launch(arch, kernel, 64, ins, [64 * 4])
    return out.reshape(64, 4)


def run_chain_probe(arch, dtype, *, seed=0, **kw):
    """Launch the chain probe.

    Returns ``(s_got, s_ref, dv_got, dv_ref)``; ``dv_ref`` uses the device ``S``
    rounded to the operand dtype, so it isolates the relabel and the K
    permutation from the rounding of ``S``.
    """
    kernel, g0, g1, _ = make_chain_probe(arch, dtype, **kw)
    rng = np.random.default_rng(seed)
    q_raw, q = _rand(rng, (g0.m, g0.k), dtype)
    k_raw, kk = _rand(rng, (g0.n, g0.k), dtype)
    do_raw, do = _rand(rng, (g1.k, g1.n), dtype)
    do_store = do_raw if kw.get("do_storage", "k_outer") == "k_outer" else do_raw.T
    s, dv = _launch(
        arch,
        kernel,
        g0.threads,
        [q_raw, k_raw, do_store],
        [g0.m * g0.n, g1.m * g1.n],
    )
    s = s.reshape(g0.m, g0.n).astype(np.float64)
    p = dec(enc(s.astype(np.float32), dtype), dtype).astype(np.float64)
    return s, q @ kk.T, dv.reshape(g1.m, g1.n).astype(np.float64), p.T @ do
