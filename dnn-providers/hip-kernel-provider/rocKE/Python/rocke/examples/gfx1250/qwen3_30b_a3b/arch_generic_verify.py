# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numeric verify for the arch-generic Qwen3 day-0 kernels on gfx1250.

Builds each kernel for gfx1250 (wave32), launches via the HIP runtime, and
compares against a numpy fp32 reference. Covers:

  * add_rmsnorm2d_bf16        (residual add + RMSNorm, bf16)
  * add_rmsnorm2d_rdquant     (add + RMSNorm + dynamic i8 quant + per-row scale)
  * topk_softmax              (router top-K then softmax, E=128/K=8)
  * moe_sorting               (histogram -> scan -> scatter, 3 kernels)
  * smoothquant               (per-channel smooth + dynamic i8 quant)
  * moe_smoothquant           (per-expert smooth + dynamic i8 quant)

Must run on a gfx1250 device:

  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.qwen3_30b_a3b.arch_generic_verify
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import ml_dtypes

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime

ARCH = "gfx1250"
bf16 = ml_dtypes.bfloat16


def _u8(a):
    a = np.ascontiguousarray(a)
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)


def _read(rt, dptr, like):
    buf = (ctypes.c_uint8 * int(like.nbytes))()
    rt.memcpy_d2h(buf, dptr, like.nbytes)
    return np.frombuffer(bytes(buf), dtype=like.dtype).reshape(like.shape)


class _Dev:
    """Tiny RAII-ish helper around Runtime alloc/copy."""

    def __init__(self, rt):
        self.rt = rt
        self.ptrs = []

    def inp(self, arr):
        d = self.rt.alloc(arr.nbytes)
        self.rt.memcpy_h2d(d, _u8(arr), arr.nbytes)
        self.ptrs.append(d)
        return d

    def out(self, arr):
        d = self.rt.alloc(arr.nbytes)
        self.rt.memset(d, 0, arr.nbytes)
        self.ptrs.append(d)
        return d

    def free(self):
        for d in self.ptrs:
            self.rt.free(d)
        self.ptrs = []


def _launch(rt, art, grid, block, packed):
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)
    rt.launch(fn, grid, block, packed)
    rt.sync()
    return module


def verify_add_rmsnorm2d_bf16(rt) -> tuple:
    from rocke.instances.common.add_rmsnorm2d_bf16 import (
        AddRMSNorm2DBF16Spec,
        build_add_rmsnorm2d_bf16,
        add_rmsnorm2d_bf16_grid,
    )

    M, N, eps = 2, 2048, 1e-6
    spec = AddRMSNorm2DBF16Spec(
        n_per_block=N,
        block_size=256,
        vec=4,
        dtype="bf16",
        save_residual=True,
        wave_size=32,
    )
    art = compile_kernel(build_add_rmsnorm2d_bf16(spec, arch=ARCH), arch=ARCH)

    rng = np.random.default_rng(1)
    A = (rng.standard_normal((M, N)) * 0.1).astype(bf16)
    B = (rng.standard_normal((M, N)) * 0.1).astype(bf16)
    Gam = (rng.uniform(0.5, 1.5, N)).astype(bf16)
    Xout = np.zeros((M, N), dtype=bf16)
    Yout = np.zeros((M, N), dtype=bf16)

    d = _Dev(rt)
    aA, aB, aG = d.inp(A), d.inp(B), d.inp(Gam)
    aX, aY = d.out(Xout), d.out(Yout)
    packed = struct.pack("<QQQQQiif", aA, aB, aG, aX, aY, M, N, eps)
    mod = _launch(
        rt, art, add_rmsnorm2d_bf16_grid(M, spec), (spec.block_size, 1, 1), packed
    )
    got_y = _read(rt, aY, Yout).astype(np.float32)
    got_x = _read(rt, aX, Xout).astype(np.float32)
    d.free()
    mod.unload()  # noqa: E702

    x = A.astype(np.float32) + B.astype(np.float32)
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    ref_y = x * inv * Gam.astype(np.float32)[None, :]
    err_y = np.abs(got_y - ref_y).max()
    err_x = np.abs(got_x - x).max()
    ok = err_y <= 4e-2 and err_x <= 4e-2
    return ("add_rmsnorm2d_bf16", ok, f"max|y|={err_y:.3e} max|resid|={err_x:.3e}")


def verify_add_rmsnorm2d_rdquant(rt) -> tuple:
    from rocke.instances.common.add_rmsnorm2d_rdquant import (
        AddRmsnorm2DRdquantSpec,
        build_add_rmsnorm2d_rdquant,
        add_rmsnorm2d_rdquant_grid,
    )

    M, N, eps_rms, eps_q = 2, 2048, 1e-6, 1e-10
    spec = AddRmsnorm2DRdquantSpec(
        n_per_block=N,
        dtype="bf16",
        out_dtype="i8",
        block_size=256,
        vec=4,
        save_residual=True,
        save_yscale=True,
        wave_size=32,
    )
    art = compile_kernel(build_add_rmsnorm2d_rdquant(spec, arch=ARCH), arch=ARCH)

    rng = np.random.default_rng(2)
    A = (rng.standard_normal((M, N)) * 0.1).astype(bf16)
    B = (rng.standard_normal((M, N)) * 0.1).astype(bf16)
    Gam = (rng.uniform(0.5, 1.5, N)).astype(bf16)
    Xout = np.zeros((M, N), dtype=bf16)
    QY = np.zeros((M, N), dtype=np.int8)
    YScale = np.zeros((M,), dtype=np.float32)

    d = _Dev(rt)
    aA, aB, aG = d.inp(A), d.inp(B), d.inp(Gam)
    aX, aQ, aS = d.out(Xout), d.out(QY), d.out(YScale)
    packed = struct.pack("<QQQQQQiiff", aA, aB, aG, aX, aQ, aS, M, N, eps_rms, eps_q)
    mod = _launch(
        rt, art, add_rmsnorm2d_rdquant_grid(M, spec), (spec.block_size, 1, 1), packed
    )
    got_q = _read(rt, aQ, QY).astype(np.int32)
    got_s = _read(rt, aS, YScale)
    d.free()
    mod.unload()  # noqa: E702

    x = A.astype(np.float32) + B.astype(np.float32)
    g = Gam.astype(np.float32)
    inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps_rms)
    xg = x * g[None, :]
    y = xg * inv
    amax = inv * np.abs(xg).max(-1, keepdims=True)
    ref_scale = np.maximum(amax[:, 0], eps_q) / 127.0
    ref_q = np.clip(np.round(y / ref_scale[:, None]), -127, 127).astype(np.int32)
    sc_err = np.abs(got_s - ref_scale).max()
    q_err = np.abs(got_q - ref_q).max()
    ok = sc_err <= 1e-3 * ref_scale.max() and q_err <= 1
    return ("add_rmsnorm2d_rdquant", ok, f"max|dq|={q_err} max|dscale|={sc_err:.3e}")


def verify_topk_softmax(rt) -> tuple:
    from rocke.instances.common.topk_softmax import (
        TopkSoftmaxSpec,
        build_topk_softmax,
        topk_softmax_grid,
    )

    T, E, K = 2, 128, 8
    spec = TopkSoftmaxSpec(
        n_per_row=E, k=K, dtype="f32", out_dtype="f32", block_size=128
    )
    art = compile_kernel(build_topk_softmax(spec, arch=ARCH), arch=ARCH)

    rng = np.random.default_rng(3)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    Y = np.zeros((T, K), dtype=np.float32)
    Idx = np.zeros((T, K), dtype=np.int32)

    d = _Dev(rt)
    aL = d.inp(logits)
    aY, aI = d.out(Y), d.out(Idx)
    packed = struct.pack("<QQQii", aL, aY, aI, T, E)
    mod = _launch(rt, art, topk_softmax_grid(T, spec), (spec.block_size, 1, 1), packed)
    got_y = _read(rt, aY, Y)
    got_i = _read(rt, aI, Idx)
    d.free()
    mod.unload()  # noqa: E702

    ref_i = np.argsort(-logits, axis=-1, kind="stable")[:, :K]
    picked = np.take_along_axis(logits, ref_i, axis=-1)
    e = np.exp(picked - picked.max(-1, keepdims=True))
    ref_y = e / e.sum(-1, keepdims=True)
    idx_ok = bool((got_i == ref_i).all())
    y_err = np.abs(got_y - ref_y).max()
    ok = idx_ok and y_err <= 1e-5
    return ("topk_softmax", ok, f"idx_match={idx_ok} max|w|={y_err:.3e}")


def verify_moe_sorting(rt) -> tuple:
    from rocke.instances.common.moe_sorting import (
        MoeSortingSpec,
        build_moe_sort_histogram,
        build_moe_sort_scan,
        build_moe_sort_scatter,
        moe_sort_histogram_grid,
        moe_sort_scan_grid,
        moe_sort_scatter_grid,
    )

    T, K, E, BS = 2, 8, 128, 128
    spec = MoeSortingSpec(tokens=T, topk=K, experts=E, block_size=BS)
    a_h = compile_kernel(build_moe_sort_histogram(spec, arch=ARCH), arch=ARCH)
    a_s = compile_kernel(build_moe_sort_scan(spec, arch=ARCH), arch=ARCH)
    a_c = compile_kernel(build_moe_sort_scatter(spec, arch=ARCH), arch=ARCH)

    rng = np.random.default_rng(4)
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    wts = rng.uniform(0, 1, size=(T, K)).astype(np.float32)
    P = T * K
    hist = np.zeros((E,), np.int32)
    offs = np.zeros((E,), np.int32)
    counts = np.zeros((E,), np.int32)
    counter = np.zeros((E,), np.int32)
    s_tok = np.full((P,), -1, np.int32)
    s_kid = np.full((P,), -1, np.int32)
    s_wts = np.zeros((P,), np.float32)

    d = _Dev(rt)
    aIds, aW = d.inp(ids), d.inp(wts)
    aHist, aOff, aCnt, aCtr = d.out(hist), d.out(offs), d.out(counts), d.out(counter)
    aSt, aSk, aSw = d.out(s_tok), d.out(s_kid), d.out(s_wts)

    m1 = _launch(
        rt,
        a_h,
        moe_sort_histogram_grid(spec),
        (BS, 1, 1),
        struct.pack("<QQii", aIds, aHist, P, E),
    )
    m2 = _launch(
        rt,
        a_s,
        moe_sort_scan_grid(spec),
        (BS, 1, 1),
        struct.pack("<QQQi", aHist, aOff, aCnt, E),
    )
    m3 = _launch(
        rt,
        a_c,
        moe_sort_scatter_grid(spec),
        (BS, 1, 1),
        struct.pack("<QQQQQQQiii", aIds, aW, aOff, aCtr, aSt, aSk, aSw, T, K, E),
    )
    got_counts = _read(rt, aCnt, counts)
    got_offs = _read(rt, aOff, offs)
    got_tok = _read(rt, aSt, s_tok)
    d.free()
    for m in (m1, m2, m3):
        m.unload()

    ref_counts = np.bincount(ids.reshape(-1), minlength=E).astype(np.int32)
    ref_offs = np.zeros((E,), np.int32)
    ref_offs[1:] = np.cumsum(ref_counts)[:-1]
    counts_ok = bool((got_counts == ref_counts).all())
    offs_ok = bool((got_offs == ref_offs).all())
    # Each expert region must hold exactly the tokens routed to that expert.
    region_ok = True
    flat_tok = ids.reshape(-1)
    for e in range(E):
        beg, cnt = ref_offs[e], ref_counts[e]
        region = sorted(int(x) for x in got_tok[beg : beg + cnt])
        want = sorted(int(p // K) for p in range(P) if flat_tok[p] == e)
        if region != want:
            region_ok = False
            break
    ok = counts_ok and offs_ok and region_ok
    return (
        "moe_sorting",
        ok,
        f"counts={counts_ok} offsets={offs_ok} regions={region_ok} sum={int(got_counts.sum())}",
    )


def verify_smoothquant(rt) -> tuple:
    from rocke.instances.common.smoothquant import (
        SmoothQuantSpec,
        build_smoothquant,
        smoothquant_grid,
    )

    M, N, eps = 2, 2048, 1e-10
    spec = SmoothQuantSpec(
        n_per_block=N,
        dtype="bf16",
        out_dtype="i8",
        block_size=256,
        vec=4,
        save_yscale=True,
        wave_size=32,
    )
    art = compile_kernel(build_smoothquant(spec, arch=ARCH), arch=ARCH)

    rng = np.random.default_rng(5)
    X = (rng.standard_normal((M, N)) * 0.5).astype(bf16)
    Sm = rng.uniform(0.5, 1.5, N).astype(np.float32)
    QY = np.zeros((M, N), np.int8)
    YS = np.zeros((M,), np.float32)

    d = _Dev(rt)
    aX, aSm = d.inp(X), d.inp(Sm)
    aQ, aS = d.out(QY), d.out(YS)
    packed = struct.pack("<QQQQiif", aX, aSm, aQ, aS, M, N, eps)
    mod = _launch(rt, art, smoothquant_grid(M, spec), (spec.block_size, 1, 1), packed)
    got_q = _read(rt, aQ, QY).astype(np.int32)
    got_s = _read(rt, aS, YS)
    d.free()
    mod.unload()  # noqa: E702

    y = X.astype(np.float32) * Sm[None, :]
    amax = np.abs(y).max(-1, keepdims=True)
    ref_s = np.maximum(amax[:, 0], eps) / 127.0
    ref_q = np.clip(np.round(y / ref_s[:, None]), -127, 127).astype(np.int32)
    sc_err = np.abs(got_s - ref_s).max()
    q_err = np.abs(got_q - ref_q).max()
    ok = sc_err <= 1e-3 * ref_s.max() and q_err <= 1
    return ("smoothquant", ok, f"max|dq|={q_err} max|dscale|={sc_err:.3e}")


def verify_moe_smoothquant(rt) -> tuple:
    from rocke.instances.common.moe_smoothquant import (
        MoeSmoothQuantSpec,
        build_moe_smoothquant,
        moe_smoothquant_grid,
    )

    T, N, K, E, eps = 2, 2048, 8, 128, 1e-10
    spec = MoeSmoothQuantSpec(
        n_per_block=N,
        topk=K,
        experts=E,
        dtype="bf16",
        out_dtype="i8",
        block_size=256,
        vec=4,
        save_yscale=True,
        wave_size=32,
        tokens=T,
    )
    art = compile_kernel(build_moe_smoothquant(spec, arch=ARCH), arch=ARCH)

    rng = np.random.default_rng(6)
    X = (rng.standard_normal((T, N)) * 0.5).astype(bf16)
    Sm = rng.uniform(0.5, 1.5, (E, N)).astype(np.float32)
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    QY = np.zeros((K * T, N), np.int8)
    YS = np.zeros((K * T,), np.float32)

    d = _Dev(rt)
    aX, aSm, aId = d.inp(X), d.inp(Sm.reshape(-1)), d.inp(ids)
    aQ, aS = d.out(QY), d.out(YS)
    packed = struct.pack("<QQQQQiif", aX, aSm, aId, aQ, aS, T, N, eps)
    mod = _launch(
        rt, art, moe_smoothquant_grid(T, spec), (spec.block_size, 1, 1), packed
    )
    got_q = _read(rt, aQ, QY).astype(np.int32)
    got_s = _read(rt, aS, YS)
    d.free()
    mod.unload()  # noqa: E702

    q_err = 0
    sc_err = 0.0
    for it in range(T):
        for ik in range(K):
            row = ik * T + it
            e = int(ids[it, ik])
            y = X[it].astype(np.float32) * Sm[e]
            amax = np.abs(y).max()
            rs = max(amax, eps) / 127.0
            rq = np.clip(np.round(y / rs), -127, 127).astype(np.int32)
            q_err = max(q_err, int(np.abs(got_q[row] - rq).max()))
            sc_err = max(sc_err, abs(float(got_s[row]) - rs))
    ok = q_err <= 1 and sc_err <= 1e-3
    return ("moe_smoothquant", ok, f"max|dq|={q_err} max|dscale|={sc_err:.3e}")


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    rt = Runtime()
    checks = [
        verify_add_rmsnorm2d_bf16,
        verify_add_rmsnorm2d_rdquant,
        verify_topk_softmax,
        verify_moe_sorting,
        verify_smoothquant,
        verify_moe_smoothquant,
    ]
    all_ok = True
    for fn in checks:
        try:
            name, ok, detail = fn(rt)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            name, ok, detail = fn.__name__, False, f"EXC {exc}"
        all_ok = all_ok and ok
        print(f"[{ARCH}] {name:24s} {'PASS' if ok else 'FAIL'}  {detail}")
    print(f"[{ARCH}] arch-generic verify: {'ALL PASS' if all_ok else 'FAILURES'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
