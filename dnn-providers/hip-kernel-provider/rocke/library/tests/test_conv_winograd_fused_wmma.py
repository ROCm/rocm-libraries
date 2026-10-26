# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the fused weight-stationary Winograd forward conv
(``kernels/common/conv_winograd_fused_wmma.py``).

CPU: spec validation, LDS / VGPR budgets, the host-side transfer bounds
simulation and IR builds of every knob.

GPU (gfx1250 + torch): K1 (filter transform) + the fused kernel on small
shapes against a float32 direct conv computed on the CPU, metric
max|out - ref| / max(max|ref|, 1) < 5e-2. U and y are poisoned (bytes 0x7F)
before every run, so an element the kernels fail to write shows up. Every
host <-> device copy goes through pinned memory with the device drained.
"""

from __future__ import annotations

import ctypes
import importlib.util

import pytest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fused_wmma import (
    WinogradFusedSpec,
    build_winograd_fused,
    fused_bounds_check,
    fused_output_coverage_check,
    fused_reason,
)


def _prob(**kw) -> ConvProblem:
    base = dict(N=1, Hi=8, Wi=8, C=32, K=32, Y=3, X=3, pH=1, pW=1)
    base.update(kw)
    return ConvProblem(**base)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(m=3), "m must be"),
        (dict(problem=_prob(Y=1, X=1, pH=0, pW=0)), "3x3"),
        (dict(problem=_prob(sH=2, sW=2)), "3x3 stride-1"),
        (dict(problem=_prob(C=12)), "multiples of 8"),
        (dict(io_dtype="bf16", operand_dtype="bf16", m=4), "bf16 operands"),
        (dict(io_dtype="bf16", m=4), "prescale"),
        (dict(ck=48), "ck must"),
        (dict(stages=1), "stages"),
        (dict(fm=3), "fm in"),
        (dict(arith="x"), "arith"),
        (dict(io_dtype="bf16", operand_dtype="fp16"), "arith='f32'"),
        (dict(tdm="x"), "tdm must"),
        (dict(tdm="producer"), "pipe='early'"),
        (dict(bh=2), "bh > 1 needs"),
        (dict(bh=3, tdm="3d"), "bh must"),
        (dict(cs=3), "cs must"),
        (dict(pipe="x"), "pipe must"),
    ],
)
def test_spec_rejects(kw, msg):
    kw = dict(kw)
    prob = kw.pop("problem", _prob())
    with pytest.raises(ValueError, match=msg):
        WinogradFusedSpec(problem=prob, **kw)


def test_budgets():
    s = WinogradFusedSpec(problem=_prob(C=128, K=128, Hi=64, Wi=64), m=2)
    assert s.waves == 8 and s.kb == 64 and s.tw == 16
    assert s.lds_bytes() <= 327680
    assert fused_reason(s) is None
    # U of 4 bins x 8 fragments x 8 k-steps does not fit 512 VGPRs
    big = WinogradFusedSpec(problem=_prob(C=512, K=128, Hi=64, Wi=64), m=2, fn=2)
    assert "VGPR" in (fused_reason(big) or "")


_BUILD = [
    dict(m=2),
    dict(m=2, fm=2, fn=1),
    dict(m=2, ck=64, stages=3),
    dict(m=2, ck=128, stages=2, fn=4, ks=1),
    dict(m=2, arith="f32"),
    dict(m=2, prefetch=False, nt_stores=False, setprio=False),
    dict(m=2, io_dtype="bf16", operand_dtype="bf16"),
    dict(m=2, io_dtype="bf16", operand_dtype="fp16", arith="f32"),
    dict(m=4, fn=1, ks=1, prefetch=False),
    dict(m=4, fn=1, ks=1, arith="f32", prefetch=False),
    dict(m=2, pipe="early"),
    dict(m=2, pipe="early", fn=4, ks=1, ck=128),
    dict(m=2, pipe="early", cs=2, fn=2, ks=1, ck=64),
    dict(m=2, pipe="early", tdm="3d", stages=3),
    dict(m=2, pipe="early", tdm="3d", bh=2, fn=4, ks=1),
    dict(m=2, pipe="early", tdm="producer", fn=2, ks=1),
    dict(m=2, pipe="early", tdm="producer", bh=4, fn=2, ks=1),
    dict(m=2, pipe="early", dpp_probe=True),
    dict(m=4, pipe="early", tdm="3d", fn=1, ks=1, prefetch=False),
]


@pytest.mark.parametrize(
    "kw", _BUILD, ids=lambda k: "_".join(f"{a}{v}" for a, v in k.items())
)
def test_ir_builds(kw):
    built = 0
    for prob in (
        _prob(C=64, K=48, Hi=13, Wi=21),
        _prob(C=40, K=24, Hi=11, Wi=11, pH=2, pW=2, dH=2, dW=2),
        _prob(N=2, C=64, K=48, Hi=10, Wi=21, groups=2),
    ):
        try:
            spec = WinogradFusedSpec(problem=prob, **kw)
        except ValueError as e:  # e.g. cs not dividing the k-steps of C/g
            assert "cs must" in str(e)
            continue
        fused_bounds_check(spec)
        k = build_winograd_fused(spec)
        assert k is not None
        built += 1
    assert built


@pytest.mark.parametrize(
    "prob",
    [
        _prob(N=2, Hi=9, Wi=7, pH=0, pW=0),
        _prob(Hi=13, Wi=37, C=96, K=40),
        _prob(Hi=11, Wi=11, C=40, K=24, pH=2, pW=2, dH=2, dW=2),
        _prob(Hi=40, Wi=67, pH=3, pW=3),
        _prob(N=3, Hi=17, Wi=130, C=64, K=64, pH=1, pW=1, dH=1, dW=3),
    ],
)
def test_bounds_simulation(prob):
    for m in (2, 4):
        for fm in (1, 2):
            for kw in (
                {},
                dict(pipe="early", tdm="3d"),
                dict(pipe="early", tdm="3d", bh=2),
                dict(pipe="early", tdm="producer", bh=4),
            ):
                spec = WinogradFusedSpec(problem=prob, m=m, fm=fm, fn=1, ks=1, **kw)
                fused_bounds_check(spec)
                fused_output_coverage_check(spec)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _gpu_arch():
    if not _HAS_TORCH:
        return None
    import torch

    torch.cuda.is_available()  # let torch claim the HIP context first
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0)
    except Exception:  # noqa: BLE001
        return None


_GPU = pytest.mark.skipif(
    _gpu_arch() != "gfx1250", reason="needs torch and a gfx1250 GPU"
)

_TOL = 5e-2

# (id, problem kwargs, m, io, operand, knobs)
_GPU_CASES = [
    ("f2_fp16_p1", dict(N=2, Hi=16, Wi=16, C=32, K=32), 2, "fp16", "fp16", {}),
    (
        "f2_fp16_p0_odd",
        dict(N=2, Hi=9, Wi=7, C=32, K=48, pH=0, pW=0),
        2,
        "fp16",
        "fp16",
        {},
    ),
    ("f2_ctail_ktail", dict(Hi=13, Wi=37, C=40, K=24), 2, "fp16", "fp16", {}),
    (
        "f2_multichunk",
        dict(Hi=13, Wi=37, C=96, K=40),
        2,
        "fp16",
        "fp16",
        dict(ck=32, stages=3),
    ),
    (
        "f2_ck128_fn4",
        dict(Hi=20, Wi=40, C=128, K=64),
        2,
        "fp16",
        "fp16",
        dict(ck=128, fn=4, ks=1),
    ),
    (
        "f2_dil2_p2",
        dict(Hi=11, Wi=11, C=40, K=24, pH=2, pW=2, dH=2, dW=2),
        2,
        "fp16",
        "fp16",
        {},
    ),
    (
        "f2_groups2",
        dict(N=2, Hi=10, Wi=21, C=64, K=48, groups=2),
        2,
        "fp16",
        "fp16",
        {},
    ),
    ("f2_bf16_opbf16", dict(Hi=12, Wi=40, C=48, K=32), 2, "bf16", "bf16", {}),
    (
        "f2_bf16_op16",
        dict(Hi=12, Wi=40, C=48, K=32),
        2,
        "bf16",
        "fp16",
        dict(arith="f32"),
    ),
    (
        "f2_f32arith_fm2",
        dict(Hi=12, Wi=70, C=32, K=32),
        2,
        "fp16",
        "fp16",
        dict(arith="f32", fm=2, fn=1),
    ),
    (
        "f4_fp16",
        dict(N=2, Hi=13, Wi=21, C=32, K=32),
        4,
        "fp16",
        "fp16",
        dict(fn=1, ks=1, prefetch=False),
    ),
    (
        "f4_fp16_dil2",
        dict(Hi=19, Wi=23, C=32, K=16, pH=2, pW=2, dH=2, dW=2),
        4,
        "fp16",
        "fp16",
        dict(fn=1, ks=1, prefetch=False),
    ),
    (
        "f2_early_cs2",
        dict(Hi=13, Wi=37, C=128, K=40),
        2,
        "fp16",
        "fp16",
        dict(pipe="early", cs=2, fn=1, ks=1, ck=64, stages=3),
    ),
    (
        "f2_early_3d",
        dict(Hi=13, Wi=37, C=96, K=40),
        2,
        "fp16",
        "fp16",
        dict(pipe="early", tdm="3d", ck=32, stages=3),
    ),
    (
        "f2_early_3d_bh2_dil",
        dict(N=2, Hi=19, Wi=23, C=64, K=48, pH=2, pW=2, dH=2, dW=2),
        2,
        "fp16",
        "fp16",
        dict(pipe="early", tdm="3d", bh=2, fn=4, ks=1, ck=64),
    ),
    (
        "f2_early_producer",
        dict(Hi=13, Wi=37, C=96, K=40),
        2,
        "fp16",
        "fp16",
        dict(pipe="early", tdm="producer", fn=2, ks=1, ck=32, stages=3),
    ),
    (
        "f2_early_producer_bh4_bf16",
        dict(N=2, Hi=16, Wi=16, C=64, K=64),
        2,
        "bf16",
        "bf16",
        dict(pipe="early", tdm="producer", bh=4, fn=2, ks=1, ck=64, stages=3),
    ),
]


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run_gpu(prob, m, io, op, knobs):
    import torch
    import torch.nn.functional as F

    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from kernels.common import conv_winograd_fwd as W
    from kernels.common.conv_winograd_fused_wmma import (
        fused_launch_values,
        fused_signature,
    )

    spec = WinogradFusedSpec(problem=prob, m=m, io_dtype=io, operand_dtype=op, **knobs)
    assert fused_reason(spec) is None
    fused_bounds_check(spec)
    import dataclasses

    k1p = dataclasses.replace(prob, N=1, Hi=8, Wi=8, pH=1, pW=1, dH=1, dW=1)
    xs = W.WinogradFwdSpec(problem=k1p, m=m, io_dtype=io, operand_dtype=op, vec=4)
    k1 = compile_kernel(W.build_winograd_filter_transform(xs), arch="gfx1250")
    kf = compile_kernel(build_winograd_fused(spec), arch="gfx1250")

    tdt = torch.float16 if io == "fp16" else torch.bfloat16
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(prob.N, prob.Hi, prob.Wi, prob.C, generator=g) * 2 - 1).to(tdt)
    w = (torch.rand(prob.K, 3, 3, prob.cpg, generator=g) * 2 - 1).to(tdt)
    ref = F.conv2d(
        x.float().permute(0, 3, 1, 2),
        w.float().permute(0, 3, 1, 2),
        padding=(prob.pH, prob.pW),
        dilation=(prob.dH, prob.dW),
        groups=prob.groups,
    ).permute(0, 2, 3, 1)

    rt = Runtime()
    sz = spec.buffer_bytes()
    sz["w"] = w.numel() * 2
    bufs = {k: int(rt.alloc(v)) for k, v in sz.items()}
    try:
        rt.sync()
        for key, t in (("x", x), ("w", w)):
            h = t.contiguous().pin_memory()
            rt.memcpy_h2d(bufs[key], _u8(h), h.nbytes)
            rt.sync()
        rt.memset(bufs["U"], 0x7F, sz["U"])
        rt.memset(bufs["y"], 0x7F, sz["y"])
        rt.sync()
        grid1 = W.filter_transform_grid(xs)
        assert min(grid1) > 0
        KernelLauncher(
            hsaco=k1.hsaco,
            kernel_name=k1.kernel_name,
            signature=W.filter_transform_signature(xs),
        )(
            dict(w=bufs["w"], u=bufs["U"], w_bytes=sz["w"], u_bytes=sz["U"]),
            config=LaunchConfig(grid=grid1, block=(W.filter_transform_block(xs), 1, 1)),
        )
        rt.sync()
        P = spec.default_wgs(256)
        grid = spec.grid(P)
        assert min(grid) > 0
        KernelLauncher(
            hsaco=kf.hsaco, kernel_name=kf.kernel_name, signature=fused_signature(spec)
        )(
            fused_launch_values(spec, bufs["x"], bufs["U"], bufs["y"], P),
            config=LaunchConfig(grid=grid, block=(spec.threads, 1, 1)),
        )
        rt.sync()
        out = torch.empty(prob.N, prob.Ho, prob.Wo, prob.K, dtype=tdt).pin_memory()
        rt.memcpy_d2h(_u8(out), bufs["y"], out.nbytes)
        rt.sync()
    finally:
        synchronize_and_release(0)
        for p in bufs.values():
            rt.free(p)
    d = (out.float() - ref).abs()
    err = float(d.max() / ref.abs().max().clamp(min=1.0))
    return err if err == err else float("inf")


@_GPU
@pytest.mark.parametrize(
    "cid,pkw,m,io,op,knobs", _GPU_CASES, ids=[c[0] for c in _GPU_CASES]
)
def test_fused_gpu(cid, pkw, m, io, op, knobs):
    base = dict(N=1, Y=3, X=3, pH=1, pW=1)
    base.update(pkw)
    err = _run_gpu(ConvProblem(**base), m, io, op, knobs)
    assert err < _TOL, f"{cid}: err {err:.3e}"
