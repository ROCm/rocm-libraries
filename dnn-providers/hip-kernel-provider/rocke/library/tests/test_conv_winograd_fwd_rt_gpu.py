# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric test of the AOT Winograd forward route: the shape-generic
transform binaries of :mod:`kernels.common.conv_winograd_fwd_rt` through
``dispatch_conv_grouped`` (pinned ``algorithm="winograd_aot"``) ->
``launch_plan_for`` -> compile every step -> launch the steps in order ->
compare y with the conv reference (max|out - ref| / max(max|ref|, 1) < 5e-2,
the conv benchmarks' metric).

The binaries are compiled once per kernel name and reused across the cases --
the point of the shape-generic kernels. Besides the default pick, the cases
force the other binaries of the AOT grid (register runs of 2 tiles, the LDS
region, the split prescale, 1-wide channel vectors) on the same pipeline.

Workspaces and y are poisoned (bytes 0x7F) first, so an element no kernel
writes shows up. Requires a gfx1250 GPU and torch.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()  # let torch claim the HIP context first

from rocke.runtime.hip_module import get_device_arch  # noqa: E402

GPU_ARCH = get_device_arch(0)

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH or GPU_ARCH != "gfx1250",
    reason="needs torch and a gfx1250 GPU",
)

_TOL = 5e-2
_ARTIFACTS: dict = {}


def _u8(t):
    import ctypes

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _artifact(step):
    """One compile per kernel name: a shape-generic binary serves every case."""
    from rocke import compile_kernel

    name = step.kernel.name
    if name.startswith("K3"):
        name = f"K3:{step.signature!r}:{step.block}"  # the GEMM config's binary
    if name not in _ARTIFACTS:
        _ARTIFACTS[name] = compile_kernel(step.kernel.build(GPU_ARCH), arch=GPU_ARCH)
    return _ARTIFACTS[name]


def _run(req, spec):
    """Run ``spec`` (an AOT Winograd plan selected for ``req``); ``err``."""
    from benchmarks.common.benchmark_winograd_conv import (
        make_inputs,
        reference,
        verify_error,
    )
    from dispatch.grouped_convolution import launch_plan_for
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    problem, dtype = spec.problem, spec.dtype
    plan = launch_plan_for(req, spec)
    launchers = []
    for step in plan.steps:
        art = _artifact(step)
        launchers.append(
            KernelLauncher(
                hsaco=art.hsaco, kernel_name=art.kernel_name, signature=step.signature
            )
        )
    x_t, w_t, y_t = make_inputs(problem, dtype, seed=7)
    ref = reference(x_t, w_t, problem, dtype)
    sizes = {"x": x_t.nbytes, "w": w_t.nbytes, "y": y_t.nbytes, **plan.workspaces}
    rt = Runtime()
    ptrs = {k: int(rt.alloc(max(int(v), 4))) for k, v in sizes.items()}
    try:
        rt.sync()
        for key, t in (("x", x_t), ("w", w_t)):
            h = t.contiguous().pin_memory()
            rt.memcpy_h2d(ptrs[key], _u8(h), h.nbytes)
            rt.sync()
            del h
        for key in (*plan.workspaces, "y"):
            rt.memset(ptrs[key], 0x7F, sizes[key])
        rt.sync()
        for step, lau in zip(plan.steps, launchers):
            lau(
                plan.values(step, ptrs, sizes),
                config=LaunchConfig(grid=step.grid, block=step.block, fence=True),
            )
        rt.sync()
        out = torch.empty_like(y_t, device="cpu").pin_memory()
        rt.memcpy_d2h(_u8(out), ptrs["y"], out.nbytes)
        rt.sync()
    finally:
        rt.sync()
        for p in ptrs.values():
            rt.free(p)
    err, _ = verify_error(out, ref)
    return err


def _request(dtype="fp16", **kw):
    from dispatch.grouped_convolution import ConvGroupedRequest
    from kernels.conv.kernel_identity import current_llvm_flavor

    return ConvGroupedRequest(
        arch=GPU_ARCH,
        algorithm="winograd_aot",
        llvm_flavor=current_llvm_flavor(),
        dtype=dtype,
        **kw,
    )


def _plans(req, tiles=None):
    from dispatch.grouped_convolution import conv_grouped_sweep_space

    out = [s for s in conv_grouped_sweep_space(req) if s.plan.kind == "unfused2d_rt"]
    if tiles is not None:
        out = [s for s in out if s.plan.tile in tiles]
    assert out, "no AOT plan"
    return out


_CASES = [
    ("fp16", dict(N=2, C=64, K=96, Hi=20, Wi=24, Y=3, X=3, pad_h=1, pad_w=1)),
    ("bf16", dict(N=2, C=64, K=96, Hi=20, Wi=24, Y=3, X=3, pad_h=1, pad_w=1)),
    (
        "fp16",
        dict(
            N=1,
            C=128,
            K=64,
            Hi=40,
            Wi=40,
            Y=3,
            X=3,
            pad_h=2,
            pad_w=2,
            dilation_h=2,
            dilation_w=2,
        ),
    ),  # fmt: skip
    ("bf16", dict(N=2, C=128, K=128, Hi=16, Wi=16, Y=3, X=3, pad_h=1, pad_w=1, G=4)),
    ("fp16", dict(N=1, C=64, K=64, Hi=17, Wi=23, Y=3, X=3, pad_h=0, pad_w=2)),
    # bf16 wide C: the fused prescale's max over more than one wave of channel
    # lanes, at blocks 256 / 512 / 1024, and the split prescale past them.
    ("bf16", dict(N=1, C=256, K=64, Hi=12, Wi=12, Y=3, X=3, pad_h=1, pad_w=1)),
    ("bf16", dict(N=1, C=2048, K=64, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1)),
    ("bf16", dict(N=1, C=4096, K=32, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1)),
]


@pytest.mark.parametrize("dtype,geom", _CASES)
def test_aot_route_every_tile(dtype, geom):
    req = _request(dtype, **geom)
    for spec in _plans(req):
        err = _run(req, spec)
        assert err < _TOL, f"{spec.name}: err {err:.3e}"


def test_aot_selection_fp16_and_bf16():
    from dispatch.grouped_convolution import dispatch_conv_grouped

    for dtype in ("fp16", "bf16"):
        req = _request(dtype, N=4, C=96, K=64, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1)
        res = dispatch_conv_grouped(req)
        assert res.spec.plan.kind == "unfused2d_rt"
        assert _run(req, res.spec) < _TOL


def test_batch_split():
    from dispatch.conv_winograd_plan import winograd_steps

    req = _request(N=4, C=64, K=64, Hi=12, Wi=12, Y=3, X=3, pad_h=1, pad_w=1)
    spec = _plans(req, tiles={(2, 2)})[0]
    split = dataclasses.replace(spec, plan=dataclasses.replace(spec.plan, chunks=2))
    winograd_steps(split.problem, split.dtype, split.plan)  # builds the plan
    assert _run(req, split) < _TOL


_VARIANTS = {
    "register_r2": lambda s, p: dataclasses.replace(
        s, tiles_per_thread=2, lds=(0, 0), block_size=256
    ),
    "lds": lambda s, p: dataclasses.replace(
        s, tiles_per_thread=4, lds=(4, 8), lds_lanes=16
    ),
    "vec1": lambda s, p: dataclasses.replace(s, vec=1),
}


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
@pytest.mark.parametrize("tile", [(2, 2), (4, 2)])
def test_forced_input_variants(monkeypatch, variant, tile):
    """The same pipeline with the K2 binary swapped for another grid member."""
    from dispatch.conv_winograd_plan import winograd_steps
    from kernels.common import conv_winograd_fwd_rt as R

    pick = R.default_stage_specs

    def forced(problem, t, io):
        specs = dict(pick(problem, t, io))
        specs["K2"] = _VARIANTS[variant](specs["K2"], problem)
        if variant == "vec1":
            specs["K1"] = dataclasses.replace(specs["K1"], vec=1)
            specs["K4"] = dataclasses.replace(specs["K4"], vec=1)
        return specs

    monkeypatch.setattr(R, "default_stage_specs", forced)
    winograd_steps.cache_clear()
    try:
        req = _request(N=2, C=64, K=64, Hi=26, Wi=30, Y=3, X=3, pad_h=1, pad_w=1)
        spec = _plans(req, tiles={tile})[0]
        assert _run(req, spec) < _TOL
    finally:
        winograd_steps.cache_clear()


def test_split_prescale(monkeypatch):
    """bf16 with the per-pixel max pass (K2a) instead of the fused prescale."""
    from dispatch.conv_winograd_plan import winograd_steps
    from kernels.common import conv_winograd_fwd_rt as R

    pick = R.default_stage_specs

    def split(problem, t, io):
        specs = dict(pick(problem, t, io))
        k2 = specs["K2"]
        if k2.prescale:
            specs["K2"] = dataclasses.replace(k2, fused=False, block_size=256)
            specs["K2a"] = R.RtPrescaleSpec(io, 8)
        return specs

    monkeypatch.setattr(R, "default_stage_specs", split)
    winograd_steps.cache_clear()
    try:
        req = _request(
            "bf16", N=2, C=64, K=96, Hi=20, Wi=24, Y=3, X=3, pad_h=1, pad_w=1
        )
        for spec in _plans(req, tiles={(4, 4), (6, 4)}):
            assert _run(req, spec) < _TOL, spec.name
    finally:
        winograd_steps.cache_clear()
