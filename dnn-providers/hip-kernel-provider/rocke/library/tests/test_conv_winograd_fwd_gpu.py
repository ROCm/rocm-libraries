# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric tests of the unfused Winograd F(m,3) forward pipeline.

Runs K1 (filter transform) -> K2 (input transform) -> K3 (grouped 1x1
implicit-GEMM conv, built on a non-pointwise probe and launched on the real
grouped 1x1 problem) -> K4 (output transform) on small shapes and compares y
with a float32 direct-conv reference, using the conv benchmarks' metric
max|out - ref| / max(max|ref|, 1) < 5e-2.

Every workspace and y are poisoned (bytes 0x7F) before the run, so an element
a kernel fails to write shows up as NaN / a huge value.

Requires a gfx1250 GPU and torch.
"""

from __future__ import annotations

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

_K3_CFG = (32, 32, 32, 1, 1, 16, "mem", "cshuffle")
_TOL = 5e-2

# (id, problem kwargs, m, io dtype, operand dtype, prescale, vec, x scale)
_CASES = [
    (
        "f2_fp16_p1",
        dict(N=2, Hi=8, Wi=8, C=16, K=24, pH=1, pW=1),
        2,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f4_fp16_p1",
        dict(N=2, Hi=8, Wi=8, C=16, K=24, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f2_fp16_p0_odd",
        dict(N=1, Hi=9, Wi=7, C=16, K=16),
        2,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f4_fp16_p0_odd",
        dict(N=1, Hi=11, Wi=13, C=16, K=16),
        4,
        "fp16",
        "fp16",
        False,
        4,
        1.0,
    ),
    (
        "f4_bf16op16_ps",
        dict(N=2, Hi=7, Wi=9, C=16, K=16, pH=1, pW=1),
        4,
        "bf16",
        "fp16",
        True,
        4,
        1.0,
    ),
    (
        "f2_bf16op16",
        dict(N=1, Hi=10, Wi=10, C=32, K=16, pH=1, pW=1),
        2,
        "bf16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f2_bf16op_bf16",
        dict(N=2, Hi=7, Wi=9, C=16, K=16, pH=1, pW=1),
        2,
        "bf16",
        "bf16",
        False,
        8,
        1.0,
    ),
    (
        "f4_dil2",
        dict(N=1, Hi=13, Wi=11, C=16, K=16, pH=2, pW=2, dH=2, dW=2),
        4,
        "fp16",
        "fp16",
        False,
        4,
        1.0,
    ),
    (
        "f2_dil2_bf16",
        dict(N=1, Hi=13, Wi=11, C=16, K=8, pH=2, pW=2, dH=2, dW=2),
        2,
        "bf16",
        "bf16",
        False,
        8,
        1.0,
    ),
    (
        "f2_g2",
        dict(N=2, Hi=8, Wi=8, C=32, K=16, pH=1, pW=1, groups=2),
        2,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f4_g2_bf16",
        dict(N=1, Hi=9, Wi=9, C=24, K=32, pH=1, pW=1, groups=2),
        4,
        "bf16",
        "fp16",
        True,
        4,
        1.0,
    ),
    (
        "f4_c12_vec4",
        dict(N=1, Hi=8, Wi=9, C=12, K=20, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f2_c6_vec2",
        dict(N=1, Hi=7, Wi=7, C=6, K=10, pH=1, pW=1),
        2,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
    ),
    (
        "f2_c5_vec1",
        dict(N=1, Hi=6, Wi=5, C=5, K=7, pH=1, pW=1),
        2,
        "bf16",
        "bf16",
        False,
        8,
        1.0,
    ),
    (
        "f2_fp16_ps",
        dict(N=1, Hi=8, Wi=8, C=16, K=16, pH=1, pW=1),
        2,
        "fp16",
        "fp16",
        True,
        8,
        1.0,
    ),
    # Large activations: fp16 V would overflow without the prescale (F4 grows
    # inputs by up to 100x), so this one exercises e[t] end to end. C > 8 * 32
    # so the prescale max is reduced across waves through LDS.
    (
        "f4_bf16_big_ps",
        dict(N=1, Hi=8, Wi=8, C=512, K=16, pH=1, pW=1),
        4,
        "bf16",
        "fp16",
        True,
        2,
        2000.0,
    ),
]


# Transform-kernel variants (id, problem kwargs, m, io, op, prescale, vec,
# x scale, extra spec kwargs). Odd tile counts so every variant has tails:
# runs of tiles along W that end past the row, LDS regions past the bottom /
# right edge, channel chunks past C.
_VARIANT_CASES = [
    (
        "f4_runs3_wtail",
        dict(N=2, Hi=9, Wi=23, C=24, K=16, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        4,
        1.0,
        dict(tiles_per_thread=3),
    ),
    (
        "f2_runs2_wg_rows2",
        dict(N=1, Hi=11, Wi=13, C=16, K=8, pH=1, pW=1),
        2,
        "bf16",
        "bf16",
        False,
        8,
        1.0,
        dict(tiles_per_thread=2, wg_tile_rows=2, block_size=64),
    ),
    (
        "f4_lds_4x8_tails",
        dict(N=2, Hi=23, Wi=37, C=40, K=24, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        4,
        1.0,
        dict(input_block_tiles=(4, 8), tiles_per_thread=4, lds_lanes=8),
    ),
    (
        "f2_lds_dil2_vec2",
        dict(N=1, Hi=17, Wi=15, C=6, K=10, pH=2, pW=2, dH=2, dW=2),
        2,
        "fp16",
        "fp16",
        False,
        8,
        1.0,
        dict(input_block_tiles=(2, 3)),
    ),
    (
        "f4_lds_g2_bf16",
        dict(N=1, Hi=10, Wi=9, C=24, K=32, pH=1, pW=1, groups=2),
        4,
        "bf16",
        "fp16",
        False,
        4,
        1.0,
        dict(input_block_tiles=(2, 2), tiles_per_thread=2),
    ),
    # Split prescale (K2a per-pixel max + LDS K2) with large activations.
    (
        "f4_bf16_big_ps_split",
        dict(N=1, Hi=9, Wi=10, C=512, K=16, pH=1, pW=1),
        4,
        "bf16",
        "fp16",
        True,
        4,
        2000.0,
        dict(prescale_mode="split", input_block_tiles=(2, 2)),
    ),
    # Fused prescale, runs of 2 tiles with a W tail, max across waves.
    (
        "f4_bf16_big_ps_runs2",
        dict(N=1, Hi=9, Wi=14, C=512, K=16, pH=1, pW=1),
        4,
        "bf16",
        "fp16",
        True,
        4,
        2000.0,
        dict(tiles_per_thread=2),
    ),
    (
        "f4_fp16_rt_stores",
        dict(N=1, Hi=8, Wi=8, C=16, K=24, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        4,
        1.0,
        dict(nt_stores=False),
    ),
]


def _run(pk, m, io, op, ps, vec, scale, *, chunks=1, merge=(), mode="serial", **extra):
    """One pipeline run on a small shape; ``(rocke metric, relL2)``.

    ``m``: int or ``(mh, mw)``. ``chunks``: batch split (the spec is built on
    one N-chunk). ``merge``: transform stages launched as one kernel.
    ``mode``: ``serial`` (``run_once``), ``two_streams`` (K1 on a second
    stream) or ``graph`` (the chain captured into a HIP graph, replayed once).
    """
    import dataclasses

    from benchmarks.common.benchmark_winograd_conv import (
        WinogradPipeline,
        build_k3_candidates,
        make_inputs,
        verify_error,
    )
    from builders.common.conv_reference import conv_reference_gfx1250
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_winograd_fwd import WinogradFwdSpec

    mh, mw = m if isinstance(m, tuple) else (m, 0)
    p = ConvProblem(Y=3, X=3, **pk)
    sub = dataclasses.replace(p, N=p.N // chunks)
    spec = WinogradFwdSpec(
        problem=sub,
        m=mh,
        m_w=mw,
        io_dtype=io,
        operand_dtype=op,
        vec=vec,
        prescale=ps,
        **{"block_size": 64, **extra},
    )
    pipe = WinogradPipeline(spec=spec, arch=GPU_ARCH, chunks=chunks, merge=merge)
    transforms = pipe.compile_transforms()
    cands = build_k3_candidates((_K3_CFG,), pipe.gemm_problem, op, GPU_ARCH, jobs=1)
    assert cands, "K3 config did not build"
    x, w, y = make_inputs(p, io, seed=7)
    if scale != 1.0:
        x = (x.float() * scale).to(x.dtype)
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[io]
    ref = conv_reference_gfx1250(x, w, p, out_dtype=tdt)
    pipe.alloc()
    try:
        pipe.upload(x, w)
        pipe.make_launchers(transforms)
        pipe.set_k3(*cands[0])
        pipe.poison(0x7F)
        if mode == "serial":
            pipe.run_once()
        elif mode == "two_streams":
            pipe.bound_two_streams()()
            torch.cuda.synchronize()
        else:
            s = torch.cuda.Stream()
            run = pipe.bound_total(stream=s.cuda_stream)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                run()
            torch.cuda.synchronize()
            pipe.poison(0x7F)  # the capture did not run the kernels
            pipe.rt.sync()
            with torch.cuda.stream(s):
                g.replay()
            torch.cuda.synchronize()
            del g
        out = pipe.read_y(y)
    finally:
        pipe.free()
    return verify_error(out, ref)


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_winograd_fwd_gpu(case):
    _id, pk, m, io, op, ps, vec, scale = case
    err, l2 = _run(pk, m, io, op, ps, vec, scale)
    assert err < _TOL, f"{_id}: err={err:.3e} relL2={l2:.3e}"


def test_winograd_fwd_gpu_large_inputs_need_prescale():
    """The large-activation case overflows fp16 V/M without the prescale.

    Guards that ``f4_bf16_big_ps`` really exercises e[t]: if this starts
    passing, the case no longer stresses the range and needs bigger inputs.
    """
    case = next(c for c in _CASES if c[0] == "f4_bf16_big_ps")
    _id, pk, m, io, op, _ps, vec, scale = case
    err, _ = _run(pk, m, io, op, False, vec, scale)
    assert not err < _TOL


@pytest.mark.parametrize("case", _VARIANT_CASES, ids=[c[0] for c in _VARIANT_CASES])
def test_winograd_fwd_gpu_transform_variants(case):
    _id, pk, m, io, op, ps, vec, scale, extra = case
    err, l2 = _run(pk, m, io, op, ps, vec, scale, **extra)
    assert err < _TOL, f"{_id}: err={err:.3e} relL2={l2:.3e}"


# Pipeline modes: batch split, merged K1 + K2 launches, a
# second stream for K1, HIP graph replay, and F(6,3) / mixed tiles.
# (id, problem kwargs, m, io, operand, prescale, vec, transform kwargs,
#  pipeline kwargs)
_TUNED_F2 = dict(
    input_block_tiles=(4, 8), tiles_per_thread=4, lds_lanes=16, filter_rows=4
)
_TUNED_F4 = dict(
    input_block_tiles=(4, 8), tiles_per_thread=4, lds_lanes=16, filter_rows=2
)
_PIPE_CASES = [
    # K2 (LDS, 64 threads) + K1 (128 threads, split over two blocks): the
    # layout that packed two K2 blocks into one workgroup before the fix.
    (
        "merge_f2_k2_64",
        dict(N=2, Hi=16, Wi=16, C=64, K=64, pH=1, pW=1),
        2,
        "fp16",
        "fp16",
        False,
        8,
        _TUNED_F2,
        dict(merge=("K2", "K1")),
    ),
    (
        "merge_k1_first_f4",
        dict(N=2, Hi=16, Wi=16, C=64, K=64, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        4,
        _TUNED_F4,
        dict(merge=("K1", "K2")),
    ),
    # Fused prescale K2 (block >= 128) with packed K1 blocks.
    (
        "merge_f4_bf16_ps",
        dict(N=1, Hi=9, Wi=14, C=512, K=16, pH=1, pW=1),
        4,
        "bf16",
        "fp16",
        True,
        4,
        dict(filter_rows=2, block_size=256),
        dict(merge=("K2", "K1")),
    ),
    (
        "chunks2_f4",
        dict(N=4, Hi=11, Wi=13, C=24, K=40, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        4,
        {},
        dict(chunks=2),
    ),
    (
        "chunks4_merge_f2_bf16",
        dict(N=4, Hi=10, Wi=9, C=24, K=16, pH=1, pW=1, groups=2),
        2,
        "bf16",
        "bf16",
        False,
        8,
        dict(filter_rows=4),
        dict(chunks=4, merge=("K2", "K1")),
    ),
    (
        "two_streams_f4",
        dict(N=2, Hi=8, Wi=8, C=16, K=24, pH=1, pW=1),
        4,
        "fp16",
        "fp16",
        False,
        4,
        {},
        dict(mode="two_streams"),
    ),
    (
        "graph_merge_chunks_f2",
        dict(N=2, Hi=8, Wi=8, C=16, K=24, pH=1, pW=1),
        2,
        "fp16",
        "fp16",
        False,
        8,
        {},
        dict(mode="graph", merge=("K2", "K1"), chunks=2),
    ),
    (
        "f6x4_fp16",
        dict(N=1, Hi=19, Wi=23, C=24, K=40, pH=1, pW=1),
        (6, 4),
        "fp16",
        "fp16",
        False,
        4,
        {},
        {},
    ),
    (
        "f4x6_bf16_ps_lds",
        dict(N=2, Hi=23, Wi=37, C=40, K=24, pH=1, pW=1),
        (4, 6),
        "bf16",
        "fp16",
        True,
        4,
        dict(prescale_mode="split", input_block_tiles=(2, 3)),
        dict(merge=("K2a", "K1")),
    ),
    (
        "f6_fp16_dil2",
        dict(N=1, Hi=17, Wi=15, C=16, K=16, pH=2, pW=2, dH=2, dW=2),
        6,
        "fp16",
        "fp16",
        False,
        4,
        {},
        {},
    ),
    (
        "f2x6_g2_lds",
        dict(N=1, Hi=10, Wi=21, C=24, K=32, pH=1, pW=1, groups=2),
        (2, 6),
        "fp16",
        "fp16",
        False,
        4,
        dict(input_block_tiles=(2, 2), tiles_per_thread=2, filter_rows=4),
        {},
    ),
    (
        "f6x2_runs3",
        dict(N=1, Hi=13, Wi=14, C=16, K=16, pH=1, pW=1),
        (6, 2),
        "bf16",
        "fp16",
        False,
        4,
        dict(tiles_per_thread=3),
        {},
    ),
]


@pytest.mark.parametrize("case", _PIPE_CASES, ids=[c[0] for c in _PIPE_CASES])
def test_winograd_fwd_gpu_pipeline_modes(case):
    _id, pk, m, io, op, ps, vec, extra, pipe_kw = case
    err, l2 = _run(pk, m, io, op, ps, vec, 1.0, **pipe_kw, **extra)
    assert err < _TOL, f"{_id}: err={err:.3e} relL2={l2:.3e}"
