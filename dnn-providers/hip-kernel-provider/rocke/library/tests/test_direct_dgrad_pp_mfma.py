# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx950 LDS-DMA ping-pong MFMA direct dgrad
(``conv_direct_dgrad_pp_mfma``, variant ``direct_dgrad_pp_mfma``).

CPU: the validator, the transposed problem, the AOT grid and the cache offer.
GPU (gfx950): dX against torch's input gradient for 3x3 and 5x5 at every
block_k, both dtypes, with image edges on every side (extents that are not a
multiple of the 16x16 tile), partial dY channel chunks (K not a multiple of
64), partial dX channel blocks, and dX channel counts that end inside a
16-byte weight granule.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_dgrad_pp_mfma.py -v
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest

from kernels.common.conv_direct_dgrad_pp_mfma import (
    DgradPpMfmaSpec,
    build_dgrad_pp_mfma,
    dgrad_pp_mfma_problem,
    dgrad_pp_mfma_reason,
    dgrad_pp_mfma_tile_efficiency,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

_VARIANT = "direct_dgrad_pp_mfma"
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _gpu_arch():
    if not _HAS_TORCH:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    from rocke.runtime.hip_module import get_device_arch

    return get_device_arch(0)


_GPU = pytest.mark.skipif(not _HAS_TORCH, reason="needs torch and a gfx950 GPU")


def _dgrad(N=2, H=37, W=41, C=136, K=96, KH=3, dtype="bf16"):
    """A "same"-padded stride-1 dgrad: dX is N x H x W x C, dY N x H x W x K."""
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KH,
        PAD=(KH - 1) // 2,
        stride=1,
        dtype=dtype,
    )


# --------------------------------------------------------------------------- CPU


def test_transposed_problem():
    tp = dgrad_pp_mfma_problem(_dgrad(KH=5))
    assert (tp.H, tp.W, tp.cpg, tp.kpg, tp.PAD) == (37, 41, 96, 136, 2)
    assert (tp.Ho, tp.Wo) == (37, 41)


def test_transposed_problem_flips_explicit_column_padding():
    """An explicit pad_w is flipped like PAD (-1 stays PAD), so dX keeps the
    dgrad's extents; a padding the kernel was not built for is refused."""
    same = replace(_dgrad(H=20, W=24), pad_w=1)
    tp = dgrad_pp_mfma_problem(same)
    assert (tp.PAD, tp.pad_w, tp.Ho, tp.Wo) == (1, 1, 20, 24)
    assert dgrad_pp_mfma_reason(DgradPpMfmaSpec(problem=tp)) is None
    valid = replace(_dgrad(H=20, W=24), PAD=0, pad_w=0)
    tp = dgrad_pp_mfma_problem(valid)
    assert (tp.PAD, tp.pad_w, tp.Ho, tp.Wo) == (2, 2, 20, 24)
    assert "'same' padding" in dgrad_pp_mfma_reason(DgradPpMfmaSpec(problem=tp))
    assert dgrad_pp_mfma_problem(_dgrad()).pad_w == -1


def test_arch_gate_is_capability_based():
    """The arch gate reads ArchTarget capabilities (wave64, ds_read_b64_tr_b16,
    16-byte LDS DMA, mfma_f32_16x16x32), not an arch name; of the known
    arches only gfx950 has them all, so the AOT set stays gfx950-only."""
    from rocke.core.arch.target import known_arches

    ok = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()))
    for dtype in ("fp16", "bf16"):
        spec = replace(ok, problem=replace(ok.problem, dtype=dtype))
        passing = [a for a in known_arches() if dgrad_pp_mfma_reason(spec, a) is None]
        assert passing == ["gfx950"], (dtype, passing)


def test_transposed_problem_flips_dilated_padding():
    """The flipped padding is dil*(K-1)-PAD per axis, so the transposed
    problem maps a dilated dgrad's dY back to its dX extents (the validators
    still refuse the dilation)."""
    for KH, PAD, dh, dw, pad_w in ((3, 2, 2, 3, 3), (5, 4, 2, 2, -1), (3, 0, 3, 1, 1)):
        d = replace(_dgrad(H=20, W=24, KH=KH), PAD=PAD, dil_h=dh, dil_w=dw, pad_w=pad_w)
        tp = dgrad_pp_mfma_problem(d)
        assert tp.PAD == dh * (KH - 1) - PAD, (d, tp)
        if pad_w < 0:
            assert tp.pad_w == -1
        else:
            assert tp.pad_w == dw * (KH - 1) - pad_w, (d, tp)
        assert (tp.H, tp.W) == (d.Ho, d.Wo)
        assert (tp.Ho, tp.Wo) == (d.H, d.W), (d, tp)
        assert dgrad_pp_mfma_reason(DgradPpMfmaSpec(problem=tp)) == "dilation must be 1"


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch (CPU)")
@pytest.mark.parametrize(
    "KH, KW, PAD, dil, pad_w",
    [
        (3, 3, 1, (2, 1), -1),
        (3, 3, 1, (2, 3), -1),
        (3, 3, 2, (2, 1), 0),
        (3, 5, 1, (1, 1), -1),
        (5, 5, 4, (2, 2), -1),
    ],
)
def test_transposed_problem_matches_cpu_dgrad(KH, KW, PAD, dil, pad_w):
    """The forward conv over the transposed problem (flipped, k<->c
    transposed filter) is the CPU dgrad: same dX extents and values. An
    implicit pad_w stays -1 only when the column flip equals the row flip
    (dil_w*(KW-1) == dil_h*(KH-1)); else it is dil_w*(KW-1)-PAD."""
    import torch

    dh, dw = dil
    d = replace(
        _dgrad(N=1, H=9, W=11, C=4, K=3, KH=KH),
        KW=KW,
        PAD=PAD,
        dil_h=dh,
        dil_w=dw,
        pad_w=pad_w,
    )
    tp = dgrad_pp_mfma_problem(d)
    if pad_w == -1 and dw * (KW - 1) == dh * (KH - 1):
        assert tp.pad_w == -1, (d, tp)
    else:
        assert tp.pad_w == dw * (KW - 1) - d.PAD_W, (d, tp)
    assert (tp.Ho, tp.Wo) == (d.H, d.W), (d, tp)

    torch.manual_seed(0)
    w = torch.randn(d.total_k, d.total_c, KH, KW, dtype=torch.float64)
    dy = torch.randn(d.N, d.total_k, d.Ho, d.Wo, dtype=torch.float64)
    ref = torch.nn.grad.conv2d_input(
        (d.N, d.total_c, d.H, d.W),
        w,
        dy,
        padding=(d.PAD, d.PAD_W),
        dilation=(dh, dw),
    )
    got = torch.nn.functional.conv2d(
        dy,
        w.flip(2, 3).transpose(0, 1),
        padding=(tp.PAD, tp.PAD_W),
        dilation=(tp.dil_h, tp.dil_w),
    )
    assert got.shape == ref.shape, (d, tp)
    torch.testing.assert_close(got, ref)


def test_reasons():
    ok = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()))
    assert dgrad_pp_mfma_reason(ok) is None
    assert dgrad_pp_mfma_reason(replace(ok, block_k=256)) is None

    def why(**kw):
        return dgrad_pp_mfma_reason(replace(ok, problem=replace(ok.problem, **kw)))

    assert "stride" in why(stride=2)
    assert "groups" in why(groups=2)
    assert why(dil_h=2) == "dilation must be 1"
    assert why(dil_w=2) == "dilation must be 1"
    assert "multiple of 8" in why(cpg=12)
    assert "multiple of 4" in why(kpg=10)
    assert why(KH=4, KW=4, PAD=1) == "square odd filters >= 3 only"
    assert why(KH=3, KW=5) == "square odd filters >= 3 only"
    assert why(KH=1, KW=1, PAD=0) == "square odd filters >= 3 only"
    # Every extent is at least 1 (before any divisibility check).
    assert why(cpg=0) == "cpg must be >= 1 (got 0)"
    assert why(kpg=0) == "kpg must be >= 1 (got 0)"
    assert why(N=0) == "N must be >= 1 (got 0)"
    # "Same" padding on both axes only.
    assert why(PAD=0) == "'same' padding only: PAD must be 1 (got 0)"
    assert why(pad_w=0) == "one padding for both axes (got PAD=1x0)"
    assert "pad_w must be -1" in why(pad_w=-3)
    assert why(pad_w=1) is None
    # Each buffer descriptor (one input image, one output image, the filter)
    # stays within the i32 range the masked offset relies on, and so does the
    # flat block id.
    big = dict(H=1024, W=2048, cpg=768, kpg=768)
    limit = "exceeds 2147483647 B, the i32 buffer range"
    assert why(**big) == f"input image {limit}"
    assert why(H=16, W=16, cpg=8, kpg=1 << 27) == f"output image {limit}"
    assert why(H=1, W=1, cpg=1 << 14, kpg=1 << 14) == f"filter {limit}"
    assert why(H=1 << 30, W=1 << 30, cpg=1 << 30, kpg=8) == f"input image {limit}"
    assert "workgroups exceed the i32 block id" in why(N=1 << 25, H=256, W=256)
    assert "fp16/bf16" in why(dtype="fp32")
    for arch in ("gfx942", "gfx90a"):
        assert dgrad_pp_mfma_reason(ok, arch=arch) == (
            f"dgrad_pp_mfma needs ds_read_b64_tr_b16 (absent on {arch})"
        )
    wave64 = "dgrad_pp_mfma is a wave64 MFMA kernel"
    assert dgrad_pp_mfma_reason(ok, arch="gfx1250") == wave64
    assert dgrad_pp_mfma_reason(replace(ok, wave_size=32)) == wave64
    assert "block_k" in dgrad_pp_mfma_reason(replace(ok, block_k=64))
    # 5x5 at block_k 256 and 7x7 at any block_k do not fit LDS.
    five = replace(ok, problem=replace(ok.problem, KH=5, KW=5, PAD=2))
    assert dgrad_pp_mfma_reason(five) is None
    assert "LDS" in dgrad_pp_mfma_reason(replace(five, block_k=256))
    seven = replace(ok, problem=replace(ok.problem, KH=7, KW=7, PAD=3))
    assert "LDS" in dgrad_pp_mfma_reason(seven)
    with pytest.raises(ValueError, match="block_k"):
        build_dgrad_pp_mfma(replace(ok, block_k=64))
    # The engine's spec dict carries the (flipped) explicit column padding.
    from kernels.conv.engine_bridge import conv_direct_dgrad_pp_mfma_spec_to_dict

    p = dgrad_pp_mfma_problem(replace(_dgrad(), pad_w=1))
    sd = conv_direct_dgrad_pp_mfma_spec_to_dict(DgradPpMfmaSpec(problem=p))
    assert sd["problem"]["pad_w"] == 1
    p = dgrad_pp_mfma_problem(replace(_dgrad(KH=5), PAD=2, pad_w=0))
    sd = conv_direct_dgrad_pp_mfma_spec_to_dict(DgradPpMfmaSpec(problem=p))
    assert sd["problem"]["pad_w"] == 4
    sd = conv_direct_dgrad_pp_mfma_spec_to_dict(
        DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()))
    )
    assert sd["problem"]["pad_w"] == -1


def test_grid_and_efficiency():
    spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()), block_k=128)
    # 2 images x 3 x 3 tiles of 16 x 2 blocks of 128 dX channels.
    assert spec.launch_geometry() == ((2 * 3 * 3 * 2, 1, 1), (512, 1, 1))
    eff = dgrad_pp_mfma_tile_efficiency(spec)
    assert eff == pytest.approx((37 * 41 * 136) / (48 * 48 * 256))


def test_kernel_name_and_attrs():
    spec = DgradPpMfmaSpec(
        problem=dgrad_pp_mfma_problem(_dgrad(KH=5)), name="rocke_dgrad_pp_mfma"
    )
    kernel = build_dgrad_pp_mfma(spec)
    assert kernel.name == "rocke_dgrad_pp_mfma_f5x5_p2_bk128_bf16"
    assert kernel.attrs["lds_alias_scopes"] is True
    assert kernel.attrs["max_workgroup_size"] == 512


def test_aot_grid():
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        _knob_grid,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

    direction, caps_list = DIRECT_CAPABILITIES[_VARIANT]
    assert direction == "dgrad"
    built = set()
    for caps in caps_list:
        assert not list(_knob_grid(_VARIANT, "gfx942", "bf16", caps))
        assert not list(_knob_grid(_VARIANT, "gfx1250", "bf16", caps))
        for dtype in ("bf16", "fp16"):
            probe = probe_problem(caps, dtype, _VARIANT)
            for knobs in _knob_grid(_VARIANT, "gfx950", dtype, caps):
                spec = make_direct_spec(_VARIANT, probe, knobs)
                if validate_direct_spec(_VARIANT, spec, "gfx950")[0]:
                    built.add((caps.KH, knobs["block_k"], dtype))
    # 5x5 at block_k 256 does not fit LDS.
    assert built == {
        (kh, bk, dt)
        for kh, bk in ((3, 128), (3, 256), (5, 128))
        for dt in ("bf16", "fp16")
    }


def test_variant_wiring():
    from kernels.conv.direct_variants import (
        DIRECT_DGRAD_FPROP_SINGLE,
        direct_dgrad_kernarg_direction,
        direct_spec_for_problem,
        make_direct_spec,
    )

    assert _VARIANT in DIRECT_DGRAD_FPROP_SINGLE
    assert direct_dgrad_kernarg_direction(_VARIANT) == "fwd"
    p = _dgrad(H=32, W=48, C=128)
    spec = make_direct_spec(_VARIANT, p, {"block_k": 256})
    assert spec.problem == dgrad_pp_mfma_problem(p)
    got, why = direct_spec_for_problem(_VARIANT, p, {"block_k": 128}, "gfx950")
    assert got is not None, why
    # Strided and grouped dgrads are not offered.
    assert (
        direct_spec_for_problem(
            _VARIANT, replace(p, stride=2), {"block_k": 128}, "gfx950"
        )[0]
        is None
    )
    # Dilated dgrads are not offered: the transposition carries the dilation
    # so the forward validator refuses it.
    for dil in ({"dil_h": 2}, {"dil_w": 2}):
        got, why = direct_spec_for_problem(
            _VARIANT, replace(p, **dil), {"block_k": 128}, "gfx950"
        )
        assert got is None and "dilation" in why, (dil, why)
    # Mostly-empty tiles are not offered (a 4x4 image in a 16x16 tile).
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, H=4, W=4), {"block_k": 128}, "gfx950"
    )
    assert got is None and "tile efficiency" in why
    # A zero channel count is refused, not divided by.
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, cpg=0), {"block_k": 128}, "gfx950"
    )
    assert got is None and why == "kpg must be >= 1 (got 0)"
    # An explicit "same" column padding keeps dX at the dgrad's extents.
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, pad_w=1), {"block_k": 128}, "gfx950"
    )
    assert got is not None, why
    assert (got.problem.Ho, got.problem.Wo) == (p.H, p.W)


def test_ir_is_shape_independent():
    """One binary serves every batch, extent and channel count."""
    from rocke.core.ir_serialize import serialize

    def ir(**kw):
        spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad(**kw)))
        return serialize(build_dgrad_pp_mfma(spec)).replace(spec.kernel_name(), "k")

    assert ir() == ir(N=5, H=64, W=17, C=256, K=512)


def test_engines_agree_on_specs():
    """The C++ engine's validator, kernel name, grid and LDS size match
    Python's over the AOT grid and a set of rejected specs (the byte-identity
    gate only covers the IR of specs both engines build)."""
    eng = pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import conv_direct_dgrad_pp_mfma_spec_to_dict

    specs = []
    for dtype in ("bf16", "fp16"):
        for kh in (3, 5, 7):
            for bk in (64, 128, 256):
                p = dgrad_pp_mfma_problem(_dgrad(N=3, H=29, W=37, KH=kh, dtype=dtype))
                specs.append(DgradPpMfmaSpec(problem=p, block_k=bk))
    base = dgrad_pp_mfma_problem(_dgrad())
    for pkw, knobs in (
        (dict(stride=2), {}),
        (dict(groups=2), {}),
        (dict(dil_h=2), {}),
        (dict(dil_w=2), {}),
        (dict(cpg=12), {}),
        (dict(kpg=10), {}),
        (dict(KH=4, KW=4, PAD=1), {}),
        (dict(KH=3, KW=5), {}),
        (dict(KH=1, KW=1, PAD=0), {}),
        (dict(dtype="fp32"), {}),
        (dict(cpg=0), {}),
        (dict(kpg=0), {}),
        (dict(N=0), {}),
        (dict(PAD=0), {}),
        (dict(pad_w=0), {}),
        (dict(pad_w=-3), {}),
        (dict(pad_w=1), {}),
        (dict(H=1024, W=2048, cpg=768, kpg=768), {}),
        (dict(H=16, W=16, cpg=8, kpg=1 << 27), {}),
        (dict(H=1, W=1, cpg=1 << 14, kpg=1 << 14), {}),
        (dict(N=1 << 25, H=256, W=256), {}),
        (dict(H=(1 << 31) - 1, W=(1 << 31) - 1, cpg=(1 << 31) - 8, kpg=8), {}),
        ({}, dict(wave_size=32)),
        ({}, dict(block_k=96)),
    ):
        specs.append(DgradPpMfmaSpec(problem=replace(base, **pkw), **knobs))
    for arch in ("gfx950", "gfx942", "gfx1250"):
        for spec in specs:
            ok, why_c, name, grid, lds = eng.conv_direct_dgrad_pp_mfma_describe(
                conv_direct_dgrad_pp_mfma_spec_to_dict(spec), arch
            )
            why = dgrad_pp_mfma_reason(spec, arch)
            assert ok == (why is None), (arch, spec, why)
            if why is not None:
                assert why_c == why, (arch, spec)
            else:
                assert tuple(grid) == spec.grid()
            assert name == spec.kernel_name()
            assert lds == spec.lds_bytes()


def test_binding_rejects_bad_input():
    """A missing, mistyped or partial problem raises in the binding."""
    eng = pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import conv_direct_dgrad_pp_mfma_spec_to_dict

    good = conv_direct_dgrad_pp_mfma_spec_to_dict(
        DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()))
    )
    partial = dict(good["problem"])
    del partial["kpg"]
    for problem in (None, "N2H13", partial):
        sd = dict(good)
        if problem is None:
            del sd["problem"]
        else:
            sd["problem"] = problem
        with pytest.raises(ValueError):
            eng.conv_direct_dgrad_pp_mfma_describe(sd, "gfx950")
        with pytest.raises(ValueError):
            eng.conv_direct_dgrad_pp_mfma_lower_llvm(sd, arch="gfx950")
    # An explicit dtype=None is refused (Python's validator refuses it too),
    # not read as the fp16 default.
    sd = dict(good, problem=dict(good["problem"], dtype=None))
    for fn in (
        lambda: eng.conv_direct_dgrad_pp_mfma_describe(sd, "gfx950"),
        lambda: eng.conv_direct_dgrad_pp_mfma_lower_llvm(sd, arch="gfx950"),
    ):
        with pytest.raises(ValueError, match="dtype must be a string, not None"):
            fn()
    spec = DgradPpMfmaSpec(problem=replace(dgrad_pp_mfma_problem(_dgrad()), dtype=None))
    assert dgrad_pp_mfma_reason(spec) == "fp16/bf16 only"


def test_backend_lowers_through_both_engines():
    """``lower_conv_direct_dgrad_pp_mfma`` gives the same .ll on the Python
    and the C++ engine."""
    pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import lower_conv_direct_dgrad_pp_mfma

    spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()), block_k=256)
    py = lower_conv_direct_dgrad_pp_mfma(spec, arch="gfx950", backend="python")
    cpp = lower_conv_direct_dgrad_pp_mfma(spec, arch="gfx950", backend="cpp")
    assert py.backend == "python" and cpp.backend == "cpp"
    assert py.llvm_text == cpp.llvm_text


def test_python_backend_needs_no_engine(monkeypatch):
    """``backend="python"`` lowers without the rocke_library_engine extension."""
    import sys

    from kernels.conv.engine_bridge import lower_conv_direct_dgrad_pp_mfma
    from rocke.core.backend import BackendError

    monkeypatch.setitem(sys.modules, "rocke_library_engine", None)  # not importable
    spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_dgrad()))
    got = lower_conv_direct_dgrad_pp_mfma(spec, arch="gfx950", backend="python")
    assert got.backend == "python" and got.llvm_text
    with pytest.raises(BackendError, match="rocke_library_engine"):
        lower_conv_direct_dgrad_pp_mfma(spec, arch="gfx950", backend="cpp")


# --------------------------------------------------------------------------- GPU


def _launch(spec, a, b, d):
    import torch
    from rocke import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs

    p = spec.problem
    art = compile_kernel(build_dgrad_pp_mfma(spec, arch="gfx950"), arch="gfx950")
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_args_signature(p.dtype),
    )
    values = ConvArgs.from_problem(p).to_launch_values(
        a.data_ptr(), b.data_ptr(), d.data_ptr(), a.nbytes, b.nbytes, d.nbytes
    )
    grid, block = spec.launch_geometry()
    launcher(values, config=LaunchConfig(grid=grid, block=block))
    torch.cuda.synchronize()


@_GPU
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "kh,block_k,C,K",
    [
        (3, 128, 136, 96),  # partial dX block, partial dY chunk
        (3, 256, 136, 96),
        (3, 128, 100, 72),  # dX channels end inside a weight granule
        (3, 256, 64, 200),  # one chunk pair plus an odd tail chunk
        (5, 128, 136, 96),
        (5, 128, 92, 136),
    ],
)
def test_dgrad_matches_torch(dtype, kh, block_k, C, K):
    if _gpu_arch() != "gfx950":
        pytest.skip("needs a gfx950 GPU")
    import torch

    p = _dgrad(C=C, K=K, KH=kh, dtype=dtype)
    spec = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(p), block_k=block_k)
    assert dgrad_pp_mfma_reason(spec) is None
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    dy = torch.empty(p.N, p.Ho, p.Wo, p.kpg, dtype=td, device="cuda").uniform_(-1, 1)
    w = torch.empty(p.kpg, kh, kh, p.cpg, dtype=td, device="cuda").uniform_(-1, 1)
    dx = torch.full((p.N, p.H, p.W, p.cpg), float("nan"), dtype=td, device="cuda")
    _launch(spec, dy, w, dx)
    ref = torch.nn.grad.conv2d_input(
        (p.N, p.cpg, p.H, p.W),
        w.permute(0, 3, 1, 2).float(),
        dy.permute(0, 3, 1, 2).float(),
        padding=p.PAD,
    ).permute(0, 2, 3, 1)
    assert not torch.isnan(dx).any()
    tol = 2e-2 if dtype == "bf16" else 5e-3
    rel = float((dx.float() - ref).abs().max() / ref.abs().max().clamp(min=1))
    assert rel < tol, rel
