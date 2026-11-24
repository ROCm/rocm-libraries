# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx950 LDS-DMA ping-pong MFMA direct forward
(``conv_direct_fwd_pp_mfma``, variant ``direct_fwd_pp_mfma``).

CPU: the validator, the AOT grid and the cache offer, and the agreement of the
two engines. GPU (gfx950): the output against torch for 3x3 at every block_k
and for 5x5 and 7x7, both dtypes, with image edges on every side (extents
that are not a multiple of the 16x16 tile), partial channel chunks (C not a
multiple of 64), an odd tail chunk, and partial output-channel blocks.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_fwd_pp_mfma.py -v
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest

from kernels.common.conv_direct_fwd_pp_mfma import (
    BLOCK_K,
    FwdPpMfmaSpec,
    build_fwd_pp_mfma,
    fwd_pp_mfma_reason,
    fwd_pp_mfma_tile_efficiency,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

_VARIANT = "direct_fwd_pp_mfma"
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


def _problem(N=2, H=37, W=41, C=96, K=136, KH=3, dtype="bf16"):
    """A "same"-padded stride-1 forward: X is N x H x W x C, Y N x H x W x K."""
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


def test_arch_gate_is_capability_based():
    """The arch gate reads ArchTarget capabilities (wave64, 16-byte LDS DMA,
    mfma_f32_16x16x32), not an arch name; of the known arches only gfx950 has
    them all, so the AOT set stays gfx950-only."""
    from rocke.core.arch.target import known_arches

    ok = FwdPpMfmaSpec(problem=_problem())
    for dtype in ("fp16", "bf16"):
        spec = replace(ok, problem=replace(ok.problem, dtype=dtype))
        passing = [a for a in known_arches() if fwd_pp_mfma_reason(spec, a) is None]
        assert passing == ["gfx950"], (dtype, passing)


def test_reasons():
    ok = FwdPpMfmaSpec(problem=_problem())
    assert fwd_pp_mfma_reason(ok) is None
    for bk in BLOCK_K:
        assert fwd_pp_mfma_reason(replace(ok, block_k=bk)) is None

    def why(**kw):
        return fwd_pp_mfma_reason(replace(ok, problem=replace(ok.problem, **kw)))

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
    # stays within the i32 range the masked offset relies on.
    limit = "exceeds 2147483647 B, the i32 buffer range"
    assert why(H=1024, W=2048, cpg=768, kpg=768) == f"input image {limit}"
    assert why(H=16, W=16, cpg=8, kpg=1 << 27) == f"output image {limit}"
    assert why(H=1, W=1, cpg=1 << 14, kpg=1 << 14) == f"filter {limit}"
    # The whole input and output sizes are i32 kernargs: 32 images that each
    # fit, but 4 GiB in all, are refused.
    kernarg = "exceeds 2147483647 B, the i32 byte-count kernarg"
    assert why(N=32, H=256, W=256, cpg=1024, kpg=1024) == f"input {kernarg}"
    assert why(N=32, H=256, W=256, cpg=8, kpg=1024) == f"output {kernarg}"
    assert "fp16/bf16" in why(dtype="fp32")
    for arch in ("gfx942", "gfx90a"):
        assert fwd_pp_mfma_reason(ok, arch=arch) == (
            f"fwd_pp_mfma needs the 16-byte LDS DMA (absent on {arch})"
        )
    wave64 = "fwd_pp_mfma is a wave64 MFMA kernel"
    assert fwd_pp_mfma_reason(ok, arch="gfx1250") == wave64
    assert fwd_pp_mfma_reason(replace(ok, wave_size=32)) == wave64
    assert fwd_pp_mfma_reason(replace(ok, block_k=96)) == (
        "block_k must be 64, 128 or 256"
    )
    # 5x5 fits LDS up to block_k 128, 7x7 only at block_k 64.
    five = replace(ok, problem=replace(ok.problem, KH=5, KW=5, PAD=2))
    assert fwd_pp_mfma_reason(five) is None
    assert "LDS" in fwd_pp_mfma_reason(replace(five, block_k=256))
    seven = replace(ok, problem=replace(ok.problem, KH=7, KW=7, PAD=3))
    assert fwd_pp_mfma_reason(replace(seven, block_k=64)) is None
    assert "LDS" in fwd_pp_mfma_reason(seven)
    with pytest.raises(ValueError, match="block_k"):
        build_fwd_pp_mfma(replace(ok, block_k=96))


def test_grid_and_efficiency():
    spec = FwdPpMfmaSpec(problem=_problem(), block_k=128)
    # 2 images x 3 x 3 tiles of 16 x 2 blocks of 128 output channels.
    assert spec.launch_geometry() == ((2 * 3 * 3 * 2, 1, 1), (512, 1, 1))
    eff = fwd_pp_mfma_tile_efficiency(spec)
    assert eff == pytest.approx((37 * 41 * 136) / (48 * 48 * 256))


def test_kernel_name_and_attrs():
    spec = FwdPpMfmaSpec(problem=_problem(KH=5), name="rocke_fwd_pp_mfma")
    kernel = build_fwd_pp_mfma(spec)
    assert kernel.name == "rocke_fwd_pp_mfma_f5x5_p2_bk128_bf16"
    assert kernel.attrs["lds_alias_scopes"] is True
    assert kernel.attrs["max_workgroup_size"] == 512
    spec = FwdPpMfmaSpec(problem=_problem(dtype="fp16"), block_k=64)
    assert build_fwd_pp_mfma(spec).name == "fwd_pp_mfma_f3x3_p1_bk64"


def _barrier_shape(region):
    """``s_barrier_bare`` count at each nesting level of ``region``: (count at
    this level, shapes of every nested region in order)."""
    top, nested = 0, []
    for op in region.ops:
        top += op.name == "tile.s_barrier_bare"
        nested += [_barrier_shape(r) for r in op.regions]
    return top, tuple(nested)


@pytest.mark.parametrize(
    "kh, block_k", [(3, 64), (3, 128), (3, 256), (5, 128), (7, 64)]
)
def test_rank_branches_hit_the_same_barriers(kh, block_k):
    """The two wave ranks run different schedules under a wave-uniform rank
    branch (the rank comes from a readfirstlane wave index). s_barrier counts
    waves, not program counters, so the schedule is sound when every wave
    executes the same number of barriers: both rank branches must have the
    same barrier count at every loop / if nesting level."""
    spec = FwdPpMfmaSpec(problem=_problem(KH=kh), block_k=block_k)
    ranks = [op for op in build_fwd_pp_mfma(spec).body.ops if op.regions]
    assert [op.name for op in ranks] == ["scf.if_else"]
    rank0, rank1 = (_barrier_shape(r) for r in ranks[0].regions)
    assert rank0 == rank1, (rank0, rank1)
    assert rank0[0] == 2 and rank0[1], rank0


def test_aot_grid():
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        _knob_grid,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

    direction, caps_list = DIRECT_CAPABILITIES[_VARIANT]
    assert direction == "fwd"
    built = set()
    for caps in caps_list:
        assert not list(_knob_grid(_VARIANT, "gfx942", "bf16", caps))
        assert not list(_knob_grid(_VARIANT, "gfx1250", "bf16", caps))
        for dtype in ("bf16", "fp16"):
            probe = probe_problem(caps, dtype, _VARIANT)
            for knobs in _knob_grid(_VARIANT, "gfx950", dtype, caps):
                spec = make_direct_spec(_VARIANT, probe, knobs)
                ok, why = validate_direct_spec(_VARIANT, spec, "gfx950")
                assert ok, (caps, knobs, why)
                built.add((caps.KH, knobs["block_k"], dtype))
    assert built == {(3, bk, dt) for bk in BLOCK_K for dt in ("bf16", "fp16")}


def test_variant_wiring():
    from kernels.conv.direct_variants import (
        direct_spec_for_problem,
        make_direct_spec,
    )

    p = _problem(H=32, W=48, C=128)
    spec = make_direct_spec(_VARIANT, p, {"block_k": 256})
    assert spec.problem == p and spec.name == "rocke_fwd_pp_mfma"
    got, why = direct_spec_for_problem(_VARIANT, p, {"block_k": 128}, "gfx950")
    assert got is not None, why
    # Strided, grouped and dilated forwards are not offered.
    for kw, reason in (
        ({"stride": 2}, "stride"),
        ({"groups": 2}, "groups"),
        ({"dil_h": 2}, "dilation"),
    ):
        got, why = direct_spec_for_problem(
            _VARIANT, replace(p, **kw), {"block_k": 128}, "gfx950"
        )
        assert got is None and reason in why, (kw, why)
    # Mostly-empty tiles are not offered (a 4x4 image in a 16x16 tile).
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, H=4, W=4), {"block_k": 128}, "gfx950"
    )
    assert got is None and "tile efficiency" in why
    # A zero channel count is refused, not divided by.
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, kpg=0), {"block_k": 128}, "gfx950"
    )
    assert got is None and why == "kpg must be >= 1 (got 0)"


def test_ir_is_shape_independent():
    """One binary serves every batch, extent and channel count."""
    from rocke.core.ir_serialize import serialize

    def ir(**kw):
        spec = FwdPpMfmaSpec(problem=_problem(**kw))
        return serialize(build_fwd_pp_mfma(spec)).replace(spec.kernel_name(), "k")

    assert ir() == ir(N=5, H=64, W=17, C=256, K=512)


def test_engines_agree_on_specs():
    """The C++ engine's validator, kernel name, grid and LDS size match
    Python's over the AOT grid and a set of rejected specs (the byte-identity
    gate only covers the IR of specs both engines build)."""
    eng = pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import conv_direct_fwd_pp_mfma_spec_to_dict

    specs = []
    for dtype in ("bf16", "fp16"):
        for kh in (3, 5, 7):
            for bk in (64, 96, 128, 256):
                p = _problem(N=3, H=29, W=37, KH=kh, dtype=dtype)
                specs.append(FwdPpMfmaSpec(problem=p, block_k=bk))
    base = _problem()
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
    ):
        specs.append(FwdPpMfmaSpec(problem=replace(base, **pkw), **knobs))
    for arch in ("gfx950", "gfx942", "gfx1250"):
        for spec in specs:
            ok, why_c, name, grid, lds = eng.conv_direct_fwd_pp_mfma_describe(
                conv_direct_fwd_pp_mfma_spec_to_dict(spec), arch
            )
            why = fwd_pp_mfma_reason(spec, arch)
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
    from kernels.conv.engine_bridge import conv_direct_fwd_pp_mfma_spec_to_dict

    good = conv_direct_fwd_pp_mfma_spec_to_dict(FwdPpMfmaSpec(problem=_problem()))
    partial = dict(good["problem"])
    del partial["kpg"]
    for problem in (None, "N2H13", partial):
        sd = dict(good)
        if problem is None:
            del sd["problem"]
        else:
            sd["problem"] = problem
        with pytest.raises(ValueError):
            eng.conv_direct_fwd_pp_mfma_describe(sd, "gfx950")
        with pytest.raises(ValueError):
            eng.conv_direct_fwd_pp_mfma_lower_llvm(sd, arch="gfx950")
    # An explicit dtype=None is refused (Python's validator refuses it too),
    # not read as the fp16 default.
    sd = dict(good, problem=dict(good["problem"], dtype=None))
    for fn in (
        lambda: eng.conv_direct_fwd_pp_mfma_describe(sd, "gfx950"),
        lambda: eng.conv_direct_fwd_pp_mfma_lower_llvm(sd, arch="gfx950"),
    ):
        with pytest.raises(ValueError, match="dtype must be a string, not None"):
            fn()
    spec = FwdPpMfmaSpec(problem=replace(_problem(), dtype=None))
    assert fwd_pp_mfma_reason(spec) == "fp16/bf16 only"


def test_backend_lowers_through_both_engines():
    """``lower_conv_direct_fwd_pp_mfma`` gives the same .ll on the Python and
    the C++ engine."""
    pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import lower_conv_direct_fwd_pp_mfma

    spec = FwdPpMfmaSpec(problem=_problem(), block_k=64)
    py = lower_conv_direct_fwd_pp_mfma(spec, arch="gfx950", backend="python")
    cpp = lower_conv_direct_fwd_pp_mfma(spec, arch="gfx950", backend="cpp")
    assert py.backend == "python" and cpp.backend == "cpp"
    assert py.llvm_text == cpp.llvm_text


def test_python_backend_needs_no_engine(monkeypatch):
    """``backend="python"`` lowers without the rocke_library_engine extension."""
    import sys

    from kernels.conv.engine_bridge import lower_conv_direct_fwd_pp_mfma
    from rocke.core.backend import BackendError

    monkeypatch.setitem(sys.modules, "rocke_library_engine", None)  # not importable
    spec = FwdPpMfmaSpec(problem=_problem())
    got = lower_conv_direct_fwd_pp_mfma(spec, arch="gfx950", backend="python")
    assert got.backend == "python" and got.llvm_text
    with pytest.raises(BackendError, match="rocke_library_engine"):
        lower_conv_direct_fwd_pp_mfma(spec, arch="gfx950", backend="cpp")


# --------------------------------------------------------------------------- GPU


def _launch(spec, a, b, d):
    import torch
    from rocke import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs

    p = spec.problem
    art = compile_kernel(build_fwd_pp_mfma(spec, arch="gfx950"), arch="gfx950")
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
        (3, 64, 96, 136),  # partial chunk, partial output block
        (3, 128, 96, 136),
        (3, 256, 96, 136),
        (3, 128, 200, 72),  # one chunk pair plus an odd tail chunk
        (3, 256, 64, 300),
        (5, 128, 136, 96),
        (7, 64, 72, 100),
    ],
)
def test_forward_matches_torch(dtype, kh, block_k, C, K):
    if _gpu_arch() != "gfx950":
        pytest.skip("needs a gfx950 GPU")
    import torch

    p = _problem(C=C, K=K, KH=kh, dtype=dtype)
    spec = FwdPpMfmaSpec(problem=p, block_k=block_k)
    assert fwd_pp_mfma_reason(spec) is None
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    x = torch.empty(p.N, p.H, p.W, p.cpg, dtype=td, device="cuda").uniform_(-1, 1)
    w = torch.empty(p.kpg, kh, kh, p.cpg, dtype=td, device="cuda").uniform_(-1, 1)
    y = torch.full((p.N, p.Ho, p.Wo, p.kpg), float("nan"), dtype=td, device="cuda")
    _launch(spec, x, w, y)
    ref = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2).float(), w.permute(0, 3, 1, 2).float(), padding=p.PAD
    ).permute(0, 2, 3, 1)
    assert not torch.isnan(y).any()
    tol = 2e-2 if dtype == "bf16" else 5e-3
    rel = float((y.float() - ref).abs().max() / ref.abs().max().clamp(min=1))
    assert rel < tol, rel
