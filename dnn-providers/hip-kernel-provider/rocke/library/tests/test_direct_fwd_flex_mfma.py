# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx950 flattened-pixel implicit-GEMM MFMA direct forward
(``conv_direct_fwd_flex_mfma``, variant ``direct_fwd_flex_mfma``).

CPU: the validator, the AOT grid and the cache offer, the C++/Python engine
agreement and the run-side re-validation of the split_c knob. GPU (gfx950):
the output against a torch reference for each layer type in FWD_FLEX_LAYERS
at small sizes; split_c 2 and 4 (with the split-reduce fold); K tails where
K % bk != 0 (K=136 with bk=128 and K=264 with bk=256); odd N*Ho*Wo pixel
tails; and C == 64 * split_c exactly.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <python> -m pytest \\
        rocke/library/tests/test_direct_fwd_flex_mfma.py -v
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest

from kernels.common.conv_direct_fwd_flex_mfma import (
    FwdFlexMfmaSpec,
    build_fwd_flex_mfma,
    fwd_flex_mfma_reason,
)
from kernels.conv.direct_aot_jobs import FWD_FLEX_LAYERS, FWD_FLEX_KNOBS
from kernels.common.conv_direct_grouped import DirectConvProblem

_VARIANT = "direct_fwd_flex_mfma"
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


def _problem(
    N=2,
    H=37,
    W=41,
    C=192,
    K=136,
    KH=3,
    KW=3,
    PAD=1,
    pad_w=-1,
    stride=2,
    dil_h=1,
    dil_w=1,
    dtype="bf16",
):
    """A stride-2 3x3 bf16 forward: X is N x H x W x C, Y N x Ho x Wo x K.
    C=192 (multiple of 64), K=136 (multiple of 8), PAD=1 for 3x3 s2."""
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=stride,
        dil_h=dil_h,
        dil_w=dil_w,
        dtype=dtype,
    )


# --------------------------------------------------------------------------- CPU


def test_arch_gate_is_capability_based():
    """The arch gate reads ArchTarget capabilities (wave64, 16-byte LDS DMA,
    mfma_f32_16x16x32), not an arch name; of the known arches only gfx950 has
    them all, so the AOT set stays gfx950-only."""
    from rocke.core.arch.target import known_arches

    ok = FwdFlexMfmaSpec(problem=_problem())
    for dtype in ("fp16", "bf16"):
        spec = replace(ok, problem=replace(ok.problem, dtype=dtype))
        passing = [a for a in known_arches() if fwd_flex_mfma_reason(spec, a) is None]
        assert passing == ["gfx950"], (dtype, passing)


def test_reasons():
    ok = FwdFlexMfmaSpec(problem=_problem())
    assert fwd_flex_mfma_reason(ok) is None
    for bk in (128, 256):
        assert fwd_flex_mfma_reason(replace(ok, bk=bk)) is None
    for sc in (1, 2, 3):
        # C=192, C/64=3: split_c in [1, 3] must pass
        assert fwd_flex_mfma_reason(replace(ok, split_c=sc)) is None, sc

    def why(**kw):
        return fwd_flex_mfma_reason(replace(ok, problem=replace(ok.problem, **kw)))

    # Channel granularity: C must be a multiple of 64
    assert "multiple of 64" in fwd_flex_mfma_reason(
        replace(ok, problem=replace(ok.problem, cpg=12))
    )
    assert "multiple of 8" in why(kpg=10)
    # split_c out of range: C=192, C/64=3, so split_c=4 is too large
    assert "split_c" in fwd_flex_mfma_reason(replace(ok, split_c=4))  # 4 > 3
    assert "split_c" in fwd_flex_mfma_reason(replace(ok, split_c=0))
    # stride and dilation must be >= 1
    assert "stride" in fwd_flex_mfma_reason(
        replace(ok, problem=replace(ok.problem, stride=0))
    )
    assert "dilation" in fwd_flex_mfma_reason(
        replace(ok, problem=replace(ok.problem, dil_h=0))
    )
    assert "dilation" in fwd_flex_mfma_reason(
        replace(ok, problem=replace(ok.problem, dil_w=0))
    )
    # groups must be 1
    assert "groups" in why(groups=2)
    # dtype
    assert "fp16/bf16" in why(dtype="fp32")
    # extent checks from problem_dims_reason
    assert why(cpg=0) == "cpg must be >= 1 (got 0)"
    assert why(kpg=0) == "kpg must be >= 1 (got 0)"
    assert why(N=0) == "N must be >= 1 (got 0)"
    # bk must be 128 or 256
    assert "bk must be 128 or 256" in fwd_flex_mfma_reason(replace(ok, bk=64))
    # pad_w: flex supports per-axis padding; only pad_w < -1 is refused
    assert (
        fwd_flex_mfma_reason(replace(ok, problem=replace(ok.problem, pad_w=0))) is None
    )
    assert "pad_w must be -1" in fwd_flex_mfma_reason(
        replace(ok, problem=replace(ok.problem, pad_w=-3))
    )
    # Buffer size limits
    limit = "exceeds 2147483647 B, the i32 buffer range"
    assert why(H=1024, W=2048, cpg=768, kpg=768).startswith("input") and limit in why(
        H=1024, W=2048, cpg=768, kpg=768
    )
    assert "output" in why(H=16, W=16, cpg=64, kpg=1 << 27)
    assert "weights" in why(H=1, W=1, cpg=1 << 14, kpg=1 << 14)
    # Arch gates
    for arch in ("gfx942", "gfx90a"):
        assert fwd_flex_mfma_reason(ok, arch=arch) == (
            f"fwd_flex_mfma needs the 16-byte LDS DMA (absent on {arch})"
        )
    wave64 = "fwd_flex_mfma is a wave64 MFMA kernel"
    assert fwd_flex_mfma_reason(ok, arch="gfx1250") == wave64
    assert fwd_flex_mfma_reason(replace(ok, wave_size=32)) == wave64
    # tap count limit: KH * KW > 32
    bigfilter = replace(
        ok,
        problem=replace(
            ok.problem, KH=6, KW=6, PAD=0, pad_w=0, stride=1, dil_h=1, dil_w=1
        ),
    )
    assert "32 taps" in fwd_flex_mfma_reason(bigfilter)
    # workspace slab size limit when split_c > 1: each slab is N*Ho*Wo*K*4 bytes;
    # input/output/weights OK but one slab > I32_MAX.
    # K=128, KH=KW=1, PAD=0, stride=1 -> Ho=H, Wo=W;
    # N=1, H=4096, W=1024: NHW=4194304; slab=4*4194304*128=2^31 > I32_MAX.
    # input=2*4194304*128=2^30 < I32_MAX; output=same; weights=2*128*1*128=32768.
    ws_spec = FwdFlexMfmaSpec(
        problem=_problem(
            N=1, H=4096, W=1024, C=128, K=128, KH=1, KW=1, PAD=0, pad_w=0, stride=1
        ),
        split_c=2,
    )
    reason = fwd_flex_mfma_reason(ws_spec)
    assert reason is not None and "workspace" in reason, reason
    # The whole workspace's size is an i32 kernarg: four 1 GiB slabs (each
    # within the buffer range) are refused.
    big_ws = FwdFlexMfmaSpec(
        problem=_problem(N=32, H=255, W=255, C=256, K=512, stride=2), split_c=4
    )
    assert big_ws.workspace_bytes == 4 << 30
    assert fwd_flex_mfma_reason(big_ws) == (
        "workspace exceeds 2147483647 B, the i32 byte-count kernarg"
    )
    # build raises when spec is invalid
    with pytest.raises(ValueError, match="bk must be 128 or 256"):
        build_fwd_flex_mfma(replace(ok, bk=64))


def test_multi_round_cases_overflow_the_persistent_grid():
    """The two GPU cases above that run more items than the persistent grid
    leave a partial last round."""
    from kernels.common.conv_direct_fwd_flex_mfma import PERSIST

    for kw, bk, items in (
        (
            dict(N=1, H=257, W=257, C=64, K=64, PAD=2, stride=1, dil_h=2, dil_w=2),
            128,
            259,
        ),
        (dict(N=2, H=91, W=91, C=128, K=512, PAD=1, stride=1), 128, 260),
    ):
        spec = FwdFlexMfmaSpec(problem=_problem(**kw), bk=bk)
        assert fwd_flex_mfma_reason(spec) is None
        assert spec.items() == items and spec.grid() == (PERSIST, 1, 1)
        assert items % PERSIST


def test_grid_and_items():
    # N=2, H=37, W=41 at stride 2: Ho=19, Wo=21; NX=2*19*21=798
    # ceil(798 / 256) = 4 pixel tiles; ceil(136 / 256) = 1 K tile; split_c=1
    # items = 4 * 1 * 1 = 4; grid = (min(4, 256), 1, 1) = (4, 1, 1)
    p = _problem()
    spec = FwdFlexMfmaSpec(problem=p, bk=256, split_c=1)
    items = spec.items()
    assert items == 4, items
    assert spec.grid() == (4, 1, 1)
    # split_c=2: items = 4 * 1 * 2 = 8
    spec2 = replace(spec, split_c=2)
    assert spec2.items() == 8
    assert spec2.grid() == (8, 1, 1)
    # workspace: 4 * split_c * N * Ho * Wo * K bytes
    assert spec.workspace_bytes == 0
    assert spec2.workspace_bytes == 4 * 2 * 2 * 19 * 21 * 136


def test_kernel_name_and_attrs():
    spec = FwdFlexMfmaSpec(problem=_problem(), name="rocke_fwd_flex_mfma")
    kernel = build_fwd_flex_mfma(spec)
    # kernel_name_join(name, f3x3, p1x1, s2, d1x1, k256, sc1) + bf16 suffix
    assert kernel.name == "rocke_fwd_flex_mfma_f3x3_p1x1_s2_d1x1_k256_sc1_bf16"
    assert kernel.attrs["lds_alias_scopes"] is True
    assert kernel.attrs["max_workgroup_size"] == 8 * 64

    spec2 = FwdFlexMfmaSpec(problem=_problem(dtype="fp16"), bk=128)
    name2 = build_fwd_flex_mfma(spec2).name
    # No bf16 suffix for fp16
    assert "fp16" not in name2 and "bf16" not in name2
    assert "k128" in name2


def test_ir_is_shape_independent():
    """One binary serves every batch, extent and channel count at runtime."""
    from rocke.core.ir_serialize import serialize

    def ir(**kw):
        spec = FwdFlexMfmaSpec(problem=_problem(**kw))
        return serialize(build_fwd_flex_mfma(spec)).replace(spec.kernel_name(), "k")

    assert ir() == ir(N=5, H=64, W=17, C=256, K=512)


def test_aot_grid():
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        _knob_grid,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

    direction, caps_list = DIRECT_CAPABILITIES[_VARIANT]
    assert direction == "fwd"
    # No knobs for non-gfx950
    for caps in caps_list:
        assert not list(_knob_grid(_VARIANT, "gfx942", "bf16", caps))
        assert not list(_knob_grid(_VARIANT, "gfx1250", "bf16", caps))

    # All 6 knobs (bk, split_c) are valid for every caps entry on gfx950
    built = set()
    for caps in caps_list:
        for dtype in ("bf16", "fp16"):
            probe = probe_problem(caps, dtype, _VARIANT)
            for knobs in _knob_grid(_VARIANT, "gfx950", dtype, caps):
                spec = make_direct_spec(_VARIANT, probe, knobs)
                ok, why = validate_direct_spec(_VARIANT, spec, "gfx950")
                assert ok, (caps, knobs, why)
                built.add(
                    (caps.KH, caps.kw or caps.KH, knobs["bk"], knobs["split_c"], dtype)
                )

    # Every (bk, split_c) combination from FWD_FLEX_KNOBS must appear
    knob_pairs = {(bk, sc) for bk, sc in FWD_FLEX_KNOBS}
    built_pairs = {(bk, sc) for _, _, bk, sc, _ in built}
    assert built_pairs == knob_pairs, (built_pairs, knob_pairs)

    # Each split_c > 1 kernel brings its split-C fold job into the AOT cell.
    import json

    from kernels.conv.direct_aot_jobs import _caps_jobs
    from kernels.conv.direct_variants import DIRECT_SPLIT_REDUCE

    _, jobs = _caps_jobs("gfx950", 64, _VARIANT, "fwd", caps_list[0], "bf16")
    folds = {
        json.loads(job.identity.knobs)["split_c"]
        for _, job in jobs
        if job.identity.algorithm == DIRECT_SPLIT_REDUCE
    }
    assert folds == {sc for _, sc in FWD_FLEX_KNOBS if sc > 1}


def test_variant_wiring():
    from kernels.conv.direct_variants import (
        DIRECT_FWD_FLEX_MFMA,
        direct_spec_for_problem,
        make_direct_spec,
    )

    p = _problem(H=32, W=48, C=256)
    spec = make_direct_spec(_VARIANT, p, {"bk": 256, "split_c": 1})
    assert spec.problem == p and spec.name == "rocke_fwd_flex_mfma"
    got, why = direct_spec_for_problem(_VARIANT, p, {"bk": 128, "split_c": 1}, "gfx950")
    assert got is not None, why
    # Groups != 1 is refused
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, groups=2), {"bk": 256, "split_c": 1}, "gfx950"
    )
    assert got is None and "groups" in why, why
    # zero channel count is refused, not divided by
    got, why = direct_spec_for_problem(
        _VARIANT, replace(p, kpg=0), {"bk": 256, "split_c": 1}, "gfx950"
    )
    assert got is None and why == "kpg must be >= 1 (got 0)"


def test_run_side_revalidation_refuses_split_c_too_large():
    """The run side rebuilds a spec on the real problem and runs the kernel's
    own validator, so a binary compiled for C=256 with split_c=4 (C/64=4,
    split_c=4 valid) is refused for C=128 (C/64=2, split_c=4 > 2)."""
    from kernels.conv.direct_variants import direct_spec_for_problem

    # Problem with C=256: split_c=4 is at the limit (C/64=4).
    p_ok = _problem(C=256)
    got, why = direct_spec_for_problem(
        _VARIANT, p_ok, {"bk": 256, "split_c": 4}, "gfx950"
    )
    assert got is not None, f"expected accepted: {why}"

    # Same knobs but C=128: C/64=2, split_c=4 > 2 -> refused.
    p_bad = _problem(C=128)
    got, why = direct_spec_for_problem(
        _VARIANT, p_bad, {"bk": 256, "split_c": 4}, "gfx950"
    )
    assert got is None, "expected refused for C=128 with split_c=4"
    assert "split_c" in why, f"unexpected refusal reason: {why!r}"


def test_engines_agree_on_specs():
    """The C++ engine's validator, kernel name, grid and workspace bytes match
    Python's over the AOT grid and a set of rejected specs."""
    eng = pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import conv_direct_fwd_flex_mfma_spec_to_dict

    specs = []
    for dtype in ("bf16", "fp16"):
        for kh, kw, pad, pw, st, dh, dw in FWD_FLEX_LAYERS[:4]:
            for bk, sc in ((256, 1), (256, 2), (128, 4)):
                p = _problem(
                    N=2,
                    H=29,
                    W=37,
                    C=192,
                    K=136,
                    KH=kh,
                    KW=kw,
                    PAD=pad,
                    pad_w=pw if pw != pad else -1,
                    stride=st,
                    dil_h=dh,
                    dil_w=dw,
                    dtype=dtype,
                )
                specs.append(FwdFlexMfmaSpec(problem=p, bk=bk, split_c=sc))
    base = _problem()
    for pkw, skw in (
        (dict(groups=2), {}),
        (dict(cpg=12), {}),
        (dict(kpg=10), {}),
        (dict(stride=0), {}),
        (dict(dil_h=0), {}),
        (dict(dtype="fp32"), {}),
        (dict(cpg=0), {}),
        (dict(kpg=0), {}),
        (dict(N=0), {}),
        (dict(H=1024, W=2048, cpg=768, kpg=768), {}),
        ({}, dict(bk=64)),
        ({}, dict(split_c=4)),  # C=192, C/64=3, split_c=4 > 3
        ({}, dict(wave_size=32)),
    ):
        specs.append(FwdFlexMfmaSpec(problem=replace(base, **pkw), **skw))

    for arch in ("gfx950", "gfx942", "gfx1250"):
        for spec in specs:
            ok, why_c, name, grid, ws = eng.conv_direct_fwd_flex_mfma_describe(
                conv_direct_fwd_flex_mfma_spec_to_dict(spec), arch
            )
            why = fwd_flex_mfma_reason(spec, arch)
            assert ok == (why is None), (arch, spec, why, why_c)
            if why is not None:
                assert why_c == why, (arch, spec, why, why_c)
            else:
                assert tuple(grid) == spec.grid(), (arch, spec)
                assert ws == spec.workspace_bytes, (arch, spec)
            assert name == spec.kernel_name()


def test_binding_rejects_bad_input():
    """A missing, mistyped or partial problem raises in the binding."""
    eng = pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import conv_direct_fwd_flex_mfma_spec_to_dict

    good = conv_direct_fwd_flex_mfma_spec_to_dict(FwdFlexMfmaSpec(problem=_problem()))
    partial = dict(good["problem"])
    del partial["kpg"]
    for problem in (None, "N2H13", partial):
        sd = dict(good)
        if problem is None:
            del sd["problem"]
        else:
            sd["problem"] = problem
        with pytest.raises(ValueError):
            eng.conv_direct_fwd_flex_mfma_describe(sd, "gfx950")
        with pytest.raises(ValueError):
            eng.conv_direct_fwd_flex_mfma_lower_llvm(sd, arch="gfx950")
    # An explicit dtype=None is refused
    sd = dict(good, problem=dict(good["problem"], dtype=None))
    for fn in (
        lambda: eng.conv_direct_fwd_flex_mfma_describe(sd, "gfx950"),
        lambda: eng.conv_direct_fwd_flex_mfma_lower_llvm(sd, arch="gfx950"),
    ):
        with pytest.raises(ValueError, match="dtype must be a string, not None"):
            fn()
    spec = FwdFlexMfmaSpec(problem=replace(_problem(), dtype=None))
    assert fwd_flex_mfma_reason(spec) == "fp16/bf16 only"


def test_backend_lowers_through_both_engines():
    """``lower_conv_direct_fwd_flex_mfma`` gives the same .ll on both backends."""
    pytest.importorskip("rocke_library_engine")
    from kernels.conv.engine_bridge import lower_conv_direct_fwd_flex_mfma

    spec = FwdFlexMfmaSpec(problem=_problem(), bk=128, split_c=1)
    py = lower_conv_direct_fwd_flex_mfma(spec, arch="gfx950", backend="python")
    cpp = lower_conv_direct_fwd_flex_mfma(spec, arch="gfx950", backend="cpp")
    assert py.backend == "python" and cpp.backend == "cpp"
    assert py.llvm_text == cpp.llvm_text


def test_python_backend_needs_no_engine(monkeypatch):
    """``backend="python"`` lowers without the rocke_library_engine extension."""
    import sys

    from kernels.conv.engine_bridge import lower_conv_direct_fwd_flex_mfma
    from rocke.core.backend import BackendError

    monkeypatch.setitem(sys.modules, "rocke_library_engine", None)
    spec = FwdFlexMfmaSpec(problem=_problem())
    got = lower_conv_direct_fwd_flex_mfma(spec, arch="gfx950", backend="python")
    assert got.backend == "python" and got.llvm_text
    with pytest.raises(BackendError, match="rocke_library_engine"):
        lower_conv_direct_fwd_flex_mfma(spec, arch="gfx950", backend="cpp")


# --------------------------------------------------------------------------- GPU


def _ref_conv(x, w, p):
    """torch reference in fp32, NHWK out (x NHWC, w KRSC)."""
    import torch

    return torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2).float(),
        w.permute(0, 3, 1, 2).float(),
        stride=p.stride,
        padding=(p.PAD, p.PAD_W),
        dilation=(p.dil_h, p.dil_w),
    ).permute(0, 2, 3, 1)


def _launch_flex(spec, x, w, y_or_ws):
    """Launch the flex kernel.  For split_c == 1, ``y_or_ws`` is the final
    output (NHWK fp16/bf16); for split_c > 1 it must be an fp32 workspace
    buffer of ``spec.workspace_bytes // 4`` elements, and the caller is
    responsible for running the split_reduce fold afterwards."""
    import torch
    from rocke import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs

    p = spec.problem
    art = compile_kernel(build_fwd_flex_mfma(spec, arch="gfx950"), arch="gfx950")
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_args_signature(p.dtype),
    )
    values = ConvArgs.from_problem(p).to_launch_values(
        x.data_ptr(),
        w.data_ptr(),
        y_or_ws.data_ptr(),
        x.nbytes,
        w.nbytes,
        y_or_ws.nbytes,
    )
    grid, block = spec.launch_geometry()
    launcher(values, config=LaunchConfig(grid=grid, block=block))
    torch.cuda.synchronize()


def _launch_split_reduce(split_c, dtype, p, ws, y):
    """Run the direct_split_reduce fold kernel over the fp32 workspace ``ws``
    and write the folded output into ``y`` (NHWK fp16/bf16)."""
    import torch
    from rocke import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_nongrouped import (
        DirectSplitReduceSpec,
        build_direct_conv_split_reduce,
    )

    rspec = DirectSplitReduceSpec(dtype=dtype, split_c=split_c)
    art = compile_kernel(
        build_direct_conv_split_reduce(rspec, arch="gfx950"), arch="gfx950"
    )
    # The split_reduce kernel takes the same direct-conv kernarg block: A is
    # the workspace (the split slabs, fp32) and D is the final output.  B is
    # unused; pass the workspace ptr for it too so the byte check is happy.
    values = ConvArgs.from_problem(p).to_launch_values(
        ws.data_ptr(),
        ws.data_ptr(),
        y.data_ptr(),
        ws.nbytes,
        ws.nbytes,
        y.nbytes,
    )
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_args_signature(dtype),
    )
    grid = rspec.grid(p)
    launcher(values, config=LaunchConfig(grid=grid, block=(rspec.threads, 1, 1)))
    torch.cuda.synchronize()


@_GPU
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "kh,kw,pad,pad_w,stride,dil_h,dil_w,bk,split_c,N,H,W,C,K",
    [
        # 3x3 s2 at both bk, K ending inside a bk block (K % bk != 0)
        (3, 3, 1, 1, 2, 1, 1, 256, 1, 2, 19, 21, 192, 136),
        (3, 3, 1, 1, 2, 1, 1, 128, 1, 2, 19, 21, 192, 136),
        (3, 3, 1, 1, 2, 1, 1, 256, 1, 2, 19, 21, 192, 264),
        (1, 1, 0, 0, 2, 1, 1, 256, 1, 2, 19, 21, 192, 136),
        (2, 2, 0, 0, 2, 1, 1, 128, 1, 2, 19, 21, 192, 136),
        # dilated 3x3, "same" padding
        (3, 3, 2, 2, 1, 2, 2, 256, 1, 2, 19, 21, 128, 128),
        (3, 3, 12, 12, 1, 12, 12, 128, 1, 1, 30, 31, 64, 72),
        # 1xS with column padding / column dilation, 7x1
        (1, 7, 0, 3, 1, 1, 1, 128, 1, 2, 19, 21, 128, 136),
        (1, 3, 0, 3, 1, 1, 3, 256, 1, 2, 19, 21, 64, 64),
        (7, 1, 3, 0, 1, 1, 1, 128, 1, 2, 19, 21, 128, 128),
        # split_c with the fold; C = 64 * split_c exactly
        (3, 3, 1, 1, 2, 1, 1, 256, 2, 2, 19, 21, 128, 128),
        (3, 3, 1, 1, 2, 1, 1, 128, 4, 2, 19, 21, 256, 136),
        (3, 3, 1, 1, 2, 1, 1, 256, 2, 2, 19, 21, 320, 128),
        # pixel tail: N * Ho * Wo = 9, far below one 256-pixel tile
        (3, 3, 1, 1, 2, 1, 1, 256, 1, 1, 5, 5, 192, 128),
        # More items than the persistent grid, with a partial last round:
        # 259 dilated pixel tiles (tap ranges change between tiles), and
        # 65 pixel tiles x 4 K tiles = 260 items.
        (3, 3, 2, 2, 1, 2, 2, 128, 1, 1, 257, 257, 64, 64),
        (3, 3, 1, 1, 1, 1, 1, 128, 1, 2, 91, 91, 128, 512),
    ],
)
def test_forward_matches_torch(
    dtype, kh, kw, pad, pad_w, stride, dil_h, dil_w, bk, split_c, N, H, W, C, K
):
    if _gpu_arch() != "gfx950":
        pytest.skip("needs a gfx950 GPU")
    import torch

    p = _problem(
        N=N,
        H=H,
        W=W,
        C=C,
        K=K,
        KH=kh,
        KW=kw,
        PAD=pad,
        pad_w=pad_w if pad_w != pad else -1,
        stride=stride,
        dil_h=dil_h,
        dil_w=dil_w,
        dtype=dtype,
    )
    spec = FwdFlexMfmaSpec(problem=p, bk=bk, split_c=split_c)
    assert fwd_flex_mfma_reason(spec) is None
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    x = torch.empty(p.N, p.H, p.W, p.cpg, dtype=td, device="cuda").uniform_(-1, 1)
    w = torch.empty(p.kpg, kh, kw, p.cpg, dtype=td, device="cuda").uniform_(-1, 1)
    y = torch.full((p.N, p.Ho, p.Wo, p.kpg), float("nan"), dtype=td, device="cuda")
    if split_c == 1:
        _launch_flex(spec, x, w, y)
    else:
        ws = torch.full(
            (spec.workspace_bytes // 4,),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        _launch_flex(spec, x, w, ws)
        _launch_split_reduce(split_c, dtype, p, ws, y)
    ref = _ref_conv(x, w, p)
    assert not torch.isnan(y).any()
    tol = 2e-2 if dtype == "bf16" else 5e-3
    rel = float((y.float() - ref).abs().max() / ref.abs().max().clamp(min=1))
    assert rel < tol, rel
