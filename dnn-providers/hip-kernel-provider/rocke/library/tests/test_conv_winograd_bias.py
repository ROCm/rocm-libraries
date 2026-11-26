# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Conv + per-output-channel bias on the Winograd forward families.

``y = round(conv_fp32 + bias[k])``: the bias is added in fp32 in the output transform
(after the prescale / output scale), before the one rounding to the output dtype.

* 2-D Winograd (``conv_winograd_fwd`` K4, ``conv_winograd_fwd_rt`` K4,
  ``conv_winograd_gen`` K4, ``conv_winograd_fused_1d_wmma``): CPU -- the kernarg tail,
  signatures, kernel names (only the stages with the output transform carry ``cbias``)
  and the bias-free stages unchanged. GPU (gfx942) -- the three stand-alone output
  transforms (their pipelines run on gfx1250, but the transforms are plain buffer
  kernels) on the numpy model's M, against its y plus the bias.
* 3-D Winograd (``conv_winograd3d`` yx kernels, bf16): CPU -- names, kernargs, the
  split-K reduction's bias, refusals. GPU (gfx942) -- the drop-in call of the shipped
  forward route specs (split-K, prescale, epi_vec stores) and a scalar-store spec against
  the f32 torch conv3d plus the bias.
"""

from __future__ import annotations

import importlib.util
import json
import re

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _p2(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    base = dict(N=2, Hi=11, Wi=13, C=16, K=24, Y=3, X=3, pH=1, pW=1)
    base.update(kw)
    return ConvProblem(**base)


def _names(k):
    return [p.name for p in k.params]


# ---------------------------------------------------------------------------
# 2-D Winograd, CPU
# ---------------------------------------------------------------------------


def test_fwd_k4_bias_tail_and_name():
    from kernels.common import conv_winograd_fwd as wf

    s = wf.WinogradFwdSpec(problem=_p2(), m=4, bias=True)
    plain = wf.WinogradFwdSpec(problem=_p2(), m=4)
    k = wf.build_winograd_output_transform(s)
    assert _names(k) == _names(wf.build_winograd_output_transform(plain)) + [
        "Bias",
        "Bias_bytes",
    ]
    assert k.name == plain.kernel_name("output") + "_cbias"
    sig = wf.output_transform_signature(s)
    assert [a["name"] for a in sig[-2:]] == ["Bias", "Bias_bytes"]
    assert sig[:-2] == wf.output_transform_signature(plain)
    # K1 / K2 (and merges of them) do not change
    for stage, build in (
        ("filter", wf.build_winograd_filter_transform),
        ("input", wf.build_winograd_input_transform),
    ):
        assert s.kernel_name(stage) == plain.kernel_name(stage)
        assert _names(build(s)) == _names(build(plain))
    assert s.kernel_name("merged_k2_k1") == plain.kernel_name("merged_k2_k1")
    assert s.kernel_name("merged_k4").endswith("_cbias")


def test_rt_gen_fused_bias_tail_and_name():
    from kernels.common import conv_winograd_fused_1d_wmma as fu
    from kernels.common import conv_winograd_fwd_rt as rt
    from kernels.common import conv_winograd_gen as gen

    o = rt.RtOutputSpec(bias=True)
    k = rt.build_output(o)
    assert _names(k)[-2:] == ["Bias", "Bias_bytes"]
    assert k.name == rt.RtOutputSpec().kernel_name() + "_cbias"
    assert [a["name"] for a in rt.output_signature(o)] == _names(k)

    p = _p2(N=1, Hi=1, Wi=60, Y=1, X=7, pH=0, pW=3)
    g = gen.WinogradGenSpec(p, gen.WinoAxis(1, 1), gen.WinoAxis(4, 7), bias=True)
    kg = gen.build_output_transform(g)
    assert _names(kg)[-2:] == ["Bias", "Bias_bytes"]
    assert kg.name.endswith("_cbias") and not g.kernel_name("input").endswith("cbias")
    assert [a["name"] for a in gen.output_transform_signature(g)] == _names(kg)

    pf = _p2(N=1, Hi=1, Wi=200, C=64, K=64, Y=1, X=11, pH=0, pW=5)
    f = fu.FusedWino1DSpec(problem=pf, ax=gen.WinoAxis(4, 11, 3), bias=True)
    assert f.kernel_name().endswith("_cbias")
    assert [a["name"] for a in fu.fused_signature(f)][-2:] == ["Bias", "Bias_bytes"]


def test_engine_bridge_forwards_bias():
    from kernels.common import conv_winograd_fwd as wf
    from kernels.common import conv_winograd_gen as gen
    from kernels.conv import engine_bridge as eb

    s = wf.WinogradFwdSpec(problem=_p2(), m=4, bias=True)
    assert eb.conv_winograd_fwd_spec_to_dict(s, "K4")["bias"] is True
    p = _p2(N=1, Hi=1, Wi=60, Y=1, X=7, pH=0, pW=3)
    g = gen.WinogradGenSpec(p, gen.WinoAxis(1, 1), gen.WinoAxis(4, 7), bias=True)
    assert eb.conv_winograd_gen_spec_to_dict(g, "K4")["bias"] is True


# ---------------------------------------------------------------------------
# 3-D Winograd, CPU
# ---------------------------------------------------------------------------


def _yx(**kw):
    from kernels.conv import conv_winograd3d as cw
    from kernels.conv import conv_winograd3d_select as sel

    d = dict(sel.SPEC_B, Di=10, Hi=18, Wi=18, C=128, K=128)
    d.update(kw)
    return cw.WinogradYxSpec(**d)


def test_yx_bias_tail_and_name():
    from kernels.conv import conv_winograd3d as cw

    plain, s = _yx(), _yx(bias=True)
    assert s.kernel_name() == plain.kernel_name() + "_cbias"
    k = cw.build_winograd_yx(s)
    assert _names(k) == _names(cw.build_winograd_yx(plain)) + ["Bias", "Bias_bytes"]


_F16 = dict(stage="lds16", op_dtype="f16")


@pytest.mark.parametrize(
    "extra",
    [
        dict(),
        dict(stage="lds"),
        dict(stage="lds", waves=8),
        dict(stage="wlds", op_dtype="f16", z_tile=2),
        dict(_F16),
        dict(_F16, b16=True),
        dict(_F16, single_buf=True),
        dict(_F16, waves=8, nsplit=True, n_tile=64, z_tile=2, nfast=True, persist=8),
        dict(_F16, epi_vec=True, yc8h=True, epi16=True, bps_out=True),
        dict(_F16, epi_vec=True, nfast=True, ksplit=2),
    ],
    ids=[
        "reg",
        "lds",
        "lds_w8",
        "wlds",
        "lds16",
        "b16",
        "sbuf",
        "persist",
        "epi16",
        "ks2",
    ],
)
def test_yx_stages_build_with_bias(extra):
    """Every main-kernel variant takes the tail (a split-K one leaves it to the reduction)."""
    from kernels.conv import conv_winograd3d as cw

    s = cw.WinogradYxSpec(Di=10, Hi=18, Wi=18, C=128, K=128, bias=True, **extra)
    ok, why = cw.is_valid_yx_spec(s)
    assert ok, why
    names = _names(cw.build_winograd_yx(s))
    if s.ksplit > 1:
        assert "Bias" not in names
    else:
        assert names[-2:] == ["Bias", "Bias_bytes"]


def test_ks_reduce_bias():
    from kernels.conv import conv_winograd3d as cw

    k = cw.build_ks_reduce(cw.KsReduceSpec(8 * 128, 2, 128, bias=True))
    assert _names(k) == ["P", "Y", "Bias", "Bias_bytes"]
    assert k.name == "wino_ks_reduce_n1024_k2_K128_cbias"
    with pytest.raises(ValueError, match="bias needs K"):
        cw.build_ks_reduce(cw.KsReduceSpec(1000, 2, 12, bias=True))


def test_yx_route_and_dgrad_refuse_bias():
    from dispatch import conv_routes as cr
    from kernels.conv import conv_winograd3d_select as sel

    spec = dict(sel.SPEC_B, bias=True)
    plan = sel.Winograd3dPlan("yx", tuple(sorted(spec.items())), 3, "dropin", "fwd")
    p = _p3()
    with pytest.raises(ValueError, match="routes carry none"):
        cr.winograd_steps(plan, p, "fwd")
    if _HAS_TORCH:
        from kernels.conv.conv_winograd3d_dropin import Winograd3dDropin

        dplan = sel.Winograd3dPlan(
            "yx", tuple(sorted(spec.items())), 3, "dropin", "dgrad"
        )
        with pytest.raises(ValueError, match="forward"):
            Winograd3dDropin(p, "dgrad", dplan)


def _p3(D=10, H=16, W=16, C=128, K=128, pd=1):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(
        N=1, Hi=H, Wi=W, C=C, K=K, Y=3, X=3, pH=1, pW=1, Di=D, Z=3, sD=1, pD=pd, dD=1
    )


# ---------------------------------------------------------------------------
# GPU (gfx942)
# ---------------------------------------------------------------------------


def _gfx942() -> bool:
    if not _HAS_TORCH:
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    return "gfx942" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


_gpu = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gfx942(), reason="needs torch and a gfx942 device"),
]


def _launch(kernel, signature, values, grid, block):
    from rocke.helpers import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    run = KernelLauncher(
        hsaco=compile_kernel(kernel, arch="gfx942").hsaco,
        kernel_name=kernel.name,
        signature=signature,
    )
    run(values, config=LaunchConfig(grid=tuple(grid), block=(block, 1, 1)))


def _peak_err(out, ref, floor=1.0):
    return float((out - ref).abs().max() / ref.abs().max().clamp(min=floor))


def _k4_case(kind, io, op, prescale, problem, tile):
    """Run one output transform with bias on the numpy model's M; returns the errors of
    y against ``round(y_model + bias)`` and against the bias-free model."""
    import numpy as np
    import torch

    from kernels.common import conv_winograd_fwd as wf
    from kernels.common import conv_winograd_fwd_rt as rt
    from kernels.common import conv_winograd_gen as gen

    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[io]
    odt = {"fp16": torch.float16, "bf16": torch.bfloat16}[op]
    p = problem
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, (p.N, p.Hi, p.Wi, p.C))
    w = rng.uniform(-1, 1, (p.K, p.Y, p.X, p.cpg)) / (p.Y * p.X * p.cpg) ** 0.5

    def rnd(name, a):  # the operand dtype of the workspaces
        return torch.from_numpy(a).to(odt).double().numpy() if name == "M" else a

    if kind == "gen":
        ay, ax = tile
        spec = gen.WinogradGenSpec(p, ay, ax, io_dtype=io, operand_dtype=op, bias=True)
        y_ref, _, _, M = gen.winograd_gen_reference_np(x, w, spec, round_fn=rnd)
    else:
        y_ref, _, _, M = wf.winograd_fwd_reference_np(x, w, p, tile, round_fn=rnd)
    T = M.shape[0]
    e = torch.zeros(T, dtype=torch.int32)
    if prescale:  # M of tile t arrives scaled by 2**-e[t]; K4 multiplies it back
        e = torch.from_numpy(rng.integers(0, 4, T).astype(np.int32))
        M = M * (2.0 ** -e.numpy().astype(np.float64))[:, None, None]
    dev = "cuda"
    Mt = torch.from_numpy(np.ascontiguousarray(M)).to(odt).to(dev)
    bias = torch.empty(p.K).uniform_(-4, 4).to(tdt)
    y = torch.full((p.N, p.Ho, p.Wo, p.K), float("nan"), dtype=tdt, device=dev)
    bd, ed = bias.to(dev), e.to(dev)
    vals = dict(
        mm=Mt,
        y=y,
        m_bytes=Mt.nbytes,
        y_bytes=y.nbytes,
        Bias=bd,
        Bias_bytes=bd.nbytes,
    )
    mh, mw = tile if kind != "gen" else (0, 0)
    if kind == "fwd":
        spec = wf.WinogradFwdSpec(
            problem=p,
            m=mh,
            m_w=mw,
            io_dtype=io,
            operand_dtype=op,
            prescale=prescale,
            bias=True,
        )
        k = wf.build_winograd_output_transform(spec)
        sig = wf.output_transform_signature(spec)
        grid, block = wf.output_transform_grid(spec), wf.output_transform_block(spec)
        vals["e"] = ed
    elif kind == "rt":
        spec = rt.RtOutputSpec(
            mh, mw, io, op, rt._pow2_div(p.K, 8), 256, prescale=prescale, bias=True
        )
        k = rt.build_output(spec)
        sig = rt.output_signature(spec)
        grid, block, scal = rt.stage_launch(spec, rt.RtGeometry(p, mh, mw))
        vals.update(scal, e=ed)
    else:
        k = gen.build_output_transform(spec)
        sig = gen.output_transform_signature(spec)
        grid, block = gen.output_transform_grid(spec), spec.block_size
    _launch(k, sig, vals, grid, block)
    torch.cuda.synchronize()
    out = y.float().cpu()
    yr = torch.from_numpy(y_ref).float()
    ref = (yr + bias.float()).to(tdt).float()
    return _peak_err(out, ref), _peak_err(out, yr.to(tdt).float())


_K4_CASES = [
    # (id, kind, io, operand, prescale, problem kwargs, tile)
    ("fwd_f4_fp16", "fwd", "fp16", "fp16", False, {}, (4, 4)),
    ("fwd_f6x4_bf16_ps", "fwd", "bf16", "fp16", True, {}, (6, 4)),
    ("fwd_f2_bf16op_g2_dil2", "fwd", "bf16", "bf16", False, dict(groups=2, dH=2, dW=2, pH=2, pW=2), (2, 2)),
    ("fwd_f4_odd_k", "fwd", "fp16", "fp16", False, dict(C=9, K=11), (4, 4)),
    ("rt_f4_fp16", "rt", "fp16", "fp16", False, {}, (4, 4)),
    ("rt_f4x2_bf16_ps", "rt", "bf16", "fp16", True, dict(K=20), (4, 2)),
    ("gen_1x7", "gen", "fp16", "fp16", False, dict(N=1, Hi=2, Wi=40, Y=1, X=7, pH=0, pW=3), "1x7"),
    ("gen_5x5_g2_bf16", "gen", "bf16", "fp16", False, dict(N=1, Hi=12, Wi=10, C=16, K=16, Y=5, X=5, pH=2, pW=2, groups=2), "5x5"),
]  # fmt: skip


@pytest.mark.parametrize("case", _K4_CASES, ids=[c[0] for c in _K4_CASES])
@pytest.mark.gpu
@pytest.mark.skipif(not _gfx942(), reason="needs torch and a gfx942 device")
def test_winograd2d_output_transform_bias_gpu(case):
    from kernels.common import conv_winograd_gen as gen

    _, kind, io, op, ps, pk, tile = case
    if tile == "1x7":
        tile = (gen.WinoAxis(1, 1), gen.WinoAxis(4, 7))
    elif tile == "5x5":
        tile = (gen.WinoAxis(2, 5), gen.WinoAxis(2, 5))
    err, no_bias = _k4_case(kind, io, op, ps, _p2(**pk), tile)
    assert err < 1e-2, err
    assert no_bias > 0.1, no_bias  # the bias moved the output


def _shipped_fwd_specs():
    from dispatch import conv_routes as cr

    out = {}
    for path in cr.shipped_conv_route_tables().values():
        data = json.loads(path.read_text())
        for mode in data["modes"].values():
            for case, row in mode["cases"].items():
                if row.get("route") != "winograd" or " -F 1 " not in row["cmd"]:
                    continue
                pd = int(re.search(r"--pad_d (\d)", row["cmd"]).group(1))
                key = (pd, json.dumps(row["winograd"]["spec"], sort_keys=True))
                out.setdefault(key, (case, int(row["winograd"]["wtv"])))
    return sorted(out.items())


_SHIPPED_FWD = _shipped_fwd_specs() if _HAS_TORCH else []


def _yx_dropin_err(spec: dict, wtv: int, pd: int, scale: float):
    import torch

    from kernels.conv import conv_winograd3d_select as sel
    from kernels.conv.conv_winograd3d_dropin import Winograd3dDropin

    C = 256 if spec.get("ksplit", 1) > 1 else 128
    p = _p3(D=32 + (2 if pd == 0 else 0), C=C, pd=pd)
    spec = dict(spec, bias=True)
    plan = sel.Winograd3dPlan("yx", tuple(sorted(spec.items())), wtv, "dropin", "fwd")
    call = Winograd3dDropin(p, "fwd", plan)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(p.Di, p.Hi, p.Wi, p.C, device="cuda", generator=g) * scale).to(
        torch.bfloat16
    )
    w = torch.randn(p.K, 3, 3, 3, p.C, device="cuda", generator=g)
    w = (w * (2 / (27 * p.C)) ** 0.5).to(torch.bfloat16)
    bias = (torch.rand(p.K, device="cuda", generator=g) * 8 - 4) * scale
    bias = bias.to(torch.bfloat16)
    y = torch.full(call.out, float("nan"), device="cuda", dtype=torch.bfloat16)
    call(x, w, y, bias=bias)
    torch.cuda.synchronize()
    conv = torch.nn.functional.conv3d(
        x.float().permute(3, 0, 1, 2)[None],
        w.float().permute(0, 4, 1, 2, 3),
        padding=(p.pD, p.pH, p.pW),
    )[0].permute(1, 2, 3, 0)
    ref = conv + bias.float()
    yf = y.float()
    return _peak_err(yf, ref, 0.0), _peak_err(yf, conv, 0.0)


@pytest.mark.parametrize(
    "key", [k for k, _ in _SHIPPED_FWD], ids=[v[0] for _, v in _SHIPPED_FWD]
)
@pytest.mark.gpu
@pytest.mark.skipif(not _gfx942(), reason="needs torch and a gfx942 device")
def test_winograd3d_shipped_fwd_specs_bias_gpu(key):
    """The shipped forward route specs with the bias epilogue (split-K: in the reduction),
    at input scales 1 and 2^-30 (the prescaled main loop)."""
    pd, spec_json = key
    wtv = dict(_SHIPPED_FWD)[key][1]
    for scale in (1.0, 2.0**-30):
        err, no_bias = _yx_dropin_err(json.loads(spec_json), wtv, pd, scale)
        assert err < 5e-2, (scale, err)
        assert no_bias > 0.2, (scale, no_bias)


@pytest.mark.parametrize(
    "variant",
    [
        dict(stage="lds16", waves=8, nsplit=True, n_tile=64, z_tile=2, op_dtype="f16"),
        dict(),  # the register stage, scalar stores
        dict(stage="lds"),
        dict(stage="lds16", op_dtype="f16", single_buf=True),
    ],
    ids=["lds16_scalar_store", "reg_scalar_store", "lds", "lds16_single_buf"],
)
@pytest.mark.gpu
@pytest.mark.skipif(not _gfx942(), reason="needs torch and a gfx942 device")
def test_winograd3d_scalar_store_bias_gpu(variant):
    err, no_bias = _yx_dropin_err(variant, 1, 1, 1.0)
    assert err < 5e-2, err
    assert no_bias > 0.2, no_bias


@pytest.mark.gpu
@pytest.mark.skipif(not _gfx942(), reason="needs torch and a gfx942 device")
def test_winograd3d_epi_vec_bias_gpu():
    from kernels.conv import conv_winograd3d_select as sel

    err, no_bias = _yx_dropin_err(dict(sel.SPEC_B), 3, 1, 1.0)
    assert err < 5e-2, err
    assert no_bias > 0.2, no_bias
