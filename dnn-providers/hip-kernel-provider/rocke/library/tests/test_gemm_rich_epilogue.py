# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks of the gfx950 rich GEMM epilogues
(``kernels.gfx950.gemm_rich_epilogue``) and their ``spec._mono`` hook.

Pure codegen (no GPU): knob validation, the hook's kernel params, that an
absent hook leaves the universal GEMM body byte-identical, that the hook
replaces the body's own tile store, and that each knob changes the emission.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from kernels.gfx950.gemm_rich_epilogue import (
    BODIES,
    EpiHooks,
    EpiKnobs,
    RichEpilogueSpec,
    body_spec,
    build_rich_epilogue,
    rich_epilogue_gemm_spec,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import build_universal_gemm, is_valid_spec


def build_epi(knobs, dtype="f16", body="HBR"):
    """(KernelDef, universal GEMM spec with the hook, hook) of one rich epilogue."""
    spec = RichEpilogueSpec(knobs=knobs, dtype=dtype, body=body)
    gs = rich_epilogue_gemm_spec(spec)
    return build_rich_epilogue(spec, arch="gfx950"), gs, gs._mono


def _ir(kd):
    return lower_kernel_to_llvm(kd, arch="gfx950").replace(kd.name, "KNAME")


@pytest.mark.parametrize("body", sorted(BODIES))
def test_bodies_are_valid_and_unhooked_body_is_unchanged(body):
    sp = body_spec("bf16", body=body)
    ok, why = is_valid_spec(sp, arch="gfx950")
    assert ok, why
    # without the hook the side channel is absent: the same body as a plain spec
    a = build_universal_gemm(sp, "gfx950")
    b = build_universal_gemm(dataclasses.replace(sp), "gfx950")
    assert _ir(a) == _ir(b)
    assert [p.name for p in a.params] == ["A", "B", "C", "M", "N", "K"]


@pytest.mark.parametrize(
    "knobs,params",
    [
        (EpiKnobs(), ["alpha", "ldd"]),
        (EpiKnobs(beta="same"), ["alpha", "ldd", "Cin", "beta", "ldc"]),
        (EpiKnobs(bias="col", act="silu"), ["alpha", "ldd", "Bias"]),
        (EpiKnobs(gate=True, resid=True), ["alpha", "ldd", "G", "R"]),
        (EpiKnobs(out="fp8"), ["alpha", "ldd", "Q", "scale"]),
        (EpiKnobs(out="fp8b", qblock=32), ["alpha", "ldd", "Q", "S"]),
        (EpiKnobs(amax="row", out="none"), ["alpha", "ldd", "Amax"]),
        (
            EpiKnobs(rowsum=True, rowsq=True, colsum=True, zero_next=True),
            ["alpha", "ldd", "RowSum", "RowSq", "ColSum", "slot"],
        ),
    ],
)
def test_hook_params(knobs, params):
    kd, sp, hooks = build_epi(knobs, "f16")
    assert [p.name for p in kd.params] == ["A", "B", "C", "M", "N", "K"] + params
    assert sp._mono is hooks
    assert kd.name.startswith("gemm_rich_epi_")


@pytest.mark.parametrize(
    "kw",
    [
        dict(swiglu=True, act="relu"),
        dict(swiglu=True, beta="same"),
        dict(out="fp8b", qblock=16),
        dict(out="fp8", swiglu=True),
    ],
)
def test_invalid_knobs_rejected(kw):
    with pytest.raises(ValueError):
        EpiHooks(EpiKnobs(**kw))


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(aux_pipe=-1), "aux_pipe must be >= 0"),
        (dict(aux_pipe=1.0), "aux_pipe must be an int"),
        (dict(aux_pipe=True), "aux_pipe must be an int"),
        (dict(qblock=96), "qblock must be 32, 64 or 128"),
        (dict(gate="false"), "gate must be a bool"),
        (dict(fast_math=1), "fast_math must be a bool"),
        (dict(zero_next=None), "zero_next must be a bool"),
        (dict(beta="T"), "beta must be one of"),
        (dict(act="gelu_tanh"), "act must be one of"),
        (dict(out=""), "out must be one of"),
        (dict(amax=False), "amax must be one of"),
    ],
)
def test_knobs_reject_instead_of_coercing(kw, match):
    """A malformed knob is an error at construction, never a different valid
    kernel (no clamping of aux_pipe, no truthiness of a string flag)."""
    with pytest.raises(ValueError, match=match):
        EpiKnobs(**kw)


def test_hook_replaces_the_tile_store_and_knobs_change_emission():
    base_kd = build_universal_gemm(body_spec("f16"), "gfx950")
    seen = {_ir(base_kd)}
    for kn in (
        EpiKnobs(),
        EpiKnobs(store_nt=False),
        EpiKnobs(bias="col"),
        EpiKnobs(bias="row"),
        EpiKnobs(act="gelu"),
        EpiKnobs(act="gelu", fast_math=False),
        EpiKnobs(beta="t"),
        EpiKnobs(beta="t", ct_vec=False),
        EpiKnobs(rowsum=True),
        EpiKnobs(rowsum=True, red_scatter=False),
        EpiKnobs(beta="same", aux_pipe=1),
        EpiKnobs(swiglu=True),
        EpiKnobs(out="fp8b", qblock=64),
    ):
        kd, _sp, _h = build_epi(kn, "f16")
        ir = _ir(kd)
        assert ir not in seen, kn.tag()
        seen.add(ir)


def test_builder_contract_and_arch_guard():
    spec = RichEpilogueSpec(knobs=EpiKnobs(bias="col"), dtype="bf16")
    kd = build_rich_epilogue(spec, arch="gfx950")
    assert kd.name.startswith(spec.kernel_name())
    assert RichEpilogueSpec().kernel_name() != spec.kernel_name()
    with pytest.raises(ValueError, match="arch"):
        build_rich_epilogue(spec, arch="gfx942")


def test_reductions_use_device_atomics():
    kd, _sp, _h = build_epi(EpiKnobs(amax="row", rowsum=True, out="none"), "bf16")
    ir = _ir(kd)
    assert "global_atomic_umax" in ir
    assert "atomicrmw fadd" in ir
    # every inline-asm VMEM atomic on an SGPR base starts with the hazard pad
    sites = re.findall(r'"([^"]*)global_atomic_umax', ir)
    assert sites and all(p.startswith("s_nop 4") for p in sites), sites


# ----------------------------------------------------------------- GPU numerics
# Run only on a gfx950 with ROCm torch (skipped elsewhere); torch is imported
# lazily so the codegen tests above stay torch-free.


def _gfx950_torch():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName.lower():
        return None
    return torch


requires_gfx950_gpu = pytest.mark.skipif(
    _gfx950_torch() is None, reason="needs a gfx950 GPU with ROCm torch"
)

_M, _N, _K = 512, 512, 128


_LAUNCHERS = {}


def _launch(knobs, dtype, vals):
    """Compile (once per knobs / dtype) and run one rich epilogue (persistent
    grid of 256 x 256)."""
    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    torch = _gfx950_torch()
    key = (knobs, dtype)
    if key not in _LAUNCHERS:
        spec = RichEpilogueSpec(knobs=knobs, dtype=dtype)
        kd = build_rich_epilogue(spec, arch="gfx950")
        art = compile_kernel(kd, arch="gfx950", backend="python", capture_ir_text=False)
        sig = []
        for p in kd.params:
            nm = p.type.name
            if isinstance(p.type, PtrType):
                nm = nm if nm.startswith("ptr<") else f"ptr<{nm}>"
            sig.append({"name": p.name, "type": nm})
        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
        )
        _LAUNCHERS[key] = (launcher, [p.name for p in kd.params])
    launcher, names = _LAUNCHERS[key]
    launcher(
        {n: vals[n] for n in names},
        config=LaunchConfig(grid=(256, 1, 1), block=(256, 1, 1)),
    )
    torch.cuda.synchronize()


def _operands(dtype, seed):
    torch = _gfx950_torch()
    tdt = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype]
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return (torch.rand(*shape, device="cuda", generator=g) * 2 - 1).to(tdt)

    A, B = rnd(_M, _K), rnd(_N, _K)
    acc = A.double() @ B.double().t()
    absacc = A.double().abs() @ B.double().abs().t()
    return tdt, rnd, A, B, acc, absacc


def _tol(dtype, y, absz):
    # one rounding at the output, f32 accumulation over K, a few ulp of f32 for
    # the activation (v_exp_f32 / v_rcp_f32), and an absolute floor where the
    # result underflows: half the smallest f16 subnormal, or the smallest f32
    # normal for bf16 (f32 results below it may be flushed to zero)
    u_out = {"f16": 2.0**-11, "bf16": 2.0**-8}[dtype]
    floor = {"f16": 2.0**-25, "bf16": 2.0**-126}[dtype]
    return (
        u_out * y.abs() + 1.25 * (_K + 8) * 2.0**-23 * absz + 2.0**-19 * y.abs() + floor
    )


@requires_gfx950_gpu
@pytest.mark.parametrize("dtype,alpha", [("f16", 1.0), ("f16", 40.0), ("bf16", 1.0e6)])
@pytest.mark.parametrize("act", ["silu", "gelu"])
@pytest.mark.parametrize("fast_math", [True, False])
def test_gpu_activation_matches_f64_over_a_wide_range(dtype, alpha, act, fast_math):
    """``_sigmoid_mul`` (SiLU, tanh-form GeLU) against an f64 reference with the
    activation argument spanning large magnitudes of both signs: the fast path's
    ``v_exp_f32`` saturates to +inf / 0 at the ends and ``x / (1 + 2^t)`` keeps
    the right limits (0 for large negative, x for large positive)."""
    torch = _gfx950_torch()
    import torch.nn.functional as F

    tdt, rnd, A, B, acc, absacc = _operands(dtype, 7)
    # a column bias sweeping the activation input over the whole range
    span = {"f16": 120.0, "bf16": 3.0e8}[dtype]
    bias = torch.linspace(-span, span, _N, device="cuda").to(tdt)
    z = alpha * acc + bias.double()
    absz = alpha * absacc + bias.double().abs()
    y = F.silu(z) if act == "silu" else F.gelu(z, approximate="tanh")
    D = torch.empty(_M, _N, dtype=tdt, device="cuda")
    knobs = EpiKnobs(bias="col", act=act, fast_math=fast_math)
    _launch(
        knobs,
        dtype,
        dict(A=A, B=B, C=D, M=_M, N=_N, K=_K, alpha=alpha, ldd=_N, Bias=bias),
    )
    finite = torch.isfinite(y.to(tdt))
    assert bool(torch.isfinite(D)[finite].all())
    err = (D.double() - y).abs()[finite]
    tol = (1.25 * _tol(dtype, y, absz))[finite]
    assert float((err / tol).max()) <= 1.0


@requires_gfx950_gpu
@pytest.mark.parametrize("red_scatter", [True, False])
def test_gpu_reductions_cross_lane_exchanges(red_scatter):
    """Column sums (reduce-scatter over the 4 row groups: lane xor 16 and 32),
    the global amax (xor 1 .. 32) and row sums (xor 1 .. 8) against f64."""
    torch = _gfx950_torch()
    dtype = "f16"
    tdt, rnd, A, B, acc, absacc = _operands(dtype, 11)
    alpha = 0.75
    y, absz = alpha * acc, alpha * absacc
    tol = _tol(dtype, y, absz)
    D = torch.empty(_M, _N, dtype=tdt, device="cuda")
    amax = torch.zeros(1, dtype=torch.int32, device="cuda")
    rowsum = torch.zeros(_M, dtype=torch.float32, device="cuda")
    colsum = torch.zeros(_N, dtype=torch.float32, device="cuda")
    knobs = EpiKnobs(amax="all", rowsum=True, colsum=True, red_scatter=red_scatter)
    _launch(
        knobs,
        dtype,
        dict(
            A=A,
            B=B,
            C=D,
            M=_M,
            N=_N,
            K=_K,
            alpha=alpha,
            ldd=_N,
            Amax=amax,
            RowSum=rowsum,
            ColSum=colsum,
        ),
    )
    assert float(((D.double() - y).abs() / tol).max()) <= 1.0
    got_amax = float(amax.view(torch.float32)[0])
    assert abs(got_amax - float(y.abs().max())) <= float(tol.max())
    col_tol = tol.sum(0) + _M * 2.0**-23 * y.abs().sum(0)
    row_tol = tol.sum(1) + _N * 2.0**-23 * y.abs().sum(1)
    assert float(((colsum.double() - y.sum(0)).abs() / col_tol).max()) <= 1.0
    assert float(((rowsum.double() - y.sum(1)).abs() / row_tol).max()) <= 1.0


@requires_gfx950_gpu
def test_gpu_global_amax_finds_a_single_outlier_anywhere():
    """One exact outlier (an all-ones row of A times an all-ones row of B gives
    K, every other output is a random sum far below it) at positions owned by
    lanes of both wave halves and of every row group: the global amax must be
    exactly alpha * K each time, so every butterfly stage (xor 1 .. 32) counts."""
    torch = _gfx950_torch()
    dtype = "f16"
    tdt, rnd, A0, B0, _acc, _absacc = _operands(dtype, 13)
    alpha = 0.75
    knobs = EpiKnobs(amax="all")
    for i, j in (
        (0, 3),
        (37, 77),
        (100, 130),
        (129, 255),
        (200, 260),
        (255, 333),
        (300, 400),
        (511, 509),
        (17, 18),
        (48, 63),
    ):
        A, B = A0.clone(), B0.clone()
        A[i].fill_(1.0)
        B[j].fill_(1.0)
        ref = (A.double() @ B.double().t()).abs()
        assert int(ref.argmax()) == i * _N + j  # the outlier is the unique max
        D = torch.empty(_M, _N, dtype=tdt, device="cuda")
        amax = torch.zeros(1, dtype=torch.int32, device="cuda")
        _launch(
            knobs,
            dtype,
            dict(A=A, B=B, C=D, M=_M, N=_N, K=_K, alpha=alpha, ldd=_N, Amax=amax),
        )
        assert float(amax.view(torch.float32)[0]) == alpha * _K, (i, j)
