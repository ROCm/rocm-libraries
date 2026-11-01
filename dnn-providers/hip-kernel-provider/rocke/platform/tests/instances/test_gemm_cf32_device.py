# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the emulated-f32 GEMM routes (cf32 and f32) on gfx950.

Every case plans through :func:`rocke.dispatch.gemm.cf32.plan_cf32_gemm`, runs
once on random data with a NaN-poisoned D, and checks every output (or every
output of the checked rows) against a complex128 reference with the f32-class
bound of the ``x6`` split.

* ``big_b``: B spans more than the 2 GiB DMA buffer range. The default
  addressing read zeros past it (wrong outputs, no error);
  ``test_default_addressing_is_wrong_past_2gib`` forces it on the same problem
  and requires wrong outputs, so the check sees the defect.
* ``big_d``: D has more than 2**31 elements (i32 store offsets would wrap).
* ``ragged``: M and 2 Nc are not tile multiples, on every body of the route;
  D sits in a larger NaN-filled buffer and the region after it must stay NaN
  (the f32 vector epilogue used to store ragged rows past D).
* ``bilinear``: D = alpha A B + beta C with complex scalars, CONJ on B,
  stream-K and data-parallel bodies.
* ``real``: the same bodies and bounds for the real f32 route
  (:func:`rocke.dispatch.gemm.cf32.plan_f32_gemm`), ragged and with alpha /
  bilinear epilogues.

Skipped unless the visible device is a gfx950; the past-i32 cases also need
free device memory.
"""

from __future__ import annotations

import pytest

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

pytestmark = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 device; device is {_ARCH}"
)

_ALPHA = complex(1.1, -0.3)
_BETA = complex(0.7, 0.2)


def _launcher(kd):
    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {
                    "name": prm.name,
                    "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>",
                }
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    art = compile_kernel(kd, arch="gfx950", capture_ir_text=False, backend="python")
    return KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)


def _free_gib():
    import torch

    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def _run(plan, *, conj_b=False, epi="", rows=None, tail=0, seed=5, spec_override=None):
    """(bad, worst, nonfinite, tail_touched) of one launch of ``plan``."""
    import torch
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.launcher import LaunchConfig

    M, Nc, Kc = plan.M, plan.Nc, plan.Kc
    spec = spec_override if spec_override is not None else plan.spec
    kd = build_universal_gemm(spec, "gfx950")
    L = _launcher(kd)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    def crnd(r, c):
        x = torch.rand(r, c, 2, device="cuda", generator=g) * 2 - 1
        return torch.view_as_complex(x)

    A, B = crnd(M, Kc), crnd(Nc, Kc)
    Cc = crnd(M, Nc) if epi == "bilinear" else None
    buf = torch.full((M * 2 * Nc + tail,), float("nan"), device="cuda")
    D = buf[: M * 2 * Nc].view(M, 2 * Nc)
    vals = {
        "A": torch.view_as_real(A).view(torch.bfloat16).view(M, 4 * Kc),
        "B": torch.view_as_real(B).view(torch.bfloat16).view(Nc, 4 * Kc),
        "C": D,
        "M": M,
        "N": 2 * Nc,
        "K": 4 * Kc,
    }
    if epi:
        vals.update(alpha_re=_ALPHA.real, alpha_im=_ALPHA.imag)
    if epi == "bilinear":
        vals.update(
            beta_re=_BETA.real,
            beta_im=_BETA.imag,
            Cin=torch.view_as_real(Cc).view(M, 2 * Nc),
        )
    if spec.trait.streamk:
        fb, pb = plan.workspace_split()
        ws = torch.zeros(fb + pb, dtype=torch.uint8, device="cuda")
        vals.update(
            SkFlags=ws[:fb].view(torch.int32), SkPart=ws[fb:].view(torch.float32)
        )
    L(
        {p.name: vals[p.name] for p in kd.params},
        config=LaunchConfig(grid=plan.grid, block=plan.block, fence=False),
    )
    torch.cuda.synchronize()
    tail_touched = int((~torch.isnan(buf[M * 2 * Nc :])).sum()) if tail else 0
    al = _ALPHA if epi else complex(1, 0)
    bad, worst = 0, 0.0
    nonfinite = 0
    row_ids = torch.arange(M, device="cuda") if rows is None else rows
    Bt = (B.conj() if conj_b else B).to(torch.complex128)
    bbr, bbi = Bt.real.abs(), Bt.imag.abs()
    for r0 in range(0, row_ids.numel(), 1024):
        rr = row_ids[r0 : r0 + 1024]
        Ar = A[rr].to(torch.complex128)
        ref = al * (Ar @ Bt.t())
        aar, aai = Ar.real.abs(), Ar.imag.abs()
        bound = abs(al) * torch.complex(
            aar @ bbr.t() + aai @ bbi.t(), aar @ bbi.t() + aai @ bbr.t()
        )
        if Cc is not None:
            Cr = Cc[rr].to(torch.complex128)
            ref = ref + _BETA * Cr
            bound = bound + abs(_BETA) * torch.complex(Cr.real.abs(), Cr.imag.abs())
        refr = torch.view_as_real(ref)
        bnd = torch.view_as_real(bound)
        tol = (
            2 * 2.0**-24 * refr.abs() + bnd * (2.0**-23 + 2 * Kc * 2.0**-23) + 2.0**-126
        )
        got = D[rr].view(-1, Nc, 2).double()
        nonfinite += int((~torch.isfinite(got)).sum())
        q = torch.nan_to_num((got - refr).abs() / tol, nan=float("inf"))
        bad += int((q > 1).sum())
        worst = max(worst, float(q.max()))
    return bad, worst, nonfinite, tail_touched


def _assert_exact(res):
    bad, worst, nonfinite, tail_touched = res
    assert nonfinite == 0, f"{nonfinite} outputs not written"
    assert bad == 0, f"{bad} wrong outputs (worst {worst:.3g} x tolerance)"
    assert tail_touched == 0, f"{tail_touched} stores past D"


def _big_b_plan():
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    return plan_cf32_gemm(256, 23040, 12288)


def test_big_b_past_dma_range_is_exact():
    if _free_gib() < 16:
        pytest.skip("needs about 16 GiB of free device memory")
    plan = _big_b_plan()
    assert plan.spec.trait.wide_addr
    _assert_exact(_run(plan))


def test_default_addressing_is_wrong_past_2gib():
    """The same problem with the default (32-bit) DMA origins must produce
    wrong outputs: the device check above can see the defect."""
    from dataclasses import replace

    if _free_gib() < 16:
        pytest.skip("needs about 16 GiB of free device memory")
    plan = _big_b_plan()
    # the body's default DMA addressing: one resource over the whole operand,
    # the tile + K origin in soffset (range checked against the 2 GiB DMA range)
    sp = replace(
        plan.spec, trait=replace(plan.spec.trait, wide_addr=False, dtl_soff0=False)
    )
    object.__setattr__(sp, "_prologue", plan.spec._prologue)
    bad, _w, nonfinite, _t = _run(plan, spec_override=sp)
    assert bad > 0 or nonfinite > 0, "default addressing unexpectedly exact past 2 GiB"


def test_big_d_past_i32_elements_is_exact():
    import torch

    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    if _free_gib() < 24:
        pytest.skip("needs about 24 GiB of free device memory")
    M, Nc, Kc = 23552, 46080, 64  # D: 2.17e9 f32 elements
    plan = plan_cf32_gemm(M, Nc, Kc)
    assert plan.spec.trait.wide_addr
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    rows = torch.cat(
        [
            torch.randperm(M, device="cuda", generator=g)[:1024],
            torch.arange(M - 512, M, device="cuda"),
        ]
    )
    rows = torch.unique(rows).sort().values
    _assert_exact(_run(plan, rows=rows))


@pytest.mark.parametrize("body", ["RIW", "RIW_SK", "Q3_SK"])
def test_ragged_bounds(body):
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    plan = plan_cf32_gemm(300, 200, 96, body=body)
    assert plan.spec.trait.pad_m and plan.spec.trait.pad_n
    _assert_exact(_run(plan, tail=1 << 20))


@pytest.mark.parametrize(
    "body,conj_b,M,Nc,Kc",
    [
        ("RIW", False, 512, 384, 256),
        ("RIW", True, 512, 384, 256),
        ("RIW_SK", False, 768, 640, 320),
        ("RIW_SK", True, 300, 200, 96),
        # scatter-epilogue bodies (ragged and aligned, CONJ)
        ("Q3_SK", False, 300, 200, 96),
        ("Q3_SK", True, 512, 384, 256),
        ("Q3_SK", False, 1536, 1440, 576),
        ("C3", True, 300, 200, 96),
    ],
)
def test_bilinear(body, conj_b, M, Nc, Kc):
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    plan = plan_cf32_gemm(M, Nc, Kc, body=body, conj_b=conj_b, epi="bilinear")
    _assert_exact(_run(plan, conj_b=conj_b, epi="bilinear", tail=1 << 16))


@pytest.mark.parametrize("body", ["RIW", "Q3_SK", "C3"])
def test_alpha_only(body):
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    plan = plan_cf32_gemm(512, 384, 256, body=body, epi="alpha")
    _assert_exact(_run(plan, epi="alpha"))


@pytest.mark.parametrize(
    "M,Nc,Kc,epi", [(1000, 1001, 256, "bilinear"), (2100, 1303, 512, "")]
)
def test_route_n_not_multiple_of_8_is_exact(M, Nc, Kc, epi):
    """2 Nc % 8 != 0: the route serves it on a scatter-epilogue body (the
    vector-epilogue bodies refuse it); every output checked, nothing written
    past D."""
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    plan = plan_cf32_gemm(M, Nc, Kc, epi=epi)
    assert plan.body in ("Q3_SK", "C3")
    _assert_exact(_run(plan, epi=epi, tail=1 << 16))


@pytest.mark.parametrize(
    "M,N,K,epi", [(2100, 1999, 512, "bilinear"), (3000, 4001, 256, "")]
)
def test_real_route_n_not_multiple_of_8_is_exact(M, N, K, epi):
    from rocke.dispatch.gemm.cf32 import plan_f32_gemm

    plan = plan_f32_gemm(M, N, K, epi=epi)
    assert plan.body in ("Q3_SK", "C3")
    _assert_exact(_run_real(plan, epi=epi, tail=1 << 16))


def test_bilinear_route_pick_small_is_exact():
    """The product route of a small problem with bilinear (scatter epilogue,
    stream-K, ragged N), every output checked."""
    from rocke.dispatch.gemm.cf32 import plan_cf32_gemm

    plan = plan_cf32_gemm(1536, 1440, 576, epi="bilinear")
    assert plan.body == "Q3_SK"
    _assert_exact(_run(plan, epi="bilinear", tail=1 << 16))


# ---- real f32 (plan_f32_gemm): the same bodies without the complex embedding


def _run_real(plan, *, epi="", tail=0, seed=9):
    """(bad, worst, nonfinite, tail_touched) of one launch of a real f32 plan."""
    import torch
    from rocke.runtime.launcher import LaunchConfig

    M, N, K = plan.M, plan.N, plan.K
    kd = plan.kernel
    L = _launcher(kd)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    A = torch.rand(M, K, device="cuda", generator=g) * 2 - 1
    B = torch.rand(N, K, device="cuda", generator=g) * 2 - 1
    Cin = (
        torch.rand(M, N, device="cuda", generator=g) * 2 - 1
        if epi == "bilinear"
        else None
    )
    al, be = (1.25, -0.5) if epi else (1.0, 0.0)
    buf = torch.full((M * N + tail,), float("nan"), device="cuda")
    D = buf[: M * N].view(M, N)
    vals = {
        "A": A.view(torch.bfloat16),
        "B": B.view(torch.bfloat16),
        "C": D,
        **plan.dims,
    }
    if epi:
        vals["alpha_re"] = al
    if epi == "bilinear":
        vals.update(beta_re=be, Cin=Cin)
    if plan.spec.trait.streamk:
        fb, pb = plan.workspace_split()
        ws = torch.zeros(fb + pb, dtype=torch.uint8, device="cuda")
        vals.update(
            SkFlags=ws[:fb].view(torch.int32), SkPart=ws[fb:].view(torch.float32)
        )
    L(
        {p.name: vals[p.name] for p in kd.params},
        config=LaunchConfig(grid=plan.grid, block=plan.block, fence=False),
    )
    torch.cuda.synchronize()
    tail_touched = int((~torch.isnan(buf[M * N :])).sum()) if tail else 0
    A64, B64 = A.double(), B.double()
    ref = al * (A64 @ B64.t())
    bnd = abs(al) * (A64.abs() @ B64.abs().t())
    if Cin is not None:
        ref = ref + be * Cin.double()
        bnd = bnd + abs(be) * Cin.double().abs()
    tol = 2 * 2.0**-24 * ref.abs() + bnd * (2.0**-23 + 2 * K * 2.0**-23) + 2.0**-126
    got = D.double()
    q = torch.nan_to_num((got - ref).abs() / tol, nan=float("inf"))
    return (
        int((q > 1).sum()),
        float(q.max()),
        int((~torch.isfinite(got)).sum()),
        tail_touched,
    )


@pytest.mark.parametrize("body", ["RIW", "RIW_SK", "Q3_SK"])
def test_real_ragged_bounds(body):
    from rocke.dispatch.gemm.cf32 import plan_f32_gemm

    plan = plan_f32_gemm(300, 392, 96, body=body)
    assert plan.spec.trait.pad_m and plan.spec.trait.pad_n
    _assert_exact(_run_real(plan, tail=1 << 20))


@pytest.mark.parametrize(
    "body,epi",
    [
        ("RIW", "bilinear"),
        ("RIW_SK", "bilinear"),
        ("RIW", "alpha"),
        ("C3", "bilinear"),
        ("C3", "alpha"),
        ("Q3_SK", "bilinear"),
    ],
)
def test_real_epi(body, epi):
    from rocke.dispatch.gemm.cf32 import plan_f32_gemm

    plan = plan_f32_gemm(520, 776, 160, body=body, epi=epi)
    _assert_exact(_run_real(plan, epi=epi, tail=1 << 16))


@pytest.mark.parametrize("epi", ["", "bilinear"])
def test_real_streamk_body_is_exact(epi):
    """The stream-K body on a shape whose stream-K tiles were wrong while the
    accumulators left the AGPRs through ``=a,0`` (partials and f32 stores;
    every output checked)."""
    from rocke.dispatch.gemm.cf32 import plan_f32_gemm

    plan = plan_f32_gemm(4608, 4608, 384, body="RIW_SK", epi=epi)
    assert plan.spec.trait.streamk and plan.spec.trait.mfma_interleave
    _assert_exact(_run_real(plan, epi=epi))
