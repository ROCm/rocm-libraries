# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device numerics of whole plan_tensor_gett plans (the launches the C++
planner, rocke/tensor_gett_planner.h, issues the same way).

Each case runs a plan's launches in order (the tensor_pack steps into
NaN-poisoned workspaces, then the GETT on the packed workspace or the user
tensor per :func:`gett_operand_sources`) on flat user buffers with the
problem's strides, NaN-poisons D first, and checks every output against a
float64 einsum of the strided views:

* the route kinds: MN-layout single launch, the pack hybrid of a K-fastest
  operand (one pack) and of two K-fastest operands, the packed K-contiguous
  route, the K-contiguous single launch on operands that already are its
  K-contiguous copies, padded strides with ragged extents and an extent-1 mode;
* scale and bilinear, alpha = 0 (D = beta Cin) and beta = 0 (D = alpha A B);
* a NaN in A and +Inf / -Inf in B reach exactly the outputs that read them;
* negative control: the K-contiguous GETT launched on MN-contiguous operands is wrong.

Skipped unless the visible device is a gfx950 (the GETT family is gfx950-only).
"""

from __future__ import annotations

import pytest

# torch's HIP runtime is initialised first (the device buffers are torch tensors; the
# kernels launch through rocke's runtime on the same device)
_ARCH = None
try:
    import torch

    if torch.cuda.is_available():
        _ARCH = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
except Exception:  # noqa: BLE001 - no torch / no device
    _ARCH = None

pytestmark = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 device; device is {_ARCH}"
)

_EXT = {"K0": 24, "K1": 24, "M0": 72, "M1": 20, "N0": 60, "N1": 12}
_CAN = (("K1", "K0", "M1", "M0"), ("K1", "K0", "N1", "N0"), ("N1", "N0", "M1", "M0"))
_KFA = (("M1", "M0", "K1", "K0"), ("K1", "K0", "N1", "N0"), ("N1", "N0", "M1", "M0"))
_KFB = (("M1", "M0", "K1", "K0"), ("N1", "N0", "K1", "K0"), ("N1", "N0", "M1", "M0"))
# ragged extents, an extent-1 mode and padded strides (in elements)
_RAG_EXT = {"K0": 40, "K1": 3, "M0": 50, "M1": 7, "N0": 36, "N1": 5, "U": 1}
_RAG = (
    ("K1", "U", "K0", "M1", "M0"),
    ("K1", "K0", "N1", "N0"),
    ("N1", "N0", "U", "M1", "M0"),
)
_RAG_STRIDES = (
    {"M0": 1, "M1": 56, "K0": 56 * 7 + 8, "U": 7, "K1": (56 * 7 + 8) * 40 + 16},
    {"N0": 1, "N1": 40, "K0": 40 * 5 + 24, "K1": (40 * 5 + 24) * 40},
    None,
)


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


def _packed(modes, ext):
    st, s = {}, 1
    for m in reversed(modes):
        st[m] = s
        s *= ext[m]
    return st


def _span(modes, ext, st):
    return 1 + sum((ext[m] - 1) * st[m] for m in modes)


def _req(modes, ext, strides, dtype, bilinear):
    from rocke.dispatch.families.tensor_gett import TensorGettRequest
    from rocke.instances.common.tensor import GettKnobs, TensorGettSpec

    a, b, d = modes
    sa, sb, sd = strides or (None, None, None)
    p = TensorGettSpec.make(
        a,
        b,
        d,
        ext,
        strides_a=sa,
        strides_b=sb,
        strides_d=sd,
        dtype=dtype,
        knobs=GettKnobs(bilinear=bilinear),
    )
    return TensorGettRequest(problem=p, arch="gfx950")


def _kc_single(req):
    """The K-contiguous GETT as one launch on the user operands (they must
    already be its K-contiguous copies)."""
    import rocke.dispatch.families.tensor_gett as TG

    for r in TG.dispatch_tensor_gett_all(req):
        if r.spec.kc_body:
            return TG.TensorGettPlan(
                packs=(), gett=r, workspace_bytes=0, explanation=()
            )
    raise AssertionError("no K-contiguous single-launch candidate")


def _run(plan, req, *, alpha, beta, poison=False):
    """Run ``plan`` on random strided operands: (D, reference) as float64 views."""
    import torch

    from rocke.dispatch.families.tensor_gett import (
        build_pack_step,
        gett_operand_sources,
    )
    from rocke.instances.common.tensor import build_tensor_gett
    from rocke.runtime.launcher import LaunchConfig

    p = req.problem
    ext = dict(p.extents)
    tdt = torch.float16 if p.dtype in ("fp16", "f16") else torch.bfloat16
    g = torch.Generator(device="cuda")
    g.manual_seed(11)
    ops, views = {}, {}
    for nm, modes, st in (
        ("A", p.modes_a, p.strides_a),
        ("B", p.modes_b, p.strides_b),
        ("D", p.modes_d, p.strides_d),
    ):
        st = _packed(modes, ext) if st is None else dict(st)
        n = _span(modes, ext, st)
        # magnitudes in [0.25, 1): no exact zeros (Inf * 0 would make the NaN pattern ambiguous)
        u = torch.rand(n, generator=g, device="cuda")
        s = torch.where(torch.rand(n, generator=g, device="cuda") < 0.5, -1.0, 1.0)
        flat = (s * (0.25 + 0.75 * u)).to(tdt)
        if nm == "D":
            cin = flat.clone()
            flat = torch.full((n,), float("nan"), device="cuda", dtype=tdt)
            ops["Cin"] = cin
        ops[nm] = flat
        views[nm] = (modes, [ext[m] for m in modes], [st[m] for m in modes])

    def view(t, nm):
        modes, size, stride = views[nm]
        return torch.as_strided(t, size, stride)

    if poison:
        va, vb = view(ops["A"], "A"), view(ops["B"], "B")
        va[(0,) * len(p.modes_a)] = float("nan")
        idx = [ext[m] - 1 for m in p.modes_b]
        vb[tuple(idx)] = float("inf")
        idx[-1] = 0
        vb[tuple(idx)] = float("-inf")

    work = []
    for step in plan.packs:
        kd = build_pack_step(step, "gfx950")
        Y = torch.full(
            (step.workspace_bytes // 2,), float("nan"), device="cuda", dtype=tdt
        )
        _launcher(kd)(
            {"X": ops[step.operand], "Y": Y},
            config=LaunchConfig(grid=tuple(step.grid), block=tuple(step.block)),
        )
        work.append(Y)
    spec = plan.gett.spec
    pl = spec.plan()
    src = {}
    for role, (kind, ref) in zip(("A", "B"), gett_operand_sources(plan)):
        src[role] = work[ref] if kind == "pack" else ops[ref]
    vals = {
        "A": src["A"],
        "B": src["B"],
        "C": ops["D"],
        "M": pl.M,
        "N": pl.N,
        "K": pl.K,
        "alpha": alpha,
        "Cin": ops["Cin"],
        "beta": beta,
    }
    kd = build_tensor_gett(spec)
    unknown = [q.name for q in kd.params if q.name not in vals]
    assert not unknown, f"unexpected GETT params {unknown}"
    _launcher(kd)(
        {q.name: vals[q.name] for q in kd.params},
        config=LaunchConfig(
            grid=tuple(plan.gett.grid), block=tuple(plan.gett.block), fence=False
        ),
    )
    torch.cuda.synchronize()

    letters = {m: chr(ord("a") + i) for i, m in enumerate(sorted(ext))}
    eq = "{},{}->{}".format(
        *("".join(letters[m] for m in ms) for ms in (p.modes_a, p.modes_b, p.modes_d))
    )
    A64 = view(ops["A"], "A").double()
    B64 = view(ops["B"], "B").double()
    acc = torch.einsum(eq, A64, B64)
    absb = torch.einsum(eq, A64.abs(), B64.abs())
    ref = alpha * acc
    bound = abs(alpha) * absb
    if p.knobs.bilinear:
        C64 = view(ops["Cin"], "D").double()
        ref = ref + beta * C64
        bound = bound + abs(beta) * C64.abs()
    return view(ops["D"], "D").double(), ref, bound, p.dtype


def _check(D, ref, bound, dtype, K):
    import torch

    u = 2.0**-11 if dtype in ("fp16", "f16") else 2.0**-8
    nan_ref, nan_d = torch.isnan(ref), torch.isnan(D)
    assert torch.equal(
        nan_ref, nan_d
    ), f"NaN pattern differs: {int(nan_ref.sum())} expected, {int(nan_d.sum())} written"
    inf_ref = torch.isinf(ref)
    assert torch.equal(inf_ref, torch.isinf(D)), "Inf pattern differs"
    assert torch.equal(ref[inf_ref], D[inf_ref]), "Inf signs differ"
    fin = torch.isfinite(ref)
    tol = 2 * u * ref[fin].abs() + 4 * 2.0**-24 * K * bound[fin] + 2.0**-24
    q = (D[fin] - ref[fin]).abs() / tol
    bad = int((q > 1).sum())
    assert bad == 0, f"{bad} wrong outputs (worst {float(q.max()):.3g} x tolerance)"


def _k(req):
    p = req.problem
    ext = dict(p.extents)
    k = 1
    for m in p.modes_a:
        if m in p.modes_b and m not in p.modes_d:
            k *= ext[m]
    return k


_CASES = {
    "mn": (_CAN, _EXT, None),
    "k_fastest_a": (_KFA, _EXT, None),
    "k_fastest_both": (_KFB, _EXT, None),
    "ragged_padded": (_RAG, _RAG_EXT, _RAG_STRIDES),
}


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("name", sorted(_CASES))
@pytest.mark.parametrize("bilinear", [False, True])
def test_plan_routes_are_exact(name, dtype, bilinear):
    from rocke.dispatch.families.tensor_gett import plan_tensor_gett

    modes, ext, strides = _CASES[name]
    req = _req(modes, ext, strides, dtype, bilinear)
    plan = plan_tensor_gett(req)
    if name.startswith("k_fastest"):
        assert plan.packs, "a K-fastest operand is read through a pack"
    _check(*_run(plan, req, alpha=0.75, beta=-1.25), _k(req))


@pytest.mark.parametrize("route", ["mn", "kc"])
def test_forced_routes_are_exact(route, monkeypatch):
    import rocke.dispatch.families.tensor_gett as TG

    req = _req(_KFB, _EXT, None, "bf16", True)
    plan = TG._plan_mn(req) if route == "mn" else TG._plan_kc(req)
    assert plan is not None
    if route == "kc":
        assert plan.gett.spec.kc_body and len(plan.packs) == 2
    _check(*_run(plan, req, alpha=1.5, beta=0.5), _k(req))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_k_contiguous_single_launch_on_its_own_copies(dtype):
    import rocke.dispatch.families.tensor_gett as TG
    from rocke.instances.common.tensor import kc_operand_modes, launch_operands

    base = _req(_CAN, _EXT, None, dtype, True)
    sp = TG._spec_kc(base)
    orders = dict(zip(launch_operands(sp), kc_operand_modes(sp)))
    st = (_packed(orders["A"], _EXT), _packed(orders["B"], _EXT), None)
    req = _req(_CAN, _EXT, st, dtype, True)
    plan = _kc_single(req)
    _check(*_run(plan, req, alpha=0.75, beta=-1.25), _k(req))


def test_k_contiguous_single_launch_on_mn_operands_is_wrong():
    """The K-contiguous GETT launched once on MN-contiguous user operands (what the registry
    candidate did before it required its own copies) must produce wrong outputs: the device
    check above can see a wrong route."""
    import torch

    import rocke.dispatch.families.tensor_gett as TG
    from rocke.dispatch.families.tensor_gett import TensorGettPlan

    req = _req(_CAN, _EXT, None, "bf16", False)
    sp = TG._spec_kc(req)
    assert not TG._kc_in_place(req.problem, sp)
    r = TG.dispatch_tensor_gett(req)
    plan = TensorGettPlan(
        packs=(), gett=_replace_spec(r, sp), workspace_bytes=0, explanation=()
    )
    D, ref, bound, dt = _run(plan, req, alpha=1.0, beta=0.0)
    u = 2.0**-8
    tol = 2 * u * ref.abs() + 4 * 2.0**-24 * _k(req) * bound + 2.0**-24
    bad = int((torch.nan_to_num((D - ref).abs() / tol, nan=float("inf")) > 1).sum())
    assert bad > 0, "a K-contiguous launch on MN-contiguous operands was exact"


def _replace_spec(r, sp):
    """The dispatch result ``r`` with the GETT spec ``sp`` (its grid and block)."""
    from dataclasses import replace

    from rocke.instances.common.tensor import tensor_gett_grid

    return replace(
        r, spec=sp, grid=tuple(tensor_gett_grid(sp)), block=(sp.body().block_size, 1, 1)
    )


@pytest.mark.parametrize("alpha,beta", [(0.0, 1.0), (0.75, 0.0)])
def test_zero_scales(alpha, beta):
    from rocke.dispatch.families.tensor_gett import plan_tensor_gett

    req = _req(_KFA, _EXT, None, "fp16", True)
    _check(*_run(plan_tensor_gett(req), req, alpha=alpha, beta=beta), _k(req))


@pytest.mark.parametrize("name", ["mn", "k_fastest_a"])
def test_nan_and_inf_reach_exactly_their_outputs(name):
    from rocke.dispatch.families.tensor_gett import plan_tensor_gett

    modes, ext, strides = _CASES[name]
    req = _req(modes, ext, strides, "bf16", False)
    D, ref, bound, dt = _run(
        plan_tensor_gett(req), req, alpha=1.0, beta=0.0, poison=True
    )
    assert int(torch_isnan_count(ref)) > 0
    _check(D, ref, bound, dt, _k(req))


def torch_isnan_count(t):
    import torch

    return torch.isnan(t).sum()
