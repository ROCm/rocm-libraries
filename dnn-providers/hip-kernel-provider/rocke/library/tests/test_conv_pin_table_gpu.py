# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The shipped conv pin and route tables' kernels on a gfx942 device.

Every distinct kernel label the shipped tables name is rebuilt from the label
alone (:func:`kernels.conv.aot_jobs.job_for_identity`), compiled into a
scratch AOT cache and launched the way a pinned route launches it
(:class:`benchmarks.common.conv_route_exec.RouteBinding`) on its table
problem with the depth cut down (the same form: padding, channels, height and
width; a tail split, whose full-tile count is the problem's own, on the table
problem), and compared with a torch reference. The x-halo y-ring dgrad kernels
are also launched repeatedly and must give bitwise-identical output.
"""

from __future__ import annotations

import json
import re

import pytest

torch = pytest.importorskip("torch")

from dispatch import conv_pins as cp  # noqa: E402
from dispatch import conv_routes as cr  # noqa: E402


def _gfx942():
    if not torch.cuda.is_available():
        return False
    return "gfx942" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


pytestmark = pytest.mark.skipif(not _gfx942(), reason="needs a gfx942 device")

_ARCH = "gfx942"
# Output depth of the cut-down problem: enough for every depth tile and box
# split of the pinned kernels to see more than one slab.
_DEPTH = 6


def _shipped_cases() -> dict:
    """Pin label (with launch) -> the first table case that names it."""
    out: dict = {}
    tables = list(cp.shipped_conv_pin_tables().values())
    tables += list(cr.shipped_conv_route_tables().values())
    for path in tables:
        doc = json.loads(path.read_text())
        for mode, body in doc["modes"].items():
            for case, row in body["cases"].items():
                if row.get("kernel") and row.get("route", "direct") == "direct":
                    out.setdefault(
                        row["kernel"].strip(),
                        (f"{path.stem}#{mode}:{case}", row["cmd"]),
                    )
    return out


_CASES = _shipped_cases()


def _cut_depth(cmd: str) -> str:
    """``cmd`` with its input depth cut so the output depth is ``_DEPTH``."""
    pd = int(re.search(r"--pad_d (\d+)", cmd).group(1))
    return re.sub(r"--in_d \d+", f"--in_d {_DEPTH + 2 - 2 * pd}", cmd)


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    from benchmarks.common.kernel_cache import KernelCache

    return KernelCache(tmp_path_factory.mktemp("pinned_cache"), _ARCH)


def _compiled(cache, label: str):
    from benchmarks.common.kernel_sweep import compile_jobs
    from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
    from kernels.conv.kernel_identity import KernelIdentity

    ident = KernelIdentity.from_label(label.split()[0])
    if cache.get(ident) is None:
        job = job_for_identity(ident)
        compile_jobs(
            cache=cache,
            all_jobs=[job],
            build=kernel_for_job,
            arch=_ARCH,
            dtype=ident.dtype_a,
            directions=[ident.direction],
            jobs=1,
            log=lambda *a, **k: None,
        )
    assert cache.get(ident) is not None, label


def _plan(label: str, cmd: str):
    routes = cr.routes_from_mapping(
        {"cases": {"pin": {"cmd": cmd, "route": "direct", "kernel": label}}},
        source="gpu-test",
    )
    req = cr.request_for_route(routes.by_case()["pin"], _ARCH)
    return cr.dispatch_conv_route(req, routes)


def _operands(plan, seed: int = 0):
    p = plan.problem
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs, ys = (p.Di, p.Hi, p.Wi, p.C), (p.Do, p.Ho, p.Wo, p.K)
    ws = (p.K, p.Z, p.Y, p.X, p.C)

    def rnd(shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(
            torch.bfloat16
        )

    wscale = (2 / (27 * p.C)) ** 0.5
    if plan.direction == "fwd":
        return rnd(xs), rnd(ws, wscale), xs, ys
    if plan.direction == "dgrad":
        return rnd(ys), rnd(ws, wscale), ys, xs
    return rnd(ys), rnd(xs), ys, ws


def _reference(plan, A, Bt):
    p = plan.problem
    pad = (p.pD, p.pH, p.pW)
    if plan.direction == "fwd":
        x, w = A.float().permute(3, 0, 1, 2)[None], Bt.float().permute(0, 4, 1, 2, 3)
        return torch.nn.functional.conv3d(x, w, padding=pad)[0].permute(1, 2, 3, 0)
    if plan.direction == "dgrad":
        dy, w = A.float().permute(3, 0, 1, 2)[None], Bt.float().permute(0, 4, 1, 2, 3)
        dx = torch.nn.grad.conv3d_input((1, p.C, p.Di, p.Hi, p.Wi), w, dy, padding=pad)
        return dx[0].permute(1, 2, 3, 0)
    dy, x = A.float().permute(3, 0, 1, 2)[None], Bt.float().permute(3, 0, 1, 2)[None]
    dw = torch.nn.grad.conv3d_weight(x, (p.K, p.C, p.Z, p.Y, p.X), dy, padding=pad)
    return dw.permute(0, 2, 3, 4, 1)


def _run(plan, cache, rt, A, Bt, out_shape, repeats: int = 1, check=None):
    """Launch the plan ``repeats`` times; the first output is returned, every
    later one handed to ``check(first, out)``."""
    from benchmarks.common.conv_route_exec import RouteBinding

    first = None
    D = torch.full(out_shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    rb = RouteBinding(
        plan,
        cache,
        rt,
        _ARCH,
        ptrs=(A.data_ptr(), Bt.data_ptr(), D.data_ptr()),
        sizes=tuple(t.numel() * 2 for t in (A, Bt, D)),
    )
    try:
        for _ in range(repeats):
            D.fill_(float("nan"))
            torch.cuda.synchronize()
            rb.run()
            rt.sync()
            if first is None:
                first = D.clone()
            else:
                check(first, D)
    finally:
        rt.sync()
        rb.free()
    return first


@pytest.mark.parametrize("label", sorted(_CASES))
def test_shipped_pinned_kernel_matches_the_reference(cache, rt, label):
    where, cmd = _CASES[label]
    _compiled(cache, label)
    # A tail split's full-tile count is the table problem's own.
    plan = _plan(label, cmd if "/full" in label else _cut_depth(cmd))
    A, Bt, _, out_shape = _operands(plan)
    D = _run(plan, cache, rt, A, Bt, out_shape)
    ref = _reference(plan, A, Bt)
    assert not torch.isnan(D).any(), f"{where}: unwritten output"
    err = (D.float() - ref).abs()
    scale = max(ref.abs().max().item(), 1.0)
    assert err.max().item() / scale < 5e-2, where
    assert err.mean().item() / ref.abs().mean().item() < 1e-2, where


# The y-ring kernels whose ring is exactly the A tile (input width 130): the
# geometry where a stray store into a ring slot races the first tile's load.
_YRING = sorted(lab for lab in _CASES if "-xyr130-" in lab and lab.startswith("dgrad-"))
# The race showed on a small fraction of calls; enough launches to see it.
_YRING_REPEATS = 400


@pytest.mark.parametrize("label", _YRING)
def test_yring_dgrad_is_bitwise_repeatable(cache, rt, label):
    """The y-ring dgrad stages input rows in an LDS ring that no other store
    may touch: repeated launches on the table problem give the same bits."""
    where, cmd = _CASES[label]
    _compiled(cache, label)
    plan = _plan(label, cmd)
    A, Bt, _, out_shape = _operands(plan, seed=1)
    bad = []

    def same(first, out):
        if not torch.equal(out.view(torch.int16), first.view(torch.int16)):
            bad.append(int((out.view(torch.int16) != first.view(torch.int16)).sum()))

    _run(plan, cache, rt, A, Bt, out_shape, repeats=_YRING_REPEATS, check=same)
    assert (
        not bad
    ), f"{where}: {len(bad)} of {_YRING_REPEATS - 1} launches differ {bad[:8]}"


def test_the_tables_name_yring_kernels():
    assert _YRING
