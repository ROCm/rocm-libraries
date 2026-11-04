# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""dispatch.conv_routes on a gfx942 device: the route kernels bound by
benchmarks.common.conv_route_exec.RouteBinding (no AOT cache needed)."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from dispatch import conv_routes as cr  # noqa: E402


def _gfx942():
    if not torch.cuda.is_available():
        return False
    return "gfx942" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


pytestmark = pytest.mark.skipif(not _gfx942(), reason="needs a gfx942 device")

ZSPEC = {"stage": "lds16", "op_dtype": "f16", "fill_il": True, "waves": 8, "nsplit": False, "z_tile": 4,
         "n_tile": 32, "nfast": True, "pkrtz": True, "epi_vec": True, "yf16": True, "xcd": 6, "bps": 1,
         "bps_halo": 1}
BSPEC = dict(ZSPEC, nsplit=True, z_tile=2, n_tile=64)


def _cmd(F, C, K, D, H, W, pd):
    return (f"MIOpenDriver convbfp16 -n 1 -c {C} --in_d {D} -H {H} -W {W} -k {K} --fil_d 3 -y 3 -x 3 "
            f"--pad_d {pd} -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 "
            f"--in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F {F} -t 1 -V 0")


ROUTES = cr.routes_from_mapping({"cases": {
    "fwd_pad1": {"cmd": _cmd(1, 128, 128, 32, 16, 16, 1), "route": "winograd",
                 "winograd": {"spec": ZSPEC, "wtv": 4}},
    "fwd_pad1_ks": {"cmd": _cmd(1, 256, 128, 32, 16, 16, 1), "route": "winograd",
                    "winograd": {"spec": dict(ZSPEC, ksplit=4), "wtv": 4}},
    "dgrad_pad011": {"cmd": _cmd(2, 128, 128, 34, 16, 16, 0), "route": "winograd",
                     "winograd": {"spec": dict(BSPEC, bps_halo_z=2), "wtv": 3}},
}}, source="gpu-test")


class _Rt:
    """The Runtime calls RouteBinding makes, on torch memory."""

    def __init__(self):
        self.keep = {}

    def alloc(self, n):
        t = torch.empty(n, dtype=torch.uint8, device="cuda")
        self.keep[t.data_ptr()] = t
        return t.data_ptr()

    def memset(self, p, v, n):
        self.keep[p].fill_(v)

    def free(self, p):
        self.keep.pop(p, None)


def _bind(plan, A, Bt, D):
    from benchmarks.common.conv_route_exec import RouteBinding

    return RouteBinding(plan, None, _Rt(), "gfx942", ptrs=(A.data_ptr(), Bt.data_ptr(), D.data_ptr()),
                        sizes=tuple(t.numel() * 2 for t in (A, Bt, D)), B=object())


def _req(case):
    return cr.request_for_route(ROUTES.by_case()[case], "gfx942")


@pytest.mark.parametrize("case", ["fwd_pad1", "fwd_pad1_ks", "dgrad_pad011"])
def test_winograd_route_matches_dropin_and_reference(case):
    from kernels.common.conv_winograd3d_dropin import Winograd3dDropin

    plan = cr.dispatch_conv_route(_req(case), ROUTES)
    p = plan.problem
    g = torch.Generator(device="cuda").manual_seed(0)
    xs, ys = (p.Di, p.Hi, p.Wi, p.C), (p.Do, p.Ho, p.Wo, p.K)
    A = torch.randn(*(xs if plan.direction == "fwd" else ys), device="cuda", generator=g).to(torch.bfloat16)
    Bt = (torch.randn(p.K, 3, 3, 3, p.C, device="cuda", generator=g) * (2 / (27 * p.C)) ** 0.5).to(torch.bfloat16)
    D = torch.full(ys if plan.direction == "fwd" else xs, float("nan"), device="cuda", dtype=torch.bfloat16)
    rb = _bind(plan, A, Bt, D)
    rb.run()
    torch.cuda.synchronize()
    ref_out = Winograd3dDropin(p, plan.direction, plan.winograd)(A, Bt)
    torch.cuda.synchronize()
    assert torch.equal(D.view(torch.int16), ref_out.view(torch.int16))
    x = A.float().permute(3, 0, 1, 2)[None]
    w = Bt.float().permute(0, 4, 1, 2, 3)
    pad = (p.pD, p.pH, p.pW)
    f = torch.nn.functional.conv3d if plan.direction == "fwd" else torch.nn.functional.conv_transpose3d
    ref = f(x, w, padding=pad)[0].permute(1, 2, 3, 0)
    err = (D.float() - ref).abs()
    assert err.max().item() / max(ref.abs().max().item(), 1.0) < 5e-2
    assert err.mean().item() / ref.abs().mean().item() < 4e-3


def test_pad_copy_step():
    # the pinned halo-form step needs an AOT cache; bind the pad copy alone
    pc = cr.PadCopySpec(D=34, H=16, W=16, C=64, halo=(0, 1, 1))
    step = cr.RouteStep("pad_copy", "kernel", (("X", "x"), ("Xp", "Xp")), build=pc.build,
                        signature=pc.signature(), grid=pc.grid(), block=pc.block)
    plan = dataclasses.replace(cr.dispatch_conv_route(_req("fwd_pad1"), ROUTES), steps=(step,),
                               workspaces=(cr.RouteWorkspace("Xp", pc.out_bytes, zero_once=True),))
    X = torch.randn(34, 16, 16, 64, device="cuda").to(torch.bfloat16)
    rb = _bind(plan, X, X, X)
    for _ in range(2):  # the halo stays zero call after call
        rb.run()
    torch.cuda.synchronize()
    Xp = rb.rt.keep[rb.workspaces["Xp"]].view(torch.bfloat16).view(34, 18, 18, 64)
    assert torch.equal(Xp, torch.nn.functional.pad(X, (0, 0, 1, 1, 1, 1, 0, 0)))
