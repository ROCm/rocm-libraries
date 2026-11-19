# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The 3-D Winograd family (kernels/common/conv_winograd3d.py) on a gfx942 device.

* every distinct Winograd spec of the shipped route tables (``xpf``, the per-axis
  prescale halo, split-K), on a reduced problem of its own padding form, at input scales
  inside and outside the f16 window of the transformed operands (the unscaled and the
  scaled copies of the main loop), against an f32 torch reference and bitwise across
  repeated calls (determinism);
* the halo-form BWD-data drop-in (``lin`` tiling, relayout halo 2);
* the pad copy at the halos the ``padcopy_halo`` routes use;
* the route executor on two same-volume halo shapes in one process (kernel names);
* the drop-in wrapper's tensor checks.
"""

import json
import re

import pytest

torch = pytest.importorskip("torch")

from dispatch import conv_routes as cr
from kernels.common import conv_winograd3d as cw
from kernels.common import conv_winograd3d_select as sel


def _gfx942():
    if not torch.cuda.is_available():
        return False
    return "gfx942" in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


pytestmark = pytest.mark.skipif(not _gfx942(), reason="needs a gfx942 device")

MAX_REL, MEAN_REL = 5e-2, 4e-3  # scale-free max / mean relative error (bf16 output)


def _cmd(F, C, K, D, H, W, pd):
    return (
        f"MIOpenDriver convbfp16 -n 1 -c {C} --in_d {D} -H {H} -W {W} -k {K} --fil_d 3 "
        f"-y 3 -x 3 --pad_d {pd} -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 "
        "-l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout "
        f"NDHWC -m conv -g 1 -F {F} -t 1 -V 0"
    )


def _shipped_specs():
    """Distinct (direction, pad_d, spec, wtv) of the shipped winograd routes."""
    out = {}
    for path in cr.shipped_conv_route_tables().values():
        data = json.loads(path.read_text())
        for mode in data["modes"].values():
            for case, row in mode["cases"].items():
                if row.get("route") != "winograd":
                    continue
                F = int(re.search(r" -F (\d)", row["cmd"]).group(1))
                pd = int(re.search(r"--pad_d (\d)", row["cmd"]).group(1))
                w = row["winograd"]
                key = (F, pd, json.dumps(w["spec"], sort_keys=True), int(w["wtv"]))
                out.setdefault(key, case)
    return sorted(out.items())


SHIPPED = _shipped_specs()


def _routed(F, pd, spec, wtv, D=32, H=16, W=16, K=128):
    """Route plan of a reduced problem in the spec's padding form (pad (1,1,1): D; pad
    (0,1,1): D + 2), C=256 for split-K (even chunk count per split), else 128."""
    C = 256 if spec.get("ksplit", 1) > 1 else 128
    row = {
        "cmd": _cmd(F, C, K, D + (2 if pd == 0 else 0), H, W, pd),
        "route": "winograd",
        "winograd": {"spec": spec, "wtv": wtv},
    }
    routes = cr.routes_from_mapping({"cases": {"t": row}}, source="gpu-test")
    req = cr.request_for_route(routes.by_case()["t"], "gfx942")
    return cr.dispatch_conv_route(req, routes)


def _data(p, direction, scale, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs, ys = (p.Di, p.Hi, p.Wi, p.C), (p.Do, p.Ho, p.Wo, p.K)
    a = torch.randn(*(xs if direction == "fwd" else ys), device="cuda", generator=g)
    a = (a * scale).to(torch.bfloat16)
    w = torch.randn(p.K, 3, 3, 3, p.C, device="cuda", generator=g)
    w = (w * (2 / (27 * p.C)) ** 0.5).to(torch.bfloat16)
    return a, w


def _reference(p, direction, a, w):
    x = a.float().permute(3, 0, 1, 2)[None]
    wt = w.float().permute(0, 4, 1, 2, 3)
    f = (
        torch.nn.functional.conv3d
        if direction == "fwd"
        else torch.nn.functional.conv_transpose3d
    )
    return f(x, wt, padding=(p.pD, p.pH, p.pW))[0].permute(1, 2, 3, 0)


def _check(out, ref):
    err = (out.float() - ref).abs()
    ra = ref.abs()
    rel = err.max().item() / ra.max().item()
    mrel = err.mean().item() / ra.mean().item()
    assert rel < MAX_REL and mrel < MEAN_REL, (rel, mrel)


def _run_twice(call, a, w):
    """Two calls into NaN-poisoned outputs: equal bit for bit (deterministic)."""
    outs = []
    for _ in range(2):
        y = torch.full(call.out, float("nan"), device="cuda", dtype=torch.bfloat16)
        call(a, w, y)
        torch.cuda.synchronize()
        outs.append(y)
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
    return outs[0]


@pytest.mark.parametrize(
    "key", [k for k, _ in SHIPPED], ids=[case for _, case in SHIPPED]
)
def test_shipped_winograd_specs(key):
    """The shipped specs at input scales 1, 2^-30 (below the f16 window: scaled loop) and
    2^13 (above it): accurate and deterministic."""
    from kernels.common.conv_winograd3d_dropin import Winograd3dDropin

    F, pd, spec_json, wtv = key
    spec = json.loads(spec_json)
    assert spec.get("xpf") == 1  # the unrolled-prefetch main loop
    plan = _routed(F, pd, spec, wtv)
    p, direction = plan.problem, plan.direction
    call = Winograd3dDropin(p, direction, plan.winograd)
    for scale in (1.0, 2.0**-30, 2.0**13):
        a, w = _data(p, direction, scale)
        out = _run_twice(call, a, w)
        _check(out, _reference(p, direction, a, w))


def test_halo_form_dgrad_lin():
    """Halo-form BWD-data (pad 0): dY relayouted with a zero halo of 2, the lin tiling
    (TX = 2^n + 1), prescale halo 2."""
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_winograd3d_dropin import Winograd3dDropin

    p = ConvProblem(
        N=1, Hi=18, Wi=18, C=128, K=128, Y=3, X=3, Di=34, Z=3, sD=1, pD=0, dD=1
    )
    spec = dict(sel.SPEC_B, lin=True, xcd=6, bps=1, bps_halo=2)
    plan = sel.Winograd3dPlan("yx", tuple(sorted(spec.items())), 3, "dropin", "dgrad")
    call = Winograd3dDropin(p, "dgrad", plan)
    assert call.spec.lin and call.src == (p.Do, p.Ho, p.Wo, p.K)
    for scale in (1.0, 2.0**-30):
        a, w = _data(p, "dgrad", scale)
        _check(_run_twice(call, a, w), _reference(p, "dgrad", a, w))


@pytest.mark.parametrize("halo", [(1, 1, 1), (0, 1, 1)])
@pytest.mark.parametrize("C", [8, 512])
def test_pad_copy(halo, C):
    from rocke.helpers import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    D, H, W = 6, 8, 10
    k = cw.build_pad_copy(cw.PadCopyKernelSpec(D, H, W, C, halo=halo))
    run = KernelLauncher(
        hsaco=compile_kernel(k, arch="gfx942").hsaco,
        kernel_name=k.name,
        signature=SignatureBuilder().ptr("X", "bf16").ptr("Xp", "bf16").build(),
    )
    X = torch.randn(D, H, W, C, device="cuda").to(torch.bfloat16)
    hz, hy, hx = halo
    Xp = torch.zeros(D + 2 * hz, H + 2 * hy, W + 2 * hx, C, device="cuda")
    Xp = Xp.to(torch.bfloat16)
    cfg = LaunchConfig(grid=cw.pad_copy_grid(D, H, W, C), block=(256, 1, 1))
    for _ in range(2):  # the halo stays zero call after call
        run({"X": X, "Xp": Xp}, config=cfg)
    torch.cuda.synchronize()
    ref = torch.nn.functional.pad(X, (0, 0, hx, hx, hy, hy, hz, hz))
    assert torch.equal(Xp, ref)


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


def test_same_volume_routes_in_one_process():
    """(64,64,64) and (256,32,32) pad (1,1,1) routes have one volume: the route executor
    (kernels memoised by name) must still run each shape's own halo relayout."""
    from benchmarks.common.conv_route_exec import RouteBinding

    fwd = [json.loads(k[2]) for k, _ in SHIPPED if k[0] == 1 and k[1] == 1]
    spec = next(s for s in fwd if s.get("ksplit", 1) == 1)
    for D, H, W in ((64, 64, 64), (256, 32, 32)):
        plan = _routed(1, 1, spec, 4, D=D, H=H, W=W, K=128)
        p = plan.problem
        a, w = _data(p, "fwd", 1.0)
        y = torch.full((p.Do, p.Ho, p.Wo, p.K), float("nan"), device="cuda")
        y = y.to(torch.bfloat16)
        rb = RouteBinding(
            plan,
            None,
            _Rt(),
            "gfx942",
            ptrs=(a.data_ptr(), w.data_ptr(), y.data_ptr()),
            sizes=tuple(t.numel() * 2 for t in (a, w, y)),
            B=object(),
        )
        rb.run()
        torch.cuda.synchronize()
        _check(y, _reference(p, "fwd", a, w))


def test_dropin_refuses_mismatched_tensors():
    """The kernels address the problem's extents, not the tensors': a y or w of another
    shape, dtype or layout is refused instead of written or read out of bounds."""
    from kernels.common.conv_winograd3d_dropin import Winograd3dDropin

    F, pd, spec_json, wtv = next(k for k, _ in SHIPPED if k[0] == 1)
    plan = _routed(F, pd, json.loads(spec_json), wtv)
    p = plan.problem
    call = Winograd3dDropin(p, "fwd", plan.winograd)
    a, w = _data(p, "fwd", 1.0)
    bf = {"device": "cuda", "dtype": torch.bfloat16}
    half = torch.empty(p.Do, p.Ho, p.Wo, p.K // 2, **bf)
    f32 = torch.empty(p.Do, p.Ho, p.Wo, p.K, device="cuda", dtype=torch.float32)
    strided = torch.empty(p.Do, p.Ho, p.Wo, 2 * p.K, **bf)[..., ::2]
    for y in (half, f32, strided):
        with pytest.raises(ValueError, match="y:"):
            call(a, w, y)
    with pytest.raises(ValueError, match="w:"):
        call(a, w[: p.K // 2], None)
    with pytest.raises(ValueError, match="w:"):
        call(a, w.float(), None)
    _check(call(a, w), _reference(p, "fwd", a, w))
