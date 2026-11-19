# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""dispatch.conv_routes: route tables -> per-request route (host only)."""

import dataclasses
import json

import pytest

from dispatch import conv_pins as cp
from dispatch import conv_routes as cr

PIN_TABLE = cp.PIN_TABLE_DIR / "gfx942_unet3d_bf16.json"


def _cmd(F=1, C=64, K=64, D=256, H=128, W=128, pd=1, p=1):
    return (
        f"MIOpenDriver convbfp16 -n 1 -c {C} --in_d {D} -H {H} -W {W} -k {K} --fil_d 3 -y 3 -x 3 "
        f"--pad_d {pd} -p {p} -q {p} --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 "
        f"--in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F {F} -t 1 -V 0"
    )


def _pin_rows():
    return json.loads(PIN_TABLE.read_text())["modes"]["deterministic"]["cases"]


ZSPEC = {
    "stage": "lds16",
    "op_dtype": "f16",
    "fill_il": True,
    "waves": 8,
    "nsplit": False,
    "z_tile": 4,
    "n_tile": 32,
    "nfast": True,
    "pkrtz": True,
    "epi_vec": True,
    "yf16": True,
    "xcd": 6,
    "bps": 1,
    "bps_halo": 1,
}
BSPEC = dict(ZSPEC, nsplit=True, z_tile=2, n_tile=64)


def _table(extra=None):
    rows = _pin_rows()
    cases = {
        # s1 forward (pad 1,1,1): pad copy + the halo-form pin of fwd_s1
        "fwd_s1_pad1": {
            "cmd": _cmd(),
            "calls": 2,
            "route": "padcopy_halo",
            "kernel": rows["fwd_s1"]["kernel"],
        },
        # s1 forward, pad (0,1,1): D+2 input
        "fwd_s1_pad011": {
            "cmd": _cmd(D=258, pd=0),
            "route": "padcopy_halo",
            "kernel": rows["fwd_s1"]["kernel"],
        },
        "dgrad_s1_pad1": {
            "cmd": _cmd(F=2),
            "route": "direct",
            "kernel": rows["dgrad_s1_pad1"]["kernel"],
        },
        "wgrad_s1_pad1": {
            "cmd": _cmd(F=4),
            "route": "direct",
            "kernel": rows["wgrad_s1_pad1"]["kernel"],
        },
        # s3 (130x66x66 halo form; 128x64x64 padded), C 256 K 128
        "fwd_s3_pad1": {
            "cmd": _cmd(C=256, K=128, D=128, H=64, W=64),
            "route": "winograd",
            "winograd": {"spec": dict(ZSPEC, variant="yx"), "wtv": 4},
        },
        "dgrad_s3_pad011": {
            "cmd": _cmd(F=2, C=256, K=128, D=130, H=64, W=64, pd=0),
            "route": "winograd",
            "winograd": {"spec": dict(BSPEC, bps_halo_z=2), "wtv": 3},
        },
        "fwd_s8_pad1": {
            "cmd": _cmd(C=1024, K=512, D=32, H=16, W=16),
            "route": "winograd",
            "winograd": {"spec": dict(ZSPEC, ksplit=4), "wtv": 4},
        },
    }
    cases.update(extra or {})
    return {
        "arch": "gfx942",
        "llvm_flavor": "llvm22",
        "modes": {"deterministic": {"deterministic": True, "cases": cases}},
    }


def _routes(extra=None, tmp_path=None):
    return cr.routes_from_mapping(_table(extra)["modes"]["deterministic"], source="t")


def _load(tmp_path, doc, mode=None):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(doc))
    return cr.load_conv_routes(f, mode)


def _req(case, routes):
    return cr.request_for_route(routes.by_case()[case], "gfx942")


def test_loader_modes(tmp_path):
    r = _load(tmp_path, _table())
    assert (r.mode, r.arch, r.llvm_flavor, len(r)) == (
        "deterministic",
        "gfx942",
        "llvm22",
        7,
    )
    doc = _table()
    doc["modes"]["fastest"] = doc["modes"]["deterministic"]
    with pytest.raises(ValueError, match="name one"):
        _load(tmp_path, doc)
    assert len(_load(tmp_path, doc, "fastest")) == 7
    with pytest.raises(ValueError, match="no mode"):
        _load(tmp_path, doc, "nope")
    with pytest.raises(ValueError, match="not a route table"):
        _load(tmp_path, {"cases": {}})


@pytest.mark.parametrize(
    "row, msg",
    [
        ({"cmd": _cmd(C=8), "route": "magic"}, "route"),
        ({"cmd": _cmd(C=8), "route": "winograd"}, "winograd.spec"),
        (
            {"cmd": _cmd(C=8, F=4), "route": "winograd", "winograd": {"spec": ZSPEC}},
            "fwd / dgrad",
        ),
        (
            {
                "cmd": _cmd(C=8),
                "route": "winograd",
                "winograd": {"spec": dict(ZSPEC, C=8)},
            },
            "derived",
        ),
        (
            {
                "cmd": _cmd(C=8),
                "route": "winograd",
                "winograd": {"spec": dict(ZSPEC, variant="x")},
            },
            "variant",
        ),
        (
            {
                "cmd": _cmd(C=8),
                "route": "winograd",
                "kernel": "k",
                "winograd": {"spec": ZSPEC},
            },
            "no pinned",
        ),
        ({"cmd": _cmd(C=8, F=2), "route": "padcopy_halo", "kernel": "k"}, "serves fwd"),
        (
            {"cmd": _cmd(C=8, pd=0, p=0), "route": "padcopy_halo", "kernel": "k"},
            "padded problem",
        ),
        (
            {
                "cmd": _cmd(C=8).replace("-n 1", "-n 2"),
                "route": "padcopy_halo",
                "kernel": "k",
            },
            "N=1",
        ),
    ],
)
def test_bad_rows_refused(row, msg):
    with pytest.raises(ValueError, match=msg):
        _routes({"bad": row})


def test_one_problem_one_route():
    with pytest.raises(ValueError, match="route one problem"):
        _routes(
            {
                "dup": {
                    "cmd": _cmd(),
                    "route": "direct",
                    "kernel": _pin_rows()["fwd_s1_pad1"]["kernel"],
                }
            }
        )


def test_lookup_exact_padded_problem():
    r = _routes()
    assert r.lookup(_req("fwd_s1_pad1", r)).case == "fwd_s1_pad1"
    assert r.lookup(_req("fwd_s1_pad011", r)).case == "fwd_s1_pad011"
    req = _req("fwd_s1_pad1", r)
    # the halo-form problem, another dtype / arch / direction: not routed
    assert r.lookup(cr.halo_request(req)) is None
    assert r.lookup(dataclasses.replace(req, dtype="fp16")) is None
    assert r.lookup(dataclasses.replace(req, direction="dgrad", Di=258)) is None
    r.arch = "gfx942"
    assert r.lookup(dataclasses.replace(req, arch="gfx950")) is None
    with pytest.raises(ValueError, match="no route"):
        cr.dispatch_conv_route(cr.halo_request(req), r)


def test_halo_request():
    r = _routes()
    h = cr.halo_request(_req("fwd_s1_pad1", r))
    assert (h.Di, h.Hi, h.Wi, h.pad_d, h.pad_h, h.pad_w) == (258, 130, 130, 0, 0, 0)
    h = cr.halo_request(_req("fwd_s1_pad011", r))
    assert (h.Di, h.Hi, h.Wi, h.pad_d, h.pad_h, h.pad_w) == (258, 130, 130, 0, 0, 0)


def test_direct_route():
    r = _routes()
    for case, row in (
        ("dgrad_s1_pad1", "dgrad_s1_pad1"),
        ("wgrad_s1_pad1", "wgrad_s1_pad1"),
    ):
        plan = cr.dispatch_conv_route(_req(case, r), r)
        assert (
            plan.route == "direct" and plan.workspaces == () and plan.pad_copy is None
        )
        assert plan.kernel_label == _pin_rows()[row]["kernel"]
        assert [s.kind for s in plan.steps] == ["pinned"]
        assert plan.steps[0].arg_map == {"A": "x", "B": "w", "D": "y"}


def test_padcopy_halo_route():
    r = _routes()
    for case, halo in (("fwd_s1_pad1", (1, 1, 1)), ("fwd_s1_pad011", (0, 1, 1))):
        plan = cr.dispatch_conv_route(_req(case, r), r)
        assert plan.route == "padcopy_halo"
        assert plan.kernel_label == _pin_rows()["fwd_s1"]["kernel"]
        kp = plan.kernel_problem
        assert (kp.Di, kp.Hi, kp.Wi, kp.pD, kp.pH, kp.pW) == (258, 130, 130, 0, 0, 0)
        assert plan.pad_copy.halo == halo and plan.pad_copy.out_shape == (
            258,
            130,
            130,
            64,
        )
        assert [(w.name, w.nbytes, w.zero_once) for w in plan.workspaces] == [
            ("Xp", 258 * 130 * 130 * 64 * 2, True)
        ]
        k, p = plan.steps
        assert (k.kind, p.kind) == ("kernel", "pinned")
        assert k.arg_map == {"X": "x", "Xp": "Xp"} and p.arg_map == {
            "A": "Xp",
            "B": "w",
            "D": "y",
        }
        kd = k.build("gfx942")
        assert kd.name.startswith("pad_copy_D") and [
            a["name"] for a in k.signature
        ] == ["X", "Xp"]
        assert (
            k.grid[0] * 256 * plan.pad_copy.unroll * 8
            >= plan.pad_copy.D * 128 * 128 * 64
        )


def test_winograd_routes():
    r = _routes()
    plan = cr.dispatch_conv_route(_req("fwd_s3_pad1", r), r)
    assert (
        plan.route == "winograd" and plan.kernel is None and plan.winograd.kind == "yx"
    )
    assert (
        plan.winograd.wtv == 4
        and plan.winograd.direction == "fwd"
        and plan.winograd.prescale == "bps"
    )
    assert [s.kind for s in plan.steps] == ["kernel", "kernel"]
    names = [s.build("gfx942").name for s in plan.steps]
    assert (
        names[1].startswith("conv3d_winoyx_D130H66W66C256K128")
        and "_h1_wt4_am" in names[0]
    )
    ws = {w.name: w for w in plan.workspaces}
    assert (
        set(ws) == {"X8", "U", "E"}
        and ws["X8"].zero_once
        and ws["X8"].nbytes == 130 * 66 * 66 * 256 * 2
    )
    assert plan.steps[1].arg_map == {"X": "X8", "U": "U", "Y": "y", "E": "E"}
    # dgrad, pad (0,1,1): relayout halo z 2, y/x 1; forward filter flipped in the aux kernel
    plan = cr.dispatch_conv_route(_req("dgrad_s3_pad011", r), r)
    names = [s.build("gfx942").name for s in plan.steps]
    assert "_flip" in names[0] and names[1].startswith(
        "conv3d_winoyx_D132H66W66C128K256"
    )
    # split-K: f32 partials + the ordered reduction into y
    plan = cr.dispatch_conv_route(_req("fwd_s8_pad1", r), r)
    assert (
        len(plan.steps) == 3
        and plan.steps[1].arg_map["Y"] == "P"
        and plan.steps[2].arg_map == {"P": "P", "Y": "y"}
    )
    assert {w.name for w in plan.workspaces} == {"X8", "U", "E", "P"}


def test_winograd_halo_mismatch_refused():
    r = _routes(
        {
            "bad": {
                "cmd": _cmd(F=2, C=256, K=128, D=130, H=64, W=64, pd=0, p=1).replace(
                    "-k 128", "-k 64"
                ),
                "route": "winograd",
                "winograd": {"spec": BSPEC, "wtv": 3},
            }
        }
    )
    with pytest.raises(ValueError, match="relayout halo"):
        cr.dispatch_conv_route(_req("bad", r), r)


def test_kernel_from_pins():
    pins = cp.load_conv_pins(PIN_TABLE, "deterministic")
    rows = _table()["modes"]["deterministic"]["cases"]
    rows = {
        k: {kk: vv for kk, vv in v.items() if kk != "kernel"} for k, v in rows.items()
    }
    r = cr.routes_from_mapping({"cases": rows})
    # padcopy_halo: the halo problem's pin; direct: the padded problem's pin
    plan = cr.dispatch_conv_route(_req("fwd_s1_pad1", r), r, pins)
    assert plan.kernel_label.split()[0] == pins.lookup(plan.kernel_request).instance
    plan = cr.dispatch_conv_route(_req("dgrad_s1_pad1", r), r, pins)
    assert (
        plan.kernel_label.split()[0]
        == _pin_rows()["dgrad_s1_pad1"]["kernel"].split()[0]
    )
    with pytest.raises(ValueError, match="names no kernel"):
        cr.dispatch_conv_route(_req("dgrad_s1_pad1", r), r)


def test_routes_then_pins():
    pins = cp.load_conv_pins(PIN_TABLE, "deterministic")
    r = _routes()
    plan = cr.dispatch_conv_routed(_req("fwd_s3_pad1", r), r, pins)
    assert plan.route == "winograd"
    # not routed, pinned: a direct plan of the pin (here the halo-form problem)
    hreq = cr.halo_request(_req("fwd_s1_pad1", r))
    plan = cr.dispatch_conv_routed(hreq, r, pins)
    assert (
        plan.route == "direct"
        and plan.kernel_label.split()[0] == pins.lookup(hreq).instance
    )
    assert plan.steps[0].kind == "pinned" and plan.calls == pins.lookup(hreq).calls
    with pytest.raises(ValueError, match="neither"):
        cr.dispatch_conv_routed(dataclasses.replace(hreq, C=8), r, pins)
    with pytest.raises(ValueError, match="neither"):
        cr.dispatch_conv_routed(hreq)


def test_default_dispatch_unchanged():
    from dispatch import grouped_convolution as gc

    before = {
        d: [c.name for c in gc._REGISTRIES[d].candidates()] for d in ("fwd", "dgrad")
    }
    r = _routes()
    for case in r.by_case():
        cr.dispatch_conv_route(_req(case, r), r)
    after = {
        d: [c.name for c in gc._REGISTRIES[d].candidates()] for d in ("fwd", "dgrad")
    }
    assert before == after


SHIPPED = cr.shipped_conv_route_tables()


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_tables_dispatch(name):
    doc = json.loads(SHIPPED[name].read_text())
    for mode in doc["modes"]:
        r = cr.load_conv_routes(SHIPPED[name], mode)
        assert len(r) == len(doc["modes"][mode]["cases"])
        for case in r.by_case():
            plan = cr.dispatch_conv_route(_req(case, r), r)
            assert plan.route in cr.ROUTES and plan.steps


def test_replay_loads_routes():
    from benchmarks.common.conv_step_replay import load_route_cases, step_sequence

    if "gfx942_unet3d_bf16" not in SHIPPED:
        pytest.skip("no shipped route table")
    path = str(SHIPPED["gfx942_unet3d_bf16"])
    for form in ("pad1", "pad011"):
        cases = load_route_cases(f"{path}#deterministic@{form}", "gfx942")
        assert {f"{d}_s{s}" for d, s in step_sequence()} <= set(cases)
        for k, v in cases.items():
            assert ("route_plan" in v) == (v["route"] != "direct")
            assert v["kernel"]


def test_shipped_route_kernels_are_the_pin_picks():
    """The shipped route table names the shipped pin table's kernels: a row with its
    ``kernel`` removed and the pin table passed resolves to the same plan."""
    if "gfx942_unet3d_bf16" not in SHIPPED:
        pytest.skip("no shipped route table")
    doc = json.loads(SHIPPED["gfx942_unet3d_bf16"].read_text())
    pins = cp.load_conv_pins(PIN_TABLE, "deterministic")
    for mode, m in doc["modes"].items():
        r = cr.load_conv_routes(SHIPPED["gfx942_unet3d_bf16"], mode)
        bare = {
            k: {f: v for f, v in row.items() if f != "kernel"}
            for k, row in m["cases"].items()
        }
        rb = cr.routes_from_mapping(dict(m, cases=bare), source="bare")
        for case in r.by_case():
            p1 = cr.dispatch_conv_route(_req(case, r), r)
            p2 = cr.dispatch_conv_route(_req(case, rb), rb, pins)
            assert (p1.route, p1.kernel_label) == (p2.route, p2.kernel_label), case


def _shipped_routes(mode="deterministic"):
    return cr.load_conv_routes(SHIPPED["gfx942_unet3d_bf16"], mode)


@pytest.mark.parametrize("case", ["dgrad_s3_pad1", "fwd_s4_pad011", "fwd_s1_pad1"])
@pytest.mark.parametrize(
    "change, msg",
    [
        ({"layout": "NCHW"}, "NHWC"),
        ({"algorithm": "implicit_gemm"}, "algorithm"),
        ({"llvm_flavor": "llvm20"}, "no route"),
    ],
)
def test_a_route_refuses_what_it_was_not_selected_for(case, change, msg):
    """Winograd routes too: the layout, an explicit algorithm and the LLVM
    flavor the table was selected under are part of the route's contract."""
    r = _shipped_routes()
    req = _req(case, r)
    cr.dispatch_conv_route(req, r)
    cr.dispatch_conv_route(dataclasses.replace(req, algorithm="pinned"), r)
    with pytest.raises(ValueError, match=msg):
        cr.dispatch_conv_route(dataclasses.replace(req, **change), r)


def _row_with(case, **change):
    row = dict(_shipped_doc()["modes"]["deterministic"]["cases"][case])
    row.update(change)
    return row


def _shipped_doc():
    return json.loads(SHIPPED["gfx942_unet3d_bf16"].read_text())


def _winograd_case(direction="fwd"):
    cases = _shipped_doc()["modes"]["deterministic"]["cases"]
    return next(
        k
        for k, v in cases.items()
        if v["route"] == "winograd" and k.startswith(direction + "_")
    )


def _route_case(route, direction):
    cases = _shipped_doc()["modes"]["deterministic"]["cases"]
    return next(
        k
        for k, v in cases.items()
        if v["route"] == route and k.startswith(direction + "_")
    )


def test_rows_whose_kernel_or_dtype_do_not_fit_are_refused():
    pad = _route_case("padcopy_halo", "fwd")
    direct = _route_case("direct", "fwd")
    dgrad = _route_case("direct", "dgrad")
    to = {"convbfp16": "conv", "fp16": "convfp16"}
    bad = {
        # the pad copy and the drop-in are bf16
        "fp32 pad copy": _row_with(
            pad, cmd=_row_with(pad)["cmd"].replace("convbfp16", to["convbfp16"])
        ),
        "fp16 winograd": _row_with(
            _winograd_case(),
            cmd=_row_with(_winograd_case())["cmd"].replace("convbfp16", to["fp16"]),
        ),
        # a bf16 kernel on an fp32 command; a dgrad kernel on a forward command
        "fp32 direct": _row_with(
            direct, cmd=_row_with(direct)["cmd"].replace("convbfp16", to["convbfp16"])
        ),
        "dgrad label": _row_with(direct, kernel=_row_with(dgrad)["kernel"]),
    }
    for name, row in bad.items():
        with pytest.raises(ValueError, match="serves bf16|kernel on a"):
            cr.routes_from_mapping({"cases": {name: row}}, source=name)


def test_routes_and_pins_of_different_modes_are_refused():
    r = _shipped_routes()
    fastest = cp.load_conv_pins(PIN_TABLE, "fastest")
    req = _req(_route_case("direct", "dgrad"), r)
    with pytest.raises(ValueError, match="mode"):
        cr.dispatch_conv_route(req, r, fastest)
    with pytest.raises(ValueError, match="mode"):
        cr.dispatch_conv_routed(req, r, fastest)
    cr.dispatch_conv_routed(req, r, cp.load_conv_pins(PIN_TABLE, "deterministic"))


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_deterministic_route_modes_launch_deterministically(name):
    """Every kernel a deterministic mode routes to is a deterministic launch
    (the f32 det_classify verdict, as for a pin table's deterministic mode);
    a Winograd route's split-K reduction is the ordered one."""
    from benchmarks.common.det_classify import is_deterministic

    doc = json.loads(SHIPPED[name].read_text())
    for mode, body in doc["modes"].items():
        r = cr.load_conv_routes(SHIPPED[name], mode)
        assert r.deterministic == bool(body.get("deterministic"))
        if not r.deterministic:
            continue
        for case in r.by_case():
            plan = cr.dispatch_conv_route(_req(case, r), r)
            if plan.route == cr.ROUTE_WINOGRAD:
                reduce = [s for s in plan.steps if s.name.startswith("split-K")]
                assert all(
                    s.build("gfx942").name.startswith("wino_ks_reduce") for s in reduce
                ), case
                continue
            spec = plan.kernel.spec
            split_k = spec.launch_params().get("split_k")
            assert is_deterministic(
                plan.kernel_request.direction, spec.identity, split_k, "f32"
            ), case
