# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The kernels the shipped gfx942 conv route and pin tables select, run on a device.

Every distinct selection of ``dispatch/conv_route_tables`` (``direct``,
``padcopy_halo`` and ``winograd`` routes) and of ``dispatch/conv_pin_tables``
(each mode) is dispatched, bound and launched exactly as the routed dispatch does it
(:func:`dispatch.conv_routes.dispatch_conv_route` /
:func:`dispatch.conv_routes.dispatch_conv_routed` and
:class:`benchmarks.common.conv_route_exec.RouteBinding`), on a reduced problem of
the case's own form: the same channels, height, width, padding and launch, fewer
depth planes where the plane is large. The pinned kernels are built from their
identity labels into a per-session AOT cache (the ``--compile-all`` build path),
so no prebuilt cache is needed.

Each call writes into a poisoned output and is compared with an f32 torch
reference. Every selection is launched again over the same workspaces, and each
rerun is checked against the reference too, so a workspace, slab or flag that a
launch fails to reset shows up on the second launch; the selections of a
deterministic mode must also reproduce the first output bitwise. The forward
``split_hybrid`` binaries are launched at degrees with and without remainder
segments, and the wgrad knobs no table selects are run on the shipped wgrad
binaries they apply to.

The CPU tests check that every shipped label still rebuilds a valid kernel
spec and that every wgrad opt-in still has a shipped binary to run on, which
the GPU tests depend on.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import re

import pytest

from dispatch import conv_pins as cp
from dispatch import conv_routes as cr

ARCH = "gfx942"
TABLE = "gfx942_unet3d_bf16"

# Scale-free error bounds: max |err| / max(max |ref|, 1) as the route check
# uses, and mean |err| / mean |ref|, a few bf16 (Winograd: f16) roundings.
MAX_REL = 5e-2
MEAN_REL = 1e-2

# A large plane keeps at least this many output voxels per depth range.
_MIN_VOXELS = 4096


def _torch():
    return pytest.importorskip("torch")


@functools.lru_cache(maxsize=None)
def _gfx942() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return ARCH in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


# ---------------------------------------------------------------------------
# Building a pinned kernel from its label
# ---------------------------------------------------------------------------


def _job_for_label(label: str):
    """The AOT ``BuildJob`` of a shipped implicit-GEMM label.

    The identity records every field that changes the emitted code, and the
    spec field of the same name carries it: fields at their identity default
    are left to the spec's default, ``lds_k_pad`` is always explicit (the
    identity's 0 is an explicit pad of 0). The capability axes are those of
    the 3-D stride-1 tables (dgrad: the x-halo binaries serve pW == 0)."""
    from kernels.conv.aot_jobs import BuildJob, spec_for_job
    from kernels.conv.kernel_identity import KernelIdentity
    from kernels.conv.kernel_label import decode, strip_launch_suffix

    ident = decode(strip_launch_suffix(label))
    assert ident.is_3d and ident.algorithm == "implicit_gemm", label
    caps = {"grouped": ident.grouped, "is_3d": ident.is_3d}
    if ident.direction == "dgrad":
        caps.update(stride=1, dilation=1)
        if ident.tapk_xhalo:
            caps["xhalo"] = True
    spec_cls = type(spec_for_job(BuildJob(ident, ident.direction, {}, caps), "bf16"))
    names = {f.name for f in dataclasses.fields(spec_cls)} - {"problem", "data"}
    kw = {}
    for f in dataclasses.fields(KernelIdentity):
        if f.name not in names:
            continue
        v = getattr(ident, f.name)
        if f.name == "lds_k_pad" or f.default is dataclasses.MISSING or v != f.default:
            kw[f.name] = v
    return BuildJob(ident, ident.direction, kw, caps)


@functools.lru_cache(maxsize=None)
def _cache(root: str):
    from benchmarks.common.kernel_cache import KernelCache

    return KernelCache(root, ARCH)


def _compiled(root: str, label: str):
    """The per-session cache holding ``label``'s binary (compiled on first use)."""
    from benchmarks.common.kernel_sweep import compile_jobs
    from kernels.conv.aot_jobs import kernel_for_job

    cache = _cache(root)
    job = _job_for_label(label)
    if not cache.has(job.identity):
        compile_jobs(
            cache=cache,
            all_jobs=[job],
            build=kernel_for_job,
            arch=ARCH,
            dtype="bf16",
            directions=[job.direction],
            jobs=1,
            log=lambda *a, **k: None,
        )
    assert cache.has(job.identity), f"{label} did not build"
    return cache


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("conv_shipped_tables_cache"))


# ---------------------------------------------------------------------------
# The selections and their reduced problems
# ---------------------------------------------------------------------------


def _table(kind: str) -> dict:
    tables = (
        cr.shipped_conv_route_tables()
        if kind == "routes"
        else cp.shipped_conv_pin_tables()
    )
    return json.loads(tables[TABLE].read_text())


def _flag(cmd: str, flag: str) -> int:
    return int(re.search(rf"(?:^| ){re.escape(flag)} (\d+)", cmd).group(1))


def _with_flag(cmd: str, flag: str, value: int) -> str:
    return re.sub(rf"((?:^| ){re.escape(flag)} )\d+", rf"\g<1>{value}", cmd)


def _out_extent(n: int, pad: int) -> int:
    return n + 2 * pad - 2  # 3-tap filter, stride 1, dilation 1


def _tiles(cmd: str, label: str) -> int:
    from dispatch.grouped_convolution import _conv_problem
    from kernels.common.conv_args import ConvArgs
    from kernels.conv.kernel_label import decode, strip_launch_suffix

    ident = decode(strip_launch_suffix(label))
    p = _conv_problem(cp.request_from_miopen_cmd(cmd, ARCH))
    a = ConvArgs.from_problem(
        p, tile_m=ident.tile_m, tile_n=ident.tile_n, tile_k=ident.tile_k
    )
    return a.num_pid_m * a.num_pid_n


def _reduced(cmd: str, label: str | None) -> tuple[str, str | None]:
    """``(cmd, label)`` of the reduced problem: fewer depth planes when the
    output plane is large, the same everything else. A split-tail launch
    (``/full<t>``) keeps its number of split (tail) tiles, on a depth large
    enough to leave full tiles beside them."""
    di, pd = _flag(cmd, "--in_d"), _flag(cmd, "--pad_d")
    ho = _out_extent(_flag(cmd, "-H"), _flag(cmd, "-p"))
    wo = _out_extent(_flag(cmd, "-W"), _flag(cmd, "-q"))
    do = _out_extent(di, pd)
    want = min(do, max(4, math.ceil(_MIN_VOXELS / (ho * wo))))
    m = re.search(r"@split_k=(\d+)/full(\d+)", label or "")
    tail = _tiles(cmd, label) - int(m.group(2)) if m else 0
    while True:
        out = _with_flag(cmd, "--in_d", want + 2 - 2 * pd)
        if not m or want >= do:
            break
        tiles = _tiles(out, label)
        if tiles > tail:
            label = label.replace(
                m.group(0), f"@split_k={m.group(1)}/full{tiles - tail}"
            )
            break
        want = min(do, 2 * want)
    if m and want >= do:
        out, label = cmd, label
    return out, label


def _form(cmd: str) -> str:
    return "pd{}ph{}pw{}".format(
        _flag(cmd, "--pad_d"), _flag(cmd, "-p"), _flag(cmd, "-q")
    )


def _route_cases():
    """One case per distinct (route, kernel or Winograd spec, padding form)."""
    out = {}
    for mode, body in _table("routes")["modes"].items():
        for case, row in body["cases"].items():
            key = (
                row["route"],
                row.get("kernel"),
                json.dumps(row.get("winograd"), sort_keys=True),
                _form(row["cmd"]),
            )
            out.setdefault(key, (mode, case, body["deterministic"]))
    return sorted(out.values())


def _pin_cases():
    """One case per distinct (mode, pinned label incl. launch, padding form)."""
    out = {}
    for mode, body in _table("pins")["modes"].items():
        for case, row in body["cases"].items():
            key = (mode, row["kernel"], _form(row["cmd"]))
            out.setdefault(key, (mode, case, bool(row["deterministic"])))
    return sorted(out.values())


# ---------------------------------------------------------------------------
# Running one call
# ---------------------------------------------------------------------------


def _check(plan, cache, *, deterministic: bool, runs: int = 2, seed: int = 1234):
    """Bind ``plan`` and launch it ``1 + runs`` times over the same buffers and
    workspaces, each time into a freshly poisoned output that must match the
    f32 reference; a deterministic plan must also reproduce its first output
    bitwise on every rerun."""
    torch = _torch()
    from benchmarks.common import benchmark_implicit_gemm_conv as B
    from benchmarks.common.conv_route_check import _operands, _ref
    from benchmarks.common.conv_route_exec import RouteBinding
    from rocke.runtime.hip_module import Runtime

    p = plan.problem
    A, Bt, D = _operands(plan.direction, p, torch, seed)
    rb = RouteBinding(
        plan,
        cache,
        Runtime(),
        ARCH,
        ptrs=(A.data_ptr(), Bt.data_ptr(), D.data_ptr()),
        sizes=tuple(t.numel() * 2 for t in (A, Bt, D)),
        stage2_kernels={},
        B=B,
    )
    try:
        ref = _ref(plan.direction, p, A, Bt, torch)
        first = None
        for launch in range(1 + runs):
            D.view(torch.int16).fill_(-1)  # bf16 NaN: every element must be written
            rb.run()
            torch.cuda.synchronize()
            out = D.float()
            assert not torch.isnan(
                out
            ).any(), f"launch {launch}: output elements left unwritten (NaN)"
            err = (out - ref).abs()
            rel = err.max().item() / max(ref.abs().max().item(), 1.0)
            mean = err.mean().item() / max(ref.abs().mean().item(), 1e-30)
            assert rel < MAX_REL and mean < MEAN_REL, (launch, rel, mean)
            if first is None:
                first = D.clone()
            elif deterministic:
                assert torch.equal(
                    D.view(torch.int16), first.view(torch.int16)
                ), f"launch {launch}: a deterministic selection is not bitwise reproducible"
    finally:
        torch.cuda.synchronize()
        rb.free(B)


def _route_plan(row: dict, case: str):
    cmd, label = _reduced(row["cmd"], row.get("kernel"))
    red = dict(row, cmd=cmd)
    if label is not None:
        red["kernel"] = label
    routes = cr.routes_from_mapping({"cases": {case: red}}, source="reduced")
    req = cp.request_from_miopen_cmd(cmd, ARCH)
    return cr.dispatch_conv_route(req, routes), label


def _pins(cases: dict, source: str):
    pins = cp.pins_from_mapping({"cases": cases}, source=source)
    pins.arch, pins.mode = ARCH, source
    return pins


def _pin_plan(row: dict, case: str):
    cmd, label = _reduced(row["cmd"], row["kernel"])
    pins = _pins({case: dict(row, cmd=cmd, kernel=label)}, "reduced")
    req = cp.request_from_miopen_cmd(cmd, ARCH)
    return cr.dispatch_conv_routed(req, None, pins), label


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _shipped_labels():
    labels = set()
    for kind in ("routes", "pins"):
        for body in _table(kind)["modes"].values():
            for row in body["cases"].values():
                if row.get("kernel"):
                    labels.add(row["kernel"].split()[0])
    return sorted(labels)


def test_every_shipped_label_rebuilds_its_spec():
    """A label the tables pin rebuilds a valid spec of its own identity (the
    GPU tests below build every pinned kernel this way)."""
    from kernels.conv.aot_jobs import _spec_is_valid, spec_for_job

    labels = _shipped_labels()
    assert labels
    for label in labels:
        job = _job_for_label(label)
        assert job.identity.label() == label
        spec_for_job(job, "bf16")
        assert _spec_is_valid(job, ARCH, "bf16"), label


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode,case,det", _route_cases(), ids=lambda v: v if isinstance(v, str) else None
)
def test_route_table_selection(mode, case, det, cache_root):
    if not _gfx942():
        pytest.skip("needs a gfx942 device")
    row = _table("routes")["modes"][mode]["cases"][case]
    plan, _ = _route_plan(row, case)
    cache = _compiled(cache_root, plan.kernel.spec.name) if plan.kernel else None
    _check(plan, cache, deterministic=det)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode,case,det", _pin_cases(), ids=lambda v: v if isinstance(v, str) else None
)
def test_pin_table_selection(mode, case, det, cache_root):
    if not _gfx942():
        pytest.skip("needs a gfx942 device")
    row = _table("pins")["modes"][mode]["cases"][case]
    plan, _ = _pin_plan(row, case)
    cache = _compiled(cache_root, plan.kernel.spec.name)
    _check(plan, cache, deterministic=det)


def _halo_cmd(cmd: str) -> str:
    """The halo form of a padded command: padding folded into the extents."""
    for ext, pad in (("--in_d", "--pad_d"), ("-H", "-p"), ("-W", "-q")):
        cmd = _with_flag(cmd, ext, _flag(cmd, ext) + 2 * _flag(cmd, pad))
        cmd = _with_flag(cmd, pad, 0)
    return cmd


def _hybrid_cases():
    """Each shipped forward ``split_hybrid`` binary (``shy<n>``, launch dropped)
    with the problem it runs: the pinned problem, or the halo problem of a
    ``padcopy_halo`` route."""
    out = {}
    for kind in ("pins", "routes"):
        for body in _table(kind)["modes"].values():
            for row in body["cases"].values():
                inst = (row.get("kernel") or "").split()
                if not inst or not re.search(r"-shy\d+", inst[0]):
                    continue
                cmd = row["cmd"]
                if row.get("route") == "padcopy_halo":
                    cmd = _halo_cmd(cmd)
                out.setdefault(inst[0], cmd)
    return sorted(out.items())


@pytest.mark.gpu
@pytest.mark.parametrize(
    "label,cmd",
    _hybrid_cases(),
    ids=[
        re.sub(r"^fwd-implicit_gemm-bf16-3d-|-wave64.*$", "", lab)
        for lab, _ in _hybrid_cases()
    ],
)
def test_split_hybrid_degrees(label, cmd, cache_root):
    """A ``split_hybrid`` binary at degrees G with no remainder segments
    (G = S * T), with remainders beside aligned slices (S >= 1, H > 0) and
    with remainders only (G < T): numerics on every launch and rerun, and
    bitwise reruns when the binary is deterministic (``sdet``)."""
    if not _gfx942():
        pytest.skip("needs a gfx942 device")
    case = "hybrid"
    cmd, _ = _reduced(cmd, label)
    T = _tiles(cmd, label)
    cache = _compiled(cache_root, label)
    det = bool(re.search(r"-sdet\d", label))
    for G in sorted({2 * T, 2 * T + max(1, T // 3), max(2, T // 2)}):
        pins = _pins({case: {"cmd": cmd, "kernel": f"{label} @split_k={G}"}}, "hybrid")
        req = cp.request_from_miopen_cmd(cmd, ARCH)
        plan = cr.dispatch_conv_routed(req, None, pins)
        _check(plan, cache, deterministic=det)


# Wgrad knobs no shipped selection uses, each on the first shipped wgrad
# binary (label order) it validates with: the reductions (ws_fuse, ws_tree,
# ws_store, ws_lc, ws_gred), the ping-pong and x-window variants and the
# epilogue stores. Some toggles also set the knobs they need or exclude.
WGRAD_OPT_INS = (
    {"ws_fuse": 1, "xwin_groups": 1, "xwin_gn": 0, "xwin_pp": 0},
    {"ws_fuse": 1, "ws_replicas": 1},
    {"ws_tree": 1},
    {"ws_store": 1, "ws_replicas": 1024},
    {"ws_lc": 1},
    {"ws_gred": 1},
    {"plain_pp": 1},
    {"plain_pp": 2},
    {"xwin_gn": 2},
    {"xwin_rot": 1},
    {"xwin_kord": 1},
    {"xwin_bq": 1},
    {"xwin_aq": 1},
    {"xwin_pps": 1},
    {"epi_store_fast": True},
    {"epi_quad_xpose": 1},
    {"sep_addr": 1},
    {"m_band": 8},
)


def _first_row(instance: str):
    """``(cmd, launch suffix)`` of the first table row pinning ``instance``."""
    for kind in ("pins", "routes"):
        for body in _table(kind)["modes"].values():
            for row in body["cases"].values():
                inst = (row.get("kernel") or "").split()
                if inst and inst[0] == instance and row.get("route") != "padcopy_halo":
                    return row["cmd"], " ".join(inst[1:])
    raise LookupError(instance)


def _opt_in_label(knobs: dict) -> tuple[str, str]:
    """``(variant label, base label)``: ``knobs`` on the first shipped wgrad
    binary they validate with."""
    from kernels.conv.aot_jobs import _spec_is_valid
    from kernels.conv.kernel_label import decode

    for base in _shipped_labels():
        if not base.startswith("wgrad-"):
            continue
        ident = dataclasses.replace(decode(base), **knobs)
        if ident == decode(base):
            continue
        label = ident.label()
        try:
            if _spec_is_valid(_job_for_label(label), ARCH, "bf16"):
                return label, base
        except (LookupError, ValueError):
            continue
    raise LookupError(f"no shipped wgrad binary takes {knobs}")


def test_every_wgrad_opt_in_has_a_shipped_base():
    for knobs in WGRAD_OPT_INS:
        _opt_in_label(knobs)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "knobs",
    WGRAD_OPT_INS,
    ids=lambda d: "-".join(f"{k}{int(v)}" for k, v in d.items()),
)
def test_wgrad_opt_in_knobs(knobs, cache_root):
    """An opt-in wgrad knob on a shipped binary, at that binary's problem
    (reduced) and launch, three reruns over the same workspaces: numerics on
    every launch, and bitwise equality with the first when deterministic."""
    if not _gfx942():
        pytest.skip("needs a gfx942 device")
    from benchmarks.common.det_classify import is_deterministic
    from kernels.conv.kernel_label import decode

    label, base = _opt_in_label(knobs)
    cmd, launch = _first_row(base)
    full = f"{label} {launch}".strip()
    cmd, full = _reduced(cmd, full)
    pins = _pins({"opt_in": {"cmd": cmd, "kernel": full}}, "opt-in")
    req = cp.request_from_miopen_cmd(cmd, ARCH)
    plan = cr.dispatch_conv_routed(req, None, pins)
    m = re.search(r"@split_k=(\d+)", full)
    det = is_deterministic(
        "wgrad", decode(label), int(m.group(1)) if m else None, "any"
    )
    _check(plan, _compiled(cache_root, label), deterministic=det, runs=3)
