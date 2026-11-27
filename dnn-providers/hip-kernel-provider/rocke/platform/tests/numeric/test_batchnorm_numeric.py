# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real-GPU numeric checks of the batchnorm2d kernel families against an fp64
numpy reference (no torch).

One test per kernel kind x dtype, on small shapes: dynamic forward (train and
inference), flat inference, dynamic split reduce + apply, single-launch backward,
split backward, NHWC (infer / train / backward), multi-channel-per-wave and the
read-once persistent-grid kernels.  Training checks ``Y``, the running mean, the
*unbiased* running variance and the saved mean / inverse std; backward checks
``dx``, ``dscale`` and ``dbias``.  Machines without a gfx9 device skip.

The read-once kernels spin on a grid-wide barrier, so their plans are capped at the
device's co-resident CTA capacity (occupancy x CUs) and every wait is bounded; a
run that exhausts its bound reports through ``Err`` and fails the test instead of
hanging.  The exhaustive candidate sweeps are not part of this file.
"""

import ctypes

import numpy as np
import pytest

from rocke.dispatch.families import batchnorm as dispatch_bn
from rocke.helpers import compile_kernel
from rocke.instances.common import batchnorm2d as bn
from rocke.instances.common import batchnorm2d_ro as ro
from rocke.numeric.batchnorm_reference import (
    batchnorm_backward_reference,
    batchnorm_forward_reference,
    round_to_dtype,
)
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import get_device_arch
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release

DTYPES = ("f16", "bf16", "f32")
EPS = 1e-5
MOMENTUM = 0.5  # large, so a missing Bessel correction shows in the running variance
# Output tolerance of the storage dtype (values are O(1)).
_Y_TOL = {"f16": (2e-2, 2e-2), "bf16": (5e-2, 5e-2), "f32": (2e-4, 2e-4)}
# f32 accumulators / f32 buffers: dtype-independent.
_STAT_TOL = (3e-4, 3e-4)

_ARTIFACTS = {}


@pytest.fixture(scope="module")
def arch():
    target = get_device_arch(0)
    if target not in ("gfx942", "gfx950"):
        pytest.skip(f"requires a gfx942 or gfx950 device; detected {target}")
    return target


def _launcher(kernel, arch):
    art = _ARTIFACTS.get((kernel.name, arch))
    if art is None:
        art = _ARTIFACTS[(kernel.name, arch)] = compile_kernel(kernel, arch=arch)
    return KernelLauncher.from_artifact(art)


def _wave(arch):
    from rocke.core.arch import ArchTarget

    return ArchTarget.from_gfx(arch).wave_size


def _close(got, want, rtol, atol, what):
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
        rtol=rtol,
        atol=atol,
        err_msg=what,
    )


def _sum_tol(want):
    """Tolerance of an f32 sum of many terms: relative to the largest magnitude."""
    return 2e-3, 2e-3 * max(1.0, float(np.abs(want).max()))


class _Fwd:
    """Device buffers + reference of one forward case (any memory layout)."""

    def __init__(self, shape, dtype, training, layout="NCHW", seed=0xBA7C):
        rng = np.random.default_rng(seed)
        self.dtype, self.training = dtype, training
        self.layout = layout
        self.axis = 1 if layout == "NCHW" else -1
        n, c, h, w = shape  # logical (N, C, H, W)
        mem = (n, c, h, w) if layout == "NCHW" else (n, h, w, c)
        self.n, self.c, self.h, self.w = n, c, h, w
        r = lambda lo, hi, sh: round_to_dtype(
            rng.uniform(lo, hi, sh), dtype
        )  # noqa: E731
        self.x = r(-1.0, 1.0, mem)
        self.gamma = r(0.5, 1.5, (c,))
        self.beta = r(-0.5, 0.5, (c,))
        self.rm = rng.uniform(-0.3, 0.3, (c,)).astype(np.float32).astype(np.float64)
        self.rv = rng.uniform(0.5, 1.5, (c,)).astype(np.float32).astype(np.float64)
        self.ref = batchnorm_forward_reference(
            self.x, self.gamma, self.beta, self.rm, self.rv,
            training=training, eps=EPS, momentum=MOMENTUM, channel_axis=self.axis,
        )  # fmt: skip
        d = DeviceArray.from_host
        self.bufs = {
            "x": d(self.x, dtype),
            "gamma": d(self.gamma, dtype),
            "beta": d(self.beta, dtype),
            "y": DeviceArray(mem, dtype),
            "rm": d(self.rm, "f32"),
            "rv": d(self.rv, "f32"),
        }
        if training:
            self.bufs["mean"] = DeviceArray((c,), dtype)
            self.bufs["invstd"] = DeviceArray((c,), dtype)

    @property
    def dims(self):
        return {"N": self.n, "C": self.c, "H": self.h, "W": self.w}

    def values(self, with_saved=None):
        p = lambda k: self.bufs[k].data_ptr()  # noqa: E731
        v = {
            "X": p("x"), "Gamma": p("gamma"), "Beta": p("beta"), "Y": p("y"),
            "RunningMean": p("rm"), "RunningVar": p("rv"),
            **self.dims, "eps": EPS, "momentum": MOMENTUM,
        }  # fmt: skip
        if self.training if with_saved is None else with_saved:
            v.update({"Mean": p("mean"), "InvStd": p("invstd")})
        return v

    def check(self, saved=True):
        rtol, atol = _Y_TOL[self.dtype]
        ref = self.ref
        _close(self.bufs["y"].read(), ref["y"], rtol, atol, "y")
        _close(self.bufs["rm"].read(), ref["running_mean"], *_STAT_TOL, "running_mean")
        _close(self.bufs["rv"].read(), ref["running_var"], *_STAT_TOL, "running_var")
        if self.training and saved:
            _close(self.bufs["mean"].read(), ref["mean"], rtol, atol, "saved mean")
            _close(
                self.bufs["invstd"].read(), ref["invstd"], rtol, atol, "saved invstd"
            )


class _Bwd:
    def __init__(self, shape, dtype, layout="NCHW", seed=0xBA7D):
        rng = np.random.default_rng(seed)
        self.dtype, self.layout = dtype, layout
        axis = 1 if layout == "NCHW" else -1
        n, c, h, w = shape
        mem = (n, c, h, w) if layout == "NCHW" else (n, h, w, c)
        self.n, self.c, self.h, self.w = n, c, h, w
        self.x = round_to_dtype(rng.uniform(-1, 1, mem), dtype)
        self.dy = round_to_dtype(rng.uniform(-1, 1, mem), dtype)
        f32 = (
            lambda lo, hi: rng.uniform(lo, hi, (c,))
            .astype(np.float32)
            .astype(np.float64)
        )  # noqa: E731
        self.scale, self.mean, self.invstd = (
            f32(0.5, 1.5),
            f32(-0.1, 0.1),
            f32(0.8, 1.2),
        )
        self.ref = batchnorm_backward_reference(
            self.x, self.dy, self.scale, self.mean, self.invstd, channel_axis=axis
        )
        d = DeviceArray.from_host
        self.bufs = {
            "x": d(self.x, dtype),
            "dy": d(self.dy, dtype),
            "scale": d(self.scale, "f32"),
            "mean": d(self.mean, "f32"),
            "invstd": d(self.invstd, "f32"),
            "dx": DeviceArray(mem, dtype),
            "dscale": DeviceArray((c,), "f32"),
            "dbias": DeviceArray((c,), "f32"),
        }

    @property
    def dims(self):
        return {"N": self.n, "C": self.c, "H": self.h, "W": self.w}

    def values(self):
        p = lambda k: self.bufs[k].data_ptr()  # noqa: E731
        return {
            "X": p("x"), "dY": p("dy"), "Scale": p("scale"), "Mean": p("mean"),
            "InvStd": p("invstd"), "dX": p("dx"), "dScale": p("dscale"),
            "dBias": p("dbias"), **self.dims,
        }  # fmt: skip

    def check(self):
        rtol, atol = _Y_TOL[self.dtype]
        _close(self.bufs["dx"].read(), self.ref["dx"], rtol, 2 * atol, "dx")
        for k in ("dscale", "dbias"):
            _close(self.bufs[k].read(), self.ref[k], *_sum_tol(self.ref[k]), k)


def _launch(launcher, values, grid, block):
    launcher(values, config=LaunchConfig(grid=grid, block=(block, 1, 1)))
    synchronize_and_release()


# --------------------------------------------------------------------- #
# Dynamic forward + flat inference
# --------------------------------------------------------------------- #

# (block_size, vec, channels_per_block, n_groups, fast_reduce)
_DYN_VARIANTS = (
    (256, 4, 1, 1, False),
    (256, 2, 1, 1, True),
    (128, 4, 2, 1, False),
    (256, 4, 1, 4, False),
)
_DYN_SHAPES = ((4, 8, 3, 5), (4, 8, 8, 8))  # tail / exact fit


@pytest.mark.parametrize("training", (True, False), ids=("train", "infer"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "variant", _DYN_VARIANTS, ids=lambda v: "b%d_v%d_cpb%d_ng%d_fr%d" % v
)
def test_dynamic_forward(arch, dtype, training, variant):
    bs, vec, cpb, ng, fr = variant
    if fr and not training:
        pytest.skip("fast_reduce is training-only")
    spec = bn.BatchNorm2DSpec(
        n=ng, c=cpb, h=1, w=1, block_size=bs, vec=vec, dtype=dtype, training=training,
        save_mean_invstd=training, channels_per_block=cpb, n_groups=ng,
        fast_reduce=fr, wave_size=_wave(arch), dynamic=True,
    )  # fmt: skip
    ok, why = bn.is_valid_spec(spec, arch)
    if not ok:
        pytest.skip(why)
    launcher = _launcher(bn.build_batchnorm2d(spec), arch)
    for shape in _DYN_SHAPES:
        case = _Fwd(shape, dtype, training)
        _launch(launcher, case.values(), bn.batchnorm2d_grid(case.c, spec), bs)
        case.check()


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize(
    "geom",
    (
        (256, 8, 1, True),
        (256, 4, 2, True),
        (128, 2, 4, True),
        (256, 4, 1, False),
        (256, 1, 1, True),
    ),
    ids=lambda g: "b%d_v%d_u%d_al%d" % g,
)
def test_flat_inference(arch, dtype, geom):
    bs, vec, unroll, aligned = geom
    spec = bn.BatchNorm2DFlatInferSpec(
        block_size=bs, vec=vec, unroll=unroll, aligned=aligned, dtype=dtype, wave_size=_wave(arch)
    )  # fmt: skip
    assert bn.is_valid_flat_infer_spec(spec, arch)[0]
    shape = (3, 8, 8, 8) if aligned else (3, 8, 3, 5)
    ok, why = bn.flat_infer_admits(spec, *shape)
    if not ok:
        pytest.skip(why)
    case = _Fwd(shape, dtype, training=False)
    launcher = _launcher(bn.build_batchnorm2d_flat_infer(spec), arch)
    grid = bn.batchnorm2d_flat_infer_grid(case.n, case.c, case.h, case.w, spec)
    _launch(launcher, case.values(), grid, bs)
    case.check()


# --------------------------------------------------------------------- #
# Split (two launches)
# --------------------------------------------------------------------- #

_SPLIT_PLANS = ((2, 1), (1, 2), (2, 2))  # (sn_req, sh_req)
_SPLIT_SHAPES = ((4, 8, 32, 32), (3, 8, 30, 30))  # exact fit / spatial tail


def _split_cases(spec, make_case):
    """``(case, plan)`` for every split plan the spec admits on the test shapes."""
    out = []
    for shape in _SPLIT_SHAPES:
        for sn_req, sh_req in _SPLIT_PLANS:
            case = make_case(shape)
            plan = bn.batchnorm2d_split_plan(
                spec, case.n, case.h, case.w, sn_req=sn_req, sh_req=sh_req
            )
            if plan is not None:
                out.append((case, plan))
    assert out, "no split plan admits the test shapes"
    return out


def _geom(plan, case):
    return {
        **case.dims, "NChunk": plan.n_chunk, "HwChunk": plan.hw_chunk,
        "SplitHw": plan.sh, "SplitN": plan.sn,
    }  # fmt: skip


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fast_reduce", (False, True), ids=("flat", "fr"))
def test_split_forward(arch, dtype, fast_reduce):
    vec = 4
    spec = bn.BatchNorm2DSplitDynSpec(
        block_size=128, vec=vec, dtype=dtype, fast_reduce=fast_reduce,
        save_mean_invstd=True, wave_size=_wave(arch),
    )  # fmt: skip
    ok, why = bn.is_valid_split_dyn_spec(spec, arch)
    if not ok:
        pytest.skip(why)
    red = _launcher(bn.build_batchnorm2d_split_dyn_reduce(spec), arch)
    app = _launcher(bn.build_batchnorm2d_split_dyn_apply(spec), arch)
    for case, plan in _split_cases(spec, lambda sh: _Fwd(sh, dtype, training=True)):
        ws = DeviceArray(
            (bn.batchnorm2d_split_dyn_workspace_floats(case.c, plan),), "f32"
        )
        geom = _geom(plan, case)
        grid = bn.batchnorm2d_split_dyn_grid(case.c, plan)
        _launch(
            red,
            {"X": case.bufs["x"].data_ptr(), "Workspace": ws.data_ptr(), **geom},
            grid,
            spec.block_size,
        )
        _launch(
            app,
            {**case.values(), "Workspace": ws.data_ptr(), **geom},
            grid,
            spec.block_size,
        )
        case.check()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fast_reduce", (False, True), ids=("flat", "fr"))
def test_split_backward(arch, dtype, fast_reduce):
    spec = bn.BatchNorm2DBwdSplitSpec(
        block_size=128, vec=4, dtype=dtype, fast_reduce=fast_reduce, save_mean_invstd=True,
        wave_size=_wave(arch),
    )  # fmt: skip
    ok, why = bn.is_valid_split_dyn_spec(spec, arch)
    if not ok:
        pytest.skip(why)
    red = _launcher(bn.build_batchnorm2d_bwd_split_reduce(spec), arch)
    app = _launcher(bn.build_batchnorm2d_bwd_split_apply(spec), arch)
    for case, plan in _split_cases(spec, lambda sh: _Bwd(sh, dtype)):
        ws = DeviceArray(
            (bn.batchnorm2d_bwd_split_workspace_floats(case.c, plan),), "f32"
        )
        geom = _geom(plan, case)
        v = case.values()
        grid = bn.batchnorm2d_split_dyn_grid(case.c, plan)
        _launch(
            red,
            {k: v[k] for k in ("X", "dY", "Mean", "InvStd")} | {"Workspace": ws.data_ptr(), **geom},
            grid, spec.block_size,
        )  # fmt: skip
        _launch(app, {**v, "Workspace": ws.data_ptr(), **geom}, grid, spec.block_size)
        case.check()


# --------------------------------------------------------------------- #
# Single-launch backward
# --------------------------------------------------------------------- #

_BWD_VARIANTS = (
    (256, 4, 1, 1, False),
    (256, 2, 1, 1, True),
    (128, 4, 2, 1, False),
    (256, 4, 1, 4, False),
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "variant", _BWD_VARIANTS, ids=lambda v: "b%d_v%d_cpb%d_ng%d_fr%d" % v
)
def test_backward_single_launch(arch, dtype, variant):
    bs, vec, cpb, ng, fr = variant
    spec = bn.BatchNorm2DBwdSpec(
        block_size=bs, vec=vec, dtype=dtype, channels_per_block=cpb, n_groups=ng,
        fast_reduce=fr, wave_size=_wave(arch),
    )  # fmt: skip
    ok, why = bn.is_valid_bwd_spec(spec, arch)
    if not ok:
        pytest.skip(why)
    launcher = _launcher(bn.build_batchnorm2d_bwd_dynamic(spec), arch)
    for shape in ((4, 8, 3, 5), (4, 8, 8, 8)):
        case = _Bwd(shape, dtype)
        _launch(launcher, case.values(), bn.batchnorm2d_bwd_grid(case.c, spec), bs)
        case.check()


# --------------------------------------------------------------------- #
# NHWC
# --------------------------------------------------------------------- #

_NHWC_SHAPE = (32, 64, 16, 16)  # N, C, H, W (memory order N, H, W, C)


def _nhwc_runs(arch, mode, dtype):
    """Yield ``(candidate, plan, spec)`` of the first shipped geometries admitting the shape."""
    n, c, h, w = _NHWC_SHAPE
    got = 0
    for cand in dispatch_bn.dynamic_batchnorm_nhwc_candidates(mode):
        spec = cand.spec(dtype, arch)
        for plan in cand.plans(n, c, h, w, dtype, arch)[:2]:
            yield cand, plan, spec
            got += 1
        if got >= 6:
            return


@pytest.mark.parametrize("mode", ("infer", "train", "bwd"))
@pytest.mark.parametrize("dtype", ("f16", "bf16"))  # no f32 NHWC kernels
def test_nhwc(arch, mode, dtype):
    runs = list(_nhwc_runs(arch, mode, dtype))
    if not runs:
        pytest.skip("no shipped NHWC geometry admits the shape")
    for cand, plan, spec in runs:
        launchers = [
            _launcher(cand.build(kind, spec, arch), arch) for kind in cand.kinds
        ]
        grid = bn.batchnorm2d_nhwc_grid(_NHWC_SHAPE[1], plan)
        ws = None
        if cand.variant == "split":
            ws = DeviceArray((cand.workspace_floats(_NHWC_SHAPE[1], plan),), "f32")
        wsp = {"Workspace": ws.data_ptr()} if ws is not None else {}
        geom = {"RowsPerCta": plan.rows_per_cta}
        if mode == "bwd":
            case = _Bwd(_NHWC_SHAPE, dtype, layout="NHWC")
            full = case.values()
            if cand.variant == "split":
                red = {
                    k: full[k]
                    for k in ("X", "dY", "Mean", "InvStd", "N", "C", "H", "W")
                }
                seq = [
                    {**red, **wsp, **geom},
                    {**full, **wsp, **geom, "Splits": plan.splits},
                ]
            else:
                seq = [full]
        else:
            case = _Fwd(_NHWC_SHAPE, dtype, training=(mode == "train"), layout="NHWC")
            full = case.values()
            if mode == "infer":
                seq = [{**full, **geom}]
            elif cand.variant == "split":
                red = {"X": full["X"], **case.dims}
                seq = [
                    {**red, **wsp, **geom},
                    {**full, **wsp, **geom, "Splits": plan.splits},
                ]
            else:
                seq = [full]
        for launcher, vals in zip(launchers, seq):
            _launch(launcher, vals, grid, spec.block_size)
        case.check()


# --------------------------------------------------------------------- #
# Multi-channel-per-wave
# --------------------------------------------------------------------- #


def _mcw_shapes(cand, spec, arch):
    """Shapes the candidate admits: its exact geometry and a ragged one."""
    lpc = _wave(arch) // cand.cpw
    hw = cand.vec * lpc * cand.slots
    out = []
    for n, h in (
        (cand.waves * cand.npw, hw),
        (cand.waves * cand.npw - 1, hw - cand.vec),
    ):
        shape = (n, 2 * cand.cpw, 1, h)
        if n > 0 and h > 0 and bn.mcw_admits(spec, *shape)[0]:
            out.append(shape)
    return out


@pytest.mark.parametrize("mode", ("train", "bwd"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_mcw_shipped(arch, mode, dtype):
    d = {"f16": "fp16", "f32": "fp32"}.get(dtype, dtype)
    cands = dispatch_bn.shipped_batchnorm_mcw_candidates(mode, d)
    assert cands, "empty shipped multi-channel-per-wave set"
    ran = 0
    for cand in cands:
        spec = cand.spec(dtype, arch)
        launcher = _launcher(bn.build_batchnorm2d_mcw(spec), arch)
        for shape in _mcw_shapes(cand, spec, arch):
            case = _Bwd(shape, dtype) if mode == "bwd" else _Fwd(shape, dtype, True)
            _launch(
                launcher,
                case.values(),
                bn.batchnorm2d_mcw_grid(case.c, spec),
                cand.block_size,
            )
            case.check()
            ran += 1
    assert ran >= len(cands)


# --------------------------------------------------------------------- #
# Read-once (persistent grid, grid-wide barrier)
# --------------------------------------------------------------------- #


def _resident_ctas(launcher, block_size):
    """CTAs of this kernel that can be co-resident on the device (occupancy x CUs)."""
    from rocke.runtime.hip_module import _resolve_hip, get_device_num_cus

    fn = _resolve_hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    blocks = ctypes.c_int(0)
    rc = fn(ctypes.byref(blocks), launcher._live_fn().p, block_size, 0)
    cus = get_device_num_cus(0) or 0
    if rc != 0 or blocks.value < 1 or cus <= 0:
        pytest.skip("occupancy query unavailable")
    return int(blocks.value) * cus


_RO_GEOMS = ((256, 8, 4), (256, 4, 16), (512, 8, 2))  # block, vec, regs
_RO_SHAPE = (4, 8, 8, 8)


@pytest.mark.parametrize("backward", (False, True), ids=("train", "bwd"))
@pytest.mark.parametrize("dtype", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("geom", _RO_GEOMS, ids=lambda g: "b%d_v%d_r%d" % g)
def test_read_once(arch, dtype, backward, geom):
    bs, vec, regs = geom
    if dtype == "f32":
        vec //= 2
    spec = ro.BatchNorm2DRoSpec(
        block_size=bs, vec=vec, regs=regs, dtype=dtype, wave_size=_wave(arch)
    )
    ok, why = ro.is_valid_ro_spec(spec, arch)
    if not ok:
        pytest.skip(why)
    kernel = (
        ro.build_batchnorm2d_ro_bwd if backward else ro.build_batchnorm2d_ro_train
    )(spec)
    launcher = _launcher(kernel, arch)
    cap = _resident_ctas(launcher, bs)
    n, c, h, w = _RO_SHAPE
    plans = []
    for sn_req, sh_req in ((1, 2), (2, 1), (2, 2), (4, 2)):
        p = ro.batchnorm2d_ro_plan(
            spec, n, c, h, w, sn_req=sn_req, sh_req=sh_req, max_resident=cap
        )
        if p is not None and p not in plans and p.splits >= 2:
            plans.append(p)
    if not plans:
        pytest.skip("no read-once plan admits the shape")
    for plan in plans:
        assert plan.grid <= cap  # co-residency: the barrier would deadlock otherwise
        case = _Bwd(_RO_SHAPE, dtype) if backward else _Fwd(_RO_SHAPE, dtype, True)
        sync = DeviceArray((ro.batchnorm2d_ro_sync_ints(c) + 64,), "i32")
        err = DeviceArray((1,), "i32")
        vals = {
            **case.values(), "SyncI": sync.data_ptr(), "SyncF": sync.data_ptr(),
            "Err": err.data_ptr(), "NChunk": plan.n_chunk, "HwChunk": plan.hw_chunk,
            "SplitHw": plan.sh, "SplitN": plan.sn, "Grid": plan.grid, "MaxSpin": 500_000,
            "Epoch": 0,
        }  # fmt: skip
        _launch(launcher, vals, (plan.grid, 1, 1), bs)
        assert int(err.read()[0]) == 0, "read-once barrier spin bound exhausted"
        case.check()
