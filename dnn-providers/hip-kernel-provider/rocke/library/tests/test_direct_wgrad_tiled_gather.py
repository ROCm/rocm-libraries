# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the tiled and the gather direct wgrad (``direct_wgrad_tiled``,
``direct_wgrad_gather``) and the slab fold (``direct_wgrad_slab_fold``) as the
gfx950 direct AOT grid serves them.

CPU: the label enumeration, the kept-set tables, the job round trip, the
launch plans of the 16-bit-dW kernels (``dw16``, ``pk16``, ``slab``) and the
grouped gather cells. GPU (gfx950): each plan class on a small layer, the dW
against a float64 numpy reference of the stored operands.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_wgrad_tiled_gather.py -v
"""

from __future__ import annotations

import functools
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.device_array import DeviceArray
from rocke.tests.gpu import current_device_arch_in

from kernels.common.conv_abi import conv_direct_args_signature, conv_direct_output_dtype
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_wgrad_gather import (
    WgradGatherSpec,
    build_wgrad_gather,
    is_valid_wgrad_gather_spec,
)
from kernels.common.conv_direct_wgrad_tiled import (
    DirectWgradTiledSpec,
    WgradSlabFoldSpec,
    is_valid_wgrad_tiled_spec,
)
from kernels.conv import direct_aot_jobs as jobs_mod

from ._direct_conv_numeric import direct_wgrad_reference, stored_uniform

_TILED = "direct_wgrad_tiled"
_GATHER = "direct_wgrad_gather"
_FOLD = "direct_wgrad_slab_fold"
_DTYPES = ("fp16", "bf16")
_FLAGS = ("dw16", "pk16", "slab", "l64", "gpack", "grouped", "one_split")


@functools.lru_cache(maxsize=None)
def _grid(dtype: str) -> tuple:
    """The gfx950 wgrad jobs of the three families (one dtype)."""
    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    return tuple(
        j
        for j in dks.enumerate_direct_jobs(
            arch="gfx950",
            target=target,
            directions=("wgrad",),
            dtype=dtype,
            jobs=1,
        )
        if j.identity.algorithm in (_TILED, _GATHER, _FOLD)
    )


def _flag(job) -> str:
    """The plan class of a job: its first set flag, or ``plain``."""
    return next((f for f in _FLAGS if job.spec_kwargs.get(f)), "plain")


class _FakeCache:
    """The two lookups a plan makes of the kernel cache, for helper kernels."""

    def __init__(self, hsacos=None):
        self.hsacos = hsacos if hsacos is not None else {}

    def entry_meta(self, identity):
        return {"kernel_name": identity.label()}

    def hsaco_path(self, identity):
        return Path(identity.label())


def _plan(job, problem):
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_cache import KernelCache

    ok, _ = KernelCache.supports_problem(job.identity, problem)
    if not ok:
        return None
    plan, _ = dks.plan_for(
        _FakeCache(),
        job.identity,
        Path("main"),
        {"kernel_name": "main", "spec_wave_size": 64},
        problem,
        "gfx950",
    )
    return plan


def _problem(dtype="fp16", **over) -> DirectConvProblem:
    kw = dict(N=2, H=14, W=14, groups=1, cpg=64, kpg=64, KH=3, KW=3, PAD=1)
    kw.update(over)
    return DirectConvProblem(dtype=dtype, **kw)


# Layers that exercise each plan class (first job of the class the layer
# supports and plans is the one run).
_CASES = {
    "plain": _problem(N=4, H=28, W=28),
    "dw16": _problem(),
    "pk16": _problem(N=4, H=28, W=28, cpg=16, kpg=16),
    "slab": _problem(N=4, H=28, W=28, cpg=16, kpg=16),
    "l64": _problem(N=4, H=28, W=28, PAD=0),
    "grouped": _problem(N=4, H=28, W=28, groups=8, cpg=8, kpg=8, stride=2),
    "gather": _problem(N=2, H=28, W=28, cpg=16, stride=2),
}


# --------------------------------------------------------------------------- CPU


def test_label_counts_per_family():
    """Tiled 286 fp16 / 234 bf16 (212 base-grid + kept points), gather 1504 per
    dtype (64 of them grouped), and the slab fold helper: two out dtypes at
    two slice counts per dtype."""
    for dtype, tiled in (("fp16", 286), ("bf16", 234)):
        labels = {}
        for j in _grid(dtype):
            labels.setdefault(j.identity.algorithm, set()).add(j.identity.label())
        assert len(labels[_TILED]) == tiled
        assert len(labels[_GATHER]) == 1504
        assert len(labels[_FOLD]) == 2
        grouped = [
            j
            for j in _grid(dtype)
            if j.identity.algorithm == _GATHER and _flag(j) == "grouped"
        ]
        assert len(grouped) == 64


def test_kept_tables_are_consistent():
    kept = jobs_mod.WGRAD_TILED_KEPT
    assert len(kept) == len(set(kept)) == 90
    field_names = {f.name for f in fields(DirectWgradTiledSpec)}
    for spec in kept:
        knobs = jobs_mod._wgrad_tiled_knobs(spec)
        assert set(knobs) <= field_names
        # a knob string is its own sorted rendering
        assert list(knobs) == sorted(knobs)
    cell = re.compile(r"^(fp16|bf16)-f\d+x\d+-p\d+x\d+$")
    for key, ids in jobs_mod.WGRAD_TILED_CELL_KEPT.items():
        assert cell.match(key), key
        idx = [int(i) for i in ids.split()]
        assert idx == sorted(set(idx))
        assert all(0 <= i < len(kept) for i in idx)
    # every kept string is offered to some cell
    used = {int(i) for v in jobs_mod.WGRAD_TILED_CELL_KEPT.values() for i in v.split()}
    assert used == set(range(len(kept)))


def test_kept_points_of_a_cell_and_of_all_cells():
    caps = jobs_mod.DirectCaps(KH=5, PAD=2, stride=1, cpg=0, kpg=0, kw=5, pad_w=2)
    cell = list(jobs_mod._wgrad_tiled_kept("fp16", caps))
    assert [jobs_mod._wgrad_tiled_knobs(jobs_mod.WGRAD_TILED_KEPT[10])] == cell[:1]
    assert len(cell) == 2
    assert list(jobs_mod._wgrad_tiled_kept("fp16", None))
    none = jobs_mod.DirectCaps(KH=7, PAD=3, stride=1, cpg=0, kpg=0)
    assert not list(jobs_mod._wgrad_tiled_kept("fp16", none))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_every_plan_class_is_in_the_grid(dtype):
    tiled = [j for j in _grid(dtype) if j.identity.algorithm == _TILED]
    assert {"plain", "dw16", "slab", "l64"} <= {_flag(j) for j in tiled}
    # packed 16-bit atomics are an fp16 class (the pair add is fp16-only)
    assert any(_flag(j) == "pk16" for j in tiled) == (dtype == "fp16")
    assert any(j.spec_kwargs.get("gpack") for j in tiled)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_jobs_round_trip_through_their_identity(dtype):
    from kernels.conv.direct_aot_jobs import direct_job_for_identity

    jobs = _grid(dtype)
    picked = [j for j in jobs if j.identity.algorithm == _FOLD]
    # every plan class of both families, plus a stride sample of the rest
    seen = set()
    for j in jobs:
        key = (j.identity.algorithm, _flag(j))
        if key not in seen:
            seen.add(key)
            picked.append(j)
    picked += jobs[:: max(1, len(jobs) // 40)]
    for job in picked:
        assert direct_job_for_identity(job.identity).identity == job.identity


def test_slab_fold_identity_is_per_out_dtype_and_slices():
    ids = {
        jobs_mod.wgrad_slab_fold_identity("gfx950", 64, out, slices, "fp16").label()
        for out in _DTYPES
        for slices in (2, 8)
    }
    assert len(ids) == 4
    assert all(_FOLD in i for i in ids)


def test_slab_fold_spec_geometry():
    spec = WgradSlabFoldSpec(out="fp16", slices=8)
    grid, block = spec.launch_geometry(36864)
    assert grid[0] >= 1 and block[0] > 0
    assert WgradSlabFoldSpec(out="bf16", slices=2).launch_geometry(36864)[0][0] >= 1


@pytest.mark.parametrize("dtype", _DTYPES)
def test_one_job_per_class_builds(dtype):
    names = set()
    seen = set()
    for job in _grid(dtype):
        key = (job.identity.algorithm, _flag(job))
        if key in seen:
            continue
        seen.add(key)
        kernel, _ = jobs_mod.direct_kernel_for_job(job, "gfx950", dtype)
        assert kernel.name not in names
        names.add(kernel.name)
    assert len(seen) >= 7


def test_kernel_name_marks_each_16bit_flag():
    base = DirectWgradTiledSpec(problem=_problem())
    for flag, mark in (("dw16", "d16"), ("pk16", "pk"), ("slab", "sl"), ("l64", "l64")):
        from dataclasses import replace

        assert f"_{mark}" in replace(base, **{flag: True}).kernel_name()
        assert f"_{mark}" not in base.kernel_name()


def test_dw16_abi_output_dtype_and_signature():
    assert conv_direct_output_dtype("fp16", "wgrad") == "fp32"
    assert conv_direct_output_dtype("fp16", "wgrad", dw16=True) == "fp16"
    assert conv_direct_output_dtype("bf16", "fwd", dw16=True) == "bf16"
    a = conv_direct_args_signature("fp16", direction="wgrad")
    b = conv_direct_args_signature("fp16", direction="wgrad", dw16=True)
    assert a != b and len(a) == len(b)


def _first_plan(dtype, algorithm, flag, problem):
    for job in _grid(dtype):
        if job.identity.algorithm == algorithm and _flag(job) == flag:
            plan = _plan(job, problem)
            if plan is not None:
                return job, plan
    raise AssertionError(f"no {algorithm}/{flag} job plans {problem}")


def test_plain_plan_adds_into_the_fp32_dw():
    _, plan = _first_plan("fp16", _TILED, "plain", _CASES["plain"])
    assert [s.buffers for s in plan.steps] == [("dy", "x", "dw")]
    assert not plan.workspaces


@pytest.mark.parametrize("flag", ["dw16", "pk16"])
def test_16bit_dw_plans_bind_the_dw16_buffer(flag):
    job, plan = _first_plan("fp16", _TILED, flag, _CASES[flag])
    (step,) = plan.steps
    assert step.buffers == ("dy", "x", "dw16")
    assert step.signature == conv_direct_args_signature(
        "fp16", direction="wgrad", dw16=True
    )
    assert flag == _flag(job)


def test_slab_plan_folds_the_slabs_into_the_16bit_dw():
    from kernels.common.conv_direct_wgrad_tiled import WGRAD_SLAB_FOLD_ARGS

    job, plan = _first_plan("fp16", _TILED, "slab", _CASES["slab"])
    main, fold = plan.steps
    p = _CASES["slab"]
    n_el = p.groups * p.kpg * p.KH * p.KW * p.cpg
    parts, rem = divmod(plan.workspaces["ws"], 4 * n_el)
    assert rem == 0 and parts >= 2
    assert main.buffers == ("dy", "x", "ws")
    assert fold.buffers == ("ws", None, "dw16")
    assert dict(fold.scalars) == dict(zip(WGRAD_SLAB_FOLD_ARGS, (parts, n_el)))
    # every slab element is stored, so nothing is zeroed
    assert not plan.zeroed


def test_pk16_plan_zeroes_its_dw16():
    _, plan = _first_plan("fp16", _TILED, "pk16", _CASES["pk16"])
    assert plan.zeroed == ("dw16",)
    _, dw16 = _first_plan("fp16", _TILED, "dw16", _CASES["dw16"])
    assert dw16.zeroed == ()


def _plain_spec(**over):
    import dataclasses

    job = next(
        j
        for j in _grid("fp16")
        if j.identity.algorithm == _TILED and _flag(j) == "plain"
    )
    spec = DirectWgradTiledSpec(problem=_CASES["plain"], **job.spec_kwargs)
    assert is_valid_wgrad_tiled_spec(spec)[0]
    return dataclasses.replace(spec, **over)


@pytest.mark.parametrize(
    "over, message",
    [
        (dict(q_groups=3), "q_groups must be 1, 2 or 4"),
        (dict(pp_sched=True, pingpong=False), "pp_sched needs the ping-pong"),
        (dict(dw16=True), "dw16 stores dW whole: needs one_split"),
        (dict(pk16=True, one_split=True), "pk16 adds atomically"),
        (dict(slab=True, dw16=True, one_split=True), "slab stores fp32 partials"),
        (dict(unfold_n=3), "unfold_n"),
        (dict(unfold_n=8), "unfold_n above 4 needs KW == 1"),
        (dict(pp_tight=True, pingpong=False), "pp_tight needs the ping-pong in halves"),
        (dict(_problem=dict(kpg=18)), "cpg must be a multiple of 8 and kpg of 4"),
        (dict(pk16=True, _problem=dict(dtype="bf16")), "pk16: bf16 pair atomics"),
        (dict(gpack=True), "gpack needs groups > 1"),
        (dict(row_tail=True, rows_per_seg=2), "row_tail needs the auto launch"),
        (
            dict(l64=True, _problem=dict(N=2, H=4096, W=4096, cpg=64, kpg=64)),
            "l64: a segment's images exceed the i32 buffer range",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "-".join(v),
)
def test_tiled_validator_refuses_each_bad_knob(over, message):
    import dataclasses

    over = dict(over)
    changes = over.pop("_problem", None)
    spec = _plain_spec(**over)
    if changes:
        spec = dataclasses.replace(
            spec, problem=dataclasses.replace(spec.problem, **changes)
        )
    ok, why = is_valid_wgrad_tiled_spec(spec)
    assert not ok
    assert message in why


def test_pk16_caps_its_split_count():
    spec = _plain_spec(pk16=True)
    ok, _ = is_valid_wgrad_tiled_spec(spec)
    assert ok == (spec.n_splits() <= 64)
    import dataclasses

    big = dataclasses.replace(spec.problem, N=64, H=56, W=56)
    many = dataclasses.replace(spec, problem=big, target_blocks=1 << 16)
    assert many.n_splits() > 64
    ok, why = is_valid_wgrad_tiled_spec(many)
    assert not ok
    assert "pk16 sums its splits in fp16" in why


def test_slab_plan_refuses_one_split():
    job = next(j for j in _grid("fp16") if _flag(j) == "slab")
    # one image strip per tile and a single block: one split
    tiny = _problem(N=1, H=3, W=3, cpg=16, kpg=16, PAD=0)
    assert _plan(job, tiny) is None


def test_gather_plans_reduce_the_workspace_into_the_fp32_dw():
    for flag in ("plain", "grouped"):
        job, plan = _first_plan(
            "fp16", _GATHER, flag, _CASES["gather" if flag == "plain" else flag]
        )
        main, fold = plan.steps
        assert main.buffers == ("dy", "x", "ws")
        assert fold.buffers == ("ws", None, "dw")
        # every partition slab is stored whole, so the scratch is not zeroed
        assert plan.workspaces == {"ws": plan.workspaces["ws"]} and not plan.zeroed


@pytest.mark.parametrize("cpg", [8, 16])
def test_grouped_gather_cells(cpg):
    p = _problem(groups=8, cpg=cpg, kpg=cpg, stride=2)
    for knobs in jobs_mod.WGRAD_GATHER_GRP:
        wave_k16, waves_k, prefetch, rps, blocks = knobs
        spec = WgradGatherSpec(
            problem=p,
            wave_k16=wave_k16,
            waves_k=waves_k,
            prefetch=prefetch,
            rows_per_seg=rps,
            target_blocks=blocks,
            grouped=True,
        )
        ok, why = is_valid_wgrad_gather_spec(spec)
        assert ok, why
        assert spec.n_atoms == p.KW
        assert spec.c_staged == spec.block_k
        assert spec.x_phase() == 0
    build_wgrad_gather(spec, arch="gfx950")


def test_grouped_gather_refuses_other_group_widths():
    # cpg 4 (kpg % 8) is not a grouped cell, nor is an unequal cpg / kpg
    for over in (dict(cpg=4, kpg=4), dict(cpg=8, kpg=16), dict(cpg=32, kpg=32)):
        spec = WgradGatherSpec(
            problem=_problem(groups=8, stride=2, **over),
            wave_k16=1,
            waves_k=4,
            grouped=True,
        )
        ok, why = is_valid_wgrad_gather_spec(spec)
        assert not ok
        assert "cpg == kpg in (8, 16)" in why


def test_grouped_cells_are_the_gather_cpg_set():
    caps = {
        j.caps["cpg"]
        for j in _grid("fp16")
        if j.identity.algorithm == _GATHER and _flag(j) == "grouped"
    }
    assert caps == set(jobs_mod.WGRAD_GATHER_GRP_CPG)


# --------------------------------------------------------------------------- GPU


def _compiled(job, dtype, cache):
    from rocke import compile_kernel

    kernel, meta = jobs_mod.direct_kernel_for_job(job, "gfx950", dtype)
    art = compile_kernel(kernel, arch="gfx950")
    return art, meta


class _CompilingCache:
    """Compiles each helper (fold / reduce) job of a plan on first lookup."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.hsacos: dict[str, bytes] = {}
        self.names: dict[str, str] = {}

    def entry_meta(self, identity):
        key = identity.label()
        if key not in self.hsacos:
            art, _ = _compiled(
                jobs_mod.direct_job_for_identity(identity), self.dtype, self
            )
            self.hsacos[key] = art.hsaco
            self.names[key] = art.kernel_name
        return {"kernel_name": self.names[key]}

    def hsaco_path(self, identity):
        return Path(identity.label())


def _run(job, problem, dtype, x, dy):
    """The whole plan of ``job`` on ``problem``; returns dW as host floats."""
    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    from benchmarks.common.kernel_cache import KernelCache

    ok, why = KernelCache.supports_problem(job.identity, problem)
    assert ok, why
    cache = _CompilingCache(dtype)
    art, meta = _compiled(job, dtype, cache)
    cache.hsacos["main"], cache.names["main"] = art.hsaco, art.kernel_name
    plan, why = dks.plan_for(
        cache,
        job.identity,
        Path("main"),
        {"kernel_name": art.kernel_name, **meta},
        problem,
        "gfx950",
    )
    assert plan is not None, why
    k_shape = (problem.total_k, problem.KH, problem.KW, problem.cpg)
    dw_fp32 = "dw" in {b for s in plan.steps for b in s.buffers}
    dw_name = "dw" if dw_fp32 else "dw16"
    out_dtype = "fp32" if dw_fp32 else dtype
    # every output starts as garbage the plan has to overwrite, except what the
    # plan itself says it zeroes or accumulates into
    accumulates = (
        any(s.buffers[2] == "dw" for s in plan.steps)
        or dw_name == "dw16"
        and job.spec_kwargs.get("pk16")
    )
    dw0 = (
        np.zeros(k_shape, np.float32)
        if dw_fp32
        else (
            np.zeros(k_shape, storage.host_dtype(dtype))
            if accumulates
            else storage.encode(np.full(k_shape, np.nan), dtype)
        )
    )
    bufs = {
        "dy": DeviceArray(dy, dtype),
        "x": DeviceArray(x, dtype),
        dw_name: DeviceArray(dw0, out_dtype),
    }
    for name, nbytes in plan.workspaces.items():
        bufs[name] = DeviceArray(np.zeros(nbytes, np.uint8), "u8")
    ptrs = {n: b.data_ptr() for n, b in bufs.items()}
    sizes = {n: b.nbytes for n, b in bufs.items()}
    for step in plan.steps:
        key = str(step.hsaco_path)
        KernelLauncher(
            hsaco=cache.hsacos[key],
            kernel_name=step.kernel_name,
            signature=step.signature,
        )(
            dks.launch_values(step, ptrs, sizes),
            config=LaunchConfig(grid=step.grid, block=step.block, fence=True),
        )
    out = bufs[dw_name].read()
    return storage.decode(out, out_dtype) if not dw_fp32 else out


_GPU_CASES = [
    ("plain", _TILED, "fp16"),
    ("plain", _TILED, "bf16"),
    ("dw16", _TILED, "fp16"),
    ("dw16", _TILED, "bf16"),
    ("pk16", _TILED, "fp16"),
    ("slab", _TILED, "fp16"),
    ("slab", _TILED, "bf16"),
    ("l64", _TILED, "fp16"),
    ("grouped", _GATHER, "fp16"),
    ("grouped", _GATHER, "bf16"),
    ("gather", _GATHER, "fp16"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("flag, algorithm, dtype", _GPU_CASES, ids=lambda v: str(v))
def test_wgrad_matches_the_reference(flag, algorithm, dtype):
    current_device_arch_in(("gfx950",), subject="the tiled and gather wgrad")
    case = "gather" if (algorithm == _GATHER and flag == "gather") else flag
    problem = _problem(dtype=dtype, **_case_kwargs(case))
    flag_key = "plain" if flag == "gather" else flag
    job = None
    for cand in _grid(dtype):
        if cand.identity.algorithm == algorithm and _flag(cand) == flag_key:
            if _plan(cand, problem) is not None:
                job = cand
                break
    assert job is not None, "no grid job of the class plans this layer"
    rng = np.random.default_rng(0)
    x = stored_uniform(rng, (problem.N, problem.H, problem.W, problem.total_c), dtype)
    dy = stored_uniform(
        rng, (problem.N, problem.Ho, problem.Wo, problem.total_k), dtype
    )
    dw = _run(job, problem, dtype, x, dy)
    ref = direct_wgrad_reference(
        problem, storage.decode(x, dtype), storage.decode(dy, dtype)
    )
    assert not np.isnan(dw).any()
    tol = 2e-2 if dtype == "bf16" else 5e-3
    assert peak_relative_error(dw, ref, floor=1.0) < tol


def _case_kwargs(case: str) -> dict:
    p = _CASES[case]
    return dict(
        N=p.N,
        H=p.H,
        W=p.W,
        groups=p.groups,
        cpg=p.cpg,
        kpg=p.kpg,
        KH=p.KH,
        KW=p.KW,
        PAD=p.PAD,
        stride=p.stride,
    )


# ------------------------------------------------- 1xN exchange and batch fold


def _wg(**over) -> DirectConvProblem:
    kw = dict(N=3, H=1, W=40, groups=1, cpg=64, kpg=64, KH=1, KW=3, PAD=0)
    kw.update(over)
    return DirectConvProblem(dtype="fp16", **kw)


def test_exchanged_hw_problem_swaps_the_axes():
    from benchmarks.common.direct_kernel_sweep import exchanged_hw_problem

    q = exchanged_hw_problem(_wg(pad_w=0))
    assert (q.H, q.W, q.KH, q.KW) == (40, 1, 3, 1)
    assert q.N == 3 and q.cpg == 64 and q.kpg == 64
    for bad in (
        dict(H=2),
        dict(KH=3),
        dict(KW=1),
        dict(PAD=1),
        dict(stride=2),
        dict(dil_w=2),
    ):
        assert exchanged_hw_problem(_wg(**bad)) is None


def test_folded_n_wgrad_problem_folds_one_row_images():
    from benchmarks.common.direct_kernel_sweep import folded_n_wgrad_problem

    q = folded_n_wgrad_problem(_wg(KW=1))
    assert (q.N, q.H, q.W) == (1, 3, 40)
    for bad in (dict(N=1), dict(H=2), dict(KH=3), dict(PAD=1), dict(stride=2)):
        assert folded_n_wgrad_problem(_wg(KW=1, **bad)) is None


class _ListCache(_FakeCache):
    """A cache serving the grid's wgrad jobs that support the asked problem."""

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs

    def compatible(self, problem, *, direction=None):
        from benchmarks.common.kernel_cache import KernelCache

        return [
            (j.identity, Path("main"), {"kernel_name": "main", "spec_wave_size": 64})
            for j in self.jobs
            if j.identity.direction == direction
            and KernelCache.supports_problem(j.identity, problem)[0]
        ]


@pytest.mark.parametrize(
    "flag, problem",
    [
        ("WGRAD_EXCHANGE_HW", _wg(pad_w=0)),
        ("WGRAD_FOLD_N", _wg()),
    ],
    ids=["exchange", "fold"],
)
def test_direct_plans_route_one_row_wgrad_through_the_cached_kernels(
    monkeypatch, flag, problem
):
    """The default-on routes of ``direct_plans``: the exchanged (1xN -> Nx1)
    and the batch-folded plans appear, and vanish with their switch off."""
    from benchmarks.common import direct_kernel_sweep as dks

    cache = _ListCache(_grid("fp16"))
    monkeypatch.setattr(dks, "WGRAD_EXCHANGE_HW", False)
    monkeypatch.setattr(dks, "WGRAD_FOLD_N", False)
    off, _ = dks.direct_plans(cache, problem, "wgrad", "gfx950")
    monkeypatch.setattr(dks, flag, True)
    on, _ = dks.direct_plans(cache, problem, "wgrad", "gfx950")
    assert len(on) > len(off)
