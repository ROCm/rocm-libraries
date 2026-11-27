# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx950 variants of the implicit-GEMM convolution.

Forward (``ImplicitGemmConvSpec``): the register-prefetch loop (``pgr``), the
16-element LDS K-row pad (``lds_k_pad=16``), the width-folded patch GEMM
(``patch``) with its XCD-grouped tile order (``chiplet_swizzle``) and its
DMA-staged two-stage K loop (``dtl``). Data gradient (``DgradConvSpec``): the
same knobs on the patch and grouped GEMMs, tile_k 128, patch split-K through the
cshuffle epilogue, the uniform-K peel and the zero-residue fill of 1x1 stride-2
shapes.

CPU: the validators (verdicts and reasons), a kernel build for every knob class
in both dtypes (``dtl`` and ``chiplet_swizzle`` included, forward and dgrad),
the AOT grid counts per knob class, the label round trip, the refusal of the new
forward identities by the packaged contract, the binary ABI, and, with the C++
engine importable, its IR against Python's. GPU (gfx950): the output against a
float64 numpy reference on small problems, both dtypes, with the output
poisoned so an element the kernel does not write fails.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_conv_implicit_gemm_gfx950_variants.py -v
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.device_array import DeviceArray
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.lanes import BINDING_LANE, import_or_skip_until

from builders.common.conv_reference import conv_dgrad_reference, conv_fwd_reference
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    implicit_gemm_conv_grid,
    is_valid_spec_for_problem,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    dgrad_fuses_dx_zero_fill,
    is_valid_dgrad_spec,
    pack_sub_gemm_buffer,
)

from ._direct_conv_numeric import poisoned, stored_uniform

_DTYPES = ("fp16", "bf16")
_ARCH = "gfx950"


# ------------------------------------------------------------------ forward rows

# The width-folded patch embedding a patch kernel runs: a 2 x 1 filter at stride
# (2, 1), C a whole number of 16-byte chunks of a K row.
_FWD_PATCH = dict(N=2, Hi=18, Wi=8, C=128, K=96, Y=2, X=1, sH=2, sW=1)
_FWD_GROUPED = dict(N=2, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2)
_FWD_PLAIN = dict(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
_WT32 = dict(warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)

# (id, problem, spec knobs, name tags): every knob class of the forward grid.
_FWD_ROWS = [
    ("patch", _FWD_PATCH, {}, ("patch",)),
    ("patch_chip", _FWD_PATCH, dict(chiplet_swizzle=True), ("patch",)),
    (
        "patch_chip_pgr_compv4",
        _FWD_PATCH,
        dict(
            _WT32,
            tile_m=64,
            tile_n=64,
            pipeline="compv4",
            chiplet_swizzle=True,
            prefetch_global_regs=True,
        ),
        ("patch", "pgr"),
    ),
    ("patch_chip_dtl", _FWD_PATCH, dict(chiplet_swizzle=True, dtl=True), ("dtl",)),
    (
        "patch_chip_dtl_wt32",
        _FWD_PATCH,
        dict(_WT32, tile_m=64, tile_n=64, chiplet_swizzle=True, dtl=True),
        ("dtl",),
    ),
    (
        "patch_tk32_compv3",
        _FWD_PATCH,
        dict(tile_m=64, tile_n=64, tile_k=32, pipeline="compv3"),
        ("patch",),
    ),
    (
        "patch_dtl_tk32",
        _FWD_PATCH,
        dict(tile_m=64, tile_n=64, tile_k=32, dtl=True),
        ("dtl",),
    ),
    ("kpad16", _FWD_PLAIN, dict(lds_k_pad=16), ()),
    (
        "grouped_kpad16",
        _FWD_GROUPED,
        dict(lds_k_pad=16, tile_n=64, warp_m=1, warp_n=4),
        (),
    ),
    (
        "grouped_kpad16_pgr_compv3",
        _FWD_GROUPED,
        dict(
            lds_k_pad=16,
            tile_m=16,
            tile_n=16,
            warp_m=1,
            warp_n=1,
            pipeline="compv3",
            prefetch_global_regs=True,
        ),
        ("pgr",),
    ),
]


def _fwd_spec(row, dtype, **extra):
    _, pkw, knobs, _ = row
    base = dict(
        problem=ConvProblem(**pkw),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=32,
        tile_n=32,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        wave_size=64,
        pipeline="mem",
        epilogue="cshuffle",
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        patch="patch" in row[0],
    )
    base.update(knobs)
    base.update(extra)
    return ImplicitGemmConvSpec(**base)


# ------------------------------------------------------------------- dgrad rows

# The patch embedding the dgrad patch kernels run (N=2 so M has a second image).
_DG_PATCH = dict(N=2, Hi=16, Wi=16, C=64, K=64, Y=2, X=2, sH=2, sW=2)
_DG_GROUPED_S2 = dict(
    N=1, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, sH=2, sW=2, pH=1, pW=1, groups=2
)
_DG_S1 = dict(N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
_DG_1X1_S2 = dict(N=2, Hi=16, Wi=16, C=64, K=96, Y=1, X=1, sH=2, sW=2)
_DG_CHIP_DTL = dict(uniform_k_decode=False, chiplet_swizzle=True, dtl=True)

# (id, problem, spec knobs, name tags)
_DGRAD_ROWS = [
    ("patch_chip_dtl", _DG_PATCH, _DG_CHIP_DTL, ("patch", "dtl", "chip")),
    (
        "patch_chip_dtl_tk128",
        _DG_PATCH,
        dict(_DG_CHIP_DTL, tile_m=32, tile_n=64, tile_k=128),
        ("patch", "dtl", "chip"),
    ),
    (
        "patch_chip_compv4",
        _DG_PATCH,
        dict(tile_m=64, tile_n=128, warp_m=4, pipeline="compv4", chiplet_swizzle=True),
        ("patch", "chip", "ukd"),
    ),
    (
        "patch_chip_pgr_kpad16",
        _DG_PATCH,
        dict(chiplet_swizzle=True, prefetch_global_regs=True, lds_k_pad=16),
        ("patch", "chip", "pgr", "ukd"),
    ),
    (
        "patch_pgr_kpad16_tk128",
        _DG_PATCH,
        dict(
            tile_m=32,
            tile_n=64,
            tile_k=128,
            prefetch_global_regs=True,
            lds_k_pad=16,
        ),
        ("patch", "pgr", "ukd"),
    ),
    (
        "grouped_s2_pgr_kpad16",
        _DG_GROUPED_S2,
        dict(
            _WT32,
            warp_n=4,
            pipeline="compv3",
            patch=False,
            prefetch_global_regs=True,
            lds_k_pad=16,
        ),
        ("pgr", "ukd"),
    ),
    (
        "s1_m_outer_pgr",
        _DG_S1,
        dict(
            _WT32,
            tile_m=64,
            tile_n=64,
            epilogue="default",
            lds_k_outer=False,
            vector_size_c=None,
            patch=False,
            prefetch_global_regs=True,
        ),
        ("pgr", "ukd"),
    ),
    # K == tile_k: the second K slice lies wholly past K.
    (
        "patch_split_k2",
        _DG_PATCH,
        dict(split_k=2, uniform_k_decode=False),
        ("patch", "spk2"),
    ),
    # Three K tiles over two slices: each slice is two whole tiles.
    (
        "patch_split_k2_k192",
        dict(_DG_PATCH, K=192),
        dict(split_k=2, uniform_k_decode=False),
        ("patch", "spk2"),
    ),
    (
        "k96_peel_zero_fill",
        _DG_1X1_S2,
        dict(tile_m=64, tile_n=64, patch=False),
        ("ukd",),
    ),
]


def _dgrad_spec(row, dtype, **extra):
    _, pkw, knobs, _ = row
    base = dict(
        problem=ConvProblem(**pkw),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=128,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        pipeline="mem",
        epilogue="cshuffle",
        lds_k_outer=True,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        patch=True,
        uniform_k_decode=True,
    )
    base.update(knobs)
    base.update(extra)
    return DgradConvSpec(**base)


# --------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _FWD_ROWS, ids=lambda r: r[0])
def test_fwd_rows_are_valid_and_build(row, dtype):
    spec = _fwd_spec(row, dtype)
    ok, why = is_valid_spec_for_problem(spec, spec.problem, _ARCH)
    assert ok, why
    kernel = build_implicit_gemm_conv(spec, arch=_ARCH)
    assert kernel.name == spec.kernel_name()
    for tag in row[3]:
        assert f"_{tag}" in kernel.name, (tag, kernel.name)
    assert ("_dtl" in kernel.name) == bool(spec.dtl)
    assert ("_patch" in kernel.name) == bool(spec.patch)


def test_dtl_ir_compiles_and_links():
    """The dtl K loops' LDS DMAs are inline asm with a buffer descriptor
    operand: a forward and a dgrad dtl row must give a module LLVM accepts."""
    from rocke.tests.ir_validity import assert_kernel_target_accepts

    fwd = _fwd_spec(next(r for r in _FWD_ROWS if r[0] == "patch_chip_dtl"), "fp16")
    assert_kernel_target_accepts(build_implicit_gemm_conv(fwd, arch=_ARCH), _ARCH)
    dg = _dgrad_spec(next(r for r in _DGRAD_ROWS if r[0] == "patch_chip_dtl"), "bf16")
    assert_kernel_target_accepts(build_implicit_gemm_conv_dgrad(dg, arch=_ARCH), _ARCH)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _DGRAD_ROWS, ids=lambda r: r[0])
def test_dgrad_rows_are_valid_and_build(row, dtype):
    spec = _dgrad_spec(row, dtype)
    ok, why = is_valid_dgrad_spec(spec, _ARCH)
    assert ok, why
    kernel = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
    assert kernel.name == spec.kernel_name()
    for tag in row[3]:
        assert f"_{tag}" in kernel.name, (tag, kernel.name)
    assert ("_dtl" in kernel.name) == bool(spec.dtl)
    assert ("_chip" in kernel.name) == bool(spec.chiplet_swizzle)


def test_variant_knobs_change_the_kernel():
    """Each knob is a distinct kernel: the same row with it off lowers to other IR,
    and with it on, to a name of its own."""
    from rocke.core.ir_serialize import serialize

    def ir(kernel):
        return serialize(kernel).replace(kernel.name, "k")

    base = _fwd_spec(_FWD_ROWS[0], "fp16")
    names = {build_implicit_gemm_conv(base, arch=_ARCH).name}
    for knobs in (
        dict(chiplet_swizzle=True),
        dict(dtl=True),
        dict(prefetch_global_regs=True),
        dict(lds_k_pad=16),
    ):
        other = replace(base, **knobs)
        a = build_implicit_gemm_conv(base, arch=_ARCH)
        b = build_implicit_gemm_conv(other, arch=_ARCH)
        assert ir(a) != ir(b), knobs
        names.add(b.name)
    # chiplet_swizzle and lds_k_pad change the IR but not the forward name (the
    # AOT label and identity carry them); dtl and pgr tag it.
    assert len(names) == 3

    dg = _dgrad_spec(_DGRAD_ROWS[2], "fp16")
    for knobs in (
        dict(chiplet_swizzle=False),
        dict(lds_k_pad=16),
        dict(prefetch_global_regs=True),
    ):
        a = build_implicit_gemm_conv_dgrad(dg, arch=_ARCH)
        b = build_implicit_gemm_conv_dgrad(replace(dg, **knobs), arch=_ARCH)
        assert ir(a) != ir(b), knobs


def test_fwd_refusals():
    """The validator verdicts and reasons of the forward knobs; the builder raises
    the same reason."""
    row = _FWD_ROWS[1]  # patch_chip

    def refused(why, **kw):
        spec = _fwd_spec(row, "fp16", **kw)
        problem = kw.get("problem", spec.problem)
        ok, reason = is_valid_spec_for_problem(spec, problem, _ARCH)
        assert not ok and why in reason, (kw, reason)
        with pytest.raises(ValueError, match=why):
            build_implicit_gemm_conv(spec, arch=_ARCH)

    refused("dtl is built on the patch GEMM only", patch=False, dtl=True)
    refused("dtl has its own K loop", dtl=True, prefetch_global_regs=True)
    refused("dtl has its own K loop", dtl=True, unroll_k=True)
    refused("dtl: 16-bit operands, packed LDS tiles", dtl=True, lds_k_pad=16)
    refused("dtl: tile_k / 8 must be a power of two", dtl=True, tile_k=96)
    refused("patch is not built with split-K", split_k=2)
    refused(
        "patch serves width-folded patch embeddings only",
        problem=ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1),
    )
    refused(
        "patch serves width-folded patch embeddings only",
        problem=ConvProblem(**dict(_FWD_PATCH, sH=1)),  # sH < Y: taps overlap
    )
    # Other architectures: no 16-byte LDS DMA on gfx942, wave32 on gfx1250.
    for arch in ("gfx942", "gfx1250"):
        spec = _fwd_spec(row, "fp16", dtl=True)
        assert not is_valid_spec_for_problem(spec, spec.problem, arch)[0]


def test_dgrad_refusals():
    """The dgrad knob gates. ``is_valid_dgrad_spec`` (the AOT grid's predicate) and
    ``validate`` (the builder's) each give their reason; the chiplet order outside
    the patch GEMM is the grid predicate's refusal only."""
    patch, split = _DGRAD_ROWS[0], _DGRAD_ROWS[7]
    s2 = ("s2", dict(_DG_S1, sH=2, sW=2), {}, ())
    for row, kw, why, why_build in (
        (patch, dict(patch=False), "dtl is built on the K-outer patch GEMM only", None),
        (
            patch,
            dict(prefetch_global_regs=True),
            "prefetch_global_regs needs the uniform_k_decode loop",
            "dtl has its own K loop (no pgr / ukd)",
        ),
        (
            patch,
            dict(prefetch_global_regs=True, uniform_k_decode=True),
            "dtl has its own K loop (no pgr / ukd)",
            None,
        ),
        (
            _DGRAD_ROWS[3],
            dict(uniform_k_decode=False),
            "prefetch_global_regs needs the uniform_k_decode loop",
            None,
        ),
        (
            split,
            dict(epilogue="default", vector_size_c=None),
            "patch with split_k > 1 needs the cshuffle epilogue",
            None,
        ),
        (patch, dict(split_k=2), "dtl has its own K loop (no split-K", None),
        (
            s2,
            dict(patch=False, chiplet_swizzle=True),
            "dgrad does not implement chiplet_swizzle",
            "",
        ),
    ):
        spec = _dgrad_spec(row, "fp16", **kw)
        ok, reason = is_valid_dgrad_spec(spec, _ARCH)
        assert not ok and why in reason, (kw, reason)
        if why_build != "":
            with pytest.raises(ValueError, match=re.escape(why_build or why)):
                spec.validate()
    # Accepted off the refused flag: the same rows with it dropped.
    for row, kw in ((patch, dict(dtl=False, tile_k=128)), (patch, dict(lds_k_pad=16))):
        ok, reason = is_valid_dgrad_spec(_dgrad_spec(row, "fp16", **kw), _ARCH)
        assert ok, reason


def test_zero_residue_fill_plan():
    """The host plan: which dgrad shapes the kernel itself zero-fills (so the plan
    needs no memset of dX), and which it does not."""
    p = ConvProblem(**_DG_1X1_S2)
    assert dgrad_fuses_dx_zero_fill(p, dtype_d="fp16")
    assert dgrad_fuses_dx_zero_fill(p, dtype_d="bf16")
    assert not dgrad_fuses_dx_zero_fill(p, dtype_d="fp32")
    assert not dgrad_fuses_dx_zero_fill(p, dtype_d="fp16", split_k=2)
    assert not dgrad_fuses_dx_zero_fill(p, dtype_d="fp16", pipeline="wavelet")
    assert not dgrad_fuses_dx_zero_fill(replace(p, sH=1, sW=1), dtype_d="fp16")
    assert not dgrad_fuses_dx_zero_fill(replace(p, pH=1, pW=1), dtype_d="fp16")
    assert not dgrad_fuses_dx_zero_fill(replace(p, C=60), dtype_d="fp16")
    assert not dgrad_fuses_dx_zero_fill(ConvProblem(**_DG_S1), dtype_d="fp16")


def test_binary_abi_is_24():
    """The forward / dgrad variants widen the kernel spec the binary takes: ABI 24."""
    from rocke.core.native_engine import BINARY_ABI

    assert BINARY_ABI == 24


# ---------------------------------------------------------------------- AOT grid

# Forward jobs per dtype that carry a knob of this PR, by knob class. The grid
# builds the register-prefetch loop and the K pad on a cover of geometries, the
# patch GEMM on its listed geometries and the dtl loop on three of them.
_FWD_CLASS_COUNTS = {
    ("pgr",): 99,
    ("kpad16",): 63,
    ("pgr", "kpad16"): 108,
    ("patch", "chip"): 14,
    ("patch", "chip", "pgr"): 14,
    ("patch", "chip", "dtl"): 3,
}
# Dgrad jobs per dtype on a knob class of this PR (patch jobs of the plain
# kind are not counted: the grid had them before).
_DGRAD_CLASS_COUNTS = {
    ("kpad16",): 1,
    ("pgr",): 1,
    ("pgr", "kpad16"): 1,
    ("patch", "chip"): 5,
    ("patch", "chip", "dtl"): 1,
    ("patch", "chip", "dtl", "tk128"): 1,
    ("patch", "chip", "tk128"): 2,
    ("patch", "chip", "kpad16"): 5,
    ("patch", "chip", "kpad16", "tk128"): 2,
    ("patch", "chip", "pgr"): 5,
    ("patch", "chip", "pgr", "tk128"): 2,
    ("patch", "chip", "pgr", "kpad16"): 5,
    ("patch", "chip", "pgr", "kpad16", "tk128"): 2,
    ("patch", "kpad16"): 9,
    ("patch", "kpad16", "tk128"): 3,
    ("patch", "tk128"): 3,
    ("patch", "pgr"): 9,
    ("patch", "pgr", "tk128"): 3,
    ("patch", "pgr", "kpad16"): 9,
    ("patch", "pgr", "kpad16", "tk128"): 3,
}


def _class_of(identity):
    flags = (
        ("patch", identity.patch),
        ("chip", identity.chiplet_swizzle),
        ("dtl", identity.dtl),
        ("pgr", identity.prefetch_global_regs),
        ("kpad16", identity.lds_k_pad == 16),
        ("tk128", identity.tile_k == 128),
    )
    return tuple(name for name, on in flags if on)


@pytest.fixture(scope="module")
def grid_jobs():
    """The gfx950 implicit-GEMM forward and dgrad AOT jobs, per dtype."""
    from benchmarks.common.kernel_sweep import enumerate_jobs
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(_ARCH)
    return {
        dtype: [
            j
            for j in enumerate_jobs(
                arch=_ARCH,
                dtype=dtype,
                target=target,
                directions=("fwd", "dgrad"),
                jobs=32,
            )
            if j.identity.algorithm == "implicit_gemm"
        ]
        for dtype in _DTYPES
    }


def _by_class(jobs, direction):
    classes: dict[tuple, list] = {}
    for job in jobs:
        i = job.identity
        if i.direction == direction and _class_of(i):
            classes.setdefault(_class_of(i), []).append(job)
    return classes


@pytest.mark.parametrize("dtype", _DTYPES)
def test_fwd_grid_counts_per_knob_class(grid_jobs, dtype):
    classes = _by_class(grid_jobs[dtype], "fwd")
    assert {k: len(v) for k, v in classes.items()} == _FWD_CLASS_COUNTS
    assert sum(_FWD_CLASS_COUNTS.values()) == 301


@pytest.mark.parametrize("dtype", _DTYPES)
def test_dgrad_grid_counts_per_knob_class(grid_jobs, dtype):
    """The trimmed dgrad grid: the variants sit on the geometries of the listed
    cover, not on the whole walk."""
    classes = _by_class(grid_jobs[dtype], "dgrad")
    got = {k: len(v) for k, v in classes.items() if k != ("patch",)}
    assert got == _DGRAD_CLASS_COUNTS
    # No variant job off gfx950 or off 16-bit data.
    from kernels.conv import aot_jobs as aj

    assert aj.CACHE_DGRAD_VARIANT_ARCHS == {_ARCH}
    assert aj.CACHE_DGRAD_VARIANT_DTYPES == ("fp16", "bf16")
    assert len(aj.CACHE_DGRAD_VARIANT_GEOMETRIES) == 12


def test_other_arches_get_none_of_the_variants():
    from benchmarks.common.kernel_sweep import enumerate_jobs
    from rocke.core.arch import ArchTarget

    for arch in ("gfx942", "gfx1250"):
        jobs = enumerate_jobs(
            arch=arch,
            dtype="fp16",
            target=ArchTarget.from_gfx(arch),
            directions=("fwd", "dgrad"),
            jobs=32,
        )
        assert jobs
        assert not [
            j
            for j in jobs
            if j.identity.algorithm == "implicit_gemm"
            and (
                j.identity.dtl
                or j.identity.chiplet_swizzle
                or j.identity.lds_k_pad == 16
                or (j.identity.direction == "dgrad" and j.identity.prefetch_global_regs)
            )
        ]


def _samples(jobs_by_class, per_class):
    picked = []
    for cls, jobs in sorted(jobs_by_class.items()):
        step = max(1, len(jobs) // per_class)
        picked.extend((cls, j) for j in jobs[::step][:per_class])
    return picked


@pytest.mark.parametrize("dtype", _DTYPES)
def test_every_knob_class_builds_a_kernel_def(grid_jobs, dtype):
    """A KernelDef for jobs of every knob class (two each, every dtl job), in
    both directions: dtl, chiplet, pgr, K pad, tile_k 128 included."""
    from kernels.conv import aot_jobs as aj

    built = set()
    for direction in ("fwd", "dgrad"):
        classes = _by_class(grid_jobs[dtype], direction)
        classes.pop(("patch",), None)  # the plain patch kernel is not this PR's
        picked = _samples(classes, 2)
        picked += [(c, j) for c, js in classes.items() if "dtl" in c for j in js]
        for cls, job in picked:
            kernel, _ = aj.kernel_for_job(job, _ARCH, dtype)
            assert kernel.name
            built.add((direction, cls))
            assert ("dtl" in cls) == ("_dtl" in kernel.name)
    want = {("fwd", c) for c in _FWD_CLASS_COUNTS}
    want |= {("dgrad", c) for c in _DGRAD_CLASS_COUNTS}
    assert built == want
    assert any("dtl" in c for d, c in built if d == "fwd")
    assert any("dtl" in c for d, c in built if d == "dgrad")
    assert any("chip" in c for d, c in built if d == "fwd")
    assert any("chip" in c for d, c in built if d == "dgrad")


# The labels of the dgrad twins the grid keeps (fp16; bf16 swaps the dtype).
_TWIN_LABELS = (
    "t128x128x64-w2x2-a16x16x32-v8x8x8-compv4-cshuffle-chip-kpad16-kouter-ukd-patch",
    "t128x128x64-w2x2-a16x16x32-v8x8x8-mem-cshuffle-chip-kpad16-kouter-pgr-ukd-patch",
    "t128x128x64-w2x4-a32x32x16-v8x8x8-compv3-cshuffle-grp-kouter-pgr-ukd",
    "t128x256x128-w2x8-a16x16x32-v8x8x8-compv3-cshuffle-kpad16-kouter-pgr-ukd-patch",
    "t128x256x64-w1x4-a16x16x32-v8x8x8-compv4-cshuffle-chip-kpad16-kouter-ukd-patch",
    "t128x256x64-w2x4-a16x16x32-v8x8x8-compv4-cshuffle-chip-kpad16-kouter-ukd-patch",
    "t128x256x64-w2x8-a16x16x32-v8x8x8-mem-cshuffle-kpad16-kouter-pgr-ukd-patch",
    "t256x256x64-w4x4-a16x16x32-v8x8x8-mem-cshuffle-kpad16-kouter-pgr-ukd-patch",
    "t32x64x128-w2x2-a16x16x32-v8x8x8-mem-cshuffle-chip-kpad16-kouter-pgr-ukd-patch",
    "t32x64x128-w2x2-a16x16x32-v8x8x8-mem-cshuffle-chip-kpad16-kouter-ukd-patch",
    "t64x128x64-w2x2-a16x16x32-v8x8x8-compv4-cshuffle-kpad16-kouter-pgr-ukd-patch",
    "t64x128x64-w4x2-a16x16x32-v8x8x8-compv4-cshuffle-chip-kpad16-kouter-pgr-ukd-patch",
    "t64x32x128-w4x2-a16x16x32-v8x8x8-mem-cshuffle-chip-kpad16-kouter-ukd-patch",
)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_dgrad_grid_keeps_the_listed_twins(grid_jobs, dtype):
    labels = {
        j.identity.label()
        for j in grid_jobs[dtype]
        if j.identity.direction == "dgrad" and _class_of(j.identity)
    }
    for tail in _TWIN_LABELS:
        want = f"dgrad-implicit_gemm-{dtype}-2d-{tail}-msg1024-wave64-gfx950"
        assert any(label.startswith(want) for label in labels), want


@pytest.mark.parametrize("dtype", _DTYPES)
def test_labels_round_trip_and_rebuild(grid_jobs, dtype):
    """Every knob class's label decodes to its identity, carries its tags, and
    maps back to the job the grid builds it from."""
    from kernels.conv import aot_jobs as aj
    from kernels.conv.kernel_identity import KernelIdentity
    from kernels.conv.kernel_label import decode

    tags = {
        "patch": "-patch",
        "chip": "-chip",
        "dtl": "-dtl",
        "pgr": "-pgr",
        "kpad16": "-kpad16",
    }
    for direction in ("fwd", "dgrad"):
        for cls, job in _samples(_by_class(grid_jobs[dtype], direction), 3):
            identity = job.identity
            label = identity.label()
            assert decode(label, KernelIdentity) == identity, label
            for name, token in tags.items():
                assert (name in cls) == (token in label), (label, name)
            assert aj.job_for_identity(identity).identity == identity


@pytest.mark.parametrize("dtype", _DTYPES)
def test_new_forward_identities_are_not_packageable(grid_jobs, dtype):
    """The packaged forward contract refuses every forward identity that carries
    one of the new knobs, each with its reason: the new forward labels are
    rocKE-cache only. A forward identity of the same walk with none of them is
    still packageable."""
    from builders.common.convolution_forward import (
        gfx950_conv_fwd_aot_spec_for_identity,
    )

    fields = {
        "patch": "patch=True",
        "chip": "chiplet_swizzle=True",
        "dtl": "dtl=True",
        "pgr": "prefetch_global_regs=True",
        "kpad16": "lds_k_pad=16",
    }
    classes = _by_class(grid_jobs[dtype], "fwd")
    for cls, job in sorted(classes.items()):
        for sample in (job[0], job[-1]):
            spec, why = gfx950_conv_fwd_aot_spec_for_identity(sample.identity)
            assert spec is None, (sample.identity.label(), why)
            assert "is not packaged" in why, why
            # The reason names the first knob the contract meets.
            assert any(fields[name] in why for name in cls), (cls, why)
    for name, field in fields.items():
        sample = next(j for c, js in sorted(classes.items()) if name in c for j in js)
        spec, why = gfx950_conv_fwd_aot_spec_for_identity(
            replace(
                sample.identity,
                **{
                    "patch": False,
                    "chiplet_swizzle": False,
                    "dtl": False,
                    "prefetch_global_regs": False,
                    "lds_k_pad": 0,
                },
            )
        )
        # Off the knobs the identity is a plain sweep one only when the plain
        # grid carries it; either way no refusal names a knob of this PR.
        assert not any(f in why for f in fields.values()), (name, why)


# ------------------------------------------------------------------ C++ engine


def _cpp_lower(eng, spec):
    from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

    return eng.conv_implicit_gemm_lower_llvm(
        conv_implicit_gemm_spec_to_dict(spec), _ARCH
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _FWD_ROWS, ids=lambda r: r[0])
def test_fwd_cpp_engine_emits_the_python_ir(row, dtype):
    """The C++ engine lowers each forward row to the Python engine's LLVM text
    (patch, chiplet, dtl, pgr, K pad); it refuses what Python refuses."""
    eng = import_or_skip_until("rocke_library_engine", BINDING_LANE)
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    spec = _fwd_spec(row, dtype)
    python = lower_kernel_to_llvm(
        build_implicit_gemm_conv(spec, arch=_ARCH), arch=_ARCH
    )
    assert _cpp_lower(eng, spec) == python


def test_fwd_cpp_engine_refuses_like_python():
    eng = import_or_skip_until("rocke_library_engine", BINDING_LANE)
    row = _FWD_ROWS[1]
    for why, kw in (
        ("dtl is built on the patch GEMM only", dict(patch=False, dtl=True)),
        ("dtl has its own K loop", dict(dtl=True, prefetch_global_regs=True)),
        ("patch is not built with split-K", dict(split_k=2)),
    ):
        spec = _fwd_spec(row, "fp16", **kw)
        with pytest.raises(RuntimeError, match=why):
            _cpp_lower(eng, spec)


# --------------------------------------------------------------------------- GPU


def _tol(dtype):
    return 2e-2 if dtype == "bf16" else 5e-3


def _launch(artifact, signature, values, grid, block):
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )(values, config=LaunchConfig(grid=grid, block=block, fence=True))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _FWD_ROWS, ids=lambda r: r[0])
def test_fwd_matches_the_reference(row, dtype):
    current_device_arch_in((_ARCH,), subject="the gfx950 implicit-GEMM forward")
    from rocke import compile_kernel

    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs

    spec = _fwd_spec(row, dtype)
    p = spec.problem
    artifact = compile_kernel(build_implicit_gemm_conv(spec, arch=_ARCH), arch=_ARCH)
    rng = np.random.default_rng(0)
    x = stored_uniform(rng, (p.N, p.Hi, p.Wi, p.C), dtype)
    w = stored_uniform(rng, (p.K, p.Y, p.X, p.C // p.groups), dtype)
    x_dev, w_dev = DeviceArray(x, dtype), DeviceArray(w, dtype)
    y_dev = DeviceArray(poisoned((p.N, p.Ho, p.Wo, p.K), dtype, 0xFF), dtype)
    values = ConvArgs.from_problem(
        p, tile_m=spec.tile_m, tile_n=spec.tile_n
    ).to_launch_values(
        x_dev.data_ptr(),
        w_dev.data_ptr(),
        y_dev.data_ptr(),
        x_dev.nbytes,
        w_dev.nbytes,
        y_dev.nbytes,
    )
    _launch(
        artifact,
        conv_args_signature(dtype),
        values,
        implicit_gemm_conv_grid(spec),
        (spec.launch_block_size, 1, 1),
    )
    y = storage.decode(y_dev.read(), dtype)
    assert not np.isnan(y).any(), "the kernel left an output element unwritten"
    ref = conv_fwd_reference(storage.decode(x, dtype), storage.decode(w, dtype), p)
    assert peak_relative_error(y, ref, floor=1.0) < _tol(dtype)


def _dgrad_run(spec, dtype, seed=0):
    """dX of one dgrad kernel on U(-1, 1) ``dY`` and ``W`` of ``seed``, with dX
    poisoned when the kernel stores it (an atomic split-K starts from zeros)."""
    from rocke import compile_kernel

    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs

    p = spec.problem
    artifact = compile_kernel(
        build_implicit_gemm_conv_dgrad(spec, arch=_ARCH), arch=_ARCH
    )
    rng = np.random.default_rng(seed)
    dy = stored_uniform(rng, (p.N, p.Ho, p.Wo, p.K), dtype)
    w = stored_uniform(rng, (p.K, p.Y, p.X, p.cpg), dtype)
    shape = (p.N, p.Hi, p.Wi, p.C)
    dx_init = (
        np.zeros(shape, dtype=dy.dtype)
        if spec.needs_atomic
        else poisoned(shape, dtype, 0xFF)
    )
    dy_dev, w_dev, dx_dev = (DeviceArray(a, dtype) for a in (dy, w, dx_init))
    sub_gemms = spec.compute_sub_gemms()
    records = np.array(pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n))
    records_dev = DeviceArray(records.astype(np.int32), "i32")
    values = ConvArgs.from_problem(
        p, direction="dgrad", tile_m=spec.tile_m, tile_n=spec.tile_n
    ).to_launch_values(
        dy_dev.data_ptr(),
        w_dev.data_ptr(),
        dx_dev.data_ptr(),
        dy_dev.nbytes,
        w_dev.nbytes,
        dx_dev.nbytes,
        sub_gemm_buf=records_dev.data_ptr(),
        num_sub_gemms=len(sub_gemms),
    )
    _launch(
        artifact,
        conv_args_signature(dtype, direction="dgrad"),
        values,
        (sub_gemms[-1].block_end, p.groups, spec.split_k),
        (spec.launch_block_size, 1, 1),
    )
    dx = storage.decode(dx_dev.read(), dtype)
    ref = conv_dgrad_reference(storage.decode(dy, dtype), storage.decode(w, dtype), p)
    return dx, ref


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("row", _DGRAD_ROWS, ids=lambda r: r[0])
def test_dgrad_matches_the_reference(row, dtype):
    current_device_arch_in((_ARCH,), subject="the gfx950 implicit-GEMM dgrad")
    spec = _dgrad_spec(row, dtype)
    dx, ref = _dgrad_run(spec, dtype)
    assert not np.isnan(dx).any(), "the kernel left a dX element unwritten"
    assert peak_relative_error(dx, ref, floor=1.0) < _tol(dtype)
    if row[0] == "k96_peel_zero_fill":
        # A 1x1 stride-2 filter writes the even pixels only; the kernel itself
        # zeroes the other three residues of each 2 x 2 (the poisoned dX has no
        # separate memset), and K = 96 is a peeled tail of the 64-wide K loop.
        assert dgrad_fuses_dx_zero_fill(spec.problem, dtype_d=dtype)
        odd = np.ones(dx.shape[:3], dtype=bool)
        odd[:, ::2, ::2] = False
        assert odd.any() and not dx[odd].any()
