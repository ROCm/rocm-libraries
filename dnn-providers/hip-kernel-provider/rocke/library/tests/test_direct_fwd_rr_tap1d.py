# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx950 rows-in-registers direct forward
(``conv_direct_fwd_rr_mfma``, variants ``direct_fwd_rr_mfma`` and its
weight-format helper ``direct_fwd_rr_wfmt``) and the 1-D tap-reuse direct
forward (``conv_direct_fwd_tap1d_mfma``, variant ``direct_fwd_tap1d_mfma``).

CPU: the validators, the kernel builds, the weight-format host reference and
the AOT grid counts. GPU (gfx950): the output against a float64 numpy
reference of the stored operands, both dtypes, with image edges (extents that
are not a multiple of the tile), partial channel chunks, odd chunk counts,
grouped and width-1 (unfolded) layers for the first family, and padded,
dilated and single-chunk layers for the second.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_fwd_rr_tap1d.py -v
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.device_array import DeviceArray
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.lanes import BUILD_VARIANT, import_or_skip_until

from kernels.common.conv_direct_fwd_rr_mfma import (
    FwdRrMfmaSpec,
    FwdRrWfmtSpec,
    build_fwd_rr_mfma,
    build_fwd_rr_wfmt,
    fwd_rr_format_weights_reference,
    fwd_rr_mfma_reason,
    fwd_rr_weight_bytes,
)
from kernels.common.conv_direct_fwd_tap1d_mfma import (
    FwdTap1dMfmaSpec,
    build_fwd_tap1d_mfma,
    fwd_tap1d_mfma_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

from ._direct_conv_numeric import (
    direct_fwd_reference,
    launch_direct_conv,
    launch_direct_kernel,
    poisoned,
    stored_uniform,
)

_RR = "direct_fwd_rr_mfma"
_RR_WFMT = "direct_fwd_rr_wfmt"
_TAP1D = "direct_fwd_tap1d_mfma"


def _rr_problem(
    N=1, H=34, W=34, C=128, K=128, KW=3, pad=1, pad_w=-1, groups=1, dtype="bf16"
):
    """A stride-1 3x3 (or 3x1) forward; ``C`` / ``K`` are per group."""
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C,
        kpg=K,
        KH=3,
        KW=KW,
        PAD=pad,
        pad_w=pad_w,
        stride=1,
        dtype=dtype,
    )


def _tap1d_problem(N=1, H=3, W=37, C=128, K=128, S=3, pad_w=1, dil_w=1, dtype="bf16"):
    """A 1 x S forward (no height padding), width padding ``pad_w``."""
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=1,
        KW=S,
        PAD=0,
        pad_w=pad_w,
        stride=1,
        dil_w=dil_w,
        dtype=dtype,
    )


# Rows-in-registers rows: (id, problem kwargs, spec knobs). Each is valid as
# listed; they cover every knob class of the AOT grid.
_RR_ROWS = [
    # plain K128, edges on the bottom and right
    ("plain", dict(H=34, W=45, C=128, K=128), {}),
    ("k96", dict(H=20, W=36, C=128, K=192), dict(kw16=3)),
    ("c64_k64", dict(H=18, W=40, C=128, K=64), dict(c64=True, kw16=2)),
    # 4 waves along K (K256), 2 and 3 chunks, C not a multiple of 64
    ("k4w", dict(H=17, W=33, C=128, K=256), dict(k4w=True)),
    ("c_tail_k4w", dict(H=18, W=34, C=200, K=256), dict(c_tail=True, k4w=True)),
    # odd chunk counts alternate the LDS buffer parity
    ("odd_c", dict(H=18, W=40, C=192, K=64), dict(kw16=2, odd_c=True)),
    ("odd_c_c64", dict(H=18, W=40, C=192, K=64), dict(c64=True, kw16=2, odd_c=True)),
    # a single chunk per tile
    ("one_c", dict(H=18, W=36, C=64, K=128), dict(one_c=True)),
    (
        "one_c_tail",
        dict(H=18, W=36, C=40, K=256),
        dict(one_c=True, c_tail=True, k4w=True),
    ),
    # grouped: a K block lies in one group
    ("grp", dict(H=18, W=34, C=128, K=128, groups=2), dict(grp=True)),
    (
        "grp_k64",
        dict(H=18, W=34, C=64, K=64, groups=3),
        dict(grp=True, kw16=2, one_c=True),
    ),
    # no and wide padding
    ("pad0", dict(H=36, W=40, C=128, K=128, pad=0), {}),
    ("pad2", dict(H=20, W=38, C=128, K=128, pad=2), {}),
    # 3x1 filters: plain, and 32 images of a width-1 layer per tile
    ("f3x1", dict(H=18, W=40, C=128, K=128, KW=1, pad=1, pad_w=0), {}),
    (
        "f3x1_unfold",
        dict(N=40, H=20, W=1, C=128, K=128, KW=1, pad=1, pad_w=0),
        dict(unfold=True),
    ),
    # chunked columns: a width that is not a multiple of 8
    ("chunk", dict(N=2, H=18, W=29, C=128, K=256), dict(k4w=True, chunk=True)),
    # XCD-local tile walk over several rounds of the persistent grid
    ("xcd", dict(N=2, H=72, W=72, C=128, K=256), dict(k4w=True, xcd=True)),
]


def _rr_spec(row, dtype):
    _, pkw, knobs = row
    return FwdRrMfmaSpec(problem=_rr_problem(dtype=dtype, **pkw), **knobs)


# Tap-reuse rows: (id, problem kwargs, spec knobs).
_TAP1D_ROWS = [
    ("s3_same", dict(S=3, pad_w=1, W=37, C=128, K=128), dict(bk=128, nwv=8, ck=64)),
    (
        "s3_two_blocks",
        dict(S=3, pad_w=1, W=300, C=128, K=128),
        dict(bk=128, nwv=8, ck=64),
    ),
    (
        "s3_d3",
        dict(S=3, pad_w=3, dil_w=3, W=70, C=192, K=128),
        dict(bk=128, nwv=4, ck=64),
    ),
    ("s3_valid", dict(S=3, pad_w=0, W=41, C=128, K=64), dict(bk=64, nwv=4, ck=64)),
    ("s7_same_ck32", dict(S=7, pad_w=3, W=53, C=96, K=64), dict(bk=64, nwv=4, ck=32)),
    (
        "s7_d5",
        dict(S=7, pad_w=15, dil_w=5, W=61, C=128, K=128),
        dict(bk=128, nwv=4, ck=64),
    ),
    (
        "s11_d3",
        dict(S=11, pad_w=15, dil_w=3, W=61, C=128, K=64),
        dict(bk=64, nwv=4, ck=64),
    ),
    (
        "s11_one",
        dict(S=11, pad_w=5, W=45, C=64, K=64),
        dict(bk=64, nwv=4, ck=64, one=True),
    ),
    (
        "s3_k32_one",
        dict(S=3, pad_w=1, W=40, C=32, K=32),
        dict(bk=32, nwv=4, ck=32, one=True),
    ),
]


def _tap1d_spec(row, dtype):
    _, pkw, knobs = row
    return FwdTap1dMfmaSpec(problem=_tap1d_problem(dtype=dtype, **pkw), **knobs)


# --------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("row", _RR_ROWS, ids=lambda r: r[0])
def test_rr_rows_are_valid_and_build(row, dtype):
    spec = _rr_spec(row, dtype)
    assert fwd_rr_mfma_reason(spec) is None
    kernel = build_fwd_rr_mfma(spec)
    assert kernel.name == spec.kernel_name()
    assert kernel.attrs["max_workgroup_size"] == 512
    assert spec.grid()[0] <= 256


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("row", _TAP1D_ROWS, ids=lambda r: r[0])
def test_tap1d_rows_are_valid_and_build(row, dtype):
    spec = _tap1d_spec(row, dtype)
    assert fwd_tap1d_mfma_reason(spec) is None
    kernel = build_fwd_tap1d_mfma(spec)
    assert kernel.name == spec.kernel_name()
    assert kernel.attrs["max_workgroup_size"] == spec.threads_per_block


def test_rr_reasons():
    ok = FwdRrMfmaSpec(problem=_rr_problem())
    assert fwd_rr_mfma_reason(ok) is None

    def why(spec=ok, **kw):
        return fwd_rr_mfma_reason(replace(spec, problem=replace(spec.problem, **kw)))

    assert "groups must be 1" in why(groups=2)
    assert "3x3 or 3x1" in why(stride=2)
    assert "3x3 or 3x1" in why(dil_h=2)
    assert "3x3 or 3x1" in why(KW=5)
    assert "padding must be" in why(PAD=3)
    assert "fp16/bf16" in why(dtype="fp32")
    assert "must be a multiple of 64" in why(cpg=100)
    assert "multiple of 128" in why(kpg=96)
    assert "odd_c" in why(cpg=192)
    limit = "exceeds 2147483647 B, the i32 buffer range"
    assert why(H=1024, W=2048, cpg=1024, kpg=128).endswith(limit)
    assert "input image" in why(H=1024, W=2048, cpg=1024, kpg=128)
    for arch in ("gfx942", "gfx90a"):
        assert fwd_rr_mfma_reason(ok, arch=arch) == (
            f"fwd_rr_mfma needs the 16-byte LDS DMA (absent on {arch})"
        )
    wave64 = "fwd_rr_mfma is a wave64 MFMA kernel"
    assert fwd_rr_mfma_reason(ok, arch="gfx1250") == wave64
    assert fwd_rr_mfma_reason(replace(ok, wave_size=32)) == wave64
    # Flag consistency.
    assert "kw16 must be" in fwd_rr_mfma_reason(replace(ok, kw16=1))
    assert "odd_c needs" in fwd_rr_mfma_reason(replace(ok, odd_c=True))
    assert "c_tail needs" in fwd_rr_mfma_reason(replace(ok, c_tail=True))
    assert "one_c needs" in fwd_rr_mfma_reason(replace(ok, one_c=True))
    assert "grp:" in fwd_rr_mfma_reason(replace(ok, grp=True))
    assert "unfold needs" in fwd_rr_mfma_reason(replace(ok, unfold=True))
    assert "chunk:" in fwd_rr_mfma_reason(replace(ok, chunk=True))
    three_by_one = _rr_problem(KW=1, pad=1, pad_w=0)
    assert "c64 needs" in fwd_rr_mfma_reason(
        FwdRrMfmaSpec(problem=three_by_one, c64=True, kw16=2)
    )
    assert "wastes most" in fwd_rr_mfma_reason(
        FwdRrMfmaSpec(problem=replace(three_by_one, W=8))
    )
    with pytest.raises(ValueError, match="invalid FwdRrMfmaSpec"):
        build_fwd_rr_mfma(replace(ok, kw16=1))


def test_tap1d_reasons():
    ok = FwdTap1dMfmaSpec(problem=_tap1d_problem())
    assert fwd_tap1d_mfma_reason(ok) is None

    def why(spec=ok, **kw):
        return fwd_tap1d_mfma_reason(replace(spec, problem=replace(spec.problem, **kw)))

    assert why(groups=2) == "groups must be 1"
    assert "1 x S filters" in why(KH=3)
    assert "1 x S filters" in why(PAD=1)
    assert "1 x S filters" in why(stride=2)
    assert "width padding" in why(pad_w=3)
    assert "pad_w must be -1" in why(pad_w=-2)
    assert "pad_w must be -1" in why(pad_w=-5)
    assert "fp16/bf16" in why(dtype="fp32")
    assert "must be a multiple of bk" in why(kpg=100)
    assert "must be 32 or 64" in why(cpg=100)
    assert "must be set exactly when C == ck" in why(cpg=64)
    assert "empty output" in why(W=1, pad_w=0, KW=3)
    assert "bk must be" in fwd_tap1d_mfma_reason(replace(ok, bk=96))
    assert "bk must be" in fwd_tap1d_mfma_reason(replace(ok, nwv=6))
    assert "must be set exactly" in fwd_tap1d_mfma_reason(replace(ok, one=True))
    assert fwd_tap1d_mfma_reason(ok, arch="gfx942") == (
        "fwd_tap1d_mfma needs the 16-byte LDS DMA (absent on gfx942)"
    )
    assert fwd_tap1d_mfma_reason(ok, arch="gfx1250") == (
        "fwd_tap1d_mfma is a wave64 MFMA kernel"
    )
    with pytest.raises(ValueError, match="invalid FwdTap1dMfmaSpec"):
        build_fwd_tap1d_mfma(replace(ok, bk=96))


def test_tap1d_geometry():
    spec = FwdTap1dMfmaSpec(problem=_tap1d_problem(N=2, H=3, W=300, K=256))
    # 8 waves, 4 along K: 2 along X -> 32 columns x 9 pixels = 288 per block.
    assert (spec.nwk, spec.nwx, spec.ncol, spec.bx) == (4, 2, 32, 288)
    assert spec.win == 288 + 2
    # 2 images x 3 rows x 2 pixel blocks x 2 channel blocks.
    assert spec.launch_geometry() == ((2 * 3 * 2 * 2, 1, 1), (512, 1, 1))
    dil = FwdTap1dMfmaSpec(problem=_tap1d_problem(S=7, pad_w=9, dil_w=3), nwv=4, bk=64)
    assert dil.ncol % 3 == 0 and dil.win == dil.bx + 3 * 6


def test_rr_grid_and_workspace():
    spec = FwdRrMfmaSpec(problem=_rr_problem(N=2, H=34, W=45, K=256))
    # 2 images x 2 pixel tiles of 32 columns x 3 row tiles of 16 x 2 K blocks.
    assert spec.tiles() == 2 * 2 * 3 * 2
    assert spec.launch_geometry() == ((24, 1, 1), (512, 1, 1))
    big = FwdRrMfmaSpec(problem=_rr_problem(N=8, H=128, W=128, K=256))
    assert big.grid() == (256, 1, 1)
    assert spec.workspace_bytes() == fwd_rr_weight_bytes(spec.problem)
    assert spec.workspace_bytes() == 2 * 256 * 3 * 3 * 128
    tail = FwdRrMfmaSpec(problem=_rr_problem(C=200, K=256), c_tail=True, k4w=True)
    assert tail.workspace_bytes() == 2 * 256 * 3 * 3 * 256  # C padded to 64


def test_rr_kernel_name_flags():
    spec = FwdRrMfmaSpec(
        problem=_rr_problem(C=200, K=256),
        name="rocke_fwd_rr_mfma",
        c_tail=True,
        k4w=True,
    )
    name = spec.kernel_name()
    assert name.startswith("rocke_fwd_rr_mfma_f3x3_p1") and name.endswith("bf16")
    assert "ctail" in name and "k4w" in name
    helper = FwdRrWfmtSpec(problem=_rr_problem(), kw16=2, fw=1, nwk=4)
    assert helper.kernel_name() == "fwd_rr_wfmt_k64_w1_wk4_bf16"
    assert build_fwd_rr_wfmt(helper).name == helper.kernel_name()


def test_rr_ir_is_shape_independent():
    """One binary serves every batch, extent and group count."""
    from rocke.core.ir_serialize import serialize

    def ir(**kw):
        spec = FwdRrMfmaSpec(problem=_rr_problem(**kw))
        return serialize(build_fwd_rr_mfma(spec)).replace(spec.kernel_name(), "k")

    assert ir() == ir(N=3, H=50, W=17, K=384)


@pytest.mark.parametrize("kw16,fw,nwk", [(4, 3, 2), (2, 3, 4), (3, 1, 2)])
def test_weight_format_reference_is_a_channel_permutation(kw16, fw, nwk):
    """With C a multiple of 64 the format only moves 8-element channel
    granules: every filter element appears exactly once."""
    C, K = 64, 16 * kw16 * nwk
    rng = np.random.default_rng(3)
    w = rng.permutation(K * 3 * fw * C).reshape(K, 3, fw, C)
    out = fwd_rr_format_weights_reference(w, C, kw16, fw, nwk)
    assert out.shape == (w.size,)
    assert np.array_equal(np.sort(out), np.sort(w.reshape(-1)))
    # The first granule is lane 0: output channel 0, tap (0, 0), channels 0..7.
    assert np.array_equal(out[:8], w[0, 0, 0, :8])


def test_weight_format_reference_zero_pads_c():
    """C = 40 formats as one 64-channel chunk; the 24 channels past C are 0."""
    C, K = 40, 128
    w = np.arange(1, K * 3 * 3 * C + 1, dtype=np.int64).reshape(K, 3, 3, C)
    out = fwd_rr_format_weights_reference(
        np.pad(w, ((0, 0), (0, 0), (0, 0), (0, 24))), 64
    )
    assert out.size == K * 3 * 3 * 64
    assert np.count_nonzero(out) == w.size


def test_aot_grid_counts():
    """The direct AOT grid offers 117 rows-in-registers and 176 tap-reuse
    identities over forward and dgrad (both dtypes), and one weight-format
    helper per distinct (kw16, filter width, waves along K) and dtype."""
    from collections import Counter

    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    seen: dict[tuple[str, str], set[str]] = {}
    for dtype in ("fp16", "bf16"):
        for job in dks.enumerate_direct_jobs(
            arch="gfx950", target=target, directions=("fwd", "dgrad"), dtype=dtype
        ):
            ident = job.identity
            if ident.algorithm in (_RR, _RR_WFMT, _TAP1D):
                seen.setdefault((dtype, ident.algorithm), set()).add(ident.label())
    count = Counter()
    for (_, algorithm), labels in seen.items():
        count[algorithm] += len(labels)
    assert count[_RR] == 117
    assert count[_TAP1D] == 176
    assert count[_RR_WFMT] > 0
    # No other architecture gets either family.
    from kernels.conv.direct_aot_jobs import grid_caches_variant

    for arch in ("gfx942", "gfx1250"):
        assert not grid_caches_variant(_RR, arch)
        assert not grid_caches_variant(_TAP1D, arch)


def test_aot_grid_jobs_round_trip():
    """Every tap-reuse job builds and maps back from its identity."""
    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.conv.direct_aot_jobs import direct_job_for_identity
    from kernels.conv.direct_variants import validate_direct_spec
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    jobs = [
        j
        for j in dks.enumerate_direct_jobs(
            arch="gfx950", target=target, directions=("fwd",), dtype="fp16"
        )
        if j.identity.algorithm in (_TAP1D, _RR)
    ]
    assert jobs
    for job in jobs[:: max(1, len(jobs) // 12)]:
        assert direct_job_for_identity(job.identity).identity == job.identity


# ------------------------------------------------------------------ C++ engine


def _rr_describe_specs():
    """The rows in both dtypes, plus one refusal per validator branch."""
    specs = [_rr_spec(row, dtype) for row in _RR_ROWS for dtype in ("bf16", "fp16")]
    ok = FwdRrMfmaSpec(problem=_rr_problem())
    for pkw in (
        dict(groups=2),
        dict(stride=2),
        dict(KW=5),
        dict(PAD=3),
        dict(pad_w=-2),
        dict(dtype="fp32"),
        dict(cpg=100),
        dict(kpg=96),
        dict(cpg=192),
        dict(N=0),
        dict(H=1024, W=2048, cpg=1024, kpg=128),
    ):
        specs.append(replace(ok, problem=replace(ok.problem, **pkw)))
    three_by_one = _rr_problem(KW=1, pad=1, pad_w=0)
    for spec in (
        replace(ok, kw16=1),
        replace(ok, odd_c=True),
        replace(ok, c_tail=True),
        replace(ok, one_c=True),
        replace(ok, grp=True),
        replace(ok, unfold=True),
        replace(ok, chunk=True),
        replace(ok, wave_size=32),
        FwdRrMfmaSpec(problem=three_by_one, c64=True, kw16=2),
        FwdRrMfmaSpec(problem=replace(three_by_one, W=8)),
        FwdRrMfmaSpec(problem=replace(ok.problem, cpg=72), one_c=True),
        FwdRrMfmaSpec(problem=replace(ok.problem, cpg=64), one_c=True, odd_c=True),
        FwdRrMfmaSpec(problem=replace(ok.problem, cpg=136), c_tail=True),
        FwdRrMfmaSpec(problem=replace(ok.problem, cpg=128), c_tail=True),
    ):
        specs.append(spec)
    return specs


def test_rr_describe_matches_python():
    """The C++ validator, kernel name, grid and workspace of the rr forward
    agree with Python's, accepted and refused specs, on three architectures."""
    eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    from kernels.conv.engine_bridge import conv_direct_fwd_rr_mfma_spec_to_dict

    for arch in ("gfx950", "gfx942", "gfx1250"):
        for spec in _rr_describe_specs():
            ok, why_c, name, grid, ws = eng.conv_direct_fwd_rr_mfma_describe(
                conv_direct_fwd_rr_mfma_spec_to_dict(spec), arch
            )
            why = fwd_rr_mfma_reason(spec, arch)
            assert ok == (why is None), (arch, spec, why, why_c)
            if why is not None:
                assert why_c == why, (arch, spec, why, why_c)
            else:
                assert tuple(grid) == spec.grid(), (arch, spec)
                assert ws == spec.workspace_bytes(), (arch, spec)
            assert name == spec.kernel_name()


def test_rr_wfmt_describe_matches_python():
    """The C++ weight-format helper names and sizes its launch like Python."""
    eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    from kernels.conv.engine_bridge import conv_direct_fwd_rr_wfmt_spec_to_dict

    for dtype in ("bf16", "fp16"):
        for kw16, fw, nwk, dg in ((4, 3, 2, False), (3, 1, 2, False), (2, 3, 4, True)):
            spec = FwdRrWfmtSpec(
                problem=_rr_problem(C=72, K=192, KW=fw, dtype=dtype),
                kw16=kw16,
                fw=fw,
                nwk=nwk,
                dg=dg,
            )
            name, grid = eng.conv_direct_fwd_rr_wfmt_describe(
                conv_direct_fwd_rr_wfmt_spec_to_dict(spec), "gfx950"
            )
            assert name == spec.kernel_name()
            assert tuple(grid) == spec.launch_geometry()[0]


def test_tap1d_describe_matches_python():
    """The C++ validator, kernel name, grid and LDS size of the tap-reuse
    forward agree with Python's, accepted and refused specs."""
    eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    from kernels.conv.engine_bridge import conv_direct_fwd_tap1d_mfma_spec_to_dict

    specs = [
        _tap1d_spec(row, dtype) for row in _TAP1D_ROWS for dtype in ("bf16", "fp16")
    ]
    ok = FwdTap1dMfmaSpec(problem=_tap1d_problem())
    for pkw in (
        dict(groups=2),
        dict(KH=3),
        dict(PAD=1),
        dict(stride=2),
        dict(pad_w=3),
        dict(pad_w=-2),
        dict(dtype="fp32"),
        dict(kpg=100),
        dict(cpg=100),
        dict(cpg=64),
        dict(W=1, pad_w=0, KW=3),
        dict(N=0),
        dict(KW=11, dil_w=20, pad_w=0, W=400),
    ):
        specs.append(replace(ok, problem=replace(ok.problem, **pkw)))
    for skw in (
        dict(bk=96),
        dict(nwv=6),
        dict(one=True),
        dict(seg=0),
        dict(ck=16),
        dict(wave_size=32),
    ):
        specs.append(replace(ok, **skw))
    for arch in ("gfx950", "gfx942", "gfx1250"):
        for spec in specs:
            got, why_c, name, grid, lds = eng.conv_direct_fwd_tap1d_mfma_describe(
                conv_direct_fwd_tap1d_mfma_spec_to_dict(spec), arch
            )
            why = fwd_tap1d_mfma_reason(spec, arch)
            assert got == (why is None), (arch, spec, why, why_c)
            if why is not None:
                assert why_c == why, (arch, spec, why, why_c)
            else:
                assert tuple(grid) == spec.grid(), (arch, spec)
                assert lds == spec.lds_bytes, (arch, spec)
            assert name == spec.kernel_name()


def test_backends_lower_identically():
    """The engine bridge gives the same .ll on the Python and C++ backends
    for one spec of each kernel (the parity emitters cover every knob)."""
    import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    from kernels.conv.engine_bridge import (
        lower_conv_direct_fwd_rr_mfma,
        lower_conv_direct_fwd_rr_wfmt,
        lower_conv_direct_fwd_tap1d_mfma,
    )

    cases = (
        (lower_conv_direct_fwd_rr_mfma, _rr_spec(_RR_ROWS[0], "bf16")),
        (
            lower_conv_direct_fwd_rr_wfmt,
            FwdRrWfmtSpec(problem=_rr_problem(C=72), kw16=3),
        ),
        (lower_conv_direct_fwd_tap1d_mfma, _tap1d_spec(_TAP1D_ROWS[2], "fp16")),
    )
    for lower, spec in cases:
        py = lower(spec, arch="gfx950", backend="python")
        cpp = lower(spec, arch="gfx950", backend="cpp")
        assert py.backend == "python" and cpp.backend == "cpp"
        assert py.llvm_text == cpp.llvm_text, spec


def test_binding_rejects_bad_input():
    """A missing, mistyped or partial problem raises in the binding."""
    eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
    from kernels.conv.engine_bridge import (
        conv_direct_fwd_rr_mfma_spec_to_dict,
        conv_direct_fwd_tap1d_mfma_spec_to_dict,
    )

    for fam, good in (
        (
            "conv_direct_fwd_rr_mfma",
            conv_direct_fwd_rr_mfma_spec_to_dict(FwdRrMfmaSpec(problem=_rr_problem())),
        ),
        (
            "conv_direct_fwd_tap1d_mfma",
            conv_direct_fwd_tap1d_mfma_spec_to_dict(
                FwdTap1dMfmaSpec(problem=_tap1d_problem())
            ),
        ),
    ):
        partial = dict(good["problem"])
        del partial["kpg"]
        for problem in (None, "N2H13", partial, dict(good["problem"], dtype=None)):
            sd = dict(good)
            if problem is None:
                del sd["problem"]
            else:
                sd["problem"] = problem
            with pytest.raises(ValueError):
                getattr(eng, f"{fam}_describe")(sd, "gfx950")
            with pytest.raises(ValueError):
                getattr(eng, f"{fam}_lower_llvm")(sd, arch="gfx950")


# --------------------------------------------------------------------------- GPU


def _operands(p, dtype):
    rng = np.random.default_rng(0)
    x = stored_uniform(rng, (p.N, p.H, p.W, p.total_c), dtype)
    w = stored_uniform(rng, (p.total_k, p.KH, p.KW, p.cpg), dtype)
    return x, w


def _check(y, p, x, w, dtype):
    ref = direct_fwd_reference(p, storage.decode(x, dtype), storage.decode(w, dtype))
    y = storage.decode(y, dtype)
    assert not np.isnan(y).any()
    tol = 2e-2 if dtype == "bf16" else 5e-3
    rel = peak_relative_error(y, ref, floor=1.0)
    assert rel < tol, rel


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("row", _RR_ROWS, ids=lambda r: r[0])
def test_rr_forward_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the rows-in-registers forward")
    from rocke import compile_kernel

    spec = _rr_spec(row, dtype)
    p = spec.problem
    assert fwd_rr_mfma_reason(spec) is None
    helper = FwdRrWfmtSpec(
        problem=p, kw16=spec.kw16, fw=p.KW, nwk=spec.nwk, name="fwd_rr_wfmt"
    )
    conv_art = compile_kernel(build_fwd_rr_mfma(spec, arch="gfx950"), arch="gfx950")
    fmt_art = compile_kernel(build_fwd_rr_wfmt(helper, arch="gfx950"), arch="gfx950")
    x, w = _operands(p, dtype)
    x_dev, w_dev = DeviceArray(x, dtype), DeviceArray(w, dtype)
    # Garbage in the formatted weights: the helper must overwrite all it reads.
    wfmt = DeviceArray(poisoned((spec.workspace_bytes(),), "u8"))
    y = DeviceArray(poisoned((p.N, p.Ho, p.Wo, p.total_k), dtype, 0xFF), dtype)

    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    from kernels.common.conv_abi import conv_direct_transform_signature

    grid, block = helper.launch_geometry()
    KernelLauncher(
        hsaco=fmt_art.hsaco,
        kernel_name=fmt_art.kernel_name,
        signature=conv_direct_transform_signature(dtype, True),
    )(
        {
            "A": w_dev.data_ptr(),
            "D": wfmt.data_ptr(),
            "A_bytes": w_dev.nbytes,
            "D_bytes": spec.workspace_bytes(),
            "p_cpg": p.cpg,
            "p_kpg": p.kpg,
        },
        config=LaunchConfig(grid=grid, block=block, fence=True),
    )
    grid, block = spec.launch_geometry()
    launch_direct_kernel(
        conv_art, p, (x_dev, wfmt, y), grid=grid, block=block, dtype=dtype
    )
    _check(y.read(), p, x, w, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("row", _TAP1D_ROWS, ids=lambda r: r[0])
def test_tap1d_forward_matches_the_reference(row, dtype):
    current_device_arch_in(("gfx950",), subject="the tap-reuse forward")
    from rocke import compile_kernel

    spec = _tap1d_spec(row, dtype)
    p = spec.problem
    assert fwd_tap1d_mfma_reason(spec) is None
    art = compile_kernel(build_fwd_tap1d_mfma(spec, arch="gfx950"), arch="gfx950")
    x, w = _operands(p, dtype)
    grid, block = spec.launch_geometry()
    y = launch_direct_conv(
        art,
        p,
        (x, w),
        poisoned((p.N, p.Ho, p.Wo, p.total_k), dtype, 0xFF),
        grid=grid,
        block=block,
    )
    _check(y, p, x, w, dtype)
