# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the gfx1250 wave32 WMMA grouped direct wgrad
(``conv_direct_grouped_wgrad_wmma``) and the block shape of the fold it runs
with (``DwWgradFoldSpec`` cols / slices / unroll).

The C++ engine's validator, kernel name, grid, LDS size and fold contract
match Python's over the whole gfx1250 AOT grid and a set of rejected specs,
and both engines serialize the same IR for a few specs (the byte-identity gate
covers the lowered IR of the parity configs). No GPU.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_grouped_wgrad_wmma.py -v
"""

from __future__ import annotations

import pytest

from kernels.common.conv_direct_dw_wgrad import (
    DwWgradFoldSpec,
    build_dw_wgrad_fold,
)
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_grouped_wgrad_wmma import (
    GroupedWgradWmmaSpec,
    build_grouped_wgrad_wmma,
    is_valid_grouped_wgrad_wmma_spec,
)

_VARIANT = "direct_wgrad_grouped_wmma"


def _problem(
    N=2,
    H=19,
    W=45,
    cpg=4,
    C=64,
    KH=3,
    KW=3,
    PAD=1,
    dtype="bf16",
    stride=1,
    dil=1,
    pad_w=-1,
):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=C // cpg,
        cpg=cpg,
        kpg=cpg,
        KH=KH,
        KW=KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=stride,
        dtype=dtype,
        dil_h=dil,
        dil_w=dil,
    )


def _grid_specs():
    """Every spec of the gfx1250 AOT grid (both dtypes), on a real-sized
    problem rather than the probe, so the grid and fold contract vary."""
    from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _knob_grid

    specs = []
    _, caps_list = DIRECT_CAPABILITIES[_VARIANT]
    for dtype in ("bf16", "fp16"):
        for caps in caps_list:
            for knobs in _knob_grid(_VARIANT, "gfx1250", dtype, caps):
                p = _problem(
                    N=3,
                    H=29,
                    W=57,
                    cpg=caps.cpg,
                    C=32 * caps.cpg,
                    KH=caps.KH,
                    KW=caps.KH,
                    PAD=caps.PAD,
                    dtype=dtype,
                )
                specs.append(
                    GroupedWgradWmmaSpec(
                        problem=p, name="rocke_direct_wgrad_grouped_wmma", **knobs
                    )
                )
    return specs


# (problem overrides, knob overrides) the validator refuses on gfx1250.
_REJECTED = (
    (dict(cpg=16), {}),
    (dict(cpg=2, C=32), {}),
    (dict(dtype="fp32"), {}),
    (dict(stride=2), {}),
    (dict(PAD=3), {}),
    (dict(KW=1, PAD=1), {}),
    (dict(C=24), {}),
    (dict(dil=2), {}),
    (dict(C=8 * 2**28), {}),  # groups * cpg past i32
    ({}, dict(waves_c=3)),
    ({}, dict(waves_c=0)),
    ({}, dict(q_steps=0)),
    ({}, dict(q_steps=5)),
    ({}, dict(rows_per_chunk=0)),
    ({}, dict(target_blocks=0)),
    ({}, dict(stages=1)),
    ({}, dict(stages=5)),
    ({}, dict(waves_c=8, q_steps=4, stages=4)),  # LDS
)


def _rejected_specs():
    return [
        GroupedWgradWmmaSpec(problem=_problem(**pkw), **knobs)
        for pkw, knobs in _REJECTED
    ]


def _describe(eng, spec, arch):
    from rocke.core.backend import conv_direct_grouped_wgrad_wmma_spec_to_dict

    return eng.conv_direct_grouped_wgrad_wmma_describe(
        conv_direct_grouped_wgrad_wmma_spec_to_dict(spec), arch
    )


def test_engines_agree_on_specs():
    """C++ describe == Python validity, reason, name, LDS bytes and, for a
    valid spec, grid and fold contract (partitions, workspace, taps,
    channels), over the AOT grid and every rejected spec, on gfx1250 and on
    gfx950 (which refuses all of them)."""
    eng = pytest.importorskip("rocke_engine")
    grid_specs = _grid_specs()
    assert len(grid_specs) == 2 * 4 * 72
    rejected = _rejected_specs()
    for arch in ("gfx1250", "gfx950"):
        for i, spec in enumerate(grid_specs + rejected):
            ok, why_c, name, grid, lds, parts, ws, taps, chans = _describe(
                eng, spec, arch
            )
            ok_py, why = is_valid_grouped_wgrad_wmma_spec(spec, arch)
            assert ok == ok_py, (arch, spec, why)
            assert why_c == why, (arch, spec)
            assert name == spec.kernel_name()
            assert lds == spec.lds_bytes()
            if i < len(grid_specs):
                # The AOT grid builds on gfx1250 and nowhere else.
                assert ok == (arch == "gfx1250"), (arch, spec, why)
            else:
                assert not ok, (arch, spec)
            if ok:
                g, block = spec.launch_geometry()
                assert tuple(grid) == g
                assert block == (spec.threads, 1, 1)
                assert parts == spec.n_partitions()
                assert ws == spec.workspace_bytes()
                assert taps == spec.fold_taps()
                assert chans == spec.fold_channels()
            else:
                assert tuple(grid) == (0, 0, 0)


def test_binding_rejects_bad_input():
    """A missing, mistyped or partial problem raises instead of reading as
    the zero default; a rejected spec does not lower."""
    eng = pytest.importorskip("rocke_engine")
    from rocke.core.backend import conv_direct_grouped_wgrad_wmma_spec_to_dict

    good = conv_direct_grouped_wgrad_wmma_spec_to_dict(
        GroupedWgradWmmaSpec(problem=_problem())
    )
    partial = dict(good["problem"])
    del partial["kpg"]
    # An explicit dtype=None is refused (Python rejects it), not read as fp16.
    no_dtype = dict(good["problem"], dtype=None)
    for problem in (None, "N2H19", partial, no_dtype):
        sd = dict(good)
        if problem is None:
            del sd["problem"]
        else:
            sd["problem"] = problem
        with pytest.raises(ValueError):
            eng.conv_direct_grouped_wgrad_wmma_describe(sd, "gfx1250")
        with pytest.raises(ValueError):
            eng.conv_direct_grouped_wgrad_wmma_lower_llvm(sd, arch="gfx1250")
    bad = conv_direct_grouped_wgrad_wmma_spec_to_dict(
        GroupedWgradWmmaSpec(problem=_problem(), stages=5)
    )
    with pytest.raises(RuntimeError, match="stages in 2..4"):
        eng.conv_direct_grouped_wgrad_wmma_lower_llvm(bad, arch="gfx1250")


@pytest.mark.parametrize(
    "pkw, knobs",
    [
        ({}, {}),
        (dict(cpg=8, PAD=0, dtype="fp16"), dict(waves_c=8, stages=4)),
        (dict(C=80), dict(q_steps=1, rows_per_chunk=8, stages=3)),
        (dict(KH=1, KW=1, PAD=0), dict(waves_c=1, q_steps=4)),
    ],
)
def test_engines_serialize_the_same_ir(pkw, knobs):
    eng = pytest.importorskip("rocke_engine")
    from rocke.core.backend import conv_direct_grouped_wgrad_wmma_spec_to_dict
    from rocke.core.ir_serialize import serialize

    spec = GroupedWgradWmmaSpec(problem=_problem(**pkw), **knobs)
    sd = conv_direct_grouped_wgrad_wmma_spec_to_dict(spec)
    py_ir = serialize(build_grouped_wgrad_wmma(spec, arch="gfx1250"))
    assert eng.conv_direct_grouped_wgrad_wmma_serialize_ir(sd, arch="gfx1250") == py_ir
    assert eng.conv_direct_grouped_wgrad_wmma_verify(sd, arch="gfx1250") == []


def test_ir_is_shape_independent():
    """One binary serves every extent and group count of a (cpg, filter)."""
    from rocke.core.ir_serialize import serialize

    a = build_grouped_wgrad_wmma(GroupedWgradWmmaSpec(problem=_problem()))
    b = build_grouped_wgrad_wmma(
        GroupedWgradWmmaSpec(problem=_problem(N=7, H=40, W=9, C=256))
    )
    assert serialize(a) == serialize(b)


def test_aot_grid_builds_on_gfx1250_only():
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        _knob_grid,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

    direction, caps_list = DIRECT_CAPABILITIES[_VARIANT]
    assert direction == "wgrad"
    assert {(c.cpg, c.PAD) for c in caps_list} == {(4, 0), (4, 1), (8, 0), (8, 1)}
    for caps in caps_list:
        assert list(_knob_grid(_VARIANT, "gfx950", "bf16", caps)) == []
        for dtype in ("bf16", "fp16"):
            probe = probe_problem(caps, dtype, _VARIANT)
            for knobs in _knob_grid(_VARIANT, "gfx1250", dtype, caps):
                spec = make_direct_spec(_VARIANT, probe, knobs, 32)
                ok, why = validate_direct_spec(_VARIANT, spec, "gfx1250")
                assert ok, (caps, knobs, why)


def test_fold_block_shape():
    """The fold's default block shape keeps its original name and kernel; the
    grouped WMMA wgrad folds with 32 outputs over 32 slices, 8 loads deep."""
    from kernels.conv.direct_variants import direct_wgrad_fold_knobs
    from rocke.core.ir_serialize import serialize

    dflt = DwWgradFoldSpec(taps=9)
    spelled = DwWgradFoldSpec(taps=9, cols=64, slices=16, unroll=4)
    assert dflt.kernel_name() == "dw_wgrad_fold_t9"
    assert spelled.kernel_name() == "dw_wgrad_fold_t9"
    assert serialize(build_dw_wgrad_fold(dflt)) == serialize(
        build_dw_wgrad_fold(spelled)
    )
    assert dflt.launch_geometry(100) == ((15, 1, 1), (1024, 1, 1))

    knobs = direct_wgrad_fold_knobs(_VARIANT, 3, 3, 8)
    assert knobs == {"taps": 72, "cols": 32, "slices": 32, "unroll": 8}
    narrow = DwWgradFoldSpec(**knobs)
    assert narrow.kernel_name() == "dw_wgrad_fold_t72_c32_s32_u8"
    assert narrow.launch_geometry(100) == ((225, 1, 1), (1024, 1, 1))
    # Only one knob off the default still names all three.
    assert DwWgradFoldSpec(taps=9, unroll=8).kernel_name() == (
        "dw_wgrad_fold_t9_c64_s16_u8"
    )
