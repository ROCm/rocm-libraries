# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection, sweep and build tests for the tensor GETT dispatcher family."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rocke.dispatch.families.tensor_gett import (
    TensorGettRequest,
    dispatch_tensor_gett,
    dispatch_tensor_gett_all,
    tensor_gett_sweep_space,
)
from rocke.instances.common.tensor import GettKnobs, TensorGettSpec, is_valid_spec

# a P3-style permuted contraction (row-major mode lists, last mode fastest)
_A = ["K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"]
_B = ["K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"]
_D = ["N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"]
_EXT = {
    "K0": 8,
    "K1": 8,
    "K2": 6,
    "K3": 4,
    "M0": 16,
    "M1": 12,
    "M2": 10,
    "M3": 8,
    "N0": 14,
    "N1": 10,
    "N2": 12,
    "N3": 8,
}


def _req(arch="gfx950", dtype="fp16", bilinear=False, **kw):
    p = TensorGettSpec.make(
        _A, _B, _D, _EXT, dtype=dtype, knobs=GettKnobs(bilinear=bilinear), **kw
    )
    return TensorGettRequest(problem=p, arch=arch)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("bilinear", [False, True])
def test_selects_the_plan_default(dtype, bilinear):
    r = dispatch_tensor_gett(_req(dtype=dtype, bilinear=bilinear))
    assert r.candidate.name == "tensor_gett_t32s3"
    assert (r.spec.tile_k, r.spec.dtl_stages, r.spec.persistent_ctas) == (32, 3, 0)
    assert r.spec.bilinear == bilinear
    assert r.block == (256, 1, 1)
    plan = r.spec.plan()
    assert r.grid == (-(-plan.N // 256), -(-plan.M // 256), 1)
    names = [a["name"] for a in r.signature]
    assert names[:7] == ["A", "B", "C", "M", "N", "K", "alpha"]
    assert (names[7:] == ["Cin", "beta"]) == bilinear
    kd = r.candidate.build(r.spec, "gfx950")
    assert [p.name for p in kd.params] == names


def test_body_levers_of_the_request_are_ignored():
    r = dispatch_tensor_gett(_req(tile_k=64, dtl_stages=2, persistent_ctas=256))
    assert (r.spec.tile_k, r.spec.dtl_stages, r.spec.persistent_ctas) == (32, 3, 0)


def test_sweep_space_is_valid_neighbourhood():
    specs = tensor_gett_sweep_space(_req())
    assert len(specs) >= 4
    assert specs[0] == dispatch_tensor_gett(_req()).spec
    assert all(is_valid_spec(s, arch="gfx950")[0] for s in specs)
    assert len(set(s.kernel_name() for s in specs)) == len(specs)


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a", "gfx1201"])
def test_other_arches_not_admitted(arch):
    assert dispatch_tensor_gett_all(_req(arch=arch)) == ()
    with pytest.raises(ValueError):
        dispatch_tensor_gett(_req(arch=arch))


def test_unplannable_problems_not_admitted():
    # batch mode L in A, B and D
    p = TensorGettSpec.make(
        ["L", "M", "K"],
        ["L", "N", "K"],
        ["L", "M", "N"],
        {"L": 4, "M": 256, "N": 256, "K": 64},
    )
    req = TensorGettRequest(problem=p, arch="gfx950")
    assert dispatch_tensor_gett_all(req) == ()
    assert dispatch_tensor_gett_all(_req(dtype="f32")) == ()


def test_request_dims():
    d = _req().dims()
    assert d == {"M": 16 * 12 * 10 * 8, "N": 14 * 10 * 12 * 8, "K": 8 * 8 * 6 * 4}
    assert replace(_req(), arch="gfx942").dims() == d
