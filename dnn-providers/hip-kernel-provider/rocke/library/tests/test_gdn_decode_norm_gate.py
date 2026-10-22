# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU contract of ``norm_gate_act``: the fused output norm's gate activation.

"auto" keeps the gate kind's default (SiLU for GDN, sigmoid for KDA); an
explicit "silu" / "sigmoid" overrides it. An override equal to the default,
or any value without ``fuse_out_norm``, emits the default kernel under the
default name.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode, request_errors
from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode, is_valid_spec
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

NORM = dict(
    num_k_heads=16,
    num_v_heads=48,
    state_dtype="f32",
    fuse_out_norm=True,
    blocks_per_v_dim=1,
)


def _ll(spec):
    return _lower_kernel_to_llvm_python(build_gdn_decode(spec), arch="gfx950")


@pytest.mark.parametrize("gate_kind, default", [("gdn", "silu"), ("kda", "sigmoid")])
def test_default_override_is_the_default_kernel(gate_kind, default):
    base = GdnDecodeSpec(**dict(NORM, num_v_heads=16), gate_kind=gate_kind)
    same = dc.replace(base, norm_gate_act=default)
    assert same.out_gate_activation == default
    assert same.kernel_name() == base.kernel_name()
    assert _ll(same) == _ll(base)


def test_sigmoid_gdn_norm_is_its_own_kernel():
    base = GdnDecodeSpec(**NORM)
    sig = dc.replace(base, norm_gate_act="sigmoid")
    assert is_valid_spec(sig) == (True, "")
    assert sig.out_gate_activation == "sigmoid"
    assert "_ngsig" in sig.kernel_name() and "_ngsig" not in base.kernel_name()
    assert _ll(sig) != _ll(base)


def test_inert_without_the_norm():
    base = GdnDecodeSpec(num_k_heads=16, num_v_heads=48, state_dtype="f32")
    sig = dc.replace(base, norm_gate_act="sigmoid")
    assert sig.kernel_name() == base.kernel_name()
    assert _ll(sig) == _ll(base)


@pytest.mark.parametrize("bad", ["relu", "", None, True])
def test_rejects_unknown_activation(bad):
    ok, why = is_valid_spec(GdnDecodeSpec(**NORM, norm_gate_act=bad))
    assert not ok and "norm_gate_act must be one of" in why
    errs = request_errors(GdnDecodeRequest(batch=1, arch="gfx950", norm_gate_act=bad))
    assert any("norm_gate_act must be one of" in e for e in errs)


def test_request_carries_the_override():
    req = GdnDecodeRequest(
        batch=2,
        arch="gfx950",
        num_k_heads=16,
        num_v_heads=48,
        state_dtype="f32",
        fuse_out_norm=True,
        norm_gate_act="sigmoid",
    )
    spec = dispatch_gdn_decode(req).spec.kernel_spec
    assert spec.norm_gate_act == "sigmoid" and spec.out_gate_activation == "sigmoid"
