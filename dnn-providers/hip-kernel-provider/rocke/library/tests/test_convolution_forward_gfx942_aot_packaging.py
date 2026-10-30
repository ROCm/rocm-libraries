# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the gfx942 shape-generic AOT forward instances
(hipkernel:Gfx942ConvFwdAot).

This is the gfx942 counterpart of ``test_convolution_forward_aot_packaging.py``
(gfx950). It does not re-derive the exhaustive sweep-permutation coverage that
file has (:mod:`builders.common.convolution_forward_gfx942` differs from the
gfx950 adapter only in ``_ARCH`` and symbol names, verified by direct
comparison when it was ported) -- instead it holds the adapter to the real
shipped catalog (``instances_new.txt``, via
``dnn-providers/hip-kernel-provider/rocke/library/dispatch/grouped_convolution_instances/gfx942_fwd.json``)
plus one representative case per packaged family.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from builders.common.convolution_forward_gfx942 import (
    AOT_DIRECT_ALGORITHMS,
    Gfx942ConvFwdAotSpec,
    Gfx942ConvFwdSpec,
    gfx942_conv_fwd_aot_capabilities,
    gfx942_conv_fwd_aot_serves,
    gfx942_conv_fwd_aot_spec_for_identity,
    supports_gfx942_conv_fwd_aot,
)
from dispatch.grouped_convolution import load_instance_list
from kernels.conv.kernel_identity import KernelIdentity

_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "dispatch"
    / "grouped_convolution_instances"
    / "gfx942_fwd.json"
)


def _catalog_identities() -> tuple[KernelIdentity, ...]:
    return load_instance_list(_CATALOG).identities


def test_the_registered_gfx942_catalog_is_681_instances():
    identities = _catalog_identities()
    assert len(identities) == 681
    assert {i.arch for i in identities} == {"gfx942"}


@pytest.mark.parametrize("identity", _catalog_identities(), ids=lambda i: i.label())
def test_every_registered_instance_is_packageable(identity: KernelIdentity):
    spec, reason = gfx942_conv_fwd_aot_spec_for_identity(identity)
    assert spec is not None, reason
    assert spec.identity() == identity
    ok, why = supports_gfx942_conv_fwd_aot(spec)
    assert ok, why


def test_the_catalog_covers_every_packaged_direct_algorithm_the_labels_use():
    identities = _catalog_identities()
    direct_algorithms = {i.algorithm for i in identities if i.is_direct}
    assert direct_algorithms <= set(AOT_DIRECT_ALGORITHMS)
    # instances_new.txt's own family mix (see mi300plan.md Faza 2/3).
    assert direct_algorithms == {
        "direct_depthwise",
        "direct_depthwise_col",
        "direct_depthwise_spatial",
        "direct_depthwise_tiled",
        "direct_grouped",
        "direct_grouped_halo",
        "direct_nongrouped",
    }


def _one_per_family(
    identities: tuple[KernelIdentity, ...],
) -> dict[str, KernelIdentity]:
    chosen: dict[str, KernelIdentity] = {}
    for identity in identities:
        chosen.setdefault(identity.algorithm, identity)
    return chosen


@pytest.fixture(scope="module")
def one_spec_per_family() -> dict[str, Gfx942ConvFwdAotSpec]:
    identities = _one_per_family(_catalog_identities())
    specs = {}
    for algorithm, identity in identities.items():
        spec, reason = gfx942_conv_fwd_aot_spec_for_identity(identity)
        assert spec is not None, reason
        specs[algorithm] = spec
    return specs


def _problem_from_probe(spec: Gfx942ConvFwdAotSpec) -> Gfx942ConvFwdSpec:
    """The exact problem rocKE's sweep built ``spec`` against, as a
    ``Gfx942ConvFwdSpec`` -- guaranteed to be a problem the instance serves,
    unlike a hand-picked shape (direct families each impose different channel/
    group constraints, e.g. ``direct_grouped``'s compiled ``(cpg, kpg)`` vs.
    ``direct_grouped_halo``'s runtime channel count that must still be a
    multiple of its ``ck`` knob)."""
    probe = spec.probe_problem()
    if spec.is_direct:
        return Gfx942ConvFwdSpec(
            N=probe.N,
            Hi=probe.H,
            Wi=probe.W,
            C=probe.cpg * probe.groups,
            K=probe.kpg * probe.groups,
            Y=probe.KH,
            X=probe.KW,
            sH=probe.stride,
            sW=probe.stride,
            pH=probe.PAD,
            pW=probe.PAD,
            dH=probe.dil_h,
            dW=probe.dil_w,
            dtype=spec.dtype,
            groups=probe.groups,
        )
    return Gfx942ConvFwdSpec(
        N=probe.N,
        Hi=probe.Hi,
        Wi=probe.Wi,
        C=probe.C,
        K=probe.K,
        Y=probe.Y,
        X=probe.X,
        sH=probe.sH,
        sW=probe.sW,
        pH=probe.pH,
        pW=probe.pW,
        dH=probe.dH,
        dW=probe.dW,
        dtype=spec.dtype,
        groups=probe.groups,
    )


def test_a_representative_instance_of_each_family_serves_its_own_probe_problem(
    one_spec_per_family: dict[str, Gfx942ConvFwdAotSpec],
):
    for algorithm, spec in one_spec_per_family.items():
        problem = _problem_from_probe(spec)
        ok, why = gfx942_conv_fwd_aot_serves(spec, problem)
        assert ok, f"{algorithm}: {why}"
        # Sanity: the capabilities and the probe problem agree (regression
        # guard, not a property of serves() alone).
        gfx942_conv_fwd_aot_capabilities(spec)


def test_an_instance_declines_the_wrong_arch():
    identity = _catalog_identities()[0]
    bad = identity.__class__(**{**identity.__dict__, "arch": "gfx950"})
    spec, reason = gfx942_conv_fwd_aot_spec_for_identity(bad)
    assert spec is None
    assert "gfx942" in reason


def test_an_instance_declines_the_wrong_dtype():
    identity = _catalog_identities()[0]
    spec, _ = gfx942_conv_fwd_aot_spec_for_identity(identity)
    other_dtype = "bf16" if spec.dtype == "fp16" else "fp16"
    problem = Gfx942ConvFwdSpec(
        N=2, Hi=16, Wi=16, C=8, K=8, Y=3, X=3, pH=1, pW=1, dtype=other_dtype
    )
    ok, why = gfx942_conv_fwd_aot_serves(spec, problem)
    assert not ok
    assert "dtype" in why
