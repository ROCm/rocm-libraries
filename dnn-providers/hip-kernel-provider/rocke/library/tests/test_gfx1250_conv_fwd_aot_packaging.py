# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the gfx1250 forward-convolution AOT packaging adapter
(``builders/gfx1250/conv_fwd_aot.py``, hipkernel:Gfx1250ConvFwdAot).

* the packaged kernel is the sweep's: for one instance of every family in each dtype,
  :func:`build_gfx1250_conv_fwd_aot` lowers to the IR the ``--compile-all`` job of the
  same identity lowers to, up to the symbol;
* :func:`supports_gfx1250_conv_fwd_aot` accepts only what the builder builds: a family
  knob the family's rocKE spec does not take (or a value it refuses) is refused by the
  predicate, not left to fail in the build.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from builders.gfx1250 import conv_fwd_aot as A
from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
from kernels.conv.direct_aot_jobs import direct_job_for_identity, direct_kernel_for_job
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


@pytest.fixture(scope="module")
def firsts():
    """``(family, dtype) -> identity``: the first registered instance of each."""
    out = {}
    for identity in A.registered_instances().values():
        out.setdefault((identity.algorithm, identity.dtype_a), identity)
    return out


def _sweep_kernel(identity):
    if identity.is_direct:
        job = direct_job_for_identity(identity)
        return direct_kernel_for_job(job, A.ARCH, identity.dtype_a)[0]
    return kernel_for_job(job_for_identity(identity), A.ARCH, identity.dtype_a)[0]


def _ir(kernel) -> str:
    text = _lower_kernel_to_llvm_python(
        kernel, arch=A.ARCH, llvm_flavor=A.PACKAGED_LLVM_FLAVOR
    )
    return text.replace(kernel.name, "@K")


def test_every_family_is_packageable_with_the_sweeps_ir(firsts):
    """One instance per family and dtype, built by the adapter and by the sweep's
    ``--compile-all`` job, lowers to the same IR up to the symbol."""
    assert {family for family, _ in firsts} == set(A.FAMILIES)
    assert {dtype for _, dtype in firsts} == set(A.DTYPES)
    for (family, dtype), identity in sorted(firsts.items()):
        spec = A.spec_from_identity(identity)
        assert spec.dtype == dtype
        ours = A.build_gfx1250_conv_fwd_aot(spec)
        assert _ir(ours) == _ir(_sweep_kernel(identity)), identity.label()


def test_the_dtype_reaches_the_kernel(firsts):
    """A bf16 instance does not lower to its fp16 twin's IR, so the IR comparison
    above tells a bf16 instance built as fp16 apart."""
    for family in A.FAMILIES:
        bf16 = firsts[(family, "bf16")]
        as_fp16 = dataclasses.replace(
            bf16, dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"
        )
        assert _ir(_sweep_kernel(bf16)) != _ir(_sweep_kernel(as_fp16)), family


def _with_knob(spec, knob: str, kind: str):
    value = {"int": 2, "bool": A.BOOL_TRUE, "string": "iterative-ilp"}[kind]
    spec = dataclasses.replace(spec, **{knob: value})
    return dataclasses.replace(spec, instance_id=spec.to_identity().stable_hash())


def test_the_predicate_accepts_only_what_the_builder_builds(firsts):
    """Every family knob an instance leaves unset, stated: the predicate either
    refuses the spec or rocKE builds its job (no ``TypeError`` from a spec that does
    not take the knob, no ``LookupError`` from one that refuses its value)."""
    refused = accepted = 0
    for (family, dtype), identity in sorted(firsts.items()):
        if dtype != "fp16":
            continue
        spec = A.spec_from_identity(identity)
        stated = set(spec.knobs())
        for knob, kind in A.FAMILY_KNOBS[family].items():
            if knob in stated:
                continue
            variant = _with_knob(spec, knob, kind)
            ok, reason = A.supports_gfx1250_conv_fwd_aot(variant)
            if not ok:
                refused += 1
                with pytest.raises(ValueError, match="invalid packaged gfx1250"):
                    A.build_gfx1250_conv_fwd_aot(variant)
                continue
            accepted += 1
            identity = variant.to_identity()
            (direct_job_for_identity if identity.is_direct else job_for_identity)(
                identity
            )
    assert refused and accepted


@pytest.mark.parametrize(
    "family, knob",
    [
        ("direct_fwd_gstream_wmma", "ck"),
        ("direct_fwd_smallc_wmma", "stages"),
        ("direct_fwd_smallk_wmma", "tiles_k"),
    ],
)
def test_a_knob_the_familys_spec_does_not_take_is_refused(firsts, family, knob):
    spec = _with_knob(A.spec_from_identity(firsts[(family, "fp16")]), knob, "int")
    ok, reason = A.supports_gfx1250_conv_fwd_aot(spec)
    assert not ok
    assert "rocKE cannot build it" in reason and knob in reason


def test_another_arch_is_refused(firsts):
    spec = A.spec_from_identity(firsts[("implicit_gemm", "fp16")])
    assert A.supports_gfx1250_conv_fwd_aot(spec) == (True, "ok")
    ok, reason = A.supports_gfx1250_conv_fwd_aot(spec, arch="gfx950")
    assert not ok and "not gfx950" in reason
    with pytest.raises(ValueError, match="not gfx950"):
        A.build_gfx1250_conv_fwd_aot(spec, arch="gfx950")


def test_an_instance_id_that_is_not_its_fields_hash_is_refused(firsts):
    """The instance_id names the identity; a spec whose fields hash elsewhere is not
    the instance it claims to be, even though rocKE would build its fields."""
    for identity in (
        firsts[("implicit_gemm", "bf16")],
        firsts[("direct_fwd_wmma", "fp16")],
    ):
        spec = A.spec_from_identity(identity)
        other = dataclasses.replace(spec, instance_id="0" * 40)
        ok, reason = A.supports_gfx1250_conv_fwd_aot(other)
        assert not ok and "is not the identity its fields describe" in reason


def test_an_int_knob_of_zero_is_refused(firsts):
    """A stated int knob must be positive: 0 is the builder default a spec leaves
    unstated, so an identity stating it would not round-trip."""
    for identity in firsts.values():
        stated = json.loads(identity.knobs) if identity.knobs else {}
        ints = [
            name
            for name, value in stated.items()
            if A.FAMILY_KNOBS[identity.algorithm][name] == "int"
        ]
        if ints:
            break
    else:
        pytest.fail("no registered instance states an int knob")
    zeroed = dataclasses.replace(
        identity, knobs=json.dumps({**stated, ints[0]: 0}, sort_keys=True)
    )
    with pytest.raises(ValueError, match=f"knob {ints[0]} must be a positive integer"):
        A.spec_from_identity(zeroed)


@pytest.mark.parametrize("field", ["tile_n", "block_h"])
def test_a_direct_identity_with_a_shared_field_set_is_refused(firsts, field):
    """A direct identity leaves the implicit GEMM's tile_n and block_h fields 0; the
    adapter carries its own knobs of those names there, so a nonzero one is not a
    packaged instance."""
    identity = dataclasses.replace(firsts[("direct_fwd_wmma", "fp16")], **{field: 4})
    with pytest.raises(ValueError, match=f"direct identity's {field} field is 4"):
        A.spec_from_identity(identity)
