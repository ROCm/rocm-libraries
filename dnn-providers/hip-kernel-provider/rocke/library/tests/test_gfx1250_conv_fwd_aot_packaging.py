# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the gfx1250 target of the forward AOT packaging adapter
(``builders/common/convolution_forward.py``, hipkernel:Gfx1250ConvFwdAot).

The gfx950 target has its own suite (``test_convolution_forward_aot_packaging.py``),
built on that arch's sweeps. This one holds the gfx1250 target to the instances the
gfx1250 forward list registers:

* every registered instance converts, round-trips its identity and has the
  dispatcher's workgroup;
* the packaged kernel is the sweep's: for one instance of every family, pipeline and
  dtype, :func:`build_gfx1250_conv_fwd_aot` lowers to the IR the ``--compile-all``
  job of the same identity lowers to, up to the symbol;
* :func:`supports_gfx1250_conv_fwd_aot` accepts only what the builder builds;
* "serves" is the dispatcher's admission (plus the launch bounds), and the launch is
  the dispatcher's plan.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest
from builders.common import convolution_forward as cf
from dispatch import grouped_convolution as gc
from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
from kernels.conv.direct_aot_jobs import direct_job_for_identity, direct_kernel_for_job
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_ARCH = "gfx1250"
_TARGET = cf.TARGETS[_ARCH]
_SEED = 2026


@pytest.fixture(scope="module")
def registered():
    """Every instance of the gfx1250 forward list, as the dispatcher registers it."""
    return gc.registered_conv_grouped_instances(_ARCH, direction="fwd")


@pytest.fixture(scope="module")
def specs(registered):
    out = []
    for instance in registered:
        spec, reason = cf.conv_fwd_aot_spec_for_identity(instance.identity)
        assert spec is not None, f"{instance.name}: {reason}"
        out.append(spec)
    return out


@pytest.fixture(scope="module")
def firsts(registered):
    """``(algorithm, pipeline, dtype) -> identity``: the first registered of each."""
    out = {}
    for instance in registered:
        identity = instance.identity
        key = (identity.algorithm, identity.pipeline, identity.dtype_a)
        out.setdefault(key, identity)
    return out


def _spec(identity):
    spec, reason = cf.conv_fwd_aot_spec_for_identity(identity)
    assert spec is not None, reason
    return spec


def _sweep_kernel(identity):
    if identity.is_direct:
        job = direct_job_for_identity(identity)
        return direct_kernel_for_job(job, _ARCH, identity.dtype_a)[0]
    return kernel_for_job(job_for_identity(identity), _ARCH, identity.dtype_a)[0]


def _ir(kernel) -> str:
    text = _lower_kernel_to_llvm_python(
        kernel, arch=_ARCH, llvm_flavor=_TARGET.llvm_flavor
    )
    return text.replace(kernel.name, "@K")


def test_every_registered_instance_converts_and_round_trips(registered, specs):
    """Each registered identity is a :class:`Gfx1250ConvFwdAotSpec` that spells it
    back (its ``instance_id`` is the identity's hash) with the dispatcher's
    workgroup, and every family of the list is one the target packages."""
    families = set(cf.GFX1250_DIRECT_ALGORITHMS) | {"implicit_gemm"}
    for instance, spec in zip(registered, specs):
        assert type(spec) is cf.Gfx1250ConvFwdAotSpec
        assert spec.identity() == instance.identity
        assert spec.instance_id == instance.identity.stable_hash()
        assert spec.block() == instance.block, instance.name
        assert spec.algorithm in families
    assert {spec.algorithm for spec in specs} == families


def test_every_family_is_packaged_with_the_sweeps_ir(firsts):
    """One instance per family, pipeline and dtype, built by the adapter and by the
    sweep's ``--compile-all`` job, lowers to the same IR up to the symbol."""
    for key, identity in sorted(firsts.items()):
        ours = cf.build_gfx1250_conv_fwd_aot(_spec(identity))
        assert ours.name.startswith("hkp_conv_fwd_"), key
        assert "_aot_gfx1250_" in ours.name, key
        assert _ir(ours) == _ir(_sweep_kernel(identity)), key


def test_the_dtype_reaches_the_kernel(firsts):
    """A bf16 instance does not lower to its fp16 twin's IR, so the IR comparison
    above tells a bf16 instance built as fp16 apart."""
    for (algorithm, pipeline, dtype), bf16 in sorted(firsts.items()):
        if dtype != "bf16":
            continue
        as_fp16 = dataclasses.replace(
            bf16, dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"
        )
        assert _ir(_sweep_kernel(bf16)) != _ir(_sweep_kernel(as_fp16)), algorithm


def test_the_spec_extends_the_gfx950_spec():
    """The gfx1250 descriptor fields are the gfx950 ones, in order, then its
    pipelines' knobs, each defaulting to its absent value."""
    base = [f.name for f in dataclasses.fields(cf.Gfx950ConvFwdAotSpec)]
    fields = {f.name: f for f in dataclasses.fields(cf.Gfx1250ConvFwdAotSpec)}
    assert list(fields)[: len(base)] == base
    added = list(fields)[len(base) :]
    assert added == list(cf._GFX1250_KNOB_DEFAULTS)
    for name, default in cf._GFX1250_KNOB_DEFAULTS.items():
        assert fields[name].default == default, name
    used = {f for knobs in cf.GFX1250_DIRECT_KNOBS.values() for f in knobs.values()}
    assert set(added) <= used


def _with_knob(spec, field_name):
    value = (
        cf._GFX1250_TRUE
        if field_name in cf._GFX1250_BOOL_FIELDS
        else "iterative-ilp" if field_name in cf._GFX1250_STRING_FIELDS else 2
    )
    stated = dataclasses.replace(spec, **{field_name: value})
    return dataclasses.replace(stated, instance_id=_identity_hash(stated))


def _identity_hash(spec) -> str:
    return cf.KernelIdentity(
        **cf._aot_identity_fields(spec), llvm_flavor=_TARGET.llvm_flavor
    ).stable_hash()


def test_the_predicate_accepts_only_what_the_builder_builds(firsts):
    """Every knob of a family an instance leaves out, stated: the predicate either
    refuses the spec (and the builder raises) or rocKE's sweep builds its job."""
    refused = accepted = 0
    for (algorithm, _, dtype), identity in sorted(firsts.items()):
        if dtype != "fp16" or not identity.is_direct:
            continue
        spec = _spec(identity)
        variant = spec.direct_variant
        stated = set(spec.direct_knobs())
        for key, field_name in cf.GFX1250_DIRECT_KNOBS[variant].items():
            if key in stated:
                continue
            other = _with_knob(spec, field_name)
            ok, reason = cf.supports_gfx1250_conv_fwd_aot(other)
            if not ok:
                refused += 1
                with pytest.raises(ValueError, match="invalid packaged AOT forward"):
                    cf.build_gfx1250_conv_fwd_aot(other)
                continue
            accepted += 1
            direct_job_for_identity(other.identity())
    assert refused and accepted


@pytest.mark.parametrize(
    "algorithm, field_name",
    [
        ("direct_fwd_gstream_wmma", "ck"),
        ("direct_fwd_smallc_wmma", "stages"),
        ("direct_fwd_smallk_wmma", "tiles_k"),
    ],
)
def test_a_knob_the_familys_spec_does_not_take_is_refused(
    firsts, algorithm, field_name
):
    spec = _with_knob(_spec(firsts[(algorithm, "direct", "fp16")]), field_name)
    ok, reason = cf.supports_gfx1250_conv_fwd_aot(spec)
    assert not ok and reason


def test_a_knob_of_another_variant_is_refused(firsts):
    """A field another gfx1250 variant uses must hold its absent value."""
    spec = _spec(firsts[("direct_dw_fwd_wmma", "direct", "fp16")])
    other = dataclasses.replace(spec, walk=4)
    ok, reason = cf.supports_gfx1250_conv_fwd_aot(other)
    assert not ok and "does not use walk" in reason


def test_another_arch_is_refused(firsts):
    spec = _spec(firsts[("implicit_gemm", "wavelet", "fp16")])
    assert cf.supports_gfx1250_conv_fwd_aot(spec) == (True, "ok")
    ok, reason = cf.supports_gfx1250_conv_fwd_aot(spec, arch="gfx950")
    assert not ok and "arch=gfx1250" in reason
    with pytest.raises(ValueError, match="arch=gfx1250"):
        cf.build_gfx1250_conv_fwd_aot(spec, arch="gfx950")
    ok, reason = cf.supports_gfx950_conv_fwd_aot(spec)
    assert not ok and "Gfx950ConvFwdAotSpec" in reason
    ok, reason = cf.gfx950_conv_fwd_aot_spec_for_identity(spec.identity())
    assert ok is None and "not a gfx950" in reason


def test_an_instance_id_that_is_not_its_fields_hash_is_refused(firsts):
    """The instance_id names the identity; a spec whose fields hash elsewhere is not
    the instance it claims to be, even though rocKE would build its fields."""
    for key in (
        ("implicit_gemm", "mem", "bf16"),
        ("direct_fwd_wmma", "direct", "fp16"),
    ):
        other = dataclasses.replace(_spec(firsts[key]), instance_id="0" * 40)
        ok, reason = cf.supports_gfx1250_conv_fwd_aot(other)
        assert not ok and "instance_id" in reason, key


def test_a_stated_int_knob_of_zero_is_refused(firsts):
    """A stated int knob must be positive: 0 is the absent value, so an identity
    stating it would not round-trip."""
    identity = firsts[("direct_fwd_wmma", "direct", "fp16")]
    stated = json.loads(identity.knobs)
    zeroed = dataclasses.replace(
        identity, knobs=json.dumps({**stated, "ck": 0}, sort_keys=True)
    )
    spec, reason = cf.conv_fwd_aot_spec_for_identity(zeroed)
    assert spec is None and "knob ck must be a positive integer" in reason


def test_a_stated_string_knob_must_be_a_name(firsts):
    """A string knob (``sched``) is a non-empty name other than the absent value:
    an empty or non-string one would not round-trip."""
    for identity in firsts.values():
        stated = json.loads(identity.knobs) if identity.is_direct else {}
        if "sched" in stated:
            break
    else:
        pytest.skip("no first-of-kind direct identity states sched")
    for bad in ("", 3):
        broken = dataclasses.replace(
            identity, knobs=json.dumps({**stated, "sched": bad}, sort_keys=True)
        )
        spec, reason = cf.conv_fwd_aot_spec_for_identity(broken)
        assert spec is None and "knob sched must be a name" in reason, bad


@pytest.mark.parametrize(
    "key",
    [("implicit_gemm", "mem", "bf16"), ("direct_fwd_wmma", "direct", "fp16")],
)
def test_a_wave_size_other_than_the_targets_is_refused(firsts, key):
    """gfx1250 is wave32: an implicit-GEMM or direct instance spelling wave64 is not
    packageable (the direct and implicit paths each check it)."""
    other = dataclasses.replace(_spec(firsts[key]), wave_size=64)
    ok, reason = cf.supports_gfx1250_conv_fwd_aot(other)
    assert not ok and "wave_size must be 32 on gfx1250" in reason, key


def test_bool_knobs_keep_an_explicit_false(registered):
    """The gfx1250 tiled grid states ``unroll_rows`` False on some instances and
    leaves it out on others: both round-trip to their own identity."""
    seen = set()
    for instance in registered:
        if instance.identity.algorithm != "direct_depthwise_tiled":
            continue
        spec = _spec(instance.identity)
        seen.add(spec.unroll_rows)
        assert spec.identity() == instance.identity
    assert seen == {0, cf._GFX1250_TRUE, cf._GFX1250_FALSE}


def _problems_near(spec, rng, count: int):
    """Problems around ``spec``'s compiled geometry: its filter, stride, padding and
    dilation (or random ones for implicit GEMM), with random extents and channels."""
    out = []
    for _ in range(count):
        depthwise = spec.is_direct and spec._direct_caps() == (1, 1)
        if spec.is_direct:
            fy, fx = spec.filter_h, spec.filter_w
            sh = sw = spec.stride_h
            ph, pw = spec.pad_h, spec.pad_w
            dh, dw = spec.dilation_h, spec.dilation_w
        else:
            fy = fx = rng.choice((1, 3, 5))
            sh = sw = rng.choice((1, 2))
            ph = pw = fy // 2
            dh = dw = 1
        cpg, kpg = spec._direct_caps() if spec.is_direct else (0, 0)
        if depthwise:
            groups = rng.choice((8, 16, 32, 64, 96))
            c = k = groups
        else:
            groups = 1 if not cpg and rng.random() < 0.7 else rng.choice((2, 4, 8))
            if spec.is_merged:
                groups = spec.group_merge * rng.choice((1, 2, 8))
                cpg = kpg = 1
            c = groups * (cpg or rng.choice((8, 16, 32, 64, 3, 4)))
            k = groups * (kpg or rng.choice((8, 16, 64, 128)))
        h = rng.choice((7, 14, 16, 28, 56, 64))
        w = rng.choice((7, 14, 16, 28, 56, 64, 250))
        out.append(
            cf.Gfx950ConvFwdSpec(
                N=rng.choice((1, 2, 4)),
                Hi=h,
                Wi=w,
                C=c,
                K=k,
                Y=fy,
                X=fx,
                sH=sh,
                sW=sw,
                pH=ph,
                pW=pw,
                dH=dh,
                dW=dw,
                groups=groups,
                dtype=spec.dtype,
            )
        )
    return out


def test_serves_is_the_dispatchers_admission_and_the_launch_its_plan(specs):
    """For a sample of problems around every instance: an instance serves exactly
    what the dispatcher admits for its identity (inside the packaged problem
    contract and launch bounds), and launches with the dispatcher's plan."""
    rng = random.Random(_SEED)
    served = declined = 0
    for spec in specs:
        for problem in _problems_near(spec, rng, 6):
            if cf._problem_error(problem, decline_grouped_pointwise=False):
                continue
            ours, why = cf.conv_fwd_aot_serves(spec, problem)
            theirs, their_why = gc._admits(spec.identity(), spec.request(problem))
            if ours != theirs:
                # Only the packaged bounds decline what the dispatcher admits.
                assert theirs and not ours, (spec.identity().label(), problem)
                assert any(s in why for s in ("exceeds", "2**23", "one dispatch")), why
                declined += 1
                continue
            if not ours:
                continue
            served += 1
            step = spec.launch_step(problem)
            plan = gc.launch_plan_for(
                spec.request(problem), gc._INSTANCE_SPECS[spec.identity().label()]
            )
            assert len(plan.steps) == 1 and not plan.workspaces
            assert step.grid == plan.steps[0].grid
            assert step.block == plan.steps[0].block
            assert step.args == plan.steps[0].args
    assert served > len(specs)
