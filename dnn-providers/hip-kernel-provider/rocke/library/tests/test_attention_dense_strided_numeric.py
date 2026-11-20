# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch-time strides of the dense attention kernels, on a device, without torch.

Every dense body takes twelve trailing strides (Q, K, V and O, each batch, head and
token). A body whose ``body_serves_strides(spec)`` is True reads them: it gets the
numeric matrix below. A body for which it is False keeps packed ``[B, S, H, D]``
addressing: a strided view is refused with the reason, and packed launches (extent-1
axes with garbage strides included) pass.

One cohort is one body, head size, dtype, mask and LSE choice: one compiled kernel
(the spec comes from the dispatch factory) launched on views of several shapes and
layouts: BSHD, BHSD, S-major, a batch slice, padded token strides, packed QKV (MHA and
GQA), an O layout unlike Q, all strides pairwise different, and extent-1 axes with
garbage strides. Every tensor is a view into one backing buffer per case; the buffers
and their guard bands are NaN-poisoned. After every launch the O view must match an
fp64 reference, every element of the O buffer outside the view must be bit-identical
to the poison, and every input buffer must be bit-identical to what was uploaded.
Negative controls launch a strided view with packed strides (and, with ``slots``, with
one of the twelve values wrong): they must fail the comparison. Refusal rows check the
reason.

Every launch, LSE and KV-split cohorts included, goes through
``run_attention_dense_torch`` of the arch's kernel module on device-buffer views, with an
allocator of device buffers for the KV-split workspace. Nothing here imports torch: the
lane that runs it has none.

Device tests skip on a GPU of another arch or none (``ROCKE_REQUIRE_<ARCH>=1`` turns
that skip into a failure). The CPU tests run the same checks against a host model of a
kernel and prove the checks catch an ignored stride, a wrong O element, a write outside
the O view, a write into an input and a shifted O base.
"""

from __future__ import annotations

import pytest

from rocke.tests.gpu import current_device_arch_in

from . import _attention_dense_strided as h

_COHORT_IDS = [c.id for c in h.COHORTS]


@pytest.fixture
def compile_log(monkeypatch):
    """The name of every kernel compiled while the test runs."""
    import rocke.helpers.compile as compile_module

    log: list = []
    real = compile_module.compile_kernel

    def counted(kernel, *args, **kwargs):
        log.append(kernel.name)
        return real(kernel, *args, **kwargs)

    monkeypatch.setattr(compile_module, "compile_kernel", counted)
    return log


@pytest.mark.gpu
@pytest.mark.parametrize("cohort", h.COHORTS, ids=_COHORT_IDS)
def test_strided_launches(cohort, compile_log):
    current_device_arch_in((cohort.arch,), subject=f"the {cohort.arch} strided lane")
    report = h.run_cohort(h.HipBackend(), cohort, compile_log=compile_log)
    print(report.text())
    assert report.ok, report.text()


# ---------------------------------------------------------------------------
# Host tests: the cohort matrix and the checks themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cohort", h.COHORTS, ids=_COHORT_IDS)
def test_cohort_specs_resolve(cohort):
    """Every shape of a cohort gets a supported factory spec on one cache key, and
    the KV-split bodies really split."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    kmod = h.kernel_module(cohort.arch)
    keys = set()
    for name in ("A", "B", "C"):
        spec = h.make_spec(
            cohort.arch,
            cohort.body,
            h.SHAPES[name],
            cohort.d,
            cohort.dtype,
            cohort.causal,
            cohort.lse,
        )
        ok, why = kmod.supports_attention_dense(spec, arch=cohort.arch)
        assert ok, why
        assert (spec.kv_segments > 1) == (cohort.body == "kvseg")
        assert spec.emit_lse == cohort.lse
        keys.add(attention_dense_cache_key(spec, arch=cohort.arch))
    assert len(keys) == 1


@pytest.mark.parametrize("cohort", h.COHORTS, ids=_COHORT_IDS)
def test_every_planned_shape_has_a_factory_kernel(cohort):
    """Every shape a cohort plans resolves to a supported factory spec, so no device
    row fails on a factory refusal (which run_cohort reports as a cohort problem)."""
    kmod = h.kernel_module(cohort.arch)

    def spec_of(shape):
        return h.make_spec(
            cohort.arch,
            cohort.body,
            shape,
            cohort.d,
            cohort.dtype,
            cohort.causal,
            cohort.lse,
        )

    first = spec_of(h.SHAPES["A"])
    served = bool(kmod.body_serves_strides(first))
    names = {shape.name for _, shape in h._plan(cohort, served, first.kv_segments > 1)}
    for name in sorted(names):
        ok, why = kmod.supports_attention_dense(
            spec_of(h.SHAPES[name]), arch=cohort.arch
        )
        assert ok, f"{name}: {why}"
    # The cross-length rows run wherever the factory serves them (gfx950, no mask).
    assert ({"X", "N"} <= names) == (
        served and not cohort.causal and cohort.arch == "gfx950"
    )


def test_matrix_covers_the_axes():
    """The matrix spans both arches, every body, both head sizes, both dtypes, both
    masks and LSE on and off."""
    for arch in ("gfx942", "gfx950"):
        cohorts = [c for c in h.COHORTS if c.arch == arch]
        assert {c.body for c in cohorts} >= {"grid", "kvseg", "persist"}
        assert {c.d for c in cohorts} == {64, 128}
        assert {c.dtype for c in cohorts} == {"fp16", "bf16"}
        assert {c.causal for c in cohorts} == {True, False}
        assert {c.lse for c in cohorts} == {True, False}
    assert len(set(_COHORT_IDS)) == len(_COHORT_IDS)


_MODEL_COHORTS = {
    # served by the gfx942 grid body: the full matrix with every control
    "served": h.Cohort("gfx942", "grid", 64, "fp16", False, slots=True),
    # the KDB pipeline keeps packed addressing by design: refusals and packed rows
    "packed_only": h.Cohort("gfx942", "kdb", 128, "bf16", True),
}


def test_model_backend_clean_served_cohort():
    report = h.run_cohort(h.ModelBackend(), _MODEL_COHORTS["served"])
    assert report.served and report.ok, report.text()
    counts = report.counts()
    assert counts == {"pass": 26, "caught": 16, "refused": 3}, counts


def test_model_backend_clean_packed_only_cohort():
    report = h.run_cohort(h.ModelBackend(), _MODEL_COHORTS["packed_only"])
    assert not report.served and report.ok, report.text()
    counts = report.counts()
    assert counts["pass"] == 5 and counts["refused"] == 2


def test_a_factory_refusal_of_a_planned_shape_fails_the_cohort(monkeypatch):
    """A planned shape the factory has no kernel for is a FAIL row and a cohort
    problem, never an ok skip that drops its launches silently."""
    real = h.make_spec

    def refusing(arch, body, shape, *rest):
        if shape.name == "C":
            raise ValueError("no kernel for this shape (test)")
        return real(arch, body, shape, *rest)

    monkeypatch.setattr(h, "make_spec", refusing)
    report = h.run_cohort(h.ModelBackend(), _MODEL_COHORTS["served"])
    rows_c = [r for r in report.rows if r.shape == "C"]
    assert rows_c and all(r.status == "FAIL" and not r.ok for r in rows_c)
    assert all("factory refuses the shape" in r.error for r in rows_c)
    assert not report.ok
    assert sum("factory refuses the shape" in p for p in report.problems) == len(rows_c)


@pytest.mark.parametrize("defect", ["blind", "element", "outside", "input", "shift_o"])
def test_checks_catch_a_faulty_kernel(defect):
    report = h.run_cohort(h.ModelBackend(defect), _MODEL_COHORTS["served"])
    assert report.problems, f"the checks missed the {defect} fault"


def test_a_stride_ignoring_kernel_fails_every_stride_control():
    """A kernel that reads packed BSHD whatever it is told passes the packed rows and
    fails every strided one."""
    report = h.run_cohort(h.ModelBackend("blind"), _MODEL_COHORTS["served"])
    failed = {r.layout for r in report.rows if r.status == "FAIL"}
    assert {"bhsd", "sbhd", "batch_slice", "pad_token", "all_distinct"} <= failed
    passed = {r.layout for r in report.rows if r.status == "pass"}
    assert {"bshd", "extent1"} <= passed


def _lse_problem():
    """A host problem with an LSE buffer: its views as the runner reads them."""
    shape, d = h.SHAPES["A"], 64
    spec = h.make_spec("gfx942", "grid", shape, d, "fp16", True, True)
    prob = h.Problem(
        h.ModelBackend(),
        h.make_case("bshd", shape, d),
        shape,
        spec,
        h.make_logical(shape, d, "fp16", True),
    )
    inputs = {n: prob.views[n] for n in ("q", "k", "v")} | {"out": prob.views["o"]}
    return prob, (shape.b, shape.hq, shape.sq, 1), inputs


def test_lse_buffer_check_takes_device_views():
    """The LSE contract needs attributes, not torch: a device-buffer view passes, and
    each way to break the contract is refused with its reason."""
    import numpy as np

    from kernels.common._lse_store import _check_lse_buffer

    prob, expected, inputs = _lse_problem()
    q = prob.views["q"]
    _check_lse_buffer(prob.lse, q, expected, inputs)
    with pytest.raises(ValueError, match="shape"):
        _check_lse_buffer(prob.lse, q, (1, *expected[1:]), inputs)
    half = h.LseView(prob.lse_buf, expected)
    half.dtype = np.dtype(np.float16)
    with pytest.raises(ValueError, match="float32"):
        _check_lse_buffer(half, q, expected, inputs)
    onto_o = h.LseView(prob.views["o"].buf, expected)
    with pytest.raises(ValueError, match="lse overlaps out"):
        _check_lse_buffer(onto_o, q, expected, inputs)
    with pytest.raises(ValueError, match="torch tensor"):
        _check_lse_buffer(np.zeros(expected, np.float32), q, expected, inputs)


def test_segment_workspace_takes_an_allocator():
    from kernels.common.attention_dense_kv_split import (
        SEGMENT_PARAMS,
        allocate_segment_workspace,
    )

    spec = h.make_spec("gfx942", "kvseg", h.SHAPES["A"], 64, "fp16", False, False)
    asked = []
    workspace = allocate_segment_workspace(
        spec, None, allocator=lambda shape: asked.append(shape) or shape
    )
    rows = spec.batch * spec.seqlen_q
    seg = (rows, spec.num_query_heads, spec.kv_segments)
    assert tuple(workspace) == SEGMENT_PARAMS
    assert asked == [seg + (spec.head_size,), seg, seg]
