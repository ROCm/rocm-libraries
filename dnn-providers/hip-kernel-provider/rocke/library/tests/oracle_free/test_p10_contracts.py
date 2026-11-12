# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the coordinate-codeword placement probe (P10)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ._analytical import UNIFORM_QUICK_CASES, UniformCase
from ._dense_subject import DENSE_SUBJECTS
from ._p10_analytical import (
    P10_GROUPED_CASES,
    P10_QUICK_CASES,
    P10_UNGROUPED_CASES,
    require_p10_scope,
)
from ._p10_subject import (
    P10_ARCHES,
    P10_SUBJECT_CASES,
    _verify_p10_resolved_spec,
    p10_kernel_key,
    resolve_p10_subject,
)

# Literal, so a change of the uniform-probability lane's specifications does not
# silently change what the placement probe covers.
_QUICK_CASE_IDS = (
    "fp16-window64-b1-sq256-sk256-hq8-hk2-d64",
    "fp16-window64-b2-sq256-sk256-hq8-hk2-d64",
    "bf16-causal-b1-sq256-sk256-hq8-hk2-d64",
    "bf16-causal-b2-sq256-sk256-hq8-hk2-d64",
    "fp16-window64-b2-sq256-sk256-hq4-hk4-d64",
    "bf16-causal-b2-sq256-sk256-hq4-hk4-d64",
)


def _subject_case_id(item: tuple[str, UniformCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


def test_quick_matrix_pins_the_grouped_and_ungrouped_cases() -> None:
    assert tuple(case.case_id for case in P10_QUICK_CASES) == _QUICK_CASE_IDS
    assert {replace(case, batch=1) for case in P10_GROUPED_CASES} == set(
        UNIFORM_QUICK_CASES
    )
    # Batch placement needs batch 2, and output-head placement needs one query
    # head per KV head, on every dtype and mask of the lane.
    pairings = {(case.dtype, case.mask) for case in UNIFORM_QUICK_CASES}
    assert {
        (case.dtype, case.mask) for case in P10_GROUPED_CASES if case.batch == 2
    } == pairings
    assert {(case.dtype, case.mask) for case in P10_UNGROUPED_CASES} == pairings
    for case in P10_GROUPED_CASES:
        assert case.nhead_q > case.nhead_k
    for case in P10_UNGROUPED_CASES:
        assert case.nhead_q == case.nhead_k
        assert case.batch == 2


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_scope_admits_the_quick_matrix(case: UniformCase) -> None:
    require_p10_scope(case)


@pytest.mark.parametrize(
    "case, message",
    [
        (
            UniformCase(3, 256, 256, 8, 2, 64, "bf16", "causal"),
            r"batch=3, expected at most 2",
        ),
        (
            UniformCase(1, 256, 256, 8, 8, 64, "bf16", "causal"),
            r"nhead_k=8, expected at most 4",
        ),
        (
            UniformCase(1, 512, 512, 8, 2, 64, "bf16", "causal"),
            r"seqlen_k=512, expected at most 256",
        ),
        (
            UniformCase(1, 256, 256, 8, 2, 128, "bf16", "causal"),
            r"head_dim=128, expected at most 64",
        ),
        (
            UniformCase(1, 256, 256, 2, 2, 2, "fp16", "causal"),
            r"head_dim=2, expected at least 4",
        ),
        # Row q of a window keeps keys (q - 64, q]; with 256 keys the first row
        # that keeps none is 319.
        (
            UniformCase(1, 384, 256, 8, 2, 64, "fp16", "window", 64),
            r"query row 319 has no legal key",
        ),
    ],
)
def test_scope_refuses_cases_outside_the_codeword(
    case: UniformCase, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        require_p10_scope(case)


def test_subject_matrix_runs_every_quick_case_on_each_derived_arch() -> None:
    assert set(P10_ARCHES) <= set(DENSE_SUBJECTS)
    assert P10_SUBJECT_CASES == tuple(
        (arch, case) for arch in P10_ARCHES for case in P10_QUICK_CASES
    )
    assert len(set(P10_SUBJECT_CASES)) == len(P10_SUBJECT_CASES)


def test_subject_refuses_an_arch_without_a_derivation() -> None:
    assert "gfx950" in DENSE_SUBJECTS
    with pytest.raises(ValueError, match=r"arch='gfx950', expected one of"):
        resolve_p10_subject("gfx950", P10_QUICK_CASES[0])


def test_subject_refuses_a_case_outside_the_codeword_before_dispatch() -> None:
    with pytest.raises(ValueError, match=r"batch=3, expected at most 2"):
        resolve_p10_subject(
            "gfx942", UniformCase(3, 256, 256, 8, 2, 64, "bf16", "causal")
        )


def _fake_tuning_spec(**overrides) -> SimpleNamespace:
    kernel_spec = SimpleNamespace(
        window_wave_skip=False, persistent=False, block_m=128, block_n=64
    )
    for name, value in overrides.items():
        setattr(kernel_spec, name, value)
    return SimpleNamespace(kernel_spec=kernel_spec)


def test_subject_refuses_the_wave_skipping_window_schedule() -> None:
    _verify_p10_resolved_spec(_fake_tuning_spec())
    with pytest.raises(AssertionError, match=r"window_wave_skip=True"):
        _verify_p10_resolved_spec(_fake_tuning_spec(window_wave_skip=True))


@pytest.mark.parametrize(
    "name,value", [("persistent", True), ("block_m", 64), ("block_n", 32)]
)
def test_tile_and_grid_knobs_stay_free(name: str, value) -> None:
    """The schedule is read from the resolved spec, so a retune inside the scope
    keeps running instead of failing a pinned geometry."""
    _verify_p10_resolved_spec(_fake_tuning_spec(**{name: value}))


@pytest.mark.parametrize("subject_case", P10_SUBJECT_CASES, ids=_subject_case_id)
def test_quick_cases_resolve_inside_the_scope(
    subject_case: tuple[str, UniformCase],
) -> None:
    arch, case = subject_case
    subject = resolve_p10_subject(arch, case)
    assert (subject.arch, subject.case_id) == (arch, case.case_id)
    assert subject.candidate_name == DENSE_SUBJECTS[arch].candidate_name
    assert not subject.tuning_spec.kernel_spec.window_wave_skip


@pytest.mark.parametrize("subject_case", P10_SUBJECT_CASES, ids=_subject_case_id)
def test_batch_and_head_counts_are_kernel_arguments(
    subject_case: tuple[str, UniformCase],
) -> None:
    """On the runtime-shape path one compiled kernel serves every batch and head
    count, so the grouped and ungrouped cases of a dtype differ only in kernel
    arguments. A baked-shape kernel must instead split its identity on them."""
    arch, case = subject_case
    spec = resolve_p10_subject(arch, case).tuning_spec.kernel_spec
    heads = (4, 4) if case.nhead_q != case.nhead_k else (8, 2)
    other = replace(
        spec,
        batch=3 - case.batch,
        num_query_heads=heads[0],
        num_kv_heads=heads[1],
    )
    same_kernel = p10_kernel_key(arch, other) == p10_kernel_key(arch, spec)
    assert same_kernel == bool(spec.runtime_shape)
