# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT backward-filter instances
(hipkernel:Gfx950ConvWgradAot).

An instance is parsed from one line of ``instances_wgrad.txt``; these tests hold the
adapter (``builders/common/convolution_backward_filter.py``) to that catalog:

* every line parses into a :class:`Gfx950ConvWgradAotSpec` with no leftover tokens;
* each family prefix maps to the ``(kernel_family, direct_variant)``
  :data:`AOT_WGRAD_DIRECT_PREFIXES` states;
* a sample instance of every family is a valid packageable instance
  (``supports_gfx950_conv_wgrad_aot``) and builds through rocKE's own entry points
  (``build_gfx950_conv_wgrad_aot``).

The native-pack cross-check (byte-identity against ``Gfx950ConvWgradNative.cpp``) is
skipped: that pack does not exist yet (Phase 3 of the wgrad AOT rollout).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from builders.common.convolution_backward_filter import (
    AOT_WGRAD_DIRECT_PREFIXES,
    AOT_WGRAD_HELPER_PREFIXES,
    HELPER_VARIANT_FOLD,
    KERNEL_FAMILY_DIRECT,
    KERNEL_FAMILY_IMPLICIT_GEMM,
    KERNEL_FAMILY_WGRAD_HELPER,
    Gfx950ConvWgradAotSpec,
    build_gfx950_conv_wgrad_aot,
    gfx950_conv_wgrad_aot_problem_for_request,
    load_wgrad_instances,
    parse_wgrad_instance_name,
    supports_gfx950_conv_wgrad_aot,
)
from dispatch.grouped_convolution import ConvGroupedRequest

_INSTANCES_PATH = Path(__file__).resolve().parents[2] / "instances_wgrad.txt"


def _lines() -> list[str]:
    with open(_INSTANCES_PATH) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_instances_file_exists_and_is_nonempty():
    lines = _lines()
    assert len(lines) > 0


def test_every_instance_line_parses():
    errors = []
    for line in _lines():
        try:
            parse_wgrad_instance_name(line)
        except ValueError as exc:
            errors.append(f"{line}: {exc}")
    assert not errors, "\n".join(errors)


def test_family_prefixes_match_known_set():
    known_direct = set(AOT_WGRAD_DIRECT_PREFIXES) | set(AOT_WGRAD_HELPER_PREFIXES)
    counts = Counter()
    for line in _lines():
        family = line.split("-")[1]
        counts[family] += 1
    assert "implicit_gemm" in counts
    for family in counts:
        assert family == "implicit_gemm" or family in known_direct, family


def test_implicit_gemm_instance_fields():
    line = next(ln for ln in _lines() if ln.split("-")[1] == "implicit_gemm")
    spec = parse_wgrad_instance_name(line)
    assert spec.kernel_family == KERNEL_FAMILY_IMPLICIT_GEMM
    assert spec.tile_m > 0 and spec.tile_n > 0 and spec.tile_k > 0
    assert spec.pipeline in ("mem", "compv3", "compv4")
    assert spec.wave_size == 64


@pytest.mark.parametrize("family", sorted(AOT_WGRAD_DIRECT_PREFIXES))
def test_direct_family_instance_fields(family):
    lines = [ln for ln in _lines() if ln.split("-")[1] == family]
    assert lines, f"no instances for family {family!r}"
    spec = parse_wgrad_instance_name(lines[0])
    assert spec.kernel_family == KERNEL_FAMILY_DIRECT
    assert spec.direct_variant == AOT_WGRAD_DIRECT_PREFIXES[family][1]
    assert spec.filter_h > 0 and spec.filter_w > 0


def test_every_knob_token_round_trips_through_the_spec():
    """Regression test for a whole class of bug: an instance name's ``key=value``
    token silently dropped because the field was missing from the spec dataclass
    and/or its variant's ``AOT_WGRAD_DIRECT_KNOBS`` entry (found three times in
    this session -- ``direct_wgrad_fold``'s ``taps``, ``direct_wgrad_gather``'s
    ``wave_k16``, ``direct_wgrad_grouped_mfma``'s ``wave_q4`` -- each one built
    silently with the dataclass default instead of the name's stated value).
    Every token on every line must land on the parsed spec unchanged."""
    errors = []
    for line in _lines():
        spec = parse_wgrad_instance_name(line)
        for token in line.split("-"):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if value == "true":
                expected = 1
            elif value == "false":
                expected = 0
            else:
                expected = int(value)
            actual = getattr(spec, key, "<no such field>")
            if actual != expected:
                errors.append(f"{line}: {key}={value!r} parsed as {actual!r}")
    assert not errors, "\n".join(errors)


def test_load_wgrad_instances_matches_line_count():
    specs = load_wgrad_instances(str(_INSTANCES_PATH))
    assert len(specs) == len(_lines())
    assert all(isinstance(s, Gfx950ConvWgradAotSpec) for s in specs)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _single_stage_implicit_gemm_lines() -> list[str]:
    # Excludes the '2stage' instances, which need the multi-launch two-stage
    # pipeline (see test_two_stage_implicit_gemm_instances_...). 'grp' alone
    # means KernelIdentity.grouped (serves >1 groups), not group_merge -- the
    # real group_merge degree is the separate 'gm<N>' token.
    return [
        ln
        for ln in _lines()
        if ln.split("-")[1] == "implicit_gemm" and "2stage" not in ln.split("-")
    ]


def test_implicit_gemm_sample_is_valid():
    spec = parse_wgrad_instance_name(_single_stage_implicit_gemm_lines()[0])
    ok, why = supports_gfx950_conv_wgrad_aot(spec)
    assert ok, why


def test_two_stage_implicit_gemm_instances_exist_and_are_flagged():
    """``group_merge`` + ``split_k`` instances need the two-stage pipeline
    (a multi-launch ``(pipeline, workspace_nbytes)`` pair), not yet packaged by
    this adapter -- see ``WGRAD_AOT_PROGRESS.md``."""
    two_stage_lines = [
        ln
        for ln in _lines()
        if ln.split("-")[1] == "implicit_gemm" and "2stage" in ln.split("-")
    ]
    assert two_stage_lines, "expected at least one two-stage instance in the catalog"
    spec = parse_wgrad_instance_name(two_stage_lines[0])
    assert spec.is_two_stage


@pytest.mark.parametrize("family", sorted(AOT_WGRAD_DIRECT_PREFIXES))
def test_direct_family_sample_is_valid(family):
    lines = [ln for ln in _lines() if ln.split("-")[1] == family]
    spec = parse_wgrad_instance_name(lines[0])
    ok, why = supports_gfx950_conv_wgrad_aot(spec)
    assert ok, why


# ---------------------------------------------------------------------------
# Build (one sample per family; the full catalog is covered by the packaging
# probe/census once the native pack exists)
# ---------------------------------------------------------------------------


def test_implicit_gemm_sample_builds():
    spec = parse_wgrad_instance_name(_single_stage_implicit_gemm_lines()[0])
    kernel = build_gfx950_conv_wgrad_aot(spec)
    assert kernel.name


@pytest.mark.parametrize("family", sorted(AOT_WGRAD_DIRECT_PREFIXES))
def test_direct_family_sample_builds(family):
    lines = [ln for ln in _lines() if ln.split("-")[1] == family]
    spec = parse_wgrad_instance_name(lines[0])
    kernel = build_gfx950_conv_wgrad_aot(spec)
    assert kernel.name


# ---------------------------------------------------------------------------
# direct_wgrad_fold helper instances (derived, not listed in instances_wgrad.txt
# directly -- appended by running tools/enumerate_conv_wgrad_aot_instances.py's
# with_helpers() over the catalog's folded direct instances)
# ---------------------------------------------------------------------------


def _fold_helper_lines() -> list[str]:
    return [ln for ln in _lines() if ln.split("-")[1] == HELPER_VARIANT_FOLD]


def test_fold_helper_instances_exist():
    lines = _fold_helper_lines()
    assert lines, "expected at least one direct_wgrad_fold helper instance"
    assert {ln.split("-")[2] for ln in lines} == {"bf16", "fp16"}


def test_fold_helper_instance_fields():
    spec = parse_wgrad_instance_name(_fold_helper_lines()[0])
    assert spec.kernel_family == KERNEL_FAMILY_WGRAD_HELPER
    assert spec.is_helper
    assert spec.direct_variant == HELPER_VARIANT_FOLD
    assert spec.taps > 0


def test_fold_helper_sample_is_valid():
    spec = parse_wgrad_instance_name(_fold_helper_lines()[0])
    ok, why = supports_gfx950_conv_wgrad_aot(spec)
    assert ok, why


def test_fold_helper_sample_builds():
    spec = parse_wgrad_instance_name(_fold_helper_lines()[0])
    kernel = build_gfx950_conv_wgrad_aot(spec)
    assert kernel.name


@pytest.mark.parametrize("family", ("direct_wgrad_gather", "direct_wgrad_grouped_mfma"))
def test_fold_helper_taps_accounts_for_per_channel_cpg(family):
    """Regression test: a per-channel folded producer's fold helper must carry
    ``taps = filter_h * filter_w * cpg`` (``direct_wgrad_fold_knobs()``'s
    definition), not ``filter_h * filter_w`` alone -- the bug
    ``to_direct_spec()`` had before it read ``taps`` off the instance name's
    ``taps=N`` token instead of recomputing it from a KMD ``cpg`` that
    ``_helper_identity`` has already zeroed by the time a helper spec is
    parsed."""
    producer = parse_wgrad_instance_name(
        next(ln for ln in _lines() if ln.split("-")[1] == family)
    )
    expected_taps = producer.filter_h * producer.filter_w * producer.cpg
    fold_taps = {
        parse_wgrad_instance_name(ln).taps
        for ln in _fold_helper_lines()
        if ln.split("-")[2] == producer.dtype
    }
    assert expected_taps in fold_taps


# ---------------------------------------------------------------------------
# Profile hooks
# ---------------------------------------------------------------------------


def test_problem_for_request_wgrad_direction():
    request = ConvGroupedRequest(
        N=8,
        C=64,
        K=64,
        Hi=56,
        Wi=56,
        Y=3,
        X=3,
        arch="gfx950",
        direction="wgrad",
        dtype="fp16",
    )
    problem = gfx950_conv_wgrad_aot_problem_for_request(request)
    assert problem.N == 8
    assert problem.dtype == "fp16"


def test_problem_for_request_rejects_other_direction():
    request = ConvGroupedRequest(
        N=8,
        C=64,
        K=64,
        Hi=56,
        Wi=56,
        Y=3,
        X=3,
        arch="gfx950",
        direction="fwd",
        dtype="fp16",
    )
    with pytest.raises(ValueError):
        gfx950_conv_wgrad_aot_problem_for_request(request)


# ---------------------------------------------------------------------------
# Native-pack cross-check (Phase 3)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Gfx950ConvWgradGeometry/Native C++ pack not implemented yet — Phase 3"
)
def test_build_is_byte_identical_to_native_pack():
    pass
