# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct unit tests for the shared dispatch helpers.

``selector_matches`` and ``make_kernel_id`` live in ``rocke.dispatch.core`` and
are shared by every operator family, so a bug in either breaks every family at
once. The family suites only exercise them *indirectly*, which means a helper
regression would surface as a confusing failure somewhere downstream. These
tests point straight at the helpers -- pure host logic, no device -- so a break
is caught at the source and named.

The inputs are faithful duck-typed stand-ins: the helpers read attributes and
``asdict(spec)``, so a namespace with the right fields and a small dataclass
spec exercise exactly the code path a real request/candidate does.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from rocke.dispatch.core import (
    CandidateRegistry,
    KernelCandidate,
    KernelId,
    make_kernel_id,
    selector_matches,
)


@dataclasses.dataclass(frozen=True)
class _Spec:
    block_size: int = 64
    value_splits: int = 8


def _candidate(**over):
    base = dict(
        family="kda_chunkwise",
        name="kda_chunkwise_gfx950_chunk_scan",
        algorithm="chunk_scan",
        spec_id="b4",
        abi_version="v1",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _registrable_candidate(**over):
    """A real ``KernelCandidate``, for the one test that goes through the
    registry rather than calling a helper directly."""
    base = dict(
        name="probe",
        family="kda_chunkwise",
        algorithm="chunk_scan",
        spec_id="b4",
        abi_version="v1",
        priority=0,
        _supports=lambda req: (True, "ok"),
        select_spec=lambda req: _Spec(),
        signature=lambda spec: (),
        grid=lambda spec, req: (1, 1, 1),
        block=lambda spec: (64, 1, 1),
        sweep_space=lambda req: (),
    )
    base.update(over)
    return KernelCandidate(**base)


def _request(**over):
    base = dict(algorithm="auto", spec_id="auto", arch="gfx950")
    base.update(over)
    ns = SimpleNamespace(**base)
    ns.normalized = lambda: {"batch": 8, "arch": ns.arch}
    return ns


# --- selector_matches -------------------------------------------------------


def test_auto_request_matches_any_candidate():
    ok, why = selector_matches(_request(), _candidate())
    assert ok and why == "ok"


def test_exact_pin_matches():
    ok, _ = selector_matches(
        _request(algorithm="chunk_scan", spec_id="b4"), _candidate()
    )
    assert ok


def test_algorithm_mismatch_is_rejected_with_reason():
    ok, why = selector_matches(
        _request(algorithm="chunk_prep"), _candidate(algorithm="chunk_scan")
    )
    assert not ok and "algorithm" in why


def test_spec_id_mismatch_is_rejected_with_reason():
    ok, why = selector_matches(_request(spec_id="b32"), _candidate(spec_id="b4"))
    assert not ok and "spec_id" in why


def test_a_request_without_the_pin_fields_raises():
    """A request type that never declared algorithm/spec_id is a family wiring
    bug. Each family's private copy read the attributes directly and raised;
    defaulting them to "auto" here would instead make every candidate match --
    the pin silently ignored, which is far harder to notice than a traceback."""
    with pytest.raises(AttributeError):
        selector_matches(SimpleNamespace(arch="gfx950"), _candidate())


def test_pin_is_case_and_whitespace_insensitive():
    # Both fields are normalized, and each normalization is its own line in
    # the helper -- so each needs its own assertion or one can be deleted with
    # the suite still green. spec_id is the likelier victim: it is a short
    # hand-typed tag like "b4" in a config or env override.
    ok, _ = selector_matches(
        _request(algorithm="  Chunk_Scan  "), _candidate(algorithm="chunk_scan")
    )
    assert ok
    ok, _ = selector_matches(_request(spec_id="  B4  "), _candidate(spec_id="b4"))
    assert ok


# --- make_kernel_id ---------------------------------------------------------


def test_kernel_id_is_deterministic():
    a = make_kernel_id(_request(), _candidate(), _Spec(), op="kda")
    b = make_kernel_id(_request(), _candidate(), _Spec(), op="kda")
    assert a == b


def test_kernel_id_tracks_the_spec():
    base = make_kernel_id(_request(), _candidate(), _Spec(), op="kda")
    other = make_kernel_id(_request(), _candidate(), _Spec(value_splits=2), op="kda")
    assert base.spec_hash != other.spec_hash
    assert base != other


def test_kernel_id_forwards_op_from_arg_and_identity_from_candidate():
    kid = make_kernel_id(_request(), _candidate(family="fam_x"), _Spec(), op="op_y")
    assert isinstance(kid, KernelId)
    assert kid.op == "op_y"  # from the argument
    assert kid.family == "fam_x"  # from the candidate
    assert kid.candidate == "kda_chunkwise_gfx950_chunk_scan"
    assert kid.algorithm == "chunk_scan" and kid.spec_id == "b4"
    assert kid.arch == "gfx950" and kid.abi_version == "v1"


# --- what actually makes the hoist behaviour-preserving ---------------------


def test_registry_rejects_a_candidate_from_another_family():
    """``make_kernel_id`` reads ``candidate.family``; the private copies it
    replaced read each family's ``_FAMILY`` constant.

    Those agree because ``CandidateRegistry.register`` REFUSES a candidate
    whose family differs from its registry, so a mismatched one never reaches
    ``candidates()``. That guard is the guarantee -- enumerating registered
    candidates and asserting the property it already enforced cannot fail, and
    a test that cannot fail hides the day someone removes the guard.
    """
    registry = CandidateRegistry("kda_chunkwise")
    with pytest.raises(ValueError, match="family"):
        registry.register(_registrable_candidate(family="attention_unified"))
