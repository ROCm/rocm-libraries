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

from rocke.dispatch.core import KernelId, make_kernel_id, selector_matches


@dataclasses.dataclass(frozen=True)
class _Spec:
    block_size: int = 64
    value_splits: int = 8


def _candidate(**over):
    base = dict(
        family="gdn_prefill",
        name="gdn_prefill_gfx950_chunk_scan",
        algorithm="chunk_scan",
        spec_id="b4",
        abi_version="v1",
    )
    base.update(over)
    return SimpleNamespace(**base)


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


def test_pin_is_case_and_whitespace_insensitive():
    ok, _ = selector_matches(
        _request(algorithm="  Chunk_Scan  "), _candidate(algorithm="chunk_scan")
    )
    assert ok


# --- make_kernel_id ---------------------------------------------------------


def test_kernel_id_is_deterministic():
    a = make_kernel_id(_request(), _candidate(), _Spec(), op="gdn_prefill")
    b = make_kernel_id(_request(), _candidate(), _Spec(), op="gdn_prefill")
    assert a == b


def test_kernel_id_tracks_the_spec():
    base = make_kernel_id(_request(), _candidate(), _Spec(), op="gdn_prefill")
    other = make_kernel_id(
        _request(), _candidate(), _Spec(value_splits=2), op="gdn_prefill"
    )
    assert base.spec_hash != other.spec_hash
    assert base != other


def test_kernel_id_forwards_op_from_arg_and_identity_from_candidate():
    kid = make_kernel_id(_request(), _candidate(family="fam_x"), _Spec(), op="op_y")
    assert isinstance(kid, KernelId)
    assert kid.op == "op_y"  # from the argument
    assert kid.family == "fam_x"  # from the candidate
    assert kid.candidate == "gdn_prefill_gfx950_chunk_scan"
    assert kid.algorithm == "chunk_scan" and kid.spec_id == "b4"
    assert kid.arch == "gfx950" and kid.abi_version == "v1"
