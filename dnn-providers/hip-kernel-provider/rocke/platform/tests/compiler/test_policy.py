# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Contracts for canonical compiler-policy selection and serialization."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from rocke.compiler import _support as support
from rocke.compiler import evidence as E
from rocke.compiler import model as M
from rocke.compiler.policy import CompilerPolicy, PolicyResolutionError, resolve_policy

_SHA = "a" * 64


def _identity(_: Path | None) -> dict[str, object]:
    return {
        "llvm_version": [23, 0, 0],
        "version_source": "test",
        "query_library": None,
        "library": "/test/libamd_comgr.so",
        "sha256": _SHA,
        "dependencies": [],
        "dependency_identity_complete": True,
    }


def _accept_all(
    requests: list[support.ProbeRequest], **_: object
) -> dict[str, support.ProbeResult]:
    return {
        request.key: support.ProbeResult(
            request.key,
            "accepted",
            evidence={
                "hsaco_sha256": _SHA,
                "hsaco_size": 64,
                "completed_stages": [
                    "compile_source_to_bc",
                    "codegen_bc_to_relocatable",
                    "link_relocatable_to_executable",
                ],
            },
        )
        for request in requests
    }


def _report(tmp_path: Path) -> dict[str, object]:
    report, cache_hit = E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=_accept_all,
    )
    assert cache_hit is False
    return report


def test_policy_resolver_selects_preferred_independent_capabilities(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    policy = resolve_policy(report)

    assert policy.data_layout == "L3"
    assert policy.buffer_resource_abi == "p8_p1_i64"
    assert policy.fp8_packed_operand_type == "i64"
    assert policy.named_barrier_address_space == 15
    assert policy.scaled_mfma_call_abi == "current_9"
    assert CompilerPolicy.from_json(policy.to_json()) == policy


def test_policy_resolver_preserves_unsupported_optional_families(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    for name in (
        M.CAPABILITY_FP8_PACKING,
        M.CAPABILITY_NAMED_BARRIER,
        M.CAPABILITY_SCALED_MFMA,
    ):
        for entry in report["capabilities"][name]["value"]["observations"]:
            entry.clear()
            entry.update(
                {
                    "family": "unsupported",
                    "candidate": "unsupported",
                    "outcome": "rejected",
                    "diagnostic": "not supported",
                }
            )

    policy = resolve_policy(report)

    assert policy.fp8_packed_operand_type is None
    assert policy.named_barrier_address_space is None
    assert policy.scaled_mfma_call_abi is None


def test_policy_resolver_rejects_missing_required_evidence(tmp_path: Path) -> None:
    report = _report(tmp_path)
    observations = report["capabilities"][M.CAPABILITY_LAYOUT]["value"]["observations"]
    for entry in observations:
        entry["outcome"] = "rejected"
        entry["diagnostic"] = "not supported"
        entry.pop("evidence")

    with pytest.raises(PolicyResolutionError, match="accepted none"):
        resolve_policy(report)


def test_policy_resolver_refuses_unknown_preferred_candidate(tmp_path: Path) -> None:
    report = _report(tmp_path)
    observations = report["capabilities"][M.CAPABILITY_LAYOUT]["value"]["observations"]
    preferred = next(entry for entry in observations if entry["candidate"] == "L3")
    preferred.clear()
    preferred.update(
        {
            "family": "module",
            "candidate": "L3",
            "outcome": "timeout",
            "diagnostic": "synthetic timeout",
        }
    )

    with pytest.raises(PolicyResolutionError, match="L3.*unknown outcome timeout"):
        resolve_policy(report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_triple", "x86_64", "target_triple='x86_64'"),
        ("requested_processor", "not-gfx", "requested_processor='not-gfx'"),
        ("data_layout", "L9", "data_layout='L9'"),
        ("named_barrier_address_space", 7, "named_barrier_address_space=7"),
        ("evidence_fingerprint", "bad", "evidence_fingerprint='bad'"),
    ],
)
def test_policy_constructor_rejects_invalid_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    policy = resolve_policy(_report(tmp_path))
    fields = dict(policy.__dict__)
    fields[field] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        CompilerPolicy(**fields)


def test_policy_manifest_rejects_tampering(tmp_path: Path) -> None:
    document = json.loads(resolve_policy(_report(tmp_path)).to_json())
    document["data_layout"] = "L0"

    with pytest.raises(ValueError, match="digest does not match"):
        CompilerPolicy.from_json(json.dumps(document))


def test_policy_manifest_matches_packaged_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(E.__file__).with_name("data") / "policy.schema.json"
    document = json.loads(resolve_policy(_report(tmp_path)).to_json())
    jsonschema.validate(document, json.loads(schema_path.read_text(encoding="utf-8")))
