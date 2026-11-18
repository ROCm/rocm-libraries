# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Contracts for the COMGR-only compiler capability report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import rocke.compiler
from rocke.compiler import _support as support
from rocke.compiler import cli
from rocke.compiler import corpus as C
from rocke.compiler import evidence as E
from rocke.compiler import model as M
from rocke.compiler.policy import CompilerPolicy, resolve_policy

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


def test_public_compiler_exports_resolve_and_unknown_names_fail() -> None:
    assert all(
        getattr(rocke.compiler, name) is not None for name in rocke.compiler.__all__
    )
    unknown = "not_an_export"
    with pytest.raises(AttributeError, match="no attribute 'not_an_export'"):
        getattr(rocke.compiler, unknown)


def test_default_cache_dir_is_persistent_and_platform_appropriate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(E.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert E.default_cache_dir() == tmp_path / "xdg" / "rocke" / "compiler_capabilities"

    monkeypatch.setattr(E.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert (
        E.default_cache_dir() == tmp_path / "local" / "rocke" / "compiler_capabilities"
    )


def test_report_is_comgr_only_and_covers_complete_corpora(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["compiler"] == {
        "route": "comgr",
        **_identity(None),
    }
    assert set(report["capabilities"]) == set(C.expected_observations("gfx950"))
    for name, expected in C.expected_observations("gfx950").items():
        result = report["capabilities"][name]
        assert result["value"]["route"] == "comgr_llvm_ir_to_hsaco"
        assert [
            (entry["family"], entry["candidate"])
            for entry in result["value"]["observations"]
        ] == expected
        assert {entry["outcome"] for entry in result["value"]["observations"]} == {
            "accepted"
        }


def test_rejection_and_crash_are_observations_not_inference(tmp_path: Path) -> None:
    def mixed(
        requests: list[support.ProbeRequest], **_: object
    ) -> dict[str, support.ProbeResult]:
        results = _accept_all(requests)
        results[requests[0].key] = support.ProbeResult(
            requests[0].key, "rejected", "synthetic COMGR rejection"
        )
        results[requests[1].key] = support.ProbeResult(
            requests[1].key, "crash", "synthetic isolated crash"
        )
        return results

    report, _ = E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=mixed,
    )
    observations = report["capabilities"][M.CAPABILITY_LAYOUT]["value"]["observations"]
    assert observations[0]["outcome"] == "rejected"
    assert observations[1]["outcome"] == "crash"
    assert "evidence" not in observations[0]


def test_timeout_report_is_not_cached(tmp_path: Path) -> None:
    calls = 0

    def timed_out(
        requests: list[support.ProbeRequest], **_: object
    ) -> dict[str, support.ProbeResult]:
        nonlocal calls
        calls += 1
        results = _accept_all(requests)
        results[requests[0].key] = support.ProbeResult(
            requests[0].key, "timeout", "synthetic timeout"
        )
        return results

    for _ in range(2):
        _, cache_hit = E.load_or_probe(
            comgr_lib=None,
            processor="gfx950",
            cache_dir=tmp_path / "cache",
            identify=_identity,
            run=timed_out,
        )
        assert cache_hit is False
    assert calls == 2


def test_identity_result_pins_every_candidate_worker(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def run(
        requests: list[support.ProbeRequest], **kwargs: object
    ) -> dict[str, support.ProbeResult]:
        seen.update(kwargs)
        return _accept_all(requests)

    E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path,
        identify=_identity,
        run=run,
    )
    assert seen["comgr_lib"] == Path("/test/libamd_comgr.so")


def test_generic_processor_fallback_reprobes_behaviors_with_effective_isa(
    tmp_path: Path,
) -> None:
    calls: list[list[support.ProbeRequest]] = []

    def run(
        requests: list[support.ProbeRequest], **_: object
    ) -> dict[str, support.ProbeResult]:
        calls.append(requests)
        results = _accept_all(requests)
        if len(calls) == 1:
            requested = next(
                request
                for request in requests
                if request.key.endswith("\0gfx950\0requested")
            )
            results[requested.key] = support.ProbeResult(
                requested.key, "rejected", "unknown processor"
            )
        return results

    report, _ = E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path,
        identify=_identity,
        run=run,
    )

    assert report["probe"]["effective_processor"] == "gfx9-4-generic"
    assert len(calls) == 2
    assert all(
        request.isa == "amdgcn-amd-amdhsa--gfx9-4-generic" for request in calls[1]
    )
    assert all(M.CAPABILITY_PROCESSOR_NAMES not in request.key for request in calls[1])


def test_cache_is_disabled_when_dependency_identity_is_incomplete(
    tmp_path: Path,
) -> None:
    calls = 0
    identity = dict(_identity(None), dependency_identity_complete=False)

    def counted(
        requests: list[support.ProbeRequest], **kwargs: object
    ) -> dict[str, support.ProbeResult]:
        nonlocal calls
        calls += 1
        return _accept_all(requests, **kwargs)

    for _ in range(2):
        _, cache_hit = E.load_or_probe(
            comgr_lib=None,
            processor="gfx950",
            cache_dir=tmp_path,
            identify=lambda _: identity,
            run=counted,
        )
        assert cache_hit is False
    assert calls == 2


def test_cache_fingerprint_covers_comgr_probe_schema_and_context(
    tmp_path: Path,
) -> None:
    calls = 0

    def counted(
        requests: list[support.ProbeRequest], **kwargs: object
    ) -> dict[str, support.ProbeResult]:
        nonlocal calls
        calls += 1
        return _accept_all(requests, **kwargs)

    first, first_hit = E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=counted,
    )
    second, second_hit = E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=counted,
    )
    assert first_hit is False
    assert second_hit is True
    assert second == first
    assert calls == 1

    other_identity = dict(_identity(None), sha256="b" * 64)
    third, third_hit = E.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=lambda _: other_identity,
        run=counted,
    )
    assert third_hit is False
    assert third["probe"]["fingerprint"] != first["probe"]["fingerprint"]
    assert calls == 2


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("probe", "implementation_sha256", "implementation_sha256"),
        ("probe", "schema_sha256", "schema_sha256"),
        ("probe", "fingerprint", "fingerprint"),
        ("compiler", "sha256", "compiler.sha256"),
    ],
)
def test_dependency_free_validator_rejects_malformed_provenance_digests(
    tmp_path: Path, section: str, field: str, message: str
) -> None:
    report = _report(tmp_path)
    report[section][field] = ""
    with pytest.raises(support.ProbeError, match=message):
        E.validate_report_structure(report)


def test_dependency_free_validator_rejects_bad_envelope_provenance(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    report["compiler"]["route"] = "clang"
    with pytest.raises(support.ProbeError, match="compiler.route"):
        E.validate_report_structure(report)

    report = _report(tmp_path / "library")
    report["compiler"]["library"] = ""
    with pytest.raises(support.ProbeError, match="compiler.library"):
        E.validate_report_structure(report)

    report = _report(tmp_path / "effective")
    report["probe"]["effective_processor"] = "gfx12-generic"
    with pytest.raises(support.ProbeError, match="effective_processor"):
        E.validate_report_structure(report)


def test_dependency_free_validator_rejects_missing_duplicate_and_bad_evidence(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    missing = copy.deepcopy(report)
    missing["capabilities"].pop(M.CAPABILITY_LAYOUT)
    with pytest.raises(support.ProbeError, match="capability set"):
        E.validate_report_structure(missing)

    duplicate = copy.deepcopy(report)
    observations = duplicate["capabilities"][M.CAPABILITY_LAYOUT]["value"][
        "observations"
    ]
    observations[1] = copy.deepcopy(observations[0])
    with pytest.raises(support.ProbeError, match="complete ordered corpus"):
        E.validate_report_structure(duplicate)

    bad_evidence = copy.deepcopy(report)
    del bad_evidence["capabilities"][M.CAPABILITY_LAYOUT]["value"]["observations"][0][
        "evidence"
    ]
    with pytest.raises(support.ProbeError, match="lacks evidence"):
        E.validate_report_structure(bad_evidence)


def test_generated_report_passes_dependency_free_contract(tmp_path: Path) -> None:
    E.validate_report_structure(_report(tmp_path))


def test_schema_accepts_generated_report_when_jsonschema_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jsonschema")
    E.validate_report(_report(tmp_path))


def test_invalid_processor_and_duplicate_request_keys_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="processor='not-a-target'.*gfx"):
        E.load_or_probe(
            comgr_lib=None,
            processor="not-a-target",
            cache_dir=tmp_path,
            identify=_identity,
            run=_accept_all,
        )
    request = support.ProbeRequest("same", "module", "isa")
    with pytest.raises(ValueError, match="keys must be unique"):
        support.probe_all([request, request], comgr_lib=None)


def test_cli_requires_an_explicit_processor() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--quiet"])


def test_main_writes_report_without_llvm_tool_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_with_policy = tmp_path / "capabilities-with-policy.json"
    output_without_policy = tmp_path / "capabilities-without-policy.json"
    policy_output = tmp_path / "compiler-policy.json"
    report = _report(tmp_path / "source")
    seen: dict[str, object] = {}

    def fake(**kwargs: object) -> tuple[dict[str, object], bool]:
        seen.update(kwargs)
        return report, False

    monkeypatch.setattr(cli, "load_or_probe", fake)
    common = [
        "--comgr-lib",
        str(tmp_path / "lib.so"),
        "--processor",
        "gfx950",
        "--quiet",
    ]
    with_policy = cli.main(
        [
            *common,
            "--output",
            str(output_with_policy),
            "--policy-output",
            str(policy_output),
        ]
    )
    without_policy = cli.main([*common, "--output", str(output_without_policy)])

    assert with_policy == without_policy == 0
    assert seen["comgr_lib"] == tmp_path / "lib.so"
    assert output_with_policy.read_bytes() == output_without_policy.read_bytes()
    assert json.loads(output_with_policy.read_text()) == report
    assert CompilerPolicy.from_json(policy_output.read_text()) == resolve_policy(report)
