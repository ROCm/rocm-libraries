# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Contracts for the COMGR-only compiler capability report."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_SOURCE_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_INSTALLED_TOOLS = Path(__file__).resolve().parent
_TOOLS = (
    _SOURCE_TOOLS
    if (_SOURCE_TOOLS / "compiler_capabilities").is_dir()
    else _INSTALLED_TOOLS
)
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compiler_capabilities import _probe_support as support
from compiler_capabilities import probe as P
from rocke.core import llvm_profile

_SHA = "a" * 64


def _identity(_: Path | None) -> dict[str, object]:
    return {
        "llvm_version": [23, 0, 0],
        "version_source": "test",
        "query_library": None,
        "library": "/test/libamd_comgr.so",
        "sha256": _SHA,
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
    report, cache_hit = P.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=_accept_all,
    )
    assert cache_hit is False
    return report


def test_report_is_comgr_only_and_covers_complete_corpora(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["compiler"] == {
        "route": "comgr",
        **_identity(None),
    }
    assert set(report["capabilities"]) == set(P._expected_observations("gfx950"))
    for name, expected in P._expected_observations("gfx950").items():
        result = report["capabilities"][name]
        assert result["value"]["route"] == "comgr_llvm_ir_to_hsaco"
        assert [
            (entry["family"], entry["candidate"])
            for entry in result["value"]["observations"]
        ] == expected
        assert {entry["outcome"] for entry in result["value"]["observations"]} == {
            "accepted"
        }


def test_current_profile_layouts_remain_in_probe_owned_archaeology() -> None:
    assert P._LAYOUT_CANDIDATES["L0"] == llvm_profile._DATALAYOUT_LLVM20
    assert P._LAYOUT_CANDIDATES["L1"] == llvm_profile._DATALAYOUT_LLVM22
    assert P._LAYOUT_CANDIDATES["L3"] == llvm_profile._DATALAYOUT_LLVM23


def test_intrinsic_candidate_corpora_are_exhaustive() -> None:
    assert len(P._buffer_resource_candidates()) == 4
    assert len(P._fp8_packing_candidates()) == 16
    assert len(P._named_barrier_candidates()) == 8
    assert len(P._scaled_mfma_candidates()) == 4
    assert {
        (candidate.family, candidate.candidate)
        for candidate in P._fp8_packing_candidates()
    } == {
        (f"{shape}_{lhs}_{rhs}", packed)
        for shape in ("16x16x32", "32x32x16")
        for lhs, rhs in (
            ("fp8", "fp8"),
            ("bf8", "bf8"),
            ("fp8", "bf8"),
            ("bf8", "fp8"),
        )
        for packed in ("v2i32", "i64")
    }


def test_generic_processor_names_are_explicit_not_digit_derived() -> None:
    assert P._GENERIC_PROCESSORS["gfx950"] == "gfx9-4-generic"
    assert P._GENERIC_PROCESSORS["gfx942"] == "gfx9-4-generic"
    assert P._GENERIC_PROCESSORS["gfx906"] == "gfx9-generic"
    assert P._GENERIC_PROCESSORS["gfx1250"] == "gfx12-generic"
    assert "gfx999" not in P._GENERIC_PROCESSORS


def test_requests_use_comgr_isa_options_and_complete_modules() -> None:
    requests, metadata = P._requests("gfx950")
    assert len(requests) == sum(
        len(entries) for entries in P._expected_observations("gfx950").values()
    )
    assert len(requests) == len(metadata) == len({request.key for request in requests})
    assert all(request.isa.startswith("amdgcn-amd-amdhsa--") for request in requests)
    assert all(
        'target triple = "amdgcn-amd-amdhsa"' in request.ir_text for request in requests
    )
    options = {
        metadata[request.key][2]: request.options
        for request in requests
        if metadata[request.key][0] == P._CAPABILITY_COMPILER_OPTIONS
    }
    assert options["mfma_vgpr_form"] == (
        "-O3",
        "-mllvm",
        "-amdgpu-mfma-vgpr-form",
    )


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

    report, _ = P.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=mixed,
    )
    observations = report["capabilities"][P._CAPABILITY_LAYOUT]["value"]["observations"]
    assert observations[0]["outcome"] == "rejected"
    assert observations[1]["outcome"] == "crash"
    assert "evidence" not in observations[0]


def test_identity_result_pins_every_candidate_worker(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def run(
        requests: list[support.ProbeRequest], **kwargs: object
    ) -> dict[str, support.ProbeResult]:
        seen.update(kwargs)
        return _accept_all(requests)

    P.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path,
        identify=_identity,
        run=run,
    )
    assert seen["comgr_lib"] == Path("/test/libamd_comgr.so")


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

    first, first_hit = P.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=_identity,
        run=counted,
    )
    second, second_hit = P.load_or_probe(
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
    third, third_hit = P.load_or_probe(
        comgr_lib=None,
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        identify=lambda _: other_identity,
        run=counted,
    )
    assert third_hit is False
    assert third["probe"]["fingerprint"] != first["probe"]["fingerprint"]
    assert calls == 2


def test_dependency_free_validator_rejects_missing_duplicate_and_bad_evidence(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    missing = copy.deepcopy(report)
    missing["capabilities"].pop(P._CAPABILITY_LAYOUT)
    with pytest.raises(support.ProbeError, match="capability set"):
        P._validate_report_completeness(missing)

    duplicate = copy.deepcopy(report)
    observations = duplicate["capabilities"][P._CAPABILITY_LAYOUT]["value"][
        "observations"
    ]
    observations[1] = copy.deepcopy(observations[0])
    with pytest.raises(support.ProbeError, match="complete ordered corpus"):
        P._validate_report_completeness(duplicate)

    bad_evidence = copy.deepcopy(report)
    del bad_evidence["capabilities"][P._CAPABILITY_LAYOUT]["value"]["observations"][0][
        "evidence"
    ]
    with pytest.raises(support.ProbeError, match="lacks evidence"):
        P._validate_report_completeness(bad_evidence)


def test_example_and_generated_report_pass_dependency_free_contract(
    tmp_path: Path,
) -> None:
    example = json.loads(
        (_TOOLS / "compiler_capabilities" / "example.capabilities.json").read_text()
    )
    P._validate_report_completeness(example)
    P._validate_report_completeness(_report(tmp_path))


def test_schema_accepts_generated_report_when_jsonschema_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jsonschema")
    P._validate_report(_report(tmp_path))


def test_invalid_processor_and_duplicate_request_keys_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="processor='not-a-target'.*gfx"):
        P.load_or_probe(
            comgr_lib=None,
            processor="not-a-target",
            cache_dir=tmp_path,
            identify=_identity,
            run=_accept_all,
        )
    request = support.ProbeRequest("same", "module", "isa")
    with pytest.raises(ValueError, match="keys must be unique"):
        support.probe_all([request, request], comgr_lib=None)


def test_main_writes_report_without_llvm_tool_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capabilities.json"
    report = _report(tmp_path / "source")
    seen: dict[str, object] = {}

    def fake(**kwargs: object) -> tuple[dict[str, object], bool]:
        seen.update(kwargs)
        return report, False

    monkeypatch.setattr(P, "load_or_probe", fake)
    assert (
        P.main(
            [
                "--comgr-lib",
                str(tmp_path / "lib.so"),
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )
    assert seen["comgr_lib"] == tmp_path / "lib.so"
    assert json.loads(output.read_text()) == report
