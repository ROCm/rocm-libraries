# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the second structural compiler-capability probe slice."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

_PLATFORM = Path(__file__).resolve().parents[2]
_TOOLS = _PLATFORM / "tools"
_CAPABILITIES_DIR = _TOOLS / "compiler_capabilities"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compiler_capabilities import _probe_support as support
from compiler_capabilities import probe as P


class FakeStructuralCompiler:
    """Accept deterministic structural candidates without invoking LLVM."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(
        self, argv: list[str], input_text: str | None
    ) -> support.CommandResult:
        self.calls.append((argv, input_text))
        tool = Path(argv[0]).name
        if argv[1:] == ["--version"] and tool == "clang":
            return support.CommandResult(
                0,
                "AMD clang version development "
                "0123456789abcdef0123456789abcdef01234567\n",
                "",
            )
        if argv[1:] == ["--version"] and tool == "opt":
            return support.CommandResult(0, "AMD LLVM development build\n", "")
        if tool == "clang" and "-ast-dump" in argv:
            return support.CommandResult(
                1,
                "|-FunctionDecl implicit used "
                "__builtin_amdgcn_mfma_f32_32x32x8f16 "
                "'float (__attribute__((ext_vector_type(4))) _Float16)'\n",
                "too few arguments\n",
            )
        if tool == "clang" and "ir" in argv:
            match = re.search(r'^target datalayout = "([^"]+)"', input_text or "")
            if match is not None:
                return support.CommandResult(
                    0,
                    f'target datalayout = "{match.group(1)}"\n'
                    'target triple = "amdgcn-amd-amdhsa"\n',
                    "",
                )
            return support.CommandResult(0, ".text\nprobe:\n  s_endpgm\n", "")
        if tool == "clang":
            return support.CommandResult(
                0,
                'target datalayout = "e-m:e-p:64:64-p8:128:128:128:48"\n'
                'target triple = "amdgcn-amd-amdhsa"\n'
                "define dso_local amdgpu_kernel void @probe() { ret void }\n"
                "!llvm.module.flags = !{!0}\n"
                '!0 = !{i32 1, !"amdgpu_code_object_version", i32 600}\n',
                "",
            )
        if tool == "opt":
            module = input_text or ""
            if "make.buffer.rsrc.p1" in module and ", i64" in module:
                return support.CommandResult(
                    1, "", "Intrinsic has incorrect argument type!\n"
                )
            declaration = next(
                line for line in module.splitlines() if line.startswith("declare ")
            )
            return support.CommandResult(0, declaration + " #0\n", "")
        raise AssertionError(argv)


def _tool_dir(tmp_path: Path) -> Path:
    llvm_bin = tmp_path / "llvm-bin"
    llvm_bin.mkdir()
    for name in ("clang", "opt"):
        (llvm_bin / name).write_bytes(f"fake {name}".encode())
    return llvm_bin


def test_schema_accepts_example_and_generated_report(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((_CAPABILITIES_DIR / "capabilities.schema.json").read_text())
    example = json.loads((_CAPABILITIES_DIR / "example.capabilities.json").read_text())
    report, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx942",
        cache_dir=tmp_path / "cache",
        run=FakeStructuralCompiler(),
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=example, schema=schema)
    jsonschema.validate(instance=report, schema=schema)
    expected_implementation = hashlib.sha256(
        (_CAPABILITIES_DIR / "probe.py").read_bytes()
        + (_CAPABILITIES_DIR / "_probe_support.py").read_bytes()
    ).hexdigest()
    assert report["probe"]["implementation_sha256"] == expected_implementation
    assert (
        report["probe"]["schema_sha256"]
        == hashlib.sha256(
            (_CAPABILITIES_DIR / "capabilities.schema.json").read_bytes()
        ).hexdigest()
    )

    missing_buffer_abi = copy.deepcopy(report)
    del missing_buffer_abi["capabilities"][P._CAPABILITY_BUFFER_RESOURCE]["value"][
        "accepted"
    ][0]["canonical_abi"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=missing_buffer_abi, schema=schema)


def test_structural_candidate_corpora_are_exhaustive() -> None:
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


def test_behavior_probe_corpora_cover_known_compiler_variations(
    tmp_path: Path,
) -> None:
    report, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx942",
        cache_dir=tmp_path / "cache",
        run=FakeStructuralCompiler(),
    )
    capabilities = report["capabilities"]

    assert capabilities[P._CAPABILITY_FRONTEND_FP16]["value"]["observations"][0][
        "evidence"
    ]["prototype"].endswith("_Float16)")
    assert [
        entry["family"]
        for entry in capabilities[P._CAPABILITY_PROCESSOR_NAMES]["value"][
            "observations"
        ]
    ] == ["gfx942", "gfx9-4-generic"]
    assert [
        entry["candidate"]
        for entry in capabilities[P._CAPABILITY_OPT_LEVELS]["value"]["observations"]
    ] == ["O0", "O1", "O2", "O3"]
    assert (
        len(capabilities[P._CAPABILITY_COMPILER_OPTIONS]["value"]["observations"]) == 3
    )
    assert (
        len(capabilities[P._CAPABILITY_FUNCTION_ATTRIBUTES]["value"]["observations"])
        == 3
    )


def test_unknown_intrinsic_roundtrip_is_rejected() -> None:
    candidate = P._buffer_resource_candidates()[0]

    def unknown(argv: list[str], input_text: str | None) -> support.CommandResult:
        del argv, input_text
        return support.CommandResult(
            0,
            "; Unknown intrinsic\n"
            f"declare {candidate.result_type} @{candidate.intrinsic}("
            + ", ".join(candidate.declaration_args)
            + ") #0\n",
            "",
        )

    result = P._probe_intrinsic_candidates(Path("opt"), (candidate,), unknown)

    assert result["value"]["accepted"] == []
    assert result["value"]["rejected"][0]["candidate"] == candidate.candidate

    def external(argv: list[str], input_text: str | None) -> support.CommandResult:
        del argv, input_text
        return support.CommandResult(
            0,
            f"declare {candidate.result_type} @{candidate.intrinsic}("
            + ", ".join(candidate.declaration_args)
            + ")\n",
            "",
        )

    external_result = P._probe_intrinsic_candidates(Path("opt"), (candidate,), external)
    assert external_result["value"]["accepted"] == []
    assert external_result["value"]["rejected"][0]["candidate"] == candidate.candidate


def test_crashed_intrinsic_and_layout_probes_are_errors() -> None:
    def crash(argv: list[str], input_text: str | None) -> support.CommandResult:
        del argv, input_text
        return support.CommandResult(-11, "", "")

    intrinsic = P._probe_intrinsic_candidates(
        Path("opt"), (P._buffer_resource_candidates()[0],), crash
    )
    layout = P._probe_layout_policy(Path("clang"), "gfx942", crash)

    assert intrinsic == {
        "status": "error",
        "diagnostic": "p1_i32: compiler process crashed",
    }
    assert layout == {"status": "error", "diagnostic": "L0: compiler process crashed"}

    def fatal(argv: list[str], input_text: str | None) -> support.CommandResult:
        del argv, input_text
        return support.CommandResult(1, "", "LLVM ERROR: synthetic backend crash")

    fatal_intrinsic = P._probe_intrinsic_candidates(
        Path("opt"), (P._buffer_resource_candidates()[0],), fatal
    )
    assert fatal_intrinsic["status"] == "error"
    assert "crashed" in fatal_intrinsic["diagnostic"]


def test_frontend_partial_output_does_not_hide_timeout() -> None:
    builtin = "__builtin_amdgcn_mfma_f32_32x32x8f16"

    def timeout(argv: list[str], input_text: str | None) -> support.CommandResult:
        del argv, input_text
        return support.CommandResult(
            None,
            f"FunctionDecl implicit used {builtin} 'illustrative prototype'\n",
            "",
        )

    result = P._probe_frontend_fp16_abi(Path("clang"), "gfx942", timeout)

    assert result["value"]["observations"][0]["outcome"] == "timeout"
    assert "evidence" not in result["value"]["observations"][0]

    def fatal(argv: list[str], input_text: str | None) -> support.CommandResult:
        del argv, input_text
        return support.CommandResult(
            1,
            f"FunctionDecl implicit used {builtin} 'illustrative prototype'\n",
            "PLEASE submit a bug report and include the crash backtrace",
        )

    fatal_result = P._probe_frontend_fp16_abi(Path("clang"), "gfx942", fatal)
    assert fatal_result["value"]["observations"][0]["outcome"] == "crash"
    assert "evidence" not in fatal_result["value"]["observations"][0]


def test_completeness_validator_rejects_missing_candidate(tmp_path: Path) -> None:
    report, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx942",
        cache_dir=tmp_path / "cache",
        run=FakeStructuralCompiler(),
    )
    incomplete = copy.deepcopy(report)
    incomplete["capabilities"][P._CAPABILITY_BUFFER_RESOURCE]["value"]["rejected"].pop()

    with pytest.raises(support.ProbeError, match="complete candidate corpus"):
        P._validate_report_completeness(incomplete)


def test_module_contract_records_preamble_and_each_layout_policy(
    tmp_path: Path,
) -> None:
    report, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx942",
        cache_dir=tmp_path / "cache",
        run=FakeStructuralCompiler(),
    )
    capabilities = report["capabilities"]

    assert capabilities[P._CAPABILITY_PREAMBLE] == {
        "status": "known",
        "value": {
            "target_triple": "amdgcn-amd-amdhsa",
            "kernel_calling_convention": "amdgpu_kernel",
            "module_flags": ['!{i32 1, !"amdgpu_code_object_version", i32 600}'],
        },
    }
    policies = capabilities[P._CAPABILITY_LAYOUT_POLICY]["value"]["candidates"]
    assert [entry["candidate"] for entry in policies] == ["L0", "L1", "L2", "L3"]
    assert {entry["outcome"] for entry in policies} == {"accepted"}


def test_cache_reuses_complete_structural_report(tmp_path: Path) -> None:
    compiler = FakeStructuralCompiler()
    llvm_bin = _tool_dir(tmp_path)
    cache_dir = tmp_path / "cache"

    first, first_hit = P.load_or_probe(
        llvm_bin=llvm_bin,
        processor="gfx942",
        cache_dir=cache_dir,
        run=compiler,
    )
    first_calls = len(compiler.calls)
    second, second_hit = P.load_or_probe(
        llvm_bin=llvm_bin,
        processor="gfx942",
        cache_dir=cache_dir,
        run=compiler,
    )

    assert first_hit is False
    assert second_hit is True
    assert second == first
    assert len(compiler.calls) > first_calls  # identities still guard cache selection
    assert len(compiler.calls) - first_calls < first_calls


@pytest.mark.parametrize(
    ("processor", "generic"),
    [
        ("gfx942", "gfx9-4-generic"),
        ("gfx950", "gfx9-5-generic"),
        ("gfx1151", "gfx11-generic"),
        ("gfx1201", "gfx12-generic"),
    ],
)
def test_generic_processor_candidates(processor: str, generic: str) -> None:
    assert P._generic_processor_name(processor) == generic


def test_python_environment_discovers_packaged_llvm_tools(tmp_path: Path) -> None:
    llvm_bin = (
        tmp_path
        / "lib"
        / "python3.14"
        / "site-packages"
        / "_rocm_sdk_core"
        / "lib"
        / "llvm"
        / "bin"
    )
    llvm_bin.mkdir(parents=True)
    for name in ("clang", "opt"):
        (llvm_bin / name).write_bytes(name.encode())

    assert P._llvm_bin_from_python_env(tmp_path) == llvm_bin.resolve()


def test_python_environment_missing_tools_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(support.ProbeError, match="install rocm-sdk-core.*--llvm-bin"):
        P._llvm_bin_from_python_env(tmp_path)


def test_human_summary_names_compact_observed_contracts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx942",
        cache_dir=tmp_path / "cache",
        run=FakeStructuralCompiler(),
    )

    P._print_summary(report, False, Path("capabilities.json"))

    summary = capsys.readouterr().out
    assert "probe implementation:" in summary
    assert "schema:" in summary
    assert "module layout: L2\n" in summary
    assert "cache: miss\n" in summary
    assert "explicit layouts: L0=accepted" in summary
    assert "output: capabilities.json\n" in summary


def test_main_writes_known_report_and_explicit_tools_override_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capabilities.json"
    explicit = tmp_path / "explicit-tools"
    seen: dict[str, object] = {}
    report = {
        "probe": {"corpus": "test"},
        "compiler": {"source_revision": None},
        "capabilities": {"test.capability": {"status": "known", "value": {}}},
    }

    def load_or_probe(**kwargs: object) -> tuple[dict[str, object], bool]:
        seen.update(kwargs)
        return report, False

    monkeypatch.setenv("ROCKE_LLVM_BIN", str(tmp_path / "environment-tools"))
    monkeypatch.setattr(P, "load_or_probe", load_or_probe)

    assert (
        P.main(
            [
                "--llvm-bin",
                str(explicit),
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )
    assert seen["llvm_bin"] == explicit
    assert json.loads(output.read_text()) == report


def test_main_setup_error_returns_two_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capabilities.json"

    def fail(**kwargs: object) -> tuple[dict[str, object], bool]:
        del kwargs
        raise support.ProbeError("missing compiler")

    monkeypatch.setattr(P, "load_or_probe", fail)

    assert P.main(["--output", str(output), "--quiet"]) == 2
    assert not output.exists()


def test_scaled_mfma_candidates_pin_nine_and_eleven_argument_forms() -> None:
    by_name = {
        (candidate.family, candidate.candidate): candidate
        for candidate in P._scaled_mfma_candidates()
    }
    for shape in ("16x16x128", "32x32x64"):
        assert len(by_name[(shape, "current_9")].call_args) == 9
        assert len(by_name[(shape, "legacy_11")].call_args) == 11
