# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused tests for the standalone compiler-capability demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PLATFORM = Path(__file__).resolve().parents[2]
_TOOLS = _PLATFORM / "tools"
_CAPABILITIES_DIR = _TOOLS / "compiler_capabilities"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compiler_capabilities import probe as P


class FakeCompiler:
    """Deterministic clang/opt behavior for the probe's observable contracts."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv: list[str], input_text: str | None) -> P.CommandResult:
        self.calls.append((argv, input_text))
        tool = Path(argv[0]).name
        if argv[1:] == ["--version"] and tool == "clang":
            return P.CommandResult(
                0,
                "AMD clang version development "
                "0123456789abcdef0123456789abcdef01234567"
                "+PATCHED:89abcdef0123456789abcdef0123456789abcdef\n",
                "",
            )
        if argv[1:] == ["--version"] and tool == "opt":
            return P.CommandResult(0, "AMD LLVM development build\n", "")
        if tool == "clang":
            return P.CommandResult(
                0,
                'target datalayout = "e-m:e-p:64:64-p8:128:128:128:48"\n'
                'target triple = "amdgcn-amd-amdhsa"\n',
                "",
            )
        if tool == "opt" and "make.buffer.rsrc.p8.p1" in (input_text or ""):
            return P.CommandResult(1, "", "Intrinsic has incorrect argument type!\n")
        if tool == "opt":
            return P.CommandResult(
                0,
                "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
                "ptr addrspace(1) readnone, i16, i32, i32) #0\n",
                "",
            )
        raise AssertionError(argv)


def _tool_dir(tmp_path: Path) -> Path:
    llvm_bin = tmp_path / "llvm-bin"
    llvm_bin.mkdir()
    for name in ("clang", "opt"):
        (llvm_bin / name).write_bytes(f"fake {name}".encode())
    return llvm_bin


def test_schema_accepts_example_and_generated_reports_when_available(
    tmp_path: Path,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((_CAPABILITIES_DIR / "capabilities-v1.schema.json").read_text())
    example = json.loads((_CAPABILITIES_DIR / "example.capabilities.json").read_text())
    generated, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        run=FakeCompiler(),
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=example, schema=schema)
    jsonschema.validate(instance=generated, schema=schema)


def test_human_summary_explains_observed_behavior(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report, _ = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        run=FakeCompiler(),
    )

    P._print_capability_summary(report["capabilities"])

    assert capsys.readouterr().out == (
        "AMDGPU module layout:\n"
        "  status: known\n"
        "  ELF mangling: e\n"
        "  address space 8: 128:128:128:48\n"
        "  additional pointer address spaces: none\n"
        "Buffer-resource intrinsic contract:\n"
        "  status: known\n"
        "  accepted canonical ABIs: basic pointer / 32-bit count\n"
        "  rejected candidates: indexed spelling / 64-bit count\n"
    )


def test_source_revisions_preserve_primary_and_patched_hashes() -> None:
    primary = "0123456789abcdef0123456789abcdef01234567"
    patched = "89abcdef0123456789abcdef0123456789abcdef"

    assert P._source_revisions(
        f"AMD clang version development ({primary}+PATCHED:{patched})"
    ) == (primary, patched)
    assert P._source_revisions("AMD clang version development") == (None, None)


def test_data_layout_parser_keeps_exact_text_and_readable_fields() -> None:
    layout = "e-m:e-p:64:64-p8:128:128:128:48"

    assert P._data_layout_value(f'target datalayout = "{layout}"\n') == {
        "text": layout,
        "elf_mangling": "e",
        "address_space_8": [128, 128, 128, 48],
        "additional_pointer_address_spaces": [],
    }


def test_data_layout_parser_refuses_output_without_a_layout() -> None:
    with pytest.raises(P.ProbeError, match="no target datalayout"):
        P._data_layout_value('target triple = "amdgcn-amd-amdhsa"\n')


def test_buffer_resource_report_distinguishes_candidate_from_canonical_abi() -> None:
    def canonicalizing_opt(argv: list[str], input_text: str | None) -> P.CommandResult:
        del argv
        if "make.buffer.rsrc.p8.p1" in (input_text or ""):
            return P.CommandResult(
                0,
                "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64("
                "ptr addrspace(1) readnone, i16, i64, i32) #0\n",
                "",
            )
        return P.CommandResult(
            0,
            "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32("
            "ptr addrspace(1) readnone, i16, i32, i32) #0\n",
            "",
        )

    result = P._probe_buffer_resource_abis(Path("opt"), canonicalizing_opt)

    assert result == {
        "status": "known",
        "value": {
            "accepted": [
                {
                    "candidate": "legacy_p1_i32",
                    "canonical_abi": "p8_p1_i32",
                    "canonical_intrinsic": "llvm.amdgcn.make.buffer.rsrc.p8.p1.i32",
                    "canonical_declaration": (
                        "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i32("
                        "ptr addrspace(1) readnone, i16, i32, i32)"
                    ),
                },
                {
                    "candidate": "p8_p1_i64",
                    "canonical_abi": "p8_p1_i64",
                    "canonical_intrinsic": "llvm.amdgcn.make.buffer.rsrc.p8.p1.i64",
                    "canonical_declaration": (
                        "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64("
                        "ptr addrspace(1) readnone, i16, i64, i32)"
                    ),
                },
            ],
            "rejected": [],
        },
    }


def test_probe_reports_behavior_instead_of_interpreting_version_label(
    tmp_path: Path,
) -> None:
    compiler = FakeCompiler()

    report, cache_hit = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        run=compiler,
    )

    assert cache_hit is False
    assert report["compiler"]["source_revision"] == (
        "0123456789abcdef0123456789abcdef01234567"
    )
    assert report["compiler"]["patched_revision"] == (
        "89abcdef0123456789abcdef0123456789abcdef"
    )
    capabilities = report["capabilities"]
    assert capabilities[P._CAPABILITY_DATA_LAYOUT]["value"]["address_space_8"] == [
        128,
        128,
        128,
        48,
    ]
    assert (
        capabilities[P._CAPABILITY_DATA_LAYOUT]["value"][
            "additional_pointer_address_spaces"
        ]
        == []
    )
    assert capabilities[P._CAPABILITY_BUFFER_RESOURCE]["value"] == {
        "accepted": [
            {
                "candidate": "legacy_p1_i32",
                "canonical_abi": "p1_i32",
                "canonical_intrinsic": "llvm.amdgcn.make.buffer.rsrc.p1",
                "canonical_declaration": (
                    "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
                    "ptr addrspace(1) readnone, i16, i32, i32)"
                ),
            }
        ],
        "rejected": [
            {
                "candidate": "p8_p1_i64",
                "diagnostic": "Intrinsic has incorrect argument type!",
            }
        ],
    }


def test_matching_cache_is_byte_repeatable_and_skips_compiler_commands(
    tmp_path: Path,
) -> None:
    compiler = FakeCompiler()
    llvm_bin = _tool_dir(tmp_path)
    cache_dir = tmp_path / "cache"

    first, first_hit = P.load_or_probe(
        llvm_bin=llvm_bin,
        processor="gfx950",
        cache_dir=cache_dir,
        run=compiler,
    )
    first_capability_calls = sum(
        argv[1:] != ["--version"] for argv, _ in compiler.calls
    )
    second, second_hit = P.load_or_probe(
        llvm_bin=llvm_bin,
        processor="gfx950",
        cache_dir=cache_dir,
        run=compiler,
    )

    assert first_hit is False
    assert second_hit is True
    assert second == first
    assert (
        sum(argv[1:] != ["--version"] for argv, _ in compiler.calls)
        == first_capability_calls
    )
    cached = next(cache_dir.glob("*.json"))
    assert json.loads(cached.read_text()) == first


def test_probe_error_is_reported_explained_and_not_cached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    compiler = FakeCompiler()

    def time_out_new_abi(argv: list[str], input_text: str | None) -> P.CommandResult:
        if "make.buffer.rsrc.p8.p1" in (input_text or ""):
            return P.CommandResult(None, "", "")
        return compiler(argv, input_text)

    report, cache_hit = P.load_or_probe(
        llvm_bin=_tool_dir(tmp_path),
        processor="gfx950",
        cache_dir=tmp_path / "cache",
        run=time_out_new_abi,
    )

    result = report["capabilities"][P._CAPABILITY_BUFFER_RESOURCE]
    assert result == {
        "status": "error",
        "diagnostic": "p8_p1_i64: compiler command exceeded 15 seconds",
    }
    assert cache_hit is False
    assert not (tmp_path / "cache").exists()

    P._print_capability_summary(report["capabilities"])
    summary = capsys.readouterr().out
    assert "Buffer-resource intrinsic contract:\n  status: error\n" in summary
    assert "  diagnostic: p8_p1_i64: compiler command exceeded 15 seconds\n" in summary


def test_refresh_reruns_a_matching_cache(tmp_path: Path) -> None:
    compiler = FakeCompiler()
    llvm_bin = _tool_dir(tmp_path)
    cache_dir = tmp_path / "cache"
    P.load_or_probe(
        llvm_bin=llvm_bin,
        processor="gfx950",
        cache_dir=cache_dir,
        run=compiler,
    )
    first_call_count = len(compiler.calls)

    _, cache_hit = P.load_or_probe(
        llvm_bin=llvm_bin,
        processor="gfx950",
        cache_dir=cache_dir,
        refresh=True,
        run=compiler,
    )

    assert cache_hit is False
    assert len(compiler.calls) > first_call_count


def test_cache_fingerprint_changes_with_every_probe_input() -> None:
    def fingerprint(
        clang: Path = Path("clang"),
        opt: Path = Path("opt"),
        clang_sha256: str = "a" * 64,
        opt_sha256: str = "b" * 64,
        clang_identity: str = "clang identity",
        opt_identity: str = "opt identity",
        processor: str = "gfx950",
    ) -> str:
        return P._fingerprint(
            clang,
            opt,
            clang_sha256,
            opt_sha256,
            clang_identity,
            opt_identity,
            processor,
        )

    baseline = fingerprint()
    assert fingerprint(clang=Path("other-clang")) != baseline
    assert fingerprint(opt=Path("other-opt")) != baseline
    assert fingerprint(clang_sha256="c" * 64) != baseline
    assert fingerprint(opt_sha256="c" * 64) != baseline
    assert fingerprint(clang_identity="different clang") != baseline
    assert fingerprint(opt_identity="different opt") != baseline
    assert fingerprint(processor="gfx942") != baseline


def test_explicit_tool_directory_reports_a_missing_tool(tmp_path: Path) -> None:
    with pytest.raises(P.ProbeError, match="opt not found"):
        P._tool_path("opt", tmp_path)


def test_probe_rejects_a_non_gfx_processor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="processor='not-a-target'.*gfx"):
        P.load_or_probe(
            llvm_bin=_tool_dir(tmp_path),
            processor="not-a-target",
            cache_dir=tmp_path / "cache",
            run=FakeCompiler(),
        )
