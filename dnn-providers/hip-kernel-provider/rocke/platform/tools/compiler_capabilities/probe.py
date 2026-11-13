#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe a selected AMD LLVM compiler and write ``capabilities.json``.

This is a demonstration tool, not a lowering input. It records enough compiler
provenance to compare focused behavior probes across local compiler builds. The
probe results, rather than the compiler's reported LLVM major, are authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_SCHEMA_VERSION = 1
_PROBE_VERSION = 1
_TARGET_TRIPLE = "amdgcn-amd-amdhsa"
_DEFAULT_PROCESSOR = "gfx950"
_REPOSITORY = "https://github.com/ROCm/llvm-project"
_TIMEOUT_SECONDS = 15
_CAPABILITY_DATA_LAYOUT = "amdgpu.ir.data_layout"
_CAPABILITY_BUFFER_RESOURCE = "amdgpu.intrinsic.make_buffer_resource.abis"
_SHA_RE = re.compile(r"\b[0-9a-fA-F]{40}\b")
_PATCHED_RE = re.compile(r"\+PATCHED:([0-9a-fA-F]{40})\b")
_PROCESSOR_RE = re.compile(r"^gfx[0-9a-z-]+$")
_DATALAYOUT_RE = re.compile(r'^target datalayout = "([^"]+)"$', re.MULTILINE)
_DECL_RE = re.compile(
    r"^(declare .+ @(?P<name>llvm\.amdgcn\.make\.buffer\.rsrc\.[^(]+)"
    r"\([^\n]+\)) #\d+$",
    re.MULTILINE,
)


class ProbeError(RuntimeError):
    """The requested compiler could not be identified or probed."""


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one bounded compiler invocation."""

    returncode: int | None
    stdout: str
    stderr: str

    @property
    def timed_out(self) -> bool:
        return self.returncode is None


@dataclass(frozen=True)
class _AbiCandidate:
    """One input spelling submitted to the compiler, not a selected rocKE ABI."""

    candidate: str
    intrinsic: str
    count_type: str

    def module(self) -> str:
        """Minimal module using this exact intrinsic declaration and call shape."""
        return (
            f'target triple = "{_TARGET_TRIPLE}"\n'
            "define amdgpu_kernel void @probe(ptr addrspace(1) %p) {\n"
            f"  %resource = call ptr addrspace(8) @{self.intrinsic}("
            f"ptr addrspace(1) %p, i16 0, {self.count_type} 4, i32 0)\n"
            "  ret void\n"
            "}\n"
            f"declare ptr addrspace(8) @{self.intrinsic}("
            f"ptr addrspace(1), i16, {self.count_type}, i32)\n"
        )


_ABI_CANDIDATES = (
    _AbiCandidate(
        "legacy_p1_i32",
        "llvm.amdgcn.make.buffer.rsrc.p1",
        "i32",
    ),
    _AbiCandidate(
        "p8_p1_i64",
        "llvm.amdgcn.make.buffer.rsrc.p8.p1",
        "i64",
    ),
)
_CANDIDATE_DESCRIPTIONS = {
    "legacy_p1_i32": "legacy spelling / 32-bit count",
    "p8_p1_i64": "indexed spelling / 64-bit count",
}
_CANONICAL_ABI_DESCRIPTIONS = {
    "p1_i32": "basic pointer / 32-bit count",
    "p8_p1_i32": "indexed pointer / 32-bit count",
    "p8_p1_i64": "indexed pointer / 64-bit count",
}

_RunCommand = Callable[[list[str], str | None], CommandResult]


def _run_command(argv: list[str], input_text: str | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            None,
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
        )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _tool_path(name: str, llvm_bin: Path | None) -> Path:
    executable = name + (".exe" if os.name == "nt" else "")
    if llvm_bin is not None:
        candidate = llvm_bin / executable
        if not candidate.is_file():
            raise ProbeError(
                f"compiler capability probe failed - {name} not found in "
                f"llvm_bin={str(llvm_bin)!r}"
            )
        return candidate.resolve()
    discovered = shutil.which(name)
    if discovered is None:
        raise ProbeError(
            f"compiler capability probe failed - {name} not found; pass --llvm-bin "
            "or set ROCKE_LLVM_BIN"
        )
    return Path(discovered).resolve()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _first_diagnostic(result: CommandResult) -> str:
    if result.timed_out:
        return f"compiler command exceeded {_TIMEOUT_SECONDS} seconds"
    for line in (result.stderr + "\n" + result.stdout).splitlines():
        if line.strip():
            return line.strip()
    return f"compiler command exited with status {result.returncode}"


def _source_revisions(version_text: str) -> tuple[str | None, str | None]:
    """Return the primary repository revision and optional patched revision."""
    patched_match = _PATCHED_RE.search(version_text)
    patched = patched_match.group(1).lower() if patched_match else None
    revisions = [match.lower() for match in _SHA_RE.findall(version_text)]
    primary = next((revision for revision in revisions if revision != patched), None)
    return primary, patched


def _data_layout_value(text: str) -> dict[str, object]:
    match = _DATALAYOUT_RE.search(text)
    if match is None:
        raise ProbeError(
            "AMDGPU data-layout probe failed - compiler output has no target datalayout"
        )
    layout = match.group(1)
    fields = layout.split("-")
    p8 = next((field for field in fields if field.startswith("p8:")), None)
    additional_pointer_address_spaces = sorted(
        int(match.group(1))
        for field in fields
        if (match := re.fullmatch(r"p(\d+):.+", field)) and int(match.group(1)) >= 10
    )
    return {
        "text": layout,
        "elf_mangling": "e" if "m:e" in fields else None,
        "address_space_8": [int(part) for part in p8.split(":")[1:]] if p8 else None,
        "additional_pointer_address_spaces": additional_pointer_address_spaces,
    }


def _probe_data_layout(
    clang: Path, processor: str, run: _RunCommand
) -> dict[str, object]:
    argv = [
        str(clang),
        "-target",
        _TARGET_TRIPLE,
        f"-mcpu={processor}",
        "-nogpulib",
        "-S",
        "-emit-llvm",
        "-x",
        "c",
        "-",
        "-o",
        "-",
    ]
    result = run(argv, "void probe(void) {}\n")
    if result.returncode != 0:
        return {
            "status": "error",
            "diagnostic": _first_diagnostic(result),
        }
    try:
        value = _data_layout_value(result.stdout)
    except ProbeError as exc:
        return {"status": "error", "diagnostic": str(exc)}
    return {"status": "known", "value": value}


def _canonical_buffer_resource_abi(intrinsic: str, declaration: str) -> str:
    """Name the canonical ABI selected by ``opt`` from its spelling and type."""
    if intrinsic == "llvm.amdgcn.make.buffer.rsrc.p1":
        return "p1_i32"
    if (
        intrinsic
        in {
            "llvm.amdgcn.make.buffer.rsrc.p8.p1",
            "llvm.amdgcn.make.buffer.rsrc.p8.p1.i32",
        }
        and ", i32, i32)" in declaration
    ):
        return "p8_p1_i32"
    if (
        intrinsic
        in {
            "llvm.amdgcn.make.buffer.rsrc.p8.p1",
            "llvm.amdgcn.make.buffer.rsrc.p8.p1.i64",
        }
        and ", i64, i32)" in declaration
    ):
        return "p8_p1_i64"
    raise ProbeError(
        "buffer-resource probe resolved an unrecognized canonical declaration "
        f"{declaration!r}"
    )


def _probe_buffer_resource_abis(opt: Path, run: _RunCommand) -> dict[str, object]:
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for candidate in _ABI_CANDIDATES:
        result = run([str(opt), "-S", "-", "-o", "-"], candidate.module())
        if result.timed_out:
            return {
                "status": "error",
                "diagnostic": f"{candidate.candidate}: {_first_diagnostic(result)}",
            }
        if result.returncode != 0:
            rejected.append(
                {
                    "candidate": candidate.candidate,
                    "diagnostic": _first_diagnostic(result),
                }
            )
            continue
        declaration = _DECL_RE.search(result.stdout)
        if declaration is None:
            rejected.append(
                {
                    "candidate": candidate.candidate,
                    "diagnostic": "opt did not resolve the declaration as an intrinsic",
                }
            )
            continue
        canonical_intrinsic = declaration.group("name")
        canonical_declaration = declaration.group(1)
        try:
            canonical_abi = _canonical_buffer_resource_abi(
                canonical_intrinsic, canonical_declaration
            )
        except ProbeError as exc:
            return {"status": "error", "diagnostic": str(exc)}
        accepted.append(
            {
                "candidate": candidate.candidate,
                "canonical_abi": canonical_abi,
                "canonical_intrinsic": canonical_intrinsic,
                "canonical_declaration": canonical_declaration,
            }
        )
    return {
        "status": "known",
        "value": {
            "accepted": accepted,
            "rejected": rejected,
        },
    }


def _fingerprint(
    clang: Path,
    opt: Path,
    clang_sha256: str,
    opt_sha256: str,
    clang_version_text: str,
    opt_version_text: str,
    processor: str,
) -> str:
    inputs = {
        "probe_version": _PROBE_VERSION,
        "target_triple": _TARGET_TRIPLE,
        "processor": processor,
        "clang": str(clang),
        "opt": str(opt),
        "clang_sha256": clang_sha256,
        "opt_sha256": opt_sha256,
        "clang_version_text": clang_version_text,
        "opt_version_text": opt_version_text,
    }
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_identity(path: Path, run: _RunCommand) -> str:
    result = run([str(path), "--version"], None)
    if result.returncode != 0:
        raise ProbeError(
            f"compiler capability probe failed - cannot read {path.name} identity: "
            + _first_diagnostic(result)
        )
    identity = result.stdout.strip() or result.stderr.strip()
    if not identity:
        raise ProbeError(
            f"compiler capability probe failed - {path.name} identity was empty"
        )
    return identity


def _compiler_report(
    clang: Path,
    opt: Path,
    processor: str,
    clang_sha256: str,
    opt_sha256: str,
    clang_version_text: str,
    opt_version_text: str,
    fingerprint: str,
    run: _RunCommand,
) -> dict[str, object]:
    source_revision, patched_revision = _source_revisions(clang_version_text)
    return {
        "schema_version": _SCHEMA_VERSION,
        "probe": {
            "name": "rocke-compiler-capabilities",
            "version": _PROBE_VERSION,
            "target_triple": _TARGET_TRIPLE,
            "processor": processor,
            "fingerprint": fingerprint,
        },
        "compiler": {
            "repository": _REPOSITORY,
            "executable": str(clang),
            "sha256": clang_sha256,
            "version_text": clang_version_text,
            "source_revision": source_revision,
            "patched_revision": patched_revision,
        },
        "tools": {
            "opt": {
                "executable": str(opt),
                "sha256": opt_sha256,
                "version_text": opt_version_text,
            }
        },
        "capabilities": {
            _CAPABILITY_DATA_LAYOUT: _probe_data_layout(clang, processor, run),
            _CAPABILITY_BUFFER_RESOURCE: _probe_buffer_resource_abis(opt, run),
        },
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_cache(path: Path, fingerprint: str) -> dict[str, object] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(document, dict):
        return None
    probe = document.get("probe")
    if not isinstance(probe, dict) or probe.get("fingerprint") != fingerprint:
        return None
    return document


def load_or_probe(
    *,
    llvm_bin: Path | None,
    processor: str,
    cache_dir: Path,
    refresh: bool = False,
    run: _RunCommand = _run_command,
) -> tuple[dict[str, object], bool]:
    """Return a probe report and whether it came from the local cache."""
    if not _PROCESSOR_RE.fullmatch(processor):
        raise ValueError(
            f"compiler capability probe failed - processor={processor!r}, "
            "expected a gfx processor identifier"
        )
    clang = _tool_path("clang", llvm_bin)
    opt = _tool_path("opt", llvm_bin)
    clang_sha256 = _sha256(clang)
    opt_sha256 = _sha256(opt)
    clang_version_text = _tool_identity(clang, run)
    opt_version_text = _tool_identity(opt, run)
    fingerprint = _fingerprint(
        clang,
        opt,
        clang_sha256,
        opt_sha256,
        clang_version_text,
        opt_version_text,
        processor,
    )
    cache_path = cache_dir / f"{fingerprint}.json"
    if not refresh:
        cached = _read_cache(cache_path, fingerprint)
        if cached is not None:
            return cached, True
    report = _compiler_report(
        clang,
        opt,
        processor,
        clang_sha256,
        opt_sha256,
        clang_version_text,
        opt_version_text,
        fingerprint,
        run,
    )
    capabilities = report["capabilities"]
    if not isinstance(capabilities, dict):
        raise TypeError("probe report capabilities must be a mapping")
    if all(result.get("status") == "known" for result in capabilities.values()):
        _write_json(cache_path, report)
    return report, False


def _default_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "rocke_compiler_capabilities"


def _print_capability_summary(capabilities: dict[str, object]) -> None:
    """Print the observed behavior in terms useful to a rocKE developer."""
    data_layout = capabilities[_CAPABILITY_DATA_LAYOUT]
    if not isinstance(data_layout, dict):
        raise TypeError("data-layout capability must be a mapping")
    print("AMDGPU module layout:")
    print(f"  status: {data_layout['status']}")
    if data_layout["status"] == "known":
        value = data_layout["value"]
        if not isinstance(value, dict):
            raise TypeError("known data-layout value must be a mapping")
        mangling = value["elf_mangling"]
        address_space_8 = value["address_space_8"]
        additional_pointer_address_spaces = value["additional_pointer_address_spaces"]
        print(f"  ELF mangling: {mangling if mangling is not None else 'absent'}")
        print(
            "  address space 8: "
            + (
                ":".join(str(part) for part in address_space_8)
                if isinstance(address_space_8, list)
                else "absent"
            )
        )
        if not isinstance(additional_pointer_address_spaces, list):
            raise TypeError("additional pointer address spaces must be a list")
        print(
            "  additional pointer address spaces: "
            + (
                ", ".join(str(space) for space in additional_pointer_address_spaces)
                or "none"
            )
        )
    else:
        print(f"  diagnostic: {data_layout['diagnostic']}")

    buffer_resource = capabilities[_CAPABILITY_BUFFER_RESOURCE]
    if not isinstance(buffer_resource, dict):
        raise TypeError("buffer-resource capability must be a mapping")
    print("Buffer-resource intrinsic contract:")
    print(f"  status: {buffer_resource['status']}")
    if buffer_resource["status"] == "known":
        value = buffer_resource["value"]
        if not isinstance(value, dict):
            raise TypeError("known buffer-resource value must be a mapping")
        accepted = value["accepted"]
        rejected = value["rejected"]
        if not isinstance(accepted, list) or not isinstance(rejected, list):
            raise TypeError(
                "buffer-resource accepted and rejected values must be lists"
            )
        accepted_descriptions = [
            _CANONICAL_ABI_DESCRIPTIONS.get(
                str(entry["canonical_abi"]), str(entry["canonical_abi"])
            )
            for entry in accepted
            if isinstance(entry, dict)
        ]
        rejected_descriptions = [
            _CANDIDATE_DESCRIPTIONS.get(
                str(entry["candidate"]), str(entry["candidate"])
            )
            for entry in rejected
            if isinstance(entry, dict)
        ]
        print(
            f"  accepted canonical ABIs: {', '.join(accepted_descriptions) or 'none'}"
        )
        print(f"  rejected candidates: {', '.join(rejected_descriptions) or 'none'}")
    else:
        print(f"  diagnostic: {buffer_resource['diagnostic']}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="probe focused AMD LLVM behavior and write capabilities.json"
    )
    parser.add_argument(
        "--llvm-bin",
        type=Path,
        default=None,
        help="directory containing clang and opt (default: ROCKE_LLVM_BIN, then PATH)",
    )
    parser.add_argument(
        "--processor",
        default=_DEFAULT_PROCESSOR,
        help=f"AMDGPU processor used by target-specific probes (default: {_DEFAULT_PROCESSOR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("capabilities.json"),
        help="manifest path (default: ./capabilities.json)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="local probe cache directory",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore a matching cached report and run the probes again",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    llvm_bin = args.llvm_bin
    if llvm_bin is None:
        raw = os.environ.get("ROCKE_LLVM_BIN")
        llvm_bin = Path(raw) if raw else None
    started = time.perf_counter()
    try:
        report, cache_hit = load_or_probe(
            llvm_bin=llvm_bin,
            processor=args.processor,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
        _write_json(args.output, report)
    except (OSError, ProbeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - started
    compiler = report["compiler"]
    capabilities = report["capabilities"]
    if not isinstance(compiler, dict) or not isinstance(capabilities, dict):
        raise TypeError("probe report has an invalid internal shape")
    revision = compiler.get("source_revision") or "unreported"
    print(f"compiler revision: {revision}")
    print(f"cache: {'hit' if cache_hit else 'miss'}")
    _print_capability_summary(capabilities)
    print(f"output: {args.output}")
    print(f"elapsed: {elapsed:.3f} s")
    return (
        0
        if all(result.get("status") == "known" for result in capabilities.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
