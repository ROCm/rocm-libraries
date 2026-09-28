# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared support for standalone gfx1250 ISA-feature validation."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rocke.core.ir import KernelDef
    from rocke.helpers.compile import KernelArtifact

try:
    from ....helpers import compile_kernel
    from ....runtime.hip_module import Runtime
except ImportError:  # Direct execution: PYTHONPATH=platform/python python file.py
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

_LLVM_FLAVOR = "llvm23"
_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


class ValidationError(RuntimeError):
    """An emitted artifact did not contain a required bridge feature."""


@dataclass(frozen=True)
class ValidatedArtifact:
    """Compiled kernel plus the ISA text used for validation."""

    artifact: KernelArtifact
    isa_text: str


@dataclass(frozen=True)
class _Result:
    name: str
    status: str
    detail: str


class Reporter:
    """Collect and print checks in a deterministic order."""

    def __init__(self, arch: str) -> None:
        self.arch = arch
        self._results: list[_Result] = []

    def passed(self, name: str, detail: str) -> None:
        self._results.append(_Result(name, _PASS, detail))

    def failed(self, name: str, detail: str) -> None:
        self._results.append(_Result(name, _FAIL, detail))

    def skipped(self, name: str, detail: str) -> None:
        self._results.append(_Result(name, _SKIP, detail))

    def finish(self) -> int:
        for result in self._results:
            print(f"[{self.arch}] {result.name}: {result.status} - {result.detail}")
        failures = sum(result.status == _FAIL for result in self._results)
        passes = sum(result.status == _PASS for result in self._results)
        skips = sum(result.status == _SKIP for result in self._results)
        status = _PASS if failures == 0 else _FAIL
        print(
            f"[{self.arch}] summary: {status} "
            f"(pass={passes}, fail={failures}, skip={skips})"
        )
        return 0 if failures == 0 else 1


def make_parser(description: str) -> argparse.ArgumentParser:
    """Return the common command-line interface for one verifier."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--arch", default="gfx1250")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="validate LLVM and ISA without loading or launching the code object",
    )
    return parser


def require_llvm23() -> None:
    """Select the LLVM flavor required by the new intrinsics."""
    os.environ["ROCKE_LLVM_FLAVOR"] = _LLVM_FLAVOR


def _objdump_path() -> str:
    explicit = os.environ.get("LLVM_OBJDUMP")
    if explicit:
        return explicit
    rocm_root = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if rocm_root:
        candidate = Path(rocm_root) / "llvm" / "bin" / "llvm-objdump"
        if candidate.is_file():
            return str(candidate)
    candidate = shutil.which("llvm-objdump")
    if candidate:
        return candidate
    raise FileNotFoundError(
        "llvm-objdump is required for ISA validation; set LLVM_OBJDUMP or ROCM_PATH"
    )


def disassemble_hsaco(hsaco: bytes, arch: str) -> str:
    """Disassemble one in-memory code object with the selected ROCm objdump."""
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as handle:
        handle.write(hsaco)
        hsaco_path = Path(handle.name)
    try:
        proc = subprocess.run(
            [
                _objdump_path(),
                "--disassemble",
                f"--mcpu={arch}",
                "--triple=amdgcn-amd-amdhsa",
                str(hsaco_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout).strip()
            raise ValidationError(f"llvm-objdump failed: {message}")
        if not proc.stdout.strip():
            raise ValidationError("llvm-objdump produced no ISA text")
        return proc.stdout
    finally:
        hsaco_path.unlink(missing_ok=True)


def _assert_llvm(llvm_text: str, required: tuple[str, ...]) -> None:
    missing = [needle for needle in required if needle not in llvm_text]
    if missing:
        raise ValidationError(f"LLVM text missing: {', '.join(missing)}")


def _assert_isa(isa_text: str, required: tuple[str, ...]) -> None:
    missing = [
        pattern
        for pattern in required
        if re.search(pattern, isa_text, flags=re.IGNORECASE | re.MULTILINE) is None
    ]
    if missing:
        raise ValidationError(f"ISA text missing patterns: {', '.join(missing)}")


def compile_and_validate(
    kernel: KernelDef,
    *,
    arch: str,
    llvm_required: tuple[str, ...],
    isa_required: tuple[str, ...],
) -> ValidatedArtifact:
    """Compile a kernel and require features in both LLVM text and final ISA."""
    require_llvm23()
    artifact = compile_kernel(kernel, arch=arch, backend="python")
    _assert_llvm(artifact.llvm_text, llvm_required)
    isa_text = disassemble_hsaco(artifact.hsaco, arch)
    _assert_isa(isa_text, isa_required)
    return ValidatedArtifact(artifact=artifact, isa_text=isa_text)


def record_compile_check(
    reporter: Reporter,
    name: str,
    kernel: KernelDef,
    *,
    arch: str,
    llvm_required: tuple[str, ...],
    isa_required: tuple[str, ...],
) -> ValidatedArtifact | None:
    """Compile, validate, and turn exceptions into one stable check result."""
    try:
        validated = compile_and_validate(
            kernel,
            arch=arch,
            llvm_required=llvm_required,
            isa_required=isa_required,
        )
    except Exception as exc:  # noqa: BLE001 - verifier must report toolchain failures
        reporter.failed(name, f"{type(exc).__name__}: {exc}")
        return None
    reporter.passed(name, "LLVM and ISA matched")
    return validated


def as_host_bytes(array: np.ndarray) -> ctypes.Array:
    """Return a ctypes byte copy accepted by Runtime.memcpy_h2d."""
    contiguous = np.ascontiguousarray(array)
    return (ctypes.c_uint8 * int(contiguous.nbytes)).from_buffer_copy(contiguous)


class DeviceArena:
    """Own device allocations for one short functional verification."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._pointers: list[int] = []

    def input(self, array: np.ndarray) -> int:
        contiguous = np.ascontiguousarray(array)
        pointer = self.runtime.alloc(contiguous.nbytes)
        self.runtime.memcpy_h2d(pointer, as_host_bytes(contiguous), contiguous.nbytes)
        self._pointers.append(pointer)
        return pointer

    def output(self, nbytes: int, *, fill: int = 0) -> int:
        pointer = self.runtime.alloc(nbytes)
        self.runtime.memset(pointer, fill, nbytes)
        self._pointers.append(pointer)
        return pointer

    def read(
        self, pointer: int, *, dtype: np.dtype, shape: tuple[int, ...]
    ) -> np.ndarray:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        buffer = (ctypes.c_uint8 * nbytes)()
        self.runtime.memcpy_d2h(buffer, pointer, nbytes)
        return np.frombuffer(bytes(buffer), dtype=dtype).reshape(shape)

    def close(self) -> None:
        for pointer in reversed(self._pointers):
            self.runtime.free(pointer)
        self._pointers.clear()

    def __enter__(self) -> DeviceArena:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


def launch(
    runtime: Runtime,
    validated: ValidatedArtifact,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    pack_format: str,
    pack_values: tuple[int, ...],
) -> None:
    """Load, launch with packed bytes, synchronize, and unload one kernel."""
    module = runtime.load_module(validated.artifact.hsaco)
    try:
        function = module.get_function(validated.artifact.kernel_name)
        packed = struct.pack(pack_format, *pack_values)
        runtime.launch(function, grid, block, packed, stream=0)
        runtime.wait_stream(0)
    finally:
        module.unload()
