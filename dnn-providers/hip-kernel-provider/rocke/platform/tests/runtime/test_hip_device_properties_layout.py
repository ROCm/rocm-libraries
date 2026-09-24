# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the ctypes ABI against installed HIP headers without loading HIP.

Set ROCM_PATH, ROCM_HOME, or HIP_PATH, or put hipcc on PATH. The test skips
if the headers or a host C++ compiler are unavailable. Layout mismatches and
compiler errors fail the test.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rocke.runtime._hip_device_properties import (
    HipDevicePropR0600,
    _HipDeviceArch,
    _HipUUID,
)


def _hip_toolchain() -> tuple[Path, str]:
    root = next(
        (
            os.environ[key]
            for key in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")
            if os.environ.get(key)
        ),
        None,
    )
    if root is None:
        hipcc = shutil.which("hipcc")
        if hipcc is None:
            pytest.skip("set ROCM_PATH or put hipcc on PATH to check the HIP layout")
        root = str(Path(hipcc).resolve().parent.parent)
    include = Path(root) / "include"
    if not (include / "hip" / "hip_runtime_api.h").is_file():
        pytest.skip(f"HIP headers unavailable under {include}")
    compiler = next(
        (
            found
            for directory in (
                Path(root) / "llvm" / "bin",
                Path(root) / "lib" / "llvm" / "bin",
            )
            if (found := shutil.which("clang++", path=str(directory)))
        ),
        None,
    )
    compiler = compiler or shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required to check the HIP layout")
    return include, compiler


def _layout_probe() -> str:
    # The compiler gets the native layout from HIP headers. ctypes supplies
    # the field names and expected layout, not the native declarations.
    lines = [
        "#include <hip/hip_runtime_api.h>",
        "#include <cstddef>",
        "#include <cstdio>",
        "#include <cstring>",
    ]
    for name, structure in (
        ("hipDeviceProp_tR0600", HipDevicePropR0600),
        ("hipUUID", _HipUUID),
        ("hipDeviceArch_t", _HipDeviceArch),
    ):
        lines.append(
            f'static_assert(sizeof({name}) == {ctypes.sizeof(structure)}, "{name} size");'
        )
        lines.append(
            f"static_assert(alignof({name}) == {ctypes.alignment(structure)}, "
            f'"{name} alignment");'
        )
        for field in structure._fields_:
            if len(field) == 3:  # C++ offsetof cannot be used with bitfields.
                continue
            member, member_type = field
            lines.append(
                f"static_assert(offsetof({name}, {member}) == "
                f'{getattr(structure, member).offset}, "{name}.{member} offset");'
            )
            lines.append(
                f"static_assert(sizeof((({name}*)nullptr)->{member}) == "
                f'{ctypes.sizeof(member_type)}, "{name}.{member} size");'
            )
    lines.append("int main() {")
    for member, _, _ in _HipDeviceArch._fields_:
        expected = _HipDeviceArch()
        setattr(expected, member, 1)
        octets = ", ".join(str(byte) for byte in bytes(expected))
        lines.extend(
            [
                "{ hipDeviceArch_t actual;",
                "std::memset(&actual, 0, sizeof(actual));",
                f"actual.{member} = 1;",
                f"const unsigned char expected[] = {{{octets}}};",
                "if (std::memcmp(&actual, expected, sizeof(actual)) != 0) {",
                f'  std::fprintf(stderr, "hipDeviceArch_t.{member} bits differ\\n");',
                "  return 1; } }",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


def test_hip_device_properties_layout(tmp_path: Path) -> None:
    include, compiler = _hip_toolchain()
    source = tmp_path / "hip_layout.cpp"
    executable = tmp_path / ("hip_layout.exe" if os.name == "nt" else "hip_layout")
    source.write_text(_layout_probe(), encoding="utf-8")
    # This is a host executable; it neither links libamdhip64 nor needs a GPU.
    built = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-D__HIP_PLATFORM_AMD__",
            f"-I{include}",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        built.returncode == 0
    ), f"{compiler}, HIP headers: {include}\n{built.stdout}\n{built.stderr}"
    checked = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
