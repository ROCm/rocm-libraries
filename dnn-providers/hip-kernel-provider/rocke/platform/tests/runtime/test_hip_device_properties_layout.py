# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the ctypes ABI against installed HIP headers without loading HIP.

Set ROCM_PATH, ROCM_HOME, or HIP_PATH, or put hipcc on PATH. The test skips
if the headers or a host C++ compiler are unavailable, and the launch-config
check skips on headers that predate cluster launch. Layout mismatches and
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
from rocke.runtime.hip_module import (
    _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
    _HipLaunchAttribute,
    _HipLaunchAttributeValue,
    _HipLaunchConfig,
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


_HEADER = [
    "#include <hip/hip_runtime_api.h>",
    "#include <cstddef>",
    "#include <cstdio>",
    "#include <cstring>",
]


def _struct_asserts(structures) -> list[str]:
    # The compiler gets the native layout from HIP headers. ctypes supplies
    # the field names and expected layout, not the native declarations.
    lines = []
    for name, structure in structures:
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
    return lines


def _layout_probe() -> str:
    lines = _HEADER + _struct_asserts(
        (
            ("hipDeviceProp_tR0600", HipDevicePropR0600),
            ("hipUUID", _HipUUID),
            ("hipDeviceArch_t", _HipDeviceArch),
        )
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


def _launch_probe() -> str:
    # The cluster launch path in hip_module.py fills these by hand and passes
    # them to hipDrvLaunchKernelEx.
    lines = _HEADER + _struct_asserts(
        (
            ("HIP_LAUNCH_CONFIG", _HipLaunchConfig),
            ("hipLaunchAttribute", _HipLaunchAttribute),
            ("hipLaunchAttributeValue", _HipLaunchAttributeValue),
        )
    )
    lines.append(
        "static_assert(hipLaunchAttributeClusterDimension == "
        f'{_HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION}, "cluster attribute id");'
    )
    lines.append("int main() { return 0; }")
    return "\n".join(lines)


def _compile_and_run(tmp_path: Path, probe: str) -> None:
    include, compiler = _hip_toolchain()
    source = tmp_path / "hip_layout.cpp"
    executable = tmp_path / ("hip_layout.exe" if os.name == "nt" else "hip_layout")
    source.write_text(probe, encoding="utf-8")
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


def test_hip_device_properties_layout(tmp_path: Path) -> None:
    _compile_and_run(tmp_path, _layout_probe())


def test_hip_launch_config_layout(tmp_path: Path) -> None:
    include, _ = _hip_toolchain()
    if not any(
        "hipLaunchAttributeClusterDimension" in header.read_text(errors="ignore")
        for header in (include / "hip").rglob("*.h")
    ):
        pytest.skip(f"HIP headers under {include} predate cluster launch")
    _compile_and_run(tmp_path, _launch_probe())
