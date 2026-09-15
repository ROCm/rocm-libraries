################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os

from pathlib import Path
from typing import List, NamedTuple

from tensilelite.Common.Utilities import isRhel8, print2
from tensilelite._runtime import executable_search_paths

osSelect = lambda linux, windows: linux if os.name != "nt" else windows


def _windowsWithExtensions(exe: str) -> List[str]:
    if not os.name == "nt":
        raise ValueError("These extensions should not be added on anything but Windows")
    files = [exe]
    files.extend([exe + ext.lower() for ext in os.environ["PATHEXT"].split(";")])
    return files


class ToolchainDefaults(NamedTuple):
    inFFMEnv = os.environ.get("HSA_MODEL_MEMFILE", "") != ""
    CXX_COMPILER = osSelect(linux="amdclang++", windows="clang++.exe")
    C_COMPILER = osSelect(linux="amdclang", windows="clang.exe")
    OFFLOAD_BUNDLER = osSelect(linux="clang-offload-bundler", windows="clang-offload-bundler.exe")
    DEVICE_ENUMERATOR = osSelect(linux="offload-arch", windows="hipinfo")
    ASSEMBLER = osSelect(linux="amdclang++", windows="clang++.exe")
    HIP_CONFIG = osSelect(linux="hipconfig", windows="hipconfig.exe")


def _supportedComponent(component: str, targets: List[str]) -> bool:
    if os.name == "nt":
        targets = [tExt for t in targets for tExt in _windowsWithExtensions(t)]
    isSupported = any([component == t for t in targets]) or any([Path(component).name == t for t in targets])
    return isSupported


def supportedCCompiler(compiler: str) -> bool:
    """
    Determine if a C compiler/assembler is supported by tensilelite.

    Args:
        compiler: The name of a compiler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(compiler, ["amdclang", "clang"])


def supportedCxxCompiler(compiler: str) -> bool:
    """
    Determine if a C++/HIP compiler/assembler is supported by tensilelite.

    Args:
        compiler: The name of a compiler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(compiler, ["amdclang++", "clang++"])


def supportedOffloadBundler(bundler: str) -> bool:
    """
    Determine if an offload bundler is supported by tensilelite.

    Args:
        bundler: The name of an offload bundler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(bundler, ["clang-offload-bundler"])


def supportedHip(hip: str) -> bool:
    """
    Determine if a hip callable binary is supported by tensilelite.

    Args:
        hip: The name of an offload bundler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(hip, ["hipcc", "hipconfig"])


def supportedDeviceEnumerator(enumerator: str) -> bool:
    """
    Determine if a device enumerator is supported by tensilelite.

    Args:
        enumerator: The name of a device enumerator to test for support.

    Return:
        If supported True; otherwise, False.
    """
    if os.name == "nt":
        return _supportedComponent(enumerator, ["hipinfo", "hipInfo"])
    return _supportedComponent(enumerator, ["offload-arch", "amdgpu-arch", "rocm_agent_enumerator"])


def _exeExists(file: Path) -> bool:
    """
    Check if a file exists and is executable.

    Args:
        file: The file to check.

    Returns:
        If the file exists and is executable, True; otherwise, False
    """
    return True if os.access(file, os.X_OK) else False


def _validateExecutable(file: str, searchPaths: List[Path]) -> str:
    """
    Validate that the given toolchain component is below the selected root and executable.

    Args:
        file: The executable to validate.
        searchPaths: List of directories to search for the executable.

    Returns:
        The validated executable with an absolute path.
    """
    print2(f"Validating {file}")

    if not any((
        supportedCxxCompiler(file),
        supportedCCompiler(file),
        supportedOffloadBundler(file),
        supportedHip(file),
        supportedDeviceEnumerator(file)
    )):
        raise ValueError(f"`{file}` is not a supported toolchain component on {'Windows' if os.name == 'nt' else 'Linux'}")

    # Check if the file is an absolute path and executable
    if Path(file).is_absolute():
        if _exeExists(Path(file)):
            return file
        raise FileNotFoundError(f"`{file}` either not found or not executable")

    # Then check the search paths
    files = _windowsWithExtensions(file) if os.name == "nt" else [file]
    for path in searchPaths:
        for f in files:
            p = path / f
            if _exeExists(p):
                return str(p)
    raise FileNotFoundError(f"`{file}` either not found or not executable in any search path: {':'.join(map(str, searchPaths))}")


def validateToolchain(*args: str):
    """
    Validate that the given toolchain components are below the selected root and executable,
    returning the absolute path to each.

    Args:
        args: List of executable toolchain components to validate.

    Returns:
        List of validated executables with absolute paths.

    Raises:
        ValueError: If no toolchain components are provided.
        FileNotFoundError: If a toolchain component is not found below the selected root.
    """
    if not args:
        raise ValueError("No toolchain components to validate, at least one argument is required")

    searchPaths = executable_search_paths()

    out = (_validateExecutable(x, searchPaths) for x in args)

    return next(out) if len(args) == 1 else tuple(out)


def deviceEnumeratorCandidates(explicit: str | None = None) -> tuple[str, ...]:
    """Return validated device-enumerator paths in fallback order."""
    if explicit is not None:
        return (validateToolchain(explicit),)
    if os.name == "nt":
        return (validateToolchain(ToolchainDefaults.DEVICE_ENUMERATOR),)

    names = ["offload-arch", "amdgpu-arch"]
    if isRhel8() or ToolchainDefaults.inFFMEnv:
        names.append("rocm_agent_enumerator")

    paths = []
    for name in names:
        try:
            paths.append(validateToolchain(name))
        except FileNotFoundError:
            continue
    if paths:
        return tuple(paths)

    raise FileNotFoundError(
        "No supported device enumerator is executable in the selected tool paths: "
        f"{':'.join(map(str, executable_search_paths()))}"
    )
