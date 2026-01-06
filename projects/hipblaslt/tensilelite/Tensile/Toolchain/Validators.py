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
import re

from pathlib import Path
from typing import List, NamedTuple, Union

from Tensile.Common.Utilities import isRhel8, print2

DEFAULT_ROCM_BIN_PATH_POSIX = Path("/opt/rocm/bin")
DEFAULT_ROCM_LLVM_BIN_PATH_POSIX = Path("/opt/rocm/lib/llvm/bin")
DEFAULT_ROCM_BIN_PATH_WINDOWS = Path("C:/Program Files/AMD/ROCm")


osSelect = lambda linux, windows: linux if os.name != "nt" else windows


def _windowsLatestRocmBin(path: Union[Path, str]) -> Path:
    """
    Get the path to the latest ROCm bin directory, on Windows.

    This function assumes that ROCm versions are differentiated with the form ``X.Y``.

    Args:
        path: The path to the ROCm root directory, typically ``C:/Program Files/AMD/ROCm``.

    Returns:
        The path to the ROCm bin directory for the latest ROCm version.
        Typically of the form ``C:/Program Files/AMD/ROCm/X.Y/bin``.
        May return ``None`` if no ``X.Y`` subdirectories exist, perhaps due to
        a partial uninstall.
    """
    path = Path(path)
    pattern = re.compile(r"^\d+\.\d+$")
    versions = list(filter(lambda d: d.is_dir() and pattern.match(d.name), path.iterdir()))
    if len(versions) == 0:
        return None
    latest = max(versions, key=lambda d: tuple(map(int, d.name.split("."))))
    return latest / "bin"


def _windowsSearchPaths() -> List[Path]:
    defaultPath = DEFAULT_ROCM_BIN_PATH_WINDOWS
    searchPaths = []

    if os.environ.get("HIP_PATH"):
        hipPaths = [Path(p) / "bin" for p in os.environ["HIP_PATH"].split(os.pathsep)]
        searchPaths.extend(hipPaths)

    if Path(defaultPath).exists():
        latestRocmBin = _windowsLatestRocmBin(defaultPath)
        if latestRocmBin:
            searchPaths.append(latestRocmBin)

    if os.environ.get("PATH"):
        envPath = [Path(p) for p in os.environ["PATH"].split(os.pathsep)]
        searchPaths.extend(envPath)

    return searchPaths


def _windowsWithExtensions(exe: str) -> List[str]:
    if not os.name == "nt":
        raise ValueError("These extensions should not be added on anything but Windows")
    files = [exe]
    files.extend([exe + ext.lower() for ext in os.environ["PATHEXT"].split(";")])
    return files


def _posixSearchPaths() -> List[Path]:

    searchPaths = []

    if os.environ.get("ROCM_PATH"):
        for p in os.environ["ROCM_PATH"].split(os.pathsep):
            searchPaths.append(Path(p) / "bin")
            searchPaths.append(Path(p) / "lib" / "llvm" / "bin")

    searchPaths.extend(
        [
            DEFAULT_ROCM_BIN_PATH_POSIX,
            DEFAULT_ROCM_LLVM_BIN_PATH_POSIX,
        ]
    )

    if os.environ.get("PATH"):
        envPath = [Path(p) for p in os.environ["PATH"].split(os.pathsep)]
        searchPaths.extend(envPath)

    return searchPaths


class ToolchainDefaults(NamedTuple):
    CXX_COMPILER = osSelect(linux="amdclang++", windows="clang++.exe")
    C_COMPILER = osSelect(linux="amdclang", windows="clang.exe")
    OFFLOAD_BUNDLER = osSelect(linux="clang-offload-bundler", windows="clang-offload-bundler.exe")
    DEVICE_ENUMERATOR = osSelect(linux="rocm_agent_enumerator" if isRhel8() else "amdgpu-arch", windows="hipinfo")
    ASSEMBLER = osSelect(linux="amdclang++", windows="clang++.exe")
    HIP_CONFIG = osSelect(linux="hipconfig", windows="hipconfig.exe")


def _supportedComponent(component: str, targets: List[str]) -> bool:
    if os.name == "nt":
        targets = [tExt for t in targets for tExt in _windowsWithExtensions(t)]
    isSupported = any([component == t for t in targets]) or any([Path(component).name == t for t in targets])
    return isSupported


def supportedCCompiler(compiler: str) -> bool:
    """
    Determine if a C compiler/assembler is supported by Tensile.

    Args:
        compiler: The name of a compiler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(compiler, ["amdclang", "clang"])


def supportedCxxCompiler(compiler: str) -> bool:
    """
    Determine if a C++/HIP compiler/assembler is supported by Tensile.

    Args:
        compiler: The name of a compiler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(compiler, ["amdclang++", "clang++"])


def supportedOffloadBundler(bundler: str) -> bool:
    """
    Determine if an offload bundler is supported by Tensile.

    Args:
        bundler: The name of an offload bundler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(bundler, ["clang-offload-bundler"])


def supportedHip(hip: str) -> bool:
    """
    Determine if a hip callable binary is supported by Tensile.

    Args:
        hip: The name of an offload bundler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    return _supportedComponent(hip, ["hipcc", "hipconfig"])


def supportedDeviceEnumerator(enumerator: str) -> bool:
    """
    Determine if a device enumerator is supported by Tensile.

    Args:
        enumerator: The name of a device enumerator to test for support.

    Return:
        If supported True; otherwise, False.
    """
    if os.name == "nt":
        return _supportedComponent(enumerator, ["hipinfo", "hipInfo"])
    return _supportedComponent(enumerator, ["rocm_agent_enumerator", "amdgpu-arch"])


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
    Validate that the given toolchain component is in the PATH and executable.

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
    Validate that the given toolchain components are in the PATH and executable,
    returning the absolute path to each.

    Args:
        args: List of executable toolchain components to validate.

    Returns:
        List of validated executables with absolute paths.

    Raises:
        ValueError: If no toolchain components are provided.
        FileNotFoundError: If a toolchain component is not found in the PATH.
    """
    if not args:
        raise ValueError("No toolchain components to validate, at least one argument is required")

    searchPaths = _windowsSearchPaths() if os.name == "nt" else _posixSearchPaths()

    out = (_validateExecutable(x, searchPaths) for x in args)

    return next(out) if len(args) == 1 else tuple(out)
