# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Choose, preload and check the libhipblaslt that hipblaslt._core binds to.

Review A's Decision 10 (AIHPBLAS-5274), based on Spike 3 (AIHPBLAS-5626). The lookup order is
HIPBLASLT_PY_LIBRARY, then the rocm-sdk wheels, then ROCM_PATH, then ROCM_HOME, then the build-time
path. The chosen library is preloaded by full path, so _core's NEEDED entry binds to it.

Everything above load() is a pure function of its arguments, so the tests can drive every case on
any platform without a second ROCm.
"""

from __future__ import annotations

import ctypes
import logging
import ntpath
import os
import posixpath
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "LIBRARY_ENV",
    "Candidate",
    "HipblasLtLoadError",
    "HipblasLtLoadWarning",
    "LoadInfo",
    "candidate_libraries",
    "check_companions",
    "check_override_can_take_effect",
    "choose_library",
    "loaded_hipblaslt",
    "override_warning",
    "version_mismatch_warning",
    "windows_dirs_to_register",
    "windows_dll_dirs",
]

LIBRARY_ENV = "HIPBLASLT_PY_LIBRARY"

_log = logging.getLogger("hipblaslt")


class HipblasLtLoadError(ImportError):
    """The process can't use the hipBLASLt that was asked for, or uses a mixed set of libraries."""


class HipblasLtLoadWarning(RuntimeWarning):
    """The loaded hipBLASLt works, but probably isn't the one you meant."""


@dataclass(frozen=True)
class Candidate:
    path: str
    source: (
        str  # "HIPBLASLT_PY_LIBRARY", "rocm-sdk", "ROCM_PATH", "ROCM_HOME" or "build"
    )


@dataclass(frozen=True)
class LoadInfo:
    path: str
    source: str
    preloaded: bool


def library_file_name(platform: str) -> str:
    return "libhipblaslt.dll" if platform == "win32" else "libhipblaslt.so.1"


def companion_stems(platform: str) -> tuple[str, ...]:
    # There's no rocRoller on Windows.
    if platform == "win32":
        return ("libtensilelite-host", "origami")
    return ("libtensilelite-host", "liborigami", "librocroller")


def _root_lib_dir(root: str, platform: str) -> str:
    return os.path.join(root, "bin" if platform == "win32" else "lib")


def candidate_libraries(
    environ: Mapping[str, str],
    platform: str,
    rocm_sdk_libraries: Sequence[str],
    build_lib_dir: str,
) -> list[Candidate]:
    """Every place to look, in Decision 10's order. Existence is checked by choose_library."""
    name = library_file_name(platform)
    out = []
    override = environ.get(LIBRARY_ENV)
    if override:
        out.append(Candidate(override, LIBRARY_ENV))
    out += [Candidate(p, "rocm-sdk") for p in rocm_sdk_libraries]
    for var in ("ROCM_PATH", "ROCM_HOME"):
        root = environ.get(var)
        if root:
            out.append(
                Candidate(os.path.join(_root_lib_dir(root, platform), name), var)
            )
    if build_lib_dir:
        out.append(Candidate(os.path.join(build_lib_dir, name), "build"))
    return out


def choose_library(
    candidates: Sequence[Candidate], exists: Callable[[str], bool]
) -> Candidate:
    """The first candidate that exists. An override that doesn't exist is an error, not a skip."""
    for c in candidates:
        if exists(c.path):
            return c
        if c.source == LIBRARY_ENV:
            raise HipblasLtLoadError(
                f"{LIBRARY_ENV} is set to {c.path!r}, which doesn't exist."
            )
    raise HipblasLtLoadError(
        "Could not find libhipblaslt. Looked in: "
        + ", ".join(f"{c.path} ({c.source})" for c in candidates)
        + f". Install the rocm-sdk wheels, set ROCM_PATH, or set {LIBRARY_ENV}."
    )


def _pathmod(platform: str):
    return ntpath if platform == "win32" else posixpath


def _canon(path: str, platform: str) -> str:
    """Normalized path for comparisons. Symlinks resolve only when `platform` is this host's."""
    pm = _pathmod(platform)
    if (platform == "win32") == (sys.platform == "win32"):
        path = os.path.realpath(path)
    return pm.normcase(pm.normpath(path))


def _base(path: str, platform: str) -> str:
    return _pathmod(platform).basename(path).lower()


def _same_file(a: str, b: str, platform: str) -> bool:
    return _canon(a, platform) == _canon(b, platform)


def loaded_hipblaslt(loaded: Iterable[str], platform: str) -> list[str]:
    """The libhipblaslt files among the process's loaded libraries, deduplicated."""
    out: list[str] = []
    for p in loaded:
        b = _base(p, platform)
        hit = (
            b == "libhipblaslt.dll"
            if platform == "win32"
            else b.startswith("libhipblaslt.so")
        )
        if hit and not any(_same_file(p, q, platform) for q in out):
            out.append(p)
    return out


def check_override_can_take_effect(
    override: str, loaded: Iterable[str], platform: str
) -> None:
    """Raise if something already loaded a different libhipblaslt than the override names."""
    others = [
        p
        for p in loaded_hipblaslt(loaded, platform)
        if not _same_file(p, override, platform)
    ]
    if others:
        raise HipblasLtLoadError(
            f"{LIBRARY_ENV}={override} can't take effect: this process already uses hipBLASLt "
            f"from {others[0]} (probably loaded by torch or another framework). Import hipblaslt "
            f"before that framework, or unset {LIBRARY_ENV}."
        )


def check_companions(hipblaslt_path: str, loaded: Iterable[str], platform: str) -> None:
    """Raise if hipBLASLt's companion libraries didn't load from libhipblaslt's own directory."""
    pm = _pathmod(platform)
    home = pm.dirname(_canon(hipblaslt_path, platform))
    stems = companion_stems(platform)
    mixed = []
    for p in loaded:
        b = _base(p, platform)
        if not any(b == s + ".dll" or b.startswith(s + ".so") for s in stems):
            continue
        if pm.dirname(_canon(p, platform)) != home:
            mixed.append(p)
    if mixed:
        raise HipblasLtLoadError(
            f"hipBLASLt from {hipblaslt_path} and its companion libraries come from different "
            f"builds: {', '.join(sorted(set(mixed)))}. Mixing builds can corrupt memory silently "
            "(ROCM-29472). Make sure one ROCm install comes first on the search path."
        )


def override_warning(override: str) -> str:
    return (
        f"{LIBRARY_ENV} is set, so hipBLASLt and its HIP runtime come from {override}. Importing a "
        "framework built against a different ROCm afterward (torch, for example) will fail to load."
    )


def _version_text(version: int, rev: str) -> str:
    major, rest = divmod(version, 100000)
    minor, patch = divmod(rest, 100)
    return f"{major}.{minor}.{patch} ({rev})"


def version_mismatch_warning(
    built: tuple[int, str], loaded: tuple[int, str] | None, path: str
) -> str | None:
    """A warning when the loaded build isn't the one the bindings were compiled against.

    Both the version and the git revision count: two nightlies can both report 1.5.0.
    """
    if loaded is None or (loaded[0], loaded[1].strip()) == (built[0], built[1].strip()):
        return None
    return (
        f"hipBLASLt {_version_text(*loaded)} is loaded from {path}, but these bindings were built "
        f"against {_version_text(*built)}. Solution indices and behavior can differ between builds."
    )


def windows_dll_dirs(
    library_path: str,
    environ: Mapping[str, str],
    hip_runtime_dirs: Sequence[str],
    build_bin_dir: str,
) -> list[str]:
    """Ordered, de-duplicated directories to register with os.add_dll_directory.

    The library's own directory first, then its HIP runtime's (in the rocm-sdk wheels that's
    _rocm_sdk_core/bin, not next to libhipblaslt.dll), then ROCM_PATH, HIP_PATH and ROCM_HOME, then the
    build-time ROCm. Modeled on rocISA's _candidate_dll_dirs; pure, so it's testable off Windows.
    """
    lib_dir = os.path.dirname(library_path)
    dirs = [lib_dir]
    # The rocm-sdk wheel layout: site-packages/_rocm_sdk_libraries/bin next to _rocm_sdk_core/bin.
    package_dir = os.path.dirname(lib_dir)
    if os.path.basename(package_dir).lower() == "_rocm_sdk_libraries":
        dirs.append(os.path.join(os.path.dirname(package_dir), "_rocm_sdk_core", "bin"))
    dirs += list(hip_runtime_dirs)
    for var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = environ.get(var)
        if root:
            dirs.append(os.path.join(root, "bin"))
    if build_bin_dir:
        dirs.append(build_bin_dir)
    ordered: list[str] = []
    seen = set()
    for d in dirs:
        key = os.path.normcase(os.path.normpath(d)) if d else ""
        if d and key not in seen:
            seen.add(key)
            ordered.append(d)
    return ordered


def windows_dirs_to_register(
    candidates: Sequence[str], has_hip_runtime: Callable[[str], bool]
) -> list[str]:
    """The library's directory, plus the first candidate that holds a HIP runtime.

    Windows doesn't define the search order among os.add_dll_directory entries, so registering every
    candidate lets a second ROCm's amdhip64 win. Spike 3's override loaded the active environment's
    runtime that way.
    """
    if not candidates:
        return []
    runtime = next((d for d in candidates if has_hip_runtime(d)), None)
    if runtime is None:
        return list(candidates)
    return [candidates[0]] if runtime == candidates[0] else [candidates[0], runtime]


def _has_hip_runtime(directory: str) -> bool:
    try:
        return any(
            n.lower().startswith("amdhip64") and n.lower().endswith(".dll")
            for n in os.listdir(directory)
        )
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------
# Side effects: what the process has loaded, and the preload itself.
# ---------------------------------------------------------------------------------------------


def loaded_libraries() -> list[str]:
    """Paths of the shared libraries loaded in this process."""
    if sys.platform == "win32":
        return _windows_modules()
    paths: list[str] = []
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    p = parts[5].strip()
                    if p not in paths:
                        paths.append(p)
    except OSError:
        pass
    return paths


def _windows_modules() -> list[str]:
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    kernel32.GetModuleFileNameW.argtypes = [
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    proc = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    count = 1024
    while True:
        mods = (wintypes.HMODULE * count)()
        if not psapi.EnumProcessModules(
            proc, mods, ctypes.sizeof(mods), ctypes.byref(needed)
        ):
            return []
        if needed.value <= ctypes.sizeof(mods):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    out = []
    buf = ctypes.create_unicode_buffer(32768)
    for h in mods[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        if h and kernel32.GetModuleFileNameW(h, buf, len(buf)):
            out.append(buf.value)
    return out


def _rocm_sdk_find(name: str) -> list[str]:
    try:
        import rocm_sdk  # type: ignore[import-not-found]
    except ImportError:
        return []
    try:
        return [str(p) for p in rocm_sdk.find_libraries(name)]
    except (
        Exception
    ):  # noqa: BLE001 - a broken rocm_sdk install just means "not found here"
        return []


def load(build_lib_dir: str, build_bin_dir: str) -> LoadInfo:
    """Pick and preload libhipblaslt. Call before importing hipblaslt._core."""
    environ = os.environ
    platform = sys.platform
    override = environ.get(LIBRARY_ENV)
    before = loaded_libraries()

    if override:
        check_override_can_take_effect(override, before, platform)
    else:
        already = loaded_hipblaslt(before, platform)
        if already:
            # Torch (or something else) got here first. _core binds to that copy by name, and
            # preloading another one would put two hipBLASLt builds in the process.
            return LoadInfo(already[0], "already loaded", preloaded=False)

    chosen = choose_library(
        candidate_libraries(
            environ, platform, _rocm_sdk_find("hipblaslt"), build_lib_dir
        ),
        os.path.isfile,
    )
    # Companions already loaded from another build would make the preload fail with an
    # "undefined symbol" error, or, between ABI-compatible builds, load and mix silently.
    check_companions(chosen.path, before, platform)
    if platform == "win32":
        hip_dirs = [os.path.dirname(p) for p in _rocm_sdk_find("amdhip64")]
        candidates = [
            d
            for d in windows_dll_dirs(chosen.path, environ, hip_dirs, build_bin_dir)
            if os.path.isdir(d)
        ]
        for d in windows_dirs_to_register(candidates, _has_hip_runtime):
            os.add_dll_directory(d)
        ctypes.WinDLL(chosen.path)
    else:
        ctypes.CDLL(chosen.path, mode=ctypes.RTLD_GLOBAL)

    if override:
        _log.info("hipBLASLt loaded from %s (%s)", chosen.path, LIBRARY_ENV)
        warnings.warn(override_warning(override), HipblasLtLoadWarning, stacklevel=3)
    else:
        _log.info("hipBLASLt loaded from %s (%s)", chosen.path, chosen.source)
    return LoadInfo(chosen.path, chosen.source, preloaded=True)


def check_after_import(info: LoadInfo, core) -> None:
    """The check that needs _core loaded: companion libraries from the same build."""
    global _library, _built_version
    _library, _built_version = info, core._built_library_version()
    check_companions(info.path, loaded_libraries(), sys.platform)


_library: LoadInfo | None = None
_built_version: tuple[int, str] | None = None
_version_checked = False


def check_loaded_version(loaded: tuple[int, str]) -> None:
    """Warn, once per process, if the loaded hipBLASLt isn't the build the bindings expect.

    Called with the version and git revision the first Handle() reads. hipblasLtGetVersion and
    hipblasLtGetGitRevision need a handle, and creating one at import would allocate GPU memory
    in every process that imports the package; hipblasLtCreate also exits the process when that
    allocation fails.
    """
    global _version_checked
    if _version_checked or _library is None or _built_version is None:
        return
    _version_checked = True
    if os.environ.get(LIBRARY_ENV):
        return
    msg = version_mismatch_warning(_built_version, loaded, _library.path)
    if msg:
        warnings.warn(msg, HipblasLtLoadWarning, stacklevel=2)
