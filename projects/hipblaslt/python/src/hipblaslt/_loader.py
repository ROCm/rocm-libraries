# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Choose, preload and check the libhipblaslt that hipblaslt._core binds to.

The lookup order is HIPBLASLT_PY_LIBRARY, then the rocm-sdk wheels, then the build-time library, then
ROCM_PATH, then ROCM_HOME. The chosen library is preloaded by full path, so _core's NEEDED entry binds to it.

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
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

__all__ = [
    "LIBRARY_ENV",
    "build_runtime_warning",
    "Candidate",
    "candidate_libraries",
    "check_companions",
    "check_override_can_take_effect",
    "check_single_hip_runtime",
    "check_single_hipblaslt",
    "check_soname",
    "choose_library",
    "compatible_already_loaded",
    "dependencies_to_preload",
    "elf_dynamic",
    "ElfDynamic",
    "hip_runtime_to_preload",
    "HipblasLtLoadError",
    "HipblasLtLoadWarning",
    "loaded_hip_runtimes",
    "loaded_hipblaslt",
    "LoadInfo",
    "matches_lib_name",
    "not_the_build_warning",
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


def library_file_name(platform: str, lib_name: str = "") -> str:
    """The file name _core's dependency uses: the build's SONAME or DLL name, if recorded."""
    return lib_name or (
        "libhipblaslt.dll" if platform == "win32" else "libhipblaslt.so.1"
    )


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
    lib_name: str = "",
) -> list[Candidate]:
    """Every place to look, in lookup order. Existence is checked by choose_library."""
    name = library_file_name(platform, lib_name)
    out = []
    override = environ.get(LIBRARY_ENV)
    if override:
        out.append(Candidate(override, LIBRARY_ENV))
    out += [Candidate(p, "rocm-sdk") for p in rocm_sdk_libraries]
    if build_lib_dir:
        out.append(Candidate(os.path.join(build_lib_dir, name), "build"))
    for var in ("ROCM_PATH", "ROCM_HOME"):
        root = environ.get(var)
        if root:
            out.append(
                Candidate(os.path.join(_root_lib_dir(root, platform), name), var)
            )
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
    if (platform == "win32") == (sys.platform == "win32"):
        try:
            return os.path.samefile(a, b)
        except OSError:
            pass
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


def matches_lib_name(path: str, lib_name: str, platform: str) -> bool:
    """True if `path` is the library `lib_name` names (libhipblaslt.so.1.5.0 is libhipblaslt.so.1)."""
    b, want = _base(path, platform), lib_name.lower()
    return b == want or (platform != "win32" and b.startswith(want + "."))


def compatible_already_loaded(
    loaded: Iterable[str], platform: str, lib_name: str
) -> str | None:
    """The libhipblaslt already in the process that _core can bind to, or None if there's none.

    Raises if the one loaded has a different name, such as libhipblaslt.so.0 when _core needs
    libhipblaslt.so.1: _core would then load a second hipBLASLt beside it.
    """
    already = loaded_hipblaslt(loaded, platform)
    if not already:
        return None
    name = library_file_name(platform, lib_name)
    for p in already:
        if matches_lib_name(p, name, platform):
            return p
    raise HipblasLtLoadError(
        f"This process already uses hipBLASLt from {already[0]}, but these bindings need {name}. "
        "Loading both would put two hipBLASLt libraries in one process; import hipblaslt before "
        "the code that loads the other one, or rebuild the bindings against that hipBLASLt."
    )


def hip_runtime_to_preload(
    chosen: Candidate, build_hip_runtime: str, loaded: Iterable[str], platform: str
) -> str | None:
    """The build-time HIP runtime to load before a build-time libhipblaslt, or None.

    A hipBLASLt installed on its own (invoke build) can sit outside the ROCm whose HIP runtime
    it needs, with nothing on the search path pointing there.
    """
    if platform == "win32" or chosen.source != "build" or not build_hip_runtime:
        return None
    if any(_base(p, platform).startswith("libamdhip64.so") for p in loaded):
        return None
    return build_hip_runtime


def dependencies_to_preload(
    path: str,
    needed_of: Callable[[str], Sequence[str]],
    isfile: Callable[[str], bool],
    loaded: Iterable[str],
    platform: str,
) -> list[str]:
    """The libraries in `path`'s own directory that it needs, dependencies before dependents.

    `needed_of` gives a library's NEEDED names. A dependency the process already has under its
    name is left alone: loading a second copy by full path would put two in the process.
    """
    pm = _pathmod(platform)
    home = pm.dirname(path)
    loaded = list(loaded)
    order: list[str] = []
    seen = {pm.basename(path)}

    def visit(lib: str) -> None:
        for name in needed_of(lib):
            if name in seen:
                continue
            seen.add(name)
            dep = pm.join(home, name)
            if not isfile(dep) or any(
                matches_lib_name(p, name, platform) for p in loaded
            ):
                continue
            visit(dep)
            order.append(dep)

    visit(path)
    return order


def check_soname(path: str, soname: str, lib_name: str) -> None:
    """Raise before loading `path` if its SONAME isn't the name _core links against.

    Loading it anyway would map it, then load a second hipBLASLt for _core, and neither can be
    unloaded: retrying in the same process would still fail.
    """
    if soname and lib_name and soname != lib_name:
        raise HipblasLtLoadError(
            f"{path} is {soname}, but these bindings need {lib_name}. Point {LIBRARY_ENV} at a "
            f"{lib_name}, or rebuild the bindings against that hipBLASLt."
        )


def _is_hip_runtime(path: str, platform: str) -> bool:
    b = _base(path, platform)
    if platform == "win32":
        return b.startswith("amdhip64") and b.endswith(".dll")
    return b.startswith("libamdhip64.so")


def loaded_hip_runtimes(loaded: Iterable[str], platform: str) -> list[str]:
    """The HIP runtime files among the process's loaded libraries, deduplicated."""
    out: list[str] = []
    for p in loaded:
        if _is_hip_runtime(p, platform) and not any(
            _same_file(p, q, platform) for q in out
        ):
            out.append(p)
    return out


def check_single_hip_runtime(loaded: Iterable[str], platform: str) -> None:
    """Raise if the process has more than one HIP runtime, such as libamdhip64.so.5 and .so.7."""
    runtimes = loaded_hip_runtimes(loaded, platform)
    if len(runtimes) > 1:
        raise HipblasLtLoadError(
            f"This process has more than one HIP runtime: {', '.join(runtimes)}. hipBLASLt and "
            "the code that loaded the other one can't share a process; import hipblaslt first, "
            "or make sure one ROCm comes first on the search path."
        )


def build_runtime_warning(
    runtimes: Sequence[str], source: str, build_hip_runtime: str, platform: str
) -> str | None:
    """A warning when the build-time hipBLASLt runs on another HIP runtime than its build's."""
    if source != "build" or not build_hip_runtime or not runtimes:
        return None
    if _same_file(runtimes[0], build_hip_runtime, platform):
        return None
    return (
        f"The HIP runtime in this process is {runtimes[0]}, not {build_hip_runtime}, the one "
        "these bindings and their hipBLASLt were built with."
    )


def check_single_hipblaslt(path: str, loaded: Iterable[str], platform: str) -> None:
    """Raise unless the only libhipblaslt in the process is `path`."""
    found = loaded_hipblaslt(loaded, platform)
    others = [p for p in found if not _same_file(p, path, platform)]
    if others or not found:
        raise HipblasLtLoadError(
            f"After importing hipblaslt._core, the process has hipBLASLt from "
            f"{', '.join(found) or 'nowhere'}, but the bindings chose {path}."
        )


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
            f"builds: {', '.join(sorted(set(mixed)))}. Mixing builds can corrupt memory silently. "
            "Make sure one ROCm install comes first on the search path."
        )


def not_the_build_warning(
    chosen: str, source: str, build_library: str, build_exists: bool, platform: str
) -> str | None:
    """A warning when import didn't get the library the bindings were built against, and nobody
    asked for another one with the override."""
    if source == LIBRARY_ENV or not build_library:
        return None
    if build_exists and _same_file(chosen, build_library, platform):
        return None
    why = {
        "already loaded": "it was already loaded, probably by torch or another framework",
        "rocm-sdk": "the rocm-sdk wheels come first, so the process shares one copy with torch",
    }.get(source, f"it was found through {source}")
    built = (
        build_library if build_exists else f"{build_library}, which no longer exists"
    )
    return (
        f"hipBLASLt is loaded from {chosen} because {why}, not from {built}, the library these "
        f"bindings were built against. Set {LIBRARY_ENV} to choose a hipBLASLt explicitly."
    )


def override_warning(override: str, runtimes: Sequence[str] = ()) -> str:
    runtime = runtimes[0] if runtimes else "not loaded yet"
    return (
        f"{LIBRARY_ENV} is set, so hipBLASLt comes from {override}; its HIP runtime is {runtime}. "
        "Importing a framework built against a different ROCm afterward (torch, for example) "
        "will fail to load."
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
    candidate lets a second ROCm's amdhip64 win: an override's libhipblaslt.dll can load next to the
    active environment's HIP runtime.
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


_DT_NULL, _DT_NEEDED, _DT_STRTAB, _DT_STRSZ, _DT_SONAME, _DT_RPATH, _DT_RUNPATH = (
    0,
    1,
    5,
    10,
    14,
    15,
    29,
)


class ElfDynamic(NamedTuple):
    needed: list[str]
    rpath_governs: (
        bool  # DT_RPATH, which LD_LIBRARY_PATH can't override, and no DT_RUNPATH
    )
    soname: str


def elf_dynamic(path: str) -> ElfDynamic | None:
    """A 64-bit little-endian ELF's NEEDED names, search-path kind and SONAME. None for anything
    else, or a file that can't be read."""
    import struct

    try:
        with open(path, "rb") as f:
            head = f.read(64)
            if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
                return None
            (phoff,) = struct.unpack_from("<Q", head, 0x20)
            phentsize, phnum = struct.unpack_from("<HH", head, 0x36)
            f.seek(phoff)
            table = f.read(phentsize * phnum)
            segments, dynamic = [], None
            for i in range(phnum):
                p_type, _, offset, vaddr, _, filesz = struct.unpack_from(
                    "<IIQQQQ", table, i * phentsize
                )
                if p_type == 1:  # PT_LOAD
                    segments.append((vaddr, offset, filesz))
                elif p_type == 2:  # PT_DYNAMIC
                    dynamic = (offset, filesz)
            if dynamic is None:
                return None
            f.seek(dynamic[0])
            data = f.read(dynamic[1])
            entries = []
            for o in range(0, len(data) - 15, 16):
                tag, value = struct.unpack_from("<qQ", data, o)
                if tag == _DT_NULL:
                    break
                entries.append((tag, value))
            tags = dict(entries)
            strtab = next(
                (
                    off + tags[_DT_STRTAB] - va
                    for va, off, size in segments
                    if va <= tags.get(_DT_STRTAB, -1) < va + size
                ),
                None,
            )
            if strtab is None:
                return None
            f.seek(strtab)
            strings = f.read(tags.get(_DT_STRSZ, 0))
    except (OSError, struct.error):
        return None

    def string(at: int) -> str:
        return strings[at : strings.index(b"\0", at)].decode("utf-8", "replace")

    needed = [string(v) for t, v in entries if t == _DT_NEEDED]
    soname = string(tags[_DT_SONAME]) if _DT_SONAME in tags else ""
    return ElfDynamic(needed, _DT_RPATH in tags and _DT_RUNPATH not in tags, soname)


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


def _preload(path: str, environ: Mapping[str, str]) -> None:
    try:
        if sys.platform == "win32":
            ctypes.WinDLL(path)
        else:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        hint = ""
        if environ.get("LD_LIBRARY_PATH"):
            hint = (
                f" LD_LIBRARY_PATH is set ({environ['LD_LIBRARY_PATH']}), and a library there "
                "may come from a different ROCm; try again without it."
            )
        raise HipblasLtLoadError(f"Couldn't load {path}: {e}.{hint}") from e


def load(
    build_lib_dir: str,
    build_bin_dir: str,
    lib_name: str = "",
    build_hip_runtime: str = "",
) -> LoadInfo:
    """Pick and preload libhipblaslt. Call before importing hipblaslt._core."""
    environ = os.environ
    platform = sys.platform
    override = environ.get(LIBRARY_ENV)
    before = loaded_libraries()
    build_library = (
        os.path.join(build_lib_dir, library_file_name(platform, lib_name))
        if build_lib_dir
        else ""
    )

    def warn_if_not_the_build(path: str, source: str) -> None:
        msg = not_the_build_warning(
            path, source, build_library, os.path.isfile(build_library), platform
        )
        if msg:
            warnings.warn(msg, HipblasLtLoadWarning, stacklevel=4)

    if override:
        check_override_can_take_effect(override, before, platform)
    else:
        already = compatible_already_loaded(before, platform, lib_name)
        if already:
            # Torch (or something else) got here first. _core binds to that copy by name, and
            # preloading another one would put two hipBLASLt builds in the process.
            warn_if_not_the_build(already, "already loaded")
            return LoadInfo(already, "already loaded", preloaded=False)

    chosen = choose_library(
        candidate_libraries(
            environ, platform, _rocm_sdk_find("hipblaslt"), build_lib_dir, lib_name
        ),
        os.path.isfile,
    )
    if platform != "win32":
        dynamic = elf_dynamic(chosen.path)
        check_soname(chosen.path, dynamic.soname if dynamic else "", lib_name)
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
        _preload(chosen.path, environ)
    else:
        runtime = hip_runtime_to_preload(chosen, build_hip_runtime, before, platform)
        first = [runtime] if runtime and os.path.isfile(runtime) else []
        if environ.get("LD_LIBRARY_PATH") and dynamic and not dynamic.rpath_governs:
            # LD_LIBRARY_PATH is searched before DT_RUNPATH, so another ROCm there would supply
            # hipBLASLt's companions. Load them from hipBLASLt's own directory first.
            first += dependencies_to_preload(
                chosen.path,
                lambda p: (elf_dynamic(p) or ElfDynamic([], False, "")).needed,
                os.path.isfile,
                before + first,
                platform,
            )
        for lib in first + [chosen.path]:
            _preload(lib, environ)

    after = loaded_libraries()
    check_single_hip_runtime(after, platform)
    runtimes = loaded_hip_runtimes(after, platform)
    if override:
        _log.info("hipBLASLt loaded from %s (%s)", chosen.path, LIBRARY_ENV)
        warnings.warn(
            override_warning(override, runtimes), HipblasLtLoadWarning, stacklevel=3
        )
    else:
        _log.info("hipBLASLt loaded from %s (%s)", chosen.path, chosen.source)
        warn_if_not_the_build(chosen.path, chosen.source)
        msg = build_runtime_warning(
            runtimes, chosen.source, build_hip_runtime, platform
        )
        if msg:
            warnings.warn(msg, HipblasLtLoadWarning, stacklevel=3)
    return LoadInfo(chosen.path, chosen.source, preloaded=True)


def check_after_import(info: LoadInfo, core) -> None:
    """The checks that need _core loaded: one hipBLASLt and one HIP runtime, with companions
    from hipBLASLt's build."""
    global _library, _built_version
    _library, _built_version = info, core._built_library_version()
    loaded = loaded_libraries()
    check_single_hipblaslt(info.path, loaded, sys.platform)
    check_single_hip_runtime(loaded, sys.platform)
    check_companions(info.path, loaded, sys.platform)


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
    # An override already warned at import, and a different build is what the caller asked for.
    if _library.source == LIBRARY_ENV:
        return
    msg = version_mismatch_warning(_built_version, loaded, _library.path)
    if msg:
        warnings.warn(msg, HipblasLtLoadWarning, stacklevel=2)
