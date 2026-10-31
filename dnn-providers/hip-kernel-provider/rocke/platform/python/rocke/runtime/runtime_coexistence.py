# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Independent compiler and HIP runtime discovery.

COMGR uses an explicit override or the installed SDK distribution's owned file.
HIP keeps its runtime ownership policy, honoring an already imported torch and
then SDK/system installations. Discovery never imports torch or SDK modules.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Any, List, Optional


_IS_WINDOWS = sys.platform == "win32"


# Read by :func:`_mapped_lib`; a module constant so tests can point it at a fixture.
_PROC_MAPS = "/proc/self/maps"


def _mapped_elf_lib_paths(stem: str) -> List[str]:
    """Loader names survive unlink/NFS renaming of an already mapped library."""
    import ctypes

    class Info(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("name", ctypes.c_char_p)]

    library = ctypes.CDLL(None)
    iterate = getattr(library, "dl_iterate_phdr", None)
    if iterate is None:
        return []
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(Info), ctypes.c_size_t, ctypes.c_void_p
    )
    paths: List[str] = []
    pattern = re.compile(rf"lib{re.escape(stem)}\.so(?:\.\d+)*")
    mappings = []
    try:
        with open(_PROC_MAPS) as fh:
            for line in fh:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) == 6:
                    start, end = fields[0].split("-", 1)
                    mappings.append((int(start, 16), int(end, 16), fields[5]))
    except FileNotFoundError:
        pass

    @callback_type
    def collect(info, size, data):
        if size >= ctypes.sizeof(Info) and info.contents.name:
            path = os.fsdecode(info.contents.name)
            if pattern.fullmatch(os.path.basename(path)):
                # The original loader name may refer to a replacement inode
                # after an SDK uninstall/reinstall. Match its actual mapping,
                # including NFS's .nfs name or the Linux deleted-file marker.
                actual = next(
                    (
                        name
                        for start, end, name in mappings
                        if start <= info.contents.address < end
                    ),
                    path + " (identity unavailable)",
                )
                if actual not in paths:
                    paths.append(actual)
        return 0

    iterate.argtypes = [callback_type, ctypes.c_void_p]
    iterate.restype = ctypes.c_int
    iterate(collect, None)
    return paths


def _mapped_lib_paths(stem: str) -> List[str]:
    """Every mapped library with this filename, independent of torch ownership."""
    if _IS_WINDOWS:
        # Platform mocks on POSIX have no Windows loader to inspect.
        if os.name != "nt":
            return []
        import ctypes
        from ctypes import wintypes

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.EnumProcessModules.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HMODULE),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        psapi.GetModuleFileNameExW.argtypes = [
            wintypes.HANDLE,
            wintypes.HMODULE,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        process = kernel.GetCurrentProcess()
        count = 256
        while True:
            modules = (wintypes.HMODULE * count)()
            needed = wintypes.DWORD()
            if not psapi.EnumProcessModules(
                process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
            ):
                raise OSError(
                    ctypes.get_last_error(), "cannot inspect mapped runtime libraries"
                )
            if needed.value <= ctypes.sizeof(modules):
                break
            count = needed.value // ctypes.sizeof(wintypes.HMODULE)
        names = []
        for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            path = ctypes.create_unicode_buffer(32768)
            size = psapi.GetModuleFileNameExW(process, module, path, len(path))
            if not size or size >= len(path):
                raise OSError(
                    ctypes.get_last_error(), "cannot identify a mapped runtime library"
                )
            if re.fullmatch(
                rf"{re.escape(stem)}\d*\.dll",
                os.path.basename(path.value),
                re.IGNORECASE,
            ):
                names.append(path.value)
        return names
    name_re = re.compile(rf"lib{re.escape(stem)}\.so(?:\.\d+)*")
    paths = list(_mapped_elf_lib_paths(stem))
    try:
        with open(_PROC_MAPS) as fh:
            for line in fh:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                path = fields[5]
                filename = os.path.basename(path.removesuffix(" (deleted)"))
                if name_re.fullmatch(filename) and path not in paths:
                    paths.append(path)
    except FileNotFoundError:
        # Keep loader evidence on POSIX systems without procfs.
        return paths
    return paths


def _mapped_lib(stem: str, *, require_unique: bool = False) -> Optional[str]:
    """Find a mapped library without guessing between distinct runtime copies.

    A library has several entries in /proc/self/maps. If distinct copies are
    mapped, prefer the one owned by imported torch or its active ROCm SDK
    wheel. Address order does not identify which copy torch loaded. Set
    require_unique to reject *all* multiple-copy cases, without changing that
    preference. Linux only; unreadable maps return None.
    """
    if _IS_WINDOWS:
        return None
    name_re = re.compile(rf"lib{re.escape(stem)}\.so(\.\d+)*")
    mapped: List[str] = []
    try:
        with open(_PROC_MAPS) as fh:
            for line in fh:
                # address perms offset dev inode [path]; the path may hold spaces.
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                path = fields[5]
                if name_re.fullmatch(os.path.basename(path)) and path not in mapped:
                    mapped.append(path)
    except OSError:
        return None
    if len(mapped) == 1:
        return mapped[0]
    if not mapped or require_unique:
        return None

    torch_mod = sys.modules.get("torch")
    torch_file = getattr(torch_mod, "__file__", None) if torch_mod else None
    roots = [os.path.realpath(os.path.dirname(torch_file))] if torch_file else []
    if "rocm_sdk" in sys.modules:
        roots.extend(os.path.realpath(loc) for loc in _rocm_sdk_core_dirs())
    preferred = []
    for path in mapped:
        resolved = os.path.realpath(path)
        if any(
            resolved == root or resolved.startswith(root + os.sep) for root in roots
        ):
            preferred.append(path)
    return preferred[0] if len(preferred) == 1 else None


def _rocm_sdk_core_dirs() -> List[str]:
    """Package dirs of the ``rocm-sdk-core`` wheel's ``_rocm_sdk_core``, or [].

    Located with ``importlib.util.find_spec``, which does not import the package.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("_rocm_sdk_core")
    except Exception:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return list(spec.submodule_search_locations)


def _rocm_sdk_core_lib(stem: str) -> Optional[str]:
    """``lib<stem>.so*`` from the ``rocm-sdk-core`` wheel's ``_rocm_sdk_core/lib``.

    TheRock torch wheels (multi-arch ``whl-next`` and the per-target-family
    indexes alike) do not ship the HIP runtime or comgr in ``torch/lib``: they
    sit in ``_rocm_sdk_core/lib`` with only a SONAME (``libamdhip64.so.7``,
    ``libamd_comgr.so.3``), and ``import torch`` preloads them through
    ``rocm_sdk.initialize_process``. Returns the shortest matching name (the
    bare ``.so`` if present, else the SONAME) or None. POSIX only; Windows uses
    :func:`_rocm_sdk_dll`.
    """
    if _IS_WINDOWS:
        return None
    for loc in _rocm_sdk_core_dirs():
        matches = glob.glob(os.path.join(loc, "lib", f"lib{stem}.so*"))
        if matches:
            return min(matches, key=lambda p: (len(p), p))
    return None


def _torch_bundled_lib(stem: str) -> Optional[str]:
    """Path of the ``lib<stem>`` torch uses, if torch is in this process.

    ROCm torch wheels bring their own ``libamdhip64`` / ``libamd_comgr`` and
    load them as a side effect of ``import torch``; they become torch's HIP
    runtime instance. A second copy of HIP loaded by rocke from
    ``/opt/rocm/lib`` is a *different* runtime instance with disjoint state --
    modules loaded via one are invisible to ``hipModuleGetFunction`` from the
    other, surfacing as ``hipError(500) named symbol not found`` even when the
    HSACO is well-formed and the symbol is present in its ELF.

    To keep both halves of the process on one runtime, return, in order:

    1. a uniquely selected mapped copy (:func:`_mapped_lib`); when several
       copies exist, only one owned by torch or its active ROCm SDK wheel wins;
    2. ``<torch>/lib/lib<stem>.so`` (wheels that bundle the libs in torch);
    3. ``_rocm_sdk_core/lib/lib<stem>.so*`` (see :func:`_rocm_sdk_core_lib`),
       only when ``rocm_sdk`` is imported, i.e. the torch is TheRock's. A
       CUDA, CPU or older ROCm torch in a venv that merely has
       ``rocm-sdk-core`` installed does not use those libs.

    Avoids importing torch as a side effect: only honors a torch that is
    *already* in :data:`sys.modules`.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    torch_file = getattr(torch_mod, "__file__", None)
    if not torch_file:
        return None
    libdir = os.path.join(os.path.dirname(torch_file), "lib")
    if _IS_WINDOWS:
        # ROCm-for-Windows torch wheels (TheRock / AMD nightlies) bundle
        # ``amdhip64.dll`` and a version-stamped ``amd_comgr*.dll`` (no
        # ``lib`` prefix). Prefer an exact match, else glob the versioned
        # comgr name.
        direct = os.path.join(libdir, f"{stem}.dll")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(libdir, f"{stem}*.dll")))
        return matches[0] if matches else None
    mapped = _mapped_lib(stem)
    if mapped is not None:
        return mapped
    candidate = os.path.join(libdir, f"lib{stem}.so")
    if os.path.exists(candidate):
        return candidate
    if "rocm_sdk" not in sys.modules:
        return None
    return _rocm_sdk_core_lib(stem)


def _rocm_sdk_dll(stem: str) -> Optional[str]:
    """Locate a ROCm runtime DLL shipped by the ``rocm-sdk-core`` wheel.

    ROCm-for-Windows torch nightlies (AMD's gfx1151 index / TheRock) put
    the HIP runtime and comgr in ``_rocm_sdk_core/bin`` with a version
    suffix (e.g. ``amdhip64_7.dll``, ``amd_comgr0702.dll``) rather than in
    ``torch/lib``. Returns the first match for ``<stem>*.dll`` or None.
    """
    if not _IS_WINDOWS:
        return None
    for loc in _rocm_sdk_core_dirs():
        bindir = os.path.join(loc, "bin")
        direct = os.path.join(bindir, f"{stem}.dll")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(bindir, f"{stem}*.dll")))
        if matches:
            return matches[0]
    return None


def _version_key(path: str) -> Any:
    """Sort key that orders ROCm install dirs newest-first.

    A plain string sort puts ``rocm-7.10`` *before* ``rocm-7.2`` (because
    ``'1' < '2'`` lexically) -- wrong for picking the newest toolkit. Extract
    every run of digits from the path and compare them as an integer tuple so
    ``7.10`` > ``7.2``. Non-numeric paths sort last. Callers reverse the result
    to get descending (newest-first) order.
    """
    nums = tuple(int(n) for n in re.findall(r"\d+", path))
    return (len(nums) > 0, nums)


def _rocm_root_libdirs() -> List[str]:
    """Existing ``<rocm>/lib`` directories discovered WITHOUT importing torch.

    This is the crux of removing rocke's accidental torch dependency. The ROCm
    torch wheel bundles ``libamdhip64.so`` / ``libamd_comgr.so`` inside
    ``torch/lib`` and, as a side effect of ``import torch``, drops that
    directory onto the process's loader search path -- which is the *only*
    reason a bare ``ctypes.CDLL("libamd_comgr.so")`` used to succeed here. A
    library must never ``import torch`` to obtain that side effect (it would
    invert the dependency and drag a multi-hundred-MB wheel into a pure-IR
    process), so we discover a real ROCm install directly instead.

    Priority, newest-version-first within each tier:
      1. ``$ROCM_PATH`` / ``$ROCM_HOME`` -> ``<root>/lib`` (operator override).
      2. Globbed real install layouts. On a packaged ROCm 7.2 there is often no
         ``/opt/rocm/lib`` with the runtime in it; the libs live under a
         versioned ``core-<X>/lib`` subdir (e.g.
         ``/opt/rocm-7.2.0/core-7.13/lib``). Cover both ``/opt/rocm*/lib`` and
         ``/opt/rocm*/core-*/lib``.

    Returns directories that exist, de-duplicated, in resolution order.
    """
    roots: List[str] = []
    seen: set = set()

    def _add(d: str) -> None:
        # De-dupe on the resolved real path so a symlinked root and its target
        # don't both get probed; keep the original string for readable candidate
        # paths.
        if not d:
            return
        rp = os.path.realpath(d)
        if rp and rp not in seen and os.path.isdir(rp):
            seen.add(rp)
            roots.append(d)

    # Tier 1: explicit env roots win over any globbed install.
    for env in ("ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(env)
        if root:
            _add(os.path.join(root, "lib"))

    # Tier 2: glob real install trees, newest version first. ``core-*/lib`` is
    # listed ahead of plain ``lib`` because that is where a packaged install
    # actually keeps the runtime .so's.
    for pattern in ("/opt/rocm*/core-*/lib", "/opt/rocm*/lib"):
        for d in sorted(glob.glob(pattern), key=_version_key, reverse=True):
            _add(d)
    return roots


def _candidate_lib_paths(stem: str, env_var: str, sonames: List[str]) -> List[str]:
    """COMGR: explicit override or SDK only. HIP: existing runtime policy."""
    if stem == "amd_comgr":
        from ._sdk_comgr import candidates

        override = os.environ.get(env_var)
        return [override] if override else candidates(windows=_IS_WINDOWS)
    paths: List[str] = []
    override = os.environ.get(env_var)
    if override:
        paths.append(override)
    bundled = _torch_bundled_lib(stem)
    if bundled is not None:
        paths.append(bundled)
    sdk = _rocm_sdk_dll(stem)
    if sdk is not None and sdk not in paths:
        paths.append(sdk)
    if _IS_WINDOWS:
        # ROCm-for-Windows / HIP SDK install: ``%HIP_PATH%\bin`` /
        # ``%ROCM_PATH%\bin`` / ``%ROCM_HOME%\bin`` then the bare DLL name
        # (resolved via the default DLL search path). The comgr DLL carries a
        # version suffix, so glob it.
        for root_env in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
            root = os.environ.get(root_env)
            if not root:
                continue
            bindir = os.path.join(root, "bin")
            paths.append(os.path.join(bindir, f"{stem}.dll"))
            paths.extend(sorted(glob.glob(os.path.join(bindir, f"{stem}*.dll"))))
        paths.append(f"{stem}.dll")
        return paths
    # POSIX: each discovered ROCm ``<root>/lib`` contributes the bare .so plus
    # SONAME-suffixed variants, newest install first.
    for libdir in _rocm_root_libdirs():
        paths.append(os.path.join(libdir, f"lib{stem}.so"))
        for soname in sonames:
            paths.append(os.path.join(libdir, f"lib{stem}.so.{soname}"))
    paths.append(f"lib{stem}.so")
    return paths


def _add_dll_dir(path: str) -> None:
    """On Windows, register a resolved DLL's own directory so its
    dependent DLLs (bundled alongside it in ``torch/lib`` or the HIP SDK
    ``bin``) are found by the loader. No-op off Windows or for bare names.
    """
    if not _IS_WINDOWS:
        return
    d = os.path.dirname(path)
    if d and os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
