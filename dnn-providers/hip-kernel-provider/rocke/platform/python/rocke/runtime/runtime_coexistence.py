# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Discover COMGR compilers separately from HIP runtime ownership.

HIP follows the runtime owned by an already imported torch, because module
handles and allocations belong to that runtime. Default COMGR selection uses
an optional installed SDK wheel before ROCm installations; torch compiler
selection is an explicit opt-in through ``comgr.prefer_bundled_lib()``.
Discovery never imports torch or initializes the SDK process. COMGR's loader
checks mapped compiler copies before opening any candidate on Linux.
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


def _rocm_sdk_core_dirs() -> list[str]:
    """Locate active SDK payloads from distribution records, without importing.

    The binary package name can carry a target/distribution suffix. RECORD
    paths are relative to the active installation, so relocation does not need
    the original wheel build path. The unsuffixed spec is a fallback for older
    packages without usable distribution records.
    """
    from importlib import metadata, util

    roots: list[str] = []
    try:
        distributions = sorted(
            (
                dist
                for dist in metadata.distributions()
                if re.fullmatch(
                    r"rocm-sdk-core(?:-[a-z0-9]+)*",
                    re.sub(r"[-_.]+", "-", dist.metadata.get("Name", "")).lower(),
                )
            ),
            key=lambda dist: dist.metadata.get("Name", ""),
        )
        for dist in distributions:
            for entry in dist.files or ():
                parts = entry.parts
                if parts and re.fullmatch(
                    r"_rocm_sdk_core(?:_[A-Za-z0-9]+)*", parts[0]
                ):
                    root = os.path.abspath(dist.locate_file(parts[0]))
                    if os.path.isdir(root) and root not in roots:
                        roots.append(root)
    except (OSError, ValueError):
        pass
    if roots:
        return roots
    try:
        spec = util.find_spec("_rocm_sdk_core")
    except (ImportError, ValueError, AttributeError):
        return []
    return list(spec.submodule_search_locations or ()) if spec else []


def _sdk_comgr_candidates() -> list[str]:
    """SDK ELF names: unversioned first, then numeric SONAME/full versions.

    Shorter SONAMEs precede their full version. Alias files are attempted once,
    retaining the readable requested spelling. An unloadable candidate does
    not remove other distinct files from the loader's fallback list. Relative
    ELF RPATHs in the complete SDK payload resolve its dependencies directly.
    """
    paths: list[str] = []
    identities: set[tuple[int, int]] = set()
    for root in _rocm_sdk_core_dirs():
        libdir = os.path.join(root, "lib")
        matches = glob.glob(os.path.join(libdir, "libamd_comgr.so*"))
        valid = [
            path
            for path in matches
            if re.fullmatch(r"libamd_comgr\.so(?:\.\d+)*", os.path.basename(path))
            and os.path.isfile(path)
        ]

        def order(path: str) -> tuple:
            version = tuple(int(n) for n in re.findall(r"\d+", os.path.basename(path)))
            return len(version), version, path

        for path in sorted(valid, key=order):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity not in identities:
                identities.add(identity)
                paths.append(path)
    return paths


def _mapped_comgr_files() -> list[tuple[str, int, int]] | None:
    """All Linux COMGR mapping identities, including deleted payloads.

    None means maps were unreadable; it is not evidence of an empty process.
    Device/inode records let the loader recognize hardlinks and deleted files
    without interpreting virtual-address order as compiler ownership.
    """
    if _IS_WINDOWS:
        return None
    records: list[tuple[str, int, int]] = []
    try:
        with open(_PROC_MAPS) as stream:
            for line in stream:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                path = fields[5].removesuffix(" (deleted)")
                if not re.fullmatch(
                    r"libamd_comgr\.so(?:\.\d+)*", os.path.basename(path)
                ):
                    continue
                try:
                    major, minor = (int(n, 16) for n in fields[3].split(":"))
                    record = (fields[5], os.makedev(major, minor), int(fields[4]))
                except ValueError:
                    continue
                if record not in records:
                    records.append(record)
    except OSError:
        return None
    return records


def _in_rocm_sdk_core(path: str) -> bool:
    """True if ``path`` lies inside the ``_rocm_sdk_core`` package (symlinks resolved)."""
    rp = os.path.realpath(path)
    for loc in _rocm_sdk_core_dirs():
        root = os.path.realpath(loc)
        if rp == root or rp.startswith(root + os.sep):
            return True
    return False


def _rocm_sdk_core_release() -> Optional[tuple]:
    """``(major, minor)`` ROCm release of TheRock's ``rocm-sdk-core`` wheel, or None.

    ``rocm_sdk.__version__`` (e.g. ``'10.1.0a20260822'``) when ``rocm_sdk`` is
    already imported (TheRock's ``import torch`` imports it), else the installed
    ``rocm-sdk-core`` distribution's version. Neither imports anything. The
    libs in ``_rocm_sdk_core`` belong to this release; ``torch.version.hip`` on
    those wheels is the HIP version, a different scheme.
    """
    version = getattr(sys.modules.get("rocm_sdk"), "__version__", None)
    if not version:
        try:
            from importlib import metadata

            version = metadata.version("rocm-sdk-core")
        except Exception:
            return None
    nums = re.findall(r"\d+", str(version))
    if len(nums) < 2:
        return None
    return (int(nums[0]), int(nums[1]))


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
        owned = os.path.realpath(os.path.dirname(torch_file))
        if (
            stem != "amd_comgr"
            or os.path.realpath(mapped).startswith(owned + os.sep)
            or ("rocm_sdk" in sys.modules and _in_rocm_sdk_core(mapped))
        ):
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


def _torch_rocm_version(bundled: Optional[str] = None) -> Optional[tuple]:
    """``(major, minor)`` ROCm **release** of torch's bundled comgr, or None.

    Informational metadata only; compiler compatibility uses loaded evidence.
    The source follows the bundled lib's location, the same way
    :func:`rocke.runtime.comgr.resolved_lib_rocm_version` classifies a path:

    - inside TheRock's ``_rocm_sdk_core``: the wheel's release
      (:func:`_rocm_sdk_core_release`). ``torch.version.hip`` on those wheels is
      the *HIP* version (``'7.16.26332'`` on ROCm 10.1), a different scheme.
    - anywhere else: ``torch.version.hip`` (e.g. ``'6.3.42134-a9a80e791'``),
      where the HIP version is the ROCm release.

    Only consults modules that are *already* imported -- never imports them.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    if bundled is not None and _in_rocm_sdk_core(bundled):
        return _rocm_sdk_core_release()
    version = getattr(getattr(torch_mod, "version", None), "hip", None)
    if not version:
        return None
    nums = re.findall(r"\d+", str(version))
    if len(nums) < 2:
        return None
    return (int(nums[0]), int(nums[1]))


_ROCM_RELEASE_DIR_RE = re.compile(r"^rocm-(\d+)\.(\d+)")


def _rocm_version_from_libdir(libdir: str) -> Optional[tuple]:
    """``(major, minor)`` ROCm **release** version for a ``<rocm>/lib`` path.

    Matches only a ``rocm-X.Y[.Z]`` directory component, and scans the whole
    path rather than just the parent of ``lib``. Two layouts make that
    necessary, and each breaks a simpler rule:

    - The distro default reaches a packaged install through an *unversioned*
      symlink (``ROCM_PATH=/opt/rocm`` -> ``/opt/rocm-7.2.3``), and
      :func:`_rocm_root_libdirs` deliberately returns the original string so
      candidate paths stay readable in errors. So the resolved target has to be
      considered too, else the version reads as unknown and
      package metadata becomes unavailable.
    - A packaged ROCm keeps its runtime under a versioned *component* subdir,
      ``/opt/rocm-7.2.0/core-7.13/lib``. ``core-7.13`` is a component version,
      not a release: taking the parent of ``lib`` would report misleading
      package metadata. Compiler compatibility comes from the loaded compiler,
      independently of this informational release number.

    Returns None when no release component is present.
    """
    for candidate in (libdir, os.path.realpath(libdir)):
        for part in candidate.split(os.sep):
            m = _ROCM_RELEASE_DIR_RE.match(part)
            if m:
                return (int(m.group(1)), int(m.group(2)))
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


def _comgr_candidate_paths(sonames: list[str], bundled: list[str] = ()) -> list[str]:
    """COMGR: override, explicit bundle opt-in, SDK, roots, then bare name."""
    paths: list[str] = []
    override = os.environ.get("ROCKE_COMGR_LIB")
    if override:
        paths.append(override)
    paths.extend(bundled)
    if _IS_WINDOWS:
        sdk = _rocm_sdk_dll("amd_comgr")
        if sdk:
            paths.append(sdk)
        for name in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
            root = os.environ.get(name)
            if root:
                bindir = os.path.join(root, "bin")
                paths.append(os.path.join(bindir, "amd_comgr.dll"))
                paths.extend(sorted(glob.glob(os.path.join(bindir, "amd_comgr*.dll"))))
        paths.append("amd_comgr.dll")
    else:
        paths.extend(_sdk_comgr_candidates())
        for libdir in _rocm_root_libdirs():
            paths.append(os.path.join(libdir, "libamd_comgr.so"))
            paths.extend(
                os.path.join(libdir, f"libamd_comgr.so.{name}") for name in sonames
            )
        paths.append("libamd_comgr.so")
    return list(dict.fromkeys(paths))


def _candidate_lib_paths(stem: str, env_var: str, sonames: List[str]) -> List[str]:
    """COMGR has independent compiler policy; HIP preserves runtime ownership.

    HIP: override, already-imported torch, SDK on Windows, ROCm roots, bare name.
    """
    if stem == "amd_comgr":
        return _comgr_candidate_paths(sonames)
    paths: List[str] = []
    override = os.environ.get(env_var)
    if override:
        paths.append(override)
    bundled = _torch_bundled_lib(stem)
    if bundled is not None:
        paths.append(bundled)
    sdk = _rocm_sdk_dll(stem)
    if sdk is not None:
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
