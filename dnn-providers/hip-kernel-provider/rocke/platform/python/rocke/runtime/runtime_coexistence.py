# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which ROCm runtime do we bind to -- and whose?

Both the HIP (`hip_module`) and comgr (`comgr`) ctypes wrappers must
resolve the *same* loaded runtime instance. The crux is process-level
coexistence with a host that may already own a ROCm runtime:

    A ROCm PyTorch wheel bundles ``libamdhip64.so`` / ``libamd_comgr.so``
    inside ``torch/lib`` and, as a side effect of ``import torch``, loads
    them into the process. A *second* copy of HIP loaded by rocke from
    ``/opt/rocm`` is a **different runtime instance** with disjoint state
    -- a module loaded via one is invisible to ``hipModuleGetFunction``
    from the other, surfacing as ``hipError(500) named symbol not found``
    even when the HSACO is well-formed. This is a loader / runtime-instance
    phenomenon (two separate HSA runtime inits, two handle tables), not a
    context-binding one.

So when torch is already in the process we prefer *its* bundled ``.so``,
sharing one loaded HIP/comgr runtime instance across both halves of the
process. A library must never ``import torch`` merely to obtain that side
effect (it would invert the dependency and drag a multi-hundred-MB wheel
into a pure-IR process): we only honor a torch that is *already* imported
(via :data:`sys.modules`) and otherwise discover a real ROCm install
directly. The sole sanctioned exception is the explicit
:func:`rocke.runtime.comgr.prefer_bundled_lib` entrypoint hook, which
imports torch deliberately to pin the bundled comgr before lowering.

This module owns only *resolution* -- candidate path discovery + the
Windows DLL-directory registration. The actual ``dlopen`` (per-family
``_load_lib``), the HIP primary-context binding (``_ensure_hip_init``),
and device-property introspection (``get_device_arch``) live with their
respective runtime wrappers.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Any, List, Optional


_IS_WINDOWS = sys.platform == "win32"


def _torch_bundled_lib(stem: str) -> Optional[str]:
    """Return path to ``<torch>/lib/lib<stem>.so`` if torch is in this process.

    Newer PyTorch+ROCm wheels (e.g. torch>=2.12 / ROCm 7.2) ship their
    own ``libamdhip64.so`` and ``libamd_comgr.so`` inside the wheel's
    ``torch/lib/`` directory. When torch is imported, those bundled
    libraries become torch's loaded HIP runtime instance.
    A second copy of HIP loaded by rocke from ``/opt/rocm/lib`` is a
    *different* runtime instance with disjoint state — modules loaded
    via one are invisible to ``hipModuleGetFunction`` from the other,
    surfacing as ``hipError(500) named symbol not found`` even when the
    HSACO is well-formed and the symbol is present in its ELF.

    To keep both halves of the process talking to the same HIP/comgr
    runtime, prefer torch's bundled lib when torch is already imported.
    Avoids importing torch as a side effect: only honors a torch that
    is *already* in :data:`sys.modules`.
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
    candidate = os.path.join(libdir, f"lib{stem}.so")
    return candidate if os.path.exists(candidate) else None


def _rocm_sdk_dll(stem: str) -> Optional[str]:
    """Locate a ROCm runtime DLL shipped by the ``rocm-sdk-core`` wheel.

    ROCm-for-Windows torch nightlies (AMD's gfx1151 index / TheRock) put
    the HIP runtime and comgr in ``_rocm_sdk_core/bin`` with a version
    suffix (e.g. ``amdhip64_7.dll``, ``amd_comgr0702.dll``) rather than in
    ``torch/lib``. Returns the first match for ``<stem>*.dll`` or None.
    """
    if not _IS_WINDOWS:
        return None
    try:
        import importlib.util

        spec = importlib.util.find_spec("_rocm_sdk_core")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for loc in spec.submodule_search_locations:
        bindir = os.path.join(loc, "bin")
        direct = os.path.join(bindir, f"{stem}.dll")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(bindir, f"{stem}*.dll")))
        if matches:
            return matches[0]
    return None


def _torch_rocm_version() -> Optional[tuple]:
    """``(major, minor)`` of the ROCm that torch was built against, or None.

    Read off ``torch.version.hip`` (e.g. ``'6.3.42134-a9a80e791'``). Only
    consults a torch that is *already* imported -- never imports it.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    hip = getattr(getattr(torch_mod, "version", None), "hip", None)
    if not hip:
        return None
    nums = re.findall(r"\d+", str(hip))
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
      :func:`_torch_comgr_is_stale` silently disables itself.
    - A packaged ROCm keeps its runtime under a versioned *component* subdir,
      ``/opt/rocm-7.2.0/core-7.13/lib``. ``core-7.13`` is a component version,
      not a release: taking the parent of ``lib`` would yield ``(7, 13)`` and
      compare it against ``torch.version.hip``, which reports the release. A
      torch on ROCm 7.10 would then look *older* than a 7.2 install, because
      ``(7, 10) < (7, 13)`` -- and comgr would be demoted backwards.

    Returns None when no release component is present; unknown must stay
    unknown, since :func:`_torch_comgr_is_stale` keeps the historical
    resolution order rather than guessing.
    """
    for candidate in (libdir, os.path.realpath(libdir)):
        for part in candidate.split(os.sep):
            m = _ROCM_RELEASE_DIR_RE.match(part)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    return None


def _newest_rocm_root_version() -> Optional[tuple]:
    """``(major, minor)`` of the ROCm install that would be loaded, or None.

    First parseable entry in *resolution* order, not the numeric maximum across
    all installs: the point of comparison is the lib this process would
    actually load, and an operator's ``ROCM_PATH`` wins that race even when a
    newer tree exists beside it.
    """
    for libdir in _rocm_root_libdirs():
        version = _rocm_version_from_libdir(libdir)
        if version is not None:
            return version
    return None


def _torch_comgr_is_stale() -> bool:
    """True when torch's bundled comgr is *older* than the newest ROCm install.

    The resolution order below prefers torch's bundled lib so both halves of the
    process share one runtime, and the surrounding comments call that lib "the
    newest". That is an assumption, not an invariant: a venv pinned to an older
    ROCm torch on a box with a newer system ROCm inverts it, and then the
    bundled comgr does not know the target ISA at all -- every compile dies with
    ``set_isa: INVALID_ARGUMENT`` for any arch newer than torch's ROCm, which is
    a confusing failure a long way from its cause.

    Only demotes when BOTH versions are positively known and torch's is strictly
    older; anything unknown keeps the historical order. Deliberately scoped to
    comgr by its one caller: comgr is a compile-only library, while loading a
    second *HIP runtime* beside torch's would be a genuine hazard.
    """
    torch_v = _torch_rocm_version()
    root_v = _newest_rocm_root_version()
    if torch_v is None or root_v is None:
        return False
    return torch_v < root_v


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
    """Resolution order for the HIP runtime / comgr shared libraries.

    Order:
      1. ``$ROCKE_HIP_LIB`` / ``$ROCKE_COMGR_LIB`` (explicit override, full path).
      2. ``<torch>/lib/lib<stem>.so`` if torch is *already* imported -- an
         opportunistic fast-path only (see :func:`_torch_bundled_lib`); we never
         import torch to populate it. For ``amd_comgr`` this tier is skipped
         when the bundled comgr is demonstrably older than the newest ROCm
         install (see :func:`_torch_comgr_is_stale`), because a stale comgr
         rejects every ISA newer than its own ROCm and would shadow a system
         comgr that handles the target fine.
      3. A real ROCm install discovered without torch (see
         :func:`_rocm_root_libdirs`): ``$ROCM_PATH``/``$ROCM_HOME`` then globbed
         ``/opt/rocm*`` trees, newest version first, each with the bare ``.so``
         and the requested SONAME variants.
      4. Bare ``lib<stem>.so`` for the dynamic linker's search path -- last
         resort. Historically this was the *only* non-torch candidate, which is
         why a torch-less process failed with ``cannot load libamd_comgr.so``:
         nothing had put the lib on the loader path. Tier 3 fixes that.
    """
    paths: List[str] = []
    override = os.environ.get(env_var)
    if override:
        paths.append(override)
    bundled = _torch_bundled_lib(stem)
    # A stale bundled comgr is demoted below the ROCm installs rather than
    # dropped: if none of them load, it is still better than nothing.
    _demote_bundled = (
        bundled is not None and stem == "amd_comgr" and _torch_comgr_is_stale()
    )
    if bundled is not None and not _demote_bundled:
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
    if _demote_bundled:
        paths.append(bundled)
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
