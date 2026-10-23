#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage the ROCm wheel's System32-shadowed DLLs app-local next to test binaries (Windows).

Why this exists: on Windows the loader resolves a DLL by name from the .exe's
own directory first, then ``C:\\Windows\\System32``, then PATH. The AMD driver
drops old copies of ROCm runtime DLLs into System32 that outrank the wheel's
copies on PATH, so the process loads a runtime older than the wheel libraries
were built against:

* ``amd_comgr.dll`` -- MIOpen's runtime JIT fails to build GCN-assembly kernels
  (Winograd and some direct/BN solvers)::

      MIOpen(HIP): Error [BuildAsm] comgr status = ERROR (1)
      lld: error: unknown emulation: no-xnack
      MIOpen Error: ... Code object build failed. Source: Conv_Winograd_v30_*.s

* ``amdhip64_<N>.dll`` (the HIP runtime) -- the wheel's rocBLAS faults with an
  access violation (SEH 0xc0000005) as soon as MIOpen's GEMM conv solvers
  (e.g. GemmFwd1x1_0_1) call into it, after which the process can hang.

Prepending the wheel's bin directory to PATH cannot beat System32; only an
app-local copy (or DLL redirection) wins. This stages each shadowed DLL from
``<rocm-bin>`` into the test binary directory (``<build>/bin``) so the loader
picks the wheel's copy first. The HIP runtime's name carries its major version,
so it is matched by the ``amdhip64_*.dll`` glob rather than a fixed name.

A copy is skipped when an up-to-date copy is already present: the PE *product*
version of source and destination is compared first (cheap, and for comgr it
matches the ``COMgr v.X.Y.Z`` string MIOpen logs), falling back to a size +
SHA-256 content comparison when version metadata is unavailable on either side.

On non-Windows platforms this is a no-op: ELF ``.so`` resolution uses
RPATH/RUNPATH and ``LD_LIBRARY_PATH`` with no System32-style shadowing, so
app-local staging is unnecessary.
"""

import argparse
import hashlib
import platform
import shutil
import sys
from pathlib import Path


SYSTEM32 = Path("C:/Windows/System32")
FIXED_DLLS = ("amd_comgr.dll",)
HIP_RUNTIME_GLOB = "amdhip64_*.dll"
LOG = "shadowed-dll-stage:"


def _fixed_file_info(path):
    """Return (file_version, product_version) tuples from a PE file, or None.

    Each version is a 4-tuple of ints. Returns None on non-Windows, when the
    file has no version resource (the driver's System32 copies have none), or on
    any lookup failure.
    """
    if platform.system() != "Windows":
        return None

    import ctypes
    from ctypes import wintypes

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [
            ("dwSignature", wintypes.DWORD),
            ("dwStrucVersion", wintypes.DWORD),
            ("dwFileVersionMS", wintypes.DWORD),
            ("dwFileVersionLS", wintypes.DWORD),
            ("dwProductVersionMS", wintypes.DWORD),
            ("dwProductVersionLS", wintypes.DWORD),
            ("dwFileFlagsMask", wintypes.DWORD),
            ("dwFileFlags", wintypes.DWORD),
            ("dwFileOS", wintypes.DWORD),
            ("dwFileType", wintypes.DWORD),
            ("dwFileSubtype", wintypes.DWORD),
            ("dwFileDateMS", wintypes.DWORD),
            ("dwFileDateLS", wintypes.DWORD),
        ]

    p = str(path)
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(p, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ctypes.windll.version.GetFileVersionInfoW(p, 0, size, buf):
            return None
        lp = ctypes.c_void_p()
        ln = wintypes.UINT()
        if not ctypes.windll.version.VerQueryValueW(
            buf, "\\", ctypes.byref(lp), ctypes.byref(ln)
        ):
            return None
        if not lp.value:
            return None
        ffi = ctypes.cast(lp, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
    except OSError:
        return None

    def split(ms, ls):
        return ((ms >> 16) & 0xFFFF, ms & 0xFFFF, (ls >> 16) & 0xFFFF, ls & 0xFFFF)

    return (
        split(ffi.dwFileVersionMS, ffi.dwFileVersionLS),
        split(ffi.dwProductVersionMS, ffi.dwProductVersionLS),
    )


def dll_version(path):
    """Return the DLL's product version 4-tuple, or None if unavailable."""
    info = _fixed_file_info(path)
    return info[1] if info else None


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fmt(version):
    return ".".join(str(x) for x in version) if version else "unknown"


def needs_stage(src, dst):
    """Decide whether dst must be (re)written from src. Returns (bool, reason)."""
    if not dst.exists():
        return True, "destination missing"

    src_ver, dst_ver = dll_version(src), dll_version(dst)
    if src_ver is not None and dst_ver is not None:
        if src_ver == dst_ver:
            return False, f"already staged (v{_fmt(dst_ver)})"
        return (
            True,
            f"version differs (wheel v{_fmt(src_ver)} vs staged v{_fmt(dst_ver)})",
        )

    # Version metadata missing on one side: compare bytes instead.
    if src.stat().st_size != dst.stat().st_size:
        return True, "size differs"
    if _sha256(src) != _sha256(dst):
        return True, "content differs"
    return False, "identical content"


def stage_dll(src, dest_bin, *, check_only=False, verbose=False):
    """Stage one DLL into dest_bin when needed.

    Returns one of: "missing-source", "up-to-date", "would-copy", "copied".
    """
    dst = dest_bin / src.name

    if not src.exists():
        print(
            f"{LOG} WARNING wheel {src.name} not found at {src}; not staging",
            file=sys.stderr,
        )
        return "missing-source"

    system32_copy = SYSTEM32 / src.name
    if verbose and system32_copy.exists():
        print(
            f"{LOG} note {system32_copy} exists (v{_fmt(dll_version(system32_copy))}) and "
            "shadows PATH; staging the wheel copy app-local overrides it"
        )

    stage, reason = needs_stage(src, dst)
    if not stage:
        if verbose:
            print(f"{LOG} up to date, {reason} -> {dst}")
        return "up-to-date"

    if check_only:
        print(f"{LOG} would copy {src} -> {dst} ({reason})")
        return "would-copy"

    dest_bin.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{LOG} staged {src} -> {dst} ({reason}; v{_fmt(dll_version(dst))})")
    return "copied"


def stage_shadowed_dlls(rocm_bin, dest_bin, *, check_only=False, verbose=False):
    """Stage every shadowed DLL from rocm_bin into dest_bin when needed.

    Returns a dict mapping each DLL name (or the HIP runtime glob, when no HIP
    runtime DLL exists in rocm_bin) to its stage_dll() action, or an empty dict
    on non-Windows platforms.
    """
    if platform.system() != "Windows":
        if verbose:
            print(f"{LOG} non-Windows platform, nothing to stage")
        return {}

    rocm_bin = Path(rocm_bin)
    dest_bin = Path(dest_bin)
    actions = {}
    hip_runtime = sorted(p.name for p in rocm_bin.glob(HIP_RUNTIME_GLOB))
    if not hip_runtime:
        print(
            f"{LOG} WARNING no {HIP_RUNTIME_GLOB} found in {rocm_bin}; not staging",
            file=sys.stderr,
        )
        actions[HIP_RUNTIME_GLOB] = "missing-source"
    for name in (*FIXED_DLLS, *hip_runtime):
        actions[name] = stage_dll(
            rocm_bin / name, dest_bin, check_only=check_only, verbose=verbose
        )
    return actions


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rocm-bin",
        required=True,
        help="ROCm wheel bin directory containing amd_comgr.dll and amdhip64_<N>.dll",
    )
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument(
        "--build-dir", help="Superbuild directory; stages into <build-dir>/bin"
    )
    dest.add_argument(
        "--dest-bin", help="Explicit destination directory for the app-local copies"
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Report what would happen without copying",
    )
    p.add_argument("--verbose", action="store_true", help="Print the staging decisions")
    args = p.parse_args()

    dest_bin = Path(args.dest_bin) if args.dest_bin else Path(args.build_dir) / "bin"
    actions = stage_shadowed_dlls(
        args.rocm_bin, dest_bin, check_only=args.check_only, verbose=args.verbose
    )
    return 1 if "missing-source" in actions.values() else 0


if __name__ == "__main__":
    sys.exit(main())
