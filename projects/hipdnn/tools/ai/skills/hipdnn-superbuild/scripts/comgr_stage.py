#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage the ROCm wheel's amd_comgr.dll app-local next to test binaries (Windows).

Why this exists: on Windows the loader resolves ``amd_comgr.dll`` by name from
the .exe's own directory first, then ``C:\\Windows\\System32``, then PATH. The
AMD driver drops an old ``amd_comgr.dll`` into System32 that outranks the
wheel's copy on PATH, so MIOpen loads the stale comgr and fails to JIT-build
GCN-assembly kernels (Winograd and some direct/BN solvers) at runtime::

    MIOpen(HIP): Error [BuildAsm] comgr status = ERROR (1)
    lld: error: unknown argument '-Wa,-defsym,ROCM_METADATA_VERSION=5'
    lld: error: unknown emulation: no-xnack
    MIOpen Error: ... Code object build failed. Source: Conv_Winograd_v30_*.s

Prepending the wheel's bin directory to PATH cannot beat System32; only an
app-local copy (or DLL redirection) wins. This stages
``<rocm-bin>/amd_comgr.dll`` into the test binary directory (``<build>/bin``)
so the loader picks the wheel's comgr first.

The copy is skipped when an up-to-date copy is already present: the PE *product*
version of source and destination is compared first (cheap, and it matches the
``COMgr v.X.Y.Z`` string MIOpen logs), falling back to a size + SHA-256 content
comparison only when version metadata is unavailable on either side.

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


COMGR_DLL = "amd_comgr.dll"
SYSTEM32_COMGR = Path("C:/Windows/System32/amd_comgr.dll")


def _fixed_file_info(path):
    """Return (file_version, product_version) tuples from a PE file, or None.

    Each version is a 4-tuple of ints. Returns None on non-Windows, when the
    file has no version resource (the driver's System32 comgr has none), or on
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


def comgr_version(path):
    """Return the comgr product version 4-tuple, or None if unavailable."""
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

    src_ver, dst_ver = comgr_version(src), comgr_version(dst)
    if src_ver is not None and dst_ver is not None:
        if src_ver == dst_ver:
            return False, f"already staged (comgr v{_fmt(dst_ver)})"
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


def stage_comgr(rocm_bin, dest_bin, *, check_only=False, verbose=False):
    """Stage <rocm_bin>/amd_comgr.dll into dest_bin when needed.

    Returns one of: "skipped-non-windows", "missing-source", "up-to-date",
    "would-copy", "copied".
    """
    if platform.system() != "Windows":
        if verbose:
            print("comgr-stage: non-Windows platform, nothing to stage")
        return "skipped-non-windows"

    rocm_bin = Path(rocm_bin)
    dest_bin = Path(dest_bin)
    src = rocm_bin / COMGR_DLL
    dst = dest_bin / COMGR_DLL

    if not src.exists():
        print(
            f"comgr-stage: WARNING wheel comgr not found at {src}; not staging",
            file=sys.stderr,
        )
        return "missing-source"

    if verbose and SYSTEM32_COMGR.exists():
        sys_ver = comgr_version(SYSTEM32_COMGR)
        print(
            f"comgr-stage: note {SYSTEM32_COMGR} exists (comgr v{_fmt(sys_ver)}) and "
            "shadows PATH; staging the wheel comgr app-local overrides it"
        )

    stage, reason = needs_stage(src, dst)
    if not stage:
        if verbose:
            print(f"comgr-stage: up to date, {reason} -> {dst}")
        return "up-to-date"

    if check_only:
        print(f"comgr-stage: would copy {src} -> {dst} ({reason})")
        return "would-copy"

    dest_bin.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(
        f"comgr-stage: staged {src} -> {dst} "
        f"({reason}; comgr v{_fmt(comgr_version(dst))})"
    )
    return "copied"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rocm-bin",
        required=True,
        help="ROCm wheel bin directory containing amd_comgr.dll",
    )
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument(
        "--build-dir", help="Superbuild directory; stages into <build-dir>/bin"
    )
    dest.add_argument(
        "--dest-bin", help="Explicit destination directory for the app-local copy"
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Report what would happen without copying",
    )
    p.add_argument("--verbose", action="store_true", help="Print the staging decision")
    args = p.parse_args()

    dest_bin = Path(args.dest_bin) if args.dest_bin else Path(args.build_dir) / "bin"
    action = stage_comgr(
        args.rocm_bin, dest_bin, check_only=args.check_only, verbose=args.verbose
    )
    return 1 if action == "missing-source" else 0


if __name__ == "__main__":
    sys.exit(main())
