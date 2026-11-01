# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cached precompiled header (PCH) of the HIP headers for the helper-kernel compile.

About 0.3 s of every helper-kernel (Kernels.cpp) compile is parsing the HIP headers: the driver's
implicit __clang_hip_runtime_wrapper.h, <hip/hip_runtime.h>, <hip/hip_ext.h> and <hip/hip_fp8.h>.
A PCH of exactly those headers is built once per compiler build, HIP install, argument list and
target, stored next to the helper-kernel cache with HelperKernelCache's pattern (SHA-256 key, a
temporary directory renamed into place, age-based eviction) and given to the compile with
-include-pch. The TensileLite static headers stay out of it: they are copied into every build
directory with a new mtime, which clang rejects.

TENSILE_HELPER_PCH_DIR overrides the location (default ~/.tensile/helper_pch).
TENSILE_DISABLE_HELPER_PCH=1/YES/ON/TRUE compiles without a PCH.
"""

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from shlex import split
from subprocess import DEVNULL, PIPE, STDOUT, CalledProcessError, run
from typing import List, Optional

from ..Common import print1
from .HelperKernelCache import _evictStale

_PCH_HEADERS = ("hip/hip_runtime.h", "hip/hip_ext.h", "hip/hip_fp8.h")
# The PCH is built from stdin: a PCH records the path of its main file and clang refuses it once
# that file is gone, which would tie an entry to the temporary directory it was built in.
PCH_SOURCE = "".join(f"#include <{header}>\n" for header in _PCH_HEADERS).encode()
_PCH_FILE = "hip.pch"
# How many times this process discarded the PCH of each key after the compiler rejected it. The
# first rejection rebuilds the PCH (a ROCm reinstall heals that way); a second means the compiler
# keeps rejecting what is built for that key (e.g. two hosts sharing the cache whose HIP headers
# differ only in mtime), so the rest of the process compiles without it.
_REJECTIONS_BEFORE_DISABLING = 2
_rejections = {}
_rejectionsLock = threading.Lock()

# Options of the driver's compile job that name a header search directory or a forced include.
_HEADER_SEARCH_OPTIONS = frozenset(
    (
        "-resource-dir",
        "-internal-isystem",
        "-internal-externc-isystem",
        "-isystem",
        "-idirafter",
        "-iquote",
        "-I",
        "-include",
    )
)


def _compilerVersion(compilerPath) -> str:
    """The full `--version` output: it names the exact compiler build (commit), which clang
    requires of a PCH, where Compiler.version keeps only major.minor.patch."""
    return run(
        [str(compilerPath), "--version"], stdin=DEVNULL, stdout=PIPE, stderr=STDOUT, check=True
    ).stdout.decode()


def _headerSearchPaths(pchArgs: List[str]) -> List[str]:
    """The header search directories and forced includes of the driver's compile job for pchArgs,
    with absolute paths resolved. This is where the HIP headers really come from: the driver
    finds them from its own location, ROCM_PATH/HIP_PATH or --rocm-path/--hip-path."""
    out = run(
        [*pchArgs, "-###"], stdin=DEVNULL, stdout=PIPE, stderr=STDOUT, check=True
    ).stdout.decode()
    paths = []
    for line in out.splitlines():
        if not line.startswith(' "'):
            continue
        args = split(line)
        for option, value in zip(args, args[1:]):
            if option in _HEADER_SEARCH_OPTIONS:
                paths.append(
                    f"{option} {os.path.realpath(value) if os.path.isabs(value) else value}"
                )
    return paths


def _computePchKey(
    pchArgs: List[str], targets: List[str], compilerVersion: str, headerSearchPaths: List[str]
) -> str:
    """SHA-256 of everything a PCH is only valid for: the exact argument list it is built with
    (every -D, -O, -std, sanitizer and target flag: a -D baked into a PCH would leak silently
    into a compile without it), the target, the exact compiler build, where the HIP headers are
    found, and the headers it contains."""
    return hashlib.sha256(
        json.dumps(
            {
                "source": PCH_SOURCE.decode(),
                "args": [str(a) for a in pchArgs],
                "targets": list(targets),
                "compilerVersion": compilerVersion,
                "headerSearchPaths": headerSearchPaths,
            }
        ).encode()
    ).hexdigest()


def _isUsable(pchPath: Path) -> bool:
    try:
        return pchPath.stat().st_size > 0
    except OSError:
        return False


class HelperKernelPch:
    """Filesystem cache of the HIP-header PCH used by the helper-kernel compile.

    Construct a fresh instance per build call to pick up the current environment.
    """

    _DEFAULT_DIR = Path.home() / ".tensile" / "helper_pch"
    _MAX_AGE_DAYS = 30

    def __init__(self):
        disabled = os.environ.get("TENSILE_DISABLE_HELPER_PCH", "").upper() in (
            "1",
            "YES",
            "ON",
            "TRUE",
        )
        self.enabled = not disabled
        self.dir = Path(os.environ.get("TENSILE_HELPER_PCH_DIR", str(self._DEFAULT_DIR)))
        if self.enabled:
            _evictStale(self.dir, self._MAX_AGE_DAYS)

    def get(self, compiler, targets: List[str]) -> Optional[Path]:
        """The PCH for compiling `targets` with `compiler`, built and stored on a miss.

        None (compile without a PCH) when disabled, for more than one target (a PCH serves one
        target, and a multi-target compile would give it to every device compile), under a
        compiler launcher (ccache does not cache a compile that uses a PCH unless configured
        to), when the key or the PCH cannot be computed or built, or once the compiler has
        rejected this key's PCH twice in this process (see discard).
        """
        if not self.enabled or len(targets) != 1 or compiler.launcher:
            return None
        pchArgs = compiler.pchArgs(targets)
        try:
            key = _computePchKey(
                pchArgs, targets, _compilerVersion(compiler.path), _headerSearchPaths(pchArgs)
            )
        except (OSError, CalledProcessError) as err:
            print1(f"# Helper kernel PCH not used, querying the compiler failed: {err}")
            return None
        with _rejectionsLock:
            if _rejections.get(key, 0) >= _REJECTIONS_BEFORE_DISABLING:
                return None

        pchPath = self.dir / key / _PCH_FILE
        if _isUsable(pchPath):
            try:
                os.utime(pchPath.parent)
            except OSError:
                pass
            return pchPath

        print1(f"# Helper kernel PCH MISS ({key[:12]}...)")
        return self._build(compiler, targets, key)

    def _build(self, compiler, targets: List[str], key: str) -> Optional[Path]:
        """Build into a temporary directory, then rename it into place: safe under concurrent
        builds of the same key (xdist workers, overlapping helper compiles)."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmpDir = Path(tempfile.mkdtemp(prefix=f".tmp_{key}_{os.getpid()}_", dir=self.dir))
        except OSError as err:
            print1(f"# Helper kernel PCH not used, cannot write {self.dir}: {err}")
            return None
        try:
            compiler.buildPch(targets, PCH_SOURCE, tmpDir / _PCH_FILE)
        except (OSError, RuntimeError) as err:
            shutil.rmtree(tmpDir, ignore_errors=True)
            print1(f"# Helper kernel PCH build failed, compiling without it: {err}")
            return None

        finalDir = self.dir / key
        try:
            tmpDir.rename(finalDir)
        except OSError:
            # Another build stored this key first.
            shutil.rmtree(tmpDir, ignore_errors=True)
        pchPath = finalDir / _PCH_FILE
        return pchPath if _isUsable(pchPath) else None

    def discard(self, pchPath: Path):
        """Remove an entry the compiler rejected (e.g. HIP headers changed since it was built), so
        that the next compile builds it again. Renamed away first, so no other build sees it
        half removed. The second rejection of a key in this process stops get() from returning
        a PCH for it for the rest of the process."""
        entry = Path(pchPath).parent
        with _rejectionsLock:
            _rejections[entry.name] = _rejections.get(entry.name, 0) + 1
            disabling = _rejections[entry.name] == _REJECTIONS_BEFORE_DISABLING
        if disabling:
            print1(
                f"# Helper kernel PCH disabled for this run ({entry.name[:12]}...): the compiler"
                " rejected it again after it was rebuilt"
            )
        staleDir = entry.with_name(f".tmp_stale_{entry.name}_{os.getpid()}_{threading.get_ident()}")
        try:
            entry.rename(staleDir)
        except OSError:
            return
        shutil.rmtree(staleDir, ignore_errors=True)
