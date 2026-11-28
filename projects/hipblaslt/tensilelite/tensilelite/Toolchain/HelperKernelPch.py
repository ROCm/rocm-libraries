# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cached precompiled header (PCH) of the HIP headers for the helper-kernel compile.

About 0.3 s of every helper-kernel (Kernels.cpp) compile is parsing the HIP headers: the driver's
implicit __clang_hip_runtime_wrapper.h, <hip/hip_runtime.h>, <hip/hip_ext.h> and <hip/hip_fp8.h>.
A PCH of exactly those headers is keyed on compiler build, current preprocessed headers and
macros, argument list and target, stored next to the helper-kernel cache with HelperKernelCache's
pattern (SHA-256 key, a temporary directory renamed into place, age-based eviction) and given to
the compile with -include-pch. The TensileLite static headers stay out of it: they are copied
into every build directory with a new mtime, which clang rejects.

TENSILE_HELPER_PCH_DIR overrides the location (default ~/.tensile/helper_pch).
TENSILE_DISABLE_HELPER_PCH=1/YES/ON/TRUE compiles without a PCH.
"""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from subprocess import DEVNULL, PIPE, STDOUT, CalledProcessError, run
from typing import List, Optional, Union

from filelock import FileLock, Timeout

from ..Common import print1
from .Component import driverJobs
from .HelperKernelCache import _evictStale

_PCH_HEADERS = ("hip/hip_runtime.h", "hip/hip_ext.h", "hip/hip_fp8.h")
# The PCH is built from stdin: a PCH records the path of its main file and clang refuses it once
# that file is gone, which would tie an entry to the temporary directory it was built in.
PCH_SOURCE = "".join(f"#include <{header}>\n" for header in _PCH_HEADERS).encode()
_PCH_FILE = "hip.pch"
_PUBLICATION_LOCK_FILE = ".publish.lock"
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
        "-c-isystem",
        "-cxx-isystem",
        "-objc-isystem",
        "-objcxx-isystem",
        "-isystem",
        "-idirafter",
        "-iquote",
        "-I",
        "-F",
        "-iframework",
        "-iframeworkwithsysroot",
        "-internal-iframework",
        "-isysroot",
        "-include",
        "-imacros",
    )
)
# Clang emits CPATH as -I<path>; other driver arguments can use joined spellings too.
_JOINED_HEADER_SEARCH_OPTIONS = (
    "-iframeworkwithsysroot",
    "-iframework",
    "-isystem",
    "-idirafter",
    "-iquote",
    "-F",
    "-I",
)
_HEADER_FILE_OPTIONS = frozenset(("-include", "-imacros"))


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
    paths = []
    for jobArgs in driverJobs(pchArgs):
        args = iter(jobArgs)
        for arg in args:
            if arg in _HEADER_SEARCH_OPTIONS:
                option = arg
                value = next(args, None)
                if value is None:
                    raise ValueError(f"Missing header search argument for {option}")
            else:
                option = next(
                    (option for option in _JOINED_HEADER_SEARCH_OPTIONS if arg.startswith(option)),
                    None,
                )
                if option is None:
                    continue
                value = arg[len(option):]
            # Directory paths are relative to the compiler's working directory. Bare forced
            # includes (e.g. __clang_hip_runtime_wrapper.h) are looked up in the search paths.
            if option not in _HEADER_FILE_OPTIONS or os.path.isabs(value):
                value = os.path.realpath(value)
            paths.append(f"{option} {value}")
    if not paths:
        raise ValueError("No header search paths found in the compiler job")
    return paths


def _computePchKey(
    pchArgs: List[str],
    targets: List[str],
    compilerVersion: str,
    headerSearchPaths: List[str],
    headerFingerprint: str,
) -> str:
    """SHA-256 of everything a PCH is only valid for: the compiler argument list
    (every -D, -O, -std, sanitizer and target flag: a -D baked into a PCH would leak silently
    into a compile without it), the target, the exact compiler build, where the HIP headers are
    found, and the freshly preprocessed header contents and macro definitions."""
    return hashlib.sha256(
        json.dumps(
            {
                "source": PCH_SOURCE.decode(),
                "args": [str(a) for a in pchArgs],
                "targets": list(targets),
                "compilerVersion": compilerVersion,
                "headerSearchPaths": headerSearchPaths,
                "headerFingerprint": headerFingerprint,
            }
        ).encode()
    ).hexdigest()


def _preprocessedHeaderHash(args: List[str]) -> str:
    """Fingerprint current header lookup, transitive contents and macro definitions.

    Clang's PCH validation checks the old dependency files. It can miss a new, higher-priority
    header in an unchanged search directory, so query the preprocessor before selecting a PCH.
    This avoids full C++ parsing while accounting for what a fresh compile would read.
    """
    result = run(args, input=PCH_SOURCE, stdout=PIPE, stderr=PIPE, check=True)
    if not result.stdout:
        raise ValueError("No preprocessed HIP headers returned by the compiler")
    return hashlib.sha256(result.stdout).hexdigest()


def _isUsable(pchPath: Path) -> bool:
    try:
        info = pchPath.stat()
        return stat.S_ISREG(info.st_mode) and info.st_size > 0
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
            try:
                _evictStale(self.dir, self._MAX_AGE_DAYS)
            except OSError as err:
                self.enabled = False
                print1(f"# Helper kernel PCH not used, cannot access {self.dir}: {err}")

    def get(
        self, compiler, targets: List[str], includeDir: Optional[Union[Path, str]] = None
    ) -> Optional[Path]:
        """The PCH for compiling `targets` with `compiler`, built and stored on a miss.

        Include the build's include directory when resolving HIP headers, so a header there
        is selected by both preprocessing and the helper compile.

        None (compile without a PCH) when disabled, for more than one target (a PCH serves one
        target, and a multi-target compile would give it to every device compile), under a
        compiler launcher (ccache does not cache a compile that uses a PCH unless configured
        to), when the key or the PCH cannot be computed or built, or once the compiler has
        rejected this key's PCH twice in this process (see discard).
        """
        if not self.enabled or len(targets) != 1 or compiler.launcher:
            return None
        includePath = None if includeDir is None else str(Path(includeDir).absolute())
        # The build-directory path itself need not split entries when it selects the same HIP
        # headers. Their actual paths, contents and macros are captured by the preprocessor.
        pchArgs = compiler.pchArgs(targets)
        try:
            key = _computePchKey(
                pchArgs, targets, _compilerVersion(compiler.path), _headerSearchPaths(pchArgs),
                _preprocessedHeaderHash(
                    compiler.pchArgs(targets, preprocess=True, include_path=includePath)
                ),
            )
        except (OSError, CalledProcessError, ValueError) as err:
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
        return self._build(compiler, targets, key, includePath)

    def _build(self, compiler, targets: List[str], key: str,
               includePath: Optional[str]=None) -> Optional[Path]:
        """Build into a temporary directory, then rename it into place: safe under concurrent
        builds of the same key (xdist workers, overlapping helper compiles). Publication and
        removal share a lock, so repairing an invalid entry preserves a valid concurrent
        winner. A busy lock falls back to compilation without the PCH."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmpDir = Path(tempfile.mkdtemp(prefix=f".tmp_{key}_{os.getpid()}_", dir=self.dir))
        except OSError as err:
            print1(f"# Helper kernel PCH not used, cannot write {self.dir}: {err}")
            return None
        try:
            if includePath is None:
                compiler.buildPch(targets, PCH_SOURCE, tmpDir / _PCH_FILE)
            else:
                compiler.buildPch(targets, PCH_SOURCE, tmpDir / _PCH_FILE, include_path=includePath)
        except (OSError, RuntimeError) as err:
            shutil.rmtree(tmpDir, ignore_errors=True)
            print1(f"# Helper kernel PCH build failed, compiling without it: {err}")
            return None

        finalDir = self.dir / key
        pchPath = finalDir / _PCH_FILE
        try:
            with FileLock(str(self.dir / _PUBLICATION_LOCK_FILE), timeout=0):
                if _isUsable(pchPath):
                    return pchPath
                if finalDir.is_dir() and not finalDir.is_symlink():
                    shutil.rmtree(finalDir)
                else:
                    finalDir.unlink(missing_ok=True)
                tmpDir.rename(finalDir)
        except (OSError, Timeout) as err:
            print1(f"# Helper kernel PCH not used, cannot publish the cache entry: {err}")
            return None
        finally:
            shutil.rmtree(tmpDir, ignore_errors=True)
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
            with FileLock(str(self.dir / _PUBLICATION_LOCK_FILE), timeout=0):
                entry.rename(staleDir)
        except (OSError, Timeout):
            return
        shutil.rmtree(staleDir, ignore_errors=True)
