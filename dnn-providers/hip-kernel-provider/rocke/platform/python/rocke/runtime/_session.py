# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One retained compiler/runtime owner, with single-flight cold admission.

Compilation never runs under the initialization lock. Native IR consumers get
this owner's flavor; they do not create another compiler-loading singleton.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ._compiler_query import ComgrError


@dataclass
class _Attempt:
    kind: str
    done: threading.Event = field(default_factory=threading.Event)
    error: tuple[type[BaseException], tuple] | None = None


class RuntimeSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.compiler = None
        self.hip = None
        self.requested_compiler = None
        self.compiler_info = None
        self.queried_compiler = None
        self.dll_directories = []
        self._pending = None
        self.environment = None

    def resolve(self, kind, loader):
        handle = getattr(self, kind)
        if handle is not None:
            return handle
        with self.lock:
            handle = getattr(self, kind)
            if handle is not None:
                return handle
            if self.environment is None:
                import os

                self.environment = dict(os.environ)
            attempt = self._pending
            initialize = attempt is None
            if initialize:
                attempt = self._pending = _Attempt(kind)
        if not initialize:
            attempt.done.wait()
            if attempt.kind == kind and attempt.error is not None:
                error_type, arguments = attempt.error
                raise error_type(*arguments)
            return self.resolve(kind, loader)
        try:
            handle = loader()
            with self.lock:
                if kind == "compiler":
                    admission = getattr(handle, "_rocke_admission", None)
                    if not isinstance(admission, tuple) or len(admission) != 2:
                        admission = (getattr(handle, "_name", None), None)
                    requested, info = admission
                    self.requested_compiler = requested
                    if info is not None:
                        self.compiler_info, self.queried_compiler = info, handle
                setattr(self, kind, handle)
            return handle
        except BaseException as error:
            attempt.error = type(error), error.args
            raise
        finally:
            with self.lock:
                self._pending = None
                attempt.done.set()

    def check_compiler_request(self, path):
        """Explicit reinitialization is not an environment change on a warm call."""
        import os

        with self.lock:
            if self.compiler is not None:
                actual = (
                    self.compiler_info.comgr_path
                    if self.compiler_info
                    else getattr(
                        self.compiler,
                        "_rocke_actual_path",
                        getattr(self.compiler, "_name", None),
                    )
                )
                try:
                    same = actual and os.path.samefile(path, actual)
                except OSError:
                    same = False
                if not same:
                    raise ComgrError(
                        f"compiler session already owns {actual!r}; cannot select {path!r}"
                    )


def paired_compiler(hip_path):
    """A declared SDK or single-compiler runtime directory, not a system search."""
    import os
    import re
    import sys
    from pathlib import Path
    from . import _sdk_comgr

    windows = sys.platform == "win32"
    compiler, runtime = _sdk_comgr.candidates(
        windows=windows
    ), _sdk_comgr.hip_candidates(windows=windows)
    if compiler and runtime:
        try:
            if os.path.samefile(hip_path, runtime[0]):
                return compiler[0]
        except OSError:
            pass
    if not hip_path or not os.path.isabs(hip_path):
        raise ComgrError("runtime pairing needs an exact HIP path")
    pattern = r"amd_comgr(?:_?\d+)?\.dll" if windows else r"libamd_comgr\.so(?:\.\d+)*"
    found = [
        path
        for path in Path(hip_path).parent.iterdir()
        if re.fullmatch(pattern, path.name, re.IGNORECASE if windows else 0)
        and path.is_file()
    ]
    distinct = {(os.stat(path).st_dev, os.stat(path).st_ino) for path in found}
    if len(distinct) != 1:
        raise ComgrError(
            f"HIP {hip_path!r} has no unambiguous declared companion compiler"
        )
    return str(min(found, key=lambda path: (len(path.name), str(path))))


def check_pair(hip_path, compiler_path):
    import os

    try:
        companion = paired_compiler(hip_path)
        same = os.path.samefile(companion, compiler_path)
    except OSError as error:
        raise ComgrError(f"runtime/compiler identity unavailable: {error}") from error
    if not same:
        raise ComgrError(
            f"HIP {hip_path!r} requires {companion!r}, not admitted COMGR {compiler_path!r}; declare a coherent exact-file pair"
        )


session = RuntimeSession()
