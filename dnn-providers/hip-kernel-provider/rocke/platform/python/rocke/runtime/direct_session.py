# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Caller-owned sessions of the direct Rust code-object compiler.

An optional path next to :func:`rocke.runtime.code_object.build_direct_rocke_ir_hsaco`:
a :class:`DirectSession` owns the compiler's persistent worker threads (so a
compile pays no thread start) and the state of a streamed input. Results are
those of ``build_direct_rocke_ir_hsaco`` on the same IR, byte for byte.

The C ABI is ``rocke_codeobj_session_*_v1`` in ``rocke/code_object_ffi.h``, and
``rocke_codeobj_session_create_v2`` for a session of another target than
gfx1151. Calls on one session must not overlap; this wrapper serializes them
with a lock, so a session may be shared between Python threads. Distinct
sessions are independent.

Usage::

    with DirectSession() as session:
        hsaco, timings, compiler, diagnostics = session.compile(ir_text)
        for piece in pieces_of_ir:          # streaming
            session.feed(piece)
        hsaco, timings, compiler, diagnostics = session.finish()
"""

from __future__ import annotations

import ctypes
import functools
import threading
import time
from typing import TYPE_CHECKING

from .code_object import (
    _DIRECT_ISA,
    CodeObjectCompilerInfo,
    CodeObjectError,
    CodeObjectTimings,
    DirectUnsupported,
    _CodeObjectCompileResultV1,
    _direct_library_functions,
    _direct_rocke_compiler_info,
    _direct_supported_targets,
    _direct_target,
    _DirectGemmCompileRequestV1,
    _rejection,
    _require_direct_library,
)

if TYPE_CHECKING:
    from typing import Self

CompileOutcome = tuple[bytes, CodeObjectTimings, CodeObjectCompilerInfo, str]


class _SessionOptionsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("worker_threads", ctypes.c_uint32),
    ]


class _SessionOptionsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("worker_threads", ctypes.c_uint32),
        ("target", ctypes.c_char_p),
    ]


@functools.cache
def _session_create_v2(path: str):
    """The library's session constructor that takes a target, or None."""
    library, _compile_gemm, _dispose = _direct_library_functions(path)
    create = getattr(library, "rocke_codeobj_session_create_v2", None)
    if create is not None:
        create.argtypes = [
            ctypes.POINTER(_SessionOptionsV2),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_int32
    return create


@functools.cache
def _session_functions(path: str):
    library, _compile_gemm, dispose = _direct_library_functions(path)
    try:
        create = library.rocke_codeobj_session_create_v1
        free = library.rocke_codeobj_session_free_v1
        compile_ir = library.rocke_codeobj_session_compile_v1
        feed = library.rocke_codeobj_session_feed_v1
        finish = library.rocke_codeobj_session_finish_v1
    except AttributeError as error:
        raise CodeObjectError(
            f"direct code-object library {path} has no compile sessions: {error}"
        ) from error
    result = ctypes.POINTER(_CodeObjectCompileResultV1)
    create.argtypes = [
        ctypes.POINTER(_SessionOptionsV1),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create.restype = ctypes.c_int32
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    compile_ir.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_DirectGemmCompileRequestV1),
        result,
    ]
    compile_ir.restype = ctypes.c_int32
    feed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    feed.restype = ctypes.c_int32
    finish.argtypes = [ctypes.c_void_p, result]
    finish.restype = ctypes.c_int32
    return create, free, compile_ir, feed, finish, dispose


def _take_result(
    status: int, result, dispose, path: str, elapsed: float
) -> CompileOutcome:
    """The image and summary of a filled result, which is disposed here."""
    try:
        diagnostics = (
            ctypes.string_at(result.diagnostics, result.diagnostics_size).decode(
                "utf-8", "replace"
            )
            if result.diagnostics
            else ""
        )
        if status != 0 or result.status != 0:
            raise _rejection(diagnostics)
        if not result.hsaco or not 0 < result.hsaco_size <= 64 * 1024 * 1024:
            raise CodeObjectError(
                f"direct compiler returned invalid image size {result.hsaco_size}"
            )
        hsaco = ctypes.string_at(result.hsaco, result.hsaco_size)
    finally:
        dispose(ctypes.byref(result))
    if not hsaco.startswith(b"\x7fELF"):
        raise CodeObjectError("direct compiler returned an invalid image")
    return (
        hsaco,
        CodeObjectTimings(executable=elapsed),
        _direct_rocke_compiler_info(path),
        diagnostics,
    )


class DirectSession:
    """A compile session of the direct compiler for one target (``isa``, an
    ISA name or target ID the library reports; gfx1151 by default); see the
    module docs.

    ``worker_threads`` is the number of threads the session starts (0 picks a
    default). Close the session (or use it as a context manager) to stop them.
    A target the library does not compile for raises :class:`DirectUnsupported`.
    """

    def __init__(self, worker_threads: int = 0, *, isa: str = _DIRECT_ISA) -> None:
        self._path = _require_direct_library()
        target = _direct_target(isa)
        supported = _direct_supported_targets(self._path)
        if target not in supported:
            raise DirectUnsupported(
                f"the direct code-object backend compiles for {', '.join(supported)}, "
                f"not {isa!r}"
            )
        (
            self._create,
            self._free,
            self._compile,
            self._feed,
            self._finish,
            self._dispose,
        ) = _session_functions(self._path)
        handle = ctypes.c_void_p()
        create_for_target = _session_create_v2(self._path)
        if create_for_target is not None:
            self._target_name = target.encode("ascii")
            options = _SessionOptionsV2(
                struct_size=ctypes.sizeof(_SessionOptionsV2),
                worker_threads=int(worker_threads),
                target=self._target_name,
            )
            created = create_for_target(ctypes.byref(options), ctypes.byref(handle))
        else:
            options = _SessionOptionsV1(
                struct_size=ctypes.sizeof(_SessionOptionsV1),
                worker_threads=int(worker_threads),
            )
            created = self._create(ctypes.byref(options), ctypes.byref(handle))
        if created != 0 or not handle:
            raise CodeObjectError("cannot create a direct compile session")
        self._handle: ctypes.c_void_p | None = handle
        self._lock = threading.Lock()

    def _live(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise CodeObjectError("the direct compile session is closed")
        return self._handle

    def compile(self, serialized_ir: str | bytes) -> CompileOutcome:
        """``build_direct_rocke_ir_hsaco`` on the session's threads."""
        if isinstance(serialized_ir, str):
            serialized_ir = serialized_ir.encode("utf-8")
        if not serialized_ir:
            raise ValueError("serialized_ir must be non-empty")
        storage = ctypes.create_string_buffer(serialized_ir, len(serialized_ir))
        request = _DirectGemmCompileRequestV1(
            struct_size=ctypes.sizeof(_DirectGemmCompileRequestV1),
            serialized_ir=ctypes.cast(storage, ctypes.c_void_p),
            serialized_ir_size=len(serialized_ir),
        )
        result = _CodeObjectCompileResultV1(
            struct_size=ctypes.sizeof(_CodeObjectCompileResultV1)
        )
        with self._lock:
            handle = self._live()
            started = time.perf_counter()
            status = int(
                self._compile(handle, ctypes.byref(request), ctypes.byref(result))
            )
            elapsed = time.perf_counter() - started
        return _take_result(status, result, self._dispose, self._path, elapsed)

    def feed(self, piece: str | bytes) -> None:
        """Appends a piece of the IR being streamed (pieces may split lines)."""
        if isinstance(piece, str):
            piece = piece.encode("utf-8")
        if not piece:
            return
        storage = ctypes.create_string_buffer(piece, len(piece))
        with self._lock:
            if (
                self._feed(
                    self._live(), ctypes.cast(storage, ctypes.c_void_p), len(piece)
                )
                != 0
            ):
                raise CodeObjectError("the direct compile session refused the piece")

    def finish(self) -> CompileOutcome:
        """Compiles everything fed since the last finish and ends the stream."""
        result = _CodeObjectCompileResultV1(
            struct_size=ctypes.sizeof(_CodeObjectCompileResultV1)
        )
        with self._lock:
            handle = self._live()
            started = time.perf_counter()
            status = int(self._finish(handle, ctypes.byref(result)))
            elapsed = time.perf_counter() - started
        return _take_result(status, result, self._dispose, self._path, elapsed)

    def close(self) -> None:
        """Stops the session's threads and frees it; later calls raise."""
        with self._lock:
            if self._handle is not None:
                self._free(self._handle)
                self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._free(handle)
            self._handle = None
