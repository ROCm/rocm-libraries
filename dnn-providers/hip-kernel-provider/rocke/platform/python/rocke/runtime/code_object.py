# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AMDGPU code-object boundaries for COMGR and direct typed lowering.

`ROCKE_BACKEND` chooses the Python/C++ engine that emits LLVM text.
`ROCKE_CODEOBJ_BACKEND` independently chooses how a kernel becomes a code
object. ``comgr`` compiles that LLVM text; its low-level API remains public.
``direct`` (gfx1151 only) skips LLVM entirely: serialized rocKE IR goes to an
in-process Rust compiler (optimizer, instruction selection, register
allocation, encoder, ELF writer) and IR it does not support is rejected. The
gfx1151 WMMA GEMM proof slice uses the same library through its own entry.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "DIRECT_GFX1151_WMMA_GEMM_SYMBOL",
    "CodeObjectCompilerInfo",
    "CodeObjectError",
    "CodeObjectProcessDied",
    "CodeObjectTimings",
    "build_direct_gfx1151_wmma_gemm_hsaco",
    "build_direct_rocke_ir_hsaco",
    "build_hsaco_from_llvm_ir",
    "build_hsaco_from_llvm_modules",
    "compiler_identity",
    "loaded_compiler_info",
    "resolve_code_object_backend",
    "supported_isas",
]

_BACKEND_ENV = "ROCKE_CODEOBJ_BACKEND"
_DIRECT_LIB_ENV = "ROCKE_CODEOBJ_DIRECT_LIB"
_BACKENDS = ("comgr", "direct")
_DIRECT_ISA = "amdgcn-amd-amdhsa--gfx1151"
DIRECT_GFX1151_WMMA_GEMM_SYMBOL = "rocke_direct_wmma_gemm_wmma16x16x16_fp16_rcr_xm"
_DIRECT_IR_KERNEL_SYMBOL = re.compile(rb"(?m)^kernel @([A-Za-z0-9_.$]+) \{$")


class _DirectGemmCompileRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("serialized_ir", ctypes.c_void_p),
        ("serialized_ir_size", ctypes.c_size_t),
    ]


class _CodeObjectCompileResultV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("hsaco", ctypes.c_void_p),
        ("hsaco_size", ctypes.c_size_t),
        ("diagnostics", ctypes.c_void_p),
        ("diagnostics_size", ctypes.c_size_t),
        ("bc_seconds", ctypes.c_double),
        ("link_seconds", ctypes.c_double),
        ("relocatable_seconds", ctypes.c_double),
        ("executable_seconds", ctypes.c_double),
    ]


class CodeObjectError(RuntimeError):
    """A selected code-object compiler rejected a request or failed."""


class CodeObjectProcessDied(CodeObjectError):
    """A fresh compiler process exited without a complete response."""


@dataclass(frozen=True)
class CodeObjectCompilerInfo:
    """LLVM version and immutable identity of one selected compiler backend."""

    backend: str
    llvm_version: tuple[int, int, int] | None
    source: str
    identity: str
    path: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        """Return provenance suitable for a compatibility diagnostic."""
        if self.backend == "rust-direct-gemm":
            compiler = "direct gfx1151 machine-code emitter"
        else:
            version = (
                ".".join(str(part) for part in self.llvm_version)
                if self.llvm_version
                else "unknown"
            )
            compiler = f"LLVM {version}"
        return (
            f"{compiler} via {self.source}; backend={self.backend!r}; "
            f"compiler={self.path!r}; identity={self.identity!r}"
        )


@dataclass
class CodeObjectTimings:
    """Per-stage wall time in seconds, matching the historical COMGR fields."""

    bc: float = 0.0
    link: float = 0.0
    relocatable: float = 0.0
    executable: float = 0.0

    @property
    def total(self) -> float:
        return self.bc + self.link + self.relocatable + self.executable


def resolve_code_object_backend(backend: str | None = None) -> str:
    """Resolve and validate the code-object backend without loading it."""
    selected = backend or os.environ.get(_BACKEND_ENV, "comgr")
    if selected not in _BACKENDS:
        raise ValueError(
            f"code-object backend={selected!r}, expected one of {_BACKENDS}; "
            f"set {_BACKEND_ENV} or pass code_object_backend="
        )
    return selected


def _path_identity(path: str | None) -> str:
    if not path:
        return "<unknown>"
    real = os.path.realpath(path)
    try:
        stat = os.stat(real)
    except OSError:
        return real
    return f"{real}:{stat.st_size}:{stat.st_mtime_ns}"


def _comgr_compiler_info() -> CodeObjectCompilerInfo:
    from . import comgr

    info = comgr._require_compiler_info()
    path = info.comgr_path or info.requested_comgr
    version = (
        ".".join(str(part) for part in info.llvm_version)
        if info.llvm_version
        else "unknown"
    )
    identity = f"comgr:{_path_identity(path)}:llvm-{version}"
    return CodeObjectCompilerInfo(
        backend="comgr",
        llvm_version=info.llvm_version,
        source=info.source,
        identity=identity,
        path=path,
        details={
            "requested_comgr": info.requested_comgr,
            "comgr_path": info.comgr_path,
            "query_library_path": info.query_library_path,
        },
    )


def _direct_library_path() -> str | None:
    explicit = os.environ.get(_DIRECT_LIB_ENV)
    if explicit:
        path = os.path.realpath(explicit)
        if not os.path.isfile(path):
            raise CodeObjectError(f"{_DIRECT_LIB_ENV}={explicit!r} is not a file")
        return path
    return ctypes.util.find_library("rocke_codeobj_ffi")


@functools.cache
def _direct_library_functions(path: str):
    try:
        library = ctypes.CDLL(path)
        abi = library.rocke_codeobj_ffi_abi_version
        compile_ir = library.rocke_codeobj_compile_gfx1151_wmma_gemm_v1
        dispose = library.rocke_codeobj_compile_result_dispose_v1
    except (OSError, AttributeError) as error:
        raise CodeObjectError(
            f"cannot load direct GEMM emitter {path}: {error}"
        ) from error
    abi.argtypes = []
    abi.restype = ctypes.c_uint32
    compile_ir.argtypes = [
        ctypes.POINTER(_DirectGemmCompileRequestV1),
        ctypes.POINTER(_CodeObjectCompileResultV1),
    ]
    compile_ir.restype = ctypes.c_int32
    dispose.argtypes = [ctypes.POINTER(_CodeObjectCompileResultV1)]
    dispose.restype = None
    if abi() != 1:
        raise CodeObjectError(f"code-object library {path} has ABI {abi()}, expected 1")
    return library, compile_ir, dispose


@functools.cache
def _direct_gemm_compiler_info(path: str) -> CodeObjectCompilerInfo:
    return CodeObjectCompilerInfo(
        backend="rust-direct-gemm",
        llvm_version=(0, 0, 0),
        source="in-process Rust direct HSACO emitter",
        identity="rust-direct-gemm:gfx1151-wmma-f16-rcr-16x16x16/v2:encoder+elf-writer",
        path=path,
        details={
            "backend": "rust-direct-gemm",
            "supported_targets": ["gfx1151"],
            "implementation": "Rust gfx1151 encoder + ELF/AMDHSA writer",
            "uses_comgr": False,
            "uses_llvm": False,
            "uses_external_tools": False,
            "uses_cache": False,
        },
    )


def _emit_direct_gemm_in_process(
    path: str,
    serialized_ir: bytes,
    expected_symbol: str,
) -> tuple[bytes, CodeObjectTimings, CodeObjectCompilerInfo]:
    _library, compile_ir, dispose = _direct_library_functions(path)
    ir_storage = ctypes.create_string_buffer(serialized_ir)
    request = _DirectGemmCompileRequestV1(
        struct_size=ctypes.sizeof(_DirectGemmCompileRequestV1),
        serialized_ir=ctypes.cast(ir_storage, ctypes.c_void_p),
        serialized_ir_size=len(serialized_ir),
    )
    result = _CodeObjectCompileResultV1(
        struct_size=ctypes.sizeof(_CodeObjectCompileResultV1)
    )
    started = time.perf_counter()
    status = int(compile_ir(ctypes.byref(request), ctypes.byref(result)))
    elapsed = time.perf_counter() - started
    try:
        diagnostics = (
            ctypes.string_at(result.diagnostics, result.diagnostics_size).decode(
                "utf-8", "replace"
            )
            if result.diagnostics
            else ""
        )
        if status != 0 or result.status != 0:
            raise CodeObjectError(
                "direct GEMM IR compiler rejected the request"
                + (f": {diagnostics}" if diagnostics else "")
            )
        if not result.hsaco or not 0 < result.hsaco_size <= 16 * 1024 * 1024:
            raise CodeObjectError(
                f"direct GEMM compiler returned invalid image size {result.hsaco_size}"
            )
        hsaco = ctypes.string_at(result.hsaco, result.hsaco_size)
    finally:
        dispose(ctypes.byref(result))
    if not hsaco.startswith(b"\x7fELF") or expected_symbol.encode() not in hsaco:
        raise CodeObjectError("direct GEMM emitter returned an invalid image")
    return (
        hsaco,
        CodeObjectTimings(executable=elapsed),
        _direct_gemm_compiler_info(path),
    )


def build_direct_gfx1151_wmma_gemm_hsaco(
    serialized_ir: str | bytes,
) -> tuple[bytes, CodeObjectTimings, CodeObjectCompilerInfo]:
    """Compile serialized gfx1151 WMMA GEMM IR without LLVM, tools, or a cache."""
    if isinstance(serialized_ir, str):
        serialized_ir = serialized_ir.encode("utf-8")
    if not serialized_ir:
        raise ValueError("serialized_ir must be non-empty")
    symbol_match = _DIRECT_IR_KERNEL_SYMBOL.search(serialized_ir)
    if symbol_match is None:
        raise ValueError("serialized_ir has no supported kernel header")
    expected_symbol = symbol_match.group(1).decode("ascii")
    direct_library = _direct_library_path()
    if direct_library is None:
        raise CodeObjectError(
            "the direct Rust code-object library (rocke_codeobj_ffi) was not "
            f"found; build it with -DROCKE_BUILD_RUST_CODEOBJ=ON and set "
            f"{_DIRECT_LIB_ENV} or put it on the library search path"
        )
    return _emit_direct_gemm_in_process(direct_library, serialized_ir, expected_symbol)


@functools.cache
def _direct_rocke_ir_function(path: str):
    library, _compile_gemm, dispose = _direct_library_functions(path)
    try:
        compile_ir = library.rocke_codeobj_compile_rocke_ir_v1
    except AttributeError as error:
        raise CodeObjectError(
            f"direct code-object library {path} has no rocKE IR compiler: {error}"
        ) from error
    compile_ir.argtypes = [
        ctypes.POINTER(_DirectGemmCompileRequestV1),
        ctypes.POINTER(_CodeObjectCompileResultV1),
    ]
    compile_ir.restype = ctypes.c_int32
    return compile_ir, dispose


@functools.cache
def _direct_rocke_compiler_info(path: str) -> CodeObjectCompilerInfo:
    return CodeObjectCompilerInfo(
        backend="direct",
        llvm_version=None,
        source="in-process Rust rocKE IR compiler",
        identity=f"rust-direct:gfx1151:{_path_identity(path)}",
        path=path,
        details={
            "backend": "direct",
            "supported_targets": ["gfx1151"],
            "implementation": "Rust rocKE IR optimizer, isel, regalloc, encoder, ELF writer",
            "uses_comgr": False,
            "uses_llvm": False,
            "uses_external_tools": False,
            "uses_cache": False,
        },
    )


def _require_direct_library() -> str:
    direct_library = _direct_library_path()
    if direct_library is None:
        raise CodeObjectError(
            "the direct Rust code-object library (rocke_codeobj_ffi) was not "
            f"found; build it with -DROCKE_BUILD_RUST_CODEOBJ=ON and set "
            f"{_DIRECT_LIB_ENV} or put it on the library search path"
        )
    return direct_library


def build_direct_rocke_ir_hsaco(
    serialized_ir: str | bytes,
    *,
    isa: str = _DIRECT_ISA,
) -> tuple[bytes, CodeObjectTimings, CodeObjectCompilerInfo, str]:
    """Compile serialized ``ck.dsl.ir/v1`` to a gfx1151 HSACO without LLVM.

    Returns the image, timings, compiler provenance and the compiler's
    resource summary. Unsupported IR raises; there is no fallback.
    """
    if isa.split(":")[0] != _DIRECT_ISA:
        raise CodeObjectError(
            f"the direct code-object backend targets {_DIRECT_ISA}, not {isa!r}"
        )
    if isinstance(serialized_ir, str):
        serialized_ir = serialized_ir.encode("utf-8")
    if not serialized_ir:
        raise ValueError("serialized_ir must be non-empty")
    path = _require_direct_library()
    compile_ir, dispose = _direct_rocke_ir_function(path)
    ir_storage = ctypes.create_string_buffer(serialized_ir)
    request = _DirectGemmCompileRequestV1(
        struct_size=ctypes.sizeof(_DirectGemmCompileRequestV1),
        serialized_ir=ctypes.cast(ir_storage, ctypes.c_void_p),
        serialized_ir_size=len(serialized_ir),
    )
    result = _CodeObjectCompileResultV1(
        struct_size=ctypes.sizeof(_CodeObjectCompileResultV1)
    )
    started = time.perf_counter()
    status = int(compile_ir(ctypes.byref(request), ctypes.byref(result)))
    elapsed = time.perf_counter() - started
    try:
        diagnostics = (
            ctypes.string_at(result.diagnostics, result.diagnostics_size).decode(
                "utf-8", "replace"
            )
            if result.diagnostics
            else ""
        )
        if status != 0 or result.status != 0:
            raise CodeObjectError(
                "direct rocKE IR compiler rejected the kernel"
                + (f": {diagnostics}" if diagnostics else "")
            )
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


def loaded_compiler_info(
    backend: str | None = None,
) -> CodeObjectCompilerInfo | None:
    """Return selected compiler evidence; failed loads remain retryable."""
    selected = resolve_code_object_backend(backend)
    try:
        if selected == "direct":
            return _direct_rocke_compiler_info(_require_direct_library())
        return _comgr_compiler_info()
    except CodeObjectError:
        return None


def _require_compiler_info(
    backend: str | None = None,
) -> CodeObjectCompilerInfo:
    if resolve_code_object_backend(backend) == "direct":
        return _direct_rocke_compiler_info(_require_direct_library())
    return _comgr_compiler_info()


def _reject_direct_for_llvm(backend: str | None) -> None:
    if resolve_code_object_backend(backend) == "direct":
        raise CodeObjectError(
            "the direct code-object backend compiles rocKE IR, not LLVM IR; "
            "use compile_kernel or build_direct_rocke_ir_hsaco"
        )


def compiler_identity(backend: str | None = None) -> str:
    """Stable cache identity of the selected compiler implementation."""
    return _require_compiler_info(backend).identity


def supported_isas(backend: str | None = None) -> list[str]:
    """Targets known by the selected backend, without querying a GPU."""
    if resolve_code_object_backend(backend) == "direct":
        return [_DIRECT_ISA]
    from .comgr import supported_isas as comgr_supported_isas

    return comgr_supported_isas()


def _timings(timings) -> CodeObjectTimings:
    return CodeObjectTimings(
        bc=timings.bc,
        link=timings.link,
        relocatable=timings.relocatable,
        executable=timings.executable,
    )


def build_hsaco_from_llvm_ir(
    ir_text: str,
    *,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    options: list[str] | None = None,
    fresh_process: bool = False,
    backend: str | None = None,
) -> tuple[bytes, CodeObjectTimings]:
    """Compile one textual LLVM module through the selected backend."""
    _reject_direct_for_llvm(backend)
    from .comgr import build_hsaco_from_llvm_ir as comgr_build

    hsaco, timings = comgr_build(
        ir_text,
        isa=isa,
        options=list(options or ["-O3"]),
        fresh_process=fresh_process,
    )
    return hsaco, _timings(timings)


def build_hsaco_from_llvm_modules(
    ir_texts: Sequence[str],
    *,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    options: list[str] | None = None,
    backend: str | None = None,
) -> tuple[bytes, CodeObjectTimings]:
    """Compile and bitcode-link two or more LLVM modules into one HSACO."""
    ir_texts = tuple(ir_texts)
    if len(ir_texts) < 2:
        raise ValueError(
            "build_hsaco_from_llvm_modules links two or more modules; use "
            "build_hsaco_from_llvm_ir for one"
        )
    _reject_direct_for_llvm(backend)
    from .comgr import build_hsaco_from_llvm_modules as comgr_build

    hsaco, timings = comgr_build(ir_texts, isa=isa, options=list(options or ["-O3"]))
    return hsaco, _timings(timings)
