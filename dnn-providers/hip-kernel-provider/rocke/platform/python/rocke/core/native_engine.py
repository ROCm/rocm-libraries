# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ engine's shared library, ``librocke``, reached through its C ABI.

The engine exports plain ``extern "C"`` entry points, so Python needs no
compiled extension to use them: ctypes loads the library and calls them. This
module is that one route for lowering. It finds the library, refuses one built
for another binary ABI, and lowers a kernel held as serialized ``ck.dsl.ir/v1``
text (``rocke.core.ir_serialize.serialize``):

- :func:`lower_serialized_ir` -- AMDGPU LLVM IR, through
  ``rocke_lower_serialized_ir_to_llvm`` (``rocke/lower_llvm.h``);
- :func:`lower_serialized_ir_to_hip` -- HIP source, through
  ``rocke_lower_serialized_ir_to_hip_text`` (``rocke/lower_hip.h``);
- :func:`native_lowering_available` / :func:`require_native_lowering` -- whether
  this process can reach the library.

Which library: the one the caller names (a path, or the ``ROCKE_ONLINE_LIB``
environment variable; a named library that does not exist is an error, never a
hint), else the one installed beside the ``rocke`` package, where the provider's
test tree puts it::

    <dir>/rocke/            the package
    <dir>/librocke.so       the engine (rocke.dll on Windows)

``ROCKE_ONLINE_LIB=none`` (:data:`NO_SHARED_ENGINE`) says this process has no
shared engine, whatever is installed: the cpp backend then falls back to the
Python engine (or raises, under ``ROCKE_CPP_STRICT``), as in an installed wheel.

:func:`load_shared_engine` is the one loader. It loads a library once per
process, refuses to switch the process to another, and refuses one built from
other sources than this tree expects (``rocke.core.engine_provenance``), for
another binary ABI, or missing an entry point a caller here declares. The
portable-IR tooling (``rocke.portable_ir.src.online``) loads the engine
through it too, for its own entry points.

A refusal keeps the engine's status and reason: :class:`ArchDomainRefusal` for
an arch-domain refusal (``ROCKE_ERR_ARCH_DOMAIN``), otherwise a ``RuntimeError``
reading ``"<C entry>: lower failed for arch '<gfx>' (status <rocke_status_t>):
<reason>"``.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

from ..assets import package_parent
from .engine_provenance import expected_build_id

__all__ = [
    "BINARY_ABI",
    "NO_SHARED_ENGINE",
    "NO_TARGET_MESSAGE",
    "SHARED_ENGINE_ENV",
    "ArchDomainRefusal",
    "SharedEngineAbiMismatch",
    "SharedEngineNotFound",
    "StaleSharedEngine",
    "bind_function",
    "check_binary_abi",
    "check_build_id",
    "installed_shared_engine_path",
    "load_shared_engine",
    "loaded_shared_engine_path",
    "lower_serialized_ir",
    "lower_serialized_ir_to_hip",
    "native_lowering_available",
    "requested_shared_engine_path",
    "require_native_lowering",
    "shared_library_name",
]

#: Binary ABI the ctypes declarations here and in ``portable_ir/src/online.py``
#: are written against. MUST equal ``ROCKE_ABI_VERSION`` in
#: ``cpp/include/rocke/abi.h``: ctypes argtypes are a hand-written mirror of the
#: C signatures, and calling a library of another ABI through them is
#: memory-unsafe rather than merely wrong, so a mismatch is refused at load.
BINARY_ABI = 24

#: The environment variable naming the shared engine to load.
SHARED_ENGINE_ENV = "ROCKE_ONLINE_LIB"

#: The :data:`SHARED_ENGINE_ENV` value that says this process has no shared
#: engine.
NO_SHARED_ENGINE = "none"

#: The refusal of a lowering called without a target, in every Python entry.
NO_TARGET_MESSAGE = (
    "lowering requires an explicit gfx target; pass arch=... (there is no default)"
)

# The directory holding the rocke package: where an install puts the shared
# engine. Not rocke.assets.platform_root(), which is the source tree's
# platform/ directory; in a source checkout the two differ.
_PACKAGE_PARENT = package_parent()

# rocke_status_t (rocke/ir.h) and rocke_llvm_flavor_t (rocke/lower_llvm.h).
_ROCKE_OK = 0
_ROCKE_ERR_ARCH_DOMAIN = 6
_LLVM_FLAVOR_AUTO = 0


class SharedEngineNotFound(RuntimeError):
    """No shared engine was named and none is installed beside the package."""


class SharedEngineAbiMismatch(RuntimeError):
    """The shared engine was built for another binary ABI than this package, or
    predates an entry point this package calls."""


class StaleSharedEngine(RuntimeError):
    """The shared engine was built from other sources than this tree expects."""


class ArchDomainRefusal(NotImplementedError):
    """The C++ engine refused the kernel because its arch-domain table says the
    target cannot lower an intrinsic the kernel needs. The message is the
    engine's (``rocke.core.arch.domain.message``); a NotImplementedError, like
    the Python lowerer's ``ArchDomainError``."""


def shared_library_name() -> str:
    """The shared engine's file name on this platform."""
    if sys.platform == "win32":
        return "rocke.dll"
    if sys.platform == "darwin":
        return "librocke.dylib"
    return "librocke.so"


def requested_shared_engine_path(
    path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """The library the caller names: ``path``, else ``$ROCKE_ONLINE_LIB``; None
    when neither names one.

    A named library is a demand: when it does not exist this raises
    ``FileNotFoundError`` rather than letting a caller fall back to another
    library, whose output it would then report as this one's.
    ``ROCKE_ONLINE_LIB=none`` raises :class:`SharedEngineNotFound`.
    """
    named = path if path is not None else os.environ.get(SHARED_ENGINE_ENV)
    if not named:
        return None
    if path is None and named == NO_SHARED_ENGINE:
        raise SharedEngineNotFound(
            f"no shared C++ engine: {SHARED_ENGINE_ENV}={NO_SHARED_ENGINE} says "
            "this process has none"
        )
    if not os.path.exists(named):
        source = "the path argument" if path is not None else SHARED_ENGINE_ENV
        raise FileNotFoundError(
            f"librocke not found at {str(named)!r} (from {source}). "
            "Refusing to fall back to another library"
        )
    return Path(named)


def installed_shared_engine_path() -> Path | None:
    """The shared engine installed beside the ``rocke`` package, or None."""
    candidate = _PACKAGE_PARENT / shared_library_name()
    return candidate if candidate.is_file() else None


def check_binary_abi(library: ctypes.CDLL, path: str | os.PathLike[str]) -> None:
    """Refuse ``library`` (loaded from ``path``) unless it reports
    :data:`BINARY_ABI`.

    Runs before any other symbol is bound, so a library from another build is
    reported as such rather than as a ctypes failure on whichever symbol came
    first. A library that predates ``rocke_abi_version`` is itself a mismatch.
    """
    try:
        abi_version = library.rocke_abi_version
    except AttributeError:
        raise SharedEngineAbiMismatch(
            f"librocke at {str(path)!r} exports no rocke_abi_version, so it predates "
            f"the binary ABI contract (v{BINARY_ABI}) this package requires. Rebuild "
            "the library from this package's source"
        ) from None
    abi_version.argtypes = []
    abi_version.restype = ctypes.c_int
    reported = abi_version()
    if reported != BINARY_ABI:
        raise SharedEngineAbiMismatch(
            f"librocke at {str(path)!r} has binary ABI v{reported}, this package is "
            f"written against v{BINARY_ABI}. The library and the package are from "
            "different builds; rebuild the library from this package's source"
        )


def check_build_id(library: ctypes.CDLL, path: str | os.PathLike[str]) -> None:
    """Refuse ``library`` (loaded from ``path``) unless it was built from the
    sources this tree expects (``engine_provenance.expected_build_id``); accept
    it when the tree cannot tell (an installed tree without a provenance
    manifest)."""
    expected = expected_build_id()
    if expected is None:
        return
    try:
        build_id = library.rocke_build_id
    except AttributeError:
        reported = "<missing>"
    else:
        build_id.argtypes = []
        build_id.restype = ctypes.c_char_p
        reported = (build_id() or b"").decode()
    if reported != expected:
        raise StaleSharedEngine(
            "stale/mixed rocKE engine: "
            f"librocke at {str(path)!r} has build-id {reported}, but the selected "
            f"environment expects {expected}. Rebuild or reinstall the library "
            "and its provenance manifest from the same source snapshot"
        )


def bind_function(
    library: ctypes.CDLL,
    path: str | os.PathLike[str],
    name: str,
    argtypes: list[object],
    restype: object,
) -> None:
    """Declare the C signature of the entry point ``name`` of ``library``
    (loaded from ``path``). A library that lacks it predates it:
    :class:`SharedEngineAbiMismatch`, not a ctypes ``AttributeError``."""
    try:
        function = getattr(library, name)
    except AttributeError:
        raise SharedEngineAbiMismatch(
            f"librocke at {str(path)!r} predates the entry point {name}; "
            "rebuild the library from this package's source"
        ) from None
    function.argtypes = argtypes
    function.restype = restype


_LOAD_LOCK = threading.Lock()
_engine: ctypes.CDLL | None = None
_engine_path: Path | None = None


def _bind(library: ctypes.CDLL, path: Path) -> None:
    """Declare the C signatures this module calls."""
    text_out = ctypes.POINTER(ctypes.c_void_p)
    c_char_p, c_int = ctypes.c_char_p, ctypes.c_int
    for name, argtypes, restype in (
        (
            "rocke_lower_serialized_ir_to_llvm",
            [c_char_p, c_char_p, c_int, text_out, text_out],
            c_int,
        ),
        (
            "rocke_lower_serialized_ir_to_hip_text",
            [c_char_p, c_char_p, text_out, text_out],
            c_int,
        ),
        ("rocke_text_free", [ctypes.c_void_p], None),
        ("rocke_llvm_flavor_from_name", [c_char_p], c_int),
        ("rocke_llvm_flavor_count", [], c_int),
        ("rocke_llvm_flavor_at", [c_int], c_char_p),
    ):
        bind_function(library, path, name, argtypes, restype)


def load_shared_engine(path: str | os.PathLike[str]) -> ctypes.CDLL:
    """The process's shared engine, loaded from ``path`` on first use.

    The first call loads it and refuses it when it is stale
    (:class:`StaleSharedEngine`), of another binary ABI or missing an entry
    point (:class:`SharedEngineAbiMismatch`); later calls return it. A later
    call naming another library raises ``RuntimeError``: a process lowers with
    one engine.
    """
    global _engine, _engine_path
    resolved = Path(path).resolve()
    with _LOAD_LOCK:
        if _engine is not None:
            if resolved != _engine_path:
                raise RuntimeError(
                    f"rocKE engine already loaded from {str(_engine_path)!r}; "
                    f"refusing to switch this process to {str(path)!r}"
                )
            return _engine
        library = ctypes.CDLL(str(resolved))
        check_build_id(library, resolved)
        check_binary_abi(library, resolved)
        _bind(library, resolved)
        _engine, _engine_path = library, resolved
        return library


def loaded_shared_engine_path() -> Path | None:
    """The shared engine this process loaded, or None before the first load."""
    return _engine_path


def _shared_engine() -> ctypes.CDLL:
    """The shared engine this module lowers with: the library already loaded,
    else the one named (``ROCKE_ONLINE_LIB``) or installed beside the package."""
    named = os.environ.get(SHARED_ENGINE_ENV)
    if _engine is not None and not named:
        return _engine
    path = requested_shared_engine_path() or installed_shared_engine_path()
    if path is None:
        raise SharedEngineNotFound(
            f"no shared C++ engine: {SHARED_ENGINE_ENV} is unset and no "
            f"{shared_library_name()} is installed beside the rocke package "
            f"({_PACKAGE_PARENT}). Build the rocke_shared target "
            f"(ROCKE_BUILD_SHARED_ENGINE=ON) and set {SHARED_ENGINE_ENV} to the "
            "library it makes"
        )
    return load_shared_engine(path)


def require_native_lowering() -> None:
    """Load the shared engine, or raise :class:`SharedEngineNotFound` saying
    where it was looked for. A library that is found but cannot be loaded, is
    stale or is from another build raises its own error: that is a broken
    setup, not a missing one."""
    _shared_engine()


def native_lowering_available() -> bool:
    """Whether :func:`lower_serialized_ir` and :func:`lower_serialized_ir_to_hip`
    can run in this process: False when no shared engine is found; a broken one
    raises, as for :func:`require_native_lowering`."""
    try:
        require_native_lowering()
    except SharedEngineNotFound:
        return False
    return True


def _flavor_names(library: ctypes.CDLL) -> tuple[str, ...]:
    """The engine's concrete LLVM flavors, oldest first."""
    return tuple(
        library.rocke_llvm_flavor_at(index).decode("ascii")
        for index in range(library.rocke_llvm_flavor_count())
    )


def _take_text(library: ctypes.CDLL, pointer: ctypes.c_void_p) -> str | None:
    """The engine-allocated string at ``pointer`` (None for NULL), freed."""
    if not pointer.value:
        return None
    try:
        return ctypes.string_at(pointer.value).decode("utf-8", errors="replace")
    finally:
        library.rocke_text_free(pointer)


def _lowered(
    library: ctypes.CDLL,
    entry: str,
    arch: str,
    status: int,
    text: ctypes.c_void_p,
    message: ctypes.c_void_p,
) -> str:
    """The text a serialized-IR entry returned, or its refusal raised."""
    lowered, reason = _take_text(library, text), _take_text(library, message)
    if status == _ROCKE_OK and lowered is not None:
        return lowered
    reason = reason or "the engine could not allocate its message"
    if status == _ROCKE_ERR_ARCH_DOMAIN:
        raise ArchDomainRefusal(reason)
    raise RuntimeError(
        f"{entry}: lower failed for arch {arch!r} (status {status}): {reason}"
    )


def lower_serialized_ir(ir_text: str, arch: str, flavor: str) -> str:
    """Lower the serialized kernel ``ir_text`` to AMDGPU LLVM IR for ``arch`` at
    LLVM ``flavor`` with the C++ engine.

    ``flavor`` is a concrete flavor name (``"llvm20"`` ...): resolving one from
    the loaded compiler is the caller's choice to make
    (``rocke.core.lower_llvm``'s resolver is what the Python lowerer uses), so
    nothing here queries a compiler. The text is byte-identical to what the
    Python lowerer emits for the parsed kernel.

    Raises ``ValueError`` for an empty ``arch`` or an unknown ``flavor``,
    :class:`ArchDomainRefusal` or ``RuntimeError`` when the engine refuses (see
    the module docstring), and :class:`SharedEngineNotFound` when there is no
    shared engine.
    """
    if not arch:
        raise ValueError(NO_TARGET_MESSAGE)
    library = _shared_engine()
    flavor_id = library.rocke_llvm_flavor_from_name(flavor.encode("ascii", "replace"))
    if flavor_id == _LLVM_FLAVOR_AUTO:
        raise ValueError(
            f"lower_serialized_ir failed - flavor={flavor!r}, expected one of "
            f"{_flavor_names(library)}"
        )
    text, message = ctypes.c_void_p(), ctypes.c_void_p()
    status = library.rocke_lower_serialized_ir_to_llvm(
        ir_text.encode("utf-8"),
        arch.encode("ascii", "replace"),
        flavor_id,
        ctypes.byref(text),
        ctypes.byref(message),
    )
    return _lowered(
        library, "rocke_lower_serialized_ir_to_llvm", arch, status, text, message
    )


def lower_serialized_ir_to_hip(ir_text: str, arch: str) -> str:
    """Lower the serialized kernel ``ir_text`` to HIP source for ``arch`` with
    the C++ engine, with the HIP lowering's defaults (prologue included, launch
    bounds from the kernel): the twin of ``rocke.core.lower_hip``'s
    ``lower_kernel_to_hip(kernel, arch=arch)``.

    Raises ``ValueError`` for an empty ``arch``, ``RuntimeError`` when the
    engine refuses, and :class:`SharedEngineNotFound` when there is no shared
    engine.
    """
    if not arch:
        raise ValueError(NO_TARGET_MESSAGE)
    library = _shared_engine()
    text, message = ctypes.c_void_p(), ctypes.c_void_p()
    status = library.rocke_lower_serialized_ir_to_hip_text(
        ir_text.encode("utf-8"),
        arch.encode("ascii", "replace"),
        ctypes.byref(text),
        ctypes.byref(message),
    )
    return _lowered(
        library, "rocke_lower_serialized_ir_to_hip_text", arch, status, text, message
    )
