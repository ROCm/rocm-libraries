# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python ctypes wrapper around `libamd_comgr` for in-process compilation.

The chain we drive:

    LLVM IR text (utf-8)
      -> AMD_COMGR_DATA_KIND_SOURCE  (lang = LLVM_IR)
      -> AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC          -> BC
      -> AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE     -> ELF relocatable
      -> AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE -> HSA code object

Several kernels destined for one code object (`build_hsaco_from_llvm_modules`)
enter as one SOURCE each and are joined by AMD_COMGR_ACTION_LINK_BC_TO_BC
before codegen, so a single module -- and a single AMDGPU metadata note
listing every kernel -- reaches the backend. Linking relocatables instead
would leave one metadata note per object, and the HIP runtime reads only one.

The resulting HSACO bytes are returned to Python and can be handed
straight to `hipModuleLoadData` (see `hip_module.py`). No subprocesses,
no `<hip/hip_runtime.h>` parsing, no clang spawn.

The library is loaded from the default ROCm library locations or the dynamic
linker search path. ABI definitions mirror ROCm's `amd_comgr.h`.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ._ctypes_bind import _LazyFn
from .runtime_coexistence import (
    _IS_WINDOWS,
    _add_dll_dir,
    _candidate_lib_paths,
    _in_rocm_sdk_core,
    _mapped_lib,
    _rocm_sdk_core_release,
)


# Status codes.
AMD_COMGR_STATUS_SUCCESS = 0

# Data kinds.
AMD_COMGR_DATA_KIND_SOURCE = 0x1
AMD_COMGR_DATA_KIND_BC = 0x6
AMD_COMGR_DATA_KIND_RELOCATABLE = 0x7
AMD_COMGR_DATA_KIND_EXECUTABLE = 0x8
AMD_COMGR_DATA_KIND_BYTES = 0x9

# Languages.
AMD_COMGR_LANGUAGE_LLVM_IR = 0x4

# Action kinds.
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 0x2
AMD_COMGR_ACTION_LINK_BC_TO_BC = 0x3
AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 0x4
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 0x7


class ComgrError(RuntimeError):
    pass


class ComgrProcessDied(ComgrError):
    """The process compiling one module (``fresh_process=True``) died before
    it reported a result or an error: a toolchain crash, not a compile error."""


def _load_lib() -> ctypes.CDLL:
    # Pair the loader with ``hip_module._load_lib`` so the two halves of
    # the process always share a single HIP/comgr runtime instance. See
    # ``_torch_bundled_lib`` in ``runtime_coexistence`` for why a
    # torch-shipped libamd_comgr is preferred over /opt/rocm when torch is
    # in the process.
    # Every candidate's failure is kept rather than overwritten. Which paths were
    # tried, in order, is the whole diagnosis when resolution lands somewhere
    # unexpected, and the interesting failure is usually the first -- the library
    # that was supposed to load -- not the last.
    # A different explicit COMGR beside an already mapped one can load a
    # second LLVM and abort the process before COMGR can return an error.
    # Only the single-copy case identifies that conflict without choosing a
    # policy for multiple mappings. A non-ELF override cannot load as COMGR;
    # let the existing candidate fallback handle invalid or unreadable files.
    override = os.environ.get("ROCKE_COMGR_LIB")
    if override and not _IS_WINDOWS and os.path.isfile(override):
        mapped = _mapped_lib("amd_comgr", require_unique=True)
        if mapped is not None:
            try:
                same_file = os.path.samefile(override, mapped)
            except OSError:
                pass
            else:
                if not same_file:
                    try:
                        with open(override, "rb") as candidate:
                            is_elf = candidate.read(4) == b"\x7fELF"
                    except OSError:
                        is_elf = False
                    if is_elf:
                        raise ComgrError(
                            f"ROCKE_COMGR_LIB={override!r} differs from the only "
                            f"mapped COMGR {mapped!r}; opening a second COMGR can "
                            "abort LLVM. Point ROCKE_COMGR_LIB at the mapped file "
                            "or compile in a separate process before loading torch."
                        )
    failures: List[Tuple[str, OSError]] = []
    for p in _candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"]):
        try:
            _add_dll_dir(p)
            return ctypes.CDLL(p)
        except OSError as e:
            failures.append((p, e))
    name = "amd_comgr.dll" if _IS_WINDOWS else "libamd_comgr.so"
    if not failures:
        raise ComgrError(
            f"cannot load {name}: no candidate path was produced. Set "
            f"ROCKE_COMGR_LIB to an explicit library, or make one discoverable."
        )
    tried = "\n".join(f"  {path}: {exc}" for path, exc in failures)
    raise ComgrError(
        f"cannot load {name}; {len(failures)} candidate(s) failed:\n{tried}"
    )


# Lazy: import order does not matter until the first query or compilation.
# See ``runtime_coexistence._torch_bundled_lib`` for discovery precedence.
_lib: Optional[ctypes.CDLL] = None
_lib_lock = threading.RLock()
_llvm_version_lib: ctypes.CDLL | None = None
_compiler_info: CompilerInfo | None = None


def _resolve_lib() -> ctypes.CDLL:
    """Retain a successful load; failed loads may recover on a later call."""
    global _lib
    with _lib_lock:
        if _lib is None:
            _lib = _load_lib()
        return _lib


def resolved_lib_path() -> Optional[str]:
    """Path of the ``libamd_comgr`` this module will load (torch-bundled
    preferred over ``/opt/rocm``; see :func:`runtime_coexistence._torch_bundled_lib`).

    Returns the already-loaded lib's path once :func:`_resolve_lib` has run,
    else the first existing candidate. Pure lookup -- does NOT ``dlopen``, so it
    is a discovery hint, not evidence of the loaded compiler version.
    """
    if _lib is not None:
        return getattr(_lib, "_name", None)
    try:
        cands = _candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"])
    except Exception:
        return None
    for p in cands:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return cands[0] if cands else None


def _parse_rocm_version(text: str) -> Optional[Tuple[int, int]]:
    head = str(text).strip().split("-", 1)[0]
    parts = head.split(".")
    try:
        return int(parts[0]), (int(parts[1]) if len(parts) >= 2 else 0)
    except (IndexError, ValueError):
        return None


def _read_rocm_version_file(path: str) -> Optional[Tuple[int, int]]:
    try:
        with open(path) as fh:
            return _parse_rocm_version(fh.read())
    except OSError:
        return None


def _torch_package_hip_text(text: str) -> Optional[str]:
    """The ``hip`` value of a torch package's ``version.py`` text
    (``hip: Optional[str] = '7.2.53211-...'``), or ``None`` when the line is
    absent or holds ``None`` (a build without ROCm)."""
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("hip"):
            continue
        rest = s[3:].lstrip()
        if not rest or rest[0] not in ":=":
            continue  # another name that starts with "hip"
        if "=" not in rest:
            return None
        value = rest.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
            return value[1:-1]
        return None
    return None


def _torch_package_rocm_version(lib_path: str) -> Optional[Tuple[int, int]]:
    """ROCm vintage of a comgr library inside a torch package
    (``<site>/torch/lib/libamd_comgr.so``), read from the package's
    ``version.py`` without importing torch; ``None`` when the library is not
    in a torch package's ``lib`` directory.

    A torch package carries no ``.info/version``, so without this an explicit
    ``ROCKE_COMGR_LIB`` naming torch's comgr in a process that has not imported
    torch was dated by ``/opt/rocm``: the node's ROCm, not the library's.
    """
    libdir = os.path.dirname(lib_path)
    pkg = os.path.dirname(libdir)
    if os.path.basename(libdir) != "lib" or os.path.basename(pkg) != "torch":
        return None
    try:
        with open(os.path.join(pkg, "version.py"), encoding="utf-8") as fh:
            hip = _torch_package_hip_text(fh.read())
    except OSError:
        return None
    return _parse_rocm_version(hip) if hip else None


def resolved_lib_rocm_version() -> Optional[Tuple[int, int]]:
    """Best-effort package metadata for diagnostics, not LLVM compatibility.

    Derived from the resolved COMGR path (:func:`resolved_lib_path`):

      * a path under the imported torch package -> ``torch.version.hip``;
      * in a torch package's ``lib`` directory (torch not imported, e.g. an
        explicit ``ROCKE_COMGR_LIB``) -> the ``hip`` entry of that package's
        ``version.py`` (read, never imported);
      * a path under TheRock's ``_rocm_sdk_core`` wheel -> the wheel's ROCm
        release (``rocm_sdk.__version__`` or ``rocm-sdk-core`` metadata);
      * a ROCm install tree (``<root>/lib/libamd_comgr.so`` or, on a packaged
        install, ``<root>/core-<X>/lib/libamd_comgr.so``) -> its root
        ``.info/version``, with ``/opt/rocm/.info/version`` as a final fallback.

    Returns ``None`` if no usable path or version is found. Compiler selection
    and validation use :func:`loaded_compiler_info` instead.
    """
    path = resolved_lib_path()
    if not path:
        return None
    try:
        rp = os.path.realpath(path)
    except Exception:
        rp = path
    # torch-bundled comgr -> torch's ROCm vintage (matches what _load_lib picks
    # when torch is in the process).
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        tfile = getattr(torch_mod, "__file__", None)
        if tfile:
            try:
                tdir = os.path.realpath(os.path.dirname(tfile))
            except Exception:
                tdir = os.path.dirname(tfile)
            if rp == tdir or rp.startswith(tdir + os.sep):
                ver = getattr(getattr(torch_mod, "version", None), "hip", None)
                return _parse_rocm_version(ver) if ver else None
    # torch-bundled comgr of a torch that is not imported -> its version.py.
    for cand in (rp, path):
        ver = _torch_package_rocm_version(cand)
        if ver is not None:
            return ver
    # TheRock wheel comgr -> the wheel's ROCm release. Without this the climb
    # below finds no .info/version inside site-packages and falls back to
    # /opt/rocm, reporting the host's release for the wheel's LLVM.
    if _in_rocm_sdk_core(rp):
        return _rocm_sdk_core_release()
    # ROCm tree -> the install *root's* ``.info/version``. Climb from the lib
    # dir collecting every ``.info/version`` we pass. This is deliberately a
    # climb, not a fixed ``dirname(dirname(path))``: a packaged ROCm 7.2 keeps
    # comgr in a versioned ``core-<X>/lib`` subdir (e.g.
    # ``/opt/rocm-7.2.0/core-7.13/lib``) which has its OWN ``.info/version``
    # recording the *component* version (7.13.0) -- NOT the ROCm release. The
    # ROCm release (7.2.0) lives in the top-level ``/opt/rocm-7.2.0/.info/
    # version``. For this informational package query, prefer the
    # ``.info/version`` sitting in a directory whose name looks like a ROCm
    # install root (``rocm`` / ``rocm-X.Y.Z``); failing that, the highest
    # (closest to ``/``) one we found.
    found: List[Tuple[str, Tuple[int, int]]] = []
    d = os.path.dirname(rp)
    prev = None
    while d and d != prev:
        ver = _read_rocm_version_file(os.path.join(d, ".info", "version"))
        if ver is not None:
            found.append((d, ver))
        prev = d
        d = os.path.dirname(d)
    for dirpath, ver in found:
        base = os.path.basename(dirpath)
        if base == "rocm" or base.startswith("rocm-") or base.startswith("rocm_"):
            return ver
    if found:
        # No obvious ``rocm`` root in the name -> the outermost match is the
        # install root (component subdirs are always deeper than it).
        return found[-1][1]
    return _read_rocm_version_file("/opt/rocm/.info/version")


def prefer_bundled_lib() -> Optional[Tuple[int, int]]:
    """Import torch if available so discovery can prefer its bundled COMGR.

    Call before the first compiler query, automatic lowering, or compilation.
    Once COMGR is loaded, later imports do not replace that retained handle.
    Ordinary library discovery never imports torch; this entrypoint hook does
    so explicitly and tolerates an unavailable torch installation.

    Returns legacy ROCm package metadata for informational callers. Use
    loaded_compiler_info() for compiler-version evidence and provenance.
    """
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401 -- pulls the bundled (newest) comgr into the process
        except Exception:
            pass
    return resolved_lib_rocm_version()


@dataclass(frozen=True)
class CompilerInfo:
    """Compiler evidence and its origin, tied to one loaded COMGR handle.

    Paths come from the dynamic loader, not filesystem version metadata.
    requested_comgr preserves the loader input when an actual module path
    cannot be obtained. A missing version or path is explicitly unknown.
    """

    llvm_version: tuple[int, int, int] | None
    source: str
    requested_comgr: str | None
    comgr_path: str | None
    query_library_path: str | None

    def describe(self) -> str:
        """Explain which loaded compiler supplied a compatibility decision."""
        version = (
            ".".join(map(str, self.llvm_version)) if self.llvm_version else "unknown"
        )
        return (
            f"LLVM {version} via {self.source}; "
            f"COMGR={self.comgr_path or self.requested_comgr!r}; "
            f"query library={self.query_library_path!r}"
        )


def _library_for_symbol(fn) -> str | None:
    """Ask the loader which mapped library owns a function address."""
    try:
        address = ctypes.cast(fn, ctypes.c_void_p)
        if _IS_WINDOWS:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            module = ctypes.c_void_p()
            get_module = kernel32.GetModuleHandleExW
            get_module.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            get_module.restype = ctypes.c_int
            # FROM_ADDRESS | UNCHANGED_REFCOUNT: inspect an existing module.
            if not get_module(0x6, address, ctypes.byref(module)):
                return None
            get_name = kernel32.GetModuleFileNameW
            get_name.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
            get_name.restype = ctypes.c_uint
            buf = ctypes.create_unicode_buffer(32768)
            size = get_name(module, buf, len(buf))
            return buf.value if 0 < size < len(buf) else None

        class DlInfo(ctypes.Structure):
            _fields_ = [
                ("filename", ctypes.c_char_p),
                ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p),
                ("address", ctypes.c_void_p),
            ]

        lookup = ctypes.CDLL(None).dladdr
        lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        lookup.restype = ctypes.c_int
        info = DlInfo()
        if lookup(address, ctypes.byref(info)) and info.filename:
            return os.fsdecode(info.filename)
    except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
        pass
    return None


def loaded_compiler_info() -> CompilerInfo | None:
    """Load COMGR and query its compiler, retaining version and provenance.

    LLVMGetVersion can be exported through COMGR's shared dependencies.
    When hidden, preprocess Clang's built-in macros through that same COMGR.
    No GPU, external compiler executable, or release metadata is needed.
    None means loading failed; llvm_version=None means this loaded compiler
    could not be queried. Neither case invents a version from another install.
    Failed loads may be retried. A successful load, including an unqueryable
    compiler, is retained for the process lifetime.
    """
    global _llvm_version_lib, _compiler_info
    with _lib_lock:
        try:
            lib = _resolve_lib()
        except (ComgrError, OSError):
            return None
        if _llvm_version_lib is lib:
            return _compiler_info
        version = None
        source = "unavailable"
        query_path = None
        try:
            query = lib.LLVMGetVersion
        except AttributeError:
            query = None
        if query is not None:
            query.argtypes = [ctypes.POINTER(ctypes.c_uint)] * 3
            query.restype = None
            values = [ctypes.c_uint() for _ in range(3)]
            query(*(ctypes.byref(v) for v in values))
            if values[0].value:
                version = tuple(v.value for v in values)
                source = "LLVMGetVersion"
                query_path = _library_for_symbol(query)
        if version is None:
            try:
                version = _probe_llvm_version(lib)
                if version:
                    source = "COMGR preprocessing"
                    query_path = _library_for_symbol(lib.amd_comgr_do_action)
            except (AttributeError, ComgrError, OSError, ValueError):
                pass
        try:
            comgr_path = _library_for_symbol(lib.amd_comgr_get_version)
        except AttributeError:
            comgr_path = None
        info = CompilerInfo(
            version, source, getattr(lib, "_name", None), comgr_path, query_path
        )
        _llvm_version_lib, _compiler_info = lib, info
        return info


def _require_compiler_info() -> CompilerInfo:
    """Require version evidence before automatic lowering or compilation."""
    info = loaded_compiler_info()
    if info is None or info.llvm_version is None:
        reason = "COMGR could not be loaded" if info is None else info.describe()
        raise ComgrError(
            f"Cannot determine LLVM flavor: {reason}. "
            "Select a queryable COMGR library for compilation, or set an explicit "
            "LLVM flavor for offline IR emission."
        )
    return info


def _assert_ir_flavor_matches_lib(ir_text: str) -> None:
    """Reject a known p8-generation mismatch against the loaded compiler.

    The guard is independent of emission overrides: selecting a flavor does
    not change the compiler that will consume it. Unknown compiler evidence
    is an error; with a known compiler, unrecognised IR layouts are left to COMGR.
    """
    from ..core.lower_llvm import (
        _datalayout_kind_for_flavor,
        _datalayout_kind_from_ir,
        _flavor_for_llvm,
    )

    info = _require_compiler_info()
    ir_kind = _datalayout_kind_from_ir(ir_text)
    if ir_kind is None:
        return
    lib_flavor = _flavor_for_llvm(info.llvm_version[0])
    lib_kind = _datalayout_kind_for_flavor(lib_flavor)
    if lib_kind is None or lib_kind is ir_kind:
        return
    raise ComgrError(
        "LLVM IR flavor / loaded compiler mismatch: IR datalayout is "
        f"{ir_kind.describe()} but the loaded compiler uses {lib_kind.describe()}: "
        f"{info.describe()}. "
        "Emit IR for the loaded compiler or select a compatible COMGR library "
        "before lowering."
    )


# Opaque handles are returned as a struct containing a single uint64_t.
class _Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


# Type aliases.
_DataSet = _Handle
_Data = _Handle
_ActionInfo = _Handle


def _bind(name: str, restype, *argtypes) -> _LazyFn:
    # Lazy ctypes wrapper: resolves on first call against the shared
    # comgr lib chosen by ``_load_lib`` above.
    return _LazyFn(name, list(argtypes), restype, _resolve_lib)


# ABI bindings. We list only what we use.
_status_string = _bind(
    "amd_comgr_status_string",
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
)

_create_data_set = _bind(
    "amd_comgr_create_data_set", ctypes.c_int, ctypes.POINTER(_DataSet)
)
_destroy_data_set = _bind("amd_comgr_destroy_data_set", ctypes.c_int, _DataSet)
_create_data = _bind(
    "amd_comgr_create_data", ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Data)
)
_release_data = _bind("amd_comgr_release_data", ctypes.c_int, _Data)
_set_data = _bind(
    "amd_comgr_set_data", ctypes.c_int, _Data, ctypes.c_size_t, ctypes.c_char_p
)
_set_data_name = _bind("amd_comgr_set_data_name", ctypes.c_int, _Data, ctypes.c_char_p)
_data_set_add = _bind("amd_comgr_data_set_add", ctypes.c_int, _DataSet, _Data)
_action_data_count = _bind(
    "amd_comgr_action_data_count",
    ctypes.c_int,
    _DataSet,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
)
_action_data_get_data = _bind(
    "amd_comgr_action_data_get_data",
    ctypes.c_int,
    _DataSet,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.POINTER(_Data),
)
_get_data = _bind(
    "amd_comgr_get_data",
    ctypes.c_int,
    _Data,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
)

_create_action_info = _bind(
    "amd_comgr_create_action_info", ctypes.c_int, ctypes.POINTER(_ActionInfo)
)
_destroy_action_info = _bind("amd_comgr_destroy_action_info", ctypes.c_int, _ActionInfo)
_action_info_set_isa_name = _bind(
    "amd_comgr_action_info_set_isa_name", ctypes.c_int, _ActionInfo, ctypes.c_char_p
)
_action_info_set_language = _bind(
    "amd_comgr_action_info_set_language", ctypes.c_int, _ActionInfo, ctypes.c_int
)
_action_info_set_options = _bind(
    "amd_comgr_action_info_set_option_list",
    ctypes.c_int,
    _ActionInfo,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_size_t,
)

_do_action = _bind(
    "amd_comgr_do_action", ctypes.c_int, ctypes.c_int, _ActionInfo, _DataSet, _DataSet
)

_get_isa_count = _bind(
    "amd_comgr_get_isa_count", ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)
)
_get_isa_name = _bind(
    "amd_comgr_get_isa_name",
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
)


def _check(s: int, where: str) -> None:
    if s != AMD_COMGR_STATUS_SUCCESS:
        msg = ctypes.c_char_p()
        _status_string(s, ctypes.byref(msg))
        raise ComgrError(
            f"{where}: status={s} ({msg.value.decode() if msg.value else ''})"
        )


def supported_isas() -> List[str]:
    """Every ISA name the loaded comgr can target, e.g. ``amdgcn-amd-amdhsa--gfx950``.

    :func:`build_hsaco_from_llvm_ir` raises the same :class:`ComgrError` for a
    target comgr does not know and for IR that fails to compile; asking first
    is how a caller tells the two apart. Raises :class:`ComgrError` when the
    library cannot be loaded.
    """
    count = ctypes.c_size_t(0)
    _check(_get_isa_count(ctypes.byref(count)), "get_isa_count")
    names = []
    for i in range(count.value):
        name = ctypes.c_char_p()
        _check(_get_isa_name(i, ctypes.byref(name)), "get_isa_name")
        names.append(name.value.decode("utf-8"))
    return names


@dataclass
class ComgrTimings:
    """Per-stage COMGR wall time in seconds, in pipeline order.

    ``bc`` compiles each IR module to bitcode, ``link`` links several bitcode
    modules into one (``LINK_BC_TO_BC``; 0.0 for a single module),
    ``relocatable`` code-generates and ``executable`` links the code object.
    Construct it by keyword: ``link`` sits between ``bc`` and ``relocatable``.
    """

    bc: float = 0.0
    link: float = 0.0
    relocatable: float = 0.0
    executable: float = 0.0

    @property
    def total(self) -> float:
        return self.bc + self.link + self.relocatable + self.executable


def _extract_first(data_set: _DataSet, kind: int) -> bytes:
    count = ctypes.c_size_t(0)
    _check(_action_data_count(data_set, kind, ctypes.byref(count)), "action_data_count")
    if count.value == 0:
        raise ComgrError(f"no output of kind {kind} produced")
    data = _Data()
    _check(
        _action_data_get_data(data_set, kind, 0, ctypes.byref(data)),
        "action_data_get_data",
    )

    try:
        size = ctypes.c_size_t(0)
        _check(_get_data(data, ctypes.byref(size), None), "get_data (size)")
        buf = ctypes.create_string_buffer(size.value)
        _check(_get_data(data, ctypes.byref(size), buf), "get_data (read)")
        out = bytes(buf.raw[: size.value])
    finally:
        _release_data(data)
    return out


def _build_in_fresh_process(
    ir_text: str, *, isa: str, options: List[str]
) -> Tuple[bytes, ComgrTimings]:
    """Compile one module in a new Python process that compiles nothing else.

    The child (:mod:`rocke.runtime._comgr_fresh_process`) loads the comgr
    library this process resolved (passed as ``ROCKE_COMGR_LIB``, so a
    torch-bundled library is kept even though the child does not import torch),
    imports this rocke package (its root is put first on ``PYTHONPATH``) and
    runs with comgr's own on-disk cache off (``AMD_COMGR_CACHE=0``).
    Raises :class:`ComgrError` with the child's message for a compile error and
    :class:`ComgrProcessDied` when the child dies.
    """
    import json
    import subprocess
    import tempfile

    # The parent's own checks first: the same library, the same flavor gate.
    lib = _resolve_lib()
    _assert_ir_flavor_matches_lib(ir_text)
    env = dict(os.environ)
    lib_path = getattr(lib, "_name", None) or resolved_lib_path()
    if lib_path:
        env["ROCKE_COMGR_LIB"] = lib_path
    # The child compiles the module itself: comgr's on-disk cache could hand
    # back an object some other process compiled with another history.
    env["AMD_COMGR_CACHE"] = "0"
    pkg_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (pkg_root, env.get("PYTHONPATH", "")) if p
    )
    with tempfile.TemporaryDirectory(prefix="rocke-comgr-") as tmp:
        out_path = os.path.join(tmp, "out.bin")
        header = {"isa": isa, "options": list(options), "out": out_path}
        proc = subprocess.run(
            [sys.executable, "-m", "rocke.runtime._comgr_fresh_process"],
            input=json.dumps(header).encode("utf-8") + b"\n" + ir_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        tail = proc.stderr.decode("utf-8", "replace")[-2000:]
        if proc.returncode == 2:  # the child's ComgrError, message on stderr
            raise ComgrError(tail.strip())
        if proc.returncode == 1:  # an exception other than ComgrError
            raise ComgrError(f"the fresh compile process failed: {tail.strip()}")
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise ComgrProcessDied(
                f"the fresh compile process died (exit status {proc.returncode})"
                + (f": {tail.strip()}" if tail.strip() else "")
            )
        with open(out_path, "rb") as fh:
            meta = json.loads(fh.readline())
            hsaco = fh.read()
    if len(hsaco) != meta["size"]:
        raise ComgrProcessDied(
            f"the fresh compile process wrote {len(hsaco)} of {meta['size']} bytes"
        )
    timings = ComgrTimings(
        bc=meta["bc"], relocatable=meta["relocatable"], executable=meta["executable"]
    )
    return hsaco, timings


def _probe_llvm_version(lib: ctypes.CDLL) -> tuple[int, int, int] | None:
    """Preprocess a version marker through COMGR when LLVM symbols are hidden."""

    def bind(name, *args):
        fn = getattr(lib, "amd_comgr_" + name)
        fn.argtypes, fn.restype = list(args), ctypes.c_int
        return fn

    hp = ctypes.POINTER(_Handle)
    create_set = bind("create_data_set", hp)
    destroy_set = bind("destroy_data_set", _Handle)
    create_data = bind("create_data", ctypes.c_int, hp)
    release_data = bind("release_data", _Handle)
    set_data = bind("set_data", _Handle, ctypes.c_size_t, ctypes.c_char_p)
    set_name = bind("set_data_name", _Handle, ctypes.c_char_p)
    add_data = bind("data_set_add", _Handle, _Handle)
    create_info = bind("create_action_info", hp)
    destroy_info = bind("destroy_action_info", _Handle)
    get_isa = bind("get_isa_name", ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p))
    set_isa = bind("action_info_set_isa_name", _Handle, ctypes.c_char_p)
    set_language = bind("action_info_set_language", _Handle, ctypes.c_int)
    action = bind("do_action", ctypes.c_int, _Handle, _Handle, _Handle)
    get_output = bind(
        "action_data_get_data", _Handle, ctypes.c_int, ctypes.c_size_t, hp
    )
    get_data = bind(
        "get_data", _Handle, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p
    )
    handles = []

    def check(status):
        if status != AMD_COMGR_STATUS_SUCCESS:
            raise ComgrError(f"COMGR compiler-version probe failed: status={status}")

    def acquire(create, destroy, *args):
        handle = _Handle()
        check(create(*args, ctypes.byref(handle)))
        handles.append((destroy, handle))
        return handle

    try:
        inputs = acquire(create_set, destroy_set)
        outputs = acquire(create_set, destroy_set)
        source = acquire(create_data, release_data, AMD_COMGR_DATA_KIND_SOURCE)
        info = acquire(create_info, destroy_info)
        payload = (
            b"ROCKE_LLVM_VERSION __clang_major__ __clang_minor__ __clang_patchlevel__\n"
        )
        check(set_data(source, len(payload), payload))
        check(set_name(source, b"rocke_compiler_version.cl"))
        check(add_data(inputs, source))
        isa = ctypes.c_char_p()
        check(get_isa(0, ctypes.byref(isa)))
        check(set_isa(info, isa.value))
        check(set_language(info, 1))  # AMD_COMGR_LANGUAGE_OPENCL_1_2
        check(action(0, info, inputs, outputs))  # SOURCE_TO_PREPROCESSOR
        output = acquire(
            get_output, release_data, outputs, AMD_COMGR_DATA_KIND_SOURCE, 0
        )
        size = ctypes.c_size_t()
        check(get_data(output, ctypes.byref(size), None))
        data = ctypes.create_string_buffer(size.value)
        check(get_data(output, ctypes.byref(size), data))
        match = re.search(rb"(?m)^ROCKE_LLVM_VERSION (\d+) (\d+) (\d+)\s*$", data.value)
        if match and int(match[1]) > 0:
            return tuple(int(match[i]) for i in (1, 2, 3))
        return None
    finally:
        for destroy, handle in reversed(handles):
            destroy(handle)


def build_hsaco_from_llvm_ir(
    ir_text: str,
    *,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    options: Optional[List[str]] = None,
    fresh_process: bool = False,
) -> Tuple[bytes, ComgrTimings]:
    """Compile LLVM IR text to a loadable HSACO blob, all in-process.

    Returns the (hsaco_bytes, timings) tuple. `hsaco_bytes` can be passed
    directly to `hipModuleLoadData`.

    ``fresh_process=True`` compiles the module in a new process instead, as
    the only compile of that process (:func:`_build_in_fresh_process`). Use it
    for IR whose code generation would otherwise depend on what this process
    compiled before. The known case is ``llvm.amdgcn.iglp.opt(i32 1)``: LLVM's
    AMDGPU single-wave small-GEMM IGroupLP strategy keeps its DS-write counts
    in file-scope statics that it adds to and never resets, so the schedule of
    a module depends on every earlier ``iglp_opt(1)`` compile in the process.
    A process that compiles one module gives the code the strategy is written
    for (its own assertion expects those counts to be zero) and the same code
    object every time. The default keeps the in-process compile.
    """
    if fresh_process:
        options = list(options or ["-O3"])
        return _build_in_fresh_process(ir_text, isa=isa, options=options)
    return _build_hsaco((ir_text,), isa=isa, options=options)


def build_hsaco_from_llvm_modules(
    ir_texts: Sequence[str],
    *,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    options: Optional[List[str]] = None,
) -> Tuple[bytes, ComgrTimings]:
    """Compile several LLVM IR modules into ONE loadable HSACO blob.

    Each text is compiled to bitcode on its own, exactly as
    :func:`build_hsaco_from_llvm_ir` would, then the bitcode modules are linked
    (``LINK_BC_TO_BC``) and the linked module is code-generated once. The
    object therefore carries one AMDGPU metadata note naming every kernel.
    ``options`` are module-wide codegen options, shared by every kernel.

    Internal-linkage globals (rocKE's LDS pools) are renamed by the linker
    when two modules spell the same name; two external definitions of one
    name -- two kernels sharing a symbol -- fail the link.
    """
    ir_texts = tuple(ir_texts)
    if len(ir_texts) < 2:
        raise ValueError(
            "build_hsaco_from_llvm_modules links two or more modules; use "
            "build_hsaco_from_llvm_ir for one"
        )
    return _build_hsaco(ir_texts, isa=isa, options=options)


def _require_count(data_set: _DataSet, kind: int, expected: int, where: str) -> None:
    count = ctypes.c_size_t(0)
    _check(_action_data_count(data_set, kind, ctypes.byref(count)), "action_data_count")
    if count.value != expected:
        raise ComgrError(
            f"{where}: produced {count.value} outputs, expected {expected}"
        )


def _build_hsaco(
    ir_texts: Tuple[str, ...],
    *,
    isa: str,
    options: Optional[List[str]],
) -> Tuple[bytes, ComgrTimings]:
    """The one comgr pipeline behind both public entry points.

    A single module takes exactly the historical path (one SOURCE named
    ``kernel.ll``, no link stage), so single-kernel objects are unchanged.
    """
    options = list(options or ["-O3"])
    linking = len(ir_texts) > 1

    # Query the loaded compiler before submitting IR: a known p8-generation
    # mismatch can abort inside codegen. The guard reports version provenance
    # and raises a catchable error instead.
    _resolve_lib()
    for ir_text in ir_texts:
        _assert_ir_flavor_matches_lib(ir_text)

    # Handles are created lazily below; declare them up front so the
    # ``finally`` block can release whatever was successfully created even
    # when an intermediate ``_check`` raises (the common case during a
    # validity/heuristics sweep over borderline specs).
    in_set = None
    sources = []
    info = None
    bc_set = None
    linked_set = None
    reloc_set = None
    exe_set = None

    try:
        # Input data set (LLVM IR text wrapped as SOURCE).
        in_set = _DataSet()
        _check(_create_data_set(ctypes.byref(in_set)), "create_data_set(in)")
        for index, ir_text in enumerate(ir_texts):
            src = _Data()
            _check(
                _create_data(AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(src)),
                "create_data(src)",
            )
            sources.append(src)
            payload = ir_text.encode("utf-8")
            name = f"kernel_{index}.ll" if linking else "kernel.ll"
            _check(_set_data(src, len(payload), payload), "set_data(src)")
            _check(_set_data_name(src, name.encode("utf-8")), "set_data_name(src)")
            _check(_data_set_add(in_set, src), "data_set_add(src)")

        # Action info.
        info = _ActionInfo()
        _check(_create_action_info(ctypes.byref(info)), "create_action_info")
        _check(_action_info_set_isa_name(info, isa.encode("utf-8")), "set_isa")
        _check(_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "set_lang")
        opt_array = (ctypes.c_char_p * len(options))(
            *[o.encode("utf-8") for o in options]
        )
        _check(_action_info_set_options(info, opt_array, len(options)), "set_options")

        timings = ComgrTimings()

        # Stage 1: LLVM IR (text/source) -> BC, one per module.
        bc_set = _DataSet()
        _check(_create_data_set(ctypes.byref(bc_set)), "create_data_set(bc)")
        t0 = time.perf_counter()
        _check(
            _do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc_set),
            "do_action(COMPILE_SOURCE_TO_BC)",
        )
        timings.bc = time.perf_counter() - t0

        codegen_input = bc_set
        if linking:
            # Stage 1b: every BC -> one linked BC.
            _require_count(
                bc_set, AMD_COMGR_DATA_KIND_BC, len(ir_texts), "COMPILE_SOURCE_TO_BC"
            )
            linked_set = _DataSet()
            _check(_create_data_set(ctypes.byref(linked_set)), "create_data_set(link)")
            t0 = time.perf_counter()
            _check(
                _do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, info, bc_set, linked_set),
                "do_action(LINK_BC_TO_BC)",
            )
            timings.link = time.perf_counter() - t0
            _require_count(linked_set, AMD_COMGR_DATA_KIND_BC, 1, "LINK_BC_TO_BC")
            codegen_input = linked_set

        # Stage 2: BC -> relocatable ELF
        reloc_set = _DataSet()
        _check(_create_data_set(ctypes.byref(reloc_set)), "create_data_set(reloc)")
        t0 = time.perf_counter()
        _check(
            _do_action(
                AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                info,
                codegen_input,
                reloc_set,
            ),
            "do_action(CODEGEN_BC_TO_RELOCATABLE)",
        )
        timings.relocatable = time.perf_counter() - t0

        # Stage 3: relocatable -> executable (HSACO).
        exe_set = _DataSet()
        _check(_create_data_set(ctypes.byref(exe_set)), "create_data_set(exe)")
        t0 = time.perf_counter()
        _check(
            _do_action(
                AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                info,
                reloc_set,
                exe_set,
            ),
            "do_action(LINK_RELOCATABLE_TO_EXECUTABLE)",
        )
        timings.executable = time.perf_counter() - t0

        hsaco = _extract_first(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE)
    finally:
        # Release every successfully-created handle in reverse order;
        # guard each so a partially-built pipeline still frees the rest.
        if exe_set is not None:
            _destroy_data_set(exe_set)
        if reloc_set is not None:
            _destroy_data_set(reloc_set)
        if linked_set is not None:
            _destroy_data_set(linked_set)
        if bc_set is not None:
            _destroy_data_set(bc_set)
        if info is not None:
            _destroy_action_info(info)
        for src in reversed(sources):
            _release_data(src)
        if in_set is not None:
            _destroy_data_set(in_set)

    return hsaco, timings
