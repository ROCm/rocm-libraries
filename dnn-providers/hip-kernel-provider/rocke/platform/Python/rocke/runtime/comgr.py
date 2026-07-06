# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python ctypes wrapper around `libamd_comgr` for in-process compilation.

The chain we drive:

    LLVM IR text (utf-8)
      -> AMD_COMGR_DATA_KIND_SOURCE  (lang = LLVM_IR)
      -> AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC          -> BC
      -> AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE     -> ELF relocatable
      -> AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE -> HSA code object

The resulting HSACO bytes are returned to Python and can be handed
straight to `hipModuleLoadData` (see `_hip_module.py`). No subprocesses,
no `<hip/hip_runtime.h>` parsing, no clang spawn.

The library is loaded from the default ROCm library locations or the dynamic
linker search path. ABI definitions mirror ROCm's `amd_comgr.h`.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .hip_module import _IS_WINDOWS, _LazyFn, _add_dll_dir, _candidate_lib_paths


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


def _load_lib() -> ctypes.CDLL:
    # Pair the loader with ``hip_module._load_lib`` so the two halves of
    # the process always share a single HIP/comgr runtime instance. See
    # ``_torch_bundled_lib`` in ``hip_module`` for why a torch-shipped
    # libamd_comgr is preferred over /opt/rocm when torch is in the
    # process.
    err = None
    for p in _candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"]):
        try:
            _add_dll_dir(p)
            return ctypes.CDLL(p)
        except OSError as e:
            err = e
    name = "amd_comgr.dll" if _IS_WINDOWS else "libamd_comgr.so"
    raise ComgrError(f"cannot load {name} ({err!r})")


# Lazy: resolved on first call so that rocke and torch can be imported
# in any order. See ``hip_module._torch_bundled_lib`` for context.
_lib: Optional[ctypes.CDLL] = None


def _resolve_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


def resolved_lib_path() -> Optional[str]:
    """Path of the ``libamd_comgr`` this module will load (torch-bundled
    preferred over ``/opt/rocm``; see :func:`hip_module._torch_bundled_lib`).

    Returns the already-loaded lib's path once :func:`_resolve_lib` has run,
    else the first existing candidate. Pure lookup -- does NOT ``dlopen``, so it
    is safe to call from flavor resolution before any compile.
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


def resolved_lib_rocm_version() -> Optional[Tuple[int, int]]:
    """ROCm ``(major, minor)`` vintage of the comgr lib that will actually
    compile the IR -- the authoritative, import-order-robust signal for LLVM
    flavor selection (the flavor MUST match the compiling comgr).

    Derived from the *resolved comgr lib path* (:func:`resolved_lib_path`):

      * torch-bundled (path under the imported torch's package dir) ->
        ``torch.version.hip``;
      * a ROCm tree (``<root>/lib/libamd_comgr.so``) -> ``<root>/.info/version``
        (with ``/opt/rocm`` as a final fallback).

    Returns ``None`` when the path or version cannot be determined.
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
    # ROCm tree: <root>/lib/libamd_comgr.so -> <root>/.info/version.
    root = os.path.dirname(os.path.dirname(rp))
    for verfile in (os.path.join(root, ".info", "version"), "/opt/rocm/.info/version"):
        ver = _read_rocm_version_file(verfile)
        if ver is not None:
            return ver
    return None


def prefer_bundled_lib() -> Optional[Tuple[int, int]]:
    """Entrypoint hook: make LLVM-flavor selection import-order-independent.

    The resolver never imports torch as a side effect (a library must not), so it
    only prefers torch's bundled (newest) ``libamd_comgr`` when torch is ALREADY
    in the process -- see :func:`hip_module._torch_bundled_lib`. A CLI / runner
    that lowers IR should call this ONCE at startup, BEFORE the first lowering, so
    the bundled comgr (e.g. ROCm 7.2 / llvm22) is in the process and the LLVM
    flavor cannot be locked to a stale ``/opt/rocm`` by import order.

    Best-effort: a no-op when torch is absent (the system comgr is then the only
    one available anyway). Returns the resolved comgr ROCm ``(major, minor)`` so
    the caller can log/verify the vintage it pinned. Idempotent and cheap once
    torch is imported.
    """
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401 -- pulls the bundled (newest) comgr into the process
        except Exception:
            pass
    return resolved_lib_rocm_version()


def _ir_flavor_is_llvm22(ir_text: str) -> Optional[bool]:
    """Infer an IR module's LLVM flavor from its ``target datalayout`` p8 field.

    ``True`` = llvm22 (``p8:128:128:128:48``), ``False`` = llvm20
    (``p8:128:128`` only), ``None`` = no recognisable datalayout. Mirrors
    ``core.lower_llvm._DATALAYOUT_LLVM20`` / ``_DATALAYOUT_LLVM22`` (kept here
    to avoid a runtime->core import).
    """
    if "p8:128:128:128:48" in ir_text:
        return True
    if "p8:128:128-" in ir_text:
        return False
    return None


def _assert_ir_flavor_matches_lib(ir_text: str) -> None:
    """Refuse to feed comgr an IR whose LLVM flavor mismatches the loaded comgr.

    An llvm22 IR on a pre-7.2 comgr SIGABRTs deep in codegen (the
    ``make.buffer.rsrc.p8.p1`` i64-stride form only the >=7.2 backend selects);
    the reverse errors. This guard turns that into a clear, catchable
    :class:`ComgrError` naming the fix. No-op when either side is unknown -- we
    never block compilation on uncertainty.
    """
    ir22 = _ir_flavor_is_llvm22(ir_text)
    if ir22 is None:
        return
    ver = resolved_lib_rocm_version()
    if ver is None:
        return
    lib22 = ver >= (7, 2)
    if ir22 != lib22:
        raise ComgrError(
            "LLVM IR flavor / comgr vintage mismatch: IR is "
            f"{'llvm22' if ir22 else 'llvm20'} but the loaded comgr is ROCm "
            f"{ver[0]}.{ver[1]} ({'llvm22' if lib22 else 'llvm20'}) at "
            f"{resolved_lib_path()!r}. Import torch before lowering so both pick "
            "the same vintage, or set ROCKE_LLVM_FLAVOR to match the comgr lib. "
            "(An llvm22 IR on a <7.2 comgr aborts in codegen; this guard turns "
            "that abort into a clean error.)"
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


def _check(s: int, where: str) -> None:
    if s != AMD_COMGR_STATUS_SUCCESS:
        msg = ctypes.c_char_p()
        _status_string(s, ctypes.byref(msg))
        raise ComgrError(
            f"{where}: status={s} ({msg.value.decode() if msg.value else ''})"
        )


@dataclass
class ComgrTimings:
    bc: float = 0.0
    relocatable: float = 0.0
    executable: float = 0.0

    @property
    def total(self) -> float:
        return self.bc + self.relocatable + self.executable


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


def build_hsaco_from_llvm_ir(
    ir_text: str,
    *,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    options: Optional[List[str]] = None,
) -> Tuple[bytes, ComgrTimings]:
    """Compile LLVM IR text to a loadable HSACO blob, all in-process.

    Returns the (hsaco_bytes, timings) tuple. `hsaco_bytes` can be passed
    directly to `hipModuleLoadData`.
    """
    options = list(options or ["-O3"])

    # Fail fast + clean on an IR-flavor / comgr-vintage mismatch rather than
    # letting comgr SIGABRT in codegen (see _assert_ir_flavor_matches_lib).
    _assert_ir_flavor_matches_lib(ir_text)

    # Handles are created lazily below; declare them up front so the
    # ``finally`` block can release whatever was successfully created even
    # when an intermediate ``_check`` raises (the common case during a
    # validity/heuristics sweep over borderline specs).
    in_set = None
    src = None
    info = None
    bc_set = None
    reloc_set = None
    exe_set = None

    try:
        # Input data set (LLVM IR text wrapped as SOURCE).
        in_set = _DataSet()
        _check(_create_data_set(ctypes.byref(in_set)), "create_data_set(in)")
        src = _Data()
        _check(
            _create_data(AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(src)),
            "create_data(src)",
        )
        payload = ir_text.encode("utf-8")
        _check(_set_data(src, len(payload), payload), "set_data(src)")
        _check(_set_data_name(src, b"kernel.ll"), "set_data_name(src)")
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

        # Stage 1: LLVM IR (text/source) -> BC
        bc_set = _DataSet()
        _check(_create_data_set(ctypes.byref(bc_set)), "create_data_set(bc)")
        t0 = time.perf_counter()
        _check(
            _do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc_set),
            "do_action(COMPILE_SOURCE_TO_BC)",
        )
        timings.bc = time.perf_counter() - t0

        # Stage 2: BC -> relocatable ELF
        reloc_set = _DataSet()
        _check(_create_data_set(ctypes.byref(reloc_set)), "create_data_set(reloc)")
        t0 = time.perf_counter()
        _check(
            _do_action(
                AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set
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
        if bc_set is not None:
            _destroy_data_set(bc_set)
        if info is not None:
            _destroy_action_info(info)
        if src is not None:
            _release_data(src)
        if in_set is not None:
            _destroy_data_set(in_set)

    return hsaco, timings
