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

The library is `/opt/rocm/lib/libamd_comgr.so`. ABI definitions come from
`/opt/rocm/include/amd_comgr/amd_comgr.h`.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


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


_lib_paths = [
    "/opt/rocm/lib/libamd_comgr.so",
    "/opt/rocm/lib/libamd_comgr.so.3",
    "libamd_comgr.so",
]


class ComgrError(RuntimeError):
    pass


def _load_lib() -> ctypes.CDLL:
    err = None
    for p in _lib_paths:
        try:
            return ctypes.CDLL(p)
        except OSError as e:
            err = e
    raise ComgrError(f"cannot load libamd_comgr.so ({err!r})")


_lib = _load_lib()


# Opaque handles are returned as a struct containing a single uint64_t.
class _Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


# Type aliases.
_DataSet = _Handle
_Data = _Handle
_ActionInfo = _Handle


def _bind(name: str, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


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

    size = ctypes.c_size_t(0)
    _check(_get_data(data, ctypes.byref(size), None), "get_data (size)")
    buf = ctypes.create_string_buffer(size.value)
    _check(_get_data(data, ctypes.byref(size), buf), "get_data (read)")
    out = bytes(buf.raw[: size.value])
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

    # Input data set (LLVM IR text wrapped as SOURCE).
    in_set = _DataSet()
    _check(_create_data_set(ctypes.byref(in_set)), "create_data_set(in)")
    src = _Data()
    _check(
        _create_data(AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(src)), "create_data(src)"
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
    opt_array = (ctypes.c_char_p * len(options))(*[o.encode("utf-8") for o in options])
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
        _do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set),
        "do_action(CODEGEN_BC_TO_RELOCATABLE)",
    )
    timings.relocatable = time.perf_counter() - t0

    # Stage 3: relocatable -> executable (HSACO).
    exe_set = _DataSet()
    _check(_create_data_set(ctypes.byref(exe_set)), "create_data_set(exe)")
    t0 = time.perf_counter()
    _check(
        _do_action(
            AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc_set, exe_set
        ),
        "do_action(LINK_RELOCATABLE_TO_EXECUTABLE)",
    )
    timings.executable = time.perf_counter() - t0

    hsaco = _extract_first(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE)

    # Cleanup.
    _release_data(src)
    _destroy_data_set(in_set)
    _destroy_data_set(bc_set)
    _destroy_data_set(reloc_set)
    _destroy_data_set(exe_set)
    _destroy_action_info(info)

    return hsaco, timings
