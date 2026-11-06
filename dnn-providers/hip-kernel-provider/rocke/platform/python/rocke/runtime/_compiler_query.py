# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compiler evidence from one handle; also a site-hook-free probe executable.

Running this file with ``python -I -S`` opens exactly the supplied compiler in a
fresh process. It has only standard-library dependencies and never discovers HIP,
imports Torch, or imports the rocKE package to obtain a version.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from dataclasses import asdict, dataclass


class ComgrError(RuntimeError):
    pass


class _Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


@dataclass(frozen=True)
class CompilerInfo:
    llvm_version: tuple[int, int, int] | None
    source: str
    requested_comgr: str | None
    comgr_path: str | None
    query_library_path: str | None

    def describe(self) -> str:
        version = (
            ".".join(map(str, self.llvm_version)) if self.llvm_version else "unknown"
        )
        return (
            f"LLVM {version} via {self.source}; "
            f"COMGR={self.comgr_path or self.requested_comgr!r}; "
            f"query library={self.query_library_path!r}"
        )


def _library_for_symbol(fn) -> str | None:
    try:
        address = ctypes.cast(fn, ctypes.c_void_p)
        if sys.platform == "win32":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            module = ctypes.c_void_p()
            kernel.GetModuleHandleExW.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            kernel.GetModuleHandleExW.restype = ctypes.c_int
            if not kernel.GetModuleHandleExW(0x6, address, ctypes.byref(module)):
                return None
            kernel.GetModuleFileNameW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_uint,
            ]
            kernel.GetModuleFileNameW.restype = ctypes.c_uint
            buf = ctypes.create_unicode_buffer(32768)
            size = kernel.GetModuleFileNameW(module, buf, len(buf))
            return buf.value if 0 < size < len(buf) else None

        if sys.platform == "linux":
            with open("/proc/self/maps") as mappings:
                for line in mappings:
                    fields = line.rstrip("\n").split(maxsplit=5)
                    if len(fields) != 6:
                        continue
                    start, end = fields[0].split("-", 1)
                    if int(start, 16) <= address.value < int(end, 16):
                        return fields[5] if os.path.isabs(fields[5]) else None
            return None

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


def _probe_llvm_version(lib) -> tuple[int, int, int] | None:
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
        if status:
            raise ComgrError(f"COMGR compiler-version probe failed: status={status}")

    def acquire(create, destroy, *args):
        handle = _Handle()
        check(create(*args, ctypes.byref(handle)))
        handles.append((destroy, handle))
        return handle

    try:
        inputs = acquire(create_set, destroy_set)
        outputs = acquire(create_set, destroy_set)
        source = acquire(create_data, release_data, 1)
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
        check(set_language(info, 1))
        check(action(0, info, inputs, outputs))
        output = acquire(get_output, release_data, outputs, 1, 0)
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


def query_info(
    lib, requested=None, *, origin=_library_for_symbol, preprocess=_probe_llvm_version
) -> CompilerInfo:
    version, source, query_path = None, "unavailable", None
    try:
        query = lib.LLVMGetVersion
    except AttributeError:
        query = None
    if query is not None:
        query.argtypes = [ctypes.POINTER(ctypes.c_uint)] * 3
        query.restype = None
        values = [ctypes.c_uint() for _ in range(3)]
        query(*(ctypes.byref(value) for value in values))
        if values[0].value:
            version = tuple(value.value for value in values)
            source, query_path = "LLVMGetVersion", origin(query)
    if version is None:
        try:
            version = preprocess(lib)
            if version:
                source, query_path = "COMGR preprocessing", origin(
                    lib.amd_comgr_do_action
                )
        except (AttributeError, ComgrError, OSError, ValueError):
            pass
    try:
        actual = origin(lib.amd_comgr_get_version)
    except AttributeError:
        actual = None
    return CompilerInfo(
        version,
        source,
        requested if requested is not None else getattr(lib, "_name", None),
        actual,
        query_path,
    )


def file_signature(path: str) -> tuple[int, int, int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _main() -> None:
    if len(sys.argv) != 2 or not os.path.isabs(sys.argv[1]):
        raise ComgrError("compiler probe requires one absolute library path")
    path = sys.argv[1]
    before = file_signature(path)
    directory = None
    if sys.platform == "win32":
        directory = os.add_dll_directory(os.path.dirname(path))
    try:
        library = ctypes.CDLL(path)
        info = query_info(library, path)
        if info.llvm_version is None or not info.comgr_path:
            raise ComgrError(
                f"compiler probe has insufficient evidence: {info.describe()}"
            )
        if not os.path.samefile(path, info.comgr_path) or before != file_signature(
            path
        ):
            raise ComgrError(
                "compiler probe file identity changed or differs from request"
            )
        print(json.dumps({"protocol": 1, "signature": before, "info": asdict(info)}))
    finally:
        if directory is not None:
            directory.close()


if __name__ == "__main__":
    _main()
